//! Integration tests for the parallel-ready dense ERI assembly ([`EriBuilder`]).
//!
//! Pure-Rust, no external library and no threading runtime (integral itself never
//! pulls in `rayon`). The "parallel" paths here are driven *serially in a chosen
//! order* — order-independence of the disjoint writes is exactly what guarantees a
//! real driver may run them concurrently.
//!
//! The headline check is the item-7 trap: the 4-fold path fills an element from its
//! own bra-pair `(ij|kl)`, whereas serial 8-fold `eri()` may fill it from the
//! bra↔ket-swapped `(kl|ij)`. The kernels are **not** bit-symmetric under bra↔ket
//! exchange, so a blanket bit-identical assertion would fail spuriously. We instead:
//!   * compare the whole tensor with the repo's tight significant-element tolerance,
//!   * and assert **bit-identical** on the subset where the element's bra-pair is
//!     lexicographically ≥ its ket-pair (there both paths invoke the *same* kernel
//!     call), proving the only divergence is the documented bra↔ket round-off.

use integral::{Basis, BraPairFill, Engine, EriBuilder, Shell};

/// Mixed s/p/d/f Cartesian shells on different centers (so most blocks are
/// genuinely distinct), with a repeated-`l` pair to exercise the `i == j` / `k == l`
/// symmetry collapses.
fn mixed_basis() -> Basis {
    Basis::new(vec![
        Shell::new(0, [0.0, 0.0, 0.0], vec![1.4, 0.5], vec![0.6, 0.5]).unwrap(),
        Shell::new(1, [0.6, -0.3, 0.2], vec![0.9], vec![1.0]).unwrap(),
        Shell::new(2, [-0.4, 0.7, -0.1], vec![1.1], vec![1.0]).unwrap(),
        Shell::new(3, [0.2, 0.5, 0.8], vec![0.7], vec![1.0]).unwrap(),
    ])
}

/// A small basis mixing Cartesian and spherical shells (covers the c2s path) with a
/// repeated shell so diagonal bra/ket pairs occur.
fn spherical_basis() -> Basis {
    Basis::new(vec![
        Shell::new(0, [0.0, 0.0, 0.0], vec![1.2, 0.4], vec![0.5, 0.6]).unwrap(),
        Shell::new_spherical(1, [0.5, 0.1, -0.2], vec![0.8], vec![1.0]).unwrap(),
        Shell::new_spherical(2, [-0.3, 0.4, 0.6], vec![1.0], vec![1.0]).unwrap(),
        Shell::new(0, [-0.3, 0.4, 0.6], vec![0.7], vec![1.0]).unwrap(),
    ])
}

/// Map each output AO index to its shell index, via the per-shell `n_func` offsets.
fn ao_to_shell(basis: &Basis) -> Vec<usize> {
    let mut map = Vec::new();
    for (s, shell) in basis.shells().iter().enumerate() {
        for _ in 0..shell.n_func() {
            map.push(s);
        }
    }
    map
}

/// Canonical (sorted, larger-first) shell pair of two shells.
fn canon(a: usize, b: usize) -> (usize, usize) {
    if a >= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Lexicographic `≥` on canonical pairs — the same ordering the 8-fold `eri()`
/// driver uses to pick which pair is the canonical bra (`ij ≥ kl`).
fn pair_ge(p: (usize, usize), q: (usize, usize)) -> bool {
    p.0 > q.0 || (p.0 == q.0 && p.1 >= q.1)
}

/// Compare an `EriBuilder` tensor against the serial `eri()` reference: a tight
/// significant-element relative tolerance + absolute floor everywhere, and
/// bit-identical on the bra-pair ≥ ket-pair subset.
///
/// Returns the count of bit-mismatched elements (all of which must be in the
/// swapped subset) so callers can sanity-check the swapped path was exercised.
fn assert_matches_reference(basis: &Basis, candidate: &[f64], reference: &[f64]) -> usize {
    assert_eq!(candidate.len(), reference.len());
    let nao = basis.nao();
    assert_eq!(reference.len(), nao.pow(4));
    let shell_of = ao_to_shell(basis);

    let peak = reference.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    let floor = 1e-3 * peak;
    let mut worst_sig = 0.0_f64;
    let mut worst_abs = 0.0_f64;
    let mut bit_mismatches = 0usize;

    for mu in 0..nao {
        for nu in 0..nao {
            let bp = canon(shell_of[mu], shell_of[nu]);
            for la in 0..nao {
                for sg in 0..nao {
                    let idx = ((mu * nao + nu) * nao + la) * nao + sg;
                    let (r, c) = (reference[idx], candidate[idx]);
                    let dv = (r - c).abs();
                    worst_abs = worst_abs.max(dv);
                    if r.abs() >= floor {
                        worst_sig = worst_sig.max(dv / r.abs());
                    }
                    let kp = canon(shell_of[la], shell_of[sg]);
                    if pair_ge(bp, kp) {
                        // Same kernel call in both paths ⇒ must be bitwise identical.
                        assert_eq!(
                            r.to_bits(),
                            c.to_bits(),
                            "non-swapped element ({mu}{nu}|{la}{sg}) must be bit-identical: \
                             ref={r:e} cand={c:e}"
                        );
                    } else if r.to_bits() != c.to_bits() {
                        bit_mismatches += 1;
                    }
                }
            }
        }
    }

    assert!(
        worst_sig < 1e-11,
        "worst significant-element relative diff {worst_sig:e} exceeds 1e-11"
    );
    assert!(
        worst_abs < 1e-11 * peak.max(1.0) + 1e-12,
        "worst absolute diff {worst_abs:e} exceeds floor (peak {peak:e})"
    );
    bit_mismatches
}

#[test]
fn build_matches_eri_auto() {
    let basis = mixed_basis();
    let reference = basis.eri();
    let built = EriBuilder::new(&basis).build();
    let mismatches = assert_matches_reference(&basis, &built, &reference);
    // This mixed basis genuinely has bra-pair < ket-pair elements (the swapped
    // subset), so the bra↔ket asymmetry must actually show up — confirming the
    // test exercises the tolerance path and isn't vacuously bit-identical.
    assert!(
        mismatches > 0,
        "expected the bra↔ket-swapped subset to differ at the bit level"
    );
}

#[test]
fn build_matches_eri_forced_engines() {
    let basis = mixed_basis();
    for engine in [Engine::OsHgp, Engine::Rys] {
        let reference = basis.eri_with(engine);
        let built = EriBuilder::with_engine(&basis, engine).build();
        assert_matches_reference(&basis, &built, &reference);
    }
}

#[test]
fn build_matches_eri_spherical() {
    let basis = spherical_basis();
    let reference = basis.eri();
    let built = EriBuilder::new(&basis).build();
    assert_matches_reference(&basis, &built, &reference);
}

#[test]
fn partition_fill_equals_build_any_order() {
    // Driving partition+fill in *forward* and *reverse* task order must both give
    // exactly the same buffer as the serial `build()` — bitwise. Order-independence
    // of the per-bra-pair writes is the run-time face of the disjointness contract:
    // a concurrent driver imposes an arbitrary order, and it must not matter.
    let basis = mixed_basis();
    let builder = EriBuilder::new(&basis);
    let serial = builder.build();

    // Forward order.
    let mut fwd = vec![0.0; builder.output_len()];
    {
        let mut tasks = builder.partition(&mut fwd);
        for t in &mut tasks {
            builder.fill(t);
        }
    }
    assert_eq!(fwd, serial, "forward partition+fill differs from build()");

    // Reverse order (simulates a different concurrent schedule).
    let mut rev = vec![0.0; builder.output_len()];
    {
        let mut tasks = builder.partition(&mut rev);
        for t in tasks.iter_mut().rev() {
            builder.fill(t);
        }
    }
    assert_eq!(
        rev, serial,
        "reverse-order partition+fill differs from build()"
    );
}

#[test]
fn partition_into_prefilled_buffer_overwrites_cleanly() {
    // A full build overwrites every element, so starting from garbage (not zero)
    // must still yield the exact tensor — proof that the union of bra-pair writes
    // covers all nao⁴ elements (no element left at its pre-fill value).
    let basis = mixed_basis();
    let builder = EriBuilder::new(&basis);
    let serial = builder.build();

    let mut buf = vec![1234.5; builder.output_len()];
    {
        let mut tasks = builder.partition(&mut buf);
        for t in &mut tasks {
            builder.fill(t);
        }
    }
    assert_eq!(buf, serial, "build did not overwrite every element");
}

#[test]
fn bra_pairs_are_canonical_and_align_with_tasks() {
    let basis = mixed_basis();
    let builder = EriBuilder::new(&basis);
    let nsh = basis.shells().len();

    // Exactly the canonical i ≥ j pairs, in (i outer, j inner) order.
    let mut expected = Vec::new();
    for i in 0..nsh {
        for j in 0..=i {
            expected.push((i, j));
        }
    }
    assert_eq!(builder.bra_pairs(), expected.as_slice());
    assert_eq!(builder.bra_pairs().len(), nsh * (nsh + 1) / 2);

    // Each partition task is the bra-pair at the same index.
    let mut out = vec![0.0; builder.output_len()];
    let tasks = builder.partition(&mut out);
    assert_eq!(tasks.len(), builder.bra_pairs().len());
    for (task, &pair) in tasks.iter().zip(builder.bra_pairs()) {
        assert_eq!(task.bra(), pair);
        assert!(pair.0 >= pair.1, "bra-pair {pair:?} is not canonical");
    }
}

#[test]
fn types_are_thread_safe_for_external_drivers() {
    // The seam's contract: a driver shares `&EriBuilder` across threads (Sync) and
    // fans `BraPairFill` tasks out to them (Send), e.g.
    // `tasks.par_iter_mut().for_each(|t| builder.fill(t))`. If a future field broke
    // this, the parallel API would be unusable — pin it down at compile time.
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<BraPairFill<'_>>();
    assert_sync::<EriBuilder<'_>>();
    assert_send::<EriBuilder<'_>>();
}

#[test]
fn output_len_is_nao_pow4() {
    let basis = mixed_basis();
    let builder = EriBuilder::new(&basis);
    assert_eq!(builder.output_len(), basis.nao().pow(4));
}

#[test]
#[should_panic(expected = "nao⁴")]
fn partition_rejects_wrong_buffer_size() {
    let basis = mixed_basis();
    let builder = EriBuilder::new(&basis);
    let mut wrong = vec![0.0; builder.output_len() - 1];
    let _ = builder.partition(&mut wrong);
}

/// `EriKernel::Erf` through the builder: the partitioned/parallel fill must
/// reproduce the serial `Basis::eri_kernel(Erf)` tensor essentially exactly —
/// per element `|Δ| ≤ 1e-14 · max(|ref|, 1)` (the only divergence allowed is
/// the documented bra↔ket round-off, orders of magnitude below this).
fn assert_matches_erf_reference(candidate: &[f64], reference: &[f64]) {
    assert_eq!(candidate.len(), reference.len());
    for (idx, (&r, &c)) in reference.iter().zip(candidate).enumerate() {
        assert!(
            (r - c).abs() <= 1e-14 * r.abs().max(1.0),
            "element {idx}: serial {r:e} vs builder {c:e}"
        );
    }
}

#[test]
fn erf_kernel_build_matches_serial_eri_kernel() {
    use integral::EriKernel;
    let omega = 0.33;
    for basis in [mixed_basis(), spherical_basis()] {
        let reference = basis.eri_kernel(EriKernel::Erf { omega });
        let builder = EriBuilder::new(&basis).kernel(EriKernel::Erf { omega });
        let built = builder.build();
        assert_matches_erf_reference(&built, &reference);

        // Partitioned fill, forward and reverse task order, must equal the
        // serial builder build bitwise (the same disjoint-write contract the
        // Coulomb seam guarantees, now over the attenuated kernel).
        for reverse in [false, true] {
            let mut out = vec![0.0; builder.output_len()];
            let mut tasks = builder.partition(&mut out);
            if reverse {
                for t in tasks.iter_mut().rev() {
                    builder.fill(t);
                }
            } else {
                for t in &mut tasks {
                    builder.fill(t);
                }
            }
            drop(tasks);
            assert_eq!(out, built, "partitioned Erf fill differs from build()");
        }
    }
}

#[test]
fn default_and_explicit_coulomb_kernel_are_bit_identical() {
    // The kernel setter must leave the Coulomb path untouched: a builder that
    // never calls `.kernel(..)` and one that selects `Coulomb` explicitly
    // produce bitwise-identical tensors (fingerprint-level check of the
    // "default Coulomb behavior unchanged" contract).
    use integral::EriKernel;
    let basis = mixed_basis();
    let plain = EriBuilder::new(&basis).build();
    let explicit = EriBuilder::new(&basis).kernel(EriKernel::Coulomb).build();
    assert_eq!(plain, explicit);
}

#[test]
#[should_panic(expected = "finite omega > 0")]
fn erf_kernel_rejects_nonpositive_omega() {
    use integral::EriKernel;
    let basis = mixed_basis();
    let _ = EriBuilder::new(&basis).kernel(EriKernel::Erf { omega: 0.0 });
}
