//! Density-fitting integrals: 2-center metric `(P|Q)`, 3-center blocks
//! `(ij|P)`, aux Schwarz bounds, and the parallel `Eri3cBuilder`.
//!
//! Oracles used here:
//! - **Closed forms** for concentric `s`-only integrals (every Gaussian-product
//!   prefactor `K = 1` and the Boys factor `F₀(0) = 1`), evaluated directly in
//!   the test — independent of both engines.
//! - **Zero-exponent limit**: the unit `s` dummy is the `α → 0` limit of a
//!   normalized-away `s` shell, so `(ij|P)` must match a 4-center `eri_block`
//!   against an explicit tiny-exponent `s` shell with effective coefficient 1.
//! - **Cross-engine**: OS/HGP and Rys must agree on the same blocks.

use integral::{Basis, Engine, Shell};

const PI: f64 = std::f64::consts::PI;

/// `N(α; 0,0,0) = (2α/π)^{3/4}` — the s-shell normalization constant.
fn s_norm(alpha: f64) -> f64 {
    (2.0 * alpha / PI).powf(0.75)
}

fn close(a: f64, b: f64, tol: f64, what: &str) {
    let denom = b.abs().max(1.0);
    assert!(
        (a - b).abs() / denom < tol,
        "{what}: got {a:.15e}, want {b:.15e} (rel {:.2e})",
        (a - b).abs() / denom
    );
}

/// Mixed main basis: contracted s, two p's sharing a center, spherical d.
fn main_basis() -> Basis {
    Basis::new(vec![
        Shell::new(0, [0.0, 0.0, 0.0], vec![1.4, 0.5], vec![0.6, 0.5]).unwrap(),
        Shell::new(1, [0.6, -0.3, 0.2], vec![0.9], vec![1.0]).unwrap(),
        Shell::new(1, [0.6, -0.3, 0.2], vec![0.4], vec![1.0]).unwrap(),
        Shell::new_spherical(2, [-0.4, 0.7, -0.1], vec![1.1], vec![1.0]).unwrap(),
    ])
}

/// Mixed aux basis on/near the same centers: s, p, spherical d, contracted s.
fn aux_basis() -> Basis {
    Basis::new(vec![
        Shell::new(0, [0.0, 0.0, 0.0], vec![2.1], vec![1.0]).unwrap(),
        Shell::new(1, [0.6, -0.3, 0.2], vec![1.3], vec![1.0]).unwrap(),
        Shell::new_spherical(2, [-0.4, 0.7, -0.1], vec![0.8], vec![1.0]).unwrap(),
        Shell::new(0, [0.3, 0.1, -0.2], vec![1.9, 0.7], vec![0.4, 0.8]).unwrap(),
    ])
}

// ---------------------------------------------------------------- closed form

#[test]
fn eri_2c_concentric_s_closed_form() {
    // Two single-primitive s aux functions at the origin: every product center
    // coincides, so (P|Q) = N_a N_b · 2π^{5/2} / (a·b·√(a+b)) exactly.
    let (a, b) = (0.9_f64, 1.7_f64);
    let aux = Basis::new(vec![
        Shell::new(0, [0.0; 3], vec![a], vec![1.0]).unwrap(),
        Shell::new(0, [0.0; 3], vec![b], vec![1.0]).unwrap(),
    ]);
    let m = aux.eri_2c();
    let pref = |x: f64, y: f64| 2.0 * PI.powf(2.5) / (x * y * (x + y).sqrt());
    close(m[1], s_norm(a) * s_norm(b) * pref(a, b), 1e-12, "(P|Q)");
    close(m[0], s_norm(a) * s_norm(a) * pref(a, a), 1e-12, "(P|P)");
    close(m[3], s_norm(b) * s_norm(b) * pref(b, b), 1e-12, "(Q|Q)");
    assert_eq!(m[1], m[2], "metric must be exactly symmetric");
}

#[test]
fn eri_3c_concentric_s_closed_form() {
    // (ss|P) with all three s functions at the origin: the bra pair contracts
    // to exponent p = a + b at the origin, so
    // (ss|P) = N_a N_b N_c · 2π^{5/2} / (p·c·√(p+c)).
    let (a, b, c) = (0.8_f64, 1.1_f64, 0.6_f64);
    let main = Basis::new(vec![
        Shell::new(0, [0.0; 3], vec![a], vec![1.0]).unwrap(),
        Shell::new(0, [0.0; 3], vec![b], vec![1.0]).unwrap(),
    ]);
    let aux = Basis::new(vec![Shell::new(0, [0.0; 3], vec![c], vec![1.0]).unwrap()]);
    let block = main.eri_3c_block(&aux, 0, 1, 0);
    assert_eq!(block.len(), 1);
    let p = a + b;
    let want = s_norm(a) * s_norm(b) * s_norm(c) * 2.0 * PI.powf(2.5) / (p * c * (p + c).sqrt());
    close(block[0], want, 1e-12, "(ss|P)");
}

// ------------------------------------------------------- zero-exponent limit

/// `(ij|P)` must equal the 4-center `(ij|P,sε)` with an explicit `s` shell of
/// exponent `ε → 0` whose *effective* coefficient is 1 (raw coefficient
/// `1/N(ε)`). With `ε = 1e-12` the truncation error is far below 1e-8.
#[test]
fn eri_3c_matches_tiny_exponent_4c_limit() {
    let main = main_basis();
    let aux = aux_basis();
    let eps = 1e-12;
    for (psh, sp) in aux.shells().iter().enumerate() {
        // 4c basis: the main shells, this aux shell, and the tiny-s dummy.
        let mut shells: Vec<Shell> = main.shells().to_vec();
        let p_idx = shells.len();
        shells.push(sp.clone());
        let tiny_idx = shells.len();
        shells.push(Shell::new(0, sp.center(), vec![eps], vec![1.0 / s_norm(eps)]).unwrap());
        let four = Basis::new(shells);
        for ish in 0..main.shells().len() {
            for jsh in 0..main.shells().len() {
                let got = main.eri_3c_block(&aux, ish, jsh, psh);
                let want = four.eri_block(ish, jsh, p_idx, tiny_idx);
                assert_eq!(got.len(), want.len());
                let peak = want.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
                for (g, w) in got.iter().zip(&want) {
                    assert!(
                        (g - w).abs() <= 1e-8 * peak + 1e-12,
                        "(i{ish} j{jsh} | P{psh}): {g:e} vs tiny-exponent {w:e}"
                    );
                }
            }
        }
    }
}

// --------------------------------------------------------------- cross-engine

#[test]
fn eri_2c_cross_engine_and_symmetry() {
    let aux = aux_basis();
    let os = aux.eri_2c_with(Engine::OsHgp);
    let rys = aux.eri_2c_with(Engine::Rys);
    let auto = aux.eri_2c();
    let n = aux.nao();
    assert_eq!(os.len(), n * n);
    let peak = os.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    for (idx, (a, b)) in os.iter().zip(&rys).enumerate() {
        assert!(
            (a - b).abs() <= 1e-11 * peak.max(1.0),
            "(P|Q) engines disagree at {idx}: {a:e} vs {b:e}"
        );
    }
    for i in 0..n {
        for j in 0..n {
            assert_eq!(auto[i * n + j], auto[j * n + i], "exact symmetry");
        }
        assert!(auto[i * n + i] > 0.0, "(P|P) diagonal must be positive");
    }
}

#[test]
fn eri_3c_cross_engine() {
    let main = main_basis();
    let aux = aux_basis();
    for ish in 0..main.shells().len() {
        for jsh in 0..main.shells().len() {
            for psh in 0..aux.shells().len() {
                let os = main.eri_3c_block_with(Engine::OsHgp, &aux, ish, jsh, psh);
                let rys = main.eri_3c_block_with(Engine::Rys, &aux, ish, jsh, psh);
                let peak = os.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
                for (a, b) in os.iter().zip(&rys) {
                    assert!(
                        (a - b).abs() <= 1e-11 * peak.max(1.0),
                        "(i{ish} j{jsh}|P{psh}) engines disagree: {a:e} vs {b:e}"
                    );
                }
            }
        }
    }
}

// ------------------------------------------------------------ metric quality

/// The metric must be symmetric positive-definite (a plain Cholesky succeeds)
/// — what chemx relies on to form (P|Q)^{-1/2}.
#[test]
fn eri_2c_is_positive_definite() {
    let aux = aux_basis();
    let n = aux.nao();
    let mut m = aux.eri_2c();
    // In-place lower Cholesky; panics on a non-positive pivot.
    for k in 0..n {
        for j in 0..k {
            let l = m[k * n + j];
            for i in k..n {
                m[i * n + k] -= m[i * n + j] * l;
            }
        }
        let d = m[k * n + k];
        assert!(d > 0.0, "non-positive Cholesky pivot {d:e} at {k}");
        let inv = 1.0 / d.sqrt();
        for i in k..n {
            m[i * n + k] *= inv;
        }
    }
}

// ------------------------------------------------------------------ screening

#[test]
fn schwarz_aux_bound_holds_for_all_3c_blocks() {
    let main = main_basis();
    let aux = aux_basis();
    let q = main.schwarz_bounds();
    let qp = aux.schwarz_aux_bounds();
    let nsh = main.shells().len();
    assert_eq!(qp.len(), aux.shells().len());
    for ish in 0..nsh {
        for jsh in 0..nsh {
            for (psh, &qpp) in qp.iter().enumerate() {
                let bound = q[ish * nsh + jsh] * qpp;
                let block = main.eri_3c_block(&aux, ish, jsh, psh);
                for &v in &block {
                    assert!(
                        v.abs() <= bound * (1.0 + 1e-10) + 1e-14,
                        "|(i{ish} j{jsh}|P{psh})| = {:e} exceeds bound {bound:e}",
                        v.abs()
                    );
                }
            }
        }
    }
}

// -------------------------------------------------------------------- builder

#[test]
fn eri_3c_builder_matches_blockwise_assembly() {
    let main = main_basis();
    let aux = aux_basis();
    let builder = main.eri_3c_builder(&aux);
    let tensor = builder.build();
    let (nao, naux) = (main.nao(), aux.nao());
    assert_eq!(tensor.len(), nao * nao * naux);

    // Reference: place every (ij|P) block independently (no symmetry reuse).
    let offs = |b: &Basis| {
        let mut o = vec![0usize];
        for s in b.shells() {
            o.push(o.last().unwrap() + s.n_func());
        }
        o
    };
    let (mo, ao) = (offs(&main), offs(&aux));
    let mut want = vec![0.0; nao * nao * naux];
    for (ish, si) in main.shells().iter().enumerate() {
        for (jsh, sj) in main.shells().iter().enumerate() {
            for (psh, sp) in aux.shells().iter().enumerate() {
                let block = main.eri_3c_block(&aux, ish, jsh, psh);
                let (nj, np) = (sj.n_func(), sp.n_func());
                for a in 0..si.n_func() {
                    for b in 0..nj {
                        for c in 0..np {
                            want[((mo[ish] + a) * nao + mo[jsh] + b) * naux + ao[psh] + c] =
                                block[(a * nj + b) * np + c];
                        }
                    }
                }
            }
        }
    }
    let peak = want.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    for (idx, (g, w)) in tensor.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-11 * peak.max(1.0),
            "builder differs from blockwise at {idx}: {g:e} vs {w:e}"
        );
    }
}

#[test]
fn eri_3c_builder_partition_covers_all_rows_once() {
    let main = main_basis();
    let aux = aux_basis();
    let builder = main.eri_3c_builder(&aux);
    let mut out = vec![0.0; builder.output_len()];
    let tasks = builder.partition(&mut out);
    assert_eq!(tasks.len(), builder.bra_pairs().len());
    let total_rows: usize = tasks
        .iter()
        .zip(builder.bra_pairs())
        .map(|(t, &(i, j))| {
            assert_eq!(t.bra(), (i, j));
            let (ni, nj) = (main.shells()[i].n_func(), main.shells()[j].n_func());
            if i == j {
                ni * nj
            } else {
                2 * ni * nj
            }
        })
        .sum();
    assert_eq!(total_rows, main.nao() * main.nao());
}

#[test]
fn fill_filtered_keep_all_is_bitwise_identical_to_fill() {
    let main = main_basis();
    let aux = aux_basis();
    let builder = main.eri_3c_builder(&aux);

    let mut plain = vec![0.0; builder.output_len()];
    for task in &mut builder.partition(&mut plain) {
        builder.fill(task);
    }
    let mut filtered = vec![0.0; builder.output_len()];
    for task in &mut builder.partition(&mut filtered) {
        builder.fill_filtered(task, |_| true);
    }
    for (idx, (a, b)) in plain.iter().zip(&filtered).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "fill_filtered(keep-all) differs from fill at {idx}: {a:e} vs {b:e}"
        );
    }
}

/// A per-task Schwarz predicate `Q[ij]·QP[p] ≥ τ` must leave kept aux shells
/// bitwise equal to the unfiltered fill and zero exactly the dropped ones —
/// and every dropped value must indeed lie below the bound τ.
#[test]
fn fill_filtered_schwarz_predicate_drops_only_below_bound() {
    let main = main_basis();
    let aux = aux_basis();
    let builder = main.eri_3c_builder(&aux);
    let q = main.schwarz_bounds();
    let qp = aux.schwarz_aux_bounds();
    let nsh = main.shells().len();
    let naux = aux.nao();
    // Pick τ strictly between the smallest and largest Schwarz product so the
    // predicate provably both keeps and drops shells on this basis.
    let mut lo = f64::INFINITY;
    let mut hi = 0.0_f64;
    for i in 0..nsh {
        for j in 0..=i {
            for &qpp in &qp {
                let b = q[i * nsh + j] * qpp;
                lo = lo.min(b);
                hi = hi.max(b);
            }
        }
    }
    assert!(lo < hi, "degenerate bounds, cannot screen");
    let tau = (lo * hi).sqrt();

    let mut full = vec![0.0; builder.output_len()];
    for task in &mut builder.partition(&mut full) {
        builder.fill(task);
    }

    let mut screened = vec![0.0; builder.output_len()];
    let mut tasks = builder.partition(&mut screened);
    let mut dropped_any = false;
    for task in &mut tasks {
        let (i, j) = task.bra();
        let qij = q[i * nsh + j];
        let mut kept = vec![false; aux.shells().len()];
        for (p, k) in kept.iter_mut().enumerate() {
            *k = qij * qp[p] >= tau;
            dropped_any |= !*k;
        }
        builder.fill_filtered(task, |p| kept[p]);
    }
    assert!(
        dropped_any,
        "τ = {tau:e} screened nothing — test is vacuous"
    );

    // Map each output element back to its (bra-pair, aux-shell) keep decision.
    let aux_offs = {
        let mut o = vec![0usize];
        for s in aux.shells() {
            o.push(o.last().unwrap() + s.n_func());
        }
        o
    };
    let main_offs = {
        let mut o = vec![0usize];
        for s in main.shells() {
            o.push(o.last().unwrap() + s.n_func());
        }
        o
    };
    let shell_of = |ao: usize, offs: &[usize]| offs.iter().rposition(|&o| o <= ao).unwrap();
    for (idx, (&s, &f)) in screened.iter().zip(&full).enumerate() {
        let p_ao = idx % naux;
        let nu = (idx / naux) % main.nao();
        let mu = idx / (naux * main.nao());
        let (ish, jsh) = (shell_of(mu, &main_offs), shell_of(nu, &main_offs));
        let psh = shell_of(p_ao, &aux_offs);
        // The canonical task is (max, min); the bound is symmetric in (i, j).
        let qij = q[ish.max(jsh) * nsh + ish.min(jsh)];
        if qij * qp[psh] >= tau {
            assert_eq!(s.to_bits(), f.to_bits(), "kept slot {idx} altered");
        } else {
            assert_eq!(s, 0.0, "dropped slot {idx} not zeroed");
            assert!(
                f.abs() <= tau * (1.0 + 1e-10),
                "dropped slot {idx} had |value| {:e} above τ = {tau:e}",
                f.abs()
            );
        }
    }
}

#[test]
fn eri_3c_builder_forced_engines_agree() {
    let main = main_basis();
    let aux = aux_basis();
    let os = integral::Eri3cBuilder::with_engine(&main, &aux, Engine::OsHgp).build();
    let rys = integral::Eri3cBuilder::with_engine(&main, &aux, Engine::Rys).build();
    let peak = os.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    for (a, b) in os.iter().zip(&rys) {
        assert!((a - b).abs() <= 1e-11 * peak.max(1.0));
    }
}
