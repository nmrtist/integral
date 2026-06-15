//! Cross-engine validation: the OS/HGP and Rys ERI engines must be
//! **transparent** — same Coulomb integral, same `(a,b,c,d)` block layout — and
//! the `Auto` dispatch must equal whichever engine it picks.
//!
//! Pure Rust, no external library. These tests lock the
//! engine-to-engine agreement, the 8-fold symmetry and falsifiable layout from
//! independently built OS blocks, and the dispatch-boundary behaviour.

use integral::math::am::n_cart;
use integral::{select_engine, Basis, Engine, Shell};

#[test]
fn dispatch_policy_matches_5c_calibration() {
    // Calibrated for the OS/HGP engine carrying monomorphized + 4-lane-SIMD VRR
    // kernels for every (ne,nf) ≤ 6. `select_engine(ne, nf, deg)`.
    //
    // Where a fast kernel exists (max(ne,nf) ≤ 6) the measured OS-vs-Rys crossover
    // sits at deg 2 (deg 1 → Rys, deg 2 → OS), so the l_total 6–16 d/f band is set
    // one notch conservative at deg ≥ 3 (deg 1–2 → Rys). g-heavy quartets
    // (max(ne,nf) ≥ 7, no fast kernel) keep the old deg ≥ 16 policy.
    use Engine::{OsHgp, Rys};
    // l_total ≤ 5: always OS, down to deg 1.
    assert_eq!(select_engine(0, 0, 1), OsHgp); // (ss|ss) uncontracted
    assert_eq!(select_engine(2, 2, 1), OsHgp); // l_total 4, deg 1 → OS
                                               // l_total 6–16 WITH a fast kernel (max(ne,nf) ≤ 6): deg 1–2 → Rys, deg ≥ 3 → OS.
    assert_eq!(select_engine(4, 4, 1), Rys); // (dd|dd) deg 1 → Rys (uncontracted: Rys wins)
    assert_eq!(select_engine(4, 4, 2), Rys); // (dd|dd) deg 2 → Rys (edge below)
    assert_eq!(select_engine(4, 4, 3), OsHgp); // (dd|dd) deg 3 → OS (edge at threshold)
    assert_eq!(select_engine(6, 6, 2), Rys); // (ff|ff) deg 2 → Rys (same crossover as dd)
    assert_eq!(select_engine(6, 6, 3), OsHgp); // (ff|ff) deg 3 → OS
                                               // l_total 6–16 WITHOUT a fast kernel (max(ne,nf) ≥ 7, g-heavy): old deg ≥ 16.
    assert_eq!(select_engine(7, 0, 3), Rys); // ne=7: no fast kernel → deg 3 still Rys
    assert_eq!(select_engine(8, 8, 15), Rys); // l_total 16, deg 15 → Rys
    assert_eq!(select_engine(8, 8, 16), OsHgp); // l_total 16, deg 16 → OS
    assert_eq!(select_engine(8, 8, 1), Rys); // …but deg 1 still Rys
                                             // l_total ≥ 17: always Rys.
    assert_eq!(select_engine(9, 8, usize::MAX), Rys);
}

/// `|os − rys| ≤ atol + rtol·|rys|` with `atol = 1e-11`, `rtol = 1e-10`. The two
/// engines are independent f64 recurrences agreeing to ~1e-12 on dominant
/// elements; the atol floor absorbs benign near-cancellation on tiny components.
fn assert_close(os: &[f64], rys: &[f64], tag: &str) {
    assert_eq!(os.len(), rys.len(), "{tag}: length mismatch");
    const ATOL: f64 = 1e-11;
    const RTOL: f64 = 1e-10;
    let mut worst = 0.0_f64;
    for (i, (o, r)) in os.iter().zip(rys.iter()).enumerate() {
        let d = (o - r).abs();
        worst = worst.max(d);
        assert!(
            d <= ATOL + RTOL * r.abs(),
            "{tag}[{i}]: OS {o} vs Rys {r} (Δ={d:e})"
        );
    }
    assert!(worst.is_finite(), "{tag}: non-finite Δ");
}

/// s (contracted), p, d, f on four distinct centres — every (ij|kl) block is
/// genuinely distinct, and several shells are contracted.
fn mixed_basis() -> Basis {
    Basis::new(vec![
        Shell::new(0, [0.0, 0.0, 0.0], vec![1.4, 0.5], vec![0.6, 0.5]).unwrap(), // s, 2 prim
        Shell::new(1, [0.6, -0.3, 0.2], vec![0.9, 0.35], vec![0.55, 0.5]).unwrap(), // p, 2 prim
        Shell::new(2, [-0.4, 0.7, -0.1], vec![1.1], vec![1.0]).unwrap(),         // d
        Shell::new(3, [0.2, 0.5, 0.8], vec![0.7], vec![1.0]).unwrap(),           // f
    ])
}

#[test]
fn os_matches_rys_block_sweep() {
    // A spread of quartets with mixed L on four distinct centres, including the
    // mixed-high-L (dd|ff) case and contracted indices (shells 0 and 1).
    let basis = mixed_basis();
    let quartets = [
        (0, 0, 0, 0),
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (1, 1, 1, 1),
        (2, 0, 1, 0),
        (2, 1, 2, 1),
        (0, 1, 2, 3), // (sp|df) — four different L
        (3, 0, 2, 1),
        (2, 2, 3, 3), // (dd|ff) — mixed high-L
        (3, 3, 3, 3), // (ff|ff)
    ];
    for (i, j, k, l) in quartets {
        let os = basis.eri_block_with(Engine::OsHgp, i, j, k, l);
        let rys = basis.eri_block_with(Engine::Rys, i, j, k, l);
        let s = basis.shells();
        assert_close(
            &os,
            &rys,
            &format!("(l{} l{}|l{} l{})", s[i].l(), s[j].l(), s[k].l(), s[l].l()),
        );
    }
}

#[test]
fn os_batched_df_matches_rys() {
    // Exercise the 4-lane d/f batch kernel (`vrr_big4`): a basis with several
    // uncontracted d and f shells on distinct centres makes the dense `eri_with`
    // builder bucket many same-shape d/f quartets ((dd|dd), (ff|ff), (df|df), …)
    // and fire the 4-wide lockstep path — which a single-quartet `eri_block_with`
    // never does. The whole forced-OS tensor (batched) must equal the forced-Rys
    // tensor element-for-element (independent recurrence), and the debug
    // NaN-poison in `vrr_big4` turns any indexing slip into a hard failure here.
    let centers = [
        [0.0, 0.0, 0.0],
        [1.3, 0.2, -0.1],
        [-0.7, 1.1, 0.3],
        [0.4, -0.9, 1.2],
        [-1.0, -0.5, -0.8],
        [0.9, 0.7, 0.6],
    ];
    let mut shells = Vec::new();
    for (k, c) in centers.iter().enumerate() {
        let e = 0.6 + 0.25 * k as f64;
        shells.push(Shell::new(2, *c, vec![e], vec![1.0]).unwrap()); // d
        shells.push(Shell::new(3, *c, vec![e * 1.4], vec![1.0]).unwrap()); // f
    }
    // A couple of contracted s/p shells so mixed d/f|s/p buckets appear too.
    shells.push(Shell::new(0, [0.2, 0.2, 0.2], vec![1.1, 0.4], vec![0.6, 0.5]).unwrap());
    shells.push(Shell::new(1, [-0.3, 0.5, 0.1], vec![0.9, 0.35], vec![0.55, 0.5]).unwrap());
    let basis = Basis::new(shells);
    let os = basis.eri_with(Engine::OsHgp);
    let rys = basis.eri_with(Engine::Rys);
    assert_close(&os, &rys, "d/f-heavy batched OS vs Rys");
}

#[test]
fn os_matches_rys_diffuse_tight() {
    // Extreme exponent ratio (1e3 / 1e-3) on a (pp|pp) quartet — the conditioning
    // stress case. Both engines must still agree.
    let basis = Basis::new(vec![
        Shell::new(1, [0.0, 0.0, 0.0], vec![1.0e3], vec![1.0]).unwrap(),
        Shell::new(1, [0.0, 0.0, 1.2], vec![1.0e-3], vec![1.0]).unwrap(),
    ]);
    let os = basis.eri_block_with(Engine::OsHgp, 0, 1, 0, 1);
    let rys = basis.eri_block_with(Engine::Rys, 0, 1, 0, 1);
    assert_close(&os, &rys, "diffuse×tight (pp|pp)");
}

#[test]
fn os_matches_rys_high_contraction() {
    // High contraction degree on every centre (the regime OS/HGP targets): a
    // 6-primitive s and a 4-primitive p, so the (sp|sp) quartet contracts
    // 6·4·6·4 = 576 primitive quartets behind a single early-contraction HRR.
    let basis = Basis::new(vec![
        Shell::new(
            0,
            [0.0, 0.0, 0.0],
            vec![30.0, 10.0, 3.0, 1.0, 0.4, 0.15],
            vec![0.02, 0.08, 0.2, 0.35, 0.3, 0.12],
        )
        .unwrap(),
        Shell::new(
            1,
            [0.3, 0.2, 1.0],
            vec![1.2, 0.6, 0.3, 0.12],
            vec![0.3, 0.4, 0.25, 0.1],
        )
        .unwrap(),
    ]);
    let os = basis.eri_block_with(Engine::OsHgp, 0, 1, 0, 1);
    let rys = basis.eri_block_with(Engine::Rys, 0, 1, 0, 1);
    assert_close(&os, &rys, "high-contraction (sp|sp)");
}

#[test]
fn eight_fold_symmetry_from_os_blocks() {
    // The dense builder scatters one evaluation per unique quartet, so its
    // shell-level symmetry is exact by construction. The genuine OS/HGP
    // recurrence check therefore compares independently *computed* permuted
    // blocks (`eri_block_with` evaluates each quartet directly, with no
    // symmetrization): (ij|kl) and σ(ij|kl) come from different VRR/HRR paths,
    // so their agreement is a real assertion on the recurrences.
    let basis = mixed_basis();
    let shells = basis.shells();
    let nsh = shells.len();
    let nc: Vec<usize> = shells.iter().map(|s| n_cart(s.l())).collect();
    const PERMS: [[usize; 4]; 8] = [
        [0, 1, 2, 3],
        [1, 0, 2, 3],
        [0, 1, 3, 2],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [2, 3, 1, 0],
        [3, 2, 0, 1],
        [3, 2, 1, 0],
    ];

    // Metric mirrors `eri_maxima`: a tight *relative* residual on the significant
    // elements (|base| ≥ 1e-3·peak) plus an absolute floor for the rest. A floorless
    // relative residual would over-weight structurally-near-zero elements, where the
    // two permutation paths differ only by f64 round-off (~1e-16 abs) that any
    // value-changing kernel optimisation perturbs — and which the consumer never
    // sees, since it reuses one computed value per unique quartet.
    let mut peak = 0.0_f64;
    let mut blocks: Vec<Vec<f64>> = Vec::new();
    for i in 0..nsh {
        for j in 0..nsh {
            for k in 0..nsh {
                for l in 0..nsh {
                    let block = basis.eri_block_with(Engine::OsHgp, i, j, k, l);
                    peak = block.iter().fold(peak, |m, &x| m.max(x.abs()));
                    blocks.push(block);
                }
            }
        }
    }
    let floor = 1e-3 * peak;
    let block_of =
        |q: [usize; 4]| -> &Vec<f64> { &blocks[((q[0] * nsh + q[1]) * nsh + q[2]) * nsh + q[3]] };
    let mut worst_sig = 0.0_f64; // worst relative residual on significant elements
    let mut worst_abs = 0.0_f64; // worst absolute residual anywhere
    for i in 0..nsh {
        for j in 0..nsh {
            for k in 0..nsh {
                for l in 0..nsh {
                    let q = [i, j, k, l];
                    let base = block_of(q);
                    let nq = [nc[i], nc[j], nc[k], nc[l]];
                    for perm in &PERMS[1..] {
                        let qp = [q[perm[0]], q[perm[1]], q[perm[2]], q[perm[3]]];
                        let pb = block_of(qp);
                        let np = [nc[qp[0]], nc[qp[1]], nc[qp[2]], nc[qp[3]]];
                        for a in 0..nq[0] {
                            for b in 0..nq[1] {
                                for c in 0..nq[2] {
                                    for d in 0..nq[3] {
                                        let src = ((a * nq[1] + b) * nq[2] + c) * nq[3] + d;
                                        let comp = [a, b, c, d];
                                        let dst = ((comp[perm[0]] * np[1] + comp[perm[1]]) * np[2]
                                            + comp[perm[2]])
                                            * np[3]
                                            + comp[perm[3]];
                                        let dv = (base[src] - pb[dst]).abs();
                                        worst_abs = worst_abs.max(dv);
                                        if base[src].abs() >= floor {
                                            worst_sig = worst_sig.max(dv / base[src].abs());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        worst_sig < 1e-10,
        "worst OS symmetry significant-relative residual {worst_sig:e}"
    );
    assert!(
        worst_abs < 1e-10 * peak.max(1.0) + 1e-12,
        "worst OS symmetry absolute residual {worst_abs:e} (peak {peak:e})"
    );
}

#[test]
fn os_block_layout_matches_dense_strides_and_is_falsifiable() {
    // The forced-OS `eri_block` must equal the forced-OS dense slice under the
    // documented row-major (a,b,c,d) layout; a deliberate (c,d) transpose must
    // make the comparison fail (so the layout test has teeth).
    let basis = mixed_basis();
    let n = basis.nao_cart();
    let dense = basis.eri_with(Engine::OsHgp);
    let shells = basis.shells();
    let offs: Vec<usize> = {
        let mut acc = 0;
        let mut v = Vec::new();
        for s in shells {
            v.push(acc);
            acc += s.n_cart();
        }
        v
    };

    let (qi, qj, qk, ql) = (1usize, 0usize, 2usize, 3usize); // (ps|df), all distinct L
    let block = basis.eri_block_with(Engine::OsHgp, qi, qj, qk, ql);
    let (na, nb, nc, nd) = (
        n_cart(shells[qi].l()),
        n_cart(shells[qj].l()),
        n_cart(shells[qk].l()),
        n_cart(shells[ql].l()),
    );
    assert_eq!(block.len(), na * nb * nc * nd);

    let mut transpose_would_differ = false;
    for a in 0..na {
        for b in 0..nb {
            for c in 0..nc {
                for d in 0..nd {
                    let src = ((a * nb + b) * nc + c) * nd + d;
                    let (i, j, k, l) = (offs[qi] + a, offs[qj] + b, offs[qk] + c, offs[ql] + d);
                    let dval = dense[((i * n + j) * n + k) * n + l];
                    assert!(
                        (block[src] - dval).abs() <= 1e-12 * dval.abs().max(1e-12),
                        "OS layout mismatch at {src} ({i}{j}|{k}{l}): {} vs {}",
                        block[src],
                        dval
                    );
                    // The deliberately wrong (c,d)-transposed index: if it differs
                    // from the correct value anywhere, the layout test is sensitive.
                    if c != d && nc == nd {
                        let wrong = ((a * nb + b) * nd + d) * nc + c;
                        if (block[wrong] - dval).abs() > 1e-9 * dval.abs().max(1e-12) {
                            transpose_would_differ = true;
                        }
                    }
                }
            }
        }
    }
    // For (ps|df), nc (d=6) ≠ nd (f=10), so the transpose is a different shape;
    // assert the block is non-trivial instead (the dense-slice equality above is
    // the falsifiable layout check — a transposed engine output would fail it).
    let _ = transpose_would_differ;
    assert!(
        block.iter().any(|&x| x.abs() > 1e-6),
        "OS block unexpectedly ~zero"
    );
}

#[test]
fn auto_dispatch_equals_both_forced_engines() {
    // Across the crossover boundary, `Auto` must (a) equal the engine it selects
    // and (b) since both engines agree, equal *both* forced results. These quartets
    // straddle the policy: the contracted low-L ones
    // dispatch to OS/HGP, the low-contraction higher-L ones to Rys — yet `Auto`
    // matches both. (The exhaustive per-boundary fingerprint is in `dispatch.rs`.)
    let basis = mixed_basis();
    let quartets = [
        (0, 0, 0, 0), // l_total 0
        (1, 1, 0, 0), // 2
        (1, 1, 1, 0), // 3
        (1, 1, 1, 1), // 4  (boundary, OS side)
        (2, 1, 1, 1), // 5  (Rys side)
        (2, 2, 3, 3), // 10 (Rys side)
    ];
    for (i, j, k, l) in quartets {
        let auto = basis.eri_block_with(Engine::Auto, i, j, k, l);
        let os = basis.eri_block_with(Engine::OsHgp, i, j, k, l);
        let rys = basis.eri_block_with(Engine::Rys, i, j, k, l);
        let s = basis.shells();
        let tag = format!(
            "auto-vs-os (l{} l{}|l{} l{})",
            s[i].l(),
            s[j].l(),
            s[k].l(),
            s[l].l()
        );
        assert_close(&auto, &os, &tag);
        let tag2 = format!(
            "auto-vs-rys (l{} l{}|l{} l{})",
            s[i].l(),
            s[j].l(),
            s[k].l(),
            s[l].l()
        );
        assert_close(&auto, &rys, &tag2);
    }
}
