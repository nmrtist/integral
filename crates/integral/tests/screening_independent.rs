//! Independent soundness of Schwarz screening.
//!
//! These are independent checks, distinct from the implementation's
//! `tests/screening.rs`: a LARGE, varied quartet sample (mixed L, exponents,
//! geometries, BOTH engines, Cartesian AND spherical) asserting the Cauchy–Schwarz
//! bound is a TRUE upper bound (ratio ≤ 1 everywhere); an independent recomputation
//! of `Q` from the full tensor diagonal; a DIFFERENT molecule than `screening.rs`;
//! a τ-sweep with the soundness contract; and τ = 0 ≡ unscreened.

use integral::{Basis, Engine, Shell, ShellKind};

/// (l, center, exponents, coefficients) for a shell.
type ShellSpec = (usize, [f64; 3], &'static [f64], &'static [f64]);

fn at(t: &[f64], nao: usize, i: usize, j: usize, k: usize, l: usize) -> f64 {
    t[((i * nao + j) * nao + k) * nao + l]
}

/// A varied, non-collinear basis: L = 0..3, several exponents, scattered centres.
/// `kind` selects Cartesian or spherical components for every shell.
fn varied_basis(kind: ShellKind) -> Basis {
    let specs: &[ShellSpec] = &[
        (0, [0.0, 0.0, 0.0], &[2.1, 0.6], &[0.5, 0.6]),
        (1, [1.3, -0.4, 0.2], &[0.9], &[1.0]),
        (2, [-0.6, 1.1, -0.3], &[1.4, 0.5], &[0.4, 0.7]),
        (0, [0.5, 0.7, 1.6], &[0.3], &[1.0]),
        (3, [-1.2, -0.5, 0.8], &[0.7], &[1.0]),
        (1, [0.2, 1.5, -1.1], &[1.8], &[1.0]),
    ];
    Basis::new(
        specs
            .iter()
            .map(|&(l, c, e, co)| Shell::with_kind(l, c, e.to_vec(), co.to_vec(), kind).unwrap())
            .collect(),
    )
}

/// C7 — the Cauchy–Schwarz bound must dominate EVERY element of EVERY quartet, for
/// both engines and both Cartesian and spherical bases. A ratio > 1 anywhere means
/// screening could drop a real integral (a Blocker), so the assertion is strict.
fn bound_is_true_upper_bound(kind: ShellKind, engine: Engine) {
    let basis = varied_basis(kind);
    let nao = basis.nao();
    let shells = basis.shells();
    let nsh = shells.len();
    let q = basis.schwarz_bounds_with(engine);
    let eri = basis.eri_with(engine);

    let mut offs = Vec::with_capacity(nsh);
    let mut acc = 0;
    for s in shells {
        offs.push(acc);
        acc += s.n_func();
    }

    let mut worst_ratio = 0.0_f64;
    let mut n_checked = 0_usize;
    for i in 0..nsh {
        for j in 0..nsh {
            for k in 0..nsh {
                for l in 0..nsh {
                    let bound = q[i * nsh + j] * q[k * nsh + l];
                    for a in 0..shells[i].n_func() {
                        for b in 0..shells[j].n_func() {
                            for c in 0..shells[k].n_func() {
                                for d in 0..shells[l].n_func() {
                                    let v = at(
                                        &eri,
                                        nao,
                                        offs[i] + a,
                                        offs[j] + b,
                                        offs[k] + c,
                                        offs[l] + d,
                                    )
                                    .abs();
                                    n_checked += 1;
                                    // True upper bound: a real violation is a Blocker.
                                    // Allow only f64 round-off slack.
                                    assert!(
                                        v <= bound * (1.0 + 1e-10) + 1e-13,
                                        "{kind:?}/{engine:?}: Schwarz bound VIOLATED at \
                                         ({i}{a},{j}{b}|{k}{c},{l}{d}): |{v:e}| > Q·Q={bound:e}"
                                    );
                                    if bound > 0.0 {
                                        worst_ratio = worst_ratio.max(v / bound);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    eprintln!(
        "{kind:?}/{engine:?}: checked {n_checked} elements, worst |element|/bound = {worst_ratio:.6}"
    );
    // Must be a genuine bound (≤ 1) and tight somewhere (the diagonal saturates).
    assert!(
        worst_ratio <= 1.0 + 1e-9,
        "ratio > 1: bound not an upper bound"
    );
    assert!(worst_ratio > 0.5, "bound implausibly loose ({worst_ratio})");
}

#[test]
fn schwarz_bound_true_upper_bound_all_combos() {
    for kind in [ShellKind::Cartesian, ShellKind::Spherical] {
        for engine in [Engine::Rys, Engine::OsHgp] {
            bound_is_true_upper_bound(kind, engine);
        }
    }
}

/// C8 — `Q[i,j]` must equal `sqrt(max_{μ∈i,ν∈j} |(μν|μν)|)` read from the *diagonal*
/// `(ij|ij)` shell block. Recompute it independently from the full ERI tensor's
/// diagonal and compare to `schwarz_bounds`.
#[test]
fn schwarz_q_is_the_diagonal_max() {
    for kind in [ShellKind::Cartesian, ShellKind::Spherical] {
        let basis = varied_basis(kind);
        let nao = basis.nao();
        let shells = basis.shells();
        let nsh = shells.len();
        let eri = basis.eri();
        let q = basis.schwarz_bounds();

        let mut offs = Vec::with_capacity(nsh);
        let mut acc = 0;
        for s in shells {
            offs.push(acc);
            acc += s.n_func();
        }
        for i in 0..nsh {
            for j in 0..nsh {
                // Independent: max over the diagonal (μν|μν) from the dense tensor.
                let mut mx = 0.0_f64;
                for a in 0..shells[i].n_func() {
                    for b in 0..shells[j].n_func() {
                        let mu = offs[i] + a;
                        let nu = offs[j] + b;
                        mx = mx.max(at(&eri, nao, mu, nu, mu, nu).abs());
                    }
                }
                let expect = mx.sqrt();
                let got = q[i * nsh + j];
                assert!(
                    (got - expect).abs() <= 1e-12 * expect.max(1.0),
                    "{kind:?} Q[{i},{j}]={got} != sqrt(diag max)={expect}"
                );
                // A self-repulsion (μν|μν) is non-negative, so Q is real & ≥ 0.
                assert!(got >= 0.0);
            }
        }
    }
}

/// A DIFFERENT molecule than the `screening.rs` z-chain: a 2-D, branched arrangement of
/// mixed s/p/d shells so distant pairs have tiny bounds and screening bites.
fn branched_molecule() -> Basis {
    let specs: &[ShellSpec] = &[
        (0, [0.0, 0.0, 0.0], &[1.5, 0.4], &[0.5, 0.5]),
        (1, [0.0, 0.0, 0.0], &[0.9], &[1.0]),
        (0, [7.0, 0.3, 0.0], &[1.2], &[1.0]),
        (2, [7.0, 0.3, 0.0], &[0.8], &[1.0]),
        (0, [0.2, 8.0, 0.0], &[1.0, 0.35], &[0.6, 0.5]),
        (1, [-6.5, -5.0, 0.4], &[0.7], &[1.0]),
    ];
    Basis::new(
        specs
            .iter()
            .map(|&(l, c, e, co)| Shell::new(l, c, e.to_vec(), co.to_vec()).unwrap())
            .collect(),
    )
}

/// C8 — on a molecule different from `screening.rs`: for a τ-sweep, every retained
/// element is bit-identical to the unscreened build and no element above τ is ever
/// dropped. Report the skipped fraction at each τ.
#[test]
fn screened_sound_across_tau_sweep_other_molecule() {
    let basis = branched_molecule();
    let unscreened = basis.eri();
    for &tau in &[1e-12_f64, 1e-10, 1e-8, 1e-6] {
        let (screened, stats) = basis.eri_screened(tau);
        assert_eq!(screened.len(), unscreened.len());
        let mut worst_dropped = 0.0_f64;
        let mut n_dropped = 0usize;
        for (s, u) in screened.iter().zip(&unscreened) {
            if s == u {
                continue;
            }
            assert_eq!(*s, 0.0, "retained element changed (τ={tau:e})");
            worst_dropped = worst_dropped.max(u.abs());
            n_dropped += 1;
        }
        eprintln!(
            "τ={tau:e}: skipped {:.1}% quartets, {n_dropped} elems zeroed, worst dropped |v|={worst_dropped:e}",
            100.0 * stats.skipped_fraction()
        );
        assert!(
            worst_dropped <= tau,
            "DROPPED AN ELEMENT ABOVE τ: {worst_dropped:e} > {tau:e}"
        );
    }
}

/// C8 — `τ = 0` must reproduce the unscreened tensor EXACTLY (bit-identical): no
/// quartet has `Q·Q < 0`, so nothing is skipped.
#[test]
fn tau_zero_reproduces_unscreened_exactly() {
    for engine in [Engine::Rys, Engine::OsHgp] {
        let basis = branched_molecule();
        let unscreened = basis.eri_with(engine);
        let (screened, stats) = basis.eri_screened_with(engine, 0.0);
        assert_eq!(stats.shell_quartets_skipped, 0, "τ=0 skipped a quartet");
        assert_eq!(screened, unscreened, "τ=0 differs from unscreened");
    }
}
