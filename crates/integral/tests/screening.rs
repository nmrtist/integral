//! Soundness of Cauchy–Schwarz (Häser–Ahlrichs) ERI screening.
//!
//! The bound `|(μν|λσ)| ≤ Q[i,j]·Q[k,l]` must be **genuine**: the screened build
//! must equal the unscreened one for every *retained* element, and every *dropped*
//! quartet must truly satisfy the bound, so screening introduces no error above
//! the threshold `τ`. These compare integral screened vs integral unscreened (no
//! external reference).

use integral::{Basis, Engine, Shell};

/// A spatially extended molecule where distant shell pairs have tiny Schwarz
/// bounds, so a meaningful fraction of quartets is skipped: four atoms strung out
/// along z with moderately tight `s`/`p` shells.
fn spread_molecule() -> Basis {
    let mut shells = Vec::new();
    for (a, z) in [0.0, 6.0, 12.0, 18.0].into_iter().enumerate() {
        let c = [0.1 * a as f64, 0.0, z];
        shells.push(Shell::new(0, c, vec![3.0, 0.8], vec![0.5, 0.5]).unwrap());
        shells.push(Shell::new(1, c, vec![1.2], vec![1.0]).unwrap());
    }
    Basis::new(shells)
}

/// Index helper into the dense row-major `nao⁴` tensor.
fn at(t: &[f64], nao: usize, i: usize, j: usize, k: usize, l: usize) -> f64 {
    t[((i * nao + j) * nao + k) * nao + l]
}

#[test]
fn schwarz_bound_holds_elementwise() {
    // The shell-pair bound Q must dominate every ERI element it covers.
    let basis = spread_molecule();
    let nao = basis.nao();
    let shells = basis.shells();
    let nsh = shells.len();
    let q = basis.schwarz_bounds();
    let unscreened = basis.eri();

    let mut offs = Vec::with_capacity(nsh);
    let mut acc = 0;
    for s in shells {
        offs.push(acc);
        acc += s.n_func();
    }

    let mut worst_ratio = 0.0_f64;
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
                                        &unscreened,
                                        nao,
                                        offs[i] + a,
                                        offs[j] + b,
                                        offs[k] + c,
                                        offs[l] + d,
                                    )
                                    .abs();
                                    // Bound must hold (allow tiny f64 slack).
                                    assert!(
                                        v <= bound + 1e-10 * bound.max(1.0) + 1e-12,
                                        "Schwarz bound violated: |{v}| > Q={bound} at \
                                         ({i}{a},{j}{b}|{k}{c},{l}{d})"
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
    // The bound is tight on the diagonal (μν|μν): worst ratio should approach 1.
    eprintln!("worst |element|/bound = {worst_ratio:.4}");
    assert!(worst_ratio <= 1.0 + 1e-9, "bound not an upper bound");
    assert!(worst_ratio > 0.5, "bound implausibly loose ({worst_ratio})");
}

#[test]
fn screened_equals_unscreened_for_retained_and_drops_only_below_tau() {
    let basis = spread_molecule();
    let tau = 1e-10;

    let unscreened = basis.eri();
    let (screened, stats) = basis.eri_screened(tau);

    assert_eq!(screened.len(), unscreened.len());

    // The error introduced by screening is exactly the dropped elements; every one
    // must be below τ (so retained elements are bit-identical, dropped ones tiny).
    let mut worst_dropped = 0.0_f64;
    let mut n_dropped_elems = 0usize;
    for (s, u) in screened.iter().zip(&unscreened) {
        if s == u {
            continue; // retained: identical (same engine, same quartet)
        }
        // Differs only because it was screened out -> screened value is 0.
        assert_eq!(*s, 0.0, "retained element changed value");
        worst_dropped = worst_dropped.max(u.abs());
        n_dropped_elems += 1;
    }
    eprintln!(
        "tau={tau:e}: skipped {}/{} quartets ({:.1}%), {} elements zeroed, \
         worst dropped |value|={worst_dropped:e}",
        stats.shell_quartets_skipped,
        stats.shell_quartets_total,
        100.0 * stats.skipped_fraction(),
        n_dropped_elems
    );

    // Soundness: no element above τ was ever dropped.
    assert!(
        worst_dropped <= tau,
        "screening dropped an element above τ: {worst_dropped:e} > {tau:e}"
    );
    // Screening must actually bite on this spread-out system.
    assert!(
        stats.skipped_fraction() > 0.2,
        "expected a meaningful skipped fraction, got {:.3}",
        stats.skipped_fraction()
    );
    // And it must retain a meaningful fraction too (not screen everything).
    assert!(stats.skipped_fraction() < 0.99);
}

#[test]
fn every_skipped_quartet_satisfies_the_bound() {
    // Directly assert the screening predicate is sound: for each skipped quartet,
    // Q·Q < τ AND every actual element of that block is ≤ Q·Q (< τ).
    let basis = spread_molecule();
    let nao = basis.nao();
    let shells = basis.shells();
    let nsh = shells.len();
    let tau = 1e-10;
    let q = basis.schwarz_bounds();
    let unscreened = basis.eri();

    let mut offs = Vec::with_capacity(nsh);
    let mut acc = 0;
    for s in shells {
        offs.push(acc);
        acc += s.n_func();
    }

    let mut checked = 0usize;
    for i in 0..nsh {
        for j in 0..nsh {
            for k in 0..nsh {
                for l in 0..nsh {
                    let bound = q[i * nsh + j] * q[k * nsh + l];
                    if bound >= tau {
                        continue; // not skipped
                    }
                    checked += 1;
                    for a in 0..shells[i].n_func() {
                        for b in 0..shells[j].n_func() {
                            for c in 0..shells[k].n_func() {
                                for d in 0..shells[l].n_func() {
                                    let v = at(
                                        &unscreened,
                                        nao,
                                        offs[i] + a,
                                        offs[j] + b,
                                        offs[k] + c,
                                        offs[l] + d,
                                    )
                                    .abs();
                                    assert!(
                                        v <= bound + 1e-12 && v < tau,
                                        "skipped quartet had element {v:e} (bound {bound:e}, τ {tau:e})"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(checked > 0, "no quartets were skipped — test is vacuous");
    eprintln!("verified {checked} skipped quartets all lie below τ");
}

#[test]
fn screening_is_engine_transparent() {
    // Screened build with each forced engine agrees with the other to tolerance.
    let basis = spread_molecule();
    let tau = 1e-10;
    let (os, sos) = basis.eri_screened_with(Engine::OsHgp, tau);
    let (rys, srys) = basis.eri_screened_with(Engine::Rys, tau);
    assert_eq!(sos.shell_quartets_skipped, srys.shell_quartets_skipped);
    let peak = rys.iter().fold(0.0f64, |m, &r| m.max(r.abs()));
    let mut worst = 0.0f64;
    for (a, b) in os.iter().zip(&rys) {
        worst = worst.max((a - b).abs());
    }
    assert!(
        worst <= 1e-9 * peak.max(1.0) + 1e-10,
        "engines disagree: {worst:e}"
    );
}
