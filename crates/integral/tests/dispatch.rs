//! Permanent guard: prove the force hooks and `Auto` dispatch route to the
//! *claimed* engine, not a silent fallback, and that `select_engine`'s thresholds
//! have no off-by-one. The two engines are independent f64 recurrences whose
//! round-off differs on near-cancellation tail elements, so their outputs are
//! NOT bit-identical — that lets us fingerprint which one actually ran.
//!
//! A permanent dispatch-routing guard.

use integral::{select_engine, Basis, Engine, Shell};

/// True iff every element is bit-for-bit equal.
fn bit_identical(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// A contracted shell with `k` primitives (so contraction_degree can be tuned).
fn shell(l: usize, c: [f64; 3], k: usize, base: f64) -> Shell {
    let exps: Vec<f64> = (0..k).map(|i| base * (1.7f64).powi(i as i32)).collect();
    let coeffs: Vec<f64> = (0..k).map(|i| 0.3 + 0.1 * i as f64).collect();
    Shell::new(l, c, exps, coeffs).unwrap()
}

#[test]
fn engines_are_not_bit_identical_so_fingerprinting_is_valid() {
    // Sanity: on a high-L quartet the two engines genuinely differ in round-off.
    let b = Basis::new(vec![
        shell(4, [0.0, 0.0, 0.0], 1, 0.9),
        shell(4, [0.5, -0.3, 0.2], 1, 1.3),
        shell(4, [-0.4, 0.6, -0.1], 1, 0.7),
        shell(4, [0.2, 0.4, 0.8], 1, 1.1),
    ]);
    let os = b.eri_block_with(Engine::OsHgp, 0, 1, 2, 3);
    let rys = b.eri_block_with(Engine::Rys, 0, 1, 2, 3);
    assert!(
        !bit_identical(&os, &rys),
        "engines bit-identical — fingerprint distinguisher would be invalid"
    );
}

#[test]
fn auto_routes_to_the_engine_select_engine_names() {
    // Build quartets straddling every documented threshold and confirm Auto's
    // output is bit-identical to the engine select_engine() names, and NOT to the
    // other. contraction_degree = product of n_prim across the four shells.
    let geo = [
        [0.0, 0.0, 0.0],
        [0.6, -0.3, 0.2],
        [-0.4, 0.5, -0.2],
        [0.2, 0.3, 0.7],
    ];
    // (l per shell, k per shell) chosen to straddle every recalibrated boundary.
    // ne = l0+l1, nf = l2+l3, deg = prod(k). With a fast kernel (max(ne,nf) ≤ 6)
    // the l_total 6–16 d/f band flips to OS at deg ≥ 3 (deg 1–2 → Rys); g-heavy
    // (max(ne,nf) ≥ 7) keeps deg ≥ 16; l_total ≥ 17 stays Rys.
    let cases: &[([usize; 4], [usize; 4])] = &[
        ([0, 0, 0, 0], [1, 1, 1, 1]), // lt0  deg1  -> OS  (always)
        ([2, 1, 1, 1], [1, 1, 1, 1]), // lt5  deg1  -> OS  (lt ≤ 5 band)
        ([2, 2, 2, 2], [1, 1, 1, 1]), // lt8  deg1  -> Rys (fast d/f kernel, deg < 3)
        ([2, 2, 2, 2], [2, 1, 1, 1]), // lt8  deg2  -> Rys (one below the deg-3 threshold)
        ([2, 2, 2, 2], [3, 1, 1, 1]), // lt8  deg3  -> OS  (at the new d/f threshold)
        ([3, 3, 3, 3], [1, 1, 1, 1]), // lt12 deg1  -> Rys (ff, deg < 3)
        ([3, 3, 3, 3], [3, 1, 1, 1]), // lt12 deg3  -> OS  (ff at threshold)
        ([4, 4, 4, 4], [1, 1, 1, 1]), // lt16 deg1  -> Rys (g-heavy: no fast kernel, deg < 16)
        ([4, 4, 4, 4], [2, 2, 2, 2]), // lt16 deg16 -> OS  (g-heavy at deg-16)
        ([4, 4, 4, 5], [2, 2, 2, 2]), // lt17 deg16 -> Rys (≥17: always Rys)
    ];
    for (ls, ks) in cases {
        let shells: Vec<Shell> = (0..4)
            .map(|i| shell(ls[i], geo[i], ks[i], 0.8 + 0.1 * i as f64))
            .collect();
        let b = Basis::new(shells);
        let lt: usize = ls.iter().sum();
        let deg: usize = ks.iter().product();
        let picked = select_engine(ls[0] + ls[1], ls[2] + ls[3], deg);
        let auto = b.eri_block_with(Engine::Auto, 0, 1, 2, 3);
        let os = b.eri_block_with(Engine::OsHgp, 0, 1, 2, 3);
        let rys = b.eri_block_with(Engine::Rys, 0, 1, 2, 3);
        let (same_as, other) = match picked {
            Engine::OsHgp => (&os, &rys),
            _ => (&rys, &os),
        };
        assert!(
            bit_identical(&auto, same_as),
            "lt={lt} deg={deg}: select={picked:?} but Auto not bit-identical to it"
        );
        // And Auto differs from the other engine (proves it didn't run the other).
        // Only assert when the two engines actually differ for this quartet.
        if !bit_identical(&os, &rys) {
            assert!(
                !bit_identical(&auto, other),
                "lt={lt} deg={deg}: Auto matches the NON-selected engine"
            );
        }
        // And both engines still agree to tolerance (engine transparency).
        let peak = rys.iter().fold(0.0f64, |m, &r| m.max(r.abs()));
        for (o, r) in os.iter().zip(&rys) {
            assert!(
                (o - r).abs() <= 1e-9 * peak.max(1.0) + 1e-10,
                "lt={lt} deg={deg}: engines disagree {o} vs {r}"
            );
        }
        eprintln!("lt={lt:2} deg={deg:4} -> {picked:?}  (auto fingerprint OK)");
    }
}

#[test]
fn select_engine_boundaries_exhaustive() {
    // Off-by-one check of every threshold edge. `select_engine(ne, nf, deg)`.
    use Engine::{OsHgp, Rys};
    // Band l_total ≤ 5: OS always (down to deg 1). One (ne,nf) per l_total 0..=5.
    for (ne, nf) in [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)] {
        assert_eq!(select_engine(ne, nf, 1), OsHgp, "lt={} deg1", ne + nf);
        assert_eq!(
            select_engine(ne, nf, usize::MAX),
            OsHgp,
            "lt={} highdeg",
            ne + nf
        );
    }
    // Band l_total 6–16 WITH a fast kernel (max(ne,nf) ≤ 6 ⇒ the measured d/f
    // sweep): OS once deg ≥ 3; deg 1–2 → Rys. Edge at deg 2/3.
    for (ne, nf) in [(4, 2), (4, 3), (4, 4), (5, 4), (6, 3), (6, 5), (6, 6)] {
        assert_eq!(select_engine(ne, nf, 1), Rys, "({ne},{nf}) deg1");
        assert_eq!(select_engine(ne, nf, 2), Rys, "({ne},{nf}) deg2");
        assert_eq!(select_engine(ne, nf, 3), OsHgp, "({ne},{nf}) deg3");
        assert_eq!(
            select_engine(ne, nf, usize::MAX),
            OsHgp,
            "({ne},{nf}) highdeg"
        );
    }
    // Band l_total 6–16 WITHOUT a fast kernel (max(ne,nf) ≥ 7, g-heavy): OS once
    // deg ≥ 16; Rys below. Edge at deg 15/16. (All these have l_total ≤ 16.)
    for (ne, nf) in [(7, 0), (7, 1), (8, 0), (7, 7), (8, 8)] {
        assert_eq!(select_engine(ne, nf, 1), Rys, "({ne},{nf}) deg1");
        assert_eq!(select_engine(ne, nf, 15), Rys, "({ne},{nf}) deg15");
        assert_eq!(select_engine(ne, nf, 16), OsHgp, "({ne},{nf}) deg16");
    }
    // Band l_total ≥ 17: always Rys, regardless of contraction.
    for (ne, nf) in [(9, 8), (10, 8), (12, 12)] {
        assert_eq!(select_engine(ne, nf, usize::MAX), Rys, "({ne},{nf}) highL");
    }
}
