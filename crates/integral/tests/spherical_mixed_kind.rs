//! Mixed-kind bases and the trivial s/p cases.
//!
//! For `l ≤ 1` the spherical count `2l+1` equals the Cartesian count `n_cart`, and
//! integral's `c2s` for `l = 1` is the identity in (x,y,z) order, so spherical and
//! Cartesian integrals must be element-wise identical. A mixed Cartesian+spherical
//! basis must size and index correctly, and the spherical sub-block must equal the
//! Cartesian block fed through the `c2s` transform by hand.

use integral::math::solid_harmonics::{c2s_matrix, monomial_to_raw_factor};
use integral::{Basis, Shell};

/// s and p spherical integrals equal their Cartesian counterparts exactly.
#[test]
fn s_and_p_spherical_equal_cartesian() {
    for l in [0usize, 1] {
        let cart = Basis::new(vec![
            Shell::new(l, [0.1, -0.2, 0.3], vec![0.9, 0.3], vec![0.6, 0.5]).unwrap(),
            Shell::new(l, [0.6, 0.4, -0.5], vec![1.2], vec![1.0]).unwrap(),
        ]);
        let sph = Basis::new(vec![
            Shell::new_spherical(l, [0.1, -0.2, 0.3], vec![0.9, 0.3], vec![0.6, 0.5]).unwrap(),
            Shell::new_spherical(l, [0.6, 0.4, -0.5], vec![1.2], vec![1.0]).unwrap(),
        ]);
        assert_eq!(cart.nao(), sph.nao(), "l={l}: nao differs");
        // One-electron operators identical element-wise.
        for (label, c, s) in [
            ("overlap", cart.overlap(), sph.overlap()),
            ("kinetic", cart.kinetic(), sph.kinetic()),
        ] {
            for (a, b) in c.iter().zip(&s) {
                assert!((a - b).abs() < 1e-13, "l={l} {label}: {a} vs {b}");
            }
        }
        // ERI tensor identical element-wise.
        let (ce, se) = (cart.eri(), sph.eri());
        let mut worst = 0.0f64;
        for (a, b) in ce.iter().zip(&se) {
            worst = worst.max((a - b).abs());
        }
        assert!(worst < 1e-12, "l={l} ERI differs by {worst:e}");
        eprintln!("l={l}: s/p spherical == cartesian, worst ERI Δ = {worst:e}");
    }
}

/// A mixed Cartesian+spherical basis sizes and indexes correctly, and the
/// spherical sub-block equals the hand-transformed Cartesian block.
#[test]
fn mixed_kind_indexing_and_values() {
    // Shell 0: s (cart=sph=1). Shell 1: d cartesian (6). Shell 2: d spherical (5).
    let s0 = Shell::new(0, [0.3, 0.1, -0.2], vec![0.7], vec![1.0]).unwrap();
    let d_cart = Shell::new(2, [0.0, 0.0, 0.0], vec![1.1], vec![1.0]).unwrap();
    let d_sph = Shell::new_spherical(2, [0.0, 0.0, 0.0], vec![1.1], vec![1.0]).unwrap();

    let basis = Basis::new(vec![s0.clone(), d_cart.clone(), d_sph.clone()]);
    // nao = 1 + 6 + 5 = 12; nao_cart = 1 + 6 + 6 = 13.
    assert_eq!(basis.nao(), 12);
    assert_eq!(basis.nao_cart(), 13);
    let n = basis.nao();
    let s = basis.overlap();
    assert_eq!(s.len(), n * n);

    // The spherical d self-overlap block (rows/cols 7..12) is the 5×5 identity
    // (single normalized primitive → unit-normalized spherical functions).
    for a in 0..5 {
        for b in 0..5 {
            let v = s[(7 + a) * n + (7 + b)];
            let e = if a == b { 1.0 } else { 0.0 };
            assert!((v - e).abs() < 1e-12, "d-sph block [{a},{b}]={v}");
        }
    }

    // Cross-check: the s–(d-sph) overlap block equals the s–(d-cart) overlap block
    // transformed by the per-index c2s matrix M = ratio·c2s_matrix(2) applied on
    // the d index. Build the s–d-cart block from a pure-Cartesian twin basis.
    let twin = Basis::new(vec![s0, d_cart]); // nao = 1 + 6 = 7
    let st = twin.overlap();
    // s(row 0) × d-cart(cols 1..7) block, shape 1×6 (row 0 of the 7×7 overlap).
    let cart_row: Vec<f64> = (0..6).map(|b| st[1 + b]).collect();
    // Apply M (5×6) to the 6-vector: m_sph[p] = Σ_b M[p,b] cart_row[b].
    let ratio = monomial_to_raw_factor(2);
    let cm = c2s_matrix(2);
    let mut hand = [0.0f64; 5];
    for p in 0..5 {
        for b in 0..6 {
            hand[p] += ratio * cm[p * 6 + b] * cart_row[b];
        }
    }
    // Compare to the actual s–(d-sph) block in the mixed basis: row 0, cols 7..12.
    for p in 0..5 {
        let got = s[7 + p];
        assert!(
            (got - hand[p]).abs() < 1e-12,
            "mixed s–dsph[{p}]: got {got}, hand-transformed {}",
            hand[p]
        );
    }
    eprintln!("mixed-kind indexing + hand-transform cross-check OK");
}
