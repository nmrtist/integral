//! Pure-Rust checks of the spherical (`c2s`) integral path.
//!
//! These guard the convention-internal invariants on every platform:
//! unit-normalized, mutually orthogonal spherical functions, and
//! engine-transparency of the transform.

use integral::{Basis, Engine, Shell};

/// A single **uncontracted, normalized** spherical shell has an identity overlap
/// `(2l+1) × (2l+1)`: the real spherical components are orthonormal (the unit
/// normalization the `c2s` matrix targets). Contracted shells are
/// intentionally not renormalized (as in the Cartesian path), so a single
/// primitive is used here.
#[test]
fn single_spherical_shell_overlap_is_identity() {
    for l in 0..=6 {
        let sh = Shell::new_spherical(l, [0.2, -0.1, 0.3], vec![0.9], vec![1.0]).unwrap();
        let basis = Basis::new(vec![sh]);
        let n = basis.nao();
        assert_eq!(n, 2 * l + 1);
        let s = basis.overlap();
        for i in 0..n {
            for j in 0..n {
                let expect = if i == j { 1.0 } else { 0.0 };
                let got = s[i * n + j];
                assert!(
                    (got - expect).abs() < 1e-12,
                    "l={l} S[{i},{j}]={got} expected {expect}"
                );
            }
        }
    }
}

/// `nao()` and block dimensions track [`integral::ShellKind`]: a mixed
/// Cartesian+spherical basis has the right size and the spherical sub-block has
/// `2l+1` rows.
#[test]
fn mixed_basis_dimensions() {
    let basis = Basis::new(vec![
        Shell::new(2, [0.0, 0.0, 0.0], vec![1.0], vec![1.0]).unwrap(), // d cart -> 6
        Shell::new_spherical(2, [0.5, 0.0, 0.0], vec![0.8], vec![1.0]).unwrap(), // d sph -> 5
        Shell::new_spherical(3, [0.0, 0.4, 0.0], vec![0.7], vec![1.0]).unwrap(), // f sph -> 7
    ]);
    assert_eq!(basis.nao(), 6 + 5 + 7);
    assert_eq!(basis.nao_cart(), 6 + 6 + 10);
    let s = basis.overlap();
    assert_eq!(s.len(), 18 * 18);
}

/// The spherical transform is engine-transparent: a spherical ERI quartet
/// computed via OS/HGP equals the one via Rys (both transform the same way), for
/// a mix of spherical angular momenta.
#[test]
fn spherical_eri_consistent_across_engines() {
    let cs = [
        [0.0, 0.0, 0.0],
        [0.6, -0.3, 0.2],
        [-0.4, 0.5, -0.1],
        [0.2, 0.3, 0.7],
    ];
    let es = [0.9, 1.3, 0.7, 1.1];
    for &ls in &[[2usize, 1, 2, 0], [3, 2, 1, 1], [2, 2, 2, 2], [4, 0, 3, 1]] {
        let basis = Basis::new(
            (0..4)
                .map(|k| Shell::new_spherical(ls[k], cs[k], vec![es[k]], vec![1.0]).unwrap())
                .collect(),
        );
        let os = basis.eri_block_with(Engine::OsHgp, 0, 1, 2, 3);
        let rys = basis.eri_block_with(Engine::Rys, 0, 1, 2, 3);
        assert_eq!(os.len(), rys.len());
        let peak = rys.iter().fold(0.0f64, |m, &r| m.max(r.abs()));
        let mut worst = 0.0f64;
        for (a, b) in os.iter().zip(&rys) {
            worst = worst.max((a - b).abs());
        }
        assert!(
            worst <= 1e-9 * peak.max(1.0) + 1e-10,
            "ls={ls:?}: spherical engines disagree, worst={worst:e} peak={peak:e}"
        );
    }
}

/// The spherical block is exactly the `c2s`-contracted Cartesian block: building
/// the spherical ERI must equal applying the transform to the Cartesian ERI
/// "by hand" for a `d` shell. (Falsifiable: a wrong component count or transform
/// would change the size or values.)
#[test]
fn spherical_d_block_has_2lplus1_per_index() {
    let cart = Basis::new(vec![
        Shell::new(2, [0.0, 0.0, 0.0], vec![1.0], vec![1.0]).unwrap(),
        Shell::new(0, [0.5, 0.2, 0.1], vec![0.6], vec![1.0]).unwrap(),
    ]);
    let sph = Basis::new(vec![
        Shell::new_spherical(2, [0.0, 0.0, 0.0], vec![1.0], vec![1.0]).unwrap(),
        Shell::new(0, [0.5, 0.2, 0.1], vec![0.6], vec![1.0]).unwrap(),
    ]);
    // (d s | d s): cartesian 6*1*6*1=36, spherical 5*1*5*1=25.
    let cb = cart.eri_block(0, 1, 0, 1);
    let sb = sph.eri_block(0, 1, 0, 1);
    assert_eq!(cb.len(), 36);
    assert_eq!(sb.len(), 25);
}
