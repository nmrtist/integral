//! Property and analytic tests for the one-electron integral drivers.
//!
//! These run on every platform with no external library.

use integral::{Basis, Shell};

/// A small mixed-angular-momentum basis: s+p on one center, s on a second,
/// d on a third. Exercises contraction and higher-l Cartesian indexing.
fn sample_basis(shift: [f64; 3]) -> Basis {
    let c = |p: [f64; 3]| [p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]];
    Basis::new(vec![
        // contracted s (3 primitives) at center A
        Shell::new(
            0,
            c([0.0, 0.0, 0.0]),
            vec![3.4, 0.9, 0.25],
            vec![0.15, 0.55, 0.45],
        )
        .unwrap(),
        // p (2 primitives) at center A
        Shell::new(1, c([0.0, 0.0, 0.0]), vec![1.2, 0.4], vec![0.6, 0.5]).unwrap(),
        // s at center B
        Shell::new(0, c([0.0, 0.0, 1.4]), vec![0.7], vec![1.0]).unwrap(),
        // d at center C
        Shell::new(2, c([0.8, -0.5, 0.3]), vec![0.55], vec![1.0]).unwrap(),
    ])
}

fn nuclei(shift: [f64; 3]) -> Vec<([f64; 3], f64)> {
    let c = |p: [f64; 3]| [p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]];
    vec![
        (c([0.0, 0.0, 0.0]), 8.0),
        (c([0.0, 0.0, 1.4]), 1.0),
        (c([0.8, -0.5, 0.3]), 6.0),
    ]
}

fn assert_symmetric(mat: &[f64], n: usize, name: &str) {
    for i in 0..n {
        for j in 0..n {
            let a = mat[i * n + j];
            let b = mat[j * n + i];
            assert!(
                (a - b).abs() < 1e-12 * a.abs().max(1.0),
                "{name} not symmetric at ({i},{j}): {a} vs {b}"
            );
        }
    }
}

#[test]
fn matrices_are_hermitian() {
    let basis = sample_basis([0.0, 0.0, 0.0]);
    let n = basis.nao_cart();
    assert_symmetric(&basis.overlap(), n, "overlap");
    assert_symmetric(&basis.kinetic(), n, "kinetic");
    assert_symmetric(&basis.nuclear(&nuclei([0.0, 0.0, 0.0])), n, "nuclear");
}

#[test]
fn translational_invariance() {
    let shift = [1.3, -2.1, 0.7];
    let (b0, b1) = (sample_basis([0.0; 3]), sample_basis(shift));
    let (n0, n1) = (nuclei([0.0; 3]), nuclei(shift));

    let cases: [(Vec<f64>, Vec<f64>, &str); 3] = [
        (b0.overlap(), b1.overlap(), "overlap"),
        (b0.kinetic(), b1.kinetic(), "kinetic"),
        (b0.nuclear(&n0), b1.nuclear(&n1), "nuclear"),
    ];
    for (m0, m1, name) in cases {
        assert_eq!(m0.len(), m1.len());
        for (k, (a, b)) in m0.iter().zip(m1.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-11 * a.abs().max(1.0),
                "{name} changed under translation at {k}: {a} vs {b}"
            );
        }
    }
}

#[test]
fn normalized_s_has_unit_self_overlap() {
    // Single-primitive s shells are unit-normalized by construction.
    for &alpha in &[0.1, 1.0, 7.5] {
        let basis = Basis::new(vec![Shell::new(
            0,
            [0.2, 0.0, -0.1],
            vec![alpha],
            vec![1.0],
        )
        .unwrap()]);
        let s = basis.overlap();
        assert!((s[0] - 1.0).abs() < 1e-12, "alpha={alpha}: {}", s[0]);
    }
}

#[test]
fn kinetic_diagonal_is_positive() {
    let basis = sample_basis([0.0; 3]);
    let n = basis.nao_cart();
    let t = basis.kinetic();
    for i in 0..n {
        assert!(
            t[i * n + i] > 0.0,
            "kinetic diagonal {i} not positive: {}",
            t[i * n + i]
        );
    }
}

#[test]
fn dipole_of_single_s_is_displacement() {
    // ⟨s|(r-O)|s⟩ for a normalized s at R equals (R-O), since ⟨s|s⟩ = 1.
    let r = [0.4, -0.3, 0.9];
    let o = [0.1, 0.2, -0.1];
    let basis = Basis::new(vec![Shell::new(0, r, vec![1.1], vec![1.0]).unwrap()]);
    let [dx, dy, dz] = basis.dipole(o);
    assert!((dx[0] - (r[0] - o[0])).abs() < 1e-12);
    assert!((dy[0] - (r[1] - o[1])).abs() < 1e-12);
    assert!((dz[0] - (r[2] - o[2])).abs() < 1e-12);
}

#[test]
fn dipole_matrices_are_symmetric() {
    let basis = sample_basis([0.0; 3]);
    let n = basis.nao_cart();
    let [dx, dy, dz] = basis.dipole([0.0, 0.0, 0.0]);
    assert_symmetric(&dx, n, "dipole_x");
    assert_symmetric(&dy, n, "dipole_y");
    assert_symmetric(&dz, n, "dipole_z");
}

/// A basis spanning the full supported angular-momentum range (s,p through
/// f,g,h,i = l=3..6) on four distinct centers given by `centers`. Mixing low and
/// high l on different centers makes every off-diagonal block — including the
/// f·i, g·h, … cross terms — non-trivial, so full-matrix symmetry is a genuine
/// transpose check across the whole l range (the builder fills the `(a,b)` and
/// `(b,a)` blocks independently; nothing is symmetrized).
fn high_l_basis(centers: &[[f64; 3]; 4]) -> Basis {
    Basis::new(vec![
        Shell::new(0, centers[0], vec![1.6, 0.5], vec![0.5, 0.6]).unwrap(), // s (contracted)
        Shell::new(1, centers[0], vec![0.9], vec![1.0]).unwrap(),           // p
        Shell::new(3, centers[1], vec![1.1], vec![1.0]).unwrap(),           // f
        Shell::new(4, centers[2], vec![0.95], vec![1.0]).unwrap(),          // g
        Shell::new(5, centers[1], vec![1.05], vec![1.0]).unwrap(),          // h
        Shell::new(6, centers[3], vec![1.2], vec![1.0]).unwrap(),           // i
    ])
}

#[test]
fn high_l_matrices_are_hermitian_across_geometries() {
    // Several genuinely distinct geometries (not mere translations): collapsed,
    // bent, near-linear, and spread out. Hermiticity must hold for every one.
    let geometries: [[[f64; 3]; 4]; 4] = [
        [
            [0.0, 0.0, 0.0],
            [1.3, 0.2, -0.4],
            [-0.6, 1.1, 0.5],
            [0.3, -0.9, 1.4],
        ],
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.8],
            [0.0, 0.0, 3.1],
            [0.0, 0.0, 4.5],
        ],
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.15, 0.05],
            [0.0, 0.25, 0.1],
            [0.15, 0.0, 0.2],
        ],
        [
            [-1.2, 0.4, 0.8],
            [1.5, -0.7, 0.3],
            [0.6, 1.9, -1.1],
            [-0.9, -1.4, 1.7],
        ],
    ];
    for (g, centers) in geometries.iter().enumerate() {
        let basis = high_l_basis(centers);
        let n = basis.nao_cart();
        // Charges on each center (varied magnitudes exercise the nuclear sum).
        let charges: Vec<([f64; 3], f64)> = vec![
            (centers[0], 8.0),
            (centers[1], 6.0),
            (centers[2], 1.0),
            (centers[3], 7.0),
        ];
        assert_symmetric(&basis.overlap(), n, &format!("overlap[geom {g}]"));
        assert_symmetric(&basis.kinetic(), n, &format!("kinetic[geom {g}]"));
        assert_symmetric(&basis.nuclear(&charges), n, &format!("nuclear[geom {g}]"));
        let [dx, dy, dz] = basis.dipole([0.0, 0.0, 0.0]);
        assert_symmetric(&dx, n, &format!("dipole_x[geom {g}]"));
        assert_symmetric(&dy, n, &format!("dipole_y[geom {g}]"));
        assert_symmetric(&dz, n, &format!("dipole_z[geom {g}]"));
    }
}

#[test]
fn high_l_translational_invariance() {
    // Translational invariance through l=6, independently from the fixed-basis
    // low-l check above.
    let base = [
        [0.0, 0.0, 0.0],
        [1.3, 0.2, -0.4],
        [-0.6, 1.1, 0.5],
        [0.3, -0.9, 1.4],
    ];
    let shift = [2.7, -1.1, 0.9];
    let shifted = base.map(|p| [p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]]);
    let (b0, b1) = (high_l_basis(&base), high_l_basis(&shifted));
    let ch0: Vec<([f64; 3], f64)> = base.iter().map(|&p| (p, 4.0)).collect();
    let ch1: Vec<([f64; 3], f64)> = shifted.iter().map(|&p| (p, 4.0)).collect();
    let cases: [(Vec<f64>, Vec<f64>, &str); 3] = [
        (b0.overlap(), b1.overlap(), "overlap"),
        (b0.kinetic(), b1.kinetic(), "kinetic"),
        (b0.nuclear(&ch0), b1.nuclear(&ch1), "nuclear"),
    ];
    for (m0, m1, name) in cases {
        assert_eq!(m0.len(), m1.len());
        for (k, (a, b)) in m0.iter().zip(m1.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-10 * a.abs().max(1.0),
                "{name} changed under translation at {k}: {a} vs {b}"
            );
        }
    }
}
