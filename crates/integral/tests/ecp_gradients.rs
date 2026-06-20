//! Validation of the analytic contracted ECP nuclear gradient
//! (`Basis::ecp_grad_contract`) by central finite differences of `Basis::ecp`
//! and by exact invariance laws (translational-invariance sum rule, rigid
//! rotation, single-atom degeneracy), plus error-path checks.

use integral::{Basis, Ecp, EcpPrimitive, IntegralError, Shell};

fn epr(n: i32, zeta: f64, coef: f64) -> EcpPrimitive {
    EcpPrimitive { n, zeta, coef }
}

/// Published def2-ECP (MWB28) parameters for Ag (same literal fixture as the
/// value tests in `ecp.rs`).
fn ag_def2_ecp(atom: usize) -> Ecp {
    Ecp {
        atom,
        n_core: 28,
        max_l: 3,
        local: vec![
            epr(2, 13.520_571, -21.097_695),
            epr(2, 6.576_015, -1.374_412),
        ],
        semilocal: vec![
            vec![
                epr(2, 13.130_309, 255.139_465),
                epr(2, 6.510_656, 36.866_806),
            ],
            vec![
                epr(2, 11.314_031, 182.176_810),
                epr(2, 5.691_475, 30.356_869),
            ],
            vec![epr(2, 9.211_745, 73.593_431), epr(2, 4.529_844, 12.702_387)],
        ],
    }
}

/// A fixed, deterministic *symmetric* density matrix.
fn symmetric_gamma(n: usize) -> Vec<f64> {
    let mut g = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let (a, b) = (i.min(j) as f64, i.max(j) as f64);
            g[i * n + j] = 0.7 / (1.0 + (b - a).powi(2)) + 0.05 * ((a + 1.0) * (b + 1.0)).sqrt();
        }
    }
    g
}

/// `E = Σ_{μν} γ_{μν} ⟨μ|U|ν⟩`.
fn energy(basis: &Basis, ecps: &[Ecp], gamma: &[f64]) -> f64 {
    basis.ecp(ecps).iter().zip(gamma).map(|(u, g)| u * g).sum()
}

/// Central finite-difference gradient (step `h`) of `energy` with respect to
/// each atom of the geometry, with the basis rebuilt at each displaced
/// geometry by `build`.
fn fd_grad(
    build: &dyn Fn(&[[f64; 3]]) -> Basis,
    atoms0: &[[f64; 3]],
    ecps: &[Ecp],
    gamma: &[f64],
    h: f64,
) -> Vec<[f64; 3]> {
    let mut out = vec![[0.0; 3]; atoms0.len()];
    for (k, slot) in out.iter_mut().enumerate() {
        for dir in 0..3 {
            let mut plus = atoms0.to_vec();
            plus[k][dir] += h;
            let mut minus = atoms0.to_vec();
            minus[k][dir] -= h;
            slot[dir] = (energy(&build(&plus), ecps, gamma) - energy(&build(&minus), ecps, gamma))
                / (2.0 * h);
        }
    }
    out
}

fn assert_close(got: &[[f64; 3]], want: &[[f64; 3]], tol: f64, label: &str) {
    for (a, (g, w)) in got.iter().zip(want).enumerate() {
        for d in 0..3 {
            assert!(
                g[d].is_finite() && (g[d] - w[d]).abs() <= tol,
                "{label}: atom {a} dir {d}: analytic {} vs reference {} (diff {:e})",
                g[d],
                w[d],
                (g[d] - w[d]).abs()
            );
        }
    }
}

/// Σ_a dE/dR_a = 0 (translational invariance of the total gradient).
fn assert_sum_rule(grad: &[[f64; 3]], tol: f64, label: &str) {
    for d in 0..3 {
        let s: f64 = grad.iter().map(|g| g[d]).sum();
        assert!(s.abs() <= tol, "{label}: Σ_a grad[{d}] = {s:e}");
    }
}

// ---------------------------------------------------------------------------
// Finite-difference validation
// ---------------------------------------------------------------------------

/// Synthetic 2-atom system (contracted s on the ECP atom, p off-center): the
/// analytic gradient matches central finite differences to ≤ 1e-8 per
/// component, and the sum rule holds to 1e-12.
#[test]
fn synthetic_two_atom_matches_finite_differences() {
    let atoms = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.3]];
    let build = |a: &[[f64; 3]]| {
        Basis::new(vec![
            Shell::new(0, a[0], vec![0.8, 2.2], vec![0.6, 0.5]).unwrap(),
            Shell::new(1, a[1], vec![1.1], vec![1.0]).unwrap(),
        ])
    };
    let ecps = [Ecp {
        atom: 0,
        n_core: 2,
        max_l: 1,
        local: vec![epr(2, 1.5, -2.0), epr(0, 2.5, 1.0)],
        semilocal: vec![vec![epr(2, 1.0, 0.7)]],
    }];
    let basis = build(&atoms);
    let gamma = symmetric_gamma(basis.nao());
    let grad = basis.ecp_grad_contract(&ecps, &gamma).unwrap();
    let fd = fd_grad(&|a| build(a), &atoms, &ecps, &gamma, 1e-5);
    assert_close(&grad, &fd, 1e-8, "synthetic 2-atom");
    assert_sum_rule(&grad, 1e-12, "synthetic 2-atom");
}

/// Realistic def2-ECP Ag fixture: s/p/d/f shells on the ECP center plus
/// contracted s and p shells off-center. Analytic vs FD ≤ 1e-8 per component
/// (the dominant magnitudes here are O(10)); sum rule to 1e-12 (relative to
/// the O(10²) summands).
#[test]
fn def2_ecp_ag_fixture_matches_finite_differences() {
    let atoms = [[0.4, -0.3, 0.2], [0.4, 0.5, 1.6]];
    let build = |a: &[[f64; 3]]| {
        Basis::new(vec![
            Shell::new(0, a[0], vec![1.2], vec![1.0]).unwrap(),
            Shell::new(1, a[0], vec![0.9], vec![1.0]).unwrap(),
            Shell::new(2, a[0], vec![1.5], vec![1.0]).unwrap(),
            Shell::new(3, a[0], vec![1.1], vec![1.0]).unwrap(),
            Shell::new(0, a[1], vec![1.3, 0.4], vec![0.4, 0.7]).unwrap(),
            Shell::new(1, a[1], vec![0.7], vec![1.0]).unwrap(),
        ])
    };
    let ecps = [ag_def2_ecp(0)];
    let basis = build(&atoms);
    let gamma = symmetric_gamma(basis.nao());
    let grad = basis.ecp_grad_contract(&ecps, &gamma).unwrap();
    let fd = fd_grad(&|a| build(a), &atoms, &ecps, &gamma, 1e-5);
    assert_close(&grad, &fd, 1e-8, "Ag def2-ECP");
    let scale: f64 = grad
        .iter()
        .flatten()
        .fold(0.0_f64, |m, &x| m.max(x.abs()))
        .max(1.0);
    assert_sum_rule(&grad, 1e-12 * scale, "Ag def2-ECP");
}

/// f and g shells off the ECP center (the highest supported AO momenta) and a
/// d-projector channel: FD agreement ≤ 1e-8.
#[test]
fn f_and_g_shells_match_finite_differences() {
    let atoms = [[0.1, -0.2, 0.3], [0.5, 0.0, 1.0]];
    let build = |a: &[[f64; 3]]| {
        Basis::new(vec![
            Shell::new(3, a[0], vec![0.9], vec![1.0]).unwrap(),
            Shell::new(4, a[1], vec![1.1], vec![1.0]).unwrap(),
        ])
    };
    let ecps = [Ecp {
        atom: 0,
        n_core: 0,
        max_l: 3,
        local: vec![epr(2, 1.2, -1.5)],
        semilocal: vec![Vec::new(), Vec::new(), vec![epr(2, 0.9, 0.8)]],
    }];
    let basis = build(&atoms);
    let gamma = symmetric_gamma(basis.nao());
    let grad = basis.ecp_grad_contract(&ecps, &gamma).unwrap();
    let fd = fd_grad(&|a| build(a), &atoms, &ecps, &gamma, 1e-5);
    assert_close(&grad, &fd, 1e-8, "f×g");
    assert_sum_rule(&grad, 1e-12, "f×g");
}

/// h (l=5) shell off the ECP center with a d-projector channel — the
/// heavy-element QZ case (5th–6th-row QZ basis sets carry h polarization). The
/// center derivative raises the h shell to i (l=6) = MAX_L, exercising the
/// gradient-cap raise from g to h. FD agreement ≤ 1e-7 (the slightly looser floor
/// reflects the larger third derivative of an h shell at the 1e-5 step).
#[test]
fn h_shell_off_center_matches_finite_differences() {
    let atoms = [[0.1, -0.2, 0.3], [0.5, 0.0, 1.0]];
    let build = |a: &[[f64; 3]]| {
        Basis::new(vec![
            Shell::new(2, a[0], vec![1.3], vec![1.0]).unwrap(), // d on the ECP atom
            Shell::new(5, a[1], vec![1.1], vec![1.0]).unwrap(), // h off-center
        ])
    };
    let ecps = [Ecp {
        atom: 0,
        n_core: 0,
        max_l: 3,
        local: vec![epr(2, 1.2, -1.5)],
        semilocal: vec![Vec::new(), Vec::new(), vec![epr(2, 0.9, 0.8)]],
    }];
    let basis = build(&atoms);
    let gamma = symmetric_gamma(basis.nao());
    let grad = basis.ecp_grad_contract(&ecps, &gamma).unwrap();
    let fd = fd_grad(&|a| build(a), &atoms, &ecps, &gamma, 1e-5);
    assert_close(&grad, &fd, 1e-7, "h off-center");
    assert_sum_rule(&grad, 1e-11, "h off-center");
}

/// Spherical shells go through the same `c2s` transform as the values: FD
/// agreement ≤ 1e-8 on a spherical-d / Cartesian-s mix, ECP on a third atom.
#[test]
fn spherical_shells_match_finite_differences() {
    let atoms = [[0.2, 0.1, -0.3], [0.9, 0.4, -0.8], [-0.5, 0.7, 0.6]];
    let build = |a: &[[f64; 3]]| {
        Basis::new(vec![
            Shell::new_spherical(2, a[0], vec![0.8], vec![1.0]).unwrap(),
            Shell::new(0, a[1], vec![1.0], vec![1.0]).unwrap(),
            Shell::new_spherical(3, a[2], vec![1.2], vec![1.0]).unwrap(),
        ])
    };
    let ecps = [Ecp {
        atom: 2,
        n_core: 0,
        max_l: 2,
        local: vec![epr(2, 1.4, -3.0)],
        semilocal: vec![vec![epr(2, 1.1, 2.0)], vec![epr(2, 0.8, 0.9)]],
    }];
    let basis = build(&atoms);
    let gamma = symmetric_gamma(basis.nao());
    let grad = basis.ecp_grad_contract(&ecps, &gamma).unwrap();
    let fd = fd_grad(&|a| build(a), &atoms, &ecps, &gamma, 1e-5);
    assert_close(&grad, &fd, 1e-8, "spherical");
    assert_sum_rule(&grad, 1e-12, "spherical");
}

/// Several ECPs at once (one per atom of a 2-atom system): the gradient is the
/// sum of the single-ECP gradients (bitwise-level additivity is not required;
/// 1e-13 numerical agreement is).
#[test]
fn additivity_over_ecps() {
    let atoms = [[0.0, 0.0, 0.0], [0.0, 0.3, 1.4]];
    let basis = Basis::new(vec![
        Shell::new(0, atoms[0], vec![0.9], vec![1.0]).unwrap(),
        Shell::new(1, atoms[1], vec![1.2], vec![1.0]).unwrap(),
    ]);
    let e0 = Ecp {
        atom: 0,
        n_core: 0,
        max_l: 1,
        local: vec![epr(2, 1.5, -2.0)],
        semilocal: vec![vec![epr(2, 1.0, 0.7)]],
    };
    let e1 = Ecp {
        atom: 1,
        n_core: 0,
        max_l: 2,
        local: vec![epr(2, 1.2, -1.0)],
        semilocal: vec![vec![epr(2, 0.9, 0.5)], vec![epr(2, 1.3, 0.4)]],
    };
    let gamma = symmetric_gamma(basis.nao());
    let both = basis
        .ecp_grad_contract(&[e0.clone(), e1.clone()], &gamma)
        .unwrap();
    let g0 = basis
        .ecp_grad_contract(std::slice::from_ref(&e0), &gamma)
        .unwrap();
    let g1 = basis
        .ecp_grad_contract(std::slice::from_ref(&e1), &gamma)
        .unwrap();
    for a in 0..2 {
        for d in 0..3 {
            assert!(
                (both[a][d] - (g0[a][d] + g1[a][d])).abs() <= 1e-13,
                "additivity at atom {a} dir {d}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Exact invariance laws
// ---------------------------------------------------------------------------

/// Rigid 90° rotation about z, `(x, y, z) → (−y, x, z)` (exactly representable
/// in floating point): the gradient of the rotated s-shell system is the
/// rotated gradient, to 1e-12.
#[test]
fn gradient_rotates_with_a_rigid_rotation() {
    let atoms: [[f64; 3]; 3] = [[0.1, -0.4, 0.3], [0.8, 0.6, -0.2], [-0.5, 0.2, 0.9]];
    let rot = |p: [f64; 3]| [-p[1], p[0], p[2]];
    let build = |a: &[[f64; 3]]| {
        Basis::new(vec![
            Shell::new(0, a[0], vec![0.9, 2.1], vec![0.5, 0.6]).unwrap(),
            Shell::new(0, a[1], vec![1.3], vec![1.0]).unwrap(),
            Shell::new(0, a[2], vec![0.7], vec![1.0]).unwrap(),
        ])
    };
    let ecps = [Ecp {
        atom: 1,
        n_core: 0,
        max_l: 1,
        local: vec![epr(2, 1.4, -2.5)],
        semilocal: vec![vec![epr(2, 1.0, 0.9)]],
    }];
    let basis = build(&atoms);
    let gamma = symmetric_gamma(basis.nao());
    let grad = basis.ecp_grad_contract(&ecps, &gamma).unwrap();
    let rotated: Vec<[f64; 3]> = atoms.iter().map(|&p| rot(p)).collect();
    let grad_rot = build(&rotated).ecp_grad_contract(&ecps, &gamma).unwrap();
    for a in 0..3 {
        let want = rot(grad[a]);
        for d in 0..3 {
            assert!(
                (grad_rot[a][d] - want[d]).abs() <= 1e-12,
                "rotation law: atom {a} dir {d}: {} vs {}",
                grad_rot[a][d],
                want[d]
            );
        }
    }
    // Translational invariance for the same system.
    assert_sum_rule(&grad, 1e-12, "rotation system");
}

/// All shells on the ECP atom (single-atom system): translational invariance
/// forces an exactly zero gradient — and the degenerate `k → 0` geometry must
/// produce no NaN.
#[test]
fn single_atom_gradient_is_exactly_zero() {
    let c = [0.3, -0.1, 0.7];
    let basis = Basis::new(vec![
        Shell::new(0, c, vec![1.2], vec![1.0]).unwrap(),
        Shell::new(1, c, vec![0.9], vec![1.0]).unwrap(),
        Shell::new(2, c, vec![1.5], vec![1.0]).unwrap(),
        Shell::new(3, c, vec![1.1], vec![1.0]).unwrap(),
    ]);
    let ecps = [ag_def2_ecp(0)];
    let gamma = symmetric_gamma(basis.nao());
    let grad = basis.ecp_grad_contract(&ecps, &gamma).unwrap();
    for g in grad.iter().flatten() {
        assert!(g.is_finite());
        assert_eq!(*g, 0.0, "single-atom gradient must vanish exactly");
    }
}

/// A non-symmetric `gamma` contributes only through its symmetric part
/// (`U` is symmetric): `grad(γ) == grad((γ + γᵀ)/2)` to 1e-13.
#[test]
fn non_symmetric_gamma_uses_symmetric_part() {
    let atoms = [[0.0, 0.0, 0.0], [0.2, -0.4, 1.1]];
    let basis = Basis::new(vec![
        Shell::new(1, atoms[0], vec![0.8], vec![1.0]).unwrap(),
        Shell::new(0, atoms[1], vec![1.4], vec![1.0]).unwrap(),
    ]);
    let ecps = [Ecp {
        atom: 0,
        n_core: 0,
        max_l: 1,
        local: vec![epr(2, 1.1, -1.7)],
        semilocal: vec![vec![epr(2, 0.9, 0.8)]],
    }];
    let n = basis.nao();
    let mut gamma = vec![0.0; n * n];
    for (k, g) in gamma.iter_mut().enumerate() {
        *g = ((k as f64) * 0.37 + 0.11).sin(); // deterministic, non-symmetric
    }
    let mut sym = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            sym[i * n + j] = 0.5 * (gamma[i * n + j] + gamma[j * n + i]);
        }
    }
    let ga = basis.ecp_grad_contract(&ecps, &gamma).unwrap();
    let gs = basis.ecp_grad_contract(&ecps, &sym).unwrap();
    for a in 0..2 {
        for d in 0..3 {
            assert!((ga[a][d] - gs[a][d]).abs() <= 1e-13);
        }
    }
}

// ---------------------------------------------------------------------------
// Error / edge paths
// ---------------------------------------------------------------------------

/// Empty ECP list: a zero gradient of the right shape, no error.
#[test]
fn empty_ecp_list_gives_zero() {
    let basis = Basis::new(vec![
        Shell::new(0, [0.0; 3], vec![1.0], vec![1.0]).unwrap(),
        Shell::new(0, [0.0, 0.0, 1.0], vec![0.7], vec![1.0]).unwrap(),
    ]);
    let gamma = symmetric_gamma(basis.nao());
    let grad = basis.ecp_grad_contract(&[], &gamma).unwrap();
    assert_eq!(grad, vec![[0.0; 3]; 2]);
}

/// Wrong `gamma` length and over-limit angular momentum produce the documented
/// errors (mirroring `eri_grad_contract` semantics).
#[test]
fn error_paths() {
    let basis = Basis::new(vec![Shell::new(0, [0.0; 3], vec![1.0], vec![1.0]).unwrap()]);
    let ecps = [Ecp {
        atom: 0,
        n_core: 0,
        max_l: 1,
        local: vec![epr(2, 1.0, -1.0)],
        semilocal: vec![Vec::new()],
    }];
    assert_eq!(
        basis.ecp_grad_contract(&ecps, &[0.0, 0.0]),
        Err(IntegralError::GammaLengthMismatch {
            expected: 1,
            got: 2
        })
    );
    // i (l=6) is above the gradient cap (the derivative would raise it to l=7,
    // outside MAX_L); h (l=5) is now supported (see `h_shell_off_center_*`).
    let high = Basis::new(vec![Shell::new(6, [0.0; 3], vec![1.0], vec![1.0]).unwrap()]);
    let gamma = vec![0.0; high.nao() * high.nao()];
    assert_eq!(
        high.ecp_grad_contract(&ecps, &gamma),
        Err(IntegralError::AngularMomentumTooHighForGradient { l: 6, max: 5 })
    );
}
