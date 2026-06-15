//! Principle-based tests for the density-contracted gradient of erf-attenuated
//! ERIs ([`Basis::eri_grad_contract_kernel`]):
//!
//!   * **Finite difference** of the contracted attenuated "energy"
//!     `E(R) = Σ Γ·(μν|λσ)_ω` (built from the value path
//!     [`Basis::eri_kernel`], which is validated independently) vs the analytic
//!     contracted force, for ω ∈ {0.3, 1.0}, on a mixed s/p/d 3-atom basis.
//!   * **Translational invariance** `Σ_atoms F = 0` for attenuated kernels.
//!   * **ω → ∞ limit**: the attenuated force approaches the Coulomb force.
//!   * **Coulomb routing**: `EriKernel::Coulomb` is **bit-identical** to
//!     [`Basis::eri_grad_contract`] for several random densities.
//!   * Full gradient l-range to l = 5 (`#[ignore]`d; run in release).
//!   * The ω-validation and error contracts of [`Basis::eri_kernel`].

use integral::{Basis, EriKernel, IntegralError, Shell};

/// Same FD tolerance as the Coulomb gradient tests (`gradients.rs::FD_TOL`).
const FD_TOL: f64 = 1e-6;
const H: f64 = 2e-4;

fn shifted(c: [f64; 3], target: [f64; 3], d: [f64; 3]) -> [f64; 3] {
    if c == target {
        [c[0] + d[0], c[1] + d[1], c[2] + d[2]]
    } else {
        c
    }
}

/// Clone `basis` with every shell on atom `target` displaced by `d`.
fn shift_basis(basis: &Basis, target: [f64; 3], d: [f64; 3]) -> Basis {
    let shells = basis
        .shells()
        .iter()
        .map(|s| {
            Shell::with_kind(
                s.l(),
                shifted(s.center(), target, d),
                s.exponents().to_vec(),
                s.coefficients().to_vec(),
                s.kind(),
            )
            .unwrap()
        })
        .collect();
    Basis::new(shells)
}

fn unit(axis: usize, h: f64) -> [f64; 3] {
    let mut d = [0.0; 3];
    d[axis] = h;
    d
}

/// Mixed s/p/d basis on three distinct non-collinear centres (as in
/// `gradients.rs::eri_test_basis`).
fn eri_test_basis() -> Basis {
    let a0 = [0.0, 0.0, 0.0];
    let a1 = [0.8, -0.2, 0.3];
    let a2 = [-0.4, 0.6, 0.1];
    Basis::new(vec![
        Shell::new(1, a0, vec![1.6, 0.5], vec![0.5, 0.6]).unwrap(),
        Shell::new(2, a1, vec![0.9], vec![1.0]).unwrap(),
        Shell::new(0, a2, vec![0.7], vec![1.0]).unwrap(),
    ])
}

/// Deterministic pseudo-random two-particle density with the full 8-fold ERI
/// permutational symmetry (LCG from `seed`).
fn random_symmetric_gamma(nao: usize, seed: u64) -> Vec<f64> {
    let mut g = vec![0.0; nao.pow(4)];
    let mut state = seed;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let idx = |i: usize, j: usize, k: usize, l: usize| ((i * nao + j) * nao + k) * nao + l;
    for i in 0..nao {
        for j in 0..=i {
            for k in 0..nao {
                for l in 0..=k {
                    let v = next();
                    for (a, b, c, d) in [
                        (i, j, k, l),
                        (j, i, k, l),
                        (i, j, l, k),
                        (j, i, l, k),
                        (k, l, i, j),
                        (l, k, i, j),
                        (k, l, j, i),
                        (l, k, j, i),
                    ] {
                        g[idx(a, b, c, d)] = v;
                    }
                }
            }
        }
    }
    g
}

/// The contracted scalar `E = Σ Γ·(μν|λσ)_k` over the value path.
fn contracted_energy(basis: &Basis, gamma: &[f64], k: EriKernel) -> f64 {
    basis
        .eri_kernel(k)
        .iter()
        .zip(gamma)
        .map(|(e, g)| e * g)
        .sum()
}

/// FD-vs-analytic worst component over all atoms/axes for kernel `k`.
fn fd_worst(basis: &Basis, gamma: &[f64], k: EriKernel) -> f64 {
    let forces = basis.eri_grad_contract_kernel(gamma, k).unwrap();
    let mut worst = 0.0_f64;
    for (ai, &atom) in basis.atoms().iter().enumerate() {
        for (axis, &force) in forces[ai].iter().enumerate() {
            let ep = contracted_energy(&shift_basis(basis, atom, unit(axis, H)), gamma, k);
            let em = contracted_energy(&shift_basis(basis, atom, unit(axis, -H)), gamma, k);
            let fd = (ep - em) / (2.0 * H);
            worst = worst.max((force - fd).abs());
        }
    }
    worst
}

/// The analytic attenuated contracted force matches the central difference of
/// the contracted attenuated value integrals (themselves cross-checked against
/// an independent MD reference in `erf_kernel.rs`).
#[test]
fn erf_grad_contract_matches_finite_difference() {
    let basis = eri_test_basis();
    let gamma = random_symmetric_gamma(basis.nao(), 0x9E37_79B9_7F4A_7C15);
    for omega in [0.3, 1.0] {
        let worst = fd_worst(&basis, &gamma, EriKernel::Erf { omega });
        eprintln!("erf grad contract FD (omega = {omega}): worst {worst:.3e}");
        assert!(
            worst < FD_TOL,
            "erf grad contract vs FD (omega = {omega}): worst {worst:.3e}"
        );
    }
}

/// `Σ_atoms F = 0` per axis — a genuine cancellation between independently
/// computed per-centre derivatives (each O(1)).
#[test]
fn erf_grad_contract_translational_invariance() {
    let basis = eri_test_basis();
    let gamma = random_symmetric_gamma(basis.nao(), 0x243F_6A88_85A3_08D3);
    for omega in [0.3, 1.0] {
        let forces = basis
            .eri_grad_contract_kernel(&gamma, EriKernel::Erf { omega })
            .unwrap();
        for axis in 0..3 {
            let s: f64 = forces.iter().map(|f| f[axis]).sum();
            assert!(
                s.abs() < 1e-11,
                "erf contracted TI residual (omega {omega}, axis {axis}) = {s:.3e}"
            );
        }
    }
}

/// ω → ∞: `s = ω²/(ρ+ω²) → 1` and the attenuated force reproduces the Coulomb
/// force (relative to the largest Coulomb component).
#[test]
fn erf_grad_contract_large_omega_reproduces_coulomb() {
    let basis = eri_test_basis();
    let gamma = random_symmetric_gamma(basis.nao(), 0x1319_8A2E_0370_7344);
    let coul = basis.eri_grad_contract(&gamma).unwrap();
    let erf = basis
        .eri_grad_contract_kernel(&gamma, EriKernel::Erf { omega: 1e5 })
        .unwrap();
    let scale = coul.iter().flatten().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(scale > 0.1, "test not meaningful: forces ~ 0");
    let mut worst = 0.0_f64;
    for (fe, fc) in erf.iter().zip(&coul) {
        for axis in 0..3 {
            worst = worst.max((fe[axis] - fc[axis]).abs());
        }
    }
    eprintln!("omega -> infinity force deviation: {worst:.3e} (scale {scale:.3e})");
    assert!(
        worst <= 1e-7 * scale,
        "omega -> infinity limit: worst {worst:.3e} vs scale {scale:.3e}"
    );
}

/// `EriKernel::Coulomb` must route to `eri_grad_contract` **bit-identically**,
/// for several random densities.
#[test]
fn coulomb_kernel_routes_bit_identically() {
    let basis = eri_test_basis();
    for seed in [1_u64, 0xDEAD_BEEF, 0x0123_4567_89AB_CDEF] {
        let gamma = random_symmetric_gamma(basis.nao(), seed);
        let direct = basis.eri_grad_contract(&gamma).unwrap();
        let routed = basis
            .eri_grad_contract_kernel(&gamma, EriKernel::Coulomb)
            .unwrap();
        assert_eq!(direct.len(), routed.len());
        for (d, r) in direct.iter().zip(&routed) {
            for axis in 0..3 {
                assert!(
                    d[axis].to_bits() == r[axis].to_bits(),
                    "Coulomb routing not bit-identical (seed {seed:#x}): {} vs {}",
                    d[axis],
                    r[axis]
                );
            }
        }
    }
}

/// Full gradient l-range: differentiate an h shell (l = 5, raised to i = 6,
/// the engines' MAX_L) on three distinct centres — FD + TI. Slow in debug;
/// run with `--release -- --include-ignored`.
#[test]
#[ignore = "high-L corner; slow in debug — run in release with --include-ignored"]
fn erf_grad_contract_l5_corner() {
    let basis = Basis::new(vec![
        Shell::new(5, [0.0, 0.0, 0.0], vec![1.0], vec![1.0]).unwrap(),
        Shell::new(1, [0.8, -0.2, 0.3], vec![0.9], vec![1.0]).unwrap(),
        Shell::new(0, [-0.4, 0.6, 0.1], vec![0.7], vec![1.0]).unwrap(),
    ]);
    let gamma = random_symmetric_gamma(basis.nao(), 0xA409_3822_299F_31D0);
    let k = EriKernel::Erf { omega: 0.5 };
    let worst = fd_worst(&basis, &gamma, k);
    eprintln!("erf grad contract FD at l = 5: worst {worst:.3e}");
    assert!(worst < FD_TOL, "l = 5 erf grad FD: worst {worst:.3e}");
    let forces = basis.eri_grad_contract_kernel(&gamma, k).unwrap();
    for axis in 0..3 {
        let s: f64 = forces.iter().map(|f| f[axis]).sum();
        assert!(s.abs() < 1e-9, "l = 5 TI residual (axis {axis}) = {s:.3e}");
    }
}

/// The `l ≤ MAX_GRAD_L` contract is shared with the Coulomb path.
#[test]
fn erf_grad_contract_rejects_i_shell() {
    let basis = Basis::new(vec![
        Shell::new(6, [0.0, 0.0, 0.0], vec![1.0], vec![1.0]).unwrap()
    ]);
    let gamma = vec![0.0; basis.nao().pow(4)];
    assert_eq!(
        basis
            .eri_grad_contract_kernel(&gamma, EriKernel::Erf { omega: 0.5 })
            .unwrap_err(),
        IntegralError::AngularMomentumTooHighForGradient { l: 6, max: 5 }
    );
}

#[test]
fn erf_grad_contract_rejects_wrong_gamma_length() {
    let basis = eri_test_basis();
    let bad = vec![0.0; basis.nao().pow(4) - 1];
    assert!(matches!(
        basis.eri_grad_contract_kernel(&bad, EriKernel::Erf { omega: 0.5 }),
        Err(IntegralError::GammaLengthMismatch { .. })
    ));
}

#[test]
#[should_panic(expected = "finite omega > 0")]
fn erf_grad_contract_rejects_nonpositive_omega() {
    let basis = eri_test_basis();
    let gamma = vec![0.0; basis.nao().pow(4)];
    let _ = basis.eri_grad_contract_kernel(&gamma, EriKernel::Erf { omega: 0.0 });
}

#[test]
#[should_panic(expected = "finite omega > 0")]
fn erf_grad_contract_rejects_nan_omega() {
    let basis = eri_test_basis();
    let gamma = vec![0.0; basis.nao().pow(4)];
    let _ = basis.eri_grad_contract_kernel(&gamma, EriKernel::Erf { omega: f64::NAN });
}
