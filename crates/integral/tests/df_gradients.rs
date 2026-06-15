//! Density-fitting gradient validation — engine-independent, principle-based.
//!
//! Checks for `eri_3c_grad_contract` / `eri_2c_grad_contract`:
//!
//! 1. **Finite difference** vs the value builders (`Eri3cBuilder::build`,
//!    `eri_2c`): displace an atom by `±h` in BOTH the orbital and aux bases and
//!    central-difference the contracted scalar `Σ Γ·(value)`. Mixed s/p/d
//!    (incl. spherical) orbital basis + s/p/d/f aux basis, 3 atoms.
//! 2. **Translational invariance** `Σ_atoms F = 0` to ≤ 1e-11.
//! 3. **4-center cross-check** via the Gaussian product theorem: an aux `s`
//!    shell of exponent `α` equals (up to a geometry-independent constant) the
//!    product of two real `s` shells of exponent `α/2` on the same center, so
//!    `Σ Γ d(μν|P)` must match `eri_grad_contract` on the equivalent quartet
//!    construction — no dummy involved on the 4c side.
//! 4. Error paths: atom-list mismatch, wrong gamma length, l > MAX_GRAD_L.
//! 5. `#[ignore]`d high-L corners: differentiated shells at l = 5 (run in
//!    release with `--include-ignored`).

use integral::{Basis, IntegralError, Shell};

/// Finite-difference agreement tolerance (see `gradients.rs`).
const FD_TOL: f64 = 1e-6;
/// Translational-invariance tolerance (acceptance bar for the DF gradients).
const TI_TOL: f64 = 1e-11;
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

/// Deterministic pseudo-random coefficients in (−0.5, 0.5) — intentionally
/// **asymmetric** (no index symmetry is imposed; the APIs must not require it).
fn random_gamma(len: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        })
        .collect()
}

/// The contracted 3-center scalar `Σ_{μν,P} Γ_{μν,P} (μν|P)` from the dense
/// value tensor — the quantity the analytic gradient must differentiate.
fn e3(main: &Basis, aux: &Basis, gamma: &[f64]) -> f64 {
    main.eri_3c_builder(aux)
        .build()
        .iter()
        .zip(gamma)
        .map(|(v, g)| v * g)
        .sum()
}

/// The contracted 2-center scalar `Σ_{PQ} γ_{PQ} (P|Q)`.
fn e2(aux: &Basis, gamma: &[f64]) -> f64 {
    aux.eri_2c().iter().zip(gamma).map(|(v, g)| v * g).sum()
}

const A0: [f64; 3] = [0.0, 0.0, 0.0];
const A1: [f64; 3] = [0.8, -0.2, 0.3];
const A2: [f64; 3] = [-0.4, 0.6, 0.1];

/// Mixed s/p/d orbital basis (with one spherical shell) on three atoms.
fn orbital_basis() -> Basis {
    Basis::new(vec![
        Shell::new(0, A0, vec![1.6, 0.5], vec![0.5, 0.6]).unwrap(),
        Shell::new(1, A1, vec![0.9], vec![1.0]).unwrap(),
        Shell::new_spherical(2, A2, vec![0.7], vec![1.0]).unwrap(),
    ])
}

/// s/p/d/f auxiliary basis on the same three atoms (same first-appearance
/// order as `orbital_basis`).
fn aux_basis() -> Basis {
    Basis::new(vec![
        Shell::new(0, A0, vec![1.2], vec![1.0]).unwrap(),
        Shell::new(3, A0, vec![0.8], vec![1.0]).unwrap(),
        Shell::new(1, A1, vec![1.1], vec![1.0]).unwrap(),
        Shell::new_spherical(2, A2, vec![0.9], vec![1.0]).unwrap(),
    ])
}

#[test]
fn eri_3c_grad_contract_matches_finite_difference() {
    let main = orbital_basis();
    let aux = aux_basis();
    let gamma = random_gamma(main.nao() * main.nao() * aux.nao(), 0x9E37_79B9);
    let forces = main.eri_3c_grad_contract(&aux, &gamma).unwrap();
    assert_eq!(forces.len(), main.atoms().len());
    let mut worst = 0.0_f64;
    for (ai, &atom) in main.atoms().iter().enumerate() {
        for (axis, &f) in forces[ai].iter().enumerate() {
            // The atom displacement moves its shells in BOTH bases.
            let plus = e3(
                &shift_basis(&main, atom, unit(axis, H)),
                &shift_basis(&aux, atom, unit(axis, H)),
                &gamma,
            );
            let minus = e3(
                &shift_basis(&main, atom, unit(axis, -H)),
                &shift_basis(&aux, atom, unit(axis, -H)),
                &gamma,
            );
            worst = worst.max((f - (plus - minus) / (2.0 * H)).abs());
        }
    }
    assert!(worst < FD_TOL, "3c grad vs FD: worst {worst:.3e}");
}

#[test]
fn eri_2c_grad_contract_matches_finite_difference() {
    let aux = aux_basis();
    // Asymmetric gamma: the full Σ_{PQ} contraction convention is what FD sees.
    let gamma = random_gamma(aux.nao() * aux.nao(), 0x1234_5678);
    let forces = aux.eri_2c_grad_contract(&gamma).unwrap();
    assert_eq!(forces.len(), aux.atoms().len());
    let mut worst = 0.0_f64;
    for (ai, &atom) in aux.atoms().iter().enumerate() {
        for (axis, &f) in forces[ai].iter().enumerate() {
            let plus = e2(&shift_basis(&aux, atom, unit(axis, H)), &gamma);
            let minus = e2(&shift_basis(&aux, atom, unit(axis, -H)), &gamma);
            worst = worst.max((f - (plus - minus) / (2.0 * H)).abs());
        }
    }
    assert!(worst < FD_TOL, "2c grad vs FD: worst {worst:.3e}");
}

/// An off-diagonal one-hot gamma isolates a single `(P|Q)` pair: confirms the
/// "no implicit symmetrization" convention — γ_{PQ} = 1 alone contracts only
/// `d(P|Q)`, and adding the transposed entry doubles the force.
#[test]
fn eri_2c_grad_contract_does_not_symmetrize_gamma() {
    let aux = aux_basis();
    let naux = aux.nao();
    // AOs on different atoms (0 = s@A0, 11 = first p@A1 component) so the
    // single-pair force is genuinely nonzero.
    let (p, q) = (0, 11);
    let mut one = vec![0.0; naux * naux];
    one[p * naux + q] = 1.0;
    let f_one = aux.eri_2c_grad_contract(&one).unwrap();
    let mut both = one.clone();
    both[q * naux + p] = 1.0;
    let f_both = aux.eri_2c_grad_contract(&both).unwrap();
    let mut worst = 0.0_f64;
    let mut largest = 0.0_f64;
    for (a, b) in f_one.iter().zip(&f_both) {
        for axis in 0..3 {
            worst = worst.max((2.0 * a[axis] - b[axis]).abs());
            largest = largest.max(a[axis].abs());
        }
    }
    assert!(
        largest > 1e-3,
        "one-hot force unexpectedly tiny: {largest:.3e}"
    );
    assert!(worst < 1e-12, "2·γ_PQ vs γ_PQ+γ_QP: worst {worst:.3e}");
}

#[test]
fn eri_3c_grad_contract_translational_invariance() {
    let main = orbital_basis();
    let aux = aux_basis();
    let gamma = random_gamma(main.nao() * main.nao() * aux.nao(), 0xDEAD_BEEF);
    let forces = main.eri_3c_grad_contract(&aux, &gamma).unwrap();
    for axis in 0..3 {
        let s: f64 = forces.iter().map(|f| f[axis]).sum();
        assert!(s.abs() < TI_TOL, "3c TI residual (axis {axis}) = {s:.3e}");
    }
}

#[test]
fn eri_2c_grad_contract_translational_invariance() {
    let aux = aux_basis();
    let gamma = random_gamma(aux.nao() * aux.nao(), 0xC0FF_EE11);
    let forces = aux.eri_2c_grad_contract(&gamma).unwrap();
    for axis in 0..3 {
        let s: f64 = forces.iter().map(|f| f[axis]).sum();
        assert!(s.abs() < TI_TOL, "2c TI residual (axis {axis}) = {s:.3e}");
    }
}

/// 3c-vs-4c cross-check via the Gaussian product theorem: a (dummy-free) aux
/// `s` shell of exponent `α` on center C satisfies
/// `(μν|P) = k · (μν|κλ)` with `κ, λ` real `s` shells of exponent `α/2` on C
/// and `k` a geometry-independent normalization ratio. So the 3c contracted
/// gradient must equal `eri_grad_contract` on the combined basis with the
/// corresponding 4c gamma — the 4c path never sees a dummy.
#[test]
fn eri_3c_grad_contract_matches_4c_product_construction() {
    let main = Basis::new(vec![
        Shell::new(0, A0, vec![0.8], vec![1.0]).unwrap(),
        Shell::new(1, A1, vec![0.9], vec![1.0]).unwrap(),
    ]);
    let aux_exps = [1.2_f64, 0.7]; // one s shell per atom, atoms shared
    let aux = Basis::new(vec![
        Shell::new(0, A0, vec![aux_exps[0]], vec![1.0]).unwrap(),
        Shell::new(0, A1, vec![aux_exps[1]], vec![1.0]).unwrap(),
    ]);
    let nao = main.nao(); // 4
    let naux = aux.nao(); // 2
    let gamma3 = random_gamma(nao * nao * naux, 0xABCD_EF01);
    let forces3 = main.eri_3c_grad_contract(&aux, &gamma3).unwrap();

    // Combined basis: main shells, then the κ/λ halves of each aux s shell.
    let centers = [A0, A1];
    let mut shells4 = main.shells().to_vec();
    for (i, &e) in aux_exps.iter().enumerate() {
        shells4.push(Shell::new(0, centers[i], vec![e / 2.0], vec![1.0]).unwrap());
        shells4.push(Shell::new(0, centers[i], vec![e / 2.0], vec![1.0]).unwrap());
    }
    let combined = Basis::new(shells4);
    let n4 = combined.nao(); // nao + 2·naux = 8
    assert_eq!(combined.atoms(), main.atoms());

    // Geometry-independent ratio k_P = (00|κ_P λ_P) / (00|P), from the values.
    let eri4 = combined.eri();
    let idx4 = |i: usize, j: usize, k: usize, l: usize| ((i * n4 + j) * n4 + k) * n4 + l;
    let k_ratio: Vec<f64> = (0..naux)
        .map(|p| {
            let v3 = main.eri_3c_block(&aux, 0, 0, p)[0];
            let (ka, la) = (nao + 2 * p, nao + 2 * p + 1);
            eri4[idx4(0, 0, ka, la)] / v3
        })
        .collect();
    // gamma4[μ,ν,κ_P,λ_P] = Γ3[μ,ν,P] / k_P; everything else zero.
    let mut gamma4 = vec![0.0; n4 * n4 * n4 * n4];
    for i in 0..nao {
        for j in 0..nao {
            for (p, &k) in k_ratio.iter().enumerate() {
                let (ka, la) = (nao + 2 * p, nao + 2 * p + 1);
                // Aux AO index = p (each aux shell is a single s function).
                gamma4[idx4(i, j, ka, la)] = gamma3[(i * nao + j) * naux + p] / k;
            }
        }
    }
    let forces4 = combined.eri_grad_contract(&gamma4).unwrap();
    assert_eq!(forces4.len(), forces3.len());
    let mut worst = 0.0_f64;
    for (f3, f4) in forces3.iter().zip(&forces4) {
        for axis in 0..3 {
            worst = worst.max((f3[axis] - f4[axis]).abs());
        }
    }
    assert!(
        worst < 1e-10,
        "3c vs 4c product construction: worst {worst:.3e}"
    );
}

// ---------------------------------------------------------------------------
// Error paths.
// ---------------------------------------------------------------------------

#[test]
fn eri_3c_grad_contract_rejects_atom_mismatch() {
    let main = orbital_basis();
    let off = [2.0, 2.0, 2.0]; // not a main-basis atom
    let aux = Basis::new(vec![Shell::new(0, off, vec![1.0], vec![1.0]).unwrap()]);
    let gamma = vec![0.0; main.nao() * main.nao() * aux.nao()];
    assert_eq!(
        main.eri_3c_grad_contract(&aux, &gamma).unwrap_err(),
        IntegralError::ChargeNotOnAtom { center: off }
    );
    // Same atoms but missing one in aux is also a mismatch.
    let aux_partial = Basis::new(vec![Shell::new(0, A0, vec![1.0], vec![1.0]).unwrap()]);
    let gamma = vec![0.0; main.nao() * main.nao() * aux_partial.nao()];
    assert!(matches!(
        main.eri_3c_grad_contract(&aux_partial, &gamma),
        Err(IntegralError::ChargeNotOnAtom { .. })
    ));
}

#[test]
fn df_grad_contract_rejects_wrong_gamma_length() {
    let main = orbital_basis();
    let aux = aux_basis();
    let bad3 = vec![0.0; main.nao() * main.nao() * aux.nao() - 1];
    assert_eq!(
        main.eri_3c_grad_contract(&aux, &bad3).unwrap_err(),
        IntegralError::GammaLengthMismatch {
            expected: main.nao() * main.nao() * aux.nao(),
            got: bad3.len()
        }
    );
    let bad2 = vec![0.0; aux.nao() * aux.nao() + 1];
    assert_eq!(
        aux.eri_2c_grad_contract(&bad2).unwrap_err(),
        IntegralError::GammaLengthMismatch {
            expected: aux.nao() * aux.nao(),
            got: bad2.len()
        }
    );
}

#[test]
fn df_grad_contract_rejects_l6_shells() {
    let i_shell = |c| Shell::new(6, c, vec![1.0], vec![1.0]).unwrap();
    let s_shell = |c| Shell::new(0, c, vec![1.0], vec![1.0]).unwrap();
    let err = IntegralError::AngularMomentumTooHighForGradient { l: 6, max: 5 };
    // l = 6 in the orbital basis.
    let main = Basis::new(vec![i_shell(A0)]);
    let aux = Basis::new(vec![s_shell(A0)]);
    let gamma = vec![0.0; main.nao() * main.nao() * aux.nao()];
    assert_eq!(main.eri_3c_grad_contract(&aux, &gamma).unwrap_err(), err);
    // l = 6 in the aux basis.
    let main = Basis::new(vec![s_shell(A0)]);
    let aux = Basis::new(vec![i_shell(A0)]);
    let gamma = vec![0.0; main.nao() * main.nao() * aux.nao()];
    assert_eq!(main.eri_3c_grad_contract(&aux, &gamma).unwrap_err(), err);
    // l = 6 in the 2c metric basis.
    let gamma = vec![0.0; aux.nao() * aux.nao()];
    assert_eq!(aux.eri_2c_grad_contract(&gamma).unwrap_err(), err);
}

// ---------------------------------------------------------------------------
// High-L corners (slow): differentiated shells at l = 5, raised to l = 6.
// ---------------------------------------------------------------------------

/// FD worst error for one (main, aux) pair over all atoms/axes.
fn fd_worst_3c(main: &Basis, aux: &Basis, gamma: &[f64]) -> f64 {
    let forces = main.eri_3c_grad_contract(aux, gamma).unwrap();
    let mut worst = 0.0_f64;
    for (ai, &atom) in main.atoms().iter().enumerate() {
        for (axis, &f) in forces[ai].iter().enumerate() {
            let plus = e3(
                &shift_basis(main, atom, unit(axis, H)),
                &shift_basis(aux, atom, unit(axis, H)),
                gamma,
            );
            let minus = e3(
                &shift_basis(main, atom, unit(axis, -H)),
                &shift_basis(aux, atom, unit(axis, -H)),
                gamma,
            );
            worst = worst.max((f - (plus - minus) / (2.0 * H)).abs());
        }
    }
    worst
}

#[test]
#[ignore = "slow high-L corner; run in release with --include-ignored"]
fn fd_3c_gradient_l5_orbital_shell() {
    let main = Basis::new(vec![
        Shell::new(5, A0, vec![1.0], vec![1.0]).unwrap(),
        Shell::new(0, A1, vec![0.7], vec![1.0]).unwrap(),
    ]);
    let aux = Basis::new(vec![
        Shell::new(0, A0, vec![1.2], vec![1.0]).unwrap(),
        Shell::new(1, A1, vec![0.9], vec![1.0]).unwrap(),
    ]);
    let gamma = random_gamma(main.nao() * main.nao() * aux.nao(), 0x5151_5151);
    let worst = fd_worst_3c(&main, &aux, &gamma);
    assert!(worst < 1e-5, "3c grad FD (orbital l=5): worst {worst:.3e}");
}

#[test]
#[ignore = "slow high-L corner; run in release with --include-ignored"]
fn fd_3c_gradient_l5_aux_shell() {
    let main = Basis::new(vec![
        Shell::new(0, A0, vec![0.8], vec![1.0]).unwrap(),
        Shell::new(1, A1, vec![0.9], vec![1.0]).unwrap(),
    ]);
    let aux = Basis::new(vec![
        Shell::new(5, A0, vec![1.0], vec![1.0]).unwrap(),
        Shell::new(0, A1, vec![0.7], vec![1.0]).unwrap(),
    ]);
    let gamma = random_gamma(main.nao() * main.nao() * aux.nao(), 0x2525_2525);
    let worst = fd_worst_3c(&main, &aux, &gamma);
    assert!(worst < 1e-5, "3c grad FD (aux l=5): worst {worst:.3e}");
}

#[test]
#[ignore = "slow high-L corner; run in release with --include-ignored"]
fn fd_2c_gradient_l5() {
    let aux = Basis::new(vec![
        Shell::new(5, A0, vec![1.0], vec![1.0]).unwrap(),
        Shell::new(1, A1, vec![0.9], vec![1.0]).unwrap(),
    ]);
    let gamma = random_gamma(aux.nao() * aux.nao(), 0x4242_4242);
    let forces = aux.eri_2c_grad_contract(&gamma).unwrap();
    let mut worst = 0.0_f64;
    for (ai, &atom) in aux.atoms().iter().enumerate() {
        for (axis, &f) in forces[ai].iter().enumerate() {
            let plus = e2(&shift_basis(&aux, atom, unit(axis, H)), &gamma);
            let minus = e2(&shift_basis(&aux, atom, unit(axis, -H)), &gamma);
            worst = worst.max((f - (plus - minus) / (2.0 * H)).abs());
        }
    }
    assert!(worst < 1e-5, "2c grad FD (l=5): worst {worst:.3e}");
    for axis in 0..3 {
        let s: f64 = forces.iter().map(|f| f[axis]).sum();
        assert!(s.abs() < TI_TOL, "2c l=5 TI residual = {s:.3e}");
    }
}
