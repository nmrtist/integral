//! Geometric first-derivative validation — **engine-independent**.
//!
//! Two independent checks (no external reference) for every derivative type
//! (S, T, V, ERI):
//!
//! 1. **Finite difference** vs the value builders: displace an atom by `±h`
//!    (its shells, and for `V` its point charge too) and central-difference the
//!    value integral; compare to the analytic per-atom gradient. This is the
//!    primary correctness check. For `V` the atom displacement moves the charge,
//!    so it rigorously exercises the Hellmann–Feynman (operator-derivative) term.
//! 2. **Translational invariance** `Σ_atoms ∂O/∂R = 0` to machine precision — an
//!    exact internal invariant needing no reference.
//!
//! Finite difference is limited by truncation (`O(h²)`) vs cancellation
//! (`O(ε/h)`) round-off, so it is checked at a **looser** tolerance than the
//! analytic `1e-11`; see `FD_TOL` and the step study in `fd_step_study`.

use integral::{Basis, Engine, Shell};

/// Finite-difference agreement tolerance (max abs elementwise). Central
/// difference at `h = 2e-4` reaches ~`1e-8`–`1e-9` here; `1e-6` leaves margin.
const FD_TOL: f64 = 1e-6;
/// Translational-invariance tolerance (machine precision, with accumulation).
const TI_TOL: f64 = 1e-10;
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

fn shift_charges(
    charges: &[([f64; 3], f64)],
    target: [f64; 3],
    d: [f64; 3],
) -> Vec<([f64; 3], f64)> {
    charges
        .iter()
        .map(|&(c, z)| (shifted(c, target, d), z))
        .collect()
}

fn unit(axis: usize, h: f64) -> [f64; 3] {
    let mut d = [0.0; 3];
    d[axis] = h;
    d
}

fn central_diff(plus: &[f64], minus: &[f64], h: f64) -> Vec<f64> {
    plus.iter()
        .zip(minus)
        .map(|(p, m)| (p - m) / (2.0 * h))
        .collect()
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// A mixed-L, distinct-non-collinear-centre molecule (carries the m4 lesson:
/// non-zero shift structure on every centre).
fn test_basis() -> Basis {
    let a0 = [0.0, 0.0, 0.0];
    let a1 = [0.9, 0.1, -0.2];
    let a2 = [-0.3, 0.8, 0.55];
    Basis::new(vec![
        Shell::new(0, a0, vec![3.4, 0.8], vec![0.4, 0.6]).unwrap(),
        Shell::new(1, a0, vec![1.1], vec![1.0]).unwrap(),
        Shell::new(2, a1, vec![0.9], vec![1.0]).unwrap(),
        Shell::new(1, a2, vec![1.3, 0.5], vec![0.5, 0.7]).unwrap(),
    ])
}

fn charges_of(basis: &Basis) -> Vec<([f64; 3], f64)> {
    basis
        .atoms()
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, (i + 1) as f64))
        .collect()
}

#[test]
fn overlap_grad_matches_finite_difference() {
    let basis = test_basis();
    let g = basis.overlap_grad().unwrap();
    let mut worst = 0.0_f64;
    for (ai, &atom) in basis.atoms().iter().enumerate() {
        for axis in 0..3 {
            let plus = shift_basis(&basis, atom, unit(axis, H)).overlap();
            let minus = shift_basis(&basis, atom, unit(axis, -H)).overlap();
            let fd = central_diff(&plus, &minus, H);
            worst = worst.max(max_abs_diff(g.block(ai, axis), &fd));
        }
    }
    assert!(worst < FD_TOL, "overlap grad vs FD: worst {worst:.3e}");
}

#[test]
fn kinetic_grad_matches_finite_difference() {
    let basis = test_basis();
    let g = basis.kinetic_grad().unwrap();
    let mut worst = 0.0_f64;
    for (ai, &atom) in basis.atoms().iter().enumerate() {
        for axis in 0..3 {
            let plus = shift_basis(&basis, atom, unit(axis, H)).kinetic();
            let minus = shift_basis(&basis, atom, unit(axis, -H)).kinetic();
            let fd = central_diff(&plus, &minus, H);
            worst = worst.max(max_abs_diff(g.block(ai, axis), &fd));
        }
    }
    assert!(worst < FD_TOL, "kinetic grad vs FD: worst {worst:.3e}");
}

#[test]
fn nuclear_grad_matches_finite_difference_incl_hellmann_feynman() {
    let basis = test_basis();
    let charges = charges_of(&basis);
    let g = basis.nuclear_grad(&charges).unwrap();
    let mut worst = 0.0_f64;
    for (ai, &atom) in basis.atoms().iter().enumerate() {
        for axis in 0..3 {
            // Displacing the atom moves BOTH its shells and its point charge, so
            // this FD includes the Hellmann–Feynman (operator) contribution.
            let bp = shift_basis(&basis, atom, unit(axis, H));
            let cp = shift_charges(&charges, atom, unit(axis, H));
            let bm = shift_basis(&basis, atom, unit(axis, -H));
            let cm = shift_charges(&charges, atom, unit(axis, -H));
            let fd = central_diff(&bp.nuclear(&cp), &bm.nuclear(&cm), H);
            worst = worst.max(max_abs_diff(g.block(ai, axis), &fd));
        }
    }
    assert!(worst < FD_TOL, "nuclear grad vs FD: worst {worst:.3e}");
}

/// Direct, isolated check of the Hellmann–Feynman term: finite-difference the
/// nuclear matrix with respect to the **charge coordinate alone** (basis fixed),
/// and compare to the implementation's HF term, recovered as the analytic
/// per-atom gradient minus the basis-only finite difference (shells moved, charge
/// fixed). All three pieces are independent; agreement isolates ∂_C.
#[test]
fn nuclear_hellmann_feynman_term_isolated() {
    let basis = test_basis();
    let charges = charges_of(&basis);
    let g = basis.nuclear_grad(&charges).unwrap();
    let mut worst = 0.0_f64;
    for (ai, &atom) in basis.atoms().iter().enumerate() {
        for axis in 0..3 {
            // HF = ∂V/∂(charge position), basis held fixed.
            let cp = shift_charges(&charges, atom, unit(axis, H));
            let cm = shift_charges(&charges, atom, unit(axis, -H));
            let hf_fd = central_diff(&basis.nuclear(&cp), &basis.nuclear(&cm), H);
            // Basis-only = ∂V/∂(shell positions), charge held fixed.
            let bp = shift_basis(&basis, atom, unit(axis, H));
            let bm = shift_basis(&basis, atom, unit(axis, -H));
            let basis_fd = central_diff(&bp.nuclear(&charges), &bm.nuclear(&charges), H);
            // Implementation's HF = analytic total − basis-only FD.
            let hf_impl: Vec<f64> = g
                .block(ai, axis)
                .iter()
                .zip(&basis_fd)
                .map(|(tot, b)| tot - b)
                .collect();
            worst = worst.max(max_abs_diff(&hf_impl, &hf_fd));
        }
    }
    assert!(worst < FD_TOL, "HF term isolated vs FD: worst {worst:.3e}");
}

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

fn eri_grad_fd_check(engine: Engine) {
    let basis = eri_test_basis();
    let g = basis.eri_grad_with(engine).unwrap();
    let mut worst = 0.0_f64;
    for (ai, &atom) in basis.atoms().iter().enumerate() {
        for axis in 0..3 {
            let plus = shift_basis(&basis, atom, unit(axis, H)).eri_with(engine);
            let minus = shift_basis(&basis, atom, unit(axis, -H)).eri_with(engine);
            let fd = central_diff(&plus, &minus, H);
            worst = worst.max(max_abs_diff(g.block(ai, axis), &fd));
        }
    }
    assert!(
        worst < FD_TOL,
        "ERI grad ({engine:?}) vs FD: worst {worst:.3e}"
    );
}

#[test]
fn eri_grad_matches_finite_difference_rys() {
    eri_grad_fd_check(Engine::Rys);
}

#[test]
fn eri_grad_matches_finite_difference_oshgp() {
    eri_grad_fd_check(Engine::OsHgp);
}

#[test]
fn eri_grad_auto_matches_finite_difference() {
    eri_grad_fd_check(Engine::Auto);
}

/// Deterministic pseudo-random two-particle density with the full 8-fold ERI
/// permutational symmetry (LCG-seeded; each canonical quadruple's value is
/// written to all eight permutation slots).
fn random_symmetric_gamma(nao: usize) -> Vec<f64> {
    let mut g = vec![0.0; nao.pow(4)];
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
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

/// `eri_grad_contract` must equal the explicit contraction of the full
/// `eri_grad` tensor with the same gamma — per engine, on a mixed s/p/d basis.
#[test]
fn eri_grad_contract_matches_explicit_contraction() {
    let basis = eri_test_basis();
    let gamma = random_symmetric_gamma(basis.nao());
    for engine in [Engine::Rys, Engine::OsHgp, Engine::Auto] {
        let full = basis.eri_grad_with(engine).unwrap();
        let forces = basis.eri_grad_contract_with(engine, &gamma).unwrap();
        assert_eq!(forces.len(), basis.atoms().len());
        let mut worst = 0.0_f64;
        for (ai, f) in forces.iter().enumerate() {
            for (axis, &got) in f.iter().enumerate() {
                let want: f64 = full
                    .block(ai, axis)
                    .iter()
                    .zip(&gamma)
                    .map(|(d, g)| d * g)
                    .sum();
                worst = worst.max((got - want).abs());
            }
        }
        assert!(
            worst < 1e-10,
            "contracted vs explicit ({engine:?}): worst {worst:.3e}"
        );
    }
}

/// The contracted force sums to zero over atoms, per component — the same
/// translational invariance as the full tensor, exact up to round-off.
#[test]
fn eri_grad_contract_translational_invariance() {
    let basis = eri_test_basis();
    let gamma = random_symmetric_gamma(basis.nao());
    for engine in [Engine::Rys, Engine::OsHgp] {
        let forces = basis.eri_grad_contract_with(engine, &gamma).unwrap();
        for axis in 0..3 {
            let s: f64 = forces.iter().map(|f| f[axis]).sum();
            assert!(
                s.abs() < TI_TOL,
                "contracted TI residual ({engine:?}, axis {axis}) = {s:.3e}"
            );
        }
    }
}

#[test]
fn eri_grad_contract_rejects_wrong_gamma_length() {
    let basis = eri_test_basis();
    let bad = vec![0.0; basis.nao().pow(4) - 1];
    assert!(matches!(
        basis.eri_grad_contract(&bad),
        Err(integral::IntegralError::GammaLengthMismatch { .. })
    ));
}

#[test]
fn translational_invariance_one_electron() {
    let basis = test_basis();
    let charges = charges_of(&basis);
    assert!(basis.overlap_grad().unwrap().max_translational_residual() < TI_TOL);
    assert!(basis.kinetic_grad().unwrap().max_translational_residual() < TI_TOL);
    assert!(
        basis
            .nuclear_grad(&charges)
            .unwrap()
            .max_translational_residual()
            < TI_TOL
    );
}

#[test]
fn translational_invariance_eri_both_engines() {
    let basis = eri_test_basis();
    for engine in [Engine::Rys, Engine::OsHgp] {
        let r = basis
            .eri_grad_with(engine)
            .unwrap()
            .max_translational_residual();
        assert!(r < TI_TOL, "ERI TI residual ({engine:?}) = {r:.3e}");
    }
}

#[test]
fn both_engines_agree_on_eri_gradient() {
    let basis = eri_test_basis();
    let rys = basis.eri_grad_with(Engine::Rys).unwrap();
    let os = basis.eri_grad_with(Engine::OsHgp).unwrap();
    let mut worst = 0.0_f64;
    for ai in 0..basis.atoms().len() {
        for axis in 0..3 {
            worst = worst.max(max_abs_diff(rys.block(ai, axis), os.block(ai, axis)));
        }
    }
    assert!(worst < 1e-10, "Rys vs OS/HGP ERI grad: worst {worst:.3e}");
}

/// Spherical shells: the c2s transform is linear and per-index, so gradients of
/// spherical integrals are the transform of the Cartesian gradients. Check FD
/// and TI hold for a spherical basis too.
#[test]
fn spherical_overlap_grad_matches_finite_difference() {
    let a0 = [0.0, 0.0, 0.0];
    let a1 = [0.7, 0.2, -0.3];
    let basis = Basis::new(vec![
        Shell::new_spherical(2, a0, vec![1.2, 0.4], vec![0.5, 0.6]).unwrap(),
        Shell::new_spherical(1, a1, vec![0.9], vec![1.0]).unwrap(),
    ]);
    let g = basis.overlap_grad().unwrap();
    let mut worst = 0.0_f64;
    for (ai, &atom) in basis.atoms().iter().enumerate() {
        for axis in 0..3 {
            let plus = shift_basis(&basis, atom, unit(axis, H)).overlap();
            let minus = shift_basis(&basis, atom, unit(axis, -H)).overlap();
            let fd = central_diff(&plus, &minus, H);
            worst = worst.max(max_abs_diff(g.block(ai, axis), &fd));
        }
    }
    assert!(
        worst < FD_TOL,
        "spherical overlap grad vs FD: worst {worst:.3e}"
    );
    assert!(g.max_translational_residual() < TI_TOL);
}

/// Documented step study (run with `--nocapture`): central-difference error vs
/// `h` for the overlap gradient, confirming the `O(h²)` truncation / `O(ε/h)`
/// cancellation trade-off behind `FD_TOL`.
#[test]
fn fd_step_study() {
    let basis = test_basis();
    let g = basis.overlap_grad().unwrap();
    let atom = basis.atoms()[0];
    eprintln!("overlap-grad central-difference error vs step h (atom 0, x):");
    for &h in &[1e-2, 1e-3, 1e-4, 1e-5, 1e-6] {
        let plus = shift_basis(&basis, atom, unit(0, h)).overlap();
        let minus = shift_basis(&basis, atom, unit(0, -h)).overlap();
        let fd = central_diff(&plus, &minus, h);
        eprintln!(
            "  h={h:.0e}  max|analytic - FD| = {:.3e}",
            max_abs_diff(g.block(0, 0), &fd)
        );
    }
}
