//! One-electron operator DSL, validation dimension 1: **reproduce the existing,
//! already-validated bespoke 1e integrals through the DSL and diff
//! element-by-element**.
//!
//! The bespoke builders (`overlap`, `dipole`, `kinetic`) are the reference here — no
//! external reference is needed: machine-precision agreement proves the
//! decomposition machinery is faithful. Each identity is checked over the **full
//! supported L range** (not just d — the recurring earlier-phase coverage gap),
//! Cartesian **and** spherical:
//!
//!   * `overlap`  ≡ DSL `identity`   (degree 0 → l ≤ 6)
//!   * `dipole`   ≡ DSL `r`          (degree 1 → l ≤ 5)
//!   * `kinetic`  ≡ DSL `½ p·p`      (degree 2 → l ≤ 4)
//!
//! All three operators are real-symmetric, so the DSL's imaginary part must be ~0.

use integral::{Basis, Operator, Shell, ShellKind};

/// Two contracted shells of momentum `l` on distinct non-collinear centres
/// (so off-diagonal blocks are non-trivial), Cartesian or spherical.
fn two_shell_basis(l: usize, sph: bool) -> Basis {
    let kind = if sph {
        ShellKind::Spherical
    } else {
        ShellKind::Cartesian
    };
    Basis::new(vec![
        Shell::with_kind(l, [0.0, 0.0, 0.0], vec![1.2, 0.45], vec![0.6, 0.5], kind).unwrap(),
        Shell::with_kind(l, [0.7, -0.3, 0.4], vec![0.9], vec![1.0], kind).unwrap(),
    ])
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// Machine-precision tolerance for the analytic identities (different float op
/// order between the bespoke recurrence and the DSL's overlap recombination).
const TOL: f64 = 1e-11;

#[test]
fn overlap_equals_dsl_identity_full_l_cart_and_sph() {
    let id = Operator::identity();
    let mut worst = 0.0_f64;
    for &sph in &[false, true] {
        for l in 0..=6 {
            let basis = two_shell_basis(l, sph);
            let bespoke = basis.overlap();
            let dsl = basis.int1e(&id).unwrap();
            let d = max_abs_diff(&bespoke, dsl.real());
            worst = worst.max(d).max(dsl.max_abs_imag());
            eprintln!(
                "overlap≡identity l={l} sph={sph}: re_diff={d:.2e} im={:.2e}",
                dsl.max_abs_imag()
            );
        }
    }
    assert!(worst < TOL, "overlap≡identity worst {worst:.3e}");
}

#[test]
fn dipole_equals_dsl_r_full_l_cart_and_sph() {
    let origin = [0.15, -0.25, 0.35];
    let mut worst = 0.0_f64;
    for &sph in &[false, true] {
        for l in 0..=5 {
            let basis = two_shell_basis(l, sph);
            let bespoke = basis.dipole(origin); // [Dx, Dy, Dz]
            for (axis, comp) in bespoke.iter().enumerate() {
                let dsl = basis.int1e(&Operator::dipole(axis, origin)).unwrap();
                let d = max_abs_diff(comp, dsl.real());
                worst = worst.max(d).max(dsl.max_abs_imag());
            }
            eprintln!("dipole≡r l={l} sph={sph}: worst so far {worst:.2e}");
        }
    }
    assert!(worst < TOL, "dipole≡r worst {worst:.3e}");
}

#[test]
fn kinetic_equals_dsl_half_p_dot_p_full_l_cart_and_sph() {
    let kin = Operator::kinetic();
    let mut worst = 0.0_f64;
    for &sph in &[false, true] {
        for l in 0..=4 {
            let basis = two_shell_basis(l, sph);
            let bespoke = basis.kinetic();
            let dsl = basis.int1e(&kin).unwrap();
            let d = max_abs_diff(&bespoke, dsl.real());
            worst = worst.max(d).max(dsl.max_abs_imag());
            eprintln!(
                "kinetic≡½p·p l={l} sph={sph}: re_diff={d:.2e} im={:.2e}",
                dsl.max_abs_imag()
            );
        }
    }
    assert!(worst < TOL, "kinetic≡½p·p worst {worst:.3e}");
}

/// The DSL must reject a shell whose momentum is too high for the operator's
/// degree (the ket is raised to `l + degree`), with a clean typed error.
#[test]
fn operator_momentum_too_high_is_clean_error() {
    use integral::IntegralError;
    // degree-2 operator (kinetic) on an i shell (l=6) → l+2 = 8 > MAX_L=6.
    let basis = Basis::new(vec![
        Shell::new(6, [0.0, 0.0, 0.0], vec![1.0], vec![1.0]).unwrap()
    ]);
    assert_eq!(
        basis.int1e(&Operator::kinetic()).unwrap_err(),
        IntegralError::OperatorMomentumTooHigh {
            l: 6,
            degree: 2,
            max: 6
        }
    );
    // degree-1 operator (dipole) on an i shell → l+1 = 7 > 6.
    assert_eq!(
        basis.int1e(&Operator::dipole(0, [0.0; 3])).unwrap_err(),
        IntegralError::OperatorMomentumTooHigh {
            l: 6,
            degree: 1,
            max: 6
        }
    );
    // identity (degree 0) is fine at l=6.
    assert!(basis.int1e(&Operator::identity()).is_ok());
}
