//! Integration tests for the periodic GPW grid + Poisson public API (`periodic` feature).
//!
//! The rigorous, machine-precision Poisson check (a band-limited periodic potential) and the
//! homogeneous-gas check live as unit tests next to the solver. Here we exercise the public
//! API end-to-end and validate against the **analytic** Hartree potential of localized
//! Gaussian charges — the physical M1 deliverable.

#![cfg(feature = "periodic")]

use integral::periodic::{collocate_density, hartree, RealSpaceGrid};
use integral::{Basis, Shell};
use latx::Cell;

/// Abramowitz & Stegun 7.1.26 rational approximation to `erf` (|err| < 1.5e-7), enough for a
/// ~1% physical comparison without pulling in a special-function dependency.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    sign * y
}

/// The isolated Hartree potential of a unit normalized Gaussian charge of exponent `alpha`
/// centred at `c`, evaluated at `r`: `erf(√α·d)/d` with `d = |r − c|`.
fn gaussian_potential(alpha: f64, c: [f64; 3], r: [f64; 3]) -> f64 {
    let d = ((r[0] - c[0]).powi(2) + (r[1] - c[1]).powi(2) + (r[2] - c[2]).powi(2)).sqrt();
    if d < 1e-12 {
        // limit erf(√α d)/d → 2√(α/π)
        2.0 * (alpha / std::f64::consts::PI).sqrt()
    } else {
        erf(alpha.sqrt() * d) / d
    }
}

#[test]
fn grid_and_poisson_public_api_smoke() {
    let grid = RealSpaceGrid::new(Cell::cubic(5.0).unwrap(), [10, 10, 10]);
    assert_eq!(grid.n_points(), 1000);
    let (v, e_h) = hartree(&vec![0.0; grid.n_points()], &grid);
    assert_eq!(v.len(), grid.n_points());
    assert_eq!(e_h, 0.0);
}

/// A charge-neutral pair of opposite Gaussians (a "dipole") in a large box: the periodic
/// Hartree potential reproduces the analytic isolated potential to ~1%. Neutrality kills the
/// monopole, so finite-cell image effects are small; comparing the potential **difference**
/// between two symmetric probe points removes any residual constant.
#[test]
fn poisson_matches_analytic_gaussian_dipole() {
    let l = 18.0;
    let n = 72; // 72 = 2³·3², spacing 0.25 bohr — resolves α = 1.5 Gaussians fully
    let cell = Cell::cubic(l).unwrap();
    let grid = RealSpaceGrid::new(cell, [n, n, n]);

    let alpha = 1.5;
    let norm = (alpha / std::f64::consts::PI).powf(1.5);
    let center = l / 2.0;
    let plus = [center + 0.7, center, center]; // +1 charge
    let minus = [center - 0.7, center, center]; // −1 charge

    let gauss = |c: [f64; 3], r: [f64; 3]| {
        let d2 = (r[0] - c[0]).powi(2) + (r[1] - c[1]).powi(2) + (r[2] - c[2]).powi(2);
        norm * (-alpha * d2).exp()
    };

    let mut n_r = vec![0.0; grid.n_points()];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let r = grid.point([i, j, k]);
                n_r[grid.linear_index([i, j, k])] = gauss(plus, r) - gauss(minus, r);
            }
        }
    }

    // Density integrates to ~0 (neutral) over the cell.
    let q: f64 = n_r.iter().sum::<f64>() * grid.dv();
    assert!(q.abs() < 1e-9, "net charge should vanish, got {q}");

    let (v, _e_h) = hartree(&n_r, &grid);

    // Probe points symmetric about the dipole, on the grid: [center±2, center, center].
    let off = (2.0 / (l / n as f64)).round() as usize; // 2 bohr in grid steps
    let mid = n / 2;
    let p_plus = [mid + off, mid, mid];
    let p_minus = [mid - off, mid, mid];
    let v_plus = v[grid.linear_index(p_plus)];
    let v_minus = v[grid.linear_index(p_minus)];

    let analytic = |idx: [usize; 3]| {
        let r = grid.point(idx);
        gaussian_potential(alpha, plus, r) - gaussian_potential(alpha, minus, r)
    };
    let want = analytic(p_plus) - analytic(p_minus);
    let got = v_plus - v_minus;

    assert!(
        want > 0.5,
        "sanity: analytic dipole potential difference should be O(1): {want}"
    );
    assert!(
        (got - want).abs() < 2.0e-2,
        "periodic Hartree vs analytic Gaussian dipole: got {got}, want {want} (Δ {})",
        (got - want).abs()
    );
}

/// Headline combined-pipeline check: collocate a single s-density `n(r) = φ₀²` (P = [[1]]),
/// solve Poisson, and confirm the grid Hartree energy `E_H = ½∫n V_H` converges to the
/// analytic self-Coulomb `½·(00|00)` (from the engine's own ERI) as the cell grows — with the
/// theoretically correct **cubic Madelung** finite-size scaling. A unit charge in a periodic
/// cell with a neutralizing background carries the Makov–Payne monopole error
/// `−ξ/(2L)` with `ξ = 2.837297`, so `residual·L → ξ/2 ≈ 1.4186`. This ties collocation +
/// Poisson + the analytic ERI together in one physical number.
#[test]
fn grid_hartree_converges_to_analytic_eri() {
    const MADELUNG_HALF: f64 = 1.418_648; // ξ/2 for a simple-cubic lattice (ξ = 2.837297)
    let alpha = 0.8;
    let shell0 = Shell::new(0, [0.0, 0.0, 0.0], vec![alpha], vec![1.0]).unwrap();
    // Analytic self-Coulomb (00|00) from the engine; E_H^iso = ½·(00|00).
    let eri = Basis::new(vec![shell0]).eri();
    let e_iso = 0.5 * eri[0];

    let spacing = 0.25;
    let lengths = [12.0_f64, 16.0, 20.0];
    let mut residuals = Vec::new();
    for &l in &lengths {
        let n = (l / spacing).round() as usize; // 48, 64, 80 (all 7-smooth)
        let c = l / 2.0;
        let basis = Basis::new(vec![
            Shell::new(0, [c, c, c], vec![alpha], vec![1.0]).unwrap()
        ]);
        let grid = RealSpaceGrid::new(Cell::cubic(l).unwrap(), [n, n, n]);
        let n_r = collocate_density(&basis, &[1.0], &grid);
        // collocated density integrates to one electron.
        let q: f64 = n_r.iter().sum::<f64>() * grid.dv();
        assert!((q - 1.0).abs() < 1e-4, "∫n should be 1, got {q} (L={l})");
        let (_v, e_h) = hartree(&n_r, &grid);
        residuals.push(e_iso - e_h); // positive: periodic energy is below isolated
    }

    // Monotonic convergence toward the analytic value.
    assert!(
        residuals[0] > residuals[1] && residuals[1] > residuals[2] && residuals[2] > 0.0,
        "Hartree energy should converge upward to ½(00|00): residuals {residuals:?}"
    );
    // The 1/L scaling matches the cubic Madelung constant (≈ within a few % at L = 20).
    let scaled = residuals[2] * lengths[2];
    assert!(
        (scaled - MADELUNG_HALF).abs() < 0.12,
        "residual·L = {scaled} should match ξ/2 = {MADELUNG_HALF} (cubic Madelung)"
    );
}
