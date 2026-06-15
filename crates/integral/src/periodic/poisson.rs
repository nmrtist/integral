//! Reciprocal-space Poisson solve for the periodic Hartree potential.
//!
//! Given a real electron density `n(r)` sampled on a [`RealSpaceGrid`], solve
//! `∇²V_H = -4π n` under periodic boundary conditions. In reciprocal space
//! `Ṽ_H(G) = 4π ñ(G)/|G|²` for `G ≠ 0`; the `G = 0` term is set to zero (a uniform
//! neutralizing background), which is the standard, well-defined choice for a charge-neutral
//! periodic cell.
//!
//! ## Normalization
//!
//! With the unnormalized DFT `N(G) = Σ_r n(r) e^{-iG·r}` (this crate's [`super::fft`]), the
//! continuum Fourier coefficient is `ñ(G) = N(G)/N_pts`, so
//! `Ṽ_H(G) = 4π N(G)/(N_pts·|G|²)`, and the **unnormalized** inverse DFT then yields
//! `V_H(r) = Σ_G Ṽ_H(G) e^{+iG·r}` directly — the usual `1/N` of the inverse transform is
//! exactly the `1/N_pts` already folded into `Ṽ_H`.

use super::fft::Fft3d;
use super::grid::RealSpaceGrid;
use rustfft::num_complex::Complex64;
use std::f64::consts::PI;

/// `|G|²` below this (bohr⁻²) is treated as the `G = 0` term.
const G2_ZERO_TOL: f64 = 1e-12;

/// Solve the periodic Poisson equation for the Hartree potential of the real density
/// `n_r` (length [`RealSpaceGrid::n_points`], row-major matching the grid).
///
/// Returns `(V_H, E_H)` where `V_H` is the Hartree potential on the grid (hartree) and
/// `E_H = ½ ∫ n V_H dr ≈ ½ dV Σ_r n(r) V_H(r)` is the Hartree energy. With the `G = 0`
/// term dropped, both are defined relative to the neutralizing background.
///
/// # Panics
/// Panics if `n_r.len()` differs from the grid point count.
#[must_use]
pub fn hartree(n_r: &[f64], grid: &RealSpaceGrid) -> (Vec<f64>, f64) {
    let npts = grid.n_points();
    assert_eq!(
        n_r.len(),
        npts,
        "density length {} must equal grid points {npts}",
        n_r.len()
    );

    let mut data: Vec<Complex64> = n_r.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    let fft = Fft3d::new(grid.n());
    fft.forward(&mut data); // data[G] = N(G)

    let inv_npts = 1.0 / npts as f64;
    let four_pi = 4.0 * PI;
    for (d, g) in data.iter_mut().zip(grid.g_vectors()) {
        let g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
        if g2 < G2_ZERO_TOL {
            *d = Complex64::new(0.0, 0.0); // neutralizing background
        } else {
            *d *= four_pi * inv_npts / g2; // Ṽ_H(G) = 4π N(G)/(N_pts |G|²)
        }
    }

    fft.inverse(&mut data); // V_H(r) = Σ_G Ṽ_H(G) e^{+iG·r}
    let v: Vec<f64> = data.iter().map(|c| c.re).collect();

    let e_h = 0.5 * grid.dv() * n_r.iter().zip(&v).map(|(&n, &vv)| n * vv).sum::<f64>();
    (v, e_h)
}

/// The reciprocal-space (metric) part of the Hartree stress `∂E_H/∂ε`,
/// `τ^recip_αβ = 2 Σ_{G≠0} e_G · G_α G_β / |G|²` with `e_G = (2π dV/N) |N(G)|² / |G|²`
/// (the per-`G` Hartree energy, `Σ_G e_G = E_H`). This is the explicit dependence on the
/// reciprocal metric (`G → F⁻ᵀ G` under a strain `F = I + ε`), with the **density values held
/// fixed**; the caller adds the volume-scaling part `δ_αβ E_H` and the density (collocation)
/// Pulay and core terms. `n_r` is the total density `ρ_tot` (length [`RealSpaceGrid::n_points`]).
///
/// Symmetric in `αβ`. The trace is `Σ_α τ^recip_αα = 2 E_H` (so the full metric trace
/// `δ_αα E_H + 2E_H = 5E_H` is the `(1+λ)⁵` scaling of `E_H` at fixed density).
///
/// # Panics
/// Panics if `n_r.len()` differs from the grid point count.
#[must_use]
pub fn hartree_reciprocal_stress(n_r: &[f64], grid: &RealSpaceGrid) -> [[f64; 3]; 3] {
    let npts = grid.n_points();
    assert_eq!(
        n_r.len(),
        npts,
        "density length {} must equal grid points {npts}",
        n_r.len()
    );
    let mut data: Vec<Complex64> = n_r.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    let fft = Fft3d::new(grid.n());
    fft.forward(&mut data); // data[G] = N(G)

    let pref = 2.0 * PI * grid.dv() / npts as f64; // 2π dV / N
    let mut tau = [[0.0_f64; 3]; 3];
    for (d, g) in data.iter().zip(grid.g_vectors()) {
        let g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
        if g2 < G2_ZERO_TOL {
            continue;
        }
        let e_g = pref * d.norm_sqr() / g2; // per-G Hartree energy
        let f = 2.0 * e_g / g2; // 2 e_G / |G|²
        for (a, ta) in tau.iter_mut().enumerate() {
            for (b, tab) in ta.iter_mut().enumerate() {
                *tab += f * g[a] * g[b];
            }
        }
    }
    tau
}

#[cfg(test)]
mod tests {
    use super::*;
    use latx::Cell;
    use std::f64::consts::PI;

    /// The analytic Hartree metric stress `δ_αβ E_H + τ^recip` matches the central FD of
    /// `E_H` under a strain that deforms only the **cell metric** (volume `Ω` and the
    /// reciprocal vectors `G`), with the density grid values held fixed — isolating the
    /// reciprocal-space (Ω, |G|²) dependence from the collocation Pulay. Checked over
    /// uniaxial, shear, and general-symmetric strains.
    #[test]
    fn hartree_metric_stress_matches_finite_difference() {
        let l = 7.0;
        let dims = [16usize, 16, 16];
        let cell = Cell::cubic(l).unwrap();
        let grid = RealSpaceGrid::new(cell, dims);
        // A fixed, neutral (zero-mean) density array on the grid.
        let two_pi_l = 2.0 * PI / l;
        let n_r: Vec<f64> = grid
            .points()
            .iter()
            .map(|r| {
                (two_pi_l * r[0]).sin() + 0.5 * (2.0 * two_pi_l * r[1]).cos()
                    - 0.3 * (two_pi_l * r[2]).sin()
            })
            .collect();
        let (_v, e_h) = hartree(&n_r, &grid);
        let tau_recip = hartree_reciprocal_stress(&n_r, &grid);

        let deform = |m: &[[f64; 3]; 3], lambda: f64, v: [f64; 3]| -> [f64; 3] {
            let mut o = v;
            for (a, oa) in o.iter_mut().enumerate() {
                for (b, &vb) in v.iter().enumerate() {
                    *oa += lambda * m[a][b] * vb;
                }
            }
            o
        };
        let dirs = [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.4, 0.2, -0.1], [0.2, -0.3, 0.15], [-0.1, 0.15, 0.5]],
        ];
        for mdir in dirs {
            // E_H with the SAME density values on a metric-deformed cell.
            let e_at = |lambda: f64| -> f64 {
                let (a1, a2, a3) = cell.vectors();
                let dcell = Cell::from_vectors(
                    deform(&mdir, lambda, a1),
                    deform(&mdir, lambda, a2),
                    deform(&mdir, lambda, a3),
                )
                .unwrap();
                hartree(&n_r, &RealSpaceGrid::new(dcell, dims)).1
            };
            let h = 1e-5;
            let fd = (e_at(h) - e_at(-h)) / (2.0 * h);
            let analytic: f64 = (0..3)
                .flat_map(|a| (0..3).map(move |b| (a, b)))
                .map(|(a, b)| {
                    let metric = if a == b { e_h } else { 0.0 };
                    (metric + tau_recip[a][b]) * mdir[a][b]
                })
                .sum();
            assert!(
                (analytic - fd).abs() < 1e-7 * fd.abs().max(1.0),
                "strain {mdir:?}: analytic {analytic} vs FD {fd}"
            );
        }
    }

    /// A constant (homogeneous-electron-gas) density has zero Hartree potential and energy
    /// after the neutralizing background removes the `G = 0` term.
    #[test]
    fn homogeneous_density_gives_zero_potential() {
        let grid = RealSpaceGrid::new(Cell::cubic(6.0).unwrap(), [12, 12, 12]);
        let n_r = vec![0.5; grid.n_points()];
        let (v, e_h) = hartree(&n_r, &grid);
        assert!(
            v.iter().all(|&x| x.abs() < 1e-10),
            "V_H must vanish for constant n"
        );
        assert!(
            e_h.abs() < 1e-10,
            "E_H must vanish for constant n, got {e_h}"
        );
    }

    /// Rigorous, exact Poisson check: pick a band-limited periodic potential
    /// `V*(r) = cos(G1·r) + ½ cos(G2·r)` (zero mean), derive its source density
    /// `n = -∇²V*/4π`, feed it through the solver, and recover `V*` to machine precision.
    #[test]
    fn recovers_band_limited_potential() {
        let l = 7.0;
        let cell = Cell::cubic(l).unwrap();
        let n = [10usize, 10, 10];
        let grid = RealSpaceGrid::new(cell, n);
        let two_pi_l = 2.0 * PI / l;
        // Reciprocal vectors for integer triples [1,0,0] and [0,1,2] in this cubic cell.
        let g1 = [two_pi_l, 0.0, 0.0];
        let g2 = [0.0, two_pi_l, 2.0 * two_pi_l];
        let g1_2 = g1[0] * g1[0] + g1[1] * g1[1] + g1[2] * g1[2];
        let g2_2 = g2[0] * g2[0] + g2[1] * g2[1] + g2[2] * g2[2];

        let dot = |g: [f64; 3], r: [f64; 3]| g[0] * r[0] + g[1] * r[1] + g[2] * r[2];
        let v_exact = |r: [f64; 3]| dot(g1, r).cos() + 0.5 * dot(g2, r).cos();
        // n = -∇²V*/4π = (|G1|² cos(G1·r) + ½|G2|² cos(G2·r)) / 4π.
        let density =
            |r: [f64; 3]| (g1_2 * dot(g1, r).cos() + 0.5 * g2_2 * dot(g2, r).cos()) / (4.0 * PI);

        let mut n_r = vec![0.0; grid.n_points()];
        for i in 0..n[0] {
            for j in 0..n[1] {
                for k in 0..n[2] {
                    let r = grid.point([i, j, k]);
                    n_r[grid.linear_index([i, j, k])] = density(r);
                }
            }
        }

        let (v, _e_h) = hartree(&n_r, &grid);
        for i in 0..n[0] {
            for j in 0..n[1] {
                for k in 0..n[2] {
                    let r = grid.point([i, j, k]);
                    let got = v[grid.linear_index([i, j, k])];
                    assert!(
                        (got - v_exact(r)).abs() < 1e-10,
                        "Poisson mismatch at {r:?}: got {got}, want {}",
                        v_exact(r)
                    );
                }
            }
        }
    }
}
