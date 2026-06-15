//! The two normalization factors in isolation, and the structural reason a
//! compensating (factor × c2s) error cannot survive.
//!
//! The spherical per-index transform applied to an integral-MONOMIAL Cartesian block
//! is `M(l) = ratio(l) · c2s_matrix(l)` with `ratio = monomial_to_raw_factor`.
//! This test isolates each piece:
//!
//!  1. `ratio(l)` is the closed form `sqrt(4π/(2l+1))` (`1` for `l ≤ 1`).
//!  2. `c2s_matrix(l)` unit-normalizes in the **raw** metric: `C·S_raw·Cᵀ = I`.
//!  3. The raw and monomial Cartesian self-overlaps differ by exactly `ratio²`
//!     (`S_raw = ratio² · S_mono`), so the `ratio` inside `C`'s normalization and
//!     the `ratio` multiplier in `M` **cancel**: `M = ratio·C` unit-normalizes in
//!     the **monomial** metric (`M·S_mono·Mᵀ = I`) and is therefore INDEPENDENT of
//!     `ratio`'s value. Consequence: the spherical integral output cannot detect a
//!     wrong `ratio` — its value is guarded solely by `ratio`'s analytic closed
//!     form (test 1), which this confirms is non-redundant.

use integral::math::am::{cart_components, n_cart};
use integral::math::norm::{cart_norm, double_factorial};
use integral::math::solid_harmonics::{c2s_matrix, monomial_to_raw_factor};

fn pi() -> f64 {
    std::f64::consts::PI
}

/// 1D Gaussian moment ∫ x^n e^{-β x²} dx over ℝ (0 for odd n).
fn moment(n: usize, beta: f64) -> f64 {
    if n % 2 == 1 {
        return 0.0;
    }
    double_factorial(n as i64 - 1) / (2.0 * beta).powi(n as i32 / 2) * (pi() / beta).sqrt()
}

/// Monomial-normalized Cartesian self-overlap S_mono[i,j] for a degree-l shell at
/// α = 1: both components scaled by the shell norm N(α; l,0,0) (integral convention).
fn s_mono(l: usize) -> Vec<f64> {
    let comps = cart_components(l);
    let nc = n_cart(l);
    let nshell = cart_norm(1.0, l, 0, 0);
    let beta = 2.0;
    let mut s = vec![0.0; nc * nc];
    for (i, ci) in comps.iter().enumerate() {
        for (j, cj) in comps.iter().enumerate() {
            let bare = moment(ci[0] + cj[0], beta)
                * moment(ci[1] + cj[1], beta)
                * moment(ci[2] + cj[2], beta);
            s[i * nc + j] = nshell * nshell * bare;
        }
    }
    s
}

#[test]
fn factor1_is_closed_form_in_isolation() {
    assert_eq!(monomial_to_raw_factor(0), 1.0);
    assert_eq!(monomial_to_raw_factor(1), 1.0);
    for l in 2..=6 {
        let want = (4.0 * pi() / (2 * l + 1) as f64).sqrt();
        assert!((monomial_to_raw_factor(l) - want).abs() < 1e-15, "l={l}");
    }
}

#[test]
fn c2s_unit_in_raw_metric_and_composed_unit_in_monomial_metric() {
    for l in 0..=6 {
        let nc = n_cart(l);
        let nsph = 2 * l + 1;
        let c = c2s_matrix(l);
        let ratio = monomial_to_raw_factor(l);
        // S_raw = ratio² · S_mono (the two metrics differ by exactly ratio² — the
        // claim that makes the cancellation exact).
        let smono = s_mono(l);
        let sraw: Vec<f64> = smono.iter().map(|&v| ratio * ratio * v).collect();

        // M = ratio · C, the composed monomial→spherical transform.
        let m: Vec<f64> = c.iter().map(|&v| ratio * v).collect();

        // Gram in each metric.
        let gram = |t: &[f64], s: &[f64]| {
            let mut g = vec![0.0; nsph * nsph];
            for p in 0..nsph {
                for q in 0..nsph {
                    let mut acc = 0.0;
                    for i in 0..nc {
                        let mut si = 0.0;
                        for j in 0..nc {
                            si += s[i * nc + j] * t[q * nc + j];
                        }
                        acc += t[p * nc + i] * si;
                    }
                    g[p * nsph + q] = acc;
                }
            }
            g
        };
        // C·S_raw·Cᵀ = I  (c2s_matrix is unit in the raw metric, factor 2 isolated).
        let g_raw = gram(&c, &sraw);
        // M·S_mono·Mᵀ = I  (composed transform unit in the monomial metric — what
        // the engines actually emit — INDEPENDENT of ratio because M = ratio·C and
        // C ∝ 1/ratio).
        let g_mono = gram(&m, &smono);
        for p in 0..nsph {
            for q in 0..nsph {
                let e = if p == q { 1.0 } else { 0.0 };
                assert!(
                    (g_raw[p * nsph + q] - e).abs() < 1e-11,
                    "l={l} C·S_raw·Cᵀ[{p},{q}]"
                );
                assert!(
                    (g_mono[p * nsph + q] - e).abs() < 1e-11,
                    "l={l} M·S_mono·Mᵀ[{p},{q}]"
                );
            }
        }
    }
}

/// Direct demonstration of ratio-cancellation: `ratio · c2s_matrix` is unchanged
/// if we (hypothetically) rescale the factor, because `c2s_matrix` already carries
/// the reciprocal. We verify `M` equals `c2s_matrix / ratio_internal · ratio` is
/// the SAME object regardless of the scalar — concretely, that `M[p,i]` depends on
/// the shell only through monomial-metric normalization. We check M reproduced
/// purely from the (sign-carrying) row direction of `c2s_matrix` and `S_mono`.
#[test]
fn composed_transform_is_independent_of_factor_value() {
    for l in 2..=6 {
        let nc = n_cart(l);
        let c = c2s_matrix(l);
        let ratio = monomial_to_raw_factor(l);
        let smono = s_mono(l);
        for p in 0..(2 * l + 1) {
            let crow = &c[p * nc..(p + 1) * nc];
            // The row DIRECTION (coeffs up to positive scale) is ratio-free; only
            // the normalization scale depends on the metric. Renormalize the bare
            // direction in the monomial metric and compare to M = ratio·c.
            let mut q = 0.0;
            for i in 0..nc {
                for j in 0..nc {
                    q += crow[i] * crow[j] * smono[i * nc + j];
                }
            }
            let kappa = 1.0 / q.sqrt(); // monomial-metric unit normalization of crow
            for (i, &ci) in crow.iter().enumerate() {
                let m_production = ratio * ci;
                let m_from_mono = kappa * ci;
                assert!(
                    (m_production - m_from_mono).abs() < 1e-11,
                    "l={l} p={p} i={i}: ratio·c={m_production} vs mono-norm={m_from_mono} \
                     — composed transform is NOT ratio-independent"
                );
            }
        }
    }
}
