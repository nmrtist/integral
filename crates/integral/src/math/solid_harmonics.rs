//! Cartesian → real solid-harmonic (`c2s`) transform coefficients.
//!
//! This module builds the matrix that maps a shell's Cartesian components to its
//! `(2l+1)` **real spherical-harmonic** components.
//!
//! Reference: H. B. Schlegel & M. J. Frisch, *Int. J. Quantum Chem.* **54**, 83
//! (1995) — equivalently the real-form solid harmonics, Racah-normalized, of
//! standard references.
//!
//! ## What the transform produces
//!
//! The `(2l+1)` components are **unit-normalized** real spherical harmonics (each
//! spherical AO has unit self-overlap), ordered and signed as described in
//! [`m_order`].
//!
//! ## Two explicitly-separable normalization factors
//!
//! The conversion from an integral Cartesian integral block to a spherical block
//! factors into **two** scalars that are kept
//! separate (not merged into one opaque matrix), so each can be documented and
//! tested on its own:
//!
//! 1. **monomial → raw** — one per-shell scalar [`monomial_to_raw_factor`]
//!    `= sqrt(4π/(2l+1))` for `l ≥ 2`, `1` otherwise. integral normalizes the
//!    Cartesian *monomial* (the `x^l` component has unit self-overlap), whereas the
//!    solid-harmonic transform assumes *solid-harmonic* (raw) radial normalization.
//!    This scalar bridges the two.
//! 2. **raw-Cartesian → real-solid-harmonic** — the `(2l+1) × n_cart(l)` matrix
//!    [`c2s_matrix`], which assumes its input Cartesian block is **raw** (solid-
//!    harmonic) normalized and yields the unit-normalized spherical block.
//!
//! The driver composes them: scale a monomial Cartesian block by factor (1) per
//! AO index, then apply matrix (2) on each index. See `integral`'s integral builders.

use crate::math::am::{cart_components, n_cart};
use crate::math::norm::cart_norm;

/// Exact factorial `n!` as `f64` (exact for the small `n` here, `l ≤ 6`).
fn factorial(n: usize) -> f64 {
    (1..=n).map(|k| k as f64).product()
}

/// Binomial coefficient `C(n, k)` as `f64` (returns `0` for `k > n`).
fn binom(n: i64, k: i64) -> f64 {
    if k < 0 || k > n || n < 0 {
        return 0.0;
    }
    factorial(n as usize) / (factorial(k as usize) * factorial((n - k) as usize))
}

/// `cos(n·π/2)` evaluated exactly on the integers (`0`, `±1`), avoiding the
/// round-off of `f64::cos` so the assembled coefficients are exact rationals.
fn cos_half_pi(n: i64) -> f64 {
    match n.rem_euclid(4) {
        0 => 1.0,
        2 => -1.0,
        _ => 0.0, // odd
    }
}

/// `sin(n·π/2)` evaluated exactly on the integers (`0`, `±1`).
fn sin_half_pi(n: i64) -> f64 {
    match n.rem_euclid(4) {
        1 => 1.0,
        3 => -1.0,
        _ => 0.0, // even
    }
}

/// Coefficient of the bare monomial `x^lx y^ly z^lz` in the **Racah-normalized**
/// real solid harmonic of degree `l` and signed order `m`.
///
/// Convention (real form, standard references): `m ≥ 0` is the **cosine** type
/// `C_l^m`, `m < 0` is the **sine** type `S_l^{|m|}`. With `M = |m|`,
///
/// ```text
///   C_l^M / S_l^M = sqrt((2 - δ_{M0})(l-M)!/(l+M)!) · Π_l^M(z) · {A_M | B_M}(x,y)
///   Π_l^M(z)      = Σ_{k=0}^{⌊(l-M)/2⌋} γ_{lk}^{(M)} r^{2k} z^{l-2k-M}
///   γ_{lk}^{(M)}  = (-1)^k 2^{-l} C(l,k) C(2l-2k,l) (l-2k)!/(l-2k-M)!
///   A_M(x,y)      = Σ_{p=0}^{M} C(M,p) x^p y^{M-p} cos((M-p)π/2)
///   B_M(x,y)      = Σ_{p=0}^{M} C(M,p) x^p y^{M-p} sin((M-p)π/2)
/// ```
///
/// `r^{2k} = (x²+y²+z²)^k` is expanded with the multinomial theorem so the whole
/// product is collected into bare monomials. Returns `0` unless the monomial's
/// powers can be produced (the parity condition `lx+ly-M` even is implicit).
fn racah_real_solid_harmonic_coeff(l: usize, m: i64, lx: usize, ly: usize, lz: usize) -> f64 {
    debug_assert_eq!(lx + ly + lz, l);
    let big_m = m.unsigned_abs() as usize;
    if big_m > l {
        return 0.0;
    }
    let cosine = m >= 0;
    let norm = if big_m == 0 {
        (factorial(l - big_m) / factorial(l + big_m)).sqrt()
    } else {
        (2.0 * factorial(l - big_m) / factorial(l + big_m)).sqrt()
    };

    let mut acc = 0.0_f64;
    // Π_l^M(z): sum over k with γ_{lk}, contributing r^{2k} z^{l-2k-M}.
    let kmax = (l - big_m) / 2;
    for k in 0..=kmax {
        let zpow = l - 2 * k - big_m; // power of z from the Π_l^M term
        let gamma = (if k % 2 == 0 { 1.0 } else { -1.0 })
            * 2f64.powi(-(l as i32))
            * binom(l as i64, k as i64)
            * binom(2 * l as i64 - 2 * k as i64, l as i64)
            * factorial(l - 2 * k)
            / factorial(l - 2 * k - big_m);
        if gamma == 0.0 {
            continue;
        }
        // r^{2k} = Σ_{a+b+c=k} k!/(a!b!c!) x^{2a} y^{2b} z^{2c}.
        for a in 0..=k {
            for b in 0..=(k - a) {
                let c = k - a - b;
                // A_M / B_M term: Σ_p C(M,p) x^p y^{M-p} {cos|sin}((M-p)π/2).
                for p in 0..=big_m {
                    let q = big_m - p; // power of y from the in-plane part
                                       // Total monomial powers from this term:
                    let tx = 2 * a + p;
                    let ty = 2 * b + q;
                    let tz = 2 * c + zpow;
                    if tx != lx || ty != ly || tz != lz {
                        continue;
                    }
                    let trig = if cosine {
                        cos_half_pi(q as i64)
                    } else {
                        sin_half_pi(q as i64)
                    };
                    if trig == 0.0 {
                        continue;
                    }
                    let multinomial = factorial(k) / (factorial(a) * factorial(b) * factorial(c));
                    acc += gamma * multinomial * binom(big_m as i64, p as i64) * trig;
                }
            }
        }
    }
    norm * acc
}

/// The signed-`m` ordering of the `(2l+1)` real spherical components. Signs follow
/// the Racah real solid harmonics here, so no Condon–Shortley flip is needed:
///
/// - **`l = 1`** uses the special `p` order `(x, y, z) = (m=+1, m=-1, m=0)`.
/// - **`l ≠ 1`** is ascending `m = -l, -l+1, …, 0, …, +l` (so `l = 0` is `[0]`).
///
/// `m > 0` selects the cosine type `C_l^m`, `m < 0` the sine type `S_l^{|m|}`.
#[must_use]
pub fn m_order(l: usize) -> Vec<i64> {
    if l == 1 {
        // p special-case: (px, py, pz) = (m=+1, m=-1, m=0).
        return vec![1, -1, 0];
    }
    (-(l as i64)..=(l as i64)).collect()
}

/// Per-shell scalar converting an **integral-monomial**-normalized Cartesian block
/// to the **raw** (solid-harmonic) normalization that [`c2s_matrix`] assumes:
/// `sqrt(4π/(2l+1))` for `l ≥ 2`, else `1`.
///
#[must_use]
pub fn monomial_to_raw_factor(l: usize) -> f64 {
    if l < 2 {
        1.0
    } else {
        (4.0 * std::f64::consts::PI / (2 * l + 1) as f64).sqrt()
    }
}

/// `1D` Gaussian moment `∫ x^n e^{-β x²} dx` over `ℝ` (`0` for odd `n`).
fn gaussian_moment_1d(n: usize, beta: f64) -> f64 {
    if n % 2 == 1 {
        return 0.0;
    }
    // (n-1)!! / (2β)^{n/2} · sqrt(π/β)
    let dfac = crate::math::norm::double_factorial(n as i64 - 1);
    dfac / (2.0 * beta).powi(n as i32 / 2) * (std::f64::consts::PI / beta).sqrt()
}

/// Self-overlap of two **raw** (solid-harmonic)-normalized Cartesian components
/// `(lx,ly,lz)` and `(lx',ly',lz')` of a degree-`l` shell at exponent `α = 1`.
///
/// In the raw normalization the per-component pattern equals integral's monomial
/// pattern and differs only by the per-shell scalar `4π/(2l+1)`, so
/// `S^{raw} = (4π/(2l+1)) · N_mono² · ⟨monomial|monomial⟩_bare`. The `α`-cancels
/// in the c2s normalization, so any fixed `α` works; `α = 1` is used.
fn raw_self_overlap(l: usize, ci: [usize; 3], cj: [usize; 3]) -> f64 {
    let beta = 2.0; // exponent sum 2α with α = 1
    let bare = gaussian_moment_1d(ci[0] + cj[0], beta)
        * gaussian_moment_1d(ci[1] + cj[1], beta)
        * gaussian_moment_1d(ci[2] + cj[2], beta);
    let n_mono = cart_norm(1.0, l, 0, 0);
    // The raw normalization differs from integral-monomial by the per-shell scalar
    // `monomial_to_raw_factor` (= 1 for l ≤ 1, sqrt(4π/(2l+1)) for l ≥ 2), so the
    // raw self-overlap is that scalar squared times the monomial overlap.
    let raw_shell = monomial_to_raw_factor(l).powi(2);
    raw_shell * n_mono * n_mono * bare
}

/// The `c2s` matrix for angular momentum `l`: a `(2l+1) × n_cart(l)` row-major
/// matrix mapping a **raw** (solid-harmonic)-normalized Cartesian block
/// (components in [`crate::math::am::cart_components`] order) to the **unit-normalized
/// real spherical** block (components in [`m_order`] order).
///
/// Apply as `O_sph = C · O_raw · Cᵀ` (one-electron) or on each index (ERI).
///
/// Construction (all analytic):
/// 1. Each spherical row is the Racah real solid harmonic
///    `racah_real_solid_harmonic_coeff` over the Cartesian monomials.
/// 2. Each row is rescaled by a single per-`(l,m)` constant so the spherical
///    function is **unit-normalized** against the raw Cartesian self-overlap
///    `raw_self_overlap` — making the diagonal of `C · S^{raw} · Cᵀ` exactly 1.
#[must_use]
pub fn c2s_matrix(l: usize) -> Vec<f64> {
    let comps = cart_components(l);
    let ncart = n_cart(l);
    let ms = m_order(l);
    let mut mat = vec![0.0_f64; ms.len() * ncart];

    for (row, &m) in ms.iter().enumerate() {
        // Racah coefficients of this spherical component over the monomials.
        let mut coeffs = vec![0.0_f64; ncart];
        for (i, c) in comps.iter().enumerate() {
            coeffs[i] = racah_real_solid_harmonic_coeff(l, m, c[0], c[1], c[2]);
        }
        // Unit-normalize against the raw-Cartesian self-overlap:
        // κ² · Σ_{i,j} coeffs_i coeffs_j S^{raw}_{ij} = 1.
        let mut q = 0.0_f64;
        for (i, ci) in comps.iter().enumerate() {
            if coeffs[i] == 0.0 {
                continue;
            }
            for (j, cj) in comps.iter().enumerate() {
                if coeffs[j] == 0.0 {
                    continue;
                }
                q += coeffs[i] * coeffs[j] * raw_self_overlap(l, *ci, *cj);
            }
        }
        let kappa = 1.0 / q.sqrt();
        for i in 0..ncart {
            mat[row * ncart + i] = kappa * coeffs[i];
        }
    }
    mat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dz2_matches_known_unnormalized_form() {
        // C_2^0 (Racah) = z² - ½(x²+y²). cart_components(2) = xx,xy,xz,yy,yz,zz.
        let c = |lx, ly, lz| racah_real_solid_harmonic_coeff(2, 0, lx, ly, lz);
        assert!((c(0, 0, 2) - 1.0).abs() < 1e-14); // z²
        assert!((c(2, 0, 0) + 0.5).abs() < 1e-14); // x²
        assert!((c(0, 2, 0) + 0.5).abs() < 1e-14); // y²
        assert!(c(1, 1, 0).abs() < 1e-14); // xy
        assert!(c(1, 0, 1).abs() < 1e-14); // xz
        assert!(c(0, 1, 1).abs() < 1e-14); // yz
    }

    #[test]
    fn p_shell_is_identity_up_to_order() {
        // l=1: real solid harmonics are just x, y, z (Racah norm = 1). The p order
        // is (px, py, pz) = (m=+1, m=-1, m=0), so the rows pick x, y, z in that
        // order. Unit-normalized against raw (= monomial for l<2) self-
        // overlap, each row is a unit vector selecting one Cartesian.
        let c = c2s_matrix(1); // 3x3 row-major; cart order x(0), y(1), z(2).
        let row = |r: usize| &c[r * 3..r * 3 + 3];
        // row0 (m=+1) ~ x
        assert!((row(0)[0].abs() - 1.0).abs() < 1e-13);
        assert!(row(0)[1].abs() < 1e-13 && row(0)[2].abs() < 1e-13);
        // row1 (m=-1) ~ y
        assert!((row(1)[1].abs() - 1.0).abs() < 1e-13);
        // row2 (m=0) ~ z
        assert!((row(2)[2].abs() - 1.0).abs() < 1e-13);
    }

    /// The spherical functions must be **orthonormal**: `C · S^{raw} · Cᵀ = I`.
    /// This pins normalization and mutual orthogonality analytically.
    #[test]
    fn spherical_functions_are_orthonormal_against_raw_overlap() {
        for l in 0..=6 {
            let c = c2s_matrix(l);
            let comps = cart_components(l);
            let ncart = n_cart(l);
            let nsph = 2 * l + 1;
            // Build S^{raw} (ncart x ncart).
            let mut s = vec![0.0; ncart * ncart];
            for (i, ci) in comps.iter().enumerate() {
                for (j, cj) in comps.iter().enumerate() {
                    s[i * ncart + j] = raw_self_overlap(l, *ci, *cj);
                }
            }
            // G = C S Cᵀ, must equal I_{nsph}.
            for p in 0..nsph {
                for q in 0..nsph {
                    let mut g = 0.0;
                    for i in 0..ncart {
                        let mut si = 0.0;
                        for j in 0..ncart {
                            si += s[i * ncart + j] * c[q * ncart + j];
                        }
                        g += c[p * ncart + i] * si;
                    }
                    let expect = if p == q { 1.0 } else { 0.0 };
                    assert!(
                        (g - expect).abs() < 1e-12,
                        "l={l} G[{p},{q}]={g} expected {expect}"
                    );
                }
            }
        }
    }

    #[test]
    fn matrix_shape_and_component_counts() {
        for l in 0..=6 {
            let c = c2s_matrix(l);
            assert_eq!(c.len(), (2 * l + 1) * n_cart(l));
            assert_eq!(m_order(l).len(), 2 * l + 1);
        }
    }

    #[test]
    fn monomial_to_raw_factor_values() {
        assert_eq!(monomial_to_raw_factor(0), 1.0);
        assert_eq!(monomial_to_raw_factor(1), 1.0);
        // d: sqrt(4π/5)
        let pi = std::f64::consts::PI;
        assert!((monomial_to_raw_factor(2) - (4.0 * pi / 5.0).sqrt()).abs() < 1e-14);
        assert!((monomial_to_raw_factor(6) - (4.0 * pi / 13.0).sqrt()).abs() < 1e-14);
    }
}
