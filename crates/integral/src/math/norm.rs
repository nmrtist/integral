//! Gaussian-type-orbital normalization constants.
//!
//! ## Conventions
//!
//! A primitive Cartesian GTO centred at the origin is
//! `g(r) = x^{lx} y^{ly} z^{lz} exp(-α r²)`. Its self-overlap is
//!
//! ```text
//!   ⟨g|g⟩ = (2lx-1)!!(2ly-1)!!(2lz-1)!! / (4α)^L · (π/2α)^{3/2},   L = lx+ly+lz
//! ```
//!
//! so the unit-normalization constant is
//!
//! ```text
//!   N(α; lx,ly,lz) = (2α/π)^{3/4} (4α)^{L/2}
//!                    / sqrt((2lx-1)!!(2ly-1)!!(2lz-1)!!).
//! ```
//!
//! Note that within a shell `N` depends on the individual `(lx,ly,lz)` only
//! through the double-factorial denominator. The component `x^l` (i.e.
//! `(l,0,0)`) has denominator `(2l-1)!!`. integral normalizes every shell by the
//! constant `N(α; l,0,0)` and lets the remaining per-component factor fall out
//! of the recurrence (the shell-level convention; see `crate::engine`).

/// Double factorial `n!! = n·(n-2)·(n-4)···`, with `(-1)!! = 0!! = 1`.
///
/// Returned as `f64` to sidestep integer overflow; exact for the small `n`
/// arising from angular momenta up to the specialization cap.
#[must_use]
pub fn double_factorial(n: i64) -> f64 {
    let mut k = n;
    let mut acc = 1.0_f64;
    while k > 1 {
        acc *= k as f64;
        k -= 2;
    }
    acc
}

/// Normalization constant `N(α; lx,ly,lz)` of a primitive Cartesian GTO.
#[must_use]
pub fn cart_norm(alpha: f64, lx: usize, ly: usize, lz: usize) -> f64 {
    let l = (lx + ly + lz) as i32;
    let df = double_factorial(2 * lx as i64 - 1)
        * double_factorial(2 * ly as i64 - 1)
        * double_factorial(2 * lz as i64 - 1);
    let two_alpha_over_pi = 2.0 * alpha / std::f64::consts::PI;
    two_alpha_over_pi.powf(0.75) * (4.0 * alpha).powi(l).sqrt() / df.sqrt()
}

/// Shell-level normalization `N(α; l,0,0)`: the constant applied to a whole
/// shell of angular momentum `l` (the `x^l` component's norm).
#[must_use]
pub fn shell_norm(alpha: f64, l: usize) -> f64 {
    cart_norm(alpha, l, 0, 0)
}

/// Purely-angular factor that rescales the Cartesian component `(lx,ly,lz)`
/// relative to the shell's `(l,0,0)` component, so that *every* Cartesian
/// component becomes individually unit-normalized:
///
/// ```text
///   g(lx,ly,lz) = sqrt( (2l-1)!! / ((2lx-1)!!(2ly-1)!!(2lz-1)!!) ),   l = lx+ly+lz.
/// ```
///
/// This is the per-component unit-normalization convention. It is `1` for `s` and
/// for every `p` component, and exceeds `1` for off-axis components of `d` and
/// above. It is independent of the exponent `α`.
#[must_use]
pub fn cart_unit_factor(lx: usize, ly: usize, lz: usize) -> f64 {
    let l = lx + ly + lz;
    let df = double_factorial(2 * lx as i64 - 1)
        * double_factorial(2 * ly as i64 - 1)
        * double_factorial(2 * lz as i64 - 1);
    (double_factorial(2 * l as i64 - 1) / df).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn double_factorial_small_values() {
        assert_eq!(double_factorial(-1), 1.0);
        assert_eq!(double_factorial(0), 1.0);
        assert_eq!(double_factorial(1), 1.0);
        assert_eq!(double_factorial(3), 3.0);
        assert_eq!(double_factorial(5), 15.0);
        assert_eq!(double_factorial(7), 105.0);
    }

    /// A normalized primitive must have unit self-overlap. Compute the overlap
    /// independently from the closed form and check `N² ⟨g|g⟩_raw = 1`.
    #[test]
    fn normalized_primitive_has_unit_self_overlap() {
        let raw_self_overlap = |alpha: f64, lx: usize, ly: usize, lz: usize| {
            let l = (lx + ly + lz) as i32;
            let df = double_factorial(2 * lx as i64 - 1)
                * double_factorial(2 * ly as i64 - 1)
                * double_factorial(2 * lz as i64 - 1);
            df / (4.0 * alpha).powi(l) * (std::f64::consts::PI / (2.0 * alpha)).powf(1.5)
        };
        for &alpha in &[0.1, 0.8, 1.0, 3.3, 25.0] {
            for &(lx, ly, lz) in &[(0, 0, 0), (1, 0, 0), (0, 1, 1), (2, 0, 0), (1, 1, 1)] {
                let n = cart_norm(alpha, lx, ly, lz);
                let s = n * n * raw_self_overlap(alpha, lx, ly, lz);
                assert!(
                    (s - 1.0).abs() < 1e-13,
                    "alpha={alpha} l=({lx}{ly}{lz}) s={s}"
                );
            }
        }
    }

    /// Tight-core regime: heavy-element basis sets carry very tight core
    /// primitives (α ~ 1e5–1e8). `cart_norm` must stay finite and the normalized
    /// primitive must keep unit self-overlap there — the `(4α)^l` intermediate
    /// (≈4e57 at α=1e9, l=6) and `N` (≈2.5e33) are both well within f64 range, and
    /// the `.powi(l).sqrt()` ordering avoids squaring a large norm. Regression
    /// guard against a future ordering change that would overflow or lose
    /// precision at the tight core.
    #[test]
    fn tight_core_normalization_is_stable() {
        let raw_self_overlap = |alpha: f64, lx: usize, ly: usize, lz: usize| {
            let l = (lx + ly + lz) as i32;
            let df = double_factorial(2 * lx as i64 - 1)
                * double_factorial(2 * ly as i64 - 1)
                * double_factorial(2 * lz as i64 - 1);
            df / (4.0 * alpha).powi(l) * (std::f64::consts::PI / (2.0 * alpha)).powf(1.5)
        };
        for &alpha in &[1e3, 1e5, 1e7, 1e8, 1e9] {
            for &(lx, ly, lz) in &[(0, 0, 0), (5, 0, 0), (6, 0, 0), (2, 2, 2), (3, 2, 1)] {
                let n = cart_norm(alpha, lx, ly, lz);
                assert!(
                    n.is_finite() && n > 0.0,
                    "N non-finite at α={alpha} l=({lx}{ly}{lz}): {n}"
                );
                let s = n * n * raw_self_overlap(alpha, lx, ly, lz);
                assert!(
                    (s - 1.0).abs() < 1e-12,
                    "α={alpha} l=({lx}{ly}{lz}) self-overlap={s}"
                );
            }
        }
    }
}
