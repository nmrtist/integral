//! Geometric (nuclear-coordinate) first derivatives of integrals (L2).
//!
//! Derivatives are assembled from **shifted-angular-momentum value integrals**
//! via the Gaussian center-derivative relation — no new recurrences. For a
//! primitive Cartesian Gaussian `g_a` on center `A`,
//!
//! ```text
//!   ∂/∂A_i g_a = 2α · g_{a+1_i} − a_i · g_{a−1_i}
//! ```
//!
//! (raise / lower the angular momentum along axis `i`), with `α` the primitive
//! exponent and `a_i` the Cartesian power along `i`. Because the operator of a
//! one- or two-electron integral does not depend on the basis-function center,
//! `∂/∂A_i ⟨g_a|O|…⟩ = ⟨∂/∂A_i g_a|O|…⟩`, so the derivative of any integral block
//! is this same fixed linear combination of the value blocks evaluated at `l±1`
//! on the differentiated index. The existing engines compute those shifted
//! blocks; this module only combines them.
//!
//! ## The `2α` weight is folded by the caller (engine-agnostic)
//!
//! The `2α` weight is **per primitive**. Rather than bake an exponent into the
//! combiner — which would only work for a per-primitive engine — the **raised**
//! block passed to [`accumulate_center_derivative`] is already weighted by `2α`.
//! Every value engine takes a `scale`, so:
//!
//! - the per-primitive **Rys** path evaluates the raised primitive block with
//!   `scale = 2α` directly;
//! - the contracted **OS/HGP** path folds `2α` into each primitive's contraction
//!   coefficient (`coeff_p · 2α_p`) when it evaluates the raised contracted block.
//!
//! Both yield the same `2α`-weighted raised block, so one combiner serves both
//! engines and the one- and two-electron paths alike.
//!
//! ## Convention
//!
//! The result is `∂/∂(center of the differentiated shell)` — the basis-function
//! (center) derivative with respect to the position of the center the
//! differentiated angular-momentum index sits on. The sign / which-center
//! convention is documented above.

use crate::math::am::{cart_components, cart_index, n_cart};

/// Geometry of one center-derivative step: which angular-momentum index of a
/// row-major block is being differentiated, along which Cartesian axis, and how
/// that index is embedded in the block.
///
/// The block's indices are grouped as `[outer, l-index, inner]` (row-major): the
/// `l`-index runs over `n_cart(l)` Cartesian components, `outer` is the product
/// of all slower index dimensions and `inner` the product of all faster ones.
#[derive(Debug, Clone, Copy)]
pub struct AxisDeriv {
    /// Angular momentum of the differentiated index.
    pub l: usize,
    /// Cartesian axis of the derivative: `0 = x`, `1 = y`, `2 = z`.
    pub cart_axis: usize,
    /// Product of the dimensions of all slower indices.
    pub outer: usize,
    /// Product of the dimensions of all faster indices.
    pub inner: usize,
}

/// Accumulate the center-derivative of one row-major block along one Cartesian
/// axis of one angular-momentum index (geometry in `d`).
///
/// The two inputs are the **value** blocks with the differentiated index raised
/// and lowered. The raised block must **already be weighted by `2α`** (see the
/// module docs):
///
/// - `raised`: `2α`-weighted, dims `outer × n_cart(l+1) × inner`,
/// - `lowered`: dims `outer × n_cart(l-1) × inner` (pass `None` when `l == 0`).
///
/// For each component `a` (axis `d.cart_axis ∈ {0,1,2}`) the result is
///
/// ```text
///   out[outer, a, inner] += scale · ( raised[outer, a+1_i, inner]
///                                     − a_i · lowered[outer, a−1_i, inner] )
/// ```
///
/// accumulated into `out` (dims `outer × n_cart(l) × inner`). `scale` is the
/// contraction-coefficient product for this primitive (set/term).
///
/// This single primitive serves every index position: a one-electron bra
/// (`outer = 1`, `inner = n_cart(lb)`), a ket (`outer = n_cart(la)`,
/// `inner = 1`), and any of the four ERI indices (`outer`/`inner` the products of
/// the dimensions before/after it).
///
/// # Panics
/// Debug-asserts the input/output lengths are consistent and that `lowered` is
/// present whenever `l > 0`.
pub fn accumulate_center_derivative(
    d: AxisDeriv,
    scale: f64,
    raised: &[f64],
    lowered: Option<&[f64]>,
    out: &mut [f64],
) {
    let AxisDeriv {
        l,
        cart_axis,
        outer,
        inner,
    } = d;
    debug_assert!(cart_axis < 3, "cart_axis must be 0, 1, or 2");
    let nl = n_cart(l);
    let np1 = n_cart(l + 1);
    debug_assert_eq!(raised.len(), outer * np1 * inner, "raised block size");
    debug_assert_eq!(out.len(), outer * nl * inner, "output block size");
    if l > 0 {
        debug_assert!(lowered.is_some(), "lowered block required for l > 0");
        debug_assert_eq!(
            lowered.unwrap().len(),
            outer * n_cart(l - 1) * inner,
            "lowered block size"
        );
    }
    let nm1 = if l > 0 { n_cart(l - 1) } else { 0 };

    for (a_idx, a) in cart_components(l).into_iter().enumerate() {
        // Raise along the axis: a + 1_i lands at `r_idx` in shell l+1.
        let mut ar = a;
        ar[cart_axis] += 1;
        let r_idx = cart_index(ar);

        // Lower along the axis: a − 1_i in shell l-1, weighted by the power a_i.
        let pw = a[cart_axis];
        let l_idx = if pw >= 1 {
            let mut al = a;
            al[cart_axis] -= 1;
            Some(cart_index(al))
        } else {
            None
        };

        for o in 0..outer {
            let out_row = (o * nl + a_idx) * inner;
            let r_row = (o * np1 + r_idx) * inner;
            let l_row = l_idx.map(|li| (o * nm1 + li) * inner);
            for s in 0..inner {
                let mut v = raised[r_row + s];
                if let Some(lr) = l_row {
                    v -= (pw as f64) * lowered.unwrap()[lr + s];
                }
                out[out_row + s] += scale * v;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(l: usize, cart_axis: usize, outer: usize, inner: usize) -> AxisDeriv {
        AxisDeriv {
            l,
            cart_axis,
            outer,
            inner,
        }
    }

    // For l = 0 (s function) the derivative is purely the raise term. With the
    // raised p-block already weighted by 2α, the x-derivative picks raised_x.
    #[test]
    fn s_function_derivative_is_raise_only() {
        let raised = vec![10.0, 20.0, 30.0]; // 2α-weighted p block (n_cart(1)=3)
        let mut out = vec![0.0; 1];
        accumulate_center_derivative(desc(0, 0, 1, 1), 1.0, &raised, None, &mut out);
        assert_eq!(out[0], 10.0);
        let mut outy = vec![0.0; 1];
        accumulate_center_derivative(desc(0, 1, 1, 1), 1.0, &raised, None, &mut outy);
        assert_eq!(outy[0], 20.0);
    }

    // For a p function, ∂_x p_x = (2α·d)_xx − 1·s ; ∂_x p_y = (2α·d)_xy − 0.
    #[test]
    fn p_function_derivative_has_raise_and_lower() {
        // raised = 2α-weighted d block over n_cart(2)=6: xx,xy,xz,yy,yz,zz
        let d = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // lowered = s block (n_cart(0)=1)
        let s = vec![7.0];
        let mut out = vec![0.0; 3]; // p block: x,y,z
        accumulate_center_derivative(desc(1, 0, 1, 1), 1.0, &d, Some(&s), &mut out);
        // ∂_x p_x = d_xx − 1·s  (p_x=[1,0,0], a_x=1, raise->xx idx0, lower->s)
        assert_eq!(out[0], 1.0 - 7.0);
        // ∂_x p_y = d_xy − 0    (p_y=[0,1,0], a_x=0, raise->xy idx1)
        assert_eq!(out[1], 2.0);
        // ∂_x p_z = d_xz − 0    (p_z=[0,0,1], a_x=0, raise->xz idx2)
        assert_eq!(out[2], 3.0);
    }

    // `scale` multiplies the whole term, and repeated calls accumulate.
    #[test]
    fn scale_multiplies_and_accumulates() {
        let raised = vec![10.0, 0.0, 0.0];
        let mut out = vec![0.0; 1];
        accumulate_center_derivative(desc(0, 0, 1, 1), 2.0, &raised, None, &mut out);
        accumulate_center_derivative(desc(0, 0, 1, 1), 3.0, &raised, None, &mut out);
        assert_eq!(out[0], (2.0 + 3.0) * 10.0);
    }

    // Outer indexing: a ket-style derivative with outer=2, inner=1 lands in the
    // right rows.
    #[test]
    fn outer_indexing_places_correctly() {
        // l=0 differentiated index with outer=2: raised dims 2*n_cart(1)*1 = 6.
        let raised = vec![1.0, 0.0, 0.0, /*row1*/ 2.0, 0.0, 0.0];
        let mut out = vec![0.0; 2];
        accumulate_center_derivative(desc(0, 0, 2, 1), 1.0, &raised, None, &mut out);
        assert_eq!(out, vec![1.0, 2.0]);
    }

    // Inner indexing (ket of a 1e pair): outer=1, the differentiated index is
    // fast (inner=1) but here we exercise inner>1 to check striding.
    #[test]
    fn inner_indexing_places_correctly() {
        // Differentiate an l=0 index sitting BEFORE a size-2 inner index.
        // raised dims outer(1)*n_cart(1)(3)*inner(2) = 6, laid out [comp][inner].
        let raised = vec![
            11.0, 12.0, // px row over inner=2
            21.0, 22.0, // py
            31.0, 32.0, // pz
        ];
        let mut out = vec![0.0; 2];
        accumulate_center_derivative(desc(0, 0, 1, 2), 1.0, &raised, None, &mut out);
        assert_eq!(out, vec![11.0, 12.0]); // x-derivative picks the px row
    }
}
