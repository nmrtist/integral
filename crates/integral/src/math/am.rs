//! Angular-momentum bookkeeping: Cartesian component counts and orderings.
//!
//! ## Cartesian ordering convention
//!
//! For a shell of angular momentum `l`, the `(l+1)(l+2)/2` Cartesian components
//! `(lx, ly, lz)` with `lx + ly + lz = l` are enumerated with `lx` descending in
//! the outer loop and `ly` descending in the inner loop. Examples:
//!
//! - `p`: `x, y, z`
//! - `d`: `xx, xy, xz, yy, yz, zz`
//! - `f`: `xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz`

/// Number of Cartesian components for angular momentum `l`: `(l+1)(l+2)/2`.
#[must_use]
pub const fn n_cart(l: usize) -> usize {
    (l + 1) * (l + 2) / 2
}

/// Cartesian power triplets `(lx, ly, lz)` for shell `l`, in the canonical order.
///
/// See the module docs for the exact ordering.
#[must_use]
pub fn cart_components(l: usize) -> Vec<[usize; 3]> {
    let mut out = Vec::with_capacity(n_cart(l));
    for lx in (0..=l).rev() {
        for ly in (0..=(l - lx)).rev() {
            out.push([lx, ly, l - lx - ly]);
        }
    }
    out
}

/// Index of the Cartesian component `[lx, ly, lz]` within [`cart_components`]`(l)`
/// for `l = lx + ly + lz`, in the canonical ordering.
///
/// Closed form for the ordering (`lx` descending outer, `ly` descending inner):
/// the components with a larger `lx'` number `(l-lx)(l-lx+1)/2`, and within a
/// fixed `lx` the offset is `(l-lx) - ly = lz`, so
///
/// ```text
///   cart_index([lx, ly, lz]) = (l-lx)(l-lx+1)/2 + lz,   l = lx+ly+lz.
/// ```
///
/// This is the inverse of indexing into `cart_components(l)`:
/// `cart_components(l)[cart_index(c)] == c`. It is the placement primitive for the
/// angular-momentum raise/lower steps used by the geometric-derivative engine
/// (`∂/∂A_i χ_a = 2α·χ_{a+1_i} − a_i·χ_{a−1_i}`): the raised component
/// `a + 1_i` lands at `cart_index` in shell `l+1`, the lowered `a − 1_i` at
/// `cart_index` in shell `l-1`.
#[must_use]
pub fn cart_index(c: [usize; 3]) -> usize {
    let l = c[0] + c[1] + c[2];
    let d = l - c[0]; // = ly + lz
    d * (d + 1) / 2 + c[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_match_formula() {
        assert_eq!(n_cart(0), 1);
        assert_eq!(n_cart(1), 3);
        assert_eq!(n_cart(2), 6);
        assert_eq!(n_cart(3), 10);
        assert_eq!(n_cart(4), 15);
    }

    #[test]
    fn p_ordering_is_xyz() {
        assert_eq!(cart_components(1), vec![[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    }

    #[test]
    fn d_ordering_is_canonical() {
        // xx, xy, xz, yy, yz, zz
        assert_eq!(
            cart_components(2),
            vec![
                [2, 0, 0],
                [1, 1, 0],
                [1, 0, 1],
                [0, 2, 0],
                [0, 1, 1],
                [0, 0, 2]
            ]
        );
    }

    #[test]
    fn component_count_is_consistent() {
        for l in 0..=6 {
            let comps = cart_components(l);
            assert_eq!(comps.len(), n_cart(l));
            for c in comps {
                assert_eq!(c[0] + c[1] + c[2], l);
            }
        }
    }

    #[test]
    fn cart_index_inverts_cart_components() {
        for l in 0..=6 {
            for (i, c) in cart_components(l).into_iter().enumerate() {
                assert_eq!(cart_index(c), i, "l={l} component {c:?}");
            }
        }
    }

    #[test]
    fn cart_index_tracks_raise_and_lower() {
        // Raising/lowering a d-component lands at the right index in l±1.
        // xy = [1,1,0] is index 1 in d; raise along x -> [2,1,0] in f.
        assert_eq!(cart_index([1, 1, 0]), 1);
        let f = cart_components(3);
        assert_eq!(f[cart_index([2, 1, 0])], [2, 1, 0]);
        // Lower xy along y -> [1,0,0] in p (index 0).
        assert_eq!(cart_index([1, 0, 0]), 0);
    }
}
