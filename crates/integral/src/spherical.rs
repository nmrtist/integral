//! Cartesian → spherical (`c2s`) transform applied to contracted integral
//! blocks.
//!
//! The integral engines always produce **Cartesian** blocks (monomial
//! normalization). For shells declared [`ShellKind::Spherical`], the driver
//! post-processes the contracted Cartesian block into the `2l+1` real
//! spherical-harmonic components by contracting each Cartesian index with the
//! shell's transform matrix. The recurrence cores are untouched.
//!
//! ## The per-shell transform (two separable factors)
//!
//! For a spherical shell of angular momentum `l`, the effective per-index
//! transform applied to an **integral-monomial** Cartesian block is the composition
//! of the two factors `crate::math` exposes and tests separately:
//!
//! ```text
//!   M(l) = monomial_to_raw_factor(l) · c2s_matrix(l)
//!          \_______ per-shell scalar _______/  \__ raw→spherical __/
//! ```
//!
//! - [`monomial_to_raw_factor`] rescales the integral-monomial block to the raw
//!   (solid-harmonic) normalization (a single per-shell scalar);
//! - [`c2s_matrix`] is the raw-Cartesian → unit-normalized real-spherical matrix.
//!
//! Keeping them separate (not pre-merged into one opaque table) lets each be
//! validated on its own — the scalar against its analytic value, the matrix
//! against the mpmath reference and orthonormality.

use crate::math::solid_harmonics::{c2s_matrix, monomial_to_raw_factor};

use crate::shell::{Shell, ShellKind};

/// The effective per-index transform for a shell, row-major `n_out × n_cart`:
/// `None` for a Cartesian shell (identity — no transform), or
/// `Some(monomial_to_raw_factor(l) · c2s_matrix(l))` for a spherical
/// shell (shape `(2l+1) × n_cart(l)`).
pub(crate) fn shell_transform(s: &Shell) -> Option<Vec<f64>> {
    match s.kind() {
        ShellKind::Cartesian => None,
        ShellKind::Spherical => {
            let l = s.l();
            let ratio = monomial_to_raw_factor(l);
            // M = ratio · C: the two separately-defined/tested `crate::math` factors,
            // composed here at application time (see module docs).
            Some(c2s_matrix(l).iter().map(|&c| ratio * c).collect())
        }
    }
}

/// Contract one axis of a row-major tensor with a `n_out × n_in` matrix `mat`,
/// where `n_in = dims[axis]`. Returns the new tensor and its updated `dims`.
///
/// `out[.., q, ..] = Σ_j mat[q, j] · tensor[.., j, ..]` over the `axis` index.
fn apply_axis(
    tensor: &[f64],
    dims: &[usize],
    axis: usize,
    mat: &[f64],
    n_out: usize,
) -> (Vec<f64>, Vec<usize>) {
    let n_in = dims[axis];
    debug_assert_eq!(mat.len(), n_out * n_in);
    let outer: usize = dims[..axis].iter().product();
    let inner: usize = dims[axis + 1..].iter().product();

    let mut new_dims = dims.to_vec();
    new_dims[axis] = n_out;
    let mut out = vec![0.0_f64; outer * n_out * inner];

    for o in 0..outer {
        for q in 0..n_out {
            let dst = (o * n_out + q) * inner;
            for j in 0..n_in {
                let m = mat[q * n_in + j];
                if m == 0.0 {
                    continue;
                }
                let src = (o * n_in + j) * inner;
                for s in 0..inner {
                    out[dst + s] += m * tensor[src + s];
                }
            }
        }
    }
    (out, new_dims)
}

/// Transform a row-major Cartesian `block` (with Cartesian dims `dims`) into the
/// function-space block by applying each axis's per-shell transform `mats[k]`
/// (`None` = keep Cartesian). Returns the transformed block; its new dims are the
/// per-shell `n_func`. Used for both 1e (2 axes) and ERI (4 axes).
///
/// Takes the transforms as borrowed slices so dense builders can compute each
/// shell's matrix **once** ([`shell_transform`] builds the `c2s` matrix — Racah
/// coefficients plus an `O(n_cart²)` normalization — so rebuilding it per quartet
/// dominated the spherical driver) and share it across the `O(n⁴)` quartet loop.
pub(crate) fn transform_block(
    block: Vec<f64>,
    dims: &[usize],
    mats: &[Option<&[f64]>],
) -> Vec<f64> {
    debug_assert_eq!(dims.len(), mats.len());
    let mut cur = block;
    let mut cur_dims = dims.to_vec();
    for (axis, m) in mats.iter().enumerate() {
        if let Some(mat) = m {
            let n_out = mat.len() / dims[axis];
            let (nt, nd) = apply_axis(&cur, &cur_dims, axis, mat, n_out);
            cur = nt;
            cur_dims = nd;
        }
    }
    cur
}

/// [`apply_axis`] into a caller-provided buffer (grown as needed, the active
/// region zero-filled before the identical accumulation), for the dense ERI
/// drivers' allocation-free pipeline. Same arithmetic in the same order as
/// [`apply_axis`] — zero-init + `+=` is exactly the fresh-`vec![0.0; …]` path —
/// so the results are bit-identical.
fn apply_axis_into(
    tensor: &[f64],
    dims: &[usize; 4],
    axis: usize,
    mat: &[f64],
    n_out: usize,
    out: &mut Vec<f64>,
) {
    let n_in = dims[axis];
    debug_assert_eq!(mat.len(), n_out * n_in);
    let outer: usize = dims[..axis].iter().product();
    let inner: usize = dims[axis + 1..].iter().product();
    let total = outer * n_out * inner;
    if out.len() < total {
        out.resize(total, 0.0);
    }
    out[..total].fill(0.0);
    for o in 0..outer {
        for q in 0..n_out {
            let dst = (o * n_out + q) * inner;
            for j in 0..n_in {
                let m = mat[q * n_in + j];
                if m == 0.0 {
                    continue;
                }
                let src = (o * n_in + j) * inner;
                for s in 0..inner {
                    out[dst + s] += m * tensor[src + s];
                }
            }
        }
    }
}

/// 4-axis [`transform_block`] over caller-owned buffers: the input block lives
/// in `block` (logical length `dims` product), each transformed axis ping-pongs
/// between `block` and `tmp` via [`apply_axis_into`] + a `Vec` swap (pointer
/// swap, no allocation), and the result is left in `block`. Returns the
/// transformed block's logical length. Bit-identical to [`transform_block`]
/// (same per-axis arithmetic and axis order; only the buffer ownership moves).
pub(crate) fn transform_block4_into(
    block: &mut Vec<f64>,
    dims: [usize; 4],
    mats: &[Option<&[f64]>; 4],
    tmp: &mut Vec<f64>,
) -> usize {
    let mut cur_dims = dims;
    let mut cur_len: usize = dims.iter().product();
    for (axis, m) in mats.iter().enumerate() {
        if let Some(mat) = m {
            let n_out = mat.len() / dims[axis];
            apply_axis_into(&block[..cur_len], &cur_dims, axis, mat, n_out, tmp);
            cur_dims[axis] = n_out;
            cur_len = cur_dims.iter().product();
            std::mem::swap(block, tmp);
        }
    }
    cur_len
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cartesian_shell_is_identity() {
        let s = Shell::new(2, [0.0, 0.0, 0.0], vec![1.0], vec![1.0]).unwrap();
        assert!(shell_transform(&s).is_none());
    }

    #[test]
    fn spherical_d_transform_shape() {
        let s = Shell::new_spherical(2, [0.0, 0.0, 0.0], vec![1.0], vec![1.0]).unwrap();
        let m = shell_transform(&s).unwrap();
        assert_eq!(m.len(), 5 * 6); // (2l+1) x n_cart(2)
    }

    #[test]
    fn apply_axis_identity_roundtrip() {
        // 2x3 block, apply 3x3 identity on axis 1 -> unchanged.
        let block = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let id = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let (out, dims) = apply_axis(&block, &[2, 3], 1, &id, 3);
        assert_eq!(dims, vec![2, 3]);
        assert_eq!(out, block);
    }

    #[test]
    fn transform_block_left_and_right_axes() {
        // A 2x2 block; transform axis 0 with [[1,1]] (2->1) and axis 1 with
        // [[1,0],[0,1]] (identity). Result is row sums as a 1x2 block.
        let block = vec![1.0, 2.0, 3.0, 4.0];
        let sum_rows = [1.0, 1.0]; // 1x2
        let id = [1.0, 0.0, 0.0, 1.0]; // 2x2
        let out = transform_block(block, &[2, 2], &[Some(&sum_rows[..]), Some(&id[..])]);
        assert_eq!(out, vec![4.0, 6.0]); // [1+3, 2+4]
    }
}
