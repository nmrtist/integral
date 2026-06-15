//! Kernel-selectable batched grid-point Coulomb matrices
//! (`Basis::grid_coulomb_kernel[_into]`): bitwise Coulomb routing, an exact
//! independent cross-check of the erf kernel through the Gaussian-charge
//! identity, the ω → ∞ / ω → 0⁺ limits, symmetry, and the parallel-partition
//! contract.

use integral::math::norm::cart_norm;
use integral::{Basis, EriKernel, Shell};

/// Mixed s/p/d/f/g basis with contraction, repeated centers, and both
/// Cartesian and spherical shells (same fixture family as `grid_coulomb.rs`).
fn mixed_basis() -> Basis {
    Basis::new(vec![
        Shell::new(0, [0.0, 0.0, 0.0], vec![1.4, 0.5], vec![0.6, 0.5]).unwrap(),
        Shell::new(1, [0.6, -0.3, 0.2], vec![0.9, 0.3], vec![0.7, 0.4]).unwrap(),
        Shell::new_spherical(2, [-0.4, 0.7, -0.1], vec![1.1], vec![1.0]).unwrap(),
        Shell::new(3, [0.2, 0.5, -0.6], vec![0.8], vec![1.0]).unwrap(),
        Shell::new_spherical(4, [-0.3, -0.2, 0.4], vec![0.7], vec![1.0]).unwrap(),
    ])
}

/// ≥ 200 deterministic points (a 6×6×6 lattice around the molecule), including
/// one point on a shell center (the T → 0 Boys corner).
fn lattice_points() -> Vec<[f64; 3]> {
    let mut pts = vec![[0.0, 0.0, 0.0]];
    for i in 0..6 {
        for j in 0..6 {
            for k in 0..6 {
                pts.push([
                    -1.5 + 0.6 * i as f64,
                    -1.4 + 0.55 * j as f64,
                    -1.6 + 0.62 * k as f64,
                ]);
            }
        }
    }
    pts
}

/// Acceptance criterion: `grid_coulomb_kernel(.., Coulomb)` is **bitwise**
/// equal to `grid_coulomb(..)` over a ≥ 200-point batch.
#[test]
fn coulomb_kernel_is_bitwise_identical() {
    let basis = mixed_basis();
    let points = lattice_points();
    assert!(points.len() >= 200);
    let plain = basis.grid_coulomb(&points);
    let kernel = basis.grid_coulomb_kernel(&points, EriKernel::Coulomb);
    assert_eq!(plain.len(), kernel.len());
    for (i, (a, b)) in plain.iter().zip(&kernel).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "bitwise mismatch at element {i}");
    }
}

/// `grid_coulomb_kernel` vs `grid_coulomb_kernel_into` (fresh buffer) are
/// identical, and disjoint point sub-ranges reassemble the same tensor —
/// the parallel-partition contract, for the erf kernel.
#[test]
fn erf_into_matches_alloc_and_subranges() {
    let basis = mixed_basis();
    let points: Vec<[f64; 3]> = lattice_points().into_iter().take(24).collect();
    let k = EriKernel::Erf { omega: 0.8 };
    let nao = basis.nao();
    let whole = basis.grid_coulomb_kernel(&points, k);
    assert!(whole.iter().all(|v| v.is_finite()));
    let mut buf = vec![f64::NAN; points.len() * nao * nao];
    basis.grid_coulomb_kernel_into(&points, k, &mut buf);
    assert_eq!(whole, buf);
    let mut parts = vec![0.0; points.len() * nao * nao];
    let split = points.len() / 2;
    let (lo, hi) = parts.split_at_mut(split * nao * nao);
    basis.grid_coulomb_kernel_into(&points[..split], k, lo);
    basis.grid_coulomb_kernel_into(&points[split..], k, hi);
    assert_eq!(whole, parts);
}

/// **Exact independent cross-check** (no limits taken): the potential of a
/// normalized Gaussian charge `(ω²/π)^{3/2} e^{−ω²|r−r_g|²}` is exactly
/// `erf(ω|r−r_g|)/|r−r_g|`, so the erf grid matrices must equal the
/// plain-Coulomb 3-center integrals `(μν|g_ω)` with a single s-type auxiliary
/// of exponent `ω²` at `r_g`, scaled to unit charge — evaluated by the
/// independent ERI machinery (`Basis::eri_3c_block`). Agreement ≤ 1e-10.
#[test]
fn erf_matches_gaussian_charge_3c_coulomb() {
    let basis = mixed_basis();
    let points = [
        [0.3, -0.2, 0.5],
        [0.0, 0.0, 0.0], // on a shell center
        [-1.1, 0.8, -0.4],
        [2.0, 1.5, -1.2],
    ];
    let nao = basis.nao();
    let offs: Vec<usize> = {
        let mut o = vec![0];
        for s in basis.shells() {
            o.push(o.last().unwrap() + s.n_func());
        }
        o
    };
    for &omega in &[0.6_f64, 2.5] {
        let om2 = omega * omega;
        // Raw coefficient that makes the effective primitive coefficient equal
        // the unit-charge Gaussian normalization (ω²/π)^{3/2}.
        let coef = (om2 / std::f64::consts::PI).powf(1.5) / cart_norm(om2, 0, 0, 0);
        let aux = Basis::new(
            points
                .iter()
                .map(|&p| Shell::new(0, p, vec![om2], vec![coef]).unwrap())
                .collect(),
        );
        let a = basis.grid_coulomb_kernel(&points, EriKernel::Erf { omega });
        for g in 0..points.len() {
            let mat = &a[g * nao * nao..(g + 1) * nao * nao];
            for (si, sa) in basis.shells().iter().enumerate() {
                for (sj, sb) in basis.shells().iter().enumerate() {
                    let blk = basis.eri_3c_block(&aux, si, sj, g);
                    let (na, nb) = (sa.n_func(), sb.n_func());
                    for ia in 0..na {
                        for ib in 0..nb {
                            let got = mat[(offs[si] + ia) * nao + offs[sj] + ib];
                            let want = blk[ia * nb + ib]; // n_p = 1, P fastest
                            assert!(
                                (got - want).abs() <= 1e-10,
                                "ω={omega} point {g} shells ({si},{sj}) elem ({ia},{ib}): \
                                 {got} vs {want}"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// ω → ∞ limit: `F_m^ω = s^{m+1/2} F_m(sT)` with `s = ω²/(ρ+ω²)` deviates from
/// Coulomb by a relative `O((m+½)·ρ/ω²)`. With ρ ≲ 3 (the largest pair
/// exponent here) and ω = 10⁶ that is ≲ 2e-11 of the element scale, so the
/// documented tolerance 1e-9 (absolute, elements are O(1)) is comfortable.
#[test]
fn large_omega_approaches_coulomb() {
    let basis = mixed_basis();
    let points: Vec<[f64; 3]> = lattice_points().into_iter().take(20).collect();
    let coulomb = basis.grid_coulomb(&points);
    let erf = basis.grid_coulomb_kernel(&points, EriKernel::Erf { omega: 1e6 });
    for (i, (c, e)) in coulomb.iter().zip(&erf).enumerate() {
        assert!(e.is_finite());
        assert!(
            (c - e).abs() <= 1e-9,
            "element {i}: coulomb {c} vs erf(ω=1e6) {e}"
        );
    }
}

/// ω → 0⁺ limit: `erf(ωr)/r ≤ 2ω/√π` pointwise, so
/// `|A^g_{μν}| ≤ (2ω/√π)·∫|φ_μ φ_ν| ≤ (2ω/√π)·√(S_μμ S_νν)` (Cauchy–Schwarz)
/// — every element is bounded by the overlap diagonal and vanishes linearly.
#[test]
fn small_omega_vanishes_with_exact_bound() {
    let basis = mixed_basis();
    let points: Vec<[f64; 3]> = lattice_points().into_iter().take(10).collect();
    let nao = basis.nao();
    let s = basis.overlap();
    for &omega in &[1e-2_f64, 1e-4, 1e-6] {
        let a = basis.grid_coulomb_kernel(&points, EriKernel::Erf { omega });
        let cap = 2.0 * omega / std::f64::consts::PI.sqrt();
        for g in 0..points.len() {
            let mat = &a[g * nao * nao..(g + 1) * nao * nao];
            for i in 0..nao {
                for j in 0..nao {
                    let bound = cap * (s[i * nao + i] * s[j * nao + j]).sqrt();
                    let v = mat[i * nao + j];
                    assert!(v.is_finite());
                    assert!(
                        v.abs() <= bound * (1.0 + 1e-12),
                        "ω={omega} point {g} ({i},{j}): |{v}| > bound {bound}"
                    );
                }
            }
        }
    }
}

/// Every erf `A^g` is symmetric, exactly like the Coulomb matrices
/// (off-diagonal shell blocks mirrored; diagonal blocks to round-off).
#[test]
fn erf_matrices_are_symmetric() {
    let basis = mixed_basis();
    let points: Vec<[f64; 3]> = lattice_points().into_iter().take(8).collect();
    let nao = basis.nao();
    let a = basis.grid_coulomb_kernel(&points, EriKernel::Erf { omega: 1.1 });
    for g in 0..points.len() {
        let mat = &a[g * nao * nao..(g + 1) * nao * nao];
        let peak = mat.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
        for i in 0..nao {
            for j in 0..i {
                let (f, r) = (mat[i * nao + j], mat[j * nao + i]);
                assert!(
                    (f - r).abs() <= 1e-12 * f.abs().max(peak * 1e-3),
                    "point {g} ({i},{j}): {f} vs {r}"
                );
            }
        }
    }
}

/// The buffer-length contract matches `grid_coulomb_into`.
#[test]
#[should_panic(expected = "grid_coulomb output buffer")]
fn wrong_buffer_length_panics() {
    let basis = mixed_basis();
    let mut buf = vec![0.0; 7];
    basis.grid_coulomb_kernel_into(&[[0.0; 3]], EriKernel::Erf { omega: 0.5 }, &mut buf);
}

/// Invalid ω is rejected up front (same contract as the other kernel APIs).
#[test]
#[should_panic(expected = "EriKernel::Erf requires a finite omega > 0")]
fn invalid_omega_panics() {
    let basis = mixed_basis();
    let _ = basis.grid_coulomb_kernel(&[[0.0; 3]], EriKernel::Erf { omega: -1.0 });
}
