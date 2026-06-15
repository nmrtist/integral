//! GTH nonlocal (Kleinman–Bylander) projector overlaps `⟨φ_μ | p_i^{lm}⟩`, analytic and
//! Bloch-summed.
//!
//! A GTH projector is `p_i^{lm}(r) = N_{il} · r^{l+2(i−1)} · e^{−r²/(2 r_l²)} · Y_lm(r̂)`
//! with `Y_lm` unit-normalized on the sphere and the radial normalization
//! `N_{il} = √2 / (r_l^{l+(4i−1)/2} √Γ(l+(4i−1)/2))` (so `∫|p|² d³r = 1`).
//!
//! The overlap is built **analytically** from the engine's exact Gaussian overlap rather
//! than on the grid, so each lattice image is available separately for the Bloch phase
//! sum `B^k_{μ,(i,m)} = Σ_R e^{ik·R} ⟨φ_μ(home) | p_i^{lm}(·−R)⟩` (the grid
//! `project_function` folds all images into one cell with no phase, so it is correct only
//! at Γ — but it is the independent oracle this module is tested against there).
//!
//! ## The reduction that makes it engine-native
//!
//! For `i = 1`, `p_1^{lm}` is *exactly* the engine's unit-normalized spherical Gaussian of
//! exponent `α = 1/(2 r_l²)` (same shape `r^l Y_lm e^{−αr²}`, both unit-norm). The engine
//! writes that AO as `cn · Σ_c M[lm,c]·(monomial c)·e^{−αr²}` with `M` the `c2s` transform
//! ([`shell_transform`](crate::spherical)) and `cn = cart_norm(α, l)`. Hence the solid
//! harmonic's raw-monomial coefficients are `a^{lm} = (cn/N_{1l}) M[lm]`, and for any `i`
//!
//! ```text
//!   ⟨φ_μ|p_i^{lm}⟩ = (N_{il}·cn/N_{1l}) · Σ_cp [ (x²+y²+z²)^{i−1} ⊗ M[lm] ]_cp · T[μ,cp]
//! ```
//!
//! where `T[μ,cp] = ⟨φ_μ | (monomial cp)·e^{−α|·−R|²}⟩` is the raw-monomial overlap from
//! [`os::overlap_into`]. For `i=1` the prefactor is `cn` and this is *identically* the
//! engine spherical overlap (so the `i=1` test is exact); higher `i` carry the
//! `(x²+y²+z²)^{i−1}` radial factor and the exact `N_{il}/N_{1l}` ratio. The real
//! spherical-harmonic convention cancels in the `m`-sum that forms `V_nl`, so it is
//! immaterial which convention `M` uses.

use crate::engine::os;
use crate::math::am::{cart_components, cart_index, n_cart};
use crate::math::norm::cart_norm;
use latx::Cell;
use rustfft::num_complex::Complex64;
use std::f64::consts::PI;

use crate::grad::pair_grad_1e;
use crate::integrals::{contract_pair, place_block, to_func_1e};
use crate::shell::{Basis, Shell};
use crate::spherical::shell_transform;

/// `Γ(n + 1/2) = √π · ∏_{j=1}^{n} (j − 1/2)` (exact for the half-integer arguments the GTH
/// normalization needs).
fn gamma_half(n: usize) -> f64 {
    let mut prod = PI.sqrt();
    for j in 1..=n {
        prod *= j as f64 - 0.5;
    }
    prod
}

/// The GTH projector normalization `N_{il} = √2 / (r_l^{l+(4i−1)/2} √Γ(l+(4i−1)/2))`.
/// The argument `l + (4i−1)/2 = (l+2i−1) + 1/2`, so `Γ = gamma_half(l+2i−1)`.
fn gth_norm(l: usize, i: usize, r_l: f64) -> f64 {
    let power = l as f64 + (4.0 * i as f64 - 1.0) / 2.0;
    let g = gamma_half(l + 2 * i - 1);
    std::f64::consts::SQRT_2 / (r_l.powf(power) * g.sqrt())
}

/// Multiply a Cartesian-monomial coefficient vector of degree `d` by `(x²+y²+z²)`, giving a
/// coefficient vector of degree `d+2` (in `crate::math::am` order).
fn raise_r2(coeffs: &[f64], d: usize) -> Vec<f64> {
    let mut out = vec![0.0; n_cart(d + 2)];
    for (ci, comp) in cart_components(d).iter().enumerate() {
        let v = coeffs[ci];
        if v == 0.0 {
            continue;
        }
        for axis in 0..3 {
            let mut nc = *comp;
            nc[axis] += 2;
            out[cart_index(nc)] += v;
        }
    }
    out
}

/// `T[μ, cp] = ⟨φ_μ | (raw Cartesian monomial cp of degree `lp`) · e^{−α|·−center|²}⟩`,
/// row-major `nao × n_cart(lp)`. Built from the engine overlap by making the projector a
/// Cartesian shell whose `primitive_coeff` is 1 (raw monomials).
fn projector_t(basis: &Basis, center: [f64; 3], alpha: f64, lp: usize) -> Vec<f64> {
    // coeff chosen so primitive_coeff = coeff · cart_norm(α, lp) = 1.
    let coeff = 1.0 / cart_norm(alpha, lp, 0, 0);
    let proj = Shell::new(lp, center, vec![alpha], vec![coeff]).expect("projector shell valid");
    let nao = basis.nao();
    let ncp = n_cart(lp);
    let offs = basis.offsets();
    let mut t = vec![0.0; nao * ncp];
    for (si, sa) in basis.shells().iter().enumerate() {
        let block = to_func_1e(contract_pair(sa, &proj, os::overlap_into), sa, &proj);
        place_block(&mut t, ncp, offs[si], 0, &block, ncp);
    }
    t
}

/// One GTH nonlocal projector channel of a single atom: the atom center, the angular
/// momentum `l`, the number of projectors `n_proj` (= the `h^l` matrix dimension), and the
/// channel radius `r_l` (bohr).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectorChannel {
    /// Atom center (bohr).
    pub center: [f64; 3],
    /// Angular momentum of the channel.
    pub l: usize,
    /// Number of projectors `i = 1..=n_proj` in the channel.
    pub n_proj: usize,
    /// Channel radius `r_l` (bohr).
    pub r_l: f64,
}

/// The real projector-overlap block for the channel placed at `center`:
/// `b[μ, (i,m)] = ⟨φ_μ | p_i^{lm}⟩`, row-major `nao × (n_proj·(2l+1))` with column
/// `(i−1)·(2l+1) + m` (`i = 1..=n_proj`, `m = 0..2l`).
fn projector_block(basis: &Basis, center: [f64; 3], ch: &ProjectorChannel) -> Vec<f64> {
    let (l, n_proj, r_l) = (ch.l, ch.n_proj, ch.r_l);
    let alpha = 1.0 / (2.0 * r_l * r_l);
    let nlm = 2 * l + 1;
    let nao = basis.nao();
    let ncol = n_proj * nlm;
    // c2s transform for angular momentum l (geometry/exponent-independent).
    let sph =
        Shell::new_spherical(l, center, vec![alpha], vec![1.0]).expect("spherical shell valid");
    let m_c2s = shell_transform(&sph).expect("spherical shell has a c2s transform");
    let ncart_l = n_cart(l);
    let cn = cart_norm(alpha, l, 0, 0);
    let n1 = gth_norm(l, 1, r_l);

    let mut out = vec![0.0; nao * ncol];
    for i in 1..=n_proj {
        let lp = l + 2 * (i - 1);
        let ncp = n_cart(lp);
        let scale = gth_norm(l, i, r_l) * cn / n1;
        let t = projector_t(basis, center, alpha, lp);
        for mm in 0..nlm {
            // Solid-harmonic coefficients (engine c2s row) raised by (x²+y²+z²)^{i−1}.
            let mut coeffs = m_c2s[mm * ncart_l..(mm + 1) * ncart_l].to_vec();
            let mut deg = l;
            for _ in 1..i {
                coeffs = raise_r2(&coeffs, deg);
                deg += 2;
            }
            let col = (i - 1) * nlm + mm;
            for mu in 0..nao {
                let trow = &t[mu * ncp..(mu + 1) * ncp];
                let mut acc = 0.0;
                for (cp, &q) in coeffs.iter().enumerate() {
                    acc += q * trow[cp];
                }
                out[mu * ncol + col] = scale * acc;
            }
        }
    }
    out
}

/// As [`projector_t`] but also returns the **bra-center** (AO-center) derivative blocks
/// `[∂T/∂A_x, ∂T/∂A_y, ∂T/∂A_z]` (each row-major `nao × n_cart(lp)`), via the exact molecular
/// center-derivative of the AO–monomial overlap ([`pair_grad_1e`]). The projector- (ket-)
/// center derivative is the negative of the bra derivative (translational invariance), so
/// only the bra block is returned.
fn projector_t_grad(
    basis: &Basis,
    center: [f64; 3],
    alpha: f64,
    lp: usize,
) -> (Vec<f64>, [Vec<f64>; 3]) {
    let coeff = 1.0 / cart_norm(alpha, lp, 0, 0);
    let proj = Shell::new(lp, center, vec![alpha], vec![coeff]).expect("projector shell valid");
    let nao = basis.nao();
    let ncp = n_cart(lp);
    let offs = basis.offsets();
    let mut t = vec![0.0; nao * ncp];
    let mut dt: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; nao * ncp]);
    for (si, sa) in basis.shells().iter().enumerate() {
        let block = to_func_1e(contract_pair(sa, &proj, os::overlap_into), sa, &proj);
        place_block(&mut t, ncp, offs[si], 0, &block, ncp);
        // Bra-center derivative of ⟨φ_μ | monomial⟩ (the AO shell sa moves).
        let (da, _db) = pair_grad_1e(sa, &proj, os::overlap_into);
        for axis in 0..3 {
            let dblock = to_func_1e(da[axis].clone(), sa, &proj);
            place_block(&mut dt[axis], ncp, offs[si], 0, &dblock, ncp);
        }
    }
    (t, dt)
}

/// As [`projector_block`] but also returns the **bra-center** (AO) derivative blocks
/// `[∂b/∂A_x, ∂b/∂A_y, ∂b/∂A_z]` (each row-major `nao × ncol`). The scale and solid-harmonic
/// combination are geometry-independent, so the derivative is the same linear combination
/// applied to the derivative of `T`. The projector-center derivative is the negative of this.
fn projector_block_grad(
    basis: &Basis,
    center: [f64; 3],
    ch: &ProjectorChannel,
) -> (Vec<f64>, [Vec<f64>; 3]) {
    let (l, n_proj, r_l) = (ch.l, ch.n_proj, ch.r_l);
    let alpha = 1.0 / (2.0 * r_l * r_l);
    let nlm = 2 * l + 1;
    let nao = basis.nao();
    let ncol = n_proj * nlm;
    let sph =
        Shell::new_spherical(l, center, vec![alpha], vec![1.0]).expect("spherical shell valid");
    let m_c2s = shell_transform(&sph).expect("spherical shell has a c2s transform");
    let ncart_l = n_cart(l);
    let cn = cart_norm(alpha, l, 0, 0);
    let n1 = gth_norm(l, 1, r_l);

    let mut out = vec![0.0; nao * ncol];
    let mut dout: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; nao * ncol]);
    for i in 1..=n_proj {
        let lp = l + 2 * (i - 1);
        let ncp = n_cart(lp);
        let scale = gth_norm(l, i, r_l) * cn / n1;
        let (t, dt) = projector_t_grad(basis, center, alpha, lp);
        for mm in 0..nlm {
            let mut coeffs = m_c2s[mm * ncart_l..(mm + 1) * ncart_l].to_vec();
            let mut deg = l;
            for _ in 1..i {
                coeffs = raise_r2(&coeffs, deg);
                deg += 2;
            }
            let col = (i - 1) * nlm + mm;
            for mu in 0..nao {
                let trow = &t[mu * ncp..(mu + 1) * ncp];
                let mut acc = 0.0;
                for (cp, &q) in coeffs.iter().enumerate() {
                    acc += q * trow[cp];
                }
                out[mu * ncol + col] = scale * acc;
                for axis in 0..3 {
                    let dtrow = &dt[axis][mu * ncp..(mu + 1) * ncp];
                    let mut dacc = 0.0;
                    for (cp, &q) in coeffs.iter().enumerate() {
                        dacc += q * dtrow[cp];
                    }
                    dout[axis][mu * ncol + col] = scale * dacc;
                }
            }
        }
    }
    (out, dout)
}

/// Bloch-summed projector overlaps `B^k` together with their **bra-center** (AO) derivatives
/// `[∂B^k/∂A_x, ∂B^k/∂A_y, ∂B^k/∂A_z]` (each row-major `nao × (n_proj·(2l+1))` complex,
/// Bloch-phased over images within `rmax`). `∂B^k_{μ,(i,m)}/∂A` is the derivative w.r.t. the
/// **AO** (bra) center of `μ`; the derivative w.r.t. the **projector** center is its negative
/// (translational invariance, image by image), so the caller uses `−db` for the projector's
/// atom. This is the gradient counterpart of [`bloch_projector_overlaps`] for the M4 nonlocal
/// projector force.
///
/// Returns `(b, [db_x, db_y, db_z])`.
#[must_use]
pub fn bloch_projector_overlaps_grad(
    basis: &Basis,
    cell: &Cell,
    ch: &ProjectorChannel,
    k_frac: [f64; 3],
    rmax: f64,
) -> (Vec<Complex64>, [Vec<Complex64>; 3]) {
    let ncol = ch.n_proj * (2 * ch.l + 1);
    let len = basis.nao() * ncol;
    let mut b = vec![Complex64::new(0.0, 0.0); len];
    let mut db: [Vec<Complex64>; 3] = std::array::from_fn(|_| vec![Complex64::new(0.0, 0.0); len]);
    for (triple, r) in cell.lattice_images(rmax) {
        let pc = [
            ch.center[0] + r[0],
            ch.center[1] + r[1],
            ch.center[2] + r[2],
        ];
        let (val, dval) = projector_block_grad(basis, pc, ch);
        let theta = 2.0
            * PI
            * (k_frac[0] * f64::from(triple[0])
                + k_frac[1] * f64::from(triple[1])
                + k_frac[2] * f64::from(triple[2]));
        let phase = Complex64::new(theta.cos(), theta.sin());
        for (o, &v) in b.iter_mut().zip(&val) {
            *o += phase * v;
        }
        for axis in 0..3 {
            for (o, &v) in db[axis].iter_mut().zip(&dval[axis]) {
                *o += phase * v;
            }
        }
    }
    (b, db)
}

/// Bloch-summed projector overlaps `B^k` together with their **strain derivative**
/// `∂B^k/∂ε_αβ` (the stress building block). Returns `(b, db_eps)` where
/// `db_eps[α][β][μ·ncol+col] = Σ_R e^{ik·R} g'^R_{μ,col,α} · sep^R_{μ,β}` with the
/// **projector-center** gradient `g'^R = −∂⟨φ_μ|p^R⟩/∂A` (the negative of the bra/AO-center
/// derivative, by translational invariance) and the separation `sep^R_μ = (R_J + R) − τ_μ`
/// (projector image center minus the AO center).
///
/// A projector overlap `⟨φ_μ(A)|p(·−C)⟩` depends only on `C−A`, so under a strain `F = I + ε`
/// (`C−A → F(C−A)`) `∂⟨⟩/∂ε_αβ = g'_α (C−A)_β`; the caller contracts `db_eps[α][β]` against
/// the force-weight `Φ` exactly as the value `B^k` is used (see the chemx-periodic projector
/// stress). Each `db_eps[α][β]` is row-major `nao × ncol` complex.
#[must_use]
pub fn bloch_projector_overlaps_strain(
    basis: &Basis,
    cell: &Cell,
    ch: &ProjectorChannel,
    k_frac: [f64; 3],
    rmax: f64,
) -> (Vec<Complex64>, [[Vec<Complex64>; 3]; 3]) {
    let nao = basis.nao();
    let ncol = ch.n_proj * (2 * ch.l + 1);
    let len = nao * ncol;
    let mut b = vec![Complex64::new(0.0, 0.0); len];
    let mut db_eps: [[Vec<Complex64>; 3]; 3] =
        std::array::from_fn(|_| std::array::from_fn(|_| vec![Complex64::new(0.0, 0.0); len]));

    // AO → shell-center map (the bra center A of each AO).
    let offs = basis.offsets();
    let mut ao_center = vec![[0.0_f64; 3]; nao];
    for (si, sh) in basis.shells().iter().enumerate() {
        for f in 0..sh.n_func() {
            ao_center[offs[si] + f] = sh.center();
        }
    }

    for (triple, r) in cell.lattice_images(rmax) {
        let pc = [
            ch.center[0] + r[0],
            ch.center[1] + r[1],
            ch.center[2] + r[2],
        ];
        let (val, dval) = projector_block_grad(basis, pc, ch);
        let theta = 2.0
            * PI
            * (k_frac[0] * f64::from(triple[0])
                + k_frac[1] * f64::from(triple[1])
                + k_frac[2] * f64::from(triple[2]));
        let phase = Complex64::new(theta.cos(), theta.sin());
        for (o, &v) in b.iter_mut().zip(&val) {
            *o += phase * v;
        }
        for (mu, &ac) in ao_center.iter().enumerate() {
            let sep = [pc[0] - ac[0], pc[1] - ac[1], pc[2] - ac[2]]; // C − A
            for col in 0..ncol {
                let idx = mu * ncol + col;
                for alpha in 0..3 {
                    // Projector-center gradient g'_α = −(bra-center derivative).
                    let pg = phase * (-dval[alpha][idx]);
                    for beta in 0..3 {
                        db_eps[alpha][beta][idx] += pg * sep[beta];
                    }
                }
            }
        }
    }
    (b, db_eps)
}

/// Real Γ-point projector overlaps `A^Γ_{μ,(i,m)} = Σ_R ⟨φ_μ(home) | p_i^{lm}(·−R)⟩` for
/// channel `ch`, summing projector lattice images within `rmax` (bohr). Row-major
/// `nao × (n_proj·(2l+1))`, column `(i−1)·(2l+1) + m`. (The home cell alone, `rmax → 0⁺`,
/// gives the molecular / large-box overlaps.)
#[must_use]
pub fn projector_overlaps(
    basis: &Basis,
    cell: &Cell,
    ch: &ProjectorChannel,
    rmax: f64,
) -> Vec<f64> {
    let ncol = ch.n_proj * (2 * ch.l + 1);
    let mut out = vec![0.0; basis.nao() * ncol];
    for (_triple, r) in cell.lattice_images(rmax) {
        let pc = [
            ch.center[0] + r[0],
            ch.center[1] + r[1],
            ch.center[2] + r[2],
        ];
        let b = projector_block(basis, pc, ch);
        for (o, v) in out.iter_mut().zip(&b) {
            *o += v;
        }
    }
    out
}

/// Bloch-summed projector overlaps `B^k_{μ,(i,m)} = Σ_R e^{ik·R} ⟨φ_μ(home) | p_i^{lm}(·−R)⟩`
/// for channel `ch` at k-point fractional coordinates `k_frac`, summing projector images
/// within `rmax` (bohr). Row-major `nao × (n_proj·(2l+1))` complex, column
/// `(i−1)·(2l+1) + m`.
///
/// The caller forms the nonlocal Bloch matrix
/// `V_nl(k)_{μν} = Σ_{i,j,m} B^k_{μ,(i,m)} h^l_{ij} (B^k_{ν,(j,m)})*` (one channel per atom,
/// the AO Bloch sum supplying the periodicity).
#[must_use]
pub fn bloch_projector_overlaps(
    basis: &Basis,
    cell: &Cell,
    ch: &ProjectorChannel,
    k_frac: [f64; 3],
    rmax: f64,
) -> Vec<Complex64> {
    let ncol = ch.n_proj * (2 * ch.l + 1);
    let mut out = vec![Complex64::new(0.0, 0.0); basis.nao() * ncol];
    for (triple, r) in cell.lattice_images(rmax) {
        let pc = [
            ch.center[0] + r[0],
            ch.center[1] + r[1],
            ch.center[2] + r[2],
        ];
        let b = projector_block(basis, pc, ch);
        let theta = 2.0
            * PI
            * (k_frac[0] * f64::from(triple[0])
                + k_frac[1] * f64::from(triple[1])
                + k_frac[2] * f64::from(triple[2]));
        let phase = Complex64::new(theta.cos(), theta.sin());
        for (o, &v) in out.iter_mut().zip(&b) {
            *o += phase * v;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::periodic::{project_function, RealSpaceGrid};
    use crate::{Basis, Shell};

    fn test_basis(center: [f64; 3]) -> Basis {
        Basis::new(vec![
            Shell::new(0, [0.0, 0.0, 0.0], vec![0.6], vec![1.0]).unwrap(),
            Shell::new_spherical(1, [1.4, 0.3, -0.2], vec![0.5], vec![1.0]).unwrap(),
            Shell::new(0, center, vec![0.9], vec![1.0]).unwrap(),
        ])
    }

    /// For `i = 1`, the analytic projector overlap is *identically* the engine's overlap
    /// with a unit-normalized spherical Gaussian shell of exponent `α = 1/(2 r_l²)`.
    #[test]
    fn first_projector_equals_engine_spherical_overlap() {
        for l in [0usize, 1, 2] {
            let center = [0.7, -0.5, 0.4];
            let basis = test_basis([2.0, 0.0, 0.0]);
            let r_l = 0.55;
            let alpha = 1.0 / (2.0 * r_l * r_l);
            let nao = basis.nao();
            let nlm = 2 * l + 1;

            let ch = ProjectorChannel {
                center,
                l,
                n_proj: 1,
                r_l,
            };
            let b = projector_block(&basis, center, &ch); // nao × nlm

            // Engine overlap: append the spherical projector shell, read the cross block.
            let mut shells = basis.shells().to_vec();
            shells.push(Shell::new_spherical(l, center, vec![alpha], vec![1.0]).unwrap());
            let aug = Basis::new(shells);
            let n_aug = aug.nao();
            let s = aug.overlap();
            for mu in 0..nao {
                for m in 0..nlm {
                    let eng = s[mu * n_aug + (nao + m)];
                    let got = b[mu * nlm + m];
                    assert!(
                        (eng - got).abs() < 1e-12,
                        "l={l} μ={mu} m={m}: engine {eng} vs analytic {got}"
                    );
                }
            }
        }
    }

    /// Cross-check the analytic route against the validated grid `project_function` for the
    /// `l = 0` channel with two projectors (Si's s channel: `i = 1` and `i = 2`, the latter
    /// carrying the `r²` radial factor and `N_{2,0}`). The closed-form projector is
    /// evaluated on the grid independently of the overlap engine.
    #[test]
    fn s_channel_matches_grid_projection() {
        let center = [4.0, 4.0, 4.0];
        let basis = test_basis([5.0, 4.0, 4.0]);
        let r_l = 0.42;
        let alpha = 1.0 / (2.0 * r_l * r_l);
        let n_proj = 2;
        let nao = basis.nao();
        // Large box so only the home image contributes; fine grid for the projector.
        let grid = RealSpaceGrid::new(latx::Cell::cubic(8.0).unwrap(), [80, 80, 80]);
        let ch = ProjectorChannel {
            center,
            l: 0,
            n_proj,
            r_l,
        };
        let b = projector_overlaps(&basis, grid.cell(), &ch, 1.0);

        let y00 = 1.0 / (4.0 * PI).sqrt();
        for i in 1..=n_proj {
            let n_il = gth_norm(0, i, r_l);
            // p_i^{00}(r) = N · r^{2(i−1)} · e^{−αr²} · Y_00.
            let pvals: Vec<f64> = grid
                .points()
                .iter()
                .map(|p| {
                    let d2 = (p[0] - center[0]).powi(2)
                        + (p[1] - center[1]).powi(2)
                        + (p[2] - center[2]).powi(2);
                    let radial = d2.powi((i - 1) as i32) * (-alpha * d2).exp();
                    n_il * radial * y00
                })
                .collect();
            let b_grid = project_function(&basis, &pvals, &grid); // nao
            for mu in 0..nao {
                let analytic = b[mu * n_proj + (i - 1)];
                assert!(
                    (analytic - b_grid[mu]).abs() < 1e-3,
                    "i={i} μ={mu}: analytic {analytic} vs grid {}",
                    b_grid[mu]
                );
            }
        }
    }

    /// The closed-form projector is unit-normalized: `∫|p_i^{00}|² d³r ≈ 1` on a fine grid
    /// (validates `N_{il}` and the `r^{2(i−1)}` radial factor for `i = 1, 2`).
    #[test]
    fn projector_is_unit_normalized() {
        let center = [4.0, 4.0, 4.0];
        let r_l = 0.42;
        let alpha = 1.0 / (2.0 * r_l * r_l);
        let grid = RealSpaceGrid::new(latx::Cell::cubic(8.0).unwrap(), [90, 90, 90]);
        let y00 = 1.0 / (4.0 * PI).sqrt();
        for i in 1..=2usize {
            let n_il = gth_norm(0, i, r_l);
            let norm: f64 = grid
                .points()
                .iter()
                .map(|p| {
                    let d2 = (p[0] - center[0]).powi(2)
                        + (p[1] - center[1]).powi(2)
                        + (p[2] - center[2]).powi(2);
                    let pv = n_il * d2.powi((i - 1) as i32) * (-alpha * d2).exp() * y00;
                    pv * pv
                })
                .sum::<f64>()
                * grid.dv();
            assert!((norm - 1.0).abs() < 1e-4, "i={i}: ∫|p|² = {norm}");
        }
    }

    /// The analytic bra-center (AO) derivative of the Bloch projector overlaps matches the
    /// central finite difference of `B^k` w.r.t. the AO center — the integral half of the M4
    /// nonlocal projector force. All AO shells share one center so moving it differentiates
    /// every `B^k_{μ,col}`; a non-Γ k-point exercises the complex phase sum.
    #[test]
    fn bloch_projector_bra_gradient_matches_finite_difference() {
        let a0 = [1.0, 0.9, 1.1];
        let proj_center = [2.3, 1.4, 0.8];
        let cell = Cell::cubic(4.5).unwrap();
        let ch = ProjectorChannel {
            center: proj_center,
            l: 1,
            n_proj: 2,
            r_l: 0.45,
        };
        let rmax = 9.0;
        let kfrac = [0.3, -0.15, 0.2];

        // Basis: an s and a p shell, both centered on the (single) movable AO atom.
        let mk = |a: [f64; 3]| {
            Basis::new(vec![
                Shell::new(0, a, vec![0.7], vec![1.0]).unwrap(),
                Shell::new_spherical(1, a, vec![0.5], vec![1.0]).unwrap(),
            ])
        };
        let basis = mk(a0);
        let (_b, db) = bloch_projector_overlaps_grad(&basis, &cell, &ch, kfrac, rmax);
        let len = db[0].len();

        let h = 1e-5;
        for axis in 0..3 {
            let mut ap = a0;
            ap[axis] += h;
            let b_plus = bloch_projector_overlaps(&mk(ap), &cell, &ch, kfrac, rmax);
            ap[axis] -= 2.0 * h;
            let b_minus = bloch_projector_overlaps(&mk(ap), &cell, &ch, kfrac, rmax);
            for idx in 0..len {
                let fd = (b_plus[idx] - b_minus[idx]) / (2.0 * h);
                assert!(
                    (db[axis][idx] - fd).norm() < 1e-6,
                    "axis {axis} idx {idx}: analytic {} vs FD {fd}",
                    db[axis][idx]
                );
            }
        }
    }

    /// The analytic strain derivative of the Bloch projector overlaps matches the central
    /// finite difference of `B^k` under a strain `F = I + λ·M_dir` (AO center, projector
    /// center, and cell deform together) — the integral half of the M5 nonlocal projector
    /// stress. Checked over uniaxial, shear, and general-symmetric strains; a non-Γ k-point
    /// exercises the phase sum.
    #[test]
    fn bloch_projector_strain_matches_finite_difference() {
        // Large box (only the home image is within rmax) so the strain derivative is tested
        // cleanly, without the lattice-image set shifting across the rmax boundary under the
        // ± strain steps (a per-element FD artifact that the energy-level chemx stress test,
        // where boundary images contribute negligibly, is robust to).
        let a0 = [3.0, 2.9, 3.1];
        let proj_center = [4.3, 3.4, 2.8];
        let cell = Cell::cubic(12.0).unwrap();
        let ch = ProjectorChannel {
            center: proj_center,
            l: 1,
            n_proj: 2,
            r_l: 0.45,
        };
        let rmax = 6.0;
        let kfrac = [0.3, -0.15, 0.2];
        let mk = |a: [f64; 3]| {
            Basis::new(vec![
                Shell::new(0, a, vec![0.7], vec![1.0]).unwrap(),
                Shell::new_spherical(1, a, vec![0.5], vec![1.0]).unwrap(),
            ])
        };
        let (_b, db_eps) = bloch_projector_overlaps_strain(&mk(a0), &cell, &ch, kfrac, rmax);
        let len = db_eps[0][0].len();

        let deform = |m: &[[f64; 3]; 3], lambda: f64, v: [f64; 3]| -> [f64; 3] {
            let mut o = v;
            for (a, oa) in o.iter_mut().enumerate() {
                for (bb, &vb) in v.iter().enumerate() {
                    *oa += lambda * m[a][bb] * vb;
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
            let bk = |lambda: f64| -> Vec<Complex64> {
                let (a1, a2, a3) = cell.vectors();
                let dcell = Cell::from_vectors(
                    deform(&mdir, lambda, a1),
                    deform(&mdir, lambda, a2),
                    deform(&mdir, lambda, a3),
                )
                .unwrap();
                let dch = ProjectorChannel {
                    center: deform(&mdir, lambda, proj_center),
                    ..ch
                };
                bloch_projector_overlaps(&mk(deform(&mdir, lambda, a0)), &dcell, &dch, kfrac, rmax)
            };
            let h = 1e-5;
            let (bp, bm) = (bk(h), bk(-h));
            for idx in 0..len {
                let fd = (bp[idx] - bm[idx]) / (2.0 * h);
                let mut analytic = Complex64::new(0.0, 0.0);
                for a in 0..3 {
                    for b in 0..3 {
                        analytic += db_eps[a][b][idx] * mdir[a][b];
                    }
                }
                assert!(
                    (analytic - fd).norm() < 1e-6,
                    "strain {mdir:?} idx {idx}: analytic {analytic} vs FD {fd}"
                );
            }
        }
    }

    /// Bloch projector overlaps reduce to the real Γ overlaps at k = Γ, and are k-periodic.
    #[test]
    fn bloch_projector_gamma_and_periodicity() {
        let center = [1.0, 1.0, 1.0];
        let cell = latx::Cell::cubic(4.0).unwrap();
        let basis = test_basis([2.2, 1.0, 1.0]);
        let ch = ProjectorChannel {
            center,
            l: 1,
            n_proj: 1,
            r_l: 0.48,
        };
        let rmax = 9.0;

        let real_gamma = projector_overlaps(&basis, &cell, &ch, rmax);
        let bloch_gamma = bloch_projector_overlaps(&basis, &cell, &ch, [0.0; 3], rmax);
        for (re, c) in real_gamma.iter().zip(&bloch_gamma) {
            assert!((c.re - re).abs() < 1e-12 && c.im.abs() < 1e-12);
        }
        // k and k + G give the same Bloch sum.
        let k1 = bloch_projector_overlaps(&basis, &cell, &ch, [0.3, 0.0, 0.0], rmax);
        let k2 = bloch_projector_overlaps(&basis, &cell, &ch, [1.3, 0.0, 0.0], rmax);
        for (a, b) in k1.iter().zip(&k2) {
            assert!((a - b).norm() < 1e-12);
        }
    }
}
