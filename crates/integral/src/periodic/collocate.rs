//! Gaussian → grid **collocation** (`P → n(r)`) and grid → matrix **integration**
//! (`V(r) → V_μν`) — the real-space half of the GPW "density-matrix ↔ grid ↔
//! potential-matrix" machinery.
//!
//! Both are built on one pointwise atomic-orbital evaluator that reproduces the engine's
//! exact normalization, so the collocated density and the integrated potential matrix are
//! consistent with the analytic [`Basis::overlap`](crate::Basis::overlap) /
//! [`Basis::kinetic`](crate::Basis::kinetic) blocks:
//!
//! - the Cartesian-monomial value of component `(lx, ly, lz)` of a shell at `A` is
//!   `[Σ_i primitive_coeff(i)·e^{-α_i ρ²}] · (x−Aₓ)^lx (y−Aᵧ)^ly (z−A_z)^lz` with
//!   `ρ = r − A` (taken as the **minimum image** under the cell, so atoms anywhere in the
//!   cell collocate correctly);
//! - for a [`ShellKind::Spherical`] shell the `(2l+1)×n_cart` transform
//!   [`shell_transform`](crate::spherical) maps those Cartesian values to the final
//!   `2l+1` real-spherical AOs.
//!
//! The two operations are adjoints: `Σ_μν V_μν P_μν = Σ_g dV·V(r_g)·n(r_g)`.
//!
//! v1 evaluates against the **minimum image** of each shell center (correct for the
//! large-box/isolated regime and the leading term in a periodic cell); a full real-space
//! lattice sum of Gaussian images for tightly-packed cells is layered in at M3.

use crate::math::am::{cart_components, n_cart};
use crate::shell::Basis;
use crate::spherical::shell_transform;

use super::grid::RealSpaceGrid;

/// Skip a primitive shell's contribution where `α_min · ρ² >` this (≈ `e^{-40}` ≈ 4e-18,
/// far under the engine's 1e-12 accuracy bar even after the monomial `ρ^l` factor).
pub(super) const SCREEN_EXP: f64 = 40.0;

/// Per-shell data hoisted out of the grid-point loop.
pub(super) struct ShellEval {
    pub(super) offset: usize,
    pub(super) n_cart: usize,
    pub(super) n_func: usize,
    pub(super) center: [f64; 3],
    /// `primitive_coeff(i)` for each primitive (raw coeff × shell norm `N(α_i, l)`).
    pub(super) prim_coeff: Vec<f64>,
    pub(super) exps: Vec<f64>,
    /// Cartesian power triplets, in canonical order.
    pub(super) comps: Vec<[usize; 3]>,
    /// `(2l+1) × n_cart` spherical transform, or `None` for a Cartesian shell.
    pub(super) transform: Option<Vec<f64>>,
    /// Smallest exponent (the radial-cutoff screen reads `α_min·ρ² > SCREEN_EXP`).
    pub(super) alpha_min: f64,
}

pub(super) fn build_shell_evals(basis: &Basis) -> Vec<ShellEval> {
    let offsets = basis.offsets();
    basis
        .shells()
        .iter()
        .zip(offsets)
        .map(|(s, offset)| {
            let l = s.l();
            let prim_coeff: Vec<f64> = (0..s.n_prim()).map(|i| s.primitive_coeff(i)).collect();
            let exps = s.exponents().to_vec();
            let alpha_min = exps.iter().copied().fold(f64::INFINITY, f64::min);
            ShellEval {
                offset,
                n_cart: n_cart(l),
                n_func: s.n_func(),
                center: s.center(),
                prim_coeff,
                exps,
                comps: cart_components(l),
                transform: shell_transform(s),
                alpha_min,
            }
        })
        .collect()
}

impl ShellEval {
    /// Append this shell's nonzero final-basis AO values at `r` to `active` as
    /// `(ao_index, value)` pairs, evaluating against the **minimum image** of the
    /// shell center. `scratch` holds the `n_cart` Cartesian values.
    fn eval_into(
        &self,
        r: [f64; 3],
        grid: &RealSpaceGrid,
        scratch: &mut Vec<f64>,
        active: &mut Vec<(usize, f64)>,
    ) {
        let dr = grid.cell().min_image([
            r[0] - self.center[0],
            r[1] - self.center[1],
            r[2] - self.center[2],
        ]);
        self.emit(dr, scratch, &mut |ao, v| active.push((ao, v)));
    }

    /// Append this shell's nonzero AO values at the **literal** displacement to
    /// `image_center` (an explicitly enumerated lattice image; *no* minimum-image
    /// wrap), via `emit(ao_index, value)`. Used by the image-resolved lattice
    /// collocation/integration ([`super::lattice`]).
    pub(super) fn eval_at<F: FnMut(usize, f64)>(
        &self,
        r: [f64; 3],
        image_center: [f64; 3],
        scratch: &mut Vec<f64>,
        emit: F,
    ) {
        let dr = [
            r[0] - image_center[0],
            r[1] - image_center[1],
            r[2] - image_center[2],
        ];
        self.emit(dr, scratch, emit);
    }

    /// Shared evaluation core: given the displacement `dr = r − center`, emit every
    /// nonzero final-basis AO value via `emit(ao_index, value)`. `scratch` holds the
    /// `n_cart` Cartesian-monomial values. The radial screen `α_min·ρ² > SCREEN_EXP`
    /// is exactly the M2 `cutoff2` screen (`cutoff2 = SCREEN_EXP/α_min`). Generic over the
    /// emit target so the hot grid loops monomorphize (no dynamic dispatch).
    pub(super) fn emit<F: FnMut(usize, f64)>(
        &self,
        dr: [f64; 3],
        scratch: &mut Vec<f64>,
        mut emit: F,
    ) {
        let rho2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
        if self.alpha_min * rho2 > SCREEN_EXP {
            return;
        }
        let mut radial = 0.0;
        for (c, &a) in self.prim_coeff.iter().zip(&self.exps) {
            radial += c * (-a * rho2).exp();
        }
        if radial == 0.0 {
            return;
        }
        // Cartesian-monomial values.
        scratch.clear();
        for comp in &self.comps {
            let mono = dr[0].powi(comp[0] as i32)
                * dr[1].powi(comp[1] as i32)
                * dr[2].powi(comp[2] as i32);
            scratch.push(radial * mono);
        }
        match &self.transform {
            None => {
                for (c, &val) in scratch.iter().enumerate() {
                    if val != 0.0 {
                        emit(self.offset + c, val);
                    }
                }
            }
            Some(m) => {
                // func_f = Σ_c M[f, c] · cart_c, M row-major (n_func × n_cart).
                for f in 0..self.n_func {
                    let row = &m[f * self.n_cart..(f + 1) * self.n_cart];
                    let mut acc = 0.0;
                    for (mc, &cart) in row.iter().zip(scratch.iter()) {
                        acc += mc * cart;
                    }
                    if acc != 0.0 {
                        emit(self.offset + f, acc);
                    }
                }
            }
        }
    }

    /// Like [`emit`](Self::emit) but also emits the **spatial gradient** `∇_r φ` of each AO
    /// (as `emit(ao_index, value, [∂x, ∂y, ∂z])`), at the literal displacement
    /// `dr = r − image_center`. For a Gaussian-times-monomial AO,
    /// `∇_r(radial·mono) = (∂radial)·mono + radial·(∇mono)` with `∂radial/∂r_a = dr_a·Σ_i
    /// c_i(−2α_i)e^{−α_i ρ²}`. The center derivative `∂φ/∂R = −∇_r φ` (used by the periodic
    /// force terms). Emits every in-screen AO — including those whose value is zero but
    /// gradient is not (e.g. a p AO at its own center). `scratch` holds the interleaved
    /// `[val, ∂x, ∂y, ∂z]` per Cartesian component. Generic over the emit target so the force
    /// loops monomorphize.
    pub(super) fn emit_grad<F: FnMut(usize, f64, [f64; 3])>(
        &self,
        r: [f64; 3],
        image_center: [f64; 3],
        scratch: &mut Vec<f64>,
        mut emit: F,
    ) {
        let dr = [
            r[0] - image_center[0],
            r[1] - image_center[1],
            r[2] - image_center[2],
        ];
        let rho2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
        if self.alpha_min * rho2 > SCREEN_EXP {
            return;
        }
        let mut radial = 0.0;
        let mut dradial = 0.0; // Σ_i c_i(−2α_i)e^{−α_i ρ²}, so ∂radial/∂r_a = dradial·dr_a
        for (c, &a) in self.prim_coeff.iter().zip(&self.exps) {
            let e = c * (-a * rho2).exp();
            radial += e;
            dradial += -2.0 * a * e;
        }
        if radial == 0.0 && dradial == 0.0 {
            return;
        }
        scratch.clear();
        for comp in &self.comps {
            let (lx, ly, lz) = (comp[0] as i32, comp[1] as i32, comp[2] as i32);
            let px = dr[0].powi(lx);
            let py = dr[1].powi(ly);
            let pz = dr[2].powi(lz);
            let mono = px * py * pz;
            let val = radial * mono;
            // ∂mono/∂r_a = l_a · dr_a^{l_a−1} · (other powers).
            let dmx = if lx == 0 {
                0.0
            } else {
                f64::from(lx) * dr[0].powi(lx - 1) * py * pz
            };
            let dmy = if ly == 0 {
                0.0
            } else {
                f64::from(ly) * px * dr[1].powi(ly - 1) * pz
            };
            let dmz = if lz == 0 {
                0.0
            } else {
                f64::from(lz) * px * py * dr[2].powi(lz - 1)
            };
            scratch.push(val);
            scratch.push(dradial * dr[0] * mono + radial * dmx);
            scratch.push(dradial * dr[1] * mono + radial * dmy);
            scratch.push(dradial * dr[2] * mono + radial * dmz);
        }
        match &self.transform {
            None => {
                for (c, chunk) in scratch.chunks_exact(4).enumerate() {
                    emit(self.offset + c, chunk[0], [chunk[1], chunk[2], chunk[3]]);
                }
            }
            Some(m) => {
                for f in 0..self.n_func {
                    let row = &m[f * self.n_cart..(f + 1) * self.n_cart];
                    let (mut v, mut gx, mut gy, mut gz) = (0.0, 0.0, 0.0, 0.0);
                    for (mc, chunk) in row.iter().zip(scratch.chunks_exact(4)) {
                        v += mc * chunk[0];
                        gx += mc * chunk[1];
                        gy += mc * chunk[2];
                        gz += mc * chunk[3];
                    }
                    emit(self.offset + f, v, [gx, gy, gz]);
                }
            }
        }
    }
}

/// Collocate the density matrix `p` (row-major `nao × nao`, the final AO basis) onto the
/// real-space grid: `n(r) = Σ_μν P_μν φ_μ(r) φ_ν(r)`. Returns `n(r)` row-major over the grid
/// (length [`RealSpaceGrid::n_points`]).
///
/// # Panics
/// Panics if `p.len() != nao²`.
#[must_use]
pub fn collocate_density(basis: &Basis, p: &[f64], grid: &RealSpaceGrid) -> Vec<f64> {
    let nao = basis.nao();
    assert_eq!(
        p.len(),
        nao * nao,
        "density matrix must be nao×nao = {}²",
        nao
    );
    let shells = build_shell_evals(basis);

    let mut n_r = vec![0.0; grid.n_points()];
    let [n0, n1, n2] = grid.n();
    let mut scratch = Vec::with_capacity(16);
    let mut active: Vec<(usize, f64)> = Vec::with_capacity(nao);
    for i in 0..n0 {
        for j in 0..n1 {
            for k in 0..n2 {
                let r = grid.point([i, j, k]);
                active.clear();
                for sh in &shells {
                    sh.eval_into(r, grid, &mut scratch, &mut active);
                }
                if active.is_empty() {
                    continue;
                }
                // n = Σ_{a,b} P[a,b] φ_a φ_b.
                let mut nn = 0.0;
                for &(mu, fmu) in &active {
                    let row = &p[mu * nao..(mu + 1) * nao];
                    let mut t = 0.0;
                    for &(nu, fnu) in &active {
                        t += row[nu] * fnu;
                    }
                    nn += fmu * t;
                }
                n_r[grid.linear_index([i, j, k])] = nn;
            }
        }
    }
    n_r
}

/// Integrate a real grid potential `v` (length [`RealSpaceGrid::n_points`]) against Gaussian
/// products: `V_μν = Σ_g dV · V(r_g) · φ_μ(r_g) φ_ν(r_g)`. Returns the symmetric matrix
/// row-major `nao × nao`. This is the adjoint of [`collocate_density`].
///
/// Passing a constant `v ≡ 1` reproduces the overlap matrix in the grid limit; passing the
/// Hartree/XC/local potential builds the local part of the KS matrix.
///
/// # Panics
/// Panics if `v.len()` differs from the grid point count.
#[must_use]
pub fn integrate_potential(basis: &Basis, v: &[f64], grid: &RealSpaceGrid) -> Vec<f64> {
    assert_eq!(
        v.len(),
        grid.n_points(),
        "potential length must equal grid points"
    );
    let nao = basis.nao();
    let shells = build_shell_evals(basis);
    let dv = grid.dv();

    let mut mat = vec![0.0; nao * nao];
    let [n0, n1, n2] = grid.n();
    let mut scratch = Vec::with_capacity(16);
    let mut active: Vec<(usize, f64)> = Vec::with_capacity(nao);
    for i in 0..n0 {
        for j in 0..n1 {
            for k in 0..n2 {
                let lin = grid.linear_index([i, j, k]);
                let vg = v[lin];
                if vg == 0.0 {
                    continue;
                }
                let r = grid.point([i, j, k]);
                active.clear();
                for sh in &shells {
                    sh.eval_into(r, grid, &mut scratch, &mut active);
                }
                let w = dv * vg;
                for &(mu, fmu) in &active {
                    let wfmu = w * fmu;
                    let row = &mut mat[mu * nao..(mu + 1) * nao];
                    for &(nu, fnu) in &active {
                        row[nu] += wfmu * fnu;
                    }
                }
            }
        }
    }
    mat
}

/// Project a real grid function `f` (length [`RealSpaceGrid::n_points`]) onto each basis
/// function: `b_μ = Σ_g dV · f(r_g) · φ_μ(r_g)`. Returns a length-`nao` vector.
///
/// This is the *linear* (one-basis-function) partner of [`integrate_potential`]'s quadratic
/// form, used to build separable Kleinman–Bylander projector overlaps `⟨φ_μ | p⟩` by feeding
/// the projector function `p(r)` evaluated on the grid.
///
/// # Panics
/// Panics if `f.len()` differs from the grid point count.
#[must_use]
pub fn project_function(basis: &Basis, f: &[f64], grid: &RealSpaceGrid) -> Vec<f64> {
    assert_eq!(
        f.len(),
        grid.n_points(),
        "function length must equal grid points"
    );
    let nao = basis.nao();
    let shells = build_shell_evals(basis);
    let dv = grid.dv();

    let mut out = vec![0.0; nao];
    let [n0, n1, n2] = grid.n();
    let mut scratch = Vec::with_capacity(16);
    let mut active: Vec<(usize, f64)> = Vec::with_capacity(nao);
    for i in 0..n0 {
        for j in 0..n1 {
            for k in 0..n2 {
                let lin = grid.linear_index([i, j, k]);
                let fg = f[lin];
                if fg == 0.0 {
                    continue;
                }
                let r = grid.point([i, j, k]);
                active.clear();
                for sh in &shells {
                    sh.eval_into(r, grid, &mut scratch, &mut active);
                }
                let w = dv * fg;
                for &(mu, fmu) in &active {
                    out[mu] += w * fmu;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Basis, Shell};
    use latx::Cell;

    /// A single normalized s function: overlap reproduction (∫1·φ² ≈ S = 1) and the density
    /// trace (∫n ≈ Σ P S).
    #[test]
    fn integrate_constant_reproduces_overlap_s() {
        let basis = Basis::new(vec![
            Shell::new(0, [8.0, 8.0, 8.0], vec![0.8], vec![1.0]).unwrap()
        ]);
        let grid = RealSpaceGrid::new(Cell::cubic(16.0).unwrap(), [64, 64, 64]);
        let s_grid = integrate_potential(&basis, &vec![1.0; grid.n_points()], &grid);
        // Analytic self-overlap of a normalized s is 1.
        assert!((s_grid[0] - 1.0).abs() < 1e-6, "S_grid = {}", s_grid[0]);
    }

    #[test]
    fn collocation_trace_matches_overlap() {
        // Two s shells; P = identity. ∫ n = Σ_μν P_μν S_μν = Tr(S).
        let basis = Basis::new(vec![
            Shell::new(0, [7.0, 8.0, 8.0], vec![0.7], vec![1.0]).unwrap(),
            Shell::new(0, [9.0, 8.0, 8.0], vec![1.1], vec![1.0]).unwrap(),
        ]);
        let s = basis.overlap();
        let grid = RealSpaceGrid::new(Cell::cubic(16.0).unwrap(), [72, 72, 72]);
        let nao = basis.nao();
        let mut p = vec![0.0; nao * nao];
        for i in 0..nao {
            p[i * nao + i] = 1.0;
        }
        let n_r = collocate_density(&basis, &p, &grid);
        let integral_n: f64 = n_r.iter().sum::<f64>() * grid.dv();
        let trace_s: f64 = (0..nao).map(|i| s[i * nao + i]).sum();
        assert!(
            (integral_n - trace_s).abs() < 1e-5,
            "∫n = {integral_n}, Tr S = {trace_s}"
        );
    }

    /// Projecting a basis function (evaluated analytically on the grid) onto the basis
    /// reproduces that column of the overlap matrix: `Σ_g dV φ_0 φ_μ ≈ S_{μ0}`.
    #[test]
    fn project_function_reproduces_overlap_column() {
        let alpha = 0.7;
        let c = [8.0, 8.0, 8.0];
        let basis = Basis::new(vec![Shell::new(0, c, vec![alpha], vec![1.0]).unwrap()]);
        let grid = RealSpaceGrid::new(Cell::cubic(16.0).unwrap(), [64, 64, 64]);
        // φ_0(r) = (2α/π)^{3/4} e^{-α|r-c|²} on the grid.
        let norm = (2.0 * alpha / std::f64::consts::PI).powf(0.75);
        let phi: Vec<f64> = grid
            .points()
            .iter()
            .map(|r| {
                let d2 = (r[0] - c[0]).powi(2) + (r[1] - c[1]).powi(2) + (r[2] - c[2]).powi(2);
                norm * (-alpha * d2).exp()
            })
            .collect();
        let b = project_function(&basis, &phi, &grid);
        assert!(
            (b[0] - 1.0).abs() < 1e-6,
            "⟨φ₀|φ₀⟩ = {} (want S₀₀ = 1)",
            b[0]
        );
    }

    /// Collocation and integration are adjoints: `Σ_μν V_μν P_μν = Σ_g dV V(r_g) n(r_g)`.
    #[test]
    fn collocate_integrate_adjoint() {
        let basis = Basis::new(vec![
            Shell::new(0, [8.0, 8.0, 8.0], vec![0.9], vec![1.0]).unwrap(),
            Shell::new_spherical(1, [8.0, 8.5, 8.0], vec![0.6], vec![1.0]).unwrap(),
        ]);
        let grid = RealSpaceGrid::new(Cell::cubic(16.0).unwrap(), [60, 60, 60]);
        let nao = basis.nao();
        // An arbitrary symmetric P.
        let mut p = vec![0.0; nao * nao];
        for a in 0..nao {
            for b in 0..nao {
                p[a * nao + b] = 0.1 * (a as f64 + 1.0) * (b as f64 + 1.0);
            }
        }
        // An arbitrary smooth potential on the grid.
        let v: Vec<f64> = grid
            .points()
            .iter()
            .map(|r| (0.3 * r[0]).sin() + 0.5 * (0.2 * r[1]).cos())
            .collect();

        let n_r = collocate_density(&basis, &p, &grid);
        let vmat = integrate_potential(&basis, &v, &grid);

        let lhs: f64 = (0..nao * nao).map(|i| vmat[i] * p[i]).sum();
        let rhs: f64 = grid.dv() * v.iter().zip(&n_r).map(|(&vv, &nn)| vv * nn).sum::<f64>();
        assert!(
            (lhs - rhs).abs() < 1e-9,
            "adjoint: Σ V·P = {lhs}, Σ dV·V·n = {rhs}"
        );
    }
}
