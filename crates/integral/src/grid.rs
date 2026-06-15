//! Batched grid-point Coulomb matrices: [`Basis::grid_coulomb`] and the
//! kernel-selectable [`Basis::grid_coulomb_kernel`].
//!
//! Semi-numerical exchange (COSX) needs, for a batch of grid points `{r_g}`,
//! the one-index Coulomb matrices `A^g_{μν} = ⟨μ| 1/|r−r_g| |ν⟩`. Each point is
//! expressible as a single-point [`Basis::nuclear`] call, but that re-walks all
//! shell pairs (Gaussian products, prefactors, recurrence bookkeeping) per
//! point. Here the shell-pair work is hoisted out of the point loop
//! ([`crate::engine::os::GridCoulombPair`]) and the points run in the **inner**
//! dimension, so thousands of points amortize one pass over the pairs.
//!
//! Range-separated COSX additionally needs the **long-range** matrices
//! `⟨μ| erf(ω·|r−r_g|)/|r−r_g| |ν⟩`: [`Basis::grid_coulomb_kernel`] accepts an
//! [`EriKernel`] and shares the same hoisted pair structure — the attenuation
//! enters only through the 0th-order Boys ladder
//! (`F_m(T) → s^{m+1/2}·F_m(sT)`, `s = ω²/(ρ+ω²)`; see
//! [`crate::engine::os::GridCoulombPair::eval_erf_into`]).

use crate::engine::os::{GridCoulombPair, GridCoulombScratch};

use crate::integrals::{check_erf_omega, effective_coeffs, place_block, to_func_1e};
use crate::shell::Basis;
use crate::EriKernel;

impl Basis {
    /// Per-grid-point Coulomb matrices `A^g_{μν} = ⟨μ| 1/|r−r_g| |ν⟩`, one
    /// dense `nao × nao` matrix per point.
    ///
    /// Output layout: row-major `[g][μ][ν]` — the point index `g` is slowest,
    /// so `&out[g·nao²..(g+1)·nao²]` is the `g`-th matrix (`nao = `
    /// [`Basis::nao`]; spherical shells contribute their `2l+1` components).
    ///
    /// # Sign convention
    /// `A^g` is the **positive** kernel `⟨μ| 1/|r−r_g| |ν⟩` — the bare Coulomb
    /// potential of a unit *positive* charge, with **no** `−Z` attraction sign.
    /// [`Basis::nuclear`] includes that sign, so per point
    /// `A^g == −nuclear(&[(r_g, 1.0)])`, element for element.
    ///
    /// Each `A^g` is symmetric (Hermitian, real basis): off-diagonal shell
    /// blocks are mirrored from one evaluation (exactly equal across the
    /// diagonal); diagonal shell blocks carry the usual round-off-level
    /// kernel asymmetry, exactly as in [`Basis::nuclear`].
    #[must_use]
    pub fn grid_coulomb(&self, points: &[[f64; 3]]) -> Vec<f64> {
        let nao = self.nao();
        let mut out = vec![0.0; points.len() * nao * nao];
        self.grid_coulomb_into(points, &mut out);
        out
    }

    /// [`Basis::grid_coulomb`] into a caller-provided buffer — the **parallel
    /// seam**: a driver splits its grid into point sub-ranges and calls this
    /// concurrently on disjoint `(points, out)` chunks (the method takes
    /// `&self`; the builder is `Send + Sync`). Every element of `out` is
    /// assigned, so the buffer need not be zeroed.
    ///
    /// Layout and sign convention as [`Basis::grid_coulomb`]; the two are
    /// identical (this is the implementation).
    ///
    /// # Panics
    /// If `out.len() != points.len() · nao²`.
    pub fn grid_coulomb_into(&self, points: &[[f64; 3]], out: &mut [f64]) {
        self.grid_kernel_into(points, None, out);
    }

    /// Like [`Basis::grid_coulomb`] but over the chosen [`EriKernel`] — the
    /// range-separated COSX ingredient.
    ///
    /// - [`EriKernel::Coulomb`] routes to the [`Basis::grid_coulomb`] code path
    ///   itself: the output is **bit-identical**.
    /// - [`EriKernel::Erf`]`{ omega }` evaluates the **long-range** matrices
    ///   `A^g_{μν} = ⟨μ| erf(ω·|r−r_g|)/|r−r_g| |ν⟩` (same positive-kernel
    ///   convention — equivalently, the plain-Coulomb matrices of a normalized
    ///   *Gaussian* charge of exponent `ω²` at `r_g`, since the potential of
    ///   that charge is exactly `erf(ωr)/r`).
    ///
    /// Layout (`[g][μ][ν]` row-major), symmetry, and the supported angular
    /// momenta (every `l ≤ `[`crate::engine::os::MAX_L`], like
    /// [`Basis::nuclear`]) are identical to [`Basis::grid_coulomb`].
    ///
    /// # Panics
    /// Panics if `k` is `Erf { omega }` with `ω ≤ 0`, NaN, or infinite.
    #[must_use]
    pub fn grid_coulomb_kernel(&self, points: &[[f64; 3]], k: EriKernel) -> Vec<f64> {
        let nao = self.nao();
        let mut out = vec![0.0; points.len() * nao * nao];
        self.grid_coulomb_kernel_into(points, k, &mut out);
        out
    }

    /// [`Basis::grid_coulomb_kernel`] into a caller-provided buffer — the same
    /// parallel-partition contract as [`Basis::grid_coulomb_into`] (every
    /// element of `out` is assigned; disjoint point sub-ranges may be filled
    /// concurrently). [`EriKernel::Coulomb`] is bit-identical to
    /// [`Basis::grid_coulomb_into`].
    ///
    /// # Panics
    /// If `out.len() != points.len() · nao²`, or if `k` is `Erf { omega }`
    /// with `ω ≤ 0`, NaN, or infinite.
    pub fn grid_coulomb_kernel_into(&self, points: &[[f64; 3]], k: EriKernel, out: &mut [f64]) {
        match k {
            EriKernel::Coulomb => self.grid_kernel_into(points, None, out),
            EriKernel::Erf { omega } => {
                check_erf_omega(omega);
                self.grid_kernel_into(points, Some(omega), out);
            }
        }
    }

    /// Shared batched-grid body: `omega = None` is the Coulomb kernel (the
    /// only kernel-dependent operation is the per-point Boys-ladder call
    /// inside [`GridCoulombPair`], so this path is bit-identical to the
    /// pre-kernel implementation), `Some(ω)` the erf-attenuated one.
    fn grid_kernel_into(&self, points: &[[f64; 3]], omega: Option<f64>, out: &mut [f64]) {
        let nao = self.nao();
        let mm = nao * nao;
        assert_eq!(
            out.len(),
            points.len() * mm,
            "grid_coulomb output buffer must be n_points · nao² = {} elements",
            points.len() * mm
        );
        let offs = self.offsets();
        let shells = self.shells();
        let eff: Vec<Vec<f64>> = shells.iter().map(effective_coeffs).collect();
        let mut scratch = GridCoulombScratch::default();
        // Shell pairs outer, grid points inner: all pair-only work (Gaussian
        // products, prefactors, recurrence plan, HRR combination) is computed
        // once per canonical pair and reused for every point.
        for (si, sa) in shells.iter().enumerate() {
            for (sj, sb) in shells.iter().enumerate().take(si + 1) {
                let mut prim_pairs = Vec::with_capacity(eff[si].len() * eff[sj].len());
                for (pi, &ca) in eff[si].iter().enumerate() {
                    for (pj, &cb) in eff[sj].iter().enumerate() {
                        prim_pairs.push((sa.exponents()[pi], sb.exponents()[pj], ca * cb));
                    }
                }
                let pair =
                    GridCoulombPair::new(sa.l(), sb.l(), sa.center(), sb.center(), &prim_pairs);
                let (na, nb) = pair.dims();
                let (naf, nbf) = (sa.n_func(), sb.n_func());
                for (g, &c) in points.iter().enumerate() {
                    let mut block = vec![0.0; na * nb];
                    match omega {
                        None => pair.eval_into(c, &mut scratch, &mut block),
                        Some(w) => pair.eval_erf_into(c, w, &mut scratch, &mut block),
                    }
                    let fb = to_func_1e(block, sa, sb);
                    let mat = &mut out[g * mm..(g + 1) * mm];
                    place_block(mat, nao, offs[si], offs[sj], &fb, nbf);
                    if si != sj {
                        // Mirror the block across the diagonal: A^g is
                        // symmetric, so the (j, i) block is the transpose.
                        for a in 0..naf {
                            for b in 0..nbf {
                                mat[(offs[sj] + b) * nao + offs[si] + a] = fb[a * nbf + b];
                            }
                        }
                    }
                }
            }
        }
    }
}
