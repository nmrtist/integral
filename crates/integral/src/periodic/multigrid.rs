//! Multigrid collocation/integration (M6): diffuse primitives on coarse grids, tight
//! primitives on the fine grid (CP2K QUICKSTEP style).
//!
//! The single-grid [`LatticeCollocator`](super::lattice::LatticeCollocator) stays as the
//! validation oracle; this module is additive. The key ideas:
//!
//! - **Bin by primitive, not contracted shell.** Each shell's `(prim_coeff, exps)` is split
//!   per primitive into level groups; the radial sum `Σ_i c_i e^{−α_i ρ²}` is linear in
//!   primitives so any partition re-sums exactly. A primitive of exponent `α` is assigned to
//!   the coarsest level whose density cutoff resolves its self-product (exponent `2α`).
//! - **Per-level new/below partition.** With `a = χ^{=ℓ}` ("new", the sharper level-ℓ
//!   primitives) and `b = χ^{<ℓ}` ("below", coarser), `s = a + b = χ^{≤ℓ}`, the level density
//!   is `n_ℓ = Σ_k w_k Σ_μν P^k_μν (a_μ s_ν* + b_μ a_ν*)` (= `Q(χ^{≤ℓ}) − Q(χ^{<ℓ})`, so
//!   cross-level products are kept). The "new" primitives define the touched set.
//! - **FFT transfer.** `prolongate` (coarse→fine, zero-pad in G) and `restrict` (fine→coarse,
//!   truncate in G) are exact adjoints (the density `n_ℓ` is band-limited to the coarse grid),
//!   so `∫ v n` is preserved and multigrid collocate/integrate stay adjoint.

use std::collections::HashMap;

use rustfft::num_complex::Complex64;

use super::collocate::{build_shell_evals, ShellEval};
use super::fft::Fft3d;
use super::grid::RealSpaceGrid;
use super::lattice::{place_images_into, BlochPhases, PlacedImage};
use crate::shell::Basis;

/// The signed DFT frequency for bin `i` of a length-`n` axis (mirrors `grid::signed_freq`).
fn signed_freq(i: usize, n: usize) -> i32 {
    if i <= n / 2 {
        i as i32
    } else {
        i as i32 - n as i32
    }
}

/// Band-limited transfer between a coarse grid and the fine grid (same cell), via FFT
/// zero-pad (prolongate) / truncate (restrict). The two are **exact adjoints**: for a field
/// band-limited to the coarse grid, `dV_f Σ_f v_f·prolongate(n_c) = dV_c Σ_c restrict(v_f)·n_c`
/// (`dV_f/N_c = dV_c/N_f`). The even-`N` coarse Nyquist bin is dropped so the embed/truncate
/// bin maps are exact transposes (the binned density decays before it).
pub(super) struct GridTransfer {
    fft_fine: Fft3d,
    fft_coarse: Fft3d,
    /// `(coarse_lin, fine_lin)` bin pairs by equal signed frequency.
    pairs: Vec<(usize, usize)>,
    n_coarse: usize,
    n_fine: usize,
    /// `coarse_n == fine_n` — transfer is the identity (the finest level).
    identity: bool,
}

impl GridTransfer {
    /// Build the transfer for coarse dims `coarse_n` into fine dims `fine_n` (each
    /// `coarse_n[a] ≤ fine_n[a]`).
    pub(super) fn new(coarse_n: [usize; 3], fine_n: [usize; 3]) -> Self {
        let identity = coarse_n == fine_n;
        // Per-axis kept coarse indices (drop the lone even-N Nyquist) paired with the fine
        // index of equal signed frequency.
        let axis_pairs = |nc: usize, nf: usize| -> Vec<(usize, usize)> {
            let mut v = Vec::with_capacity(nc);
            for ic in 0..nc {
                if nc % 2 == 0 && ic == nc / 2 {
                    continue; // ambiguous Nyquist
                }
                let f = signed_freq(ic, nc);
                let fi = if f >= 0 {
                    f as usize
                } else {
                    (nf as i32 + f) as usize
                };
                v.push((ic, fi));
            }
            v
        };
        let ap0 = axis_pairs(coarse_n[0], fine_n[0]);
        let ap1 = axis_pairs(coarse_n[1], fine_n[1]);
        let ap2 = axis_pairs(coarse_n[2], fine_n[2]);
        let mut pairs = Vec::with_capacity(ap0.len() * ap1.len() * ap2.len());
        for &(c0, f0) in &ap0 {
            for &(c1, f1) in &ap1 {
                for &(c2, f2) in &ap2 {
                    let clin = (c0 * coarse_n[1] + c1) * coarse_n[2] + c2;
                    let flin = (f0 * fine_n[1] + f1) * fine_n[2] + f2;
                    pairs.push((clin, flin));
                }
            }
        }
        Self {
            fft_fine: Fft3d::new(fine_n),
            fft_coarse: Fft3d::new(coarse_n),
            pairs,
            n_coarse: coarse_n[0] * coarse_n[1] * coarse_n[2],
            n_fine: fine_n[0] * fine_n[1] * fine_n[2],
            identity,
        }
    }

    /// Prolongate a coarse field to the fine grid (band-limited interpolation):
    /// `(1/N_c)·Re IDFT_fine( embed( DFT_coarse(n_c) ) )`.
    pub(super) fn prolongate(&self, coarse: &[f64]) -> Vec<f64> {
        debug_assert_eq!(coarse.len(), self.n_coarse);
        if self.identity {
            return coarse.to_vec();
        }
        let mut c: Vec<Complex64> = coarse.iter().map(|&x| Complex64::new(x, 0.0)).collect();
        self.fft_coarse.forward(&mut c);
        let mut f = vec![Complex64::new(0.0, 0.0); self.n_fine];
        for &(ci, fi) in &self.pairs {
            f[fi] = c[ci];
        }
        self.fft_fine.inverse(&mut f);
        let inv = 1.0 / self.n_coarse as f64;
        f.iter().map(|z| z.re * inv).collect()
    }

    /// Restrict (low-pass) a fine field to the coarse grid — the adjoint of [`prolongate`]:
    /// `(1/N_f)·Re IDFT_coarse( truncate( DFT_fine(v) ) )`.
    pub(super) fn restrict(&self, fine: &[f64]) -> Vec<f64> {
        debug_assert_eq!(fine.len(), self.n_fine);
        if self.identity {
            return fine.to_vec();
        }
        let mut f: Vec<Complex64> = fine.iter().map(|&x| Complex64::new(x, 0.0)).collect();
        self.fft_fine.forward(&mut f);
        let mut c = vec![Complex64::new(0.0, 0.0); self.n_coarse];
        for &(ci, fi) in &self.pairs {
            c[ci] = f[fi];
        }
        self.fft_coarse.inverse(&mut c);
        let inv = 1.0 / self.n_fine as f64;
        c.iter().map(|z| z.re * inv).collect()
    }
}

/// Relative accuracy target for the per-level band-limit (`K = 4·ln(1/ε)`). Chosen at
/// `1e-8`: validated agreement with the single-grid oracle is far tighter (~1e-11–1e-13, since
/// the 7-smooth grid rounding and the factor-4 level steps add conservatism), while keeping
/// only genuinely tight primitives on the fine grid (the diffuse/semi-diffuse ones, the cost
/// driver, drop to coarse grids).
const BIN_EPS: f64 = 1e-8;

/// Minimum per-axis grid dimension for a coarse level (FFT/collocation sanity floor).
const MIN_LEVEL_DIM: usize = 4;

/// The density plane-wave cutoff (hartree) needed to resolve a primitive of exponent `α` —
/// i.e. its self-*product* of exponent `2α`: `K·α` with `K = 4·ln(1/ε)` (product Fourier
/// `e^{−G²/4ζ}`, `ζ = 2α`, screened to `ε` at `G_max = √(2 E_cut)`).
fn cutoff_for_exp(alpha: f64) -> f64 {
    4.0 * (1.0 / BIN_EPS).ln() * alpha
}

/// The grid dimensions of the coarsest level (`e_cut/4^j`) that still resolves a primitive of
/// exponent `α` (cutoff `≥ K·α`), with each dimension floored at [`MIN_LEVEL_DIM`] and capped
/// at the fine grid's dimensions.
fn level_dims_for(cell: &latx::Cell, e_cut: f64, fine_n: [usize; 3], alpha: f64) -> [usize; 3] {
    let needed = cutoff_for_exp(alpha).min(e_cut);
    let ratio = (e_cut / needed).max(1.0);
    let mut j = (ratio.ln() / 4.0_f64.ln()).floor().max(0.0) as i32;
    loop {
        let cutoff = (e_cut / 4.0_f64.powi(j)).max(needed);
        let dims = RealSpaceGrid::from_cutoff(*cell, cutoff).n();
        let ok = dims.iter().all(|&d| d >= MIN_LEVEL_DIM);
        if ok || j == 0 {
            // Never exceed the fine grid (monotone in cutoff, but clamp defensively).
            return [
                dims[0].min(fine_n[0]),
                dims[1].min(fine_n[1]),
                dims[2].min(fine_n[2]),
            ];
        }
        j -= 1; // too coarse — go one level finer
    }
}

/// Clone a base [`ShellEval`] keeping only the primitives flagged in `keep` (aligned with the
/// shell's exponents). Returns `None` if no primitive is kept. The radial sum is linear in
/// primitives, so a subset is the exact partial contraction.
fn subset_shell(base: &ShellEval, keep: &[bool]) -> Option<ShellEval> {
    let prim_coeff: Vec<f64> = base
        .prim_coeff
        .iter()
        .zip(keep)
        .filter_map(|(&c, &k)| k.then_some(c))
        .collect();
    if prim_coeff.is_empty() {
        return None;
    }
    let exps: Vec<f64> = base
        .exps
        .iter()
        .zip(keep)
        .filter_map(|(&e, &k)| k.then_some(e))
        .collect();
    let alpha_min = exps.iter().copied().fold(f64::INFINITY, f64::min);
    Some(ShellEval {
        offset: base.offset,
        n_cart: base.n_cart,
        n_func: base.n_func,
        center: base.center,
        prim_coeff,
        exps,
        comps: base.comps.clone(),
        transform: base.transform.clone(),
        alpha_min,
    })
}

/// One grid level of the multigrid hierarchy: its own resolution grid, the FFT transfer up to
/// the fine grid, and the "new" (this level's primitives) / "below" (all coarser primitives)
/// shell evaluators and their placed lattice images, sharing one triple list.
struct Level {
    grid: RealSpaceGrid,
    transfer: GridTransfer,
    new_shells: Vec<ShellEval>,
    below_shells: Vec<ShellEval>,
    new_placed: Vec<PlacedImage>,
    below_placed: Vec<PlacedImage>,
    unique: Vec<[i32; 3]>,
}

impl Level {
    /// Collocate the level density `n_ℓ = Σ_k w_k Σ_μν P^k_μν (a_μ s_ν* + b_μ a_ν*)` on this
    /// level's grid: `a = χ^{=ℓ}` (new, defines the touched set), `b = χ^{<ℓ}` (below),
    /// `s = a + b`.
    fn collocate(&self, nao: usize, p_k: &[Vec<Complex64>], phases: &BlochPhases) -> Vec<f64> {
        use rayon::prelude::*;
        let grid = &self.grid;
        let nk = phases.nk;
        let nu = phases.n_unique;
        let [_, n1, n2] = grid.n();
        let slab = n1 * n2;
        let mut n_r = vec![0.0; grid.n_points()];
        n_r.par_chunks_mut(slab).enumerate().for_each(|(i, out)| {
            let mut scratch = Vec::with_capacity(16);
            let mut a = vec![Complex64::new(0.0, 0.0); nk * nao];
            let mut b = vec![Complex64::new(0.0, 0.0); nk * nao];
            let mut seen = vec![false; nao];
            let mut touched_new: Vec<usize> = Vec::new();
            let mut touched_all: Vec<usize> = Vec::new();
            for j in 0..n1 {
                for k in 0..n2 {
                    touched_new.clear();
                    touched_all.clear();
                    let r = grid.point([i, j, k]);
                    // "new" (sharper) primitives — define the touched set.
                    for pl in &self.new_placed {
                        let slot = pl.bucket;
                        self.new_shells[pl.shell].eval_at(r, pl.center, &mut scratch, |ao, v| {
                            if !seen[ao] {
                                seen[ao] = true;
                                touched_new.push(ao);
                            }
                            for kk in 0..nk {
                                a[kk * nao + ao] += phases.phase[kk * nu + slot] * v;
                            }
                        });
                    }
                    if touched_new.is_empty() {
                        continue;
                    }
                    touched_all.extend_from_slice(&touched_new);
                    // "below" (diffuse) primitives — only at the localized new-touched points.
                    for pl in &self.below_placed {
                        let slot = pl.bucket;
                        self.below_shells[pl.shell].eval_at(r, pl.center, &mut scratch, |ao, v| {
                            if !seen[ao] {
                                seen[ao] = true;
                                touched_all.push(ao);
                            }
                            for kk in 0..nk {
                                b[kk * nao + ao] += phases.phase[kk * nu + slot] * v;
                            }
                        });
                    }
                    let mut nn_val = 0.0;
                    for kk in 0..nk {
                        let ak = &a[kk * nao..(kk + 1) * nao];
                        let bk = &b[kk * nao..(kk + 1) * nao];
                        let pk = &p_k[kk];
                        let mut term = Complex64::new(0.0, 0.0);
                        // term1 = Σ_{μ∈new} a_μ Σ_{ν∈all} P_μν s_ν*  (s = a + b)
                        for &mu in &touched_new {
                            let prow = &pk[mu * nao..mu * nao + nao];
                            let mut rd = Complex64::new(0.0, 0.0);
                            for &nu_ in &touched_all {
                                rd += prow[nu_] * (ak[nu_] + bk[nu_]).conj();
                            }
                            term += ak[mu] * rd;
                        }
                        // term2 = Σ_{ν∈new} a_ν* Σ_{μ∈all} P_μν b_μ
                        for &nu_ in &touched_new {
                            let mut cd = Complex64::new(0.0, 0.0);
                            for &mu in &touched_all {
                                cd += pk[mu * nao + nu_] * bk[mu];
                            }
                            term += ak[nu_].conj() * cd;
                        }
                        nn_val += phases.weights[kk] * term.re;
                    }
                    out[j * n2 + k] = nn_val;
                    for &ao in &touched_all {
                        seen[ao] = false;
                        for kk in 0..nk {
                            a[kk * nao + ao] = Complex64::new(0.0, 0.0);
                            b[kk * nao + ao] = Complex64::new(0.0, 0.0);
                        }
                    }
                }
            }
        });
        n_r
    }

    /// Integrate the level potential `v` (on this level's grid) to the per-k local matrices —
    /// the adjoint of [`collocate`](Self::collocate):
    /// `V_ℓ(k)_μν = Σ_g dV_ℓ v(g) (a_μ* s_ν + b_μ* a_ν)` (Hermitian).
    fn integrate(&self, nao: usize, v: &[f64], phases: &BlochPhases) -> Vec<Vec<Complex64>> {
        use rayon::prelude::*;
        let grid = &self.grid;
        let nk = phases.nk;
        let nu = phases.n_unique;
        let nn = nao * nao;
        let dv = grid.dv();
        let [n0, n1, n2] = grid.n();
        let slab = n1 * n2;
        let zero = || vec![Complex64::new(0.0, 0.0); nk * nn];
        let flat = (0..n0)
            .into_par_iter()
            .fold(zero, |mut acc, i| {
                let mut scratch = Vec::with_capacity(16);
                let mut a = vec![Complex64::new(0.0, 0.0); nk * nao];
                let mut b = vec![Complex64::new(0.0, 0.0); nk * nao];
                let mut seen = vec![false; nao];
                let mut touched_new: Vec<usize> = Vec::new();
                let mut touched_all: Vec<usize> = Vec::new();
                for j in 0..n1 {
                    for k in 0..n2 {
                        let vg = v[i * slab + j * n2 + k];
                        if vg == 0.0 {
                            continue;
                        }
                        touched_new.clear();
                        touched_all.clear();
                        let r = grid.point([i, j, k]);
                        for pl in &self.new_placed {
                            let slot = pl.bucket;
                            self.new_shells[pl.shell].eval_at(
                                r,
                                pl.center,
                                &mut scratch,
                                |ao, val| {
                                    if !seen[ao] {
                                        seen[ao] = true;
                                        touched_new.push(ao);
                                    }
                                    for kk in 0..nk {
                                        a[kk * nao + ao] += phases.phase[kk * nu + slot] * val;
                                    }
                                },
                            );
                        }
                        if touched_new.is_empty() {
                            continue;
                        }
                        touched_all.extend_from_slice(&touched_new);
                        for pl in &self.below_placed {
                            let slot = pl.bucket;
                            self.below_shells[pl.shell].eval_at(
                                r,
                                pl.center,
                                &mut scratch,
                                |ao, val| {
                                    if !seen[ao] {
                                        seen[ao] = true;
                                        touched_all.push(ao);
                                    }
                                    for kk in 0..nk {
                                        b[kk * nao + ao] += phases.phase[kk * nu + slot] * val;
                                    }
                                },
                            );
                        }
                        let w = dv * vg;
                        for kk in 0..nk {
                            let ak = &a[kk * nao..(kk + 1) * nao];
                            let bk = &b[kk * nao..(kk + 1) * nao];
                            let block = &mut acc[kk * nn..(kk + 1) * nn];
                            // term A: V[μ,ν] += w·a_μ*·s_ν  (μ∈new, ν∈all)
                            for &mu in &touched_new {
                                let wam = Complex64::new(w, 0.0) * ak[mu].conj();
                                let row = &mut block[mu * nao..mu * nao + nao];
                                for &nu_ in &touched_all {
                                    row[nu_] += wam * (ak[nu_] + bk[nu_]);
                                }
                            }
                            // term B: V[μ,ν] += w·b_μ*·a_ν  (μ∈all, ν∈new)
                            for &mu in &touched_all {
                                let bm = bk[mu];
                                if bm == Complex64::new(0.0, 0.0) {
                                    continue;
                                }
                                let wbm = Complex64::new(w, 0.0) * bm.conj();
                                let row = &mut block[mu * nao..mu * nao + nao];
                                for &nu_ in &touched_new {
                                    row[nu_] += wbm * ak[nu_];
                                }
                            }
                        }
                        for &ao in &touched_all {
                            seen[ao] = false;
                            for kk in 0..nk {
                                a[kk * nao + ao] = Complex64::new(0.0, 0.0);
                                b[kk * nao + ao] = Complex64::new(0.0, 0.0);
                            }
                        }
                    }
                }
                acc
            })
            .reduce(zero, |mut x, y| {
                for (p, q) in x.iter_mut().zip(&y) {
                    *p += q;
                }
                x
            });
        (0..nk)
            .map(|kk| flat[kk * nn..(kk + 1) * nn].to_vec())
            .collect()
    }

    /// The level's contribution to the collocation **Pulay force** `F^coll = −∫ v_ℓ ∂n_ℓ/∂R`.
    /// With the new/below partition (`a = χ^{=ℓ}`, `b = χ^{<ℓ}`, `s = a + b`), the per-atom
    /// force is:
    /// `F_I = Σ_g v_ℓ dV_ℓ Σ_k w_k 2 Re[ Σ_{μ∈I} (a'_μ D^s_μ + b'_μ D^a_μ) ]`,
    /// `D^s_μ = Σ_ν P_μν s_ν*`, `D^a_μ = Σ_ν P_μν a_ν*` (`'` = spatial AO gradient; `∂χ/∂R_I =
    /// −χ'`). `v` is this level's potential (`restrict` of the fine potential).
    #[allow(clippy::too_many_arguments)]
    fn collocation_pulay_force(
        &self,
        nao: usize,
        p_k: &[Vec<Complex64>],
        v: &[f64],
        phases: &BlochPhases,
        ao_atom: &[usize],
        natom: usize,
    ) -> Vec<[f64; 3]> {
        use rayon::prelude::*;
        let grid = &self.grid;
        let nk = phases.nk;
        let nu = phases.n_unique;
        let dv = grid.dv();
        let [n0, n1, n2] = grid.n();
        let slab = n1 * n2;
        let zero = || vec![[0.0_f64; 3]; natom];
        (0..n0)
            .into_par_iter()
            .fold(zero, |mut force, i| {
                let mut scratch = Vec::with_capacity(16);
                let mut a = vec![Complex64::new(0.0, 0.0); nk * nao];
                let mut ag = vec![Complex64::new(0.0, 0.0); nk * nao * 3];
                let mut b = vec![Complex64::new(0.0, 0.0); nk * nao];
                let mut bg = vec![Complex64::new(0.0, 0.0); nk * nao * 3];
                let mut seen = vec![false; nao];
                let mut touched_new: Vec<usize> = Vec::new();
                let mut touched_all: Vec<usize> = Vec::new();
                let mut ds = vec![Complex64::new(0.0, 0.0); nao];
                let mut da = vec![Complex64::new(0.0, 0.0); nao];
                for j in 0..n1 {
                    for k in 0..n2 {
                        let vg = v[i * slab + j * n2 + k];
                        if vg == 0.0 {
                            continue;
                        }
                        touched_new.clear();
                        touched_all.clear();
                        let r = grid.point([i, j, k]);
                        for pl in &self.new_placed {
                            let slot = pl.bucket;
                            self.new_shells[pl.shell].emit_grad(
                                r,
                                pl.center,
                                &mut scratch,
                                |ao, val, g| {
                                    if !seen[ao] {
                                        seen[ao] = true;
                                        touched_new.push(ao);
                                    }
                                    for kk in 0..nk {
                                        let ph = phases.phase[kk * nu + slot];
                                        a[kk * nao + ao] += ph * val;
                                        let base = (kk * nao + ao) * 3;
                                        ag[base] += ph * g[0];
                                        ag[base + 1] += ph * g[1];
                                        ag[base + 2] += ph * g[2];
                                    }
                                },
                            );
                        }
                        if touched_new.is_empty() {
                            continue;
                        }
                        touched_all.extend_from_slice(&touched_new);
                        for pl in &self.below_placed {
                            let slot = pl.bucket;
                            self.below_shells[pl.shell].emit_grad(
                                r,
                                pl.center,
                                &mut scratch,
                                |ao, val, g| {
                                    if !seen[ao] {
                                        seen[ao] = true;
                                        touched_all.push(ao);
                                    }
                                    for kk in 0..nk {
                                        let ph = phases.phase[kk * nu + slot];
                                        b[kk * nao + ao] += ph * val;
                                        let base = (kk * nao + ao) * 3;
                                        bg[base] += ph * g[0];
                                        bg[base + 1] += ph * g[1];
                                        bg[base + 2] += ph * g[2];
                                    }
                                },
                            );
                        }
                        let wv = 2.0 * dv * vg;
                        for (kk, &wk) in phases.weights.iter().enumerate() {
                            let ak = &a[kk * nao..(kk + 1) * nao];
                            let bk = &b[kk * nao..(kk + 1) * nao];
                            let pk = &p_k[kk];
                            for &mu in &touched_all {
                                let prow = &pk[mu * nao..mu * nao + nao];
                                let mut s_acc = Complex64::new(0.0, 0.0);
                                let mut a_acc = Complex64::new(0.0, 0.0);
                                for &nu_ in &touched_all {
                                    s_acc += prow[nu_] * (ak[nu_] + bk[nu_]).conj();
                                }
                                for &nu_ in &touched_new {
                                    a_acc += prow[nu_] * ak[nu_].conj();
                                }
                                ds[mu] = s_acc;
                                da[mu] = a_acc;
                            }
                            let wkv = wk * wv;
                            for &mu in &touched_all {
                                let atom = ao_atom[mu];
                                let base = (kk * nao + mu) * 3;
                                let dsm = ds[mu];
                                let dam = da[mu];
                                for axis in 0..3 {
                                    force[atom][axis] +=
                                        wkv * (ag[base + axis] * dsm + bg[base + axis] * dam).re;
                                }
                            }
                        }
                        for &ao in &touched_all {
                            seen[ao] = false;
                            for kk in 0..nk {
                                a[kk * nao + ao] = Complex64::new(0.0, 0.0);
                                b[kk * nao + ao] = Complex64::new(0.0, 0.0);
                                let base = (kk * nao + ao) * 3;
                                for t in 0..3 {
                                    ag[base + t] = Complex64::new(0.0, 0.0);
                                    bg[base + t] = Complex64::new(0.0, 0.0);
                                }
                            }
                        }
                    }
                }
                force
            })
            .reduce(zero, |mut x, y| {
                for (xa, ya) in x.iter_mut().zip(&y) {
                    for t in 0..3 {
                        xa[t] += ya[t];
                    }
                }
                x
            })
    }

    /// The level's contribution to the collocation **Pulay stress** `τ_αβ = ∫ v_ℓ ∂n_ℓ/∂ε_αβ`.
    /// Same structure as [`collocation_pulay_force`](Self::collocation_pulay_force) but with the
    /// AO **gradient ⊗ displacement** `∂χ/∂ε_αβ = (∇φ)_α·(r−τ−S)_β` and a sum over **all** AOs
    /// (the strain deforms every center): `τ_αβ = Σ_g v_ℓ dV_ℓ Σ_k w_k 2 Re[ Σ_μ (a_disp_μ[α][β]
    /// D^s_μ + b_disp_μ[α][β] D^a_μ) ]`.
    fn collocation_pulay_stress(
        &self,
        nao: usize,
        p_k: &[Vec<Complex64>],
        v: &[f64],
        phases: &BlochPhases,
    ) -> [[f64; 3]; 3] {
        use rayon::prelude::*;
        let grid = &self.grid;
        let nk = phases.nk;
        let nu = phases.n_unique;
        let dv = grid.dv();
        let [n0, n1, n2] = grid.n();
        let slab = n1 * n2;
        let zero = || [[0.0_f64; 3]; 3];
        // Per-image gradient-⊗-displacement accumulation into a 9-vector per (k, ao).
        let emit_disp = |chi: &mut [Complex64],
                         cd: &mut [Complex64],
                         seen: &mut [bool],
                         touched: &mut Vec<usize>,
                         placed: &[PlacedImage],
                         shells: &[ShellEval],
                         r: [f64; 3],
                         scratch: &mut Vec<f64>| {
            for pl in placed {
                let slot = pl.bucket;
                let disp = [
                    r[0] - pl.center[0],
                    r[1] - pl.center[1],
                    r[2] - pl.center[2],
                ];
                shells[pl.shell].emit_grad(r, pl.center, scratch, |ao, val, g| {
                    if !seen[ao] {
                        seen[ao] = true;
                        touched.push(ao);
                    }
                    for kk in 0..nk {
                        let ph = phases.phase[kk * nu + slot];
                        chi[kk * nao + ao] += ph * val;
                        let base = (kk * nao + ao) * 9;
                        for (alpha, &ga) in g.iter().enumerate() {
                            let pg = ph * ga;
                            for (beta, &db) in disp.iter().enumerate() {
                                cd[base + alpha * 3 + beta] += pg * db;
                            }
                        }
                    }
                });
            }
        };
        (0..n0)
            .into_par_iter()
            .fold(zero, |mut tau, i| {
                let mut scratch = Vec::with_capacity(16);
                let mut a = vec![Complex64::new(0.0, 0.0); nk * nao];
                let mut adisp = vec![Complex64::new(0.0, 0.0); nk * nao * 9];
                let mut b = vec![Complex64::new(0.0, 0.0); nk * nao];
                let mut bdisp = vec![Complex64::new(0.0, 0.0); nk * nao * 9];
                let mut seen = vec![false; nao];
                let mut touched_new: Vec<usize> = Vec::new();
                let mut touched_all: Vec<usize> = Vec::new();
                let mut ds = vec![Complex64::new(0.0, 0.0); nao];
                let mut da = vec![Complex64::new(0.0, 0.0); nao];
                for j in 0..n1 {
                    for k in 0..n2 {
                        let vg = v[i * slab + j * n2 + k];
                        if vg == 0.0 {
                            continue;
                        }
                        touched_new.clear();
                        touched_all.clear();
                        let r = grid.point([i, j, k]);
                        emit_disp(
                            &mut a,
                            &mut adisp,
                            &mut seen,
                            &mut touched_new,
                            &self.new_placed,
                            &self.new_shells,
                            r,
                            &mut scratch,
                        );
                        if touched_new.is_empty() {
                            continue;
                        }
                        touched_all.extend_from_slice(&touched_new);
                        emit_disp(
                            &mut b,
                            &mut bdisp,
                            &mut seen,
                            &mut touched_all,
                            &self.below_placed,
                            &self.below_shells,
                            r,
                            &mut scratch,
                        );
                        let wv = 2.0 * dv * vg;
                        for (kk, &wk) in phases.weights.iter().enumerate() {
                            let ak = &a[kk * nao..(kk + 1) * nao];
                            let bk = &b[kk * nao..(kk + 1) * nao];
                            let pk = &p_k[kk];
                            for &mu in &touched_all {
                                let prow = &pk[mu * nao..mu * nao + nao];
                                let mut s_acc = Complex64::new(0.0, 0.0);
                                let mut a_acc = Complex64::new(0.0, 0.0);
                                for &nu_ in &touched_all {
                                    s_acc += prow[nu_] * (ak[nu_] + bk[nu_]).conj();
                                }
                                for &nu_ in &touched_new {
                                    a_acc += prow[nu_] * ak[nu_].conj();
                                }
                                ds[mu] = s_acc;
                                da[mu] = a_acc;
                            }
                            let wkv = wk * wv;
                            for &mu in &touched_all {
                                let base = (kk * nao + mu) * 9;
                                let dsm = ds[mu];
                                let dam = da[mu];
                                for (alpha, ta) in tau.iter_mut().enumerate() {
                                    for (beta, tab) in ta.iter_mut().enumerate() {
                                        let idx = base + alpha * 3 + beta;
                                        *tab += wkv * (adisp[idx] * dsm + bdisp[idx] * dam).re;
                                    }
                                }
                            }
                        }
                        for &ao in &touched_all {
                            seen[ao] = false;
                            for kk in 0..nk {
                                a[kk * nao + ao] = Complex64::new(0.0, 0.0);
                                b[kk * nao + ao] = Complex64::new(0.0, 0.0);
                                let base = (kk * nao + ao) * 9;
                                for t in 0..9 {
                                    adisp[base + t] = Complex64::new(0.0, 0.0);
                                    bdisp[base + t] = Complex64::new(0.0, 0.0);
                                }
                            }
                        }
                    }
                }
                tau
            })
            .reduce(zero, |mut x, y| {
                for (xa, ya) in x.iter_mut().zip(&y) {
                    for (p, q) in xa.iter_mut().zip(ya) {
                        *p += q;
                    }
                }
                x
            })
    }
}

/// Precomputed Bloch phases for every multigrid level (one [`BlochPhases`] per level, keyed by
/// that level's shared lattice-triple list).
pub struct MultiBlochPhases {
    per_level: Vec<BlochPhases>,
}

/// A multigrid collocator: diffuse primitives on coarse grids, tight primitives on the fine
/// grid. Drop-in replacement for [`LatticeCollocator`](super::lattice::LatticeCollocator)'s
/// `collocate_k` / `integrate_k` value path — the density is returned on, and the potential
/// consumed on, the **fine** grid ([`fine_grid`](Self::fine_grid)).
pub struct MultiGridCollocator {
    nao: usize,
    fine_grid: RealSpaceGrid,
    levels: Vec<Level>,
}

impl MultiGridCollocator {
    /// Build the multigrid collocator for `basis` on `cell` at density cutoff `e_cut` (hartree).
    /// The fine grid is `RealSpaceGrid::from_cutoff(cell, e_cut)`; coarser levels are derived
    /// per primitive.
    ///
    /// # Panics
    /// Panics if `e_cut` is not positive.
    #[must_use]
    pub fn new(basis: &Basis, cell: &latx::Cell, e_cut: f64) -> Self {
        let fine_grid = RealSpaceGrid::from_cutoff(*cell, e_cut);
        let fine_n = fine_grid.n();
        let base_shells = build_shell_evals(basis);

        // Per-primitive target grid dims; distinct dims ranked by point count (coarsest first).
        let prim_dims: Vec<Vec<[usize; 3]>> = base_shells
            .iter()
            .map(|sh| {
                sh.exps
                    .iter()
                    .map(|&a| level_dims_for(cell, e_cut, fine_n, a))
                    .collect()
            })
            .collect();
        let mut distinct: Vec<[usize; 3]> = Vec::new();
        for pd in &prim_dims {
            for d in pd {
                if !distinct.contains(d) {
                    distinct.push(*d);
                }
            }
        }
        distinct.sort_by_key(|d| d[0] * d[1] * d[2]);
        let rank_of = |d: [usize; 3]| distinct.iter().position(|x| *x == d).unwrap();

        let levels = distinct
            .iter()
            .enumerate()
            .map(|(rank, &dims)| {
                // new = primitives whose level is exactly this rank; below = coarser ranks.
                let mut new_shells = Vec::new();
                let mut below_shells = Vec::new();
                for (si, base) in base_shells.iter().enumerate() {
                    let new_keep: Vec<bool> =
                        prim_dims[si].iter().map(|&d| rank_of(d) == rank).collect();
                    let below_keep: Vec<bool> =
                        prim_dims[si].iter().map(|&d| rank_of(d) < rank).collect();
                    if let Some(s) = subset_shell(base, &new_keep) {
                        new_shells.push(s);
                    }
                    if let Some(s) = subset_shell(base, &below_keep) {
                        below_shells.push(s);
                    }
                }
                let grid = RealSpaceGrid::new(*cell, dims);
                let mut unique: Vec<[i32; 3]> = Vec::new();
                let mut seen: HashMap<[i32; 3], usize> = HashMap::new();
                let new_placed = place_images_into(&new_shells, &grid, &mut unique, &mut seen);
                let below_placed = place_images_into(&below_shells, &grid, &mut unique, &mut seen);
                Level {
                    grid,
                    transfer: GridTransfer::new(dims, fine_n),
                    new_shells,
                    below_shells,
                    new_placed,
                    below_placed,
                    unique,
                }
            })
            .collect();

        Self {
            nao: basis.nao(),
            fine_grid,
            levels,
        }
    }

    /// The fine real-space grid (where the density is returned and the potential consumed).
    #[must_use]
    pub fn fine_grid(&self) -> &RealSpaceGrid {
        &self.fine_grid
    }

    /// The matrix dimension.
    #[must_use]
    pub fn nao(&self) -> usize {
        self.nao
    }

    /// Number of grid levels (1 = single grid, i.e. all primitives need the fine grid).
    #[must_use]
    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }

    /// The grid dimensions of each level (coarsest first) — for diagnostics/tests.
    #[must_use]
    pub fn level_dims(&self) -> Vec<[usize; 3]> {
        self.levels.iter().map(|l| l.grid.n()).collect()
    }

    /// Precompute the per-level Bloch phases for the `collocate_k`/`integrate_k` fast path.
    #[must_use]
    pub fn bloch_phases(&self, k_fracs: &[[f64; 3]], weights: &[f64]) -> MultiBlochPhases {
        MultiBlochPhases {
            per_level: self
                .levels
                .iter()
                .map(|l| BlochPhases::from_unique(&l.unique, k_fracs, weights))
                .collect(),
        }
    }

    /// Collocate the per-k density matrices to `n(r)` on the **fine** grid:
    /// `n_fine = Σ_ℓ prolongate_ℓ(n_ℓ)`.
    ///
    /// # Panics
    /// Panics if `p_k.len()` differs from the k-count.
    #[must_use]
    pub fn collocate_k(&self, p_k: &[Vec<Complex64>], phases: &MultiBlochPhases) -> Vec<f64> {
        let mut n_fine = vec![0.0; self.fine_grid.n_points()];
        for (lvl, ph) in self.levels.iter().zip(&phases.per_level) {
            let n_lvl = lvl.collocate(self.nao, p_k, ph);
            let prol = lvl.transfer.prolongate(&n_lvl);
            for (a, b) in n_fine.iter_mut().zip(&prol) {
                *a += b;
            }
        }
        n_fine
    }

    /// Integrate the fine-grid potential `v` to the per-k local matrices `V_loc(k)`:
    /// `V_loc(k) = Σ_ℓ integrate_ℓ(restrict_ℓ(v))`. Each is row-major `nao²` complex
    /// Hermitian.
    ///
    /// # Panics
    /// Panics if `v.len()` differs from the fine grid point count.
    #[must_use]
    pub fn integrate_k(&self, v: &[f64], phases: &MultiBlochPhases) -> Vec<Vec<Complex64>> {
        assert_eq!(
            v.len(),
            self.fine_grid.n_points(),
            "potential length must equal fine grid points"
        );
        let nn = self.nao * self.nao;
        let nk = phases.per_level.first().map_or(0, |p| p.nk);
        let mut v_loc = vec![vec![Complex64::new(0.0, 0.0); nn]; nk];
        for (lvl, ph) in self.levels.iter().zip(&phases.per_level) {
            let v_lvl = lvl.transfer.restrict(v);
            let v_k = lvl.integrate(self.nao, &v_lvl, ph);
            for (dst, src) in v_loc.iter_mut().zip(&v_k) {
                for (d, s) in dst.iter_mut().zip(src) {
                    *d += s;
                }
            }
        }
        v_loc
    }

    /// The **collocation Pulay force** `F^coll = −∫ V_loc ∂n/∂R`: the fine-grid potential `v`
    /// is restricted to each level, and per-level forces are summed.
    /// Returns the physical force `[x, y, z]` per atom (`ao_atom[μ]` labels each AO's atom).
    /// Drop-in for [`LatticeCollocator::collocation_pulay_forces`](super::lattice::LatticeCollocator::collocation_pulay_forces).
    ///
    /// # Panics
    /// Panics if `v.len()` differs from the fine grid point count, `p_k.len()` differs from the
    /// k-count, or `ao_atom.len() != nao`.
    #[must_use]
    pub fn collocation_pulay_forces(
        &self,
        p_k: &[Vec<Complex64>],
        v: &[f64],
        phases: &MultiBlochPhases,
        ao_atom: &[usize],
        natom: usize,
    ) -> Vec<[f64; 3]> {
        assert_eq!(
            v.len(),
            self.fine_grid.n_points(),
            "potential length must equal fine grid points"
        );
        assert_eq!(ao_atom.len(), self.nao, "ao_atom must label every AO");
        let mut force = vec![[0.0_f64; 3]; natom];
        for (lvl, ph) in self.levels.iter().zip(&phases.per_level) {
            let v_lvl = lvl.transfer.restrict(v);
            let f_lvl = lvl.collocation_pulay_force(self.nao, p_k, &v_lvl, ph, ao_atom, natom);
            for (dst, src) in force.iter_mut().zip(&f_lvl) {
                for ax in 0..3 {
                    dst[ax] += src[ax];
                }
            }
        }
        force
    }

    /// The **collocation Pulay stress** `τ_αβ = ∫ V_loc ∂n/∂ε_αβ`: the fine-grid potential `v`
    /// is restricted to each level and the per-level stresses summed.
    /// Generally non-symmetric (the caller symmetrizes). Drop-in for
    /// [`LatticeCollocator::collocation_pulay_stress`](super::lattice::LatticeCollocator::collocation_pulay_stress).
    ///
    /// # Panics
    /// Panics if `v.len()` differs from the fine grid point count or `p_k.len()` ≠ the k-count.
    #[must_use]
    pub fn collocation_pulay_stress(
        &self,
        p_k: &[Vec<Complex64>],
        v: &[f64],
        phases: &MultiBlochPhases,
    ) -> [[f64; 3]; 3] {
        assert_eq!(
            v.len(),
            self.fine_grid.n_points(),
            "potential length must equal fine grid points"
        );
        let mut tau = [[0.0_f64; 3]; 3];
        for (lvl, ph) in self.levels.iter().zip(&phases.per_level) {
            let v_lvl = lvl.transfer.restrict(v);
            let t_lvl = lvl.collocation_pulay_stress(self.nao, p_k, &v_lvl, ph);
            for (ta, sa) in tau.iter_mut().zip(&t_lvl) {
                for (x, &y) in ta.iter_mut().zip(sa) {
                    *x += y;
                }
            }
        }
        tau
    }
}

#[cfg(test)]
mod collocator_tests {
    use super::*;
    use crate::periodic::lattice::LatticeCollocator;
    use crate::{Basis, Shell};
    use latx::Cell;

    /// A multi-exponent Si-like basis (contracted s/p with a diffuse tail + a d shell) on two
    /// nearby atoms in a small cell, so the exponent spread forces several grid levels and the
    /// lattice image sum genuinely engages.
    fn spread_basis(c0: [f64; 3], c1: [f64; 3]) -> Basis {
        let exps = vec![1.20, 0.47, 0.17, 0.058];
        let s = vec![0.33, -0.25, -0.79, -0.19];
        let p = vec![0.047, -0.26, -0.54, -0.36];
        let mut shells = Vec::new();
        for c in [c0, c1] {
            shells.push(Shell::new_spherical(0, c, exps.clone(), s.clone()).unwrap());
            shells.push(Shell::new_spherical(1, c, exps.clone(), p.clone()).unwrap());
            shells.push(Shell::new_spherical(2, c, vec![0.45], vec![1.0]).unwrap());
        }
        Basis::new(shells)
    }

    /// Deterministic Hermitian per-k density matrices.
    fn hermitian_pk(nao: usize, nk: usize) -> Vec<Vec<Complex64>> {
        (0..nk)
            .map(|kk| {
                let mut p = vec![Complex64::new(0.0, 0.0); nao * nao];
                for a in 0..nao {
                    for b in 0..nao {
                        let re = 0.2 * (((a * 3 + b + kk) as f64) * 0.13).sin();
                        let im = 0.1 * (((a + 2 * b + kk) as f64) * 0.21).cos();
                        p[a * nao + b] = Complex64::new(re, im);
                    }
                }
                let mut h = vec![Complex64::new(0.0, 0.0); nao * nao];
                for a in 0..nao {
                    for b in 0..nao {
                        h[a * nao + b] =
                            (p[a * nao + b] + p[b * nao + a].conj()) * Complex64::new(0.5, 0.0);
                    }
                }
                h
            })
            .collect()
    }

    const A: f64 = 7.0;
    const ECUT: f64 = 80.0;

    fn setup() -> (Basis, Cell) {
        let cell = Cell::cubic(A).unwrap();
        let basis = spread_basis([0.0, 0.0, 0.0], [1.8, 1.8, 1.8]);
        (basis, cell)
    }

    /// The multigrid actually splits the basis across more than one grid level.
    #[test]
    fn multigrid_has_several_levels() {
        let (basis, cell) = setup();
        let mgc = MultiGridCollocator::new(&basis, &cell, ECUT);
        eprintln!("[multigrid] levels = {:?}", mgc.level_dims());
        assert!(
            mgc.n_levels() >= 2,
            "expected multiple levels, got {:?}",
            mgc.level_dims()
        );
    }

    /// Multigrid `collocate_k` matches the single-grid oracle on the fine grid: both at Γ and
    /// at non-Γ k-points, to the per-level band-limit accuracy.
    #[test]
    fn collocate_k_matches_single_grid() {
        let (basis, cell) = setup();
        let nao = basis.nao();
        let kfracs = [[0.0, 0.0, 0.0], [0.3, -0.1, 0.2]];
        let weights = [0.6, 0.4];
        let p_k = hermitian_pk(nao, kfracs.len());

        let grid = RealSpaceGrid::from_cutoff(cell, ECUT);
        let lc = LatticeCollocator::new(&basis, &grid);
        let lph = lc.bloch_phases(&kfracs, &weights);
        let n_single = lc.collocate_k(&grid, &p_k, &lph);

        let mgc = MultiGridCollocator::new(&basis, &cell, ECUT);
        assert_eq!(mgc.fine_grid().n(), grid.n(), "fine grid must match oracle");
        let mph = mgc.bloch_phases(&kfracs, &weights);
        let n_multi = mgc.collocate_k(&p_k, &mph);

        let maxn = n_single.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
        let maxdiff = n_single
            .iter()
            .zip(&n_multi)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        eprintln!(
            "[multigrid] collocate: max|n| = {maxn:.4}, max|Δ| = {maxdiff:.3e}, rel = {:.3e}",
            maxdiff / maxn
        );
        assert!(maxdiff < 1e-8 * maxn.max(1.0), "collocate Δ = {maxdiff}");
    }

    /// Multigrid `integrate_k` matches the single-grid oracle, the result is Hermitian, and it
    /// is the adjoint of `collocate_k` (`Σ_k Re Tr(V_loc(k) P^k) = Σ_g dV·v·n`).
    #[test]
    fn integrate_k_matches_single_grid_and_is_adjoint() {
        use std::f64::consts::PI;
        let (basis, cell) = setup();
        let nao = basis.nao();
        let kfracs = [[0.0, 0.0, 0.0], [0.25, 0.15, -0.2]];
        let weights = [0.5, 0.5];
        let p_k = hermitian_pk(nao, kfracs.len());

        let grid = RealSpaceGrid::from_cutoff(cell, ECUT);
        // A smooth periodic potential on the fine grid.
        let v: Vec<f64> = grid
            .points()
            .iter()
            .map(|r| (2.0 * PI * r[0] / A).cos() + 0.4 * (2.0 * PI * r[1] / A).sin() + 1.0)
            .collect();

        let lc = LatticeCollocator::new(&basis, &grid);
        let lph = lc.bloch_phases(&kfracs, &weights);
        let v_single = lc.integrate_k(&grid, &v, &lph);

        let mgc = MultiGridCollocator::new(&basis, &cell, ECUT);
        let mph = mgc.bloch_phases(&kfracs, &weights);
        let v_multi = mgc.integrate_k(&v, &mph);

        // Match the oracle.
        let mut maxdiff = 0.0_f64;
        let mut maxv = 0.0_f64;
        for (vs, vm) in v_single.iter().zip(&v_multi) {
            for (a, b) in vs.iter().zip(vm) {
                maxdiff = maxdiff.max((a - b).norm());
                maxv = maxv.max(a.norm());
            }
        }
        eprintln!(
            "[multigrid] integrate: max|V| = {maxv:.4}, max|Δ| = {maxdiff:.3e}, rel = {:.3e}",
            maxdiff / maxv
        );
        assert!(maxdiff < 1e-8 * maxv.max(1.0), "integrate Δ = {maxdiff}");

        // Hermitian.
        let mut maxherm = 0.0_f64;
        for vk in &v_multi {
            for a in 0..nao {
                for b in 0..nao {
                    maxherm = maxherm.max((vk[a * nao + b] - vk[b * nao + a].conj()).norm());
                }
            }
        }
        assert!(maxherm < 1e-10, "V_loc(k) not Hermitian: {maxherm}");

        // Adjoint with multigrid collocate: Σ_k Re Tr(V P) = Σ_g dV v n.
        let n_multi = mgc.collocate_k(&p_k, &mph);
        let mut lhs = 0.0;
        for (kk, &w) in weights.iter().enumerate() {
            let mut tr = 0.0;
            for mu in 0..nao {
                for nu in 0..nao {
                    tr += (v_multi[kk][mu * nao + nu] * p_k[kk][nu * nao + mu]).re;
                }
            }
            lhs += w * tr;
        }
        let rhs: f64 = grid.dv()
            * v.iter()
                .zip(&n_multi)
                .map(|(&vv, &nn)| vv * nn)
                .sum::<f64>();
        eprintln!("[multigrid] adjoint: Tr(VP) = {lhs:.8}, Σdv·v·n = {rhs:.8}");
        assert!(
            (lhs - rhs).abs() < 1e-9,
            "multigrid adjoint: {lhs} vs {rhs}"
        );
    }

    /// AO → atom map for the two-atom spread basis (shells on atom 0 vs atom 1 by center).
    fn ao_atom(basis: &Basis, c0: [f64; 3]) -> Vec<usize> {
        let mut map = Vec::with_capacity(basis.nao());
        for sh in basis.shells() {
            let c = sh.center();
            let atom = usize::from((0..3).map(|x| (c[x] - c0[x]).powi(2)).sum::<f64>() >= 1e-12);
            for _ in 0..sh.n_func() {
                map.push(atom);
            }
        }
        map
    }

    /// Multigrid collocation **Pulay force** matches the single-grid oracle (itself FD-validated):
    /// both compute `−∫ V_loc ∂n/∂R` from the same potential and `P^k`.
    #[test]
    fn collocation_pulay_forces_match_single_grid() {
        use std::f64::consts::PI;
        let c0 = [0.0, 0.0, 0.0];
        let c1 = [1.8, 1.8, 1.8];
        let cell = Cell::cubic(A).unwrap();
        let basis = spread_basis(c0, c1);
        let nao = basis.nao();
        let aoatom = ao_atom(&basis, c0);
        let kfracs = [[0.0, 0.0, 0.0], [0.3, -0.1, 0.2]];
        let weights = [0.6, 0.4];
        let p_k = hermitian_pk(nao, kfracs.len());

        let grid = RealSpaceGrid::from_cutoff(cell, ECUT);
        let v: Vec<f64> = grid
            .points()
            .iter()
            .map(|r| (2.0 * PI * r[0] / A).cos() + 0.4 * (2.0 * PI * r[1] / A).sin() + 1.0)
            .collect();

        let lc = LatticeCollocator::new(&basis, &grid);
        let lph = lc.bloch_phases(&kfracs, &weights);
        let f_single = lc.collocation_pulay_forces(&grid, &p_k, &v, &lph, &aoatom, 2);

        let mgc = MultiGridCollocator::new(&basis, &cell, ECUT);
        let mph = mgc.bloch_phases(&kfracs, &weights);
        let f_multi = mgc.collocation_pulay_forces(&p_k, &v, &mph, &aoatom, 2);

        let mut maxdiff = 0.0_f64;
        let mut maxf = 0.0_f64;
        for (fs, fm) in f_single.iter().zip(&f_multi) {
            for ax in 0..3 {
                maxdiff = maxdiff.max((fs[ax] - fm[ax]).abs());
                maxf = maxf.max(fs[ax].abs());
            }
        }
        eprintln!("[multigrid] pulay force: max|F| = {maxf:.4}, max|Δ| = {maxdiff:.3e}");
        assert!(maxdiff < 1e-8 * maxf.max(1.0), "force Δ = {maxdiff}");
    }

    /// Multigrid collocation **Pulay stress** matches the single-grid oracle (FD-validated).
    #[test]
    fn collocation_pulay_stress_matches_single_grid() {
        use std::f64::consts::PI;
        let c0 = [0.0, 0.0, 0.0];
        let c1 = [1.8, 1.8, 1.8];
        let cell = Cell::cubic(A).unwrap();
        let basis = spread_basis(c0, c1);
        let nao = basis.nao();
        let kfracs = [[0.0, 0.0, 0.0], [0.25, 0.15, -0.2]];
        let weights = [0.5, 0.5];
        let p_k = hermitian_pk(nao, kfracs.len());

        let grid = RealSpaceGrid::from_cutoff(cell, ECUT);
        let v: Vec<f64> = grid
            .points()
            .iter()
            .map(|r| (2.0 * PI * r[2] / A).cos() + 0.3 * (2.0 * PI * r[0] / A).sin() + 1.0)
            .collect();

        let lc = LatticeCollocator::new(&basis, &grid);
        let lph = lc.bloch_phases(&kfracs, &weights);
        let t_single = lc.collocation_pulay_stress(&grid, &p_k, &v, &lph);

        let mgc = MultiGridCollocator::new(&basis, &cell, ECUT);
        let mph = mgc.bloch_phases(&kfracs, &weights);
        let t_multi = mgc.collocation_pulay_stress(&p_k, &v, &mph);

        let mut maxdiff = 0.0_f64;
        let mut maxt = 0.0_f64;
        for a in 0..3 {
            for b in 0..3 {
                maxdiff = maxdiff.max((t_single[a][b] - t_multi[a][b]).abs());
                maxt = maxt.max(t_single[a][b].abs());
            }
        }
        eprintln!("[multigrid] pulay stress: max|τ| = {maxt:.4}, max|Δ| = {maxdiff:.3e}");
        assert!(maxdiff < 1e-8 * maxt.max(1.0), "stress Δ = {maxdiff}");
    }
}

#[cfg(test)]
mod transfer_tests {
    use super::*;
    use latx::Cell;
    use std::f64::consts::PI;

    /// A field band-limited to the coarse grid prolongates to the fine grid and restricts back
    /// to itself (round-trip exact), and the prolongation samples the same continuous function.
    #[test]
    fn prolongate_restrict_round_trip_band_limited() {
        let l = 8.0;
        let cell = Cell::cubic(l).unwrap();
        let coarse_n = [9usize, 9, 9]; // odd → no Nyquist drop
        let fine_n = [27usize, 27, 27];
        let coarse = RealSpaceGrid::new(cell, coarse_n);
        let fine = RealSpaceGrid::new(cell, fine_n);
        let tr = GridTransfer::new(coarse_n, fine_n);

        // A low-frequency field (well inside the coarse band): cos/sin of the first few G.
        let two_pi_l = 2.0 * PI / l;
        let field = |r: [f64; 3]| {
            1.0 + (two_pi_l * r[0]).cos() + 0.5 * (2.0 * two_pi_l * r[1]).sin()
                - 0.3 * (two_pi_l * r[2]).cos()
        };
        let nc: Vec<f64> = coarse.points().iter().map(|&r| field(r)).collect();
        let nf_exact: Vec<f64> = fine.points().iter().map(|&r| field(r)).collect();

        let nf = tr.prolongate(&nc);
        let maxp = nf
            .iter()
            .zip(&nf_exact)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(maxp < 1e-10, "prolongate vs exact: {maxp}");

        let nc_back = tr.restrict(&nf);
        let maxr = nc_back
            .iter()
            .zip(&nc)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(maxr < 1e-10, "restrict∘prolongate round trip: {maxr}");
    }

    /// Prolongate and restrict are exact adjoints across the two quadratures:
    /// `dV_f Σ_f v·P(n_c) = dV_c Σ_c R(v)·n_c` for an arbitrary fine `v` and coarse `n_c`.
    #[test]
    fn prolongate_restrict_are_adjoint() {
        let l = 6.0;
        let cell = Cell::cubic(l).unwrap();
        let coarse_n = [8usize, 10, 12]; // even → exercises the Nyquist drop
        let fine_n = [24usize, 24, 24];
        let coarse = RealSpaceGrid::new(cell, coarse_n);
        let fine = RealSpaceGrid::new(cell, fine_n);
        let tr = GridTransfer::new(coarse_n, fine_n);

        // Arbitrary (NOT band-limited) fine v and arbitrary coarse n_c.
        let vf: Vec<f64> = (0..fine.n_points())
            .map(|i| (i as f64 * 0.31).sin() + 0.2 * (i as f64 * 0.07).cos())
            .collect();
        let ncoarse: Vec<f64> = (0..coarse.n_points())
            .map(|i| (i as f64 * 0.19 + 0.5).cos())
            .collect();

        let lhs: f64 = fine.dv()
            * vf.iter()
                .zip(&tr.prolongate(&ncoarse))
                .map(|(&a, &b)| a * b)
                .sum::<f64>();
        let rhs: f64 = coarse.dv()
            * tr.restrict(&vf)
                .iter()
                .zip(&ncoarse)
                .map(|(&a, &b)| a * b)
                .sum::<f64>();
        assert!((lhs - rhs).abs() < 1e-10, "adjoint: {lhs} vs {rhs}");
    }

    /// The identity transfer (coarse == fine) is a no-op copy.
    #[test]
    fn identity_transfer_is_copy() {
        let n = [16usize, 16, 16];
        let tr = GridTransfer::new(n, n);
        let f: Vec<f64> = (0..n[0] * n[1] * n[2]).map(|i| i as f64 * 0.5).collect();
        assert_eq!(tr.prolongate(&f), f);
        assert_eq!(tr.restrict(&f), f);
    }
}
