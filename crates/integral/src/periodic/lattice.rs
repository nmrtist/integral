//! Lattice-image-resolved collocation and integration (the M3 generalization of the
//! M2 minimum-image [`collocate_density`](super::collocate_density) /
//! [`integrate_potential`](super::integrate_potential)).
//!
//! In a tightly packed periodic cell a Gaussian reaches several periodic images, so the
//! density and the local potential matrix are **real-space lattice sums** over images
//! `S` (integer lattice triples). Both operations share one per-grid-point kernel: at
//! each point, evaluate every shell at **all of its images within its Gaussian cutoff**
//! (no minimum-image wrap), giving `active = [(ao, S, value)]`. Then
//!
//! - **collocation** `n(r_g) = Σ_{a,b} P^{S_a−S_b}_{μ_a ν_b} · val_a · val_b`, where
//!   `P^S` is the image-resolved (real-space) density matrix
//!   `P^S_{μν} = Σ_k w_k e^{−ik·S} P^k_{μν}`;
//! - **integration** accumulates `W^{S_b−S_a}_{μ_a ν_b} += dV·V(r_g)·val_a·val_b`, and the
//!   Bloch local matrix is `V_loc(k) = Σ_S e^{ik·S} W^S` (formed by the caller).
//!
//! The derivation `n(r) = Σ_{μν,R,R'} P^{R−R'}_{μν} φ_μ(r−τ_μ−R) φ_ν(r−τ_ν−R')` shows the
//! symmetric double image sum collapses to the `active×active` fold above, and that the
//! same per-point image evaluation serves both operations. At a single Γ point `P^S=P^Γ`
//! for every `S`, so this path also builds the ordinary Γ density from the periodic-summed
//! (Bloch-Γ) AOs — there is no separate Γ collocation.

use std::collections::HashMap;

use rustfft::num_complex::Complex64;

use super::collocate::{build_shell_evals, ShellEval, SCREEN_EXP};
use super::grid::RealSpaceGrid;
use crate::shell::Basis;

/// Precomputed Bloch phases `e^{ik·S}` for each (k-point, distinct reaching-image triple)
/// plus the k-weights. Built once per k-mesh and reused across SCF iterations by the
/// per-k collocation/integration ([`LatticeCollocator::collocate_k`] /
/// [`LatticeCollocator::integrate_k`]).
pub struct BlochPhases {
    pub(super) nk: usize,
    pub(super) n_unique: usize,
    pub(super) weights: Vec<f64>,
    /// `[k * n_unique + slot]` — the phase of unique image `slot` at k-point `k`.
    pub(super) phase: Vec<Complex64>,
}

impl BlochPhases {
    /// Build phases `e^{ik·S}` for the `unique` lattice triples at each k-point. Shared by the
    /// single-grid [`LatticeCollocator::bloch_phases`] and the multigrid collocator.
    pub(super) fn from_unique(unique: &[[i32; 3]], k_fracs: &[[f64; 3]], weights: &[f64]) -> Self {
        assert_eq!(
            k_fracs.len(),
            weights.len(),
            "k-points and weights must align"
        );
        let n_unique = unique.len();
        let nk = k_fracs.len();
        let mut phase = vec![Complex64::new(0.0, 0.0); nk * n_unique];
        for (ik, k) in k_fracs.iter().enumerate() {
            for (slot, s) in unique.iter().enumerate() {
                let theta = 2.0
                    * std::f64::consts::PI
                    * (k[0] * f64::from(s[0]) + k[1] * f64::from(s[1]) + k[2] * f64::from(s[2]));
                phase[ik * n_unique + slot] = Complex64::new(theta.cos(), theta.sin());
            }
        }
        Self {
            nk,
            n_unique,
            weights: weights.to_vec(),
            phase,
        }
    }
}

/// A real-space, image-resolved set of `nao × nao` matrix blocks keyed by lattice
/// triple `S` (the density matrix `P^S` for collocation, or the integrated `W^S`).
///
/// Each block is row-major `nao × nao`. Missing triples are treated as zero.
#[derive(Debug, Clone)]
pub struct ImageBlocks {
    nao: usize,
    triples: Vec<[i32; 3]>,
    blocks: Vec<Vec<f64>>,
    index: HashMap<[i32; 3], usize>,
}

impl ImageBlocks {
    /// Blocks built from a closure of the triple — `block(s)` returns the row-major
    /// `nao × nao` data for triple `s`. Used by the per-k SCF to lay down the
    /// real-space density matrix `P^S = Σ_k w_k e^{−ik·S} P^k`.
    #[must_use]
    pub fn from_fn(
        nao: usize,
        triples: &[[i32; 3]],
        mut block: impl FnMut([i32; 3]) -> Vec<f64>,
    ) -> Self {
        let mut index = HashMap::new();
        let mut tris = Vec::new();
        let mut blocks = Vec::new();
        for &t in triples {
            if index.insert(t, tris.len()).is_none() {
                let b = block(t);
                assert_eq!(
                    b.len(),
                    nao * nao,
                    "block for {t:?} must be nao² = {}",
                    nao * nao
                );
                tris.push(t);
                blocks.push(b);
            }
        }
        Self {
            nao,
            triples: tris,
            blocks,
            index,
        }
    }

    /// The same `nao × nao` block for every triple — the Γ-point density matrix `P^Γ`,
    /// which is `P^S` for *every* image `S` when the mesh is a single Γ point.
    #[must_use]
    pub fn constant(nao: usize, triples: &[[i32; 3]], block: &[f64]) -> Self {
        assert_eq!(block.len(), nao * nao, "block must be nao² = {}", nao * nao);
        Self::from_fn(nao, triples, |_| block.to_vec())
    }

    /// Zero-initialized blocks for the given lattice triples (deduplicated).
    #[must_use]
    pub fn zeros(nao: usize, triples: &[[i32; 3]]) -> Self {
        let mut index = HashMap::new();
        let mut tris = Vec::new();
        let mut blocks = Vec::new();
        for &t in triples {
            if index.insert(t, tris.len()).is_none() {
                tris.push(t);
                blocks.push(vec![0.0; nao * nao]);
            }
        }
        Self {
            nao,
            triples: tris,
            blocks,
            index,
        }
    }

    /// Matrix dimension.
    #[must_use]
    pub fn nao(&self) -> usize {
        self.nao
    }

    /// The lattice triples carried, in insertion order.
    #[must_use]
    pub fn triples(&self) -> &[[i32; 3]] {
        &self.triples
    }

    /// Number of blocks (= number of distinct triples).
    #[must_use]
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    /// Whether there are no blocks.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// The block for triple `s` (row-major `nao × nao`), or `None` if absent.
    #[must_use]
    pub fn get(&self, s: [i32; 3]) -> Option<&[f64]> {
        self.index.get(&s).map(|&i| self.blocks[i].as_slice())
    }

    /// The block at storage index `i` (aligned with [`triples`](Self::triples)).
    #[must_use]
    pub fn block(&self, i: usize) -> &[f64] {
        &self.blocks[i]
    }

    /// The triple at storage index `i`.
    #[must_use]
    pub fn triple(&self, i: usize) -> [i32; 3] {
        self.triples[i]
    }
}

/// The radius (bohr) beyond which a shell's contribution is below `REACH_EPS`, used to
/// bound the lattice images enumerated for it. Coefficient-aware: a contracted shell's
/// diffuse primitive has a small effective coefficient `|c_i·N(α_i)|`, so its true reach is
/// much shorter than the uniform `√(SCREEN_EXP/α_min)`. Per primitive, the value
/// `|c_i·N|·ρ^l·e^{−α_i ρ²}` drops below `REACH_EPS` at `ρ² = (ln(|c_i·N|/REACH_EPS)+2l)/α_i`;
/// the reach is the max over primitives, capped at the emit screen so nothing the
/// per-point screen keeps is dropped.
pub(super) fn shell_reach(sh: &ShellEval) -> f64 {
    const REACH_EPS: f64 = 1e-12;
    let l = sh.comps.first().map_or(0, |c| c[0] + c[1] + c[2]); // (l,0,0) is first
    let margin = 2.0 * l as f64;
    let mut r2 = 0.0_f64;
    for (&pc, &a) in sh.prim_coeff.iter().zip(&sh.exps) {
        if a <= 0.0 {
            return f64::INFINITY; // constant (zero-exponent) shell — no decay
        }
        let abs = pc.abs();
        if abs <= REACH_EPS {
            continue;
        }
        let screen = ((abs / REACH_EPS).ln() + margin).min(SCREEN_EXP);
        r2 = r2.max(screen / a);
    }
    r2.sqrt()
}

/// A shell evaluator placed at one lattice image: the integer triple `S`, the
/// (un-wrapped) Cartesian center `τ_shell + S·A`, and the fixed bucket slot of its triple
/// (its index in [`LatticeCollocator::unique`]).
pub(super) struct PlacedImage {
    pub(super) shell: usize,
    pub(super) center: [f64; 3],
    pub(super) bucket: usize,
}

/// The cell circumradius (max corner distance from the cell center).
pub(super) fn cell_circumradius(cell: &latx::Cell) -> f64 {
    let center = cell.frac_to_cart([0.5, 0.5, 0.5]);
    let mut r_circ = 0.0_f64;
    for c in [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ] {
        let p = cell.frac_to_cart(c);
        let d =
            ((p[0] - center[0]).powi(2) + (p[1] - center[1]).powi(2) + (p[2] - center[2]).powi(2))
                .sqrt();
        r_circ = r_circ.max(d);
    }
    r_circ
}

/// Place every shell's reaching lattice images into the home cell, bucketing each distinct
/// lattice triple `S` into `unique` (shared across calls through `seen`, so several shell sets
/// — e.g. the multigrid "new"/"below" groups on one level — index a common triple list).
/// Returns the placed images, whose `shell` field indexes `shells`.
pub(super) fn place_images_into(
    shells: &[ShellEval],
    grid: &RealSpaceGrid,
    unique: &mut Vec<[i32; 3]>,
    seen: &mut HashMap<[i32; 3], usize>,
) -> Vec<PlacedImage> {
    let cell = grid.cell();
    let center = cell.frac_to_cart([0.5, 0.5, 0.5]);
    let r_circ = cell_circumradius(cell);
    let mut placed = Vec::new();
    for (si, sh) in shells.iter().enumerate() {
        // Radius beyond which the shell's value is negligible (coefficient-aware, capped at
        // the emit screen). See [`shell_reach`].
        let rcut = shell_reach(sh);
        let reach = rcut + r_circ;
        for (triple, r) in cell.lattice_images(reach + r_circ) {
            let ic = [
                sh.center[0] + r[0],
                sh.center[1] + r[1],
                sh.center[2] + r[2],
            ];
            let d = ((ic[0] - center[0]).powi(2)
                + (ic[1] - center[1]).powi(2)
                + (ic[2] - center[2]).powi(2))
            .sqrt();
            if d <= reach + 1e-9 {
                let bucket = *seen.entry(triple).or_insert_with(|| {
                    let s = unique.len();
                    unique.push(triple);
                    s
                });
                placed.push(PlacedImage {
                    shell: si,
                    center: ic,
                    bucket,
                });
            }
        }
    }
    placed
}

/// A flat lookup table mapping a small lattice triple to its block index, replacing a
/// `HashMap` in the per-grid-point hot loop. Triples outside `[−b, b]³` (or not present)
/// return `None`.
struct FlatTripleIndex {
    b: i32,
    span: usize,
    table: Vec<i32>,
}

impl FlatTripleIndex {
    fn new(triples: &[[i32; 3]]) -> Self {
        let b = triples
            .iter()
            .flat_map(|t| t.iter().map(|x| x.abs()))
            .max()
            .unwrap_or(0)
            + 1;
        let span = (2 * b + 1) as usize;
        let mut table = vec![-1i32; span * span * span];
        for (i, t) in triples.iter().enumerate() {
            table[Self::pack(b, span, *t)] = i as i32;
        }
        Self { b, span, table }
    }

    #[inline]
    fn pack(b: i32, span: usize, s: [i32; 3]) -> usize {
        (((s[0] + b) as usize * span) + (s[1] + b) as usize) * span + (s[2] + b) as usize
    }

    #[inline]
    fn get(&self, s: [i32; 3]) -> Option<usize> {
        if s[0].abs() > self.b || s[1].abs() > self.b || s[2].abs() > self.b {
            return None;
        }
        let i = self.table[Self::pack(self.b, self.span, s)];
        (i >= 0).then_some(i as usize)
    }
}

/// A geometry-fixed cache of the Bloch atomic-orbital values `χ_μk(r)` at every grid point,
/// in CSR form over the **touched** AOs (those with nonzero `χ`) — the expensive part of
/// [`collocate_k`](LatticeCollocator::collocate_k) / [`integrate_k`](LatticeCollocator::integrate_k)
/// (the `exp` over hundreds of placed Gaussian images per point). Built **once per geometry**
/// ([`build_chi_cache`](LatticeCollocator::build_chi_cache)) and reused across all SCF
/// iterations, where only the cheap density/potential contraction varies; the cached
/// `collocate_k`/`integrate_k` skip the `χ` evaluation entirely.
///
/// The touched-AO set at a grid point is k-independent (it is the set of AOs any lattice image
/// reaches there), so one CSR row per point carries `nk` complex `χ` values per touched AO.
/// **Memory** ≈ `npts · ⟨touched⟩ · nk · 16` bytes — large at high cutoff and many k-points,
/// so this is best for Γ / small k-meshes.
pub struct ChiCache {
    nk: usize,
    nao: usize,
    npts: usize,
    /// CSR row offsets into [`aos`](Self::aos) (length `npts + 1`).
    offsets: Vec<usize>,
    /// Touched AO indices, concatenated per grid point.
    aos: Vec<u32>,
    /// `χ` values: for touched entry `t`, `chi[t·nk + k]` is `χ` at k-point `k`.
    chi: Vec<Complex64>,
    /// BZ weights (aligned with the k-points the cache was built for).
    weights: Vec<f64>,
}

impl ChiCache {
    /// The k-point count the cache was built for.
    #[must_use]
    pub fn nk(&self) -> usize {
        self.nk
    }

    /// The total number of cached `(point, touched-AO)` entries — `chi` holds `nk` complex
    /// values per entry. Useful for a memory estimate.
    #[must_use]
    pub fn n_entries(&self) -> usize {
        self.aos.len()
    }
}

/// Precomputed image-collocation state for one basis on one cell: the per-shell
/// evaluators, the lattice images each shell reaches into the home cell, and the set of
/// difference triples `S_a − S_b` the matrix blocks span. Built once per geometry and
/// reused across SCF iterations.
pub struct LatticeCollocator {
    nao: usize,
    shells: Vec<ShellEval>,
    placed: Vec<PlacedImage>,
    /// Distinct reaching-image triples `{S}`; each is one fixed bucket slot
    /// (`PlacedImage::bucket` indexes here).
    unique: Vec<[i32; 3]>,
    /// Difference triples `{S_a − S_b}` over the reaching images — the keys collocation
    /// looks up in `P^S` and the keys integration fills in `W^S`.
    triples: Vec<[i32; 3]>,
}

impl LatticeCollocator {
    /// Build the collocator for `basis` on the cell of `grid`.
    #[must_use]
    pub fn new(basis: &Basis, grid: &RealSpaceGrid) -> Self {
        let shells = build_shell_evals(basis);

        let mut unique: Vec<[i32; 3]> = Vec::new();
        let mut seen: HashMap<[i32; 3], usize> = HashMap::new();
        let placed = place_images_into(&shells, grid, &mut unique, &mut seen);

        // Difference set {S_a − S_b}: the triples that collocation queries in P^S and
        // integration fills in W^S (symmetric under negation).
        let mut diff_seen: HashMap<[i32; 3], ()> = HashMap::new();
        let mut triples = Vec::new();
        for a in &unique {
            for b in &unique {
                let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
                if diff_seen.insert(d, ()).is_none() {
                    triples.push(d);
                }
            }
        }

        Self {
            nao: basis.nao(),
            shells,
            placed,
            unique,
            triples,
        }
    }

    /// The lattice triples `{S}` the matrix blocks span — the caller builds `P^S` over
    /// exactly these for [`collocate`](Self::collocate), and [`integrate`](Self::integrate)
    /// returns `W^S` over them.
    #[must_use]
    pub fn image_triples(&self) -> &[[i32; 3]] {
        &self.triples
    }

    /// Evaluate every reaching shell image at grid point `r` into the fixed per-triple
    /// `buckets` (one slot per [`unique`](Self::unique) triple, reused across points).
    /// `touched` is filled with the slot indices that received any value; callers iterate
    /// it. Returns nothing; the touched slots' vecs hold `(ao, value)` pairs.
    fn active_buckets(
        &self,
        r: [f64; 3],
        scratch: &mut Vec<f64>,
        buckets: &mut [Vec<(usize, f64)>],
        touched: &mut Vec<usize>,
    ) {
        for &t in touched.iter() {
            buckets[t].clear();
        }
        touched.clear();
        for pl in &self.placed {
            let slot = pl.bucket;
            let was_empty = buckets[slot].is_empty();
            self.shells[pl.shell].eval_at(r, pl.center, scratch, &mut |ao, v| {
                buckets[slot].push((ao, v));
            });
            if was_empty && !buckets[slot].is_empty() {
                touched.push(slot);
            }
        }
    }

    /// Collocate the image-resolved density matrix `dm` onto the grid:
    /// `n(r_g) = Σ_{a,b} P^{S_a−S_b}_{μ_a ν_b} φ_a φ_b`. Returns `n(r)` row-major over the
    /// grid.
    ///
    /// `dm` must carry every triple in [`image_triples`](Self::image_triples) (missing
    /// ones contribute zero).
    #[must_use]
    pub fn collocate(&self, grid: &RealSpaceGrid, dm: &ImageBlocks) -> Vec<f64> {
        use rayon::prelude::*;
        assert_eq!(dm.nao(), self.nao, "density matrix dimension mismatch");
        let nao = self.nao;
        let [_, n1, n2] = grid.n();
        let slab = n1 * n2;
        // Flat index from a difference triple S_a − S_b to dm's block (shared, read-only).
        let index = FlatTripleIndex::new(dm.triples());
        let mut n_r = vec![0.0; grid.n_points()];
        // Parallel over slabs `i` (the grid points are independent); per-thread buffers.
        n_r.par_chunks_mut(slab).enumerate().for_each(|(i, out)| {
            let mut scratch = Vec::with_capacity(16);
            let mut buckets: Vec<Vec<(usize, f64)>> = vec![Vec::new(); self.unique.len()];
            let mut touched: Vec<usize> = Vec::new();
            for j in 0..n1 {
                for k in 0..n2 {
                    let r = grid.point([i, j, k]);
                    self.active_buckets(r, &mut scratch, &mut buckets, &mut touched);
                    if touched.is_empty() {
                        continue;
                    }
                    let mut nn = 0.0;
                    for &ta in touched.iter() {
                        let sa = self.unique[ta];
                        for &tb in touched.iter() {
                            let sb = self.unique[tb];
                            let s = [sa[0] - sb[0], sa[1] - sb[1], sa[2] - sb[2]];
                            let Some(bi) = index.get(s) else { continue };
                            let block = dm.block(bi);
                            for &(ao_a, va) in &buckets[ta] {
                                let row = &block[ao_a * nao..ao_a * nao + nao];
                                for &(ao_b, vb) in &buckets[tb] {
                                    nn += row[ao_b] * va * vb;
                                }
                            }
                        }
                    }
                    out[j * n2 + k] = nn;
                }
            }
        });
        n_r
    }

    /// Integrate a real grid potential `v` against image-resolved Gaussian products:
    /// `W^S_{μν} = ⟨φ_μ(home) | v | φ_ν(·−S)⟩ = Σ_g dV·v(r_g)·φ_μ(r_g)·φ_ν(r_g−S)`,
    /// returned over [`image_triples`](Self::image_triples). The caller forms the Bloch
    /// local matrix `V_loc(k) = Σ_S e^{ik·S} W^S`.
    ///
    /// # Panics
    /// Panics if `v.len()` differs from the grid point count.
    #[must_use]
    pub fn integrate(&self, grid: &RealSpaceGrid, v: &[f64]) -> ImageBlocks {
        assert_eq!(
            v.len(),
            grid.n_points(),
            "potential length must equal grid points"
        );
        use rayon::prelude::*;
        let nao = self.nao;
        let dv = grid.dv();
        let ntri = self.triples.len();
        let nn = nao * nao;
        let index = FlatTripleIndex::new(&self.triples);
        let [n0, n1, n2] = grid.n();
        let slab = n1 * n2;
        let zero = || vec![0.0; ntri * nn]; // flat: [triple][μ][ν]
                                            // Parallel over slabs with per-thread accumulators, summed at the end.
        let flat = (0..n0)
            .into_par_iter()
            .fold(zero, |mut acc, i| {
                let mut scratch = Vec::with_capacity(16);
                let mut buckets: Vec<Vec<(usize, f64)>> = vec![Vec::new(); self.unique.len()];
                let mut touched: Vec<usize> = Vec::new();
                for j in 0..n1 {
                    for k in 0..n2 {
                        let vg = v[i * slab + j * n2 + k];
                        if vg == 0.0 {
                            continue;
                        }
                        let r = grid.point([i, j, k]);
                        self.active_buckets(r, &mut scratch, &mut buckets, &mut touched);
                        if touched.is_empty() {
                            continue;
                        }
                        let w = dv * vg;
                        for &ta in touched.iter() {
                            let sa = self.unique[ta];
                            for &tb in touched.iter() {
                                let sb = self.unique[tb];
                                // W^{S_b − S_a}: bra image S_a (home), ket image S_b.
                                let s = [sb[0] - sa[0], sb[1] - sa[1], sb[2] - sa[2]];
                                let Some(bi) = index.get(s) else { continue };
                                let block = &mut acc[bi * nn..(bi + 1) * nn];
                                for &(ao_a, va) in &buckets[ta] {
                                    let wva = w * va;
                                    let row = &mut block[ao_a * nao..ao_a * nao + nao];
                                    for &(ao_b, vb) in &buckets[tb] {
                                        row[ao_b] += wva * vb;
                                    }
                                }
                            }
                        }
                    }
                }
                acc
            })
            .reduce(zero, |mut a, b| {
                for (x, y) in a.iter_mut().zip(&b) {
                    *x += y;
                }
                a
            });
        // Repack the flat accumulator into per-triple blocks.
        let mut out = ImageBlocks::zeros(nao, &self.triples);
        for bi in 0..ntri {
            out.blocks[bi].copy_from_slice(&flat[bi * nn..(bi + 1) * nn]);
        }
        out
    }

    /// Precompute the Bloch phases `e^{ik·S}` for the `collocate_k`/`integrate_k` fast path.
    #[must_use]
    pub fn bloch_phases(&self, k_fracs: &[[f64; 3]], weights: &[f64]) -> BlochPhases {
        BlochPhases::from_unique(&self.unique, k_fracs, weights)
    }

    /// Evaluate the Bloch atomic orbitals `χ_μk(r) = Σ_S e^{ik·S} φ_μ(r − τ_μ − S)` at grid
    /// point `r` for every AO `μ` and k-point, accumulating into `chi` (`[k*nao + μ]`) and
    /// recording the **touched** AO indices (those with a nonzero `χ`) in `touched`. `chi`
    /// entries for the previously-touched AOs and `seen` must be zero/false on entry; the
    /// caller clears exactly the touched entries afterwards (incremental zeroing — the bulk
    /// of `chi` stays zero, so neither the full `chi` reset nor the dense `nao²` contraction
    /// is paid). `seen` is a per-thread `nao`-length scratch bitmap.
    fn point_chi(
        &self,
        r: [f64; 3],
        phases: &BlochPhases,
        scratch: &mut Vec<f64>,
        chi: &mut [Complex64],
        seen: &mut [bool],
        touched: &mut Vec<usize>,
    ) {
        let nao = self.nao;
        let nk = phases.nk;
        let phase = &phases.phase;
        let n_unique = phases.n_unique;
        for pl in &self.placed {
            let slot = pl.bucket;
            self.shells[pl.shell].eval_at(r, pl.center, scratch, |ao, v| {
                if !seen[ao] {
                    seen[ao] = true;
                    touched.push(ao);
                }
                for k in 0..nk {
                    chi[k * nao + ao] += phase[k * n_unique + slot] * v;
                }
            });
        }
    }

    /// Collocate the per-k density matrices `p_k` (each row-major `nao²` complex, Hermitian)
    /// onto the grid: `n(r) = Σ_k w_k Σ_{μν} P^k_{μν} χ_μk(r) χ_νk(r)*`. The Bloch-AO (`χ`)
    /// form folds the lattice image sum per AO (linear), then contracts only over the AOs
    /// **touched** at each point (the few with nonzero `χ`), far cheaper than the image-pair
    /// fold in a dense cell. Returns `n(r)` row-major.
    #[must_use]
    pub fn collocate_k(
        &self,
        grid: &RealSpaceGrid,
        p_k: &[Vec<Complex64>],
        phases: &BlochPhases,
    ) -> Vec<f64> {
        use rayon::prelude::*;
        let nao = self.nao;
        let nk = phases.nk;
        assert_eq!(p_k.len(), nk, "p_k count must match k-points");
        let [_, n1, n2] = grid.n();
        let slab = n1 * n2;
        let mut n_r = vec![0.0; grid.n_points()];
        n_r.par_chunks_mut(slab).enumerate().for_each(|(i, out)| {
            let mut scratch = Vec::with_capacity(16);
            let mut chi = vec![Complex64::new(0.0, 0.0); nk * nao];
            let mut seen = vec![false; nao];
            let mut touched: Vec<usize> = Vec::new();
            for j in 0..n1 {
                for k in 0..n2 {
                    touched.clear();
                    let r = grid.point([i, j, k]);
                    self.point_chi(r, phases, &mut scratch, &mut chi, &mut seen, &mut touched);
                    let mut nn = 0.0;
                    for kk in 0..nk {
                        let xk = &chi[kk * nao..(kk + 1) * nao];
                        let pk = &p_k[kk];
                        // Sparse contraction over the touched (nonzero-χ) AOs only.
                        let mut acc = Complex64::new(0.0, 0.0);
                        for &mu in touched.iter() {
                            let prow = &pk[mu * nao..mu * nao + nao];
                            let mut row = Complex64::new(0.0, 0.0);
                            for &nu in touched.iter() {
                                row += prow[nu] * xk[nu].conj();
                            }
                            acc += xk[mu] * row;
                        }
                        nn += phases.weights[kk] * acc.re;
                    }
                    out[j * n2 + k] = nn;
                    // Incremental zeroing of only the touched entries.
                    for &ao in touched.iter() {
                        seen[ao] = false;
                        for kk in 0..nk {
                            chi[kk * nao + ao] = Complex64::new(0.0, 0.0);
                        }
                    }
                }
            }
        });
        n_r
    }

    /// Like [`point_chi`](Self::point_chi) but also accumulates the **spatial gradient** of
    /// each Bloch AO, `χ'_μk(r) = Σ_S e^{ik·S} ∇_r φ_μ(r − τ_μ − S)`, into `chi_g`
    /// (`[(k·nao + μ)·3 + axis]`). The center derivative is `∂χ_μk/∂R_μ = −χ'_μk`. `chi`,
    /// `chi_g`, and `seen` for the previously-touched AOs must be zero/false on entry; the
    /// caller clears exactly the touched entries afterwards.
    #[allow(clippy::too_many_arguments)]
    fn point_chi_grad(
        &self,
        r: [f64; 3],
        phases: &BlochPhases,
        scratch: &mut Vec<f64>,
        chi: &mut [Complex64],
        chi_g: &mut [Complex64],
        seen: &mut [bool],
        touched: &mut Vec<usize>,
    ) {
        let nao = self.nao;
        let nk = phases.nk;
        let phase = &phases.phase;
        let n_unique = phases.n_unique;
        for pl in &self.placed {
            let slot = pl.bucket;
            self.shells[pl.shell].emit_grad(r, pl.center, scratch, |ao, v, g| {
                if !seen[ao] {
                    seen[ao] = true;
                    touched.push(ao);
                }
                for k in 0..nk {
                    let ph = phase[k * n_unique + slot];
                    chi[k * nao + ao] += ph * v;
                    let base = (k * nao + ao) * 3;
                    chi_g[base] += ph * g[0];
                    chi_g[base + 1] += ph * g[1];
                    chi_g[base + 2] += ph * g[2];
                }
            });
        }
    }

    /// The **collocation Pulay** force — the basis-derivative part of the local grid energy.
    /// With the local potential `v` and the collocated density `n(r) = Σ_k w_k Σ_{μν} P^k_μν
    /// χ_μk χ_νk*`, the physical term is `F^coll_I = −∫ v(r) ∂n(r)/∂R_I dr`, and since only the
    /// AOs on atom `I` move (`∂χ_μk/∂R_I = −χ'_μk`, the Bloch sum of the spatial AO gradient),
    ///
    /// ```text
    ///   F^coll_I[a] = 2·dV·Σ_g v(r_g) Σ_k w_k Re[ Σ_{μ∈I} χ'^{(a)}_μk(r_g) · D^k_μ(r_g) ] ,
    ///   D^k_μ(r) = Σ_ν P^k_μν χ_νk(r)* .
    /// ```
    ///
    /// **This returns the physical force `F^coll = −∫ v ∂n/∂R`** (pass `v = V_loc_grid`
    /// directly). `p_k` are the per-k density matrices (row-major `nao²` complex Hermitian),
    /// `ao_atom[μ]` the atom index of AO `μ`, `natom` the atom count. Atom order is the
    /// caller's `ao_atom` labelling.
    ///
    /// # Panics
    /// Panics if `v.len()` differs from the grid point count, `p_k.len()` differs from the
    /// k-count, or `ao_atom.len() != nao`.
    #[must_use]
    pub fn collocation_pulay_forces(
        &self,
        grid: &RealSpaceGrid,
        p_k: &[Vec<Complex64>],
        v: &[f64],
        phases: &BlochPhases,
        ao_atom: &[usize],
        natom: usize,
    ) -> Vec<[f64; 3]> {
        use rayon::prelude::*;
        assert_eq!(
            v.len(),
            grid.n_points(),
            "potential length must equal grid points"
        );
        let nao = self.nao;
        let nk = phases.nk;
        assert_eq!(p_k.len(), nk, "p_k count must match k-points");
        assert_eq!(ao_atom.len(), nao, "ao_atom must label every AO");
        let dv = grid.dv();
        let [n0, n1, n2] = grid.n();
        let slab = n1 * n2;
        let zero = || vec![[0.0_f64; 3]; natom];
        let acc = (0..n0)
            .into_par_iter()
            .fold(zero, |mut force, i| {
                let mut scratch = Vec::with_capacity(16);
                let mut chi = vec![Complex64::new(0.0, 0.0); nk * nao];
                let mut chi_g = vec![Complex64::new(0.0, 0.0); nk * nao * 3];
                let mut seen = vec![false; nao];
                let mut touched: Vec<usize> = Vec::new();
                let mut dvec = vec![Complex64::new(0.0, 0.0); nao]; // D^k_μ scratch
                for j in 0..n1 {
                    for k in 0..n2 {
                        let vg = v[i * slab + j * n2 + k];
                        if vg == 0.0 {
                            continue;
                        }
                        touched.clear();
                        let r = grid.point([i, j, k]);
                        self.point_chi_grad(
                            r,
                            phases,
                            &mut scratch,
                            &mut chi,
                            &mut chi_g,
                            &mut seen,
                            &mut touched,
                        );
                        let wv = 2.0 * dv * vg;
                        for (kk, &wk) in phases.weights.iter().enumerate() {
                            let xk = &chi[kk * nao..(kk + 1) * nao];
                            let pk = &p_k[kk];
                            // D^k_μ = Σ_ν P^k_μν χ_νk*, over touched ν only.
                            for &mu in touched.iter() {
                                let prow = &pk[mu * nao..mu * nao + nao];
                                let mut d = Complex64::new(0.0, 0.0);
                                for &nu in touched.iter() {
                                    d += prow[nu] * xk[nu].conj();
                                }
                                dvec[mu] = d;
                            }
                            let wkv = wk * wv;
                            for &mu in touched.iter() {
                                let atom = ao_atom[mu];
                                let base = (kk * nao + mu) * 3;
                                let d = dvec[mu];
                                force[atom][0] += wkv * (chi_g[base] * d).re;
                                force[atom][1] += wkv * (chi_g[base + 1] * d).re;
                                force[atom][2] += wkv * (chi_g[base + 2] * d).re;
                            }
                        }
                        // Incremental zeroing of only the touched entries.
                        for &ao in touched.iter() {
                            seen[ao] = false;
                            for kk in 0..nk {
                                chi[kk * nao + ao] = Complex64::new(0.0, 0.0);
                                let base = (kk * nao + ao) * 3;
                                chi_g[base] = Complex64::new(0.0, 0.0);
                                chi_g[base + 1] = Complex64::new(0.0, 0.0);
                                chi_g[base + 2] = Complex64::new(0.0, 0.0);
                            }
                        }
                    }
                }
                force
            })
            .reduce(zero, |mut a, b| {
                for (x, y) in a.iter_mut().zip(&b) {
                    for ax in 0..3 {
                        x[ax] += y[ax];
                    }
                }
                a
            });
        acc
    }

    /// Like [`point_chi_grad`](Self::point_chi_grad) but accumulates the AO **gradient ⊗
    /// displacement** `χ_disp_μk[α][β] = Σ_S e^{ik·S} (∇φ_μ)_α(r−τ_μ−S) · (r−τ_μ−S)_β` into
    /// `chi_disp` (`[(k·nao + μ)·9 + (3α + β)]`) — the strain derivative `∂χ_μk/∂ε_αβ =
    /// χ_disp_μk[α][β]` (under a strain the AO–grid displacement scales). `chi` carries the
    /// values for the `D^k_μ` contraction. `chi`, `chi_disp`, `seen` for previously-touched
    /// AOs must be zero/false on entry; the caller clears exactly the touched entries.
    #[allow(clippy::too_many_arguments)]
    fn point_chi_grad_disp(
        &self,
        r: [f64; 3],
        phases: &BlochPhases,
        scratch: &mut Vec<f64>,
        chi: &mut [Complex64],
        chi_disp: &mut [Complex64],
        seen: &mut [bool],
        touched: &mut Vec<usize>,
    ) {
        let nao = self.nao;
        let nk = phases.nk;
        let phase = &phases.phase;
        let n_unique = phases.n_unique;
        for pl in &self.placed {
            let slot = pl.bucket;
            let disp = [
                r[0] - pl.center[0],
                r[1] - pl.center[1],
                r[2] - pl.center[2],
            ];
            self.shells[pl.shell].emit_grad(r, pl.center, scratch, |ao, v, g| {
                if !seen[ao] {
                    seen[ao] = true;
                    touched.push(ao);
                }
                for k in 0..nk {
                    let ph = phase[k * n_unique + slot];
                    chi[k * nao + ao] += ph * v;
                    let base = (k * nao + ao) * 9;
                    for (alpha, &ga) in g.iter().enumerate() {
                        let pg = ph * ga;
                        for (beta, &db) in disp.iter().enumerate() {
                            chi_disp[base + alpha * 3 + beta] += pg * db;
                        }
                    }
                }
            });
        }
    }

    /// The **collocation Pulay stress** — the basis-derivative (strain) part of the local grid
    /// energy, `τ_αβ = ∫ v(r) ∂n(r)/∂ε_αβ dr`. Under a strain the AO–grid displacement scales
    /// (`∂χ_μk/∂ε_αβ = χ_disp_μk[α][β]`), so
    ///
    /// ```text
    ///   τ_αβ = 2·dV·Σ_g v(r_g) Σ_k w_k Re[ Σ_μ χ_disp_μk[α][β](r_g) · D^k_μ(r_g) ] ,
    ///   D^k_μ(r) = Σ_ν P^k_μν χ_νk(r)* .
    /// ```
    ///
    /// This is the contribution to `∂E/∂ε` (not negated — stress is `∂E/∂ε`, unlike the force
    /// `−∂E/∂R`); pass `v = V_loc_grid`. Generally non-symmetric in `αβ` (the caller
    /// symmetrizes for the physical stress).
    ///
    /// # Panics
    /// Panics if `v.len()` differs from the grid point count or `p_k.len()` ≠ the k-count.
    #[must_use]
    pub fn collocation_pulay_stress(
        &self,
        grid: &RealSpaceGrid,
        p_k: &[Vec<Complex64>],
        v: &[f64],
        phases: &BlochPhases,
    ) -> [[f64; 3]; 3] {
        use rayon::prelude::*;
        assert_eq!(
            v.len(),
            grid.n_points(),
            "potential length must equal grid points"
        );
        let nao = self.nao;
        let nk = phases.nk;
        assert_eq!(p_k.len(), nk, "p_k count must match k-points");
        let dv = grid.dv();
        let [n0, n1, n2] = grid.n();
        let slab = n1 * n2;
        let zero = || [[0.0_f64; 3]; 3];
        let acc = (0..n0)
            .into_par_iter()
            .fold(zero, |mut tau, i| {
                let mut scratch = Vec::with_capacity(16);
                let mut chi = vec![Complex64::new(0.0, 0.0); nk * nao];
                let mut chi_disp = vec![Complex64::new(0.0, 0.0); nk * nao * 9];
                let mut seen = vec![false; nao];
                let mut touched: Vec<usize> = Vec::new();
                let mut dvec = vec![Complex64::new(0.0, 0.0); nao];
                for j in 0..n1 {
                    for k in 0..n2 {
                        let vg = v[i * slab + j * n2 + k];
                        if vg == 0.0 {
                            continue;
                        }
                        touched.clear();
                        let r = grid.point([i, j, k]);
                        self.point_chi_grad_disp(
                            r,
                            phases,
                            &mut scratch,
                            &mut chi,
                            &mut chi_disp,
                            &mut seen,
                            &mut touched,
                        );
                        let wv = 2.0 * dv * vg;
                        for (kk, &wk) in phases.weights.iter().enumerate() {
                            let xk = &chi[kk * nao..(kk + 1) * nao];
                            let pk = &p_k[kk];
                            for &mu in touched.iter() {
                                let prow = &pk[mu * nao..mu * nao + nao];
                                let mut d = Complex64::new(0.0, 0.0);
                                for &nu in touched.iter() {
                                    d += prow[nu] * xk[nu].conj();
                                }
                                dvec[mu] = d;
                            }
                            let wkv = wk * wv;
                            for &mu in touched.iter() {
                                let base = (kk * nao + mu) * 9;
                                let d = dvec[mu];
                                for (alpha, ta) in tau.iter_mut().enumerate() {
                                    for (beta, tab) in ta.iter_mut().enumerate() {
                                        *tab += wkv * (chi_disp[base + alpha * 3 + beta] * d).re;
                                    }
                                }
                            }
                        }
                        for &ao in touched.iter() {
                            seen[ao] = false;
                            for kk in 0..nk {
                                chi[kk * nao + ao] = Complex64::new(0.0, 0.0);
                                let base = (kk * nao + ao) * 9;
                                for s in &mut chi_disp[base..base + 9] {
                                    *s = Complex64::new(0.0, 0.0);
                                }
                            }
                        }
                    }
                }
                tau
            })
            .reduce(zero, |mut a, b| {
                for (ra, rb) in a.iter_mut().zip(&b) {
                    for (x, y) in ra.iter_mut().zip(rb) {
                        *x += y;
                    }
                }
                a
            });
        acc
    }

    /// Integrate a real grid potential `v` to the per-k local Kohn–Sham matrices
    /// `V_loc(k)_{μν} = Σ_g dV·v(r_g)·χ_μk(r_g)* χ_νk(r_g)` (each row-major `nao²` complex
    /// Hermitian). Returns one matrix per k-point. The Bloch-AO form builds `V_loc(k)`
    /// directly — no `W^S` intermediate.
    ///
    /// # Panics
    /// Panics if `v.len()` differs from the grid point count.
    #[must_use]
    pub fn integrate_k(
        &self,
        grid: &RealSpaceGrid,
        v: &[f64],
        phases: &BlochPhases,
    ) -> Vec<Vec<Complex64>> {
        use rayon::prelude::*;
        assert_eq!(
            v.len(),
            grid.n_points(),
            "potential length must equal grid points"
        );
        let nao = self.nao;
        let nk = phases.nk;
        let nn = nao * nao;
        let dv = grid.dv();
        let [n0, n1, n2] = grid.n();
        let slab = n1 * n2;
        let zero = || vec![Complex64::new(0.0, 0.0); nk * nn];
        let flat = (0..n0)
            .into_par_iter()
            .fold(zero, |mut acc, i| {
                let mut scratch = Vec::with_capacity(16);
                let mut chi = vec![Complex64::new(0.0, 0.0); nk * nao];
                let mut seen = vec![false; nao];
                let mut touched: Vec<usize> = Vec::new();
                for j in 0..n1 {
                    for k in 0..n2 {
                        let vg = v[i * slab + j * n2 + k];
                        if vg == 0.0 {
                            continue;
                        }
                        touched.clear();
                        let r = grid.point([i, j, k]);
                        self.point_chi(r, phases, &mut scratch, &mut chi, &mut seen, &mut touched);
                        let w = dv * vg;
                        for kk in 0..nk {
                            let xk = &chi[kk * nao..(kk + 1) * nao];
                            let block = &mut acc[kk * nn..(kk + 1) * nn];
                            // Sparse rank-1 update over the touched AOs only.
                            for &mu in touched.iter() {
                                let cxmu = Complex64::new(w, 0.0) * xk[mu].conj();
                                let row = &mut block[mu * nao..mu * nao + nao];
                                for &nu in touched.iter() {
                                    row[nu] += cxmu * xk[nu];
                                }
                            }
                        }
                        // Incremental zeroing of only the touched entries.
                        for &ao in touched.iter() {
                            seen[ao] = false;
                            for kk in 0..nk {
                                chi[kk * nao + ao] = Complex64::new(0.0, 0.0);
                            }
                        }
                    }
                }
                acc
            })
            .reduce(zero, |mut a, b| {
                for (x, y) in a.iter_mut().zip(&b) {
                    *x += y;
                }
                a
            });
        (0..nk)
            .map(|kk| flat[kk * nn..(kk + 1) * nn].to_vec())
            .collect()
    }

    /// Build the geometry-fixed [`ChiCache`]: evaluate every Bloch AO `χ_μk(r)` at every grid
    /// point once (the `exp`-over-images bottleneck) and store the touched-AO values in CSR
    /// form, to be reused across SCF iterations by [`collocate_k_cached`](Self::collocate_k_cached)
    /// / [`integrate_k_cached`](Self::integrate_k_cached). Parallel over grid slabs.
    #[must_use]
    pub fn build_chi_cache(&self, grid: &RealSpaceGrid, phases: &BlochPhases) -> ChiCache {
        use rayon::prelude::*;
        let nao = self.nao;
        let nk = phases.nk;
        let [n0, n1, n2] = grid.n();
        let slab = n1 * n2;
        let npts = grid.n_points();
        // Per-slab CSR fragments: (aos, chi, per-point counts), built independently.
        let per_slab: Vec<(Vec<u32>, Vec<Complex64>, Vec<usize>)> = (0..n0)
            .into_par_iter()
            .map(|i| {
                let mut scratch = Vec::with_capacity(16);
                let mut chi = vec![Complex64::new(0.0, 0.0); nk * nao];
                let mut seen = vec![false; nao];
                let mut touched: Vec<usize> = Vec::new();
                let mut aos: Vec<u32> = Vec::new();
                let mut vals: Vec<Complex64> = Vec::new();
                let mut counts = vec![0usize; slab];
                for j in 0..n1 {
                    for k in 0..n2 {
                        touched.clear();
                        let r = grid.point([i, j, k]);
                        self.point_chi(r, phases, &mut scratch, &mut chi, &mut seen, &mut touched);
                        counts[j * n2 + k] = touched.len();
                        for &ao in &touched {
                            aos.push(ao as u32);
                            for kk in 0..nk {
                                vals.push(chi[kk * nao + ao]);
                            }
                        }
                        // Incremental zeroing of only the touched entries.
                        for &ao in &touched {
                            seen[ao] = false;
                            for kk in 0..nk {
                                chi[kk * nao + ao] = Complex64::new(0.0, 0.0);
                            }
                        }
                    }
                }
                (aos, vals, counts)
            })
            .collect();

        let mut offsets = vec![0usize; npts + 1];
        let total: usize = per_slab.iter().map(|(a, _, _)| a.len()).sum();
        let mut aos = Vec::with_capacity(total);
        let mut chi = Vec::with_capacity(total * nk);
        let mut g = 0usize;
        for (a, v, counts) in per_slab {
            for c in counts {
                offsets[g + 1] = offsets[g] + c;
                g += 1;
            }
            aos.extend(a);
            chi.extend(v);
        }
        ChiCache {
            nk,
            nao,
            npts,
            offsets,
            aos,
            chi,
            weights: phases.weights.clone(),
        }
    }

    /// Collocate the per-k density matrices to `n(r)` using a prebuilt [`ChiCache`] — identical
    /// result to [`collocate_k`](Self::collocate_k) but with the `χ` evaluation skipped (only
    /// the `nao²`-style contraction over touched AOs is paid).
    ///
    /// # Panics
    /// Panics if `p_k.len()` differs from the cache's k-count.
    #[must_use]
    pub fn collocate_k_cached(&self, cache: &ChiCache, p_k: &[Vec<Complex64>]) -> Vec<f64> {
        use rayon::prelude::*;
        let nao = self.nao;
        let nk = cache.nk;
        assert_eq!(p_k.len(), nk, "p_k count must match the cache k-points");
        assert_eq!(cache.nao, nao, "cache built for a different basis");
        let mut n_r = vec![0.0; cache.npts];
        n_r.par_iter_mut().enumerate().for_each(|(g, out)| {
            let s = cache.offsets[g];
            let e = cache.offsets[g + 1];
            if s == e {
                return;
            }
            let aos = &cache.aos[s..e];
            let mut nn = 0.0;
            for (kk, pk) in p_k.iter().enumerate() {
                let mut acc = Complex64::new(0.0, 0.0);
                for (ti, &mu_) in aos.iter().enumerate() {
                    let mu = mu_ as usize;
                    let xmu = cache.chi[(s + ti) * nk + kk];
                    let prow = &pk[mu * nao..mu * nao + nao];
                    let mut row = Complex64::new(0.0, 0.0);
                    for (tj, &nu_) in aos.iter().enumerate() {
                        row += prow[nu_ as usize] * cache.chi[(s + tj) * nk + kk].conj();
                    }
                    acc += xmu * row;
                }
                nn += cache.weights[kk] * acc.re;
            }
            *out = nn;
        });
        n_r
    }

    /// Integrate a real grid potential `v` to the per-k local matrices `V_loc(k)` using a
    /// prebuilt [`ChiCache`] — identical result to [`integrate_k`](Self::integrate_k) but with
    /// the `χ` evaluation skipped.
    ///
    /// # Panics
    /// Panics if `v.len()` differs from the cached grid point count.
    #[must_use]
    pub fn integrate_k_cached(
        &self,
        cache: &ChiCache,
        grid: &RealSpaceGrid,
        v: &[f64],
    ) -> Vec<Vec<Complex64>> {
        use rayon::prelude::*;
        assert_eq!(
            v.len(),
            cache.npts,
            "potential length must equal grid points"
        );
        let nao = self.nao;
        let nk = cache.nk;
        let nn = nao * nao;
        let dv = grid.dv();
        let [n0, n1, n2] = grid.n();
        let slab = n1 * n2;
        let zero = || vec![Complex64::new(0.0, 0.0); nk * nn];
        let flat = (0..n0)
            .into_par_iter()
            .fold(zero, |mut acc, i| {
                for j in 0..n1 {
                    for k in 0..n2 {
                        let g = i * slab + j * n2 + k;
                        let vg = v[g];
                        if vg == 0.0 {
                            continue;
                        }
                        let s = cache.offsets[g];
                        let e = cache.offsets[g + 1];
                        if s == e {
                            continue;
                        }
                        let aos = &cache.aos[s..e];
                        let w = dv * vg;
                        for kk in 0..nk {
                            let block = &mut acc[kk * nn..(kk + 1) * nn];
                            for (ti, &mu_) in aos.iter().enumerate() {
                                let cxmu =
                                    Complex64::new(w, 0.0) * cache.chi[(s + ti) * nk + kk].conj();
                                let row = &mut block[mu_ as usize * nao..mu_ as usize * nao + nao];
                                for (tj, &nu_) in aos.iter().enumerate() {
                                    row[nu_ as usize] += cxmu * cache.chi[(s + tj) * nk + kk];
                                }
                            }
                        }
                    }
                }
                acc
            })
            .reduce(zero, |mut a, b| {
                for (x, y) in a.iter_mut().zip(&b) {
                    *x += y;
                }
                a
            });
        (0..nk)
            .map(|kk| flat[kk * nn..(kk + 1) * nn].to_vec())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::periodic::{bloch_overlap, collocate_density, integrate_potential, RealSpaceGrid};
    use crate::{Basis, Shell};
    use latx::Cell;
    use rustfft::num_complex::Complex64;
    use std::f64::consts::PI;

    fn identity(nao: usize) -> Vec<f64> {
        let mut p = vec![0.0; nao * nao];
        for i in 0..nao {
            p[i * nao + i] = 1.0;
        }
        p
    }

    /// In a large box (no image but the home cell reaches), the image-resolved
    /// collocation reduces to the M2 minimum-image `collocate_density`.
    #[test]
    fn large_box_collocation_matches_minimum_image() {
        let basis = Basis::new(vec![
            Shell::new(0, [7.0, 8.0, 8.0], vec![0.7], vec![1.0]).unwrap(),
            Shell::new_spherical(1, [9.0, 8.0, 8.0], vec![0.9], vec![1.0]).unwrap(),
        ]);
        let grid = RealSpaceGrid::new(Cell::cubic(16.0).unwrap(), [48, 48, 48]);
        let nao = basis.nao();
        let p = identity(nao);

        let coll = LatticeCollocator::new(&basis, &grid);
        let dm = ImageBlocks::constant(nao, coll.image_triples(), &p);
        let n_lat = coll.collocate(&grid, &dm);
        let n_m2 = collocate_density(&basis, &p, &grid);

        let max = n_lat
            .iter()
            .zip(&n_m2)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(max < 1e-12, "lattice vs min-image collocation: {max}");
    }

    /// Same large-box reduction for integration: `Σ_S W^S` equals the M2 minimum-image
    /// `integrate_potential`.
    #[test]
    fn large_box_integration_matches_minimum_image() {
        let basis = Basis::new(vec![
            Shell::new(0, [8.0, 8.0, 8.0], vec![0.8], vec![1.0]).unwrap(),
            Shell::new(0, [8.0, 9.5, 8.0], vec![1.1], vec![1.0]).unwrap(),
        ]);
        let grid = RealSpaceGrid::new(Cell::cubic(16.0).unwrap(), [50, 50, 50]);
        let nao = basis.nao();
        // An arbitrary smooth periodic potential.
        let v: Vec<f64> = grid
            .points()
            .iter()
            .map(|r| (2.0 * PI * r[0] / 16.0).sin() + 0.4 * (2.0 * PI * r[1] / 16.0).cos())
            .collect();

        let coll = LatticeCollocator::new(&basis, &grid);
        let w = coll.integrate(&grid, &v);
        // Σ_S W^S (real Γ value).
        let mut sum = vec![0.0; nao * nao];
        for i in 0..w.len() {
            for (s, wij) in sum.iter_mut().zip(w.block(i)) {
                *s += wij;
            }
        }
        let m2 = integrate_potential(&basis, &v, &grid);
        let max = sum
            .iter()
            .zip(&m2)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(max < 1e-12, "Σ_S W^S vs min-image integration: {max}");
    }

    /// Collocation and integration are adjoints across the lattice sum:
    /// `Σ_S Σ_μν P_μν W^S_μν = Σ_g dV·V·n` for a constant (Γ) density matrix `P`, in a
    /// **small** cell where several images genuinely contribute.
    #[test]
    fn lattice_collocate_integrate_adjoint_small_cell() {
        let basis = Basis::new(vec![
            Shell::new(0, [0.0, 0.0, 0.0], vec![0.45], vec![1.0]).unwrap(),
            Shell::new(0, [2.0, 2.0, 2.0], vec![0.55], vec![1.0]).unwrap(),
        ]);
        let cell = Cell::cubic(4.5).unwrap();
        let grid = RealSpaceGrid::new(cell, [40, 40, 40]);
        let nao = basis.nao();
        let coll = LatticeCollocator::new(&basis, &grid);
        // The image machinery must actually engage (more than the home image).
        assert!(
            coll.image_triples().len() > 1,
            "expected multiple lattice images in a small cell, got {}",
            coll.image_triples().len()
        );

        // An arbitrary symmetric P.
        let mut p = vec![0.0; nao * nao];
        for a in 0..nao {
            for b in 0..nao {
                p[a * nao + b] = 0.3 * (a as f64 + 1.0) * (b as f64 + 1.0);
            }
        }
        let dm = ImageBlocks::constant(nao, coll.image_triples(), &p);
        let n_r = coll.collocate(&grid, &dm);
        // A smooth periodic potential.
        let v: Vec<f64> = grid
            .points()
            .iter()
            .map(|r| (2.0 * PI * r[0] / 4.5).cos() + 0.5)
            .collect();
        let w = coll.integrate(&grid, &v);

        // lhs = Σ_S Σ_μν P_μν W^S_μν.
        let mut lhs = 0.0;
        for i in 0..w.len() {
            let block = w.block(i);
            for (pij, wij) in p.iter().zip(block) {
                lhs += pij * wij;
            }
        }
        // rhs = Σ_g dV·V·n.
        let rhs: f64 = grid.dv() * v.iter().zip(&n_r).map(|(&vv, &nn)| vv * nn).sum::<f64>();
        assert!(
            (lhs - rhs).abs() < 1e-9,
            "adjoint: Σ P·W = {lhs}, Σ dV·V·n = {rhs}"
        );
    }

    /// The Bloch local matrix `V_loc(k) = Σ_S e^{ik·S} W^S` is Hermitian for any k.
    #[test]
    fn bloch_local_matrix_is_hermitian() {
        let basis = Basis::new(vec![
            Shell::new(0, [0.0, 0.0, 0.0], vec![0.5], vec![1.0]).unwrap(),
            Shell::new_spherical(1, [1.8, 1.6, 1.7], vec![0.6], vec![1.0]).unwrap(),
        ]);
        let cell = Cell::cubic(4.0).unwrap();
        let grid = RealSpaceGrid::new(cell, [36, 36, 36]);
        let nao = basis.nao();
        let coll = LatticeCollocator::new(&basis, &grid);
        let v: Vec<f64> = grid
            .points()
            .iter()
            .map(|r| (2.0 * PI * r[2] / 4.0).sin() + 0.7)
            .collect();
        let w = coll.integrate(&grid, &v);

        let kfrac = [0.3, -0.15, 0.2];
        let mut vk = vec![Complex64::new(0.0, 0.0); nao * nao];
        for i in 0..w.len() {
            let s = w.triple(i);
            let theta = 2.0
                * PI
                * (kfrac[0] * f64::from(s[0])
                    + kfrac[1] * f64::from(s[1])
                    + kfrac[2] * f64::from(s[2]));
            let phase = Complex64::new(theta.cos(), theta.sin());
            for (vij, &wij) in vk.iter_mut().zip(w.block(i)) {
                *vij += phase * wij;
            }
        }
        for a in 0..nao {
            for b in 0..nao {
                let diff = vk[a * nao + b] - vk[b * nao + a].conj();
                assert!(diff.norm() < 1e-12, "V_loc(k) not Hermitian at ({a},{b})");
            }
        }
    }

    /// The Bloch-AO (`χ`) collocation reduces to the M2 minimum-image `collocate_density`
    /// at a single Γ point in a large box (`P^Γ = identity`).
    #[test]
    fn collocate_k_gamma_matches_minimum_image() {
        let basis = Basis::new(vec![
            Shell::new(0, [7.0, 8.0, 8.0], vec![0.7], vec![1.0]).unwrap(),
            Shell::new_spherical(1, [9.0, 8.0, 8.0], vec![0.9], vec![1.0]).unwrap(),
        ]);
        let grid = RealSpaceGrid::new(Cell::cubic(16.0).unwrap(), [48, 48, 48]);
        let nao = basis.nao();
        let coll = LatticeCollocator::new(&basis, &grid);
        let phases = coll.bloch_phases(&[[0.0, 0.0, 0.0]], &[1.0]);
        // P^Γ = identity (complex).
        let mut pk = vec![Complex64::new(0.0, 0.0); nao * nao];
        for i in 0..nao {
            pk[i * nao + i] = Complex64::new(1.0, 0.0);
        }
        let n_chi = coll.collocate_k(&grid, &[pk], &phases);
        let n_m2 = collocate_density(&basis, &identity(nao), &grid);
        let max = n_chi
            .iter()
            .zip(&n_m2)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(max < 1e-12, "collocate_k(Γ) vs min-image: {max}");
    }

    /// The `χ` collocation and integration are adjoints at a single Γ point:
    /// `Re Tr(V_loc(Γ) P^Γ) = Σ_g dV·v·n` with `n = collocate_k(P)`, `V_loc(Γ) =
    /// integrate_k(v)`, in a small cell where several images contribute.
    #[test]
    fn collocate_k_integrate_k_adjoint() {
        let basis = Basis::new(vec![
            Shell::new(0, [0.0, 0.0, 0.0], vec![0.45], vec![1.0]).unwrap(),
            Shell::new_spherical(1, [2.0, 2.0, 2.0], vec![0.55], vec![1.0]).unwrap(),
        ]);
        let cell = Cell::cubic(4.5).unwrap();
        let grid = RealSpaceGrid::new(cell, [40, 40, 40]);
        let nao = basis.nao();
        let coll = LatticeCollocator::new(&basis, &grid);
        let phases = coll.bloch_phases(&[[0.0, 0.0, 0.0]], &[1.0]);
        // An arbitrary Hermitian (here real-symmetric) P^Γ.
        let mut pk = vec![Complex64::new(0.0, 0.0); nao * nao];
        for a in 0..nao {
            for b in 0..nao {
                pk[a * nao + b] = Complex64::new(0.3 * (a as f64 + 1.0) * (b as f64 + 1.0), 0.0);
            }
        }
        let n_r = coll.collocate_k(&grid, &[pk.clone()], &phases);
        let v: Vec<f64> = grid
            .points()
            .iter()
            .map(|r| (2.0 * PI * r[0] / 4.5).cos() + 0.5)
            .collect();
        let vloc = coll.integrate_k(&grid, &v, &phases);
        // lhs = Re Σ_{μν} V_loc(Γ)_{μν} P^Γ_{νμ}.
        let mut lhs = 0.0;
        for mu in 0..nao {
            for nu in 0..nao {
                lhs += (vloc[0][mu * nao + nu] * pk[nu * nao + mu]).re;
            }
        }
        let rhs: f64 = grid.dv() * v.iter().zip(&n_r).map(|(&vv, &nn)| vv * nn).sum::<f64>();
        assert!(
            (lhs - rhs).abs() < 1e-9,
            "χ adjoint: Tr(V P) = {lhs}, Σ dV v n = {rhs}"
        );
    }

    /// The [`ChiCache`] reproduces `collocate_k` / `integrate_k` exactly (same `χ` values and
    /// contraction order), at two k-points in a small cell where several images contribute.
    #[test]
    fn chi_cache_matches_uncached() {
        let basis = Basis::new(vec![
            Shell::new(0, [0.0, 0.0, 0.0], vec![0.45], vec![1.0]).unwrap(),
            Shell::new_spherical(1, [2.0, 2.0, 2.0], vec![0.55], vec![1.0]).unwrap(),
        ]);
        let cell = Cell::cubic(4.5).unwrap();
        let grid = RealSpaceGrid::new(cell, [40, 40, 40]);
        let nao = basis.nao();
        let coll = LatticeCollocator::new(&basis, &grid);
        let kfracs = [[0.0, 0.0, 0.0], [0.3, -0.1, 0.2]];
        let weights = [0.6, 0.4];
        let phases = coll.bloch_phases(&kfracs, &weights);

        // Hermitian per-k density matrices.
        let p_k: Vec<Vec<Complex64>> = (0..2)
            .map(|kk| {
                let mut p = vec![Complex64::new(0.0, 0.0); nao * nao];
                for a in 0..nao {
                    for b in 0..nao {
                        p[a * nao + b] = Complex64::new(
                            0.2 * ((a + 1) as f64) + 0.1 * (a * b) as f64 + kk as f64 * 0.05,
                            0.0,
                        );
                    }
                }
                for a in 0..nao {
                    for b in (a + 1)..nao {
                        let s = (p[a * nao + b] + p[b * nao + a]) * Complex64::new(0.5, 0.0);
                        p[a * nao + b] = s;
                        p[b * nao + a] = s;
                    }
                }
                p
            })
            .collect();
        let v: Vec<f64> = grid
            .points()
            .iter()
            .map(|r| (2.0 * PI * r[0] / 4.5).cos() + 0.4 * (2.0 * PI * r[1] / 4.5).sin() + 1.0)
            .collect();

        let cache = coll.build_chi_cache(&grid, &phases);
        let n_un = coll.collocate_k(&grid, &p_k, &phases);
        let n_ca = coll.collocate_k_cached(&cache, &p_k);
        let dn = n_un
            .iter()
            .zip(&n_ca)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(dn < 1e-12, "cached collocate Δ = {dn}");

        let v_un = coll.integrate_k(&grid, &v, &phases);
        let v_ca = coll.integrate_k_cached(&cache, &grid, &v);
        let mut dv_max = 0.0_f64;
        for (a, b) in v_un.iter().zip(&v_ca) {
            for (x, y) in a.iter().zip(b) {
                dv_max = dv_max.max((x - y).norm());
            }
        }
        assert!(dv_max < 1e-12, "cached integrate Δ = {dv_max}");
    }

    /// Build the AO→atom map for a basis: AO `μ` belongs to the atom of its shell.
    fn ao_atom_map(basis: &Basis) -> Vec<usize> {
        let satom = basis.shell_atom();
        let offs = basis.offsets();
        let mut map = vec![0usize; basis.nao()];
        for (si, sh) in basis.shells().iter().enumerate() {
            for f in 0..sh.n_func() {
                map[offs[si] + f] = satom[si];
            }
        }
        map
    }

    /// The collocation Pulay force matches the central finite difference of the local grid
    /// energy `E(R) = dV Σ_g v(r_g) n(r_g; R)` with the density matrix `P^k` and potential `v`
    /// held fixed (the basis moves with the atoms) — the per-term M4 validation of the
    /// `∂n/∂R` collocation derivative. Two k-points exercise the complex `χ` path.
    #[test]
    fn collocation_pulay_force_matches_finite_difference() {
        let cell = Cell::cubic(5.0).unwrap();
        let grid = RealSpaceGrid::new(cell, [36, 36, 36]);
        // s + p on each of two atoms.
        let mk_basis = |c0: [f64; 3], c1: [f64; 3]| {
            Basis::new(vec![
                Shell::new(0, c0, vec![0.7], vec![1.0]).unwrap(),
                Shell::new_spherical(1, c0, vec![0.6], vec![1.0]).unwrap(),
                Shell::new(0, c1, vec![0.8], vec![1.0]).unwrap(),
                Shell::new_spherical(1, c1, vec![0.5], vec![1.0]).unwrap(),
            ])
        };
        let c0 = [1.2, 1.0, 1.1];
        let c1 = [2.7, 2.6, 2.8];
        let basis = mk_basis(c0, c1);
        let nao = basis.nao();
        let ao_atom = ao_atom_map(&basis);

        let kfracs = [[0.0, 0.0, 0.0], [0.3, -0.1, 0.2]];
        let weights = [0.55, 0.45];
        // A fixed real-symmetric P^k (a valid Hermitian density matrix), same for both k.
        let mut p = vec![Complex64::new(0.0, 0.0); nao * nao];
        for a in 0..nao {
            for b in 0..nao {
                let v = 0.15 * ((a + 1) as f64) * 0.1
                    + 0.07 * (a * b) as f64
                    + 0.2 * ((b + 1) as f64) * 0.1;
                p[a * nao + b] = Complex64::new(v, 0.0);
            }
        }
        for a in 0..nao {
            for b in (a + 1)..nao {
                let s = 0.5 * (p[a * nao + b] + p[b * nao + a]);
                p[a * nao + b] = s;
                p[b * nao + a] = s;
            }
        }
        let p_k = vec![p.clone(), p];
        // A fixed smooth periodic potential.
        let v: Vec<f64> = grid
            .points()
            .iter()
            .map(|r| (2.0 * PI * r[0] / 5.0).cos() + 0.4 * (2.0 * PI * r[1] / 5.0).sin() + 1.0)
            .collect();

        let coll = LatticeCollocator::new(&basis, &grid);
        let phases = coll.bloch_phases(&kfracs, &weights);
        let analytic = coll.collocation_pulay_forces(&grid, &p_k, &v, &phases, &ao_atom, 2);

        // FD: E(R) = dV Σ_g v(g) n(g; R), F^coll = −∂E/∂R.
        let energy = |c0: [f64; 3], c1: [f64; 3]| -> f64 {
            let b = mk_basis(c0, c1);
            let coll = LatticeCollocator::new(&b, &grid);
            let ph = coll.bloch_phases(&kfracs, &weights);
            let n = coll.collocate_k(&grid, &p_k, &ph);
            grid.dv() * v.iter().zip(&n).map(|(&vv, &nn)| vv * nn).sum::<f64>()
        };
        let h = 1e-5;
        let centers = [c0, c1];
        for atom in 0..2 {
            for axis in 0..3 {
                let mut cp = centers;
                cp[atom][axis] += h;
                let e_plus = energy(cp[0], cp[1]);
                cp[atom][axis] -= 2.0 * h;
                let e_minus = energy(cp[0], cp[1]);
                let fd = -(e_plus - e_minus) / (2.0 * h);
                assert!(
                    (analytic[atom][axis] - fd).abs() < 1e-6,
                    "atom {atom} axis {axis}: analytic {} vs FD {fd}",
                    analytic[atom][axis]
                );
            }
        }
    }

    /// The collocation Pulay **stress** matches the central FD of `∫ v·n(ε)` (the local grid
    /// energy with the fixed potential `v` and density matrix `P^k`, the basis + cell deforming
    /// with the strain, the volume element `dV` held fixed so only the `∂n/∂ε` part is tested)
    /// — the M5 collocation Pulay stress validation. Two k-points; uniaxial/shear/general
    /// strains.
    #[test]
    fn collocation_pulay_stress_matches_finite_difference() {
        let cell = Cell::cubic(5.0).unwrap();
        let dims = [36usize, 36, 36];
        let grid = RealSpaceGrid::new(cell, dims);
        let mk_basis = |c0: [f64; 3], c1: [f64; 3]| {
            Basis::new(vec![
                Shell::new(0, c0, vec![0.7], vec![1.0]).unwrap(),
                Shell::new_spherical(1, c0, vec![0.6], vec![1.0]).unwrap(),
                Shell::new(0, c1, vec![0.8], vec![1.0]).unwrap(),
                Shell::new_spherical(1, c1, vec![0.5], vec![1.0]).unwrap(),
            ])
        };
        let c0 = [1.2, 1.0, 1.1];
        let c1 = [2.7, 2.6, 2.8];
        let basis = mk_basis(c0, c1);
        let nao = basis.nao();
        let kfracs = [[0.0, 0.0, 0.0], [0.3, -0.1, 0.2]];
        let weights = [0.55, 0.45];
        let mut p = vec![Complex64::new(0.0, 0.0); nao * nao];
        for a in 0..nao {
            for b in 0..nao {
                let v = 0.15 * ((a + 1) as f64) * 0.1 + 0.07 * (a * b) as f64;
                p[a * nao + b] = Complex64::new(v, 0.0);
            }
        }
        for a in 0..nao {
            for b in (a + 1)..nao {
                let s = 0.5 * (p[a * nao + b] + p[b * nao + a]);
                p[a * nao + b] = s;
                p[b * nao + a] = s;
            }
        }
        let p_k = vec![p.clone(), p];
        let v: Vec<f64> = grid
            .points()
            .iter()
            .map(|r| (2.0 * PI * r[0] / 5.0).cos() + 0.4 * (2.0 * PI * r[1] / 5.0).sin() + 1.0)
            .collect();
        let dv0 = grid.dv();

        let coll = LatticeCollocator::new(&basis, &grid);
        let phases = coll.bloch_phases(&kfracs, &weights);
        let tau = coll.collocation_pulay_stress(&grid, &p_k, &v, &phases);

        let deform = |m: &[[f64; 3]; 3], lambda: f64, vv: [f64; 3]| -> [f64; 3] {
            let mut o = vv;
            for (a, oa) in o.iter_mut().enumerate() {
                for (b, &vb) in vv.iter().enumerate() {
                    *oa += lambda * m[a][b] * vb;
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
            // E(λ) = dv0 Σ_g v_g n_g(λ): density collocated from the strained basis on the
            // strained grid, V and dV held fixed (so the FD isolates ∫v ∂n/∂ε).
            let energy = |lambda: f64| -> f64 {
                let (a1, a2, a3) = cell.vectors();
                let dcell = Cell::from_vectors(
                    deform(&mdir, lambda, a1),
                    deform(&mdir, lambda, a2),
                    deform(&mdir, lambda, a3),
                )
                .unwrap();
                let dgrid = RealSpaceGrid::new(dcell, dims);
                let db = mk_basis(deform(&mdir, lambda, c0), deform(&mdir, lambda, c1));
                let dcoll = LatticeCollocator::new(&db, &dgrid);
                let dph = dcoll.bloch_phases(&kfracs, &weights);
                let n = dcoll.collocate_k(&dgrid, &p_k, &dph);
                dv0 * v.iter().zip(&n).map(|(&vv, &nn)| vv * nn).sum::<f64>()
            };
            let h = 1e-5;
            let fd = (energy(h) - energy(-h)) / (2.0 * h);
            let analytic: f64 = (0..3)
                .flat_map(|a| (0..3).map(move |b| (a, b)))
                .map(|(a, b)| tau[a][b] * mdir[a][b])
                .sum();
            assert!(
                (analytic - fd).abs() < 1e-6,
                "strain {mdir:?}: analytic {analytic} vs FD {fd}"
            );
        }
    }

    /// Cross-check the grid image-integration against the analytic Bloch engine: with a
    /// constant potential `v ≡ 1`, `Σ_S W^S` is the Γ overlap `Σ_R⟨μ|ν+R⟩`, which the
    /// analytic [`bloch_overlap`] computes exactly. They agree to grid accuracy.
    #[test]
    fn lattice_overlap_matches_analytic_bloch_at_gamma() {
        let basis = Basis::new(vec![
            Shell::new(0, [0.0, 0.0, 0.0], vec![0.6], vec![1.0]).unwrap(),
            Shell::new(0, [2.1, 0.0, 0.0], vec![0.8], vec![1.0]).unwrap(),
        ]);
        let cell = Cell::cubic(4.2).unwrap();
        let grid = RealSpaceGrid::new(cell, [56, 56, 56]);
        let nao = basis.nao();
        let coll = LatticeCollocator::new(&basis, &grid);
        let w = coll.integrate(&grid, &vec![1.0; grid.n_points()]);
        // Σ_S W^S.
        let mut s_grid = vec![0.0; nao * nao];
        for i in 0..w.len() {
            for (acc, wij) in s_grid.iter_mut().zip(w.block(i)) {
                *acc += wij;
            }
        }
        // Analytic Bloch overlap at Γ over a generous image radius.
        let s_an = bloch_overlap(&basis, &cell, [0.0, 0.0, 0.0], 20.0);
        let max = (0..nao * nao)
            .map(|i| (s_grid[i] - s_an[i].re).abs())
            .fold(0.0, f64::max);
        assert!(max < 2e-3, "grid Σ_S W^S vs analytic Bloch overlap: {max}");
    }
}
