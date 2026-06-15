//! Density-fitting (RI) integrals: 3-center `(μν|P)` and 2-center `(P|Q)`
//! Coulomb integrals over a main basis plus an auxiliary fitting basis.
//!
//! Both reduce to the existing 4-center Coulomb kernels by attaching a **unit
//! `s` dummy** to the auxiliary index: a zero-exponent, coefficient-1 `s`
//! primitive is the constant function `1`, so
//!
//! ```text
//!   (μν|P) = (μν|P·1ₛ)        (P|Q) = (P·1ₛ|Q·1ₛ)
//! ```
//!
//! *exactly* (no limit is taken — the Gaussian-product machinery is evaluated
//! at `α = 0`, where every pair quantity is finite: the pair exponent is
//! `ζ = α_P + 0 = α_P`, the prefactor `K = exp(0) = 1`, and the pair center is
//! the aux center itself). The dummy is exempt from primitive normalization
//! (see [`Shell::primitive_coeff`]). Both ERI engines (OS/HGP and Rys) divide
//! only by *pair* exponent sums, never by an individual primitive exponent, so
//! the zero-exponent index is safe in either.
//!
//! The public surface mirrors the 4-center family: dense + `_with(Engine)`
//! variants, kind-aware spherical sizes, row-major blocks with the **last
//! index fastest**, an aux-side Schwarz bound, and a parallel-ready
//! [`Eri3cBuilder`] with the same partition/fill contract as
//! [`crate::EriBuilder`].

use crate::integrals::{
    canonical_shell_pairs, effective_coeffs, quartet_into_scratch, quartet_into_scratch_erf,
    Engine, EriKernel, QuartetScratch,
};
use crate::shell::{Basis, Shell};
use crate::spherical::shell_transform;

/// The unit `s` dummy at `center`: one zero-exponent primitive, coefficient 1,
/// i.e. the constant function `1` (normalization-exempt; see module docs).
/// Shared with the density-fitting gradient builders in `grad.rs`.
pub(crate) fn unit_s(center: [f64; 3]) -> Shell {
    Shell::new(0, center, vec![0.0], vec![1.0]).expect("unit s dummy shell is always valid")
}

impl Basis {
    /// 2-center Coulomb metric `(P|Q) = ∫∫ φ_P(1) r₁₂⁻¹ φ_Q(2) d1 d2` over
    /// `self` as the **auxiliary** basis.
    ///
    /// Row-major `[naux, naux]` with `naux = self.nao()`, kind-aware (spherical
    /// shells contribute their `2l+1` components). The matrix is **exactly
    /// symmetric**: each canonical shell pair is evaluated once and mirrored,
    /// so `(P|Q)` and `(Q|P)` are the same `f64`.
    #[must_use]
    pub fn eri_2c(&self) -> Vec<f64> {
        self.eri_2c_with(Engine::Auto)
    }

    /// Like [`Basis::eri_2c`] but forces a specific [`Engine`] (or
    /// [`Engine::Auto`]). Both engines produce the same metric to tolerance.
    #[must_use]
    pub fn eri_2c_with(&self, engine: Engine) -> Vec<f64> {
        let naux = self.nao();
        let offs = self.offsets();
        let shells = self.shells();
        let eff: Vec<Vec<f64>> = shells.iter().map(effective_coeffs).collect();
        let c2s: Vec<Option<Vec<f64>>> = shells.iter().map(shell_transform).collect();
        let dummies: Vec<Shell> = shells.iter().map(|s| unit_s(s.center())).collect();
        let unit_eff = [1.0];
        let mut out = vec![0.0; naux * naux];
        let mut scratch = QuartetScratch::default();
        for (p, sp) in shells.iter().enumerate() {
            for (q, sq) in shells.iter().enumerate().take(p + 1) {
                let len = quartet_into_scratch(
                    &mut scratch,
                    engine,
                    [sp, &dummies[p], sq, &dummies[q]],
                    [&eff[p], &unit_eff, &eff[q], &unit_eff],
                    [c2s[p].as_deref(), None, c2s[q].as_deref(), None],
                );
                let (np, nq) = (sp.n_func(), sq.n_func());
                debug_assert_eq!(len, np * nq);
                for a in 0..np {
                    for b in 0..nq {
                        let v = scratch.block[a * nq + b];
                        out[(offs[p] + a) * naux + offs[q] + b] = v;
                        out[(offs[q] + b) * naux + offs[p] + a] = v;
                    }
                }
            }
        }
        out
    }

    /// 2-center metric over the chosen [`EriKernel`] — layout identical to
    /// [`Basis::eri_2c`] (row-major `[naux, naux]`, exactly symmetric).
    ///
    /// [`EriKernel::Coulomb`] routes to [`Basis::eri_2c`] itself (bit-identical
    /// output); [`EriKernel::Erf`] evaluates `(P| erf(ω·r₁₂)/r₁₂ |Q)` via the
    /// same zero-exponent unit-`s` dummy construction, on the Rys engine (see
    /// [`Basis::eri_kernel`]).
    ///
    /// # Panics
    /// Panics if `k` is `Erf { omega }` with `ω ≤ 0`, NaN, or infinite.
    #[must_use]
    pub fn eri_2c_kernel(&self, k: EriKernel) -> Vec<f64> {
        let omega = match k {
            EriKernel::Coulomb => return self.eri_2c(),
            EriKernel::Erf { omega } => {
                crate::integrals::check_erf_omega(omega);
                omega
            }
        };
        let naux = self.nao();
        let offs = self.offsets();
        let shells = self.shells();
        let eff: Vec<Vec<f64>> = shells.iter().map(effective_coeffs).collect();
        let c2s: Vec<Option<Vec<f64>>> = shells.iter().map(shell_transform).collect();
        let dummies: Vec<Shell> = shells.iter().map(|s| unit_s(s.center())).collect();
        let unit_eff = [1.0];
        let mut out = vec![0.0; naux * naux];
        let mut scratch = QuartetScratch::default();
        for (p, sp) in shells.iter().enumerate() {
            for (q, sq) in shells.iter().enumerate().take(p + 1) {
                let len = quartet_into_scratch_erf(
                    &mut scratch,
                    [sp, &dummies[p], sq, &dummies[q]],
                    [&eff[p], &unit_eff, &eff[q], &unit_eff],
                    [c2s[p].as_deref(), None, c2s[q].as_deref(), None],
                    omega,
                );
                let (np, nq) = (sp.n_func(), sq.n_func());
                debug_assert_eq!(len, np * nq);
                for a in 0..np {
                    for b in 0..nq {
                        let v = scratch.block[a * nq + b];
                        out[(offs[p] + a) * naux + offs[q] + b] = v;
                        out[(offs[q] + b) * naux + offs[p] + a] = v;
                    }
                }
            }
        }
        out
    }

    /// One 3-center Coulomb shell block `(ij|P)`: shells `ish, jsh` over `self`
    /// (the main basis), shell `psh` over `aux` (the auxiliary basis).
    ///
    /// Row-major `[n_i, n_j, n_p]` with **`P` fastest-varying** (so a
    /// per-shell-pair GEMM against the metric is contiguous), kind-aware like
    /// [`Basis::eri_block`]: `n_x` is the shell's `n_func` (`n_cart` for
    /// Cartesian, `2l+1` for spherical) in the usual component order.
    #[must_use]
    pub fn eri_3c_block(&self, aux: &Basis, ish: usize, jsh: usize, psh: usize) -> Vec<f64> {
        self.eri_3c_block_with(Engine::Auto, aux, ish, jsh, psh)
    }

    /// Like [`Basis::eri_3c_block`] but forces a specific [`Engine`] (or
    /// [`Engine::Auto`]). Both engines produce the same block to tolerance.
    #[must_use]
    pub fn eri_3c_block_with(
        &self,
        engine: Engine,
        aux: &Basis,
        ish: usize,
        jsh: usize,
        psh: usize,
    ) -> Vec<f64> {
        let s = self.shells();
        let (si, sj) = (&s[ish], &s[jsh]);
        let sp = &aux.shells()[psh];
        let dummy = unit_s(sp.center());
        let (mi, mj, mp) = (
            shell_transform(si),
            shell_transform(sj),
            shell_transform(sp),
        );
        let mut scratch = QuartetScratch::default();
        let len = quartet_into_scratch(
            &mut scratch,
            engine,
            [si, sj, sp, &dummy],
            [
                &effective_coeffs(si),
                &effective_coeffs(sj),
                &effective_coeffs(sp),
                &[1.0],
            ],
            [mi.as_deref(), mj.as_deref(), mp.as_deref(), None],
        );
        scratch.block[..len].to_vec()
    }

    /// One 3-center shell block over the chosen [`EriKernel`] — layout identical
    /// to [`Basis::eri_3c_block`] (row-major `[n_i, n_j, n_p]`, `P` fastest).
    ///
    /// [`EriKernel::Coulomb`] routes to [`Basis::eri_3c_block`] itself
    /// (bit-identical output); [`EriKernel::Erf`] evaluates
    /// `(ij| erf(ω·r₁₂)/r₁₂ |P)` via the same zero-exponent unit-`s` dummy
    /// construction, on the Rys engine (see [`Basis::eri_kernel`]). The Coulomb
    /// Schwarz factors ([`Basis::schwarz_bounds`] / [`Basis::schwarz_aux_bounds`])
    /// remain valid upper bounds for the attenuated blocks (`erf(ωr)/r ≤ 1/r`).
    ///
    /// # Panics
    /// Panics if `k` is `Erf { omega }` with `ω ≤ 0`, NaN, or infinite.
    #[must_use]
    pub fn eri_3c_block_kernel(
        &self,
        aux: &Basis,
        ish: usize,
        jsh: usize,
        psh: usize,
        k: EriKernel,
    ) -> Vec<f64> {
        let omega = match k {
            EriKernel::Coulomb => return self.eri_3c_block(aux, ish, jsh, psh),
            EriKernel::Erf { omega } => {
                crate::integrals::check_erf_omega(omega);
                omega
            }
        };
        let s = self.shells();
        let (si, sj) = (&s[ish], &s[jsh]);
        let sp = &aux.shells()[psh];
        let dummy = unit_s(sp.center());
        let (mi, mj, mp) = (
            shell_transform(si),
            shell_transform(sj),
            shell_transform(sp),
        );
        let mut scratch = QuartetScratch::default();
        let len = quartet_into_scratch_erf(
            &mut scratch,
            [si, sj, sp, &dummy],
            [
                &effective_coeffs(si),
                &effective_coeffs(sj),
                &effective_coeffs(sp),
                &[1.0],
            ],
            [mi.as_deref(), mj.as_deref(), mp.as_deref(), None],
            omega,
        );
        scratch.block[..len].to_vec()
    }

    /// Auxiliary-side Schwarz factors over `self` as the aux basis, one per
    /// shell: `QP[p] = sqrt(max_{μ∈p} (μ|μ))` with `(μ|μ)` the diagonal of the
    /// 2-center block `(p|p)`.
    ///
    /// Together with the main-basis [`Basis::schwarz_bounds`] this bounds every
    /// 3-center integral: `|(μν|P)| ≤ Q[i,j] · QP[p]` for `μν` in shell pair
    /// `(i, j)` and `P` in aux shell `p` (Cauchy–Schwarz in the Coulomb inner
    /// product). Kind-aware, like the 4-center bounds.
    #[must_use]
    pub fn schwarz_aux_bounds(&self) -> Vec<f64> {
        self.schwarz_aux_bounds_with(Engine::Auto)
    }

    /// Like [`Basis::schwarz_aux_bounds`] but with a forced [`Engine`]. The
    /// bound is engine-independent to tolerance.
    #[must_use]
    pub fn schwarz_aux_bounds_with(&self, engine: Engine) -> Vec<f64> {
        let shells = self.shells();
        let eff: Vec<Vec<f64>> = shells.iter().map(effective_coeffs).collect();
        let c2s: Vec<Option<Vec<f64>>> = shells.iter().map(shell_transform).collect();
        let unit_eff = [1.0];
        let mut scratch = QuartetScratch::default();
        let mut bounds = Vec::with_capacity(shells.len());
        for (p, sp) in shells.iter().enumerate() {
            let dummy = unit_s(sp.center());
            let len = quartet_into_scratch(
                &mut scratch,
                engine,
                [sp, &dummy, sp, &dummy],
                [&eff[p], &unit_eff, &eff[p], &unit_eff],
                [c2s[p].as_deref(), None, c2s[p].as_deref(), None],
            );
            let np = sp.n_func();
            debug_assert_eq!(len, np * np);
            let mut mx = 0.0_f64;
            for mu in 0..np {
                mx = mx.max(scratch.block[mu * np + mu].abs());
            }
            bounds.push(mx.sqrt());
        }
        bounds
    }

    /// Create a parallel-ready [`Eri3cBuilder`] filling `(ij|P)` with `ij` over
    /// `self` (the main basis) and `P` over `aux`, with the default
    /// [`Engine::Auto`] dispatch. Equivalent to [`Eri3cBuilder::new`].
    #[must_use]
    pub fn eri_3c_builder<'a>(&'a self, aux: &'a Basis) -> Eri3cBuilder<'a> {
        Eri3cBuilder::new(self, aux)
    }
}

/// A reusable plan for assembling the dense 3-center tensor `(ij|P)` —
/// row-major `[nao, nao, naux]`, `P` fastest — in parallel over canonical
/// **bra shell-pairs** of the main basis, with no in-crate threading runtime.
///
/// Same contract as [`crate::EriBuilder`]: [`Eri3cBuilder::partition`] slices
/// the caller's buffer into one `naux` row slab per `(μ, ν)` AO pair via
/// `chunks_exact_mut` (provably disjoint `&mut` views, no `unsafe`), and hands
/// each canonical bra-pair `(i ≥ j)` exactly the rows it owns — the `(i, j)`
/// band and, when `i ≠ j`, the `(j, i)` band. A driver (e.g. rayon at the call
/// site) fills the returned tasks concurrently with [`Eri3cBuilder::fill`];
/// chemx's existing LPT dispatch over [`crate::EriBuilder`] tasks carries over
/// unchanged.
///
/// Unlike the 4-center builder there is no bra↔ket exchange to trade away: the
/// only symmetry is the bra swap `(μν|P) = (νμ|P)`, so each canonical bra-pair
/// evaluates every aux shell once and writes both orderings.
///
/// # Example — serial
/// ```
/// use integral::{Basis, Shell};
/// let main = Basis::new(vec![
///     Shell::new(0, [0.0, 0.0, 0.0], vec![0.8], vec![1.0]).unwrap(),
///     Shell::new(1, [0.1, 0.0, 0.0], vec![0.6], vec![1.0]).unwrap(),
/// ]);
/// let aux = Basis::new(vec![
///     Shell::new(0, [0.0, 0.0, 0.0], vec![1.2], vec![1.0]).unwrap(),
/// ]);
/// let b = main.eri_3c_builder(&aux);
/// let tensor = b.build();
/// assert_eq!(tensor.len(), main.nao() * main.nao() * aux.nao());
/// ```
#[derive(Debug)]
pub struct Eri3cBuilder<'b> {
    main: &'b [Shell],
    aux: &'b [Shell],
    engine: Engine,
    /// Output-AO offset of each main-basis shell (function-space).
    offs: Vec<usize>,
    /// `n_func` of each main-basis shell.
    nfunc: Vec<usize>,
    /// Total main-basis output AOs.
    nao: usize,
    /// Aux-shell AO offsets and total aux AOs.
    aux_offs: Vec<usize>,
    naux: usize,
    /// Effective contraction coefficients per shell (`d_i · N(α_i, l)`).
    eff: Vec<Vec<f64>>,
    aux_eff: Vec<Vec<f64>>,
    /// Cached `c2s` transform per shell (`None` = Cartesian).
    c2s: Vec<Option<Vec<f64>>>,
    aux_c2s: Vec<Option<Vec<f64>>>,
    /// One unit-`s` dummy per aux shell, at that shell's center.
    dummies: Vec<Shell>,
    /// Canonical main-basis shell pairs `(i ≥ j)`, the parallel grain.
    pairs: Vec<(usize, usize)>,
}

impl<'b> Eri3cBuilder<'b> {
    /// Build a plan for `(ij|P)` over `main` × `aux` with the default
    /// [`Engine::Auto`] dispatch.
    #[must_use]
    pub fn new(main: &'b Basis, aux: &'b Basis) -> Self {
        Self::with_engine(main, aux, Engine::Auto)
    }

    /// Build a plan that forces a specific [`Engine`] (or [`Engine::Auto`]).
    #[must_use]
    pub fn with_engine(main: &'b Basis, aux: &'b Basis, engine: Engine) -> Self {
        let shells = main.shells();
        let aux_shells = aux.shells();
        Eri3cBuilder {
            main: shells,
            aux: aux_shells,
            engine,
            offs: main.offsets(),
            nfunc: shells.iter().map(Shell::n_func).collect(),
            nao: main.nao(),
            aux_offs: aux.offsets(),
            naux: aux.nao(),
            eff: shells.iter().map(effective_coeffs).collect(),
            aux_eff: aux_shells.iter().map(effective_coeffs).collect(),
            c2s: shells.iter().map(shell_transform).collect(),
            aux_c2s: aux_shells.iter().map(shell_transform).collect(),
            dummies: aux_shells.iter().map(|s| unit_s(s.center())).collect(),
            pairs: canonical_shell_pairs(shells.len()),
        }
    }

    /// The canonical bra shell-pairs `(i, j)` with `i ≥ j`, in build order —
    /// the **external parallel grain**, aligned index-for-index with the tasks
    /// returned by [`Eri3cBuilder::partition`].
    #[must_use]
    pub fn bra_pairs(&self) -> &[(usize, usize)] {
        &self.pairs
    }

    /// Length of the dense output buffer, `nao² · naux`. Allocate
    /// `vec![0.0; output_len()]` before [`Eri3cBuilder::partition`].
    #[must_use]
    pub fn output_len(&self) -> usize {
        self.nao * self.nao * self.naux
    }

    /// Partition a freshly-zeroed `nao²·naux` output buffer into one
    /// [`Bra3cFill`] task per canonical bra-pair (aligned with
    /// [`Eri3cBuilder::bra_pairs`]). Each task borrows **only** the `(μ, ν)`
    /// row slabs it owns; the borrows are mutually disjoint, so the tasks may
    /// be filled concurrently into the same buffer.
    ///
    /// # Panics
    /// If `out.len() != output_len()`.
    #[must_use]
    pub fn partition<'o>(&self, out: &'o mut [f64]) -> Vec<Bra3cFill<'o>> {
        let nao = self.nao;
        assert_eq!(
            out.len(),
            nao * nao * self.naux,
            "3c output buffer must be nao²·naux = {} elements",
            nao * nao * self.naux
        );

        // One mutable slab of naux elements per (μ, ν) AO row, row = μ·nao + ν.
        let mut slabs: Vec<Option<&'o mut [f64]>> =
            out.chunks_exact_mut(self.naux).map(Some).collect();
        debug_assert_eq!(slabs.len(), nao * nao);

        let mut tasks = Vec::with_capacity(self.pairs.len());
        for &(i, j) in &self.pairs {
            let (ni, nj) = (self.nfunc[i], self.nfunc[j]);
            let (oi, oj) = (self.offs[i], self.offs[j]);

            // (i, j) band: rows (μ∈i, ν∈j), row-major (a, b).
            let mut ij_band = Vec::with_capacity(ni * nj);
            for a in 0..ni {
                for b in 0..nj {
                    ij_band.push(claim_row(&mut slabs, (oi + a) * nao + (oj + b)));
                }
            }

            // (j, i) band: rows (μ∈j, ν∈i), row-major (b, a). Empty when i == j.
            let mut ji_band = Vec::new();
            if i != j {
                ji_band.reserve(nj * ni);
                for b in 0..nj {
                    for a in 0..ni {
                        ji_band.push(claim_row(&mut slabs, (oj + b) * nao + (oi + a)));
                    }
                }
            }

            tasks.push(Bra3cFill {
                bra: (i, j),
                ij_band,
                ji_band,
            });
        }

        debug_assert!(
            slabs.iter().all(Option::is_none),
            "partition left {} output rows unclaimed",
            slabs.iter().filter(|s| s.is_some()).count()
        );

        tasks
    }

    /// Fill one bra-pair's owned rows: evaluate `(ij|P)` once per aux shell and
    /// write the block into the `(i, j)` band and (when `i ≠ j`) the bra-swapped
    /// `(j, i)` band. Writes touch only `task`'s rows, so this may run
    /// concurrently with [`Eri3cBuilder::fill`] on every *other* task.
    pub fn fill(&self, task: &mut Bra3cFill<'_>) {
        self.fill_filtered(task, |_| true);
    }

    /// Like [`Eri3cBuilder::fill`] but evaluates only the aux shells `p` for
    /// which `keep(p)` is true, skipping the rest entirely — the hook for
    /// per-aux-shell Schwarz screening (`Q[ij]·QP[p] < τ`) *inside* a bra-pair
    /// task. Skipped shells' output slots are left untouched, i.e. zero in the
    /// freshly-zeroed buffer [`Eri3cBuilder::partition`] requires.
    ///
    /// With `keep = |_| true` this is the identical code path to
    /// [`Eri3cBuilder::fill`] (bitwise-equal output).
    pub fn fill_filtered(&self, task: &mut Bra3cFill<'_>, keep: impl Fn(usize) -> bool) {
        let (i, j) = task.bra;
        let (si, sj) = (&self.main[i], &self.main[j]);
        let (ni, nj) = (self.nfunc[i], self.nfunc[j]);
        let mut scratch = QuartetScratch::default();
        for (p, sp) in self.aux.iter().enumerate() {
            if !keep(p) {
                continue;
            }
            let len = quartet_into_scratch(
                &mut scratch,
                self.engine,
                [si, sj, sp, &self.dummies[p]],
                [&self.eff[i], &self.eff[j], &self.aux_eff[p], &[1.0]],
                [
                    self.c2s[i].as_deref(),
                    self.c2s[j].as_deref(),
                    self.aux_c2s[p].as_deref(),
                    None,
                ],
            );
            let np = sp.n_func();
            debug_assert_eq!(len, ni * nj * np);
            let op = self.aux_offs[p];
            let block = &scratch.block[..len];
            for a in 0..ni {
                for b in 0..nj {
                    let row = &block[(a * nj + b) * np..(a * nj + b + 1) * np];
                    task.ij_band[a * nj + b][op..op + np].copy_from_slice(row);
                    if i != j {
                        task.ji_band[b * ni + a][op..op + np].copy_from_slice(row);
                    }
                }
            }
        }
    }

    /// Assemble the whole dense `(ij|P)` tensor on the current thread by
    /// filling every bra-pair in sequence — the identical code path a parallel
    /// driver runs, just serially.
    #[must_use]
    pub fn build(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.output_len()];
        let mut tasks = self.partition(&mut out);
        for task in &mut tasks {
            self.fill(task);
        }
        out
    }
}

/// One unit of parallel 3-center work: the `(μ, ν)` output rows owned by a
/// single canonical bra-pair `(i, j)`, handed out by [`Eri3cBuilder::partition`].
/// Distinct tasks borrow disjoint regions of the same buffer, so a driver may
/// fill them across threads with no synchronisation ([`Eri3cBuilder::fill`]).
#[derive(Debug)]
pub struct Bra3cFill<'o> {
    bra: (usize, usize),
    /// Slabs for rows `(μ∈i, ν∈j)`, row-major `(a, b)` → index `a·n_j + b`;
    /// each slab is the `naux`-long `P` row of that `(μ, ν)`.
    ij_band: Vec<&'o mut [f64]>,
    /// Slabs for rows `(μ∈j, ν∈i)`, row-major `(b, a)` → index `b·n_i + a`.
    /// Empty when `i == j`.
    ji_band: Vec<&'o mut [f64]>,
}

impl Bra3cFill<'_> {
    /// The canonical bra shell-pair `(i, j)` (`i ≥ j`) this task fills.
    #[must_use]
    pub fn bra(&self) -> (usize, usize) {
        self.bra
    }
}

/// Take the slab for output `row`, asserting it has not already been claimed
/// (a double-claim would violate the disjointness contract).
fn claim_row<'o>(slabs: &mut [Option<&'o mut [f64]>], row: usize) -> &'o mut [f64] {
    slabs[row]
        .take()
        .expect("output row claimed by two bra-pairs (disjointness violated)")
}
