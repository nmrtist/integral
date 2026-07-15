//! Parallel-ready dense ERI assembly: [`EriBuilder`].
//!
//! [`Basis::eri`](crate::Basis::eri) is the single-threaded dense build — it walks
//! the *fully* canonical 8-fold quartets (`i ≥ j`, `k ≥ l`, `ij ≥ kl`) and
//! scatters each block to all eight permutation-equivalent slots. That last
//! constraint (`ij ≥ kl`, the bra↔ket exchange) is what makes it inherently
//! serial: a single computed block writes into *both* the bra rows and the ket
//! rows, so two quartets that share a shell pair race on the shared output.
//!
//! [`EriBuilder`] trades that exchange away to make the build embarrassingly
//! parallel **without any in-crate threading runtime** (no `rayon`; integral
//! stays dependency-free at runtime and in its tests). The grain is the canonical
//! **bra shell-pair** `(i, j)`, `i ≥ j` ([`EriBuilder::bra_pairs`]). For one such
//! bra-pair the fill sweeps *all* canonical ket-pairs `(k, l)`, `k ≥ l` (the full
//! set — there is no `ij ≥ kl` pruning here), evaluates `(ij|kl)` once, and
//! scatters it into the **four** bra/ket-internal symmetry slots only:
//!
//! ```text
//!   (μν|λσ)  (μν|σλ)  (νμ|λσ)  (νμ|σλ)
//! ```
//!
//! i.e. the swaps `μ↔ν` (within the bra) and `λ↔σ` (within the ket), but **never**
//! the bra↔ket exchange `(μν|λσ) → (λσ|μν)`. Every slot a bra-pair `(i, j)` writes
//! therefore lands in a row `(μ, ν)` whose shell pair is `{i, j}` (ordered either
//! way). Two *distinct* canonical bra-pairs own *disjoint* unordered shell pairs,
//! hence disjoint output rows — so distinct bra-pairs can be filled concurrently
//! into one shared buffer with no synchronisation.
//!
//! ## Disjointness contract (how the public surface stays safe)
//!
//! integral owns the partitioning. [`EriBuilder::partition`] slices the caller's
//! `nao⁴` buffer into one `nao²` "row slab" per `(μ, ν)` AO pair (via
//! [`slice::chunks_exact_mut`], so the slabs are provably non-overlapping `&mut`
//! views), then hands each canonical bra-pair exactly the slabs for the rows it
//! owns. Because a bra-pair `(i, j)` with `i ≠ j` writes the `(i, j)` band **and**
//! the `(j, i)` band, its rows are *not* one contiguous slab — the partition
//! routes both bands. The result is a `Vec<`[`BraPairFill`]`>`, one per bra-pair,
//! each holding only its own disjoint `&mut` slabs. An external driver then runs
//! them concurrently (e.g. `rayon::par_iter_mut` at the call site) and calls
//! [`EriBuilder::fill`] on each.
//!
//! No `unsafe` is needed: the disjointness that would normally require raw-pointer
//! partitioning is expressed entirely through `chunks_exact_mut`, so integral keeps
//! its crate-wide `#![forbid(unsafe_code)]`. The split-band index math lives here,
//! inside integral; the public surface (`partition`, `fill`) is fully safe.
//!
//! ## Relationship to [`Basis::eri`](crate::Basis::eri)
//!
//! The assembled tensor equals the serial `eri()` tensor. It is **not** bitwise
//! identical everywhere, by design — see [`EriBuilder::build`] for the exact
//! tolerance contract and why.

use std::collections::BTreeMap;

use crate::engine::os_eri::{self, ShellPairData, ShellRef};
use crate::eri_batch::{batch_key, evaluate_batch4, EriBatchScratch};
use crate::integrals::{
    canonical_shell_pairs, check_erf_omega, effective_coeffs, quartet_into_scratch_erf,
    quartet_into_scratch_pairs, Engine, EriKernel, QuartetScratch, PERMS8,
};
use crate::shell::{Basis, Shell};
use crate::spherical::shell_transform;

/// A reusable plan for assembling the dense `(ij|kl)` ERI tensor in parallel over
/// canonical bra shell-pairs, with no in-crate threading runtime.
///
/// Construct once per [`Basis`] ([`EriBuilder::new`] / [`EriBuilder::with_engine`]);
/// it caches the per-shell data the dense build needs (effective contraction
/// coefficients, `c2s` transforms, AO offsets, the canonical pair list). The build
/// is then either serial ([`EriBuilder::build`]) or driven concurrently by the
/// caller over [`EriBuilder::partition`] + [`EriBuilder::fill`].
///
/// # Example — serial
/// ```
/// use integral::{Basis, EriBuilder, Shell};
/// let basis = Basis::new(vec![
///     Shell::new(0, [0.0, 0.0, 0.0], vec![0.8], vec![1.0]).unwrap(),
///     Shell::new(1, [0.1, 0.0, 0.0], vec![0.6], vec![1.0]).unwrap(),
/// ]);
/// let tensor = EriBuilder::new(&basis).build();
/// assert_eq!(tensor.len(), basis.nao().pow(4));
/// ```
///
/// # Example — parallel grain (driver-supplied threads)
/// ```
/// use integral::{Basis, EriBuilder, Shell};
/// let basis = Basis::new(vec![
///     Shell::new(0, [0.0, 0.0, 0.0], vec![0.8], vec![1.0]).unwrap(),
///     Shell::new(1, [0.1, 0.0, 0.0], vec![0.6], vec![1.0]).unwrap(),
/// ]);
/// let builder = EriBuilder::new(&basis);
/// let mut out = vec![0.0; builder.output_len()];
/// let mut tasks = builder.partition(&mut out);
/// // A driver (e.g. rayon) would run these concurrently; each task writes a
/// // disjoint set of rows, so a shared `&mut out` is safe by construction.
/// for task in &mut tasks {
///     builder.fill(task);
/// }
/// ```
#[derive(Debug)]
pub struct EriBuilder<'b> {
    shells: &'b [Shell],
    engine: Engine,
    /// The two-electron operator the build evaluates ([`EriKernel::Coulomb`]
    /// by default — that path is untouched and bit-identical to before the
    /// kernel was selectable).
    kernel: EriKernel,
    /// Output-AO offset of each shell (function-space).
    offs: Vec<usize>,
    /// `n_func` of each shell.
    nfunc: Vec<usize>,
    /// Total output AOs.
    nao: usize,
    /// Effective contraction coefficients per shell (`d_i · N(α_i, l)`).
    eff: Vec<Vec<f64>>,
    /// Cached `c2s` transform per shell (`None` = Cartesian).
    c2s: Vec<Option<Vec<f64>>>,
    /// Canonical shell pairs `(i ≥ j)`, the parallel grain and the ket sweep.
    pairs: Vec<(usize, usize)>,
    /// Primitive-pair data reused by every quartet containing the shell pair.
    pair_data: Vec<ShellPairData>,
    /// Precomputed Coulomb dispatch for every bra-pair. Each quartet is represented
    /// by its compact ket-pair index and appears exactly once, either in a compatible
    /// batch4 group or in the scalar/tail list. This keeps dispatch, bucketing, and
    /// queue allocation out of the parallel fill hot path.
    batch_schedule: EriBatchSchedule,
}

/// Compact, flat batch schedule shared by all bra-pair tasks.
///
/// The two offset arrays delimit one slice per canonical bra-pair. Storing a ket
/// pair as a `u32` rather than its four shell indices keeps the schedule close to
/// one word per quartet; shell indices and the fixed bra orientation are recovered
/// from `EriBuilder::pairs` at evaluation time.
#[derive(Debug)]
struct EriBatchSchedule {
    batch_offsets: Vec<usize>,
    batches: Vec<[u32; 4]>,
    scalar_offsets: Vec<usize>,
    scalars: Vec<u32>,
}

impl EriBatchSchedule {
    fn new(
        shells: &[Shell],
        engine: Engine,
        pairs: &[(usize, usize)],
        pair_data: &[ShellPairData],
    ) -> Self {
        assert!(
            u32::try_from(pairs.len()).is_ok(),
            "too many canonical shell pairs for dense ERI schedule"
        );

        let mut schedule = Self {
            batch_offsets: Vec::with_capacity(pairs.len() + 1),
            batches: Vec::new(),
            scalar_offsets: Vec::with_capacity(pairs.len() + 1),
            scalars: Vec::new(),
        };
        schedule.batch_offsets.push(0);
        schedule.scalar_offsets.push(0);

        // For a fixed bra pair, a batch key's bra angular momentum and primitive
        // count are constants. Ket pairs can therefore be grouped once globally by
        // the remaining two key fields and reused for every bra. This avoids even
        // construction-time per-bra maps and temporary bucket vectors.
        let mut ket_buckets: BTreeMap<(usize, usize), Vec<u32>> = BTreeMap::new();
        for (ket_pair_index, &(k, l)) in pairs.iter().enumerate() {
            ket_buckets
                .entry((
                    shells[k].l() + shells[l].l(),
                    pair_data[ket_pair_index].len(),
                ))
                .or_default()
                .push(ket_pair_index as u32);
        }

        // The global buckets and this fixed four-entry pending array exist only
        // while constructing the immutable plan. Parallel fill tasks subsequently
        // read flat slices and perform no dispatch lookup, map insertion, or queue
        // growth.
        for (bra_pair_index, &(i, j)) in pairs.iter().enumerate() {
            let bra_pair_count = pair_data[bra_pair_index].len();
            for bucket in ket_buckets.values() {
                let mut pending = [0_u32; 4];
                let mut pending_len = 0;
                for &ket_pair_index in bucket {
                    let ket = ket_pair_index as usize;
                    let (k, l) = pairs[ket];
                    if batch_key(
                        engine,
                        [&shells[i], &shells[j], &shells[k], &shells[l]],
                        [bra_pair_count, pair_data[ket].len()],
                    )
                    .is_some()
                    {
                        pending[pending_len] = ket_pair_index;
                        pending_len += 1;
                        if pending_len == 4 {
                            schedule.batches.push(pending);
                            pending_len = 0;
                        }
                    } else {
                        schedule.scalars.push(ket_pair_index);
                    }
                }
                schedule.scalars.extend_from_slice(&pending[..pending_len]);
            }
            schedule.batch_offsets.push(schedule.batches.len());
            schedule.scalar_offsets.push(schedule.scalars.len());
        }
        schedule
    }

    #[inline]
    fn batches(&self, bra_pair_index: usize) -> &[[u32; 4]] {
        &self.batches[self.batch_offsets[bra_pair_index]..self.batch_offsets[bra_pair_index + 1]]
    }

    #[inline]
    fn scalars(&self, bra_pair_index: usize) -> &[u32] {
        &self.scalars[self.scalar_offsets[bra_pair_index]..self.scalar_offsets[bra_pair_index + 1]]
    }
}

impl<'b> EriBuilder<'b> {
    /// Build a plan for `basis` with the default [`Engine::Auto`] dispatch.
    #[must_use]
    pub fn new(basis: &'b Basis) -> Self {
        Self::with_engine(basis, Engine::Auto)
    }

    /// Build a plan that forces a specific [`Engine`] (or [`Engine::Auto`]). Both
    /// engines produce the same tensor to tolerance; forcing exists so tests/CI
    /// exercise each path. See [`Basis::eri_with`](crate::Basis::eri_with).
    #[must_use]
    pub fn with_engine(basis: &'b Basis, engine: Engine) -> Self {
        let shells = basis.shells();
        let offs = basis.offsets();
        let nfunc: Vec<usize> = shells.iter().map(Shell::n_func).collect();
        let nao = basis.nao();
        let eff: Vec<Vec<f64>> = shells.iter().map(effective_coeffs).collect();
        let c2s: Vec<Option<Vec<f64>>> = shells.iter().map(shell_transform).collect();
        let pairs = canonical_shell_pairs(shells.len());
        let pair_data: Vec<ShellPairData> = pairs
            .iter()
            .map(|&(i, j)| {
                os_eri::shell_pair_data(
                    ShellRef {
                        center: shells[i].center(),
                        l: shells[i].l(),
                        exps: shells[i].exponents(),
                        coeffs: &eff[i],
                    },
                    ShellRef {
                        center: shells[j].center(),
                        l: shells[j].l(),
                        exps: shells[j].exponents(),
                        coeffs: &eff[j],
                    },
                )
            })
            .collect();
        let batch_schedule = EriBatchSchedule::new(shells, engine, &pairs, &pair_data);
        EriBuilder {
            shells,
            engine,
            kernel: EriKernel::Coulomb,
            offs,
            nfunc,
            nao,
            eff,
            c2s,
            pairs,
            pair_data,
            batch_schedule,
        }
    }

    /// Select the two-electron operator ([`EriKernel`]) the build evaluates —
    /// builder-style, so `EriBuilder::new(&basis).kernel(EriKernel::Erf {
    /// omega }).partition(..)` / `.fill(..)` / `.build()` assemble the
    /// **attenuated** tensor through the identical parallel seam.
    ///
    /// - [`EriKernel::Coulomb`] (the default) leaves every code path exactly
    ///   as before this setter existed: the Coulomb build is **bit-identical**
    ///   to a builder that never called `kernel`.
    /// - [`EriKernel::Erf`]`{ omega }` evaluates `erf(ω·r₁₂)/r₁₂` per quartet
    ///   via the Rys attenuated-quadrature transform (the same kernel as
    ///   [`Basis::eri_kernel`](crate::Basis::eri_kernel); the `engine` choice
    ///   is ignored — attenuation is Rys-only). The assembled tensor equals
    ///   the serial `eri_kernel(Erf)` tensor under the same bra↔ket round-off
    ///   contract as [`EriBuilder::build`] documents for Coulomb.
    ///
    /// # Panics
    /// If `k` is `Erf { omega }` with `ω ≤ 0`, NaN, or infinite.
    #[must_use]
    pub fn kernel(mut self, k: EriKernel) -> Self {
        if let EriKernel::Erf { omega } = k {
            check_erf_omega(omega);
        }
        self.kernel = k;
        self
    }

    /// The canonical bra shell-pairs `(i, j)` with `i ≥ j`, in build order — the
    /// **external parallel grain**. Index `p` here corresponds to the `p`-th task
    /// returned by [`EriBuilder::partition`]. A driver fans these out across
    /// threads; each owns a disjoint set of output rows.
    #[must_use]
    pub fn bra_pairs(&self) -> &[(usize, usize)] {
        &self.pairs
    }

    /// Length of the dense output buffer, `nao⁴`. Allocate `vec![0.0; output_len()]`
    /// before [`EriBuilder::partition`].
    #[must_use]
    pub fn output_len(&self) -> usize {
        let n = self.nao;
        n * n * n * n
    }

    /// Partition a freshly-zeroed `nao⁴` output buffer into one [`BraPairFill`] task
    /// per canonical bra-pair (aligned with [`EriBuilder::bra_pairs`]). Each task
    /// borrows **only** the row slabs it owns; the borrows are mutually disjoint, so
    /// the returned tasks may be filled concurrently into the same buffer.
    ///
    /// The caller must zero `out` first (the fill *assigns*, but only into the
    /// elements a bra-pair owns — collectively all `nao⁴`, so a correct full build
    /// overwrites everything; zeroing is the safe default and matters if a driver
    /// fills only a subset of tasks).
    ///
    /// # Panics
    /// If `out.len() != output_len()`.
    #[must_use]
    pub fn partition<'o>(&self, out: &'o mut [f64]) -> Vec<BraPairFill<'o>> {
        let nao = self.nao;
        let plane = nao * nao; // one (μ,ν) row spans the whole (λ,σ) plane
        assert_eq!(
            out.len(),
            plane * plane,
            "ERI output buffer must be nao⁴ = {} elements",
            plane * plane
        );

        // One mutable slab per (μ, ν) AO row, indexed by `row = μ·nao + ν`. These
        // are provably non-overlapping &mut views (chunks_exact_mut), the safe
        // substitute for raw-pointer partitioning.
        let mut slabs: Vec<Option<&'o mut [f64]>> = out.chunks_exact_mut(plane).map(Some).collect();
        debug_assert_eq!(slabs.len(), plane);

        let mut tasks = Vec::with_capacity(self.pairs.len());
        for &(i, j) in &self.pairs {
            let (ni, nj) = (self.nfunc[i], self.nfunc[j]);
            let (oi, oj) = (self.offs[i], self.offs[j]);

            // (i, j) band: rows (μ∈i, ν∈j), row-major (a, b).
            let mut ij_band = Vec::with_capacity(ni * nj);
            for a in 0..ni {
                for b in 0..nj {
                    let row = (oi + a) * nao + (oj + b);
                    ij_band.push(claim_row(&mut slabs, row));
                }
            }

            // (j, i) band: rows (μ∈j, ν∈i), row-major (b, a). Empty when i == j —
            // the bra-swap slot then coincides with the identity and the single
            // (i, i) band already covers both orderings (see `scatter_4fold`).
            let mut ji_band = Vec::new();
            if i != j {
                ji_band.reserve(nj * ni);
                for b in 0..nj {
                    for a in 0..ni {
                        let row = (oj + b) * nao + (oi + a);
                        ji_band.push(claim_row(&mut slabs, row));
                    }
                }
            }

            tasks.push(BraPairFill {
                bra: (i, j),
                ij_band,
                ji_band,
            });
        }

        // Every row must have been claimed exactly once: the take() in `claim_row`
        // already rejects a double-claim (overlap), and a leftover `Some` here would
        // mean an uncovered row (under-write). This turns the row-level disjointness
        // + coverage contract into a runtime-checked invariant.
        debug_assert!(
            slabs.iter().all(Option::is_none),
            "partition left {} output rows unclaimed",
            slabs.iter().filter(|s| s.is_some()).count()
        );

        tasks
    }

    /// Fill one bra-pair's owned rows: sweep all canonical ket-pairs `(k, l)`,
    /// evaluate `(ij|kl)` once each, and scatter into the four bra/ket-internal
    /// slots within this task's slabs. Writes touch only `task`'s rows, so this may
    /// run concurrently with [`EriBuilder::fill`] on every *other* task.
    pub fn fill(&self, task: &mut BraPairFill<'_>) {
        let (i, j) = task.bra;
        let mut sink = BandSink {
            nao: self.nao,
            off_i: self.offs[i],
            off_j: self.offs[j],
            n_i: self.nfunc[i],
            n_j: self.nfunc[j],
            ij: &mut task.ij_band,
            ji: &mut task.ji_band,
        };
        self.run_bra_pair(i, j, &mut sink);
    }

    /// Assemble the whole dense `(ij|kl)` tensor on the current thread by filling
    /// every bra-pair in sequence. Convenience over [`EriBuilder::partition`] +
    /// [`EriBuilder::fill`]; it drives the *identical* code path a parallel driver
    /// would, just serially.
    ///
    /// # Equality to [`Basis::eri`](crate::Basis::eri) and the tolerance contract
    ///
    /// The result equals `Basis::eri()` (same [`Engine`]) but is **not** bitwise
    /// identical everywhere. The OS/HGP and Rys kernels are *not* bit-symmetric
    /// under bra↔ket exchange — `(ij|kl)` and `(kl|ij)` agree only to ~1 ULP (a
    /// direct probe over a mixed s/p/d/f basis: ~69% of elements differ at the bit
    /// level, worst absolute diff ≈ 5.6e-16, worst significant-element relative diff
    /// far below 1e-11). The serial 8-fold `eri()` fills an element `(μν|λσ)` from
    /// whichever of `(ij|kl)` / `(kl|ij)` is the *lexicographically larger* shell
    /// pair (its canonical bra), whereas this 4-fold path always fills it from the
    /// element's own bra-pair `(ij|kl)`. So:
    ///
    /// - Elements whose bra-pair is `≥` (lexicographically) their ket-pair are
    ///   filled by the *same* kernel call in both paths ⇒ **bitwise identical**.
    /// - The rest differ only by the bra↔ket round-off above.
    ///
    /// **Tolerance standard (shared with the chemx driver):** compare with the
    /// repo's significant-element convention — relative residual `< 1e-11` on
    /// elements with `|ref| ≥ 1e-3 · peak`, plus an absolute floor
    /// `< 1e-11 · max(peak, 1) + 1e-12` for the rest — *not* a bit-identical
    /// assertion over the whole tensor. (The bra-pair `≥` ket-pair subset may still
    /// be asserted bit-identical, and is, in `tests/eri_builder.rs`.)
    #[must_use]
    pub fn build(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.output_len()];
        let mut tasks = self.partition(&mut out);
        for task in &mut tasks {
            self.fill(task);
        }
        out
    }

    /// Core sweep shared by [`EriBuilder::fill`] (writing into disjoint slabs) and
    /// the internal coverage/disjointness checks (writing into a counter): for the
    /// bra-pair `(i, j)`, evaluate `(ij|kl)` over every canonical ket-pair and
    /// scatter into `sink`.
    fn run_bra_pair<S: EriSink>(&self, i: usize, j: usize, sink: &mut S) {
        // One reusable block/transform buffer pair per bra-pair task (each task
        // runs on one thread), instead of fresh `Vec`s per quartet.
        let mut scratch = QuartetScratch::default();
        if self.kernel != EriKernel::Coulomb {
            // The attenuated Rys path deliberately remains scalar and keeps its
            // original canonical ket traversal unchanged.
            for &(k, l) in &self.pairs {
                self.evaluate_scalar(i, j, k, l, &mut scratch, sink);
            }
            return;
        }

        let bra_pair_index = pair_index(i, j);
        let mut batch_scratch = EriBatchScratch::default();
        for &ket_group in self.batch_schedule.batches(bra_pair_index) {
            let group = ket_group.map(|ket_pair_index| {
                let (k, l) = self.pairs[ket_pair_index as usize];
                [i, j, k, l]
            });
            evaluate_batch4(
                &mut batch_scratch,
                group,
                self.shells,
                &self.eff,
                &self.pair_data,
                &self.c2s,
                |idx, block| {
                    scatter_4fold(sink, idx, &self.offs, idx.map(|q| self.nfunc[q]), block);
                },
            );
        }
        for &ket_pair_index in self.batch_schedule.scalars(bra_pair_index) {
            let (k, l) = self.pairs[ket_pair_index as usize];
            self.evaluate_scalar(i, j, k, l, &mut scratch, sink);
        }
    }

    fn evaluate_scalar<S: EriSink>(
        &self,
        i: usize,
        j: usize,
        k: usize,
        l: usize,
        scratch: &mut QuartetScratch,
        sink: &mut S,
    ) {
        let s = self.shells;
        let quartet = [&s[i], &s[j], &s[k], &s[l]];
        let eff = [&self.eff[i][..], &self.eff[j], &self.eff[k], &self.eff[l]];
        let mats = [
            self.c2s[i].as_deref(),
            self.c2s[j].as_deref(),
            self.c2s[k].as_deref(),
            self.c2s[l].as_deref(),
        ];
        let len = match self.kernel {
            EriKernel::Coulomb => quartet_into_scratch_pairs(
                scratch,
                self.engine,
                quartet,
                eff,
                mats,
                &self.pair_data[pair_index(i, j)],
                &self.pair_data[pair_index(k, l)],
            ),
            EriKernel::Erf { omega } => {
                quartet_into_scratch_erf(scratch, quartet, eff, mats, omega)
            }
        };
        scatter_4fold(
            sink,
            [i, j, k, l],
            &self.offs,
            [self.nfunc[i], self.nfunc[j], self.nfunc[k], self.nfunc[l]],
            &scratch.block[..len],
        );
    }
}

#[inline]
fn pair_index(i: usize, j: usize) -> usize {
    debug_assert!(i >= j);
    i * (i + 1) / 2 + j
}

/// One unit of parallel work: the output rows owned by a single canonical bra-pair
/// `(i, j)`, handed out by [`EriBuilder::partition`].
///
/// Holds only this bra-pair's disjoint `&mut` row slabs (the `(i, j)` band and, when
/// `i ≠ j`, the `(j, i)` band). Distinct `BraPairFill`s borrow disjoint regions of
/// the same output buffer, so a driver may hold a `Vec<BraPairFill>` and fill them
/// across threads with no synchronisation. Fill it with [`EriBuilder::fill`].
#[derive(Debug)]
pub struct BraPairFill<'o> {
    bra: (usize, usize),
    /// Slabs for rows `(μ∈i, ν∈j)`, row-major `(a, b)` → index `a·n_j + b`.
    ij_band: Vec<&'o mut [f64]>,
    /// Slabs for rows `(μ∈j, ν∈i)`, row-major `(b, a)` → index `b·n_i + a`.
    /// Empty when `i == j`.
    ji_band: Vec<&'o mut [f64]>,
}

impl BraPairFill<'_> {
    /// The canonical bra shell-pair `(i, j)` (`i ≥ j`) this task fills.
    #[must_use]
    pub fn bra(&self) -> (usize, usize) {
        self.bra
    }
}

impl Basis {
    /// Create a parallel-ready [`EriBuilder`] for this basis (default
    /// [`Engine::Auto`] dispatch). Equivalent to [`EriBuilder::new`].
    #[must_use]
    pub fn eri_builder(&self) -> EriBuilder<'_> {
        EriBuilder::new(self)
    }
}

/// Take the slab for output `row`, asserting it has not already been claimed.
///
/// A second claim of the same row returns `None` → panic: that can only happen if
/// the bra-pair → row mapping overlaps, which would violate the disjointness
/// contract. The check is always on (cheap, `O(nao²)` total across a build).
fn claim_row<'o>(slabs: &mut [Option<&'o mut [f64]>], row: usize) -> &'o mut [f64] {
    slabs[row]
        .take()
        .expect("output row claimed by two bra-pairs (disjointness violated)")
}

/// Destination of a scattered ERI value. Abstracts *where* `(μν|λσ) = v` lands so
/// the one scatter routine ([`scatter_4fold`]) serves both the real disjoint-slab
/// write ([`BandSink`]) and the numerics-independent coverage/disjointness check
/// ([`CountSink`], in tests).
trait EriSink {
    /// Record output element `(μ, ν | λ, σ)` (global AO indices) = `v`.
    fn put(&mut self, mu: usize, nu: usize, la: usize, sg: usize, v: f64);
}

/// Writes into a bra-pair's disjoint row slabs. A `(μ, ν)` row maps to its slab by
/// band membership; within the slab the `(λ, σ)` plane is row-major (`λ·nao + σ`).
struct BandSink<'a, 'o> {
    nao: usize,
    off_i: usize,
    off_j: usize,
    n_i: usize,
    n_j: usize,
    ij: &'a mut [&'o mut [f64]],
    ji: &'a mut [&'o mut [f64]],
}

impl EriSink for BandSink<'_, '_> {
    #[inline]
    fn put(&mut self, mu: usize, nu: usize, la: usize, sg: usize, v: f64) {
        let col = la * self.nao + sg;
        // Classify the row into the (i, j) band or the (j, i) band. When i == j the
        // two shell ranges coincide, the first test always succeeds, and `ji` is
        // empty — every write lands in the single band, as `scatter_4fold` intends.
        if mu >= self.off_i
            && mu - self.off_i < self.n_i
            && nu >= self.off_j
            && nu - self.off_j < self.n_j
        {
            let a = mu - self.off_i;
            let b = nu - self.off_j;
            self.ij[a * self.n_j + b][col] = v;
        } else {
            // The only other band a scattered write can target.
            let b = mu - self.off_j;
            let a = nu - self.off_i;
            self.ji[b * self.n_i + a][col] = v;
        }
    }
}

/// Scatter a contracted `(ij|kl)` block into the **four bra/ket-internal** symmetry
/// slots — the `μ↔ν` and `λ↔σ` swaps, *not* the bra↔ket exchange.
///
/// This is the `PERMS8[..4]` half of [`crate::integrals`]'s 8-fold scatter, reusing
/// the same permutation table and the same dedup-by-permuted-shell-tuple rule, so
/// the symmetry definition is not duplicated. It differs only in the *target*: each
/// surviving permutation's destination is split into a row coordinate `(μ, ν)` and a
/// column coordinate `(λ, σ)` and handed to the [`EriSink`], rather than written at
/// one flat `nao⁴` offset — because the parallel build's output is partitioned by
/// row.
///
/// Degenerate collapses are handled by the dedup exactly as in the 8-fold scatter:
/// the ket-swap slot is dropped when `k == l`, the bra-swap when `i == j`, and the
/// double-swap when either holds — so no element is written twice.
fn scatter_4fold<S: EriSink>(
    sink: &mut S,
    sidx: [usize; 4],
    offs: &[usize],
    n: [usize; 4],
    block: &[f64],
) {
    // Dedup distinct permutations by the shell-index tuple they produce, identical
    // to `scatter_eri_block_s8`. At most four survive.
    let mut seen: [[usize; 4]; 4] = [[usize::MAX; 4]; 4];
    let mut n_seen = 0;
    for perm in &PERMS8[..4] {
        let tup = [sidx[perm[0]], sidx[perm[1]], sidx[perm[2]], sidx[perm[3]]];
        if seen[..n_seen].contains(&tup) {
            continue;
        }
        seen[n_seen] = tup;
        n_seen += 1;

        // Output-position offsets: position q is shell `sidx[perm[q]]`. Positions
        // 0,1 form the bra row (μ, ν); positions 2,3 the ket column (λ, σ).
        let o = [offs[tup[0]], offs[tup[1]], offs[tup[2]], offs[tup[3]]];
        // perm[0], perm[1] ∈ {0,1} (bra source axes a,b); perm[2], perm[3] ∈ {2,3}
        // (ket source axes c,d). Map each output coord back to its source component.
        let (m_ax, n_ax) = (perm[0], perm[1]); // ∈ {0,1}
        let (l_ax, s_ax) = (perm[2] - 2, perm[3] - 2); // ∈ {0,1}

        let mut src = 0usize;
        for a in 0..n[0] {
            for b in 0..n[1] {
                let ab = [a, b];
                let mu = o[0] + ab[m_ax];
                let nu = o[1] + ab[n_ax];
                for c in 0..n[2] {
                    for d in 0..n[3] {
                        let cd = [c, d];
                        let la = o[2] + cd[l_ax];
                        let sg = o[3] + cd[s_ax];
                        sink.put(mu, nu, la, sg, block[src]);
                        src += 1;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A counting/ownership sink over a flat `nao⁴` buffer: records which bra-pair
    /// wrote each element and *panics on any second write*. Used to verify, with no
    /// dependence on the numerical values, that the 4-fold scatter writes each
    /// output element **exactly once** (no second write) and that the per-bra-pair
    /// write sets are **mutually disjoint** (the previous owner, if any, is a
    /// different bra-pair — also caught by the same single-write panic, since the
    /// buffer is shared across all bra-pairs in the sweep).
    struct CountSink<'a> {
        nao: usize,
        owner: &'a mut [i64],
        current: i64,
    }

    impl EriSink for CountSink<'_> {
        fn put(&mut self, mu: usize, nu: usize, la: usize, sg: usize, _v: f64) {
            let idx = ((mu * self.nao + nu) * self.nao + la) * self.nao + sg;
            assert_eq!(
                self.owner[idx], -1,
                "element {idx} written twice: now by bra-pair {}, previously by {}",
                self.current, self.owner[idx]
            );
            self.owner[idx] = self.current;
        }
    }

    fn mixed_basis() -> Basis {
        // Distinct L on distinct centers so no accidental symmetry hides a bug, plus
        // a repeated-L pair to exercise i == j / k == l collapses, plus a spherical
        // shell so the c2s path is covered.
        Basis::new(vec![
            Shell::new(0, [0.0, 0.0, 0.0], vec![1.4, 0.5], vec![0.6, 0.5]).unwrap(),
            Shell::new(1, [0.6, -0.3, 0.2], vec![0.9], vec![1.0]).unwrap(),
            Shell::new(1, [0.6, -0.3, 0.2], vec![0.4], vec![1.0]).unwrap(),
            Shell::new_spherical(2, [-0.4, 0.7, -0.1], vec![1.1], vec![1.0]).unwrap(),
        ])
    }

    #[test]
    fn write_coverage_exactly_once_and_disjoint() {
        // Numerics-independent: drive the real scatter with an ownership counter and
        // assert every output element is written exactly once across the whole build
        // (items 8 & 9: exactly-once coverage + mutually-disjoint bra-pair spans).
        let basis = mixed_basis();
        let builder = EriBuilder::new(&basis);
        let n4 = builder.output_len();
        let mut owner = vec![-1i64; n4];
        for (p, &(i, j)) in builder.bra_pairs().iter().enumerate() {
            let mut sink = CountSink {
                nao: builder.nao,
                owner: &mut owner,
                current: p as i64,
            };
            builder.run_bra_pair(i, j, &mut sink);
        }
        // Full coverage: no element left unwritten (combined with the single-write
        // panic in `put`, this proves exactly-once over all nao⁴ elements).
        let unwritten = owner.iter().filter(|&&o| o == -1).count();
        assert_eq!(unwritten, 0, "{unwritten} output elements never written");
    }

    #[test]
    fn partition_claims_every_row_once() {
        // The row-level disjointness/coverage invariant: partition hands every (μ,ν)
        // row to exactly one bra-pair. `claim_row` panics on a double-claim; here we
        // assert full coverage (the debug_assert inside partition checks the same,
        // but we make it explicit and build-independent).
        let basis = mixed_basis();
        let builder = EriBuilder::new(&basis);
        let mut out = vec![0.0; builder.output_len()];
        let tasks = builder.partition(&mut out);

        let nao = builder.nao;
        let total_rows: usize = tasks
            .iter()
            .map(|t| t.ij_band.len() + t.ji_band.len())
            .sum();
        assert_eq!(total_rows, nao * nao, "bra-pairs do not cover all rows");
        assert_eq!(tasks.len(), builder.bra_pairs().len());
    }

    #[test]
    fn serial_build_matches_basis_eri_tolerance() {
        // Quick in-module value sanity (the rigorous comparison, incl. the
        // bit-identical subset, lives in tests/eri_builder.rs).
        let basis = mixed_basis();
        let reference = basis.eri();
        let built = EriBuilder::new(&basis).build();
        assert_eq!(reference.len(), built.len());
        let peak = reference.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
        let floor = 1e-3 * peak;
        let mut worst_sig = 0.0_f64;
        let mut worst_abs = 0.0_f64;
        for (&r, &b) in reference.iter().zip(&built) {
            let dv = (r - b).abs();
            worst_abs = worst_abs.max(dv);
            if r.abs() >= floor {
                worst_sig = worst_sig.max(dv / r.abs());
            }
        }
        assert!(
            worst_sig < 1e-11,
            "worst significant relative diff {worst_sig:e}"
        );
        assert!(
            worst_abs < 1e-11 * peak.max(1.0) + 1e-12,
            "worst absolute diff {worst_abs:e} (peak {peak:e})"
        );
    }
}
