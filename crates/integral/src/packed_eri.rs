//! Parallel-ready assembly of the symmetry-unique ERI tensor: [`PackedEriBuilder`].
//!
//! The dense [`EriBuilder`](crate::EriBuilder) stores all `nao⁴` elements and, to
//! keep bra-pair tasks disjoint, evaluates both `(ij|kl)` and `(kl|ij)`. This
//! builder stores each of the `nao⁴/8` symmetry-unique integrals once, and a
//! fill evaluates each canonical shell quartet (`i ≥ j`, `k ≥ l`, `ij ≥ kl`)
//! once. Screening adds one `(ij|ij)` block per canonical shell pair, evaluated
//! on the first screened fill; an unscreened build never computes them.
//!
//! ## Layout
//!
//! AO pairs `(p, q)`, `p ≥ q`, are numbered **shell-pair blocked**: canonical shell
//! pairs `(i ≥ j)` in row-major order, and inside one shell pair the function pairs
//! `(a, b)` row-major (`b ≤ a` only when `i == j`). [`PackedEriBuilder::ao_pairs`]
//! lists `(p, q)` for every pair index `t`. The output is the row-major lower
//! triangle over pair indices: `(t|u)`, `u ≤ t`, lives at `t·(t+1)/2 + u`.
//!
//! Blocking by shell pair (rather than the usual `p·(p+1)/2 + q`) is what makes
//! the build parallel without `unsafe`: every integral a bra shell pair produces
//! lands in a row owned by that shell pair, and those rows are one contiguous
//! slice of the triangle.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use crate::engine::os_eri::{self, ShellPairData, ShellRef};
use crate::eri_batch::{batch_key, evaluate_batch4, BatchKey, EriBatchScratch};
use crate::integrals::{
    canonical_shell_pairs, effective_coeffs, quartet_into_scratch_pairs, Engine, QuartetScratch,
};
use crate::shell::{Basis, Shell};
use crate::spherical::shell_transform;

/// Reusable worker-local scratch for [`PackedEriBuilder::fill`].
#[derive(Default)]
pub struct PackedEriWorkspace {
    quartet: QuartetScratch,
    batch: EriBatchScratch,
    pending: BTreeMap<BatchKey, Vec<usize>>,
}

impl PackedEriWorkspace {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// One unit of parallel work: the packed rows owned by a single canonical bra
/// shell pair, handed out by [`PackedEriBuilder::partition`].
#[derive(Debug)]
pub struct PackedBraFill<'o> {
    bra: usize,
    rows: &'o mut [f64],
}

impl PackedBraFill<'_> {
    /// Index of this task's bra shell pair in [`PackedEriBuilder::bra_pairs`].
    #[must_use]
    pub fn bra(&self) -> usize {
        self.bra
    }
}

/// An owned plan for building the packed, symmetry-unique ERI tensor.
#[derive(Debug)]
pub struct PackedEriBuilder {
    shells: Vec<Shell>,
    engine: Engine,
    nfunc: Vec<usize>,
    eff: Vec<Vec<f64>>,
    c2s: Vec<Option<Vec<f64>>>,
    pairs: Vec<(usize, usize)>,
    pair_data: Vec<ShellPairData>,
    /// Schwarz bound of each canonical shell pair, built on the first screened fill.
    schwarz: OnceLock<Vec<f64>>,
    /// First AO-pair index of each canonical shell pair, plus the total count.
    pair_start: Vec<usize>,
    ao_pairs: Vec<(usize, usize)>,
}

impl PackedEriBuilder {
    #[must_use]
    pub fn new(basis: &Basis) -> Self {
        Self::with_engine(basis, Engine::Auto)
    }

    #[must_use]
    pub fn with_engine(basis: &Basis, engine: Engine) -> Self {
        let shells = basis.shells().to_vec();
        let offsets = basis.offsets();
        let nfunc: Vec<usize> = shells.iter().map(Shell::n_func).collect();
        let eff: Vec<Vec<f64>> = shells.iter().map(effective_coeffs).collect();
        let c2s = shells.iter().map(shell_transform).collect();
        let pairs = canonical_shell_pairs(shells.len());
        let pair_data = pairs
            .iter()
            .map(|&(i, j)| {
                os_eri::shell_pair_data(
                    shell_ref(&shells[i], &eff[i]),
                    shell_ref(&shells[j], &eff[j]),
                )
            })
            .collect();
        let mut pair_start = Vec::with_capacity(pairs.len() + 1);
        let mut ao_pairs = Vec::with_capacity(basis.nao() * (basis.nao() + 1) / 2);
        for &(i, j) in &pairs {
            pair_start.push(ao_pairs.len());
            for a in 0..nfunc[i] {
                let b_end = if i == j { a + 1 } else { nfunc[j] };
                for b in 0..b_end {
                    ao_pairs.push((offsets[i] + a, offsets[j] + b));
                }
            }
        }
        pair_start.push(ao_pairs.len());
        Self {
            shells,
            engine,
            nfunc,
            eff,
            c2s,
            pairs,
            pair_data,
            schwarz: OnceLock::new(),
            pair_start,
            ao_pairs,
        }
    }

    /// The canonical bra shell pairs `(i, j)`, `i ≥ j` — the parallel grain.
    #[must_use]
    pub fn bra_pairs(&self) -> &[(usize, usize)] {
        &self.pairs
    }

    /// `(p, q)`, `p ≥ q`, of every AO-pair index, in storage order.
    #[must_use]
    pub fn ao_pairs(&self) -> &[(usize, usize)] {
        &self.ao_pairs
    }

    /// Length of the packed output, `m·(m+1)/2` with `m = nao·(nao+1)/2`.
    #[must_use]
    pub fn output_len(&self) -> usize {
        tri(self.ao_pairs.len())
    }

    /// Split a zeroed output buffer into one task per bra shell pair, aligned
    /// with [`Self::bra_pairs`]. The tasks borrow disjoint row ranges and may be
    /// filled concurrently.
    ///
    /// # Panics
    /// If `out.len() != output_len()`.
    #[must_use]
    pub fn partition<'o>(&self, out: &'o mut [f64]) -> Vec<PackedBraFill<'o>> {
        assert_eq!(
            out.len(),
            self.output_len(),
            "packed ERI output buffer must hold {} elements",
            self.output_len()
        );
        let mut rest = out;
        let mut tasks = Vec::with_capacity(self.pairs.len());
        for bra in 0..self.pairs.len() {
            let len = tri(self.pair_start[bra + 1]) - tri(self.pair_start[bra]);
            let (rows, tail) = rest.split_at_mut(len);
            rest = tail;
            tasks.push(PackedBraFill { bra, rows });
        }
        tasks
    }

    /// Fill one bra shell pair's rows. With `threshold > 0`, quartets whose
    /// Schwarz bound `Q_ij·Q_kl` is below it are skipped, leaving the zeros the
    /// caller initialized. With `threshold = 0.0` nothing is screened and every
    /// quartet is evaluated, including those whose bound underflows to zero.
    pub fn fill(
        &self,
        workspace: &mut PackedEriWorkspace,
        task: &mut PackedBraFill<'_>,
        threshold: f64,
    ) {
        let bra = task.bra;
        let (i, j) = self.pairs[bra];
        let schwarz = (threshold > 0.0).then(|| self.schwarz_bounds());
        workspace.pending.clear();
        for ket in 0..=bra {
            let (k, l) = self.pairs[ket];
            if schwarz.is_some_and(|q| q[bra] * q[ket] < threshold) {
                continue;
            }
            let key = batch_key(
                self.engine,
                [i, j, k, l].map(|q| &self.shells[q]),
                [self.pair_data[bra].len(), self.pair_data[ket].len()],
            );
            let Some(key) = key else {
                self.fill_scalar(workspace, bra, ket, task.rows);
                continue;
            };
            let pending = workspace.pending.entry(key).or_default();
            pending.push(ket);
            if pending.len() == 4 {
                let kets = [pending[0], pending[1], pending[2], pending[3]];
                pending.clear();
                let group = kets.map(|ket| {
                    let (k, l) = self.pairs[ket];
                    [i, j, k, l]
                });
                let mut lane = 0;
                evaluate_batch4(
                    &mut workspace.batch,
                    group,
                    &self.shells,
                    &self.eff,
                    &self.pair_data,
                    &self.c2s,
                    |_, block| {
                        self.store_block(bra, kets[lane], block, task.rows);
                        lane += 1;
                    },
                );
            }
        }
        let tails = std::mem::take(&mut workspace.pending);
        for &ket in tails.values().flatten() {
            self.fill_scalar(workspace, bra, ket, task.rows);
        }
        workspace.pending = tails;
    }

    /// Assemble the whole packed tensor on the current thread.
    #[must_use]
    pub fn build(&self, threshold: f64) -> Vec<f64> {
        let mut out = vec![0.0; self.output_len()];
        let mut workspace = PackedEriWorkspace::new();
        for task in &mut self.partition(&mut out) {
            self.fill(&mut workspace, task, threshold);
        }
        out
    }

    /// `Q = sqrt(max |(μν|μν)|)` per canonical shell pair, from this builder's engine.
    fn schwarz_bounds(&self) -> &[f64] {
        self.schwarz.get_or_init(|| {
            let mut scratch = QuartetScratch::default();
            self.pairs
                .iter()
                .enumerate()
                .map(|(pair, &(i, j))| {
                    let idx = [i, j, i, j];
                    quartet_into_scratch_pairs(
                        &mut scratch,
                        self.engine,
                        idx.map(|q| &self.shells[q]),
                        idx.map(|q| self.eff[q].as_slice()),
                        idx.map(|q| self.c2s[q].as_deref()),
                        &self.pair_data[pair],
                        &self.pair_data[pair],
                    );
                    let (ni, nj) = (self.nfunc[i], self.nfunc[j]);
                    let mut max = 0.0_f64;
                    for a in 0..ni {
                        for b in 0..nj {
                            max = max.max(scratch.block[((a * nj + b) * ni + a) * nj + b].abs());
                        }
                    }
                    max.sqrt()
                })
                .collect()
        })
    }

    fn fill_scalar(
        &self,
        workspace: &mut PackedEriWorkspace,
        bra: usize,
        ket: usize,
        rows: &mut [f64],
    ) {
        let (i, j) = self.pairs[bra];
        let (k, l) = self.pairs[ket];
        let idx = [i, j, k, l];
        let len = quartet_into_scratch_pairs(
            &mut workspace.quartet,
            self.engine,
            idx.map(|q| &self.shells[q]),
            idx.map(|q| self.eff[q].as_slice()),
            idx.map(|q| self.c2s[q].as_deref()),
            &self.pair_data[bra],
            &self.pair_data[ket],
        );
        self.store_block(bra, ket, &workspace.quartet.block[..len], rows);
    }

    fn store_block(&self, bra: usize, ket: usize, block: &[f64], rows: &mut [f64]) {
        let (i, j) = self.pairs[bra];
        let (k, l) = self.pairs[ket];
        let [na, nb, nc, nd] = [i, j, k, l].map(|q| self.nfunc[q]);
        let ncd = nc * nd;
        let t0 = self.pair_start[bra];
        let u0 = self.pair_start[ket];
        let base = tri(t0);
        let mut t = t0;
        for a in 0..na {
            let b_end = if i == j { a + 1 } else { nb };
            for b in 0..b_end {
                let src = &block[(a * nb + b) * ncd..][..ncd];
                let row = &mut rows[tri(t) - base..][..t + 1];
                if k != l && ket != bra {
                    row[u0..u0 + ncd].copy_from_slice(src);
                } else {
                    let mut u = u0;
                    'ket: for c in 0..nc {
                        let d_end = if k == l { c + 1 } else { nd };
                        for d in 0..d_end {
                            if u > t {
                                break 'ket;
                            }
                            row[u] = src[c * nd + d];
                            u += 1;
                        }
                    }
                }
                t += 1;
            }
        }
    }
}

impl Basis {
    /// Create a [`PackedEriBuilder`] for this basis (default [`Engine::Auto`]).
    #[must_use]
    pub fn packed_eri_builder(&self) -> PackedEriBuilder {
        PackedEriBuilder::new(self)
    }
}

#[inline]
fn shell_ref<'a>(shell: &'a Shell, eff: &'a [f64]) -> ShellRef<'a> {
    ShellRef {
        center: shell.center(),
        l: shell.l(),
        exps: shell.exponents(),
        coeffs: eff,
    }
}

/// Number of lower-triangle elements in the first `n` rows.
#[inline]
fn tri(n: usize) -> usize {
    n * (n + 1) / 2
}
