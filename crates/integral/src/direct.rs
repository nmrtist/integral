//! Cached, batched direct Coulomb/exchange contraction.

use std::collections::BTreeMap;

use crate::engine::os_eri::{self, ShellPairData, ShellRef};
use crate::eri_batch::{batch_key, evaluate_batch4, BatchKey, EriBatchScratch};
use crate::integrals::{
    canonical_shell_pairs, effective_coeffs, quartet_into_scratch_pairs, Engine, QuartetScratch,
};
use crate::shell::{Basis, Shell};
use crate::spherical::shell_transform;

/// Caller-owned matrices for one density in a direct contraction.
pub struct DirectBuffers<'a> {
    pub coulomb: &'a mut [f64],
    pub exchange: Option<&'a mut [f64]>,
}

/// Reusable worker-local scratch. Construct one per Rayon fold worker and reuse
/// it across every bra shell-pair assigned to that worker.
#[derive(Default)]
pub struct DirectWorkspace {
    quartet: QuartetScratch,
    batch: EriBatchScratch,
    pending: BTreeMap<BatchKey, Vec<([usize; 4], u64)>>,
}

impl DirectWorkspace {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// An owned direct-contraction plan. Shell metadata, effective coefficients,
/// transforms, primitive-pair lists, Schwarz bounds, and dispatch inputs are
/// computed once when the plan is created.
#[derive(Debug)]
pub struct DirectContractor {
    shells: Vec<Shell>,
    engine: Engine,
    offsets: Vec<usize>,
    nfunc: Vec<usize>,
    nao: usize,
    eff: Vec<Vec<f64>>,
    c2s: Vec<Option<Vec<f64>>>,
    pairs: Vec<(usize, usize)>,
    pair_data: Vec<ShellPairData>,
    schwarz: Vec<f64>,
}

impl DirectContractor {
    #[must_use]
    pub fn new(basis: &Basis) -> Self {
        Self::with_engine(basis, Engine::Auto)
    }

    #[must_use]
    pub fn with_engine(basis: &Basis, engine: Engine) -> Self {
        let shells = basis.shells().to_vec();
        let offsets = basis.offsets();
        let nfunc = shells.iter().map(Shell::n_func).collect();
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
        Self {
            shells,
            engine,
            offsets,
            nfunc,
            nao: basis.nao(),
            eff,
            c2s,
            pairs,
            pair_data,
            schwarz: basis.schwarz_bounds(),
        }
    }

    #[must_use]
    pub fn nao(&self) -> usize {
        self.nao
    }

    #[must_use]
    pub fn bra_pairs(&self) -> &[(usize, usize)] {
        &self.pairs
    }

    /// Maximum absolute density element in every ordered shell block, once per
    /// input density. The returned bounds are accepted directly by
    /// [`Self::accumulate_bra_pair`].
    #[must_use]
    pub fn density_shell_bounds(&self, densities: &[&[f64]]) -> Vec<Vec<f64>> {
        let nsh = self.shells.len();
        densities
            .iter()
            .map(|density| {
                assert_eq!(density.len(), self.nao * self.nao);
                let mut bounds = vec![0.0; nsh * nsh];
                for i in 0..nsh {
                    for j in 0..nsh {
                        let mut max = 0.0_f64;
                        for a in 0..self.nfunc[i] {
                            let row = (self.offsets[i] + a) * self.nao + self.offsets[j];
                            for b in 0..self.nfunc[j] {
                                max = max.max(density[row + b].abs());
                            }
                        }
                        bounds[i * nsh + j] = max;
                    }
                }
                bounds
            })
            .collect()
    }

    /// Contract all canonical ket pairs for one canonical bra pair. Each
    /// accepted quartet is evaluated once and reused across every density whose
    /// density-weighted Schwarz test accepted it.
    #[allow(clippy::too_many_arguments)]
    pub fn accumulate_bra_pair(
        &self,
        workspace: &mut DirectWorkspace,
        bra: (usize, usize),
        densities: &[&[f64]],
        density_bounds: &[&[f64]],
        threshold: f64,
        want_exchange: bool,
        outputs: &mut [DirectBuffers<'_>],
    ) {
        assert!(
            !densities.is_empty(),
            "direct contraction needs at least one density"
        );
        assert!(
            densities.len() <= 64,
            "at most 64 simultaneous densities are supported"
        );
        assert_eq!(densities.len(), density_bounds.len());
        assert_eq!(densities.len(), outputs.len());
        let n2 = self.nao * self.nao;
        for ((d, b), out) in densities.iter().zip(density_bounds).zip(outputs.iter()) {
            assert_eq!(d.len(), n2);
            assert_eq!(b.len(), self.shells.len().pow(2));
            assert_eq!(out.coulomb.len(), n2);
            if want_exchange {
                assert_eq!(out.exchange.as_ref().map(|v| v.len()), Some(n2));
            }
        }

        workspace.pending.clear();
        let (i, j) = bra;
        debug_assert!(i >= j);
        let nsh = self.shells.len();
        let qab = self.schwarz[i * nsh + j];
        if qab == 0.0 {
            return;
        }
        for k in 0..=i {
            let l_top = if k == i { j } else { k };
            for l in 0..=l_top {
                let pair_bound = qab * self.schwarz[k * nsh + l];
                if pair_bound < threshold {
                    continue;
                }
                let mut mask = 0_u64;
                for (di, bounds) in density_bounds.iter().enumerate() {
                    let dbound = bounds[i * nsh + j]
                        .max(bounds[k * nsh + l])
                        .max(bounds[i * nsh + k])
                        .max(bounds[i * nsh + l])
                        .max(bounds[j * nsh + k])
                        .max(bounds[j * nsh + l]);
                    if pair_bound * dbound >= threshold {
                        mask |= 1_u64 << di;
                    }
                }
                if mask == 0 {
                    continue;
                }
                let idx = [i, j, k, l];
                let shells = idx.map(|q| &self.shells[q]);
                let key = batch_key(
                    self.engine,
                    shells,
                    [
                        self.pair_data[tri(i, j)].len(),
                        self.pair_data[tri(k, l)].len(),
                    ],
                );
                if let Some(key) = key {
                    let pending = workspace.pending.entry(key).or_default();
                    pending.push((idx, mask));
                    if pending.len() == 4 {
                        let group = [pending[0], pending[1], pending[2], pending[3]];
                        pending.clear();
                        self.consume_batch(workspace, group, densities, want_exchange, outputs);
                    }
                } else {
                    self.consume_scalar(workspace, idx, mask, densities, want_exchange, outputs);
                }
            }
        }
        let tails = std::mem::take(&mut workspace.pending);
        for (idx, mask) in tails.into_values().flatten() {
            self.consume_scalar(workspace, idx, mask, densities, want_exchange, outputs);
        }
    }

    fn consume_batch(
        &self,
        workspace: &mut DirectWorkspace,
        group: [([usize; 4], u64); 4],
        densities: &[&[f64]],
        want_exchange: bool,
        outputs: &mut [DirectBuffers<'_>],
    ) {
        let quartets = group.map(|x| x.0);
        let masks = group.map(|x| x.1);
        let mut lane = 0;
        evaluate_batch4(
            &mut workspace.batch,
            quartets,
            &self.shells,
            &self.eff,
            &self.pair_data,
            &self.c2s,
            |idx, block| {
                self.scatter_block(idx, block, masks[lane], densities, want_exchange, outputs);
                lane += 1;
            },
        );
    }

    fn consume_scalar(
        &self,
        workspace: &mut DirectWorkspace,
        idx: [usize; 4],
        mask: u64,
        densities: &[&[f64]],
        want_exchange: bool,
        outputs: &mut [DirectBuffers<'_>],
    ) {
        let [i, j, k, l] = idx;
        let len = quartet_into_scratch_pairs(
            &mut workspace.quartet,
            self.engine,
            [
                &self.shells[i],
                &self.shells[j],
                &self.shells[k],
                &self.shells[l],
            ],
            [&self.eff[i], &self.eff[j], &self.eff[k], &self.eff[l]],
            [
                self.c2s[i].as_deref(),
                self.c2s[j].as_deref(),
                self.c2s[k].as_deref(),
                self.c2s[l].as_deref(),
            ],
            &self.pair_data[tri(i, j)],
            &self.pair_data[tri(k, l)],
        );
        self.scatter_block(
            idx,
            &workspace.quartet.block[..len],
            mask,
            densities,
            want_exchange,
            outputs,
        );
    }

    fn scatter_block(
        &self,
        [si, sj, sk, sl]: [usize; 4],
        block: &[f64],
        mask: u64,
        densities: &[&[f64]],
        want_exchange: bool,
        outputs: &mut [DirectBuffers<'_>],
    ) {
        let [na, nb, nc, nd] = [si, sj, sk, sl].map(|q| self.nfunc[q]);
        let [oa, ob, oc, od] = [si, sj, sk, sl].map(|q| self.offsets[q]);
        for a in 0..na {
            let mu = oa + a;
            let b_hi = if si == sj { a + 1 } else { nb };
            for b in 0..b_hi {
                let nu = ob + b;
                for c in 0..nc {
                    let lam = oc + c;
                    let d_hi = if sk == sl { c + 1 } else { nd };
                    let base = ((a * nb + b) * nc + c) * nd;
                    for d in 0..d_hi {
                        let sig = od + d;
                        if si == sk && sj == sl && (mu, nu) < (lam, sig) {
                            continue;
                        }
                        let g = block[base + d];
                        for di in 0..densities.len() {
                            if mask & (1_u64 << di) != 0 {
                                scatter_unique(
                                    &mut outputs[di],
                                    densities[di],
                                    self.nao,
                                    g,
                                    [mu, nu, lam, sig],
                                    want_exchange,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

impl Basis {
    #[must_use]
    pub fn direct_contractor(&self) -> DirectContractor {
        DirectContractor::new(self)
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

#[inline]
fn tri(i: usize, j: usize) -> usize {
    debug_assert!(i >= j);
    i * (i + 1) / 2 + j
}

#[inline]
fn scatter_unique(
    out: &mut DirectBuffers<'_>,
    density: &[f64],
    n: usize,
    g: f64,
    [a, b, c, d]: [usize; 4],
    want_exchange: bool,
) {
    add_contraction(out, density, n, g, a, b, c, d, want_exchange);
    if a != b {
        add_contraction(out, density, n, g, b, a, c, d, want_exchange);
    }
    if c != d {
        add_contraction(out, density, n, g, a, b, d, c, want_exchange);
        if a != b {
            add_contraction(out, density, n, g, b, a, d, c, want_exchange);
        }
    }
    if (a, b) != (c, d) {
        add_contraction(out, density, n, g, c, d, a, b, want_exchange);
        if c != d {
            add_contraction(out, density, n, g, d, c, a, b, want_exchange);
        }
        if a != b {
            add_contraction(out, density, n, g, c, d, b, a, want_exchange);
            if c != d {
                add_contraction(out, density, n, g, d, c, b, a, want_exchange);
            }
        }
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn add_contraction(
    out: &mut DirectBuffers<'_>,
    density: &[f64],
    n: usize,
    g: f64,
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    want_exchange: bool,
) {
    out.coulomb[a * n + b] += g * density[c * n + d];
    if want_exchange {
        out.exchange
            .as_deref_mut()
            .expect("exchange buffer required")[a * n + c] += g * density[b * n + d];
    }
}
