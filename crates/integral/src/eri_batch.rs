//! Shared four-lane shell-quartet batching used by dense and direct drivers.

use std::collections::BTreeMap;

use crate::engine::os_eri::{self, ShellPairData, ShellRef};
use crate::integrals::{select_engine, Engine};
use crate::shell::Shell;
use crate::spherical::transform_block4_into;

pub(crate) type BatchKey = (usize, usize, usize, usize);

/// Reusable queue and scratch for a shell-quartet sweep. A caller owns one of
/// these per independent task/worker, so no synchronization is required.
#[derive(Default)]
pub(crate) struct EriBatchQueue {
    pub(crate) buckets: BTreeMap<BatchKey, Vec<[usize; 4]>>,
    pub(crate) scratch: EriBatchScratch,
}

/// Reusable four-lane evaluation buffers, separate from the dynamic pending
/// queue. Pre-scheduled callers such as `EriBuilder` need only this state and
/// therefore never construct a map or quartet queue in their hot path.
#[derive(Default)]
pub(crate) struct EriBatchScratch {
    blocks: [Vec<f64>; 4],
    transform_tmp: Vec<f64>,
    core: os_eri::EriBatch4Scratch,
}

#[inline]
pub(crate) fn shell_ref<'a>(s: &'a Shell, eff: &'a [f64]) -> ShellRef<'a> {
    ShellRef {
        center: s.center(),
        l: s.l(),
        exps: s.exponents(),
        coeffs: eff,
    }
}

/// Return the lockstep bucket for an OS/HGP quartet. The angular-momentum
/// shape and both surviving primitive-pair counts are part of the key.
pub(crate) fn batch_key(
    engine: Engine,
    s: [&Shell; 4],
    pair_counts: [usize; 2],
) -> Option<BatchKey> {
    let resolved = match engine {
        Engine::Auto => select_engine(
            s[0].l() + s[1].l(),
            s[2].l() + s[3].l(),
            s[0].n_prim() * s[1].n_prim() * s[2].n_prim() * s[3].n_prim(),
        ),
        forced => forced,
    };
    if resolved != Engine::OsHgp {
        return None;
    }
    let ne = s[0].l() + s[1].l();
    let nf = s[2].l() + s[3].l();
    (ne <= 6 && nf <= 6).then_some((ne, nf, pair_counts[0], pair_counts[1]))
}

/// Evaluate and transform four compatible quartets. `consume` is invoked in
/// lane order with each function-space block, allowing dense and direct callers
/// to retain their own scatter orientation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_batch4(
    scratch: &mut EriBatchScratch,
    group: [[usize; 4]; 4],
    shells: &[Shell],
    eff: &[Vec<f64>],
    pair_data: &[ShellPairData],
    c2s: &[Option<Vec<f64>>],
    mut consume: impl FnMut([usize; 4], &[f64]),
) {
    let quartets: [[ShellRef<'_>; 4]; 4] = group.map(|idx| {
        [
            shell_ref(&shells[idx[0]], &eff[idx[0]]),
            shell_ref(&shells[idx[1]], &eff[idx[1]]),
            shell_ref(&shells[idx[2]], &eff[idx[2]]),
            shell_ref(&shells[idx[3]], &eff[idx[3]]),
        ]
    });
    let dims: [[usize; 4]; 4] = group.map(|idx| [0, 1, 2, 3].map(|q| shells[idx[q]].n_cart()));
    for (lane, d) in dims.iter().enumerate() {
        let n = d.iter().product();
        if scratch.blocks[lane].len() < n {
            scratch.blocks[lane].resize(n, 0.0);
        }
        scratch.blocks[lane][..n].fill(0.0);
    }
    {
        let [b0, b1, b2, b3] = &mut scratch.blocks;
        let mut outs: [&mut [f64]; 4] = [
            &mut b0[..dims[0].iter().product()],
            &mut b1[..dims[1].iter().product()],
            &mut b2[..dims[2].iter().product()],
            &mut b3[..dims[3].iter().product()],
        ];
        os_eri::coulomb_shell_batch4_pairs_into_scratch(
            &mut scratch.core,
            &quartets,
            group.map(|idx| &pair_data[tri_idx(idx[0], idx[1])]),
            group.map(|idx| &pair_data[tri_idx(idx[2], idx[3])]),
            &mut outs,
        );
    }
    for (lane, idx) in group.into_iter().enumerate() {
        let mats = [
            c2s[idx[0]].as_deref(),
            c2s[idx[1]].as_deref(),
            c2s[idx[2]].as_deref(),
            c2s[idx[3]].as_deref(),
        ];
        let len = transform_block4_into(
            &mut scratch.blocks[lane],
            dims[lane],
            &mats,
            &mut scratch.transform_tmp,
        );
        consume(idx, &scratch.blocks[lane][..len]);
    }
}

#[inline]
pub(crate) fn tri_idx(i: usize, j: usize) -> usize {
    debug_assert!(i >= j);
    i * (i + 1) / 2 + j
}
