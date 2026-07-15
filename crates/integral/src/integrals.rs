//! One- and two-electron integral builders over a [`Basis`].
//!
//! Each one-electron builder loops over shell pairs, contracts primitives with
//! their effective coefficients (`d_i · N(α_i, l)`), and places the resulting
//! Cartesian block into a dense `nao × nao` matrix (row-major). Dipole returns
//! three such matrices for the `x`, `y`, `z` components about a chosen origin.
//!
//! The two-electron builder ([`Basis::eri`]) produces the dense repulsion
//! tensor by evaluating each canonical shell quartet once (engine-dispatched)
//! and scattering it to all permutation-equivalent slots; see its docs for the
//! layout.

use crate::engine::os_eri::{self, ShellRef};
use crate::engine::{os, rys};
use crate::eri_batch::{batch_key, evaluate_batch4, shell_ref, tri_idx, EriBatchQueue};

use crate::shell::{Basis, Shell};
use crate::spherical::{shell_transform, transform_block, transform_block4_into};

/// Transform a contracted Cartesian one-electron block (`na_cart × nb_cart`,
/// row-major) into the function-space block (`na_func × nb_func`) by applying
/// each shell's `c2s` transform (identity for Cartesian shells).
pub(crate) fn to_func_1e(block: Vec<f64>, sa: &Shell, sb: &Shell) -> Vec<f64> {
    let mats = [shell_transform(sa), shell_transform(sb)];
    transform_block(
        block,
        &[sa.n_cart(), sb.n_cart()],
        &[mats[0].as_deref(), mats[1].as_deref()],
    )
}

/// Transform a contracted Cartesian ERI quartet block into function space by
/// applying each of the four shells' `c2s` transforms, computed on the spot.
/// The dense `O(n⁴)` builders precompute the transforms once per shell and call
/// [`to_func_eri_cached`] instead (building a `c2s` matrix costs far more than
/// applying it).
pub(crate) fn to_func_eri(
    block: Vec<f64>,
    sa: &Shell,
    sb: &Shell,
    sc: &Shell,
    sd: &Shell,
) -> Vec<f64> {
    let mats = [
        shell_transform(sa),
        shell_transform(sb),
        shell_transform(sc),
        shell_transform(sd),
    ];
    to_func_eri_cached(
        block,
        [sa, sb, sc, sd],
        [
            mats[0].as_deref(),
            mats[1].as_deref(),
            mats[2].as_deref(),
            mats[3].as_deref(),
        ],
    )
}

/// Like [`to_func_eri`] but with each shell's transform supplied by the caller
/// (`None` = Cartesian shell, identity). Results are identical.
pub(crate) fn to_func_eri_cached(
    block: Vec<f64>,
    s: [&Shell; 4],
    mats: [Option<&[f64]>; 4],
) -> Vec<f64> {
    transform_block(
        block,
        &[s[0].n_cart(), s[1].n_cart(), s[2].n_cart(), s[3].n_cart()],
        &mats,
    )
}

/// Reusable per-quartet buffers for the dense ERI drivers: the contracted
/// Cartesian block and the c2s ping-pong partner. One instance lives across a
/// whole quartet loop (or one per bra-pair task), replacing the former
/// per-quartet block/axis `Vec` allocations (~5 per quartet, ~45k per
/// ethylene/cc-pVDZ build). Results are bit-identical to the allocating path:
/// the buffers are zero-filled over the active region before the identical
/// accumulation.
#[derive(Default)]
pub(crate) struct QuartetScratch {
    /// The contracted block (Cartesian, then transformed in place via the
    /// ping-pong with `tmp`); after [`quartet_into_scratch`] the result is
    /// `block[..len]`.
    pub(crate) block: Vec<f64>,
    tmp: Vec<f64>,
}

/// [`contract_quartet_cached`] + [`to_func_eri_cached`] into
/// [`QuartetScratch::block`] with no per-quartet allocation. Returns the
/// function-space block's logical length; the block is `scratch.block[..len]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn quartet_into_scratch(
    scratch: &mut QuartetScratch,
    engine: Engine,
    s: [&Shell; 4],
    eff: [&[f64]; 4],
    mats: [Option<&[f64]>; 4],
) -> usize {
    let dims = [s[0].n_cart(), s[1].n_cart(), s[2].n_cart(), s[3].n_cart()];
    let n_cart: usize = dims.iter().product();
    if scratch.block.len() < n_cart {
        scratch.block.resize(n_cart, 0.0);
    }
    scratch.block[..n_cart].fill(0.0);
    contract_quartet_cached_into(
        engine,
        s[0],
        eff[0],
        s[1],
        eff[1],
        s[2],
        eff[2],
        s[3],
        eff[3],
        &mut scratch.block[..n_cart],
    );
    transform_block4_into(&mut scratch.block, dims, &mats, &mut scratch.tmp)
}

/// Like [`quartet_into_scratch`] but with the quartet's bra/ket primitive-pair
/// data supplied by the caller (precomputed once per canonical shell pair by
/// the dense drivers). An OS/HGP-resolved quartet evaluates through the
/// borrowed-pairs engine entry; a Rys-resolved quartet ignores the pair data
/// (the Rys engine contracts per primitive quartet). Results are bit-identical
/// to [`quartet_into_scratch`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn quartet_into_scratch_pairs(
    scratch: &mut QuartetScratch,
    engine: Engine,
    s: [&Shell; 4],
    eff: [&[f64]; 4],
    mats: [Option<&[f64]>; 4],
    bra_pairs: &os_eri::ShellPairData,
    ket_pairs: &os_eri::ShellPairData,
) -> usize {
    let dims = [s[0].n_cart(), s[1].n_cart(), s[2].n_cart(), s[3].n_cart()];
    let n_cart: usize = dims.iter().product();
    if scratch.block.len() < n_cart {
        scratch.block.resize(n_cart, 0.0);
    }
    scratch.block[..n_cart].fill(0.0);
    let resolved = match engine {
        Engine::Auto => select_engine(
            s[0].l() + s[1].l(),
            s[2].l() + s[3].l(),
            s[0].n_prim() * s[1].n_prim() * s[2].n_prim() * s[3].n_prim(),
        ),
        forced => forced,
    };
    match resolved {
        Engine::OsHgp => os_eri::coulomb_shell_pairs_into(
            shell_ref(s[0], eff[0]),
            shell_ref(s[1], eff[1]),
            shell_ref(s[2], eff[2]),
            shell_ref(s[3], eff[3]),
            bra_pairs,
            ket_pairs,
            &mut scratch.block[..n_cart],
        ),
        // `Auto` is resolved above; treat anything else as Rys.
        _ => contract_quartet_rys(
            s[0],
            eff[0],
            s[1],
            eff[1],
            s[2],
            eff[2],
            s[3],
            eff[3],
            &mut scratch.block[..n_cart],
        ),
    }
    transform_block4_into(&mut scratch.block, dims, &mats, &mut scratch.tmp)
}

/// Class-bucketed batching state for the dense s8 drivers: canonical quartets
/// whose VRR shape `(ne, nf) ≤ (3, 3)` resolves to the OS/HGP engine are queued
/// by `(ne, nf, bra-pair count, ket-pair count)` and evaluated **four at a
/// time** through [`os_eri::coulomb_shell_batch4_into_scratch`], one quartet
/// per SIMD lane. Equal pair counts keep the four lanes' primitive loops in
/// true lockstep (no padding), and each lane's arithmetic is bit-identical to
/// the scalar path; leftover quartets (< 4 in a bucket at the end of the loop)
/// drain through the scalar path. Only *when* a quartet's block is computed
/// and scattered moves; every output slot is still written exactly once with
/// the same value, so the tensor is bit-identical to the unbatched loop.
/// Evaluate four queued quartets through the 4-lane batch kernel, then
/// transform and scatter each lane in queue order.
#[allow(clippy::too_many_arguments)]
fn flush_batch4(
    queue: &mut EriBatchQueue,
    group: [[usize; 4]; 4],
    shells: &[Shell],
    eff: &[Vec<f64>],
    pair_data: &[os_eri::ShellPairData],
    c2s: &[Option<Vec<f64>>],
    out: &mut [f64],
    nao: usize,
    offs: &[usize],
) {
    evaluate_batch4(
        &mut queue.scratch,
        group,
        shells,
        eff,
        pair_data,
        c2s,
        |sidx, block| {
            scatter_eri_block_s8(
                out,
                nao,
                sidx,
                offs,
                [0, 1, 2, 3].map(|q| shells[sidx[q]].n_func()),
                block,
            );
        },
    );
}

/// Screened primitive-pair data for every canonical shell pair (`i ≥ j`,
/// indexed by [`tri_idx`]), computed once per dense build (`O(n_shells²)` work
/// and memory — negligible next to the quartet loop). The canonical s8 loop
/// only ever forms bra/ket pairs in the `i ≥ j` orientation, which is the
/// orientation stored here (the pair data is order-sensitive). Replaces both
/// the per-quartet `build_pairs` at evaluation time and the former
/// `surviving_pair_count` duplicate exponent loop (bucket keys now read the
/// precomputed lists' lengths).
fn shell_pair_datas(shells: &[Shell], eff: &[Vec<f64>]) -> Vec<os_eri::ShellPairData> {
    let nsh = shells.len();
    let mut data = Vec::with_capacity(nsh * (nsh + 1) / 2);
    for i in 0..nsh {
        for j in 0..=i {
            data.push(os_eri::shell_pair_data(
                shell_ref(&shells[i], &eff[i]),
                shell_ref(&shells[j], &eff[j]),
            ));
        }
    }
    data
}

/// Evaluate every quartet still queued (the < 4-deep bucket tails) through the
/// immediate scalar path.
#[allow(clippy::too_many_arguments)]
fn drain_batch_queue(
    queue: &mut EriBatchQueue,
    scratch: &mut QuartetScratch,
    engine: Engine,
    shells: &[Shell],
    eff: &[Vec<f64>],
    pair_data: &[os_eri::ShellPairData],
    c2s: &[Option<Vec<f64>>],
    out: &mut [f64],
    nao: usize,
    offs: &[usize],
) {
    let buckets = std::mem::take(&mut queue.buckets);
    for sidx in buckets.into_values().flatten() {
        let [si, sj, sk, sl] = sidx;
        let len = quartet_into_scratch_pairs(
            scratch,
            engine,
            [&shells[si], &shells[sj], &shells[sk], &shells[sl]],
            [&eff[si], &eff[sj], &eff[sk], &eff[sl]],
            [
                c2s[si].as_deref(),
                c2s[sj].as_deref(),
                c2s[sk].as_deref(),
                c2s[sl].as_deref(),
            ],
            &pair_data[tri_idx(si, sj)],
            &pair_data[tri_idx(sk, sl)],
        );
        scatter_eri_block_s8(
            out,
            nao,
            sidx,
            offs,
            [0, 1, 2, 3].map(|q| shells[sidx[q]].n_func()),
            &scratch.block[..len],
        );
    }
}

/// The two-electron interaction operator an ERI family is evaluated over.
///
/// [`EriKernel::Coulomb`] is the full-range `1/r₁₂` and routes to **exactly**
/// the existing code paths — [`Basis::eri_kernel`] and friends return
/// bit-identical output to [`Basis::eri`] / [`Basis::eri_2c`] /
/// [`Basis::eri_3c_block`]. [`EriKernel::Erf`] is the **long-range**
/// (range-separated) operator `erf(ω·r₁₂)/r₁₂`, evaluated by the Rys engine
/// via the attenuated quadrature transform (`s = ω²/(ρ+ω²)`; roots/weights at
/// `s·T` with `x → s·x`, `w → √s·w` — see `crate::engine::rys::erf_coulomb_into`;
/// Gill & Adamson, CPL **261**, 105 (1996); Ahlrichs, PCCP **8**, 3072 (2006)).
///
/// Since `erf(ωr)/r ≤ 1/r` pointwise, the Coulomb Schwarz bounds
/// ([`Basis::schwarz_bounds`]) remain valid (looser) upper bounds for the
/// attenuated integrals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EriKernel {
    /// The full-range Coulomb operator `1/r₁₂`.
    Coulomb,
    /// The long-range operator `erf(ω·r₁₂)/r₁₂`; requires a finite `ω > 0`.
    Erf {
        /// Range-separation parameter `ω` (bohr⁻¹), finite and `> 0`.
        omega: f64,
    },
}

/// Validate the `Erf` range-separation parameter (the public kernel entry
/// points panic on misuse; the integral values themselves stay panic-free).
pub(crate) fn check_erf_omega(omega: f64) {
    assert!(
        omega.is_finite() && omega > 0.0,
        "EriKernel::Erf requires a finite omega > 0, got {omega}"
    );
}

/// Which two-electron engine evaluates an ERI quartet.
///
/// Correctness is **engine-transparent**: both engines compute the same Coulomb
/// integral to the documented tolerance, so [`Engine::OsHgp`] and [`Engine::Rys`]
/// can be forced (e.g. from tests/CI) to exercise both paths on the same cases.
/// [`Engine::Auto`] applies the dispatch policy ([`select_engine`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Engine {
    /// Dispatch by `(total angular momentum, contraction degree)`.
    #[default]
    Auto,
    /// Force the Obara–Saika / Head-Gordon–Pople engine (low-L, high-contraction).
    OsHgp,
    /// Force the Rys-quadrature engine (general; the high-L fallback).
    Rys,
}

/// Dispatch policy: pick OS/HGP vs Rys from a quartet's bra/ket half-momenta
/// `ne = la+lb`, `nf = lc+ld` (so `l_total = ne+nf`) and its primitive-quartet
/// `contraction_degree` (`n_prim_a·n_prim_b·n_prim_c·n_prim_d`).
///
/// HGP's per-primitive VRR is cheaper than Rys's per-primitive roots/weights and
/// its geometry-only HRR is amortised once per shell quartet, so it wins at low L
/// and/or high contraction; Rys's small-footprint 2D recurrences win as L grows
/// and the HGP VRR/HRR tables blow up.
///
/// The crossover is calibrated from an on-host benchmark with the current Rys
/// engine and the OS/HGP engine carrying **monomorphized + 4-lane-SIMD VRR
/// kernels for every `(ne, nf) ≤ 6`** (`vrr_small`/`vrr_small4` for ≤ 3,
/// `vrr_big`/`vrr_big4` for the d/f range 4..=6). Wherever a fast kernel exists
/// (`max(ne, nf) ≤ 6`) OS/HGP wins from a very low contraction: a measured d/f
/// sweep ((dd|dd) and (ff|ff), mixed contraction shapes) places the OS-vs-Rys
/// crossover **sharply at `deg 2`** (deg 1 → Rys ~1.3–1.6×, deg 2 → OS ~1.2–1.4×,
/// rising to ~5.6× by deg 16) — **shape- and class-insensitive** (bra-contracted
/// ≡ ket-contracted; (dd|dd) and (ff|ff) cross at the same deg). The threshold is
/// set one notch conservative at **`deg ≥ 3`** (≈1.7× measured margin) to absorb
/// dispatch overhead, partially-filled batch tails, and run-to-run variance.
///
/// | band                                  | OS/HGP when `contraction_degree ≥` |
/// |---------------------------------------|------------------------------------|
/// | `l_total ≤ 5`                         | 1   (always OS)                    |
/// | `l_total 6–16`, `max(ne,nf) ≤ 6`      | 3   (fast d/f kernel; deg 1–2 → Rys)|
/// | `l_total 6–16`, `max(ne,nf) ≥ 7`      | 16  (no fast kernel; g-heavy)       |
/// | `l_total ≥ 17`                        | never (Rys)                        |
///
/// The `max(ne,nf) ≤ 6` gate is exactly "a monomorphized OS/HGP kernel exists";
/// g-heavy quartets (`ne` or `nf ≥ 7`) keep the old Rys-favoring policy so they
/// never fall on the slow generic m-marching OS path. Thresholds are
/// **calibrated to this engine state**; a future change to either engine
/// re-opens the calibration.
#[must_use]
pub fn select_engine(ne: usize, nf: usize, contraction_degree: usize) -> Engine {
    let l_total = ne + nf;
    let has_fast_kernel = ne.max(nf) <= 6; // vrr_small (≤3) / vrr_big{,4} (4..=6)
    let threshold = match l_total {
        0..=5 => 1,                     // OS wins down to deg 1 (small-class VRR)
        6..=16 if has_fast_kernel => 3, // d/f with a fast SIMD kernel: OS from deg 3 (xover deg 2)
        6..=16 => 16,                   // no fast kernel (g-heavy): OS once deg clears 16
        _ => return Engine::Rys,        // l_total ≥ 17: Rys
    };
    if contraction_degree >= threshold {
        Engine::OsHgp
    } else {
        Engine::Rys
    }
}

/// Place a row-major `na × nb` block at `(row_off, col_off)` in a row-major
/// `n × n` matrix.
pub(crate) fn place_block(
    mat: &mut [f64],
    n: usize,
    row_off: usize,
    col_off: usize,
    block: &[f64],
    nb: usize,
) {
    let na = block.len() / nb;
    for i in 0..na {
        for j in 0..nb {
            mat[(row_off + i) * n + col_off + j] = block[i * nb + j];
        }
    }
}

/// Contract one shell pair into a fresh `na × nb` block using `prim_op`, which
/// accumulates `scale · ⟨a|·|b⟩` for a primitive pair into the block.
///
/// `pub(crate)` so the periodic module can reuse it for cross-cell (Bloch) one-electron
/// blocks between a shell and a lattice-translated shell.
pub(crate) fn contract_pair<F>(sa: &Shell, sb: &Shell, mut prim_op: F) -> Vec<f64>
where
    F: FnMut(os::Prim, os::Prim, f64, &mut [f64]),
{
    let mut block = vec![0.0; sa.n_cart() * sb.n_cart()];
    for pi in 0..sa.n_prim() {
        for pj in 0..sb.n_prim() {
            let scale = sa.primitive_coeff(pi) * sb.primitive_coeff(pj);
            prim_op(sa.prim(pi), sb.prim(pj), scale, &mut block);
        }
    }
    block
}

impl Basis {
    /// Overlap matrix `S_{μν} = ⟨μ|ν⟩`.
    #[must_use]
    pub fn overlap(&self) -> Vec<f64> {
        let n = self.nao();
        let offs = self.offsets();
        let mut mat = vec![0.0; n * n];
        for (si, sa) in self.shells().iter().enumerate() {
            for (sj, sb) in self.shells().iter().enumerate() {
                let block = to_func_1e(contract_pair(sa, sb, os::overlap_into), sa, sb);
                place_block(&mut mat, n, offs[si], offs[sj], &block, sb.n_func());
            }
        }
        mat
    }

    /// Kinetic-energy matrix `T_{μν} = ⟨μ| -½∇² |ν⟩`.
    #[must_use]
    pub fn kinetic(&self) -> Vec<f64> {
        let n = self.nao();
        let offs = self.offsets();
        let mut mat = vec![0.0; n * n];
        for (si, sa) in self.shells().iter().enumerate() {
            for (sj, sb) in self.shells().iter().enumerate() {
                let block = to_func_1e(contract_pair(sa, sb, os::kinetic_into), sa, sb);
                place_block(&mut mat, n, offs[si], offs[sj], &block, sb.n_func());
            }
        }
        mat
    }

    /// Nuclear-attraction matrix `V_{μν} = Σ_C ⟨μ| −Z_C/|r−C| |ν⟩` for the given
    /// point charges `charges = [(center, Z)]`.
    #[must_use]
    pub fn nuclear(&self, charges: &[([f64; 3], f64)]) -> Vec<f64> {
        let n = self.nao();
        let offs = self.offsets();
        let mut mat = vec![0.0; n * n];
        for (si, sa) in self.shells().iter().enumerate() {
            for (sj, sb) in self.shells().iter().enumerate() {
                let block = to_func_1e(
                    contract_pair(sa, sb, |a, b, scale, out| {
                        os::nuclear_into(a, b, charges, scale, out);
                    }),
                    sa,
                    sb,
                );
                place_block(&mut mat, n, offs[si], offs[sj], &block, sb.n_func());
            }
        }
        mat
    }

    /// Spin-free scalar-relativistic integral `W_{μν} = Σ_k ⟨∂_k μ| V |∂_k ν⟩` with
    /// `V = Σ_A −Z_A/|r−R_A|` over this basis's atoms (same charges/sign convention
    /// as [`Basis::nuclear`]). `nao×nao` row-major, same layout/order/normalization
    /// as [`Basis::overlap`]/[`Basis::nuclear`].
    ///
    /// A [`Basis`] carries shell centers but no nuclear charges, so the atom list
    /// is [`Basis::atoms`] (distinct shell centers, first-appearance order) with
    /// **unit charge `Z_A = 1` on every atom**. For physical nuclear charges use
    /// [`Basis::pvp_charges`], of which this is exactly
    /// `pvp_charges(&[(atom, 1.0), …])`.
    ///
    /// `W` is the spin-free part of the `p·Vp` operator (`p = −i∇`) the
    /// one-electron X2C transformation needs; it is symmetric and is **built
    /// symmetric** (`W = Wᵀ` bitwise; see [`Basis::pvp_charges`]).
    ///
    /// # Panics
    /// If any shell has `l > MAX_L − 1` (= 5): the gradient raises the shell to
    /// `l + 1`, which must stay within the engines' validated `MAX_L`.
    #[must_use]
    pub fn pvp(&self) -> Vec<f64> {
        let charges: Vec<([f64; 3], f64)> = self.atoms().into_iter().map(|c| (c, 1.0)).collect();
        self.pvp_charges(&charges)
    }

    /// Spin-free pVp matrix `W_{μν} = Σ_{k∈{x,y,z}} ⟨∂_k μ| V |∂_k ν⟩` for the
    /// given point charges `charges = [(center, Z)]` — the same charge list,
    /// units, and `−Z/|r−C|` sign convention as [`Basis::nuclear`].
    ///
    /// Each `⟨∂_k μ|V|∂_k ν⟩` is assembled at the primitive level from up to four
    /// ordinary nuclear-attraction integrals with the bra/ket angular momenta
    /// shifted by ±1 (`∂_k g = 2α·g_{l+1_k} − l_k·g_{l−1_k}`; see
    /// [`crate::engine::os::pvp_nuclear_into`]). The matrix is symmetric by
    /// construction and is built **bitwise symmetric**: only canonical shell pairs
    /// are evaluated, the diagonal shell block is symmetrized, and the lower
    /// triangle is mirrored.
    ///
    /// # Panics
    /// If any shell has `l > MAX_L − 1` (= 5), since the derivative raises the
    /// shell to `l + 1` (so AO angular momenta through `g` (`l = 4`) — and `h` —
    /// are fully supported).
    #[must_use]
    pub fn pvp_charges(&self, charges: &[([f64; 3], f64)]) -> Vec<f64> {
        let n = self.nao();
        let offs = self.offsets();
        let mut mat = vec![0.0; n * n];
        for (si, sa) in self.shells().iter().enumerate() {
            for (sj, sb) in self.shells().iter().enumerate().take(si + 1) {
                let mut block = to_func_1e(
                    contract_pair(sa, sb, |a, b, scale, out| {
                        os::pvp_nuclear_into(a, b, charges, scale, out);
                    }),
                    sa,
                    sb,
                );
                let (na, nbf) = (sa.n_func(), sb.n_func());
                if si == sj {
                    // Diagonal shell block: symmetrize (round-off-level asymmetry
                    // of one kernel evaluation) so W = Wᵀ holds bitwise.
                    for i in 0..na {
                        for j in 0..i {
                            let m = 0.5 * (block[i * nbf + j] + block[j * nbf + i]);
                            block[i * nbf + j] = m;
                            block[j * nbf + i] = m;
                        }
                    }
                    place_block(&mut mat, n, offs[si], offs[sj], &block, nbf);
                } else {
                    place_block(&mut mat, n, offs[si], offs[sj], &block, nbf);
                    // Mirror the transpose into the (sj, si) block.
                    for i in 0..na {
                        for j in 0..nbf {
                            mat[(offs[sj] + j) * n + offs[si] + i] = block[i * nbf + j];
                        }
                    }
                }
            }
        }
        mat
    }

    /// Cartesian dipole matrices `[D_x, D_y, D_z]`, `D_k = ⟨μ| (r−O)_k |ν⟩`,
    /// about the origin `o`.
    #[must_use]
    pub fn dipole(&self, o: [f64; 3]) -> [Vec<f64>; 3] {
        let n = self.nao();
        let offs = self.offsets();
        let mut dx = vec![0.0; n * n];
        let mut dy = vec![0.0; n * n];
        let mut dz = vec![0.0; n * n];
        for (si, sa) in self.shells().iter().enumerate() {
            for (sj, sb) in self.shells().iter().enumerate() {
                let (na, nb) = (sa.n_cart(), sb.n_cart());
                let (mut bx, mut by, mut bz) =
                    (vec![0.0; na * nb], vec![0.0; na * nb], vec![0.0; na * nb]);
                for pi in 0..sa.n_prim() {
                    for pj in 0..sb.n_prim() {
                        let scale = sa.primitive_coeff(pi) * sb.primitive_coeff(pj);
                        os::dipole_into(
                            sa.prim(pi),
                            sb.prim(pj),
                            o,
                            scale,
                            &mut bx,
                            &mut by,
                            &mut bz,
                        );
                    }
                }
                let bx = to_func_1e(bx, sa, sb);
                let by = to_func_1e(by, sa, sb);
                let bz = to_func_1e(bz, sa, sb);
                let nbf = sb.n_func();
                place_block(&mut dx, n, offs[si], offs[sj], &bx, nbf);
                place_block(&mut dy, n, offs[si], offs[sj], &by, nbf);
                place_block(&mut dz, n, offs[si], offs[sj], &bz, nbf);
            }
        }
        [dx, dy, dz]
    }

    /// Contracted Cartesian ERI block for the four shells `(i, j, k, l)` in
    /// chemists' notation `(ij|kl) = ∫∫ φ_i(1)φ_j(1) r₁₂⁻¹ φ_k(2)φ_l(2) d1 d2`.
    ///
    /// The returned block is **row-major over the four Cartesian component
    /// indices** `(a, b, c, d)` of shells `(i, j, k, l)`:
    ///
    /// ```text
    ///   block[((a · n_j + b) · n_k + c) · n_l + d]
    /// ```
    ///
    /// with `n_x = self.shells()[x].n_func()` (`n_cart` for a Cartesian shell,
    /// `2l+1` for a spherical one) and the Cartesian component order of
    /// `crate::math::am` (or the `crate::math::solid_harmonics::m_order` spherical
    /// order for spherical shells) — the
    /// fastest-varying index is `d`, slowest is `a`. The block length is
    /// `n_i · n_j · n_k · n_l`.
    #[must_use]
    pub fn eri_block(&self, i: usize, j: usize, k: usize, l: usize) -> Vec<f64> {
        self.eri_block_with(Engine::Auto, i, j, k, l)
    }

    /// Like [`Basis::eri_block`] but forces a specific [`Engine`] (or [`Engine::Auto`]
    /// for the dispatch policy). Both engines produce the same block to tolerance;
    /// forcing exists so tests/CI exercise each path on the same quartets.
    #[must_use]
    pub fn eri_block_with(
        &self,
        engine: Engine,
        i: usize,
        j: usize,
        k: usize,
        l: usize,
    ) -> Vec<f64> {
        let s = self.shells();
        let (sa, sb, sc, sd) = (&s[i], &s[j], &s[k], &s[l]);
        let block = contract_quartet(engine, sa, sb, sc, sd);
        // Spherical shells are transformed to their `2l+1` components; Cartesian
        // shells pass through unchanged. Block dims become per-shell `n_func`.
        to_func_eri(block, sa, sb, sc, sd)
    }

    /// Dense electron-repulsion tensor `(ij|kl)` over the whole basis, in
    /// chemists' notation. Shells declared [`crate::ShellKind::Spherical`]
    /// contribute their `2l+1` spherical components; Cartesian shells their
    /// `n_cart`.
    ///
    /// Shape `[nao, nao, nao, nao]` flattened **row-major**:
    ///
    /// ```text
    ///   eri[((i · nao + j) · nao + k) · nao + l] = (ij|kl)
    /// ```
    ///
    /// where `nao = self.nao()` and `i, j, k, l` are global AO indices (shell
    /// blocks placed at the offsets from `offsets()`). The tensor obeys the
    /// 8-fold permutational symmetry `(ij|kl) = (ji|kl) = (ij|lk) = (kl|ij) = …`.
    ///
    /// The build exploits that symmetry: only the canonical shell quartets
    /// (`i ≥ j`, `k ≥ l`, pair index `ij ≥ kl`) are evaluated, and each computed
    /// block is scattered to every distinct permutation-equivalent position. Slots
    /// related by a *shell-level* permutation are therefore bitwise-equal copies of
    /// one evaluation; within a block whose bra (or ket) shells coincide, the usual
    /// round-off-level (`~1e-16` relative) asymmetry of one kernel evaluation
    /// remains, exactly as for an unsymmetrized build.
    #[must_use]
    pub fn eri(&self) -> Vec<f64> {
        self.eri_with(Engine::Auto)
    }

    /// Like [`Basis::eri`] but forces a specific [`Engine`] (or [`Engine::Auto`]).
    /// Both engines produce the same tensor to tolerance.
    #[must_use]
    pub fn eri_with(&self, engine: Engine) -> Vec<f64> {
        let nao = self.nao();
        let offs = self.offsets();
        let shells = self.shells();
        // Effective coefficients depend only on the shell; compute once per shell
        // instead of re-running `cart_norm` (a `powf`) for all four shells of every
        // quartet. Same for the `c2s` transforms: building one runs the
        // Racah-coefficient/normalization machinery, so rebuilding them per quartet
        // dominated the spherical driver (~35% of a cc-pVDZ build).
        let eff: Vec<Vec<f64>> = shells.iter().map(effective_coeffs).collect();
        let c2s: Vec<Option<Vec<f64>>> = shells.iter().map(shell_transform).collect();
        let mut out = vec![0.0; nao * nao * nao * nao];
        // Canonical s8 loop: i ≥ j, k ≥ l, pair index (ij) ≥ (kl). Each computed
        // block is scattered to all distinct permutation-equivalent slots, so the
        // kernel runs once per *unique* quartet — ~8× fewer evaluations than the
        // full shell loop on large bases.
        let mut scratch = QuartetScratch::default();
        let mut queue = EriBatchQueue::default();
        let pair_data = shell_pair_datas(shells, &eff);
        for (si, sa) in shells.iter().enumerate() {
            for (sj, sb) in shells.iter().enumerate().take(si + 1) {
                let bra_data = &pair_data[tri_idx(si, sj)];
                for (sk, sc) in shells.iter().enumerate().take(si + 1) {
                    let l_top = if sk == si { sj } else { sk };
                    for (sl, sd) in shells.iter().enumerate().take(l_top + 1) {
                        let ket_data = &pair_data[tri_idx(sk, sl)];
                        // Queue OS-routed small-class quartets for the 4-lane
                        // batch kernel; everything else evaluates immediately.
                        if let Some(key) =
                            batch_key(engine, [sa, sb, sc, sd], [bra_data.len(), ket_data.len()])
                        {
                            let pending = queue.buckets.entry(key).or_default();
                            pending.push([si, sj, sk, sl]);
                            if pending.len() == 4 {
                                let group: [[usize; 4]; 4] =
                                    [pending[0], pending[1], pending[2], pending[3]];
                                pending.clear();
                                flush_batch4(
                                    &mut queue, group, shells, &eff, &pair_data, &c2s, &mut out,
                                    nao, &offs,
                                );
                            }
                            continue;
                        }
                        let len = quartet_into_scratch_pairs(
                            &mut scratch,
                            engine,
                            [sa, sb, sc, sd],
                            [&eff[si], &eff[sj], &eff[sk], &eff[sl]],
                            [
                                c2s[si].as_deref(),
                                c2s[sj].as_deref(),
                                c2s[sk].as_deref(),
                                c2s[sl].as_deref(),
                            ],
                            bra_data,
                            ket_data,
                        );
                        scatter_eri_block_s8(
                            &mut out,
                            nao,
                            [si, sj, sk, sl],
                            &offs,
                            [sa.n_func(), sb.n_func(), sc.n_func(), sd.n_func()],
                            &scratch.block[..len],
                        );
                    }
                }
            }
        }
        drain_batch_queue(
            &mut queue,
            &mut scratch,
            engine,
            shells,
            &eff,
            &pair_data,
            &c2s,
            &mut out,
            nao,
            &offs,
        );
        out
    }

    /// Dense two-electron tensor over the chosen [`EriKernel`] — layout, ordering,
    /// normalization, and units identical to [`Basis::eri`].
    ///
    /// - [`EriKernel::Coulomb`] routes to [`Basis::eri`] itself: the output is
    ///   **bit-identical** (the same code path).
    /// - [`EriKernel::Erf`]`{ omega }` evaluates the long-range operator
    ///   `erf(ω·r₁₂)/r₁₂`. The attenuation is implemented in the **Rys engine
    ///   only** (the quadrature transform is a few lines there, whereas the
    ///   OS/HGP recurrences would need their own attenuated path), so the
    ///   engine dispatch is bypassed and every quartet runs through Rys over
    ///   the same canonical-quartet (8-fold) loop. The Coulomb
    ///   [`Basis::schwarz_bounds`] remain valid upper bounds for screening
    ///   attenuated integrals at the call site (`erf(ωr)/r ≤ 1/r`).
    ///
    /// # Panics
    /// Panics if `k` is `Erf { omega }` with `ω ≤ 0`, NaN, or infinite.
    #[must_use]
    pub fn eri_kernel(&self, k: EriKernel) -> Vec<f64> {
        let omega = match k {
            EriKernel::Coulomb => return self.eri(),
            EriKernel::Erf { omega } => {
                check_erf_omega(omega);
                omega
            }
        };
        let nao = self.nao();
        let offs = self.offsets();
        let shells = self.shells();
        let eff: Vec<Vec<f64>> = shells.iter().map(effective_coeffs).collect();
        let c2s: Vec<Option<Vec<f64>>> = shells.iter().map(shell_transform).collect();
        let mut out = vec![0.0; nao * nao * nao * nao];
        let mut scratch = QuartetScratch::default();
        // Same canonical s8 loop as `eri_with` (i ≥ j, k ≥ l, pair ij ≥ kl, each
        // block scattered to all permutation-equivalent slots); Rys-erf only, so
        // no OS/HGP batch queue or shell-pair data.
        for (si, sa) in shells.iter().enumerate() {
            for (sj, sb) in shells.iter().enumerate().take(si + 1) {
                for (sk, sc) in shells.iter().enumerate().take(si + 1) {
                    let l_top = if sk == si { sj } else { sk };
                    for (sl, sd) in shells.iter().enumerate().take(l_top + 1) {
                        let len = quartet_into_scratch_erf(
                            &mut scratch,
                            [sa, sb, sc, sd],
                            [&eff[si], &eff[sj], &eff[sk], &eff[sl]],
                            [
                                c2s[si].as_deref(),
                                c2s[sj].as_deref(),
                                c2s[sk].as_deref(),
                                c2s[sl].as_deref(),
                            ],
                            omega,
                        );
                        scatter_eri_block_s8(
                            &mut out,
                            nao,
                            [si, sj, sk, sl],
                            &offs,
                            [sa.n_func(), sb.n_func(), sc.n_func(), sd.n_func()],
                            &scratch.block[..len],
                        );
                    }
                }
            }
        }
        out
    }

    /// Cauchy–Schwarz shell-pair bound matrix `Q` (Häser–Ahlrichs 1989), row-major
    /// `n_shells × n_shells`:
    ///
    /// ```text
    ///   Q[i, j] = sqrt( max_{μ∈i, ν∈j} (μν|μν) ).
    /// ```
    ///
    /// Each diagonal self-repulsion `(μν|μν) ≥ 0` is read from the `(ij|ij)` shell
    /// block, so `Q` bounds every ERI by `|(μν|λσ)| ≤ Q[i,j]·Q[k,l]` for `μν` in
    /// shell pair `(i,j)` and `λσ` in `(k,l)`. Kind-aware: spherical shells use
    /// their `2l+1` components, so `Q` bounds the spherical integrals directly.
    #[must_use]
    pub fn schwarz_bounds(&self) -> Vec<f64> {
        self.schwarz_bounds_with(Engine::Auto)
    }

    /// Like [`Basis::schwarz_bounds`] but with a forced [`Engine`] (the diagonal
    /// blocks are evaluated with it). The bound is engine-independent to tolerance.
    #[must_use]
    pub fn schwarz_bounds_with(&self, engine: Engine) -> Vec<f64> {
        let shells = self.shells();
        let nsh = shells.len();
        let mut q = vec![0.0; nsh * nsh];
        for i in 0..nsh {
            for j in 0..nsh {
                let (ni, nj) = (shells[i].n_func(), shells[j].n_func());
                let block = self.eri_block_with(engine, i, j, i, j);
                let mut mx = 0.0_f64;
                for mu in 0..ni {
                    for nu in 0..nj {
                        // Diagonal element (μν|μν) of the (ij|ij) block.
                        let idx = ((mu * nj + nu) * ni + mu) * nj + nu;
                        mx = mx.max(block[idx].abs());
                    }
                }
                q[i * nsh + j] = mx.sqrt();
            }
        }
        q
    }

    /// Schwarz-screened dense ERI tensor: identical to [`Basis::eri`] except a
    /// shell quartet `(ij|kl)` is **skipped** (left zero) when its Cauchy–Schwarz
    /// bound `Q[i,j]·Q[k,l] < τ` (`tau`). Because every element of a skipped block
    /// satisfies `|(μν|λσ)| ≤ Q[i,j]·Q[k,l] < τ`, screening introduces **no error
    /// above `τ`**. Returns the tensor and [`ScreeningStats`].
    ///
    /// `tau` is the documented screening threshold; smaller `τ` retains more
    /// quartets (more accurate, slower). A typical production value is `1e-10`–
    /// `1e-12`.
    #[must_use]
    pub fn eri_screened(&self, tau: f64) -> (Vec<f64>, ScreeningStats) {
        self.eri_screened_with(Engine::Auto, tau)
    }

    /// Like [`Basis::eri_screened`] but with a forced [`Engine`].
    #[must_use]
    pub fn eri_screened_with(&self, engine: Engine, tau: f64) -> (Vec<f64>, ScreeningStats) {
        let nao = self.nao();
        let offs = self.offsets();
        let shells = self.shells();
        let nsh = shells.len();
        let q = self.schwarz_bounds_with(engine);
        let eff: Vec<Vec<f64>> = shells.iter().map(effective_coeffs).collect();
        let c2s: Vec<Option<Vec<f64>>> = shells.iter().map(shell_transform).collect();
        let mut out = vec![0.0; nao * nao * nao * nao];
        let mut total = 0_usize;
        let mut skipped = 0_usize;
        let mut scratch = QuartetScratch::default();
        let mut queue = EriBatchQueue::default();
        let pair_data = shell_pair_datas(shells, &eff);
        // Same canonical s8 loop as `eri_with` (including the 4-lane batch
        // queue); a skipped canonical quartet leaves
        // all 8 permutation-equivalent slots zero (the Schwarz bound is
        // permutation-invariant, so the guarantee `|(μν|λσ)| < τ` covers them all).
        for si in 0..nsh {
            for sj in 0..=si {
                let qij = q[si * nsh + sj];
                let bra_data = &pair_data[tri_idx(si, sj)];
                for sk in 0..=si {
                    let l_top = if sk == si { sj } else { sk };
                    for sl in 0..=l_top {
                        total += 1;
                        if qij * q[sk * nsh + sl] < tau {
                            skipped += 1;
                            continue;
                        }
                        let ket_data = &pair_data[tri_idx(sk, sl)];
                        if let Some(key) = batch_key(
                            engine,
                            [&shells[si], &shells[sj], &shells[sk], &shells[sl]],
                            [bra_data.len(), ket_data.len()],
                        ) {
                            let pending = queue.buckets.entry(key).or_default();
                            pending.push([si, sj, sk, sl]);
                            if pending.len() == 4 {
                                let group: [[usize; 4]; 4] =
                                    [pending[0], pending[1], pending[2], pending[3]];
                                pending.clear();
                                flush_batch4(
                                    &mut queue, group, shells, &eff, &pair_data, &c2s, &mut out,
                                    nao, &offs,
                                );
                            }
                            continue;
                        }
                        let len = quartet_into_scratch_pairs(
                            &mut scratch,
                            engine,
                            [&shells[si], &shells[sj], &shells[sk], &shells[sl]],
                            [&eff[si], &eff[sj], &eff[sk], &eff[sl]],
                            [
                                c2s[si].as_deref(),
                                c2s[sj].as_deref(),
                                c2s[sk].as_deref(),
                                c2s[sl].as_deref(),
                            ],
                            bra_data,
                            ket_data,
                        );
                        scatter_eri_block_s8(
                            &mut out,
                            nao,
                            [si, sj, sk, sl],
                            &offs,
                            [
                                shells[si].n_func(),
                                shells[sj].n_func(),
                                shells[sk].n_func(),
                                shells[sl].n_func(),
                            ],
                            &scratch.block[..len],
                        );
                    }
                }
            }
        }
        drain_batch_queue(
            &mut queue,
            &mut scratch,
            engine,
            shells,
            &eff,
            &pair_data,
            &c2s,
            &mut out,
            nao,
            &offs,
        );
        (
            out,
            ScreeningStats {
                shell_quartets_total: total,
                shell_quartets_skipped: skipped,
                tau,
            },
        )
    }
}

/// Outcome of a Schwarz-screened ERI build ([`Basis::eri_screened`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreeningStats {
    /// Total *canonical* shell quartets considered (`i ≥ j`, `k ≥ l`, `ij ≥ kl`
    /// — the unique quartets under the 8-fold permutational symmetry).
    pub shell_quartets_total: usize,
    /// Canonical shell quartets skipped by the Schwarz bound.
    pub shell_quartets_skipped: usize,
    /// The screening threshold `τ` used.
    pub tau: f64,
}

impl ScreeningStats {
    /// Fraction of shell quartets skipped, in `[0, 1]`.
    #[must_use]
    pub fn skipped_fraction(&self) -> f64 {
        if self.shell_quartets_total == 0 {
            0.0
        } else {
            self.shell_quartets_skipped as f64 / self.shell_quartets_total as f64
        }
    }
}

/// Contract one shell quartet into a fresh `na·nb·nc·nd` block (row-major over
/// the four Cartesian component indices), dispatching to the requested engine.
/// [`Engine::Auto`] resolves via [`select_engine`]. Computes each shell's effective
/// coefficients on the spot; the dense `O(nao⁴)` builders precompute them once per
/// shell and call [`contract_quartet_cached`] instead.
fn contract_quartet(engine: Engine, sa: &Shell, sb: &Shell, sc: &Shell, sd: &Shell) -> Vec<f64> {
    contract_quartet_cached(
        engine,
        sa,
        &effective_coeffs(sa),
        sb,
        &effective_coeffs(sb),
        sc,
        &effective_coeffs(sc),
        sd,
        &effective_coeffs(sd),
    )
}

/// Like [`contract_quartet`] but with each shell's effective coefficients supplied
/// by the caller (e.g. precomputed once per shell across a dense build), avoiding
/// the per-quartet `cart_norm`/allocation. Results are identical.
#[allow(clippy::too_many_arguments)]
pub(crate) fn contract_quartet_cached(
    engine: Engine,
    sa: &Shell,
    ea: &[f64],
    sb: &Shell,
    eb: &[f64],
    sc: &Shell,
    ec: &[f64],
    sd: &Shell,
    ed: &[f64],
) -> Vec<f64> {
    let mut block = vec![0.0; sa.n_cart() * sb.n_cart() * sc.n_cart() * sd.n_cart()];
    contract_quartet_cached_into(engine, sa, ea, sb, eb, sc, ec, sd, ed, &mut block);
    block
}

/// [`contract_quartet_cached`] into a caller-provided (zeroed) block — the
/// allocation-free core both the wrapper above and the dense drivers'
/// [`quartet_into_scratch`] share.
#[allow(clippy::too_many_arguments)]
fn contract_quartet_cached_into(
    engine: Engine,
    sa: &Shell,
    ea: &[f64],
    sb: &Shell,
    eb: &[f64],
    sc: &Shell,
    ec: &[f64],
    sd: &Shell,
    ed: &[f64],
    block: &mut [f64],
) {
    let resolved = match engine {
        Engine::Auto => select_engine(
            sa.l() + sb.l(),
            sc.l() + sd.l(),
            sa.n_prim() * sb.n_prim() * sc.n_prim() * sd.n_prim(),
        ),
        forced => forced,
    };
    match resolved {
        Engine::OsHgp => contract_quartet_oshgp(sa, ea, sb, eb, sc, ec, sd, ed, block),
        // `Auto` is resolved above; treat anything else as Rys.
        _ => contract_quartet_rys(sa, ea, sb, eb, sc, ec, sd, ed, block),
    }
}

/// Effective contraction coefficients `d_i · N(α_i, l)` of a shell, in primitive
/// order — what both engines multiply into the contracted block.
pub(crate) fn effective_coeffs(s: &Shell) -> Vec<f64> {
    (0..s.n_prim()).map(|i| s.primitive_coeff(i)).collect()
}

/// Rys path: accumulate the Coulomb engine over every primitive quartet, using the
/// caller-supplied effective coefficients (`e* [p] = d_p · N(α_p, l)`).
#[allow(clippy::too_many_arguments)]
fn contract_quartet_rys(
    sa: &Shell,
    ea: &[f64],
    sb: &Shell,
    eb: &[f64],
    sc: &Shell,
    ec: &[f64],
    sd: &Shell,
    ed: &[f64],
    block: &mut [f64],
) {
    for (pa, &ca) in ea.iter().enumerate() {
        for (pb, &cb) in eb.iter().enumerate() {
            for (pc, &cc) in ec.iter().enumerate() {
                for (pd, &cd) in ed.iter().enumerate() {
                    let scale = ca * cb * cc * cd;
                    rys::coulomb_into(
                        sa.prim(pa),
                        sb.prim(pb),
                        sc.prim(pc),
                        sd.prim(pd),
                        scale,
                        block,
                    );
                }
            }
        }
    }
}

/// Long-range (erf-attenuated) Rys path: like [`contract_quartet_rys`] but over
/// the kernel `erf(ω·r₁₂)/r₁₂`. The attenuation is Rys-only (see
/// [`Basis::eri_kernel`]), so there is no OS/HGP counterpart.
#[allow(clippy::too_many_arguments)]
fn contract_quartet_rys_erf(
    sa: &Shell,
    ea: &[f64],
    sb: &Shell,
    eb: &[f64],
    sc: &Shell,
    ec: &[f64],
    sd: &Shell,
    ed: &[f64],
    omega: f64,
    block: &mut [f64],
) {
    for (pa, &ca) in ea.iter().enumerate() {
        for (pb, &cb) in eb.iter().enumerate() {
            for (pc, &cc) in ec.iter().enumerate() {
                for (pd, &cd) in ed.iter().enumerate() {
                    let scale = ca * cb * cc * cd;
                    rys::erf_coulomb_into(
                        sa.prim(pa),
                        sb.prim(pb),
                        sc.prim(pc),
                        sd.prim(pd),
                        omega,
                        scale,
                        block,
                    );
                }
            }
        }
    }
}

/// [`quartet_into_scratch`] for the long-range kernel: contract one shell
/// quartet of `erf(ω·r₁₂)/r₁₂` integrals (Rys engine) into
/// [`QuartetScratch::block`] and transform to function space. Returns the
/// function-space block's logical length.
pub(crate) fn quartet_into_scratch_erf(
    scratch: &mut QuartetScratch,
    s: [&Shell; 4],
    eff: [&[f64]; 4],
    mats: [Option<&[f64]>; 4],
    omega: f64,
) -> usize {
    let dims = [s[0].n_cart(), s[1].n_cart(), s[2].n_cart(), s[3].n_cart()];
    let n_cart: usize = dims.iter().product();
    if scratch.block.len() < n_cart {
        scratch.block.resize(n_cart, 0.0);
    }
    scratch.block[..n_cart].fill(0.0);
    contract_quartet_rys_erf(
        s[0],
        eff[0],
        s[1],
        eff[1],
        s[2],
        eff[2],
        s[3],
        eff[3],
        omega,
        &mut scratch.block[..n_cart],
    );
    transform_block4_into(&mut scratch.block, dims, &mats, &mut scratch.tmp)
}

/// OS/HGP path: one early-contraction call over the whole shell quartet, with the
/// caller-supplied effective coefficients.
#[allow(clippy::too_many_arguments)]
fn contract_quartet_oshgp(
    sa: &Shell,
    ea: &[f64],
    sb: &Shell,
    eb: &[f64],
    sc: &Shell,
    ec: &[f64],
    sd: &Shell,
    ed: &[f64],
    block: &mut [f64],
) {
    os_eri::coulomb_shell_into(
        ShellRef {
            center: sa.center(),
            l: sa.l(),
            exps: sa.exponents(),
            coeffs: ea,
        },
        ShellRef {
            center: sb.center(),
            l: sb.l(),
            exps: sb.exponents(),
            coeffs: eb,
        },
        ShellRef {
            center: sc.center(),
            l: sc.l(),
            exps: sc.exponents(),
            coeffs: ec,
        },
        ShellRef {
            center: sd.center(),
            l: sd.l(),
            exps: sd.exponents(),
            coeffs: ed,
        },
        block,
    );
}

/// The 8 index permutations of chemists' `(ij|kl)`: bra swap, ket swap, and
/// bra↔ket exchange. `PERMS8[p][q]` is the source axis (`0..4` ⇒ `a,b,c,d`)
/// that lands at output position `q`.
///
/// The single source of truth for the ERI permutational symmetry. The first
/// four rows (`PERMS8[..4]`) are exactly the **bra/ket-internal** permutations
/// (no bra↔ket exchange); the parallel [`crate::EriBuilder`] reuses that half to
/// scatter a canonical bra-pair's writes (see `eri_builder::scatter_4fold`).
pub(crate) const PERMS8: [[usize; 4]; 8] = [
    [0, 1, 2, 3], // (ij|kl)
    [1, 0, 2, 3], // (ji|kl)
    [0, 1, 3, 2], // (ij|lk)
    [1, 0, 3, 2], // (ji|lk)
    [2, 3, 0, 1], // (kl|ij)
    [2, 3, 1, 0], // (lk|ij)
    [3, 2, 0, 1], // (kl|ji)
    [3, 2, 1, 0], // (lk|ji)
];

/// The canonical shell pairs `(i, j)` with `i ≥ j` — the unique pairs under the
/// bra (and ket) index-swap symmetry, in row-major order (`i` outer, `j` inner).
///
/// This is the same `i ≥ j` enumeration the dense [`Basis::eri`] driver runs in
/// its outer two loops; the parallel [`crate::EriBuilder`] reuses it for both the
/// bra grain and the ket sweep so the canonical-pair definition lives in one place.
pub(crate) fn canonical_shell_pairs(nsh: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::with_capacity(nsh * (nsh + 1) / 2);
    for i in 0..nsh {
        for j in 0..=i {
            pairs.push((i, j));
        }
    }
    pairs
}

/// Scatter one computed canonical quartet block into every *distinct*
/// permutation-equivalent position of the dense `nao⁴` row-major tensor.
///
/// `sidx` are the four shell indices, `offs` the per-shell AO offsets, and
/// `n = [na, nb, nc, nd]` the block's component dims (row-major over `a,b,c,d`).
/// Two permutations write the same slot set iff they map the shell-index tuple
/// to the same tuple, so deduplicating on the permuted tuple makes every output
/// slot written exactly once across a canonical build (identity first, so a
/// quartet's own slots always carry its directly computed values).
fn scatter_eri_block_s8(
    out: &mut [f64],
    nao: usize,
    sidx: [usize; 4],
    offs: &[usize],
    n: [usize; 4],
    block: &[f64],
) {
    let mut seen: [[usize; 4]; 8] = [[usize::MAX; 4]; 8];
    let mut n_seen = 0;
    for perm in &PERMS8 {
        let tup = [sidx[perm[0]], sidx[perm[1]], sidx[perm[2]], sidx[perm[3]]];
        if seen[..n_seen].contains(&tup) {
            continue;
        }
        seen[n_seen] = tup;
        n_seen += 1;
        // Output strides/base per *source* axis: source axis `perm[q]` lands at
        // output position `q`, whose stride is `nao^(3−q)`.
        let mut stride = [0usize; 4];
        let mut base = 0usize;
        for (q, &src_axis) in perm.iter().enumerate() {
            let s = nao.pow(3 - q as u32);
            stride[src_axis] = s;
            base += offs[sidx[src_axis]] * s;
        }
        let mut src = 0usize;
        for a in 0..n[0] {
            let oa = base + a * stride[0];
            for b in 0..n[1] {
                let ob = oa + b * stride[1];
                for c in 0..n[2] {
                    let oc = ob + c * stride[2];
                    for d in 0..n[3] {
                        out[oc + d * stride[3]] = block[src];
                        src += 1;
                    }
                }
            }
        }
    }
}
