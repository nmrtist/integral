//! Two-electron repulsion integrals (ERIs) by the Obara–Saika / Head-Gordon–Pople
//! (OS/HGP) recurrence — the second ERI engine.
//!
//! A **vertical recurrence (VRR)** builds the
//! intermediate `[e0|f0]^(m)` classes from `[00|00]^(m)` per primitive quartet,
//! the primitives are **contracted** into AO space, and a **horizontal recurrence
//! (HRR)** then shifts angular momentum `A→B` and `C→D` *in the contracted space*.
//! Doing the HRR after contraction is the HGP "early contraction" trick: the
//! geometry-only HRR runs **once per shell quartet** instead of once per
//! primitive quartet, which is the win at high contraction degree.
//!
//! It is the engineering counterpart of the [`crate::engine::rys`] engine: same Coulomb
//! kernel, same row-major `(a,b,c,d)` block layout, validated element-for-element
//! against it (and against an independent McMurchie–Davidson path).
//!
//! ## Method
//!
//! For a primitive quartet with exponents `α,β` on the bra centres `A,B` and
//! `γ,δ` on the ket centres `C,D`, with `p=α+β`, `q=γ+δ`, `P,Q` the Gaussian
//! product centres, `W=(pP+qQ)/(p+q)`, `ρ=pq/(p+q)`, and `T=ρ|P−Q|²`:
//!
//! ```text
//!   [00|00]^(m) = 2π^{5/2}/(p q √(p+q)) · K_ab · K_cd · F_m(T)
//! ```
//!
//! The VRR raises angular momentum on `A` (the bra, index `e`) and `C` (the ket,
//! index `f`) using the standard OS ERI relations (Obara–Saika 1986; HGP 1988):
//!
//! ```text
//!   [e+1_i,0|f0]^(m) = (P−A)_i[e0|f0]^(m) + (W−P)_i[e0|f0]^(m+1)
//!       + e_i/2p ( [e−1_i,0|f0]^(m) − (q/(p+q))[e−1_i,0|f0]^(m+1) )
//!       + f_i/2(p+q) [e0|f−1_i,0]^(m+1)
//!   [e0|f+1_i,0]^(m) = (Q−C)_i[e0|f0]^(m) + (W−Q)_i[e0|f0]^(m+1)
//!       + f_i/2q ( [e0|f−1_i,0]^(m) − (p/(p+q))[e0|f−1_i,0]^(m+1) )
//!       + e_i/2(p+q) [e−1_i,0|f0]^(m+1)
//! ```
//!
//! After contracting `[e0|f0]^(0)` over the primitive quartet, the HRR builds the
//! Cartesian shell block:
//!
//! ```text
//!   (a,b+1_i | f0) = (a+1_i,b | f0) + (A−B)_i (a,b | f0)   [bra, A→B]
//!   (ab | c,d+1_i) = (ab | c+1_i,d) + (C−D)_i (ab | c,d)   [ket, C→D]
//! ```
//!
//! ## Buffers
//!
//! The VRR table `[e0|f0]^(m)` is **3D and `W`-coupled** (the recurrence mixes
//! Cartesian axes), so unlike the axis-separable 1D tables of the one-electron and
//! Rys engines it cannot be a small fixed stack array — a `MAX_L` stack buffer
//! would be tens of MB, and even a heap copy of the *full* table is ~41 MB at
//! `(ii|ii)`. The current engine removes that:
//!
//! - **m-marching VRR** (`vrr_primitive`): the VRR never materialises the full
//!   `e×f×m` table. It marches over the ket `f`-degree keeping only a rolling window
//!   of **3 consecutive `f`-degree levels**, each **triangle-packed** in the Boys
//!   index `m`. Resident VRR footprint is `3·max_k[n_cart(k)·slab_k]` (≈ 4.3 MB at
//!   `(ii|ii)`, vs the old 41 MB single table), plus the `n_e·n_f` contracted table.
//! - **flat-array HRR** (`hrr_and_scatter`): contiguous arrays indexed by `addr`
//!   / [`cart_index`] with two rolling degree layers, replacing the former HashMap
//!   memoisation.
//! - **reusable arena** ([`EriScratch`]): all buffers are allocated once and reused
//!   across quartets (thread-local by default), not re-allocated per quartet.
//!
//! All of this stays in safe Rust (`#![forbid(unsafe_code)]`) and reproduces the
//! former full-table engine's values, cross-checked against the Rys engine and an
//! independent McMurchie–Davidson path (`tests/eri_cross_algorithm.rs`).

use std::cell::RefCell;

use crate::math::am::{cart_components, cart_index, n_cart};
use crate::math::boys::{boys_array, boys_array2, boys_array4};

use crate::engine::os::{Vec3, MAX_L};

/// A contracted Cartesian shell as seen by the HGP engine: its centre, angular
/// momentum, and primitive `(exponent, effective-coefficient)` data. The
/// coefficients are the driver's *effective* coefficients `d_i · N(α_i, l)` — the
/// engine itself works on un-normalised monomials and only multiplies the four
/// coefficients into the contracted accumulator.
#[derive(Debug, Clone, Copy)]
pub struct ShellRef<'a> {
    /// Shell centre (bohr).
    pub center: Vec3,
    /// Angular momentum.
    pub l: usize,
    /// Primitive exponents.
    pub exps: &'a [f64],
    /// Effective contraction coefficients, aligned with `exps`.
    pub coeffs: &'a [f64],
}

/// Number of Cartesian triples of degree `0..=lmax`: `(L+1)(L+2)(L+3)/6`.
#[inline]
fn n_addr(lmax: usize) -> usize {
    (lmax + 1) * (lmax + 2) * (lmax + 3) / 6
}

/// Number of Cartesian triples of degree **strictly less than** `d`:
/// `d(d+1)(d+2)/6`. This is the cumulative base of degree `d` in `addr`
/// (`addr(t) = tri_below(deg) + cart_index(t)`) and `n_addr(L) = tri_below(L+1)`.
#[inline]
fn tri_below(d: usize) -> usize {
    d * (d + 1) * (d + 2) / 6
}

/// Global address of a Cartesian triple within the cumulative `0..=` ordering,
/// consistent with [`crate::math::am`]'s per-degree canonical order. Bijective onto
/// `0..n_addr(deg)`.
#[inline]
fn addr(t: [usize; 3]) -> usize {
    let n = t[0] + t[1] + t[2];
    let base = n * (n + 1) * (n + 2) / 6;
    // within-degree index: (n-lx)(n-lx+1)/2 + lz
    let within = (n - t[0]) * (n - t[0] + 1) / 2 + t[2];
    base + within
}

/// Reusable scratch arena for the OS/HGP ERI engine.
///
/// Holds the engine's working buffers so they are allocated **once and reused**
/// across shell quartets, instead of a fresh heap allocation per quartet inside the
/// O(n⁴) shell loop. All buffers are plain `Vec<f64>` grown on demand in safe Rust
/// (`#![forbid(unsafe_code)]` holds); there is no shared mutable state, so each
/// thread uses its own instance — the thread-local default behind
/// [`coulomb_shell_into`], or one passed explicitly to
/// [`coulomb_shell_into_scratch`] (the `&mut` makes cross-thread sharing a compile
/// error).
///
/// **Memory-correctness.** `c_ef` is an accumulator and is **zeroed per quartet**.
/// The VRR `levels` and HRR `bra`/layer buffers are fully overwritten in the region
/// they are later read, so they need no functional zeroing — but in debug builds
/// `levels` and `bra` are NaN-filled per quartet, so any accidental out-of-range
/// read (e.g. outside the VRR `m`-triangle) poisons the output and trips the tests
/// rather than silently reading a stale value from a previous quartet. Results are
/// therefore independent of evaluation order and of arena reuse (asserted by
/// `tests/arena.rs`).
#[derive(Debug, Default, Clone)]
pub struct EriScratch {
    /// VRR rolling window: 3 triangle-packed f-degree levels (slot = `k % 3`).
    levels: Vec<f64>,
    /// Contracted `[e0|f0]^(0)` accumulator, shape `[n_e·n_f]`.
    c_ef: Vec<f64>,
    /// HRR bra intermediate `(a b | f 0)`, shape `[na·nb·nf_range]`.
    bra: Vec<f64>,
    /// HRR bra rolling `b`-degree layers.
    bra_prev: Vec<f64>,
    bra_cur: Vec<f64>,
    /// HRR ket rolling `d`-degree layers.
    ket_prev: Vec<f64>,
    ket_cur: Vec<f64>,
    /// Precomputed bra/ket primitive-pair data (combined exponent, product
    /// centre, `K` prefactor, coeff product) — built once per shell quartet so
    /// the per-pair `exp`/centre work is not repeated inside the deep primitive
    /// loop. Reused across quartets like the other buffers.
    bra_pairs: Vec<PrimPair>,
    ket_pairs: Vec<PrimPair>,
    /// Shared Cartesian-triple table `tri_all[d] = cart_components(d)` for every
    /// degree the engine needs (`0..=2·MAX_L`). Built once and sliced for the VRR
    /// `e`/`f` lists and the HRR triples, instead of re-`collect`ing per quartet.
    tri_all: Vec<Vec<[usize; 3]>>,
    /// Flat VRR offset table: `eoff[k·n_e + ae]` is `e`'s packed offset in level
    /// `k`. Rebuilt per quartet into this reused buffer (no per-quartet `Vec<Vec>`
    /// allocation); `slab[k]` is one `f`-component's packed size for level `k`.
    eoff: Vec<usize>,
    slab: Vec<usize>,
    /// VRR working tables for the extended monomorphized kernel [`vrr_big`]
    /// (classes with `ne` or `nf` in `4..=6`): the bra e-table plus the per-degree
    /// ket f-level tables, carved out of one reused arena slab (grown on demand)
    /// instead of stack arrays — the `(ff|ff)` worst case is ~480 KB, far past a
    /// sane stack budget, and per-shell-quartet zeroing of that would swamp the
    /// small classes that share the array shape. Overwritten in the region read
    /// per primitive quartet (NaN-poisoned in debug), like `levels`.
    big: Vec<f64>,
}

/// One primitive pair's reusable geometry/prefactor, shared by every primitive
/// quartet that uses it: the combined exponent `zeta = α+β`, the Gaussian product
/// centre, the overlap prefactor `K = exp(−αβ/ζ·|A−B|²)`, and the two contraction
/// coefficients (`c1` outer, `c2` inner).
///
/// The coefficients are kept **un-multiplied** so the contracted `scale` can be
/// formed in exactly the former evaluation order `((c_a·c_b)·c_c)·c_d` — making the
/// engine's output bit-identical to the pre-precomputation code (the per-element
/// and 8-fold-symmetry tests are floorless-relative and sensitive to the last ULP
/// on near-cancellation elements).
#[derive(Debug, Clone, Copy, Default)]
struct PrimPair {
    zeta: f64,
    center: Vec3,
    kappa: f64,
    c1: f64,
    c2: f64,
    /// `1/(2ζ)` — the VRR `inv_2p`(bra)/`inv_2q`(ket) coefficient. Pure per-pair.
    inv_2zeta: f64,
    /// `centre − s1.centre` — the VRR `P−A`(bra)/`Q−C`(ket) shift. Pure per-pair.
    r1: Vec3,
}

/// A primitive pair whose weighted overlap prefactor `K·|c₁·c₂|` falls below
/// this bound contributes at most ~`1e-24` (absolutely) to any tensor element —
/// roughly twelve orders below the engine's `1e-12` accuracy bar — so it is
/// dropped from the pair list before the quartet loop. Every `[e0|f0]^(m)` term
/// the pair could produce is bounded by `K·|c₁·c₂|` times the other pair's
/// weight (≤ its own bound), the `2π^{5/2}/(pq√(p+q))` prefactor, `F_m ≤ 1`,
/// and the VRR's geometric raise factors; the product stays ≤ ~1e-24 for any
/// chemically sensible exponents/geometry. The deeply contracted cross-centre
/// pairs this removes have `K` underflowing to `0.0` outright in typical bases
/// (tight×tight pairs at bonding distances reach `exp(-10⁴)`), where skipping
/// is *exactly* lossless.
const PAIR_NEGLIGIBLE: f64 = 1e-32;

/// Fill `out` with the `PrimPair` data for every primitive combination of two
/// shells (outer `s1`, inner `s2`), matching the former in-loop computation
/// element-for-element. The inter-centre distance is hoisted out of the pair loop.
/// `inv_2zeta` and `r1` are the pair-only VRR coefficients (`1/2ζ` and the `P−A` /
/// `Q−C` shift) that the kernel formerly recomputed on every primitive quartet.
/// Pairs below `PAIR_NEGLIGIBLE` are omitted; the surviving pairs keep their
/// order, so the contraction accumulates the remaining terms in the same
/// sequence as before.
fn build_pairs(out: &mut Vec<PrimPair>, s1: ShellRef<'_>, s2: ShellRef<'_>) {
    out.clear();
    out.reserve(s1.exps.len() * s2.exps.len());
    let d2 = dist2(s1.center, s2.center);
    for (&e1, &c1) in s1.exps.iter().zip(s1.coeffs.iter()) {
        for (&e2, &c2) in s2.exps.iter().zip(s2.coeffs.iter()) {
            let zeta = e1 + e2;
            let kappa = (-(e1 * e2 / zeta) * d2).exp();
            if kappa * (c1 * c2).abs() < PAIR_NEGLIGIBLE {
                continue;
            }
            let center = combine(e1, s1.center, e2, s2.center, zeta);
            out.push(PrimPair {
                zeta,
                center,
                kappa,
                c1,
                c2,
                inv_2zeta: 0.5 / zeta,
                r1: sub(center, s1.center),
            });
        }
    }
}

/// Precomputed, screened `PrimPair` list of one **ordered** shell pair —
/// libcint's pair-data ("optimizer") precompute. A dense driver builds one per
/// canonical shell pair once per build (`O(n_shells²)`, trivial memory) and
/// passes borrowed lists to [`coulomb_shell_pairs_into_scratch`] /
/// [`coulomb_shell_batch4_pairs_into_scratch`], instead of the engine rebuilding
/// the same pairs on every quartet that shares the pair.
///
/// The contents and order are exactly what the self-building entries compute
/// internally (`build_pairs`), so routing through borrowed lists is
/// bit-identical. **Orientation matters**: `build_pairs(s1, s2)` is
/// order-sensitive (`r1 = P − A` uses `s1`'s centre; `c1`/`c2` keep the
/// `s1`-outer/`s2`-inner coefficient association), so a list built for `(i, j)`
/// must only be used where the engine would have built `(i, j)` — the canonical
/// drivers use the `i ≥ j` orientation for both bra and ket throughout.
#[derive(Debug, Clone, Default)]
pub struct ShellPairData {
    pairs: Vec<PrimPair>,
}

impl ShellPairData {
    /// Number of surviving primitive pairs (the `PAIR_NEGLIGIBLE` screen
    /// applied) — what [`surviving_pair_count`] returns for the same pair.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// `true` when no primitive pair survives the negligibility screen.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }
}

/// Build the screened pair list for the **ordered** shell pair `(s1, s2)` —
/// element-for-element what the self-building engine entries compute per
/// quartet (see [`ShellPairData`] for the orientation contract).
#[must_use]
pub fn shell_pair_data(s1: ShellRef<'_>, s2: ShellRef<'_>) -> ShellPairData {
    let mut pairs = Vec::new();
    build_pairs(&mut pairs, s1, s2);
    ShellPairData { pairs }
}

impl EriScratch {
    /// A fresh, empty arena; it grows to fit on first use.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Total `f64` elements currently held across all buffers — the resident
    /// working set, for memory reporting/tests.
    #[must_use]
    pub fn resident_f64(&self) -> usize {
        self.levels.len()
            + self.c_ef.len()
            + self.bra.len()
            + self.bra_prev.len()
            + self.bra_cur.len()
            + self.ket_prev.len()
            + self.ket_cur.len()
    }

    /// Largest single buffer (`f64` elements). Demonstrates the former ~41 MB
    /// monolithic VRR `[e0|f0]^(m)` table is no longer resident: the m-marching
    /// window keeps only `3·max_k[n_cart(k)·slab_k]`.
    #[must_use]
    pub fn largest_buffer_f64(&self) -> usize {
        [
            self.levels.len(),
            self.c_ef.len(),
            self.bra.len(),
            self.bra_prev.len(),
            self.bra_cur.len(),
            self.ket_prev.len(),
            self.ket_cur.len(),
        ]
        .into_iter()
        .max()
        .unwrap_or(0)
    }
}

thread_local! {
    /// Per-thread default arena backing [`coulomb_shell_into`].
    static ERI_SCRATCH: RefCell<EriScratch> = RefCell::new(EriScratch::new());
}

/// Grow `v` to at least `n` elements (reusing existing capacity); never shrinks.
#[inline]
fn ensure_len(v: &mut Vec<f64>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
}

/// `ensure_len` for the `usize` offset/slab buffers.
#[inline]
fn ensure_usize(v: &mut Vec<usize>, n: usize) {
    if v.len() < n {
        v.resize(n, 0);
    }
}

/// Accumulate the contracted Coulomb block `(ab|cd)` for four shells into the
/// row-major `out` block (shape `[n_cart(la)·n_cart(lb)·n_cart(lc)·n_cart(ld)]`,
/// the same `(a,b,c,d)` layout as [`crate::engine::rys::coulomb_into`]).
///
/// The engine owns the primitive-quartet loop (the HGP contraction happens between
/// VRR and HRR), so unlike the Rys engine the driver calls this **once per shell
/// quartet** with all primitives. This uses a **thread-local** [`EriScratch`]; for
/// explicit control (e.g. one arena per worker thread) call
/// [`coulomb_shell_into_scratch`].
///
/// # Panics
/// Panics (in debug builds) if any `l > MAX_L` or `out` is too short.
pub fn coulomb_shell_into(
    a: ShellRef<'_>,
    b: ShellRef<'_>,
    c: ShellRef<'_>,
    d: ShellRef<'_>,
    out: &mut [f64],
) {
    ERI_SCRATCH.with(|s| coulomb_shell_into_scratch(&mut s.borrow_mut(), a, b, c, d, out));
}

/// Like [`coulomb_shell_into`] (thread-local arena) but with the bra/ket
/// primitive-pair data supplied by the caller — the borrowed-pairs analogue,
/// see [`ShellPairData`]. Bit-identical to [`coulomb_shell_into`].
///
/// # Panics
/// Panics (in debug builds) if any `l > MAX_L` or `out` is too short.
pub fn coulomb_shell_pairs_into(
    a: ShellRef<'_>,
    b: ShellRef<'_>,
    c: ShellRef<'_>,
    d: ShellRef<'_>,
    bra_pairs: &ShellPairData,
    ket_pairs: &ShellPairData,
    out: &mut [f64],
) {
    ERI_SCRATCH.with(|s| {
        coulomb_shell_pairs_into_scratch(
            &mut s.borrow_mut(),
            a,
            b,
            c,
            d,
            bra_pairs,
            ket_pairs,
            out,
        );
    });
}

/// Like [`coulomb_shell_into`] but evaluates into the caller-provided
/// [`EriScratch`], reused across quartets to avoid per-quartet heap allocation. Use
/// **one instance per thread** (sharing a `&mut EriScratch` across threads is a
/// compile error); the result is bit-identical regardless of which arena is used or
/// what it last held.
///
/// # Panics
/// Panics (in debug builds) if any `l > MAX_L` or `out` is too short.
pub fn coulomb_shell_into_scratch(
    scratch: &mut EriScratch,
    a: ShellRef<'_>,
    b: ShellRef<'_>,
    c: ShellRef<'_>,
    d: ShellRef<'_>,
    out: &mut [f64],
) {
    // Build the pair lists into the arena's reused buffers, then run the core
    // on borrowed slices. The vecs are moved out for the duration of the call
    // (a pointer swap) so the core can take `&mut` of the rest of the arena.
    let mut bra_pairs = std::mem::take(&mut scratch.bra_pairs);
    let mut ket_pairs = std::mem::take(&mut scratch.ket_pairs);
    build_pairs(&mut bra_pairs, a, b);
    build_pairs(&mut ket_pairs, c, d);
    coulomb_shell_core(scratch, a, b, c, d, &bra_pairs, &ket_pairs, out);
    scratch.bra_pairs = bra_pairs;
    scratch.ket_pairs = ket_pairs;
}

/// Like [`coulomb_shell_into_scratch`] but with the bra/ket primitive-pair data
/// supplied by the caller (precomputed once per shell pair across a dense
/// build, see [`ShellPairData`]), instead of rebuilt per quartet. Bit-identical:
/// the pair values and their order are exactly what the self-building entry
/// computes — only *where* they are computed moves.
///
/// # Panics
/// Panics (in debug builds) if any `l > MAX_L` or `out` is too short.
#[allow(clippy::too_many_arguments)]
pub fn coulomb_shell_pairs_into_scratch(
    scratch: &mut EriScratch,
    a: ShellRef<'_>,
    b: ShellRef<'_>,
    c: ShellRef<'_>,
    d: ShellRef<'_>,
    bra_pairs: &ShellPairData,
    ket_pairs: &ShellPairData,
    out: &mut [f64],
) {
    coulomb_shell_core(scratch, a, b, c, d, &bra_pairs.pairs, &ket_pairs.pairs, out);
}

/// The quartet kernel shared by the self-building and borrowed-pairs entries:
/// fast path / VRR dispatch / HRR over caller-provided pair slices.
#[allow(clippy::too_many_arguments)]
fn coulomb_shell_core(
    scratch: &mut EriScratch,
    a: ShellRef<'_>,
    b: ShellRef<'_>,
    c: ShellRef<'_>,
    d: ShellRef<'_>,
    bra_pairs: &[PrimPair],
    ket_pairs: &[PrimPair],
    out: &mut [f64],
) {
    let (la, lb, lc, ld) = (a.l, b.l, c.l, d.l);
    debug_assert!(
        la <= MAX_L && lb <= MAX_L && lc <= MAX_L && ld <= MAX_L,
        "angular momentum exceeds MAX_L"
    );
    let (na, nb, nc, nd) = (n_cart(la), n_cart(lb), n_cart(lc), n_cart(ld));
    debug_assert!(out.len() >= na * nb * nc * nd, "ERI output block too short");

    let ne = la + lb; // max bra (A-side) degree built by VRR
    let nf = lc + ld; // max ket (C-side) degree built by VRR
    let l_total = ne + nf;
    let n_e = n_addr(ne);
    let n_f = n_addr(nf);

    let EriScratch {
        levels,
        c_ef,
        bra,
        bra_prev,
        bra_cur,
        ket_prev,
        ket_cur,
        tri_all,
        eoff,
        slab,
        big,
        ..
    } = scratch;

    // Fast path: (ss|ss). With no angular momentum to raise, the whole quartet is a
    // single contracted [00|00]^(0). Skip the VRR/HRR machinery (the dead raise
    // coefficients, the `levels` buffer round-trip, the per-primitive call) and sum
    // straight into a register. Bit-identical to the general path: same `pref`/`t`
    // expressions, same `((c_a·c_b)·c_c)·c_d` scale, same `scale·(pref·F_0)` term,
    // same bra-outer/ket-inner order — just without the scaffolding.
    if l_total == 0 {
        let two_pi_2_5 =
            2.0 * std::f64::consts::PI * std::f64::consts::PI * std::f64::consts::PI.sqrt();
        let mut acc = 0.0;
        let mut f0 = [0.0f64; 1];
        let mut f1 = [0.0f64; 1];
        for bra in bra_pairs.iter() {
            let bc = bra.c1 * bra.c2;
            let p = bra.zeta;
            // Kets in PAIRS through the interleaved `boys_array2` (per-lane
            // arithmetic and the lane-0-then-lane-1 accumulation order are
            // exactly the former sequential loop's, so this is bit-identical).
            let mut kets = ket_pairs.chunks_exact(2);
            for pair in kets.by_ref() {
                let (k0, k1) = (&pair[0], &pair[1]);
                let pq0 = p + k0.zeta;
                let pq1 = p + k1.zeta;
                let t0 = (p * k0.zeta / pq0) * dist2(bra.center, k0.center);
                let t1 = (p * k1.zeta / pq1) * dist2(bra.center, k1.center);
                let pref0 = two_pi_2_5 / (p * k0.zeta * pq0.sqrt()) * bra.kappa * k0.kappa;
                let pref1 = two_pi_2_5 / (p * k1.zeta * pq1.sqrt()) * bra.kappa * k1.kappa;
                boys_array2(0, t0, t1, &mut f0, &mut f1);
                acc += (bc * k0.c1) * k0.c2 * (pref0 * f0[0]);
                acc += (bc * k1.c1) * k1.c2 * (pref1 * f1[0]);
            }
            for ket in kets.remainder() {
                let q = ket.zeta;
                let pq = p + q;
                let t = (p * q / pq) * dist2(bra.center, ket.center);
                let pref = two_pi_2_5 / (p * q * pq.sqrt()) * bra.kappa * ket.kappa;
                boys_array(0, t, &mut f0);
                acc += (bc * ket.c1) * ket.c2 * (pref * f0[0]);
            }
        }
        out[0] += acc;
        return;
    }

    // Contracted [e0|f0]^(0): zeroed per quartet (it is a `+=` accumulator).
    ensure_len(c_ef, n_e * n_f);
    c_ef[..n_e * n_f].fill(0.0);

    // Shared Cartesian-triple table, built once (degrees `0..=2·MAX_L` cover every
    // `ne`/`nf`/`max(ne,nf)` the engine reaches) and sliced below — no per-quartet
    // `cart_components` allocation.
    if tri_all.len() <= 2 * MAX_L {
        *tri_all = (0..=2 * MAX_L).map(cart_components).collect();
    }

    // VRR. The small classes (`ne, nf ≤ 3` — every quartet shape of an
    // spd basis except the d,d-bra/ket pairs, ~96% of a cc-pVDZ-style build)
    // dispatch to the monomorphized kernels, where the whole structural walk
    // (triples, addresses, branches, m-ranges) is resolved at compile time;
    // everything else runs the general m-marching `vrr_primitive`.
    // `(0,0)` already returned through the (ss|ss) fast path above.
    match (ne, nf) {
        (0, 1) => contract_class::<0, 1>(bra_pairs, ket_pairs, c_ef),
        (1, 0) => contract_class::<1, 0>(bra_pairs, ket_pairs, c_ef),
        (1, 1) => contract_class::<1, 1>(bra_pairs, ket_pairs, c_ef),
        (0, 2) => contract_class::<0, 2>(bra_pairs, ket_pairs, c_ef),
        (2, 0) => contract_class::<2, 0>(bra_pairs, ket_pairs, c_ef),
        (1, 2) => contract_class::<1, 2>(bra_pairs, ket_pairs, c_ef),
        (2, 1) => contract_class::<2, 1>(bra_pairs, ket_pairs, c_ef),
        (2, 2) => contract_class::<2, 2>(bra_pairs, ket_pairs, c_ef),
        (0, 3) => contract_class::<0, 3>(bra_pairs, ket_pairs, c_ef),
        (3, 0) => contract_class::<3, 0>(bra_pairs, ket_pairs, c_ef),
        (1, 3) => contract_class::<1, 3>(bra_pairs, ket_pairs, c_ef),
        (3, 1) => contract_class::<3, 1>(bra_pairs, ket_pairs, c_ef),
        (2, 3) => contract_class::<2, 3>(bra_pairs, ket_pairs, c_ef),
        (3, 2) => contract_class::<3, 2>(bra_pairs, ket_pairs, c_ef),
        (3, 3) => contract_class::<3, 3>(bra_pairs, ket_pairs, c_ef),
        // Extended monomorphized kernel for the d/f-heavy classes (one of
        // `ne`/`nf` in `4..=6`), covering every quartet of a cc-pVTZ Cartesian
        // build. Classes with `ne` or `nf` ≥ 7 still fall to the generic path.
        (0, 4) => contract_class_big::<0, 4>(bra_pairs, ket_pairs, c_ef, big),
        (0, 5) => contract_class_big::<0, 5>(bra_pairs, ket_pairs, c_ef, big),
        (0, 6) => contract_class_big::<0, 6>(bra_pairs, ket_pairs, c_ef, big),
        (1, 4) => contract_class_big::<1, 4>(bra_pairs, ket_pairs, c_ef, big),
        (1, 5) => contract_class_big::<1, 5>(bra_pairs, ket_pairs, c_ef, big),
        (1, 6) => contract_class_big::<1, 6>(bra_pairs, ket_pairs, c_ef, big),
        (2, 4) => contract_class_big::<2, 4>(bra_pairs, ket_pairs, c_ef, big),
        (2, 5) => contract_class_big::<2, 5>(bra_pairs, ket_pairs, c_ef, big),
        (2, 6) => contract_class_big::<2, 6>(bra_pairs, ket_pairs, c_ef, big),
        (3, 4) => contract_class_big::<3, 4>(bra_pairs, ket_pairs, c_ef, big),
        (3, 5) => contract_class_big::<3, 5>(bra_pairs, ket_pairs, c_ef, big),
        (3, 6) => contract_class_big::<3, 6>(bra_pairs, ket_pairs, c_ef, big),
        (4, 0) => contract_class_big::<4, 0>(bra_pairs, ket_pairs, c_ef, big),
        (4, 1) => contract_class_big::<4, 1>(bra_pairs, ket_pairs, c_ef, big),
        (4, 2) => contract_class_big::<4, 2>(bra_pairs, ket_pairs, c_ef, big),
        (4, 3) => contract_class_big::<4, 3>(bra_pairs, ket_pairs, c_ef, big),
        (4, 4) => contract_class_big::<4, 4>(bra_pairs, ket_pairs, c_ef, big),
        (4, 5) => contract_class_big::<4, 5>(bra_pairs, ket_pairs, c_ef, big),
        (4, 6) => contract_class_big::<4, 6>(bra_pairs, ket_pairs, c_ef, big),
        (5, 0) => contract_class_big::<5, 0>(bra_pairs, ket_pairs, c_ef, big),
        (5, 1) => contract_class_big::<5, 1>(bra_pairs, ket_pairs, c_ef, big),
        (5, 2) => contract_class_big::<5, 2>(bra_pairs, ket_pairs, c_ef, big),
        (5, 3) => contract_class_big::<5, 3>(bra_pairs, ket_pairs, c_ef, big),
        (5, 4) => contract_class_big::<5, 4>(bra_pairs, ket_pairs, c_ef, big),
        (5, 5) => contract_class_big::<5, 5>(bra_pairs, ket_pairs, c_ef, big),
        (5, 6) => contract_class_big::<5, 6>(bra_pairs, ket_pairs, c_ef, big),
        (6, 0) => contract_class_big::<6, 0>(bra_pairs, ket_pairs, c_ef, big),
        (6, 1) => contract_class_big::<6, 1>(bra_pairs, ket_pairs, c_ef, big),
        (6, 2) => contract_class_big::<6, 2>(bra_pairs, ket_pairs, c_ef, big),
        (6, 3) => contract_class_big::<6, 3>(bra_pairs, ket_pairs, c_ef, big),
        (6, 4) => contract_class_big::<6, 4>(bra_pairs, ket_pairs, c_ef, big),
        (6, 5) => contract_class_big::<6, 5>(bra_pairs, ket_pairs, c_ef, big),
        (6, 6) => contract_class_big::<6, 6>(bra_pairs, ket_pairs, c_ef, big),
        _ => {
            // m-marching VRR scratch. Instead of the full `n_e·n_f·nm`
            // table (~41 MB at `(ii|ii)`), keep only a rolling window of **3 consecutive
            // f-degree levels**, each **triangle-packed** in the Boys index `m`: for an
            // `f`-degree `k`, an `e` of degree `de` stores only `m ∈ 0..=(l_total−de−k)`.
            // Level `k` lives in slot `k % 3` of `levels`; `slab[k]` is one `f`-component's
            // packed size and `eoff[k·n_e + ae]` is `e`'s offset within it (flat, reused).
            ensure_usize(slab, nf + 1);
            ensure_usize(eoff, (nf + 1) * n_e);
            // `k` indexes both `slab` and the flat `eoff` block base `k·n_e`, so a range
            // loop is the clear form here.
            #[allow(clippy::needless_range_loop)]
            for k in 0..=nf {
                let base = k * n_e;
                let mut run = 0usize;
                let mut ae = 0usize;
                for de in 0..=ne {
                    let mlen = l_total - de - k + 1; // ≥ 1: de + k ≤ ne + nf = l_total
                    for _ in 0..n_cart(de) {
                        eoff[base + ae] = run;
                        ae += 1;
                        run += mlen;
                    }
                }
                slab[k] = run;
            }
            let maxlevel = (0..=nf).map(|k| n_cart(k) * slab[k]).max().unwrap_or(1);
            ensure_len(levels, 3 * maxlevel);
            // Debug guard: poison the reused VRR window so any out-of-triangle / stale read
            // surfaces as a NaN in the output (caught by the golden + cross-engine tests),
            // instead of silently using a previous quartet's value.
            #[cfg(debug_assertions)]
            levels[..3 * maxlevel].fill(f64::NAN);

            // Reborrow the offset/triple buffers as shared slices for the read-only kernel.
            let (eoff_s, slab_s, tri_s) = (&eoff[..], &slab[..], &tri_all[..]);
            for bra in bra_pairs.iter() {
                let bra_coef = bra.c1 * bra.c2; // = c_a·c_b, hoisted out of the ket loop
                for ket in ket_pairs.iter() {
                    // ((c_a·c_b)·c_c)·c_d — the former left-to-right product, bit for bit.
                    let scale = (bra_coef * ket.c1) * ket.c2;
                    vrr_primitive(
                        bra.zeta,
                        ket.zeta,
                        bra.center,
                        ket.center,
                        bra.kappa,
                        ket.kappa,
                        bra.r1,
                        ket.r1,
                        bra.inv_2zeta,
                        ket.inv_2zeta,
                        ne,
                        nf,
                        l_total,
                        n_e,
                        n_f,
                        maxlevel,
                        eoff_s,
                        slab_s,
                        tri_s,
                        levels,
                        scale,
                        c_ef,
                    );
                }
            }
        }
    }

    // HRR in contracted space, then scatter the block.
    let ab = sub(a.center, b.center); // A − B
    let cd = sub(c.center, d.center); // C − D
    hrr_and_scatter(
        la,
        lb,
        lc,
        ld,
        n_f,
        c_ef,
        ab,
        cd,
        out,
        bra,
        bra_prev,
        bra_cur,
        ket_prev,
        ket_cur,
        &tri_all[..],
    );
}

/// Cartesian triples of degree 1, 2 and 3 in [`cart_components`] order, as `const`
/// tables so the monomorphized small-class VRR kernels fully unroll over them
/// (the structural walk — axes, addresses, `has2`/cross branches — then resolves
/// at compile time instead of being re-derived on every primitive quartet).
const TRI1: [[usize; 3]; 3] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
const TRI2: [[usize; 3]; 6] = [
    [2, 0, 0],
    [1, 1, 0],
    [1, 0, 1],
    [0, 2, 0],
    [0, 1, 1],
    [0, 0, 2],
];
const TRI3: [[usize; 3]; 10] = [
    [3, 0, 0],
    [2, 1, 0],
    [2, 0, 1],
    [1, 2, 0],
    [1, 1, 1],
    [1, 0, 2],
    [0, 3, 0],
    [0, 2, 1],
    [0, 1, 2],
    [0, 0, 3],
];
/// Degree-4/5/6 Cartesian triples in [`cart_components`] / [`addr`] order, used by
/// the extended monomorphized VRR kernel [`vrr_big`] (classes with `ne` or `nf`
/// in `4..=6`, i.e. the d/f-heavy quartets a cc-pVTZ+ build spends most of its
/// kernel time on). Same const-table role as `TRI1..TRI3`.
const TRI4: [[usize; 3]; 15] = [
    [4, 0, 0],
    [3, 1, 0],
    [3, 0, 1],
    [2, 2, 0],
    [2, 1, 1],
    [2, 0, 2],
    [1, 3, 0],
    [1, 2, 1],
    [1, 1, 2],
    [1, 0, 3],
    [0, 4, 0],
    [0, 3, 1],
    [0, 2, 2],
    [0, 1, 3],
    [0, 0, 4],
];
const TRI5: [[usize; 3]; 21] = [
    [5, 0, 0],
    [4, 1, 0],
    [4, 0, 1],
    [3, 2, 0],
    [3, 1, 1],
    [3, 0, 2],
    [2, 3, 0],
    [2, 2, 1],
    [2, 1, 2],
    [2, 0, 3],
    [1, 4, 0],
    [1, 3, 1],
    [1, 2, 2],
    [1, 1, 3],
    [1, 0, 4],
    [0, 5, 0],
    [0, 4, 1],
    [0, 3, 2],
    [0, 2, 3],
    [0, 1, 4],
    [0, 0, 5],
];
const TRI6: [[usize; 3]; 28] = [
    [6, 0, 0],
    [5, 1, 0],
    [5, 0, 1],
    [4, 2, 0],
    [4, 1, 1],
    [4, 0, 2],
    [3, 3, 0],
    [3, 2, 1],
    [3, 1, 2],
    [3, 0, 3],
    [2, 4, 0],
    [2, 3, 1],
    [2, 2, 2],
    [2, 1, 3],
    [2, 0, 4],
    [1, 5, 0],
    [1, 4, 1],
    [1, 3, 2],
    [1, 2, 3],
    [1, 1, 4],
    [1, 0, 5],
    [0, 6, 0],
    [0, 5, 1],
    [0, 4, 2],
    [0, 3, 3],
    [0, 2, 4],
    [0, 1, 5],
    [0, 0, 6],
];
/// The e-components `0..n_addr(6)` in [`addr`] order with their degree — the bra
/// index set of the monomorphized kernels. `vrr_small` (`NE ≤ 3`) reads the first
/// 20 entries (unchanged); `vrr_big` (`NE ≤ 6`) reads all 84.
const E_COMP: [([usize; 3], usize); 84] = [
    ([0, 0, 0], 0),
    ([1, 0, 0], 1),
    ([0, 1, 0], 1),
    ([0, 0, 1], 1),
    ([2, 0, 0], 2),
    ([1, 1, 0], 2),
    ([1, 0, 1], 2),
    ([0, 2, 0], 2),
    ([0, 1, 1], 2),
    ([0, 0, 2], 2),
    ([3, 0, 0], 3),
    ([2, 1, 0], 3),
    ([2, 0, 1], 3),
    ([1, 2, 0], 3),
    ([1, 1, 1], 3),
    ([1, 0, 2], 3),
    ([0, 3, 0], 3),
    ([0, 2, 1], 3),
    ([0, 1, 2], 3),
    ([0, 0, 3], 3),
    ([4, 0, 0], 4),
    ([3, 1, 0], 4),
    ([3, 0, 1], 4),
    ([2, 2, 0], 4),
    ([2, 1, 1], 4),
    ([2, 0, 2], 4),
    ([1, 3, 0], 4),
    ([1, 2, 1], 4),
    ([1, 1, 2], 4),
    ([1, 0, 3], 4),
    ([0, 4, 0], 4),
    ([0, 3, 1], 4),
    ([0, 2, 2], 4),
    ([0, 1, 3], 4),
    ([0, 0, 4], 4),
    ([5, 0, 0], 5),
    ([4, 1, 0], 5),
    ([4, 0, 1], 5),
    ([3, 2, 0], 5),
    ([3, 1, 1], 5),
    ([3, 0, 2], 5),
    ([2, 3, 0], 5),
    ([2, 2, 1], 5),
    ([2, 1, 2], 5),
    ([2, 0, 3], 5),
    ([1, 4, 0], 5),
    ([1, 3, 1], 5),
    ([1, 2, 2], 5),
    ([1, 1, 3], 5),
    ([1, 0, 4], 5),
    ([0, 5, 0], 5),
    ([0, 4, 1], 5),
    ([0, 3, 2], 5),
    ([0, 2, 3], 5),
    ([0, 1, 4], 5),
    ([0, 0, 5], 5),
    ([6, 0, 0], 6),
    ([5, 1, 0], 6),
    ([5, 0, 1], 6),
    ([4, 2, 0], 6),
    ([4, 1, 1], 6),
    ([4, 0, 2], 6),
    ([3, 3, 0], 6),
    ([3, 2, 1], 6),
    ([3, 1, 2], 6),
    ([3, 0, 3], 6),
    ([2, 4, 0], 6),
    ([2, 3, 1], 6),
    ([2, 2, 2], 6),
    ([2, 1, 3], 6),
    ([2, 0, 4], 6),
    ([1, 5, 0], 6),
    ([1, 4, 1], 6),
    ([1, 3, 2], 6),
    ([1, 2, 3], 6),
    ([1, 1, 4], 6),
    ([1, 0, 5], 6),
    ([0, 6, 0], 6),
    ([0, 5, 1], 6),
    ([0, 4, 2], 6),
    ([0, 3, 3], 6),
    ([0, 2, 4], 6),
    ([0, 1, 5], 6),
    ([0, 0, 6], 6),
];

/// Per-primitive-quartet scalar header consumed by [`vrr_small`]: everything the
/// kernel needs that depends on the *pair of pairs* (not on either pair alone).
/// The expressions are the former `vrr_small` prologue, verbatim — only *where*
/// they are evaluated moves (into the strip loop of [`contract_class`]).
#[derive(Clone, Copy, Default)]
struct QuartetHeader {
    wp: [f64; 3], // W − P
    wq: [f64; 3], // W − Q
    pref: f64,
    inv_2pq: f64,
    q_over_pq: f64,
    p_over_pq: f64,
    scale: f64, // ((c_a·c_b)·c_c)·c_d
    t: f64,     // Boys argument ρ·|P−Q|²
}

/// Ket-strip width of the strip-mined primitive loop in [`contract_class`].
const STRIP: usize = 4;

/// Contract a whole shell quartet of VRR shape `(NE, NF)` (with `NE, NF ≤ 3`)
/// into `c_ef` using the monomorphized small-class kernel [`vrr_small`].
///
/// Same bra-outer / ket-inner primitive order and the same left-to-right
/// `((c_a·c_b)·c_c)·c_d` scale as the general path, so the accumulation into
/// `c_ef` is bit-identical. The ket loop is **strip-mined**: for a strip of up
/// to [`STRIP`] ket pairs, first compute every scalar header (the serial
/// sqrt/divide chain) plus its Boys row in one lane loop, then run the VRR
/// body per lane — breaking the per-quartet header→Boys→VRR dependency chain
/// so a lane's Boys/header work overlaps another lane's VRR in the pipeline.
/// (A measured note: splitting Boys into its own third pass *regressed* —
/// the extra `t` store/reload and loop overhead cost more than the split
/// bought; the two-pass form is what wins.) Each lane's arithmetic
/// (expressions, evaluation order, accumulation order into `c_ef`) is
/// unchanged, so the result stays bit-identical.
/// The VRR working tables live here (overwritten per primitive
/// quartet in the region read, like the general path's `levels` arena) so
/// their one-time zero-init is per shell quartet, not per primitive.
fn contract_class<const NE: usize, const NF: usize>(
    bra_pairs: &[PrimPair],
    ket_pairs: &[PrimPair],
    c_ef: &mut [f64],
) {
    // Worst-case (NE,NF)=(3,3) sizes: e-table 20 components × m ∈ 0..=6;
    // f-degree-1 level 20×3 × m ∈ 0..=5; f-degree-2 level 20×6 × m ∈ 0..=4;
    // f-degree-3 level 20×10 × m ∈ 0..=3.
    let mut ea = [0.0f64; 140];
    let mut eb1 = [0.0f64; 360];
    let mut eb2 = [0.0f64; 600];
    let mut eb3 = [0.0f64; 800];
    let lt = NE + NF;
    let mut hdr = [QuartetHeader::default(); STRIP];
    let mut fm = [[0.0f64; 7]; STRIP];
    for bra in bra_pairs {
        let bra_coef = bra.c1 * bra.c2;
        let (p, pc) = (bra.zeta, bra.center);
        let mut start = 0;
        while start < ket_pairs.len() {
            let strip = &ket_pairs[start..(start + STRIP).min(ket_pairs.len())];
            // Phase 1 — headers + Boys rows: identical expressions to
            // `vrr_primitive`, independent iterations. Lanes are processed in
            // PAIRS so the two Boys ladders evaluate through the manually
            // interleaved [`boys_array2`] (two overlapping Horner/recurrence
            // chains); each lane's own arithmetic is unchanged.
            let header = |h: &mut QuartetHeader, ket: &PrimPair| {
                let (q, qc) = (ket.zeta, ket.center);
                let pq = p + q;
                let rho = p * q / pq;
                let pq_vec = sub(pc, qc); // P − Q
                h.t = rho * norm2(pq_vec);
                let w = [
                    (p * pc[0] + q * qc[0]) / pq,
                    (p * pc[1] + q * qc[1]) / pq,
                    (p * pc[2] + q * qc[2]) / pq,
                ];
                h.wp = sub(w, pc);
                h.wq = sub(w, qc);
                use std::f64::consts::PI;
                h.pref = 2.0 * PI * PI * PI.sqrt() / (p * q * pq.sqrt()) * bra.kappa * ket.kappa;
                h.inv_2pq = 0.5 / pq;
                h.q_over_pq = q / pq;
                h.p_over_pq = p / pq;
                h.scale = (bra_coef * ket.c1) * ket.c2;
            };
            let len = strip.len();
            let mut k = 0;
            while k + 2 <= len {
                header(&mut hdr[k], &strip[k]);
                header(&mut hdr[k + 1], &strip[k + 1]);
                let (f0, f1) = fm.split_at_mut(k + 1);
                boys_array2(
                    lt,
                    hdr[k].t,
                    hdr[k + 1].t,
                    &mut f0[k][..=lt],
                    &mut f1[0][..=lt],
                );
                k += 2;
            }
            if k < len {
                header(&mut hdr[k], &strip[k]);
                boys_array(lt, hdr[k].t, &mut fm[k][..=lt]);
            }
            // Phase 2 — VRR body per lane, fed from the precomputed header.
            for (k, ket) in strip.iter().enumerate() {
                vrr_small::<NE, NF>(
                    bra, ket, &hdr[k], &fm[k], &mut ea, &mut eb1, &mut eb2, &mut eb3, c_ef,
                );
            }
            start += STRIP;
        }
    }
}

/// One primitive quartet of the OS VRR for the small classes `NE, NF ≤ 3`,
/// monomorphized per `(NE, NF)` — a per-L-class specialized kernel expressed
/// through const generics: every loop bound, Cartesian triple, address, stride,
/// and `has2`/cross branch below is a compile-time constant in each
/// instantiation, so the emitted code is a straight unrolled FMA chain.
///
/// **Bit-identical to [`vrr_primitive`] by construction**: the same recurrence
/// expressions with the same term order and association — base
/// `pref·F_m`, raise `pa_i·s1[m] + wp_i·s1[m+1]`, then `+= coef2·(s2[m] −
/// (q/pq)·s2[m+1])`, then `+= cross·s3[m+1]` — the same m-ranges
/// (`m ≤ l_total − de − k`) for every *consumed* value, and the same
/// `c_ef[ae·n_f + af] += scale·v` extraction. Two things differ, neither of
/// which touches a kept value's arithmetic: the storage (fixed per-degree
/// tables instead of the triangle-packed rolling window), and the final
/// f-level computing only its consumed `m = 0` row (the general path also
/// fills the dead `m > 0` triangle there).
/// Guarded by the full-tensor XOR fingerprint and the golden/cross-engine tests.
///
/// Per-degree row strides: the m-rows of `ea` hold `m ∈ 0..=lt` (`sa = lt+1`),
/// of `eb1` `m ∈ 0..=lt−1` (`s1 = lt`), of `eb2` `m ∈ 0..=lt−2`, of `eb3`
/// `m ∈ 0..=lt−3` — each level's longest row (its degree-0 e-component).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn vrr_small<const NE: usize, const NF: usize>(
    bra: &PrimPair,
    ket: &PrimPair,
    hdr: &QuartetHeader,
    fm: &[f64; 7],
    ea: &mut [f64; 140],
    eb1: &mut [f64; 360],
    eb2: &mut [f64; 600],
    eb3: &mut [f64; 800],
    c_ef: &mut [f64],
) {
    let lt = NE + NF;
    let n_e = n_addr(NE);
    let n_f = n_addr(NF);
    // Row strides (compile-time constants per instantiation).
    let sa = lt + 1;
    let s1 = lt;
    let s2 = lt.saturating_sub(1);
    let s3 = lt.saturating_sub(2);
    let pa = bra.r1; // P − A
    let qcen = ket.r1; // Q − C
    let (inv_2p, inv_2q) = (bra.inv_2zeta, ket.inv_2zeta);

    // Scalar header + Boys row: precomputed by `contract_class` (phases 1–2).
    let QuartetHeader {
        wp,
        wq,
        pref,
        inv_2pq,
        q_over_pq,
        p_over_pq,
        scale,
        t: _,
    } = *hdr;
    for m in 0..=lt {
        ea[m] = pref * fm[m];
    }

    // Phase A — bra ladder: [e0|00]^(m), e-degrees 1..=NE, m ≤ lt − de.
    if NE >= 1 {
        for (iw, te) in TRI1.iter().enumerate() {
            let i = lower_axis(*te);
            // Source is [0,0,0] at addr 0; no has2 term at degree 1.
            for m in 0..=(lt - 1) {
                ea[(1 + iw) * sa + m] = pa[i] * ea[m] + wp[i] * ea[m + 1];
            }
        }
    }
    if NE >= 2 {
        for (iw, te) in TRI2.iter().enumerate() {
            let i = lower_axis(*te);
            let s1a = addr(dec(*te, i));
            let has2 = te[i] >= 2;
            let coef2 = ((te[i] - 1) as f64) * inv_2p;
            let s2a = if has2 { addr(dec(dec(*te, i), i)) } else { 0 };
            for m in 0..=(lt - 2) {
                let mut v = pa[i] * ea[s1a * sa + m] + wp[i] * ea[s1a * sa + m + 1];
                if has2 {
                    v += coef2 * (ea[s2a * sa + m] - q_over_pq * ea[s2a * sa + m + 1]);
                }
                ea[(4 + iw) * sa + m] = v;
            }
        }
    }
    if NE >= 3 {
        for (iw, te) in TRI3.iter().enumerate() {
            let i = lower_axis(*te);
            let s1a = addr(dec(*te, i));
            let has2 = te[i] >= 2;
            let coef2 = ((te[i] - 1) as f64) * inv_2p;
            let s2a = if has2 { addr(dec(dec(*te, i), i)) } else { 0 };
            for m in 0..=(lt - 3) {
                let mut v = pa[i] * ea[s1a * sa + m] + wp[i] * ea[s1a * sa + m + 1];
                if has2 {
                    v += coef2 * (ea[s2a * sa + m] - q_over_pq * ea[s2a * sa + m + 1]);
                }
                ea[(10 + iw) * sa + m] = v;
            }
        }
    }
    // Extract f-degree 0: contract m = 0 of every e-component.
    for ae in 0..n_e {
        c_ef[ae * n_f] += scale * ea[ae * sa];
    }

    // Phase B — ket raise, f-degree k: backward liveness gives every f-level-k
    // row a needed m-range of `m ≤ NF − k`, independent of the e-degree: the
    // final level (k = NF) only feeds the m = 0 extraction, and each level
    // below it is read at m, m+1 by the next level's raise (its cross and the
    // k+2 has2 reach no further). The general path's full triangle
    // (`m ≤ lt − de − k`) computes `NE − de` dead values per row beyond that;
    // skipping them removes whole computations and leaves every kept value
    // bit-identical (fingerprint-guarded). Phase A has no such slack — its
    // liveness works out to exactly the `lt − de` triangle it computes.
    if NF >= 1 {
        for (fw, tf) in TRI1.iter().enumerate() {
            let j = lower_axis(*tf);
            // Source f-component is [0,0,0] (level 0 = the phase-A table).
            for ae in 0..n_e {
                let (te, _) = E_COMP[ae];
                let has_cross = te[j] >= 1;
                let cross_coef = te[j] as f64 * inv_2pq;
                let cs = if has_cross { addr(dec(te, j)) } else { 0 };
                let mtop = NF - 1;
                for m in 0..=mtop {
                    let mut v = qcen[j] * ea[ae * sa + m] + wq[j] * ea[ae * sa + m + 1];
                    if has_cross {
                        v += cross_coef * ea[cs * sa + m + 1];
                    }
                    eb1[(ae * 3 + fw) * s1 + m] = v;
                }
            }
        }
        for ae in 0..n_e {
            for fw in 0..3 {
                c_ef[ae * n_f + 1 + fw] += scale * eb1[(ae * 3 + fw) * s1];
            }
        }
    }

    // Phase B — ket raise, f-degree 2: m ≤ NF − 2 (liveness bound, see above).
    if NF >= 2 {
        for (fw, tf) in TRI2.iter().enumerate() {
            let j = lower_axis(*tf);
            let f1 = dec(*tf, j);
            let lf1 = cart_index(f1); // f-component index in the degree-1 level
            let has2 = tf[j] >= 2; // its k−2 source is then [0,0,0] (phase A)
            let coef2 = ((tf[j] - 1) as f64) * inv_2q;
            for ae in 0..n_e {
                let (te, _) = E_COMP[ae];
                let has_cross = te[j] >= 1;
                let cross_coef = te[j] as f64 * inv_2pq;
                let cs = if has_cross { addr(dec(te, j)) } else { 0 };
                let mtop = NF - 2;
                for m in 0..=mtop {
                    let mut v = qcen[j] * eb1[(ae * 3 + lf1) * s1 + m]
                        + wq[j] * eb1[(ae * 3 + lf1) * s1 + m + 1];
                    if has2 {
                        v += coef2 * (ea[ae * sa + m] - p_over_pq * ea[ae * sa + m + 1]);
                    }
                    if has_cross {
                        v += cross_coef * eb1[(cs * 3 + lf1) * s1 + m + 1];
                    }
                    eb2[(ae * 6 + fw) * s2 + m] = v;
                }
            }
        }
        for ae in 0..n_e {
            for fw in 0..6 {
                c_ef[ae * n_f + 4 + fw] += scale * eb2[(ae * 6 + fw) * s2];
            }
        }
    }

    // Phase B — ket raise, f-degree 3: always the final f-level (NF ≤ 3), so
    // only m = 0 is computed (same dead-row trim as above).
    if NF >= 3 {
        for (fw, tf) in TRI3.iter().enumerate() {
            let j = lower_axis(*tf);
            let f1 = dec(*tf, j);
            let lf1 = cart_index(f1); // f-component index in the degree-2 level
            let has2 = tf[j] >= 2; // its k−2 source is f1 lowered again (degree 1)
            let coef2 = ((tf[j] - 1) as f64) * inv_2q;
            let lf2 = if has2 { cart_index(dec(f1, j)) } else { 0 };
            for ae in 0..n_e {
                let (te, _) = E_COMP[ae];
                let has_cross = te[j] >= 1;
                let cross_coef = te[j] as f64 * inv_2pq;
                let cs = if has_cross { addr(dec(te, j)) } else { 0 };
                let mtop = 0; // final level: m = 0 only
                for m in 0..=mtop {
                    let mut v = qcen[j] * eb2[(ae * 6 + lf1) * s2 + m]
                        + wq[j] * eb2[(ae * 6 + lf1) * s2 + m + 1];
                    if has2 {
                        v += coef2
                            * (eb1[(ae * 3 + lf2) * s1 + m]
                                - p_over_pq * eb1[(ae * 3 + lf2) * s1 + m + 1]);
                    }
                    if has_cross {
                        v += cross_coef * eb2[(cs * 6 + lf1) * s2 + m + 1];
                    }
                    eb3[(ae * 10 + fw) * s3 + m] = v;
                }
            }
        }
        for ae in 0..n_e {
            for fw in 0..10 {
                c_ef[ae * n_f + 10 + fw] += scale * eb3[(ae * 10 + fw) * s3];
            }
        }
    }
}

/// Per-degree byte-budget of the [`vrr_big`] arena buffer for VRR shape
/// `(NE, NF)`: `sizes[0]` is the bra e-table, `sizes[k]` (`1 ≤ k ≤ NF`) the
/// `f`-degree-`k` ket level, the rest `0`. Returns `(sizes, total)`. Pure
/// function of the const generics, so every value folds at compile time.
#[inline(always)]
fn big_sizes<const NE: usize, const NF: usize>() -> ([usize; 7], usize) {
    const NCART: [usize; 7] = [1, 3, 6, 10, 15, 21, 28];
    let lt = NE + NF;
    let n_e = n_addr(NE);
    let mut sizes = [0usize; 7];
    sizes[0] = n_e * (lt + 1); // e-table: addr(e) ∈ 0..n_e, stride lt+1
    let mut total = sizes[0];
    let mut k = 1;
    while k <= NF {
        let sk = lt - k + 1; // f-level k row length (m ∈ 0..=lt−k)
        sizes[k] = n_e * NCART[k] * sk;
        total += sizes[k];
        k += 1;
    }
    (sizes, total)
}

/// Contract a whole shell quartet of VRR shape `(NE, NF)` with `ne` or `nf` in
/// `4..=6` (both `≤ 6`) into `c_ef` using the extended monomorphized kernel
/// [`vrr_big`]. The d/f-heavy classes a cc-pVTZ+ build spends most of its kernel
/// time on, which the `≤ 3` [`contract_class`] does not cover and which therefore
/// fell onto the general m-marching [`vrr_primitive`].
///
/// Identical strip-mining (header→Boys→VRR two-pass) and bit-identical
/// accumulation order to [`contract_class`]; the only structural difference is
/// that the VRR working tables live in the caller's reused arena slab `big`
/// (sized by [`big_sizes`]) rather than stack arrays, since the `(ff|ff)` tables
/// are ~480 KB.
#[allow(clippy::too_many_arguments)]
fn contract_class_big<const NE: usize, const NF: usize>(
    bra_pairs: &[PrimPair],
    ket_pairs: &[PrimPair],
    c_ef: &mut [f64],
    big: &mut Vec<f64>,
) {
    let lt = NE + NF;
    let (_sizes, total) = big_sizes::<NE, NF>();
    ensure_len(big, total);
    let buf = &mut big[..total];
    let mut hdr = [QuartetHeader::default(); STRIP];
    // Boys row length is `lt+1 ≤ 4·MAX_L+1`; the generic bound keeps it uniform.
    let mut fm = [[0.0f64; 4 * MAX_L + 1]; STRIP];
    for bra in bra_pairs {
        let bra_coef = bra.c1 * bra.c2;
        let (p, pc) = (bra.zeta, bra.center);
        let mut start = 0;
        while start < ket_pairs.len() {
            let strip = &ket_pairs[start..(start + STRIP).min(ket_pairs.len())];
            // Phase 1 — headers + Boys rows (lanes in pairs through `boys_array2`),
            // exactly as in `contract_class`.
            let header = |h: &mut QuartetHeader, ket: &PrimPair| {
                let (q, qc) = (ket.zeta, ket.center);
                let pq = p + q;
                let rho = p * q / pq;
                let pq_vec = sub(pc, qc);
                h.t = rho * norm2(pq_vec);
                let w = [
                    (p * pc[0] + q * qc[0]) / pq,
                    (p * pc[1] + q * qc[1]) / pq,
                    (p * pc[2] + q * qc[2]) / pq,
                ];
                h.wp = sub(w, pc);
                h.wq = sub(w, qc);
                use std::f64::consts::PI;
                h.pref = 2.0 * PI * PI * PI.sqrt() / (p * q * pq.sqrt()) * bra.kappa * ket.kappa;
                h.inv_2pq = 0.5 / pq;
                h.q_over_pq = q / pq;
                h.p_over_pq = p / pq;
                h.scale = (bra_coef * ket.c1) * ket.c2;
            };
            let len = strip.len();
            let mut k = 0;
            while k + 2 <= len {
                header(&mut hdr[k], &strip[k]);
                header(&mut hdr[k + 1], &strip[k + 1]);
                let (f0, f1) = fm.split_at_mut(k + 1);
                boys_array2(
                    lt,
                    hdr[k].t,
                    hdr[k + 1].t,
                    &mut f0[k][..=lt],
                    &mut f1[0][..=lt],
                );
                k += 2;
            }
            if k < len {
                header(&mut hdr[k], &strip[k]);
                boys_array(lt, hdr[k].t, &mut fm[k][..=lt]);
            }
            // Phase 2 — VRR body per lane.
            for (k, ket) in strip.iter().enumerate() {
                vrr_big::<NE, NF>(bra, ket, &hdr[k], &fm[k], buf, c_ef);
            }
            start += STRIP;
        }
    }
}

/// One primitive quartet of the OS VRR for the extended classes `NE, NF ≤ 6`,
/// monomorphized per `(NE, NF)` like [`vrr_small`] (every loop bound, triple,
/// address, stride, and `has2`/cross branch is a compile-time constant), but with
/// the bra e-table and the `f`-degree levels carved out of the caller's arena
/// slab `buf` (laid out by [`big_sizes`]) instead of fixed stack arrays.
///
/// **Bit-identical to [`vrr_primitive`]/[`vrr_small`] by construction**: the same
/// recurrence expressions, term order and association, the same backward-liveness
/// `m`-ranges (`m ≤ NF − k` per `f`-level `k`, `m ≤ lt − de` per bra degree), and
/// the same `c_ef[ae·n_f + af] += scale·v` extraction. Guarded by the cross-engine
/// (`os_matches_rys_block_sweep` covers `(dd|ff)` and `(ff|ff)`) and golden tests.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn vrr_big<const NE: usize, const NF: usize>(
    bra: &PrimPair,
    ket: &PrimPair,
    hdr: &QuartetHeader,
    fm: &[f64],
    buf: &mut [f64],
    c_ef: &mut [f64],
) {
    let lt = NE + NF;
    let n_e = n_addr(NE);
    let n_f = n_addr(NF);
    // Row strides (compile-time constants per instantiation): e-table holds
    // m ∈ 0..=lt; f-level k holds m ∈ 0..=lt−k.
    let sa = lt + 1;
    let s1 = lt;
    let s2 = lt.saturating_sub(1);
    let s3 = lt.saturating_sub(2);
    let s4 = lt.saturating_sub(3);
    let s5 = lt.saturating_sub(4);
    let s6 = lt.saturating_sub(5);

    // Carve the arena slab into the e-table and the per-degree f-levels. Sizes
    // beyond NF are 0 (their `if NF >= k` blocks are compiled out), so the splits
    // hand back empty slices that are never indexed.
    let (sizes, _total) = big_sizes::<NE, NF>();
    let (ea, r) = buf.split_at_mut(sizes[0]);
    let (eb1, r) = r.split_at_mut(sizes[1]);
    let (eb2, r) = r.split_at_mut(sizes[2]);
    let (eb3, r) = r.split_at_mut(sizes[3]);
    let (eb4, r) = r.split_at_mut(sizes[4]);
    let (eb5, r) = r.split_at_mut(sizes[5]);
    let (eb6, _r) = r.split_at_mut(sizes[6]);
    // Debug: poison the reused tables so any out-of-range / stale read surfaces as
    // a NaN in the output (caught by the cross-engine + golden tests).
    #[cfg(debug_assertions)]
    {
        for b in [&mut *ea, eb1, eb2, eb3, eb4, eb5, eb6] {
            b.fill(f64::NAN);
        }
    }

    let pa = bra.r1; // P − A
    let qcen = ket.r1; // Q − C
    let (inv_2p, inv_2q) = (bra.inv_2zeta, ket.inv_2zeta);
    let QuartetHeader {
        wp,
        wq,
        pref,
        inv_2pq,
        q_over_pq,
        p_over_pq,
        scale,
        t: _,
    } = *hdr;
    // Base [00|00]^(m) = pref·F_m (e-component addr 0, stride sa → ea[m]).
    for m in 0..=lt {
        ea[m] = pref * fm[m];
    }

    // Phase A — bra ladder: [e0|00]^(m), e-degrees 1..=NE, m ≤ lt − de. The
    // e-component at `addr(te)` lives at `ea[addr(te)·sa + m]`.
    if NE >= 1 {
        for te in TRI1.iter() {
            let i = lower_axis(*te);
            let dst = addr(*te) * sa; // source is [0,0,0] at addr 0; no has2 at degree 1
            for m in 0..=(lt - 1) {
                ea[dst + m] = pa[i] * ea[m] + wp[i] * ea[m + 1];
            }
        }
    }
    if NE >= 2 {
        big_bra_degree(&TRI2, lt - 2, sa, pa, wp, q_over_pq, inv_2p, ea);
    }
    if NE >= 3 {
        big_bra_degree(&TRI3, lt - 3, sa, pa, wp, q_over_pq, inv_2p, ea);
    }
    if NE >= 4 {
        big_bra_degree(&TRI4, lt - 4, sa, pa, wp, q_over_pq, inv_2p, ea);
    }
    if NE >= 5 {
        big_bra_degree(&TRI5, lt - 5, sa, pa, wp, q_over_pq, inv_2p, ea);
    }
    if NE >= 6 {
        big_bra_degree(&TRI6, lt - 6, sa, pa, wp, q_over_pq, inv_2p, ea);
    }
    // Extract f-degree 0: contract m = 0 of every e-component.
    for ae in 0..n_e {
        c_ef[ae * n_f] += scale * ea[ae * sa];
    }

    // Phase B — ket raise. Per `f`-level `k`, the needed `m`-range is `m ≤ NF − k`
    // (backward liveness), independent of the e-degree — the final level (k = NF)
    // feeds only the m = 0 extraction, and each level below is read at m, m+1 by
    // the next level's raise. f-component `tf` of level k lives at
    // `ebk[(ae·n_cart(k) + cart_index(tf))·s_k + m]`.
    if NF >= 1 {
        // Level 1 raises from the e-table (level 0); no has2 (k−2 < 0).
        for (fw, tf) in TRI1.iter().enumerate() {
            let j = lower_axis(*tf);
            for ae in 0..n_e {
                let (te, _) = E_COMP[ae];
                let has_cross = te[j] >= 1;
                let cross_coef = te[j] as f64 * inv_2pq;
                let cs = if has_cross { addr(dec(te, j)) } else { 0 };
                for m in 0..=(NF - 1) {
                    let mut v = qcen[j] * ea[ae * sa + m] + wq[j] * ea[ae * sa + m + 1];
                    if has_cross {
                        v += cross_coef * ea[cs * sa + m + 1];
                    }
                    eb1[(ae * 3 + fw) * s1 + m] = v;
                }
            }
        }
        for ae in 0..n_e {
            for fw in 0..3 {
                c_ef[ae * n_f + 1 + fw] += scale * eb1[(ae * 3 + fw) * s1];
            }
        }
    }
    if NF >= 2 {
        // Level 2: k−1 source eb1, k−2 source the e-table (level 0).
        for (fw, tf) in TRI2.iter().enumerate() {
            let j = lower_axis(*tf);
            let lf1 = cart_index(dec(*tf, j));
            let has2 = tf[j] >= 2;
            let coef2 = ((tf[j] - 1) as f64) * inv_2q;
            for ae in 0..n_e {
                let (te, _) = E_COMP[ae];
                let has_cross = te[j] >= 1;
                let cross_coef = te[j] as f64 * inv_2pq;
                let cs = if has_cross { addr(dec(te, j)) } else { 0 };
                for m in 0..=(NF - 2) {
                    let mut v = qcen[j] * eb1[(ae * 3 + lf1) * s1 + m]
                        + wq[j] * eb1[(ae * 3 + lf1) * s1 + m + 1];
                    if has2 {
                        v += coef2 * (ea[ae * sa + m] - p_over_pq * ea[ae * sa + m + 1]);
                    }
                    if has_cross {
                        v += cross_coef * eb1[(cs * 3 + lf1) * s1 + m + 1];
                    }
                    eb2[(ae * 6 + fw) * s2 + m] = v;
                }
            }
        }
        for ae in 0..n_e {
            for fw in 0..6 {
                c_ef[ae * n_f + 4 + fw] += scale * eb2[(ae * 6 + fw) * s2];
            }
        }
    }
    if NF >= 3 {
        big_ket_degree(
            &TRI3,
            NF - 3,
            10,
            6,
            3,
            s3,
            s2,
            s1,
            inv_2q,
            inv_2pq,
            p_over_pq,
            qcen,
            wq,
            n_e,
            n_f,
            10,
            scale,
            eb2,
            eb1,
            eb3,
            c_ef,
        );
    }
    if NF >= 4 {
        big_ket_degree(
            &TRI4,
            NF - 4,
            15,
            10,
            6,
            s4,
            s3,
            s2,
            inv_2q,
            inv_2pq,
            p_over_pq,
            qcen,
            wq,
            n_e,
            n_f,
            20,
            scale,
            eb3,
            eb2,
            eb4,
            c_ef,
        );
    }
    if NF >= 5 {
        big_ket_degree(
            &TRI5,
            NF - 5,
            21,
            15,
            10,
            s5,
            s4,
            s3,
            inv_2q,
            inv_2pq,
            p_over_pq,
            qcen,
            wq,
            n_e,
            n_f,
            35,
            scale,
            eb4,
            eb3,
            eb5,
            c_ef,
        );
    }
    if NF >= 6 {
        big_ket_degree(
            &TRI6,
            NF - 6,
            28,
            21,
            15,
            s6,
            s5,
            s4,
            inv_2q,
            inv_2pq,
            p_over_pq,
            qcen,
            wq,
            n_e,
            n_f,
            56,
            scale,
            eb5,
            eb4,
            eb6,
            c_ef,
        );
    }
}

/// One bra-ladder degree of [`vrr_big`]'s Phase A: raise every triple in `tri`
/// (a `TRI{2..6}` table) into the e-table `ea`, with `mtop = lt − degree` the
/// (compile-time) top `m`. Factored out so degrees 2..6 share one body; the same
/// expressions/order as `vrr_small`'s degree-2/3 blocks.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn big_bra_degree(
    tri: &[[usize; 3]],
    mtop: usize,
    sa: usize,
    pa: Vec3,
    wp: [f64; 3],
    q_over_pq: f64,
    inv_2p: f64,
    ea: &mut [f64],
) {
    for te in tri.iter() {
        let i = lower_axis(*te);
        let s1a = addr(dec(*te, i)); // source degree − 1
        let has2 = te[i] >= 2;
        let coef2 = ((te[i] - 1) as f64) * inv_2p;
        let s2a = if has2 { addr(dec(dec(*te, i), i)) } else { 0 };
        let dst = addr(*te) * sa;
        for m in 0..=mtop {
            let mut v = pa[i] * ea[s1a * sa + m] + wp[i] * ea[s1a * sa + m + 1];
            if has2 {
                v += coef2 * (ea[s2a * sa + m] - q_over_pq * ea[s2a * sa + m + 1]);
            }
            ea[dst + m] = v;
        }
    }
}

/// One ket-raise degree `k ≥ 3` of [`vrr_big`]'s Phase B: build f-level `k` (table
/// `tri = TRI{k}`) into `dst` from the k−1 level `src1` and (for `has2`) the k−2
/// level `src2`, then extract its `m = 0` row into `c_ef` at base `cbase =
/// tri_below(k)`. `nck/nck1/nck2` are `n_cart` of degrees k/k−1/k−2 and
/// `sk/sk1/sk2` their row strides. Same recurrence as `vrr_small`'s degree-3
/// block, generalized over the level. (Degrees 1 and 2 are inlined in `vrr_big`
/// because their k−2 source is the e-table, not an `eb` level.)
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn big_ket_degree(
    tri: &[[usize; 3]],
    mtop: usize,
    nck: usize,
    nck1: usize,
    nck2: usize,
    sk: usize,
    sk1: usize,
    sk2: usize,
    inv_2q: f64,
    inv_2pq: f64,
    p_over_pq: f64,
    qcen: Vec3,
    wq: [f64; 3],
    n_e: usize,
    n_f: usize,
    cbase: usize,
    scale: f64,
    src1: &[f64], // k−1 level
    src2: &[f64], // k−2 level
    dst: &mut [f64],
    c_ef: &mut [f64],
) {
    for (fw, tf) in tri.iter().enumerate() {
        let j = lower_axis(*tf);
        let f1 = dec(*tf, j);
        let lf1 = cart_index(f1); // f-component index in level k−1
        let has2 = tf[j] >= 2;
        let coef2 = ((tf[j] - 1) as f64) * inv_2q;
        let lf2 = if has2 { cart_index(dec(f1, j)) } else { 0 }; // level k−2
        for ae in 0..n_e {
            let (te, _) = E_COMP[ae];
            let has_cross = te[j] >= 1;
            let cross_coef = te[j] as f64 * inv_2pq;
            let cs = if has_cross { addr(dec(te, j)) } else { 0 };
            for m in 0..=mtop {
                let mut v = qcen[j] * src1[(ae * nck1 + lf1) * sk1 + m]
                    + wq[j] * src1[(ae * nck1 + lf1) * sk1 + m + 1];
                if has2 {
                    v += coef2
                        * (src2[(ae * nck2 + lf2) * sk2 + m]
                            - p_over_pq * src2[(ae * nck2 + lf2) * sk2 + m + 1]);
                }
                if has_cross {
                    v += cross_coef * src1[(cs * nck1 + lf1) * sk1 + m + 1];
                }
                dst[(ae * nck + fw) * sk + m] = v;
            }
        }
    }
    for ae in 0..n_e {
        for fw in 0..nck {
            c_ef[ae * n_f + cbase + fw] += scale * dst[(ae * nck + fw) * sk];
        }
    }
}

// --- 4-lane batch across shell quartets (libint2-style class vectorization) ---

/// Four-wide value: one shell quartet per lane. All lane arithmetic below is
/// elementwise, so each quartet's numbers never mix across lanes — the batch
/// path is bit-identical to four scalar evaluations by construction.
type V4 = [f64; 4];

#[inline(always)]
fn v4_mul(a: V4, b: V4) -> V4 {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]]
}

#[inline(always)]
fn v4_add(a: V4, b: V4) -> V4 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

#[inline(always)]
fn v4_sub(a: V4, b: V4) -> V4 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}

#[inline(always)]
fn v4_scale(s: f64, a: V4) -> V4 {
    [s * a[0], s * a[1], s * a[2], s * a[3]]
}

/// Grow a [`V4`] arena to at least `n` lanes-of-four (the [`vrr_big4`] tables).
fn ensure_len_v4(v: &mut Vec<V4>, n: usize) {
    if v.len() < n {
        v.resize(n, [0.0; 4]);
    }
}

/// Lane-major [`QuartetHeader`] for the 4-quartet batch kernel: the same scalar
/// header fields, one lane per quartet. Each lane is filled by the exact
/// expression sequence of the scalar header (see [`contract_class`]).
#[derive(Clone, Copy, Default)]
struct QuartetHeader4 {
    wp: [V4; 3],
    wq: [V4; 3],
    pref: V4,
    inv_2pq: V4,
    q_over_pq: V4,
    p_over_pq: V4,
    scale: V4,
    t: V4,
}

/// Contract four shell quartets of the same VRR shape `(NE, NF)` (`NE, NF ≤ 3`)
/// in lockstep, one quartet per lane. Every lane must have the **same bra and
/// ket pair counts** (the caller buckets quartets that way), so the lanes run
/// the identical primitive loop with no padding — no `0·∞`/`-0.0` hazards.
///
/// Per lane the primitive order (bra outer, ket inner), header expressions,
/// Boys ladder ([`boys_array4`], lanes bitwise equal to [`boys_array`]) and the
/// VRR/accumulation arithmetic of [`vrr_small4`] are exactly the scalar
/// [`contract_class`]/[`vrr_small`] sequence, so each lane's `c_ef` is
/// bit-identical to a scalar evaluation of that quartet (fingerprint-guarded).
// `ib`/`ik` index four parallel pair slices in lockstep — a range loop is the
// clear form (no single iterable to zip without allocation).
#[allow(clippy::needless_range_loop)]
fn contract_class4<const NE: usize, const NF: usize>(
    bra_pairs: [&[PrimPair]; 4],
    ket_pairs: [&[PrimPair]; 4],
    c_ef: &mut [&mut [f64]; 4],
) {
    let lt = NE + NF;
    let n_bra = bra_pairs[0].len();
    let n_ket = ket_pairs[0].len();
    debug_assert!(bra_pairs.iter().all(|b| b.len() == n_bra));
    debug_assert!(ket_pairs.iter().all(|k| k.len() == n_ket));
    // Worst-case (NE,NF)=(3,3) sizes, the scalar `contract_class` tables widened
    // to V4 (~61 KB of stack total — fine on both the main thread and the
    // `EriBuilder` workers): e-table 20 components × m ∈ 0..=6; f-degree-1
    // level 20×3 × m ∈ 0..=5; f-degree-2 level 20×6 × m ∈ 0..=4; f-degree-3
    // level 20×10 × m ∈ 0..=3.
    let mut ea = [[0.0f64; 4]; 140];
    let mut eb1 = [[0.0f64; 4]; 360];
    let mut eb2 = [[0.0f64; 4]; 600];
    let mut eb3 = [[0.0f64; 4]; 800];
    let mut hdr = QuartetHeader4::default();
    let mut fm = [[0.0f64; 4]; 7];
    for ib in 0..n_bra {
        let bras = [
            &bra_pairs[0][ib],
            &bra_pairs[1][ib],
            &bra_pairs[2][ib],
            &bra_pairs[3][ib],
        ];
        let mut pa = [[0.0f64; 4]; 3];
        let mut inv_2p = [0.0f64; 4];
        let mut bra_coef = [0.0f64; 4];
        for (lane, bra) in bras.iter().enumerate() {
            for (dst, &src) in pa.iter_mut().zip(bra.r1.iter()) {
                dst[lane] = src;
            }
            inv_2p[lane] = bra.inv_2zeta;
            bra_coef[lane] = bra.c1 * bra.c2;
        }
        for ik in 0..n_ket {
            let kets = [
                &ket_pairs[0][ik],
                &ket_pairs[1][ik],
                &ket_pairs[2][ik],
                &ket_pairs[3][ik],
            ];
            let mut qcen = [[0.0f64; 4]; 3];
            let mut inv_2q = [0.0f64; 4];
            for (lane, (bra, ket)) in bras.iter().zip(kets.iter()).enumerate() {
                // The scalar header expressions of `contract_class`, verbatim,
                // one lane per quartet.
                let (p, pc) = (bra.zeta, bra.center);
                let (q, qc) = (ket.zeta, ket.center);
                let pq = p + q;
                let rho = p * q / pq;
                let pq_vec = sub(pc, qc); // P − Q
                hdr.t[lane] = rho * norm2(pq_vec);
                let w = [
                    (p * pc[0] + q * qc[0]) / pq,
                    (p * pc[1] + q * qc[1]) / pq,
                    (p * pc[2] + q * qc[2]) / pq,
                ];
                let wp = sub(w, pc);
                let wq = sub(w, qc);
                for ax in 0..3 {
                    hdr.wp[ax][lane] = wp[ax];
                    hdr.wq[ax][lane] = wq[ax];
                    qcen[ax][lane] = ket.r1[ax];
                }
                use std::f64::consts::PI;
                hdr.pref[lane] =
                    2.0 * PI * PI * PI.sqrt() / (p * q * pq.sqrt()) * bra.kappa * ket.kappa;
                hdr.inv_2pq[lane] = 0.5 / pq;
                hdr.q_over_pq[lane] = q / pq;
                hdr.p_over_pq[lane] = p / pq;
                hdr.scale[lane] = (bra_coef[lane] * ket.c1) * ket.c2;
                inv_2q[lane] = ket.inv_2zeta;
            }
            boys_array4(lt, hdr.t, &mut fm[..=lt]);
            vrr_small4::<NE, NF>(
                &pa, &qcen, inv_2p, inv_2q, &hdr, &fm, &mut ea, &mut eb1, &mut eb2, &mut eb3, c_ef,
            );
        }
    }
}

/// Four primitive quartets of the small-class OS VRR (`NE, NF ≤ 3`) in lockstep,
/// one quartet per [`V4`] lane — the 4-wide counterpart of [`vrr_small`].
///
/// **Bit-identical to [`vrr_small`] per lane by construction**: every statement
/// below is the scalar kernel's statement with each `f64` widened to a [`V4`]
/// whose lanes never interact — same expressions, same term order and
/// association, same m-ranges (the round-6/7 liveness trims included), and the
/// same per-lane `c_ef[ae·n_f + af] += scale·v` extraction order.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn vrr_small4<const NE: usize, const NF: usize>(
    pa: &[V4; 3],
    qcen: &[V4; 3],
    inv_2p: V4,
    inv_2q: V4,
    hdr: &QuartetHeader4,
    fm: &[V4; 7],
    ea: &mut [V4; 140],
    eb1: &mut [V4; 360],
    eb2: &mut [V4; 600],
    eb3: &mut [V4; 800],
    c_ef: &mut [&mut [f64]; 4],
) {
    const { assert!(NE <= 3 && NF <= 3, "batch kernel covers NE, NF <= 3") };
    let lt = NE + NF;
    let n_e = n_addr(NE);
    let n_f = n_addr(NF);
    let sa = lt + 1;
    let s1 = lt;
    let s2 = lt.saturating_sub(1);
    let s3 = lt.saturating_sub(2);
    let QuartetHeader4 {
        wp,
        wq,
        pref,
        inv_2pq,
        q_over_pq,
        p_over_pq,
        scale,
        t: _,
    } = *hdr;
    for m in 0..=lt {
        ea[m] = v4_mul(pref, fm[m]);
    }

    // Phase A — bra ladder: [e0|00]^(m), e-degrees 1..=NE, m ≤ lt − de.
    if NE >= 1 {
        for (iw, te) in TRI1.iter().enumerate() {
            let i = lower_axis(*te);
            for m in 0..=(lt - 1) {
                ea[(1 + iw) * sa + m] = v4_add(v4_mul(pa[i], ea[m]), v4_mul(wp[i], ea[m + 1]));
            }
        }
    }
    if NE >= 2 {
        for (iw, te) in TRI2.iter().enumerate() {
            let i = lower_axis(*te);
            let s1a = addr(dec(*te, i));
            let has2 = te[i] >= 2;
            let coef2 = v4_scale((te[i] - 1) as f64, inv_2p);
            let s2a = if has2 { addr(dec(dec(*te, i), i)) } else { 0 };
            for m in 0..=(lt - 2) {
                let mut v = v4_add(
                    v4_mul(pa[i], ea[s1a * sa + m]),
                    v4_mul(wp[i], ea[s1a * sa + m + 1]),
                );
                if has2 {
                    v = v4_add(
                        v,
                        v4_mul(
                            coef2,
                            v4_sub(ea[s2a * sa + m], v4_mul(q_over_pq, ea[s2a * sa + m + 1])),
                        ),
                    );
                }
                ea[(4 + iw) * sa + m] = v;
            }
        }
    }
    if NE >= 3 {
        for (iw, te) in TRI3.iter().enumerate() {
            let i = lower_axis(*te);
            let s1a = addr(dec(*te, i));
            let has2 = te[i] >= 2;
            let coef2 = v4_scale((te[i] - 1) as f64, inv_2p);
            let s2a = if has2 { addr(dec(dec(*te, i), i)) } else { 0 };
            for m in 0..=(lt - 3) {
                let mut v = v4_add(
                    v4_mul(pa[i], ea[s1a * sa + m]),
                    v4_mul(wp[i], ea[s1a * sa + m + 1]),
                );
                if has2 {
                    v = v4_add(
                        v,
                        v4_mul(
                            coef2,
                            v4_sub(ea[s2a * sa + m], v4_mul(q_over_pq, ea[s2a * sa + m + 1])),
                        ),
                    );
                }
                ea[(10 + iw) * sa + m] = v;
            }
        }
    }
    // Extract f-degree 0: contract m = 0 of every e-component, per lane.
    for ae in 0..n_e {
        let v = ea[ae * sa];
        for lane in 0..4 {
            c_ef[lane][ae * n_f] += scale[lane] * v[lane];
        }
    }

    // Phase B — ket raise, f-degree k: liveness m-range `m ≤ NF − k` (see
    // [`vrr_small`]'s Phase B comment for the derivation).
    if NF >= 1 {
        for (fw, tf) in TRI1.iter().enumerate() {
            let j = lower_axis(*tf);
            for ae in 0..n_e {
                let (te, _) = E_COMP[ae];
                let has_cross = te[j] >= 1;
                let cross_coef = v4_scale(te[j] as f64, inv_2pq);
                let cs = if has_cross { addr(dec(te, j)) } else { 0 };
                let mtop = NF - 1;
                for m in 0..=mtop {
                    let mut v = v4_add(
                        v4_mul(qcen[j], ea[ae * sa + m]),
                        v4_mul(wq[j], ea[ae * sa + m + 1]),
                    );
                    if has_cross {
                        v = v4_add(v, v4_mul(cross_coef, ea[cs * sa + m + 1]));
                    }
                    eb1[(ae * 3 + fw) * s1 + m] = v;
                }
            }
        }
        for ae in 0..n_e {
            for fw in 0..3 {
                let v = eb1[(ae * 3 + fw) * s1];
                for lane in 0..4 {
                    c_ef[lane][ae * n_f + 1 + fw] += scale[lane] * v[lane];
                }
            }
        }
    }

    if NF >= 2 {
        for (fw, tf) in TRI2.iter().enumerate() {
            let j = lower_axis(*tf);
            let f1 = dec(*tf, j);
            let lf1 = cart_index(f1);
            let has2 = tf[j] >= 2; // its k−2 source is then [0,0,0] (phase A)
            let coef2 = v4_scale((tf[j] - 1) as f64, inv_2q);
            for ae in 0..n_e {
                let (te, _) = E_COMP[ae];
                let has_cross = te[j] >= 1;
                let cross_coef = v4_scale(te[j] as f64, inv_2pq);
                let cs = if has_cross { addr(dec(te, j)) } else { 0 };
                let mtop = NF - 2;
                for m in 0..=mtop {
                    let mut v = v4_add(
                        v4_mul(qcen[j], eb1[(ae * 3 + lf1) * s1 + m]),
                        v4_mul(wq[j], eb1[(ae * 3 + lf1) * s1 + m + 1]),
                    );
                    if has2 {
                        v = v4_add(
                            v,
                            v4_mul(
                                coef2,
                                v4_sub(ea[ae * sa + m], v4_mul(p_over_pq, ea[ae * sa + m + 1])),
                            ),
                        );
                    }
                    if has_cross {
                        v = v4_add(v, v4_mul(cross_coef, eb1[(cs * 3 + lf1) * s1 + m + 1]));
                    }
                    eb2[(ae * 6 + fw) * s2 + m] = v;
                }
            }
        }
        for ae in 0..n_e {
            for fw in 0..6 {
                let v = eb2[(ae * 6 + fw) * s2];
                for lane in 0..4 {
                    c_ef[lane][ae * n_f + 4 + fw] += scale[lane] * v[lane];
                }
            }
        }
    }

    // Phase B — ket raise, f-degree 3: always the final f-level (NF ≤ 3), so
    // only m = 0 is computed (the scalar `vrr_small` block, widened to V4).
    if NF >= 3 {
        for (fw, tf) in TRI3.iter().enumerate() {
            let j = lower_axis(*tf);
            let f1 = dec(*tf, j);
            let lf1 = cart_index(f1); // f-component index in the degree-2 level
            let has2 = tf[j] >= 2; // its k−2 source is f1 lowered again (degree 1)
            let coef2 = v4_scale((tf[j] - 1) as f64, inv_2q);
            let lf2 = if has2 { cart_index(dec(f1, j)) } else { 0 };
            for ae in 0..n_e {
                let (te, _) = E_COMP[ae];
                let has_cross = te[j] >= 1;
                let cross_coef = v4_scale(te[j] as f64, inv_2pq);
                let cs = if has_cross { addr(dec(te, j)) } else { 0 };
                let mtop = 0; // final level: m = 0 only
                for m in 0..=mtop {
                    let mut v = v4_add(
                        v4_mul(qcen[j], eb2[(ae * 6 + lf1) * s2 + m]),
                        v4_mul(wq[j], eb2[(ae * 6 + lf1) * s2 + m + 1]),
                    );
                    if has2 {
                        v = v4_add(
                            v,
                            v4_mul(
                                coef2,
                                v4_sub(
                                    eb1[(ae * 3 + lf2) * s1 + m],
                                    v4_mul(p_over_pq, eb1[(ae * 3 + lf2) * s1 + m + 1]),
                                ),
                            ),
                        );
                    }
                    if has_cross {
                        v = v4_add(v, v4_mul(cross_coef, eb2[(cs * 6 + lf1) * s2 + m + 1]));
                    }
                    eb3[(ae * 10 + fw) * s3 + m] = v;
                }
            }
        }
        for ae in 0..n_e {
            for fw in 0..10 {
                let v = eb3[(ae * 10 + fw) * s3];
                for lane in 0..4 {
                    c_ef[lane][ae * n_f + 10 + fw] += scale[lane] * v[lane];
                }
            }
        }
    }
}

/// Contract four shell quartets of the same VRR shape `(NE, NF)` with `ne` or
/// `nf` in `4..=6` in lockstep — the 4-lane counterpart of [`contract_class_big`]
/// (and the d/f extension of [`contract_class4`]). Per lane bit-identical to a
/// scalar [`vrr_big`] evaluation: every value is the scalar kernel's, widened to a
/// [`V4`] whose lanes never interact. The VRR tables live in the reused `big4`
/// arena slab (the [`vrr_big`] layout widened to `V4`), not on the stack.
#[allow(clippy::needless_range_loop)]
#[allow(clippy::too_many_arguments)]
fn contract_class_big4<const NE: usize, const NF: usize>(
    bra_pairs: [&[PrimPair]; 4],
    ket_pairs: [&[PrimPair]; 4],
    c_ef: &mut [&mut [f64]; 4],
    big4: &mut Vec<V4>,
) {
    let lt = NE + NF;
    let n_bra = bra_pairs[0].len();
    let n_ket = ket_pairs[0].len();
    debug_assert!(bra_pairs.iter().all(|b| b.len() == n_bra));
    debug_assert!(ket_pairs.iter().all(|k| k.len() == n_ket));
    let (_sizes, total) = big_sizes::<NE, NF>();
    ensure_len_v4(big4, total);
    let buf = &mut big4[..total];
    let mut hdr = QuartetHeader4::default();
    let mut fm = [[0.0f64; 4]; 4 * MAX_L + 1];
    for ib in 0..n_bra {
        let bras = [
            &bra_pairs[0][ib],
            &bra_pairs[1][ib],
            &bra_pairs[2][ib],
            &bra_pairs[3][ib],
        ];
        let mut pa = [[0.0f64; 4]; 3];
        let mut inv_2p = [0.0f64; 4];
        let mut bra_coef = [0.0f64; 4];
        for (lane, bra) in bras.iter().enumerate() {
            for (dst, &src) in pa.iter_mut().zip(bra.r1.iter()) {
                dst[lane] = src;
            }
            inv_2p[lane] = bra.inv_2zeta;
            bra_coef[lane] = bra.c1 * bra.c2;
        }
        for ik in 0..n_ket {
            let kets = [
                &ket_pairs[0][ik],
                &ket_pairs[1][ik],
                &ket_pairs[2][ik],
                &ket_pairs[3][ik],
            ];
            let mut qcen = [[0.0f64; 4]; 3];
            let mut inv_2q = [0.0f64; 4];
            for (lane, (bra, ket)) in bras.iter().zip(kets.iter()).enumerate() {
                // The scalar header expressions of `contract_class_big`, verbatim,
                // one lane per quartet (identical to `contract_class4`'s header).
                let (p, pc) = (bra.zeta, bra.center);
                let (q, qc) = (ket.zeta, ket.center);
                let pq = p + q;
                let rho = p * q / pq;
                let pq_vec = sub(pc, qc);
                hdr.t[lane] = rho * norm2(pq_vec);
                let w = [
                    (p * pc[0] + q * qc[0]) / pq,
                    (p * pc[1] + q * qc[1]) / pq,
                    (p * pc[2] + q * qc[2]) / pq,
                ];
                let wp = sub(w, pc);
                let wq = sub(w, qc);
                for ax in 0..3 {
                    hdr.wp[ax][lane] = wp[ax];
                    hdr.wq[ax][lane] = wq[ax];
                    qcen[ax][lane] = ket.r1[ax];
                }
                use std::f64::consts::PI;
                hdr.pref[lane] =
                    2.0 * PI * PI * PI.sqrt() / (p * q * pq.sqrt()) * bra.kappa * ket.kappa;
                hdr.inv_2pq[lane] = 0.5 / pq;
                hdr.q_over_pq[lane] = q / pq;
                hdr.p_over_pq[lane] = p / pq;
                hdr.scale[lane] = (bra_coef[lane] * ket.c1) * ket.c2;
                inv_2q[lane] = ket.inv_2zeta;
            }
            boys_array4(lt, hdr.t, &mut fm[..=lt]);
            vrr_big4::<NE, NF>(&pa, &qcen, inv_2p, inv_2q, &hdr, &fm, buf, c_ef);
        }
    }
}

/// Four primitive quartets of the d/f OS VRR (`NE, NF ≤ 6`) in lockstep, one
/// quartet per [`V4`] lane — the 4-wide counterpart of [`vrr_big`]. Bit-identical
/// to [`vrr_big`] per lane (every `f64` widened to a [`V4`]; same expressions,
/// term order, m-ranges and per-lane extraction). Tables carved from the arena
/// `buf` by [`big_sizes`], exactly as in [`vrr_big`].
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn vrr_big4<const NE: usize, const NF: usize>(
    pa: &[V4; 3],
    qcen: &[V4; 3],
    inv_2p: V4,
    inv_2q: V4,
    hdr: &QuartetHeader4,
    fm: &[V4],
    buf: &mut [V4],
    c_ef: &mut [&mut [f64]; 4],
) {
    let lt = NE + NF;
    let n_e = n_addr(NE);
    let n_f = n_addr(NF);
    let sa = lt + 1;
    let s1 = lt;
    let s2 = lt.saturating_sub(1);
    let s3 = lt.saturating_sub(2);
    let s4 = lt.saturating_sub(3);
    let s5 = lt.saturating_sub(4);
    let s6 = lt.saturating_sub(5);
    let (sizes, _total) = big_sizes::<NE, NF>();
    let (ea, r) = buf.split_at_mut(sizes[0]);
    let (eb1, r) = r.split_at_mut(sizes[1]);
    let (eb2, r) = r.split_at_mut(sizes[2]);
    let (eb3, r) = r.split_at_mut(sizes[3]);
    let (eb4, r) = r.split_at_mut(sizes[4]);
    let (eb5, r) = r.split_at_mut(sizes[5]);
    let (eb6, _r) = r.split_at_mut(sizes[6]);
    #[cfg(debug_assertions)]
    {
        for b in [&mut *ea, eb1, eb2, eb3, eb4, eb5, eb6] {
            b.fill([f64::NAN; 4]);
        }
    }
    let QuartetHeader4 {
        wp,
        wq,
        pref,
        inv_2pq,
        q_over_pq,
        p_over_pq,
        scale,
        t: _,
    } = *hdr;
    for m in 0..=lt {
        ea[m] = v4_mul(pref, fm[m]);
    }

    // Phase A — bra ladder.
    if NE >= 1 {
        for te in TRI1.iter() {
            let i = lower_axis(*te);
            let dst = addr(*te) * sa;
            for m in 0..=(lt - 1) {
                ea[dst + m] = v4_add(v4_mul(pa[i], ea[m]), v4_mul(wp[i], ea[m + 1]));
            }
        }
    }
    if NE >= 2 {
        big_bra_degree4(&TRI2, lt - 2, sa, pa, &wp, q_over_pq, inv_2p, ea);
    }
    if NE >= 3 {
        big_bra_degree4(&TRI3, lt - 3, sa, pa, &wp, q_over_pq, inv_2p, ea);
    }
    if NE >= 4 {
        big_bra_degree4(&TRI4, lt - 4, sa, pa, &wp, q_over_pq, inv_2p, ea);
    }
    if NE >= 5 {
        big_bra_degree4(&TRI5, lt - 5, sa, pa, &wp, q_over_pq, inv_2p, ea);
    }
    if NE >= 6 {
        big_bra_degree4(&TRI6, lt - 6, sa, pa, &wp, q_over_pq, inv_2p, ea);
    }
    for ae in 0..n_e {
        let v = ea[ae * sa];
        for lane in 0..4 {
            c_ef[lane][ae * n_f] += scale[lane] * v[lane];
        }
    }

    // Phase B — ket raise.
    if NF >= 1 {
        for (fw, tf) in TRI1.iter().enumerate() {
            let j = lower_axis(*tf);
            for ae in 0..n_e {
                let (te, _) = E_COMP[ae];
                let has_cross = te[j] >= 1;
                let cross_coef = v4_scale(te[j] as f64, inv_2pq);
                let cs = if has_cross { addr(dec(te, j)) } else { 0 };
                for m in 0..=(NF - 1) {
                    let mut v = v4_add(
                        v4_mul(qcen[j], ea[ae * sa + m]),
                        v4_mul(wq[j], ea[ae * sa + m + 1]),
                    );
                    if has_cross {
                        v = v4_add(v, v4_mul(cross_coef, ea[cs * sa + m + 1]));
                    }
                    eb1[(ae * 3 + fw) * s1 + m] = v;
                }
            }
        }
        for ae in 0..n_e {
            for fw in 0..3 {
                let v = eb1[(ae * 3 + fw) * s1];
                for lane in 0..4 {
                    c_ef[lane][ae * n_f + 1 + fw] += scale[lane] * v[lane];
                }
            }
        }
    }
    if NF >= 2 {
        for (fw, tf) in TRI2.iter().enumerate() {
            let j = lower_axis(*tf);
            let lf1 = cart_index(dec(*tf, j));
            let has2 = tf[j] >= 2;
            let coef2 = v4_scale((tf[j] - 1) as f64, inv_2q);
            for ae in 0..n_e {
                let (te, _) = E_COMP[ae];
                let has_cross = te[j] >= 1;
                let cross_coef = v4_scale(te[j] as f64, inv_2pq);
                let cs = if has_cross { addr(dec(te, j)) } else { 0 };
                for m in 0..=(NF - 2) {
                    let mut v = v4_add(
                        v4_mul(qcen[j], eb1[(ae * 3 + lf1) * s1 + m]),
                        v4_mul(wq[j], eb1[(ae * 3 + lf1) * s1 + m + 1]),
                    );
                    if has2 {
                        v = v4_add(
                            v,
                            v4_mul(
                                coef2,
                                v4_sub(ea[ae * sa + m], v4_mul(p_over_pq, ea[ae * sa + m + 1])),
                            ),
                        );
                    }
                    if has_cross {
                        v = v4_add(v, v4_mul(cross_coef, eb1[(cs * 3 + lf1) * s1 + m + 1]));
                    }
                    eb2[(ae * 6 + fw) * s2 + m] = v;
                }
            }
        }
        for ae in 0..n_e {
            for fw in 0..6 {
                let v = eb2[(ae * 6 + fw) * s2];
                for lane in 0..4 {
                    c_ef[lane][ae * n_f + 4 + fw] += scale[lane] * v[lane];
                }
            }
        }
    }
    if NF >= 3 {
        big_ket_degree4(
            &TRI3,
            NF - 3,
            10,
            6,
            3,
            s3,
            s2,
            s1,
            inv_2q,
            inv_2pq,
            p_over_pq,
            qcen,
            wq,
            n_e,
            n_f,
            10,
            scale,
            eb2,
            eb1,
            eb3,
            c_ef,
        );
    }
    if NF >= 4 {
        big_ket_degree4(
            &TRI4,
            NF - 4,
            15,
            10,
            6,
            s4,
            s3,
            s2,
            inv_2q,
            inv_2pq,
            p_over_pq,
            qcen,
            wq,
            n_e,
            n_f,
            20,
            scale,
            eb3,
            eb2,
            eb4,
            c_ef,
        );
    }
    if NF >= 5 {
        big_ket_degree4(
            &TRI5,
            NF - 5,
            21,
            15,
            10,
            s5,
            s4,
            s3,
            inv_2q,
            inv_2pq,
            p_over_pq,
            qcen,
            wq,
            n_e,
            n_f,
            35,
            scale,
            eb4,
            eb3,
            eb5,
            c_ef,
        );
    }
    if NF >= 6 {
        big_ket_degree4(
            &TRI6,
            NF - 6,
            28,
            21,
            15,
            s6,
            s5,
            s4,
            inv_2q,
            inv_2pq,
            p_over_pq,
            qcen,
            wq,
            n_e,
            n_f,
            56,
            scale,
            eb5,
            eb4,
            eb6,
            c_ef,
        );
    }
}

/// 4-lane counterpart of [`big_bra_degree`] (Phase A degrees 2..6), widened to
/// [`V4`].
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn big_bra_degree4(
    tri: &[[usize; 3]],
    mtop: usize,
    sa: usize,
    pa: &[V4; 3],
    wp: &[V4; 3],
    q_over_pq: V4,
    inv_2p: V4,
    ea: &mut [V4],
) {
    for te in tri.iter() {
        let i = lower_axis(*te);
        let s1a = addr(dec(*te, i));
        let has2 = te[i] >= 2;
        let coef2 = v4_scale((te[i] - 1) as f64, inv_2p);
        let s2a = if has2 { addr(dec(dec(*te, i), i)) } else { 0 };
        let dst = addr(*te) * sa;
        for m in 0..=mtop {
            let mut v = v4_add(
                v4_mul(pa[i], ea[s1a * sa + m]),
                v4_mul(wp[i], ea[s1a * sa + m + 1]),
            );
            if has2 {
                v = v4_add(
                    v,
                    v4_mul(
                        coef2,
                        v4_sub(ea[s2a * sa + m], v4_mul(q_over_pq, ea[s2a * sa + m + 1])),
                    ),
                );
            }
            ea[dst + m] = v;
        }
    }
}

/// 4-lane counterpart of [`big_ket_degree`] (Phase B degrees `k ≥ 3`), widened to
/// [`V4`]; extracts each level's `m = 0` row into the per-lane `c_ef`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn big_ket_degree4(
    tri: &[[usize; 3]],
    mtop: usize,
    nck: usize,
    nck1: usize,
    nck2: usize,
    sk: usize,
    sk1: usize,
    sk2: usize,
    inv_2q: V4,
    inv_2pq: V4,
    p_over_pq: V4,
    qcen: &[V4; 3],
    wq: [V4; 3],
    n_e: usize,
    n_f: usize,
    cbase: usize,
    scale: V4,
    src1: &[V4],
    src2: &[V4],
    dst: &mut [V4],
    c_ef: &mut [&mut [f64]; 4],
) {
    for (fw, tf) in tri.iter().enumerate() {
        let j = lower_axis(*tf);
        let f1 = dec(*tf, j);
        let lf1 = cart_index(f1);
        let has2 = tf[j] >= 2;
        let coef2 = v4_scale((tf[j] - 1) as f64, inv_2q);
        let lf2 = if has2 { cart_index(dec(f1, j)) } else { 0 };
        for ae in 0..n_e {
            let (te, _) = E_COMP[ae];
            let has_cross = te[j] >= 1;
            let cross_coef = v4_scale(te[j] as f64, inv_2pq);
            let cs = if has_cross { addr(dec(te, j)) } else { 0 };
            for m in 0..=mtop {
                let mut v = v4_add(
                    v4_mul(qcen[j], src1[(ae * nck1 + lf1) * sk1 + m]),
                    v4_mul(wq[j], src1[(ae * nck1 + lf1) * sk1 + m + 1]),
                );
                if has2 {
                    v = v4_add(
                        v,
                        v4_mul(
                            coef2,
                            v4_sub(
                                src2[(ae * nck2 + lf2) * sk2 + m],
                                v4_mul(p_over_pq, src2[(ae * nck2 + lf2) * sk2 + m + 1]),
                            ),
                        ),
                    );
                }
                if has_cross {
                    v = v4_add(v, v4_mul(cross_coef, src1[(cs * nck1 + lf1) * sk1 + m + 1]));
                }
                dst[(ae * nck + fw) * sk + m] = v;
            }
        }
    }
    for ae in 0..n_e {
        for fw in 0..nck {
            let v = dst[(ae * nck + fw) * sk];
            for lane in 0..4 {
                c_ef[lane][ae * n_f + cbase + fw] += scale[lane] * v[lane];
            }
        }
    }
}

/// Four `(ss|ss)` shell quartets in lockstep — the 4-lane counterpart of the
/// scalar `(ss|ss)` fast path in [`coulomb_shell_into_scratch`]: per lane the
/// same `t`/`pref` expressions, the same `((c_a·c_b)·c_c)·c_d` scale, the same
/// `scale·(pref·F_0)` term and the same bra-outer/ket-inner accumulation order
/// into a single per-lane register, so each lane is bit-identical to the
/// scalar path. Lanes must have equal bra/ket pair counts (caller-bucketed).
#[allow(clippy::needless_range_loop)] // ib/ik index four parallel slices in lockstep
fn contract_ssss4(
    bra_pairs: [&[PrimPair]; 4],
    ket_pairs: [&[PrimPair]; 4],
    outs: &mut [&mut [f64]; 4],
) {
    let n_bra = bra_pairs[0].len();
    let n_ket = ket_pairs[0].len();
    debug_assert!(bra_pairs.iter().all(|b| b.len() == n_bra));
    debug_assert!(ket_pairs.iter().all(|k| k.len() == n_ket));
    let two_pi_2_5 =
        2.0 * std::f64::consts::PI * std::f64::consts::PI * std::f64::consts::PI.sqrt();
    let mut acc = [0.0f64; 4];
    let mut fm = [[0.0f64; 4]; 1];
    for ib in 0..n_bra {
        let bras = [
            &bra_pairs[0][ib],
            &bra_pairs[1][ib],
            &bra_pairs[2][ib],
            &bra_pairs[3][ib],
        ];
        let mut bc = [0.0f64; 4];
        for (lane, bra) in bras.iter().enumerate() {
            bc[lane] = bra.c1 * bra.c2;
        }
        for ik in 0..n_ket {
            let kets = [
                &ket_pairs[0][ik],
                &ket_pairs[1][ik],
                &ket_pairs[2][ik],
                &ket_pairs[3][ik],
            ];
            let mut t = [0.0f64; 4];
            let mut pref = [0.0f64; 4];
            for (lane, (bra, ket)) in bras.iter().zip(kets.iter()).enumerate() {
                let p = bra.zeta;
                let q = ket.zeta;
                let pq = p + q;
                t[lane] = (p * q / pq) * dist2(bra.center, ket.center);
                pref[lane] = two_pi_2_5 / (p * q * pq.sqrt()) * bra.kappa * ket.kappa;
            }
            boys_array4(0, t, &mut fm);
            for (lane, ket) in kets.iter().enumerate() {
                acc[lane] += (bc[lane] * ket.c1) * ket.c2 * (pref[lane] * fm[0][lane]);
            }
        }
    }
    for (lane, out) in outs.iter_mut().enumerate() {
        out[0] += acc[lane];
    }
}

/// Number of primitive pairs of two shells that survive the
/// `PAIR_NEGLIGIBLE` screen — the pair count `build_pairs` would produce.
/// Drivers bucketing shell quartets for [`coulomb_shell_batch4_into_scratch`]
/// use this (once per shell pair) to group quartets whose lanes run the
/// primitive loop in true lockstep.
#[must_use]
pub fn surviving_pair_count(s1: ShellRef<'_>, s2: ShellRef<'_>) -> usize {
    let d2 = dist2(s1.center, s2.center);
    let mut n = 0;
    for (&e1, &c1) in s1.exps.iter().zip(s1.coeffs.iter()) {
        for (&e2, &c2) in s2.exps.iter().zip(s2.coeffs.iter()) {
            let zeta = e1 + e2;
            let kappa = (-(e1 * e2 / zeta) * d2).exp();
            if kappa * (c1 * c2).abs() >= PAIR_NEGLIGIBLE {
                n += 1;
            }
        }
    }
    n
}

/// Reusable buffers for the 4-quartet batch entry
/// [`coulomb_shell_batch4_into_scratch`]: per-lane pair lists and `c_ef`
/// accumulators, plus a scalar [`EriScratch`] for the per-lane HRR (and the
/// defensive scalar fallback).
#[derive(Debug, Default)]
pub struct EriBatch4Scratch {
    bra_pairs: [Vec<PrimPair>; 4],
    ket_pairs: [Vec<PrimPair>; 4],
    c_ef: [Vec<f64>; 4],
    /// 4-lane VRR working tables for the d/f batch kernel [`vrr_big4`] (`ne` or
    /// `nf` in `4..=6`): the [`vrr_big`] arena slab, widened to [`V4`]. Grown on
    /// demand, reused across quartets; the `(ff|ff)` worst case is ~1.9 MB (4×
    /// the scalar 480 KB), past any sane stack budget.
    big4: Vec<V4>,
    scalar: EriScratch,
}

/// Evaluate **four shell quartets in lockstep** — `quartets[lane] = [a, b, c, d]`
/// — accumulating each lane's contracted Cartesian block into `outs[lane]`
/// (same layout/contract as [`coulomb_shell_into`]).
///
/// All four quartets must share the VRR shape `(ne, nf) = (la+lb, lc+ld)` with
/// `ne, nf ≤ 3`, and have equal surviving bra and ket pair
/// counts (see [`surviving_pair_count`]) so the lanes run the primitive loop in
/// true lockstep. Quartets violating that (which a class-bucketing driver never
/// sends) are evaluated through the scalar path instead — same values either
/// way: each lane's result is **bit-identical** to a [`coulomb_shell_into`]
/// call for that quartet.
///
/// # Panics
/// Panics (in debug builds) if any `l > MAX_L` or any `outs[lane]` is too short.
pub fn coulomb_shell_batch4_into_scratch(
    scratch: &mut EriBatch4Scratch,
    quartets: &[[ShellRef<'_>; 4]; 4],
    outs: &mut [&mut [f64]; 4],
) {
    // Build the per-lane pair lists into the reused buffers (moved out for the
    // duration of the core call, like the scalar entry), then run the shared
    // core on borrowed slices.
    let mut bra_pairs = std::mem::take(&mut scratch.bra_pairs);
    let mut ket_pairs = std::mem::take(&mut scratch.ket_pairs);
    for (lane, &[a, b, c, d]) in quartets.iter().enumerate() {
        build_pairs(&mut bra_pairs[lane], a, b);
        build_pairs(&mut ket_pairs[lane], c, d);
    }
    let bra = [
        &bra_pairs[0][..],
        &bra_pairs[1][..],
        &bra_pairs[2][..],
        &bra_pairs[3][..],
    ];
    let ket = [
        &ket_pairs[0][..],
        &ket_pairs[1][..],
        &ket_pairs[2][..],
        &ket_pairs[3][..],
    ];
    coulomb_shell_batch4_core(scratch, quartets, bra, ket, outs);
    scratch.bra_pairs = bra_pairs;
    scratch.ket_pairs = ket_pairs;
}

/// Like [`coulomb_shell_batch4_into_scratch`] but with each lane's bra/ket
/// primitive-pair data supplied by the caller (precomputed once per shell pair
/// across a dense build, see [`ShellPairData`]). Bit-identical to the
/// self-building entry: same pair values and order, only computed elsewhere.
///
/// # Panics
/// Panics (in debug builds) if any `l > MAX_L` or any `outs[lane]` is too short.
pub fn coulomb_shell_batch4_pairs_into_scratch(
    scratch: &mut EriBatch4Scratch,
    quartets: &[[ShellRef<'_>; 4]; 4],
    bra_pairs: [&ShellPairData; 4],
    ket_pairs: [&ShellPairData; 4],
    outs: &mut [&mut [f64]; 4],
) {
    coulomb_shell_batch4_core(
        scratch,
        quartets,
        bra_pairs.map(|p| &p.pairs[..]),
        ket_pairs.map(|p| &p.pairs[..]),
        outs,
    );
}

/// The 4-lane batch kernel shared by the self-building and borrowed-pairs
/// entries. Lanes that violate the lockstep contract (mismatched VRR shape,
/// `ne`/`nf > 3`, or unequal pair counts — which a class-bucketing driver never
/// sends) drain through the scalar core with the same pair lists.
fn coulomb_shell_batch4_core(
    scratch: &mut EriBatch4Scratch,
    quartets: &[[ShellRef<'_>; 4]; 4],
    bra: [&[PrimPair]; 4],
    ket: [&[PrimPair]; 4],
    outs: &mut [&mut [f64]; 4],
) {
    let [a0, b0, c0, d0] = quartets[0];
    let ne = a0.l + b0.l;
    let nf = c0.l + d0.l;
    let lanes_match = quartets
        .iter()
        .all(|[a, b, c, d]| a.l + b.l == ne && c.l + d.l == nf);
    let n_bra = bra[0].len();
    let n_ket = ket[0].len();
    let lockstep = lanes_match
        && ne <= 6
        && nf <= 6
        && bra.iter().all(|p| p.len() == n_bra)
        && ket.iter().all(|p| p.len() == n_ket);
    if !lockstep {
        for (lane, &[a, b, c, d]) in quartets.iter().enumerate() {
            coulomb_shell_core(
                &mut scratch.scalar,
                a,
                b,
                c,
                d,
                bra[lane],
                ket[lane],
                outs[lane],
            );
        }
        return;
    }

    // (ss|ss): the 4-lane fast path (no VRR/HRR scaffolding, like the scalar
    // fast path).
    if ne + nf == 0 {
        contract_ssss4(bra, ket, outs);
        return;
    }

    let n_e = n_addr(ne);
    let n_f = n_addr(nf);
    for lane in 0..4 {
        ensure_len(&mut scratch.c_ef[lane], n_e * n_f);
        scratch.c_ef[lane][..n_e * n_f].fill(0.0);
    }
    {
        let [c0, c1, c2, c3] = &mut scratch.c_ef;
        let big4 = &mut scratch.big4;
        let mut c_ef: [&mut [f64]; 4] = [
            &mut c0[..n_e * n_f],
            &mut c1[..n_e * n_f],
            &mut c2[..n_e * n_f],
            &mut c3[..n_e * n_f],
        ];
        match (ne, nf) {
            (0, 1) => contract_class4::<0, 1>(bra, ket, &mut c_ef),
            (1, 0) => contract_class4::<1, 0>(bra, ket, &mut c_ef),
            (1, 1) => contract_class4::<1, 1>(bra, ket, &mut c_ef),
            (0, 2) => contract_class4::<0, 2>(bra, ket, &mut c_ef),
            (2, 0) => contract_class4::<2, 0>(bra, ket, &mut c_ef),
            (1, 2) => contract_class4::<1, 2>(bra, ket, &mut c_ef),
            (2, 1) => contract_class4::<2, 1>(bra, ket, &mut c_ef),
            (2, 2) => contract_class4::<2, 2>(bra, ket, &mut c_ef),
            (0, 3) => contract_class4::<0, 3>(bra, ket, &mut c_ef),
            (3, 0) => contract_class4::<3, 0>(bra, ket, &mut c_ef),
            (1, 3) => contract_class4::<1, 3>(bra, ket, &mut c_ef),
            (3, 1) => contract_class4::<3, 1>(bra, ket, &mut c_ef),
            (2, 3) => contract_class4::<2, 3>(bra, ket, &mut c_ef),
            (3, 2) => contract_class4::<3, 2>(bra, ket, &mut c_ef),
            (3, 3) => contract_class4::<3, 3>(bra, ket, &mut c_ef),
            // d/f classes (one of ne/nf in 4..=6): 4-lane d/f kernel.
            (0, 4) => contract_class_big4::<0, 4>(bra, ket, &mut c_ef, big4),
            (0, 5) => contract_class_big4::<0, 5>(bra, ket, &mut c_ef, big4),
            (0, 6) => contract_class_big4::<0, 6>(bra, ket, &mut c_ef, big4),
            (1, 4) => contract_class_big4::<1, 4>(bra, ket, &mut c_ef, big4),
            (1, 5) => contract_class_big4::<1, 5>(bra, ket, &mut c_ef, big4),
            (1, 6) => contract_class_big4::<1, 6>(bra, ket, &mut c_ef, big4),
            (2, 4) => contract_class_big4::<2, 4>(bra, ket, &mut c_ef, big4),
            (2, 5) => contract_class_big4::<2, 5>(bra, ket, &mut c_ef, big4),
            (2, 6) => contract_class_big4::<2, 6>(bra, ket, &mut c_ef, big4),
            (3, 4) => contract_class_big4::<3, 4>(bra, ket, &mut c_ef, big4),
            (3, 5) => contract_class_big4::<3, 5>(bra, ket, &mut c_ef, big4),
            (3, 6) => contract_class_big4::<3, 6>(bra, ket, &mut c_ef, big4),
            (4, 0) => contract_class_big4::<4, 0>(bra, ket, &mut c_ef, big4),
            (4, 1) => contract_class_big4::<4, 1>(bra, ket, &mut c_ef, big4),
            (4, 2) => contract_class_big4::<4, 2>(bra, ket, &mut c_ef, big4),
            (4, 3) => contract_class_big4::<4, 3>(bra, ket, &mut c_ef, big4),
            (4, 4) => contract_class_big4::<4, 4>(bra, ket, &mut c_ef, big4),
            (4, 5) => contract_class_big4::<4, 5>(bra, ket, &mut c_ef, big4),
            (4, 6) => contract_class_big4::<4, 6>(bra, ket, &mut c_ef, big4),
            (5, 0) => contract_class_big4::<5, 0>(bra, ket, &mut c_ef, big4),
            (5, 1) => contract_class_big4::<5, 1>(bra, ket, &mut c_ef, big4),
            (5, 2) => contract_class_big4::<5, 2>(bra, ket, &mut c_ef, big4),
            (5, 3) => contract_class_big4::<5, 3>(bra, ket, &mut c_ef, big4),
            (5, 4) => contract_class_big4::<5, 4>(bra, ket, &mut c_ef, big4),
            (5, 5) => contract_class_big4::<5, 5>(bra, ket, &mut c_ef, big4),
            (5, 6) => contract_class_big4::<5, 6>(bra, ket, &mut c_ef, big4),
            (6, 0) => contract_class_big4::<6, 0>(bra, ket, &mut c_ef, big4),
            (6, 1) => contract_class_big4::<6, 1>(bra, ket, &mut c_ef, big4),
            (6, 2) => contract_class_big4::<6, 2>(bra, ket, &mut c_ef, big4),
            (6, 3) => contract_class_big4::<6, 3>(bra, ket, &mut c_ef, big4),
            (6, 4) => contract_class_big4::<6, 4>(bra, ket, &mut c_ef, big4),
            (6, 5) => contract_class_big4::<6, 5>(bra, ket, &mut c_ef, big4),
            (6, 6) => contract_class_big4::<6, 6>(bra, ket, &mut c_ef, big4),
            _ => unreachable!("guarded above"),
        }
    }

    // Per-lane HRR + scatter, exactly the scalar tail of
    // [`coulomb_shell_into_scratch`].
    let EriScratch {
        bra,
        bra_prev,
        bra_cur,
        ket_prev,
        ket_cur,
        tri_all,
        ..
    } = &mut scratch.scalar;
    if tri_all.len() <= 2 * MAX_L {
        *tri_all = (0..=2 * MAX_L).map(cart_components).collect();
    }
    for (lane, &[a, b, c, d]) in quartets.iter().enumerate() {
        let ab = sub(a.center, b.center);
        let cd = sub(c.center, d.center);
        hrr_and_scatter(
            a.l,
            b.l,
            c.l,
            d.l,
            n_f,
            &scratch.c_ef[lane],
            ab,
            cd,
            outs[lane],
            bra,
            bra_prev,
            bra_cur,
            ket_prev,
            ket_cur,
            &tri_all[..],
        );
    }
}

/// Build `[e0|f0]^(m)` for one primitive quartet by **m-marching** over the `f`
/// degree and accumulate `scale · [e0|f0]^(0)` into the contracted `c_ef` table.
///
/// Algebraically identical to the former full-table VRR (same OS recurrences, same
/// term order) — only the storage changes, so it is bit-identical
/// (B0 golden snapshot). `levels` holds 3 rolling `f`-degree levels in slots
/// `k % 3`, each `maxlevel` long; within slot `k`, element `[e0|f0]^(m)` for the
/// `f`-component with local index `lf` (= `cart_index(f)`) lives at
/// `(k%3)·maxlevel + lf·slab[k] + eoff[k][addr(e)] + m`. The `m`-rows are
/// triangle-packed (`m ∈ 0..=(l_total−de−k)`), so the full `[e0|f0]^(m)` table is
/// never resident. Each level's `m=0` slice is extracted into `c_ef` before its
/// buffer is recycled.
#[allow(clippy::too_many_arguments)]
fn vrr_primitive(
    p: f64,
    q: f64,
    pc: Vec3,
    qc: Vec3,
    kab: f64,
    kcd: f64,
    pa: Vec3,    // P − A  (bra pair, precomputed)
    qcen: Vec3,  // Q − C  (ket pair, precomputed)
    inv_2p: f64, // 1/(2p) (bra pair, precomputed)
    inv_2q: f64, // 1/(2q) (ket pair, precomputed)
    ne: usize,
    nf: usize,
    l_total: usize,
    n_e: usize,
    n_f: usize,
    maxlevel: usize,
    eoff: &[usize],
    slab: &[usize],
    tri: &[Vec<[usize; 3]>],
    levels: &mut [f64],
    scale: f64,
    c_ef: &mut [f64],
) {
    let pq = p + q;
    let rho = p * q / pq;
    let pq_vec = sub(pc, qc); // P − Q
    let t = rho * norm2(pq_vec);
    let w = [
        (p * pc[0] + q * qc[0]) / pq,
        (p * pc[1] + q * qc[1]) / pq,
        (p * pc[2] + q * qc[2]) / pq,
    ];
    // `pa` (P−A) and `qcen` (Q−C) are now passed in (pure per-pair).
    let wp = sub(w, pc); // W − P
    let wq = sub(w, qc); // W − Q

    // Index of `[e0|f0]^(m)` (f-component local index `lf`, e at `addr(e)=ae`) in
    // the rolling buffer for f-degree `k`. Reads/writes `levels` separately (f64 is
    // Copy, so indexing never holds a borrow), so the same `levels` slice serves as
    // source and destination across the distinct slots `k`, `k−1`, `k−2`.
    let off = |k: usize, lf: usize, ae: usize, m: usize| -> usize {
        (k % 3) * maxlevel + lf * slab[k] + eoff[k * n_e + ae] + m
    };

    // Base [00|00]^(m) = pref · F_m(T). The Boys index m reaches
    // l_total = la+lb+lc+ld ≤ 4·MAX_L (NOT 2·MAX_L — that is the per-electron
    // bound; an ERI couples both electrons).
    use std::f64::consts::PI;
    let pref = 2.0 * PI * PI * PI.sqrt() / (p * q * pq.sqrt()) * kab * kcd;
    let mut fm = [0.0f64; 4 * MAX_L + 1];
    boys_array(l_total, t, &mut fm[..=l_total]);
    for m in 0..=l_total {
        levels[off(0, 0, 0, m)] = pref * fm[m];
    }

    // `inv_2p` / `inv_2q` are now passed in (pure per-pair).
    let inv_2pq = 0.5 / pq;
    let q_over_pq = q / pq;
    let p_over_pq = p / pq;

    // Phase A — bra ladder (f-degree 0, level 0): build [e0|00]^(m) by raising e.
    // Level 0 lives in slot 0, f-component 0, so the buffer offset of `[e0|00]^(m)`
    // is `eoff0[addr(e)] + m`; the `e`-offsets are hoisted out of the hot m-loop.
    let eoff0 = &eoff[..n_e];
    for (na, te_list) in tri.iter().enumerate().take(ne + 1).skip(1) {
        for &te in te_list {
            let i = lower_axis(te);
            let s1a = addr(dec(te, i));
            let mmax = l_total - na;
            let coef2 = ((te[i] - 1) as f64) * inv_2p; // (e_i of source)/2p
            let has2 = te[i] >= 2;
            let s1_0 = eoff0[s1a];
            let s2_0 = if has2 {
                eoff0[addr(dec(dec(te, i), i))]
            } else {
                0
            };
            let dst_0 = eoff0[addr(te)];
            for m in 0..=mmax {
                let mut v = pa[i] * levels[s1_0 + m] + wp[i] * levels[s1_0 + m + 1];
                if has2 {
                    v += coef2 * (levels[s2_0 + m] - q_over_pq * levels[s2_0 + m + 1]);
                }
                levels[dst_0 + m] = v;
            }
        }
    }
    // Extract level 0 (f = [0,0,0], local 0, addr 0): contract m = 0.
    for ae in 0..n_e {
        c_ef[ae * n_f] += scale * levels[off(0, 0, ae, 0)];
    }

    // Phase B — ket raise: march f-degree k = 1..=nf, keeping levels {k, k−1, k−2}.
    // Slot/slab/eoff for the three active levels are hoisted out of the hot loops; the
    // m-loop touches only `levels[base + m]` (base computed once per (tf, te)), matching
    // the former full-table inner-loop cost. The k−2 level is read only when the axis
    // power `tf[j] ≥ 2` (which forces k ≥ 2), so its `k−2` indices never underflow.
    for k in 1..=nf {
        let slot_k = (k % 3) * maxlevel;
        let slot_k1 = ((k - 1) % 3) * maxlevel;
        let (slab_k, slab_k1) = (slab[k], slab[k - 1]);
        let eoff_k = &eoff[k * n_e..];
        let eoff_k1 = &eoff[(k - 1) * n_e..];
        let (slot_k2, slab_k2, eoff_k2): (usize, usize, &[usize]) = if k >= 2 {
            (
                ((k - 2) % 3) * maxlevel,
                slab[k - 2],
                &eoff[(k - 2) * n_e..],
            )
        } else {
            (0, 0, &[])
        };
        for &tf in &tri[k] {
            let j = lower_axis(tf);
            let f1 = dec(tf, j);
            let lf1 = cart_index(f1); // local index in level k−1
            let coef2 = ((tf[j] - 1) as f64) * inv_2q;
            let has2 = tf[j] >= 2;
            let lf2 = if has2 { cart_index(dec(f1, j)) } else { 0 }; // level k−2
            let lf = cart_index(tf); // local index in level k
            let sk = slot_k + lf * slab_k;
            let sk1 = slot_k1 + lf1 * slab_k1;
            let sk2 = slot_k2 + lf2 * slab_k2;
            for (nadeg, te_list) in tri.iter().enumerate().take(ne + 1) {
                for &te in te_list {
                    let ea = addr(te);
                    // cross term e_j/2(p+q) · [e−1_j,0|f−1_j,0]^(m+1)
                    let has_cross = te[j] >= 1;
                    let (cross_coef, cross_0) = if has_cross {
                        (te[j] as f64 * inv_2pq, sk1 + eoff_k1[addr(dec(te, j))])
                    } else {
                        (0.0, 0)
                    };
                    let mmax = l_total - nadeg - k;
                    let dst_0 = sk + eoff_k[ea];
                    let src1_0 = sk1 + eoff_k1[ea];
                    let src2_0 = if has2 { sk2 + eoff_k2[ea] } else { 0 };
                    for m in 0..=mmax {
                        let mut v = qcen[j] * levels[src1_0 + m] + wq[j] * levels[src1_0 + m + 1];
                        if has2 {
                            v += coef2 * (levels[src2_0 + m] - p_over_pq * levels[src2_0 + m + 1]);
                        }
                        if has_cross {
                            v += cross_coef * levels[cross_0 + m + 1];
                        }
                        levels[dst_0 + m] = v;
                    }
                }
            }
        }
        // Extract level k: contract each f-component's m = 0 slice into c_ef.
        for &tf in &tri[k] {
            let lf = cart_index(tf);
            let af = addr(tf);
            for ae in 0..n_e {
                c_ef[ae * n_f + af] += scale * levels[off(k, lf, ae, 0)];
            }
        }
    }
}

/// HRR in contracted space (bra `A→B`, then ket `C→D`) followed by scatter into
/// the row-major output block.
///
/// Flat-array HGP horizontal recurrence, replacing the earlier
/// HashMap memoisation. The recurrence math is unchanged — same `lower_axis`
/// choice, same `(A−B)/(C−D)` geometric shifts, same `[raised] + shift·[same]`
/// term order — so it is **bit-identical** to the recursive version (guarded by
/// the B0 golden snapshot). Only the storage changes: contiguous arrays indexed
/// by `addr` / [`cart_index`] instead of hashed `(triple,…)` keys.
///
/// The bra recurrence `(a,b|f) = (a+1_i,b−1_i|f) + (A−B)_i (a,b−1_i|f)` (axis
/// `i = lower_axis(b)`) is built by ascending `b`-degree with two rolling layers,
/// independently per ket index `f` (a spectator), into the `bra` intermediate.
/// The ket recurrence `(ab|c,d) = (ab|c+1_j,d−1_j) + (C−D)_j (ab|c,d−1_j)` is the
/// symmetric pass over `c,d`, run per `(a,b)` output pair and scattered into `out`.
/// Resident scratch is `O(na·nb·nf_range)` for the bra intermediate plus two small
/// rolling layers — never the dense `(a,b,f)`/`(a,b,c,d)` key space.
#[allow(clippy::too_many_arguments)]
fn hrr_and_scatter<'a>(
    la: usize,
    lb: usize,
    lc: usize,
    ld: usize,
    n_f: usize,
    c_ef: &[f64],
    ab: Vec3,
    cd: Vec3,
    out: &mut [f64],
    bra: &mut Vec<f64>,
    mut prev: &'a mut Vec<f64>,
    mut cur: &'a mut Vec<f64>,
    mut kprev: &'a mut Vec<f64>,
    mut kcur: &'a mut Vec<f64>,
    // Shared Cartesian-triple table (degrees `0..=2·MAX_L`); HRR indexes it up to
    // `max(ne, nf)`. Passed in so it is not re-`collect`ed per quartet.
    tri: &[Vec<[usize; 3]>],
) {
    let (na, nb, nc, nd) = (n_cart(la), n_cart(lb), n_cart(lc), n_cart(ld));
    let ne = la + lb; // max bra (A-side) degree present in c_ef
    let nf = lc + ld; // max ket (C-side) degree present in c_ef
    let n_e = n_addr(ne);

    // Ket index range actually used: f-degrees [lc..=nf]. The bra intermediate and
    // the ket base are keyed by the global `addr(f)` offset by `f_base`.
    let f_base = tri_below(lc);
    let nf_range = n_f - f_base;

    // --- Bra HRR: bra[(ia·nb+ib)·nf_range + (addr(f)−f_base)] = (a_ia b_ib | f 0).
    // Reused arena buffers; bra is fully overwritten below, the rolling
    // layers in the region read — debug NaN-fill bra so a missed write surfaces.
    let bra_len = na * nb * nf_range;
    ensure_len(bra, bra_len);
    #[cfg(debug_assertions)]
    bra[..bra_len].fill(f64::NAN);
    // Two rolling b-degree layers, indexed [cart_index(b)·n_e + addr(a)].
    let layer_len = n_cart(lb) * n_e;
    ensure_len(prev, layer_len);
    ensure_len(cur, layer_len);
    for f_global in f_base..n_f {
        let jf = f_global - f_base;
        // Base b-degree 0 (one component, within-index 0): (a,0|f) = c_ef[a][f].
        for &a in tri[la..=ne].iter().flatten() {
            let ae = addr(a);
            prev[ae] = c_ef[ae * n_f + f_global];
        }
        for kb in 1..=lb {
            for (ibw, &b) in tri[kb].iter().enumerate() {
                let i = lower_axis(b);
                let b1w = cart_index(dec(b, i));
                // a-degrees that can still reach (la, lb): [la..=ne−kb].
                for &a in tri[la..=(ne - kb)].iter().flatten() {
                    let ae = addr(a);
                    let a1e = addr(inc(a, i));
                    cur[ibw * n_e + ae] = prev[b1w * n_e + a1e] + ab[i] * prev[b1w * n_e + ae];
                }
            }
            std::mem::swap(&mut prev, &mut cur);
        }
        // `prev` now holds b-degree lb (or the base when lb = 0): extract.
        for (ib, &b) in tri[lb].iter().enumerate() {
            let ibw = cart_index(b); // == ib (tri[lb] is in cart order)
            for (ia, &a) in tri[la].iter().enumerate() {
                bra[(ia * nb + ib) * nf_range + jf] = prev[ibw * n_e + addr(a)];
            }
        }
    }

    // --- Ket HRR: per (ia,ib), build (c,d) and scatter into out.
    let klayer_len = n_cart(ld) * n_f;
    ensure_len(kprev, klayer_len);
    ensure_len(kcur, klayer_len);
    for ia in 0..na {
        for ib in 0..nb {
            let brarow = (ia * nb + ib) * nf_range;
            // Base d-degree 0: (ab|c,0) = bra[ia][ib][c], c-degrees [lc..=nf].
            for &c in tri[lc..=nf].iter().flatten() {
                let ce = addr(c);
                kprev[ce] = bra[brarow + (ce - f_base)];
            }
            for kd in 1..=ld {
                for (idw, &d) in tri[kd].iter().enumerate() {
                    let j = lower_axis(d);
                    let d1w = cart_index(dec(d, j));
                    for &c in tri[lc..=(nf - kd)].iter().flatten() {
                        let ce = addr(c);
                        let c1e = addr(inc(c, j));
                        kcur[idw * n_f + ce] =
                            kprev[d1w * n_f + c1e] + cd[j] * kprev[d1w * n_f + ce];
                    }
                }
                std::mem::swap(&mut kprev, &mut kcur);
            }
            // `kprev` now holds d-degree ld (or the base when ld = 0): scatter into out.
            for (ic, &c) in tri[lc].iter().enumerate() {
                let ce = addr(c);
                for id in 0..nd {
                    out[((ia * nb + ib) * nc + ic) * nd + id] += kprev[id * n_f + ce];
                }
            }
        }
    }
}

// --- small helpers ---

/// First nonzero axis of a triple (the lowering direction).
#[inline]
fn lower_axis(t: [usize; 3]) -> usize {
    if t[0] > 0 {
        0
    } else if t[1] > 0 {
        1
    } else {
        2
    }
}

#[inline]
fn dec(mut t: [usize; 3], i: usize) -> [usize; 3] {
    t[i] -= 1;
    t
}

#[inline]
fn inc(mut t: [usize; 3], i: usize) -> [usize; 3] {
    t[i] += 1;
    t
}

#[inline]
fn combine(a: f64, ca: Vec3, b: f64, cb: Vec3, p: f64) -> Vec3 {
    [
        (a * ca[0] + b * cb[0]) / p,
        (a * ca[1] + b * cb[1]) / p,
        (a * ca[2] + b * cb[2]) / p,
    ]
}

#[inline]
fn sub(u: Vec3, v: Vec3) -> Vec3 {
    [u[0] - v[0], u[1] - v[1], u[2] - v[2]]
}

#[inline]
fn dist2(u: Vec3, v: Vec3) -> f64 {
    norm2(sub(u, v))
}

#[inline]
fn norm2(u: Vec3) -> f64 {
    u[0] * u[0] + u[1] * u[1] + u[2] * u[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Single-primitive `ShellRef` helper.
    fn s(center: Vec3, l: usize, exp: f64) -> (Vec3, usize, [f64; 1], [f64; 1]) {
        (center, l, [exp], [1.0])
    }

    /// (ss|ss) over four unit s primitives must equal the closed form
    /// `2π^{5/2}/(p q √(p+q)) · K_ab · K_cd · F_0(T)` — the same check the Rys
    /// engine passes, so the two share a verified base case.
    #[test]
    fn ssss_matches_closed_form() {
        let (ac, al, ae, acf) = s([0.0, 0.0, 0.0], 0, 0.8);
        let (bc, bl, be, bcf) = s([0.0, 0.0, 0.7], 0, 1.3);
        let (cc, cl, ce, ccf) = s([0.4, 0.0, 0.0], 0, 0.5);
        let (dc, dl, de, dcf) = s([0.0, 0.6, 0.2], 0, 1.1);
        let mut out = [0.0; 1];
        coulomb_shell_into(
            ShellRef {
                center: ac,
                l: al,
                exps: &ae,
                coeffs: &acf,
            },
            ShellRef {
                center: bc,
                l: bl,
                exps: &be,
                coeffs: &bcf,
            },
            ShellRef {
                center: cc,
                l: cl,
                exps: &ce,
                coeffs: &ccf,
            },
            ShellRef {
                center: dc,
                l: dl,
                exps: &de,
                coeffs: &dcf,
            },
            &mut out,
        );

        let p = 0.8 + 1.3;
        let q = 0.5 + 1.1;
        let pcen = combine(0.8, ac, 1.3, bc, p);
        let qcen = combine(0.5, cc, 1.1, dc, q);
        let kab = (-(0.8 * 1.3 / p) * dist2(ac, bc)).exp();
        let kcd = (-(0.5 * 1.1 / q) * dist2(cc, dc)).exp();
        let rho = p * q / (p + q);
        let t = rho * dist2(pcen, qcen);
        let mut fm = [0.0; 1];
        boys_array(0, t, &mut fm);
        use std::f64::consts::PI;
        let expect = 2.0 * PI * PI * PI.sqrt() / (p * q * (p + q).sqrt()) * kab * kcd * fm[0];
        assert!(
            (out[0] - expect).abs() < 1e-14 * expect.abs(),
            "ssss {} vs {}",
            out[0],
            expect
        );
    }

    use crate::engine::os::Prim;
    use crate::engine::rys::coulomb_into;

    /// Build a single-primitive OS/HGP block for a quartet of given `(l, exp,
    /// center)`, for cross-checking against the Rys engine.
    #[allow(clippy::too_many_arguments)]
    fn os_block(
        la: usize,
        ea: f64,
        ca: Vec3,
        lb: usize,
        eb: f64,
        cb: Vec3,
        lc: usize,
        recc: f64,
        ccc: Vec3,
        ld: usize,
        ed: f64,
        cdd: Vec3,
    ) -> Vec<f64> {
        let mut out = vec![0.0; n_cart(la) * n_cart(lb) * n_cart(lc) * n_cart(ld)];
        let (ea1, eb1, ec1, ed1) = ([ea], [eb], [recc], [ed]);
        let one = [1.0];
        coulomb_shell_into(
            ShellRef {
                center: ca,
                l: la,
                exps: &ea1,
                coeffs: &one,
            },
            ShellRef {
                center: cb,
                l: lb,
                exps: &eb1,
                coeffs: &one,
            },
            ShellRef {
                center: ccc,
                l: lc,
                exps: &ec1,
                coeffs: &one,
            },
            ShellRef {
                center: cdd,
                l: ld,
                exps: &ed1,
                coeffs: &one,
            },
            &mut out,
        );
        out
    }

    /// OS/HGP must reproduce the Rys engine element-for-element on a single
    /// primitive quartet, across a sweep of angular momenta (including the
    /// bug-prone mixed-high-L and "all four different L on four centers" cases).
    #[test]
    fn matches_rys_engine_primitive_sweep() {
        let ca = [0.0, 0.0, 0.0];
        let cb = [0.5, -0.3, 0.2];
        let cc = [-0.4, 0.6, -0.1];
        let cd = [0.2, 0.4, 0.8];
        let (ea, eb, ec, ed) = (0.9, 1.3, 0.7, 1.1);

        let quartets = [
            (0usize, 0usize, 0usize, 0usize),
            (1, 0, 0, 0),
            (0, 0, 1, 0),
            (1, 1, 0, 0),
            (1, 0, 1, 0),
            (1, 1, 1, 1),
            (2, 0, 0, 0),
            (2, 1, 0, 0),
            (2, 1, 2, 1),
            (0, 1, 2, 3), // four different L on four centers
            (2, 2, 3, 3), // (dd|ff) mixed high-L
            (3, 0, 0, 1),
            // l_total ≥ 13 guards: the Boys aux index m reaches la+lb+lc+ld, which
            // exceeds 2·MAX_L. These panicked the under-sized fm buffer (a real bug
            // a ≤(dd|ff) sweep missed) and are kept as permanent regression guards.
            (4, 4, 4, 1), // l_total = 13
            (6, 6, 1, 0), // l_total = 13, two i-shells
            (3, 3, 3, 3), // ffff: the cancellation-heavy mixed case
        ];
        for (la, lb, lc, ld) in quartets {
            let os = os_block(la, ea, ca, lb, eb, cb, lc, ec, cc, ld, ed, cd);
            let mut rys = vec![0.0; os.len()];
            coulomb_into(
                Prim::new(ea, ca, la),
                Prim::new(eb, cb, lb),
                Prim::new(ec, cc, lc),
                Prim::new(ed, cd, ld),
                1.0,
                &mut rys,
            );
            assert_cross_engine_close(&os, &rys, &format!("({la}{lb}|{lc}{ld})"));
        }
    }

    /// Assert two ERI blocks (here OS/HGP vs Rys, two independent f64 recurrences)
    /// agree under `|o − r| ≤ atol + rtol·|r|` with `atol = 1e-11`, `rtol = 1e-10`.
    /// The atol floor absorbs benign near-cancellation on structurally tiny
    /// components (where the *relative* OS/Rys difference reaches ~1e-8 at high L
    /// even though both engines are correct — their dominant elements agree to
    /// ~1e-12); the rtol catches any real divergence.
    fn assert_cross_engine_close(os: &[f64], rys: &[f64], tag: &str) {
        const ATOL: f64 = 1e-11;
        const RTOL: f64 = 1e-10;
        for (o, r) in os.iter().zip(rys.iter()) {
            assert!(
                (o - r).abs() <= ATOL + RTOL * r.abs(),
                "{tag} OS vs Rys mismatch: {o} vs {r} (Δ={:e})",
                (o - r).abs()
            );
        }
    }

    /// HGP early contraction (VRR per primitive, HRR once in contracted space)
    /// must equal the per-primitive Rys sum for a genuinely **contracted**
    /// quartet.
    ///
    /// Note: doing the HRR after contraction is a *performance*
    /// choice, **not** a correctness fork. The HRR operator is linear with
    /// exponent-independent geometric coefficients (`A−B`, `C−D`), so
    /// `HRR(Σ_p w_p·[e0|f0]_p) = Σ_p w_p·HRR([e0|f0]_p)` identically — per-primitive
    /// and post-contraction HRR give the *same* answer. This remains a valuable
    /// end-to-end check (a broken VRR cross-term, a mis-applied contraction
    /// coefficient, or a wrong HRR sign would all fail it), but it does not
    /// distinguish HRR *ordering*, because the two orders are algebraically equal.
    #[test]
    fn contracted_quartet_matches_rys_sum() {
        // Two p shells and a d shell, each with multiple primitives, on distinct
        // centres (so HRR shift vectors A−B, C−D are all non-trivial).
        let ca = [0.0, 0.0, 0.0];
        let cb = [0.6, -0.2, 0.1];
        let cc = [-0.3, 0.5, -0.2];
        let cd = [0.2, 0.3, 0.7];
        let (la, lb, lc, ld) = (1usize, 1usize, 2usize, 0usize);
        let ax = [1.4, 0.45];
        let acf = [0.6, 0.5];
        let bx = [0.9, 0.3];
        let bcf = [0.55, 0.5];
        let cx = [1.1, 0.4];
        let ccf = [0.7, 0.4];
        let dx = [0.8];
        let dcf = [1.0];

        let mut os = vec![0.0; n_cart(la) * n_cart(lb) * n_cart(lc) * n_cart(ld)];
        coulomb_shell_into(
            ShellRef {
                center: ca,
                l: la,
                exps: &ax,
                coeffs: &acf,
            },
            ShellRef {
                center: cb,
                l: lb,
                exps: &bx,
                coeffs: &bcf,
            },
            ShellRef {
                center: cc,
                l: lc,
                exps: &cx,
                coeffs: &ccf,
            },
            ShellRef {
                center: cd,
                l: ld,
                exps: &dx,
                coeffs: &dcf,
            },
            &mut os,
        );

        // Reference: sum the Rys primitive engine over the quartet with the same
        // effective coefficients.
        let mut rys = vec![0.0; os.len()];
        for (&ea, &wa) in ax.iter().zip(acf.iter()) {
            for (&eb, &wb) in bx.iter().zip(bcf.iter()) {
                for (&ec, &wc) in cx.iter().zip(ccf.iter()) {
                    for (&ed, &wd) in dx.iter().zip(dcf.iter()) {
                        coulomb_into(
                            Prim::new(ea, ca, la),
                            Prim::new(eb, cb, lb),
                            Prim::new(ec, cc, lc),
                            Prim::new(ed, cd, ld),
                            wa * wb * wc * wd,
                            &mut rys,
                        );
                    }
                }
            }
        }

        for (o, r) in os.iter().zip(rys.iter()) {
            assert!(
                (o - r).abs() <= 1e-10 * r.abs().max(1e-12),
                "contracted OS vs Rys mismatch: {o} vs {r}"
            );
        }
    }

    /// The triple-address map must be a bijection onto `0..n_addr(L)`.
    #[test]
    fn addr_is_bijective() {
        for lmax in 0..=6 {
            let mut seen = vec![false; n_addr(lmax)];
            for n in 0..=lmax {
                for t in cart_components(n) {
                    let a = addr(t);
                    assert!(a < n_addr(lmax), "addr {a} out of range for lmax {lmax}");
                    assert!(!seen[a], "addr collision at {t:?}");
                    seen[a] = true;
                }
            }
            assert!(
                seen.iter().all(|&x| x),
                "addr not surjective at lmax {lmax}"
            );
        }
    }
}
