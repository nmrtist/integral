//! One-electron integrals by Obara–Saika (OS) recurrence.
//!
//! All routines here work on **primitive** Cartesian Gaussians and accumulate
//! their (coefficient-weighted) contribution into a caller-provided output
//! block. Normalization, contraction, and shell bookkeeping live in the `integral`
//! driver crate. The output block for a shell pair `(la, lb)` is row-major with
//! shape `[n_cart(la), n_cart(lb)]`, using the Cartesian component order defined
//! in [`crate::math::am`].
//!
//! ## Conventions
//!
//! A primitive is `g(r) = (x-A_x)^{lx}(y-A_y)^{ly}(z-A_z)^{lz} e^{-α|r-A|²}`
//! (no normalization). For a pair `(α, A)`, `(β, B)`:
//!
//! ```text
//!   p = α + β,   P = (αA + βB)/p,   μ = αβ/p,   K_AB = e^{-μ|A-B|²}.
//! ```
//!
//! ## Buffer sizing
//!
//! Angular momentum is capped at [`MAX_L`]. The separable 1D tables use fixed
//! stack arrays sized to the cap (strategy (a): over-allocate to the cap rather
//! than the nightly-gated `generic_const_exprs`). The nuclear-attraction path
//! uses small heap maps for its auxiliary `(triple, m)` table; tightening that
//! to stack buffers / const-generic monomorphization is a later optimization.

use std::collections::HashMap;

use crate::math::am::{cart_components, cart_index, n_cart};
use crate::math::boys::{boys_array, boys_array_erf};

/// Maximum supported angular momentum per shell for the one-electron OS engine.
/// Higher angular momentum is rejected by the public API.
pub const MAX_L: usize = 6;

/// 1D table dimension: indices `0..=MAX_L+2` (the `+2` is needed by the kinetic
/// recurrence, which reaches `j+2`).
const TBL: usize = MAX_L + 3;

/// A 3-vector (atom center or origin), in atomic units (bohr).
pub type Vec3 = [f64; 3];

/// A single primitive Gaussian: exponent, center, and angular momentum.
#[derive(Debug, Clone, Copy)]
pub struct Prim {
    pub exp: f64,
    pub center: Vec3,
    pub l: usize,
}

impl Prim {
    /// Construct a primitive from its exponent, center, and angular momentum.
    #[must_use]
    pub fn new(exp: f64, center: Vec3, l: usize) -> Self {
        Prim { exp, center, l }
    }
}

/// Gaussian-product quantities for a primitive pair.
#[derive(Debug, Clone, Copy)]
struct Pair {
    p: f64,
    one_over_2p: f64,
    p_center: Vec3,
    mu: f64,
    ab2: f64,
}

fn make_pair(alpha: f64, a: Vec3, beta: f64, b: Vec3) -> Pair {
    let p = alpha + beta;
    let p_center = [
        (alpha * a[0] + beta * b[0]) / p,
        (alpha * a[1] + beta * b[1]) / p,
        (alpha * a[2] + beta * b[2]) / p,
    ];
    let mu = alpha * beta / p;
    let ab2 = (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
    Pair {
        p,
        one_over_2p: 0.5 / p,
        p_center,
        mu,
        ab2,
    }
}

/// Fill the 1D overlap (and, with `e_max > 0`, multipole) table along one axis.
///
/// `table[i][j][e]` is the 1D integral `⟨ (x-A)^i | (x-O)^e | (x-B)^j ⟩` with the
/// Gaussian weight, where the multipole origin enters through `po = P - O`.
/// `e = 0` is the plain overlap factor `S_x(i,j)`. The recurrences are
///
/// ```text
///   S(i+1,j) = pa·S(i,j) + (1/2p)[ i·S(i-1,j) + j·S(i,j-1) ]
///   S(i,j+1) = pb·S(i,j) + (1/2p)[ i·S(i-1,j) + j·S(i,j-1) ]
///   M_e(i,j,e+1) = po·M(i,j,e) + (1/2p)[ i·M(i-1,j,e) + j·M(i,j-1,e) + e·M(i,j,e-1) ]
/// ```
fn overlap_1d(
    pair: &Pair,
    pa: f64,
    pb: f64,
    base: f64,
    i_max: usize,
    j_max: usize,
) -> [[f64; TBL]; TBL] {
    let mut s = [[0.0f64; TBL]; TBL];
    let h = pair.one_over_2p;
    s[0][0] = base;
    // Build the j = 0 column by raising i.
    for i in 1..=i_max {
        let prev2 = if i >= 2 {
            (i - 1) as f64 * s[i - 2][0]
        } else {
            0.0
        };
        s[i][0] = pa * s[i - 1][0] + h * prev2;
    }
    // Raise j for every i.
    for j in 1..=j_max {
        for i in 0..=i_max {
            let from_i = if i >= 1 {
                i as f64 * s[i - 1][j - 1]
            } else {
                0.0
            };
            let from_j = if j >= 2 {
                (j - 1) as f64 * s[i][j - 2]
            } else {
                0.0
            };
            s[i][j] = pb * s[i][j - 1] + h * (from_i + from_j);
        }
    }
    s
}

/// Per-axis base overlap factor `S(0,0) = sqrt(π/p) · e^{-μ d²}` where `d` is the
/// separation along that axis.
fn axis_base(pair: &Pair, d: f64) -> f64 {
    (std::f64::consts::PI / pair.p).sqrt() * (-pair.mu * d * d).exp()
}

/// Accumulate `scale · ⟨a|b⟩` (overlap) into `out` for the shell pair.
pub fn overlap_into(a: Prim, b: Prim, scale: f64, out: &mut [f64]) {
    let Prim {
        exp: alpha,
        center: a,
        l: la,
    } = a;
    let Prim {
        exp: beta,
        center: b,
        l: lb,
    } = b;
    let pair = make_pair(alpha, a, beta, b);
    let sx = overlap_1d(
        &pair,
        pair.p_center[0] - a[0],
        pair.p_center[0] - b[0],
        axis_base(&pair, a[0] - b[0]),
        la,
        lb,
    );
    let sy = overlap_1d(
        &pair,
        pair.p_center[1] - a[1],
        pair.p_center[1] - b[1],
        axis_base(&pair, a[1] - b[1]),
        la,
        lb,
    );
    let sz = overlap_1d(
        &pair,
        pair.p_center[2] - a[2],
        pair.p_center[2] - b[2],
        axis_base(&pair, a[2] - b[2]),
        la,
        lb,
    );

    let comps_a = cart_components(la);
    let comps_b = cart_components(lb);
    let nb = n_cart(lb);
    for (ia, ca) in comps_a.iter().enumerate() {
        for (ib, cb) in comps_b.iter().enumerate() {
            let v = sx[ca[0]][cb[0]] * sy[ca[1]][cb[1]] * sz[ca[2]][cb[2]];
            out[ia * nb + ib] += scale * v;
        }
    }
}

/// Accumulate `scale · ⟨a| -½∇² |b⟩` (kinetic energy) into `out`.
///
/// Uses the 1D kinetic factor (acting on the ket exponent `β`)
/// `T_x(i,j) = β(2j+1)S(i,j) - 2β²S(i,j+2) - ½ j(j-1)S(i,j-2)` and
/// `T = T_x S_y S_z + S_x T_y S_z + S_x S_y T_z`.
pub fn kinetic_into(a: Prim, b: Prim, scale: f64, out: &mut [f64]) {
    let Prim {
        exp: alpha,
        center: a,
        l: la,
    } = a;
    let Prim {
        exp: beta,
        center: b,
        l: lb,
    } = b;
    let pair = make_pair(alpha, a, beta, b);
    // Overlap tables need j up to lb+2 for the kinetic recurrence.
    let s: [[[f64; TBL]; TBL]; 3] = [
        overlap_1d(
            &pair,
            pair.p_center[0] - a[0],
            pair.p_center[0] - b[0],
            axis_base(&pair, a[0] - b[0]),
            la,
            lb + 2,
        ),
        overlap_1d(
            &pair,
            pair.p_center[1] - a[1],
            pair.p_center[1] - b[1],
            axis_base(&pair, a[1] - b[1]),
            la,
            lb + 2,
        ),
        overlap_1d(
            &pair,
            pair.p_center[2] - a[2],
            pair.p_center[2] - b[2],
            axis_base(&pair, a[2] - b[2]),
            la,
            lb + 2,
        ),
    ];

    let kin = |axis: usize, i: usize, j: usize| -> f64 {
        let t = &s[axis];
        let term_plus = 2.0 * beta * beta * t[i][j + 2];
        let term_mid = beta * (2 * j + 1) as f64 * t[i][j];
        let term_minus = if j >= 2 {
            0.5 * (j * (j - 1)) as f64 * t[i][j - 2]
        } else {
            0.0
        };
        term_mid - term_plus - term_minus
    };

    let comps_a = cart_components(la);
    let comps_b = cart_components(lb);
    let nb = n_cart(lb);
    for (ia, ca) in comps_a.iter().enumerate() {
        for (ib, cb) in comps_b.iter().enumerate() {
            let sx = s[0][ca[0]][cb[0]];
            let sy = s[1][ca[1]][cb[1]];
            let sz = s[2][ca[2]][cb[2]];
            let tx = kin(0, ca[0], cb[0]);
            let ty = kin(1, ca[1], cb[1]);
            let tz = kin(2, ca[2], cb[2]);
            let v = tx * sy * sz + sx * ty * sz + sx * sy * tz;
            out[ia * nb + ib] += scale * v;
        }
    }
}

/// Accumulate `scale · ⟨a| (r-O)_k |b⟩` (Cartesian dipole) into the three
/// component blocks `out_x`, `out_y`, `out_z`, with multipole origin `o`.
pub fn dipole_into(
    a: Prim,
    b: Prim,
    o: Vec3,
    scale: f64,
    out_x: &mut [f64],
    out_y: &mut [f64],
    out_z: &mut [f64],
) {
    let Prim {
        exp: alpha,
        center: a,
        l: la,
    } = a;
    let Prim {
        exp: beta,
        center: b,
        l: lb,
    } = b;
    let pair = make_pair(alpha, a, beta, b);
    // Multipole tables along each axis: m[axis][i][j][e], e in {0,1}.
    let mut m = [[[[0.0f64; 2]; TBL]; TBL]; 3];
    for axis in 0..3 {
        let pa = pair.p_center[axis] - a[axis];
        let pb = pair.p_center[axis] - b[axis];
        let po = pair.p_center[axis] - o[axis];
        let base = axis_base(&pair, a[axis] - b[axis]);
        let s = overlap_1d(&pair, pa, pb, base, la, lb);
        let h = pair.one_over_2p;
        for i in 0..=la {
            for j in 0..=lb {
                m[axis][i][j][0] = s[i][j];
                // e = 1 multipole: M(i,j,1) = po·S(i,j) + (1/2p)[i·S(i-1,j)+j·S(i,j-1)]
                let from_i = if i >= 1 { i as f64 * s[i - 1][j] } else { 0.0 };
                let from_j = if j >= 1 { j as f64 * s[i][j - 1] } else { 0.0 };
                m[axis][i][j][1] = po * s[i][j] + h * (from_i + from_j);
            }
        }
    }

    let comps_a = cart_components(la);
    let comps_b = cart_components(lb);
    let nb = n_cart(lb);
    for (ia, ca) in comps_a.iter().enumerate() {
        for (ib, cb) in comps_b.iter().enumerate() {
            let s0 = [
                m[0][ca[0]][cb[0]][0],
                m[1][ca[1]][cb[1]][0],
                m[2][ca[2]][cb[2]][0],
            ];
            let m1 = [
                m[0][ca[0]][cb[0]][1],
                m[1][ca[1]][cb[1]][1],
                m[2][ca[2]][cb[2]][1],
            ];
            let idx = ia * nb + ib;
            out_x[idx] += scale * m1[0] * s0[1] * s0[2];
            out_y[idx] += scale * s0[0] * m1[1] * s0[2];
            out_z[idx] += scale * s0[0] * s0[1] * m1[2];
        }
    }
}

/// Accumulate `scale · Σ_C (−Z_C) ⟨a| 1/|r−C| |b⟩` (nuclear attraction) into
/// `out`, summed over the given point charges `charges = [(C, Z)]`.
///
/// Builds `Θ^m(a, 0)` by the OS vertical recurrence with the Boys auxiliary
/// index, then transfers angular momentum to centre B by the horizontal
/// recurrence `(a, b+1_j) = (a+1_j, b) + (A−B)_j (a, b)`.
pub fn nuclear_into(a: Prim, b: Prim, charges: &[(Vec3, f64)], scale: f64, out: &mut [f64]) {
    let Prim {
        exp: alpha,
        center: a,
        l: la,
    } = a;
    let Prim {
        exp: beta,
        center: b,
        l: lb,
    } = b;
    let pair = make_pair(alpha, a, beta, b);
    let p = pair.p;
    let k_ab = (-pair.mu * pair.ab2).exp();
    let total_l = la + lb;
    let ab = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];

    let comps_a = cart_components(la);
    let comps_b = cart_components(lb);
    let nb = n_cart(lb);

    for &(c, z) in charges {
        // Boys arguments: U = p |P - C|².
        let pc = [
            pair.p_center[0] - c[0],
            pair.p_center[1] - c[1],
            pair.p_center[2] - c[2],
        ];
        let u = p * (pc[0] * pc[0] + pc[1] * pc[1] + pc[2] * pc[2]);
        let mut fm = vec![0.0f64; total_l + 1];
        boys_array(total_l, u, &mut fm);

        // Base Θ^m(0,0,0) = (2π/p) K_AB F_m(U).
        let pref = 2.0 * std::f64::consts::PI / p * k_ab;
        let pa = [
            pair.p_center[0] - a[0],
            pair.p_center[1] - a[1],
            pair.p_center[2] - a[2],
        ];

        // Vertical recurrence: theta[triple] = vec over m of Θ^m(triple, 0).
        let theta = nuclear_vertical(total_l, p, pa, pc, &fm, pref);

        // Horizontal recurrence on B, memoized within this nucleus.
        let mut hrr_cache: HashMap<([u8; 3], [u8; 3]), f64> = HashMap::new();
        for (ia, ca) in comps_a.iter().enumerate() {
            for (ib, cb) in comps_b.iter().enumerate() {
                let g = hrr(
                    [ca[0] as u8, ca[1] as u8, ca[2] as u8],
                    [cb[0] as u8, cb[1] as u8, cb[2] as u8],
                    ab,
                    &theta,
                    &mut hrr_cache,
                );
                out[ia * nb + ib] += scale * (-z) * g;
            }
        }
    }
}

/// Accumulate `scale · Σ_{k∈{x,y,z}} ⟨∂_k a| V |∂_k b⟩` (the spin-free **pVp**
/// primitive block) into `out`, with `V = Σ_C (−Z_C)/|r−C|` over the point
/// charges `charges = [(C, Z)]` — the same charge list and sign convention as
/// [`nuclear_into`].
///
/// Each Cartesian gradient component is the standard angular-momentum shift at
/// the primitive level,
///
/// ```text
///   ∂_k g_a = 2α · g_{a+1_k} − a_k · g_{a−1_k},
/// ```
///
/// applied on both bra and ket, so every `(a, b)` element is a fixed linear
/// combination of up to four **ordinary nuclear-attraction integrals** with the
/// bra/ket angular momenta shifted by ±1. The four shifted shell-pair blocks
/// are evaluated once per primitive pair by [`nuclear_into`] (the existing
/// kernel — no new recurrences) and combined here:
///
/// ```text
///   Σ_k [ 4αβ · N(a+1_k, b+1_k) − 2α·b_k · N(a+1_k, b−1_k)
///         − 2β·a_k · N(a−1_k, b+1_k) + a_k·b_k · N(a−1_k, b−1_k) ].
/// ```
///
/// The output block is row-major `n_cart(a.l) × n_cart(b.l)`, as for the other
/// one-electron primitives in this module.
///
/// # Panics
/// If `a.l` or `b.l` exceeds `MAX_L − 1`: the derivative raises the shell to
/// `l + 1`, which must stay within the engine's validated [`MAX_L`].
pub fn pvp_nuclear_into(a: Prim, b: Prim, charges: &[(Vec3, f64)], scale: f64, out: &mut [f64]) {
    let (la, lb) = (a.l, b.l);
    assert!(
        la < MAX_L && lb < MAX_L,
        "pVp requires shell l <= {} (the derivative raises l by 1), got la={la}, lb={lb}",
        MAX_L - 1
    );
    let (alpha, beta) = (a.exp, b.exp);
    let shifted = |li: usize, lj: usize| {
        let mut blk = vec![0.0; n_cart(li) * n_cart(lj)];
        nuclear_into(
            Prim::new(a.exp, a.center, li),
            Prim::new(b.exp, b.center, lj),
            charges,
            1.0,
            &mut blk,
        );
        blk
    };
    let n_pp = shifted(la + 1, lb + 1);
    let n_pm = (lb > 0).then(|| shifted(la + 1, lb - 1));
    let n_mp = (la > 0).then(|| shifted(la - 1, lb + 1));
    let n_mm = (la > 0 && lb > 0).then(|| shifted(la - 1, lb - 1));

    let nb = n_cart(lb);
    let nbp = n_cart(lb + 1);
    let nbm = if lb > 0 { n_cart(lb - 1) } else { 0 };
    for (ia, ca) in cart_components(la).iter().enumerate() {
        for (ib, cb) in cart_components(lb).iter().enumerate() {
            let mut v = 0.0;
            for k in 0..3 {
                let mut cap = *ca;
                cap[k] += 1;
                let rp = cart_index(cap);
                let mut cbp = *cb;
                cbp[k] += 1;
                let cp = cart_index(cbp);
                v += 4.0 * alpha * beta * n_pp[rp * nbp + cp];
                let cm = (cb[k] > 0).then(|| {
                    let mut cbm = *cb;
                    cbm[k] -= 1;
                    cart_index(cbm)
                });
                if let Some(cm) = cm {
                    v -= 2.0 * alpha * cb[k] as f64 * n_pm.as_ref().unwrap()[rp * nbm + cm];
                }
                if ca[k] > 0 {
                    let mut cam = *ca;
                    cam[k] -= 1;
                    let rm = cart_index(cam);
                    v -= 2.0 * beta * ca[k] as f64 * n_mp.as_ref().unwrap()[rm * nbp + cp];
                    if let Some(cm) = cm {
                        v += (ca[k] * cb[k]) as f64 * n_mm.as_ref().unwrap()[rm * nbm + cm];
                    }
                }
            }
            out[ia * nb + ib] += scale * v;
        }
    }
}

/// Build `Θ^0(a, 0)` for every Cartesian triple `a` with `|a| <= total_l`.
///
/// Returns a map from triple to `Θ^0`. Intermediate orders `Θ^m` are kept only
/// while building. The recurrence (b = 0 branch) is
///
/// ```text
///   Θ^m(t,0) = (P−A)_i Θ^m(a,0) − (P−C)_i Θ^{m+1}(a,0)
///            + (a_i/2p)[ Θ^m(a−1_i,0) − Θ^{m+1}(a−1_i,0) ],   a = t − 1_i.
/// ```
fn nuclear_vertical(
    total_l: usize,
    p: f64,
    pa: Vec3,
    pc: Vec3,
    fm: &[f64],
    pref: f64,
) -> HashMap<[u8; 3], f64> {
    // Full m-ladder per triple while building; collapse to Θ^0 at the end.
    let mut ladder: HashMap<[u8; 3], Vec<f64>> = HashMap::new();
    let one_over_2p = 0.5 / p;

    // |a| = 0.
    let base: Vec<f64> = (0..=total_l).map(|m| pref * fm[m]).collect();
    ladder.insert([0, 0, 0], base);

    for n in 1..=total_l {
        for t in triples_with_sum(n) {
            // Choose lowering direction: first nonzero component.
            let i = if t[0] > 0 {
                0
            } else if t[1] > 0 {
                1
            } else {
                2
            };
            let mut a = t;
            a[i] -= 1;
            let a_i = a[i] as f64; // exponent in direction i of the source triple

            let mut aa = a; // a - 1_i
            let have_second = aa[i] > 0;
            if have_second {
                aa[i] -= 1;
            }

            let m_count = total_l - n + 1;
            let mut vals = vec![0.0f64; m_count];
            let src = &ladder[&a];
            let src2 = if have_second {
                Some(ladder[&aa].clone())
            } else {
                None
            };
            for m in 0..m_count {
                let mut v = pa[i] * src[m] - pc[i] * src[m + 1];
                if let Some(ref s2) = src2 {
                    v += a_i * one_over_2p * (s2[m] - s2[m + 1]);
                }
                vals[m] = v;
            }
            ladder.insert(t, vals);
        }
    }

    ladder.into_iter().map(|(k, v)| (k, v[0])).collect()
}

/// Horizontal recurrence transferring angular momentum from A to B:
/// `(a, b) = (a+1_j, b−1_j) + (A−B)_j (a, b−1_j)`, with base `(a,0) = Θ^0(a,0)`.
fn hrr(
    a: [u8; 3],
    b: [u8; 3],
    ab: Vec3,
    theta: &HashMap<[u8; 3], f64>,
    cache: &mut HashMap<([u8; 3], [u8; 3]), f64>,
) -> f64 {
    if b[0] == 0 && b[1] == 0 && b[2] == 0 {
        return theta[&a];
    }
    if let Some(&v) = cache.get(&(a, b)) {
        return v;
    }
    // Lower b along its first nonzero component.
    let j = if b[0] > 0 {
        0
    } else if b[1] > 0 {
        1
    } else {
        2
    };
    let mut b_low = b;
    b_low[j] -= 1;
    let mut a_up = a;
    a_up[j] += 1;

    let v = hrr(a_up, b_low, ab, theta, cache) + ab[j] * hrr(a, b_low, ab, theta, cache);
    cache.insert((a, b), v);
    v
}

/// Enumerate Cartesian triples `(i,j,k)` with `i+j+k == n`, canonical order.
fn triples_with_sum(n: usize) -> Vec<[u8; 3]> {
    let mut out = Vec::new();
    for lx in (0..=n).rev() {
        for ly in (0..=(n - lx)).rev() {
            out.push([lx as u8, ly as u8, (n - lx - ly) as u8]);
        }
    }
    out
}

/// One step of the precomputed nuclear-attraction vertical-recurrence plan used
/// by [`GridCoulombPair`]: build the Boys-auxiliary ladder entry of one
/// Cartesian triple from its already-built lower neighbours. The plan depends
/// only on the pair's total angular momentum (the recurrence *structure*); the
/// geometry (`P−A`, `P−C`) enters at evaluation time.
#[derive(Debug, Clone, Copy)]
struct VrrStep {
    /// Ladder offset of the destination triple `t`.
    dst: usize,
    /// Ladder offset of the source triple `a = t − 1_i`.
    src: usize,
    /// Ladder offset of `a − 1_i`, or `usize::MAX` when `a_i == 0`.
    src2: usize,
    /// Lowering direction `i ∈ {0,1,2}`.
    dir: usize,
    /// `a_i` of the source triple (the second-term coefficient).
    a_i: f64,
    /// Number of auxiliary orders `m` to fill at this triple.
    m_count: usize,
}

/// Reusable buffers for [`GridCoulombPair::eval_into`], so a caller looping
/// over many grid points performs no per-point allocation. One instance per
/// thread/loop; buffers grow on first use and are reused thereafter.
#[derive(Debug, Default)]
pub struct GridCoulombScratch {
    fm: Vec<f64>,
    ladder: Vec<f64>,
    theta0: Vec<f64>,
}

/// Precomputed shell-pair data for **batched grid-point Coulomb integrals**
/// `⟨a| 1/|r−C| |b⟩` — the per-point matrices a semi-numerical exchange (COSX)
/// build needs, evaluated for many points `C` against one shell pair.
///
/// All point-independent work is hoisted into [`GridCoulombPair::new`]:
/// the Gaussian-product quantities of every primitive pair (`p`, `P`, `P−A`,
/// prefactor `scale · 2π/p · K_AB`), the vertical-recurrence *plan* (a flat
/// triple-indexed schedule replacing the per-call `HashMap` ladder of
/// [`nuclear_into`]), and the horizontal recurrence collapsed to a fixed
/// **linear combination** per Cartesian component pair: `(a, b)` is, by the
/// HRR `(a,b) = (a+1_j, b−1_j) + (A−B)_j (a, b−1_j)`, an exact geometry-only
/// linear combination of the `Θ⁰(t, 0)` (and the combination is the same for
/// every primitive pair and every point). [`GridCoulombPair::eval_into`] then
/// does only the point-dependent work: one Boys evaluation and one flat-array
/// VRR sweep per primitive pair, then `n_a·n_b` short dot products.
///
/// Sign convention: the **positive** kernel `1/|r−C|` (no charge, no `−Z`
/// attraction sign): for a unit charge at `C`, the block equals the *negative*
/// of what [`nuclear_into`] accumulates.
#[derive(Debug)]
pub struct GridCoulombPair {
    total_l: usize,
    /// Per primitive pair: `(p, one_over_2p, P, P−A, pref)` with
    /// `pref = scale · (2π/p) · K_AB`.
    prims: Vec<(f64, f64, Vec3, Vec3, f64)>,
    plan: Vec<VrrStep>,
    /// Ladder offset of each triple (triple index order: level-major,
    /// [`triples_with_sum`] order within a level).
    ladder_offs: Vec<usize>,
    ladder_len: usize,
    /// Per output component pair `(ia, ib)` (row-major `ia·n_b + ib`): the HRR
    /// linear combination `Σ coef · Θ⁰(t)` as `(triple index, coef)` terms.
    hrr: Vec<Vec<(usize, f64)>>,
    na: usize,
    nb: usize,
}

impl GridCoulombPair {
    /// Precompute the pair data for shells `(la, A)` and `(lb, B)` whose
    /// surviving primitive pairs are given as `(α, β, scale)` triples, where
    /// `scale` is the product of the two primitives' effective contraction
    /// coefficients. The output block of [`GridCoulombPair::eval_into`] is
    /// row-major `n_cart(la) × n_cart(lb)`.
    ///
    /// # Panics
    /// If `la + lb` exceeds the engine's validated range (`la, lb ≤ MAX_L`
    /// each is the caller's contract, as for the other OS entry points).
    #[must_use]
    pub fn new(la: usize, lb: usize, a: Vec3, b: Vec3, prim_pairs: &[(f64, f64, f64)]) -> Self {
        let total_l = la + lb;
        // Triple indexing: level-major, `triples_with_sum` order within a level.
        let mut triples: Vec<[u8; 3]> = Vec::new();
        for n in 0..=total_l {
            triples.extend(triples_with_sum(n));
        }
        let index: HashMap<[u8; 3], usize> =
            triples.iter().enumerate().map(|(i, &t)| (t, i)).collect();
        // Ladder layout: triple `t` holds `total_l − |t| + 1` auxiliary orders.
        let mut ladder_offs = Vec::with_capacity(triples.len());
        let mut acc = 0usize;
        for t in &triples {
            ladder_offs.push(acc);
            acc += total_l - (t[0] + t[1] + t[2]) as usize + 1;
        }
        let ladder_len = acc;
        // The VRR plan (same lowering choice and arithmetic as
        // `nuclear_vertical`, scheduled level by level).
        let mut plan = Vec::with_capacity(triples.len().saturating_sub(1));
        for (ti, &t) in triples.iter().enumerate() {
            let n = (t[0] + t[1] + t[2]) as usize;
            if n == 0 {
                continue;
            }
            let i = if t[0] > 0 {
                0
            } else if t[1] > 0 {
                1
            } else {
                2
            };
            let mut a_tr = t;
            a_tr[i] -= 1;
            let a_i = f64::from(a_tr[i]);
            let src2 = if a_tr[i] > 0 {
                let mut aa = a_tr;
                aa[i] -= 1;
                ladder_offs[index[&aa]]
            } else {
                usize::MAX
            };
            plan.push(VrrStep {
                dst: ladder_offs[ti],
                src: ladder_offs[index[&a_tr]],
                src2,
                dir: i,
                a_i,
                m_count: total_l - n + 1,
            });
        }
        // HRR collapsed to per-(ia, ib) linear combinations over Θ⁰ triples.
        let ab = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let comps_a = cart_components(la);
        let comps_b = cart_components(lb);
        let mut hrr = Vec::with_capacity(comps_a.len() * comps_b.len());
        for ca in &comps_a {
            for cb in &comps_b {
                let mut acc_map: HashMap<[u8; 3], f64> = HashMap::new();
                hrr_expand(
                    [ca[0] as u8, ca[1] as u8, ca[2] as u8],
                    [cb[0] as u8, cb[1] as u8, cb[2] as u8],
                    1.0,
                    ab,
                    &mut acc_map,
                );
                let mut terms: Vec<(usize, f64)> =
                    acc_map.into_iter().map(|(t, c)| (index[&t], c)).collect();
                terms.sort_unstable_by_key(|&(i, _)| i);
                hrr.push(terms);
            }
        }
        // Gaussian-product data per primitive pair (the point-independent half
        // of `nuclear_into`).
        let prims = prim_pairs
            .iter()
            .map(|&(alpha, beta, scale)| {
                let pair = make_pair(alpha, a, beta, b);
                let pa = [
                    pair.p_center[0] - a[0],
                    pair.p_center[1] - a[1],
                    pair.p_center[2] - a[2],
                ];
                let pref =
                    scale * 2.0 * std::f64::consts::PI / pair.p * (-pair.mu * pair.ab2).exp();
                (pair.p, pair.one_over_2p, pair.p_center, pa, pref)
            })
            .collect();
        GridCoulombPair {
            total_l,
            prims,
            plan,
            ladder_offs,
            ladder_len,
            hrr,
            na: n_cart(la),
            nb: n_cart(lb),
        }
    }

    /// Evaluate the **positive-kernel** block `⟨a| 1/|r−C| |b⟩` for one point
    /// `C` into `out` (row-major `n_cart(la) × n_cart(lb)`, every element
    /// *assigned*). Only point-dependent work runs here; see the type docs.
    ///
    /// # Panics
    /// If `out.len() < n_cart(la) · n_cart(lb)`.
    pub fn eval_into(&self, c: Vec3, scratch: &mut GridCoulombScratch, out: &mut [f64]) {
        self.eval_kernel_into(c, None, scratch, out);
    }

    /// Like [`GridCoulombPair::eval_into`] but for the **long-range** kernel
    /// `erf(ω·|r−C|)/|r−C|`: the attenuation enters only through the 0th-order
    /// kernel, `F_m(T) → F_m^ω(T) = s^{m+1/2}·F_m(sT)` with `s = ω²/(p+ω²)`
    /// (`p` the pair total exponent — the point charge is the zero-width limit
    /// of the second charge distribution, so `ρ → p`), realized by
    /// [`boys_array_erf`]; the VRR/HRR structure is unchanged (Gill & Adamson,
    /// CPL **261**, 105 (1996); Ahlrichs, PCCP **8**, 3072 (2006)).
    ///
    /// The caller validates `omega` (finite, `> 0`); [`GridCoulombPair::eval_into`]
    /// is the `ω → ∞` (Coulomb) path and is **not** routed through this method.
    ///
    /// # Panics
    /// If `out.len() < n_cart(la) · n_cart(lb)`.
    pub fn eval_erf_into(
        &self,
        c: Vec3,
        omega: f64,
        scratch: &mut GridCoulombScratch,
        out: &mut [f64],
    ) {
        self.eval_kernel_into(c, Some(omega), scratch, out);
    }

    /// Shared point-evaluation body: `omega = None` is the Coulomb kernel
    /// (bit-identical to the pre-kernel implementation — the only branch is
    /// the Boys-ladder call), `Some(ω)` the erf-attenuated one.
    fn eval_kernel_into(
        &self,
        c: Vec3,
        omega: Option<f64>,
        scratch: &mut GridCoulombScratch,
        out: &mut [f64],
    ) {
        let l = self.total_l;
        if scratch.fm.len() < l + 1 {
            scratch.fm.resize(l + 1, 0.0);
        }
        if scratch.ladder.len() < self.ladder_len {
            scratch.ladder.resize(self.ladder_len, 0.0);
        }
        if scratch.theta0.len() < self.ladder_offs.len() {
            scratch.theta0.resize(self.ladder_offs.len(), 0.0);
        }
        scratch.theta0[..self.ladder_offs.len()].fill(0.0);
        for &(p, one_over_2p, p_center, pa, pref) in &self.prims {
            let pc = [p_center[0] - c[0], p_center[1] - c[1], p_center[2] - c[2]];
            let u = p * (pc[0] * pc[0] + pc[1] * pc[1] + pc[2] * pc[2]);
            match omega {
                None => boys_array(l, u, &mut scratch.fm),
                Some(w) => boys_array_erf(l, u, p, w, &mut scratch.fm),
            }
            let ladder = &mut scratch.ladder;
            for (slot, &f) in ladder.iter_mut().zip(&scratch.fm[..=l]) {
                *slot = pref * f;
            }
            for step in &self.plan {
                for m in 0..step.m_count {
                    let mut v = pa[step.dir] * ladder[step.src + m]
                        - pc[step.dir] * ladder[step.src + m + 1];
                    if step.src2 != usize::MAX {
                        v += step.a_i
                            * one_over_2p
                            * (ladder[step.src2 + m] - ladder[step.src2 + m + 1]);
                    }
                    ladder[step.dst + m] = v;
                }
            }
            for (th, &off) in scratch.theta0.iter_mut().zip(&self.ladder_offs) {
                *th += ladder[off];
            }
        }
        for (slot, terms) in out[..self.na * self.nb].iter_mut().zip(&self.hrr) {
            let mut v = 0.0;
            for &(t, coef) in terms {
                v += coef * scratch.theta0[t];
            }
            *slot = v;
        }
    }

    /// The output block dimensions `(n_cart(la), n_cart(lb))`.
    #[must_use]
    pub fn dims(&self) -> (usize, usize) {
        (self.na, self.nb)
    }
}

/// Expand the HRR `(a, b) = (a+1_j, b−1_j) + (A−B)_j (a, b−1_j)` down to `b = 0`,
/// accumulating the geometry-only coefficients of each base triple `Θ⁰(t, 0)`.
fn hrr_expand(a: [u8; 3], b: [u8; 3], coef: f64, ab: Vec3, acc: &mut HashMap<[u8; 3], f64>) {
    if b == [0, 0, 0] {
        *acc.entry(a).or_insert(0.0) += coef;
        return;
    }
    let j = if b[0] > 0 {
        0
    } else if b[1] > 0 {
        1
    } else {
        2
    };
    let mut b_low = b;
    b_low[j] -= 1;
    let mut a_up = a;
    a_up[j] += 1;
    hrr_expand(a_up, b_low, coef, ab, acc);
    hrr_expand(a, b_low, coef * ab[j], ab, acc);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::norm::cart_norm;

    const PI: f64 = std::f64::consts::PI;

    fn norm_s(alpha: f64) -> f64 {
        cart_norm(alpha, 0, 0, 0)
    }

    #[test]
    fn s_s_overlap_matches_analytic() {
        // Normalized s|s overlap between two centres.
        let (a, b) = (1.2, 0.8);
        let ca = [0.0, 0.0, 0.0];
        let cb = [0.0, 0.0, 1.3];
        let mut out = [0.0];
        let scale = norm_s(a) * norm_s(b);
        overlap_into(Prim::new(a, ca, 0), Prim::new(b, cb, 0), scale, &mut out);

        let p = a + b;
        let mu = a * b / p;
        let r2 = 1.3_f64.powi(2);
        let raw = (PI / p).powf(1.5) * (-mu * r2).exp();
        let expected = norm_s(a) * norm_s(b) * raw;
        assert!(
            (out[0] - expected).abs() < 1e-13,
            "{} vs {}",
            out[0],
            expected
        );
    }

    #[test]
    fn same_center_normalized_s_overlap_is_one() {
        let a = 0.9;
        let c = [0.2, -0.4, 0.5];
        let mut out = [0.0];
        overlap_into(
            Prim::new(a, c, 0),
            Prim::new(a, c, 0),
            norm_s(a) * norm_s(a),
            &mut out,
        );
        assert!((out[0] - 1.0).abs() < 1e-13, "{}", out[0]);
    }

    #[test]
    fn s_s_kinetic_matches_analytic() {
        // T_ss = μ(3 - 2μR²) S_ss  (for unnormalized primitives).
        let (a, b) = (1.1, 0.7);
        let ca = [0.0, 0.0, 0.0];
        let cb = [0.0, 0.6, 0.0];
        let mut t = [0.0];
        kinetic_into(Prim::new(a, ca, 0), Prim::new(b, cb, 0), 1.0, &mut t);

        let p = a + b;
        let mu = a * b / p;
        let r2 = 0.6_f64.powi(2);
        let s_ss = (PI / p).powf(1.5) * (-mu * r2).exp();
        let expected = mu * (3.0 - 2.0 * mu * r2) * s_ss;
        assert!((t[0] - expected).abs() < 1e-13, "{} vs {}", t[0], expected);
    }

    #[test]
    fn nuclear_s_s_matches_analytic() {
        // (s|1/r_C|s) = (2π/p) e^{-μR²} F_0(p|P-C|²)  (unnormalized).
        use crate::math::boys::boys;
        let (a, b) = (1.3, 0.9);
        let ca = [0.0, 0.0, 0.0];
        let cb = [0.0, 0.0, 1.0];
        let c = [0.0, 0.0, 0.4];
        let mut v = [0.0];
        // charge Z = 1 → integrand is -(1)·(s|1/r|s); compare magnitude.
        nuclear_into(
            Prim::new(a, ca, 0),
            Prim::new(b, cb, 0),
            &[(c, 1.0)],
            1.0,
            &mut v,
        );

        let p = a + b;
        let mu = a * b / p;
        let pcenter = (a * 0.0 + b * 1.0) / p;
        let r2 = 1.0_f64.powi(2);
        let pc2 = (pcenter - 0.4).powi(2);
        let expected = -(2.0 * PI / p) * (-mu * r2).exp() * boys(0, p * pc2);
        assert!((v[0] - expected).abs() < 1e-13, "{} vs {}", v[0], expected);
    }

    /// `⟨a|O|b⟩` blocks must satisfy `block(la,lb) == block(lb,la)^T` for any
    /// symmetric one-electron operator. Exercises higher-`l` Cartesian indexing
    /// (the s-only analytic tests above cannot). Validated here for overlap,
    /// kinetic, and nuclear at p and d.
    #[test]
    fn blocks_are_transpose_symmetric() {
        let pa = Prim::new(1.4, [0.1, 0.0, -0.2], 0);
        let pb = Prim::new(0.6, [0.0, 0.5, 0.3], 0);
        let charges = [([0.2, -0.1, 0.4], 3.0), ([-0.3, 0.2, 0.0], 1.0)];

        for (la, lb) in [(1, 0), (1, 1), (2, 1), (2, 2)] {
            let a = Prim { l: la, ..pa };
            let b = Prim { l: lb, ..pb };
            let (na, nb) = (n_cart(la), n_cart(lb));

            for kind in 0..3 {
                let mut fwd = vec![0.0; na * nb];
                let mut rev = vec![0.0; nb * na];
                match kind {
                    0 => {
                        overlap_into(a, b, 1.0, &mut fwd);
                        overlap_into(b, a, 1.0, &mut rev);
                    }
                    1 => {
                        kinetic_into(a, b, 1.0, &mut fwd);
                        kinetic_into(b, a, 1.0, &mut rev);
                    }
                    _ => {
                        nuclear_into(a, b, &charges, 1.0, &mut fwd);
                        nuclear_into(b, a, &charges, 1.0, &mut rev);
                    }
                }
                for i in 0..na {
                    for j in 0..nb {
                        let f = fwd[i * nb + j];
                        let r = rev[j * na + i];
                        assert!(
                            (f - r).abs() < 1e-12 * f.abs().max(1.0),
                            "kind={kind} (la={la},lb={lb}) [{i},{j}]: {f} vs {r}"
                        );
                    }
                }
            }
        }
    }

    /// [`GridCoulombPair`] must reproduce the per-point positive Coulomb block
    /// `−1 ×` [`nuclear_into`] with a unit charge, primitive pair by primitive
    /// pair, across mixed angular momenta (independent code path: HashMap
    /// ladder + recursive HRR vs flat plan + collapsed linear combination).
    #[test]
    fn grid_pair_matches_negated_nuclear() {
        let a_c = [0.1, -0.3, 0.2];
        let b_c = [-0.4, 0.5, 0.7];
        let prim_pairs = [(1.3, 0.9, 0.8), (0.5, 0.9, -0.4), (1.3, 0.3, 1.1)];
        let points = [[0.3, 0.2, -0.5], [-1.0, 0.4, 0.9], [0.1, -0.3, 0.2]];
        for (la, lb) in [(0, 0), (1, 0), (2, 1), (3, 2), (4, 4)] {
            let pair = GridCoulombPair::new(la, lb, a_c, b_c, &prim_pairs);
            let (na, nb) = pair.dims();
            let mut scratch = GridCoulombScratch::default();
            let mut got = vec![0.0; na * nb];
            for &c in &points {
                pair.eval_into(c, &mut scratch, &mut got);
                let mut reference = vec![0.0; na * nb];
                for &(alpha, beta, scale) in &prim_pairs {
                    nuclear_into(
                        Prim::new(alpha, a_c, la),
                        Prim::new(beta, b_c, lb),
                        &[(c, 1.0)],
                        scale,
                        &mut reference,
                    );
                }
                let peak = reference.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
                for (idx, (&g, &r)) in got.iter().zip(&reference).enumerate() {
                    let want = -r; // positive kernel vs the −Z attraction sign
                    assert!(
                        (g - want).abs() <= 1e-12 * want.abs().max(1e-3 * peak),
                        "(la={la},lb={lb}) elem {idx}: {g} vs {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn dipole_same_center_s_is_position_times_overlap() {
        // ⟨s| (r-O) |s⟩ with both functions at center R: equals (R-O)·⟨s|s⟩.
        let a = 1.0;
        let r = [0.3, -0.2, 0.7];
        let o = [0.1, 0.1, 0.1];
        let scale = norm_s(a) * norm_s(a);
        let (mut dx, mut dy, mut dz) = ([0.0], [0.0], [0.0]);
        dipole_into(
            Prim::new(a, r, 0),
            Prim::new(a, r, 0),
            o,
            scale,
            &mut dx,
            &mut dy,
            &mut dz,
        );
        // Overlap is 1 (normalized, same center), so dipole = R - O.
        assert!((dx[0] - (r[0] - o[0])).abs() < 1e-13, "{}", dx[0]);
        assert!((dy[0] - (r[1] - o[1])).abs() < 1e-13, "{}", dy[0]);
        assert!((dz[0] - (r[2] - o[2])).abs() < 1e-13, "{}", dz[0]);
    }
}
