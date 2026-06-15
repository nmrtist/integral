//! Confirm the **interpolated** production Rys path and the **Stieltjes
//! reference** path produce the SAME ERIs to the tight (1e-11) tolerance, i.e.
//! the table-interpolation optimization changed nothing observable downstream.
//!
//! Strategy: a faithful copy of `integral::engine::rys::coulomb_into` / `build_axis`,
//! parameterized by the roots/weights function. We (1) verify the copy fed with
//! the production `rys_roots_weights` reproduces the real public `coulomb_into`
//! to machine precision (so the copy is faithful), then (2) diff the copy fed
//! with `rys_roots_weights` vs the copy fed with `rys_roots_weights_reference`
//! over primitive quartets across the full L range — the only difference between
//! the two runs is interp-vs-Stieltjes roots/weights.

use integral::engine::os::{Prim, MAX_L};
use integral::engine::rys::coulomb_into;
use integral::math::am::{cart_components, n_cart};
use integral::math::rys::{rys_roots_weights, rys_roots_weights_reference, MAX_RYS_ROOTS};

const L1: usize = MAX_L + 1;
const LT1: usize = 2 * MAX_L + 1;
type Axis4 = [[[[f64; L1]; L1]; L1]; L1];

fn zeroed() -> Axis4 {
    [[[[0.0; L1]; L1]; L1]; L1]
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn norm2(a: [f64; 3]) -> f64 {
    a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
}
fn dist2(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm2(sub(a, b))
}
fn combine(a: f64, ca: [f64; 3], b: f64, cb: [f64; 3], p: f64) -> [f64; 3] {
    [
        (a * ca[0] + b * cb[0]) / p,
        (a * ca[1] + b * cb[1]) / p,
        (a * ca[2] + b * cb[2]) / p,
    ]
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn build_axis(
    la: usize,
    lb: usize,
    lc: usize,
    ld: usize,
    c00: f64,
    cp0: f64,
    b10: f64,
    b01: f64,
    b00: f64,
    ab: f64,
    cd: f64,
    out4: &mut Axis4,
) {
    let nbra = la + lb;
    let nket = lc + ld;
    let mut g = [[0.0f64; LT1]; LT1];
    g[0][0] = 1.0;
    if nbra >= 1 {
        g[1][0] = c00;
    }
    for n in 1..nbra {
        g[n + 1][0] = c00 * g[n][0] + (n as f64) * b10 * g[n - 1][0];
    }
    for m in 0..nket {
        for n in 0..=nbra {
            let mut term = cp0 * g[n][m];
            if m >= 1 {
                term += (m as f64) * b01 * g[n][m - 1];
            }
            if n >= 1 {
                term += (n as f64) * b00 * g[n - 1][m];
            }
            g[n][m + 1] = term;
        }
    }
    let mut h = [[[0.0f64; LT1]; L1]; LT1];
    for i in 0..=nbra {
        for m in 0..=nket {
            h[i][0][m] = g[i][m];
        }
    }
    for j in 1..=lb {
        for i in 0..=(nbra - j) {
            for m in 0..=nket {
                h[i][j][m] = h[i + 1][j - 1][m] + ab * h[i][j - 1][m];
            }
        }
    }
    for i in 0..=la {
        for j in 0..=lb {
            let mut t = [[0.0f64; L1]; LT1];
            for k in 0..=nket {
                t[k][0] = h[i][j][k];
            }
            for l in 1..=ld {
                for k in 0..=(nket - l) {
                    t[k][l] = t[k + 1][l - 1] + cd * t[k][l - 1];
                }
            }
            for k in 0..=lc {
                for l in 0..=ld {
                    out4[i][j][k][l] = t[k][l];
                }
            }
        }
    }
}

/// Faithful copy of `coulomb_into`, but the roots/weights come from `rw` — a plain
/// `fn` for the production path, or a memoizing closure for the slow Stieltjes
/// reference (hence the `FnMut` bound rather than the former `fn`-pointer type).
#[allow(clippy::too_many_arguments)]
fn coulomb_into_with<F: FnMut(usize, f64, &mut [f64], &mut [f64])>(
    mut rw: F,
    a: Prim,
    b: Prim,
    c: Prim,
    d: Prim,
    scale: f64,
    out: &mut [f64],
) {
    let (la, lb, lc, ld) = (a.l, b.l, c.l, d.l);
    let (nb, nc, nd) = (n_cart(lb), n_cart(lc), n_cart(ld));
    let p = a.exp + b.exp;
    let q = c.exp + d.exp;
    let p_ctr = combine(a.exp, a.center, b.exp, b.center, p);
    let q_ctr = combine(c.exp, c.center, d.exp, d.center, q);
    let kab = (-(a.exp * b.exp / p) * dist2(a.center, b.center)).exp();
    let kcd = (-(c.exp * d.exp / q) * dist2(c.center, d.center)).exp();
    let pq = p + q;
    let rho = p * q / pq;
    let pq_vec = sub(p_ctr, q_ctr);
    let t = rho * norm2(pq_vec);
    use std::f64::consts::PI;
    let pref = 2.0 * PI * PI * PI.sqrt() / (p * q * pq.sqrt()) * kab * kcd;
    let l_total = la + lb + lc + ld;
    let nroots = l_total / 2 + 1;
    let mut roots = [0.0f64; MAX_RYS_ROOTS];
    let mut wts = [0.0f64; MAX_RYS_ROOTS];
    rw(nroots, t, &mut roots, &mut wts);
    let pa = sub(p_ctr, a.center);
    let qc = sub(q_ctr, c.center);
    let ab = sub(a.center, b.center);
    let cd = sub(c.center, d.center);
    let comps_a = cart_components(la);
    let comps_b = cart_components(lb);
    let comps_c = cart_components(lc);
    let comps_d = cart_components(ld);
    let mut ix: Axis4 = zeroed();
    let mut iy: Axis4 = zeroed();
    let mut iz: Axis4 = zeroed();
    for r in 0..nroots {
        let x = roots[r];
        let b00 = x / (2.0 * pq);
        let b10 = (1.0 - x * q / pq) / (2.0 * p);
        let b01 = (1.0 - x * p / pq) / (2.0 * q);
        let cq = x * q / pq;
        let cp = x * p / pq;
        for (axis, out4) in [&mut ix, &mut iy, &mut iz].into_iter().enumerate() {
            let c00 = pa[axis] - cq * pq_vec[axis];
            let cp0 = qc[axis] + cp * pq_vec[axis];
            build_axis(
                la, lb, lc, ld, c00, cp0, b10, b01, b00, ab[axis], cd[axis], out4,
            );
        }
        let pw = scale * pref * wts[r];
        for (ia, ca) in comps_a.iter().enumerate() {
            for (ib, cb) in comps_b.iter().enumerate() {
                for (ic, cc) in comps_c.iter().enumerate() {
                    for (id, cd_) in comps_d.iter().enumerate() {
                        let v = ix[ca[0]][cb[0]][cc[0]][cd_[0]]
                            * iy[ca[1]][cb[1]][cc[1]][cd_[1]]
                            * iz[ca[2]][cb[2]][cc[2]][cd_[2]];
                        out[((ia * nb + ib) * nc + ic) * nd + id] += pw * v;
                    }
                }
            }
        }
    }
}

fn prim(exp: f64, c: [f64; 3], l: usize) -> Prim {
    Prim::new(exp, c, l)
}

/// Four non-collinear centres, varied exponents.
fn centres() -> [[f64; 3]; 4] {
    [
        [0.0, 0.0, 0.0],
        [0.7, -0.3, 0.2],
        [-0.4, 0.8, -0.1],
        [0.2, 0.5, 0.9],
    ]
}

#[test]
fn copy_is_faithful_to_public_coulomb_into() {
    // Production rw through the copy must equal the real public coulomb_into.
    let ctr = centres();
    let exps = [1.2, 0.9, 1.1, 0.7];
    let mut worst = 0.0_f64;
    for la in 0..=3 {
        for lb in 0..=3 {
            for lc in 0..=3 {
                for ld in 0..=3 {
                    let a = prim(exps[0], ctr[0], la);
                    let b = prim(exps[1], ctr[1], lb);
                    let c = prim(exps[2], ctr[2], lc);
                    let d = prim(exps[3], ctr[3], ld);
                    let len = n_cart(la) * n_cart(lb) * n_cart(lc) * n_cart(ld);
                    let mut real = vec![0.0; len];
                    let mut copy = vec![0.0; len];
                    coulomb_into(a, b, c, d, 1.0, &mut real);
                    coulomb_into_with(rys_roots_weights, a, b, c, d, 1.0, &mut copy);
                    for (x, y) in real.iter().zip(copy.iter()) {
                        worst = worst.max((x - y).abs());
                    }
                }
            }
        }
    }
    eprintln!("copy vs public coulomb_into: worst abs {worst:.3e}");
    assert!(
        worst < 1e-18,
        "copy is not faithful to the engine: {worst:e}"
    );
}

#[test]
fn interp_path_eri_matches_reference_path_eri() {
    // The decisive check: interp roots/weights vs Stieltjes-reference
    // roots/weights, all else identical, full L range, distinct centres.
    let ctr = centres();
    let exps = [1.2, 0.9, 1.1, 0.7];
    let mut worst_abs = 0.0_f64;
    let mut worst_ratio = 0.0_f64; // allclose atol=rtol=1e-11
    let mut worst_signif = 0.0_f64;
    let mut at = (0, 0, 0, 0);

    // The Stieltjes reference roots/weights are a deterministic pure function of
    // (nroots, T). Here T = ρ·|P−Q|² is *constant* across the whole sweep (exps and
    // centres are fixed) and nroots = l_total/2+1 takes only the values 1..=13, so the
    // discretized-Stieltjes + Golub–Welsch solve is run at most once per nroots instead
    // of once per quartet (~2401 calls → ~13). This is bit-identical (memoizing a pure
    // function cannot change its result) and preserves the full exhaustive L-sweep. The
    // win is modest: the test is dominated by the per-quartet `build_axis` recurrence
    // (run for both the interp and reference paths), not by the oracle — measured, the
    // isolated test goes ~17.3s → ~15.0s at opt-0 and ~3.34s → ~3.18s at opt-2 (it is
    // `opt-level = 2`, not this cache, that does the heavy lifting: 17.3s → 3.3s). The
    // cache is keyed on nroots via an O(1) array index; the `debug_assert` pins the
    // constant-T assumption so a future geometry-varying edit fails loudly rather than
    // silently returning a stale oracle.
    let mut ref_t: Option<u64> = None;
    let mut ref_cache: [Option<([f64; MAX_RYS_ROOTS], [f64; MAX_RYS_ROOTS])>; MAX_RYS_ROOTS + 1] =
        std::array::from_fn(|_| None);
    let mut ref_rw = |nroots: usize, t: f64, roots: &mut [f64], wts: &mut [f64]| {
        match ref_t {
            Some(bits) => debug_assert_eq!(bits, t.to_bits(), "ref cache assumes constant T"),
            None => ref_t = Some(t.to_bits()),
        }
        let entry = ref_cache[nroots].get_or_insert_with(|| {
            let mut r = [0.0; MAX_RYS_ROOTS];
            let mut w = [0.0; MAX_RYS_ROOTS];
            rys_roots_weights_reference(nroots, t, &mut r, &mut w);
            (r, w)
        });
        roots[..nroots].copy_from_slice(&entry.0[..nroots]);
        wts[..nroots].copy_from_slice(&entry.1[..nroots]);
    };

    for la in 0..=MAX_L {
        for lb in 0..=MAX_L {
            // keep l_total within engine cap (4*MAX_L=24, nroots<=13); also cap
            // the loop work — sample a representative spread.
            for lc in 0..=MAX_L {
                for ld in 0..=MAX_L {
                    if la + lb + lc + ld > 4 * MAX_L {
                        continue;
                    }
                    let a = prim(exps[0], ctr[0], la);
                    let b = prim(exps[1], ctr[1], lb);
                    let c = prim(exps[2], ctr[2], lc);
                    let d = prim(exps[3], ctr[3], ld);
                    let len = n_cart(la) * n_cart(lb) * n_cart(lc) * n_cart(ld);
                    let mut ip = vec![0.0; len];
                    let mut rf = vec![0.0; len];
                    coulomb_into_with(rys_roots_weights, a, b, c, d, 1.0, &mut ip);
                    coulomb_into_with(&mut ref_rw, a, b, c, d, 1.0, &mut rf);
                    let peak = rf.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
                    for (&x, &y) in ip.iter().zip(rf.iter()) {
                        let abse = (x - y).abs();
                        if abse > worst_abs {
                            worst_abs = abse;
                            at = (la, lb, lc, ld);
                        }
                        worst_ratio = worst_ratio.max(abse / (1e-11 + 1e-11 * y.abs()));
                        if y.abs() >= 1e-3 * peak {
                            worst_signif = worst_signif.max(abse / y.abs().max(1e-300));
                        }
                    }
                }
            }
        }
    }
    eprintln!(
        "interp-path vs reference-path ERI: worst_abs={worst_abs:.3e} at {at:?}; \
         allclose_ratio(1e-11)={worst_ratio:.3e}; max_rel_signif={worst_signif:.3e}"
    );
    assert!(
        worst_ratio <= 1.0,
        "interp and reference ERI paths differ beyond 1e-11 allclose: ratio={worst_ratio:e} abs={worst_abs:e} at {at:?}"
    );
}
