//! Extra ERI checks beyond the core suite.
//!
//!  * C8 stride SENSITIVITY — confirm the layout comparison can actually fail:
//!    integral's `(s p | d d')` block matches an independent McMurchie–Davidson
//!    block under the documented row-major `(a,b,c,d)` layout, but does NOT match
//!    once the last two axes are deliberately transposed. A test that cannot fail
//!    proves nothing; this shows it can.
//!  * C10 independent third source to f — the same MD path (Hermite E-coeffs +
//!    Coulomb R-tensor, no external library) is extended to f-containing quartets,
//!    incl. all-distinct mixed-AM (s p | d f) and (f f | s s).
//!
//! The MD implementation here is a self-contained copy (so this file is a fully
//! independent cross-check), mirroring `tests/eri_md_cross_check.rs`.

use integral::math::am::{cart_components, n_cart};
use integral::math::boys::boys_array;
use integral::math::norm::cart_norm;
use integral::{Basis, Shell};

fn e_coeff(i: i64, j: i64, t: i64, q: f64, a: f64, b: f64) -> f64 {
    let p = a + b;
    let mu = a * b / p;
    if t < 0 || t > i + j {
        return 0.0;
    }
    if i == 0 && j == 0 && t == 0 {
        return (-mu * q * q).exp();
    }
    if j == 0 {
        (1.0 / (2.0 * p)) * e_coeff(i - 1, j, t - 1, q, a, b)
            - (mu * q / a) * e_coeff(i - 1, j, t, q, a, b)
            + (t as f64 + 1.0) * e_coeff(i - 1, j, t + 1, q, a, b)
    } else {
        (1.0 / (2.0 * p)) * e_coeff(i, j - 1, t - 1, q, a, b)
            + (mu * q / b) * e_coeff(i, j - 1, t, q, a, b)
            + (t as f64 + 1.0) * e_coeff(i, j - 1, t + 1, q, a, b)
    }
}

fn hermite_r(t: i64, u: i64, v: i64, n: usize, fm: &[f64], two_rho: f64, pq: [f64; 3]) -> f64 {
    if t < 0 || u < 0 || v < 0 {
        return 0.0;
    }
    if t == 0 && u == 0 && v == 0 {
        return (-two_rho).powi(n as i32) * fm[n];
    }
    if t > 0 {
        (t as f64 - 1.0) * hermite_r(t - 2, u, v, n + 1, fm, two_rho, pq)
            + pq[0] * hermite_r(t - 1, u, v, n + 1, fm, two_rho, pq)
    } else if u > 0 {
        (u as f64 - 1.0) * hermite_r(t, u - 2, v, n + 1, fm, two_rho, pq)
            + pq[1] * hermite_r(t, u - 1, v, n + 1, fm, two_rho, pq)
    } else {
        (v as f64 - 1.0) * hermite_r(t, u, v - 2, n + 1, fm, two_rho, pq)
            + pq[2] * hermite_r(t, u, v - 1, n + 1, fm, two_rho, pq)
    }
}

#[derive(Clone, Copy)]
struct P {
    e: f64,
    c: [f64; 3],
    l: usize,
}

fn combine(a: P, b: P, p: f64) -> [f64; 3] {
    [
        (a.e * a.c[0] + b.e * b.c[0]) / p,
        (a.e * a.c[1] + b.e * b.c[1]) / p,
        (a.e * a.c[2] + b.e * b.c[2]) / p,
    ]
}

fn md_primitive(a: P, b: P, c: P, d: P) -> Vec<f64> {
    let p = a.e + b.e;
    let q = c.e + d.e;
    let pc = combine(a, b, p);
    let qc = combine(c, d, q);
    let rho = p * q / (p + q);
    let pq = [pc[0] - qc[0], pc[1] - qc[1], pc[2] - qc[2]];
    let t_param = rho * (pq[0] * pq[0] + pq[1] * pq[1] + pq[2] * pq[2]);
    let lbra = a.l + b.l;
    let lket = c.l + d.l;
    let lmax = lbra + lket;
    let mut fm = vec![0.0; lmax + 1];
    boys_array(lmax, t_param, &mut fm);
    let two_rho = 2.0 * rho;
    let pref = 2.0 * std::f64::consts::PI.powf(2.5) / (p * q * (p + q).sqrt());
    let (na, nb, nc, nd) = (n_cart(a.l), n_cart(b.l), n_cart(c.l), n_cart(d.l));
    let (ca, cb, cc, cd) = (
        cart_components(a.l),
        cart_components(b.l),
        cart_components(c.l),
        cart_components(d.l),
    );
    let ab = [a.c[0] - b.c[0], a.c[1] - b.c[1], a.c[2] - b.c[2]];
    let cdv = [c.c[0] - d.c[0], c.c[1] - d.c[1], c.c[2] - d.c[2]];
    let mut out = vec![0.0; na * nb * nc * nd];
    for (ia, la) in ca.iter().enumerate() {
        for (ib, lb) in cb.iter().enumerate() {
            for (ic, lc) in cc.iter().enumerate() {
                for (id, ld) in cd.iter().enumerate() {
                    let mut sum = 0.0;
                    for tx in 0..=(la[0] + lb[0]) {
                        let ex = e_coeff(la[0] as i64, lb[0] as i64, tx as i64, ab[0], a.e, b.e);
                        for ty in 0..=(la[1] + lb[1]) {
                            let ey =
                                e_coeff(la[1] as i64, lb[1] as i64, ty as i64, ab[1], a.e, b.e);
                            for tz in 0..=(la[2] + lb[2]) {
                                let ez =
                                    e_coeff(la[2] as i64, lb[2] as i64, tz as i64, ab[2], a.e, b.e);
                                let ebra = ex * ey * ez;
                                if ebra == 0.0 {
                                    continue;
                                }
                                for sx in 0..=(lc[0] + ld[0]) {
                                    let fx = e_coeff(
                                        lc[0] as i64,
                                        ld[0] as i64,
                                        sx as i64,
                                        cdv[0],
                                        c.e,
                                        d.e,
                                    );
                                    for sy in 0..=(lc[1] + ld[1]) {
                                        let fy = e_coeff(
                                            lc[1] as i64,
                                            ld[1] as i64,
                                            sy as i64,
                                            cdv[1],
                                            c.e,
                                            d.e,
                                        );
                                        for sz in 0..=(lc[2] + ld[2]) {
                                            let fz = e_coeff(
                                                lc[2] as i64,
                                                ld[2] as i64,
                                                sz as i64,
                                                cdv[2],
                                                c.e,
                                                d.e,
                                            );
                                            let eket = fx * fy * fz;
                                            if eket == 0.0 {
                                                continue;
                                            }
                                            let sign =
                                                if (sx + sy + sz) % 2 == 0 { 1.0 } else { -1.0 };
                                            let r = hermite_r(
                                                (tx + sx) as i64,
                                                (ty + sy) as i64,
                                                (tz + sz) as i64,
                                                0,
                                                &fm,
                                                two_rho,
                                                pq,
                                            );
                                            sum += ebra * eket * sign * r;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    out[((ia * nb + ib) * nc + ic) * nd + id] = pref * sum;
                }
            }
        }
    }
    out
}

fn md_block(sa: &Shell, sb: &Shell, sc: &Shell, sd: &Shell) -> Vec<f64> {
    let prims = |s: &Shell| -> Vec<(f64, P)> {
        (0..s.n_prim())
            .map(|i| {
                let e = s.exponents()[i];
                let coeff = s.coefficients()[i] * cart_norm(e, s.l(), 0, 0);
                (
                    coeff,
                    P {
                        e,
                        c: s.center(),
                        l: s.l(),
                    },
                )
            })
            .collect()
    };
    let (pa, pb, pc, pd) = (prims(sa), prims(sb), prims(sc), prims(sd));
    let len = sa.n_cart() * sb.n_cart() * sc.n_cart() * sd.n_cart();
    let mut acc = vec![0.0; len];
    for (wa, a) in &pa {
        for (wb, b) in &pb {
            for (wc, c) in &pc {
                for (wd, d) in &pd {
                    let blk = md_primitive(*a, *b, *c, *d);
                    let w = wa * wb * wc * wd;
                    for (o, v) in acc.iter_mut().zip(blk.iter()) {
                        *o += w * v;
                    }
                }
            }
        }
    }
    acc
}

fn max_rel(x: &[f64], y: &[f64]) -> f64 {
    x.iter()
        .zip(y.iter())
        .map(|(&a, &b)| (a - b).abs() / b.abs().max(1e-300))
        .fold(0.0_f64, f64::max)
}

/// numpy-`allclose` worst ratio: `max |a-b| / (atol + rtol·|b|)`; ≤ 1 ⇔ pass.
/// This is the project's grounding ERI criterion: an `atol` floor
/// admits benign near-cancellation elements (tiny `|b|` with tiny absolute
/// error) while the `rtol` term tightly bounds the dominant elements. A
/// floor-less pure-relative metric, by contrast, is dominated by the cancellation
/// tail — a sub-ulp perturbation of a structurally-small element (e.g. the
/// Phase-5a Rys interpolation, ~2e-14 in roots/weights) inflates its *relative*
/// error without any real loss of accuracy.
fn allclose_ratio(x: &[f64], y: &[f64], atol: f64, rtol: f64) -> f64 {
    x.iter()
        .zip(y.iter())
        .map(|(&a, &b)| (a - b).abs() / (atol + rtol * b.abs()))
        .fold(0.0_f64, f64::max)
}

/// Worst relative error among **significant** elements (`|ref| ≥ 1e-3 · block
/// peak`) — the dominant regime `rtol` governs (the `max_rel_signif` metric).
fn max_rel_signif(x: &[f64], y: &[f64]) -> f64 {
    let peak = y.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    x.iter()
        .zip(y.iter())
        .filter(|(_, &b)| b.abs() >= 1e-3 * peak)
        .map(|(&a, &b)| (a - b).abs() / b.abs().max(1e-300))
        .fold(0.0_f64, f64::max)
}

/// s, p, d, f on four distinct centers (every block non-trivial), plus a 2nd d.
fn fbasis() -> Basis {
    Basis::new(vec![
        Shell::new(0, [0.0, 0.0, 0.0], vec![1.2, 0.5], vec![0.6, 0.5]).unwrap(), // 0 s
        Shell::new(1, [0.7, -0.3, 0.2], vec![0.9], vec![1.0]).unwrap(),          // 1 p
        Shell::new(2, [-0.4, 0.8, -0.1], vec![1.1], vec![1.0]).unwrap(),         // 2 d
        Shell::new(3, [0.2, 0.5, 0.9], vec![0.7], vec![1.0]).unwrap(),           // 3 f
        Shell::new(2, [0.3, -0.6, 0.4], vec![0.75], vec![1.0]).unwrap(),         // 4 d' (≠ d)
    ])
}

#[test]
fn eri_matches_md_through_f() {
    // C10: independent MD source extended to f-containing quartets.
    let basis = fbasis();
    let s = basis.shells();
    let quartets = [
        (0, 1, 2, 3), // (s p | d f)  — all distinct, mixed AM (the shape/stride case)
        (3, 0, 1, 2), // (f s | p d)
        (3, 3, 0, 0), // (f f | s s)
        (2, 3, 3, 1), // (d f | f p)
        (1, 3, 2, 3), // (p f | d f)
    ];
    for (i, j, k, l) in quartets {
        let ox = basis.eri_block(i, j, k, l);
        let md = md_block(&s[i], &s[j], &s[k], &s[l]);
        // Cancellation-robust gate: allclose at atol=rtol=1e-11
        // (tighter than the former floor-less max_rel<1e-10 on the dominant
        // elements), plus the significant-element relative for grounding.
        let ratio = allclose_ratio(&ox, &md, 1e-11, 1e-11);
        let signif = max_rel_signif(&ox, &md);
        assert!(
            ratio <= 1.0 && signif < 1e-10,
            "(l{} l{}|l{} l{}) vs MD: allclose_ratio={ratio:e} max_rel_signif={signif:e}",
            s[i].l(),
            s[j].l(),
            s[k].l(),
            s[l].l()
        );
    }
}

#[test]
fn stride_sensitivity_transpose_is_detected() {
    // C8: the layout comparison must be able to FAIL. Build (s p | d d') where the
    // last two shells are both d (n_cart 6) but DIFFERENT (distinct exps/centers),
    // so the block is genuinely non-symmetric in (c,d). integral matches MD under the
    // documented layout; transposing the last two axes must break the match.
    let basis = fbasis();
    let s = basis.shells();
    let (i, j, k, l) = (0usize, 1usize, 2usize, 4usize); // s p | d d'
    let ox = basis.eri_block(i, j, k, l);
    let md = md_block(&s[i], &s[j], &s[k], &s[l]);

    let (na, nb, nc, nd) = (
        n_cart(s[i].l()),
        n_cart(s[j].l()),
        n_cart(s[k].l()),
        n_cart(s[l].l()),
    );
    assert_eq!(
        nc, nd,
        "test requires equal last-axis dims for a clean transpose"
    );

    // Correct layout: must agree (cancellation-robust allclose).
    let ratio_ok = allclose_ratio(&ox, &md, 1e-11, 1e-11);
    assert!(
        ratio_ok <= 1.0,
        "correct layout should match MD: allclose_ratio={ratio_ok:e}"
    );

    // Transpose the last two axes (c <-> d) of MD, then compare to integral.
    let mut md_t = vec![0.0; ox.len()];
    for a in 0..na {
        for b in 0..nb {
            for c in 0..nc {
                for d in 0..nd {
                    let dst = ((a * nb + b) * nc + c) * nd + d;
                    let src = ((a * nb + b) * nc + d) * nd + c; // swapped c,d
                    md_t[dst] = md[src];
                }
            }
        }
    }
    let re_swapped = max_rel(&ox, &md_t);
    assert!(
        re_swapped > 1e-3,
        "transposing c,d should make the blocks DISAGREE, but max_rel was only \
         {re_swapped:e} — the layout test is not sensitive to a transposition"
    );
}
