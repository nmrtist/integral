//! Independent cross-check of the Rys ERI engine against a **McMurchie–Davidson**
//! Coulomb implementation (Hermite expansion + Hermite Coulomb `R`-tensor).
//!
//! MD is a genuinely different algorithm from Rys quadrature, so agreement
//! validates the engine's *values*, not just its internal consistency — the
//! two-electron analogue of the MD overlap/kinetic cross-check. Runs on every
//! platform (no external library).

use integral::math::am::{cart_components, n_cart};
use integral::math::boys::boys_array;
use integral::math::norm::cart_norm;
use integral::{Basis, Engine, Shell};

/// McMurchie–Davidson Hermite expansion coefficient `E^t_{ij}` for a 1D Gaussian
/// product with center separation `q = A − B` and exponents `a, b` (same
/// recurrence as the MD overlap cross-check).
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

/// Hermite Coulomb integral `R_{tuv} = R^0_{tuv}` via the downward auxiliary
/// recursion `R^n_{000} = (-2ρ)^n F_n(T)`,
/// `R^n_{t+1,u,v} = t R^{n+1}_{t-1,u,v} + X R^{n+1}_{t,u,v}` (and `u, v`).
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

/// One primitive (ab|cd) over Cartesian components `(la,lb,lc,ld)` via MD,
/// returned row-major over the four component indices (matching integral).
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
    let comps = |l: usize| cart_components(l);
    let (ca, cb, cc, cd) = (comps(a.l), comps(b.l), comps(c.l), comps(d.l));
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

fn combine(a: P, b: P, p: f64) -> [f64; 3] {
    [
        (a.e * a.c[0] + b.e * b.c[0]) / p,
        (a.e * a.c[1] + b.e * b.c[1]) / p,
        (a.e * a.c[2] + b.e * b.c[2]) / p,
    ]
}

/// Contracted MD ERI block for four shells, normalized like integral (per-primitive
/// `cart_norm(α, l, 0, 0)`), row-major over `(a, b, c, d)`.
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

/// A basis with four shells of distinct angular momenta on distinct centers, so
/// every (ij|kl) block is non-trivial.
fn mixed_basis() -> Basis {
    Basis::new(vec![
        Shell::new(0, [0.0, 0.0, 0.0], vec![1.2, 0.5], vec![0.6, 0.5]).unwrap(), // s (contracted)
        Shell::new(1, [0.7, -0.3, 0.2], vec![0.9], vec![1.0]).unwrap(),          // p
        Shell::new(2, [-0.4, 0.8, -0.1], vec![1.1], vec![1.0]).unwrap(),         // d
        Shell::new(1, [0.2, 0.5, 0.9], vec![0.65], vec![1.0]).unwrap(),          // p
    ])
}

#[test]
fn eri_matches_mcmurchie_davidson_through_d() {
    let basis = mixed_basis();
    let s = basis.shells();
    // A spread of quartets with mixed L on each center, up to d.
    let quartets = [
        (0, 0, 0, 0), // (ss|ss)
        (1, 0, 0, 0), // (ps|ss)
        (1, 1, 1, 1), // (pp|pp)
        (2, 0, 1, 0), // (ds|ps)
        (2, 1, 2, 1), // (dp|dp)
        (0, 2, 1, 2), // (sd|pd)
        (2, 2, 2, 2), // (dd|dd)
    ];
    for (i, j, k, l) in quartets {
        let ox = basis.eri_block(i, j, k, l);
        let md = md_block(&s[i], &s[j], &s[k], &s[l]);
        let re = max_rel(&ox, &md);
        assert!(
            re < 1e-11,
            "(l{} l{} | l{} l{}) vs MD max_rel = {re:e}",
            s[i].l(),
            s[j].l(),
            s[k].l(),
            s[l].l()
        );
    }
}

/// A basis including an f shell on a distinct centre, for the OS/HGP × MD check.
fn f_basis() -> Basis {
    Basis::new(vec![
        Shell::new(0, [0.0, 0.0, 0.0], vec![1.2, 0.5], vec![0.6, 0.5]).unwrap(), // s (contracted)
        Shell::new(1, [0.7, -0.3, 0.2], vec![0.9], vec![1.0]).unwrap(),          // p
        Shell::new(2, [-0.4, 0.8, -0.1], vec![1.1], vec![1.0]).unwrap(),         // d
        Shell::new(3, [0.2, 0.5, 0.9], vec![0.65], vec![1.0]).unwrap(),          // f
    ])
}

/// The **OS/HGP** engine (forced) must match the independent McMurchie–Davidson
/// path through f-containing quartets — the independent value check for the
/// second engine, including the contracted bra and `(dd|ff)` mixed high-L.
#[test]
fn oshgp_matches_mcmurchie_davidson_through_f() {
    let basis = f_basis();
    let s = basis.shells();
    let quartets = [
        (3, 0, 0, 0), // (fs|ss)
        (0, 1, 2, 3), // (sp|df) — all four L distinct
        (3, 0, 1, 2), // (fs|pd)
        (3, 3, 0, 0), // (ff|ss)
        (2, 2, 3, 3), // (dd|ff) — mixed high-L
        (1, 3, 2, 3), // (pf|df)
    ];
    for (i, j, k, l) in quartets {
        let ox = basis.eri_block_with(Engine::OsHgp, i, j, k, l);
        let md = md_block(&s[i], &s[j], &s[k], &s[l]);
        let re = max_rel(&ox, &md);
        assert!(
            re < 1e-10,
            "OS (l{} l{} | l{} l{}) vs MD max_rel = {re:e}",
            s[i].l(),
            s[j].l(),
            s[k].l(),
            s[l].l()
        );
    }
}
