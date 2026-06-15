//! Cross-algorithm corroboration of the OS/HGP ERI engine: for a representative
//! set of shell quartets (spanning low to high angular momentum, including
//! `(dd|ff)`, `(ff|ff)`, `(sf|gi)`, and `(ii|ii)`), each block computed by the
//! OS/HGP engine is recomputed with two **independent algorithms** and asserted
//! to agree at the tight `1e-11` tolerance:
//!   (a) an **independent McMurchie–Davidson** Coulomb path (Hermite E-coeffs +
//!       Hermite R-tensor — a different algorithm), and
//!   (b) the **Rys** engine (quadrature — also a different algorithm).
//! Three independent algorithms agreeing pins the values from mathematical
//! principle, with no external reference software.
//!
//! MD is `O(deg(bra)³·deg(ket)³)` per element with deep recursion, so it is run only
//! on the feasible cases (`l_total ≤ 10`); the high-L `(sf|gi)`/`(ii|ii)` cases are
//! corroborated by Rys (whose own MD agreement is established elsewhere up through f).

use integral::engine::os::{Prim, Vec3};
use integral::engine::os_eri::{coulomb_shell_into, ShellRef};
use integral::engine::rys::coulomb_into;
use integral::math::am::{cart_components, n_cart};
use integral::math::boys::boys_array;

// ----- the representative quartet set -----

const CA: Vec3 = [0.0, 0.0, 0.0];
const CB: Vec3 = [0.5, -0.3, 0.2];
const CC: Vec3 = [-0.4, 0.6, -0.1];
const CD: Vec3 = [0.2, 0.4, 0.8];

#[derive(Clone)]
struct ShellSpec {
    l: usize,
    center: Vec3,
    exps: Vec<f64>,
    coeffs: Vec<f64>,
}
impl ShellSpec {
    fn single(l: usize, center: Vec3, exp: f64) -> Self {
        ShellSpec {
            l,
            center,
            exps: vec![exp],
            coeffs: vec![1.0],
        }
    }
}

struct Case {
    label: &'static str,
    md_feasible: bool,
    shells: [ShellSpec; 4],
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            label: "sp_df_0123",
            md_feasible: true,
            shells: [
                ShellSpec::single(0, CA, 0.9),
                ShellSpec::single(1, CB, 1.3),
                ShellSpec::single(2, CC, 0.7),
                ShellSpec::single(3, CD, 1.1),
            ],
        },
        Case {
            label: "ddff",
            md_feasible: true,
            shells: [
                ShellSpec::single(2, CA, 0.9),
                ShellSpec::single(2, CB, 1.3),
                ShellSpec::single(3, CC, 0.7),
                ShellSpec::single(3, CD, 1.1),
            ],
        },
        Case {
            label: "ffff",
            md_feasible: true,
            shells: [
                ShellSpec::single(3, CA, 0.9),
                ShellSpec::single(3, CB, 1.3),
                ShellSpec::single(3, CC, 0.7),
                ShellSpec::single(3, CD, 1.1),
            ],
        },
        Case {
            label: "isis",
            md_feasible: true, // l_total = 12 but bra/ket each one s shell → small Hermite sums
            shells: [
                ShellSpec::single(6, CA, 0.9),
                ShellSpec::single(0, CB, 1.3),
                ShellSpec::single(6, CC, 0.7),
                ShellSpec::single(0, CD, 1.1),
            ],
        },
        Case {
            label: "sfgi",
            md_feasible: false, // l_total = 13, lket = 10 → MD too slow; Rys-corroborated
            shells: [
                ShellSpec::single(0, CA, 0.9),
                ShellSpec::single(3, CB, 1.3),
                ShellSpec::single(4, CC, 0.7),
                ShellSpec::single(6, CD, 1.1),
            ],
        },
        Case {
            label: "iiii",
            md_feasible: false, // l_total = 24 → MD wholly infeasible; Rys-corroborated
            shells: [
                ShellSpec::single(6, CA, 0.9),
                ShellSpec::single(6, CB, 1.3),
                ShellSpec::single(6, CC, 0.7),
                ShellSpec::single(6, CD, 1.1),
            ],
        },
        Case {
            label: "contracted_ppds",
            md_feasible: true,
            shells: [
                ShellSpec {
                    l: 1,
                    center: CA,
                    exps: vec![1.4, 0.45],
                    coeffs: vec![0.6, 0.5],
                },
                ShellSpec {
                    l: 1,
                    center: CB,
                    exps: vec![0.9, 0.3],
                    coeffs: vec![0.55, 0.5],
                },
                ShellSpec {
                    l: 2,
                    center: CC,
                    exps: vec![1.1, 0.4],
                    coeffs: vec![0.7, 0.4],
                },
                ShellSpec::single(0, CD, 0.8),
            ],
        },
    ]
}

fn block_len(c: &Case) -> usize {
    let s = &c.shells;
    n_cart(s[0].l) * n_cart(s[1].l) * n_cart(s[2].l) * n_cart(s[3].l)
}

/// The OS/HGP engine block for a case, in the raw-monomial convention.
fn os_block(c: &Case) -> Vec<f64> {
    let s = &c.shells;
    let mut out = vec![0.0; block_len(c)];
    let r = |i: usize| ShellRef {
        center: s[i].center,
        l: s[i].l,
        exps: &s[i].exps,
        coeffs: &s[i].coeffs,
    };
    coulomb_shell_into(r(0), r(1), r(2), r(3), &mut out);
    out
}

/// Rys block: sum the Rys primitive engine over the quartet with product coeffs
/// (same raw-monomial convention — coeffs as given, no extra normalization).
fn rys_block(c: &Case) -> Vec<f64> {
    let s = &c.shells;
    let mut out = vec![0.0; block_len(c)];
    for (&ea, &wa) in s[0].exps.iter().zip(&s[0].coeffs) {
        for (&eb, &wb) in s[1].exps.iter().zip(&s[1].coeffs) {
            for (&ec, &wc) in s[2].exps.iter().zip(&s[2].coeffs) {
                for (&ed, &wd) in s[3].exps.iter().zip(&s[3].coeffs) {
                    coulomb_into(
                        Prim::new(ea, s[0].center, s[0].l),
                        Prim::new(eb, s[1].center, s[1].l),
                        Prim::new(ec, s[2].center, s[2].l),
                        Prim::new(ed, s[3].center, s[3].l),
                        wa * wb * wc * wd,
                        &mut out,
                    );
                }
            }
        }
    }
    out
}

// ----- independent McMurchie–Davidson Coulomb (self-contained) -----

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
    let lmax = a.l + b.l + c.l + d.l;
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

fn md_block(c: &Case) -> Vec<f64> {
    let s = &c.shells;
    let mk = |i: usize, e: f64| P {
        e,
        c: s[i].center,
        l: s[i].l,
    };
    let mut out = vec![0.0; block_len(c)];
    for (&ea, &wa) in s[0].exps.iter().zip(&s[0].coeffs) {
        for (&eb, &wb) in s[1].exps.iter().zip(&s[1].coeffs) {
            for (&ec, &wc) in s[2].exps.iter().zip(&s[2].coeffs) {
                for (&ed, &wd) in s[3].exps.iter().zip(&s[3].coeffs) {
                    let blk = md_primitive(mk(0, ea), mk(1, eb), mk(2, ec), mk(3, ed));
                    let w = wa * wb * wc * wd;
                    for (o, v) in out.iter_mut().zip(&blk) {
                        *o += w * v;
                    }
                }
            }
        }
    }
    out
}

/// Worst `(|x−y| − atol) / |y|` over significant elements; ≤ 0 means all pass
/// `|x−y| ≤ atol + rtol·|y|`. Returns (worst_signif_rel, worst_abs).
fn agree(x: &[f64], y: &[f64], atol: f64, rtol: f64) -> (f64, f64) {
    let peak = y.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    let mut worst_rel = 0.0_f64;
    let mut worst_abs = 0.0_f64;
    for (&a, &b) in x.iter().zip(y) {
        let d = (a - b).abs();
        worst_abs = worst_abs.max(d);
        // significant element: not a structurally-tiny near-cancellation component.
        if b.abs() >= 1e-6 * peak && b.abs() > 0.0 {
            let slack = (d - atol) / b.abs();
            worst_rel = worst_rel.max(slack);
        }
        assert!(
            d <= atol + rtol * b.abs(),
            "element disagreement Δ={d:e} (a={a:e} b={b:e})"
        );
    }
    (worst_rel, worst_abs)
}

/// The OS/HGP block agrees with the Rys engine for **every** case, including
/// (sf|gi) and (ii|ii), at the tight tolerance.
#[test]
fn blocks_match_rys() {
    const ATOL: f64 = 1e-11;
    const RTOL: f64 = 1e-10;
    for case in cases() {
        let os = os_block(&case);
        let rys = rys_block(&case);
        let (wr, wa) = agree(&os, &rys, ATOL, RTOL);
        eprintln!(
            "{:<16} vs Rys: worst_signif_slack={wr:.2e} worst_abs={wa:.2e}",
            case.label
        );
    }
}

/// The OS/HGP block agrees with an independent McMurchie–Davidson Coulomb path on
/// the MD-feasible cases — an algorithm with no shared recurrence machinery and no
/// external-library dependence.
#[test]
fn blocks_match_independent_md() {
    const ATOL: f64 = 1e-11;
    const RTOL: f64 = 1e-10;
    let mut covered = Vec::new();
    for case in cases() {
        if !case.md_feasible {
            continue;
        }
        let os = os_block(&case);
        let md = md_block(&case);
        let (wr, wa) = agree(&os, &md, ATOL, RTOL);
        eprintln!(
            "{:<16} vs MD : worst_signif_slack={wr:.2e} worst_abs={wa:.2e}",
            case.label
        );
        covered.push(case.label);
    }
    assert!(
        covered.contains(&"ffff") && covered.contains(&"ddff") && covered.contains(&"sp_df_0123"),
        "MD corroboration must cover the high-L stored cases"
    );
}
