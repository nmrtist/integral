//! Permanent guard: OS-vs-Rys agreement at the *true* angular-momentum maxima
//! (`l_total` up to 24 — `gggg`/`hhhh`/`iiii`), the asymmetric maxima, and an
//! independent McMurchie-Davidson cross-check pushed to a g-containing quartet.
//!
//! These exercise the true max and the non-zero HRR-shift geometries (four
//! distinct centres), spanning the full angular-momentum range up to
//! `l_total = 24`.
//! Run with `--release` — the forced-OS `(iiii)` corner is slow in debug.

use integral::math::am::{cart_components, n_cart};
use integral::math::boys::boys_array;
use integral::math::norm::cart_norm;
use integral::{Basis, Engine, Shell};

/// metrics: (worst_abs, worst_rel_signif over |ref| >= 1e-3*peak, peak)
fn metrics(os: &[f64], rys: &[f64]) -> (f64, f64, f64) {
    let peak = rys.iter().fold(0.0f64, |m, &r| m.max(r.abs()));
    let floor = 1e-3 * peak;
    let mut wabs = 0.0f64;
    let mut wsig = 0.0f64;
    for (o, r) in os.iter().zip(rys.iter()) {
        let a = (o - r).abs();
        wabs = wabs.max(a);
        if r.abs() >= floor {
            wsig = wsig.max(a / r.abs());
        }
    }
    (wabs, wsig, peak)
}

fn quartet(ls: [usize; 4]) -> Basis {
    let cs = [
        [0.0, 0.0, 0.0],
        [0.5, -0.3, 0.2],
        [-0.4, 0.6, -0.1],
        [0.2, 0.4, 0.8],
    ];
    let es = [0.9, 1.3, 0.7, 1.1];
    Basis::new(
        (0..4)
            .map(|k| Shell::new(ls[k], cs[k], vec![es[k]], vec![1.0]).unwrap())
            .collect(),
    )
}

/// Run the forced OS-vs-Rys cross-engine check over a set of `(la,lb,lc,ld)`
/// quartets on four distinct non-collinear centres, asserting agreement.
fn run_cross_engine(cases: &[[usize; 4]]) {
    let mut bad = Vec::new();
    for &ls in cases {
        let b = quartet(ls);
        let os = b.eri_block_with(Engine::OsHgp, 0, 1, 2, 3);
        let rys = b.eri_block_with(Engine::Rys, 0, 1, 2, 3);
        let (wabs, wsig, peak) = metrics(&os, &rys);
        eprintln!(
            "({},{},{},{}) lt={:2} n={:6} peak={:.2e} worst_abs={:.2e} worst_rel_signif={:.2e}",
            ls[0],
            ls[1],
            ls[2],
            ls[3],
            ls.iter().sum::<usize>(),
            os.len(),
            peak,
            wabs,
            wsig
        );
        if wsig > 1e-9 {
            bad.push(format!("{ls:?} worst_rel_signif={wsig:e}"));
        }
        if wabs > 1e-9 * peak.max(1.0) + 1e-10 {
            bad.push(format!("{ls:?} worst_abs={wabs:e} peak={peak:e}"));
        }
    }
    assert!(bad.is_empty(), "cross-engine divergence: {bad:?}");
}

/// Default-suite cross-engine maxima: `gggg` (l_total 16) and the asymmetric /
/// mixed quartets whose forced-OS cost is modest in debug. The heavier corners
/// (`hhhh`/`iiii` and the l_total 18–20 cases) live in
/// [`cross_engine_high_l_corners`], `#[ignore]`d so `cargo test` stays fast; CI
/// runs them in release (see `.github/workflows/ci.yml`).
#[test]
fn cross_engine_maxima_and_asymmetric() {
    run_cross_engine(&[
        [4, 4, 4, 4], // gggg  l_total 16
        [6, 0, 6, 0], // is|is  ne=6 nf=6
        [6, 6, 0, 0], // ii|ss  ne=12 nf=0 (max bra, zero ket)
        [0, 0, 6, 6], // ss|ii  ne=0 nf=12 (zero bra, max ket)
        [0, 3, 4, 6], // sf|gi  l_total 13 mixed all-distinct
        [6, 4, 3, 0], // ig|fs
    ]);
}

/// The expensive high-L corners — `hhhh` (l_total 20), `iiii` (l_total 24, the
/// true maximum), and the l_total 18–20 asymmetric cases. Forced-OS at the
/// `iiii` corner is ~3.7 s in release (≈30 s in debug), so this is
/// `#[ignore]`d by default and run in CI's dedicated release step
/// (`cargo test -p integral --release -- --include-ignored`). It is the only check
/// that exercises OS≡Rys at the true `l_total = 24` maximum with non-zero HRR
/// shifts (four distinct centres).
#[test]
#[ignore = "expensive high-L corner (l_total up to 24); run in release via --include-ignored"]
fn cross_engine_high_l_corners() {
    run_cross_engine(&[
        [5, 5, 5, 5], // hhhh  l_total 20
        [6, 6, 6, 6], // iiii  l_total 24 (the true max)
        [6, 6, 6, 0], // iii s  ne=12 nf=6  l_total 18
        [2, 6, 6, 6], // d iii  ne=8 nf=12  l_total 20
    ]);
}

// ---- independent McMurchie-Davidson primitive (self-contained) ----

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

/// Hermite Coulomb integrals `R^0_{tuv}` for all `t+u+v ≤ lmax`, returned as a
/// flat table `[(t·D + u)·D + v]`, `D = lmax+1`. Built iteratively from the same
/// MD auxiliary recursion as the (validated) recursive form, but tabulated so the
/// cost is `O(lmax⁴)` instead of exponential — which makes the (gg|gg)/(hh|hh)
/// corners feasible. `R^n_{000} = (−2ρ)^n F_n`; for `t>0`,
/// `R^n_{t,u,v} = (t−1)·R^{n+1}_{t−2,u,v} + X_PQ·R^{n+1}_{t−1,u,v}` (then `u`, then
/// `v`) — the exact relation the recursive version uses.
fn hermite_r_table(lmax: usize, fm: &[f64], two_rho: f64, pq: [f64; 3]) -> Vec<f64> {
    let d = lmax + 1;
    let mut layers: Vec<Vec<f64>> = (0..=lmax)
        .map(|n| {
            let mut l = vec![0.0f64; d * d * d];
            l[0] = (-two_rho).powi(n as i32) * fm[n]; // R^n_{000}
            l
        })
        .collect();
    for s in 1..=lmax {
        for n in 0..=(lmax - s) {
            for t in 0..=s {
                for u in 0..=(s - t) {
                    let v = s - t - u;
                    let next = &layers[n + 1];
                    let val = if t > 0 {
                        pq[0] * next[((t - 1) * d + u) * d + v]
                            + if t >= 2 {
                                (t as f64 - 1.0) * next[((t - 2) * d + u) * d + v]
                            } else {
                                0.0
                            }
                    } else if u > 0 {
                        pq[1] * next[(u - 1) * d + v]
                            + if u >= 2 {
                                (u as f64 - 1.0) * next[(u - 2) * d + v]
                            } else {
                                0.0
                            }
                    } else {
                        pq[2] * next[v - 1]
                            + if v >= 2 {
                                (v as f64 - 1.0) * next[v - 2]
                            } else {
                                0.0
                            }
                    };
                    layers[n][(t * d + u) * d + v] = val;
                }
            }
        }
    }
    layers.swap_remove(0)
}

/// MD Hermite-expansion coefficients `E^t_{ij}` for one axis, tabulated as a flat
/// `[(i·(jmax+1) + j)·(imax+jmax+1) + t]` array — removes the exponential cost of
/// re-recursing `e_coeff` for every output component.
fn e_table(imax: usize, jmax: usize, q: f64, a: f64, b: f64) -> Vec<f64> {
    let tdim = imax + jmax + 1;
    let mut tbl = vec![0.0; (imax + 1) * (jmax + 1) * tdim];
    for i in 0..=imax {
        for j in 0..=jmax {
            for t in 0..=(i + j) {
                tbl[(i * (jmax + 1) + j) * tdim + t] =
                    e_coeff(i as i64, j as i64, t as i64, q, a, b);
            }
        }
    }
    tbl
}

#[allow(clippy::too_many_arguments)]
fn md_primitive(
    ea: f64,
    ca: [f64; 3],
    la: usize,
    eb: f64,
    cb: [f64; 3],
    lb: usize,
    ec: f64,
    cc: [f64; 3],
    lc: usize,
    ed: f64,
    cd: [f64; 3],
    ld: usize,
) -> Vec<f64> {
    let p = ea + eb;
    let q = ec + ed;
    let comb = |e1: f64, c1: [f64; 3], e2: f64, c2: [f64; 3], s: f64| {
        [
            (e1 * c1[0] + e2 * c2[0]) / s,
            (e1 * c1[1] + e2 * c2[1]) / s,
            (e1 * c1[2] + e2 * c2[2]) / s,
        ]
    };
    let pc = comb(ea, ca, eb, cb, p);
    let qc = comb(ec, cc, ed, cd, q);
    let rho = p * q / (p + q);
    let pq = [pc[0] - qc[0], pc[1] - qc[1], pc[2] - qc[2]];
    let tparam = rho * (pq[0] * pq[0] + pq[1] * pq[1] + pq[2] * pq[2]);
    let lmax = la + lb + lc + ld;
    let mut fm = vec![0.0; lmax + 1];
    boys_array(lmax, tparam, &mut fm);
    let two_rho = 2.0 * rho;
    let pref = 2.0 * std::f64::consts::PI.powf(2.5) / (p * q * (p + q).sqrt());
    let (na, nb, nc, nd) = (n_cart(la), n_cart(lb), n_cart(lc), n_cart(ld));
    let (cca, ccb, ccc, ccd) = (
        cart_components(la),
        cart_components(lb),
        cart_components(lc),
        cart_components(ld),
    );
    let ab = [ca[0] - cb[0], ca[1] - cb[1], ca[2] - cb[2]];
    let cdv = [cc[0] - cd[0], cc[1] - cd[1], cc[2] - cd[2]];
    // Tabulate the Hermite-R tensor and the per-axis E-coefficients once.
    let dd = lmax + 1;
    let rtab = hermite_r_table(lmax, &fm, two_rho, pq);
    let tb = la + lb + 1; // bra t-dimension
    let tk = lc + ld + 1; // ket s-dimension
    let ebx = e_table(la, lb, ab[0], ea, eb);
    let eby = e_table(la, lb, ab[1], ea, eb);
    let ebz = e_table(la, lb, ab[2], ea, eb);
    let ekx = e_table(lc, ld, cdv[0], ec, ed);
    let eky = e_table(lc, ld, cdv[1], ec, ed);
    let ekz = e_table(lc, ld, cdv[2], ec, ed);
    let be = |tbl: &[f64], i: usize, j: usize, t: usize| tbl[(i * (lb + 1) + j) * tb + t];
    let ke = |tbl: &[f64], i: usize, j: usize, s: usize| tbl[(i * (ld + 1) + j) * tk + s];

    let mut out = vec![0.0; na * nb * nc * nd];
    for (ia, va) in cca.iter().enumerate() {
        for (ib, vb) in ccb.iter().enumerate() {
            for (ic, vc) in ccc.iter().enumerate() {
                for (id, vd) in ccd.iter().enumerate() {
                    let mut sum = 0.0;
                    for tx in 0..=(va[0] + vb[0]) {
                        let ex = be(&ebx, va[0], vb[0], tx);
                        if ex == 0.0 {
                            continue;
                        }
                        for ty in 0..=(va[1] + vb[1]) {
                            let ey = be(&eby, va[1], vb[1], ty);
                            if ey == 0.0 {
                                continue;
                            }
                            for tz in 0..=(va[2] + vb[2]) {
                                let ez = be(&ebz, va[2], vb[2], tz);
                                let ebra = ex * ey * ez;
                                if ebra == 0.0 {
                                    continue;
                                }
                                for sx in 0..=(vc[0] + vd[0]) {
                                    let fx = ke(&ekx, vc[0], vd[0], sx);
                                    if fx == 0.0 {
                                        continue;
                                    }
                                    for sy in 0..=(vc[1] + vd[1]) {
                                        let fy = ke(&eky, vc[1], vd[1], sy);
                                        if fy == 0.0 {
                                            continue;
                                        }
                                        for sz in 0..=(vc[2] + vd[2]) {
                                            let fz = ke(&ekz, vc[2], vd[2], sz);
                                            let eket = fx * fy * fz;
                                            if eket == 0.0 {
                                                continue;
                                            }
                                            let sign =
                                                if (sx + sy + sz) % 2 == 0 { 1.0 } else { -1.0 };
                                            let r =
                                                rtab[((tx + sx) * dd + (ty + sy)) * dd + (tz + sz)];
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

/// Forced-OS block of a single-primitive quartet `ls` (on the four fixed
/// `quartet` centers) vs the independent MD primitive engine: `(worst_abs,
/// worst_rel_signif, peak)`.
fn md_vs_os_metrics(ls: [usize; 4]) -> (f64, f64, f64) {
    let cs = [
        [0.0, 0.0, 0.0],
        [0.5, -0.3, 0.2],
        [-0.4, 0.6, -0.1],
        [0.2, 0.4, 0.8],
    ];
    let es = [0.9, 1.3, 0.7, 1.1];
    let b = quartet(ls);
    let s = b.shells();
    let os = b.eri_block_with(Engine::OsHgp, 0, 1, 2, 3);
    let w: Vec<f64> = (0..4)
        .map(|k| s[k].coefficients()[0] * cart_norm(es[k], ls[k], 0, 0))
        .collect();
    let md0 = md_primitive(
        es[0], cs[0], ls[0], es[1], cs[1], ls[1], es[2], cs[2], ls[2], es[3], cs[3], ls[3],
    );
    let scale = w[0] * w[1] * w[2] * w[3];
    let md: Vec<f64> = md0.iter().map(|x| x * scale).collect();
    metrics(&os, &md)
}

fn run_md_value_cases(cases: &[[usize; 4]]) {
    let mut bad = Vec::new();
    for &ls in cases {
        let (wabs, wsig, peak) = md_vs_os_metrics(ls);
        eprintln!(
            "MD ({},{},{},{}) lt={:2} peak={:.2e} worst_abs={:.2e} worst_rel_signif={:.2e}",
            ls[0],
            ls[1],
            ls[2],
            ls[3],
            ls.iter().sum::<usize>(),
            peak,
            wabs,
            wsig
        );
        if wsig > 1e-9 || wabs > 1e-9 * peak.max(1.0) + 1e-10 {
            bad.push(format!("{ls:?} abs={wabs:e} sig={wsig:e}"));
        }
    }
    assert!(bad.is_empty(), "OS vs MD divergence: {bad:?}");
}

#[test]
fn os_matches_independent_md_g_quartets() {
    // Through g (l=4) and into h (l=5) at modest total angular momentum — the
    // cheap cases that stay fast in the default (debug) suite.
    run_md_value_cases(&[
        [4, 0, 0, 0],
        [4, 1, 0, 0],
        [4, 0, 4, 0],
        [4, 1, 2, 0],
        [2, 2, 4, 0],
        [4, 4, 0, 0],
        [5, 0, 0, 0], // h
        [5, 1, 0, 0],
        [5, 4, 0, 0], // (hg|ss) l_total 9
        [5, 0, 5, 0], // (h|h)   l_total 10
    ]);
}

/// Independent-reference validation of the *high-l corners* the engines are
/// positioned for: (gg|gg) (l_total 16) and (hh|hh) (l_total 20). Previously
/// these were validated only OS-vs-Rys (a shared-convention blind spot: both
/// engines share the Boys table, prefactor, and overall convention, so an
/// OS≡Rys agreement cannot catch a systematic error common to both). The MD
/// Hermite-tensor path is a genuinely different algorithm. `#[ignore]`d because
/// the recursion-based MD reference at l_total 20 is slow; run in release via
/// `--include-ignored`.
#[test]
#[ignore = "expensive: independent MD reference at l_total up to 20 ((hh|hh)); run in release"]
fn os_matches_independent_md_high_l_corners() {
    run_md_value_cases(&[
        [4, 4, 4, 4], // gggg l_total 16
        [5, 5, 4, 4], // hhgg l_total 18
        [5, 5, 5, 5], // hhhh l_total 20
    ]);
}

#[test]
fn boys_m24_matches_quadrature_reference() {
    for &t in &[0.0, 0.5, 5.0, 13.0, 30.0, 36.0, 60.0, 120.0] {
        let mut out = vec![0.0; 25];
        boys_array(24, t, &mut out);
        let m = 24usize;
        let n = 2_000_000usize;
        let h = 1.0 / n as f64;
        let f = |x: f64| x.powi(2 * m as i32) * (-t * x * x).exp();
        let mut sum = f(0.0) + f(1.0);
        let mut comp = 0.0f64;
        for i in 1..n {
            let x = i as f64 * h;
            let wt = if i % 2 == 0 { 2.0 } else { 4.0 };
            let y = wt * f(x) - comp;
            let nw = sum + y;
            comp = (nw - sum) - y;
            sum = nw;
        }
        let q = sum * h / 3.0;
        let rel = (out[24] - q).abs() / q.abs().max(1e-300);
        eprintln!("F_24({t}) = {:.6e} ref {:.6e} rel {:e}", out[24], q, rel);
        assert!(rel < 1e-7, "F_24({t}) rel err {rel:e}");
    }
}

// ---- C5: contracted quartet vs INDEPENDENT MD sum (not Rys) ----
#[test]
fn contracted_quartet_matches_independent_md_sum() {
    // Genuinely contracted: K>1 on three shells, distinct non-collinear centers.
    // Reference is the MD primitive engine summed over the primitive quartet with
    // the same effective (radial-normalized) coefficients — fully independent of
    // both integral engines. Confirms VRR-contract-HRR equals per-primitive truth.
    use integral::Shell;
    let ca = [0.0, 0.0, 0.0];
    let cb = [0.6, -0.2, 0.1];
    let cc = [-0.3, 0.5, -0.2];
    let cd = [0.2, 0.3, 0.7];
    let (la, lb, lc, ld) = (1usize, 1, 2, 0);
    let ax = vec![1.4, 0.45];
    let acf = vec![0.6, 0.5];
    let bx = vec![0.9, 0.3];
    let bcf = vec![0.55, 0.5];
    let cx = vec![1.1, 0.4];
    let ccf = vec![0.7, 0.4];
    let dx = vec![0.8];
    let dcf = vec![1.0];

    let basis = Basis::new(vec![
        Shell::new(la, ca, ax.clone(), acf.clone()).unwrap(),
        Shell::new(lb, cb, bx.clone(), bcf.clone()).unwrap(),
        Shell::new(lc, cc, cx.clone(), ccf.clone()).unwrap(),
        Shell::new(ld, cd, dx.clone(), dcf.clone()).unwrap(),
    ]);
    let os = basis.eri_block_with(Engine::OsHgp, 0, 1, 2, 3);

    // MD sum with effective coeffs (radial cart_norm * contraction coeff).
    let nrm = |e: f64, l: usize| cart_norm(e, l, 0, 0);
    let mut md = vec![0.0; os.len()];
    for (&ea, &wa) in ax.iter().zip(&acf) {
        for (&eb, &wb) in bx.iter().zip(&bcf) {
            for (&ec, &wc) in cx.iter().zip(&ccf) {
                for (&ed, &wd) in dx.iter().zip(&dcf) {
                    let blk = md_primitive(ea, ca, la, eb, cb, lb, ec, cc, lc, ed, cd, ld);
                    let w = (wa * nrm(ea, la))
                        * (wb * nrm(eb, lb))
                        * (wc * nrm(ec, lc))
                        * (wd * nrm(ed, ld));
                    for (o, v) in md.iter_mut().zip(&blk) {
                        *o += w * v;
                    }
                }
            }
        }
    }
    let (wabs, wsig, peak) = metrics(&os, &md);
    eprintln!(
        "contracted OS vs MD: peak={peak:.2e} worst_abs={wabs:.2e} worst_rel_signif={wsig:.2e}"
    );
    assert!(
        wsig < 1e-9 && wabs < 1e-9 * peak.max(1.0) + 1e-10,
        "abs={wabs:e} sig={wsig:e}"
    );
}

/// Heavy-element wide-exponent / deep-contraction stress. Each shell carries a
/// contraction spanning ~9 orders of magnitude in exponent (a tight core 1e7–1e8
/// alongside diffuse 1e-2) — the regime where tight cores drive the Boys argument
/// `T = ρ|P−Q|²` into 1e6–1e8 (asymptotic branch, `e^{−T}=0`) and the contraction
/// sum mixes vastly different magnitudes. Forces BOTH engines AND an independent
/// MD primitive sum to agree, guarding: contraction-sum precision, the
/// `PAIR_NEGLIGIBLE` screen on tight×diffuse cross terms, and the large-T Boys
/// path under realistic heavy-element conditions. No existing test combined wide
/// exponents with the ERI path.
#[test]
fn wide_exponent_deep_contraction_matches_md_and_rys() {
    use integral::Shell;
    let ca = [0.0, 0.0, 0.0];
    let cb = [0.7, -0.3, 0.2];
    let cc = [-0.4, 0.6, -0.1];
    let cd = [0.3, 0.2, 0.8];
    let (la, lb, lc, ld) = (2usize, 1, 2, 0);
    // Contractions spanning tight core → diffuse.
    let ax = vec![1.0e7, 1.0e2, 1.0, 1.0e-2];
    let acf = vec![0.2, 0.5, 0.6, 0.4];
    let bx = vec![5.0e6, 3.0, 0.05];
    let bcf = vec![0.3, 0.6, 0.5];
    let cx = vec![1.0e8, 10.0, 0.3];
    let ccf = vec![0.1, 0.5, 0.7];
    let dx = vec![2.0e7, 0.7];
    let dcf = vec![0.2, 0.8];
    let basis = Basis::new(vec![
        Shell::new(la, ca, ax.clone(), acf.clone()).unwrap(),
        Shell::new(lb, cb, bx.clone(), bcf.clone()).unwrap(),
        Shell::new(lc, cc, cx.clone(), ccf.clone()).unwrap(),
        Shell::new(ld, cd, dx.clone(), dcf.clone()).unwrap(),
    ]);
    let os = basis.eri_block_with(Engine::OsHgp, 0, 1, 2, 3);
    let rys = basis.eri_block_with(Engine::Rys, 0, 1, 2, 3);

    let nrm = |e: f64, l: usize| cart_norm(e, l, 0, 0);
    let mut md = vec![0.0; os.len()];
    for (&ea, &wa) in ax.iter().zip(&acf) {
        for (&eb, &wb) in bx.iter().zip(&bcf) {
            for (&ec, &wc) in cx.iter().zip(&ccf) {
                for (&ed, &wd) in dx.iter().zip(&dcf) {
                    let blk = md_primitive(ea, ca, la, eb, cb, lb, ec, cc, lc, ed, cd, ld);
                    let w = (wa * nrm(ea, la))
                        * (wb * nrm(eb, lb))
                        * (wc * nrm(ec, lc))
                        * (wd * nrm(ed, ld));
                    for (o, v) in md.iter_mut().zip(&blk) {
                        *o += w * v;
                    }
                }
            }
        }
    }
    let (oa, osig, peak) = metrics(&os, &md);
    let (ra, rsig, _) = metrics(&rys, &md);
    eprintln!("wide-exp OS vs MD:  peak={peak:.3e} worst_abs={oa:.3e} worst_sig={osig:.3e}");
    eprintln!("wide-exp Rys vs MD: worst_abs={ra:.3e} worst_sig={rsig:.3e}");
    assert!(
        osig < 1e-8 && oa < 1e-8 * peak.max(1.0) + 1e-12,
        "OS vs MD: abs={oa:e} sig={osig:e}"
    );
    assert!(
        rsig < 1e-8 && ra < 1e-8 * peak.max(1.0) + 1e-12,
        "Rys vs MD: abs={ra:e} sig={rsig:e}"
    );
}

// ---- F13: forced-OS layout falsifiability with nc==nd ----
#[test]
fn forced_os_layout_transpose_is_detected() {
    use integral::Shell;
    // (s p | d d') with nc==nd==6 but the two d shells DISTINCT, so the block is
    // non-symmetric in (c,d). Forced-OS dense slice must match the forced-OS block
    // under row-major (a,b,c,d); a (c,d) transpose must break it.
    let basis = Basis::new(vec![
        Shell::new(0, [0.0, 0.0, 0.0], vec![1.2], vec![1.0]).unwrap(),
        Shell::new(1, [0.7, -0.3, 0.2], vec![0.9], vec![1.0]).unwrap(),
        Shell::new(2, [-0.4, 0.8, -0.1], vec![1.1], vec![1.0]).unwrap(),
        Shell::new(2, [0.3, -0.6, 0.4], vec![0.75], vec![1.0]).unwrap(),
    ]);
    let (i, j, k, l) = (0usize, 1, 2, 3);
    let block = basis.eri_block_with(Engine::OsHgp, i, j, k, l);
    let (na, nb, nc, nd) = (1usize, 3, 6, 6);
    assert_eq!(block.len(), na * nb * nc * nd);
    // Transpose c,d:
    let mut t = vec![0.0; block.len()];
    for a in 0..na {
        for b in 0..nb {
            for c in 0..nc {
                for d in 0..nd {
                    t[((a * nb + b) * nc + c) * nd + d] = block[((a * nb + b) * nc + d) * nd + c];
                }
            }
        }
    }
    let diff = block
        .iter()
        .zip(&t)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max);
    eprintln!("forced-OS (sp|dd') c,d-transpose max diff = {diff:e}");
    assert!(
        diff > 1e-3,
        "transpose should change the block (non-symmetric in c,d)"
    );
}
