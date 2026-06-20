//! Independent cross-check of s/p overlap & kinetic using a McMurchie–Davidson
//! (MD) Hermite-expansion implementation — a *different* recurrence than integral's
//! Obara–Saika engine — so agreement is not an artifact of a shared algorithm.
//! Runs natively (no external library).
//!
//! The MD primitive overlap uses Hermite coefficients E^t_{ij}; the kinetic uses
//! the standard ∇² expansion in terms of MD 1D overlaps. Both are contracted and
//! normalized with integral's documented convention (per-primitive `cart_norm(α,l,0,0)`)
//! and compared against the public `Basis` API.

use integral::math::am::cart_components;
use integral::math::norm::cart_norm;
use integral::{Basis, Shell};

/// McMurchie–Davidson Hermite expansion coefficient E^t_{ij} for two 1D
/// Gaussians (exponent `a` at `A`, exponent `b` at `B`), `q = A - B`.
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
        // decrement i
        (1.0 / (2.0 * p)) * e_coeff(i - 1, j, t - 1, q, a, b)
            - (mu * q / a) * e_coeff(i - 1, j, t, q, a, b)
            + (t as f64 + 1.0) * e_coeff(i - 1, j, t + 1, q, a, b)
    } else {
        // decrement j
        (1.0 / (2.0 * p)) * e_coeff(i, j - 1, t - 1, q, a, b)
            + (mu * q / b) * e_coeff(i, j - 1, t, q, a, b)
            + (t as f64 + 1.0) * e_coeff(i, j - 1, t + 1, q, a, b)
    }
}

/// 1D primitive overlap S(i,j) via MD.
fn s1(i: i64, j: i64, q: f64, a: f64, b: f64) -> f64 {
    e_coeff(i, j, 0, q, a, b) * (std::f64::consts::PI / (a + b)).sqrt()
}

/// 1D kinetic factor K(i,j) acting on the ket (exponent `b`), standard ∇² form.
fn k1(i: i64, j: i64, q: f64, a: f64, b: f64) -> f64 {
    let term_minus = if j >= 2 {
        (j * (j - 1)) as f64 * s1(i, j - 2, q, a, b)
    } else {
        0.0
    };
    let term_mid = 2.0 * b * (2 * j + 1) as f64 * s1(i, j, q, a, b);
    let term_plus = 4.0 * b * b * s1(i, j + 2, q, a, b);
    -0.5 * (term_minus - term_mid + term_plus)
}

/// MD primitive 3D overlap for Cartesian powers `la=(lx,ly,lz)`, `lb=(...)`.
fn md_overlap(la: [usize; 3], lb: [usize; 3], a: f64, ca: [f64; 3], b: f64, cb: [f64; 3]) -> f64 {
    (0..3)
        .map(|ax| s1(la[ax] as i64, lb[ax] as i64, ca[ax] - cb[ax], a, b))
        .product()
}

/// MD primitive 3D kinetic.
fn md_kinetic(la: [usize; 3], lb: [usize; 3], a: f64, ca: [f64; 3], b: f64, cb: [f64; 3]) -> f64 {
    let q = [ca[0] - cb[0], ca[1] - cb[1], ca[2] - cb[2]];
    let sx = s1(la[0] as i64, lb[0] as i64, q[0], a, b);
    let sy = s1(la[1] as i64, lb[1] as i64, q[1], a, b);
    let sz = s1(la[2] as i64, lb[2] as i64, q[2], a, b);
    let kx = k1(la[0] as i64, lb[0] as i64, q[0], a, b);
    let ky = k1(la[1] as i64, lb[1] as i64, q[1], a, b);
    let kz = k1(la[2] as i64, lb[2] as i64, q[2], a, b);
    kx * sy * sz + sx * ky * sz + sx * sy * kz
}

/// A shell description: (l, center, exponents, contraction coeffs).
type ShellSpec = (usize, [f64; 3], Vec<f64>, Vec<f64>);

/// Reference contracted, normalized matrix (overlap or kinetic) for a 2-shell
/// basis, using the MD primitive routines and integral's normalization convention.
fn md_matrix(shells: &[ShellSpec], kinetic: bool) -> Vec<f64> {
    // AO offsets.
    let ncart = |l: usize| (l + 1) * (l + 2) / 2;
    let offs: Vec<usize> = {
        let mut acc = 0;
        shells
            .iter()
            .map(|(l, ..)| {
                let o = acc;
                acc += ncart(*l);
                o
            })
            .collect()
    };
    let n: usize = shells.iter().map(|(l, ..)| ncart(*l)).sum();
    let mut mat = vec![0.0; n * n];

    for (si, (la, ca, ea, da)) in shells.iter().enumerate() {
        for (sj, (lb, cb, eb, db)) in shells.iter().enumerate() {
            let comps_a = cart_components(*la);
            let comps_b = cart_components(*lb);
            for (ia, compa) in comps_a.iter().enumerate() {
                for (ib, compb) in comps_b.iter().enumerate() {
                    let mut acc = 0.0;
                    for (pa, &alpha) in ea.iter().enumerate() {
                        for (pb, &beta) in eb.iter().enumerate() {
                            let na = da[pa] * cart_norm(alpha, *la, 0, 0);
                            let nb = db[pb] * cart_norm(beta, *lb, 0, 0);
                            let prim = if kinetic {
                                md_kinetic(*compa, *compb, alpha, *ca, beta, *cb)
                            } else {
                                md_overlap(*compa, *compb, alpha, *ca, beta, *cb)
                            };
                            acc += na * nb * prim;
                        }
                    }
                    let (r, c) = (offs[si] + ia, offs[sj] + ib);
                    mat[r * n + c] = acc;
                }
            }
        }
    }
    mat
}

fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs() / y.abs().max(1e-12))
        .fold(0.0_f64, f64::max)
}

/// Significance-weighted comparison: `(worst_abs, worst_rel over |ref| ≥ 1e-3·peak,
/// peak)`. Avoids the near-zero blow-up of [`max_rel`] for high-l multi-center
/// bases that legitimately have ~0 elements.
fn sig_metrics(got: &[f64], reference: &[f64]) -> (f64, f64, f64) {
    let peak = reference.iter().fold(0.0f64, |m, &r| m.max(r.abs()));
    let floor = 1e-3 * peak;
    let mut wabs = 0.0f64;
    let mut wsig = 0.0f64;
    for (&g, &r) in got.iter().zip(reference) {
        let d = (g - r).abs();
        wabs = wabs.max(d);
        if r.abs() >= floor {
            wsig = wsig.max(d / r.abs());
        }
    }
    (wabs, wsig, peak)
}

// ---- Independent MD nuclear-attraction and dipole references ----

/// Hermite Coulomb integral `R^n_{tuv}` for nuclear attraction (Helgaker MD):
/// base `R^n_{000} = (−2p)^n F_n(p·R_PC²)`, descending recursion in `t,u,v`. This
/// is the MD Hermite-tensor path — a different recurrence than the engine's
/// Obara–Saika vertical/horizontal build.
fn hermite_r(t: i64, u: i64, v: i64, n: usize, fm: &[f64], two_p: f64, rpc: [f64; 3]) -> f64 {
    if t < 0 || u < 0 || v < 0 {
        return 0.0;
    }
    if t == 0 && u == 0 && v == 0 {
        return (-two_p).powi(n as i32) * fm[n];
    }
    if t > 0 {
        (t as f64 - 1.0) * hermite_r(t - 2, u, v, n + 1, fm, two_p, rpc)
            + rpc[0] * hermite_r(t - 1, u, v, n + 1, fm, two_p, rpc)
    } else if u > 0 {
        (u as f64 - 1.0) * hermite_r(t, u - 2, v, n + 1, fm, two_p, rpc)
            + rpc[1] * hermite_r(t, u - 1, v, n + 1, fm, two_p, rpc)
    } else {
        (v as f64 - 1.0) * hermite_r(t, u, v - 2, n + 1, fm, two_p, rpc)
            + rpc[2] * hermite_r(t, u, v - 1, n + 1, fm, two_p, rpc)
    }
}

/// MD primitive nuclear attraction `Σ_C (−Z_C) ⟨a| 1/|r−C| |b⟩` for the Cartesian
/// power triplets `la`, `lb`. Uses `integral`'s validated Boys ladder as the shared
/// radial kernel (separately quadrature-checked); the angular path is independent.
fn md_nuclear(
    la: [usize; 3],
    lb: [usize; 3],
    a: f64,
    ca: [f64; 3],
    b: f64,
    cb: [f64; 3],
    charges: &[([f64; 3], f64)],
) -> f64 {
    let p = a + b;
    let pcent = [
        (a * ca[0] + b * cb[0]) / p,
        (a * ca[1] + b * cb[1]) / p,
        (a * ca[2] + b * cb[2]) / p,
    ];
    let ab = [ca[0] - cb[0], ca[1] - cb[1], ca[2] - cb[2]];
    let lmax = la[0] + la[1] + la[2] + lb[0] + lb[1] + lb[2];
    let two_p = 2.0 * p;
    let mut total = 0.0;
    for &(c, z) in charges {
        let rpc = [pcent[0] - c[0], pcent[1] - c[1], pcent[2] - c[2]];
        let r2 = rpc[0] * rpc[0] + rpc[1] * rpc[1] + rpc[2] * rpc[2];
        let mut fm = vec![0.0; lmax + 1];
        integral::math::boys::boys_array(lmax, p * r2, &mut fm);
        let mut sum = 0.0;
        for t in 0..=(la[0] + lb[0]) {
            let ex = e_coeff(la[0] as i64, lb[0] as i64, t as i64, ab[0], a, b);
            if ex == 0.0 {
                continue;
            }
            for u in 0..=(la[1] + lb[1]) {
                let ey = e_coeff(la[1] as i64, lb[1] as i64, u as i64, ab[1], a, b);
                if ey == 0.0 {
                    continue;
                }
                for w in 0..=(la[2] + lb[2]) {
                    let ez = e_coeff(la[2] as i64, lb[2] as i64, w as i64, ab[2], a, b);
                    if ez == 0.0 {
                        continue;
                    }
                    let r = hermite_r(t as i64, u as i64, w as i64, 0, &fm, two_p, rpc);
                    sum += ex * ey * ez * r;
                }
            }
        }
        total += (-z) * (2.0 * std::f64::consts::PI / p) * sum;
    }
    total
}

/// MD primitive dipole `⟨a| (r−O)_k |b⟩` for `k = x,y,z` about origin `o`. Uses
/// the identity `(x−O) = (x−B) + (B−O)`, so the moment is `S(i,j+1)+(B−O)·S(i,j)`
/// from the same 1D MD overlaps — independent of the engine's multipole recurrence.
fn md_dipole(
    la: [usize; 3],
    lb: [usize; 3],
    a: f64,
    ca: [f64; 3],
    b: f64,
    cb: [f64; 3],
    o: [f64; 3],
) -> [f64; 3] {
    let q = [ca[0] - cb[0], ca[1] - cb[1], ca[2] - cb[2]];
    let s = [
        s1(la[0] as i64, lb[0] as i64, q[0], a, b),
        s1(la[1] as i64, lb[1] as i64, q[1], a, b),
        s1(la[2] as i64, lb[2] as i64, q[2], a, b),
    ];
    let mom = [
        s1(la[0] as i64, lb[0] as i64 + 1, q[0], a, b) + (cb[0] - o[0]) * s[0],
        s1(la[1] as i64, lb[1] as i64 + 1, q[1], a, b) + (cb[1] - o[1]) * s[1],
        s1(la[2] as i64, lb[2] as i64 + 1, q[2], a, b) + (cb[2] - o[2]) * s[2],
    ];
    [
        mom[0] * s[1] * s[2],
        s[0] * mom[1] * s[2],
        s[0] * s[1] * mom[2],
    ]
}

/// Reference contracted, normalized nuclear-attraction matrix for a basis of
/// Cartesian shells, via the MD primitive routine and `integral`'s normalization.
fn md_nuclear_matrix(shells: &[ShellSpec], charges: &[([f64; 3], f64)]) -> Vec<f64> {
    let ncart = |l: usize| (l + 1) * (l + 2) / 2;
    let mut acc = 0;
    let offs: Vec<usize> = shells
        .iter()
        .map(|(l, ..)| {
            let o = acc;
            acc += ncart(*l);
            o
        })
        .collect();
    let n: usize = shells.iter().map(|(l, ..)| ncart(*l)).sum();
    let mut mat = vec![0.0; n * n];
    for (si, (la, ca, ea, da)) in shells.iter().enumerate() {
        for (sj, (lb, cb, eb, db)) in shells.iter().enumerate() {
            for (ia, compa) in cart_components(*la).iter().enumerate() {
                for (ib, compb) in cart_components(*lb).iter().enumerate() {
                    let mut v = 0.0;
                    for (pa, &alpha) in ea.iter().enumerate() {
                        for (pb, &beta) in eb.iter().enumerate() {
                            let na = da[pa] * cart_norm(alpha, *la, 0, 0);
                            let nb = db[pb] * cart_norm(beta, *lb, 0, 0);
                            v += na
                                * nb
                                * md_nuclear(*compa, *compb, alpha, *ca, beta, *cb, charges);
                        }
                    }
                    mat[(offs[si] + ia) * n + offs[sj] + ib] = v;
                }
            }
        }
    }
    mat
}

/// Reference contracted, normalized dipole matrices `[Dx, Dy, Dz]` about `o`.
fn md_dipole_matrix(shells: &[ShellSpec], o: [f64; 3]) -> [Vec<f64>; 3] {
    let ncart = |l: usize| (l + 1) * (l + 2) / 2;
    let mut acc = 0;
    let offs: Vec<usize> = shells
        .iter()
        .map(|(l, ..)| {
            let off = acc;
            acc += ncart(*l);
            off
        })
        .collect();
    let n: usize = shells.iter().map(|(l, ..)| ncart(*l)).sum();
    let mut mats = [vec![0.0; n * n], vec![0.0; n * n], vec![0.0; n * n]];
    for (si, (la, ca, ea, da)) in shells.iter().enumerate() {
        for (sj, (lb, cb, eb, db)) in shells.iter().enumerate() {
            for (ia, compa) in cart_components(*la).iter().enumerate() {
                for (ib, compb) in cart_components(*lb).iter().enumerate() {
                    let mut v = [0.0; 3];
                    for (pa, &alpha) in ea.iter().enumerate() {
                        for (pb, &beta) in eb.iter().enumerate() {
                            let na = da[pa] * cart_norm(alpha, *la, 0, 0);
                            let nb = db[pb] * cart_norm(beta, *lb, 0, 0);
                            let d = md_dipole(*compa, *compb, alpha, *ca, beta, *cb, o);
                            for k in 0..3 {
                                v[k] += na * nb * d[k];
                            }
                        }
                    }
                    let idx = (offs[si] + ia) * n + offs[sj] + ib;
                    for k in 0..3 {
                        mats[k][idx] = v[k];
                    }
                }
            }
        }
    }
    mats
}

/// A heavy-element-shaped basis: shells s…i (l = 0..=6) across three centers,
/// some contracted, off-axis geometry. The shared fixture for the high-l value
/// cross-checks.
fn high_l_shells() -> Vec<ShellSpec> {
    vec![
        (0usize, [0.0, 0.0, 0.0], vec![1.6, 0.5], vec![0.55, 0.45]),
        (2, [0.0, 0.0, 0.0], vec![0.9], vec![1.0]),
        (4, [0.7, -0.4, 0.3], vec![0.8], vec![1.0]),  // g
        (5, [0.7, -0.4, 0.3], vec![0.6], vec![1.0]),  // h
        (6, [-0.5, 0.6, -0.2], vec![0.7], vec![1.0]), // i
        (3, [-0.5, 0.6, -0.2], vec![1.1, 0.4], vec![0.5, 0.5]), // f, contracted
    ]
}

fn high_l_basis() -> Basis {
    Basis::new(
        high_l_shells()
            .iter()
            .map(|(l, c, e, d)| Shell::new(*l, *c, e.clone(), d.clone()).unwrap())
            .collect(),
    )
}

#[test]
fn sp_overlap_and_kinetic_match_mcmurchie_davidson() {
    // s + p mix on two centers, contracted, off-axis geometry.
    let shells = vec![
        (0usize, [0.0, 0.0, 0.0], vec![1.8, 0.5], vec![0.6, 0.4]),
        (1usize, [0.7, -0.4, 0.9], vec![1.2, 0.3], vec![0.5, 0.55]),
    ];
    let basis = Basis::new(
        shells
            .iter()
            .map(|(l, c, e, d)| Shell::new(*l, *c, e.clone(), d.clone()).unwrap())
            .collect(),
    );

    let s_ox = basis.overlap();
    let s_md = md_matrix(&shells, false);
    let re_s = max_rel(&s_ox, &s_md);
    assert!(re_s < 1e-12, "overlap vs MD max_rel = {re_s:e}");

    let t_ox = basis.kinetic();
    let t_md = md_matrix(&shells, true);
    let re_t = max_rel(&t_ox, &t_md);
    assert!(re_t < 1e-12, "kinetic vs MD max_rel = {re_t:e}");

    println!("MD cross-check: overlap max_rel={re_s:e}, kinetic max_rel={re_t:e}");
}

#[test]
fn pp_overlap_matches_mcmurchie_davidson() {
    // pure p·p, single primitives, generic geometry.
    let shells = vec![
        (1usize, [0.1, 0.2, -0.3], vec![0.9], vec![1.0]),
        (1usize, [-0.5, 0.8, 0.4], vec![1.7], vec![1.0]),
    ];
    let basis = Basis::new(
        shells
            .iter()
            .map(|(l, c, e, d)| Shell::new(*l, *c, e.clone(), d.clone()).unwrap())
            .collect(),
    );
    let re = max_rel(&basis.overlap(), &md_matrix(&shells, false));
    assert!(re < 1e-12, "p·p overlap vs MD max_rel = {re:e}");
}

/// Overlap and kinetic across the FULL l range (s…i, l = 0..=6) against the
/// independent MD reference. This is the first *analytic value* check above l=1
/// for these operators — previously h/i values were guarded only by symmetry +
/// translational invariance, which a consistent recurrence/normalization bug can
/// satisfy. Covers every `(la, lb)` pair up to `(6, 6)` in one basis.
#[test]
fn high_l_overlap_kinetic_match_md() {
    let shells = high_l_shells();
    let basis = high_l_basis();

    let (sa, ss, sp) = sig_metrics(&basis.overlap(), &md_matrix(&shells, false));
    println!("high-l overlap vs MD: peak={sp:e} worst_abs={sa:e} worst_sig={ss:e}");
    assert!(
        ss < 1e-10 && sa < 1e-10 * sp.max(1.0),
        "overlap abs={sa:e} sig={ss:e}"
    );

    let (ta, ts, tp) = sig_metrics(&basis.kinetic(), &md_matrix(&shells, true));
    println!("high-l kinetic vs MD: peak={tp:e} worst_abs={ta:e} worst_sig={ts:e}");
    assert!(
        ts < 1e-10 && ta < 1e-10 * tp.max(1.0),
        "kinetic abs={ta:e} sig={ts:e}"
    );
}

/// Nuclear attraction across l = 0..=6 against an independent MD Hermite-tensor
/// reference (multi-charge, off-center), closing the gap where the only prior
/// analytic nuclear check was a single s|s pair.
#[test]
fn high_l_nuclear_matches_md() {
    let shells = high_l_shells();
    let basis = high_l_basis();
    // Physical-style charges sitting on (and off) the shell centers.
    let charges = [
        ([0.0, 0.0, 0.0], 3.0),
        ([0.7, -0.4, 0.3], 5.0),
        ([-0.5, 0.6, -0.2], 2.0),
        ([0.9, 0.2, -0.6], 1.5),
    ];
    let got = basis.nuclear(&charges);
    let reference = md_nuclear_matrix(&shells, &charges);
    let (a, sig, peak) = sig_metrics(&got, &reference);
    println!("high-l nuclear vs MD: peak={peak:e} worst_abs={a:e} worst_sig={sig:e}");
    assert!(
        sig < 1e-9 && a < 1e-9 * peak.max(1.0),
        "nuclear abs={a:e} sig={sig:e}"
    );
}

/// Dipole `[Dx, Dy, Dz]` across l = 0..=6 against an independent MD reference,
/// off-origin. Previously the only analytic dipole check was a single s case.
#[test]
fn high_l_dipole_matches_md() {
    let shells = high_l_shells();
    let basis = high_l_basis();
    let o = [0.15, -0.25, 0.35];
    let got = basis.dipole(o);
    let reference = md_dipole_matrix(&shells, o);
    for (k, axis) in ["x", "y", "z"].iter().enumerate() {
        let (a, sig, peak) = sig_metrics(&got[k], &reference[k]);
        println!("high-l dipole_{axis} vs MD: peak={peak:e} worst_abs={a:e} worst_sig={sig:e}");
        assert!(
            sig < 1e-10 && a < 1e-10 * peak.max(1.0),
            "dipole_{axis} abs={a:e} sig={sig:e}"
        );
    }
}
