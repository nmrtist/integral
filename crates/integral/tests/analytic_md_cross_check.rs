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
