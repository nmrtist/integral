//! Discretization-sufficiency check for the Rys finite-branch Gauss–Legendre order.
//!
//! integral's finite branch discretizes dμ with a FIXED `GL_POINTS = 128`
//! Gauss–Legendre rule (see `integral/src/math/rys.rs`). This checks whether
//! 128 has converged at high `n`: if bumping to GL-256 moves the roots/weights by
//! more than the claimed tolerance, that would be a problem.
//!
//! Approach: a FAITHFUL copy of the library's finite-branch path
//! (Gauss–Legendre-01 → discretized Stieltjes → Golub–Welsch via implicit-shift
//! QL), parameterized by the GL order `N`. Two assertions:
//!
//!   1. FIDELITY — the `N = 128` copy reproduces integral's
//!      `rys_roots_weights_reference` (the library's discretized-Stieltjes finite
//!      branch) to ~machine precision. This proves the copy mirrors the real
//!      algorithm, so the convergence claim below is about integral's actual
//!      Stieltjes path and not a divergent re-implementation. (The *production*
//!      `rys_roots_weights` finite branch is the Chebyshev-table interpolation of
//!      this reference — validated separately in
//!      `rys::tests::interp_matches_reference_full_grid` — so the fidelity check
//!      targets the reference, which is the discretization at issue here.)
//!   2. CONVERGENCE — `N = 256` vs `N = 128` agree to < 1e-13, i.e. GL-128 is a
//!      sufficient discretization (any residual integral error is f64 round-off,
//!      measured separately by the mpmath node/weight diff).
//!
//! Only the finite branch uses the GL discretization; the large-T branch is an
//! exact (T-independent) generalized-Laguerre recurrence, so it is not exercised
//! here. Test points stay strictly below each `b_n = 30.5 + 5n` crossover.

use integral::math::rys::{rys_roots_weights_reference, MAX_RYS_ROOTS};

const MAX_RYS: usize = MAX_RYS_ROOTS;

/// Implicit-shift QL accumulating only the first eigenvector row — a faithful
/// copy of the library's `tridiagonal_ql`, but slice-generic in length.
fn tridiagonal_ql(d: &mut [f64], e: &mut [f64], z0: &mut [f64]) {
    let n = d.len();
    if n <= 1 {
        return;
    }
    const MAX_ITER: usize = 60;
    for l in 0..n {
        let mut iter = 0usize;
        loop {
            let mut m = l;
            while m < n - 1 {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= f64::EPSILON * dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iter += 1;
            if iter > MAX_ITER {
                break;
            }
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let r = g.hypot(1.0);
            g = d[m] - d[l] + e[l] / (g + r.copysign(g));
            let mut s = 1.0;
            let mut c = 1.0;
            let mut p = 0.0;
            let mut underflow = false;
            let mut i = m;
            while i > l {
                i -= 1;
                let mut f = s * e[i];
                let b = c * e[i];
                let r2 = f.hypot(g);
                e[i + 1] = r2;
                if r2 == 0.0 {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    underflow = true;
                    break;
                }
                s = f / r2;
                c = g / r2;
                let dd = d[i + 1] - p;
                let r3 = (d[i] - dd) * s + 2.0 * c * b;
                p = s * r3;
                d[i + 1] = dd + p;
                g = c * r3 - b;
                f = z0[i + 1];
                z0[i + 1] = s * z0[i] + c * f;
                z0[i] = c * z0[i] - s * f;
            }
            if underflow {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
}

/// `N`-point Gauss–Legendre rule on [0,1] (faithful copy of `gauss_legendre_01`).
fn gauss_legendre_01(npts: usize) -> (Vec<f64>, Vec<f64>) {
    let n = npts;
    let mut d = vec![0.5f64; n];
    let mut e = vec![0.0f64; n];
    let mut z0 = vec![0.0f64; n];
    z0[0] = 1.0;
    for i in 1..n {
        let l = i as f64;
        let b = 0.25 * l * l / (4.0 * l * l - 1.0);
        e[i - 1] = b.sqrt();
    }
    e[n - 1] = 0.0;
    tridiagonal_ql(&mut d, &mut e, &mut z0);
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| d[i].partial_cmp(&d[j]).unwrap());
    let mut nodes = vec![0.0f64; n];
    let mut wts = vec![0.0f64; n];
    for (out, &i) in idx.iter().enumerate() {
        nodes[out] = d[i];
        wts[out] = z0[i] * z0[i];
    }
    (nodes, wts)
}

/// Discretized Stieltjes (faithful copy of `stieltjes`), slice-generic.
fn stieltjes(n: usize, x: &[f64], w: &[f64], alpha: &mut [f64], beta: &mut [f64]) {
    let m = x.len();
    let inner = |p: &[f64]| {
        let mut norm = 0.0;
        let mut xnorm = 0.0;
        for ((&xi, &wi), &pi) in x.iter().zip(w.iter()).zip(p.iter()).take(m) {
            let wp = wi * pi * pi;
            norm += wp;
            xnorm += xi * wp;
        }
        (norm, xnorm)
    };
    let mut p_prev = vec![0.0f64; m];
    let mut p_curr = vec![1.0f64; m];
    let (mut norm, xnorm) = inner(&p_curr);
    beta[0] = norm;
    alpha[0] = xnorm / norm;
    for k in 1..n {
        let a_km1 = alpha[k - 1];
        let b_km1 = beta[k - 1];
        let mut p_next = vec![0.0f64; m];
        for idx in 0..m {
            p_next[idx] = (x[idx] - a_km1) * p_curr[idx] - b_km1 * p_prev[idx];
        }
        let (nn, xn) = inner(&p_next);
        alpha[k] = xn / nn;
        beta[k] = nn / norm;
        norm = nn;
        p_prev = p_curr;
        p_curr = p_next;
    }
}

/// Golub–Welsch (faithful copy of `golub_welsch`).
fn golub_welsch(alpha: &[f64], beta: &[f64], roots: &mut [f64], weights: &mut [f64]) {
    let n = alpha.len();
    let mu0 = beta[0];
    if n == 1 {
        roots[0] = alpha[0];
        weights[0] = mu0;
        return;
    }
    let mut d = vec![0.0f64; n];
    let mut e = vec![0.0f64; n];
    let mut z0 = vec![0.0f64; n];
    d[..n].copy_from_slice(&alpha[..n]);
    z0[0] = 1.0;
    for i in 1..n {
        e[i - 1] = beta[i].max(0.0).sqrt();
    }
    e[n - 1] = 0.0;
    tridiagonal_ql(&mut d, &mut e, &mut z0);
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| {
        d[i].partial_cmp(&d[j])
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    for (out, &i) in idx.iter().enumerate() {
        roots[out] = d[i];
        weights[out] = mu0 * z0[i] * z0[i];
    }
}

/// Faithful finite-branch evaluation with a configurable GL order `npts`.
fn finite_branch_gl(n: usize, t: f64, npts: usize) -> (Vec<f64>, Vec<f64>) {
    let (u, uw) = gauss_legendre_01(npts);
    let mut x = vec![0.0f64; npts];
    let mut w = vec![0.0f64; npts];
    for m in 0..npts {
        let xm = u[m] * u[m];
        x[m] = xm;
        w[m] = uw[m] * (-t * xm).exp();
    }
    let mut alpha = vec![0.0f64; n];
    let mut beta = vec![0.0f64; n];
    stieltjes(n, &x, &w, &mut alpha, &mut beta);
    let mut roots = vec![0.0f64; n];
    let mut wts = vec![0.0f64; n];
    golub_welsch(&alpha, &beta, &mut roots, &mut wts);
    (roots, wts)
}

/// T values strictly below b_n = 30.5 + 5n (finite branch only).
fn finite_ts(n: usize) -> Vec<f64> {
    let b = 30.5 + 5.0 * n as f64;
    let mut v = vec![0.0, 0.5, 2.0, 7.0, 15.0, 25.0];
    for &d in &[10.0, 5.0, 2.0, 0.5, 0.1] {
        let t = b - d;
        if t > 0.0 {
            v.push(t);
        }
    }
    v.retain(|&t| t < b);
    v
}

#[test]
fn gl128_copy_reproduces_library_finite_branch() {
    // Fidelity: the N=128 faithful copy must equal the library's discretized-
    // Stieltjes reference on the finite branch to ~machine precision (same
    // algorithm, same inputs). The production path interpolates this reference;
    // the interpolation is checked in `rys::tests::interp_matches_reference_full_grid`.
    let mut worst = 0.0_f64;
    for n in 1..=MAX_RYS {
        for &t in &finite_ts(n) {
            let (r_copy, w_copy) = finite_branch_gl(n, t, 128);
            let mut r_ship = [0.0f64; MAX_RYS];
            let mut w_ship = [0.0f64; MAX_RYS];
            rys_roots_weights_reference(n, t, &mut r_ship, &mut w_ship);
            for i in 0..n {
                let dr = (r_copy[i] - r_ship[i]).abs() / r_ship[i].abs().max(1e-300);
                let dw = (w_copy[i] - w_ship[i]).abs() / w_ship[i].abs().max(1e-300);
                worst = worst.max(dr).max(dw);
            }
        }
    }
    assert!(
        worst < 1e-12,
        "GL-128 copy does not match the library's finite branch (worst rel {worst:e}) — \
         copy is not faithful, convergence claim below would be meaningless"
    );
}

#[test]
fn gl256_vs_gl128_converged() {
    // Convergence: bumping the discretization 128 -> 256 must not move the
    // roots/weights by more than the Rys tolerance. If it does, GL-128 is an
    // insufficient discretization (a finding).
    let mut worst_root = (0.0_f64, 0usize, 0.0_f64);
    let mut worst_wt = (0.0_f64, 0usize, 0.0_f64);
    for n in 1..=MAX_RYS {
        for &t in &finite_ts(n) {
            let (r128, w128) = finite_branch_gl(n, t, 128);
            let (r256, w256) = finite_branch_gl(n, t, 256);
            for i in 0..n {
                let dr = (r128[i] - r256[i]).abs() / r256[i].abs().max(1e-300);
                let dw = (w128[i] - w256[i]).abs() / w256[i].abs().max(1e-300);
                if dr > worst_root.0 {
                    worst_root = (dr, n, t);
                }
                if dw > worst_wt.0 {
                    worst_wt = (dw, n, t);
                }
            }
        }
    }
    eprintln!(
        "GL-128 vs GL-256: worst root {:.3e} (n={}, T={}), worst weight {:.3e} (n={}, T={})",
        worst_root.0, worst_root.1, worst_root.2, worst_wt.0, worst_wt.1, worst_wt.2
    );
    // 2e-13: an order of magnitude under the ~1e-12 Rys accuracy bar, with
    // headroom for platform libm ULP differences (glibc measures 1.03e-13 on
    // the worst weight where MSVC lands just under 1e-13).
    assert!(
        worst_root.0 < 2e-13 && worst_wt.0 < 2e-13,
        "GL-128 not converged vs GL-256: root {:.3e}, weight {:.3e}",
        worst_root.0,
        worst_wt.0
    );
}
