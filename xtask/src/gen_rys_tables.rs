//! Generate `crates/integral/src/math/rys_tables.rs` — the precomputed
//! Chebyshev-in-`T` tables for the Rys finite-branch roots/weights.
//!
//! For each `nroots = 1..=13` the finite interval `T ∈ [0, asymptotic_boundary(
//! 2n-1)]` is split into `n_sub` **equal-width** subintervals. On each
//! subinterval every root `x_i(T)` and weight `w_i(T)` (both smooth, analytic
//! functions of `T` valued in `(0,1)`) is fit by a degree-`RYS_CHEB_DEGREE`
//! Chebyshev polynomial sampled at the Chebyshev nodes of
//! [`integral::math::rys::rys_finite_reference`] (the discretized-Stieltjes path).
//! `n_sub` is chosen **adaptively** per `n`: the smallest count whose worst
//! interpolation error over a dense in-cell grid is `≤ FIT_TOL`.
//!
//! The emitted file is committed; `rys::tests::interp_matches_reference_full_grid`
//! re-checks it against the live reference so it cannot drift. Run with
//! `cargo run -p xtask -- gen-rys-tables` (pure Rust, no external library).

use std::fmt::Write as _;

use integral::math::boys::asymptotic_boundary;
use integral::math::rys::{rys_finite_reference, MAX_RYS_ROOTS};

/// Chebyshev degree per series (`DEGREE + 1` coefficients). Fixed across all
/// `(n, subinterval)`; `n_sub` absorbs the per-`n` difficulty.
const DEGREE: usize = 16;
/// Coefficients per series.
const NC: usize = DEGREE + 1;
/// Target worst **absolute** interpolation error (roots and weights are both in
/// `(0,1)`, so this is also ~relative on the O(1) parts). Set a few ulps above
/// the f64 Clenshaw floor so the fit is essentially exact vs the reference and
/// adds negligibly to the reference's own ~5e-14 node / ~1.3e-12 weight error.
const FIT_TOL: f64 = 2e-14;
/// Largest `n_sub` the search will try before giving up (a safety stop; real
/// tables need far fewer).
const MAX_SUB: usize = 256;
/// In-cell validation samples (uniform, endpoints included) used to accept a
/// chosen `n_sub`.
const CHECK_PER_CELL: usize = 33;

pub(crate) fn run() {
    let mut body = String::new();
    let mut summary: Vec<(usize, usize, f64)> = Vec::new();
    let mut root_table_refs = String::new();
    let mut weight_table_refs = String::new();
    let mut total_coeffs = 0usize;

    for n in 1..=MAX_RYS_ROOTS {
        let t_hi = asymptotic_boundary(2 * n - 1);
        let (n_sub, worst) = choose_n_sub(n, t_hi);
        summary.push((n, n_sub, worst));

        // Build the flattened coefficient tables [sub*n + i].
        let mut root_coeffs: Vec<[f64; NC]> = Vec::with_capacity(n_sub * n);
        let mut weight_coeffs: Vec<[f64; NC]> = Vec::with_capacity(n_sub * n);
        let width = t_hi / n_sub as f64;
        for s in 0..n_sub {
            let a = s as f64 * width;
            let b = a + width;
            let (rc, wc) = fit_cell(n, a, b);
            root_coeffs.extend(rc);
            weight_coeffs.extend(wc);
        }
        total_coeffs += 2 * root_coeffs.len() * NC;

        emit_static(&mut body, &format!("ROOT_N{n}"), &root_coeffs);
        emit_static(&mut body, &format!("WEIGHT_N{n}"), &weight_coeffs);
        let sep = if n == 1 { "" } else { ",\n" };
        write!(root_table_refs, "{sep}    &ROOT_N{n}").unwrap();
        write!(weight_table_refs, "{sep}    &WEIGHT_N{n}").unwrap();
    }

    // Assemble the file.
    let mut out = String::new();
    out.push_str(&header(&summary, total_coeffs));
    writeln!(
        out,
        "/// Chebyshev interpolation degree per root/weight series (`degree + 1` coeffs)."
    )
    .unwrap();
    writeln!(out, "pub(crate) const RYS_CHEB_DEGREE: usize = {DEGREE};\n").unwrap();
    writeln!(out, "const NC: usize = RYS_CHEB_DEGREE + 1;\n").unwrap();
    out.push_str(&body);
    writeln!(
        out,
        "/// Per-`nroots` (index `n-1`) finite-branch **root** Chebyshev coefficients,\n\
         /// flattened `[sub * n + i]` over `n_sub` equal-width `T`-subintervals and root\n\
         /// index `i`. `n_sub = slice.len() / n`.\n\
         #[rustfmt::skip]\n\
         pub(crate) static RYS_ROOT_COEFFS: [&[[f64; NC]]; 13] = [\n{root_table_refs},\n];\n"
    )
    .unwrap();
    writeln!(
        out,
        "/// Per-`nroots` finite-branch **weight** Chebyshev coefficients, same layout as\n\
         /// [`RYS_ROOT_COEFFS`].\n\
         #[rustfmt::skip]\n\
         pub(crate) static RYS_WEIGHT_COEFFS: [&[[f64; NC]]; 13] = [\n{weight_table_refs},\n];"
    )
    .unwrap();

    let path = format!(
        "{}/../crates/integral/src/math/rys_tables.rs",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::write(&path, out).expect("write rys_tables.rs");

    eprintln!("wrote {path}");
    eprintln!("  degree = {DEGREE}, fit_tol = {FIT_TOL:e}");
    let bytes = total_coeffs * 8;
    eprintln!(
        "  total coeffs = {total_coeffs} f64 ({:.1} KB)",
        bytes as f64 / 1024.0
    );
    eprintln!("  n  n_sub  worst_abs_err");
    for (n, n_sub, worst) in &summary {
        eprintln!("  {n:2}  {n_sub:4}   {worst:.2e}");
    }
}

/// Pick the smallest `n_sub` whose worst in-cell interpolation error is `≤
/// FIT_TOL`, returning `(n_sub, worst_error)`.
fn choose_n_sub(n: usize, t_hi: f64) -> (usize, f64) {
    let mut n_sub = 1usize;
    loop {
        let worst = max_error(n, t_hi, n_sub);
        if worst <= FIT_TOL || n_sub >= MAX_SUB {
            return (n_sub, worst);
        }
        n_sub += 1;
    }
}

/// Worst absolute root/weight interpolation error over all subintervals for a
/// candidate `n_sub`, sampled on a dense in-cell grid.
fn max_error(n: usize, t_hi: f64, n_sub: usize) -> f64 {
    let width = t_hi / n_sub as f64;
    let mut worst = 0.0f64;
    let mut ref_r = [0.0f64; MAX_RYS_ROOTS];
    let mut ref_w = [0.0f64; MAX_RYS_ROOTS];
    for s in 0..n_sub {
        let a = s as f64 * width;
        let b = a + width;
        let (rc, wc) = fit_cell(n, a, b);
        for j in 0..CHECK_PER_CELL {
            let frac = j as f64 / (CHECK_PER_CELL - 1) as f64;
            let t = a + frac * (b - a);
            rys_finite_reference(n, t, &mut ref_r, &mut ref_w);
            let xi = (2.0 * t - (a + b)) / (b - a);
            for i in 0..n {
                worst = worst.max((chebev(&rc[i], xi) - ref_r[i]).abs());
                worst = worst.max((chebev(&wc[i], xi) - ref_w[i]).abs());
            }
        }
    }
    worst
}

/// Fit degree-`DEGREE` Chebyshev coefficients to each of the `n` roots and `n`
/// weights on `[a, b]`, sampling [`rys_finite_reference`] at the Chebyshev nodes
/// (NR `chebft`). Returns `(root_series, weight_series)`, each length `n`.
fn fit_cell(n: usize, a: f64, b: f64) -> (Vec<[f64; NC]>, Vec<[f64; NC]>) {
    // Sample the reference at the NC Chebyshev nodes of [a, b].
    let bma = 0.5 * (b - a);
    let bpa = 0.5 * (b + a);
    let mut node_roots = [[0.0f64; MAX_RYS_ROOTS]; NC];
    let mut node_wts = [[0.0f64; MAX_RYS_ROOTS]; NC];
    let mut r = [0.0f64; MAX_RYS_ROOTS];
    let mut w = [0.0f64; MAX_RYS_ROOTS];
    for (j, (nr, nw)) in node_roots.iter_mut().zip(node_wts.iter_mut()).enumerate() {
        let y = ((std::f64::consts::PI * (j as f64 + 0.5)) / NC as f64).cos();
        let t = y * bma + bpa;
        rys_finite_reference(n, t, &mut r, &mut w);
        nr[..n].copy_from_slice(&r[..n]);
        nw[..n].copy_from_slice(&w[..n]);
    }

    // c_k = (2/NC) Σ_j f_j cos(π k (j+0.5)/NC) for each root/weight function.
    let mut root_series = vec![[0.0f64; NC]; n];
    let mut weight_series = vec![[0.0f64; NC]; n];
    let fac = 2.0 / NC as f64;
    for k in 0..NC {
        for j in 0..NC {
            let cos_kj = ((std::f64::consts::PI * k as f64 * (j as f64 + 0.5)) / NC as f64).cos();
            for i in 0..n {
                root_series[i][k] += fac * node_roots[j][i] * cos_kj;
                weight_series[i][k] += fac * node_wts[j][i] * cos_kj;
            }
        }
    }
    (root_series, weight_series)
}

/// Clenshaw evaluation matching `integral::math::rys::chebev` exactly (half-`c_0`).
fn chebev(c: &[f64; NC], xi: f64) -> f64 {
    let two_xi = 2.0 * xi;
    let mut d = 0.0f64;
    let mut dd = 0.0f64;
    for &ck in c[1..].iter().rev() {
        let sv = d;
        d = two_xi * d - dd + ck;
        dd = sv;
    }
    xi * d - dd + 0.5 * c[0]
}

/// Emit one `static NAME: [[f64; NC]; LEN] = [ ... ];`.
fn emit_static(out: &mut String, name: &str, coeffs: &[[f64; NC]]) {
    out.push_str("#[rustfmt::skip]\n");
    writeln!(out, "static {name}: [[f64; NC]; {}] = [", coeffs.len()).unwrap();
    for series in coeffs {
        out.push_str("    [");
        for (k, &c) in series.iter().enumerate() {
            if k > 0 {
                out.push_str(", ");
            }
            write!(out, "{c:.17e}").unwrap();
        }
        out.push_str("],\n");
    }
    out.push_str("];\n\n");
}

fn header(summary: &[(usize, usize, f64)], total_coeffs: usize) -> String {
    let mut h = String::new();
    writeln!(
        h,
        "//! Precomputed Chebyshev-in-`T` tables for the Rys finite-branch\n\
         //! roots/weights.\n\
         //!\n\
         //! **GENERATED by `cargo run -p xtask -- gen-rys-tables`. DO NOT EDIT BY HAND.**\n\
         //!\n\
         //! Each `nroots = 1..=13` covers `T ∈ [0, asymptotic_boundary(2n-1)]` with\n\
         //! `n_sub` equal-width subintervals; on each, every root/weight is a degree-{DEGREE}\n\
         //! Chebyshev fit to the discretized-Stieltjes reference. `n_sub` is chosen so the\n\
         //! worst in-cell interpolation error is ≤ {FIT_TOL:e} (abs). Evaluation: Clenshaw\n\
         //! (`integral::math::rys::chebev`). Total = {total_coeffs} f64 ({:.1} KB).\n\
         //!\n\
         //! Per-n table (n, n_sub, worst_abs_err vs reference):",
        total_coeffs * 8 / 1024
    )
    .unwrap();
    for (n, n_sub, worst) in summary {
        writeln!(h, "//!   n={n:2}  n_sub={n_sub:3}  err={worst:.2e}").unwrap();
    }
    h.push_str("\n#![allow(clippy::all)]\n#![allow(clippy::unreadable_literal)]\n\n");
    h
}
