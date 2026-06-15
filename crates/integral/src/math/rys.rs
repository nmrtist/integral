//! Rys-quadrature roots and weights.
//!
//! The Rys quadrature is the engine of the two-electron repulsion integral
//! and provides nodes
//! `{x_i}` and weights `{w_i}` such that, for the parameter `T = ρ·R²_PQ ≥ 0`,
//!
//! ```text
//!   ∫_0^1 exp(-T u²) p(u²) du  =  Σ_i w_i · p(x_i)
//! ```
//!
//! holds **exactly** for every polynomial `p` of degree `< 2n`. Substituting
//! `x = u²` turns the left side into a Gauss quadrature for the measure
//!
//! ```text
//!   dμ(x) = ½ · x^{-1/2} · exp(-T x) dx   on   x ∈ [0, 1],
//! ```
//!
//! whose **power moments are exactly the Boys functions**:
//!
//! ```text
//!   μ_k = ∫_0^1 x^k dμ(x) = F_k(T).
//! ```
//!
//! So the defining, testable property of the output is
//!
//! ```text
//!   Σ_i w_i · x_i^k = F_k(T),   k = 0, 1, …, 2n-1.                        (★)
//! ```
//!
//! `x_i ∈ (0, 1)` is the **squared** Rys root (`= t_i²`); the ERI 2D recurrences
//! consume `x_i` directly. The weights satisfy `Σ_i w_i = F_0(T)`.
//!
//! ## Two paths: tabulated production vs Stieltjes reference
//!
//! [`rys_roots_weights`] is the **production** path: a fast piecewise
//! **Chebyshev interpolation in `T`** of the finite-branch roots/weights from
//! precomputed `rys_tables`, plus the cached large-`T` Laguerre limit. The
//! discretized-Stieltjes solve described below is **kept** as
//! [`rys_roots_weights_reference`] — the test reference, the table's source
//! ([`rys_finite_reference`]), and the documented higher-precision/DD fallback.
//! The interpolation reproduces the reference to `≤ 1.85e-14` (abs) and external
//! 60-digit mpmath to `1.66e-14` (node) / `1.83e-14` (weight, abs); the downstream
//! ERIs still agree with the independent cross-checks at `1e-11`.
//! The remainder of this module documents the
//! **reference** method.
//!
//! ## Method (the hardest numerical part)
//!
//! The nodes/weights are the Gauss quadrature for `dμ`, obtained by the
//! **Golub–Welsch** algorithm: build the three-term recurrence coefficients
//! `(α_k, β_k)` of the polynomials orthogonal w.r.t. `dμ`, then read the nodes
//! as eigenvalues of the symmetric tridiagonal Jacobi matrix and the weights
//! from the first components of its eigenvectors.
//!
//! The `(α_k, β_k)` are computed by the **discretized Stieltjes procedure**
//! rather than from the power moments `F_k` directly (which make the classical
//! Chebyshev map catastrophically ill-conditioned) or from analytic modified
//! moments (whose monomial form reintroduces the same cancellation). The key is
//! the substitution `x = u²`, which turns `dμ` into the **smooth** measure
//! `e^{-Tu²} du` on `[0, 1]` with *no* endpoint singularity. We discretize it
//! with a cached high-order Gauss–Legendre rule `{u_m, ω_m}` — giving the
//! discrete nodes `X_m = u_m²` and weights `W_m = ω_m e^{-T u_m²}` — and run
//! Stieltjes on it. The monic orthogonal polynomials stay bounded by 1 on
//! `[0, 1]`, so the procedure is numerically stable and pushes the usable f64
//! range far higher than a moment-based map. This `(α, β)` step is written
//! generic over a [`Scalar`] so it can be re-run in higher precision if a
//! `(T, n)` regime is ever found to degrade — the downstream eigensolve stays in
//! f64 (see the module-level accuracy notes below).
//!
//! For **large `T`** the measure collapses toward `x = 0` and the finite-interval
//! map conditions poorly, but there the tail beyond `x = 1` is negligible and
//! `dμ` is, to ~`exp(-T)`, the generalized-Laguerre measure `x^{-1/2}e^{-Tx}` on
//! `[0, ∞)`; we then use the (exact-recurrence, well-conditioned) generalized
//! Gauss–Laguerre(−½) nodes scaled by `1/T`. The crossover is placed where the
//! Boys asymptotic form is itself accurate for the highest required order, so
//! the asymptotic moments match the true `F_k` to ~machine precision.
//!
//! ## Validated f64 accuracy
//!
//! Cross-checked against a 60-digit mpmath Gauss quadrature over 819 `(n, T)`
//! cases (`n = 1..13`, `T = 0..1000`). Worst **node**
//! relative error `4.9e-14`; worst **weight** relative error `1.35e-12`. f64
//! holds below `1e-12` everywhere except a few weights in the high-`n` corner
//! just *below* each crossover (`n=9,T≈76`; `n=11,T≈86`; `n=13,T≈96`), where the
//! extremely concentrated measure costs ~3–4 digits and weights reach
//! `~1.3e-12`. That is still inside the `1e-11` ERI tolerance, so the
//! precision-generic `(α, β)` step runs in plain **f64** — no
//! double-double fallback is required at `MAX_L = 6`.

use std::sync::OnceLock;

use crate::math::boys::asymptotic_boundary;

/// Precomputed Chebyshev-in-`T` tables for the finite-branch roots/weights.
/// **Generated** by `cargo xtask gen-rys-tables` from
/// [`rys_finite_reference`]; do not edit by hand. See [`interp_finite`].
#[path = "rys_tables.rs"]
mod rys_tables;

/// Maximum number of Rys roots the engine supports.
///
/// `nroots = ⌊L_total/2⌋ + 1` with `L_total = la+lb+lc+ld`. At the engine cap
/// `L_total = 4·MAX_L = 24` this is `⌊24/2⌋ + 1 = 13`.
pub const MAX_RYS_ROOTS: usize = 13;

/// Number of Gauss–Legendre points used to discretize `dμ` in the finite-T
/// branch. Generated once and cached; chosen large enough that the rule resolves
/// `e^{-Tu²}` (width `~1/√T`) and the degree-`≤4·MAX_L` polynomials for every `T`
/// below the large-T crossover (`T ≲ 96`). Endpoint-clustered GL nodes give ample
/// resolution near `u = 0` where the large-T measure concentrates.
const GL_POINTS: usize = 128;

/// Minimal real-scalar abstraction for the precision-sensitive `(α, β)` step.
///
/// Only the four field operations are needed (Wheeler's algorithm uses no roots
/// or transcendental functions); `sqrt` and the eigensolve live downstream in
/// f64. Implementing this trait for a double-double type is the documented
/// fallback path if the mpmath cross-check ever locates an f64 break-down.
pub trait Scalar:
    Copy
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
{
    /// Inject an f64 constant.
    fn from_f64(x: f64) -> Self;
    /// Project back to f64 (used once the `(α, β)` are formed).
    fn to_f64(self) -> f64;
}

impl Scalar for f64 {
    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
}

/// Evaluate the `n`-point Rys quadrature for parameter `t = T ≥ 0`
/// (**production path**).
///
/// Writes the squared roots `x_i ∈ (0, 1)` into `roots[0..n]` and the weights
/// `w_i` into `weights[0..n]`, satisfying property (★) above. Nodes are returned
/// in ascending order.
///
/// ## Method
///
/// This is the fast tabulated path that replaces the per-quartet discretized
/// Stieltjes solve for production use:
///
/// - **Finite branch** (`T < `[`asymptotic_boundary`]`(2n-1)`): a piecewise
///   **Chebyshev interpolation in `T`** of the roots and weights, read from the
///   precomputed `rys_tables` (Clenshaw evaluation). The tables are generated
///   from [`rys_finite_reference`] (the Stieltjes path) by `cargo xtask
///   gen-rys-tables`; a unit test re-checks them against the live reference so
///   they cannot drift.
/// - **Large-`T` branch** (`T ≥ asymptotic_boundary(2n-1)`): the exact
///   generalized-Laguerre(−½) limit, **bit-identical** to the reference — the
///   `T`-independent nodes/weights are cached once per `n` and scaled by `1/T`,
///   `1/(2√T)` (no interpolation).
///
/// The branch seam is the same `asymptotic_boundary(2n-1)` as the reference, so
/// the two paths agree to the interpolation tolerance everywhere. The validated
/// Stieltjes path remains available as [`rys_roots_weights_reference`] (the test
/// reference and the documented higher-precision fallback).
///
/// # Panics
/// Panics (in debug builds) if `n == 0`, `n > MAX_RYS_ROOTS`, the output slices
/// are too short, or `t < 0`.
pub fn rys_roots_weights(n: usize, t: f64, roots: &mut [f64], weights: &mut [f64]) {
    debug_assert!(n >= 1, "Rys quadrature needs at least one root");
    debug_assert!(n <= MAX_RYS_ROOTS, "n exceeds MAX_RYS_ROOTS");
    debug_assert!(roots.len() >= n && weights.len() >= n, "output too short");
    debug_assert!(t >= 0.0, "Rys parameter T must be >= 0");

    if t >= asymptotic_boundary(2 * n - 1) {
        laguerre_half_branch(n, t, roots, weights);
    } else {
        interp_finite(n, t, roots, weights);
    }
}

/// Evaluate the `n`-point Rys quadrature by the **discretized-Stieltjes
/// reference**: the validated, slower path used as the test reference for the
/// tabulated production path and as the documented higher-precision fallback.
///
/// Identical branch structure and output contract as [`rys_roots_weights`]; only
/// the finite branch differs (Stieltjes solve vs table interpolation). The
/// large-`T` branch is the shared `laguerre_half_branch`.
///
/// # Panics
/// Same as [`rys_roots_weights`].
pub fn rys_roots_weights_reference(n: usize, t: f64, roots: &mut [f64], weights: &mut [f64]) {
    debug_assert!(n >= 1, "Rys quadrature needs at least one root");
    debug_assert!(n <= MAX_RYS_ROOTS, "n exceeds MAX_RYS_ROOTS");
    debug_assert!(roots.len() >= n && weights.len() >= n, "output too short");
    debug_assert!(t >= 0.0, "Rys parameter T must be >= 0");

    if t >= asymptotic_boundary(2 * n - 1) {
        laguerre_half_branch(n, t, roots, weights);
    } else {
        finite_branch::<f64>(n, t, roots, weights);
    }
}

/// The **finite-branch** Stieltjes reference in isolation (always the
/// `[0, 1]`-interval discretized-Stieltjes solve, regardless of `T`). This is the
/// function the Chebyshev tables are fit to and validated against, so the
/// production finite branch reproduces *exactly this* quantity for
/// `T < asymptotic_boundary(2n-1)`. Used by `xtask gen-rys-tables` and the
/// table-vs-reference unit test.
///
/// # Panics
/// Same as [`rys_roots_weights`].
pub fn rys_finite_reference(n: usize, t: f64, roots: &mut [f64], weights: &mut [f64]) {
    debug_assert!(n >= 1, "Rys quadrature needs at least one root");
    debug_assert!(n <= MAX_RYS_ROOTS, "n exceeds MAX_RYS_ROOTS");
    debug_assert!(roots.len() >= n && weights.len() >= n, "output too short");
    debug_assert!(t >= 0.0, "Rys parameter T must be >= 0");
    finite_branch::<f64>(n, t, roots, weights);
}

/// Finite-branch production path: piecewise Chebyshev interpolation in `T` of the
/// roots/weights from `rys_tables`. Locates the equal-width `T`-subinterval for
/// `n`, maps `T` to `ξ ∈ [-1, 1]`, and Clenshaw-evaluates each root/weight series.
fn interp_finite(n: usize, t: f64, roots: &mut [f64], weights: &mut [f64]) {
    let t_hi = asymptotic_boundary(2 * n - 1);
    let root_coeffs = rys_tables::RYS_ROOT_COEFFS[n - 1];
    let weight_coeffs = rys_tables::RYS_WEIGHT_COEFFS[n - 1];
    let n_sub = root_coeffs.len() / n;
    let width = t_hi / n_sub as f64;

    // Subinterval index (clamp the seam into the last cell).
    let mut s = (t / width) as usize;
    if s >= n_sub {
        s = n_sub - 1;
    }
    let a = s as f64 * width;
    let b = a + width;
    // Map T ∈ [a, b] → ξ ∈ [-1, 1].
    let xi = (2.0 * t - (a + b)) / (b - a);

    let base = s * n;
    for i in 0..n {
        roots[i] = chebev(&root_coeffs[base + i], xi);
        weights[i] = chebev(&weight_coeffs[base + i], xi);
    }
}

/// Clenshaw evaluation of a Chebyshev series `Σ_k c_k T_k(ξ)` on `ξ ∈ [-1, 1]`,
/// using the standard half-`c_0` convention (Numerical Recipes `chebev`): the
/// `c_k` are the coefficients produced by `rys_tables` (and the generator),
/// where `c_0` is the full DCT term and contributes `½ c_0` to the sum.
#[inline]
fn chebev(c: &[f64; rys_tables::RYS_CHEB_DEGREE + 1], xi: f64) -> f64 {
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

/// Finite-interval Gauss quadrature for `dμ` on `[0, 1]` via discretized
/// Stieltjes + Golub–Welsch. With `x = u²`, `dμ` becomes the smooth measure
/// `e^{-Tu²} du`; we discretize it with the cached Gauss–Legendre rule and run
/// Stieltjes in `S` precision.
fn finite_branch<S: Scalar>(n: usize, t: f64, roots: &mut [f64], weights: &mut [f64]) {
    let (u_nodes, u_wts) = gauss_legendre_01();

    // Discrete measure for dμ: X_m = u_m², W_m = ω_m · e^{-T u_m²}.
    let mut x = [0.0f64; GL_POINTS];
    let mut w = [0.0f64; GL_POINTS];
    for m in 0..GL_POINTS {
        let u = u_nodes[m];
        let xm = u * u;
        x[m] = xm;
        w[m] = u_wts[m] * (-t * xm).exp();
    }

    let mut alpha = [0.0f64; MAX_RYS_ROOTS];
    let mut beta = [0.0f64; MAX_RYS_ROOTS];
    stieltjes::<S>(n, &x, &w, &mut alpha, &mut beta);

    golub_welsch(&alpha[..n], &beta[..n], roots, weights);
}

/// Discretized Stieltjes procedure (Gautschi). Given a discrete inner product
/// `⟨f,g⟩ = Σ_m w_m f(x_m) g(x_m)`, compute the recurrence coefficients
/// `(α_k, β_k)` of the monic orthogonal polynomials, `k = 0..n-1`, with
/// `β_0 = Σ_m w_m`. The monic polynomials are evaluated at the nodes via their
/// own three-term recurrence (bounded by 1 on `[0,1]`, hence stable). All
/// arithmetic runs in `S` precision; only the `(α, β)` are projected to f64.
fn stieltjes<S: Scalar>(n: usize, x: &[f64], w: &[f64], alpha: &mut [f64], beta: &mut [f64]) {
    let m = x.len();
    let zero = S::from_f64(0.0);

    // ⟨f,g⟩ = Σ_m w_m f(x_m) g(x_m) over the sampled polynomials `p`. Returns
    // (⟨p,p⟩, ⟨x p,p⟩).
    let moments = |p: &[S]| {
        let mut norm = zero;
        let mut xnorm = zero;
        for ((&xi, &wi), &pi) in x.iter().zip(w.iter()).zip(p.iter()).take(m) {
            let wp = S::from_f64(wi) * pi * pi;
            norm = norm + wp;
            xnorm = xnorm + S::from_f64(xi) * wp;
        }
        (norm, xnorm)
    };

    // π_{k-1} and π_k sampled at the nodes; start k=0: π_0 = 1, π_{-1} = 0.
    let mut p_prev = [zero; GL_POINTS];
    let mut p_curr = [zero; GL_POINTS];
    for v in p_curr.iter_mut().take(m) {
        *v = S::from_f64(1.0);
    }

    let (mut norm, xnorm) = moments(&p_curr); // ⟨π_0,π_0⟩, ⟨x π_0,π_0⟩
    beta[0] = norm.to_f64();
    alpha[0] = (xnorm / norm).to_f64();

    for k in 1..n {
        // π_k = (x − α_{k-1}) π_{k-1} − β_{k-1} π_{k-2}.
        let a_km1 = S::from_f64(alpha[k - 1]);
        let b_km1 = S::from_f64(beta[k - 1]);
        let mut p_next = [zero; GL_POINTS];
        for (((pn, &xi), &pc), &pp) in p_next
            .iter_mut()
            .zip(x.iter())
            .zip(p_curr.iter())
            .zip(p_prev.iter())
            .take(m)
        {
            *pn = (S::from_f64(xi) - a_km1) * pc - b_km1 * pp;
        }

        let (nn, xn) = moments(&p_next); // ⟨π_k,π_k⟩, ⟨x π_k,π_k⟩
        alpha[k] = (xn / nn).to_f64();
        beta[k] = (nn / norm).to_f64(); // ⟨π_k,π_k⟩ / ⟨π_{k-1},π_{k-1}⟩

        norm = nn;
        p_prev = p_curr;
        p_curr = p_next;
    }
}

/// Cached `GL_POINTS`-point Gauss–Legendre rule on `[0, 1]`: nodes `u_m` and
/// weights `ω_m` for `∫_0^1 f(u) du ≈ Σ_m ω_m f(u_m)`. Built once via
/// Golub–Welsch on the (smooth, well-conditioned) monic shifted-Legendre
/// recurrence and reused by every finite-T quadrature.
fn gauss_legendre_01() -> &'static ([f64; GL_POINTS], [f64; GL_POINTS]) {
    static RULE: OnceLock<([f64; GL_POINTS], [f64; GL_POINTS])> = OnceLock::new();
    RULE.get_or_init(|| {
        let n = GL_POINTS;
        // Monic shifted-Legendre on [0,1]: a_l = 1/2, b_l = ¼ l²/(4l²−1), b_0 = 1.
        let mut d = [0.5f64; GL_POINTS];
        let mut e = [0.0f64; GL_POINTS];
        let mut z0 = [0.0f64; GL_POINTS];
        z0[0] = 1.0;
        for i in 1..n {
            let l = i as f64;
            let b = 0.25 * l * l / (4.0 * l * l - 1.0);
            e[i - 1] = b.sqrt();
        }
        e[n - 1] = 0.0;
        tridiagonal_ql(&mut d, &mut e, &mut z0);

        // Sort ascending; weights = β_0 · z0² with β_0 = ∫_0^1 du = 1.
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&i, &j| d[i].partial_cmp(&d[j]).unwrap());
        let mut nodes = [0.0f64; GL_POINTS];
        let mut wts = [0.0f64; GL_POINTS];
        for (out, &i) in idx.iter().enumerate() {
            nodes[out] = d[i];
            wts[out] = z0[i] * z0[i];
        }
        (nodes, wts)
    })
}

/// Golub–Welsch: given Jacobi coefficients `(α, β)` of a measure, return the
/// Gauss nodes (eigenvalues of the symmetric tridiagonal Jacobi matrix `J`) and
/// weights `w_i = β_0 · z_{0,i}²` (`z_0` = first row of `J`'s eigenvectors).
/// Nodes are returned ascending.
fn golub_welsch(alpha: &[f64], beta: &[f64], roots: &mut [f64], weights: &mut [f64]) {
    let n = alpha.len();
    let mu0 = beta[0];

    if n == 1 {
        roots[0] = alpha[0];
        weights[0] = mu0;
        return;
    }

    // Diagonal d and off-diagonal e (e[i] couples rows i,i+1; e[n-1] sentinel 0).
    let mut d = [0.0f64; MAX_RYS_ROOTS];
    let mut e = [0.0f64; MAX_RYS_ROOTS];
    let mut z0 = [0.0f64; MAX_RYS_ROOTS];
    for i in 0..n {
        d[i] = alpha[i];
        z0[i] = 0.0;
    }
    z0[0] = 1.0;
    for i in 1..n {
        // β_i is guaranteed > 0 for a positive measure; guard against tiny
        // negatives from rounding before the sqrt.
        e[i - 1] = beta[i].max(0.0).sqrt();
    }
    e[n - 1] = 0.0;

    tridiagonal_ql(&mut d[..n], &mut e[..n], &mut z0[..n]);

    // Pair up (node, weight) and sort ascending by node.
    let mut idx = [0usize; MAX_RYS_ROOTS];
    for (i, s) in idx.iter_mut().enumerate().take(n) {
        *s = i;
    }
    idx[..n].sort_by(|&i, &j| {
        d[i].partial_cmp(&d[j])
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    for (out, &i) in idx[..n].iter().enumerate() {
        roots[out] = d[i];
        weights[out] = mu0 * z0[i] * z0[i];
    }
}

/// Implicit-shift QL for a symmetric tridiagonal matrix, accumulating only the
/// first row `z0` of the eigenvector matrix (the Golub–Welsch shortcut).
///
/// `d` holds the diagonal (overwritten with eigenvalues); `e` the off-diagonal
/// with `e[i]` coupling rows `i, i+1` and `e[n-1] = 0`; `z0` starts as the first
/// row of the identity. Standard Wilkinson-shift formulation.
fn tridiagonal_ql(d: &mut [f64], e: &mut [f64], z0: &mut [f64]) {
    let n = d.len();
    if n <= 1 {
        return;
    }
    const MAX_ITER: usize = 60;

    for l in 0..n {
        let mut iter = 0usize;
        loop {
            // Find a small off-diagonal element to split at.
            let mut m = l;
            while m < n - 1 {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= f64::EPSILON * dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break; // d[l] has converged.
            }
            iter += 1;
            debug_assert!(iter <= MAX_ITER, "tridiagonal QL failed to converge");
            if iter > MAX_ITER {
                break;
            }

            // Wilkinson shift.
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
                    // Underflow: deflate this element and restart the QL sweep
                    // for index l (the outer loop re-finds the split point).
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

                // Accumulate the rotation onto the tracked first row.
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

/// Large-`T` branch: `dμ` is, to `O(e^{-T})`, the generalized-Laguerre measure
/// `x^{-1/2} e^{-Tx}` on `[0, ∞)`. With `s = T x` it becomes
/// `(2√T)^{-1} s^{-1/2} e^{-s} ds`, whose Gauss nodes are the
/// generalized-Gauss–Laguerre(−½) nodes `ξ_j` and weights `ω_j`
/// (`Σ ω_j = Γ(½) = √π`). Hence `x_j = ξ_j / T` and `w_j = ω_j / (2√T)`, which
/// reproduce the asymptotic Boys moments `F_k(T) ≈ ½ Γ(k+½) T^{-(k+½)}` exactly.
fn laguerre_half_branch(n: usize, t: f64, roots: &mut [f64], weights: &mut [f64]) {
    let (xi, om) = laguerre_half_nodes(n);
    let inv_t = 1.0 / t;
    let scale = 0.5 / t.sqrt();
    for i in 0..n {
        roots[i] = xi[i] * inv_t;
        weights[i] = om[i] * scale;
    }
}

/// The `T`-independent generalized-Gauss–Laguerre(−½) nodes `ξ_j` and weights
/// `ω_j` for `j = 0..n-1`, computed once per `n` and cached. The large-`T` Rys
/// branch is `x_j = ξ_j/T`, `w_j = ω_j/(2√T)`; only this scaling is per-call.
///
/// Returns slices of length `n`. Caching turns the large-`T` branch into a few
/// multiplies (it previously ran a Golub–Welsch eigensolve on every quartet).
fn laguerre_half_nodes(n: usize) -> (&'static [f64], &'static [f64]) {
    #[allow(clippy::type_complexity)]
    static CACHE: OnceLock<([Vec<f64>; MAX_RYS_ROOTS], [Vec<f64>; MAX_RYS_ROOTS])> =
        OnceLock::new();
    let (nodes, wts) = CACHE.get_or_init(|| {
        let mut all_nodes: [Vec<f64>; MAX_RYS_ROOTS] = Default::default();
        let mut all_wts: [Vec<f64>; MAX_RYS_ROOTS] = Default::default();
        for m in 1..=MAX_RYS_ROOTS {
            // Exact monic generalized-Laguerre(α=−½) recurrence: a_k = 2k+α+1,
            // b_k = k(k+α). Well-conditioned and T-independent; β_0 = Γ(½) = √π.
            let alpha_param = -0.5;
            let mut a = [0.0f64; MAX_RYS_ROOTS];
            let mut b = [0.0f64; MAX_RYS_ROOTS];
            for k in 0..m {
                let kf = k as f64;
                a[k] = 2.0 * kf + alpha_param + 1.0;
                b[k] = kf * (kf + alpha_param);
            }
            b[0] = core::f64::consts::PI.sqrt();

            let mut xi = [0.0f64; MAX_RYS_ROOTS];
            let mut om = [0.0f64; MAX_RYS_ROOTS];
            golub_welsch(&a[..m], &b[..m], &mut xi[..m], &mut om[..m]);
            all_nodes[m - 1] = xi[..m].to_vec();
            all_wts[m - 1] = om[..m].to_vec();
        }
        (all_nodes, all_wts)
    });
    (&nodes[n - 1], &wts[n - 1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::boys::boys_array;

    /// Worst relative error of the defining property (★): `Σ w_i x_i^k = F_k(T)`
    /// for k = 0..2n-1, over an `n`-point quadrature at parameter `t`.
    fn moment_residual(n: usize, t: f64) -> f64 {
        let mut roots = [0.0; MAX_RYS_ROOTS];
        let mut wts = [0.0; MAX_RYS_ROOTS];
        rys_roots_weights(n, t, &mut roots, &mut wts);

        let mut fm = [0.0; 2 * MAX_RYS_ROOTS];
        boys_array(2 * n - 1, t, &mut fm[..2 * n]);

        let mut worst = 0.0_f64;
        for (k, &fk) in fm.iter().enumerate().take(2 * n) {
            let mut q = 0.0;
            for (&wi, &ri) in wts.iter().zip(roots.iter()).take(n) {
                q += wi * ri.powi(k as i32);
            }
            let rel = (q - fk).abs() / fk.abs().max(1e-300);
            worst = worst.max(rel);
        }
        worst
    }

    #[test]
    fn nodes_in_unit_interval_and_weights_positive() {
        for &t in &[0.0, 1.0, 13.0, 80.0, 500.0] {
            for n in 1..=MAX_RYS_ROOTS {
                let mut roots = [0.0; MAX_RYS_ROOTS];
                let mut wts = [0.0; MAX_RYS_ROOTS];
                rys_roots_weights(n, t, &mut roots, &mut wts);
                for i in 0..n {
                    assert!(
                        roots[i] > 0.0 && roots[i] < 1.0,
                        "node {i} out of (0,1) at n={n} T={t}: {}",
                        roots[i]
                    );
                    assert!(
                        wts[i] > 0.0,
                        "weight {i} not positive at n={n} T={t}: {}",
                        wts[i]
                    );
                    if i > 0 {
                        assert!(
                            roots[i] > roots[i - 1],
                            "nodes not ascending at n={n} T={t}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn reproduces_boys_moments_low_order() {
        // Low orders should be at machine precision across a wide T range.
        for &t in &[0.0, 1e-4, 0.5, 2.0, 7.0, 20.0] {
            for n in 1..=6 {
                let r = moment_residual(n, t);
                assert!(r < 1e-11, "moment residual {r:e} at n={n} T={t}");
            }
        }
    }

    #[test]
    fn reproduces_boys_moments_large_t_branch() {
        // Above the crossover the Laguerre(−½) branch must reproduce moments.
        for &t in &[120.0, 300.0, 1000.0] {
            for n in 1..=MAX_RYS_ROOTS {
                let r = moment_residual(n, t);
                assert!(r < 1e-10, "large-T moment residual {r:e} at n={n} T={t}");
            }
        }
    }

    #[test]
    fn one_point_quadrature_is_mean_and_mass() {
        // n=1: node = F_1/F_0, weight = F_0.
        for &t in &[0.3, 5.0] {
            let mut roots = [0.0; 1];
            let mut wts = [0.0; 1];
            rys_roots_weights(1, t, &mut roots, &mut wts);
            let mut fm = [0.0; 2];
            boys_array(1, t, &mut fm);
            assert!((roots[0] - fm[1] / fm[0]).abs() < 1e-14);
            assert!((wts[0] - fm[0]).abs() < 1e-14);
        }
    }

    // ----- interpolated production path vs Stieltjes reference -----

    /// **Table-staleness guard + B2.1.** The interpolated production
    /// [`rys_roots_weights`] must reproduce the discretized-Stieltjes
    /// [`rys_roots_weights_reference`] across the full finite-branch
    /// `(n, T)` grid. A regenerated/edited table that drifts from the live
    /// reference fails here. Reports the worst error and where (the spec asks for
    /// the branch-seam and high-`n` corner specifically).
    #[test]
    fn interp_matches_reference_full_grid() {
        let mut worst = 0.0_f64;
        let mut at = (0usize, 0.0f64);
        for n in 1..=MAX_RYS_ROOTS {
            let t_hi = asymptotic_boundary(2 * n - 1);
            // 600 points spanning [0, t_hi): dense through the small-T variation
            // and right up against the finite/large-T seam.
            let steps = 600usize;
            for k in 0..steps {
                let t = t_hi * (k as f64) / (steps as f64);
                let mut rp = [0.0; MAX_RYS_ROOTS];
                let mut wp = [0.0; MAX_RYS_ROOTS];
                let mut rr = [0.0; MAX_RYS_ROOTS];
                let mut wr = [0.0; MAX_RYS_ROOTS];
                rys_roots_weights(n, t, &mut rp, &mut wp);
                rys_roots_weights_reference(n, t, &mut rr, &mut wr);
                for i in 0..n {
                    let e = (rp[i] - rr[i]).abs().max((wp[i] - wr[i]).abs());
                    if e > worst {
                        worst = e;
                        at = (n, t);
                    }
                }
            }
        }
        eprintln!(
            "interp vs Stieltjes reference: worst abs err {worst:.3e} at n={} T={:.3}",
            at.0, at.1
        );
        // The generator fits to ≤ 2e-14; allow a small margin for the off-node
        // sampling here. Far inside the 1e-11 ERI tolerance.
        assert!(worst < 1e-13, "interp vs reference {worst:.3e} at {at:?}");
    }

    /// The large-`T` branch shares the cached Laguerre(−½) limit, so the
    /// production and reference paths are **bit-identical** there (no
    /// interpolation, only the per-call `1/T`, `1/(2√T)` scaling).
    #[test]
    fn large_t_branch_is_bit_identical_to_reference() {
        for n in 1..=MAX_RYS_ROOTS {
            let seam = asymptotic_boundary(2 * n - 1);
            for &t in &[seam, seam + 1.0, 120.0, 300.0, 1000.0] {
                let mut rp = [0.0; MAX_RYS_ROOTS];
                let mut wp = [0.0; MAX_RYS_ROOTS];
                let mut rr = [0.0; MAX_RYS_ROOTS];
                let mut wr = [0.0; MAX_RYS_ROOTS];
                rys_roots_weights(n, t, &mut rp, &mut wp);
                rys_roots_weights_reference(n, t, &mut rr, &mut wr);
                for i in 0..n {
                    assert_eq!(rp[i], rr[i], "root {i} differs at n={n} T={t}");
                    assert_eq!(wp[i], wr[i], "weight {i} differs at n={n} T={t}");
                }
            }
        }
    }

    /// **B2.4 invariants on the production path**, dense `(n, T)` grid spanning
    /// both branches: every root ∈ (0,1), ascending, every weight > 0.
    #[test]
    fn production_invariants_dense_grid() {
        for n in 1..=MAX_RYS_ROOTS {
            let t_hi = asymptotic_boundary(2 * n - 1);
            for k in 0..=500 {
                // 0 .. 1.3·t_hi: cover the finite branch and well into large-T.
                let t = 1.3 * t_hi * (k as f64) / 500.0;
                let mut roots = [0.0; MAX_RYS_ROOTS];
                let mut wts = [0.0; MAX_RYS_ROOTS];
                rys_roots_weights(n, t, &mut roots, &mut wts);
                for i in 0..n {
                    assert!(
                        roots[i] > 0.0 && roots[i] < 1.0,
                        "node {i} out of (0,1) at n={n} T={t}: {}",
                        roots[i]
                    );
                    assert!(wts[i] > 0.0, "weight {i} ≤ 0 at n={n} T={t}: {}", wts[i]);
                    if i > 0 {
                        assert!(roots[i] > roots[i - 1], "nodes not ascending n={n} T={t}");
                    }
                }
            }
        }
    }

    /// **B2 moment property on the production path.** The interpolated
    /// roots/weights still satisfy the defining identity (★)
    /// `Σ w_i x_i^k = F_k(T)` across the full grid, within the documented budget
    /// (3 orders inside the 1e-11 ERI tolerance). This is the end-to-end tie to
    /// truth that the interpolation does not degrade the quadrature.
    #[test]
    fn moment_property_holds_full_grid() {
        let mut worst = 0.0_f64;
        let mut at = (0usize, 0.0f64);
        for n in 1..=MAX_RYS_ROOTS {
            let t_hi = asymptotic_boundary(2 * n - 1);
            for k in 0..=260 {
                let t = 1.2 * t_hi * (k as f64) / 260.0;
                let r = moment_residual(n, t);
                if r > worst {
                    worst = r;
                    at = (n, t);
                }
            }
        }
        eprintln!(
            "production moment (★) worst rel residual {worst:.3e} at n={} T={:.3}",
            at.0, at.1
        );
        assert!(worst < 1e-9, "moment residual {worst:.3e} at {at:?}");
    }
}
