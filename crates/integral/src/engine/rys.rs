//! Two-electron repulsion integrals (ERIs) by Rys quadrature.
//!
//! This is the general ERI engine. It evaluates a
//! Cartesian shell-quartet block `(ab|cd)` of the Coulomb kernel `1/r₁₂` over
//! four **primitive** Gaussians and accumulates the coefficient-weighted result
//! into a caller-provided block, mirroring the one-electron [`crate::engine::os`] engine.
//!
//! ## Method
//!
//! With the Gaussian-product centres `P` (bra, exponent `p = a+b`) and `Q` (ket,
//! `q = c+d`), `ρ = pq/(p+q)`, and `T = ρ|P−Q|²`, the Coulomb integral is
//!
//! ```text
//!   (ab|cd) = 2π^{5/2}/(p q √(p+q)) · K_ab · K_cd · Σ_α w_α Iˣ_α Iʸ_α Iᶻ_α,
//! ```
//!
//! where `{x_α, w_α}` are the `nroots = ⌊L_total/2⌋+1` Rys roots/weights for `T`
//! ([`crate::math::rys`], `L_total = la+lb+lc+ld`) and `I^w_α` is the 2D integral
//! along axis `w` for the four angular-momentum powers on that axis, normalized
//! so the all-`s` case gives `I = 1`.
//!
//! Per root and axis the 2D integrals are built by the standard Rys recurrences:
//! a **vertical** recurrence (VRR) raises the combined bra power `n` and ket
//! power `m` from `(0,0)`, then two **horizontal** recurrences (HRR) split
//! `n → (i on A, j on B)` using `A−B` and `m → (k on C, l on D)` using `C−D`.
//! The recurrence coefficients reduce, in the appropriate limits, to the
//! Obara–Saika `(W−P)`/`(W−Q)`/`1/2p`/`1/2(p+q)` terms (verified analytically for
//! `(ps|ss)`, `(ss|ps)`, `(ds|ss)`, `(ps|ps)`).
//!
//! ## Block layout
//!
//! The output block for the quartet `(la, lb, lc, ld)` is **row-major over the
//! four Cartesian component indices** `(a, b, c, d)`:
//!
//! ```text
//!   out[((ia · n_b + ib) · n_c + ic) · n_d + id] += scale · (ab|cd)[ia,ib,ic,id]
//! ```
//!
//! with `n_x = n_cart(lx)` and the component order of [`crate::math::am`]
//! (fastest-varying index is `d`, slowest is `a`). The driver crate documents the
//! same convention for the contracted builder.

use crate::math::am::{cart_components, n_cart};
use crate::math::rys::{rys_roots_weights, MAX_RYS_ROOTS};

use crate::engine::os::{Prim, Vec3, MAX_L};

/// Per-axis component-power range `0..=MAX_L` ⇒ dimension `MAX_L + 1`.
const L1: usize = MAX_L + 1;
/// Combined bra/ket power range `0..=2·MAX_L` ⇒ dimension `2·MAX_L + 1`.
const LT1: usize = 2 * MAX_L + 1;

/// Per-axis 4-index 2D-integral table `I[i][j][k][l]` (`i≤la, j≤lb, k≤lc, l≤ld`).
type Axis4 = [[[[f64; L1]; L1]; L1]; L1];

/// Accumulate `scale · (ab|cd)` for the Coulomb kernel over four primitives into
/// the row-major block `out` (shape `[n_cart(la)·n_cart(lb)·n_cart(lc)·n_cart(ld)]`,
/// see the module-level "Block layout").
///
/// `scale` is the product of the four contraction/normalization coefficients,
/// supplied by the driver (the engine itself works on un-normalized monomials).
///
/// # Panics
/// Panics (in debug builds) if any `l > MAX_L` or `out` is too short.
pub fn coulomb_into(a: Prim, b: Prim, c: Prim, d: Prim, scale: f64, out: &mut [f64]) {
    eri_into_impl(a, b, c, d, None, scale, out);
}

/// Accumulate `scale · (ab|cd)` for the **long-range (erf-attenuated) Coulomb
/// kernel** `erf(ω·r₁₂)/r₁₂` over four primitives — same block layout, scaling
/// and conventions as [`coulomb_into`].
///
/// ## Method
///
/// The Coulomb path rests on the Gaussian transform
/// `1/r = (2/√π) ∫₀^∞ exp(−u²r²) du` and the Rys substitution
/// `t² = u²/(ρ + u²)`, which maps `u ∈ [0, ∞)` onto `t² ∈ [0, 1)` with a flat
/// measure, so `(ab|cd) = pref · ∫₀¹ P(t²) e^{−T t²} dt` with `P` a polynomial
/// of degree `≤ L_total` in `t²` (the recurrence coefficients are linear in
/// `t²`). The attenuated kernel only truncates the transform at `u = ω`:
/// `erf(ωr)/r = (2/√π) ∫₀^ω exp(−u²r²) du`, i.e. `t² ∈ [0, s]` with
/// `s = ω²/(ρ + ω²)`. Rescaling `t = √s·v` gives
///
/// ```text
///   (ab|cd)_ω = pref · √s · ∫₀¹ P(s·v²) e^{−(sT) v²} dv
///             = pref · √s · Σ_α w'_α P(s·x'_α),
/// ```
///
/// where `{x'_α, w'_α}` are the ordinary Rys roots/weights at the **scaled
/// argument** `s·T`. So the attenuated integral is the Coulomb expression with
/// roots `x → s·x'(sT)` and weights `w → √s·w'(sT)` — exact (the same `nroots`
/// quadrature integrates `P(s·v²)` exactly), no new recurrences. Equivalently
/// this realizes the attenuated Boys function `F_m^ω(T) = s^{m+1/2} F_m(sT)`
/// ([`crate::math::boys::boys_array_erf`]). References: P. M. W. Gill,
/// R. D. Adamson, *Chem. Phys. Lett.* **261**, 105 (1996); R. Ahlrichs,
/// *Phys. Chem. Chem. Phys.* **8**, 3072 (2006).
///
/// # Panics
/// Panics (in debug builds) if `ω` is not finite and `> 0`, any `l > MAX_L`,
/// or `out` is too short. The driver crate validates `ω` for release builds.
pub fn erf_coulomb_into(
    a: Prim,
    b: Prim,
    c: Prim,
    d: Prim,
    omega: f64,
    scale: f64,
    out: &mut [f64],
) {
    debug_assert!(
        omega.is_finite() && omega > 0.0,
        "erf attenuation requires a finite ω > 0"
    );
    eri_into_impl(a, b, c, d, Some(omega), scale, out);
}

/// Shared Rys evaluation: `attenuation = None` is the Coulomb kernel (the exact
/// instruction sequence of the original [`coulomb_into`] — bit-identical);
/// `Some(ω)` applies the root/weight transformation documented on
/// [`erf_coulomb_into`].
fn eri_into_impl(
    a: Prim,
    b: Prim,
    c: Prim,
    d: Prim,
    attenuation: Option<f64>,
    scale: f64,
    out: &mut [f64],
) {
    let (la, lb, lc, ld) = (a.l, b.l, c.l, d.l);
    debug_assert!(
        la <= MAX_L && lb <= MAX_L && lc <= MAX_L && ld <= MAX_L,
        "angular momentum exceeds MAX_L"
    );

    let (na, nb, nc, nd) = (n_cart(la), n_cart(lb), n_cart(lc), n_cart(ld));
    debug_assert!(out.len() >= na * nb * nc * nd, "ERI output block too short");

    // Bra / ket Gaussian products.
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

    // (ss|ss) prefactor 2π^{5/2}/(p q √(p+q)) · K_ab · K_cd. Combined with
    // Σ_α w_α = F_0(T) this is exactly the (ss|ss) Coulomb integral.
    use std::f64::consts::PI;
    let pref = 2.0 * PI * PI * PI.sqrt() / (p * q * pq.sqrt()) * kab * kcd;

    let l_total = la + lb + lc + ld;
    let nroots = l_total / 2 + 1;
    let mut roots = [0.0f64; MAX_RYS_ROOTS];
    let mut wts = [0.0f64; MAX_RYS_ROOTS];
    match attenuation {
        None => rys_roots_weights(nroots, t, &mut roots, &mut wts),
        Some(omega) => {
            // Long-range kernel: roots/weights at the scaled argument s·T with
            // x → s·x, w → √s·w (see `erf_coulomb_into`'s docs for the
            // derivation and references). Exactness is preserved: the quadrature
            // integrand P(s·v²) is a polynomial of the same degree in v².
            let s = omega * omega / (rho + omega * omega);
            rys_roots_weights(nroots, s * t, &mut roots, &mut wts);
            let sqrt_s = s.sqrt();
            for r in roots[..nroots].iter_mut() {
                *r *= s;
            }
            for w in wts[..nroots].iter_mut() {
                *w *= sqrt_s;
            }
        }
    }

    // Per-axis geometric scalars (root-independent parts).
    let pa = sub(p_ctr, a.center); // P − A
    let qc = sub(q_ctr, c.center); // Q − C
    let ab = sub(a.center, b.center); // A − B  (bra HRR)
    let cd = sub(c.center, d.center); // C − D  (ket HRR)

    let comps_a = cart_components(la);
    let comps_b = cart_components(lb);
    let comps_c = cart_components(lc);
    let comps_d = cart_components(ld);

    let mut ix: Axis4 = zeroed();
    let mut iy: Axis4 = zeroed();
    let mut iz: Axis4 = zeroed();

    for r in 0..nroots {
        let x = roots[r];
        // Root-dependent recurrence coefficients (shared across axes).
        let b00 = x / (2.0 * pq);
        let b10 = (1.0 - x * q / pq) / (2.0 * p);
        let b01 = (1.0 - x * p / pq) / (2.0 * q);
        let cq = x * q / pq; // weight of (P−Q) in the bra shift C00
        let cp = x * p / pq; // weight of (P−Q) in the ket shift CP0

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

/// Build the per-axis 4-index 2D-integral table `out4[i][j][k][l]` for one
/// Cartesian axis, given the axis-projected recurrence coefficients.
//
// Index-based loops are intrinsic to these recurrences (neighbour access like
// `i+1`, `k-1` across several parallel tables), so the range-loop lint is off.
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
    let nbra = la + lb; // combined bra power
    let nket = lc + ld; // combined ket power

    // --- VRR: g[n][m], n = 0..=nbra, m = 0..=nket. ---
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

    // --- HRR bra: h[i][j][m] = h[i+1][j-1][m] + (A−B)·h[i][j-1][m]. ---
    // h[i][0][m] = g[i][m]; final i ≤ la, j ≤ lb (intermediate i up to nbra).
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

    // --- HRR ket: per (i,j), t[k][l] = t[k+1][l-1] + (C−D)·t[k][l-1]. ---
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

#[inline]
fn zeroed() -> Axis4 {
    [[[[0.0; L1]; L1]; L1]; L1]
}

#[inline]
fn combine(a: f64, ca: Vec3, b: f64, cb: Vec3, p: f64) -> Vec3 {
    [
        (a * ca[0] + b * cb[0]) / p,
        (a * ca[1] + b * cb[1]) / p,
        (a * ca[2] + b * cb[2]) / p,
    ]
}

#[inline]
fn sub(u: Vec3, v: Vec3) -> Vec3 {
    [u[0] - v[0], u[1] - v[1], u[2] - v[2]]
}

#[inline]
fn dist2(u: Vec3, v: Vec3) -> f64 {
    norm2(sub(u, v))
}

#[inline]
fn norm2(u: Vec3) -> f64 {
    u[0] * u[0] + u[1] * u[1] + u[2] * u[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (ss|ss) over four unit s primitives must equal the closed form
    /// `2π^{5/2}/(p q √(p+q)) · K_ab · K_cd · F_0(T)`.
    #[test]
    fn ssss_matches_closed_form() {
        let a = Prim::new(0.8, [0.0, 0.0, 0.0], 0);
        let b = Prim::new(1.3, [0.0, 0.0, 0.7], 0);
        let c = Prim::new(0.5, [0.4, 0.0, 0.0], 0);
        let d = Prim::new(1.1, [0.0, 0.6, 0.2], 0);
        let mut out = [0.0; 1];
        coulomb_into(a, b, c, d, 1.0, &mut out);

        // Independent closed form.
        let p = a.exp + b.exp;
        let q = c.exp + d.exp;
        let pc = combine(a.exp, a.center, b.exp, b.center, p);
        let qc = combine(c.exp, c.center, d.exp, d.center, q);
        let kab = (-(a.exp * b.exp / p) * dist2(a.center, b.center)).exp();
        let kcd = (-(c.exp * d.exp / q) * dist2(c.center, d.center)).exp();
        let rho = p * q / (p + q);
        let t = rho * dist2(pc, qc);
        let mut fm = [0.0; 1];
        crate::math::boys::boys_array(0, t, &mut fm);
        use std::f64::consts::PI;
        let expect = 2.0 * PI * PI * PI.sqrt() / (p * q * (p + q).sqrt()) * kab * kcd * fm[0];
        assert!(
            (out[0] - expect).abs() < 1e-14 * expect.abs(),
            "ssss {} vs {}",
            out[0],
            expect
        );
    }

    /// The `scale` argument multiplies the whole block and the engine accumulates.
    #[test]
    fn scale_and_accumulate() {
        let a = Prim::new(0.8, [0.0, 0.0, 0.0], 0);
        let b = Prim::new(1.3, [0.0, 0.0, 0.7], 0);
        let c = Prim::new(0.5, [0.4, 0.0, 0.0], 0);
        let d = Prim::new(1.1, [0.0, 0.6, 0.2], 0);
        let mut one = [0.0; 1];
        coulomb_into(a, b, c, d, 1.0, &mut one);
        let mut acc = [0.0; 1];
        coulomb_into(a, b, c, d, 2.0, &mut acc);
        coulomb_into(a, b, c, d, 3.0, &mut acc);
        assert!((acc[0] - 5.0 * one[0]).abs() < 1e-12 * one[0].abs());
    }

    /// (ss|ss) over the erf-attenuated kernel must equal the closed form
    /// `pref · F_0^ω(T) = pref · √s · F_0(s·T)` with `s = ω²/(ρ+ω²)`.
    #[test]
    fn erf_ssss_matches_closed_form() {
        let a = Prim::new(0.8, [0.0, 0.0, 0.0], 0);
        let b = Prim::new(1.3, [0.0, 0.0, 0.7], 0);
        let c = Prim::new(0.5, [0.4, 0.0, 0.0], 0);
        let d = Prim::new(1.1, [0.0, 0.6, 0.2], 0);
        for omega in [0.2, 0.7, 1.5, 30.0] {
            let mut out = [0.0; 1];
            erf_coulomb_into(a, b, c, d, omega, 1.0, &mut out);

            let p = a.exp + b.exp;
            let q = c.exp + d.exp;
            let pc = combine(a.exp, a.center, b.exp, b.center, p);
            let qc = combine(c.exp, c.center, d.exp, d.center, q);
            let kab = (-(a.exp * b.exp / p) * dist2(a.center, b.center)).exp();
            let kcd = (-(c.exp * d.exp / q) * dist2(c.center, d.center)).exp();
            let rho = p * q / (p + q);
            let t = rho * dist2(pc, qc);
            let s = omega * omega / (rho + omega * omega);
            let mut fm = [0.0; 1];
            crate::math::boys::boys_array(0, s * t, &mut fm);
            use std::f64::consts::PI;
            let expect =
                2.0 * PI * PI * PI.sqrt() / (p * q * (p + q).sqrt()) * kab * kcd * s.sqrt() * fm[0];
            assert!(
                (out[0] - expect).abs() < 1e-13 * expect.abs(),
                "erf ssss ω={omega}: {} vs {}",
                out[0],
                expect
            );
        }
    }

    /// ω → ∞ limit: the attenuated kernel reproduces the Coulomb kernel
    /// (`s → 1`), here on a (pd|sp)-type primitive quartet.
    #[test]
    fn erf_large_omega_reproduces_coulomb_primitive() {
        let a = Prim::new(0.9, [0.1, 0.0, 0.0], 1);
        let b = Prim::new(0.7, [0.0, 0.2, 0.0], 2);
        let c = Prim::new(1.2, [0.0, 0.0, 0.3], 0);
        let d = Prim::new(0.6, [0.2, 0.1, 0.0], 1);
        let n = 3 * 6 * 3; // n_a · n_b · n_c · n_d = 3 · 6 · 1 · 3
        let mut coul = vec![0.0; n];
        coulomb_into(a, b, c, d, 1.0, &mut coul);
        let mut erf = vec![0.0; n];
        erf_coulomb_into(a, b, c, d, 1e6, 1.0, &mut erf);
        let scale_ref = coul.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for (x, y) in erf.iter().zip(coul.iter()) {
            assert!(
                (x - y).abs() <= 1e-10 * y.abs().max(1e-3 * scale_ref),
                "ω→∞ limit: {x} vs {y}"
            );
        }
    }

    /// Bra–ket exchange symmetry of a primitive block: `(ab|cd) = (cd|ab)` under
    /// the corresponding index transpose. Uses a `(ps|ss)`-type block so the
    /// p-shell makes the test non-trivial.
    #[test]
    fn bra_ket_exchange_primitive() {
        let a = Prim::new(0.9, [0.1, 0.0, 0.0], 1); // p
        let b = Prim::new(0.7, [0.0, 0.2, 0.0], 0); // s
        let c = Prim::new(1.2, [0.0, 0.0, 0.3], 0); // s
        let d = Prim::new(0.6, [0.2, 0.1, 0.0], 0); // s
        let mut abcd = [0.0; 3]; // (p s| s s): na=3, others 1
        coulomb_into(a, b, c, d, 1.0, &mut abcd);
        // (cd|ab) = (s s| p s): the p block is the last index -> nd=3.
        let mut cdab = [0.0; 3];
        coulomb_into(c, d, a, b, 1.0, &mut cdab);
        for i in 0..3 {
            assert!(
                (abcd[i] - cdab[i]).abs() < 1e-13 * abcd[i].abs().max(1e-300),
                "bra-ket exchange mismatch at {i}: {} vs {}",
                abcd[i],
                cdab[i]
            );
        }
    }
}
