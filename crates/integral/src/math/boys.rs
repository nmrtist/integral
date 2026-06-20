//! The Boys function
//!
//! ```text
//!   F_m(T) = ∫_0^1 t^{2m} exp(-T t^2) dt,   T >= 0,  m = 0, 1, 2, ...
//! ```
//!
//! It is the radial kernel of nuclear-attraction and electron-repulsion
//! integrals. We evaluate the whole ladder `F_0(T) .. F_{m_max}(T)` at once,
//! since the recurrences couple neighbouring orders.
//!
//! ## Method
//!
//! 1. Compute the top order `F_{m_max}(T)` directly:
//!    - **`T < `[`asymptotic_boundary`]`(m_max)`:** a non-alternating
//!      (all-positive) power series that converges for every `T` and avoids the
//!      catastrophic cancellation of the naive alternating Taylor series.
//!    - **otherwise:** the large-`T` closed form
//!      `F_m(T) ≈ (2m-1)!!/2^{m+1} · sqrt(π) · T^{-(m+1/2)}`, i.e. extending the
//!      integral to `∞`.
//! 2. Fill the lower orders by **downward** recurrence
//!    `F_{m-1}(T) = (2T·F_m(T) + e^{-T}) / (2m-1)`, which is the numerically
//!    stable direction (it preserves the relative accuracy of the top order).
//!
//! ### Why the asymptotic cutoff depends on `m`
//!
//! Extending the integral to `∞` neglects `∫_1^∞ t^{2m} e^{-T t^2} dt`. In
//! *absolute* terms this tail is `O(e^{-T})`, but `F_m(T)` itself shrinks rapidly
//! with `m` (e.g. `F_8(35) ≈ 5e-10`), so the *relative* error of the asymptotic
//! form grows with `m`. The power series, by contrast, stays exact and retains
//! full `f64` precision well past `T = 50` for these orders, so we only switch to
//! the asymptotic form once `T` is large enough for the chosen `m`.
//!
//! ### Fast path: a Taylor-interpolation table
//!
//! The non-alternating series above is correct but slow — it needs `O(T)` terms
//! (each with a division) before the tail is negligible, so it dominated the ERI
//! kernel for the common `T ∈ [5, 30]` band. [`boys_array`] now uses it only to
//! **generate** a table once; at runtime, for `T < BOYS_TABLE_TMAX`, it reads
//! the whole ladder from that table by a short Taylor expansion about the nearest
//! grid point:
//!
//! ```text
//!   F_m(T0 + d) = Σ_{k=0..K} F_{m+k}(T0) · (−d)^k / k!     (since dF_m/dT = −F_{m+1})
//! ```
//!
//! Because the remainder scales with `F_{m+K+1}`, which shrinks with `m` exactly
//! as `F_m` does, the **relative** error is bounded by `(h/2)^{K+1}/(K+1)!`
//! *uniformly in m* — ~1e-15 at `h = 0.1`, `K = 7`. The table is built (lazily,
//! once) from `boys_reference` (the series/asymptotic path), so the values it
//! interpolates are the same validated ones; large `T` still uses the asymptotic
//! form directly. Validated against `boys_reference` across a dense `(m, T)` grid
//! and by all the property tests below.

/// Smallest `T` at which the large-`T` asymptotic form is used for order `m`.
///
/// Below this, the power series is used. The linear-in-`m` boundary keeps the
/// asymptotic relative error near machine precision across the full order range
/// the engines need: validated against arbitrary-precision mpmath to ~1.5e-15
/// for `m` up to **24 = 4·MAX_L** — the highest Boys order an ERI shell quartet
/// requires (`m_max = la+lb+lc+ld`) — while staying within the range where the
/// series itself keeps full `f64` precision.
#[must_use]
pub fn asymptotic_boundary(m: usize) -> f64 {
    33.0 + 2.5 * m as f64
}

/// Highest Boys order an ERI shell quartet needs: `m_max = la+lb+lc+ld ≤ 4·MAX_L`
/// (`MAX_L = 6`). The interpolation table carries `K` extra orders above this for
/// the Taylor expansion.
const BOYS_MAX_M: usize = 24;
/// Number of Taylor terms beyond the value itself (`K+1 = 9` total). The relative
/// interpolation error is `≈ (H/2)^{K+1}/(K+1)!` ≈ 1e-16 here — deliberately an
/// order of magnitude below the engine's intrinsic ~1e-14 per-element rounding, so
/// the table is invisible to the 8-fold-symmetry / cross-engine residuals.
const BOYS_K: usize = 8;
/// Orders stored per grid point: `0..=(BOYS_MAX_M + BOYS_K)`.
const BOYS_STRIDE: usize = BOYS_MAX_M + BOYS_K + 1;
/// Row length in the table: the `BOYS_STRIDE` Boys orders plus one trailing
/// `e^{-T0}` entry, so the downward recurrence's `e^{-T}` can be reconstructed as
/// `e^{-T0}·e^{neg_d}` (a short Taylor in the same `neg_d`) without a libm `exp`
/// call — the recurrence is on the ERI kernel's critical path.
const BOYS_ROW: usize = BOYS_STRIDE + 1;
/// Grid spacing in `T`. With `BOYS_K = 8` this leaves the top-order Taylor remainder
/// (`≈ (H/2)^{K+1}/(K+1)!`) far below the table values' own ~1e-15 rounding.
const BOYS_H: f64 = 0.1;

/// `1/k!` — the Taylor coefficients for the Horner evaluation of the top order.
const INV_FACT: [f64; BOYS_K + 1] = {
    let mut a = [1.0f64; BOYS_K + 1]; // a[0] = 1/0! = 1
    let mut k = 1;
    while k <= BOYS_K {
        a[k] = a[k - 1] / k as f64; // 1/k! = (1/(k-1)!)/k
        k += 1;
    }
    a
};

/// `1/(2m-1)` — reciprocals for the downward recurrence
/// `F_{m-1} = (2T·F_m + e^{-T}) · 1/(2m-1)`. A multiply replaces the former
/// division: each division sat on the recurrence's serial dependency chain
/// (~13-cycle latency per order). Entry `0` is unused padding.
const INV_ODD: [f64; BOYS_MAX_M + 1] = {
    let mut a = [1.0f64; BOYS_MAX_M + 1];
    let mut m = 1;
    while m <= BOYS_MAX_M {
        a[m] = 1.0 / (2 * m - 1) as f64;
        m += 1;
    }
    a
};

/// Above this `T` the table is not used (the large-`T` asymptotic form is accurate
/// for every order up to [`BOYS_MAX_M`], see [`asymptotic_boundary`]). Also the
/// upper end of the tabulated grid.
const BOYS_TABLE_TMAX: f64 = 33.0 + 2.5 * BOYS_MAX_M as f64; // = asymptotic_boundary(24) = 93

/// Number of grid intervals; grid points are `i·BOYS_H` for `i = 0..=BOYS_NGRID`.
const BOYS_NGRID: usize = (BOYS_TABLE_TMAX / BOYS_H) as usize; // 930

/// Lazily-built interpolation table, row-major `[grid_point · BOYS_ROW + order]`;
/// the last entry of each row is `e^{-T0}` for that grid point.
static BOYS_TABLE: std::sync::OnceLock<Vec<f64>> = std::sync::OnceLock::new();

/// Build the table: `F_0(T0)..F_{STRIDE-1}(T0)` plus `e^{-T0}` at every grid
/// point `T0 = i·H`, from [`boys_reference`] (the validated series/asymptotic
/// path) and libm `exp` respectively.
fn build_boys_table() -> Vec<f64> {
    let mut table = vec![0.0; (BOYS_NGRID + 1) * BOYS_ROW];
    for i in 0..=BOYS_NGRID {
        let t0 = i as f64 * BOYS_H;
        let row = &mut table[i * BOYS_ROW..(i + 1) * BOYS_ROW];
        boys_reference(BOYS_STRIDE - 1, t0, &mut row[..BOYS_STRIDE]);
        row[BOYS_STRIDE] = (-t0).exp();
    }
    table
}

/// Evaluate `F_0(T) .. F_{m_max}(T)` into `out[0..=m_max]`.
///
/// `out` must have length at least `m_max + 1`; only that prefix is written. For
/// `T < BOYS_TABLE_TMAX` the ladder is read from the Taylor-interpolation table
/// (see the module docs); otherwise it falls back to `boys_reference`.
///
/// # Panics
/// Panics (in debug builds) if `out` is too short or `t` is negative.
pub fn boys_array(m_max: usize, t: f64, out: &mut [f64]) {
    debug_assert!(out.len() > m_max, "output buffer too short for m_max");
    debug_assert!(t >= 0.0, "Boys function requires T >= 0");

    // Table covers T < TMAX and orders up to BOYS_MAX_M (+K headroom); anything
    // outside (large T, or an unexpected very high order) uses the reference path.
    if t < BOYS_TABLE_TMAX && m_max + BOYS_K < BOYS_STRIDE {
        // Interpolate ONLY the top order from the table, then descend by the *same*
        // downward recurrence the reference uses. Recurring (rather than Taylor-ing
        // every order) keeps the lower orders within ~1 ULP of the reference path.
        //
        // The top order is `Σ_k table[base+k]·(neg_d)^k/k!` evaluated by **Horner**
        // in `neg_d`: a single length-K fused multiply-add dependency chain, instead
        // of two chains (build the weights, then the dot product). This roughly halves
        // the Boys latency, which the step profile showed is ~50% of an (ss|ss) quartet.
        let table = BOYS_TABLE.get_or_init(build_boys_table);
        let i = (t * (1.0 / BOYS_H)).round() as usize; // nearest grid point
        let neg_d = i as f64 * BOYS_H - t; // = -(t - T0), |neg_d| ≤ H/2
        let row = i * BOYS_ROW;
        let base = row + m_max;
        let mut top = table[base + BOYS_K] * INV_FACT[BOYS_K];
        for k in (0..BOYS_K).rev() {
            top = top * neg_d + table[base + k] * INV_FACT[k];
        }
        out[m_max] = top;
        // The downward recurrence (and its e^{-T}) is only needed for m_max ≥ 1;
        // the very common ss/`m_max = 0` case skips it entirely.
        if m_max > 0 {
            // e^{-T} = e^{-T0}·e^{neg_d} (T = T0 − neg_d): the tabulated e^{-T0}
            // times an 8-term Horner for e^{neg_d} — truncation ≤ (H/2)^9/9! ≈
            // 5e-18 — instead of a libm `exp` call. This chain is independent of
            // the top-order chain above, so the two overlap in the pipeline.
            let mut ep = INV_FACT[BOYS_K];
            for k in (0..BOYS_K).rev() {
                ep = ep * neg_d + INV_FACT[k];
            }
            let et = table[row + BOYS_STRIDE] * ep;
            let two_t = 2.0 * t;
            let mut m = m_max;
            while m > 0 {
                out[m - 1] = (two_t * out[m] + et) * INV_ODD[m];
                m -= 1;
            }
        }
    } else {
        boys_reference(m_max, t, out);
    }
}

/// Evaluate two independent Boys ladders `F_0..F_{m_max}` at `t0` and `t1` with
/// the two table-path computations **manually interleaved**: each lane's
/// arithmetic is the exact expression sequence of [`boys_array`] (so each lane's
/// results are bitwise identical to two separate calls — fingerprint-guarded),
/// but the two Horner/recurrence dependency chains alternate in program order,
/// so they overlap in the pipeline instead of running back-to-back. This roughly
/// halves the per-ladder latency in the ERI primitive-quartet loop, where the
/// Boys row is on the critical path. Falls back to two sequential [`boys_array`]
/// calls when either lane is outside the table region.
///
/// # Panics
/// Panics (in debug builds) if either buffer is too short or either `t` is
/// negative.
pub fn boys_array2(m_max: usize, t0: f64, t1: f64, out0: &mut [f64], out1: &mut [f64]) {
    debug_assert!(out0.len() > m_max && out1.len() > m_max);
    debug_assert!(t0 >= 0.0 && t1 >= 0.0, "Boys function requires T >= 0");
    if !(t0 < BOYS_TABLE_TMAX && t1 < BOYS_TABLE_TMAX && m_max + BOYS_K < BOYS_STRIDE) {
        boys_array(m_max, t0, out0);
        boys_array(m_max, t1, out1);
        return;
    }
    let table = BOYS_TABLE.get_or_init(build_boys_table);
    let i0 = (t0 * (1.0 / BOYS_H)).round() as usize;
    let i1 = (t1 * (1.0 / BOYS_H)).round() as usize;
    let nd0 = i0 as f64 * BOYS_H - t0;
    let nd1 = i1 as f64 * BOYS_H - t1;
    let row0 = i0 * BOYS_ROW;
    let row1 = i1 * BOYS_ROW;
    let base0 = row0 + m_max;
    let base1 = row1 + m_max;
    let mut top0 = table[base0 + BOYS_K] * INV_FACT[BOYS_K];
    let mut top1 = table[base1 + BOYS_K] * INV_FACT[BOYS_K];
    for k in (0..BOYS_K).rev() {
        top0 = top0 * nd0 + table[base0 + k] * INV_FACT[k];
        top1 = top1 * nd1 + table[base1 + k] * INV_FACT[k];
    }
    out0[m_max] = top0;
    out1[m_max] = top1;
    if m_max > 0 {
        let mut ep0 = INV_FACT[BOYS_K];
        let mut ep1 = INV_FACT[BOYS_K];
        for k in (0..BOYS_K).rev() {
            ep0 = ep0 * nd0 + INV_FACT[k];
            ep1 = ep1 * nd1 + INV_FACT[k];
        }
        let et0 = table[row0 + BOYS_STRIDE] * ep0;
        let et1 = table[row1 + BOYS_STRIDE] * ep1;
        let two_t0 = 2.0 * t0;
        let two_t1 = 2.0 * t1;
        let mut m = m_max;
        while m > 0 {
            out0[m - 1] = (two_t0 * out0[m] + et0) * INV_ODD[m];
            out1[m - 1] = (two_t1 * out1[m] + et1) * INV_ODD[m];
            m -= 1;
        }
    }
}

/// Evaluate four independent Boys ladders `F_0..F_{m_max}` at `t[0..4]` into the
/// lane-major rows `out[m][lane]` — the four table-path computations run in
/// lockstep so the Horner / `e^{-T}` / downward-recurrence arithmetic
/// autovectorizes across lanes. Each lane's expression sequence is exactly
/// [`boys_array`]'s (so each lane's results are bitwise identical to four
/// separate calls — fingerprint-guarded); only the storage layout (lane-major
/// rows) differs. Falls back to four sequential [`boys_array`] calls when any
/// lane is outside the table region.
///
/// # Panics
/// Panics (in debug builds) if `out` is too short or any `t` is negative.
pub fn boys_array4(m_max: usize, t: [f64; 4], out: &mut [[f64; 4]]) {
    debug_assert!(out.len() > m_max, "output buffer too short for m_max");
    debug_assert!(t.iter().all(|&x| x >= 0.0), "Boys function requires T >= 0");
    if !(t.iter().all(|&x| x < BOYS_TABLE_TMAX) && m_max + BOYS_K < BOYS_STRIDE) {
        let mut row = [0.0f64; BOYS_STRIDE];
        for lane in 0..4 {
            boys_array(m_max, t[lane], &mut row[..=m_max]);
            for m in 0..=m_max {
                out[m][lane] = row[m];
            }
        }
        return;
    }
    let table = BOYS_TABLE.get_or_init(build_boys_table);
    let mut row = [0usize; 4];
    let mut base = [0usize; 4];
    let mut nd = [0.0f64; 4];
    for lane in 0..4 {
        let i = (t[lane] * (1.0 / BOYS_H)).round() as usize;
        nd[lane] = i as f64 * BOYS_H - t[lane];
        row[lane] = i * BOYS_ROW;
        base[lane] = row[lane] + m_max;
    }
    let mut top = [0.0f64; 4];
    for lane in 0..4 {
        top[lane] = table[base[lane] + BOYS_K] * INV_FACT[BOYS_K];
    }
    for k in (0..BOYS_K).rev() {
        for lane in 0..4 {
            top[lane] = top[lane] * nd[lane] + table[base[lane] + k] * INV_FACT[k];
        }
    }
    out[m_max] = top;
    if m_max > 0 {
        let mut ep = [INV_FACT[BOYS_K]; 4];
        for k in (0..BOYS_K).rev() {
            for lane in 0..4 {
                ep[lane] = ep[lane] * nd[lane] + INV_FACT[k];
            }
        }
        let mut et = [0.0f64; 4];
        let mut two_t = [0.0f64; 4];
        for lane in 0..4 {
            et[lane] = table[row[lane] + BOYS_STRIDE] * ep[lane];
            two_t[lane] = 2.0 * t[lane];
        }
        let mut m = m_max;
        while m > 0 {
            let cur = out[m];
            let mut lower = [0.0f64; 4];
            for lane in 0..4 {
                lower[lane] = (two_t[lane] * cur[lane] + et[lane]) * INV_ODD[m];
            }
            out[m - 1] = lower;
            m -= 1;
        }
    }
}

/// Evaluate the **erf-attenuated** Boys ladder `F_0^ω(T) .. F_{m_max}^ω(T)` for
/// the long-range Coulomb kernel `erf(ω·r₁₂)/r₁₂` into `out[0..=m_max]`.
///
/// With the reduced pair exponent `ρ` and `s = ω²/(ρ + ω²)`,
///
/// ```text
///   F_m^ω(T) = s^{m+1/2} · F_m(s·T) = ∫_0^{√s} t^{2m} exp(−T t²) dt,
/// ```
///
/// i.e. the Boys integral with its upper limit truncated at `√s` — exactly what
/// the Gaussian transform of `erf(ωr)/r` (range `u ∈ [0, ω]` instead of
/// `[0, ∞)`) produces after the Rys substitution `t² = u²/(ρ + u²)`. Feeding
/// `F_m^ω` in place of `F_m` turns any Coulomb ERI/Hermite machinery into its
/// long-range counterpart; the derivative identity `dF_m^ω/dT = −F_{m+1}^ω`
/// carries over unchanged. References: P. M. W. Gill, R. D. Adamson,
/// *Chem. Phys. Lett.* **261**, 105 (1996); R. Ahlrichs, *Phys. Chem. Chem.
/// Phys.* **8**, 3072 (2006).
///
/// `ω → ∞` gives `s → 1` (plain Boys); `ω → 0⁺` gives `F_m^ω → 0` like
/// `s^{m+1/2} ≈ (ω²/ρ)^{m+1/2}`.
///
/// # Panics
/// Panics (in debug builds) if `out` is too short, `t` is negative, `rho` is
/// not positive, or `omega` is not finite and `> 0`.
pub fn boys_array_erf(m_max: usize, t: f64, rho: f64, omega: f64, out: &mut [f64]) {
    debug_assert!(rho > 0.0, "attenuated Boys requires rho > 0");
    debug_assert!(
        omega.is_finite() && omega > 0.0,
        "attenuated Boys requires a finite omega > 0"
    );
    let s = omega * omega / (rho + omega * omega);
    boys_array(m_max, s * t, out);
    let mut factor = s.sqrt(); // s^{m+1/2}, built up order by order
    for v in out[..=m_max].iter_mut() {
        *v *= factor;
        factor *= s;
    }
}

/// Reference Boys ladder: top order by the non-alternating series (`T` below the
/// [`asymptotic_boundary`]) or the large-`T` asymptotic form, then the stable
/// downward recurrence. This is the validated path that [`boys_array`]'s table is
/// generated from and that it falls back to for large `T`.
pub(crate) fn boys_reference(m_max: usize, t: f64, out: &mut [f64]) {
    debug_assert!(out.len() > m_max, "output buffer too short for m_max");
    debug_assert!(t >= 0.0, "Boys function requires T >= 0");

    let et = (-t).exp();

    out[m_max] = if t < asymptotic_boundary(m_max) {
        boys_series(m_max, t, et)
    } else {
        boys_asymptotic(m_max, t)
    };

    // Downward recurrence: F_{m-1} = (2T F_m + e^{-T}) / (2m-1).
    let mut m = m_max;
    while m > 0 {
        out[m - 1] = (2.0 * t * out[m] + et) / ((2 * m - 1) as f64);
        m -= 1;
    }
}

/// Convenience scalar accessor for a single order. Prefer [`boys_array`] when
/// more than one order is needed.
#[must_use]
pub fn boys(m: usize, t: f64) -> f64 {
    let mut buf = vec![0.0; m + 1];
    boys_array(m, t, &mut buf);
    buf[m]
}

/// Non-alternating power series for the top order:
/// `F_m(T) = e^{-T} Σ_{k>=0} (2T)^k (2m-1)!! / (2m+2k+1)!!`.
///
/// Term ratio `term_{k}/term_{k-1} = 2T / (2m+2k+1)`, with `term_0 = 1/(2m+1)`.
fn boys_series(m: usize, t: f64, et: f64) -> f64 {
    let two_t = 2.0 * t;
    let mut term = 1.0 / (2 * m + 1) as f64;
    let mut sum = term;
    let mut k = 1usize;
    loop {
        term *= two_t / (2 * m + 2 * k + 1) as f64;
        sum += term;
        // All terms positive: stop when the tail is negligible vs the partial sum.
        if term <= sum * 1e-17 {
            break;
        }
        k += 1;
        debug_assert!(k < 1000, "Boys series failed to converge");
        if k >= 1000 {
            break;
        }
    }
    et * sum
}

/// Large-`T` closed form `F_m(T) ≈ (2m-1)!!/2^{m+1} · sqrt(π) · T^{-(m+1/2)}`,
/// built from `F_0 = 0.5 sqrt(π/T)` via the exact asymptotic ratio
/// `F_m/F_{m-1} = (2m-1)/(2T)`.
fn boys_asymptotic(m: usize, t: f64) -> f64 {
    let mut val = 0.5 * (std::f64::consts::PI / t).sqrt();
    let two_t = 2.0 * t;
    for k in 1..=m {
        val *= (2 * k - 1) as f64 / two_t;
    }
    val
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent reference: composite Simpson quadrature of the Boys integral, for
    /// all orders `m = 0..=8` in a single pass. The integrand `x^{2m} e^{−T x²}` shares
    /// the transcendental `e^{−T x²}` and the node `x` across every order, so computing
    /// them once (instead of re-running the 1e6-panel loop per `m`) is ~9× faster and
    /// **bit-identical** to the per-order quadrature: each order keeps its own
    /// `x.powi(2m)` factor and an independent Kahan accumulator fed the same term
    /// sequence in the same order. Kahan compensation keeps the 1e6-term accumulation
    /// accurate to ~1e-12, which matters at large `T` where the integrand is sharp.
    fn boys_quadrature_all(t: f64) -> [f64; 9] {
        let n: usize = 1_000_000; // even number of panels
        let h = 1.0 / n as f64;
        let term = |m: usize, x: f64, et: f64| x.powi(2 * m as i32) * et;

        // Endpoints x = 0 and x = 1 carry Simpson weight 1.
        let (et0, et1) = ((-t * 0.0 * 0.0).exp(), (-t * 1.0 * 1.0).exp());
        let mut sum: [f64; 9] = std::array::from_fn(|m| term(m, 0.0, et0) + term(m, 1.0, et1));
        let mut comp = [0.0f64; 9]; // per-order Kahan compensation
        for i in 1..n {
            let x = i as f64 * h;
            let et = (-t * x * x).exp();
            let weight = if i % 2 == 0 { 2.0 } else { 4.0 };
            for m in 0..9 {
                let y = weight * term(m, x, et) - comp[m];
                let new = sum[m] + y;
                comp[m] = (new - sum[m]) - y;
                sum[m] = new;
            }
        }
        std::array::from_fn(|m| sum[m] * h / 3.0)
    }

    #[test]
    fn value_at_zero_is_reciprocal_odd() {
        // F_m(0) = 1/(2m+1).
        let mut out = [0.0; 9];
        boys_array(8, 0.0, &mut out);
        for (m, &fm) in out.iter().enumerate() {
            let expected = 1.0 / (2 * m + 1) as f64;
            assert!((fm - expected).abs() < 1e-15, "m={m}: {fm} vs {expected}");
        }
    }

    /// The interpolation-table path ([`boys_array`]) must reproduce the reference
    /// series/asymptotic path ([`boys_reference`]) to ~1e-13 relative across a dense
    /// `(m, T)` grid spanning the table region (including off-grid `T`, the worst
    /// case for Taylor interpolation) and the asymptotic fallback. This is the
    /// guard that the speed change did not move any computed Boys value beyond the
    /// crate's tolerance.
    #[test]
    fn table_matches_reference_across_grid() {
        let mut worst = 0.0f64;
        let mut tab = [0.0f64; BOYS_MAX_M + 1];
        let mut refv = [0.0f64; BOYS_MAX_M + 1];
        // Step 0.037 lands between grid points (grid is 0.1) to exercise interpolation;
        // go past TMAX to cover the asymptotic fallback too.
        let mut t = 0.0f64;
        while t <= BOYS_TABLE_TMAX + 20.0 {
            for m_max in [0usize, 1, 2, 4, 8, 12, 16, 20, 24] {
                boys_array(m_max, t, &mut tab[..=m_max]);
                boys_reference(m_max, t, &mut refv[..=m_max]);
                for m in 0..=m_max {
                    let rel = (tab[m] - refv[m]).abs() / refv[m].abs().max(1e-300);
                    worst = worst.max(rel);
                    assert!(
                        rel < 1e-12,
                        "m={m} T={t}: table {} vs ref {} (rel {rel:e})",
                        tab[m],
                        refv[m]
                    );
                }
            }
            t += 0.037;
        }
        eprintln!("worst boys table-vs-reference rel error: {worst:e}");
        assert!(
            worst < 1e-12,
            "worst table-vs-reference rel error {worst:e}"
        );
    }

    #[test]
    fn f0_at_one_matches_known_value() {
        // F_0(1) = (sqrt(pi)/2) erf(1) = 0.7468241328124271.
        assert!((boys(0, 1.0) - 0.746_824_132_812_427_1).abs() < 1e-14);
    }

    #[test]
    fn matches_quadrature_across_regimes() {
        let ts = [
            0.0, 1e-6, 0.01, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 34.9, 35.0, 35.1, 50.0,
            100.0,
        ];
        for &t in &ts {
            let mut out = [0.0; 9];
            boys_array(8, t, &mut out);
            let quad = boys_quadrature_all(t);
            for (m, &fm) in out.iter().enumerate() {
                let q = quad[m];
                let err = (fm - q).abs() / q.abs().max(1e-300);
                assert!(err < 1e-9, "m={m} T={t}: boys={fm} quad={q} relerr={err:e}");
            }
        }
    }

    #[test]
    fn satisfies_upward_recurrence() {
        // (2m+1) F_m = 2T F_{m+1} + e^{-T}.
        for &t in &[0.3, 3.0, 15.0, 37.0, 80.0] {
            let mut out = [0.0; 9];
            boys_array(8, t, &mut out);
            let et = (-t).exp();
            for m in 0..8 {
                let lhs = (2 * m + 1) as f64 * out[m];
                let rhs = 2.0 * t * out[m + 1] + et;
                assert!(
                    (lhs - rhs).abs() < 1e-12 * lhs.abs().max(1.0),
                    "T={t} m={m}"
                );
            }
        }
    }

    /// Independent reference for the attenuated ladder: composite Simpson
    /// quadrature of the truncated Boys integral `∫_0^{√s} t^{2m} e^{−T t²} dt`
    /// (the definition, evaluated directly).
    fn boys_erf_quadrature(m: usize, t: f64, rho: f64, omega: f64) -> f64 {
        let s = omega * omega / (rho + omega * omega);
        let upper = s.sqrt();
        let n: usize = 200_000; // even number of panels
        let h = upper / n as f64;
        let f = |x: f64| x.powi(2 * m as i32) * (-t * x * x).exp();
        let mut sum = f(0.0) + f(upper);
        for i in 1..n {
            let w = if i % 2 == 0 { 2.0 } else { 4.0 };
            sum += w * f(i as f64 * h);
        }
        sum * h / 3.0
    }

    /// `boys_array_erf` matches the truncated-integral definition by direct
    /// quadrature across regimes of `T`, `ω`, and `m`.
    #[test]
    fn erf_matches_truncated_quadrature() {
        for &t in &[0.0, 0.3, 2.5, 12.0, 40.0] {
            for &omega in &[0.2, 0.5, 1.0, 4.0] {
                let rho = 0.9;
                let mut out = [0.0; 7];
                boys_array_erf(6, t, rho, omega, &mut out);
                for (m, &fm) in out.iter().enumerate() {
                    let q = boys_erf_quadrature(m, t, rho, omega);
                    let err = (fm - q).abs() / q.abs().max(1e-300);
                    assert!(err < 1e-9, "m={m} T={t} ω={omega}: {fm} vs {q} ({err:e})");
                }
            }
        }
    }

    /// ω → ∞ recovers the plain Boys ladder (`s → 1`); the attenuated values are
    /// positive and strictly below the Coulomb ones for finite ω.
    #[test]
    fn erf_limits_and_bounds() {
        let rho = 1.3;
        for &t in &[0.0, 1.0, 20.0] {
            let mut plain = [0.0; 9];
            boys_array(8, t, &mut plain);
            let mut large = [0.0; 9];
            boys_array_erf(8, t, rho, 1e8, &mut large);
            for m in 0..=8 {
                assert!(
                    (large[m] - plain[m]).abs() < 1e-7 * plain[m],
                    "ω→∞ m={m} T={t}"
                );
            }
            let mut att = [0.0; 9];
            boys_array_erf(8, t, rho, 0.5, &mut att);
            for m in 0..=8 {
                assert!(att[m] > 0.0 && att[m] < plain[m], "bounds m={m} T={t}");
            }
        }
    }

    /// The attenuated ladder satisfies the shifted recurrence
    /// `(2m+1) F_m^ω = 2T F_{m+1}^ω + s^{m+1/2} e^{−sT}` (the plain recurrence
    /// at `sT`, scaled by `s^{m+1/2}`).
    #[test]
    fn erf_satisfies_recurrence() {
        let (rho, omega) = (0.8, 0.6);
        let s = omega * omega / (rho + omega * omega);
        for &t in &[0.4, 3.0, 25.0] {
            let mut out = [0.0; 9];
            boys_array_erf(8, t, rho, omega, &mut out);
            let est = (-s * t).exp();
            for m in 0..8 {
                let lhs = (2 * m + 1) as f64 * out[m];
                let rhs = 2.0 * t * out[m + 1] + s.powf(m as f64 + 0.5) * est;
                assert!(
                    (lhs - rhs).abs() < 1e-12 * lhs.abs().max(1e-300),
                    "T={t} m={m}"
                );
            }
        }
    }

    #[test]
    fn monotonic_decreasing_in_m() {
        // For fixed T, F_m strictly decreases with m (since 0 <= t <= 1).
        for &t in &[0.0, 1.0, 25.0, 60.0] {
            let mut out = [0.0; 9];
            boys_array(8, t, &mut out);
            for m in 0..8 {
                assert!(out[m] > out[m + 1], "T={t} m={m}");
            }
        }
    }

    /// Tight-core (very large `T`) regime: heavy elements have tight core
    /// exponents (α ~ 1e6–1e8) that drive `T = ρ|P−C|²` to ~1e6–1e8, where
    /// `e^{−T}` underflows to exactly 0 and the ladder is the exact asymptotic
    /// closed form `F_m(T) = (2m−1)!!/2^{m+1} · √π · T^{−(m+1/2)}`. This locks in
    /// that the full ladder (top order by the asymptotic form, lower orders by the
    /// downward recurrence with `e^{−T}=0`) stays accurate up to `m = 24 = 4·MAX_L`
    /// — the regime no other Boys test reaches (the table/quadrature tests stop at
    /// `T ≈ 100`). The double-factorial closed form is an independent reference.
    #[test]
    fn large_t_matches_asymptotic_closed_form() {
        // (2m−1)!! with (−1)!! = 1.
        let odd_df = |m: usize| -> f64 {
            let mut p = 1.0;
            let mut i = 2 * m as i64 - 1;
            while i > 1 {
                p *= i as f64;
                i -= 2;
            }
            p
        };
        let sqrt_pi = std::f64::consts::PI.sqrt();
        for &t in &[1e3_f64, 1e4, 1e6, 1e8] {
            assert_eq!((-t).exp(), 0.0, "expected e^-T underflow at T={t}");
            let mut out = [0.0; 25];
            boys_array(24, t, &mut out);
            for (m, &got) in out.iter().enumerate() {
                let expect =
                    odd_df(m) / 2f64.powi(m as i32 + 1) * sqrt_pi * t.powf(-(m as f64 + 0.5));
                let rel = (got - expect).abs() / expect.abs().max(1e-300);
                assert!(rel < 1e-12, "F_{m}({t}) = {got} vs {expect} (rel {rel:e})");
            }
        }
    }
}
