//! One-electron operator DSL over polynomials in `r` and `p` (L2).
//!
//! This module provides a typed
//! representation of a one-electron integrand as a **polynomial in the position
//! operator `r` and the momentum operator `p = −i∇`**, evaluated by
//! **decomposition into the base Cartesian integrals the engines already
//! produce** — no new recurrences. Adding a new integral type is then a single
//! [`Operator`] *declaration*, not an engine change.
//!
//! ## Decomposition (fold the operator onto the ket)
//!
//! For a one-electron integral `⟨χ_a | O | χ_b⟩`, the operator `O` (a product of
//! multiplication operators `r` and derivative operators `p`) acts on the ket
//! function `χ_b`. Each elementary factor expands `χ_b` into a sum of pure
//! Cartesian monomials:
//!
//! - **`r_i` about origin `O`:** `(r − O)_i χ_b = χ_{b+1_i} + (B − O)_i · χ_b`
//!   — raise the ket monomial by one along axis `i`, plus a shifted copy. (This
//!   is exactly the multipole step [`crate::engine::os::dipole_into`] performs.)
//! - **`p_i = −i ∂_{r_i}`:** since `χ_b` depends on `r − B`, `∂_{r_i} = −∂_{B_i}`,
//!   so `p_i χ_b = i · ∂_{B_i} χ_b = i · (2β · χ_{b+1_i} − b_i · χ_{b−1_i})`. The
//!   real part is the **center-derivative** combiner; the operator
//!   carries an explicit factor of `i`.
//!
//! Applying every factor (rightmost first — operators act right-to-left) turns
//! `χ_b` into `Σ_{m'} c(m') · χ_{m'}`, so
//!
//! ```text
//!   ⟨a | O | b⟩ = i^{n_p} · Σ_{m'} c(m') · ⟨a | m'⟩,
//! ```
//!
//! where `n_p` is the number of momentum factors and each `⟨a | m'⟩` is a plain
//! **overlap** integral the [`crate::engine::os`] engine already computes at a shifted ket
//! momentum. The bra stays at `la`; the only engine call is `overlap_into`.
//!
//! ## Hermiticity / the imaginary unit (watch this)
//!
//! Every `p` factor contributes one `i`, so a term with `n_p` momentum factors
//! contributes the **phase** `i^{n_p} ∈ {1, i, −1, −i}`. The real-part expansion
//! above is routed to a real or imaginary output buffer with a sign accordingly:
//!
//! | `n_p mod 4` | phase | buffer | sign |
//! |-------------|-------|--------|------|
//! | 0           |  `1`  | real   | `+`  |
//! | 1           |  `i`  | imag   | `+`  |
//! | 2           | `−1`  | real   | `−`  |
//! | 3           | `−i`  | imag   | `−`  |
//!
//! Consequently `r`, `r_i r_j`, and `p·p` (kinetic) are **real-symmetric** while
//! `p` and `r × p` are **imaginary/antisymmetric** (anti-Hermitian real
//! representation) — the Hermiticity character falls out of `n_p` and is not
//! assumed. The `i`/sign convention is documented above.
//!
//! ## Angular-momentum limit
//!
//! Folding onto the ket raises it to `l_b + degree`, so the overlap engine must
//! stay inside its validated `MAX_L`: `l_b + degree ≤ MAX_L`. The driver enforces
//! this with a typed error; here it is a debug assertion.

use crate::math::am::{cart_components, cart_index, n_cart};

use crate::engine::os::{overlap_into, Prim, Vec3, MAX_L};

/// An elementary factor of a one-electron operator product, along a Cartesian
/// axis (`0 = x`, `1 = y`, `2 = z`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Factor {
    /// Position `r_axis` (relative to the operator's origin). Raises the ket
    /// monomial by one along `axis` (plus the origin-shift copy).
    R(usize),
    /// Momentum `p_axis = −i ∂_{r_axis}`. The center-derivative combiner along
    /// `axis`, carrying a factor of `i`.
    P(usize),
}

/// One product term of an operator: a real scalar `coeff` times an ordered
/// product of [`Factor`]s, applied **right-to-left** to the ket.
#[derive(Debug, Clone, PartialEq)]
pub struct Term {
    /// Real scalar coefficient.
    pub coeff: f64,
    /// Ordered factors (operators act right-to-left, i.e. the last factor acts
    /// on the ket first).
    pub factors: Vec<Factor>,
}

impl Term {
    /// Construct a term from its coefficient and ordered factors.
    #[must_use]
    pub fn new(coeff: f64, factors: Vec<Factor>) -> Self {
        Term { coeff, factors }
    }

    /// Number of momentum (`p`) factors — sets the term's `i^{n_p}` phase.
    #[must_use]
    fn n_momentum(&self) -> usize {
        self.factors
            .iter()
            .filter(|f| matches!(f, Factor::P(_)))
            .count()
    }
}

/// A one-electron operator: a **sum of [`Term`]s**, polynomial in `r` and `p`,
/// with a documented `origin` for its `r` factors (the multipole / gauge origin).
///
/// Adding a new integral type is constructing one of these — see the
/// constructors ([`Operator::dipole`], [`Operator::quadrupole`],
/// [`Operator::momentum`], [`Operator::angular_momentum`], …). The engines are
/// untouched.
#[derive(Debug, Clone, PartialEq)]
pub struct Operator {
    /// The terms summed to form the operator.
    pub terms: Vec<Term>,
    /// Origin for the `r` factors (multipole / gauge origin), in bohr.
    pub origin: Vec3,
}

impl Operator {
    /// Construct an operator from its terms and `r`-origin.
    #[must_use]
    pub fn new(terms: Vec<Term>, origin: Vec3) -> Self {
        Operator { terms, origin }
    }

    /// The **identity** operator — its integral is the plain overlap `⟨a|b⟩`.
    #[must_use]
    pub fn identity() -> Self {
        Operator::new(vec![Term::new(1.0, vec![])], [0.0; 3])
    }

    /// The dipole component `r_axis` about `origin` — `⟨a|(r−O)_axis|b⟩`.
    /// Real-symmetric.
    #[must_use]
    pub fn dipole(axis: usize, origin: Vec3) -> Self {
        Operator::new(vec![Term::new(1.0, vec![Factor::R(axis)])], origin)
    }

    /// The second-moment / quadrupole component `r_i r_j` about `origin` —
    /// `⟨a|(r−O)_i (r−O)_j|b⟩`. Real-symmetric.
    #[must_use]
    pub fn quadrupole(i: usize, j: usize, origin: Vec3) -> Self {
        Operator::new(
            vec![Term::new(1.0, vec![Factor::R(i), Factor::R(j)])],
            origin,
        )
    }

    /// The kinetic-energy operator `−½∇² = ½ p·p`. Real-symmetric; reproduces
    /// [`crate::engine::os::kinetic_into`].
    #[must_use]
    pub fn kinetic() -> Self {
        let terms = (0..3)
            .map(|ax| Term::new(0.5, vec![Factor::P(ax), Factor::P(ax)]))
            .collect();
        Operator::new(terms, [0.0; 3])
    }

    /// The momentum component `p_axis = −i∂_axis`. **Imaginary**/antisymmetric.
    #[must_use]
    pub fn momentum(axis: usize) -> Self {
        Operator::new(vec![Term::new(1.0, vec![Factor::P(axis)])], [0.0; 3])
    }

    /// The orbital-angular-momentum component `(r × p)_axis` about `origin`,
    /// e.g. `L_x = (r−O)_y p_z − (r−O)_z p_y`. **Imaginary**/antisymmetric (the
    /// `i` is carried explicitly).
    #[must_use]
    pub fn angular_momentum(axis: usize, origin: Vec3) -> Self {
        // Cyclic (j, k) after `axis`: x→(y,z), y→(z,x), z→(x,y).
        let (j, k) = match axis {
            0 => (1, 2),
            1 => (2, 0),
            _ => (0, 1),
        };
        Operator::new(
            vec![
                Term::new(1.0, vec![Factor::R(j), Factor::P(k)]),
                Term::new(-1.0, vec![Factor::R(k), Factor::P(j)]),
            ],
            origin,
        )
    }

    /// The operator's polynomial degree: the largest number of factors in any
    /// term. The ket is raised by at most this much, so a shell pair is
    /// evaluable iff `l_ket + degree ≤ MAX_L`.
    #[must_use]
    pub fn degree(&self) -> usize {
        self.terms
            .iter()
            .map(|t| t.factors.len())
            .max()
            .unwrap_or(0)
    }
}

/// Which part of the complex matrix a term contributes to, and with which sign,
/// from its `i^{n_p}` phase.
fn phase(n_p: usize) -> (bool, f64) {
    // (is_imaginary, sign)
    match n_p % 4 {
        0 => (false, 1.0),
        1 => (true, 1.0),
        2 => (false, -1.0),
        _ => (true, -1.0),
    }
}

/// Apply one factor to a sparse monomial expansion `Σ c·χ_m` (real
/// coefficients), returning the expanded `Σ c'·χ_{m'}`. `beta` is the ket
/// exponent (for `p`); `b_minus_o = B − O` is the ket-center offset from the
/// operator origin (for `r`).
fn apply_factor(
    factor: Factor,
    beta: f64,
    b_minus_o: Vec3,
    src: &[([usize; 3], f64)],
) -> Vec<([usize; 3], f64)> {
    let mut out = Vec::with_capacity(src.len() * 2);
    match factor {
        // (r − O)_ax · χ_m = χ_{m+1_ax} + (B − O)_ax · χ_m
        Factor::R(ax) => {
            for &(m, c) in src {
                let mut up = m;
                up[ax] += 1;
                out.push((up, c));
                out.push((m, c * b_minus_o[ax]));
            }
        }
        // Re(p_ax) · χ_m = 2β · χ_{m+1_ax} − m_ax · χ_{m−1_ax}
        Factor::P(ax) => {
            for &(m, c) in src {
                let mut up = m;
                up[ax] += 1;
                out.push((up, c * 2.0 * beta));
                if m[ax] >= 1 {
                    let mut dn = m;
                    dn[ax] -= 1;
                    out.push((dn, -c * m[ax] as f64));
                }
            }
        }
    }
    out
}

/// Accumulate `scale · ⟨a | Op | b⟩` for one **primitive** pair into the real and
/// imaginary Cartesian shell-pair blocks `out_re`/`out_im` (each row-major
/// `n_cart(la) × n_cart(lb)`, the same layout as [`crate::engine::os::overlap_into`]).
///
/// The only engine call is `overlap_into`, at ket momenta `0..=l_b + degree`;
/// the operator's `r`/`p` factors are decomposed symbolically (see the module
/// docs). `scale` is the contraction-coefficient product for this primitive
/// pair, supplied by the driver.
///
/// # Panics
/// Debug-asserts the block sizes and that the operator does not raise the ket
/// beyond `MAX_L` (`l_b + Op::degree() ≤ MAX_L`); the driver enforces the latter
/// with a typed error.
pub fn operator_into(
    op: &Operator,
    a: Prim,
    b: Prim,
    scale: f64,
    out_re: &mut [f64],
    out_im: &mut [f64],
) {
    let (la, lb) = (a.l, b.l);
    let beta = b.exp;
    let deg = op.degree();
    let lmax_ket = lb + deg;
    debug_assert!(lmax_ket <= MAX_L, "operator raises ket beyond MAX_L");
    let na = n_cart(la);
    let nb = n_cart(lb);
    debug_assert_eq!(out_re.len(), na * nb, "real block size");
    debug_assert_eq!(out_im.len(), na * nb, "imag block size");

    // Overlap blocks ⟨la | l'⟩ for every ket momentum the expansion can reach.
    let ov: Vec<Vec<f64>> = (0..=lmax_ket)
        .map(|lp| {
            let mut blk = vec![0.0; na * n_cart(lp)];
            overlap_into(
                Prim::new(a.exp, a.center, la),
                Prim::new(beta, b.center, lp),
                1.0,
                &mut blk,
            );
            blk
        })
        .collect();

    let b_minus_o = [
        b.center[0] - op.origin[0],
        b.center[1] - op.origin[1],
        b.center[2] - op.origin[2],
    ];
    let comps_b = cart_components(lb);

    for term in &op.terms {
        let (is_imag, psign) = phase(term.n_momentum());
        let out = if is_imag { &mut *out_im } else { &mut *out_re };
        for (ib, &bm) in comps_b.iter().enumerate() {
            // Expand the operator acting on this ket monomial (rightmost first).
            let mut exp = vec![(bm, 1.0)];
            for factor in term.factors.iter().rev() {
                exp = apply_factor(*factor, beta, b_minus_o, &exp);
            }
            for (m, rc) in exp {
                let lp = m[0] + m[1] + m[2];
                let idx = cart_index(m);
                let np = n_cart(lp);
                let blk = &ov[lp];
                let w = scale * psign * term.coeff * rc;
                for ia in 0..na {
                    out[ia * nb + ib] += w * blk[ia * np + idx];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Identity reproduces the overlap block element-for-element.
    #[test]
    fn identity_is_overlap() {
        let a = Prim::new(1.1, [0.0, 0.0, 0.0], 1);
        let b = Prim::new(0.7, [0.3, -0.2, 0.5], 1);
        let mut s = vec![0.0; 9];
        overlap_into(a, b, 1.0, &mut s);
        let (mut re, mut im) = (vec![0.0; 9], vec![0.0; 9]);
        operator_into(&Operator::identity(), a, b, 1.0, &mut re, &mut im);
        for k in 0..9 {
            assert!((re[k] - s[k]).abs() < 1e-14, "{} vs {}", re[k], s[k]);
            assert_eq!(im[k], 0.0);
        }
    }

    // Dipole r_x about the ket center equals ⟨a|b+1_x⟩ (raise) exactly.
    #[test]
    fn dipole_about_ket_center_is_pure_raise() {
        let a = Prim::new(1.0, [0.1, 0.0, 0.0], 0);
        let bc = [0.4, -0.2, 0.3];
        let b = Prim::new(0.8, bc, 0);
        // Origin at the ket center → no shift term, pure raise to p_x.
        let op = Operator::dipole(0, bc);
        let (mut re, mut im) = (vec![0.0; 1], vec![0.0; 1]);
        operator_into(&op, a, b, 1.0, &mut re, &mut im);
        // Compare with ⟨a | (s-raised-to p_x) ⟩ = overlap(la=0, lb=1)[x].
        let mut raise = vec![0.0; 3];
        overlap_into(a, Prim::new(0.8, bc, 1), 1.0, &mut raise);
        assert!((re[0] - raise[0]).abs() < 1e-14);
        assert_eq!(im[0], 0.0);
    }

    // Momentum p_x is purely imaginary and antisymmetric for same-center s|s
    // (⟨s|p|s⟩ = 0 at same center).
    #[test]
    fn momentum_same_center_s_is_zero() {
        let c = [0.2, 0.1, -0.3];
        let a = Prim::new(1.0, c, 0);
        let b = Prim::new(1.0, c, 0);
        let (mut re, mut im) = (vec![0.0; 1], vec![0.0; 1]);
        operator_into(&Operator::momentum(0), a, b, 1.0, &mut re, &mut im);
        assert_eq!(re[0], 0.0);
        assert!(im[0].abs() < 1e-14, "im={}", im[0]);
    }

    // ½ p·p reproduces the kinetic block (the analytic identity), for d|d.
    #[test]
    fn half_p_dot_p_is_kinetic() {
        use crate::engine::os::kinetic_into;
        let a = Prim::new(1.3, [0.0, 0.0, 0.0], 2);
        let b = Prim::new(0.6, [0.4, -0.3, 0.5], 2);
        let mut t = vec![0.0; 36];
        kinetic_into(a, b, 1.0, &mut t);
        let (mut re, mut im) = (vec![0.0; 36], vec![0.0; 36]);
        operator_into(&Operator::kinetic(), a, b, 1.0, &mut re, &mut im);
        for k in 0..36 {
            assert!((re[k] - t[k]).abs() < 1e-12, "k={k}: {} vs {}", re[k], t[k]);
            assert_eq!(im[k], 0.0);
        }
    }

    // Degree drives the L reach.
    #[test]
    fn degree_is_max_factors() {
        assert_eq!(Operator::identity().degree(), 0);
        assert_eq!(Operator::dipole(0, [0.0; 3]).degree(), 1);
        assert_eq!(Operator::momentum(1).degree(), 1);
        assert_eq!(Operator::quadrupole(0, 1, [0.0; 3]).degree(), 2);
        assert_eq!(Operator::kinetic().degree(), 2);
        assert_eq!(Operator::angular_momentum(2, [0.0; 3]).degree(), 2);
    }
}
