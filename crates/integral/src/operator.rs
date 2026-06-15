//! One-electron operator-DSL driver (L3).
//!
//! [`Basis::int1e`] evaluates any one-electron [`Operator`] — a polynomial in the
//! position `r` and momentum `p = −i∇` operators — over the basis, returning the
//! complex matrix as separate real and imaginary parts ([`OperatorMatrix`]). The
//! decomposition into base overlap integrals lives in [`crate::engine::operator`];
//! this driver loops shell pairs, contracts primitives, applies the
//! Cartesian→spherical transform to **each** part, and places the blocks.
//!
//! ## Additive layer
//!
//! This is purely additive: the bespoke fast paths ([`Basis::overlap`],
//! [`Basis::kinetic`], [`Basis::dipole`]) stay as-is. The DSL exists for
//! generality (a new integral type is one [`Operator`] declaration, not an engine
//! change) and is validated *against* those bespoke paths (`tests/operator_dsl`).
//!
//! ## Hermiticity / the imaginary unit
//!
//! `r`, `r_i r_j`, and `p·p` (kinetic) are real-symmetric → [`OperatorMatrix::imag`]
//! is ~0; `p` and `r × p` are imaginary/antisymmetric → [`OperatorMatrix::real`]
//! is ~0. The `i`/sign convention is documented in the module docs.

use crate::engine::operator::operator_into;
use crate::engine::os::MAX_L;

pub use crate::engine::operator::{Factor, Operator, Term};

use crate::integrals::{place_block, to_func_1e};
use crate::shell::Basis;
use crate::IntegralError;

/// The complex matrix of a one-electron operator, stored as separate real and
/// imaginary parts. Both are dense row-major `nao × nao` (same ordering/strides
/// as the value builders).
///
/// A Hermitian operator with a real Gaussian representation has
/// [`OperatorMatrix::imag`] ≈ 0 (e.g. `r`, kinetic); an anti-Hermitian one (e.g.
/// `p`, `r × p`) has [`OperatorMatrix::real`] ≈ 0 and an antisymmetric imaginary
/// part.
#[derive(Debug, Clone, PartialEq)]
pub struct OperatorMatrix {
    nao: usize,
    re: Vec<f64>,
    im: Vec<f64>,
}

impl OperatorMatrix {
    /// Matrix dimension `nao`.
    #[must_use]
    pub fn nao(&self) -> usize {
        self.nao
    }

    /// The real part `Re⟨μ|O|ν⟩` (row-major `nao × nao`).
    #[must_use]
    pub fn real(&self) -> &[f64] {
        &self.re
    }

    /// The imaginary part `Im⟨μ|O|ν⟩` (row-major `nao × nao`).
    #[must_use]
    pub fn imag(&self) -> &[f64] {
        &self.im
    }

    /// Largest `|Im⟨μ|O|ν⟩|` — ~0 for a real (Hermitian, real-symmetric)
    /// operator. A diagnostic for the operator's Hermiticity character.
    #[must_use]
    pub fn max_abs_imag(&self) -> f64 {
        self.im.iter().fold(0.0_f64, |m, &v| m.max(v.abs()))
    }

    /// Largest `|Re⟨μ|O|ν⟩|` — ~0 for a purely imaginary (anti-Hermitian)
    /// operator.
    #[must_use]
    pub fn max_abs_real(&self) -> f64 {
        self.re.iter().fold(0.0_f64, |m, &v| m.max(v.abs()))
    }
}

impl Basis {
    /// Evaluate a one-electron [`Operator`] over the basis, returning its complex
    /// matrix as real + imaginary parts ([`OperatorMatrix`]).
    ///
    /// The operator is decomposed into base overlap integrals
    /// ([`crate::engine::operator`]); spherical shells are transformed to their
    /// `2l+1` components (the transform is applied to each part). This is the
    /// generic operator-DSL path; the bespoke [`Basis::overlap`] /
    /// [`Basis::kinetic`] / [`Basis::dipole`] builders remain the fast paths.
    ///
    /// # Errors
    /// [`IntegralError::OperatorMomentumTooHigh`] if any shell has
    /// `l + op.degree() > MAX_L` (the DSL raises the ket to `l + degree`).
    pub fn int1e(&self, op: &Operator) -> Result<OperatorMatrix, IntegralError> {
        let deg = op.degree();
        for s in self.shells() {
            if s.l() + deg > MAX_L {
                return Err(IntegralError::OperatorMomentumTooHigh {
                    l: s.l(),
                    degree: deg,
                    max: MAX_L,
                });
            }
        }

        let n = self.nao();
        let offs = self.offsets();
        let mut re = vec![0.0; n * n];
        let mut im = vec![0.0; n * n];
        let shells = self.shells();

        for (si, sa) in shells.iter().enumerate() {
            for (sj, sb) in shells.iter().enumerate() {
                let (na, nb) = (sa.n_cart(), sb.n_cart());
                let mut br = vec![0.0; na * nb];
                let mut bi = vec![0.0; na * nb];
                for pi in 0..sa.n_prim() {
                    for pj in 0..sb.n_prim() {
                        let scale = sa.primitive_coeff(pi) * sb.primitive_coeff(pj);
                        operator_into(op, sa.prim(pi), sb.prim(pj), scale, &mut br, &mut bi);
                    }
                }
                // c2s each part independently (the transform is real & linear).
                let br = to_func_1e(br, sa, sb);
                let bi = to_func_1e(bi, sa, sb);
                let nbf = sb.n_func();
                place_block(&mut re, n, offs[si], offs[sj], &br, nbf);
                place_block(&mut im, n, offs[si], offs[sj], &bi, nbf);
            }
        }
        Ok(OperatorMatrix { nao: n, re, im })
    }
}
