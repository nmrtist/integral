//! `integral` — native-Rust Gaussian integrals for quantum mechanics.
//!
//! This is the public, Layer-3 crate: basis/molecule description and the
//! one-electron integral drivers. It exposes overlap,
//! kinetic, nuclear-attraction, dipole, and spin-free scalar-relativistic pVp
//! ([`Basis::pvp`] / [`Basis::pvp_charges`]) integrals over contracted Cartesian
//! (and real-spherical) Gaussian shells, two ERI engines, geometric first
//! derivatives, and a one-electron **operator DSL** ([`Operator`] /
//! [`Basis::int1e`]) over polynomials in the position `r` and momentum `p = −i∇`
//! operators — adding a new 1e integral type is a single [`Operator`]
//! declaration, not an engine change.
//!
//! ## Quick start
//!
//! ```
//! use integral::{Basis, Shell};
//!
//! // A single normalized s function (one primitive) at the origin.
//! let s = Shell::new(0, [0.0, 0.0, 0.0], vec![0.8], vec![1.0]).unwrap();
//! let basis = Basis::new(vec![s]);
//! let ovlp = basis.overlap();
//! assert_eq!(ovlp.len(), 1);
//! assert!((ovlp[0] - 1.0).abs() < 1e-12); // self-overlap of a normalized s = 1
//! ```
//!
//! ## Conventions
//!
//! - **Storage.** Dense `f64`, row-major. Square one-electron matrices are
//!   `nao × nao` with `nao = `[`Basis::nao_cart`]; dipole returns three such
//!   matrices `[x, y, z]`.
//! - **Cartesian or spherical.** Shells default to Cartesian, with components in
//!   [`crate::math::am`] ordering (e.g. `d`: `xx, xy, xz, yy, yz, zz`).
//!   [`Shell::new_spherical`] requests the `2l+1` real spherical-harmonic
//!   components instead (see [`ShellKind`]); a basis may mix the two.
//! - **Normalization.** Each primitive is scaled by the shell-level constant
//!   `N(α, l)` (see [`Shell::primitive_coeff`]) and the user-supplied
//!   contraction coefficients. The Cartesian convention normalizes each
//!   **monomial** so that the stretched component `x^l` has unit self-overlap
//!   (`cart_norm`); off-axis components such as `d_xy` therefore have a smaller
//!   self-overlap. (This differs from the *solid-harmonic* Cartesian
//!   normalization by a single scalar per shell — `1` for `s`/`p`; the relative
//!   pattern of components is the same in both.)
//! - **Units.** Atomic units (bohr) throughout.
//!
//! The C ABI lives in the separate `integral-sys` crate.

#![forbid(unsafe_code)]

pub mod engine;
pub mod math;

mod df;
mod direct;
mod ecp;
mod eri_batch;
mod eri_builder;
mod grad;
mod grid;
mod integrals;
mod operator;
#[cfg(feature = "periodic")]
pub mod periodic;
mod shell;
mod spherical;

use std::fmt;

pub use df::{Bra3cFill, Eri3cBuilder};
pub use direct::{DirectBuffers, DirectContractor, DirectWorkspace};
pub use ecp::{Ecp, EcpPrimitive, MAX_ECP_GRAD_L};
pub use eri_builder::{BraPairFill, EriBuilder};
pub use grad::{Gradient1e, GradientEri, MAX_GRAD_L};
pub use integrals::{select_engine, Engine, EriKernel, ScreeningStats};
pub use operator::{Factor, Operator, OperatorMatrix, Term};
pub use shell::{Basis, Shell, ShellKind};

/// Errors returned when constructing or using a [`Shell`] / [`Basis`].
#[derive(Debug, Clone, PartialEq)]
pub enum IntegralError {
    /// A shell's exponent and coefficient vectors had different lengths.
    MismatchedContraction {
        exponents: usize,
        coefficients: usize,
    },
    /// A shell's angular momentum exceeds the engine's supported maximum.
    AngularMomentumTooHigh { l: usize, max: usize },
    /// A shell was constructed with no primitives.
    EmptyContraction,
    /// A geometric-derivative build was requested for a shell whose angular
    /// momentum exceeds the gradient maximum. The center-derivative relation
    /// raises the differentiated shell to `l + 1`, so gradients require
    /// `l ≤ MAX_L − 1` to keep the raised shell inside the engines' validated
    /// `MAX_L` range.
    AngularMomentumTooHighForGradient { l: usize, max: usize },
    /// A center that must coincide with a basis atom ([`Basis::atoms`]) does
    /// not. Two builders return this:
    ///
    /// - [`Basis::nuclear_grad`]: a point charge's center is not a basis atom
    ///   (the Hellmann–Feynman term is placed on the charge's atom, so each
    ///   charge must sit on a basis center — the physical molecular case).
    /// - [`Basis::eri_3c_grad_contract`]: the auxiliary basis's atom list
    ///   differs from the orbital basis's (`aux.atoms() != self.atoms()`);
    ///   `center` is the first non-shared (or out-of-order) atom. This doubles
    ///   as the DF-gradient atom-mismatch error because the enum is exhaustive
    ///   (not `#[non_exhaustive]`), so adding a variant would be a breaking
    ///   change.
    ChargeNotOnAtom { center: [f64; 3] },
    /// A density-contracted ERI gradient was given a `gamma` slice of the wrong
    /// length: `nao⁴` for [`Basis::eri_grad_contract`], `nao²·naux` for
    /// [`Basis::eri_3c_grad_contract`], `naux²` for
    /// [`Basis::eri_2c_grad_contract`].
    GammaLengthMismatch { expected: usize, got: usize },
    /// A one-electron operator-DSL build ([`Basis::int1e`]) was requested for a
    /// shell whose angular momentum is too high for the operator's degree. The
    /// DSL folds the operator onto the ket, raising it to `l + degree`, which
    /// must stay inside the engines' validated `MAX_L`; so a degree-`d` operator
    /// requires every shell to have `l ≤ MAX_L − d`.
    OperatorMomentumTooHigh { l: usize, degree: usize, max: usize },
}

impl fmt::Display for IntegralError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntegralError::MismatchedContraction {
                exponents,
                coefficients,
            } => write!(
                f,
                "contraction length mismatch: {exponents} exponents but {coefficients} coefficients"
            ),
            IntegralError::AngularMomentumTooHigh { l, max } => {
                write!(f, "angular momentum l={l} exceeds supported maximum {max}")
            }
            IntegralError::EmptyContraction => write!(f, "shell has no primitives"),
            IntegralError::AngularMomentumTooHighForGradient { l, max } => write!(
                f,
                "angular momentum l={l} exceeds the gradient maximum {max} \
                 (the derivative raises the shell to l+1)"
            ),
            IntegralError::ChargeNotOnAtom { center } => write!(
                f,
                "gradient center {center:?} is not a basis atom (point charge \
                 off-atom, or aux/orbital basis atom-list mismatch)"
            ),
            IntegralError::GammaLengthMismatch { expected, got } => write!(
                f,
                "density coefficient array has {got} elements, expected {expected}"
            ),
            IntegralError::OperatorMomentumTooHigh { l, degree, max } => write!(
                f,
                "angular momentum l={l} is too high for a degree-{degree} operator \
                 (the operator raises the ket to l+{degree}, which must stay ≤ {max})"
            ),
        }
    }
}

impl std::error::Error for IntegralError {}
