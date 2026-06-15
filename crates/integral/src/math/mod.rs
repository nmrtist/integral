//! Layer 0 shared numerical primitives.
//!
//! This module hosts the foundation used by both integral engines:
//!
//! - [`boys`]   — the Boys function `F_m(T)`.
//! - [`rys`]    — Rys-quadrature roots & weights (for ERIs).
//! - [`norm`]   — Gaussian-type-orbital normalization constants.
//! - [`am`]     — angular-momentum bookkeeping (Cartesian component counts/orderings).
//! - [`solid_harmonics`] — Cartesian → real spherical-harmonic (`c2s`) transform.
//!
//! It is intentionally **dependency-free**.

pub mod am;
pub mod boys;
pub mod norm;
pub mod rys;
pub mod solid_harmonics;
