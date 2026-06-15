//! Layer 1 (integral engines) + Layer 2 (operator/derivative layer).
//!
//! - [`os`] — the one-electron Obara–Saika engine (overlap, kinetic,
//!   nuclear-attraction, and multipole/dipole integrals).
//! - [`os_eri`] — the OS/HGP two-electron (ERI) engine.
//! - [`rys`] — the Rys-quadrature ERI engine.
//! - [`operator`] — the L2 one-electron operator DSL over `r` and `p`.
//! - [`deriv`] — geometric (nuclear-coordinate) first derivatives.

pub mod deriv;
pub mod operator;
pub mod os;
pub mod os_eri;
pub mod rys;
