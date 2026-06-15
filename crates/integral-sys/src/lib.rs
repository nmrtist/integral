//! `integral-sys` — `extern "C"` ABI for the `integral` crate.
//!
//! This crate currently exposes a minimal ABI surface for integration and
//! version checks.
//!
//! This is the one crate where `unsafe` is permitted (FFI). It uses
//! `#![deny(unsafe_code)]` rather than `forbid`, so any future `unsafe` block
//! must be opted in with an explicit, reviewable local `#[allow(unsafe_code)]`
//! and carry a `// SAFETY:` justification.
#![deny(unsafe_code)]

/// Minimal ABI version symbol.
///
/// Returns the ABI version so a C caller can verify linkage end-to-end.
// SAFETY: `#[no_mangle]` exports an unmangled symbol, which the `unsafe_code`
// lint flags because duplicate symbol names across libraries are UB. This is the
// crate's intended C-ABI surface; the explicit `allow` is the reviewable opt-in
// the crate-level `deny(unsafe_code)` requires.
#[allow(unsafe_code)]
#[no_mangle]
pub extern "C" fn integral_abi_version() -> u32 {
    0
}
