# integral-sys

`extern "C"` ABI for [integral](https://github.com/nmrtist/integral), a native-Rust
library for Gaussian integrals in quantum mechanics.

This crate currently exposes a minimal ABI surface for integration and version
checks. It is the workspace home for the C-facing interface.

This is the one integral crate where `unsafe` is permitted (FFI). It uses
`#![deny(unsafe_code)]` rather than `forbid`, so any `unsafe` must be opted in
with an explicit local `#[allow(unsafe_code)]` and a `// SAFETY:` note.

Licensed under Apache-2.0 or MIT.
