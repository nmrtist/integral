# integral-codegen

Pure-Rust build-time code generation for
[integral](https://github.com/nmrtist/integral), a native-Rust library for Gaussian
integrals in quantum mechanics.

Intended to be invoked from `build.rs` scripts to emit specialized
per-angular-momentum kernels and coefficient tables as plain Rust source —
keeping all build-time code generation in one language rather than a separate
external generator toolchain.

This crate currently provides the workspace's build-time code generation seam.

Licensed under Apache-2.0 or MIT.
