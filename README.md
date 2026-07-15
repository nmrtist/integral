# integral

Native-Rust Gaussian integrals for quantum chemistry.

`integral` evaluates overlap, kinetic, nuclear-attraction, multipole, and
two-electron repulsion integrals over contracted Gaussian shells in safe,
stable Rust. The workspace has no dependency on external quantum-chemistry
libraries or C/Fortran integral backends.

[![Crates.io](https://img.shields.io/crates/v/integral.svg)](https://crates.io/crates/integral)
[![Documentation](https://docs.rs/integral/badge.svg)](https://docs.rs/integral)

## Highlights

- Pure Rust implementation of one-electron and two-electron Gaussian integrals.
- Cartesian and real-spherical shells through the public `integral` crate.
- Two ERI engines: OS/HGP — with monomorphized, SIMD-batched (4 quartets/lane)
  kernels for the d/f Cartesian classes — for low-order contracted work, and Rys
  quadrature for the general path.
- First derivatives for one-electron integrals and ERIs.
- Operator DSL for one-electron integrals built from `r` and `p`.
- Dense ERI builder API that supports caller-managed parallel execution.

## Quick Start

```toml
[dependencies]
integral = "0.4.1"
```

```rust
use integral::{Basis, Shell};

let exps = vec![3.425250914, 0.623913730, 0.168855404];
let coef = vec![0.154328967, 0.535328142, 0.444634542];
let basis = Basis::new(vec![
    Shell::new(0, [0.0, 0.0, 0.0], exps.clone(), coef.clone()).unwrap(),
    Shell::new(0, [0.0, 0.0, 1.4], exps, coef).unwrap(),
]);

let n = basis.nao();
let s = basis.overlap();
let t = basis.kinetic();
let v = basis.nuclear(&[([0.0, 0.0, 0.0], 1.0), ([0.0, 0.0, 1.4], 1.0)]);
let eri = basis.eri();

println!("nao = {n}");
println!("S01 = {:.6}", s[1]);
println!("T00 = {:.6}", t[0]);
println!("V00 = {:.6}", v[0]);
println!("(00|11) = {:.6}", eri[(0 * n + 0) * n * n + (1 * n + 1)]);
```

## Capabilities

| Capability | Status | Limit |
|---|---|---|
| Overlap, kinetic, nuclear attraction | Stable | up to `l = 6` |
| Dipole and multipole integrals | Stable | up to `l = 6` |
| Coulomb ERIs | Stable | each shell up to `l = 6` |
| Cartesian output | Stable | default |
| Real-spherical output | Stable | via spherical shells |
| Schwarz-screened ERIs | Stable | caller-selected threshold |
| First derivatives | Stable | shells up to `l = 5` |
| One-electron operator DSL | Stable | public API |
| `integral-sys` C ABI | Minimal | ABI version surface only |

## Parallel Use

The library is single-threaded but thread-safe. Parallelism is intended at the
call site. For dense ERIs, `EriBuilder` partitions a shared output tensor into
disjoint tasks so a caller can evaluate shell-pair work concurrently without
unsafe code inside the library.

## Validation

The test suite checks:

- closed-form analytic identities for core one-electron quantities,
- exact symmetry and invariance properties,
- agreement between independent ERI implementations,
- finite-difference checks for derivative code, and
- platform-independent numerical invariants for the low-level math routines.

For repository verification, use release builds:

```bash
cargo test --workspace --release
```

## Workspace

| Crate | Role |
|---|---|
| `crates/integral` | The library (the only crate published to crates.io). Public API — shells, bases, drivers, gradients, screening — plus the internal `math` module (Boys function, Rys quadrature, normalization, spherical transforms) and `engine` module (one- and two-electron engines) |
| `crates/integral-sys` | C ABI (`extern "C"`); built as a C artifact, not published to crates.io |
| `crates/integral-codegen` | Build-time Rust code generation support (workspace-internal, not published) |
| `benches` | Benchmarks and profiling harnesses |
| `xtask` | Maintainer utilities |

## Safety And Toolchain

- All library crates forbid unsafe code except `integral-sys`, which keeps the
  FFI surface behind `#![deny(unsafe_code)]`.
- MSRV is Rust `1.82`.

## License

Licensed under Apache-2.0 or MIT. See `LICENSE-APACHE` and `LICENSE-MIT`.
