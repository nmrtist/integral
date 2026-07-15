# integral

Native-Rust Gaussian integrals for quantum chemistry.

`integral` computes the integrals quantum-chemistry methods are built on — overlap,
kinetic energy, nuclear attraction, multipole/dipole, and two-electron repulsion
integrals (ERIs) — over contracted Cartesian and real-spherical Gaussian shells,
in pure, safe, stable Rust with no dependency on external quantum-chemistry
libraries.

[![Crates.io](https://img.shields.io/crates/v/integral.svg)](https://crates.io/crates/integral)
[![Documentation](https://docs.rs/integral/badge.svg)](https://docs.rs/integral)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/nmrtist/integral/blob/main/LICENSE-APACHE)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/nmrtist/integral/blob/main/LICENSE-MIT)

## Highlights

integral ships *two* complementary two-electron integral engines behind a measured
dispatch policy:

- an **Obara–Saika / Head-Gordon–Pople (OS/HGP)** engine with specialized,
  per–angular-momentum recurrences — monomorphized and SIMD-batched (4
  quartets/lane) for the d/f Cartesian classes — fastest at low angular momentum
  and high contraction; and
- a **Rys-quadrature** engine that stays accurate and compact across *all*
  angular momenta with a very small memory footprint.

A `(angular-momentum, contraction)` policy picks the faster engine per shell
quartet; the two agree to tolerance, so the choice is purely about speed.
One-electron integrals are built from an operator DSL over the position `r` and
momentum `p` operators, and all compile-time specialization and table generation
is done in pure Rust.

The integral algorithms follow standard Gaussian-integral literature and keep
output conventions interoperable with common downstream tooling.

## Quick start

```toml
# Cargo.toml
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

println!("S01 = {:.6}", s[1]);
println!("T00 = {:.6}", t[0]);
println!("V00 = {:.6}", v[0]);
println!("(00|11) = {:.6}", eri[(0 * n + 0) * n * n + (1 * n + 1)]);
```

Spherical-harmonic output (`2l+1` real components) is opt-in per
shell via `Shell::new_spherical`; Schwarz-screened ERIs via
`basis.eri_screened(tau)`; geometric gradients via `basis.overlap_grad()`,
`basis.eri_grad()`, etc.; and arbitrary one-electron operators via
`basis.int1e(&Operator)`.

## Feature matrix

| Capability | Status | Limit |
|---|---|---|
| Overlap `S`, kinetic `T`, nuclear attraction `V` | ✅ | up to `l = 6` (i shells) |
| Dipole / multipole | ✅ | up to `l = 6` |
| Two-electron ERIs (Coulomb) — OS/HGP **and** Rys engines | ✅ | each shell up to `l = 6` (`l_total` ≤ 24) |
| Measured engine dispatch (`(l_total, contraction)`) | ✅ | — |
| Cartesian output | ✅ (default) | up to `l = 6` |
| Real-spherical (`c2s`) output | ✅ | up to `l = 6` |
| Schwarz (Cauchy–Schwarz) ERI screening | ✅ | error ≤ `τ` (caller-set) |
| Geometric first derivatives (gradients) of `S/T/V/ERI` | ✅ | shells up to `l = 5` (raised to 6) |
| One-electron operator DSL over `r` and `p` | ✅ | dipole, quadrupole, momentum, angular momentum, kinetic, custom |
| C ABI (`integral-sys`) | Minimal | ABI version surface only |
| Parallel dense-ERI seam (`EriBuilder`, call-site threads) | ✅ | safe, no in-crate runtime |

**Threading.** The library is **single-threaded but thread-safe**: integral
builders take `&self` and share no mutable state, so the recommended way to
parallelize is **at the call site**. For the dense ERI tensor, `EriBuilder`
provides a ready-made **parallel seam**: its grain is the canonical bra shell-pair
`(i, j)` (`bra_pairs()`), and `partition()` hands each bra-pair a *disjoint* set of
output rows, so a driver can fill them concurrently into one buffer with no
synchronisation —

```rust
let builder = basis.eri_builder();
let mut out = vec![0.0; builder.output_len()];
let mut tasks = builder.partition(&mut out);
tasks.par_iter_mut().for_each(|t| builder.fill(t)); // rayon lives in the caller
```

The disjointness is owned inside integral (safe `chunks_exact_mut` partitioning, no
`unsafe`); integral itself pulls in **no** threading runtime.

## Validation

integral's correctness is established **from physical and mathematical principles**,
not by requiring users to install reference software:

- **Closed-form analytic checks** — s-Gaussian overlap/kinetic/nuclear/dipole,
  normalization, the Boys function vs `erf`/incomplete-gamma and its recurrence.
- **Exact symmetry & invariance laws** — Hermiticity of one-electron matrices,
  operator-character (anti)symmetry, ERI 8-fold permutational symmetry,
  translational invariance of gradients, gauge/origin operator identities.
- **Three independent in-repo algorithms agreeing** — the OS/HGP and Rys ERI
  engines are cross-checked against each other *and* against an independent
  **McMurchie–Davidson** implementation, to the tight `1e-11` tolerance, across
  the full angular-momentum range.
- **Finite-difference derivative consistency** and numeric invariants (Rys roots
  ∈ (0,1), weights > 0, …).
Every check above runs in plain `cargo test` on Windows, macOS, and Linux with
**no external library**.

## Module layout

`integral` is a single crate, layered internally as:

| Module | Layer | Role |
|---|---|---|
| [`integral::math`](https://docs.rs/integral/latest/integral/math/) | L0 | Boys function, Rys roots/weights, normalization, `c2s` coefficients |
| [`integral::engine`](https://docs.rs/integral/latest/integral/engine/) | L1/L2 | OS one-electron engine, OS/HGP + Rys ERI engines, operator DSL, derivatives |
| [`integral`](https://docs.rs/integral/latest/integral/) | L3 | basis/molecule description, drivers, screening, transforms — the public API |

The `extern "C"` ABI lives in the separate `integral-sys` crate (consumed as a
compiled C artifact, not published to crates.io).

## Safety & MSRV

- `#![forbid(unsafe_code)]` in every crate except `integral-sys` (the FFI surface),
  which uses `#![deny(unsafe_code)]` so each `unsafe` must be an explicit,
  reviewed opt-in with a `// SAFETY:` note.
- Stable Rust only. The crates target a low MSRV (**1.82**) for broad downstream
  compatibility, and are developed on recent stable Rust.

## License

Licensed under Apache-2.0 or MIT.
