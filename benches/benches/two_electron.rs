//! Two-electron benchmarks: Rys roots/weights and the Rys ERI engine on
//! representative shell quartets (one primitive per shell, so each measures a
//! single engine quartet of the given angular-momentum class).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use integral::math::rys::{rys_roots_weights, rys_roots_weights_reference, MAX_RYS_ROOTS};
use integral::{Basis, Engine, Shell};

/// Before/after: the interpolated production [`rys_roots_weights`] vs the
/// discretized-Stieltjes [`rys_roots_weights_reference`], on the same
/// representative `(nroots, T)` points across the low/finite/large-T branches.
/// The ratio of medians is the roots/weights speedup.
fn bench_rys_roots(c: &mut Criterion) {
    let mut group = c.benchmark_group("rys_roots");
    let mut roots = [0.0f64; MAX_RYS_ROOTS];
    let mut wts = [0.0f64; MAX_RYS_ROOTS];
    for &(n, t) in &[
        (1usize, 0.5_f64),
        (3, 5.0),
        (7, 30.0),
        (13, 90.0),
        (13, 500.0),
    ] {
        let label = format!("n{n}_T{t}");
        group.bench_with_input(BenchmarkId::new("interp", &label), &t, |b, &t| {
            b.iter(|| {
                rys_roots_weights(n, black_box(t), &mut roots, &mut wts);
                black_box(roots[0])
            });
        });
        group.bench_with_input(BenchmarkId::new("reference", &label), &t, |b, &t| {
            b.iter(|| {
                rys_roots_weights_reference(n, black_box(t), &mut roots, &mut wts);
                black_box(roots[0])
            });
        });
    }
    group.finish();
}

/// Four single-primitive shells of equal angular momentum on four centers; the
/// (0,1,2,3) block is one engine quartet of that class.
fn quartet_basis(l: usize) -> Basis {
    Basis::new(vec![
        Shell::new(l, [0.0, 0.0, 0.0], vec![1.2], vec![1.0]).unwrap(),
        Shell::new(l, [0.7, -0.3, 0.2], vec![0.9], vec![1.0]).unwrap(),
        Shell::new(l, [-0.4, 0.6, -0.1], vec![1.1], vec![1.0]).unwrap(),
        Shell::new(l, [0.2, 0.5, 0.8], vec![0.7], vec![1.0]).unwrap(),
    ])
}

fn bench_eri_block(c: &mut Criterion) {
    let mut group = c.benchmark_group("eri_block");
    for (name, l) in [("ssss", 0usize), ("pppp", 1), ("dddd", 2), ("ffff", 3)] {
        let basis = quartet_basis(l);
        group.bench_function(name, |b| b.iter(|| black_box(basis.eri_block(0, 1, 2, 3))));
    }
    group.finish();
}

/// Four single-L shells on four centres, each contracted with `k` primitives —
/// `contraction_degree = k⁴` primitive quartets behind the block.
fn quartet_basis_k(l: usize, k: usize) -> Basis {
    let centers = [
        [0.0, 0.0, 0.0],
        [0.7, -0.3, 0.2],
        [-0.4, 0.6, -0.1],
        [0.2, 0.5, 0.8],
    ];
    // A spread of exponents (tight→diffuse) with smooth coefficients.
    let exps: Vec<f64> = (0..k).map(|i| 3.0 * 0.45f64.powi(i as i32)).collect();
    let coeffs: Vec<f64> = (0..k).map(|i| 0.5 + 0.1 * i as f64).collect();
    Basis::new(
        centers
            .iter()
            .map(|&c| Shell::new(l, c, exps.clone(), coeffs.clone()).unwrap())
            .collect(),
    )
}

/// OS/HGP vs Rys across `(total angular momentum, contraction degree)` — the
/// measurement that sets the dispatch crossover. For
/// each `(L, K)` the same `(0,1,2,3)` quartet is timed under both forced engines.
/// `l_total = 4·L`, `contraction_degree = K⁴`.
///
/// The grid pins the crossovers across the contraction range: the `K=2` (deg-16)
/// rung at each L to locate the mid-contraction crossover, the high-L/high-contraction
/// corners (`L3_K6`, `L4_K3`), and `K=1` probes up through `i` shells (`L6_K1`,
/// `l_total 24`) to confirm Rys still wins low-contraction high-L. The very-high-L
/// high-contraction corners (`L5_K3`, `L6_K3`) are omitted — seconds per iteration,
/// and the `K=1`/`K=3` rungs already bracket the decision there.
fn bench_engine_crossover(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine_crossover");
    // (label, l, k): contraction swept at each L; L raised to the i-shell cap.
    let cases = [
        ("L0_K1", 0usize, 1usize),
        ("L0_K3", 0, 3),
        ("L0_K6", 0, 6),
        ("L1_K1", 1, 1),
        ("L1_K2", 1, 2),
        ("L1_K3", 1, 3),
        ("L1_K6", 1, 6),
        ("L2_K1", 2, 1),
        ("L2_K2", 2, 2),
        ("L2_K3", 2, 3),
        ("L2_K6", 2, 6),
        ("L3_K1", 3, 1),
        ("L3_K2", 3, 2),
        ("L3_K3", 3, 3),
        ("L3_K6", 3, 6),
        ("L4_K1", 4, 1),
        ("L4_K2", 4, 2),
        ("L4_K3", 4, 3),
        ("L5_K1", 5, 1),
        ("L6_K1", 6, 1),
    ];
    for (label, l, k) in cases {
        let basis = quartet_basis_k(l, k);
        for (eng_name, engine) in [("rys", Engine::Rys), ("oshgp", Engine::OsHgp)] {
            group.bench_function(BenchmarkId::new(eng_name, label), |b| {
                b.iter(|| black_box(basis.eri_block_with(engine, 0, 1, 2, 3)));
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_rys_roots,
    bench_eri_block,
    bench_engine_crossover
);
criterion_main!(benches);
