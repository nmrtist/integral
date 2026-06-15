//! Phase-1 benchmarks: the Boys function and the one-electron integral matrix
//! builders, on representative shells.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use integral::math::boys::boys_array;
use integral::{Basis, Shell};

fn bench_boys(c: &mut Criterion) {
    let mut group = c.benchmark_group("boys");
    let mut out = [0.0f64; 9];
    for &t in &[0.05_f64, 5.0, 40.0] {
        group.bench_with_input(BenchmarkId::new("m=8", t), &t, |b, &t| {
            b.iter(|| {
                boys_array(8, black_box(t), &mut out);
                black_box(out[0])
            });
        });
    }
    group.finish();
}

/// A pair of contracted shells of equal angular momentum on two centers,
/// giving a representative shell-pair workload for each class.
fn pair_basis(l: usize) -> Basis {
    Basis::new(vec![
        Shell::new(l, [0.0, 0.0, 0.0], vec![3.2, 0.8, 0.2], vec![0.2, 0.5, 0.4]).unwrap(),
        Shell::new(l, [0.0, 0.0, 1.5], vec![2.1, 0.6], vec![0.4, 0.7]).unwrap(),
    ])
}

fn bench_matrices(c: &mut Criterion) {
    let charges = [([0.0, 0.0, 0.0], 8.0), ([0.0, 0.0, 1.5], 1.0)];
    for (name, l) in [("s", 0usize), ("p", 1), ("d", 2), ("f", 3)] {
        let basis = pair_basis(l);
        let mut group = c.benchmark_group(format!("1e/{name}{name}"));
        group.bench_function("overlap", |b| b.iter(|| black_box(basis.overlap())));
        group.bench_function("kinetic", |b| b.iter(|| black_box(basis.kinetic())));
        group.bench_function("nuclear", |b| b.iter(|| black_box(basis.nuclear(&charges))));
        group.finish();
    }
}

criterion_group!(benches, bench_boys, bench_matrices);
criterion_main!(benches);
