//! B3 — reusable `EriScratch` arena: memory-correctness.
//!
//! The Phase-5b arena reuses its buffers across shell quartets, so the new failure
//! mode is **state leaking between quartets**. These tests assert directly that the
//! result is independent of evaluation order and of arena reuse, that a threaded run
//! (per-thread arenas) equals the serial one bit-for-bit, and that the m-marching
//! window has actually eliminated the ~41 MB resident VRR table.
//!
//! All comparisons are **bitwise** (`to_bits`), the right bar for "reuse changed
//! nothing": a tolerance check could hide a stale-buffer contribution.

use std::thread;

use integral::engine::os::Vec3;
use integral::engine::os_eri::{
    coulomb_shell_into, coulomb_shell_into_scratch, EriScratch, ShellRef,
};
use integral::math::am::n_cart;

#[derive(Clone)]
struct ShellSpec {
    l: usize,
    center: Vec3,
    exps: Vec<f64>,
    coeffs: Vec<f64>,
}

impl ShellSpec {
    fn s(l: usize, center: Vec3, exp: f64) -> Self {
        ShellSpec {
            l,
            center,
            exps: vec![exp],
            coeffs: vec![1.0],
        }
    }
    fn as_ref(&self) -> ShellRef<'_> {
        ShellRef {
            center: self.center,
            l: self.l,
            exps: &self.exps,
            coeffs: &self.coeffs,
        }
    }
}

type Quartet = [ShellSpec; 4];

fn block_len(q: &Quartet) -> usize {
    n_cart(q[0].l) * n_cart(q[1].l) * n_cart(q[2].l) * n_cart(q[3].l)
}

/// Evaluate one quartet through a given (possibly already-used) arena.
fn eval(scratch: &mut EriScratch, q: &Quartet) -> Vec<f64> {
    let mut out = vec![0.0; block_len(q)];
    coulomb_shell_into_scratch(
        scratch,
        q[0].as_ref(),
        q[1].as_ref(),
        q[2].as_ref(),
        q[3].as_ref(),
        &mut out,
    );
    out
}

fn bits_eq(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// A spread of quartets covering different sizes, L mixes, and contraction, so a
/// stale-buffer leak between dissimilar quartets would show up.
fn corpus() -> Vec<Quartet> {
    let ca = [0.0, 0.0, 0.0];
    let cb = [0.5, -0.3, 0.2];
    let cc = [-0.4, 0.6, -0.1];
    let cd = [0.2, 0.4, 0.8];
    vec![
        [
            ShellSpec::s(0, ca, 0.9),
            ShellSpec::s(0, cb, 1.3),
            ShellSpec::s(0, cc, 0.7),
            ShellSpec::s(0, cd, 1.1),
        ],
        [
            ShellSpec::s(0, ca, 0.9),
            ShellSpec::s(1, cb, 1.3),
            ShellSpec::s(2, cc, 0.7),
            ShellSpec::s(3, cd, 1.1),
        ],
        [
            ShellSpec::s(2, ca, 0.9),
            ShellSpec::s(2, cb, 1.3),
            ShellSpec::s(3, cc, 0.7),
            ShellSpec::s(3, cd, 1.1),
        ],
        [
            ShellSpec::s(3, ca, 0.9),
            ShellSpec::s(3, cb, 1.3),
            ShellSpec::s(3, cc, 0.7),
            ShellSpec::s(3, cd, 1.1),
        ],
        [
            ShellSpec::s(0, ca, 0.9),
            ShellSpec::s(3, cb, 1.3),
            ShellSpec::s(4, cc, 0.7),
            ShellSpec::s(6, cd, 1.1),
        ],
        [
            // contracted (pp|ds)
            ShellSpec {
                l: 1,
                center: ca,
                exps: vec![1.4, 0.45],
                coeffs: vec![0.6, 0.5],
            },
            ShellSpec {
                l: 1,
                center: cb,
                exps: vec![0.9, 0.3],
                coeffs: vec![0.55, 0.5],
            },
            ShellSpec {
                l: 2,
                center: cc,
                exps: vec![1.1, 0.4],
                coeffs: vec![0.7, 0.4],
            },
            ShellSpec::s(0, cd, 0.8),
        ],
    ]
}

/// Evaluating quartet X through a **fresh** arena must equal evaluating it through an
/// arena previously used for every other quartet (any order) — bit for bit. A reused
/// buffer that leaked state would break this.
#[test]
fn arena_reuse_is_order_independent() {
    let corpus = corpus();
    for (i, target) in corpus.iter().enumerate() {
        // Reference: target alone through a brand-new arena.
        let mut fresh = EriScratch::new();
        let reference = eval(&mut fresh, target);

        // Through one arena, run every other quartet first (forward), then target.
        let mut used = EriScratch::new();
        for (j, q) in corpus.iter().enumerate() {
            if j != i {
                let _ = eval(&mut used, q);
            }
        }
        let after_forward = eval(&mut used, target);
        assert!(
            bits_eq(&reference, &after_forward),
            "quartet {i}: result changed after forward arena reuse"
        );

        // And again with the predecessors in reverse order.
        let mut used_rev = EriScratch::new();
        for (j, q) in corpus.iter().enumerate().rev() {
            if j != i {
                let _ = eval(&mut used_rev, q);
            }
        }
        let after_reverse = eval(&mut used_rev, target);
        assert!(
            bits_eq(&reference, &after_reverse),
            "quartet {i}: result changed after reverse arena reuse"
        );
    }
}

/// Re-running the same quartet many times through one arena must give the identical
/// block every time (the `c_ef` accumulator is re-zeroed; nothing accumulates).
#[test]
fn arena_repeated_eval_is_stable() {
    let corpus = corpus();
    let mut s = EriScratch::new();
    for q in &corpus {
        let first = eval(&mut s, q);
        for _ in 0..3 {
            let again = eval(&mut s, q);
            assert!(bits_eq(&first, &again), "repeated eval drifted");
        }
    }
}

/// A multi-threaded run with **one arena per thread** must reproduce the serial
/// results bit-for-bit. Each thread owns its `EriScratch` (a `&mut` cannot be shared
/// across threads — that is a compile error), so there is no shared mutable arena.
#[test]
fn threaded_per_thread_arena_matches_serial() {
    let corpus = corpus();

    // Serial reference (thread-local arena via the public convenience entry).
    let serial: Vec<Vec<f64>> = corpus
        .iter()
        .map(|q| {
            let mut out = vec![0.0; block_len(q)];
            coulomb_shell_into(
                q[0].as_ref(),
                q[1].as_ref(),
                q[2].as_ref(),
                q[3].as_ref(),
                &mut out,
            );
            out
        })
        .collect();

    // Threaded: 4 worker threads, each with its own arena, each handling a stripe.
    let n_threads = 4;
    let threaded: Vec<Vec<f64>> = thread::scope(|scope| {
        let handles: Vec<_> = (0..n_threads)
            .map(|t| {
                let corpus = &corpus;
                scope.spawn(move || {
                    let mut scratch = EriScratch::new();
                    let mut results = Vec::new();
                    for (idx, q) in corpus.iter().enumerate() {
                        if idx % n_threads == t {
                            results.push((idx, eval(&mut scratch, q)));
                        }
                    }
                    results
                })
            })
            .collect();
        let mut all: Vec<(usize, Vec<f64>)> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();
        all.sort_by_key(|(idx, _)| *idx);
        all.into_iter().map(|(_, v)| v).collect()
    });

    for (i, (a, b)) in serial.iter().zip(&threaded).enumerate() {
        assert!(bits_eq(a, b), "quartet {i}: threaded != serial");
    }
}

/// The m-marching window (B2) plus the arena (B3) must have eliminated the ~41 MB
/// monolithic VRR table. For `(ii|ii)` the old full `[e0|f0]^(m)` table was
/// `n_addr(12)² · 25 = 455² · 25 = 5_175_625` f64 (~41 MB) in a single allocation;
/// assert the largest single buffer is now a small fraction of that and the whole
/// resident working set is far below it.
#[test]
fn iiii_resident_footprint_is_far_below_old_table() {
    const OLD_TABLE_F64: usize = 455 * 455 * 25; // 5_175_625, ~41 MB
    let ii = |center: Vec3, exp: f64| ShellSpec::s(6, center, exp);
    let q: Quartet = [
        ii([0.0, 0.0, 0.0], 0.9),
        ii([0.5, -0.3, 0.2], 1.3),
        ii([-0.4, 0.6, -0.1], 0.7),
        ii([0.2, 0.4, 0.8], 1.1),
    ];
    let mut s = EriScratch::new();
    let _ = eval(&mut s, &q);

    let largest = s.largest_buffer_f64();
    let resident = s.resident_f64();
    eprintln!(
        "(ii|ii) arena: largest single buffer {largest} f64 ({:.2} MB), resident {resident} f64 \
         ({:.2} MB); old monolithic g table was {OLD_TABLE_F64} f64 (~{:.1} MB).",
        largest as f64 * 8.0 / 1e6,
        resident as f64 * 8.0 / 1e6,
        OLD_TABLE_F64 as f64 * 8.0 / 1e6,
    );
    // Largest single allocation: the 3-level m-marching window, well under half the
    // old table. (Bound: 3·max_k[n_cart(k)·slab_k] ≈ 540_540 f64 at (ii|ii).)
    assert!(
        largest < OLD_TABLE_F64 / 2,
        "largest buffer {largest} f64 not far below the old {OLD_TABLE_F64}-f64 table"
    );
    // Whole resident working set still well under the single old table.
    assert!(
        resident < OLD_TABLE_F64,
        "resident {resident} f64 should be below the old single-table {OLD_TABLE_F64} f64"
    );
}
