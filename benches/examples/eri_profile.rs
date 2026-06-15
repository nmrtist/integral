//! Standalone ERI profiling and benchmark harness.
//!
//! Builds faithful cc-pVDZ bases (water = 24 spherical bf, ethylene = 48) and a
//! per-angular-momentum-class microbenchmark, then reports median wall-clock over
//! N reps. Single-thread by design — it measures the per-quartet kernel speed
//! that dominates a dense ERI build.
//!
//! Usage:
//!   cargo run --release -p integral-benches --example eri_profile -- water [reps]
//!   cargo run --release -p integral-benches --example eri_profile -- ethylene [reps]
//!   cargo run --release -p integral-benches --example eri_profile -- micro [reps]
//!   cargo run --release -p integral-benches --example eri_profile -- classes [water|ethylene] [passes]
//!   cargo run --release -p integral-benches --example eri_profile -- fp | boys | prof [secs]

use std::time::Instant;

use integral::math::boys::boys_array;
use integral::{Basis, Engine, Shell};

/// Counting allocator: verifies the hot kernel loop is allocation-free after
/// warm-up. Counts only; delegates to the system allocator. The lib crates
/// stay `#![forbid(unsafe_code)]`; this instrumentation lives in a separate
/// compilation root.
mod alloc_count {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicU64, Ordering};

    static ALLOCS: AtomicU64 = AtomicU64::new(0);

    pub(crate) struct Counting;

    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            unsafe { System.alloc(l) }
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            unsafe { System.dealloc(p, l) }
        }
    }

    pub(crate) fn count() -> u64 {
        ALLOCS.load(Ordering::Relaxed)
    }
}

#[global_allocator]
static ALLOC: alloc_count::Counting = alloc_count::Counting;

/// Build a shell as Cartesian or spherical depending on `sph`.
fn mk(sph: bool, l: usize, c: [f64; 3], e: Vec<f64>, co: Vec<f64>) -> Shell {
    if sph {
        Shell::new_spherical(l, c, e, co).unwrap()
    } else {
        Shell::new(l, c, e, co).unwrap()
    }
}

/// cc-pVDZ hydrogen: [2s1p].
fn ccpvdz_h(center: [f64; 3], sph: bool) -> Vec<Shell> {
    vec![
        mk(
            sph,
            0,
            center,
            vec![13.01, 1.962, 0.4446],
            vec![0.019685, 0.137977, 0.478148],
        ),
        mk(sph, 0, center, vec![0.122], vec![1.0]),
        mk(sph, 1, center, vec![0.727], vec![1.0]),
    ]
}

/// cc-pVDZ first-row atom [3s2p1d] with the general-contraction s/p split into
/// separate shells (how a single-function `Shell` API represents it).
#[allow(clippy::too_many_arguments)]
fn ccpvdz_firstrow(
    center: [f64; 3],
    sph: bool,
    s_exps: Vec<f64>,
    s_c1: Vec<f64>,
    s_c2: Vec<f64>,
    p_exps: Vec<f64>,
    p_c1: Vec<f64>,
    d_exp: f64,
) -> Vec<Shell> {
    let s_diffuse = *s_exps.last().unwrap();
    let p_diffuse = *p_exps.last().unwrap();
    vec![
        mk(sph, 0, center, s_exps.clone(), s_c1),
        mk(sph, 0, center, s_exps, s_c2),
        mk(sph, 0, center, vec![s_diffuse], vec![1.0]),
        mk(sph, 1, center, p_exps.clone(), p_c1),
        mk(sph, 1, center, vec![p_diffuse], vec![1.0]),
        mk(sph, 2, center, vec![d_exp], vec![1.0]),
    ]
}

fn ccpvdz_o(center: [f64; 3], sph: bool) -> Vec<Shell> {
    ccpvdz_firstrow(
        center,
        sph,
        vec![
            11720.0, 1759.0, 400.8, 113.7, 37.03, 13.27, 5.025, 1.013, 0.3023,
        ],
        vec![
            0.00071, 0.00547, 0.027837, 0.10480, 0.283062, 0.448719, 0.270952, 0.015458, -0.002585,
        ],
        vec![
            -0.00016, -0.001263, -0.006267, -0.025716, -0.070924, -0.165411, -0.116955, 0.557368,
            0.572759,
        ],
        vec![17.70, 3.854, 1.046, 0.2753],
        vec![0.043018, 0.228913, 0.508728, 0.460531],
        1.185,
    )
}

fn ccpvdz_c(center: [f64; 3], sph: bool) -> Vec<Shell> {
    ccpvdz_firstrow(
        center,
        sph,
        vec![
            6665.0, 1000.0, 228.0, 64.71, 21.06, 7.495, 2.797, 0.5215, 0.1596,
        ],
        vec![
            0.000692, 0.005329, 0.027077, 0.101718, 0.27474, 0.448564, 0.285074, 0.015204,
            -0.003191,
        ],
        vec![
            -0.000146, -0.001154, -0.005725, -0.023312, -0.063955, -0.149981, -0.127262, 0.544529,
            0.580496,
        ],
        vec![9.439, 2.002, 0.5456, 0.1517],
        vec![0.038109, 0.20948, 0.508557, 0.468842],
        0.55,
    )
}

/// Water at a realistic geometry (bohr). 24 spherical / 25 Cartesian bf.
fn water(sph: bool) -> Basis {
    let mut shells = Vec::new();
    shells.extend(ccpvdz_o([0.0, 0.0, 0.0], sph));
    shells.extend(ccpvdz_h([0.0, 1.4305, 1.1075], sph));
    shells.extend(ccpvdz_h([0.0, -1.4305, 1.1075], sph));
    Basis::new(shells)
}

/// Ethylene C2H4, planar (bohr). 48 spherical bf.
fn ethylene(sph: bool) -> Basis {
    let mut shells = Vec::new();
    shells.extend(ccpvdz_c([0.0, 0.0, 1.26], sph));
    shells.extend(ccpvdz_c([0.0, 0.0, -1.26], sph));
    shells.extend(ccpvdz_h([0.0, 1.74, 2.33], sph));
    shells.extend(ccpvdz_h([0.0, -1.74, 2.33], sph));
    shells.extend(ccpvdz_h([0.0, 1.74, -2.33], sph));
    shells.extend(ccpvdz_h([0.0, -1.74, -2.33], sph));
    Basis::new(shells)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Time `f` over `reps` reps; return (median_s, min_s). Uses a checksum to defeat
/// dead-code elimination.
fn time_it<F: FnMut() -> f64>(reps: usize, mut f: F) -> (f64, f64, f64) {
    let mut times = Vec::with_capacity(reps);
    let mut checksum = 0.0;
    for _ in 0..reps {
        let t0 = Instant::now();
        checksum += f();
        times.push(t0.elapsed().as_secs_f64());
    }
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    (median(times), min, checksum)
}

fn run_basis(name: &str, basis: Basis, reps: usize) {
    println!(
        "\n== {name}: {} shells, {} bf (nao_cart={}) ==",
        basis.shells().len(),
        basis.nao(),
        basis.nao_cart()
    );
    for (eng_name, eng) in [
        ("Auto", Engine::Auto),
        ("OsHgp", Engine::OsHgp),
        ("Rys", Engine::Rys),
    ] {
        let (med, min, sum) = time_it(reps, || {
            let t = basis.eri_with(eng);
            t.iter().take(1).sum::<f64>()
        });
        println!("  eri_with({eng_name:5})  median={med:8.4}s  min={min:8.4}s  (chk {sum:.3e})");
    }
}

/// Per-L-class microbench: 4 single-L shells on 4 centers, contraction depth k
/// per shell. Times the (0,1,2,3) quartet under Auto.
fn micro(reps: usize) {
    println!("\n== per-(L, K) quartet microbench (median over {reps} reps) ==");
    let centers = [
        [0.0, 0.0, 0.0],
        [0.7, -0.3, 0.2],
        [-0.4, 0.6, -0.1],
        [0.2, 0.5, 0.8],
    ];
    let names = ["s", "p", "d", "f", "g"];
    println!(
        "  {:<10} {:>12} {:>12} {:>12}",
        "class", "Auto(us)", "OsHgp(us)", "Rys(us)"
    );
    for (l, &lname) in names.iter().enumerate().take(4) {
        for k in [1usize, 3, 8] {
            let exps: Vec<f64> = (0..k).map(|i| 3.0 * 0.45f64.powi(i as i32)).collect();
            let coeffs: Vec<f64> = (0..k).map(|i| 0.5 + 0.1 * i as f64).collect();
            let basis = Basis::new(
                centers
                    .iter()
                    .map(|&c| Shell::new(l, c, exps.clone(), coeffs.clone()).unwrap())
                    .collect(),
            );
            let cls = format!("({0}{0}|{0}{0})k{k}", lname);
            let mut us = [0.0; 3];
            let mut chks = [0.0f64; 3];
            for (idx, eng) in [Engine::Auto, Engine::OsHgp, Engine::Rys]
                .into_iter()
                .enumerate()
            {
                // Auto-calibrate inner reps to ~15 ms/sample so no class blows up.
                let t0 = Instant::now();
                let one = basis.eri_block_with(eng, 0, 1, 2, 3)[0];
                let single = t0.elapsed().as_secs_f64().max(1e-9);
                let inner = ((0.015 / single) as usize).clamp(1, 5000);
                let (med, _min, _chk) = time_it(reps, || {
                    let mut s = 0.0;
                    for _ in 0..inner {
                        s += basis.eri_block_with(eng, 0, 1, 2, 3)[0];
                    }
                    s
                });
                us[idx] = med / inner as f64 * 1e6;
                chks[idx] = one;
            }
            // Cross-engine sanity: OS/HGP vs Rys must agree on element 0.
            let div = (chks[1] - chks[2]).abs() / chks[2].abs().max(1e-300);
            let flag = if div > 1e-9 { " <-- DIVERGENCE" } else { "" };
            println!(
                "  {cls:<10} {:>12.3} {:>12.3} {:>12.3}{flag}",
                us[0], us[1], us[2]
            );
        }
    }
}

/// Order-independent bitwise fingerprint of the full ERI tensor (XOR-fold of every
/// element's `to_bits`), for proving a kernel change is bit-identical.
fn fingerprint(basis: &Basis, eng: Engine) -> u64 {
    basis
        .eri_with(eng)
        .iter()
        .fold(0u64, |h, &x| h ^ x.to_bits().rotate_left((h & 63) as u32))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("water");
    let reps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);

    match mode {
        "prof" => {
            // Long-running single-engine loop so a sampling profiler has time to work.
            let basis = water(true);
            let secs: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8.0);
            let t0 = Instant::now();
            let mut n = 0u64;
            let mut chk = 0.0;
            while t0.elapsed().as_secs_f64() < secs {
                chk += basis.eri_with(Engine::OsHgp)[0];
                n += 1;
            }
            println!(
                "ran {n} water/OsHgp builds in {:.2}s (chk {chk:.3e})",
                t0.elapsed().as_secs_f64()
            );
        }
        "boys" => {
            // Cost of boys_array per call across the regimes the ERI kernel hits:
            // m=0 (ss), m=4 (pp..dd), m=12 (ff), small/mid/large T, series vs asymptotic.
            println!("== boys_array ns/call (median over {reps} reps) ==");
            let inner = 200_000usize;
            let mut buf = [0.0f64; 32];
            for m in [0usize, 2, 4, 8, 12] {
                let mut line = format!("  m={m:<2}");
                for t in [0.3_f64, 5.0, 25.0, 80.0] {
                    let (med, _min, _chk) = time_it(reps, || {
                        let mut s = 0.0;
                        for i in 0..inner {
                            // vary t a hair so the optimizer can't hoist the call
                            boys_array(m, t + (i & 1) as f64 * 1e-9, &mut buf[..=m]);
                            s += buf[0];
                        }
                        s
                    });
                    line.push_str(&format!("  T={t:<5}:{:7.1}ns", med / inner as f64 * 1e9));
                }
                println!("{line}");
            }
        }
        "fp" => {
            for (name, b) in [("water", water(true)), ("ethylene", ethylene(true))] {
                for (en, e) in [
                    ("Auto", Engine::Auto),
                    ("OsHgp", Engine::OsHgp),
                    ("Rys", Engine::Rys),
                ] {
                    println!("fp {name:10} {en:6} = {:016x}", fingerprint(&b, e));
                }
            }
        }
        "classes" => {
            // Per-quartet-class kernel time attribution: replicate the dense OsHgp
            // loop (cached effective coefficients, one explicit scratch, Cartesian
            // shells so no c2s) and bucket wall time by the (la lb|lc ld) class and
            // contraction degree. Timer overhead (~2×Instant per quartet) is shared
            // evenly, so it cannot promote a µs-scale class.
            use std::collections::HashMap;

            use integral::engine::os_eri::{coulomb_shell_into_scratch, EriScratch, ShellRef};

            let which = args.get(2).map(String::as_str).unwrap_or("water");
            let passes: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
            let basis = if which == "ethylene" {
                ethylene(false)
            } else {
                water(false)
            };
            let shells = basis.shells();
            let eff: Vec<Vec<f64>> = shells
                .iter()
                .map(|s| (0..s.n_prim()).map(|i| s.primitive_coeff(i)).collect())
                .collect();
            let sref = |i: usize| ShellRef {
                center: shells[i].center(),
                l: shells[i].l(),
                exps: shells[i].exponents(),
                coeffs: &eff[i],
            };
            let names = ["s", "p", "d", "f", "g"];
            let mut scratch = EriScratch::new();
            let maxn = shells.iter().map(|s| s.n_cart()).max().unwrap();
            let mut out = vec![0.0; maxn * maxn * maxn * maxn];
            // key: (la, lb, lc, ld, n_prim_product) -> (count, seconds)
            type Buckets = HashMap<(usize, usize, usize, usize, usize), (u64, f64)>;
            let mut buckets: Buckets = HashMap::new();
            let t_all = Instant::now();
            let mut chk = 0.0;
            let mut allocs_warm = 0u64;
            for pass in 0..passes {
                if pass == 1 {
                    allocs_warm = alloc_count::count();
                }
                for si in 0..shells.len() {
                    for sj in 0..shells.len() {
                        for sk in 0..shells.len() {
                            for sl in 0..shells.len() {
                                let (sa, sb, sc, sd) =
                                    (&shells[si], &shells[sj], &shells[sk], &shells[sl]);
                                let n = sa.n_cart() * sb.n_cart() * sc.n_cart() * sd.n_cart();
                                let np = sa.n_prim() * sb.n_prim() * sc.n_prim() * sd.n_prim();
                                out[..n].fill(0.0);
                                let t0 = Instant::now();
                                coulomb_shell_into_scratch(
                                    &mut scratch,
                                    sref(si),
                                    sref(sj),
                                    sref(sk),
                                    sref(sl),
                                    &mut out[..n],
                                );
                                let dt = t0.elapsed().as_secs_f64();
                                chk += out[0];
                                let e = buckets
                                    .entry((sa.l(), sb.l(), sc.l(), sd.l(), np))
                                    .or_insert((0, 0.0));
                                e.0 += 1;
                                e.1 += dt;
                            }
                        }
                    }
                }
            }
            let total = t_all.elapsed().as_secs_f64();
            let mut v: Vec<_> = buckets.into_iter().collect();
            v.sort_by(|a, b| b.1 .1.partial_cmp(&a.1 .1).unwrap());
            let sum: f64 = v.iter().map(|x| x.1 .1).sum();
            let allocs_after_warm = alloc_count::count() - allocs_warm;
            println!(
                "== {which} per-class OsHgp kernel attribution: {passes} passes, loop {total:.4}s, bucketed {sum:.4}s (chk {chk:.3e}) =="
            );
            println!("  heap allocations in passes 2..{passes} (post-warmup): {allocs_after_warm}");
            // Aggregate by l_total: share of kernel time, primitive-quartet count,
            // and ns per primitive quartet (is the cost per-primitive-bound?).
            let mut by_lt: Vec<(f64, u64)> = vec![(0.0, 0); 4 * 4 + 1];
            for ((la, lb, lc, ld, np), (cnt, t)) in &v {
                let lt = la + lb + lc + ld;
                by_lt[lt].0 += t;
                by_lt[lt].1 += *np as u64 * cnt;
            }
            for (lt, (t, npq)) in by_lt.iter().enumerate() {
                if *npq > 0 {
                    println!(
                        "  l_total={lt}: {:8.2} ms {:5.1}%  {:>12} prim-quartets  {:7.1} ns/pq",
                        t * 1e3 / passes as f64,
                        t / sum * 100.0,
                        npq / passes as u64,
                        t / *npq as f64 * 1e9
                    );
                }
            }
            // Aggregate by the VRR shape (ne, nf) = (la+lb, lc+ld) — what a
            // monomorphized per-class kernel would dispatch on.
            let mut by_ef: HashMap<(usize, usize), f64> = HashMap::new();
            for ((la, lb, lc, ld, _np), (_cnt, t)) in &v {
                *by_ef.entry((la + lb, lc + ld)).or_insert(0.0) += t;
            }
            let mut ef: Vec<_> = by_ef.into_iter().collect();
            ef.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let mut cum = 0.0;
            for ((ne, nf), t) in &ef {
                cum += t / sum * 100.0;
                println!(
                    "  (ne,nf)=({ne},{nf}): {:8.2} ms {:5.1}%  cum {:5.1}%",
                    t * 1e3 / passes as f64,
                    t / sum * 100.0,
                    cum
                );
            }
            let mut cum = 0.0;
            for ((la, lb, lc, ld, np), (cnt, t)) in v.iter().take(40) {
                cum += t / sum * 100.0;
                println!(
                    "  ({}{}|{}{}) np={:<4} nq={:<5} {:9.2} ms {:5.1}%  cum {:5.1}%  {:9.3} us/q",
                    names[*la],
                    names[*lb],
                    names[*lc],
                    names[*ld],
                    np,
                    cnt / passes as u64,
                    t * 1e3 / passes as f64,
                    t / sum * 100.0,
                    cum,
                    t / *cnt as f64 * 1e6
                );
            }
        }
        "water" => run_basis("water/cc-pVDZ (spherical)", water(true), reps),
        "waterc" => run_basis("water/cc-pVDZ (Cartesian, kernel-only)", water(false), reps),
        "ethylene" => run_basis("ethylene/cc-pVDZ (spherical)", ethylene(true), reps),
        "ethylenec" => run_basis(
            "ethylene/cc-pVDZ (Cartesian, kernel-only)",
            ethylene(false),
            reps,
        ),
        "micro" => micro(reps),
        "all" => {
            run_basis("water/cc-pVDZ", water(true), reps);
            micro(reps);
        }
        other => {
            eprintln!(
                "unknown mode {other:?}; use water|waterc|ethylene|ethylenec|micro|all|fp|boys"
            )
        }
    }
}
