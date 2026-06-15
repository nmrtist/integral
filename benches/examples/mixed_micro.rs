//! Per-engine timing of mixed-L shell quartets in the `l_total` 5–8 band,
//! where the OS/HGP-vs-Rys dispatch crossover sits — complementing
//! `eri_profile`'s same-L `micro` mode, which cannot produce these shapes.
//!
//! `select_engine` keys on `(l_total, contraction_degree)`, so the measured
//! per-shape winners here decide the band thresholds.
//!
//! Usage:
//!   cargo run --release -p integral-benches --example mixed_micro -- [reps]

use std::time::Instant;

use integral::{Basis, Engine, Shell};

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let reps: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);
    let centers = [
        [0.0, 0.0, 0.0],
        [0.7, -0.3, 0.2],
        [-0.4, 0.6, -0.1],
        [0.2, 0.5, 0.8],
    ];
    // (la,lb,lc,ld) quartets in the lt 5..=8 band, deg 1 and 3 per shell.
    let cases: [(usize, usize, usize, usize); 6] = [
        (2, 1, 1, 1), // lt5, (ne,nf)=(3,2)
        (2, 1, 2, 1), // lt6, (3,3)
        (2, 2, 1, 1), // lt6, (4,2) — uncovered bra
        (2, 2, 2, 1), // lt7, (4,3)
        (2, 2, 2, 2), // lt8, (4,4)
        (2, 0, 2, 1), // lt5, (2,3)
    ];
    println!(
        "{:<14} {:>5} {:>12} {:>12}",
        "class", "K", "OsHgp(us)", "Rys(us)"
    );
    for (la, lb, lc, ld) in cases {
        for k in [1usize, 3] {
            let exps: Vec<f64> = (0..k).map(|i| 3.0 * 0.45f64.powi(i as i32)).collect();
            let coeffs: Vec<f64> = (0..k).map(|i| 0.5 + 0.1 * i as f64).collect();
            let ls = [la, lb, lc, ld];
            let basis = Basis::new(
                (0..4)
                    .map(|s| Shell::new(ls[s], centers[s], exps.clone(), coeffs.clone()).unwrap())
                    .collect(),
            );
            let mut us = [0.0f64; 2];
            for (idx, eng) in [Engine::OsHgp, Engine::Rys].into_iter().enumerate() {
                let t0 = Instant::now();
                let _ = basis.eri_block_with(eng, 0, 1, 2, 3);
                let single = t0.elapsed().as_secs_f64().max(1e-9);
                let inner = ((0.01 / single) as usize).clamp(1, 5000);
                let times: Vec<f64> = (0..reps)
                    .map(|_| {
                        let t0 = Instant::now();
                        let mut s = 0.0;
                        for _ in 0..inner {
                            s += basis.eri_block_with(eng, 0, 1, 2, 3)[0];
                        }
                        std::hint::black_box(s);
                        t0.elapsed().as_secs_f64() / inner as f64
                    })
                    .collect();
                us[idx] = median(times) * 1e6;
            }
            println!(
                "({}{}|{}{})k{k}     {:>12.3} {:>12.3}   {}",
                ["s", "p", "d", "f"][la],
                ["s", "p", "d", "f"][lb],
                ["s", "p", "d", "f"][lc],
                ["s", "p", "d", "f"][ld],
                us[0],
                us[1],
                if us[0] < us[1] { "OS" } else { "Rys" }
            );
        }
    }
}
