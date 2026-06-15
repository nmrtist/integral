//! Untracked scoping tool: lane occupancy of class-bucketed quartet batching.
//! Replays the canonical s8 loop, buckets OS-routed quartets by
//! (ne, nf, n_bra_pairs, n_ket_pairs) and by (ne, nf) alone, and reports how
//! many quartets land in full groups of 4 under flush-when-4 semantics.

use std::collections::HashMap;

use integral::{Basis, Shell};

fn mk(l: usize, c: [f64; 3], e: Vec<f64>, co: Vec<f64>) -> Shell {
    Shell::new_spherical(l, c, e, co).unwrap()
}

fn ccpvdz_h(center: [f64; 3]) -> Vec<Shell> {
    vec![
        mk(
            0,
            center,
            vec![13.01, 1.962, 0.4446],
            vec![0.019685, 0.137977, 0.478148],
        ),
        mk(0, center, vec![0.122], vec![1.0]),
        mk(1, center, vec![0.727], vec![1.0]),
    ]
}

#[allow(clippy::too_many_arguments)]
fn ccpvdz_firstrow(
    center: [f64; 3],
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
        mk(0, center, s_exps.clone(), s_c1),
        mk(0, center, s_exps, s_c2),
        mk(0, center, vec![s_diffuse], vec![1.0]),
        mk(1, center, p_exps.clone(), p_c1),
        mk(1, center, vec![p_diffuse], vec![1.0]),
        mk(2, center, vec![d_exp], vec![1.0]),
    ]
}

fn ccpvdz_c(center: [f64; 3]) -> Vec<Shell> {
    ccpvdz_firstrow(
        center,
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

fn ethylene() -> Basis {
    let mut shells = Vec::new();
    shells.extend(ccpvdz_c([0.0, 0.0, 1.26]));
    shells.extend(ccpvdz_c([0.0, 0.0, -1.26]));
    shells.extend(ccpvdz_h([0.0, 1.74, 2.33]));
    shells.extend(ccpvdz_h([0.0, -1.74, 2.33]));
    shells.extend(ccpvdz_h([0.0, 1.74, -2.33]));
    shells.extend(ccpvdz_h([0.0, -1.74, -2.33]));
    Basis::new(shells)
}

/// Mirror of os_eri::build_pairs' surviving-pair count (PAIR_NEGLIGIBLE screen
/// on kappa * |c1*c2| with effective coefficients).
fn pair_count(s1: &Shell, s2: &Shell) -> usize {
    let d2: f64 = s1
        .center()
        .iter()
        .zip(s2.center())
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    let mut n = 0;
    for i in 0..s1.n_prim() {
        for j in 0..s2.n_prim() {
            let (e1, e2) = (s1.exponents()[i], s2.exponents()[j]);
            let kappa = (-(e1 * e2 / (e1 + e2)) * d2).exp();
            if kappa * (s1.primitive_coeff(i) * s2.primitive_coeff(j)).abs() >= 1e-32 {
                n += 1;
            }
        }
    }
    n
}

fn main() {
    let basis = ethylene();
    let shells = basis.shells();
    let nsh = shells.len();
    // pair counts per (i,j)
    let mut pc = vec![0usize; nsh * nsh];
    for i in 0..nsh {
        for j in 0..nsh {
            pc[i * nsh + j] = pair_count(&shells[i], &shells[j]);
        }
    }
    // canonical loop; OS routing per dispatch (lt<=5 deg>=1; 6..=16 deg>=16)
    let mut total = 0usize;
    let mut eligible = 0usize; // OS, lt>0, ne<=2 && nf<=2
    let mut elig3 = 0usize; // same but ne,nf<=3
    let mut buckets: HashMap<(usize, usize, usize, usize), usize> = HashMap::new();
    let mut class_only: HashMap<(usize, usize), usize> = HashMap::new();
    let mut prim_cost_batched = 0usize;
    let mut prim_cost_total = 0usize;
    for si in 0..nsh {
        for sj in 0..=si {
            for sk in 0..=si {
                let l_top = if sk == si { sj } else { sk };
                for sl in 0..=l_top {
                    total += 1;
                    let lt = shells[si].l() + shells[sj].l() + shells[sk].l() + shells[sl].l();
                    let deg = shells[si].n_prim()
                        * shells[sj].n_prim()
                        * shells[sk].n_prim()
                        * shells[sl].n_prim();
                    let os = match lt {
                        0..=5 => true,
                        6..=16 => deg >= 16,
                        _ => false,
                    };
                    if !os {
                        continue;
                    }
                    let ne = shells[si].l() + shells[sj].l();
                    let nf = shells[sk].l() + shells[sl].l();
                    let nb = pc[si * nsh + sj];
                    let nk = pc[sk * nsh + sl];
                    let cost = nb * nk * (ne + 1) * (nf + 1); // crude weight
                    prim_cost_total += cost;
                    if ne <= 3 && nf <= 3 {
                        elig3 += 1;
                    }
                    if ne <= 2 && nf <= 2 {
                        eligible += 1;
                        *buckets.entry((ne, nf, nb, nk)).or_default() += 1;
                        *class_only.entry((ne, nf)).or_default() += 1;
                        prim_cost_batched += cost;
                    }
                }
            }
        }
    }
    println!("canonical quartets: {total}; OS lt>0: cost_total={prim_cost_total}");
    println!("eligible (ne,nf<=2): {eligible}  (ne,nf<=3: {elig3})");
    let full: usize = buckets.values().map(|&n| n - n % 4).sum();
    let tail: usize = buckets.values().map(|&n| n % 4).sum();
    println!(
        "bucket key (ne,nf,nbra,nket): {} buckets, batched {} ({:.1}%), scalar tail {}",
        buckets.len(),
        full,
        100.0 * full as f64 / eligible as f64,
        tail
    );
    let fullc: usize = class_only.values().map(|&n| n - n % 4).sum();
    println!(
        "bucket key (ne,nf) only:      {} buckets, batched {} ({:.1}%)",
        class_only.len(),
        fullc,
        100.0 * fullc as f64 / eligible as f64
    );
    // distribution of pair-count keys per class
    let mut per_class: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (&(ne, nf, _, _), &n) in &buckets {
        per_class.entry((ne, nf)).or_default().push(n);
    }
    let mut keys: Vec<_> = per_class.keys().copied().collect();
    keys.sort_unstable();
    for k in keys {
        let v = &per_class[&k];
        let tot: usize = v.iter().sum();
        let full: usize = v.iter().map(|&n| n - n % 4).sum();
        println!(
            "  class {k:?}: {} pair-count buckets, {} quartets, {:.1}% in full lanes",
            v.len(),
            tot,
            100.0 * full as f64 / tot as f64
        );
    }
    println!(
        "cost share of eligible classes: {:.1}%",
        100.0 * prim_cost_batched as f64 / prim_cost_total as f64
    );
}
