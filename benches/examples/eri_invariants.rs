//! Cross-library validation: print order-insensitive invariants of the dense
//! spherical ERI tensor — sum, Frobenius norm², `(00|00)`, `Σ(ii|jj)`,
//! `Σ(ij|ij)` — for the same water/ethylene cc-pVDZ systems as `eri_profile`.
//!
//! The invariants are insensitive to per-shell signed permutations of the
//! spherical m components, so they can be compared digit-for-digit against
//! another integrals library evaluating the identical basis (e.g. a libcint
//! harness with coefficients scaled by `CINTgto_norm` and the contraction
//! left unrenormalized — integral's convention), or across integral versions
//! as a value-stability oracle. Agreement to ~13–14 significant digits means
//! the two builds compute the same tensor.
//!
//! Usage:
//!   cargo run --release -p integral-benches --example eri_invariants -- [water|ethylene]

use integral::{Basis, Engine, Shell};

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

fn ccpvdz_o(center: [f64; 3]) -> Vec<Shell> {
    ccpvdz_firstrow(
        center,
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

fn water() -> Basis {
    let mut shells = Vec::new();
    shells.extend(ccpvdz_o([0.0, 0.0, 0.0]));
    shells.extend(ccpvdz_h([0.0, 1.4305, 1.1075]));
    shells.extend(ccpvdz_h([0.0, -1.4305, 1.1075]));
    Basis::new(shells)
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

fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "water".into());
    let basis = if which == "ethylene" {
        ethylene()
    } else {
        water()
    };
    let nao = basis.nao();
    let t = basis.eri_with(Engine::Auto);
    let (mut sum, mut sumsq) = (0.0f64, 0.0f64);
    for &x in &t {
        sum += x;
        sumsq += x * x;
    }
    let (mut sum_iijj, mut sum_ijij) = (0.0f64, 0.0f64);
    for i in 0..nao {
        for j in 0..nao {
            sum_iijj += t[((i * nao + i) * nao + j) * nao + j];
            sum_ijij += t[((i * nao + j) * nao + i) * nao + j];
        }
    }
    println!("integral {which}: nao={nao}");
    println!("  invariants: sum={sum:.15e}");
    println!("              frob2={sumsq:.15e}");
    println!("              (00|00)={:.15e}", t[0]);
    println!("              sum_iijj={sum_iijj:.15e}");
    println!("              sum_ijij={sum_ijij:.15e}");
}
