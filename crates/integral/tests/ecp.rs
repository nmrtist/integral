//! Validation of the semilocal ECP integrals (`Basis::ecp`) against an
//! independent brute-force numerical-quadrature reference, plus exact
//! symmetry/invariance laws.
//!
//! The reference evaluates `⟨μ|U|ν⟩` directly on a radial × spherical product
//! grid (composite Gauss–Legendre in `r`, Gauss–Legendre in `cosθ` × uniform
//! `φ`), including the projector terms through explicit `⟨μ|S_lm⟩(r)`
//! projections with a **hard-coded closed-form table** of real spherical
//! harmonics. It shares no code with the analytic path beyond basis-function
//! conventions (component ordering, primitive normalization).

use integral::math::am::cart_components;
use integral::{Basis, Ecp, EcpPrimitive, Shell};

const PI: f64 = std::f64::consts::PI;

fn epr(n: i32, zeta: f64, coef: f64) -> EcpPrimitive {
    EcpPrimitive { n, zeta, coef }
}

// ---------------------------------------------------------------------------
// Brute-force quadrature reference
// ---------------------------------------------------------------------------

/// Gauss–Legendre nodes/weights on [−1, 1] (test-local Newton iteration).
fn gl(n: usize) -> Vec<(f64, f64)> {
    let mut out = Vec::with_capacity(n);
    for i in 1..=n {
        let mut x = (PI * (i as f64 - 0.25) / (n as f64 + 0.5)).cos();
        let mut dp = 1.0;
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0, x);
            for k in 2..=n {
                let kf = k as f64;
                let p2 = ((2.0 * kf - 1.0) * x * p1 - (kf - 1.0) * p0) / kf;
                p0 = p1;
                p1 = p2;
            }
            dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
            let dx = p1 / dp;
            x -= dx;
            if dx.abs() < 5e-16 {
                break;
            }
        }
        out.push((x, 2.0 / ((1.0 - x * x) * dp * dp)));
    }
    out
}

/// Closed-form real spherical harmonics on the unit sphere, `l ≤ 4`
/// (standard table; any orthonormal convention gives the same projector).
fn ref_rsh(l: usize, m: i32, u: [f64; 3]) -> f64 {
    let (x, y, z) = (u[0], u[1], u[2]);
    match (l, m) {
        (0, 0) => 0.5 * (1.0 / PI).sqrt(),
        (1, -1) => (3.0 / (4.0 * PI)).sqrt() * y,
        (1, 0) => (3.0 / (4.0 * PI)).sqrt() * z,
        (1, 1) => (3.0 / (4.0 * PI)).sqrt() * x,
        (2, -2) => 0.5 * (15.0 / PI).sqrt() * x * y,
        (2, -1) => 0.5 * (15.0 / PI).sqrt() * y * z,
        (2, 0) => 0.25 * (5.0 / PI).sqrt() * (3.0 * z * z - 1.0),
        (2, 1) => 0.5 * (15.0 / PI).sqrt() * x * z,
        (2, 2) => 0.25 * (15.0 / PI).sqrt() * (x * x - y * y),
        (3, -3) => 0.25 * (35.0 / (2.0 * PI)).sqrt() * y * (3.0 * x * x - y * y),
        (3, -2) => 0.5 * (105.0 / PI).sqrt() * x * y * z,
        (3, -1) => 0.25 * (21.0 / (2.0 * PI)).sqrt() * y * (5.0 * z * z - 1.0),
        (3, 0) => 0.25 * (7.0 / PI).sqrt() * (5.0 * z * z * z - 3.0 * z),
        (3, 1) => 0.25 * (21.0 / (2.0 * PI)).sqrt() * x * (5.0 * z * z - 1.0),
        (3, 2) => 0.25 * (105.0 / PI).sqrt() * z * (x * x - y * y),
        (3, 3) => 0.25 * (35.0 / (2.0 * PI)).sqrt() * x * (x * x - 3.0 * y * y),
        (4, -4) => 0.75 * (35.0 / PI).sqrt() * x * y * (x * x - y * y),
        (4, -3) => 0.75 * (35.0 / (2.0 * PI)).sqrt() * y * z * (3.0 * x * x - y * y),
        (4, -2) => 0.75 * (5.0 / PI).sqrt() * x * y * (7.0 * z * z - 1.0),
        (4, -1) => 0.75 * (5.0 / (2.0 * PI)).sqrt() * y * z * (7.0 * z * z - 3.0),
        (4, 0) => 3.0 / 16.0 * (1.0 / PI).sqrt() * (35.0 * z.powi(4) - 30.0 * z * z + 3.0),
        (4, 1) => 0.75 * (5.0 / (2.0 * PI)).sqrt() * x * z * (7.0 * z * z - 3.0),
        (4, 2) => 3.0 / 8.0 * (5.0 / PI).sqrt() * (x * x - y * y) * (7.0 * z * z - 1.0),
        (4, 3) => 0.75 * (35.0 / (2.0 * PI)).sqrt() * x * z * (x * x - 3.0 * y * y),
        (4, 4) => 3.0 / 16.0 * (35.0 / PI).sqrt() * (x.powi(4) - 6.0 * x * x * y * y + y.powi(4)),
        _ => unreachable!("reference harmonics only tabulated to l = 4"),
    }
}

/// Evaluate all (Cartesian) AOs of `basis` at the point `p`.
fn ao_values(basis: &Basis, p: [f64; 3], out: &mut [f64]) {
    let mut idx = 0;
    for s in basis.shells() {
        let c = s.center();
        let d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
        let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        let rad: f64 = (0..s.n_prim())
            .map(|i| s.primitive_coeff(i) * (-s.exponents()[i] * r2).exp())
            .sum();
        for comp in cart_components(s.l()) {
            out[idx] = d[0].powi(comp[0] as i32)
                * d[1].powi(comp[1] as i32)
                * d[2].powi(comp[2] as i32)
                * rad;
            idx += 1;
        }
    }
    assert_eq!(idx, basis.nao());
}

/// `U(r) = Σ_k d_k r^{n_k − 2} e^{−ζ_k r²}` for one channel's expansion.
fn eval_radial(prims: &[EcpPrimitive], r: f64) -> f64 {
    prims
        .iter()
        .map(|p| p.coef * r.powi(p.n - 2) * (-p.zeta * r * r).exp())
        .sum()
}

/// Brute-force `⟨μ|Σ_A U_A|ν⟩` on a radial × angular product grid: composite
/// 32-point Gauss–Legendre on geometric radial panels up to 25.6 bohr,
/// `nth`-point Gauss–Legendre in `cosθ`, `nph` uniform points in `φ`.
fn ref_ecp(basis: &Basis, ecps: &[Ecp], nth: usize, nph: usize) -> Vec<f64> {
    let nao = basis.nao();
    assert_eq!(nao, basis.nao_cart(), "reference handles Cartesian bases");
    let atoms = basis.atoms();
    let bounds = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6];
    let g32 = gl(32);
    let mut radial = Vec::new();
    for win in bounds.windows(2) {
        let (c1, c2) = (0.5 * (win[1] - win[0]), 0.5 * (win[1] + win[0]));
        for &(x, w) in &g32 {
            radial.push((c1 * x + c2, c1 * w));
        }
    }
    let gth = gl(nth);
    let mut dirs = Vec::new();
    for &(ct, wt) in &gth {
        let st = (1.0 - ct * ct).max(0.0).sqrt();
        for jp in 0..nph {
            let phi = 2.0 * PI * jp as f64 / nph as f64;
            dirs.push((
                [st * phi.cos(), st * phi.sin(), ct],
                wt * 2.0 * PI / nph as f64,
            ));
        }
    }
    let mut mat = vec![0.0; nao * nao];
    let mut phi = vec![0.0; nao];
    for ecp in ecps {
        let c = atoms[ecp.atom];
        let nlm: usize = (0..ecp.semilocal.len()).map(|l| 2 * l + 1).sum();
        let mut proj = vec![0.0; nlm.max(1) * nao];
        for &(r, wr) in &radial {
            let ul = eval_radial(&ecp.local, r);
            let uls: Vec<f64> = ecp.semilocal.iter().map(|p| eval_radial(p, r)).collect();
            if ul.abs() < 1e-40 && uls.iter().all(|u| u.abs() < 1e-40) {
                continue;
            }
            proj.iter_mut().for_each(|v| *v = 0.0);
            let w1 = wr * r * r * ul;
            for &(u, wang) in &dirs {
                let p = [c[0] + r * u[0], c[1] + r * u[1], c[2] + r * u[2]];
                ao_values(basis, p, &mut phi);
                if w1 != 0.0 {
                    let s = w1 * wang;
                    for i in 0..nao {
                        let fi = s * phi[i];
                        if fi != 0.0 {
                            for j in 0..=i {
                                mat[i * nao + j] += fi * phi[j];
                            }
                        }
                    }
                }
                let mut off = 0;
                for (l, ul_chan) in uls.iter().enumerate() {
                    if *ul_chan == 0.0 {
                        off += 2 * l + 1;
                        continue;
                    }
                    for m in -(l as i32)..=l as i32 {
                        let sv = ref_rsh(l, m, u) * wang;
                        for i in 0..nao {
                            proj[off * nao + i] += sv * phi[i];
                        }
                        off += 1;
                    }
                }
            }
            let mut off = 0;
            for (l, ul_chan) in uls.iter().enumerate() {
                let wl = wr * r * r * ul_chan;
                for _m in 0..(2 * l + 1) {
                    if wl != 0.0 {
                        let pr = &proj[off * nao..(off + 1) * nao];
                        for i in 0..nao {
                            for j in 0..=i {
                                mat[i * nao + j] += wl * pr[i] * pr[j];
                            }
                        }
                    }
                    off += 1;
                }
            }
        }
    }
    for i in 0..nao {
        for j in 0..i {
            mat[j * nao + i] = mat[i * nao + j];
        }
    }
    mat
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn assert_no_nan(m: &[f64]) {
    assert!(m.iter().all(|v| v.is_finite()), "matrix contains NaN/Inf");
}

/// Published def2-ECP (MWB28) parameters for Ag: the f channel is the local
/// part `U_L` (L = 3); the s/p/d entries are the tabulated `U_l − U_L`
/// differences (literal in-test fixture; no file parsing).
fn ag_def2_ecp(atom: usize) -> Ecp {
    Ecp {
        atom,
        n_core: 28,
        max_l: 3,
        local: vec![
            epr(2, 13.520_571, -21.097_695),
            epr(2, 6.576_015, -1.374_412),
        ],
        semilocal: vec![
            vec![
                epr(2, 13.130_309, 255.139_465),
                epr(2, 6.510_656, 36.866_806),
            ],
            vec![
                epr(2, 11.314_031, 182.176_810),
                epr(2, 5.691_475, 30.356_869),
            ],
            vec![epr(2, 9.211_745, 73.593_431), epr(2, 4.529_844, 12.702_387)],
        ],
    }
}

/// Published def2-ECP for iodine (I, ECP28MDF): 28-electron core, local L = f,
/// s/p/d projectors. The s/p/d blocks are the `U_l − U_f` differences in the
/// gaussian94 tabulation (each ends with the negated local terms), matching the
/// crate's `semilocal[l]` convention. Source: Basis Set Exchange `def2-ecp`.
fn i_def2_ecp(atom: usize) -> Ecp {
    Ecp {
        atom,
        n_core: 28,
        max_l: 3,
        local: vec![
            epr(2, 19.458_609_00, -21.842_040_00),
            epr(2, 19.349_260_00, -28.468_191_00),
            epr(2, 4.823_767_00, -0.243_713_00),
            epr(2, 4.884_315_00, -0.320_804_00),
        ],
        semilocal: vec![
            vec![
                epr(2, 40.015_835_00, 49.994_293_00),
                epr(2, 17.429_747_00, 281.025_317_00),
                epr(2, 9.005_484_00, 61.573_326_00),
                epr(2, 19.458_609_00, 21.842_040_00),
                epr(2, 19.349_260_00, 28.468_191_00),
                epr(2, 4.823_767_00, 0.243_713_00),
                epr(2, 4.884_315_00, 0.320_804_00),
            ],
            vec![
                epr(2, 15.355_466_00, 67.442_841_00),
                epr(2, 14.971_833_00, 134.881_137_00),
                epr(2, 8.960_164_00, 14.675_051_00),
                epr(2, 8.259_096_00, 29.375_666_00),
                epr(2, 19.458_609_00, 21.842_040_00),
                epr(2, 19.349_260_00, 28.468_191_00),
                epr(2, 4.823_767_00, 0.243_713_00),
                epr(2, 4.884_315_00, 0.320_804_00),
            ],
            vec![
                epr(2, 15.068_908_00, 35.439_529_00),
                epr(2, 14.555_322_00, 53.176_057_00),
                epr(2, 6.718_647_00, 9.067_195_00),
                epr(2, 6.456_393_00, 13.206_937_00),
                epr(2, 1.191_779_00, 0.089_335_00),
                epr(2, 1.291_157_00, 0.052_380_00),
                epr(2, 19.458_609_00, 21.842_040_00),
                epr(2, 19.349_260_00, 28.468_191_00),
                epr(2, 4.823_767_00, 0.243_713_00),
                epr(2, 4.884_315_00, 0.320_804_00),
            ],
        ],
    }
}

/// Published def2-ECP for lead (Pb, ECP60MDF): 60-electron core, local L = f,
/// s/p/d projectors (the 6th-row main-group case). Source: BSE `def2-ecp`.
fn pb_def2_ecp(atom: usize) -> Ecp {
    Ecp {
        atom,
        n_core: 60,
        max_l: 3,
        local: vec![
            epr(2, 3.887_512_00, 12.209_892_00),
            epr(2, 3.811_963_00, 16.190_291_00),
        ],
        semilocal: vec![
            vec![
                epr(2, 12.296_303_00, 281.285_499_00),
                epr(2, 8.632_634_00, 62.520_217_00),
                epr(2, 3.887_512_00, -12.209_892_00),
                epr(2, 3.811_963_00, -16.190_291_00),
            ],
            vec![
                epr(2, 10.241_790_00, 72.276_897_00),
                epr(2, 8.924_176_00, 144.591_083_00),
                epr(2, 6.581_342_00, 4.758_693_00),
                epr(2, 6.255_403_00, 9.940_621_00),
                epr(2, 3.887_512_00, -12.209_892_00),
                epr(2, 3.811_963_00, -16.190_291_00),
            ],
            vec![
                epr(2, 7.754_336_00, 35.848_507_00),
                epr(2, 7.720_281_00, 53.724_342_00),
                epr(2, 4.970_264_00, 10.115_256_00),
                epr(2, 4.563_789_00, 14.833_731_00),
                epr(2, 3.887_512_00, -12.209_892_00),
                epr(2, 3.811_963_00, -16.190_291_00),
            ],
        ],
    }
}

/// Published def2-ECP for ytterbium (Yb, ECP28MWB, 4f-in-valence lanthanide):
/// 28-electron core, local L = h (a zero reference potential), and projectors
/// s/p/d/f/**g** — the highest projector momentum and largest local L in the
/// def2 range, exercising the type-2 angular path at λ up to `l_proj + l_basis`.
/// Source: BSE `def2-ecp`.
fn yb_def2_ecp(atom: usize) -> Ecp {
    Ecp {
        atom,
        n_core: 28,
        max_l: 5,
        local: vec![epr(2, 1.000_000_00, 0.000_000_00)],
        semilocal: vec![
            vec![epr(2, 32.424_484_00, 891.013_777_00)],
            vec![epr(2, 18.656_232_00, 264.036_953_00)],
            vec![epr(2, 10.490_222_00, 73.923_919_00)],
            vec![epr(2, 20.774_183_00, -39.592_173_00)],
            vec![epr(2, 28.431_028_00, -34.638_638_00)],
        ],
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// (a) H-like 2-atom system with a synthetic single-channel ECP (contracted
/// s on the ECP atom, p off-center; local includes an n = 0 term).
#[test]
fn synthetic_two_atom_matches_quadrature_reference() {
    let basis = Basis::new(vec![
        Shell::new(0, [0.0, 0.0, 0.0], vec![0.8, 2.2], vec![0.6, 0.5]).unwrap(),
        Shell::new(1, [0.0, 0.0, 1.3], vec![1.1], vec![1.0]).unwrap(),
    ]);
    let ecp = Ecp {
        atom: 0,
        n_core: 2,
        max_l: 1,
        local: vec![epr(2, 1.5, -2.0), epr(0, 2.5, 1.0)],
        semilocal: vec![vec![epr(2, 1.0, 0.7)]],
    };
    let got = basis.ecp(std::slice::from_ref(&ecp));
    assert_no_nan(&got);
    let want = ref_ecp(&basis, std::slice::from_ref(&ecp), 40, 80);
    let diff = max_abs_diff(&got, &want);
    assert!(diff <= 1e-8, "max |analytic − reference| = {diff:e}");
}

/// (b) Realistic def2-ECP Ag fixture with mixed s/p/d/f shells on the ECP
/// center plus contracted s and p shells off-center.
#[test]
fn def2_ecp_ag_fixture_matches_quadrature_reference() {
    let ag = [0.4, -0.3, 0.2];
    let off = [0.4, 0.5, 1.6];
    let basis = Basis::new(vec![
        Shell::new(0, ag, vec![1.2], vec![1.0]).unwrap(),
        Shell::new(1, ag, vec![0.9], vec![1.0]).unwrap(),
        Shell::new(2, ag, vec![1.5], vec![1.0]).unwrap(),
        Shell::new(3, ag, vec![1.1], vec![1.0]).unwrap(),
        Shell::new(0, off, vec![1.3, 0.4], vec![0.4, 0.7]).unwrap(),
        Shell::new(1, off, vec![0.7], vec![1.0]).unwrap(),
    ]);
    let ecps = [ag_def2_ecp(0)];
    let got = basis.ecp(&ecps);
    assert_no_nan(&got);
    let want = ref_ecp(&basis, &ecps, 48, 96);
    let diff = max_abs_diff(&got, &want);
    assert!(diff <= 1e-8, "max |analytic − reference| = {diff:e}");
}

/// (c) Projector channels l = 0..=4 individually toggled (no local part),
/// with an on-center d shell and an off-center s shell.
#[test]
fn projector_channels_individually_match_quadrature_reference() {
    let c = [0.2, 0.1, -0.3];
    let basis = Basis::new(vec![
        Shell::new(2, c, vec![0.8], vec![1.0]).unwrap(),
        Shell::new(0, [0.9, 0.4, -0.8], vec![1.0], vec![1.0]).unwrap(),
    ]);
    for l in 0..=4usize {
        let mut semilocal = vec![Vec::new(); l + 1];
        semilocal[l] = vec![epr(2, 1.3, 1.9)];
        let ecp = Ecp {
            atom: 0,
            n_core: 0,
            max_l: l + 1,
            local: Vec::new(),
            semilocal,
        };
        let got = basis.ecp(std::slice::from_ref(&ecp));
        assert_no_nan(&got);
        let want = ref_ecp(&basis, std::slice::from_ref(&ecp), 36, 72);
        let diff = max_abs_diff(&got, &want);
        assert!(diff <= 1e-8, "l={l}: max |analytic − reference| = {diff:e}");
    }
}

/// g shells (L = 4) on and off the ECP center, with both a local part and a
/// d-projector channel.
#[test]
fn g_shells_on_and_off_center_match_quadrature_reference() {
    let c = [0.1, -0.2, 0.3];
    let basis = Basis::new(vec![
        Shell::new(4, c, vec![0.9], vec![1.0]).unwrap(),
        Shell::new(4, [0.5, 0.0, 1.0], vec![1.1], vec![1.0]).unwrap(),
    ]);
    let ecp = Ecp {
        atom: 0,
        n_core: 0,
        max_l: 3,
        local: vec![epr(2, 1.2, -1.5)],
        semilocal: vec![Vec::new(), Vec::new(), vec![epr(2, 0.9, 0.8)]],
    };
    let got = basis.ecp(std::slice::from_ref(&ecp));
    assert_no_nan(&got);
    let want = ref_ecp(&basis, std::slice::from_ref(&ecp), 40, 80);
    let diff = max_abs_diff(&got, &want);
    assert!(diff <= 1e-8, "max |analytic − reference| = {diff:e}");
}

/// h (l=5) and i (l=6) basis functions on and off the ECP center, validated
/// against the brute-force quadrature reference. Heavy-element QZ basis sets
/// carry h (and the engine supports i); the type-1 local channel drives the
/// angular order to `2·l = 12`, exercising the high-λ real-spherical-harmonic
/// normalization — the regime where the previous `factorial(l+m)` form lost
/// ~1e-7 of precision (see the `rsh_poly` normalization fix). Projector ≤ g
/// (l=4, the def2 maximum), so the reference's closed-form l≤4 harmonics suffice.
/// Closes the gap where no basis above g (l=4) was ever validated.
#[test]
fn h_and_i_basis_match_quadrature_reference() {
    for l in [5usize, 6] {
        let c = [0.15, -0.2, 0.3];
        let basis = Basis::new(vec![
            Shell::new(l, c, vec![1.1], vec![1.0]).unwrap(), // on the ECP center
            Shell::new(l, [0.5, 0.1, 1.0], vec![0.9], vec![1.0]).unwrap(), // off-center
        ]);
        // L=5 (h) local channel drives type-1 λ up to 2l; a g (l=4) projector
        // exercises type-2 λ up to l_proj + l_basis.
        let ecp = Ecp {
            atom: 0,
            n_core: 0,
            max_l: 5,
            local: vec![epr(2, 1.3, -1.8)],
            semilocal: vec![
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                vec![epr(2, 0.9, 0.7)],
            ],
        };
        let got = basis.ecp(std::slice::from_ref(&ecp));
        assert_no_nan(&got);
        // Angular GL is exact for these polynomial integrands well below (40, 80);
        // accuracy is radial-grid-limited (~1e-13), so a tight guard is safe and
        // catches the ~1e-7 high-λ normalization regression by orders of magnitude.
        let want = ref_ecp(&basis, std::slice::from_ref(&ecp), 40, 80);
        let diff = max_abs_diff(&got, &want);
        eprintln!("l={l} h/i-basis ECP vs quadrature: max diff = {diff:e}");
        assert!(diff <= 1e-9, "l={l}: max |analytic − reference| = {diff:e}");
    }
}

/// Real published def2-ECPs spanning the Rb–Rn range — I (ECP28, 5th-row
/// p-block), Yb (ECP28, lanthanide: local L=5/h with a g projector), Pb (ECP60,
/// 6th-row p-block) — validated against the brute-force quadrature reference for
/// the ACTUAL heavy-element potentials (deep/tight exponents, real channel
/// structures), not just synthetic ones. Together with the Ag (ECP28) fixtures
/// this covers 5th-row, 6th-row, and lanthanide ECPs.
#[test]
fn def2_ecp_range_matches_quadrature_reference() {
    let center = [0.2, -0.1, 0.3];
    let off = [0.4, 0.5, 1.4];
    let make_basis = || {
        Basis::new(vec![
            Shell::new(0, center, vec![1.5], vec![1.0]).unwrap(),
            Shell::new(1, center, vec![1.1], vec![1.0]).unwrap(),
            Shell::new(2, center, vec![0.9], vec![1.0]).unwrap(),
            Shell::new(3, center, vec![1.2], vec![1.0]).unwrap(), // f on the ECP atom
            Shell::new(0, off, vec![0.8], vec![1.0]).unwrap(),
            Shell::new(1, off, vec![0.6], vec![1.0]).unwrap(),
        ])
    };
    for (name, ecp) in [
        ("I", i_def2_ecp(0)),
        ("Yb", yb_def2_ecp(0)),
        ("Pb", pb_def2_ecp(0)),
    ] {
        let basis = make_basis();
        let got = basis.ecp(std::slice::from_ref(&ecp));
        assert_no_nan(&got);
        let want = ref_ecp(&basis, std::slice::from_ref(&ecp), 40, 80);
        let diff = max_abs_diff(&got, &want);
        eprintln!("{name} def2-ECP vs quadrature: max diff = {diff:e}");
        assert!(
            diff <= 1e-7,
            "{name}: max |analytic − reference| = {diff:e}"
        );
    }
}

/// The ECP matrix is exactly symmetric (bitwise, by construction), including
/// for mixed Cartesian/spherical bases.
#[test]
fn matrix_is_exactly_symmetric() {
    let ag = [0.4, -0.3, 0.2];
    let basis = Basis::new(vec![
        Shell::new(0, ag, vec![1.2], vec![1.0]).unwrap(),
        Shell::new_spherical(2, ag, vec![1.5], vec![1.0]).unwrap(),
        Shell::new_spherical(3, [0.4, 0.5, 1.6], vec![0.9], vec![1.0]).unwrap(),
        Shell::new(1, [0.4, 0.5, 1.6], vec![0.7], vec![1.0]).unwrap(),
    ]);
    let m = basis.ecp(&[ag_def2_ecp(0)]);
    assert_no_nan(&m);
    let n = basis.nao();
    for i in 0..n {
        for j in 0..n {
            assert!(
                (m[i * n + j] - m[j * n + i]).abs() <= 1e-12,
                "asymmetry at ({i},{j})"
            );
        }
    }
}

/// Rigid translation of the whole system leaves the matrix unchanged.
#[test]
fn translation_invariance() {
    let a = [0.5, -0.25, 1.5];
    let b = [1.25, 0.75, -0.5];
    let t = [16.0, -8.0, 4.0];
    let build = |s: [f64; 3]| {
        Basis::new(vec![
            Shell::new(
                0,
                [a[0] + s[0], a[1] + s[1], a[2] + s[2]],
                vec![0.8, 2.0],
                vec![0.5, 0.6],
            )
            .unwrap(),
            Shell::new(
                2,
                [a[0] + s[0], a[1] + s[1], a[2] + s[2]],
                vec![1.1],
                vec![1.0],
            )
            .unwrap(),
            Shell::new(
                1,
                [b[0] + s[0], b[1] + s[1], b[2] + s[2]],
                vec![0.9],
                vec![1.0],
            )
            .unwrap(),
        ])
    };
    let ecp = Ecp {
        atom: 0,
        n_core: 0,
        max_l: 2,
        local: vec![epr(2, 1.4, -3.0)],
        semilocal: vec![vec![epr(2, 1.1, 2.0)], vec![epr(2, 0.8, 0.9)]],
    };
    let m0 = build([0.0; 3]).ecp(std::slice::from_ref(&ecp));
    let m1 = build(t).ecp(std::slice::from_ref(&ecp));
    let diff = max_abs_diff(&m0, &m1);
    assert!(diff <= 1e-12, "translation moved the matrix by {diff:e}");
}

/// Empty ECP lists and zero-coefficient ECPs give an exactly-zero matrix; the
/// contribution of several ECPs is the sum of their individual matrices.
#[test]
fn empty_zero_and_multiple_ecps() {
    let a = [0.0, 0.0, 0.0];
    let b = [0.0, 0.0, 1.4];
    let basis = Basis::new(vec![
        Shell::new(0, a, vec![0.9], vec![1.0]).unwrap(),
        Shell::new(1, b, vec![1.2], vec![1.0]).unwrap(),
    ]);
    let n = basis.nao();
    // no ECPs at all
    assert!(basis.ecp(&[]).iter().all(|&v| v == 0.0));
    // empty expansions
    let empty = Ecp {
        atom: 0,
        n_core: 0,
        max_l: 1,
        local: Vec::new(),
        semilocal: vec![Vec::new()],
    };
    assert!(basis.ecp(&[empty]).iter().all(|&v| v == 0.0));
    // zero coefficients
    let zero = Ecp {
        atom: 1,
        n_core: 0,
        max_l: 1,
        local: vec![epr(2, 1.0, 0.0)],
        semilocal: vec![vec![epr(2, 1.0, 0.0)]],
    };
    assert!(basis.ecp(&[zero]).iter().all(|&v| v == 0.0));
    // additivity over atoms (Cartesian basis: bitwise-exact sum)
    let e0 = Ecp {
        atom: 0,
        n_core: 0,
        max_l: 1,
        local: vec![epr(2, 1.5, -2.0)],
        semilocal: vec![vec![epr(2, 1.0, 0.7)]],
    };
    let e1 = Ecp {
        atom: 1,
        n_core: 0,
        max_l: 2,
        local: vec![epr(2, 1.2, -1.0)],
        semilocal: vec![vec![epr(2, 0.9, 0.5)], vec![epr(2, 1.3, 0.4)]],
    };
    let both = basis.ecp(&[e0.clone(), e1.clone()]);
    let m0 = basis.ecp(std::slice::from_ref(&e0));
    let m1 = basis.ecp(std::slice::from_ref(&e1));
    for i in 0..n * n {
        assert!(
            (both[i] - (m0[i] + m1[i])).abs() <= 1e-12,
            "additivity broken at {i}"
        );
    }
}

/// For s and p shells the spherical transform is the identity, so the
/// spherical-shell ECP matrix must equal the Cartesian one (wiring check of
/// the shared `c2s` path).
#[test]
fn spherical_s_p_matches_cartesian() {
    let a = [0.1, 0.2, -0.3];
    let b = [0.8, -0.4, 0.9];
    let cart = Basis::new(vec![
        Shell::new(0, a, vec![0.9], vec![1.0]).unwrap(),
        Shell::new(1, b, vec![1.2], vec![1.0]).unwrap(),
    ]);
    let sph = Basis::new(vec![
        Shell::new_spherical(0, a, vec![0.9], vec![1.0]).unwrap(),
        Shell::new_spherical(1, b, vec![1.2], vec![1.0]).unwrap(),
    ]);
    let ecp = Ecp {
        atom: 0,
        n_core: 0,
        max_l: 1,
        local: vec![epr(2, 1.5, -2.0)],
        semilocal: vec![vec![epr(2, 1.0, 0.7)]],
    };
    let mc = cart.ecp(std::slice::from_ref(&ecp));
    let ms = sph.ecp(std::slice::from_ref(&ecp));
    let diff = max_abs_diff(&mc, &ms);
    assert!(diff <= 1e-12, "spherical s/p path deviates by {diff:e}");
}
