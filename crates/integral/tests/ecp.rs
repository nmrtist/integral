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
