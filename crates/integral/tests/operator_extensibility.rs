//! Public-API extensibility test.
//!
//! Claim: "adding a new integral type is a single declaration, not an
//! engine change." The base library demonstrates this by adding three *constructors*
//! (`quadrupole`/`momentum`/`angular_momentum`). This file pushes the claim
//! HARDER: it adds THREE genuinely new operator types **not** built into the
//! library — built **entirely from the public `Operator`/`Term`/`Factor`
//! constructors in this test file**, touching **no** `integral`
//! source at all (not even a constructor):
//!
//!   * third moment      `r_i r_j r_k`  (degree 3, real)     — vs analytic
//!   * mixed momentum     `p_i p_j`      (degree 2, real)     — vs 2nd FD of overlap
//!   * position×momentum  `r_i p_j`      (degree 2, imaginary)— vs FD of dipole
//!
//! `r_i p_j` is NOT `angular_momentum` (which is the antisymmetric combination
//! `r_j p_k − r_k p_j`); a bare `r_x p_y` is a fresh type. Each is validated
//! engine-free (analytic or finite-difference of the already-validated VALUE
//! builders), and each character claim (real⇒Im bit-zero, imaginary⇒Re bit-zero)
//! is asserted at machine zero.
//!
//! If this file compiles and passes with no change to the library, the
//! "single declaration" property is Confirmed at its strongest form.

use integral::{Basis, Factor, Operator, Shell, ShellKind, Term};

const H: f64 = 1.5e-3;

fn two_shell_basis(l: usize, sph: bool) -> Basis {
    let kind = if sph {
        ShellKind::Spherical
    } else {
        ShellKind::Cartesian
    };
    Basis::new(vec![
        Shell::with_kind(l, [0.0, 0.0, 0.0], vec![1.2, 0.45], vec![0.6, 0.5], kind).unwrap(),
        Shell::with_kind(l, [0.7, -0.3, 0.4], vec![0.9], vec![1.0], kind).unwrap(),
    ])
}

fn shift_ket(basis: &Basis, d: [f64; 3]) -> Basis {
    let shells: Vec<Shell> = basis
        .shells()
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let c = if i == 1 {
                [
                    s.center()[0] + d[0],
                    s.center()[1] + d[1],
                    s.center()[2] + d[2],
                ]
            } else {
                s.center()
            };
            Shell::with_kind(
                s.l(),
                c,
                s.exponents().to_vec(),
                s.coefficients().to_vec(),
                s.kind(),
            )
            .unwrap()
        })
        .collect();
    Basis::new(shells)
}

/// `(rows of shell 0) × (cols of shell 1)` sub-block of an nao×nao row-major mat.
fn block01(mat: &[f64], n0: usize, n1: usize) -> Vec<f64> {
    let nao = n0 + n1;
    let mut out = Vec::with_capacity(n0 * n1);
    for i in 0..n0 {
        for j in 0..n1 {
            out.push(mat[i * nao + (n0 + j)]);
        }
    }
    out
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn d3(axis: usize, h: f64) -> [f64; 3] {
    let mut d = [0.0; 3];
    d[axis] = h;
    d
}

// ===========================================================================
// NEW TYPE 1 — third moment r_i r_j r_k (degree 3, real-symmetric).
// Declaration is literally `Operator::new(vec![Term::new(1.0, vec![R,R,R])],O)`.
// ===========================================================================

#[test]
fn third_moment_same_center_s_matches_analytic() {
    // ⟨s|(r-O)_i(r-O)_j(r-O)_k|s⟩ / S =
    //   δ_iδ_jδ_k + [δ_i·[j=k] + δ_j·[i=k] + δ_k·[i=j]]/(2p),  δ = R-O, p=α+β.
    let alpha = 1.3;
    let r = [0.3, -0.2, 0.5];
    let o = [0.1, 0.15, -0.05];
    let basis = Basis::new(vec![Shell::new(0, r, vec![alpha], vec![1.0]).unwrap()]);
    let s = basis.overlap()[0];
    let p = 2.0 * alpha;
    let del = [r[0] - o[0], r[1] - o[1], r[2] - o[2]];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                // The whole "new type": one declaration, public API only.
                let op = Operator::new(
                    vec![Term::new(
                        1.0,
                        vec![Factor::R(i), Factor::R(j), Factor::R(k)],
                    )],
                    o,
                );
                let m = basis.int1e(&op).unwrap();
                let expect = (del[i] * del[j] * del[k]
                    + (del[i] * f64::from(u8::from(j == k))
                        + del[j] * f64::from(u8::from(i == k))
                        + del[k] * f64::from(u8::from(i == j)))
                        / (2.0 * p))
                    * s;
                assert!(
                    (m.real()[0] - expect).abs() < 1e-13,
                    "rrr[{i}{j}{k}]: {} vs {}",
                    m.real()[0],
                    expect
                );
                // Real operator (n_p=0) ⇒ imaginary part is bit-exact zero.
                assert!(m.imag().iter().all(|&v| v == 0.0), "rrr imag not bit-zero");
            }
        }
    }
}

#[test]
fn third_moment_is_fully_symmetric_full_l() {
    // r_i r_j r_k is invariant under any permutation of (i,j,k) and the matrix is
    // symmetric (real Hermitian operator). Full L, Cart + sph.
    let o = [0.2, -0.1, 0.3];
    let mut worst_perm = 0.0_f64;
    let mut worst_sym = 0.0_f64;
    for &sph in &[false, true] {
        for l in 0..=3 {
            let basis = two_shell_basis(l, sph);
            let nao = basis.nao();
            let mk = |i: usize, j: usize, k: usize| {
                basis
                    .int1e(&Operator::new(
                        vec![Term::new(
                            1.0,
                            vec![Factor::R(i), Factor::R(j), Factor::R(k)],
                        )],
                        o,
                    ))
                    .unwrap()
            };
            let ref_ = mk(0, 1, 2);
            for &(i, j, k) in &[(0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)] {
                worst_perm = worst_perm.max(max_abs_diff(ref_.real(), mk(i, j, k).real()));
            }
            worst_perm = worst_perm.max(ref_.max_abs_imag());
            for a in 0..nao {
                for b in 0..nao {
                    worst_sym =
                        worst_sym.max((ref_.real()[a * nao + b] - ref_.real()[b * nao + a]).abs());
                }
            }
        }
    }
    eprintln!("third moment: worst permutation {worst_perm:.2e}, worst asymmetry {worst_sym:.2e}");
    assert!(worst_perm < 1e-12, "rrr permutation {worst_perm:.3e}");
    assert!(worst_sym < 1e-12, "rrr symmetry {worst_sym:.3e}");
}

// ===========================================================================
// NEW TYPE 2 — mixed momentum product p_i p_j (degree 2, real-symmetric).
// Re⟨a|p_i p_j|b⟩ = -∂_{B_i}∂_{B_j}⟨a|b⟩  (n_p=2 ⇒ phase -1, real).
// Validated against an independent mixed 2nd central difference of OVERLAP.
// ===========================================================================

/// Mixed 2nd central difference ∂_{B_i}∂_{B_j} of the overlap 0,1 block.
fn fd2_overlap_block(basis: &Basis, i: usize, j: usize, n0: usize, n1: usize) -> Vec<f64> {
    let ov = |d: [f64; 3]| block01(&shift_ket(basis, d).overlap(), n0, n1);
    if i == j {
        // ∂² f ≈ [f(+h) - 2 f(0) + f(-h)] / h²
        let fp = ov(d3(i, H));
        let f0 = ov([0.0; 3]);
        let fm = ov(d3(i, -H));
        (0..fp.len())
            .map(|t| (fp[t] - 2.0 * f0[t] + fm[t]) / (H * H))
            .collect()
    } else {
        // ∂²/∂i∂j ≈ [f(++) - f(+-) - f(-+) + f(--)] / (4h²)
        let mut dpp = [0.0; 3];
        dpp[i] = H;
        dpp[j] = H;
        let mut dpm = [0.0; 3];
        dpm[i] = H;
        dpm[j] = -H;
        let mut dmp = [0.0; 3];
        dmp[i] = -H;
        dmp[j] = H;
        let mut dmm = [0.0; 3];
        dmm[i] = -H;
        dmm[j] = -H;
        let (a, b, c, d) = (ov(dpp), ov(dpm), ov(dmp), ov(dmm));
        (0..a.len())
            .map(|t| (a[t] - b[t] - c[t] + d[t]) / (4.0 * H * H))
            .collect()
    }
}

#[test]
fn momentum_product_real_matches_overlap_2nd_fd_full_l() {
    let mut worst = 0.0_f64;
    for &sph in &[false, true] {
        for l in 0..=3 {
            let basis = two_shell_basis(l, sph);
            let (n0, n1) = (basis.shells()[0].n_func(), basis.shells()[1].n_func());
            for i in 0..3 {
                for j in 0..3 {
                    // The whole "new type": one declaration, public API only.
                    let op = Operator::new(
                        vec![Term::new(1.0, vec![Factor::P(i), Factor::P(j)])],
                        [0.0; 3],
                    );
                    let m = basis.int1e(&op).unwrap();
                    // n_p=2 ⇒ imaginary part bit-exact zero.
                    assert!(
                        m.imag().iter().all(|&v| v == 0.0),
                        "p_i p_j imag not bit-zero"
                    );
                    let re01 = block01(m.real(), n0, n1);
                    // Re = -∂_{B_i}∂_{B_j} overlap.
                    let fd: Vec<f64> = fd2_overlap_block(&basis, i, j, n0, n1)
                        .iter()
                        .map(|v| -v)
                        .collect();
                    worst = worst.max(max_abs_diff(&re01, &fd));
                }
            }
        }
    }
    eprintln!("p_i p_j: worst Re-vs-(-2nd-FD overlap) = {worst:.2e}");
    // 2nd-order mixed FD is cancellation-limited; ~1e-5 is clean agreement.
    assert!(worst < 1e-4, "p_i p_j worst {worst:.3e}");
}

#[test]
fn momentum_product_diag_sum_is_two_kinetic() {
    // Σ_i p_i p_i = 2·(½ p·p) = -∇² = 2·kinetic. Independent cross-check that the
    // bare p_i p_i declarations sum to the bespoke kinetic builder.
    for &sph in &[false, true] {
        for l in 0..=4 {
            let basis = two_shell_basis(l, sph);
            let nao = basis.nao();
            let kin = basis.kinetic();
            let mut sum = vec![0.0; nao * nao];
            for i in 0..3 {
                let op = Operator::new(
                    vec![Term::new(1.0, vec![Factor::P(i), Factor::P(i)])],
                    [0.0; 3],
                );
                let m = basis.int1e(&op).unwrap();
                for (s, v) in sum.iter_mut().zip(m.real()) {
                    *s += v;
                }
            }
            let worst = sum
                .iter()
                .zip(&kin)
                .map(|(s, k)| (0.5 * s - k).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                worst < 1e-11,
                "½Σp_ip_i≡kinetic l={l} sph={sph}: {worst:.3e}"
            );
        }
    }
}

// ===========================================================================
// NEW TYPE 3 — position×momentum r_i p_j (degree 2, imaginary/anti-character).
// Im⟨a|r_i p_j|b⟩ = ∂_{B_j}⟨a|(r-O)_i|b⟩  (FD of dipole component i, ket axis j).
// Distinct from angular_momentum (which is r_j p_k - r_k p_j).
// ===========================================================================

fn fd_dipole_block(
    basis: &Basis,
    comp: usize,
    axis: usize,
    o: [f64; 3],
    n0: usize,
    n1: usize,
) -> Vec<f64> {
    // First derivative: a finer step than the 2nd-difference tests (less
    // cancellation pressure, better truncation), matching the value-builder FD step.
    const HD: f64 = 2e-4;
    let dip = |d: [f64; 3]| block01(&shift_ket(basis, d).dipole(o)[comp], n0, n1);
    let p = dip(d3(axis, HD));
    let m = dip(d3(axis, -HD));
    (0..p.len()).map(|t| (p[t] - m[t]) / (2.0 * HD)).collect()
}

#[test]
fn pos_times_momentum_imag_matches_dipole_fd_full_l() {
    let o = [0.15, -0.25, 0.2];
    let mut worst = 0.0_f64;
    for &sph in &[false, true] {
        for l in 0..=4 {
            let basis = two_shell_basis(l, sph);
            let (n0, n1) = (basis.shells()[0].n_func(), basis.shells()[1].n_func());
            for i in 0..3 {
                for j in 0..3 {
                    // The whole "new type": one declaration, public API only.
                    let op =
                        Operator::new(vec![Term::new(1.0, vec![Factor::R(i), Factor::P(j)])], o);
                    let m = basis.int1e(&op).unwrap();
                    // n_p=1 ⇒ real part bit-exact zero.
                    assert!(
                        m.real().iter().all(|&v| v == 0.0),
                        "r_i p_j real not bit-zero"
                    );
                    let im01 = block01(m.imag(), n0, n1);
                    let fd = fd_dipole_block(&basis, i, j, o, n0, n1);
                    worst = worst.max(max_abs_diff(&im01, &fd));
                }
            }
        }
    }
    eprintln!("r_i p_j: worst Im-vs-FD(dipole) = {worst:.2e}");
    assert!(worst < 1e-6, "r_i p_j worst {worst:.3e}");
}

#[test]
fn angular_momentum_origin_shift_is_wired() {
    // L_a(o) = (r-o)×p = r×p - o×p = L_a(0) - (o × p)_a.  For a = x:
    //   L_x(o) - L_x(0) = -o_y·p_z + o_z·p_y.
    // This falsifies a broken r-origin shift (the `b_minus_o` term) engine-free:
    // if the origin were ignored, the LHS would be 0 but the RHS is not.
    let o = [0.35, -0.2, 0.45];
    for l in 0..=3 {
        let basis = two_shell_basis(l, false);
        let nao = basis.nao();
        for axis in 0..3 {
            let (j, k) = match axis {
                0 => (1, 2),
                1 => (2, 0),
                _ => (0, 1),
            };
            let l_o = basis.int1e(&Operator::angular_momentum(axis, o)).unwrap();
            let l_0 = basis
                .int1e(&Operator::angular_momentum(axis, [0.0; 3]))
                .unwrap();
            let pj = basis.int1e(&Operator::momentum(j)).unwrap();
            let pk = basis.int1e(&Operator::momentum(k)).unwrap();
            // L_a(o) - L_a(0) = -o_j·p_k + o_k·p_j  (cyclic j,k after a).
            let worst = (0..nao * nao)
                .map(|t| {
                    let lhs = l_o.imag()[t] - l_0.imag()[t];
                    let rhs = -o[j] * pk.imag()[t] + o[k] * pj.imag()[t];
                    (lhs - rhs).abs()
                })
                .fold(0.0_f64, f64::max);
            assert!(worst < 1e-12, "L_{axis} origin shift wiring: {worst:.3e}");
            // Sanity: the shift is actually non-trivial (origin matters).
            let shift_mag = (0..nao * nao)
                .map(|t| (l_o.imag()[t] - l_0.imag()[t]).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                shift_mag > 1e-6,
                "origin shift vanished (l={l}) — not a real test"
            );
        }
    }
}

#[test]
fn pos_times_momentum_antisymmetrized_equals_angular_momentum() {
    // Cross-check the bare r_i p_j against the library's angular_momentum: the
    // antisymmetric combination r_j p_k - r_k p_j must reproduce L_axis exactly.
    let o = [0.1, 0.2, -0.15];
    for l in 0..=3 {
        let basis = two_shell_basis(l, false);
        let nao = basis.nao();
        for axis in 0..3 {
            let (j, k) = match axis {
                0 => (1, 2),
                1 => (2, 0),
                _ => (0, 1),
            };
            let rjpk = basis
                .int1e(&Operator::new(
                    vec![Term::new(1.0, vec![Factor::R(j), Factor::P(k)])],
                    o,
                ))
                .unwrap();
            let rkpj = basis
                .int1e(&Operator::new(
                    vec![Term::new(1.0, vec![Factor::R(k), Factor::P(j)])],
                    o,
                ))
                .unwrap();
            let lop = basis.int1e(&Operator::angular_momentum(axis, o)).unwrap();
            let worst = (0..nao * nao)
                .map(|t| ((rjpk.imag()[t] - rkpj.imag()[t]) - lop.imag()[t]).abs())
                .fold(0.0_f64, f64::max);
            assert!(worst < 1e-12, "L_{axis} from r_jp_k - r_kp_j: {worst:.3e}");
        }
    }
}
