//! Operator DSL, validation dimensions 2 & 3: **genuinely new
//! integral types added as single [`Operator`] declarations**, with engine-free
//! checks (no external library).
//!
//! The three new types — each one constructor in `integral::engine::operator`, **no
//! engine code** — are:
//!
//!   * quadrupole / second moment `r_i r_j`  ([`Operator::quadrupole`]) — real
//!   * momentum `p_i = −i∂_i`                 ([`Operator::momentum`]) — imaginary
//!   * angular momentum `(r × p)_i`           ([`Operator::angular_momentum`]) — imaginary
//!
//! ## Engine-free anchors (no external library)
//!
//! - **Quadrupole**: same-center `s|s` analytic
//!   `⟨s|(r−O)_i(r−O)_j|s⟩ = [(R−O)_i(R−O)_j + δ_ij/(2p)] · ⟨s|s⟩`, plus symmetry
//!   and `r_i r_j = r_j r_i` over the full L range.
//! - **Momentum / angular momentum** (imaginary): their imaginary block is a
//!   **center-derivative of a value builder**, so it is finite-differenced
//!   against the (independently cross-checked) value integrals — the same FD-anchoring
//!   used for geometric derivatives, and their real part must vanish
//!   (anti-Hermitian character):
//!
//! ```text
//!   Im⟨a|p_k|b⟩      = ∂_{B_k}⟨a|b⟩                 (FD of overlap, ket center)
//!   Im⟨a|(r×p)_x|b⟩  = ∂_{B_z}⟨a|(r−O)_y|b⟩ − ∂_{B_y}⟨a|(r−O)_z|b⟩
//!                                                   (FD of dipole, ket center)
//! ```

use integral::{Basis, Operator, Shell, ShellKind};

const H: f64 = 2e-4;
const FD_TOL: f64 = 1e-6;

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

/// Move shell index 1's center by `d`, keeping shell 0 fixed.
fn shift_ket_shell(basis: &Basis, d: [f64; 3]) -> Basis {
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

fn unit(axis: usize, h: f64) -> [f64; 3] {
    let mut d = [0.0; 3];
    d[axis] = h;
    d
}

/// The `(rows of shell 0) × (cols of shell 1)` sub-block of an `nao × nao`
/// row-major matrix — i.e. `⟨χ∈shell0 | O | χ∈shell1⟩`, where shell 1 is the ket.
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

/// Central difference of a value-builder block (extracted as the 0,1 block) w.r.t.
/// moving the ket shell along `axis`.
fn fd_ket_block(
    basis: &Basis,
    axis: usize,
    n0: usize,
    n1: usize,
    value: impl Fn(&Basis) -> Vec<f64>,
) -> Vec<f64> {
    let plus = value(&shift_ket_shell(basis, unit(axis, H)));
    let minus = value(&shift_ket_shell(basis, unit(axis, -H)));
    let bp = block01(&plus, n0, n1);
    let bm = block01(&minus, n0, n1);
    bp.iter()
        .zip(&bm)
        .map(|(p, m)| (p - m) / (2.0 * H))
        .collect()
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

// ---------------------------------------------------------------------------
// Quadrupole r_i r_j (real-symmetric)
// ---------------------------------------------------------------------------

#[test]
fn quadrupole_same_center_s_matches_analytic() {
    // Single-primitive s shell at R; second moment about O.
    let alpha = 1.3;
    let r = [0.3, -0.2, 0.5];
    let o = [0.1, 0.15, -0.05];
    let basis = Basis::new(vec![Shell::new(0, r, vec![alpha], vec![1.0]).unwrap()]);
    let s = basis.overlap()[0];
    let p = 2.0 * alpha; // α + β with β = α
    for i in 0..3 {
        for j in 0..3 {
            let q = basis.int1e(&Operator::quadrupole(i, j, o)).unwrap();
            let expect =
                ((r[i] - o[i]) * (r[j] - o[j]) + if i == j { 1.0 / (2.0 * p) } else { 0.0 }) * s;
            assert!(
                (q.real()[0] - expect).abs() < 1e-13,
                "rr[{i}{j}]: {} vs {}",
                q.real()[0],
                expect
            );
            assert_eq!(q.max_abs_imag(), 0.0);
        }
    }
}

#[test]
fn quadrupole_is_symmetric_and_commutes_full_l() {
    let o = [0.2, -0.1, 0.3];
    let mut worst_comm = 0.0_f64;
    let mut worst_sym = 0.0_f64;
    for &sph in &[false, true] {
        for l in 0..=4 {
            let basis = two_shell_basis(l, sph);
            let nao = basis.nao();
            for i in 0..3 {
                for j in 0..3 {
                    let rij = basis.int1e(&Operator::quadrupole(i, j, o)).unwrap();
                    let rji = basis.int1e(&Operator::quadrupole(j, i, o)).unwrap();
                    // r_i r_j = r_j r_i (factors commute).
                    worst_comm = worst_comm.max(max_abs_diff(rij.real(), rji.real()));
                    worst_comm = worst_comm.max(rij.max_abs_imag());
                    // Matrix is symmetric (Hermitian real operator).
                    for a in 0..nao {
                        for b in 0..nao {
                            let d = (rij.real()[a * nao + b] - rij.real()[b * nao + a]).abs();
                            worst_sym = worst_sym.max(d);
                        }
                    }
                }
            }
        }
    }
    eprintln!("quadrupole: worst commute {worst_comm:.2e}, worst asymmetry {worst_sym:.2e}");
    assert!(worst_comm < 1e-12, "rr commute {worst_comm:.3e}");
    assert!(worst_sym < 1e-12, "rr symmetry {worst_sym:.3e}");
}

// ---------------------------------------------------------------------------
// Momentum p_i (imaginary / anti-Hermitian)
// ---------------------------------------------------------------------------

#[test]
fn momentum_imag_matches_overlap_ket_fd_full_l() {
    let mut worst = 0.0_f64;
    for &sph in &[false, true] {
        for l in 0..=5 {
            let basis = two_shell_basis(l, sph);
            let (n0, n1) = (basis.shells()[0].n_func(), basis.shells()[1].n_func());
            for axis in 0..3 {
                let p = basis.int1e(&Operator::momentum(axis)).unwrap();
                // Real part must vanish (anti-Hermitian).
                worst = worst.max(p.max_abs_real());
                // Im ⟨a|p_axis|b⟩ on the 0,1 block = ∂_{B_axis} ⟨a|b⟩ (FD overlap).
                let im01 = block01(p.imag(), n0, n1);
                let fd = fd_ket_block(&basis, axis, n0, n1, Basis::overlap);
                worst = worst.max(max_abs_diff(&im01, &fd));
            }
        }
    }
    eprintln!("momentum: worst (real-leak ∪ Im-vs-FD) = {worst:.2e}");
    assert!(worst < FD_TOL, "momentum worst {worst:.3e}");
}

#[test]
fn momentum_matrix_is_antisymmetric_imaginary() {
    // p is Hermitian → i·M Hermitian → M (the imaginary part) is antisymmetric.
    for l in 0..=3 {
        let basis = two_shell_basis(l, false);
        let nao = basis.nao();
        for axis in 0..3 {
            let p = basis.int1e(&Operator::momentum(axis)).unwrap();
            let im = p.imag();
            let mut worst = 0.0_f64;
            let mut peak = 0.0_f64;
            for a in 0..nao {
                for b in 0..nao {
                    worst = worst.max((im[a * nao + b] + im[b * nao + a]).abs());
                    peak = peak.max(im[a * nao + b].abs());
                }
            }
            // Antisymmetry alone is vacuously satisfied if the imaginary buffer is
            // empty — exactly the symptom of an `i`-routing/parity bug that sends a
            // `p` term to the real buffer. Require the imaginary part to be
            // *non-trivially populated* so this test has teeth against that bug
            // `p` on a non-trivial shell pair has
            // an O(1) imaginary block, so a healthy `peak` is well away from zero.
            assert!(
                peak > 1e-6,
                "p_{axis} l={l} imaginary part vacuously empty (peak {peak:.3e}) \
                 — parity/`i`-routing regression"
            );
            assert!(worst < 1e-11, "p_{axis} l={l} antisymmetry {worst:.3e}");
        }
    }
}

// ---------------------------------------------------------------------------
// Angular momentum (r × p)_i (imaginary / anti-Hermitian)
// ---------------------------------------------------------------------------

#[test]
fn angular_momentum_imag_matches_dipole_ket_fd_full_l() {
    let o = [0.2, -0.15, 0.1];
    let mut worst = 0.0_f64;
    for &sph in &[false, true] {
        for l in 0..=4 {
            let basis = two_shell_basis(l, sph);
            let (n0, n1) = (basis.shells()[0].n_func(), basis.shells()[1].n_func());
            for axis in 0..3 {
                let (j, k) = match axis {
                    0 => (1, 2),
                    1 => (2, 0),
                    _ => (0, 1),
                };
                let lop = basis.int1e(&Operator::angular_momentum(axis, o)).unwrap();
                worst = worst.max(lop.max_abs_real()); // anti-Hermitian → Re ~ 0
                let im01 = block01(lop.imag(), n0, n1);
                // Im⟨a|(r×p)_axis|b⟩ = ∂_{B_k}⟨a|(r−O)_j|b⟩ − ∂_{B_j}⟨a|(r−O)_k|b⟩.
                let fd_j_k = fd_ket_block(&basis, k, n0, n1, |bb| bb.dipole(o)[j].clone());
                let fd_k_j = fd_ket_block(&basis, j, n0, n1, |bb| bb.dipole(o)[k].clone());
                let fd: Vec<f64> = fd_j_k.iter().zip(&fd_k_j).map(|(a, b)| a - b).collect();
                worst = worst.max(max_abs_diff(&im01, &fd));
            }
        }
    }
    eprintln!("angular momentum: worst (real-leak ∪ Im-vs-FD) = {worst:.2e}");
    assert!(worst < FD_TOL, "angular momentum worst {worst:.3e}");
}
