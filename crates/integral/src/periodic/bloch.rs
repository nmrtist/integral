//! Bloch (lattice) sums of one-electron integrals.
//!
//! A periodic Bloch basis function is `χ_μk(r) = Σ_R e^{ik·R} φ_μ(r − τ_μ − R)`. The Bloch
//! overlap and kinetic matrices are then lattice sums of the ordinary (molecular) one-electron
//! blocks between a home shell and a lattice-translated shell:
//!
//! ```text
//!   M_μν(k) = Σ_R e^{ik·R} ⟨φ_μ(·−τ_μ) | Ô | φ_ν(·−τ_ν−R)⟩.
//! ```
//!
//! Each per-image block reuses the engine's exact contracted one-electron machinery
//! (`contract_pair` + `to_func_1e`), so at Γ with only `R = 0` the result is bit-identical to
//! the molecular [`Basis::overlap`](crate::Basis::overlap) / [`Basis::kinetic`](crate::Basis::kinetic).
//! The phase uses `k·R = 2π (k_frac · n)` for integer triple `n` (see `latx`).

use crate::engine::os::{self, Prim, MAX_L};
use latx::Cell;
use rustfft::num_complex::Complex64;
use std::f64::consts::PI;

use crate::grad::pair_grad_1e;
use crate::integrals::{contract_pair, place_block, to_func_1e};
use crate::shell::{Basis, Shell};

/// A copy of shell `s` translated by `shift` (bohr).
fn shifted_shell(s: &Shell, shift: [f64; 3]) -> Shell {
    let c = s.center();
    Shell::with_kind(
        s.l(),
        [c[0] + shift[0], c[1] + shift[1], c[2] + shift[2]],
        s.exponents().to_vec(),
        s.coefficients().to_vec(),
        s.kind(),
    )
    .expect("translation preserves shell validity")
}

/// The cross one-electron matrix `M[μ,ν] = ⟨φ_μ(home) | Ô | φ_ν(shifted by `shift`)⟩`,
/// row-major `nao × nao`, for a primitive operator `prim_op` (e.g. `os::overlap_into`).
fn cross_matrix<F>(basis: &Basis, shift: [f64; 3], prim_op: F) -> Vec<f64>
where
    F: FnMut(Prim, Prim, f64, &mut [f64]) + Copy,
{
    let n = basis.nao();
    let offs = basis.offsets();
    let mut mat = vec![0.0; n * n];
    for (si, sa) in basis.shells().iter().enumerate() {
        for (sj, sb) in basis.shells().iter().enumerate() {
            let sb_shifted = shifted_shell(sb, shift);
            let block = to_func_1e(contract_pair(sa, &sb_shifted, prim_op), sa, &sb_shifted);
            place_block(&mut mat, n, offs[si], offs[sj], &block, sb.n_func());
        }
    }
    mat
}

/// Bloch sum `M_μν(k) = Σ_R e^{2πi k·n_R} ⟨μ | Ô | ν+R⟩` over all lattice images `R` within
/// `rmax` (bohr). Returns a row-major `nao × nao` complex matrix.
fn bloch_matrix<F>(
    basis: &Basis,
    cell: &Cell,
    k_frac: [f64; 3],
    rmax: f64,
    prim_op: F,
) -> Vec<Complex64>
where
    F: FnMut(Prim, Prim, f64, &mut [f64]) + Copy,
{
    let n = basis.nao();
    let mut m = vec![Complex64::new(0.0, 0.0); n * n];
    for (triple, r) in cell.lattice_images(rmax) {
        let block = cross_matrix(basis, r, prim_op);
        let theta = 2.0
            * PI
            * (k_frac[0] * f64::from(triple[0])
                + k_frac[1] * f64::from(triple[1])
                + k_frac[2] * f64::from(triple[2]));
        let phase = Complex64::new(theta.cos(), theta.sin());
        for (mij, &bij) in m.iter_mut().zip(&block) {
            *mij += phase * bij;
        }
    }
    m
}

/// The Bloch overlap matrix `S_μν(k)` for k-point fractional coordinates `k_frac`, summing
/// lattice images within `rmax` (bohr). Row-major `nao × nao` complex.
#[must_use]
pub fn bloch_overlap(basis: &Basis, cell: &Cell, k_frac: [f64; 3], rmax: f64) -> Vec<Complex64> {
    bloch_matrix(basis, cell, k_frac, rmax, os::overlap_into)
}

/// The Bloch kinetic matrix `T_μν(k)`. Row-major `nao × nao` complex.
#[must_use]
pub fn bloch_kinetic(basis: &Basis, cell: &Cell, k_frac: [f64; 3], rmax: f64) -> Vec<Complex64> {
    bloch_matrix(basis, cell, k_frac, rmax, os::kinetic_into)
}

/// Per-atom energy-gradient contribution of a Bloch one-electron operator,
/// `g_I[axis] = Σ_k w_k Re Tr(∂O(k)/∂R_{I,axis} · M^k)`.
///
/// This is the **Pulay** (basis-derivative) gradient of a Bloch-summed 1e energy term. It is
/// assembled in real space: with `O(k)_μν = Σ_R e^{2πi k·n_R} o^R_μν` and
/// `o^R_μν = ⟨φ_μ(home) | Ô | φ_ν(·−R)⟩`,
///
/// ```text
///   Σ_k w_k Tr(∂O(k) M^k) = Σ_R Σ_{μν} ∂o^R_μν · M^R_νμ ,   M^R_νμ = Σ_k w_k e^{ik·R} M^k_νμ .
/// ```
///
/// The lattice vectors `R` are fixed by the cell (atoms move within it), so there is no phase
/// derivative — only the bra/ket basis centers move, differentiated by the exact molecular
/// center-derivative blocks ([`pair_grad_1e`]) per image. The per-k weight matrix `m_k[ik]`
/// is row-major `nao²` complex: pass the density matrix `P^k` for the kinetic term, the
/// energy-weighted density `W^k` for the overlap (Pulay) term. Atom order is [`Basis::atoms`];
/// the caller applies the physical force sign. `k_fracs`/`weights`/`m_k` are aligned per
/// k-point; images are summed within `rmax` (bohr).
///
/// # Panics
/// Panics if `k_fracs`, `weights`, `m_k` lengths disagree, if any `m_k` block is not `nao²`,
/// or if any shell has `l ≥ MAX_L` (the derivative raises it to `l + 1`).
fn bloch_grad_1e_contract<F>(
    basis: &Basis,
    cell: &Cell,
    k_fracs: &[[f64; 3]],
    weights: &[f64],
    m_k: &[Vec<Complex64>],
    rmax: f64,
    op: F,
) -> Vec<[f64; 3]>
where
    F: Fn(Prim, Prim, f64, &mut [f64]) + Copy,
{
    let nk = k_fracs.len();
    assert_eq!(nk, weights.len(), "k-points and weights must align");
    assert_eq!(nk, m_k.len(), "k-points and weight matrices must align");
    let n = basis.nao();
    assert!(
        m_k.iter().all(|m| m.len() == n * n),
        "each weight matrix must be nao² = {}",
        n * n
    );
    let shells = basis.shells();
    assert!(
        shells.iter().all(|s| s.l() < MAX_L),
        "Bloch 1e gradient needs l < MAX_L (the derivative raises a shell to l+1)"
    );
    let offs = basis.offsets();
    let satom = basis.shell_atom();
    let natom = basis.atoms().len();

    let mut g = vec![[0.0_f64; 3]; natom];
    let mut mr = vec![Complex64::new(0.0, 0.0); n * n]; // M^R for the current image
    for (triple, r) in cell.lattice_images(rmax) {
        // M^R_νμ = Σ_k w_k e^{ik·R} M^k_νμ (per element, same row-major layout).
        for v in &mut mr {
            *v = Complex64::new(0.0, 0.0);
        }
        for ik in 0..nk {
            let theta = 2.0
                * PI
                * (k_fracs[ik][0] * f64::from(triple[0])
                    + k_fracs[ik][1] * f64::from(triple[1])
                    + k_fracs[ik][2] * f64::from(triple[2]));
            let phase = Complex64::new(theta.cos(), theta.sin()) * weights[ik];
            for (mrv, &mkv) in mr.iter_mut().zip(&m_k[ik]) {
                *mrv += phase * mkv;
            }
        }
        for (si, sa) in shells.iter().enumerate() {
            let nba = sa.n_func();
            let (atom_a, ri) = (satom[si], offs[si]);
            for (sj, sb) in shells.iter().enumerate() {
                let sb_shifted = shifted_shell(sb, r);
                let (da, db) = pair_grad_1e(sa, &sb_shifted, op);
                let nbf = sb.n_func();
                let (atom_b, ci) = (satom[sj], offs[sj]);
                for axis in 0..3 {
                    let fa = to_func_1e(da[axis].clone(), sa, &sb_shifted);
                    let fb = to_func_1e(db[axis].clone(), sa, &sb_shifted);
                    // Σ_{ij} ∂o^R_{μ_i ν_j} · M^R_{ν_j μ_i}; o^R is real, so take Re(M^R).
                    let (mut ca, mut cb) = (0.0, 0.0);
                    for i in 0..nba {
                        for j in 0..nbf {
                            let mre = mr[(ci + j) * n + (ri + i)].re;
                            ca += fa[i * nbf + j] * mre;
                            cb += fb[i * nbf + j] * mre;
                        }
                    }
                    g[atom_a][axis] += ca;
                    g[atom_b][axis] += cb;
                }
            }
        }
    }
    g
}

/// Per-atom Pulay gradient of the Bloch kinetic energy,
/// `g_I = Σ_k w_k Re Tr(∂T(k)/∂R_I · P^k)` (pass the per-k density matrices `p_k`). See
/// [`bloch_grad_1e_contract`]; atom order is [`Basis::atoms`].
///
/// # Panics
/// As [`bloch_grad_1e_contract`].
#[must_use]
pub fn bloch_kinetic_grad_contract(
    basis: &Basis,
    cell: &Cell,
    k_fracs: &[[f64; 3]],
    weights: &[f64],
    p_k: &[Vec<Complex64>],
    rmax: f64,
) -> Vec<[f64; 3]> {
    bloch_grad_1e_contract(basis, cell, k_fracs, weights, p_k, rmax, os::kinetic_into)
}

/// Per-atom Pulay gradient of the Bloch overlap energy,
/// `g_I = Σ_k w_k Re Tr(∂S(k)/∂R_I · W^k)` (pass the per-k **energy-weighted** density
/// matrices `w_k`, `W^k_μν = Σ_i f_i ε_i C_μi C*_νi`). The orthonormality (Pulay) force term
/// is `+g_I`. See [`bloch_grad_1e_contract`]; atom order is [`Basis::atoms`].
///
/// # Panics
/// As [`bloch_grad_1e_contract`].
#[must_use]
pub fn bloch_overlap_grad_contract(
    basis: &Basis,
    cell: &Cell,
    k_fracs: &[[f64; 3]],
    weights: &[f64],
    w_k: &[Vec<Complex64>],
    rmax: f64,
) -> Vec<[f64; 3]> {
    bloch_grad_1e_contract(basis, cell, k_fracs, weights, w_k, rmax, os::overlap_into)
}

/// The strain-derivative (stress) contraction of a Bloch one-electron operator,
/// `τ_αβ = Σ_k w_k Re Tr(∂O(k)/∂ε_αβ · M^k)`.
///
/// A two-center integral `⟨φ_μ(A)|Ô|φ_ν(C)⟩` depends only on the separation `C−A`, so under a
/// strain `F = I + ε` (both centers scale) `∂⟨Ô⟩/∂ε_αβ = (∂⟨Ô⟩/∂C_α)·(C−A)_β = db_α · sep_β`
/// — the **ket-center** gradient times the separation vector. Summing over lattice images
/// (the phase `e^{ik·R}` is strain-invariant; `R` deforms with the cell) and folding the per-k
/// weight `M^k` into the real-space `M^R_νμ = Σ_k w_k e^{ik·R} M^k_νμ` gives
///
/// ```text
///   τ_αβ = Σ_R Σ_{μν} db^R_{μν,α} · sep^R_{μν,β} · M^R_νμ ,   sep^R_{μν} = (τ_ν + R) − τ_μ .
/// ```
///
/// Returns the (generally non-symmetric) `∂(Σ_k w_k Tr(O(k) M^k))/∂ε`; pass `P^k` for the
/// kinetic term, `W^k` for overlap. The caller applies the physical sign and symmetrizes.
///
/// # Panics
/// As [`bloch_grad_1e_contract`].
fn bloch_grad_1e_stress_contract<F>(
    basis: &Basis,
    cell: &Cell,
    k_fracs: &[[f64; 3]],
    weights: &[f64],
    m_k: &[Vec<Complex64>],
    rmax: f64,
    op: F,
) -> [[f64; 3]; 3]
where
    F: Fn(Prim, Prim, f64, &mut [f64]) + Copy,
{
    let nk = k_fracs.len();
    assert_eq!(nk, weights.len(), "k-points and weights must align");
    assert_eq!(nk, m_k.len(), "k-points and weight matrices must align");
    let n = basis.nao();
    assert!(
        m_k.iter().all(|m| m.len() == n * n),
        "each weight matrix must be nao² = {}",
        n * n
    );
    let shells = basis.shells();
    assert!(
        shells.iter().all(|s| s.l() < MAX_L),
        "Bloch 1e stress needs l < MAX_L (the derivative raises a shell to l+1)"
    );
    let offs = basis.offsets();

    let mut tau = [[0.0_f64; 3]; 3];
    let mut mr = vec![Complex64::new(0.0, 0.0); n * n];
    for (triple, r) in cell.lattice_images(rmax) {
        for v in &mut mr {
            *v = Complex64::new(0.0, 0.0);
        }
        for ik in 0..nk {
            let theta = 2.0
                * PI
                * (k_fracs[ik][0] * f64::from(triple[0])
                    + k_fracs[ik][1] * f64::from(triple[1])
                    + k_fracs[ik][2] * f64::from(triple[2]));
            let phase = Complex64::new(theta.cos(), theta.sin()) * weights[ik];
            for (mrv, &mkv) in mr.iter_mut().zip(&m_k[ik]) {
                *mrv += phase * mkv;
            }
        }
        for (si, sa) in shells.iter().enumerate() {
            let nba = sa.n_func();
            let (ri, ac) = (offs[si], sa.center());
            for (sj, sb) in shells.iter().enumerate() {
                let sb_shifted = shifted_shell(sb, r);
                let nbf = sb.n_func();
                let ci = offs[sj];
                let bc = sb.center();
                let sep = [
                    bc[0] + r[0] - ac[0],
                    bc[1] + r[1] - ac[1],
                    bc[2] + r[2] - ac[2],
                ];
                let (_da, db) = pair_grad_1e(sa, &sb_shifted, op);
                for (alpha, ta) in tau.iter_mut().enumerate() {
                    let fb = to_func_1e(db[alpha].clone(), sa, &sb_shifted);
                    // c_α = Σ_{ij} db^R_{μ_i ν_j, α} · M^R_{ν_j μ_i} (M^R real-projected).
                    let mut ca = 0.0;
                    for i in 0..nba {
                        for j in 0..nbf {
                            ca += fb[i * nbf + j] * mr[(ci + j) * n + (ri + i)].re;
                        }
                    }
                    for (beta, tab) in ta.iter_mut().enumerate() {
                        *tab += ca * sep[beta];
                    }
                }
            }
        }
    }
    tau
}

/// The kinetic stress contraction `τ_αβ = Σ_k w_k Re Tr(∂T(k)/∂ε_αβ · P^k)` (pass the per-k
/// density matrices `p_k`). The kinetic contribution to `∂E/∂ε` is `+τ`. See
/// [`bloch_grad_1e_stress_contract`].
///
/// # Panics
/// As [`bloch_grad_1e_stress_contract`].
#[must_use]
pub fn bloch_kinetic_stress_contract(
    basis: &Basis,
    cell: &Cell,
    k_fracs: &[[f64; 3]],
    weights: &[f64],
    p_k: &[Vec<Complex64>],
    rmax: f64,
) -> [[f64; 3]; 3] {
    bloch_grad_1e_stress_contract(basis, cell, k_fracs, weights, p_k, rmax, os::kinetic_into)
}

/// The overlap stress contraction `τ_αβ = Σ_k w_k Re Tr(∂S(k)/∂ε_αβ · W^k)` (pass the per-k
/// **energy-weighted** density matrices `w_k`). The orthonormality (Pulay) contribution to
/// `∂E/∂ε` is `−τ`. See [`bloch_grad_1e_stress_contract`].
///
/// # Panics
/// As [`bloch_grad_1e_stress_contract`].
#[must_use]
pub fn bloch_overlap_stress_contract(
    basis: &Basis,
    cell: &Cell,
    k_fracs: &[[f64; 3]],
    weights: &[f64],
    w_k: &[Vec<Complex64>],
    rmax: f64,
) -> [[f64; 3]; 3] {
    bloch_grad_1e_stress_contract(basis, cell, k_fracs, weights, w_k, rmax, os::overlap_into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Basis, Shell};

    fn two_s_basis() -> Basis {
        Basis::new(vec![
            Shell::new(0, [1.0, 1.0, 1.0], vec![0.8], vec![1.0]).unwrap(),
            Shell::new(0, [2.0, 1.0, 1.0], vec![1.2], vec![1.0]).unwrap(),
        ])
    }

    /// At Γ with only the home cell (`rmax` excludes all R ≠ 0), the Bloch overlap is exactly
    /// the molecular overlap (real, zero imaginary).
    #[test]
    fn gamma_reduces_to_molecular_overlap() {
        let basis = two_s_basis();
        let cell = Cell::cubic(8.0).unwrap();
        let s_mol = basis.overlap();
        let s_k = bloch_overlap(&basis, &cell, [0.0, 0.0, 0.0], 0.1); // only R = 0
        for (k, &m) in s_k.iter().zip(&s_mol) {
            assert!((k.re - m).abs() < 1e-13 && k.im.abs() < 1e-13, "{k} vs {m}");
        }
    }

    /// Same for the kinetic matrix.
    #[test]
    fn gamma_reduces_to_molecular_kinetic() {
        let basis = two_s_basis();
        let cell = Cell::cubic(8.0).unwrap();
        let t_mol = basis.kinetic();
        let t_k = bloch_kinetic(&basis, &cell, [0.0, 0.0, 0.0], 0.1);
        for (k, &m) in t_k.iter().zip(&t_mol) {
            assert!((k.re - m).abs() < 1e-13 && k.im.abs() < 1e-13);
        }
    }

    /// `S(k)` is Hermitian: `S[μ,ν] = conj(S[ν,μ])`, including the cross-cell images.
    #[test]
    fn bloch_overlap_is_hermitian() {
        let basis = two_s_basis();
        // A small cell so several images contribute at this k.
        let cell = Cell::cubic(3.0).unwrap();
        let n = basis.nao();
        let s = bloch_overlap(&basis, &cell, [0.3, -0.1, 0.2], 12.0);
        for i in 0..n {
            for j in 0..n {
                let a = s[i * n + j];
                let b = s[j * n + i];
                assert!(
                    (a - b.conj()).norm() < 1e-12,
                    "S not Hermitian at ({i},{j})"
                );
            }
        }
    }

    /// A small cell with an s and a p shell on two distinct atoms — exercises the spherical
    /// transform and the `l → l+1` raise in the Pulay gradient.
    fn s_p_basis(c0: [f64; 3], c1: [f64; 3]) -> Basis {
        Basis::new(vec![
            Shell::new(0, c0, vec![0.7], vec![1.0]).unwrap(),
            Shell::new_spherical(1, c1, vec![0.5], vec![1.0]).unwrap(),
        ])
    }

    /// `Σ_k w_k Re Tr(O(k) M)` for a real-symmetric weight `m` (row-major `nao²`), the energy
    /// whose analytic gradient [`bloch_grad_1e_contract`] computes.
    fn weighted_trace<F>(
        basis: &Basis,
        cell: &Cell,
        kfracs: &[[f64; 3]],
        weights: &[f64],
        m: &[f64],
        rmax: f64,
        op: F,
    ) -> f64
    where
        F: Fn(&Basis, &Cell, [f64; 3], f64) -> Vec<Complex64>,
    {
        let n = basis.nao();
        let mut e = 0.0;
        for (k, &w) in kfracs.iter().zip(weights) {
            let ok = op(basis, cell, *k, rmax);
            let mut tr = 0.0;
            for mu in 0..n {
                for nu in 0..n {
                    tr += (ok[mu * n + nu] * m[nu * n + mu]).re;
                }
            }
            e += w * tr;
        }
        e
    }

    /// The analytic Bloch kinetic/overlap Pulay gradients match the central finite difference
    /// of `Σ_k w_k Re Tr(O(k;R) M)` with the weight matrix `M` held fixed — the per-term M4
    /// validation. Two non-Γ k-points exercise the phase sum; translational invariance
    /// (`Σ_I g_I = 0`) is checked too.
    #[test]
    fn bloch_pulay_gradients_match_finite_difference() {
        let cell = Cell::cubic(5.0).unwrap();
        let c0 = [1.1, 1.0, 0.9];
        let c1 = [2.6, 2.4, 2.7];
        let basis = s_p_basis(c0, c1);
        let n = basis.nao();
        let rmax = 12.0;
        let kfracs = [[0.0, 0.0, 0.0], [0.3, -0.1, 0.2]];
        let weights = [0.55, 0.45];

        // A fixed real-symmetric weight matrix (a valid Hermitian M).
        let mut m = vec![0.0; n * n];
        for a in 0..n {
            for b in 0..n {
                m[a * n + b] =
                    0.2 * ((a + 1) as f64) + 0.1 * ((b + 1) as f64) + 0.05 * (a * b) as f64;
            }
        }
        for a in 0..n {
            for b in (a + 1)..n {
                let s = 0.5 * (m[a * n + b] + m[b * n + a]);
                m[a * n + b] = s;
                m[b * n + a] = s;
            }
        }
        let m_k: Vec<Vec<Complex64>> = (0..kfracs.len())
            .map(|_| m.iter().map(|&x| Complex64::new(x, 0.0)).collect())
            .collect();

        type BlochOp = fn(&Basis, &Cell, [f64; 3], f64) -> Vec<Complex64>;
        let cases: [(&str, Vec<[f64; 3]>, BlochOp); 2] = [
            (
                "overlap",
                bloch_overlap_grad_contract(&basis, &cell, &kfracs, &weights, &m_k, rmax),
                bloch_overlap as BlochOp,
            ),
            (
                "kinetic",
                bloch_kinetic_grad_contract(&basis, &cell, &kfracs, &weights, &m_k, rmax),
                bloch_kinetic as BlochOp,
            ),
        ];
        for (op_name, grad, op) in cases {
            let h = 1e-5;
            let centers = [c0, c1];
            for atom in 0..2 {
                for axis in 0..3 {
                    let mut cp = centers;
                    cp[atom][axis] += h;
                    let e_plus = weighted_trace(
                        &s_p_basis(cp[0], cp[1]),
                        &cell,
                        &kfracs,
                        &weights,
                        &m,
                        rmax,
                        op,
                    );
                    cp[atom][axis] -= 2.0 * h;
                    let e_minus = weighted_trace(
                        &s_p_basis(cp[0], cp[1]),
                        &cell,
                        &kfracs,
                        &weights,
                        &m,
                        rmax,
                        op,
                    );
                    let fd = (e_plus - e_minus) / (2.0 * h);
                    assert!(
                        (grad[atom][axis] - fd).abs() < 1e-6,
                        "{op_name} atom {atom} axis {axis}: analytic {} vs FD {fd}",
                        grad[atom][axis]
                    );
                }
            }
            // Translational invariance: moving every atom together leaves O(k) unchanged.
            for axis in 0..3 {
                let s: f64 = grad.iter().map(|g| g[axis]).sum();
                assert!(s.abs() < 1e-9, "{op_name} Σ_I g_I[{axis}] = {s} ≠ 0");
            }
        }
    }

    /// Deform a Cartesian vector by `F = I + λ·m`.
    fn deform_vec(m: &[[f64; 3]; 3], lambda: f64, v: [f64; 3]) -> [f64; 3] {
        let mut o = v;
        for (a, oa) in o.iter_mut().enumerate() {
            for (b, &vb) in v.iter().enumerate() {
                *oa += lambda * m[a][b] * vb;
            }
        }
        o
    }

    /// The analytic Bloch kinetic/overlap **stress** contractions match the central finite
    /// difference of `Σ_k w_k Re Tr(O(k;ε) M)` under a strain `F = I + λ·M_dir` (cell and shell
    /// centers deform together), with the weight `M` held fixed — the per-term M5 stress
    /// validation. Checked over uniaxial, shear, and general-symmetric strain directions; two
    /// non-Γ k-points exercise the phase sum.
    #[test]
    fn bloch_pulay_stress_match_finite_difference() {
        let cell = Cell::cubic(5.0).unwrap();
        let c0 = [1.1, 1.0, 0.9];
        let c1 = [2.6, 2.4, 2.7];
        let basis = s_p_basis(c0, c1);
        let n = basis.nao();
        let rmax = 12.0;
        let kfracs = [[0.0, 0.0, 0.0], [0.3, -0.1, 0.2]];
        let weights = [0.55, 0.45];

        let mut m = vec![0.0; n * n];
        for a in 0..n {
            for b in 0..n {
                m[a * n + b] =
                    0.2 * ((a + 1) as f64) + 0.1 * ((b + 1) as f64) + 0.05 * (a * b) as f64;
            }
        }
        for a in 0..n {
            for b in (a + 1)..n {
                let s = 0.5 * (m[a * n + b] + m[b * n + a]);
                m[a * n + b] = s;
                m[b * n + a] = s;
            }
        }
        let m_k: Vec<Vec<Complex64>> = (0..kfracs.len())
            .map(|_| m.iter().map(|&x| Complex64::new(x, 0.0)).collect())
            .collect();

        let dirs = [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.4, 0.2, -0.1], [0.2, -0.3, 0.15], [-0.1, 0.15, 0.5]],
        ];

        type StressFn =
            fn(&Basis, &Cell, &[[f64; 3]], &[f64], &[Vec<Complex64>], f64) -> [[f64; 3]; 3];
        type BlochOp = fn(&Basis, &Cell, [f64; 3], f64) -> Vec<Complex64>;
        let cases: [(&str, StressFn, BlochOp); 2] = [
            ("kinetic", bloch_kinetic_stress_contract, bloch_kinetic),
            ("overlap", bloch_overlap_stress_contract, bloch_overlap),
        ];
        for (name, stress_fn, op) in cases {
            let tau = stress_fn(&basis, &cell, &kfracs, &weights, &m_k, rmax);
            for mdir in dirs {
                let energy = |lambda: f64| -> f64 {
                    let dcell = Cell::from_vectors(
                        deform_vec(&mdir, lambda, cell.vectors().0),
                        deform_vec(&mdir, lambda, cell.vectors().1),
                        deform_vec(&mdir, lambda, cell.vectors().2),
                    )
                    .unwrap();
                    let db =
                        s_p_basis(deform_vec(&mdir, lambda, c0), deform_vec(&mdir, lambda, c1));
                    weighted_trace(&db, &dcell, &kfracs, &weights, &m, rmax, op)
                };
                let h = 1e-5;
                let fd = (energy(h) - energy(-h)) / (2.0 * h);
                let analytic: f64 = (0..3)
                    .flat_map(|a| (0..3).map(move |b| (a, b)))
                    .map(|(a, b)| tau[a][b] * mdir[a][b])
                    .sum();
                assert!(
                    (analytic - fd).abs() < 1e-6,
                    "{name} strain {mdir:?}: analytic {analytic} vs FD {fd}"
                );
            }
        }
    }

    /// The Bloch overlap is periodic in k (k and k + reciprocal-lattice vector give the same
    /// matrix), a basic consistency property.
    #[test]
    fn bloch_overlap_periodic_in_k() {
        let basis = two_s_basis();
        let cell = Cell::cubic(4.0).unwrap();
        let s1 = bloch_overlap(&basis, &cell, [0.25, 0.0, 0.0], 10.0);
        let s2 = bloch_overlap(&basis, &cell, [1.25, 0.0, 0.0], 10.0);
        for (a, b) in s1.iter().zip(&s2) {
            assert!((a - b).norm() < 1e-12);
        }
    }
}
