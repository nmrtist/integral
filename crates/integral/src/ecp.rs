//! Scalar semilocal effective-core-potential (ECP) integrals `⟨μ|U|ν⟩`.
//!
//! An ECP of the def2-ECP type on atom `A` is
//!
//! ```text
//!   U_A(r) = U_L(r) + Σ_{l=0}^{L−1} Σ_{m=−l}^{l} |S_lm⟩ [U_l(r) − U_L(r)] ⟨S_lm|
//! ```
//!
//! with real spherical harmonics `S_lm` centered on `A` and radial expansions
//! `U(r) = Σ_k d_k r^{n_k−2} e^{−ζ_k r²}` ([`EcpPrimitive`]). [`Basis::ecp`]
//! returns the matrix of `Σ_A U_A` over the basis, ready to be added to the
//! core Hamiltonian as-is (attractive local channels carry negative tabulated
//! coefficients; no extra signs are inserted). [`Basis::ecp_grad_contract`]
//! is the density-contracted analytic nuclear gradient of that matrix.
//!
//! ## Method
//!
//! The standard McMurchie–Davidson (1981) / Kahn–Goddard semilocal ECP
//! factorization into **analytic angular** × **radial** integrals is used:
//!
//! - Each Cartesian Gaussian is expanded about the ECP center `C`: the shifted
//!   exponential becomes `e^{k·r̂ r} = 4π Σ_{λμ} M_λ(kr) S_λμ(k̂) S_λμ(r̂)`
//!   with modified spherical Bessel functions `M_λ = i_λ`, and the Cartesian
//!   prefactor becomes a polynomial in `r` and the unit vector `r̂`.
//! - The angular factors `∫ S_λμ(r̂) r̂^t dΩ` (type 1) and
//!   `∫ S_lm S_λμ r̂^t dΩ` (type 2) are evaluated **exactly** from monomial
//!   expansions of the real spherical harmonics and the closed-form
//!   unit-sphere monomial integral.
//! - The radial integrals `Q_λ^N = ∫_0^∞ r^N e^{−αr²} M_λ(kr) dr` (type 1) and
//!   their two-Bessel type-2 analogues are evaluated by a 128-point
//!   Gauss–Legendre rule on the window `[max(0, r₀−w), r₀+w]` with
//!   `r₀ = k/(2α)`, `w = √(80/α)` — outside the window the integrand is below
//!   `e^{−80}` of its peak. The Bessel functions are evaluated in the **scaled**
//!   form `e^{−x} i_λ(x)` (all-positive power series for `x < 100`, upward
//!   recurrence from closed-form `λ = 0, 1` above), and the bra/ket Gaussian
//!   prefactors `e^{−α_a|CA|² − α_b|CB|²}` are folded into the quadrature
//!   exponent, so no factor `e^{+kr}` is ever formed and the path is
//!   overflow/NaN-free for any geometry, including shells sitting exactly on
//!   the ECP center (`k → 0` gives `M_λ(0) = δ_{λ0}` from the series).
//!
//! ## Screening
//!
//! A primitive pair × ECP-primitive contribution is skipped only when the
//! exact peak bound of its radial integrand,
//! `exp(−α_a|CA|² − α_b|CB|² + k²/(4α))`, is below `e^{−120} ≈ 7.7·10⁻⁵³` —
//! far tighter than the documented `1e-16` budget, so screening is
//! conservative by a wide margin.

use std::f64::consts::PI;
use std::sync::OnceLock;

use crate::math::am::cart_components;
use crate::math::norm::double_factorial;

use crate::integrals::{place_block, to_func_1e};
use crate::shell::{Basis, Shell};
use crate::IntegralError;

/// Maximum AO angular momentum supported by [`Basis::ecp_grad_contract`]
/// (`l ≤ 4`, s…g — the range validated by the principle-based gradient tests;
/// the derivative raises the differentiated shell to `l + 1`).
pub const MAX_ECP_GRAD_L: usize = 4;

/// Maximum number of projector channels (`l = 0..=4`) supported by
/// [`Basis::ecp_grad_contract`].
const MAX_ECP_GRAD_PROJ: usize = 5;

/// Skip a primitive contribution when its peak exponent bound is below this
/// (natural-log) threshold: `e^{−120} ≈ 7.7e-53`, conservatively below the
/// documented `1e-16` screening budget.
const ARG_SCREEN: f64 = -120.0;
/// Half-width of the radial quadrature window in units of `1/√α`: the Gaussian
/// factor is `≤ e^{−80}` of its peak outside `r₀ ± √(WINDOW_LOG/α)`.
const WINDOW_LOG: f64 = 80.0;
/// Gauss–Legendre points for the radial integrals.
const N_QUAD: usize = 128;
/// Below this `|k|`, the expansion vector is treated as zero (`k̂ = ẑ`).
const K_TINY: f64 = 1e-12;

const FOUR_PI: f64 = 4.0 * PI;

/// One radial primitive `d · r^(n−2) · exp(−ζ r²)` of an ECP expansion
/// (`n ∈ {0, 1, 2, ...}`).
#[derive(Debug, Clone, PartialEq)]
pub struct EcpPrimitive {
    /// Power `n` of the `r^{n−2}` prefactor (as tabulated; `n ≥ 0`).
    pub n: i32,
    /// Gaussian exponent `ζ` (bohr⁻²), `> 0`.
    pub zeta: f64,
    /// Expansion coefficient `d`, with the tabulated sign.
    pub coef: f64,
}

/// Semilocal ECP on one atom:
/// `U(r) = U_L(r) + Σ_{l<L} Σ_m |lm⟩ (U_l(r) − U_L(r)) ⟨lm|`.
///
/// `local` holds `U_L`; `semilocal[l]` holds the **already-subtracted**
/// `(U_l − U_L)` expansion for projector channel `l` (the def2-ECP tabulation
/// convention), for `l = 0, …, max_l − 1` (`semilocal.len() == max_l`).
/// `n_core` (the number of replaced core electrons) is bookkeeping for the
/// caller and does not enter the integrals.
#[derive(Debug, Clone, PartialEq)]
pub struct Ecp {
    /// Index of the ECP's atom in [`Basis::atoms`].
    pub atom: usize,
    /// Number of core electrons the ECP replaces (caller bookkeeping).
    pub n_core: u32,
    /// The local channel's angular momentum `L`; projectors run `l < max_l`.
    pub max_l: usize,
    /// The local potential `U_L`.
    pub local: Vec<EcpPrimitive>,
    /// Per-channel `(U_l − U_L)` expansions, `l = 0..max_l`.
    pub semilocal: Vec<Vec<EcpPrimitive>>,
}

impl Basis {
    /// ECP matrix `⟨μ| Σ_A U_A |ν⟩`, `nao × nao` row-major (same layout, AO
    /// ordering, and normalization as [`Basis::overlap`]). Contributions from
    /// all listed ECPs are summed; an empty list (or ECPs with empty / all-zero
    /// expansions) yields an exactly-zero matrix. The matrix is exactly
    /// symmetric by construction (the upper triangle is the transpose of the
    /// computed lower triangle).
    ///
    /// See the module docs for the evaluation method, accuracy,
    /// and screening.
    ///
    /// # Panics
    /// Panics if an [`Ecp::atom`] index is out of range of [`Basis::atoms`].
    #[must_use]
    pub fn ecp(&self, ecps: &[Ecp]) -> Vec<f64> {
        let n = self.nao();
        let offs = self.offsets();
        let atoms = self.atoms();
        let shells = self.shells();
        let lsh = shells.iter().map(Shell::l).max().unwrap_or(0);
        let lproj = ecps.iter().map(|e| e.semilocal.len()).max().unwrap_or(0);
        // Spherical-harmonic order needed: 2·l_shell (type 1) and
        // l_channel + l_shell (type 2); the channel harmonics are ≤ that too.
        let lam_max = (2 * lsh).max(lproj.saturating_sub(1) + lsh);
        let rsh = Rsh::new(lam_max);
        let mut mat = vec![0.0; n * n];
        for (si, sa) in shells.iter().enumerate() {
            for (sj, sb) in shells.iter().enumerate().take(si + 1) {
                let mut block = vec![0.0; sa.n_cart() * sb.n_cart()];
                for ecp in ecps {
                    ecp_pair_into(sa, sb, atoms[ecp.atom], ecp, &rsh, &mut block);
                }
                let fa = to_func_1e(block, sa, sb);
                let (nfa, nfb) = (sa.n_func(), sb.n_func());
                place_block(&mut mat, n, offs[si], offs[sj], &fa, nfb);
                if si != sj {
                    let mut ft = vec![0.0; nfa * nfb];
                    for i in 0..nfa {
                        for j in 0..nfb {
                            ft[j * nfa + i] = fa[i * nfb + j];
                        }
                    }
                    place_block(&mut mat, n, offs[sj], offs[si], &ft, nfa);
                }
            }
        }
        mat
    }

    /// Contracted ECP nuclear gradient: per-atom `dE/dR_a` for
    /// `E = Σ_{μν} γ_{μν} ⟨μ| Σ_A U_A |ν⟩`, never materializing the per-atom
    /// derivative matrices.
    ///
    /// Mirrors [`Basis::eri_grad_contract`]: the result is the **raw
    /// derivative** `dE/dR_a` (not the force `−dE/dR_a`), one `[x, y, z]`
    /// triple per atom in [`Basis::atoms`] order, atomic units (hartree/bohr).
    /// `gamma` is the row-major `nao × nao` density (same layout as
    /// [`Basis::ecp`]); a non-symmetric `gamma` is handled exactly — only its
    /// symmetric part contributes, as in the energy expression.
    ///
    /// ## Method
    ///
    /// The bra/ket Gaussian centers are differentiated with the standard shift
    /// relation `∂/∂A_i χ_a = 2α·χ_{a+1_i} − a_i·χ_{a−1_i}`, evaluated through
    /// the same analytic-angular × radial-quadrature machinery as
    /// [`Basis::ecp`] (the raised/lowered blocks reuse the value path with
    /// per-primitive weights `2α·w` / `w`). The ECP-center contribution is
    /// obtained from **translational invariance** per shell-pair × ECP
    /// triplet: `∂/∂C = −(∂/∂A + ∂/∂B)` — exact for these integrals, and it
    /// makes `Σ_a dE/dR_a = 0` hold to round-off by construction (each
    /// contribution is added to a shell's atom and subtracted from the ECP's
    /// atom). Degenerate geometries (shells on the ECP center) need no special
    /// casing — the radial/angular path is NaN-free there.
    ///
    /// ## Screening
    ///
    /// A shell pair is skipped only when its symmetrized density block
    /// `γ + γᵀ` is exactly zero; within a pair the value path's conservative
    /// `e^{−120}` peak-bound primitive screen applies (see the
    /// module docs).
    ///
    /// # Errors
    /// [`IntegralError::AngularMomentumTooHighForGradient`] if any shell has
    /// `l > `[`MAX_ECP_GRAD_L`], or [`IntegralError::GammaLengthMismatch`] if
    /// `gamma.len() != nao²`.
    ///
    /// # Panics
    /// Panics if an [`Ecp::atom`] index is out of range of [`Basis::atoms`]
    /// (as [`Basis::ecp`]), or if an ECP has more than 5 projector channels
    /// (`l > 4` projectors are outside the validated gradient range).
    pub fn ecp_grad_contract(
        &self,
        ecps: &[Ecp],
        gamma: &[f64],
    ) -> Result<Vec<[f64; 3]>, IntegralError> {
        let shells = self.shells();
        for s in shells {
            if s.l() > MAX_ECP_GRAD_L {
                return Err(IntegralError::AngularMomentumTooHighForGradient {
                    l: s.l(),
                    max: MAX_ECP_GRAD_L,
                });
            }
        }
        let n = self.nao();
        if gamma.len() != n * n {
            return Err(IntegralError::GammaLengthMismatch {
                expected: n * n,
                got: gamma.len(),
            });
        }
        let atoms = self.atoms();
        let mut grad = vec![[0.0; 3]; atoms.len()];
        for e in ecps {
            assert!(
                e.atom < atoms.len(),
                "Ecp::atom index {} out of range ({} atoms)",
                e.atom,
                atoms.len()
            );
            assert!(
                e.semilocal.len() <= MAX_ECP_GRAD_PROJ,
                "ecp_grad_contract supports projector channels l <= 4, \
                 got {} channels",
                e.semilocal.len()
            );
        }
        if ecps.is_empty() {
            return Ok(grad);
        }
        let offs = self.offsets();
        let shell_atom = self.shell_atom();
        let lsh = shells.iter().map(Shell::l).max().unwrap_or(0);
        let lproj = ecps.iter().map(|e| e.semilocal.len()).max().unwrap_or(0);
        // As in `ecp`, but the differentiated bra is raised to `l + 1`.
        let lam_max = (2 * (lsh + 1)).max(lproj.saturating_sub(1) + lsh + 1);
        let rsh = Rsh::new(lam_max);
        let wts: Vec<Vec<f64>> = shells
            .iter()
            .map(|s| (0..s.n_prim()).map(|i| s.primitive_coeff(i)).collect())
            .collect();
        for (si, sa) in shells.iter().enumerate() {
            let la = sa.l();
            let a = sa.center();
            let (naf, nca) = (sa.n_func(), sa.n_cart());
            let comps_a = cart_components(la);
            let comps_up = cart_components(la + 1);
            let comps_dn = if la > 0 {
                cart_components(la - 1)
            } else {
                Vec::new()
            };
            // Raised-side weights carry the per-primitive `2α` of the shift
            // relation; the lowered side reuses the value weights.
            let wup: Vec<f64> = sa
                .exponents()
                .iter()
                .zip(&wts[si])
                .map(|(&e, &w)| 2.0 * e * w)
                .collect();
            for (sj, sb) in shells.iter().enumerate() {
                let (nbf, ncb) = (sb.n_func(), sb.n_cart());
                // Symmetrized density block (γ + γᵀ): differentiating only the
                // bra over *ordered* pairs then accounts for both the bra- and
                // ket-center derivatives (U is symmetric, so the ket derivative
                // of (μ, ν) is the bra derivative of (ν, μ)).
                let mut gs = vec![0.0; naf * nbf];
                let mut any = false;
                for i in 0..naf {
                    for j in 0..nbf {
                        let v = gamma[(offs[si] + i) * n + offs[sj] + j]
                            + gamma[(offs[sj] + j) * n + offs[si] + i];
                        any |= v != 0.0;
                        gs[i * nbf + j] = v;
                    }
                }
                if !any {
                    continue;
                }
                let b = sb.center();
                for ecp in ecps {
                    let c = atoms[ecp.atom];
                    let cav = [a[0] - c[0], a[1] - c[1], a[2] - c[2]];
                    let cbv = [b[0] - c[0], b[1] - c[1], b[2] - c[2]];
                    let side_b = EcpSide {
                        l: sb.l(),
                        cv: cbv,
                        exps: sb.exponents(),
                        wts: &wts[sj],
                    };
                    let mut bup = vec![0.0; comps_up.len() * ncb];
                    ecp_sides_into(
                        &EcpSide {
                            l: la + 1,
                            cv: cav,
                            exps: sa.exponents(),
                            wts: &wup,
                        },
                        &side_b,
                        ecp,
                        &rsh,
                        &mut bup,
                    );
                    let bdn = if la > 0 {
                        let mut bd = vec![0.0; comps_dn.len() * ncb];
                        ecp_sides_into(
                            &EcpSide {
                                l: la - 1,
                                cv: cav,
                                exps: sa.exponents(),
                                wts: &wts[si],
                            },
                            &side_b,
                            ecp,
                            &rsh,
                            &mut bd,
                        );
                        bd
                    } else {
                        Vec::new()
                    };
                    for dir in 0..3 {
                        // ∂/∂A_dir block from the shift relation, in the
                        // original `nca × ncb` Cartesian component layout.
                        let mut dcart = vec![0.0; nca * ncb];
                        for (ia, pa) in comps_a.iter().enumerate() {
                            let mut up = *pa;
                            up[dir] += 1;
                            let iu = comps_up
                                .iter()
                                .position(|p| p == &up)
                                .expect("raised Cartesian component");
                            let idn = (pa[dir] > 0).then(|| {
                                let mut dn = *pa;
                                dn[dir] -= 1;
                                comps_dn
                                    .iter()
                                    .position(|p| p == &dn)
                                    .expect("lowered Cartesian component")
                            });
                            for ib in 0..ncb {
                                let mut v = bup[iu * ncb + ib];
                                if let Some(id) = idn {
                                    v -= pa[dir] as f64 * bdn[id * ncb + ib];
                                }
                                dcart[ia * ncb + ib] = v;
                            }
                        }
                        let df = to_func_1e(dcart, sa, sb);
                        let v: f64 = df.iter().zip(&gs).map(|(x, g)| x * g).sum();
                        // Translational invariance: ∂/∂C = −(∂/∂A + ∂/∂B);
                        // every bra contribution is mirrored onto the ECP atom.
                        grad[shell_atom[si]][dir] += v;
                        grad[ecp.atom][dir] -= v;
                    }
                }
            }
        }
        Ok(grad)
    }
}

/// One side (bra or ket) of a primitive-level ECP evaluation: angular
/// momentum, center relative to the ECP center, and per-primitive exponents
/// with their **effective weights** (contraction coefficient × normalization
/// — or any caller-chosen weight, e.g. `2α·w` for the raised derivative side).
struct EcpSide<'a> {
    l: usize,
    cv: [f64; 3],
    exps: &'a [f64],
    wts: &'a [f64],
}

/// Accumulate one ECP's contribution to a contracted Cartesian shell-pair
/// block (`na_cart × nb_cart`, row-major).
fn ecp_pair_into(sa: &Shell, sb: &Shell, c: [f64; 3], ecp: &Ecp, rsh: &Rsh, block: &mut [f64]) {
    let (a, b) = (sa.center(), sb.center());
    let wa: Vec<f64> = (0..sa.n_prim()).map(|i| sa.primitive_coeff(i)).collect();
    let wb: Vec<f64> = (0..sb.n_prim()).map(|i| sb.primitive_coeff(i)).collect();
    let side_a = EcpSide {
        l: sa.l(),
        cv: [a[0] - c[0], a[1] - c[1], a[2] - c[2]],
        exps: sa.exponents(),
        wts: &wa,
    };
    let side_b = EcpSide {
        l: sb.l(),
        cv: [b[0] - c[0], b[1] - c[1], b[2] - c[2]],
        exps: sb.exponents(),
        wts: &wb,
    };
    ecp_sides_into(&side_a, &side_b, ecp, rsh, block);
}

/// Accumulate one ECP's contribution for two primitive sides (the shared core
/// of the value and gradient paths).
fn ecp_sides_into(sa: &EcpSide<'_>, sb: &EcpSide<'_>, ecp: &Ecp, rsh: &Rsh, block: &mut [f64]) {
    type1_into(sa, sb, &ecp.local, rsh, block);
    for (l, prims) in ecp.semilocal.iter().enumerate() {
        if !prims.is_empty() {
            type2_into(l, prims, sa, sb, rsh, block);
        }
    }
}

// ---------------------------------------------------------------------------
// Type-1 (local) integrals  ⟨a| U_L(r_C) |b⟩
// ---------------------------------------------------------------------------

fn type1_into(
    sa: &EcpSide<'_>,
    sb: &EcpSide<'_>,
    prims: &[EcpPrimitive],
    rsh: &Rsh,
    block: &mut [f64],
) {
    if prims.is_empty() {
        return;
    }
    let (la, lb) = (sa.l, sb.l);
    let (cav, cbv) = (sa.cv, sb.cv);
    let ltot = la + lb;
    let d = ltot + 1;
    let comps_a = cart_components(la);
    let comps_b = cart_components(lb);
    let nb = comps_b.len();
    let ca2 = dot(cav, cav);
    let cb2 = dot(cbv, cbv);
    let mut mb = vec![0.0; d];
    for (pi, &aa) in sa.exps.iter().enumerate() {
        let wa = sa.wts[pi];
        for (pj, &ab) in sb.exps.iter().enumerate() {
            let w = wa * sb.wts[pj];
            let arg0 = -aa * ca2 - ab * cb2;
            let kvec = [
                2.0 * (aa * cav[0] + ab * cbv[0]),
                2.0 * (aa * cav[1] + ab * cbv[1]),
                2.0 * (aa * cav[2] + ab * cbv[2]),
            ];
            let kk = dot(kvec, kvec).sqrt();
            let asum = aa + ab;
            if asum > 0.0 && arg0 + kk * kk / (4.0 * asum) < ARG_SCREEN {
                continue;
            }
            let (k, khat) = if kk > K_TINY {
                (kk, [kvec[0] / kk, kvec[1] / kk, kvec[2] / kk])
            } else {
                (0.0, [0.0, 0.0, 1.0])
            };
            let ang = build_ang1(rsh, ltot, khat);
            let v = build_v1(&comps_a, &comps_b, cav, cbv, ltot, &ang);
            for ep in prims {
                if ep.coef == 0.0 {
                    continue;
                }
                let alpha = asum + ep.zeta;
                if alpha <= 0.0 || arg0 + k * k / (4.0 * alpha) < ARG_SCREEN {
                    continue;
                }
                let q = radial_type1(alpha, k, arg0, ep.n, ltot, &mut mb);
                let scale = w * ep.coef * FOUR_PI;
                for ia in 0..comps_a.len() {
                    for ib in 0..nb {
                        let vsl = &v[(ia * nb + ib) * d * d..][..d * d];
                        let mut acc = 0.0;
                        for (vv, qq) in vsl.iter().zip(&q) {
                            if *vv != 0.0 {
                                acc += vv * qq;
                            }
                        }
                        block[ia * nb + ib] += scale * acc;
                    }
                }
            }
        }
    }
}

/// `ang[λ][(tx·d + ty)·d + tz] = Σ_μ S_λμ(k̂) ∫ S_λμ(r̂) r̂^t dΩ` for all
/// `t` with `tx + ty + tz ≤ ltot` (`d = ltot + 1`).
fn build_ang1(rsh: &Rsh, ltot: usize, khat: [f64; 3]) -> Vec<Vec<f64>> {
    let d = ltot + 1;
    let mut ang = vec![vec![0.0; d * d * d]; d];
    for (lam, slot) in ang.iter_mut().enumerate() {
        let lam_i = lam as i32;
        let svals: Vec<f64> = (-lam_i..=lam_i)
            .map(|m| eval_poly(rsh.poly(lam, m), khat))
            .collect();
        for tx in 0..d {
            for ty in 0..(d - tx) {
                for tz in 0..(d - tx - ty) {
                    let ts = tx + ty + tz;
                    // ∫ S_λμ r̂^t dΩ vanishes unless λ ≤ |t| with equal parity.
                    if ts < lam || (ts + lam) % 2 == 1 {
                        continue;
                    }
                    let mut acc = 0.0;
                    for (sv, m) in svals.iter().zip(-lam_i..=lam_i) {
                        if *sv != 0.0 {
                            acc += sv * omega1(rsh.poly(lam, m), [tx, ty, tz]);
                        }
                    }
                    slot[(tx * d + ty) * d + tz] = acc;
                }
            }
        }
    }
    ang
}

/// Per-component-pair contraction of the Cartesian-prefactor expansion with
/// the type-1 angular table: `v[(ia·nb + ib)·d² + λ·d + s] = Σ_{|t|=s} c_t ·
/// ang[λ][t]`, where `c_t` is the binomial expansion of
/// `Π_c (r û_c − CA_c)^{a_c} (r û_c − CB_c)^{b_c}`.
fn build_v1(
    comps_a: &[[usize; 3]],
    comps_b: &[[usize; 3]],
    cav: [f64; 3],
    cbv: [f64; 3],
    ltot: usize,
    ang: &[Vec<f64>],
) -> Vec<f64> {
    let d = ltot + 1;
    let (na, nb) = (comps_a.len(), comps_b.len());
    let mut v = vec![0.0; na * nb * d * d];
    for (ia, pa) in comps_a.iter().enumerate() {
        let cca: Vec<Vec<f64>> = (0..3).map(|c| coord_coeffs(pa[c], cav[c])).collect();
        for (ib, pb) in comps_b.iter().enumerate() {
            let prod: Vec<Vec<f64>> = (0..3)
                .map(|c| convolve(&cca[c], &coord_coeffs(pb[c], cbv[c])))
                .collect();
            let base = (ia * nb + ib) * d * d;
            for (tx, &px) in prod[0].iter().enumerate() {
                for (ty, &py) in prod[1].iter().enumerate() {
                    for (tz, &pz) in prod[2].iter().enumerate() {
                        let cc = px * py * pz;
                        if cc == 0.0 {
                            continue;
                        }
                        let s = tx + ty + tz;
                        let tidx = (tx * d + ty) * d + tz;
                        for (lam, a) in ang.iter().enumerate() {
                            if a[tidx] != 0.0 {
                                v[base + lam * d + s] += cc * a[tidx];
                            }
                        }
                    }
                }
            }
        }
    }
    v
}

/// Type-1 radial integrals
/// `q[λ·(ltot+1) + s] = e^{arg0} ∫_0^∞ r^{n+s} e^{−α r²} M_λ(k r) dr`
/// for `λ, s = 0..=ltot`, by Gauss–Legendre quadrature on the peak window.
fn radial_type1(alpha: f64, k: f64, arg0: f64, n: i32, ltot: usize, mb: &mut [f64]) -> Vec<f64> {
    let d = ltot + 1;
    let mut q = vec![0.0; d * d];
    let (c1, c2) = quad_window(alpha, k);
    for &(x, w) in gl_nodes() {
        let r = c1 * x + c2;
        let e = (arg0 + (k - alpha * r) * r).exp();
        if e == 0.0 {
            continue;
        }
        msb_scaled(k * r, ltot, mb);
        let wr = c1 * w * e;
        let mut rs = r.powi(n);
        for s in 0..d {
            let v = wr * rs;
            for lam in 0..d {
                q[lam * d + s] += v * mb[lam];
            }
            rs *= r;
        }
    }
    q
}

// ---------------------------------------------------------------------------
// Type-2 (semilocal projector) integrals  Σ_m ⟨a|S_lm⟩ U_l ⟨S_lm|b⟩
// ---------------------------------------------------------------------------

fn type2_into(
    l: usize,
    prims: &[EcpPrimitive],
    sa: &EcpSide<'_>,
    sb: &EcpSide<'_>,
    rsh: &Rsh,
    block: &mut [f64],
) {
    let (la, lb) = (sa.l, sb.l);
    let (cav, cbv) = (sa.cv, sb.cv);
    let (lama, lamb) = (l + la, l + lb);
    let comps_a = cart_components(la);
    let comps_b = cart_components(lb);
    let nb = comps_b.len();
    let ra2 = dot(cav, cav);
    let rb2 = dot(cbv, cbv);
    let ra = ra2.sqrt();
    let rb = rb2.sqrt();
    let (ra_eff, khat_a) = if ra > K_TINY {
        (ra, [cav[0] / ra, cav[1] / ra, cav[2] / ra])
    } else {
        (0.0, [0.0, 0.0, 1.0])
    };
    let (rb_eff, khat_b) = if rb > K_TINY {
        (rb, [cbv[0] / rb, cbv[1] / rb, cbv[2] / rb])
    } else {
        (0.0, [0.0, 0.0, 1.0])
    };
    // Angular tables are exponent-independent (k̂ is the CA/CB direction), so
    // they are built once per shell pair and channel.
    let fa = build_f2(rsh, l, la, khat_a);
    let fb = build_f2(rsh, l, lb, khat_b);
    let ga = build_g2(&fa, &comps_a, cav, la, l);
    let gb = build_g2(&fb, &comps_b, cbv, lb, l);
    let nm = 2 * l + 1;
    let smax = la + lb;
    let scale16 = 16.0 * PI * PI;
    let mut mba = vec![0.0; lama + 1];
    let mut mbb = vec![0.0; lamb + 1];
    for (pi, &aa) in sa.exps.iter().enumerate() {
        let wa = sa.wts[pi];
        for (pj, &ab) in sb.exps.iter().enumerate() {
            let w = wa * sb.wts[pj];
            let ka = 2.0 * aa * ra_eff;
            let kb = 2.0 * ab * rb_eff;
            let ksum = ka + kb;
            let arg0 = -aa * ra2 - ab * rb2;
            let asum = aa + ab;
            if asum > 0.0 && arg0 + ksum * ksum / (4.0 * asum) < ARG_SCREEN {
                continue;
            }
            for ep in prims {
                if ep.coef == 0.0 {
                    continue;
                }
                let alpha = asum + ep.zeta;
                if alpha <= 0.0 || arg0 + ksum * ksum / (4.0 * alpha) < ARG_SCREEN {
                    continue;
                }
                let q = radial_type2(
                    alpha, ka, kb, arg0, ep.n, smax, lama, lamb, &mut mba, &mut mbb,
                );
                let scale = w * ep.coef * scale16;
                for ia in 0..comps_a.len() {
                    for ib in 0..nb {
                        let mut acc = 0.0;
                        for mi in 0..nm {
                            for laa in 0..=lama {
                                for pa in 0..=la {
                                    let g = ga[((ia * nm + mi) * (lama + 1) + laa) * (la + 1) + pa];
                                    if g == 0.0 {
                                        continue;
                                    }
                                    for lbb in 0..=lamb {
                                        for pb in 0..=lb {
                                            let h = gb[((ib * nm + mi) * (lamb + 1) + lbb)
                                                * (lb + 1)
                                                + pb];
                                            if h != 0.0 {
                                                acc += g
                                                    * h
                                                    * q[(laa * (lamb + 1) + lbb) * (smax + 1)
                                                        + pa
                                                        + pb];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        block[ia * nb + ib] += scale * acc;
                    }
                }
            }
        }
    }
}

/// Type-2 angular table for one side:
/// `f[mi][λ][(tx·d + ty)·d + tz] = Σ_μ S_λμ(k̂) ∫ S_lm S_λμ r̂^t dΩ`
/// for `|t| ≤ lside` (`d = lside + 1`, `mi = m + l`).
#[allow(clippy::needless_range_loop)] // `lam` indexes the *inner* table dimension
fn build_f2(rsh: &Rsh, l: usize, lside: usize, khat: [f64; 3]) -> Vec<Vec<Vec<f64>>> {
    let d = lside + 1;
    let lam_n = l + lside + 1;
    let l_i = l as i32;
    let mut f = vec![vec![vec![0.0; d * d * d]; lam_n]; 2 * l + 1];
    for lam in 0..lam_n {
        let lam_i = lam as i32;
        let svals: Vec<f64> = (-lam_i..=lam_i)
            .map(|m| eval_poly(rsh.poly(lam, m), khat))
            .collect();
        for tx in 0..d {
            for ty in 0..(d - tx) {
                for tz in 0..(d - tx - ty) {
                    let ts = tx + ty + tz;
                    // ∫ S_lm S_λμ r̂^t dΩ vanishes outside the triangle
                    // |l − |t|| ≤ λ ≤ l + |t| with parity l + |t| + λ even.
                    if lam + ts < l || lam > l + ts || (l + ts + lam) % 2 == 1 {
                        continue;
                    }
                    let tidx = (tx * d + ty) * d + tz;
                    for mm in -l_i..=l_i {
                        let mut acc = 0.0;
                        for (sv, mu) in svals.iter().zip(-lam_i..=lam_i) {
                            if *sv != 0.0 {
                                acc +=
                                    sv * omega2(rsh.poly(l, mm), rsh.poly(lam, mu), [tx, ty, tz]);
                            }
                        }
                        f[(mm + l_i) as usize][lam][tidx] = acc;
                    }
                }
            }
        }
    }
    f
}

/// Contract one side's angular table with its component binomial expansion:
/// `g[((ic·nm + mi)·lam_n + λ)·(lside+1) + p] = Σ_{|t|=p} c_t(ic) f[mi][λ][t]`.
fn build_g2(
    f: &[Vec<Vec<f64>>],
    comps: &[[usize; 3]],
    cv: [f64; 3],
    lside: usize,
    l: usize,
) -> Vec<f64> {
    let d = lside + 1;
    let nm = 2 * l + 1;
    let lam_n = l + lside + 1;
    let mut g = vec![0.0; comps.len() * nm * lam_n * d];
    for (ic, p) in comps.iter().enumerate() {
        let cc: Vec<Vec<f64>> = (0..3).map(|c| coord_coeffs(p[c], cv[c])).collect();
        for (tx, &px) in cc[0].iter().enumerate() {
            for (ty, &py) in cc[1].iter().enumerate() {
                for (tz, &pz) in cc[2].iter().enumerate() {
                    let coef = px * py * pz;
                    if coef == 0.0 {
                        continue;
                    }
                    let pw = tx + ty + tz;
                    let tidx = (tx * d + ty) * d + tz;
                    for (mi, fm) in f.iter().enumerate() {
                        for (lam, fl) in fm.iter().enumerate() {
                            if fl[tidx] != 0.0 {
                                g[((ic * nm + mi) * lam_n + lam) * d + pw] += coef * fl[tidx];
                            }
                        }
                    }
                }
            }
        }
    }
    g
}

/// Type-2 radial integrals
/// `q[(λa·(lamb+1) + λb)·(smax+1) + s] = e^{arg0} ∫_0^∞ r^{n+s} e^{−α r²}
/// M_λa(ka r) M_λb(kb r) dr`.
#[allow(clippy::too_many_arguments)]
fn radial_type2(
    alpha: f64,
    ka: f64,
    kb: f64,
    arg0: f64,
    n: i32,
    smax: usize,
    lama: usize,
    lamb: usize,
    mba: &mut [f64],
    mbb: &mut [f64],
) -> Vec<f64> {
    let mut q = vec![0.0; (lama + 1) * (lamb + 1) * (smax + 1)];
    let ksum = ka + kb;
    let (c1, c2) = quad_window(alpha, ksum);
    for &(x, w) in gl_nodes() {
        let r = c1 * x + c2;
        let e = (arg0 + (ksum - alpha * r) * r).exp();
        if e == 0.0 {
            continue;
        }
        msb_scaled(ka * r, lama, mba);
        msb_scaled(kb * r, lamb, mbb);
        let wr = c1 * w * e;
        let mut rs = r.powi(n);
        for s in 0..=smax {
            let v = wr * rs;
            for laa in 0..=lama {
                let va = v * mba[laa];
                for lbb in 0..=lamb {
                    q[(laa * (lamb + 1) + lbb) * (smax + 1) + s] += va * mbb[lbb];
                }
            }
            rs *= r;
        }
    }
    q
}

// ---------------------------------------------------------------------------
// Shared numerical helpers
// ---------------------------------------------------------------------------

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Quadrature window for `∫ r^N e^{−αr² + kr} … dr`: scale/offset `(c1, c2)`
/// mapping GL nodes on `[−1, 1]` to `[max(0, r₀−w), r₀+w]`, `r₀ = k/(2α)`,
/// `w = √(WINDOW_LOG/α)`.
fn quad_window(alpha: f64, k: f64) -> (f64, f64) {
    let r0 = k / (2.0 * alpha);
    let hw = (WINDOW_LOG / alpha).sqrt();
    let lo = (r0 - hw).max(0.0);
    let hi = r0 + hw;
    (0.5 * (hi - lo), 0.5 * (hi + lo))
}

/// Binomial-expansion coefficients of `(r û − c)^p` in powers of `r û`:
/// `out[i] = C(p, i)·(−c)^{p−i}`.
fn coord_coeffs(p: usize, c: f64) -> Vec<f64> {
    (0..=p)
        .map(|i| binom(p, i) * (-c).powi((p - i) as i32))
        .collect()
}

fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        for (j, &y) in b.iter().enumerate() {
            out[i + j] += x * y;
        }
    }
    out
}

fn binom(n: usize, k: usize) -> f64 {
    let mut acc = 1.0;
    for i in 0..k {
        acc = acc * ((n - i) as f64) / ((i + 1) as f64);
    }
    acc
}

fn factorial(n: usize) -> f64 {
    (1..=n).map(|i| i as f64).product()
}

/// Scaled modified spherical Bessel functions `out[λ] = e^{−x} i_λ(x)` for
/// `λ = 0..=lmax`, `x ≥ 0`. All-positive power series (no cancellation) below
/// `x = 100`; upward recurrence from the closed forms `λ = 0, 1` above (stable
/// there since `x ≫ λ`). `x = 0` yields `δ_{λ0}` exactly.
fn msb_scaled(x: f64, lmax: usize, out: &mut [f64]) {
    debug_assert!(x >= 0.0);
    if x < 100.0 {
        let emx = (-x).exp();
        let x2 = x * x;
        for (lam, slot) in out.iter_mut().enumerate().take(lmax + 1) {
            let mut term = 1.0 / double_factorial(2 * lam as i64 + 1);
            let mut sum = term;
            let mut t = 0usize;
            while term > 1e-18 * sum && t < 500 {
                term *= x2 / (((2 * t + 2) * (2 * (lam + t) + 3)) as f64);
                sum += term;
                t += 1;
            }
            *slot = x.powi(lam as i32) * sum * emx;
        }
    } else {
        let e2 = (-2.0 * x).exp();
        out[0] = (1.0 - e2) / (2.0 * x);
        if lmax >= 1 {
            out[1] = ((x - 1.0) + (x + 1.0) * e2) / (2.0 * x * x);
        }
        for lam in 2..=lmax {
            out[lam] = out[lam - 2] - (2 * lam - 1) as f64 / x * out[lam - 1];
        }
    }
}

/// Gauss–Legendre nodes/weights on `[−1, 1]` by Newton iteration on `P_n`.
fn gauss_legendre(n: usize) -> Vec<(f64, f64)> {
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

fn gl_nodes() -> &'static [(f64, f64)] {
    static GL: OnceLock<Vec<(f64, f64)>> = OnceLock::new();
    GL.get_or_init(|| gauss_legendre(N_QUAD))
}

// ---------------------------------------------------------------------------
// Real spherical harmonics as unit-sphere monomial expansions
// ---------------------------------------------------------------------------

/// Monomial expansion `Σ c · x̂^i ŷ^j ẑ^k` of a function on the unit sphere.
type Poly = Vec<(f64, [usize; 3])>;

/// Orthonormal real spherical harmonics `S_{l m}` as unit-sphere monomial
/// expansions, for `l = 0..=lmax`. Built from Legendre-polynomial derivatives
/// (`π_{lm}(ẑ) = d^m P_l/dẑ^m`) and the Chebyshev expansion of
/// `Re/Im[(x̂ + iŷ)^m]`; any fixed orthonormal convention is valid here, since
/// only `Σ_m |S_lm⟩⟨S_lm|` (convention-invariant) enters the integrals.
struct Rsh {
    /// `tables[l][(m + l)]` — monomial expansion of `S_{l m}`.
    tables: Vec<Vec<Poly>>,
}

impl Rsh {
    fn new(lmax: usize) -> Self {
        let mut tables = Vec::with_capacity(lmax + 1);
        for l in 0..=lmax {
            let pl = legendre_coeffs(l);
            let l_i = l as i32;
            tables.push((-l_i..=l_i).map(|m| rsh_poly(l, m, &pl)).collect());
        }
        Rsh { tables }
    }

    fn poly(&self, l: usize, m: i32) -> &Poly {
        &self.tables[l][(m + l as i32) as usize]
    }
}

/// Coefficients of the Legendre polynomial `P_l(z)`; `c[p]` multiplies `z^p`.
fn legendre_coeffs(l: usize) -> Vec<f64> {
    let mut c = vec![0.0; l + 1];
    let scale = 0.5f64.powi(l as i32);
    for k in 0..=(l / 2) {
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        c[l - 2 * k] = scale * sign * binom(l, k) * binom(2 * l - 2 * k, l);
    }
    c
}

/// Monomial expansion of the real spherical harmonic `S_{l m}` on the unit
/// sphere, given the Legendre coefficients of `P_l`.
fn rsh_poly(l: usize, m: i32, pl: &[f64]) -> Poly {
    let ma = m.unsigned_abs() as usize;
    // π_{l,|m|}(z) = d^|m| P_l / dz^|m| (associated Legendre without the
    // (1 − z²)^{|m|/2} factor and without Condon–Shortley).
    let mut pi_lm = pl.to_vec();
    for _ in 0..ma {
        let mut der = vec![0.0; pi_lm.len().saturating_sub(1)];
        for (p, slot) in der.iter_mut().enumerate() {
            *slot = pi_lm[p + 1] * (p + 1) as f64;
        }
        pi_lm = der;
    }
    let mut out = Poly::new();
    if m == 0 {
        let nrm = ((2 * l + 1) as f64 / (4.0 * PI)).sqrt();
        for (p, &cz) in pi_lm.iter().enumerate() {
            if cz != 0.0 {
                out.push((nrm * cz, [0, 0, p]));
            }
        }
        return out;
    }
    let nrm = ((2 * l + 1) as f64 / (2.0 * PI) * factorial(l - ma) / factorial(l + ma)).sqrt();
    // sin^|m|θ · cos/sin(|m|φ) = Re/Im[(x̂ + iŷ)^|m|].
    let p_start = usize::from(m < 0);
    for p in (p_start..=ma).step_by(2) {
        let sign = if (p / 2) % 2 == 0 { 1.0 } else { -1.0 };
        let ct = sign * binom(ma, p);
        for (q, &cz) in pi_lm.iter().enumerate() {
            if cz != 0.0 {
                out.push((nrm * ct * cz, [ma - p, p, q]));
            }
        }
    }
    out
}

/// `∫_{S²} x̂^i ŷ^j ẑ^k dΩ`: zero unless all powers are even, else
/// `4π (i−1)!!(j−1)!!(k−1)!! / (i+j+k+1)!!`.
fn sphere_monomial(i: usize, j: usize, k: usize) -> f64 {
    if i % 2 == 1 || j % 2 == 1 || k % 2 == 1 {
        return 0.0;
    }
    FOUR_PI
        * double_factorial(i as i64 - 1)
        * double_factorial(j as i64 - 1)
        * double_factorial(k as i64 - 1)
        / double_factorial((i + j + k) as i64 + 1)
}

/// `∫ S_λμ(r̂) r̂^t dΩ` from the monomial expansion.
fn omega1(p: &Poly, t: [usize; 3]) -> f64 {
    p.iter()
        .map(|&(c, m)| c * sphere_monomial(t[0] + m[0], t[1] + m[1], t[2] + m[2]))
        .sum()
}

/// `∫ S_lm(r̂) S_λμ(r̂) r̂^t dΩ` from the two monomial expansions.
fn omega2(p1: &Poly, p2: &Poly, t: [usize; 3]) -> f64 {
    let mut acc = 0.0;
    for &(c1, m1) in p1 {
        for &(c2, m2) in p2 {
            let i = t[0] + m1[0] + m2[0];
            let j = t[1] + m1[1] + m2[1];
            let k = t[2] + m1[2] + m2[2];
            if i % 2 == 0 && j % 2 == 0 && k % 2 == 0 {
                acc += c1 * c2 * sphere_monomial(i, j, k);
            }
        }
    }
    acc
}

fn eval_poly(p: &Poly, u: [f64; 3]) -> f64 {
    p.iter()
        .map(|&(c, m)| c * u[0].powi(m[0] as i32) * u[1].powi(m[1] as i32) * u[2].powi(m[2] as i32))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The generated real spherical harmonics are orthonormal on the sphere:
    /// `∫ S_{l1 m1} S_{l2 m2} dΩ = δ` — checked exactly through the same
    /// monomial-integral primitive the angular factors use.
    #[test]
    fn rsh_orthonormal() {
        let rsh = Rsh::new(8);
        for l1 in 0..=8usize {
            for l2 in 0..=8usize {
                for m1 in -(l1 as i32)..=l1 as i32 {
                    for m2 in -(l2 as i32)..=l2 as i32 {
                        let s = omega2(rsh.poly(l1, m1), rsh.poly(l2, m2), [0, 0, 0]);
                        let expect = if l1 == l2 && m1 == m2 { 1.0 } else { 0.0 };
                        assert!((s - expect).abs() < 1e-12, "⟨{l1},{m1}|{l2},{m2}⟩ = {s}");
                    }
                }
            }
        }
    }

    /// Gauss–Legendre sanity: exact for low-order polynomials.
    #[test]
    fn gauss_legendre_integrates_polynomials() {
        for &n in &[8usize, 64, 128] {
            let gl = gauss_legendre(n);
            let s0: f64 = gl.iter().map(|&(_, w)| w).sum();
            let s2: f64 = gl.iter().map(|&(x, w)| w * x * x).sum();
            let s6: f64 = gl.iter().map(|&(x, w)| w * x.powi(6)).sum();
            assert!((s0 - 2.0).abs() < 1e-14, "n={n} Σw={s0}");
            assert!((s2 - 2.0 / 3.0).abs() < 1e-14, "n={n}");
            assert!((s6 - 2.0 / 7.0).abs() < 1e-14, "n={n}");
        }
    }

    /// Scaled Bessel: `δ_{λ0}` at `x = 0`, the three-term recurrence holds in
    /// the series branch, and the series/recurrence branches agree at the
    /// `x = 100` switchover.
    #[test]
    fn scaled_bessel_consistency() {
        let mut m = [0.0; 13];
        msb_scaled(0.0, 12, &mut m);
        assert_eq!(m[0], 1.0);
        for &v in &m[1..] {
            assert_eq!(v, 0.0);
        }
        // recurrence identity i_{λ−1} − i_{λ+1} = (2λ+1)/x · i_λ at x = 5.
        msb_scaled(5.0, 12, &mut m);
        for lam in 1..12 {
            let lhs = m[lam - 1] - m[lam + 1];
            let rhs = (2 * lam + 1) as f64 / 5.0 * m[lam];
            assert!((lhs - rhs).abs() < 1e-15 * m[0], "λ={lam}");
        }
        // series branch (x = 90 < 100) vs the closed-form upward recurrence
        // the x ≥ 100 branch uses, evaluated at the same x.
        let x = 90.0_f64;
        let mut series = [0.0; 13];
        msb_scaled(x, 12, &mut series);
        let e2 = (-2.0 * x).exp();
        let mut rec = [0.0; 13];
        rec[0] = (1.0 - e2) / (2.0 * x);
        rec[1] = ((x - 1.0) + (x + 1.0) * e2) / (2.0 * x * x);
        for lam in 2..=12 {
            rec[lam] = rec[lam - 2] - (2 * lam - 1) as f64 / x * rec[lam - 1];
        }
        for lam in 0..=12 {
            assert!(
                (series[lam] - rec[lam]).abs() < 1e-12 * rec[lam].abs(),
                "λ={lam}: {} vs {}",
                series[lam],
                rec[lam]
            );
        }
        // closed form for λ = 0: e^{−x} sinh(x)/x = (1 − e^{−2x})/(2x).
        for &x in &[0.3, 4.0, 40.0, 99.0, 150.0] {
            msb_scaled(x, 2, &mut m);
            let exact = (1.0 - (-2.0 * x).exp()) / (2.0 * x);
            assert!((m[0] - exact).abs() < 1e-14, "x={x}");
        }
    }

    /// Type-1 radial quadrature vs an independent composite-Simpson evaluation
    /// of the same integral, including `k = 0` and a sharp exponent.
    #[test]
    fn radial_type1_matches_simpson() {
        let mut mb = vec![0.0; 5];
        for &(alpha, k) in &[(1.7_f64, 0.0_f64), (1.7, 3.0), (24.0, 11.0), (0.4, 1.2)] {
            let q = radial_type1(alpha, k, 0.0, 1, 4, &mut mb);
            // Simpson on [0, 40] with 80000 panels; scaled Bessel via series.
            for lam in 0..=4usize {
                for s in 0..=4usize {
                    let f = |r: f64| {
                        let mut b = vec![0.0; lam + 1];
                        msb_scaled(k * r, lam, &mut b);
                        r.powi(1 + s as i32) * ((k - alpha * r) * r).exp() * b[lam]
                    };
                    let n = 80_000;
                    let h = 40.0 / n as f64;
                    let mut acc = f(0.0) + f(40.0);
                    for i in 1..n {
                        let w = if i % 2 == 1 { 4.0 } else { 2.0 };
                        acc += w * f(i as f64 * h);
                    }
                    acc *= h / 3.0;
                    let got = q[lam * 5 + s];
                    assert!(
                        (got - acc).abs() < 1e-10 * (1.0 + acc.abs()),
                        "alpha={alpha} k={k} λ={lam} s={s}: {got} vs {acc}"
                    );
                }
            }
        }
    }

    /// Unit-sphere monomial integrals: a few closed-form values.
    #[test]
    fn sphere_monomial_closed_forms() {
        assert!((sphere_monomial(0, 0, 0) - FOUR_PI).abs() < 1e-14);
        assert!((sphere_monomial(2, 0, 0) - FOUR_PI / 3.0).abs() < 1e-14);
        assert!((sphere_monomial(2, 2, 0) - FOUR_PI / 15.0).abs() < 1e-15);
        assert!((sphere_monomial(4, 0, 0) - FOUR_PI / 5.0).abs() < 1e-14);
        assert_eq!(sphere_monomial(1, 2, 0), 0.0);
        assert_eq!(sphere_monomial(2, 3, 1), 0.0);
    }
}
