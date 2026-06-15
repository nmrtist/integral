//! Geometric (nuclear-coordinate) first-derivative builders.
//!
//! These produce the **per-atom** first derivatives of the integral matrices /
//! tensors with respect to nuclear coordinates — the ingredients for energy
//! gradients (forces). Each derivative is assembled from shifted-angular-momentum
//! *value* integrals via the Gaussian center-derivative relation
//! ([`crate::engine::deriv`]); the recurrence engines are reused, not extended.
//! The density-fitting families get density-contracted gradients
//! ([`Basis::eri_3c_grad_contract`], [`Basis::eri_2c_grad_contract`]) via the
//! same zero-exponent unit-`s` dummy as the value builders in [`crate::df`] —
//! the dummy is the constant function `1`, so only the real centers are
//! differentiated (the dummy's center derivative is exactly zero and is never
//! evaluated).
//!
//! ## What "atom" means
//!
//! Derivatives are grouped by **distinct shell center** ([`Basis::atoms`], in
//! first-appearance order). Atom `c`'s gradient block is the sum of the
//! basis-function derivatives of every shell sitting on `c` (and, for the nuclear
//! attraction, the Hellmann–Feynman term of the charge on `c`).
//!
//! ## Convention
//!
//! [`Gradient1e`]/[`GradientEri`] hold `∂O/∂R_c` — the derivative of the integral
//! with respect to moving atom `c`. The full molecular sum `Σ_c ∂O/∂R_c` is zero
//! (translational invariance); see [`Gradient1e::max_translational_residual`].
//! The sign/center convention relating the center derivative to the bra-gradient
//! block is documented above.
//!
//! ## Angular-momentum limit
//!
//! The center-derivative raises the differentiated shell to `l + 1`, so gradients
//! require every shell to have `l ≤ MAX_L − 1` (the raised shell then stays
//! within the engines' validated `MAX_L`). Builders return
//! [`IntegralError::AngularMomentumTooHighForGradient`] otherwise.

use crate::engine::deriv::{accumulate_center_derivative, AxisDeriv};
use crate::engine::os::{self, Prim, Vec3, MAX_L};
use crate::engine::os_eri::{self, ShellRef};
use crate::engine::rys;
use crate::math::am::n_cart;

use crate::df::unit_s;
use crate::integrals::{check_erf_omega, to_func_1e, to_func_eri, Engine, EriKernel};
use crate::shell::{Basis, Shell};
use crate::IntegralError;

/// Maximum shell angular momentum for which a gradient can be built. The
/// derivative raises the shell to `l + 1`, which must stay `≤ MAX_L`.
pub const MAX_GRAD_L: usize = MAX_L - 1;

/// Per-atom geometric gradient of a one-electron matrix.
///
/// Layout: dense row-major `f64`, shape `[natom, 3, nao, nao]`. Atom order is
/// [`Basis::atoms`]; axis order is `x, y, z`; the trailing `nao × nao` is the
/// usual one-electron matrix (same ordering/strides as the value builders, with
/// `nao = `[`Basis::nao`]). Use [`Gradient1e::block`] for one `(atom, axis)`
/// matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Gradient1e {
    natom: usize,
    nao: usize,
    data: Vec<f64>,
}

impl Gradient1e {
    fn zeros(natom: usize, nao: usize) -> Self {
        Gradient1e {
            natom,
            nao,
            data: vec![0.0; natom * 3 * nao * nao],
        }
    }

    /// Number of atoms (distinct shell centers).
    #[must_use]
    pub fn natom(&self) -> usize {
        self.natom
    }

    /// Matrix dimension `nao`.
    #[must_use]
    pub fn nao(&self) -> usize {
        self.nao
    }

    /// The `nao × nao` derivative matrix `∂O/∂(R_atom)_axis` (row-major),
    /// `axis ∈ {0,1,2}` for `x,y,z`.
    #[must_use]
    pub fn block(&self, atom: usize, axis: usize) -> &[f64] {
        let nn = self.nao * self.nao;
        let off = (atom * 3 + axis) * nn;
        &self.data[off..off + nn]
    }

    fn block_mut(&mut self, atom: usize, axis: usize) -> &mut [f64] {
        let nn = self.nao * self.nao;
        let off = (atom * 3 + axis) * nn;
        &mut self.data[off..off + nn]
    }

    /// Largest `|Σ_atom ∂O/∂R_atom|` over all elements and axes — the
    /// translational-invariance residual, exactly zero in infinite precision.
    #[must_use]
    pub fn max_translational_residual(&self) -> f64 {
        let nn = self.nao * self.nao;
        let mut worst = 0.0_f64;
        for axis in 0..3 {
            for e in 0..nn {
                let mut s = 0.0;
                for c in 0..self.natom {
                    s += self.data[(c * 3 + axis) * nn + e];
                }
                worst = worst.max(s.abs());
            }
        }
        worst
    }
}

/// Per-atom geometric gradient of the electron-repulsion tensor.
///
/// Layout: dense row-major `f64`, shape `[natom, 3, nao, nao, nao, nao]`. Atom
/// order is [`Basis::atoms`]; axis order is `x, y, z`; the trailing `nao⁴` is the
/// usual ERI tensor (same ordering/strides as [`Basis::eri`]). Use
/// [`GradientEri::block`] for one `(atom, axis)` tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct GradientEri {
    natom: usize,
    nao: usize,
    data: Vec<f64>,
}

impl GradientEri {
    fn zeros(natom: usize, nao: usize) -> Self {
        let nao4 = nao * nao * nao * nao;
        GradientEri {
            natom,
            nao,
            data: vec![0.0; natom * 3 * nao4],
        }
    }

    /// Number of atoms (distinct shell centers).
    #[must_use]
    pub fn natom(&self) -> usize {
        self.natom
    }

    /// Tensor dimension `nao`.
    #[must_use]
    pub fn nao(&self) -> usize {
        self.nao
    }

    /// The `nao⁴` derivative tensor `∂(ij|kl)/∂(R_atom)_axis` (row-major),
    /// `axis ∈ {0,1,2}`.
    #[must_use]
    pub fn block(&self, atom: usize, axis: usize) -> &[f64] {
        let nn = self.nao.pow(4);
        let off = (atom * 3 + axis) * nn;
        &self.data[off..off + nn]
    }

    /// Largest `|Σ_atom ∂(ij|kl)/∂R_atom|` over all elements and axes — the
    /// translational-invariance residual.
    #[must_use]
    pub fn max_translational_residual(&self) -> f64 {
        let nn = self.nao.pow(4);
        let mut worst = 0.0_f64;
        for axis in 0..3 {
            for e in 0..nn {
                let mut s = 0.0;
                for c in 0..self.natom {
                    s += self.data[(c * 3 + axis) * nn + e];
                }
                worst = worst.max(s.abs());
            }
        }
        worst
    }
}

/// `outer`/`inner` strides of index `pos` of a 4-index block with the given
/// per-index Cartesian dimensions.
fn eri_outer_inner(pos: usize, dims: [usize; 4]) -> (usize, usize) {
    match pos {
        0 => (1, dims[1] * dims[2] * dims[3]),
        1 => (dims[0], dims[2] * dims[3]),
        2 => (dims[0] * dims[1], dims[3]),
        _ => (dims[0] * dims[1] * dims[2], 1),
    }
}

/// `outer`/`inner` strides of index `pos` of a 2-index (bra=0, ket=1) block.
fn pair_outer_inner(pos: usize, na: usize, nb: usize) -> (usize, usize) {
    if pos == 0 {
        (1, nb)
    } else {
        (na, 1)
    }
}

impl Basis {
    fn check_grad_l(&self) -> Result<(), IntegralError> {
        for s in self.shells() {
            if s.l() > MAX_GRAD_L {
                return Err(IntegralError::AngularMomentumTooHighForGradient {
                    l: s.l(),
                    max: MAX_GRAD_L,
                });
            }
        }
        Ok(())
    }

    /// Per-atom gradient of the overlap matrix, `∂S/∂R_c`.
    ///
    /// # Errors
    /// [`IntegralError::AngularMomentumTooHighForGradient`] if any shell has
    /// `l > MAX_GRAD_L`.
    pub fn overlap_grad(&self) -> Result<Gradient1e, IntegralError> {
        self.grad_1e(os::overlap_into)
    }

    /// Per-atom gradient of the kinetic-energy matrix, `∂T/∂R_c`.
    ///
    /// # Errors
    /// As [`Basis::overlap_grad`].
    pub fn kinetic_grad(&self) -> Result<Gradient1e, IntegralError> {
        self.grad_1e(os::kinetic_into)
    }

    /// Per-atom gradient of the nuclear-attraction matrix, `∂V/∂R_c`, for the
    /// point charges `charges = [(center, Z)]`.
    ///
    /// Includes both the basis-function derivatives and the operator
    /// (Hellmann–Feynman) term: the `1/|r−C|` operator depends on the charge
    /// position `C`, so moving the atom carrying charge `C` contributes
    /// `∂_C ⟨a|V_C|b⟩`. By the exact single-charge translational identity
    /// `∂_C = −(∂_A + ∂_B)`, this term is assembled from the same basis-center
    /// derivatives, placed on the charge's atom.
    ///
    /// # Errors
    /// [`IntegralError::AngularMomentumTooHighForGradient`] as above, or
    /// [`IntegralError::ChargeNotOnAtom`] if a charge center is not a basis atom.
    pub fn nuclear_grad(&self, charges: &[(Vec3, f64)]) -> Result<Gradient1e, IntegralError> {
        self.check_grad_l()?;
        let nao = self.nao();
        let atoms = self.atoms();
        let satom = self.shell_atom();
        let offs = self.offsets();
        let shells = self.shells();

        // Each charge must sit on a basis atom so its HF term has a home.
        let mut charge_atom = Vec::with_capacity(charges.len());
        for (c, _) in charges {
            match self.atom_at(*c) {
                Some(idx) => charge_atom.push(idx),
                None => return Err(IntegralError::ChargeNotOnAtom { center: *c }),
            }
        }

        let mut g = Gradient1e::zeros(atoms.len(), nao);
        for (si, sa) in shells.iter().enumerate() {
            for (sj, sb) in shells.iter().enumerate() {
                // Sum the per-charge single-charge derivatives. The full basis
                // derivative is Σ_charge of the single-charge one; the HF term
                // for the charge on atom cX is −(∂_A+∂_B) of that single-charge
                // integral, placed on cX.
                for (ci, &(cc, cz)) in charges.iter().enumerate() {
                    let one = [(cc, cz)];
                    let (da, db) = pair_grad_1e(sa, sb, |a, b, scale, out| {
                        os::nuclear_into(a, b, &one, scale, out);
                    });
                    let cx = charge_atom[ci];
                    let nbf = sb.n_func();
                    let (ri, ci_) = (offs[si], offs[sj]);
                    for axis in 0..3 {
                        let fa = to_func_1e(da[axis].clone(), sa, sb);
                        let fb = to_func_1e(db[axis].clone(), sa, sb);
                        let pa = |atom| Place1e {
                            atom,
                            axis,
                            row_off: ri,
                            col_off: ci_,
                        };
                        // Basis-function derivatives on the shells' own atoms.
                        add_block_1e(&mut g, pa(satom[si]), nbf, &fa, 1.0);
                        add_block_1e(&mut g, pa(satom[sj]), nbf, &fb, 1.0);
                        // Hellmann–Feynman term −(∂_A+∂_B) on the charge's atom.
                        add_block_1e(&mut g, pa(cx), nbf, &fa, -1.0);
                        add_block_1e(&mut g, pa(cx), nbf, &fb, -1.0);
                    }
                }
            }
        }
        Ok(g)
    }

    /// Shared 1e gradient driver for operators that do not depend on nuclear
    /// position (overlap, kinetic): only the basis-function derivatives appear.
    fn grad_1e<F>(&self, eval: F) -> Result<Gradient1e, IntegralError>
    where
        F: Fn(Prim, Prim, f64, &mut [f64]),
    {
        self.check_grad_l()?;
        let nao = self.nao();
        let atoms = self.atoms();
        let satom = self.shell_atom();
        let offs = self.offsets();
        let shells = self.shells();

        let mut g = Gradient1e::zeros(atoms.len(), nao);
        for (si, sa) in shells.iter().enumerate() {
            for (sj, sb) in shells.iter().enumerate() {
                let (da, db) = pair_grad_1e(sa, sb, &eval);
                let nbf = sb.n_func();
                let (ri, ci) = (offs[si], offs[sj]);
                for axis in 0..3 {
                    let fa = to_func_1e(da[axis].clone(), sa, sb);
                    let fb = to_func_1e(db[axis].clone(), sa, sb);
                    let pa = |atom| Place1e {
                        atom,
                        axis,
                        row_off: ri,
                        col_off: ci,
                    };
                    add_block_1e(&mut g, pa(satom[si]), nbf, &fa, 1.0);
                    add_block_1e(&mut g, pa(satom[sj]), nbf, &fb, 1.0);
                }
            }
        }
        Ok(g)
    }

    /// Per-atom gradient of the electron-repulsion tensor, `∂(ij|kl)/∂R_c`.
    ///
    /// Uses the dispatch policy ([`Engine::Auto`]); see [`Basis::eri_grad_with`]
    /// to force an engine.
    ///
    /// # Errors
    /// [`IntegralError::AngularMomentumTooHighForGradient`] if any shell has
    /// `l > MAX_GRAD_L`.
    pub fn eri_grad(&self) -> Result<GradientEri, IntegralError> {
        self.eri_grad_with(Engine::Auto)
    }

    /// Like [`Basis::eri_grad`] but forces a specific [`Engine`]. Both engines
    /// produce the same gradient to tolerance; forcing exists so tests exercise
    /// each derivative path on the same quartets.
    ///
    /// # Errors
    /// As [`Basis::eri_grad`].
    pub fn eri_grad_with(&self, engine: Engine) -> Result<GradientEri, IntegralError> {
        self.check_grad_l()?;
        let nao = self.nao();
        let atoms = self.atoms();
        let satom = self.shell_atom();
        let offs = self.offsets();
        let shells = self.shells();

        let mut g = GradientEri::zeros(atoms.len(), nao);
        for (si, sa) in shells.iter().enumerate() {
            for (sj, sb) in shells.iter().enumerate() {
                for (sk, sc) in shells.iter().enumerate() {
                    for (sl, sd) in shells.iter().enumerate() {
                        let grads = quartet_grad_eri(engine, sa, sb, sc, sd);
                        let shells4 = [sa, sb, sc, sd];
                        let atoms4 = [satom[si], satom[sj], satom[sk], satom[sl]];
                        let offs4 = [offs[si], offs[sj], offs[sk], offs[sl]];
                        for (pos, axes) in grads.iter().enumerate() {
                            for (axis, blk) in axes.iter().enumerate() {
                                let f = to_func_eri(blk.clone(), sa, sb, sc, sd);
                                add_block_eri(
                                    &mut g,
                                    atoms4[pos],
                                    axis,
                                    offs4,
                                    [
                                        shells4[0].n_func(),
                                        shells4[1].n_func(),
                                        shells4[2].n_func(),
                                        shells4[3].n_func(),
                                    ],
                                    &f,
                                );
                            }
                        }
                    }
                }
            }
        }
        Ok(g)
    }

    /// Contraction of the ERI geometric derivative with a two-particle density
    /// `gamma`, never materializing the `nao⁴` derivative tensor:
    ///
    /// ```text
    ///   F_c = Σ_{μνλσ} Γ_{μνλσ} · ∂(μν|λσ)/∂R_c        (one [x, y, z] per atom)
    /// ```
    ///
    /// `gamma` uses the same row-major `nao⁴` layout (and hence the same 8-fold
    /// symmetry convention) as [`Basis::eri`]. The result equals contracting
    /// `gamma` against each [`GradientEri::block`] of [`Basis::eri_grad`], but
    /// peak memory is one shell-quartet block instead of `natom·3·nao⁴`. Atom
    /// order is [`Basis::atoms`].
    ///
    /// Uses the dispatch policy ([`Engine::Auto`]); see
    /// [`Basis::eri_grad_contract_with`] to force an engine.
    ///
    /// # Errors
    /// [`IntegralError::AngularMomentumTooHighForGradient`] if any shell has
    /// `l > MAX_GRAD_L`, or [`IntegralError::GammaLengthMismatch`] if
    /// `gamma.len() != nao⁴`.
    pub fn eri_grad_contract(&self, gamma: &[f64]) -> Result<Vec<[f64; 3]>, IntegralError> {
        self.eri_grad_contract_with(Engine::Auto, gamma)
    }

    /// Like [`Basis::eri_grad_contract`] but forces a specific [`Engine`].
    /// Both engines produce the same contraction to tolerance.
    ///
    /// # Errors
    /// As [`Basis::eri_grad_contract`].
    pub fn eri_grad_contract_with(
        &self,
        engine: Engine,
        gamma: &[f64],
    ) -> Result<Vec<[f64; 3]>, IntegralError> {
        self.check_grad_l()?;
        let nao = self.nao();
        let nao4 = nao.pow(4);
        if gamma.len() != nao4 {
            return Err(IntegralError::GammaLengthMismatch {
                expected: nao4,
                got: gamma.len(),
            });
        }
        let atoms = self.atoms();
        let satom = self.shell_atom();
        let offs = self.offsets();
        let shells = self.shells();

        let mut forces = vec![[0.0_f64; 3]; atoms.len()];
        for (si, sa) in shells.iter().enumerate() {
            for (sj, sb) in shells.iter().enumerate() {
                for (sk, sc) in shells.iter().enumerate() {
                    for (sl, sd) in shells.iter().enumerate() {
                        let grads = quartet_grad_eri(engine, sa, sb, sc, sd);
                        let atoms4 = [satom[si], satom[sj], satom[sk], satom[sl]];
                        let offs4 = [offs[si], offs[sj], offs[sk], offs[sl]];
                        let n4 = [sa.n_func(), sb.n_func(), sc.n_func(), sd.n_func()];
                        for (pos, axes) in grads.iter().enumerate() {
                            for (axis, blk) in axes.iter().enumerate() {
                                let f = to_func_eri(blk.clone(), sa, sb, sc, sd);
                                forces[atoms4[pos]][axis] +=
                                    dot_block_eri(gamma, nao, offs4, n4, &f);
                            }
                        }
                    }
                }
            }
        }
        Ok(forces)
    }

    /// Like [`Basis::eri_grad_contract`] but over the chosen [`EriKernel`] —
    /// the density-contracted geometric ERI derivative of the long-range
    /// (range-separated) operator, for forces of RSH SCF energies.
    ///
    /// - [`EriKernel::Coulomb`] routes to [`Basis::eri_grad_contract`] itself:
    ///   the output is **bit-identical**.
    /// - [`EriKernel::Erf`]`{ omega }` evaluates the derivative of
    ///   `erf(ω·r₁₂)/r₁₂` integrals on the Rys engine. The attenuation enters
    ///   **only** through the 0th-order kernel (`F_m → F_m^ω`, realized as the
    ///   root/weight transform `x → s·x(sT)`, `w → √s·w(sT)` with
    ///   `s = ω²/(ρ+ω²)`); the Gaussian center-derivative relation
    ///   `∂/∂A χ = 2α·χ_{l+1} − l·χ_{l−1}` acts on the basis functions, not the
    ///   two-electron operator, so the transform commutes with the gradient
    ///   structure and the Coulomb recurrences are reused unchanged
    ///   (Gill & Adamson, *Chem. Phys. Lett.* **261**, 105 (1996); Ahlrichs,
    ///   *Phys. Chem. Chem. Phys.* **8**, 3072 (2006)).
    ///
    /// Same units, layout (`Vec<[f64; 3]>` in [`Basis::atoms`] order), `gamma`
    /// convention, and `l ≤ MAX_GRAD_L` limit as [`Basis::eri_grad_contract`].
    ///
    /// # Errors
    /// As [`Basis::eri_grad_contract`].
    ///
    /// # Panics
    /// Panics if `k` is `Erf { omega }` with `ω ≤ 0`, NaN, or infinite.
    pub fn eri_grad_contract_kernel(
        &self,
        gamma: &[f64],
        k: EriKernel,
    ) -> Result<Vec<[f64; 3]>, IntegralError> {
        let omega = match k {
            EriKernel::Coulomb => return self.eri_grad_contract(gamma),
            EriKernel::Erf { omega } => {
                check_erf_omega(omega);
                omega
            }
        };
        self.check_grad_l()?;
        let nao = self.nao();
        let nao4 = nao.pow(4);
        if gamma.len() != nao4 {
            return Err(IntegralError::GammaLengthMismatch {
                expected: nao4,
                got: gamma.len(),
            });
        }
        let atoms = self.atoms();
        let satom = self.shell_atom();
        let offs = self.offsets();
        let shells = self.shells();

        let mut forces = vec![[0.0_f64; 3]; atoms.len()];
        for (si, sa) in shells.iter().enumerate() {
            for (sj, sb) in shells.iter().enumerate() {
                for (sk, sc) in shells.iter().enumerate() {
                    for (sl, sd) in shells.iter().enumerate() {
                        let grads = quartet_grad_eri_erf(omega, sa, sb, sc, sd);
                        let atoms4 = [satom[si], satom[sj], satom[sk], satom[sl]];
                        let offs4 = [offs[si], offs[sj], offs[sk], offs[sl]];
                        let n4 = [sa.n_func(), sb.n_func(), sc.n_func(), sd.n_func()];
                        for (pos, axes) in grads.iter().enumerate() {
                            for (axis, blk) in axes.iter().enumerate() {
                                let f = to_func_eri(blk.clone(), sa, sb, sc, sd);
                                forces[atoms4[pos]][axis] +=
                                    dot_block_eri(gamma, nao, offs4, n4, &f);
                            }
                        }
                    }
                }
            }
        }
        Ok(forces)
    }

    /// Density-contracted geometric gradient of the **3-center** density-fitting
    /// Coulomb integrals `(μν|P)`:
    ///
    /// ```text
    ///   F_c = Σ_{μν,P} Γ_{μν,P} · ∂(μν|P)/∂R_c        (one [x, y, z] per atom)
    /// ```
    ///
    /// `μν` run over `self` (the orbital basis) and `P` over `aux`; `gamma` is
    /// row-major `[μ, ν, P]` with `P` fastest — the same layout as the dense
    /// [`Eri3cBuilder::build`](crate::Eri3cBuilder::build) tensor, length `nao²·naux`. No symmetry of
    /// `gamma` is assumed (the full `μν` sweep is contracted; a `μν`-symmetric
    /// `gamma` simply works).
    ///
    /// **Atom convention:** `aux` must sit on exactly the atoms of `self`
    /// ([`Basis::atoms`] lists bitwise equal, including order — the practical
    /// RI case where the fitting basis shares the orbital basis's centers).
    /// The result has one `[x, y, z]` per shared atom, in [`Basis::atoms`]
    /// order, and sums to zero over atoms (translational invariance).
    ///
    /// The derivative is taken via the same zero-exponent unit-`s` dummy
    /// construction as [`Basis::eri_3c_block`]; only the three **real** centers
    /// (`μ`, `ν`, `P`) are differentiated — the dummy is the constant function
    /// `1`, which does not move with any atom, so its center derivative is
    /// exactly zero by construction (it is never evaluated). Uses the
    /// [`Engine::Auto`] dispatch policy.
    ///
    /// # Errors
    /// - [`IntegralError::AngularMomentumTooHighForGradient`] if any shell of
    ///   `self` or `aux` has `l > MAX_GRAD_L`.
    /// - [`IntegralError::ChargeNotOnAtom`] if `aux.atoms() != self.atoms()`
    ///   (reused as the atom-mismatch error — the enum is exhaustive, so no new
    ///   variant is added; `center` is the first non-shared / out-of-order
    ///   atom).
    /// - [`IntegralError::GammaLengthMismatch`] if
    ///   `gamma.len() != nao²·naux`.
    pub fn eri_3c_grad_contract(
        &self,
        aux: &Basis,
        gamma: &[f64],
    ) -> Result<Vec<[f64; 3]>, IntegralError> {
        self.check_grad_l()?;
        aux.check_grad_l()?;
        let atoms = self.atoms();
        let aux_atoms = aux.atoms();
        if aux_atoms != atoms {
            return Err(IntegralError::ChargeNotOnAtom {
                center: first_mismatched_atom(&atoms, &aux_atoms),
            });
        }
        let nao = self.nao();
        let naux = aux.nao();
        let expected = nao * nao * naux;
        if gamma.len() != expected {
            return Err(IntegralError::GammaLengthMismatch {
                expected,
                got: gamma.len(),
            });
        }
        let satom = self.shell_atom();
        let aatom = aux.shell_atom();
        let offs = self.offsets();
        let aoffs = aux.offsets();
        let shells = self.shells();

        let mut forces = vec![[0.0_f64; 3]; atoms.len()];
        for (si, sa) in shells.iter().enumerate() {
            for (sj, sb) in shells.iter().enumerate() {
                for (sp, sx) in aux.shells().iter().enumerate() {
                    let dummy = unit_s(sx.center());
                    let quartet = [sa, sb, sx, &dummy];
                    let atoms3 = [satom[si], satom[sj], aatom[sp]];
                    let off3 = [offs[si], offs[sj], aoffs[sp]];
                    let n3 = [sa.n_func(), sb.n_func(), sx.n_func()];
                    // Differentiate the three real centers only; the dummy
                    // (pos 3) has exactly zero derivative and is skipped.
                    for pos in 0..3 {
                        let axes = quartet_grad_eri_pos(Engine::Auto, quartet, pos);
                        for (axis, blk) in axes.into_iter().enumerate() {
                            let f = to_func_eri(blk, sa, sb, sx, &dummy);
                            forces[atoms3[pos]][axis] +=
                                dot_block_3c(gamma, nao, naux, off3, n3, &f);
                        }
                    }
                }
            }
        }
        Ok(forces)
    }

    /// Density-contracted geometric gradient of the **2-center** density-fitting
    /// Coulomb metric `(P|Q)` over `self` as the auxiliary basis:
    ///
    /// ```text
    ///   F_c = Σ_{PQ} γ_{PQ} · ∂(P|Q)/∂R_c             (one [x, y, z] per atom)
    /// ```
    ///
    /// `gamma` is row-major `[naux, naux]` (`naux = `[`Basis::nao`]), the same
    /// layout as [`Basis::eri_2c`]. The contraction is the **full** double sum
    /// `Σ_{PQ} γ_{PQ} ∂(P|Q)` with **no implicit symmetrization** — an
    /// asymmetric `gamma` is contracted as given (`(P|Q) = (Q|P)`, so a caller
    /// holding only a triangle should symmetrize first).
    ///
    /// Both centers `P` and `Q` are differentiated with the same
    /// center-derivative recurrences as [`Basis::eri_grad_contract`]; the two
    /// zero-exponent unit-`s` dummies are constant functions with exactly zero
    /// derivative and are never differentiated. Result is per atom in
    /// [`Basis::atoms`] order and sums to zero over atoms. Uses the
    /// [`Engine::Auto`] dispatch policy.
    ///
    /// # Errors
    /// [`IntegralError::AngularMomentumTooHighForGradient`] if any shell has
    /// `l > MAX_GRAD_L`, or [`IntegralError::GammaLengthMismatch`] if
    /// `gamma.len() != naux²`.
    pub fn eri_2c_grad_contract(&self, gamma: &[f64]) -> Result<Vec<[f64; 3]>, IntegralError> {
        self.check_grad_l()?;
        let naux = self.nao();
        let expected = naux * naux;
        if gamma.len() != expected {
            return Err(IntegralError::GammaLengthMismatch {
                expected,
                got: gamma.len(),
            });
        }
        let atoms = self.atoms();
        let satom = self.shell_atom();
        let offs = self.offsets();
        let shells = self.shells();
        let dummies: Vec<Shell> = shells.iter().map(|s| unit_s(s.center())).collect();

        let mut forces = vec![[0.0_f64; 3]; atoms.len()];
        for (p, sp) in shells.iter().enumerate() {
            for (q, sq) in shells.iter().enumerate() {
                let quartet = [sp, &dummies[p], sq, &dummies[q]];
                let off2 = [offs[p], offs[q]];
                let n2 = [sp.n_func(), sq.n_func()];
                // Differentiate the two real centers (quartet positions 0 and
                // 2); the dummies (positions 1 and 3) have exactly zero
                // derivative and are skipped.
                for (pos, atom) in [(0, satom[p]), (2, satom[q])] {
                    let axes = quartet_grad_eri_pos(Engine::Auto, quartet, pos);
                    for (axis, blk) in axes.into_iter().enumerate() {
                        let f = to_func_eri(blk, sp, &dummies[p], sq, &dummies[q]);
                        forces[atom][axis] += dot_block_2c(gamma, naux, off2, n2, &f);
                    }
                }
            }
        }
        Ok(forces)
    }
}

/// Bra- and ket-center derivative Cartesian blocks `([∂A_x,∂A_y,∂A_z],
/// [∂B_x,∂B_y,∂B_z])` of one shell pair for a one-electron operator `eval`
/// (`eval(prim_a, prim_b, scale, out)` accumulates `scale·⟨a|O|b⟩`). Each block
/// is row-major `n_cart(la) × n_cart(lb)`.
///
/// Exposed to the crate so the periodic Bloch 1e-gradients ([`crate::periodic::bloch`])
/// can reuse the exact molecular center-derivative blocks per lattice image.
pub(crate) fn pair_grad_1e<F>(sa: &Shell, sb: &Shell, eval: F) -> ([Vec<f64>; 3], [Vec<f64>; 3])
where
    F: Fn(Prim, Prim, f64, &mut [f64]),
{
    let (la, lb) = (sa.l(), sb.l());
    let (na, nb) = (sa.n_cart(), sb.n_cart());
    let mut da: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; na * nb]);
    let mut db: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; na * nb]);

    for pi in 0..sa.n_prim() {
        for pj in 0..sb.n_prim() {
            let alpha = sa.exponents()[pi];
            let beta = sb.exponents()[pj];
            let scale = sa.primitive_coeff(pi) * sb.primitive_coeff(pj);
            let (a, b) = (sa.center(), sb.center());

            // --- bra (center A): raise/lower la ---
            let mut raised = vec![0.0; n_cart(la + 1) * nb];
            eval(
                Prim::new(alpha, a, la + 1),
                Prim::new(beta, b, lb),
                2.0 * alpha,
                &mut raised,
            );
            let lowered = (la > 0).then(|| {
                let mut t = vec![0.0; n_cart(la - 1) * nb];
                eval(
                    Prim::new(alpha, a, la - 1),
                    Prim::new(beta, b, lb),
                    1.0,
                    &mut t,
                );
                t
            });
            let (outer, inner) = pair_outer_inner(0, na, nb);
            for (axis, slot) in da.iter_mut().enumerate() {
                accumulate_center_derivative(
                    AxisDeriv {
                        l: la,
                        cart_axis: axis,
                        outer,
                        inner,
                    },
                    scale,
                    &raised,
                    lowered.as_deref(),
                    slot,
                );
            }

            // --- ket (center B): raise/lower lb ---
            let mut raised = vec![0.0; na * n_cart(lb + 1)];
            eval(
                Prim::new(alpha, a, la),
                Prim::new(beta, b, lb + 1),
                2.0 * beta,
                &mut raised,
            );
            let lowered = (lb > 0).then(|| {
                let mut t = vec![0.0; na * n_cart(lb - 1)];
                eval(
                    Prim::new(alpha, a, la),
                    Prim::new(beta, b, lb - 1),
                    1.0,
                    &mut t,
                );
                t
            });
            let (outer, inner) = pair_outer_inner(1, na, nb);
            for (axis, slot) in db.iter_mut().enumerate() {
                accumulate_center_derivative(
                    AxisDeriv {
                        l: lb,
                        cart_axis: axis,
                        outer,
                        inner,
                    },
                    scale,
                    &raised,
                    lowered.as_deref(),
                    slot,
                );
            }
        }
    }
    (da, db)
}

/// The four center-derivative Cartesian blocks `[∂A, ∂B, ∂C, ∂D]` (each
/// `[x,y,z]`) of one ERI shell quartet. Each block is row-major
/// `n_cart(la)·n_cart(lb)·n_cart(lc)·n_cart(ld)`.
fn quartet_grad_eri(
    engine: Engine,
    sa: &Shell,
    sb: &Shell,
    sc: &Shell,
    sd: &Shell,
) -> [[Vec<f64>; 3]; 4] {
    quartet_grad_eri_with([sa, sb, sc, sd], |shells, pos| {
        eri_pos_blocks(engine, shells, pos)
    })
}

/// As [`quartet_grad_eri`], for the erf-attenuated kernel `erf(ω·r₁₂)/r₁₂`
/// (Rys engine only). The attenuation modifies only the 0th-order two-electron
/// kernel (`F_m → F_m^ω` via the per-primitive root/weight transform — ρ is
/// fixed within a primitive quartet, so `s = ω²/(ρ+ω²)` is a per-quartet
/// constant), while the center-derivative relation differentiates the basis
/// functions; the two therefore commute and the Coulomb derivative structure
/// is reused verbatim (Gill & Adamson, CPL **261**, 105 (1996); Ahlrichs,
/// PCCP **8**, 3072 (2006)).
fn quartet_grad_eri_erf(
    omega: f64,
    sa: &Shell,
    sb: &Shell,
    sc: &Shell,
    sd: &Shell,
) -> [[Vec<f64>; 3]; 4] {
    quartet_grad_eri_with([sa, sb, sc, sd], |shells, pos| {
        eri_pos_blocks_erf(shells, pos, omega)
    })
}

/// Shared center-derivative combiner: applies the raise/lower relation to the
/// per-position `(raised, lowered)` blocks produced by `blocks`.
fn quartet_grad_eri_with<F>(shells: [&Shell; 4], blocks: F) -> [[Vec<f64>; 3]; 4]
where
    F: Fn([&Shell; 4], usize) -> (Vec<f64>, Option<Vec<f64>>),
{
    std::array::from_fn(|pos| {
        let (raised, lowered) = blocks(shells, pos);
        combine_center_derivative(shells, pos, &raised, lowered.as_deref())
    })
}

/// `[x, y, z]` center-derivative blocks of ERI index `pos`, combined from a
/// `2α`-weighted raised block and an optional lowered block at the quartet's
/// own Cartesian dimensions.
fn combine_center_derivative(
    shells: [&Shell; 4],
    pos: usize,
    raised: &[f64],
    lowered: Option<&[f64]>,
) -> [Vec<f64>; 3] {
    let dims = [
        shells[0].n_cart(),
        shells[1].n_cart(),
        shells[2].n_cart(),
        shells[3].n_cart(),
    ];
    let nblk = dims[0] * dims[1] * dims[2] * dims[3];
    let (outer, inner) = eri_outer_inner(pos, dims);
    let mut out: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; nblk]);
    for (axis, slot) in out.iter_mut().enumerate() {
        accumulate_center_derivative(
            AxisDeriv {
                l: shells[pos].l(),
                cart_axis: axis,
                outer,
                inner,
            },
            1.0,
            raised,
            lowered,
            slot,
        );
    }
    out
}

/// Coulomb center-derivative `[x, y, z]` blocks of a single ERI index `pos`
/// (engine-dispatched) — the per-position counterpart of [`quartet_grad_eri`],
/// used by the density-fitting gradients to differentiate only the **real**
/// centers of a dummy-bearing quartet.
fn quartet_grad_eri_pos(engine: Engine, shells: [&Shell; 4], pos: usize) -> [Vec<f64>; 3] {
    let (raised, lowered) = eri_pos_blocks(engine, shells, pos);
    combine_center_derivative(shells, pos, &raised, lowered.as_deref())
}

/// Contracted, `2α`-weighted **raised** block and the plain **lowered** block of
/// ERI index `pos`, used by [`quartet_grad_eri`]. Both already include the full
/// contraction over all four shells, so the combiner applies them with
/// `scale = 1`.
fn eri_pos_blocks(engine: Engine, shells: [&Shell; 4], pos: usize) -> (Vec<f64>, Option<Vec<f64>>) {
    let l = shells[pos].l();
    // Cartesian dims with index `pos` raised / lowered.
    let dims_with = |lp: usize| {
        let mut d = [
            shells[0].n_cart(),
            shells[1].n_cart(),
            shells[2].n_cart(),
            shells[3].n_cart(),
        ];
        d[pos] = n_cart(lp);
        d
    };
    let len_with = |lp: usize| {
        let d = dims_with(lp);
        d[0] * d[1] * d[2] * d[3]
    };

    let resolved = match engine {
        Engine::Auto => crate::integrals::select_engine(
            shells[0].l() + shells[1].l(),
            shells[2].l() + shells[3].l(),
            shells[0].n_prim() * shells[1].n_prim() * shells[2].n_prim() * shells[3].n_prim(),
        ),
        forced => forced,
    };

    let mut raised = vec![0.0; len_with(l + 1)];
    let mut lowered = (l > 0).then(|| vec![0.0; len_with(l - 1)]);

    match resolved {
        Engine::OsHgp => {
            // Build the (raised) block with `pos`'s contraction folding 2α per
            // primitive, and the (lowered) block with the plain contraction.
            let eff: Vec<Vec<f64>> = shells
                .iter()
                .map(|s| (0..s.n_prim()).map(|i| s.primitive_coeff(i)).collect())
                .collect();
            let raised_coeffs: Vec<f64> = (0..shells[pos].n_prim())
                .map(|i| shells[pos].primitive_coeff(i) * 2.0 * shells[pos].exponents()[i])
                .collect();

            os_eri_shifted(shells, pos, l + 1, &eff, &raised_coeffs, &mut raised);
            if let Some(lo) = lowered.as_mut() {
                os_eri_shifted(shells, pos, l - 1, &eff, &eff[pos], lo);
            }
        }
        _ => {
            // Rys: accumulate over primitive quartets, folding 2α into the
            // raised block's scale.
            rys_shifted(shells, pos, l + 1, true, None, &mut raised);
            if let Some(lo) = lowered.as_mut() {
                rys_shifted(shells, pos, l - 1, false, None, lo);
            }
        }
    }
    (raised, lowered)
}

/// As [`eri_pos_blocks`] for the erf-attenuated kernel — Rys engine only (the
/// attenuated quadrature transform lives there; see [`quartet_grad_eri_erf`]).
fn eri_pos_blocks_erf(shells: [&Shell; 4], pos: usize, omega: f64) -> (Vec<f64>, Option<Vec<f64>>) {
    let l = shells[pos].l();
    let len_with = |lp: usize| {
        let mut d = [
            shells[0].n_cart(),
            shells[1].n_cart(),
            shells[2].n_cart(),
            shells[3].n_cart(),
        ];
        d[pos] = n_cart(lp);
        d[0] * d[1] * d[2] * d[3]
    };

    let mut raised = vec![0.0; len_with(l + 1)];
    rys_shifted(shells, pos, l + 1, true, Some(omega), &mut raised);
    let lowered = (l > 0).then(|| {
        let mut lo = vec![0.0; len_with(l - 1)];
        rys_shifted(shells, pos, l - 1, false, Some(omega), &mut lo);
        lo
    });
    (raised, lowered)
}

/// One OS/HGP contracted shell-quartet evaluation with shell `pos` at angular
/// momentum `lp` and contraction coefficients `pos_coeffs` (the other shells use
/// their effective coeffs `eff`). Accumulates into `out`.
fn os_eri_shifted(
    shells: [&Shell; 4],
    pos: usize,
    lp: usize,
    eff: &[Vec<f64>],
    pos_coeffs: &[f64],
    out: &mut [f64],
) {
    let make = |i: usize| {
        let s = shells[i];
        let (l, coeffs) = if i == pos {
            (lp, pos_coeffs)
        } else {
            (s.l(), eff[i].as_slice())
        };
        ShellRef {
            center: s.center(),
            l,
            exps: s.exponents(),
            coeffs,
        }
    };
    os_eri::coulomb_shell_into(make(0), make(1), make(2), make(3), out);
}

/// Rys per-primitive accumulation of the shell quartet with shell `pos` at
/// angular momentum `lp`; if `weight_2alpha`, fold `2·α_pos` into each primitive
/// quartet's scale (the raised block). `attenuation = None` is the Coulomb
/// kernel; `Some(ω)` the long-range `erf(ω·r₁₂)/r₁₂` kernel — the same
/// shifted-angular-momentum evaluation over the attenuated primitive engine
/// ([`rys::erf_coulomb_into`]; valid per primitive quartet since ρ, and hence
/// `s = ω²/(ρ+ω²)`, is fixed within it). Accumulates into `out`.
fn rys_shifted(
    shells: [&Shell; 4],
    pos: usize,
    lp: usize,
    weight_2alpha: bool,
    attenuation: Option<f64>,
    out: &mut [f64],
) {
    let lvals = [shells[0].l(), shells[1].l(), shells[2].l(), shells[3].l()];
    let np = [
        shells[0].n_prim(),
        shells[1].n_prim(),
        shells[2].n_prim(),
        shells[3].n_prim(),
    ];
    for pa in 0..np[0] {
        for pb in 0..np[1] {
            for pc in 0..np[2] {
                for pd in 0..np[3] {
                    let p = [pa, pb, pc, pd];
                    let mut scale = 1.0;
                    for i in 0..4 {
                        scale *= shells[i].primitive_coeff(p[i]);
                    }
                    if weight_2alpha {
                        scale *= 2.0 * shells[pos].exponents()[p[pos]];
                    }
                    let prim = |i: usize| {
                        let l = if i == pos { lp } else { lvals[i] };
                        Prim::new(shells[i].exponents()[p[i]], shells[i].center(), l)
                    };
                    match attenuation {
                        None => rys::coulomb_into(prim(0), prim(1), prim(2), prim(3), scale, out),
                        Some(omega) => rys::erf_coulomb_into(
                            prim(0),
                            prim(1),
                            prim(2),
                            prim(3),
                            omega,
                            scale,
                            out,
                        ),
                    }
                }
            }
        }
    }
}

/// Dot `block` (row-major over the four function-space component indices,
/// dims `n`) against the matching `gamma` slots at AO offsets `off` — the
/// contraction counterpart of [`add_block_eri`].
fn dot_block_eri(gamma: &[f64], nao: usize, off: [usize; 4], n: [usize; 4], block: &[f64]) -> f64 {
    let mut acc = 0.0;
    for a in 0..n[0] {
        for b in 0..n[1] {
            for c in 0..n[2] {
                for d in 0..n[3] {
                    let src = ((a * n[1] + b) * n[2] + c) * n[3] + d;
                    let i = off[0] + a;
                    let j = off[1] + b;
                    let k = off[2] + c;
                    let l = off[3] + d;
                    acc += gamma[((i * nao + j) * nao + k) * nao + l] * block[src];
                }
            }
        }
    }
    acc
}

/// Dot a 3-center derivative block (row-major `[n_i, n_j, n_p]`, `P` fastest —
/// the trailing unit dummy dimension collapses away) against the matching
/// `gamma` slots of a row-major `[nao, nao, naux]` coefficient array.
fn dot_block_3c(
    gamma: &[f64],
    nao: usize,
    naux: usize,
    off: [usize; 3],
    n: [usize; 3],
    block: &[f64],
) -> f64 {
    let mut acc = 0.0;
    for a in 0..n[0] {
        for b in 0..n[1] {
            for p in 0..n[2] {
                let src = (a * n[1] + b) * n[2] + p;
                let g = ((off[0] + a) * nao + off[1] + b) * naux + off[2] + p;
                acc += gamma[g] * block[src];
            }
        }
    }
    acc
}

/// Dot a 2-center derivative block (row-major `[n_p, n_q]` — both unit dummy
/// dimensions collapse away) against the matching `gamma` slots of a row-major
/// `[naux, naux]` coefficient array.
fn dot_block_2c(gamma: &[f64], naux: usize, off: [usize; 2], n: [usize; 2], block: &[f64]) -> f64 {
    let mut acc = 0.0;
    for a in 0..n[0] {
        for b in 0..n[1] {
            acc += gamma[(off[0] + a) * naux + off[1] + b] * block[a * n[1] + b];
        }
    }
    acc
}

/// The first atom that witnesses `aux.atoms() != main.atoms()`: an atom present
/// in only one list, else the first positional (ordering) difference.
fn first_mismatched_atom(main: &[[f64; 3]], aux: &[[f64; 3]]) -> [f64; 3] {
    if let Some(c) = aux.iter().find(|c| !main.contains(c)) {
        return *c;
    }
    if let Some(c) = main.iter().find(|c| !aux.contains(c)) {
        return *c;
    }
    aux.iter()
        .zip(main)
        .find(|(x, y)| x != y)
        .map(|(x, _)| *x)
        .unwrap_or([f64::NAN; 3])
}

/// Where to scatter a derivative block: target `(atom, axis)` matrix and the
/// AO offsets of the block.
#[derive(Clone, Copy)]
struct Place1e {
    atom: usize,
    axis: usize,
    row_off: usize,
    col_off: usize,
}

/// Accumulate `factor · block` (row-major `na × nb`) into the `(atom, axis)`
/// matrix of `g` at `(row_off, col_off)`.
fn add_block_1e(g: &mut Gradient1e, p: Place1e, nb: usize, block: &[f64], factor: f64) {
    let nao = g.nao;
    let na = block.len() / nb;
    let mat = g.block_mut(p.atom, p.axis);
    for i in 0..na {
        for j in 0..nb {
            mat[(p.row_off + i) * nao + p.col_off + j] += factor * block[i * nb + j];
        }
    }
}

/// Accumulate `block` (row-major over the four function-space component indices,
/// dims `n`) into the `(atom, axis)` tensor of `g` at AO offsets `off`.
fn add_block_eri(
    g: &mut GradientEri,
    atom: usize,
    axis: usize,
    off: [usize; 4],
    n: [usize; 4],
    block: &[f64],
) {
    let nao = g.nao;
    let nn = nao.pow(4);
    let base = (atom * 3 + axis) * nn;
    for a in 0..n[0] {
        for b in 0..n[1] {
            for c in 0..n[2] {
                for d in 0..n[3] {
                    let src = ((a * n[1] + b) * n[2] + c) * n[3] + d;
                    let i = off[0] + a;
                    let j = off[1] + b;
                    let k = off[2] + c;
                    let l = off[3] + d;
                    g.data[base + ((i * nao + j) * nao + k) * nao + l] += block[src];
                }
            }
        }
    }
}
