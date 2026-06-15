//! Basis-set description: contracted Cartesian Gaussian shells.

use crate::engine::os::{Prim, MAX_L};
use crate::math::am::n_cart;
use crate::math::norm::cart_norm;

use crate::IntegralError;

/// Whether a shell's integrals are returned over Cartesian or real
/// spherical-harmonic components.
///
/// - [`ShellKind::Cartesian`] — `(l+1)(l+2)/2` Cartesian components in
///   [`crate::math::am`] order (e.g. `d`: `xx, xy, xz, yy, yz, zz`). This is the
///   native output of both engines.
/// - [`ShellKind::Spherical`] — `2l+1` real spherical-harmonic components,
///   produced by applying the `c2s` transform
///   ([`crate::math::solid_harmonics`]) to the Cartesian block. The components,
///   ordering, sign, and unit normalization follow the real spherical-harmonic
///   convention implemented by this workspace's `c2s` transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShellKind {
    /// Cartesian components (the engine-native basis).
    #[default]
    Cartesian,
    /// Real spherical-harmonic components.
    Spherical,
}

/// A contracted Cartesian Gaussian shell: several primitives sharing a center
/// and angular momentum, combined with contraction coefficients.
///
/// All primitives in the shell have the same angular momentum `l` and center.
///
/// # Normalization convention
///
/// Each **primitive** is normalized: the stored coefficient is multiplied by the
/// shell-level Cartesian norm `N(α_i, l)` (see [`Shell::primitive_coeff`]) so the
/// `x^l` monomial of every primitive has unit self-overlap. The **contraction as
/// a whole is *not* renormalized** — integral uses the raw coefficients you pass in,
/// by design. This differs from codes that rescale each contracted function to
/// exactly unit self-overlap. With *rounded* tabulated coefficients
/// (e.g. published STO-3G) a contracted integral AO can therefore have a
/// self-overlap of `1 ± ~3e-8` rather than exactly 1, producing a uniform
/// ~`2.7e-8` difference from a renormalizing code that vanishes under any
/// per-AO (symmetric) normalization of the resulting matrices. If you need
/// unit-normalized contractions, pre-scale your coefficients before constructing
/// the shell.
#[derive(Debug, Clone, PartialEq)]
pub struct Shell {
    l: usize,
    center: [f64; 3],
    exponents: Vec<f64>,
    coefficients: Vec<f64>,
    kind: ShellKind,
}

impl Shell {
    /// Build a **Cartesian** shell from its angular momentum, center, primitive
    /// exponents, and raw contraction coefficients.
    ///
    /// # Errors
    /// Returns [`IntegralError`] if the exponent and coefficient counts differ, the
    /// contraction is empty, or `l` exceeds the supported maximum ([`MAX_L`]).
    pub fn new(
        l: usize,
        center: [f64; 3],
        exponents: Vec<f64>,
        coefficients: Vec<f64>,
    ) -> Result<Self, IntegralError> {
        Self::with_kind(l, center, exponents, coefficients, ShellKind::Cartesian)
    }

    /// Build a **spherical** (real solid-harmonic) shell. Integrals over this
    /// shell are returned as `2l+1` real spherical-harmonic components (see
    /// [`ShellKind::Spherical`]).
    ///
    /// # Errors
    /// As [`Shell::new`].
    pub fn new_spherical(
        l: usize,
        center: [f64; 3],
        exponents: Vec<f64>,
        coefficients: Vec<f64>,
    ) -> Result<Self, IntegralError> {
        Self::with_kind(l, center, exponents, coefficients, ShellKind::Spherical)
    }

    /// Build a shell with an explicit [`ShellKind`].
    ///
    /// # Errors
    /// As [`Shell::new`].
    pub fn with_kind(
        l: usize,
        center: [f64; 3],
        exponents: Vec<f64>,
        coefficients: Vec<f64>,
        kind: ShellKind,
    ) -> Result<Self, IntegralError> {
        if exponents.is_empty() {
            return Err(IntegralError::EmptyContraction);
        }
        if exponents.len() != coefficients.len() {
            return Err(IntegralError::MismatchedContraction {
                exponents: exponents.len(),
                coefficients: coefficients.len(),
            });
        }
        if l > MAX_L {
            return Err(IntegralError::AngularMomentumTooHigh { l, max: MAX_L });
        }
        Ok(Shell {
            l,
            center,
            exponents,
            coefficients,
            kind,
        })
    }

    /// Angular momentum of the shell.
    #[must_use]
    pub fn l(&self) -> usize {
        self.l
    }

    /// Whether this shell yields Cartesian or spherical components.
    #[must_use]
    pub fn kind(&self) -> ShellKind {
        self.kind
    }

    /// Shell center (bohr).
    #[must_use]
    pub fn center(&self) -> [f64; 3] {
        self.center
    }

    /// Primitive exponents.
    #[must_use]
    pub fn exponents(&self) -> &[f64] {
        &self.exponents
    }

    /// Raw (user-supplied) contraction coefficients.
    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// Number of Cartesian components in the shell, `(l+1)(l+2)/2`.
    #[must_use]
    pub fn n_cart(&self) -> usize {
        n_cart(self.l)
    }

    /// Number of output (function-space) components: `n_cart` for a Cartesian
    /// shell, `2l+1` for a spherical shell. This is the block dimension the
    /// integral builders place into the AO matrices/tensors.
    #[must_use]
    pub fn n_func(&self) -> usize {
        match self.kind {
            ShellKind::Cartesian => n_cart(self.l),
            ShellKind::Spherical => 2 * self.l + 1,
        }
    }

    /// Number of primitives.
    #[must_use]
    pub fn n_prim(&self) -> usize {
        self.exponents.len()
    }

    /// Effective coefficient of primitive `i`: the raw contraction coefficient
    /// times the shell-level normalization `N(α_i, l)`.
    ///
    /// A **zero-exponent** primitive is exempt from normalization (its norm
    /// constant would be 0): the raw coefficient is used as-is. A zero-exponent
    /// `s` primitive with coefficient 1 is then the constant function `1` — the
    /// unit dummy index the density-fitting builders use to obtain 3-center
    /// `(μν|P)` and 2-center `(P|Q)` Coulomb integrals from the 4-center kernels.
    #[must_use]
    pub fn primitive_coeff(&self, i: usize) -> f64 {
        if self.exponents[i] == 0.0 {
            return self.coefficients[i];
        }
        self.coefficients[i] * cart_norm(self.exponents[i], self.l, 0, 0)
    }

    /// The `i`-th primitive as an engine [`Prim`].
    #[must_use]
    pub(crate) fn prim(&self, i: usize) -> Prim {
        Prim::new(self.exponents[i], self.center, self.l)
    }
}

/// An ordered collection of shells defining the atomic-orbital basis.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Basis {
    shells: Vec<Shell>,
}

impl Basis {
    /// Create a basis from a list of shells. AO ordering follows shell order.
    #[must_use]
    pub fn new(shells: Vec<Shell>) -> Self {
        Basis { shells }
    }

    /// The shells, in AO order.
    #[must_use]
    pub fn shells(&self) -> &[Shell] {
        &self.shells
    }

    /// Total number of Cartesian atomic orbitals (counting every shell as
    /// Cartesian, regardless of [`ShellKind`]).
    #[must_use]
    pub fn nao_cart(&self) -> usize {
        self.shells.iter().map(Shell::n_cart).sum()
    }

    /// Total number of output atomic orbitals: the sum of each shell's
    /// [`Shell::n_func`] (`n_cart` for Cartesian shells, `2l+1` for spherical).
    /// This is the dimension of the matrices/tensors the builders return.
    #[must_use]
    pub fn nao(&self) -> usize {
        self.shells.iter().map(Shell::n_func).sum()
    }

    /// Output-AO offset (row/column index of the first component) of each shell,
    /// using [`Shell::n_func`] per shell.
    #[must_use]
    pub(crate) fn offsets(&self) -> Vec<usize> {
        let mut offs = Vec::with_capacity(self.shells.len());
        let mut acc = 0;
        for s in &self.shells {
            offs.push(acc);
            acc += s.n_func();
        }
        offs
    }

    /// Distinct shell centers ("atoms"), in order of first appearance.
    ///
    /// Two shells belong to the same atom iff their centers are **bitwise equal**
    /// (`[f64; 3]` equality). This is the natural grouping when shells on one
    /// nucleus share the same center value, and it defines the per-atom ordering
    /// of the geometric-derivative (gradient) builders.
    #[must_use]
    pub fn atoms(&self) -> Vec<[f64; 3]> {
        let mut atoms: Vec<[f64; 3]> = Vec::new();
        for s in &self.shells {
            if !atoms.contains(&s.center) {
                atoms.push(s.center);
            }
        }
        atoms
    }

    /// Map from shell index to its atom index in [`Basis::atoms`].
    #[must_use]
    pub(crate) fn shell_atom(&self) -> Vec<usize> {
        let atoms = self.atoms();
        self.shells
            .iter()
            .map(|s| atoms.iter().position(|c| *c == s.center).unwrap())
            .collect()
    }

    /// Index of the atom (in [`Basis::atoms`]) at `center`, by bitwise equality,
    /// or `None` if no atom sits there.
    #[must_use]
    pub(crate) fn atom_at(&self, center: [f64; 3]) -> Option<usize> {
        self.atoms().iter().position(|c| *c == center)
    }
}
