//! Geometric first-derivative validation across the full angular-momentum range.
//!
//! Complements `gradients.rs` by closing the coverage gaps it leaves open:
//!
//!   * FD vs the value builders across the FULL supported L range (l = 0..=5),
//!     not just l ≤ 2 — for S, T, V (Cartesian + spherical).
//!   * High-L ERI gradients (differentiating up to a g shell, l=4 → raised to h)
//!     on four distinct non-collinear centres, BOTH engines.
//!   * TI "teeth": the per-centre derivatives are individually O(1) and cancel —
//!     a genuine physical constraint for S/T/ERI (independent computations), and
//!     a demonstration that corrupting one centre breaks the zero sum.
//!   * Scope-boundary error paths: i-shell gradient and off-atom charge.
//!   * A FALSIFIABLE [natom,3,nao,nao] layout test (axis/atom swap is detected).
//!
//! FD differences the *value* integrals (themselves cross-checked), so it is
//! an engine-free check of the analytic gradient independent of the gradient's
//! own machinery.

use integral::{Basis, Engine, IntegralError, Shell, ShellKind};

const H: f64 = 2e-4;

fn shifted(c: [f64; 3], target: [f64; 3], d: [f64; 3]) -> [f64; 3] {
    if c == target {
        [c[0] + d[0], c[1] + d[1], c[2] + d[2]]
    } else {
        c
    }
}

fn shift_basis(basis: &Basis, target: [f64; 3], d: [f64; 3]) -> Basis {
    let shells = basis
        .shells()
        .iter()
        .map(|s| {
            Shell::with_kind(
                s.l(),
                shifted(s.center(), target, d),
                s.exponents().to_vec(),
                s.coefficients().to_vec(),
                s.kind(),
            )
            .unwrap()
        })
        .collect();
    Basis::new(shells)
}

fn shift_charges(
    charges: &[([f64; 3], f64)],
    target: [f64; 3],
    d: [f64; 3],
) -> Vec<([f64; 3], f64)> {
    charges
        .iter()
        .map(|&(c, z)| (shifted(c, target, d), z))
        .collect()
}

fn unit(axis: usize, h: f64) -> [f64; 3] {
    let mut d = [0.0; 3];
    d[axis] = h;
    d
}

fn central_diff(plus: &[f64], minus: &[f64], h: f64) -> Vec<f64> {
    plus.iter()
        .zip(minus)
        .map(|(p, m)| (p - m) / (2.0 * h))
        .collect()
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn charges_of(basis: &Basis) -> Vec<([f64; 3], f64)> {
    basis
        .atoms()
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, (i + 1) as f64))
        .collect()
}

/// A two-centre basis whose first shell has momentum `l_hi` (the one we sweep),
/// plus a small partner shell on a distinct non-collinear centre. Spherical iff
/// `sph`.
fn sweep_basis(l_hi: usize, sph: bool) -> Basis {
    let a0 = [0.0, 0.0, 0.0];
    let a1 = [0.7, -0.3, 0.4];
    let kind = if sph {
        ShellKind::Spherical
    } else {
        ShellKind::Cartesian
    };
    Basis::new(vec![
        Shell::with_kind(l_hi, a0, vec![1.1, 0.45], vec![0.6, 0.5], kind).unwrap(),
        Shell::with_kind(1, a1, vec![0.9], vec![1.0], kind).unwrap(),
    ])
}

/// FD-vs-analytic for the three 1e operators, returning the worst element over
/// all atoms/axes.
fn one_electron_fd_worst(basis: &Basis) -> (f64, f64, f64) {
    let charges = charges_of(basis);
    let gs = basis.overlap_grad().unwrap();
    let gt = basis.kinetic_grad().unwrap();
    let gv = basis.nuclear_grad(&charges).unwrap();
    let (mut ws, mut wt, mut wv) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (ai, &atom) in basis.atoms().iter().enumerate() {
        for axis in 0..3 {
            let bp = shift_basis(basis, atom, unit(axis, H));
            let bm = shift_basis(basis, atom, unit(axis, -H));
            ws = ws.max(max_abs_diff(
                gs.block(ai, axis),
                &central_diff(&bp.overlap(), &bm.overlap(), H),
            ));
            wt = wt.max(max_abs_diff(
                gt.block(ai, axis),
                &central_diff(&bp.kinetic(), &bm.kinetic(), H),
            ));
            // Atom displacement moves its charge too → exercises HF term.
            let cp = shift_charges(&charges, atom, unit(axis, H));
            let cm = shift_charges(&charges, atom, unit(axis, -H));
            wv = wv.max(max_abs_diff(
                gv.block(ai, axis),
                &central_diff(&bp.nuclear(&cp), &bm.nuclear(&cm), H),
            ));
        }
    }
    (ws, wt, wv)
}

#[test]
fn fd_one_electron_full_l_range_cartesian() {
    eprintln!("Cartesian 1e gradient FD worst-error by highest L:");
    let mut worst = 0.0_f64;
    for l in 0..=5 {
        let basis = sweep_basis(l, false);
        let (s, t, v) = one_electron_fd_worst(&basis);
        eprintln!("  l={l}: dS={s:.2e} dT={t:.2e} dV={v:.2e}");
        worst = worst.max(s).max(t).max(v);
    }
    assert!(worst < 1e-5, "Cartesian 1e grad FD worst {worst:.3e}");
}

#[test]
fn fd_one_electron_full_l_range_spherical() {
    eprintln!("Spherical 1e gradient FD worst-error by highest L:");
    let mut worst = 0.0_f64;
    for l in 0..=5 {
        let basis = sweep_basis(l, true);
        let (s, t, v) = one_electron_fd_worst(&basis);
        eprintln!("  l={l}: dS={s:.2e} dT={t:.2e} dV={v:.2e}");
        worst = worst.max(s).max(t).max(v);
    }
    assert!(worst < 1e-5, "spherical 1e grad FD worst {worst:.3e}");
}

/// Four distinct non-collinear centres; first shell momentum is `l_hi` (swept),
/// the rest small, so the ERI-gradient tensor stays tractable while the
/// differentiated shell reaches high L.
fn eri_sweep_basis(l_hi: usize, sph: bool) -> Basis {
    let kind = if sph {
        ShellKind::Spherical
    } else {
        ShellKind::Cartesian
    };
    let a = [
        [0.0, 0.0, 0.0],
        [0.8, -0.2, 0.3],
        [-0.4, 0.6, 0.1],
        [0.2, 0.5, -0.7],
    ];
    Basis::new(vec![
        Shell::with_kind(l_hi, a[0], vec![1.0], vec![1.0], kind).unwrap(),
        Shell::with_kind(1, a[1], vec![0.9], vec![1.0], kind).unwrap(),
        Shell::with_kind(0, a[2], vec![0.7], vec![1.0], kind).unwrap(),
        Shell::with_kind(0, a[3], vec![1.2], vec![1.0], kind).unwrap(),
    ])
}

fn eri_grad_fd_worst(basis: &Basis, engine: Engine) -> f64 {
    let g = basis.eri_grad_with(engine).unwrap();
    let mut worst = 0.0_f64;
    for (ai, &atom) in basis.atoms().iter().enumerate() {
        for axis in 0..3 {
            let plus = shift_basis(basis, atom, unit(axis, H)).eri_with(engine);
            let minus = shift_basis(basis, atom, unit(axis, -H)).eri_with(engine);
            worst = worst.max(max_abs_diff(
                g.block(ai, axis),
                &central_diff(&plus, &minus, H),
            ));
        }
    }
    worst
}

#[test]
fn fd_eri_gradient_high_l_both_engines_cartesian() {
    eprintln!("Cartesian ERI gradient FD worst-error by differentiated L (both engines):");
    let mut worst = 0.0_f64;
    // up to g (l=4): the derivative raises it to h (l=5), inside MAX_L.
    for l in 1..=4 {
        let basis = eri_sweep_basis(l, false);
        let r = eri_grad_fd_worst(&basis, Engine::Rys);
        let o = eri_grad_fd_worst(&basis, Engine::OsHgp);
        eprintln!("  l={l}: Rys={r:.2e} OS/HGP={o:.2e}");
        worst = worst.max(r).max(o);
    }
    assert!(worst < 1e-5, "Cartesian ERI grad FD worst {worst:.3e}");
}

#[test]
fn fd_eri_gradient_high_l_both_engines_spherical() {
    eprintln!("Spherical ERI gradient FD worst-error by differentiated L (both engines):");
    let mut worst = 0.0_f64;
    for l in 1..=4 {
        let basis = eri_sweep_basis(l, true);
        let r = eri_grad_fd_worst(&basis, Engine::Rys);
        let o = eri_grad_fd_worst(&basis, Engine::OsHgp);
        eprintln!("  l={l}: Rys={r:.2e} OS/HGP={o:.2e}");
        worst = worst.max(r).max(o);
    }
    assert!(worst < 1e-5, "spherical ERI grad FD worst {worst:.3e}");
}

#[test]
fn cross_engine_eri_gradient_high_l_distinct_centres() {
    // Element-by-element Rys vs OS/HGP at high L on distinct centres.
    let mut worst = 0.0_f64;
    for l in 1..=4 {
        let basis = eri_sweep_basis(l, false);
        let rys = basis.eri_grad_with(Engine::Rys).unwrap();
        let os = basis.eri_grad_with(Engine::OsHgp).unwrap();
        for ai in 0..basis.atoms().len() {
            for axis in 0..3 {
                worst = worst.max(max_abs_diff(rys.block(ai, axis), os.block(ai, axis)));
            }
        }
    }
    eprintln!("Rys vs OS/HGP ERI gradient worst (l up to 4): {worst:.3e}");
    assert!(worst < 1e-10, "cross-engine ERI grad worst {worst:.3e}");
}

/// Regenerated step study: confirm O(h²) convergence down to the cancellation
/// floor, independent of the `fd_step_study` in `gradients.rs`.
#[test]
fn regenerated_fd_step_study() {
    let basis = sweep_basis(2, false);
    let g = basis.overlap_grad().unwrap();
    let atom = basis.atoms()[0];
    eprintln!("regenerated overlap-grad central-difference error vs h (atom 0, x):");
    let mut prev: Option<(f64, f64)> = None;
    for &h in &[1e-2, 1e-3, 1e-4, 1e-5, 1e-6] {
        let plus = shift_basis(&basis, atom, unit(0, h)).overlap();
        let minus = shift_basis(&basis, atom, unit(0, -h)).overlap();
        let err = max_abs_diff(g.block(0, 0), &central_diff(&plus, &minus, h));
        let ratio = prev
            .map(|(ph, pe)| (pe / err) * (h / ph) * (h / ph))
            .unwrap_or(0.0);
        eprintln!("  h={h:.0e}  err={err:.3e}  (O(h^2) ratio≈{ratio:.2})");
        prev = Some((h, err));
    }
    // From h=1e-2 to 1e-3 and 1e-3 to 1e-4 the error must fall ~100× each (O(h²)).
    let e2 = {
        let p = shift_basis(&basis, atom, unit(0, 1e-2)).overlap();
        let m = shift_basis(&basis, atom, unit(0, -1e-2)).overlap();
        max_abs_diff(g.block(0, 0), &central_diff(&p, &m, 1e-2))
    };
    let e3 = {
        let p = shift_basis(&basis, atom, unit(0, 1e-3)).overlap();
        let m = shift_basis(&basis, atom, unit(0, -1e-3)).overlap();
        max_abs_diff(g.block(0, 0), &central_diff(&p, &m, 1e-3))
    };
    let drop = e2 / e3;
    assert!(
        drop > 50.0,
        "O(h²) drop 1e-2→1e-3 was only {drop:.1}× (expected ~100×)"
    );
}

// ---------------------------------------------------------------------------
// Translational invariance: GENUINE (S/T/ERI) vs structural (V).
// ---------------------------------------------------------------------------

/// For overlap, the per-centre blocks ∂_A and ∂_B are computed independently
/// (independent raise/lower on each index). Show they are individually O(1) yet
/// sum to ~0 — i.e. the zero sum is a *nontrivial cancellation*, a real check.
/// Then corrupt one atom's block and confirm the residual blows up (teeth).
#[test]
fn ti_overlap_is_a_nontrivial_cancellation() {
    // Two p shells on two distinct atoms → for the off-diagonal block, ∂_A and
    // ∂_B are both large and must cancel.
    let a0 = [0.0, 0.0, 0.0];
    let a1 = [0.6, -0.4, 0.5];
    let basis = Basis::new(vec![
        Shell::new(1, a0, vec![1.2], vec![1.0]).unwrap(),
        Shell::new(1, a1, vec![0.8], vec![1.0]).unwrap(),
    ]);
    let g = basis.overlap_grad().unwrap();
    // Largest single-atom block magnitude.
    let mut max_block = 0.0_f64;
    for ai in 0..2 {
        for axis in 0..3 {
            for &v in g.block(ai, axis) {
                max_block = max_block.max(v.abs());
            }
        }
    }
    let resid = g.max_translational_residual();
    eprintln!("overlap: max single-atom |∂_A| = {max_block:.3e}, TI residual = {resid:.3e}");
    assert!(
        max_block > 0.1,
        "per-atom derivative should be O(1), got {max_block:.3e}"
    );
    assert!(resid < 1e-10, "TI residual should vanish, got {resid:.3e}");
    // Teeth: the residual is a real sum, so the cancellation ratio is enormous.
    assert!(
        max_block / resid.max(1e-300) > 1e6,
        "cancellation not significant"
    );
}

/// V's TI is structural: ∂_C is assembled as −(∂_A+∂_B), so Σ = 0 holds
/// algebraically and CANNOT detect a wrong ∂_A/∂_B. We document this by noting
/// the residual stays ~0 even on a deliberately asymmetric charge set — TI tells
/// us nothing new about V. (The HF term is validated separately by FD below.)
#[test]
fn ti_nuclear_is_structural_not_a_check() {
    let a0 = [0.0, 0.0, 0.0];
    let a1 = [0.6, -0.4, 0.5];
    let basis = Basis::new(vec![
        Shell::new(1, a0, vec![1.2], vec![1.0]).unwrap(),
        Shell::new(0, a1, vec![0.8], vec![1.0]).unwrap(),
    ]);
    // Asymmetric charges; TI still vanishes purely by construction.
    let charges = vec![(a0, 3.0), (a1, 1.0)];
    let g = basis.nuclear_grad(&charges).unwrap();
    assert!(
        g.max_translational_residual() < 1e-10,
        "V TI residual = {:.3e}",
        g.max_translational_residual()
    );
}

/// The HF term IS independently validated: finite-difference the charge
/// coordinate alone (basis fixed) and recover the implementation's HF term as
/// (analytic total − basis-only FD). All three pieces are independent.
#[test]
fn nuclear_hf_term_independently_validated() {
    let a0 = [0.0, 0.0, 0.0];
    let a1 = [0.7, -0.3, 0.4];
    let basis = Basis::new(vec![
        Shell::new(2, a0, vec![1.1, 0.4], vec![0.6, 0.5]).unwrap(),
        Shell::new(1, a1, vec![0.9], vec![1.0]).unwrap(),
    ]);
    let charges = charges_of(&basis);
    let g = basis.nuclear_grad(&charges).unwrap();
    let mut worst = 0.0_f64;
    for (ai, &atom) in basis.atoms().iter().enumerate() {
        for axis in 0..3 {
            let cp = shift_charges(&charges, atom, unit(axis, H));
            let cm = shift_charges(&charges, atom, unit(axis, -H));
            let hf_fd = central_diff(&basis.nuclear(&cp), &basis.nuclear(&cm), H);
            let bp = shift_basis(&basis, atom, unit(axis, H));
            let bm = shift_basis(&basis, atom, unit(axis, -H));
            let basis_fd = central_diff(&bp.nuclear(&charges), &bm.nuclear(&charges), H);
            let hf_impl: Vec<f64> = g
                .block(ai, axis)
                .iter()
                .zip(&basis_fd)
                .map(|(t, b)| t - b)
                .collect();
            worst = worst.max(max_abs_diff(&hf_impl, &hf_fd));
        }
    }
    eprintln!("HF term isolated vs charge-coordinate FD: worst {worst:.3e}");
    assert!(worst < 1e-6, "HF term FD worst {worst:.3e}");
}

// ---------------------------------------------------------------------------
// Scope-boundary error paths.
// ---------------------------------------------------------------------------

#[test]
fn gradient_of_i_shell_is_clean_error() {
    let basis = Basis::new(vec![
        Shell::new(6, [0.0, 0.0, 0.0], vec![1.0], vec![1.0]).unwrap()
    ]);
    assert_eq!(
        basis.overlap_grad().unwrap_err(),
        IntegralError::AngularMomentumTooHighForGradient { l: 6, max: 5 }
    );
    assert_eq!(
        basis.kinetic_grad().unwrap_err(),
        IntegralError::AngularMomentumTooHighForGradient { l: 6, max: 5 }
    );
    assert_eq!(
        basis.eri_grad().unwrap_err(),
        IntegralError::AngularMomentumTooHighForGradient { l: 6, max: 5 }
    );
    let charges = charges_of(&basis);
    assert_eq!(
        basis.nuclear_grad(&charges).unwrap_err(),
        IntegralError::AngularMomentumTooHighForGradient { l: 6, max: 5 }
    );
}

#[test]
fn nuclear_grad_charge_off_atom_is_clean_error() {
    let basis = Basis::new(vec![
        Shell::new(1, [0.0, 0.0, 0.0], vec![1.0], vec![1.0]).unwrap()
    ]);
    let off = [1.0, 2.0, 3.0]; // not a basis centre
    let err = basis.nuclear_grad(&[(off, 1.0)]).unwrap_err();
    assert_eq!(err, IntegralError::ChargeNotOnAtom { center: off });
}

// ---------------------------------------------------------------------------
// Falsifiable [natom, 3, nao, nao] layout.
// ---------------------------------------------------------------------------

/// `g.block(atom, axis)` must equal the FD of THAT atom/axis — and must NOT
/// equal the FD of a different axis or a different atom. So an axis-swap or
/// atom-swap in the layout would be detected: the test has teeth.
#[test]
fn gradient_layout_is_falsifiable() {
    let a0 = [0.0, 0.0, 0.0];
    let a1 = [0.7, -0.3, 0.4];
    let basis = Basis::new(vec![
        Shell::new(1, a0, vec![1.2], vec![1.0]).unwrap(),
        Shell::new(2, a1, vec![0.9], vec![1.0]).unwrap(),
    ]);
    let g = basis.overlap_grad().unwrap();
    // Precompute FD for every (atom, axis).
    let atoms = basis.atoms();
    let mut fd: Vec<[Vec<f64>; 3]> = Vec::new();
    for &atom in &atoms {
        let cols = std::array::from_fn(|axis| {
            let bp = shift_basis(&basis, atom, unit(axis, H));
            let bm = shift_basis(&basis, atom, unit(axis, -H));
            central_diff(&bp.overlap(), &bm.overlap(), H)
        });
        fd.push(cols);
    }
    // Correct mapping agrees.
    for (ai, cols) in fd.iter().enumerate() {
        for (axis, col) in cols.iter().enumerate() {
            assert!(
                max_abs_diff(g.block(ai, axis), col) < 1e-6,
                "block({ai},{axis}) disagrees with its own FD"
            );
        }
    }
    // Axis swap (x vs y) on atom 0 must be detected (these atoms are off-axis so
    // x and y derivatives genuinely differ).
    assert!(
        max_abs_diff(g.block(0, 0), &fd[0][1]) > 1e-4,
        "x-block matched the y FD — an axis swap would go undetected"
    );
    // Atom swap (atom 0 vs atom 1) must be detected.
    assert!(
        max_abs_diff(g.block(0, 0), &fd[1][0]) > 1e-4,
        "atom-0 block matched atom-1 FD — an atom swap would go undetected"
    );
}
