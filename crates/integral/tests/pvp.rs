//! Principle-based tests for the spin-free pVp matrix
//! `W_{μν} = Σ_k ⟨∂_k μ| V |∂_k ν⟩` ([`Basis::pvp`] / [`Basis::pvp_charges`]).
//!
//! No external reference values: a hand-derivable closed form, an independent
//! analytic assembly through the public `nuclear()` builder over an explicit
//! derivative basis, a finite-difference law on the bra center, and exact
//! symmetry / translation / rotation / charge-linearity laws.

use integral::math::am::{cart_components, cart_index, n_cart};
use integral::math::norm::cart_norm;
use integral::{Basis, Shell};

const PI: f64 = std::f64::consts::PI;

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, &x| m.max(x.abs()))
}

/// Shell-level norm `N(α, l)` — the constant `Shell` folds into every
/// primitive coefficient (the `x^l` monomial's norm).
fn shell_norm(alpha: f64, l: usize) -> f64 {
    cart_norm(alpha, l, 0, 0)
}

/// A single-primitive Cartesian shell whose **effective** coefficient is
/// `N(α, l_ref)` instead of its own `N(α, l)`: the building block for spelling
/// out the derivative `∂_k g = 2α·g_{l+1_k} − l_k·g_{l−1_k}` of a normalized
/// `l_ref` primitive as explicit basis functions.
fn shifted_shell(l: usize, l_ref: usize, alpha: f64, center: [f64; 3]) -> Shell {
    let coeff = shell_norm(alpha, l_ref) / shell_norm(alpha, l);
    Shell::new(l, center, vec![alpha], vec![coeff]).unwrap()
}

/// The derivative expansion of component `c` of a normalized primitive
/// `(l, α)` along axis `k`, as `(which_block, component_index, coefficient)`
/// terms with `which_block ∈ {0 = raised, 1 = lowered}`.
fn deriv_terms(c: [usize; 3], alpha: f64, k: usize) -> Vec<(usize, usize, f64)> {
    let mut up = c;
    up[k] += 1;
    let mut terms = vec![(0, cart_index(up), 2.0 * alpha)];
    if c[k] > 0 {
        let mut dn = c;
        dn[k] -= 1;
        terms.push((1, cart_index(dn), -(c[k] as f64)));
    }
    terms
}

// ---------------------------------------------------------------------------
// Closed form
// ---------------------------------------------------------------------------

/// Two s primitives and the nucleus all at one point: each `∂_k s = 2α·p_k`
/// (unnormalized), and `Σ_k ⟨p_k|1/r|p_k⟩ = ∫ r e^{−p r²} d³r = 2π/p²`, so
/// `W = −Z · N(α)N(β) · 4αβ · 2π/(α+β)²` — a fully hand-derivable value.
#[test]
fn s_s_on_nucleus_matches_closed_form() {
    let (a, b) = (0.9, 1.7);
    let c = [0.0; 3];
    let basis = Basis::new(vec![
        Shell::new(0, c, vec![a], vec![1.0]).unwrap(),
        Shell::new(0, c, vec![b], vec![1.0]).unwrap(),
    ]);
    let w = basis.pvp_charges(&[(c, 1.0)]);
    let expect = |x: f64, y: f64| {
        -shell_norm(x, 0) * shell_norm(y, 0) * 4.0 * x * y * 2.0 * PI / (x + y).powi(2)
    };
    let cases = [(0, a, a), (1, a, b), (3, b, b)];
    for (idx, x, y) in cases {
        let e = expect(x, y);
        assert!(
            (w[idx] - e).abs() <= 1e-13 * e.abs(),
            "element {idx}: {} vs closed form {e}",
            w[idx]
        );
    }
}

// ---------------------------------------------------------------------------
// Independent analytic cross-check (derivative-basis assembly)
// ---------------------------------------------------------------------------

/// Assemble `Σ_k ⟨∂_k μ|V|∂_k ν⟩` through the **public `nuclear()` builder**
/// over an explicit four-shell derivative basis (raised/lowered shells with
/// norm-compensated coefficients), and compare to `pvp_charges` element-wise —
/// a structurally different evaluation (driver-level matrix assembly vs the
/// primitive-level block combination), through angular momenta up to h×h
/// (`l = 5`, the documented maximum — internal shifted integrals reach l = 6).
/// The h cases close the gap between the kernel's claimed `l ≤ 5` reach and the
/// previous l ≤ 4 cross-check (heavy-element QZ basis sets carry h functions).
#[test]
fn matches_derivative_basis_assembly_up_to_h() {
    let a_c = [0.1, -0.3, 0.2];
    let b_c = [-0.4, 0.5, 0.7];
    let charges: [([f64; 3], f64); 2] = [([0.3, 0.2, -0.1], 3.0), ([-0.2, -0.5, 0.6], 1.0)];
    let mut worst = 0.0_f64;
    for (la, lb, alpha, beta) in [
        (0, 0, 1.3, 0.9),
        (1, 0, 0.8, 1.1),
        (1, 1, 1.3, 0.6),
        (2, 1, 0.9, 1.4),
        (2, 2, 0.7, 0.5),
        (3, 2, 1.1, 0.8),
        (4, 3, 0.9, 0.7),
        (4, 4, 0.8, 0.6),
        (5, 4, 0.7, 0.9), // h × g
        (5, 5, 0.6, 0.8), // h × h (the documented maximum)
    ] {
        let bra = Shell::new(la, a_c, vec![alpha], vec![1.0]).unwrap();
        let ket = Shell::new(lb, b_c, vec![beta], vec![1.0]).unwrap();
        let pair = Basis::new(vec![bra, ket]);
        let (na, nb) = (n_cart(la), n_cart(lb));
        let n = na + nb;
        let w = pair.pvp_charges(&charges);

        // Derivative basis: [bra l+1, bra l−1?, ket l+1, ket l−1?].
        let mut big = vec![shifted_shell(la + 1, la, alpha, a_c)];
        let mut offs = vec![0usize, n_cart(la + 1)];
        if la > 0 {
            big.push(shifted_shell(la - 1, la, alpha, a_c));
            offs.push(offs[1] + n_cart(la - 1));
        } else {
            offs.push(offs[1]); // lowered block absent: zero width
        }
        let ket_p_off = *offs.last().unwrap();
        big.push(shifted_shell(lb + 1, lb, beta, b_c));
        let ket_m_off = ket_p_off + n_cart(lb + 1);
        if lb > 0 {
            big.push(shifted_shell(lb - 1, lb, beta, b_c));
        }
        let bra_offs = [0, n_cart(la + 1)]; // raised, lowered row offsets
        let ket_offs = [ket_p_off, ket_m_off];
        let big_basis = Basis::new(big);
        let nbig = big_basis.nao();
        let v = big_basis.nuclear(&charges);

        let comps_a = cart_components(la);
        let comps_b = cart_components(lb);
        let mut reference = vec![0.0; na * nb];
        for (ia, &ca) in comps_a.iter().enumerate() {
            for (ib, &cb) in comps_b.iter().enumerate() {
                let mut acc = 0.0;
                for k in 0..3 {
                    for &(rb, ri, rc) in &deriv_terms(ca, alpha, k) {
                        for &(cbk, ci, cc) in &deriv_terms(cb, beta, k) {
                            let row = bra_offs[rb] + ri;
                            let col = ket_offs[cbk] + ci;
                            acc += rc * cc * v[row * nbig + col];
                        }
                    }
                }
                reference[ia * nb + ib] = acc;
            }
        }
        let peak = max_abs(&reference);
        for ia in 0..na {
            for ib in 0..nb {
                let got = w[ia * n + na + ib];
                let want = reference[ia * nb + ib];
                let err = (got - want).abs() / peak;
                worst = worst.max(err);
                assert!(
                    err <= 1e-11,
                    "(la={la},lb={lb}) [{ia},{ib}]: {got} vs {want} (rel {err:.2e})"
                );
            }
        }
    }
    eprintln!("derivative-basis cross-check worst relative error: {worst:.2e}");
}

// ---------------------------------------------------------------------------
// Finite-difference law on the bra center
// ---------------------------------------------------------------------------

/// `⟨∂_k μ|V|∂_k ν⟩ = d/dA_k ⟨μ(A)|V|∂_k ν⟩` with the nuclei held fixed
/// (`∂_k g = 2α g_{l+1} − l_k g_{l−1}` is exactly the A-derivative of the
/// primitive). The ket derivative is expanded analytically as explicit
/// shells; the bra center is displaced by central differences with `h = 1e-5`
/// while the point charges stay put — `nuclear()` takes the charges
/// explicitly, so the shell center is naturally detached from the charge.
#[test]
fn matches_bra_center_finite_difference() {
    let h = 1e-5;
    let a_c = [0.2, -0.1, 0.4];
    let b_c = [-0.3, 0.6, 0.1];
    let charges: [([f64; 3], f64); 2] = [(a_c, 2.0), ([0.5, 0.3, -0.2], 1.0)];
    let mut worst = 0.0_f64;
    for (la, lb, alpha, beta) in [
        (0, 0, 1.1, 0.7),
        (1, 1, 0.9, 1.2),
        (2, 2, 0.8, 0.6),
        (3, 1, 1.0, 0.9),
    ] {
        let (na, nb) = (n_cart(la), n_cart(lb));
        let n = na + nb;
        let pair = Basis::new(vec![
            Shell::new(la, a_c, vec![alpha], vec![1.0]).unwrap(),
            Shell::new(lb, b_c, vec![beta], vec![1.0]).unwrap(),
        ]);
        let w = pair.pvp_charges(&charges);

        // g_k(t)[ia, ib] = ⟨μ(A + t e_k)|V|∂_k ν⟩, ket derivative analytic.
        let eval = |k: usize, t: f64| -> Vec<f64> {
            let mut center = a_c;
            center[k] += t;
            let mut shells = vec![
                Shell::new(la, center, vec![alpha], vec![1.0]).unwrap(),
                shifted_shell(lb + 1, lb, beta, b_c),
            ];
            if lb > 0 {
                shells.push(shifted_shell(lb - 1, lb, beta, b_c));
            }
            let ket_offs = [na, na + n_cart(lb + 1)];
            let big = Basis::new(shells);
            let nbig = big.nao();
            let v = big.nuclear(&charges);
            let comps_b = cart_components(lb);
            let mut out = vec![0.0; na * nb];
            for ia in 0..na {
                for (ib, &cb) in comps_b.iter().enumerate() {
                    let mut acc = 0.0;
                    for &(blk, ci, cc) in &deriv_terms(cb, beta, k) {
                        acc += cc * v[ia * nbig + ket_offs[blk] + ci];
                    }
                    out[ia * nb + ib] = acc;
                }
            }
            out
        };

        let mut fd = vec![0.0; na * nb];
        for k in 0..3 {
            let plus = eval(k, h);
            let minus = eval(k, -h);
            for (f, (p, m)) in fd.iter_mut().zip(plus.iter().zip(&minus)) {
                *f += (p - m) / (2.0 * h);
            }
        }
        let scale = max_abs(&fd).max(1.0);
        for ia in 0..na {
            for ib in 0..nb {
                let got = w[ia * n + na + ib];
                let want = fd[ia * nb + ib];
                let err = (got - want).abs() / scale;
                worst = worst.max(err);
                assert!(
                    err <= 1e-9,
                    "(la={la},lb={lb}) [{ia},{ib}]: {got} vs FD {want} (rel {err:.2e})"
                );
            }
        }
    }
    eprintln!("bra-center FD cross-check worst relative error: {worst:.2e}");
}

// ---------------------------------------------------------------------------
// Exact laws
// ---------------------------------------------------------------------------

/// An asymmetric mixed Cartesian/spherical, contracted basis for the law tests.
fn law_basis(shift: [f64; 3]) -> (Basis, Vec<([f64; 3], f64)>) {
    let t = |c: [f64; 3]| [c[0] + shift[0], c[1] + shift[1], c[2] + shift[2]];
    let c0 = t([0.0, 0.0, 0.0]);
    let c1 = t([1.1, -0.4, 0.8]);
    let shells = vec![
        Shell::new(0, c0, vec![1.3, 0.4], vec![0.7, 0.5]).unwrap(),
        Shell::new(1, c0, vec![0.9], vec![1.0]).unwrap(),
        Shell::new_spherical(2, c1, vec![0.8, 0.3], vec![0.6, 0.7]).unwrap(),
        Shell::new(3, c1, vec![0.5], vec![1.0]).unwrap(),
    ];
    let charges = vec![(c0, 6.0), (c1, 1.0)];
    (Basis::new(shells), charges)
}

#[test]
fn w_is_bitwise_symmetric() {
    let (basis, charges) = law_basis([0.0; 3]);
    for w in [basis.pvp_charges(&charges), basis.pvp()] {
        let n = basis.nao();
        for i in 0..n {
            for j in 0..n {
                assert_eq!(
                    w[i * n + j].to_bits(),
                    w[j * n + i].to_bits(),
                    "W not bitwise symmetric at ({i},{j})"
                );
            }
        }
    }
}

#[test]
fn translational_invariance() {
    let (b0, q0) = law_basis([0.0; 3]);
    let (b1, q1) = law_basis([1.7, -2.3, 0.9]);
    let (w0, w1) = (b0.pvp_charges(&q0), b1.pvp_charges(&q1));
    let peak = max_abs(&w0);
    let worst = w0
        .iter()
        .zip(&w1)
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        worst <= 1e-12 * peak,
        "translation residual {worst:.2e} vs peak {peak:.2e}"
    );
}

/// 90° rotation about z (`x → −y'`, i.e. centers map `(x,y,z) → (−y,x,z)`):
/// a Cartesian monomial component `[i,j,k]` of the rotated system equals
/// `(−1)^i ×` component `[j,i,k]` of the original, and `Σ_k ∂_k(·)∂_k(·)` and
/// `V` are rotation invariant, so `W'[a,b] = (−1)^{i_a+i_b} W[σa, σb]`.
#[test]
fn rotation_consistency_90deg_about_z() {
    let rot = |c: [f64; 3]| [-c[1], c[0], c[2]];
    let centers = [[0.3, -0.2, 0.5], [-0.7, 0.4, -0.1]];
    let mk = |r: bool| {
        let p = |c: [f64; 3]| if r { rot(c) } else { c };
        let shells = vec![
            Shell::new(0, p(centers[0]), vec![1.1], vec![1.0]).unwrap(),
            Shell::new(1, p(centers[0]), vec![0.7], vec![1.0]).unwrap(),
            Shell::new(2, p(centers[1]), vec![0.9], vec![1.0]).unwrap(),
        ];
        let charges = vec![(p(centers[0]), 4.0), (p([0.6, 0.1, -0.8]), 1.5)];
        (Basis::new(shells), charges)
    };
    let (b, q) = mk(false);
    let (br, qr) = mk(true);
    let (w, wr) = (b.pvp_charges(&q), br.pvp_charges(&qr));
    let n = b.nao();

    // Per-AO map: source index and sign under the rotation.
    let mut map: Vec<(usize, f64)> = Vec::with_capacity(n);
    let mut off = 0;
    for l in [0usize, 1, 2] {
        for c in cart_components(l) {
            let sigma = [c[1], c[0], c[2]];
            let sign = if c[0] % 2 == 0 { 1.0 } else { -1.0 };
            map.push((off + cart_index(sigma), sign));
        }
        off += n_cart(l);
    }
    let peak = max_abs(&w);
    let mut worst = 0.0_f64;
    for a in 0..n {
        for b_ in 0..n {
            let (sa, fa) = map[a];
            let (sb, fb) = map[b_];
            let want = fa * fb * w[sa * n + sb];
            worst = worst.max((wr[a * n + b_] - want).abs());
        }
    }
    assert!(
        worst <= 1e-12 * peak,
        "rotation residual {worst:.2e} vs peak {peak:.2e}"
    );
}

/// `W` is linear in the nuclear charges: scaling `Z` scales `W`, and the
/// multi-charge matrix is the sum of the single-charge ones.
#[test]
fn linear_in_nuclear_charge() {
    let (basis, charges) = law_basis([0.0; 3]);
    let scaled: Vec<_> = charges.iter().map(|&(c, z)| (c, 2.5 * z)).collect();
    let w = basis.pvp_charges(&charges);
    let ws = basis.pvp_charges(&scaled);
    let peak = max_abs(&w);
    for (a, b) in w.iter().zip(&ws) {
        assert!((2.5 * a - b).abs() <= 1e-12 * peak, "Z-scaling violated");
    }
    let sum: Vec<f64> = basis
        .pvp_charges(&charges[..1])
        .iter()
        .zip(&basis.pvp_charges(&charges[1..]))
        .map(|(a, b)| a + b)
        .collect();
    for (a, b) in w.iter().zip(&sum) {
        assert!((a - b).abs() <= 1e-12 * peak, "charge additivity violated");
    }
}

// ---------------------------------------------------------------------------
// Edge cases and API contract
// ---------------------------------------------------------------------------

/// Shells of every supported `l` (0..=5) sitting exactly on the nucleus:
/// the Boys argument hits `T = 0`; nothing may overflow or NaN.
#[test]
fn on_nucleus_shells_are_finite() {
    let c = [0.4, -0.2, 0.3];
    let shells: Vec<Shell> = (0..=5)
        .map(|l| Shell::new(l, c, vec![0.9], vec![1.0]).unwrap())
        .collect();
    let basis = Basis::new(shells);
    let w = basis.pvp_charges(&[(c, 7.0)]);
    assert!(w.iter().all(|x| x.is_finite()), "non-finite pVp element");
    // The fully on-center diagonal must be strictly negative (attraction).
    let n = basis.nao();
    assert!(w[0] < 0.0 && w[(n - 1) * n + n - 1] < 0.0);
}

/// `pvp()` is exactly `pvp_charges` over the distinct shell centers with unit
/// charges, in `Basis::atoms` order.
#[test]
fn pvp_is_unit_charges_on_atoms() {
    let (basis, _) = law_basis([0.3, 0.1, -0.2]);
    let unit: Vec<([f64; 3], f64)> = basis.atoms().into_iter().map(|c| (c, 1.0)).collect();
    let (a, b) = (basis.pvp(), basis.pvp_charges(&unit));
    assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
}

/// The derivative raises shells to `l + 1`, so `l = 6` (the engine cap) must
/// be rejected with the documented panic.
#[test]
#[should_panic(expected = "pVp requires shell l <= 5")]
fn l6_shell_panics() {
    let basis = Basis::new(vec![Shell::new(6, [0.0; 3], vec![0.8], vec![1.0]).unwrap()]);
    let _ = basis.pvp();
}
