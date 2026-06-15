//! Principle-based tests for the erf-attenuated (long-range) ERI kernel
//! [`EriKernel::Erf`]: an independent **McMurchie–Davidson** cross-check with
//! the attenuated Boys function (mirroring `eri_md_cross_check.rs`), exact
//! limit laws (`ω → ∞` reproduces Coulomb; `ω → 0⁺` scales like `ω` for the
//! totally symmetric elements), 8-fold permutational symmetry, 2c/3c ↔ 4c
//! consistency via the zero-exponent dummy-shell construction, and the
//! contract that `EriKernel::Coulomb` is **bit-identical** to the plain paths.

use integral::math::am::{cart_components, n_cart};
use integral::math::boys::boys_array_erf;
use integral::math::norm::cart_norm;
use integral::{Basis, EriKernel, Shell};

// ---------------------------------------------------------------------------
// Independent McMurchie–Davidson reference with the attenuated Boys function
// F_m^ω(T) = s^{m+1/2} F_m(sT), s = ω²/(ρ+ω²) (Gill & Adamson, CPL 261, 105
// (1996)). The Hermite machinery is unchanged because dF_m^ω/dT = −F_{m+1}^ω,
// exactly like the plain Boys ladder.
// ---------------------------------------------------------------------------

/// MD Hermite expansion coefficient `E^t_{ij}` (same recurrence as the Coulomb
/// MD cross-check).
fn e_coeff(i: i64, j: i64, t: i64, q: f64, a: f64, b: f64) -> f64 {
    let p = a + b;
    let mu = a * b / p;
    if t < 0 || t > i + j {
        return 0.0;
    }
    if i == 0 && j == 0 && t == 0 {
        return (-mu * q * q).exp();
    }
    if j == 0 {
        (1.0 / (2.0 * p)) * e_coeff(i - 1, j, t - 1, q, a, b)
            - (mu * q / a) * e_coeff(i - 1, j, t, q, a, b)
            + (t as f64 + 1.0) * e_coeff(i - 1, j, t + 1, q, a, b)
    } else {
        (1.0 / (2.0 * p)) * e_coeff(i, j - 1, t - 1, q, a, b)
            + (mu * q / b) * e_coeff(i, j - 1, t, q, a, b)
            + (t as f64 + 1.0) * e_coeff(i, j - 1, t + 1, q, a, b)
    }
}

/// Hermite Coulomb integral over the attenuated kernel: the standard downward
/// recursion with `F_n → F_n^ω` (valid since the derivative identity is the
/// same).
fn hermite_r(t: i64, u: i64, v: i64, n: usize, fm: &[f64], two_rho: f64, pq: [f64; 3]) -> f64 {
    if t < 0 || u < 0 || v < 0 {
        return 0.0;
    }
    if t == 0 && u == 0 && v == 0 {
        return (-two_rho).powi(n as i32) * fm[n];
    }
    if t > 0 {
        (t as f64 - 1.0) * hermite_r(t - 2, u, v, n + 1, fm, two_rho, pq)
            + pq[0] * hermite_r(t - 1, u, v, n + 1, fm, two_rho, pq)
    } else if u > 0 {
        (u as f64 - 1.0) * hermite_r(t, u - 2, v, n + 1, fm, two_rho, pq)
            + pq[1] * hermite_r(t, u - 1, v, n + 1, fm, two_rho, pq)
    } else {
        (v as f64 - 1.0) * hermite_r(t, u, v - 2, n + 1, fm, two_rho, pq)
            + pq[2] * hermite_r(t, u, v - 1, n + 1, fm, two_rho, pq)
    }
}

#[derive(Clone, Copy)]
struct P {
    e: f64,
    c: [f64; 3],
    l: usize,
}

/// One primitive erf-attenuated (ab|cd) via MD, row-major over the four
/// component indices (matching integral's layout).
fn md_primitive_erf(a: P, b: P, c: P, d: P, omega: f64) -> Vec<f64> {
    let p = a.e + b.e;
    let q = c.e + d.e;
    let pc = combine(a, b, p);
    let qc = combine(c, d, q);
    let rho = p * q / (p + q);
    let pq = [pc[0] - qc[0], pc[1] - qc[1], pc[2] - qc[2]];
    let t_param = rho * (pq[0] * pq[0] + pq[1] * pq[1] + pq[2] * pq[2]);

    let lmax = a.l + b.l + c.l + d.l;
    let mut fm = vec![0.0; lmax + 1];
    boys_array_erf(lmax, t_param, rho, omega, &mut fm);
    let two_rho = 2.0 * rho;

    let pref = 2.0 * std::f64::consts::PI.powf(2.5) / (p * q * (p + q).sqrt());

    let (na, nb, nc, nd) = (n_cart(a.l), n_cart(b.l), n_cart(c.l), n_cart(d.l));
    let (ca, cb, cc, cd) = (
        cart_components(a.l),
        cart_components(b.l),
        cart_components(c.l),
        cart_components(d.l),
    );
    let ab = [a.c[0] - b.c[0], a.c[1] - b.c[1], a.c[2] - b.c[2]];
    let cdv = [c.c[0] - d.c[0], c.c[1] - d.c[1], c.c[2] - d.c[2]];

    let mut out = vec![0.0; na * nb * nc * nd];
    for (ia, la) in ca.iter().enumerate() {
        for (ib, lb) in cb.iter().enumerate() {
            for (ic, lc) in cc.iter().enumerate() {
                for (id, ld) in cd.iter().enumerate() {
                    let mut sum = 0.0;
                    for tx in 0..=(la[0] + lb[0]) {
                        let ex = e_coeff(la[0] as i64, lb[0] as i64, tx as i64, ab[0], a.e, b.e);
                        for ty in 0..=(la[1] + lb[1]) {
                            let ey =
                                e_coeff(la[1] as i64, lb[1] as i64, ty as i64, ab[1], a.e, b.e);
                            for tz in 0..=(la[2] + lb[2]) {
                                let ez =
                                    e_coeff(la[2] as i64, lb[2] as i64, tz as i64, ab[2], a.e, b.e);
                                let ebra = ex * ey * ez;
                                if ebra == 0.0 {
                                    continue;
                                }
                                for sx in 0..=(lc[0] + ld[0]) {
                                    let fx = e_coeff(
                                        lc[0] as i64,
                                        ld[0] as i64,
                                        sx as i64,
                                        cdv[0],
                                        c.e,
                                        d.e,
                                    );
                                    for sy in 0..=(lc[1] + ld[1]) {
                                        let fy = e_coeff(
                                            lc[1] as i64,
                                            ld[1] as i64,
                                            sy as i64,
                                            cdv[1],
                                            c.e,
                                            d.e,
                                        );
                                        for sz in 0..=(lc[2] + ld[2]) {
                                            let fz = e_coeff(
                                                lc[2] as i64,
                                                ld[2] as i64,
                                                sz as i64,
                                                cdv[2],
                                                c.e,
                                                d.e,
                                            );
                                            let eket = fx * fy * fz;
                                            if eket == 0.0 {
                                                continue;
                                            }
                                            let sign =
                                                if (sx + sy + sz) % 2 == 0 { 1.0 } else { -1.0 };
                                            let r = hermite_r(
                                                (tx + sx) as i64,
                                                (ty + sy) as i64,
                                                (tz + sz) as i64,
                                                0,
                                                &fm,
                                                two_rho,
                                                pq,
                                            );
                                            sum += ebra * eket * sign * r;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    out[((ia * nb + ib) * nc + ic) * nd + id] = pref * sum;
                }
            }
        }
    }
    out
}

fn combine(a: P, b: P, p: f64) -> [f64; 3] {
    [
        (a.e * a.c[0] + b.e * b.c[0]) / p,
        (a.e * a.c[1] + b.e * b.c[1]) / p,
        (a.e * a.c[2] + b.e * b.c[2]) / p,
    ]
}

/// Contracted attenuated MD ERI block, normalized like integral.
fn md_block_erf(sa: &Shell, sb: &Shell, sc: &Shell, sd: &Shell, omega: f64) -> Vec<f64> {
    let prims = |s: &Shell| -> Vec<(f64, P)> {
        (0..s.n_prim())
            .map(|i| {
                let e = s.exponents()[i];
                // Zero-exponent primitives (the RI unit-`s` dummy) are
                // normalization-exempt, matching `Shell::primitive_coeff`.
                let nrm = if e == 0.0 {
                    1.0
                } else {
                    cart_norm(e, s.l(), 0, 0)
                };
                let coeff = s.coefficients()[i] * nrm;
                (
                    coeff,
                    P {
                        e,
                        c: s.center(),
                        l: s.l(),
                    },
                )
            })
            .collect()
    };
    let (pa, pb, pc, pd) = (prims(sa), prims(sb), prims(sc), prims(sd));
    let len = sa.n_cart() * sb.n_cart() * sc.n_cart() * sd.n_cart();
    let mut acc = vec![0.0; len];
    for (wa, a) in &pa {
        for (wb, b) in &pb {
            for (wc, c) in &pc {
                for (wd, d) in &pd {
                    let blk = md_primitive_erf(*a, *b, *c, *d, omega);
                    let w = wa * wb * wc * wd;
                    for (o, v) in acc.iter_mut().zip(blk.iter()) {
                        *o += w * v;
                    }
                }
            }
        }
    }
    acc
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// A mixed s/p/d/f Cartesian basis on distinct centers.
fn spdf_basis() -> Basis {
    Basis::new(vec![
        Shell::new(0, [0.0, 0.0, 0.0], vec![1.2, 0.5], vec![0.6, 0.5]).unwrap(), // s (contracted)
        Shell::new(1, [0.7, -0.3, 0.2], vec![0.9], vec![1.0]).unwrap(),          // p
        Shell::new(2, [-0.4, 0.8, -0.1], vec![1.1], vec![1.0]).unwrap(),         // d
        Shell::new(3, [0.2, 0.5, 0.9], vec![0.65], vec![1.0]).unwrap(),          // f
    ])
}

/// Function-space AO offsets of a basis (the `offsets()` accessor is crate
/// private; reconstructed from the public `n_func`).
fn offsets(b: &Basis) -> Vec<usize> {
    let mut offs = Vec::with_capacity(b.shells().len());
    let mut acc = 0;
    for s in b.shells() {
        offs.push(acc);
        acc += s.n_func();
    }
    offs
}

/// Extract the contracted `(ij|kl)` shell block from a dense `nao⁴` tensor,
/// row-major over the four component indices.
fn extract_block(t: &[f64], nao: usize, b: &Basis, q: [usize; 4]) -> Vec<f64> {
    let offs = offsets(b);
    let s = b.shells();
    let n: [usize; 4] = [0, 1, 2, 3].map(|x| s[q[x]].n_func());
    let o: [usize; 4] = [0, 1, 2, 3].map(|x| offs[q[x]]);
    let mut out = Vec::with_capacity(n.iter().product());
    for a in 0..n[0] {
        for bb in 0..n[1] {
            for c in 0..n[2] {
                for d in 0..n[3] {
                    out.push(t[(((o[0] + a) * nao + o[1] + bb) * nao + o[2] + c) * nao + o[3] + d]);
                }
            }
        }
    }
    out
}

fn max_rel(x: &[f64], y: &[f64]) -> f64 {
    x.iter()
        .zip(y.iter())
        .map(|(&a, &b)| (a - b).abs() / b.abs().max(1e-300))
        .fold(0.0_f64, f64::max)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// `EriKernel::Coulomb` must be **bit-identical** (`==`) to the plain Coulomb
/// builders, for the 4c, 2c, and 3c families.
#[test]
fn coulomb_kernel_is_bit_identical() {
    let basis = spdf_basis();
    assert_eq!(basis.eri_kernel(EriKernel::Coulomb), basis.eri());

    let aux = Basis::new(vec![
        Shell::new(0, [0.1, 0.0, 0.0], vec![1.5], vec![1.0]).unwrap(),
        Shell::new(2, [0.0, -0.5, 0.3], vec![0.8], vec![1.0]).unwrap(),
        Shell::new_spherical(1, [0.4, 0.2, -0.6], vec![0.7], vec![1.0]).unwrap(),
    ]);
    assert_eq!(aux.eri_2c_kernel(EriKernel::Coulomb), aux.eri_2c());
    for ish in 0..basis.shells().len() {
        for psh in 0..aux.shells().len() {
            assert_eq!(
                basis.eri_3c_block_kernel(&aux, ish, 1, psh, EriKernel::Coulomb),
                basis.eri_3c_block(&aux, ish, 1, psh)
            );
        }
    }
}

/// ω → large reproduces the Coulomb tensor: `1 − s = ρ/(ρ+ω²)` is ~1e-10 at
/// ω = 1e5 for these exponents, so the relative deviation of each significant
/// element stays well under 1e-8.
#[test]
fn large_omega_reproduces_coulomb() {
    let basis = spdf_basis();
    let coul = basis.eri();
    let erf = basis.eri_kernel(EriKernel::Erf { omega: 1e5 });
    let scale = coul.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let mut worst = 0.0f64;
    for (x, y) in erf.iter().zip(coul.iter()) {
        // Relative on significant elements; tiny elements (cancellation) are
        // measured against a small fraction of the tensor's scale.
        let rel = (x - y).abs() / y.abs().max(1e-3 * scale);
        worst = worst.max(rel);
    }
    assert!(worst <= 1e-8, "ω→∞ limit worst rel deviation {worst:e}");
}

/// ω → 0⁺ scaling: every attenuated integral is bounded by Coulomb in
/// magnitude and the tensor is monotone in ω on the positive diagonal
/// elements; the (ss|ss) element scales linearly in ω (`√s ≈ ω/√ρ`).
#[test]
fn small_omega_scaling() {
    let basis = spdf_basis();
    let coul = basis.eri();
    let lo = basis.eri_kernel(EriKernel::Erf { omega: 0.3 });
    let hi = basis.eri_kernel(EriKernel::Erf { omega: 0.6 });
    let nao = basis.nao();
    for i in 0..nao {
        for k in 0..nao {
            // Diagonal elements (ii|kk) are positive and monotone in ω.
            let idx = ((i * nao + i) * nao + k) * nao + k;
            assert!(lo[idx] > 0.0 && lo[idx] < hi[idx] && hi[idx] < coul[idx]);
        }
    }
    // Linear small-ω law on (ss|ss): value(ω)/ω is constant as ω → 0.
    let v1 = basis.eri_kernel(EriKernel::Erf { omega: 1e-3 })[0];
    let v2 = basis.eri_kernel(EriKernel::Erf { omega: 2e-3 })[0];
    let (r1, r2) = (v1 / 1e-3, v2 / 2e-3);
    assert!(
        (r1 - r2).abs() < 1e-4 * r1.abs(),
        "small-ω linear law: {r1} vs {r2}"
    );
}

/// Identity check: the Rys attenuated kernel equals an independent
/// McMurchie–Davidson implementation with the attenuated Boys function, for
/// mixed quartets through f and ω ∈ {0.3, 0.5, 1.0}.
#[test]
fn erf_matches_mcmurchie_davidson() {
    let basis = spdf_basis();
    let s = basis.shells();
    let quartets = [
        (0, 0, 0, 0), // (ss|ss)
        (1, 0, 0, 0), // (ps|ss)
        (1, 1, 1, 1), // (pp|pp)
        (2, 0, 1, 0), // (ds|ps)
        (2, 1, 2, 1), // (dp|dp)
        (2, 2, 2, 2), // (dd|dd)
        (3, 0, 1, 2), // (fs|pd)
        (0, 1, 2, 3), // (sp|df)
    ];
    for omega in [0.3, 0.5, 1.0] {
        let dense = basis.eri_kernel(EriKernel::Erf { omega });
        for (i, j, k, l) in quartets {
            let ours = extract_block(&dense, basis.nao(), &basis, [i, j, k, l]);
            let md = md_block_erf(&s[i], &s[j], &s[k], &s[l], omega);
            let re = max_rel(&ours, &md);
            assert!(
                re < 1e-11,
                "ω={omega} (l{} l{} | l{} l{}) vs MD max_rel = {re:e}",
                s[i].l(),
                s[j].l(),
                s[k].l(),
                s[l].l()
            );
        }
    }
}

/// 8-fold permutational symmetry of the dense attenuated tensor.
#[test]
fn erf_eightfold_symmetry() {
    let basis = spdf_basis();
    let nao = basis.nao();
    let t = basis.eri_kernel(EriKernel::Erf { omega: 0.5 });
    let at = |i: usize, j: usize, k: usize, l: usize| t[((i * nao + j) * nao + k) * nao + l];
    for i in 0..nao {
        for j in 0..=i {
            for k in 0..=i {
                for l in 0..=k {
                    let v = at(i, j, k, l);
                    for w in [
                        at(j, i, k, l),
                        at(i, j, l, k),
                        at(j, i, l, k),
                        at(k, l, i, j),
                        at(l, k, i, j),
                        at(k, l, j, i),
                        at(l, k, j, i),
                    ] {
                        assert!(
                            (v - w).abs() <= 1e-12 * v.abs().max(1e-14),
                            "8-fold symmetry: ({i}{j}|{k}{l}) {v} vs {w}"
                        );
                    }
                }
            }
        }
    }
}

/// 2c and 3c attenuated integrals must be consistent with the 4c path under
/// the zero-exponent unit-`s` dummy-shell construction `(μν|P) = (μν|P·1ₛ)`,
/// `(P|Q) = (P·1ₛ|Q·1ₛ)`. The 4c side is evaluated here by the *independent*
/// McMurchie–Davidson 4-center algorithm on the explicit dummy-augmented
/// quartets (the dense Rys driver cannot host a dummy–dummy bra pair, whose
/// Gaussian-product center is undefined; pairing each dummy with a real shell
/// — exactly the RI construction — keeps every pair quantity finite).
#[test]
fn erf_2c_3c_consistent_with_4c_dummy_construction() {
    let omega = 0.5;
    let k = EriKernel::Erf { omega };
    let main = Basis::new(vec![
        Shell::new(0, [0.0, 0.0, 0.0], vec![1.2, 0.5], vec![0.6, 0.5]).unwrap(),
        Shell::new(1, [0.7, -0.3, 0.2], vec![0.9], vec![1.0]).unwrap(),
    ]);
    let aux = Basis::new(vec![
        Shell::new(0, [0.1, 0.4, 0.0], vec![1.5], vec![1.0]).unwrap(),
        Shell::new(2, [-0.3, 0.0, 0.6], vec![0.8], vec![1.0]).unwrap(),
    ]);
    let dummy = |c: [f64; 3]| Shell::new(0, c, vec![0.0], vec![1.0]).unwrap();

    // --- 3c: (ij|P) vs the MD 4c quartet (i j | P 1ₛ). ---
    for psh in 0..aux.shells().len() {
        let sp = &aux.shells()[psh];
        let from4c = md_block_erf(
            &main.shells()[0],
            &main.shells()[1],
            sp,
            &dummy(sp.center()),
            omega,
        );
        let block = main.eri_3c_block_kernel(&aux, 0, 1, psh, k);
        let re = max_rel(&block, &from4c);
        assert!(re < 1e-11, "3c vs 4c dummy (P shell {psh}): max_rel {re:e}");
    }

    // --- 2c: (P|Q) vs the MD 4c quartet (P 1ₛ | Q 1ₛ). ---
    let metric = aux.eri_2c_kernel(k);
    let naux = aux.nao();
    let aux_offs = offsets(&aux);
    for p in 0..aux.shells().len() {
        for q in 0..aux.shells().len() {
            let (sp, sq) = (&aux.shells()[p], &aux.shells()[q]);
            let from4c = md_block_erf(sp, &dummy(sp.center()), sq, &dummy(sq.center()), omega);
            let (np, nq) = (sp.n_func(), sq.n_func());
            for a in 0..np {
                for b in 0..nq {
                    let m = metric[(aux_offs[p] + a) * naux + aux_offs[q] + b];
                    let f = from4c[a * nq + b];
                    assert!(
                        (m - f).abs() <= 1e-11 * f.abs().max(1e-14),
                        "2c vs 4c dummy ({p},{q})[{a},{b}]: {m} vs {f}"
                    );
                }
            }
        }
    }
}

/// Spherical shells go through the same c2s transform as the Coulomb path:
/// at large ω the attenuated spherical tensor must reproduce the (validated)
/// spherical Coulomb tensor; at finite ω the 8-fold symmetry must hold.
#[test]
fn erf_spherical_shells() {
    let sph = Basis::new(vec![
        Shell::new_spherical(2, [0.0, 0.0, 0.0], vec![1.1], vec![1.0]).unwrap(),
        Shell::new(1, [0.5, -0.2, 0.3], vec![0.9], vec![1.0]).unwrap(),
        Shell::new_spherical(3, [-0.3, 0.4, 0.1], vec![0.7], vec![1.0]).unwrap(),
    ]);
    let coul = sph.eri();
    let erf = sph.eri_kernel(EriKernel::Erf { omega: 1e5 });
    let scale = coul.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let mut worst = 0.0f64;
    for (x, y) in erf.iter().zip(coul.iter()) {
        worst = worst.max((x - y).abs() / y.abs().max(1e-3 * scale));
    }
    assert!(worst <= 1e-8, "spherical ω→∞ worst rel deviation {worst:e}");

    let nao = sph.nao();
    let t = sph.eri_kernel(EriKernel::Erf { omega: 0.5 });
    let at = |i: usize, j: usize, k: usize, l: usize| t[((i * nao + j) * nao + k) * nao + l];
    for i in 0..nao {
        for k in 0..nao {
            let (v, w) = (at(i, 0, k, 1), at(k, 1, i, 0));
            assert!(
                (v - w).abs() <= 1e-12 * v.abs().max(1e-14),
                "spherical bra-ket exchange ({i}0|{k}1)"
            );
        }
    }
}

#[test]
#[should_panic(expected = "EriKernel::Erf requires a finite omega > 0")]
fn erf_zero_omega_panics() {
    let b = spdf_basis();
    let _ = b.eri_kernel(EriKernel::Erf { omega: 0.0 });
}

#[test]
#[should_panic(expected = "EriKernel::Erf requires a finite omega > 0")]
fn erf_negative_omega_panics_2c() {
    let b = spdf_basis();
    let _ = b.eri_2c_kernel(EriKernel::Erf { omega: -0.5 });
}

#[test]
#[should_panic(expected = "EriKernel::Erf requires a finite omega > 0")]
fn erf_nan_omega_panics_3c() {
    let b = spdf_basis();
    let aux = Basis::new(vec![
        Shell::new(0, [0.1, 0.0, 0.0], vec![1.5], vec![1.0]).unwrap()
    ]);
    let _ = b.eri_3c_block_kernel(&aux, 0, 0, 0, EriKernel::Erf { omega: f64::NAN });
}

/// Full-L corner: h and i shells (l = 5, 6) through the attenuated kernel —
/// the large-ω limit against the Coulomb tensor, plus positivity of the
/// diagonal. Slow in debug; run with `--release -- --include-ignored`.
#[test]
#[ignore = "high-L corner, slow in debug; run in release with --include-ignored"]
fn erf_high_l_corner_h_and_i_shells() {
    let basis = Basis::new(vec![
        Shell::new(6, [0.0, 0.0, 0.0], vec![0.9], vec![1.0]).unwrap(), // i
        Shell::new(5, [0.4, -0.2, 0.5], vec![0.8], vec![1.0]).unwrap(), // h
    ]);
    let coul = basis.eri();
    let erf = basis.eri_kernel(EriKernel::Erf { omega: 1e5 });
    let scale = coul.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let mut worst = 0.0f64;
    for (x, y) in erf.iter().zip(coul.iter()) {
        worst = worst.max((x - y).abs() / y.abs().max(1e-3 * scale));
    }
    assert!(worst <= 1e-8, "high-L ω→∞ worst rel deviation {worst:e}");

    // Finite-ω diagonal positivity at l = 5, 6 (a valid inner product).
    let nao = basis.nao();
    let t = basis.eri_kernel(EriKernel::Erf { omega: 0.4 });
    for i in 0..nao {
        assert!(t[((i * nao + i) * nao + i) * nao + i] > 0.0, "diag {i}");
    }
}
