//! Batched grid-point Coulomb matrices (`Basis::grid_coulomb[_into]`):
//! per-point equivalence to the negated single-point nuclear attraction,
//! `grid_coulomb` / `grid_coulomb_into` consistency, Hermiticity of every
//! `A^g`, an independent closed-form s|s check, and the batched-vs-naive
//! speedup micro-comparison (`#[ignore]`, release).

use integral::{Basis, Shell};

/// Mixed s/p/d/f/g basis with contraction, repeated centers, and both
/// Cartesian and spherical shells (l up to 4, spherical included).
fn mixed_basis() -> Basis {
    Basis::new(vec![
        Shell::new(0, [0.0, 0.0, 0.0], vec![1.4, 0.5], vec![0.6, 0.5]).unwrap(),
        Shell::new(1, [0.6, -0.3, 0.2], vec![0.9, 0.3], vec![0.7, 0.4]).unwrap(),
        Shell::new_spherical(2, [-0.4, 0.7, -0.1], vec![1.1], vec![1.0]).unwrap(),
        Shell::new(3, [0.2, 0.5, -0.6], vec![0.8], vec![1.0]).unwrap(),
        Shell::new_spherical(4, [-0.3, -0.2, 0.4], vec![0.7], vec![1.0]).unwrap(),
    ])
}

fn grid_points() -> Vec<[f64; 3]> {
    // A small deterministic cloud around the molecule, including a point that
    // coincides with a shell center (the U → 0 Boys corner).
    let mut pts = vec![[0.0, 0.0, 0.0], [0.6, -0.3, 0.2]];
    let mut x = 0.123_f64;
    for _ in 0..10 {
        // Cheap deterministic pseudo-random walk (no dependencies).
        let next = |v: f64| (v * 997.0 + 0.371).sin() * 2.5;
        let p = [x, next(x), next(next(x))];
        pts.push(p);
        x = next(p[2]) + 0.17;
    }
    pts
}

/// Acceptance criterion 1: each `A^g` equals `−nuclear(&[(r_g, 1.0)])` to
/// 1e-12 relative on significant elements (mixed s/p/d/f/g, spherical
/// included), per point.
#[test]
fn matches_negated_single_point_nuclear() {
    let basis = mixed_basis();
    let points = grid_points();
    let nao = basis.nao();
    let a = basis.grid_coulomb(&points);
    assert_eq!(a.len(), points.len() * nao * nao);
    for (g, &p) in points.iter().enumerate() {
        let v = basis.nuclear(&[(p, 1.0)]);
        let mat = &a[g * nao * nao..(g + 1) * nao * nao];
        let peak = v.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
        for (idx, (&got, &nuc)) in mat.iter().zip(&v).enumerate() {
            let want = -nuc; // positive kernel vs the −Z attraction sign
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1e-3 * peak),
                "point {g} elem {idx}: {got} vs {want}"
            );
        }
    }
}

/// Acceptance criterion 2: `grid_coulomb` and `grid_coulomb_into` into a fresh
/// buffer are identical (the former is implemented by the latter — assert
/// bitwise equality), and a sub-range fill matches the full batch (the
/// caller-side parallel-partition contract).
#[test]
fn into_matches_alloc_and_subranges() {
    let basis = mixed_basis();
    let points = grid_points();
    let nao = basis.nao();
    let whole = basis.grid_coulomb(&points);
    let mut buf = vec![f64::NAN; points.len() * nao * nao];
    basis.grid_coulomb_into(&points, &mut buf);
    assert_eq!(whole, buf);
    // Disjoint point sub-ranges into disjoint output chunks reassemble the
    // same tensor bitwise — what a parallel driver does.
    let mut parts = vec![0.0; points.len() * nao * nao];
    let split = points.len() / 2;
    let (lo, hi) = parts.split_at_mut(split * nao * nao);
    basis.grid_coulomb_into(&points[..split], lo);
    basis.grid_coulomb_into(&points[split..], hi);
    assert_eq!(whole, parts);
}

/// Acceptance criterion 5: every `A^g` is symmetric. Off-diagonal shell blocks
/// are mirrored (exact); diagonal shell blocks carry only round-off-level
/// kernel asymmetry.
#[test]
fn each_point_matrix_is_hermitian() {
    let basis = mixed_basis();
    let points = grid_points();
    let nao = basis.nao();
    let a = basis.grid_coulomb(&points);
    for g in 0..points.len() {
        let mat = &a[g * nao * nao..(g + 1) * nao * nao];
        let peak = mat.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
        for i in 0..nao {
            for j in 0..i {
                let (f, r) = (mat[i * nao + j], mat[j * nao + i]);
                assert!(
                    (f - r).abs() <= 1e-12 * f.abs().max(peak * 1e-3),
                    "point {g} ({i},{j}): {f} vs {r}"
                );
            }
        }
    }
}

/// Independent closed-form check (not a cross-check against integral's own
/// nuclear path): for normalized s primitives,
/// `⟨s_α(A)| 1/|r−C| |s_β(B)⟩ = N_α N_β (2π/p) e^{−μ|AB|²} F₀(p|P−C|²)`,
/// with the Boys function evaluated through `erf` for `F₀`.
#[test]
fn s_s_closed_form() {
    let (alpha, beta) = (1.3, 0.9);
    let a = [0.0, 0.0, 0.0];
    let b = [0.0, 0.0, 1.0];
    let basis = Basis::new(vec![
        Shell::new(0, a, vec![alpha], vec![1.0]).unwrap(),
        Shell::new(0, b, vec![beta], vec![1.0]).unwrap(),
    ]);
    let c = [0.0, 0.0, 0.4];
    let out = basis.grid_coulomb(&[c]);

    let pi = std::f64::consts::PI;
    let p = alpha + beta;
    let mu = alpha * beta / p;
    let pz = (alpha * a[2] + beta * b[2]) / p;
    let u: f64 = p * (pz - c[2]).powi(2);
    // F₀(u) = ½ √(π/u) erf(√u); erf via the Boys-free numeric integral is
    // overkill — use the F₀ power series for the moderate u here.
    let f0 = {
        // Converged Taylor/continued series for F₀ at this u (u ≈ 0.27).
        let mut s = 0.0;
        let mut term = 1.0;
        let mut k = 0;
        loop {
            let add = term / (2 * k + 1) as f64;
            s += add;
            if add.abs() < 1e-18 {
                break;
            }
            k += 1;
            term *= -u / k as f64;
        }
        s
    };
    let norm = |e: f64| (2.0 * e / pi).powf(0.75); // s-primitive norm
    let expected = norm(alpha) * norm(beta) * (2.0 * pi / p) * (-mu * 1.0).exp() * f0;
    assert!(
        (out[1] - expected).abs() < 1e-13 * expected.abs(),
        "{} vs {}",
        out[1],
        expected
    );
    // Diagonals are positive (positive kernel).
    assert!(out[0] > 0.0 && out[3] > 0.0);
}

/// Acceptance criterion 3 (run in release with `--include-ignored`): the
/// batched build must be ≥ 5× faster than the naive per-point
/// `−nuclear(&[(g,1)])` loop for ≥ 500 points on a small molecule. Prints the
/// A/B medians (3 repetitions each, same session) for the report.
#[test]
#[ignore = "timing comparison; run in release"]
fn batched_is_5x_faster_than_naive_nuclear_loop() {
    // A water-like molecule with a modest contracted s/p/d basis.
    let o = [0.0, 0.0, 0.2217];
    let h1 = [0.0, 1.4309, -0.8867];
    let h2 = [0.0, -1.4309, -0.8867];
    let basis = Basis::new(vec![
        Shell::new(0, o, vec![130.7, 23.81, 6.44], vec![0.154, 0.535, 0.444]).unwrap(),
        Shell::new(0, o, vec![5.03, 1.17, 0.38], vec![-0.1, 0.4, 0.7]).unwrap(),
        Shell::new(1, o, vec![5.03, 1.17, 0.38], vec![0.156, 0.607, 0.392]).unwrap(),
        Shell::new_spherical(2, o, vec![1.2], vec![1.0]).unwrap(),
        Shell::new(0, h1, vec![3.43, 0.62, 0.17], vec![0.155, 0.535, 0.444]).unwrap(),
        Shell::new(1, h1, vec![0.8], vec![1.0]).unwrap(),
        Shell::new(0, h2, vec![3.43, 0.62, 0.17], vec![0.155, 0.535, 0.444]).unwrap(),
        Shell::new(1, h2, vec![0.8], vec![1.0]).unwrap(),
    ]);
    let n_points = 600;
    let mut points = Vec::with_capacity(n_points);
    let mut v = 0.317_f64;
    for _ in 0..n_points {
        let next = |x: f64| (x * 997.0 + 0.371).sin() * 3.0;
        points.push([v, next(v), next(next(v))]);
        v = next(points.last().unwrap()[2]) + 0.13;
    }
    let nao = basis.nao();

    let median3 = |f: &mut dyn FnMut() -> Vec<f64>| {
        let mut times: Vec<(std::time::Duration, Vec<f64>)> = (0..3)
            .map(|_| {
                let t0 = std::time::Instant::now();
                let out = f();
                (t0.elapsed(), out)
            })
            .collect();
        times.sort_by_key(|(t, _)| *t);
        let (t, out) = times.swap_remove(1);
        (t, out)
    };

    let (t_batched, batched) = median3(&mut || basis.grid_coulomb(&points));
    let (t_naive, naive) = median3(&mut || {
        let mut out = vec![0.0; points.len() * nao * nao];
        for (g, &p) in points.iter().enumerate() {
            let v = basis.nuclear(&[(p, 1.0)]);
            for (slot, x) in out[g * nao * nao..(g + 1) * nao * nao].iter_mut().zip(&v) {
                *slot = -x;
            }
        }
        out
    });

    // Same values (the timing comparison must be apples-to-apples).
    let peak = naive.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    for (&a, &b) in batched.iter().zip(&naive) {
        assert!((a - b).abs() <= 1e-12 * a.abs().max(1e-3 * peak));
    }

    let speedup = t_naive.as_secs_f64() / t_batched.as_secs_f64();
    println!(
        "grid_coulomb micro-comparison ({n_points} points, nao={nao}): \
         batched median {t_batched:?}, naive per-point nuclear median {t_naive:?}, \
         speedup {speedup:.1}x"
    );
    assert!(
        speedup >= 5.0,
        "batched grid_coulomb only {speedup:.2}x faster than the naive loop \
         (batched {t_batched:?}, naive {t_naive:?})"
    );
}
