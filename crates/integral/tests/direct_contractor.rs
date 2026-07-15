use integral::{Basis, DirectBuffers, DirectWorkspace, Engine, Shell, ShellKind};

fn basis(kind: ShellKind) -> Basis {
    Basis::new(vec![
        Shell::with_kind(0, [0.0, 0.0, 0.0], vec![1.2, 0.35], vec![0.7, 0.4], kind).unwrap(),
        Shell::with_kind(1, [0.2, -0.1, 0.5], vec![0.8], vec![1.0], kind).unwrap(),
        Shell::with_kind(2, [-0.3, 0.4, -0.2], vec![0.6], vec![1.0], kind).unwrap(),
        Shell::with_kind(3, [0.1, 0.3, 0.7], vec![0.5], vec![1.0], kind).unwrap(),
    ])
}

fn dense_jk(basis: &Basis, d: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = basis.nao();
    let eri = basis.eri();
    let mut j = vec![0.0; n * n];
    let mut k = vec![0.0; n * n];
    for a in 0..n {
        for b in 0..n {
            for c in 0..n {
                for e in 0..n {
                    j[a * n + b] += eri[((a * n + b) * n + c) * n + e] * d[c * n + e];
                    k[a * n + c] += eri[((a * n + b) * n + c) * n + e] * d[b * n + e];
                }
            }
        }
    }
    (j, k)
}

fn check(kind: ShellKind, engine: Engine) {
    let basis = basis(kind);
    let n = basis.nao();
    let densities: Vec<Vec<f64>> = (0..2)
        .map(|shift| {
            (0..n * n)
                .map(|p| (((p + 3 * shift) % 17) as f64 - 8.0) * 0.013)
                .collect()
        })
        .collect();
    let drefs: Vec<&[f64]> = densities.iter().map(Vec::as_slice).collect();
    let contractor = integral::DirectContractor::with_engine(&basis, engine);
    let bounds = contractor.density_shell_bounds(&drefs);
    let brefs: Vec<&[f64]> = bounds.iter().map(Vec::as_slice).collect();
    let mut js = vec![vec![0.0; n * n]; densities.len()];
    let mut ks = vec![vec![0.0; n * n]; densities.len()];
    let mut workspace = DirectWorkspace::new();
    for &bra in contractor.bra_pairs() {
        let mut outputs: Vec<_> = js
            .iter_mut()
            .zip(&mut ks)
            .map(|(j, k)| DirectBuffers {
                coulomb: j,
                exchange: Some(k),
            })
            .collect();
        contractor.accumulate_bra_pair(
            &mut workspace,
            bra,
            &drefs,
            &brefs,
            0.0,
            true,
            &mut outputs,
        );
    }
    for di in 0..densities.len() {
        let (j_ref, k_ref) = dense_jk(&basis, &densities[di]);
        let worst = js[di]
            .iter()
            .zip(&j_ref)
            .chain(ks[di].iter().zip(&k_ref))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            worst < 1e-9,
            "{kind:?} {engine:?} density {di}: ΔJK={worst:e}"
        );
    }
}

#[test]
fn cartesian_multiple_densities_forced_engines() {
    check(ShellKind::Cartesian, Engine::OsHgp);
    check(ShellKind::Cartesian, Engine::Rys);
}

#[test]
fn spherical_multiple_densities_forced_engines() {
    check(ShellKind::Spherical, Engine::OsHgp);
    check(ShellKind::Spherical, Engine::Rys);
}
