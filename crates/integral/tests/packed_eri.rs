use integral::{Basis, Engine, PackedEriBuilder, PackedEriWorkspace, Shell, ShellKind};

fn basis(kind: ShellKind) -> Basis {
    Basis::new(vec![
        Shell::with_kind(0, [0.0, 0.0, 0.0], vec![1.2, 0.35], vec![0.7, 0.4], kind).unwrap(),
        Shell::with_kind(1, [0.2, -0.1, 0.5], vec![0.8], vec![1.0], kind).unwrap(),
        Shell::with_kind(1, [0.2, -0.1, 0.5], vec![0.3], vec![1.0], kind).unwrap(),
        Shell::with_kind(2, [-0.3, 0.4, -0.2], vec![0.6], vec![1.0], kind).unwrap(),
        Shell::with_kind(3, [0.1, 0.3, 0.7], vec![0.5], vec![1.0], kind).unwrap(),
    ])
}

fn check(kind: ShellKind, engine: Engine) {
    let basis = basis(kind);
    let n = basis.nao();
    let dense = basis.eri_with(engine);
    let builder = PackedEriBuilder::with_engine(&basis, engine);
    let pairs = builder.ao_pairs();
    assert_eq!(pairs.len(), n * (n + 1) / 2);
    let mut seen = vec![false; n * n];
    for &(p, q) in pairs {
        assert!(p >= q && !seen[p * n + q]);
        seen[p * n + q] = true;
    }

    let serial = builder.build(0.0);
    assert_eq!(serial.len(), builder.output_len());
    let peak = dense.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    for (t, &(p, q)) in pairs.iter().enumerate() {
        for (u, &(r, s)) in pairs[..=t].iter().enumerate() {
            let got = serial[t * (t + 1) / 2 + u];
            let want = dense[((p * n + q) * n + r) * n + s];
            assert!(
                (got - want).abs() < 1e-11 * peak.max(1.0),
                "{kind:?} {engine:?} ({p}{q}|{r}{s}): {got:e} vs {want:e}"
            );
        }
    }

    // Tasks filled in reverse order reproduce the serial build bit for bit.
    let mut out = vec![0.0; builder.output_len()];
    let mut workspace = PackedEriWorkspace::new();
    let mut tasks = builder.partition(&mut out);
    assert_eq!(tasks.len(), builder.bra_pairs().len());
    for task in tasks.iter_mut().rev() {
        builder.fill(&mut workspace, task, 0.0);
    }
    assert_eq!(out, serial);
}

#[test]
fn packed_matches_dense_cartesian() {
    check(ShellKind::Cartesian, Engine::Auto);
    check(ShellKind::Cartesian, Engine::Rys);
    check(ShellKind::Cartesian, Engine::OsHgp);
}

#[test]
fn packed_matches_dense_spherical() {
    check(ShellKind::Spherical, Engine::Auto);
    check(ShellKind::Spherical, Engine::Rys);
    check(ShellKind::Spherical, Engine::OsHgp);
}

#[test]
fn screening_only_drops_integrals_below_the_bound() {
    let basis = Basis::new(vec![
        Shell::new(0, [0.0, 0.0, 0.0], vec![1.1], vec![1.0]).unwrap(),
        Shell::new(1, [0.0, 0.0, 0.4], vec![0.7], vec![1.0]).unwrap(),
        Shell::new(0, [0.0, 0.0, 5.0], vec![1.3], vec![1.0]).unwrap(),
        Shell::new(1, [0.0, 0.0, 5.4], vec![0.9], vec![1.0]).unwrap(),
    ]);
    let builder = PackedEriBuilder::new(&basis);
    let exact = builder.build(0.0);
    let tau = 1e-9;
    let screened = builder.build(tau);
    let mut dropped = 0;
    for (a, b) in exact.iter().zip(&screened) {
        if a != b {
            assert_eq!(*b, 0.0);
            assert!(a.abs() <= tau, "dropped integral {a:e} exceeds tau");
            dropped += 1;
        }
    }
    assert!(dropped > 0, "the far-apart shells must trigger screening");
}

#[test]
fn unscreened_build_keeps_integrals_below_the_schwarz_underflow() {
    // Two unit-exponent s shells 13 bohr apart: the Schwarz bound of the mixed
    // pair underflows, yet (10|00) is a nonzero ~3e-38 on the Rys engine.
    let basis = Basis::new(vec![
        Shell::new(0, [0.0, 0.0, 0.0], vec![1.0], vec![1.0]).unwrap(),
        Shell::new(0, [0.0, 0.0, 13.0], vec![1.0], vec![1.0]).unwrap(),
    ]);
    let n = basis.nao();
    let dense = basis.eri_with(Engine::Rys);
    let builder = PackedEriBuilder::with_engine(&basis, Engine::Rys);
    let packed = builder.build(0.0);
    let mut tiny = 0;
    for (t, &(p, q)) in builder.ao_pairs().iter().enumerate() {
        for (u, &(r, s)) in builder.ao_pairs()[..=t].iter().enumerate() {
            let want = dense[((p * n + q) * n + r) * n + s];
            assert_eq!(packed[t * (t + 1) / 2 + u], want, "({p}{q}|{r}{s})");
            tiny += usize::from(want != 0.0 && want.abs() < 1e-30);
        }
    }
    assert!(
        tiny > 0,
        "the test needs a nonzero integral below the bound"
    );
}
