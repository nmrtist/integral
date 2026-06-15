//! Property and layout tests for the two-electron (ERI) builder.
//!
//! Pure-Rust, no external library. These tests assert the
//! permutational symmetry and the documented block layout/strides — the latter
//! matters now that ERI blocks are *not* index-symmetric the way the Phase-1
//! operators were.

use integral::math::am::n_cart;
use integral::{Basis, Shell};

/// Four shells of different angular momenta on different centers, so every
/// (ij|kl) block is genuinely distinct (no accidental symmetry to hide a
/// transposition bug).
fn mixed_basis() -> Basis {
    Basis::new(vec![
        Shell::new(0, [0.0, 0.0, 0.0], vec![1.4, 0.5], vec![0.6, 0.5]).unwrap(), // s
        Shell::new(1, [0.6, -0.3, 0.2], vec![0.9], vec![1.0]).unwrap(),          // p
        Shell::new(2, [-0.4, 0.7, -0.1], vec![1.1], vec![1.0]).unwrap(),         // d
        Shell::new(3, [0.2, 0.5, 0.8], vec![0.7], vec![1.0]).unwrap(),           // f
    ])
}

fn nao(b: &Basis) -> usize {
    b.nao_cart()
}

#[test]
fn ssss_is_positive_and_finite() {
    // A single s shell: (00|00) must be a finite positive number.
    let basis = Basis::new(vec![
        Shell::new(0, [0.0, 0.0, 0.0], vec![0.8], vec![1.0]).unwrap()
    ]);
    let eri = basis.eri();
    assert_eq!(eri.len(), 1);
    assert!(eri[0].is_finite() && eri[0] > 0.0, "(ss|ss) = {}", eri[0]);
}

#[test]
fn eight_fold_permutational_symmetry() {
    // The dense tensor evaluates one canonical quartet per symmetry class and
    // scatters it, so its shell-level 8-fold symmetry is exact by construction.
    // The genuine kernel assertion therefore compares independently *computed*
    // blocks: `eri_block(σ(i,j,k,l))` evaluates its own quartet with no
    // symmetrization, so agreement across all 8 σ is a real recurrence check.
    let basis = mixed_basis();
    let shells = basis.shells();
    let nsh = shells.len();
    let nc: Vec<usize> = shells.iter().map(|s| n_cart(s.l())).collect();

    // The 8 permutations of (i,j,k,l): bra swap, ket swap, bra↔ket exchange.
    const PERMS: [[usize; 4]; 8] = [
        [0, 1, 2, 3],
        [1, 0, 2, 3],
        [0, 1, 3, 2],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [2, 3, 1, 0],
        [3, 2, 0, 1],
        [3, 2, 1, 0],
    ];

    // Tight *relative* residual on significant elements (|base| ≥ 1e-3·peak) plus an
    // absolute floor for the rest (the `eri_maxima` convention). The two permutation
    // paths of a structurally-near-zero element differ only by f64 round-off
    // (~1e-16 abs); a floorless relative residual would over-weight those and reject
    // any value-changing kernel optimisation, while the consumer reuses one value per
    // unique quartet and never observes the difference.
    let mut peak = 0.0_f64;
    let mut blocks: Vec<Vec<f64>> = Vec::new();
    let mut quartets: Vec<[usize; 4]> = Vec::new();
    for i in 0..nsh {
        for j in 0..nsh {
            for k in 0..nsh {
                for l in 0..nsh {
                    let block = basis.eri_block(i, j, k, l);
                    peak = block.iter().fold(peak, |m, &x| m.max(x.abs()));
                    blocks.push(block);
                    quartets.push([i, j, k, l]);
                }
            }
        }
    }
    let floor = 1e-3 * peak;
    let block_of =
        |q: [usize; 4]| -> &Vec<f64> { &blocks[((q[0] * nsh + q[1]) * nsh + q[2]) * nsh + q[3]] };
    let mut worst_sig = 0.0_f64;
    let mut worst_abs = 0.0_f64;
    for &q in &quartets {
        let base = block_of(q);
        let nq = [nc[q[0]], nc[q[1]], nc[q[2]], nc[q[3]]];
        for perm in &PERMS[1..] {
            let qp = [q[perm[0]], q[perm[1]], q[perm[2]], q[perm[3]]];
            let pb = block_of(qp);
            let np = [nc[qp[0]], nc[qp[1]], nc[qp[2]], nc[qp[3]]];
            for a in 0..nq[0] {
                for b in 0..nq[1] {
                    for c in 0..nq[2] {
                        for d in 0..nq[3] {
                            let src = ((a * nq[1] + b) * nq[2] + c) * nq[3] + d;
                            let comp = [a, b, c, d];
                            let (pa, pbx, pc, pd) =
                                (comp[perm[0]], comp[perm[1]], comp[perm[2]], comp[perm[3]]);
                            let dst = ((pa * np[1] + pbx) * np[2] + pc) * np[3] + pd;
                            let dv = (base[src] - pb[dst]).abs();
                            worst_abs = worst_abs.max(dv);
                            if base[src].abs() >= floor {
                                worst_sig = worst_sig.max(dv / base[src].abs());
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        worst_sig < 1e-11,
        "worst symmetry relative residual {worst_sig:e}"
    );
    assert!(
        worst_abs < 1e-11 * peak.max(1.0) + 1e-12,
        "worst symmetry absolute residual {worst_abs:e} (peak {peak:e})"
    );
}

#[test]
fn block_layout_matches_dense_strides() {
    // `eri_block(i,j,k,l)` must equal the dense tensor slice at the four shell
    // offsets under the documented row-major (a,b,c,d) layout. Use all-distinct,
    // different-L shells so a stride/transposition bug cannot stay hidden.
    let basis = mixed_basis();
    let n = nao(&basis);
    let dense = basis.eri();
    let shells = basis.shells();
    let offs: Vec<usize> = {
        let mut acc = 0;
        let mut v = Vec::new();
        for s in shells {
            v.push(acc);
            acc += s.n_cart();
        }
        v
    };

    for (qi, qj, qk, ql) in [(1usize, 0usize, 2usize, 3usize), (0, 1, 3, 2), (2, 3, 1, 0)] {
        let block = basis.eri_block(qi, qj, qk, ql);
        let (nb, nc, nd) = (
            n_cart(shells[qj].l()),
            n_cart(shells[qk].l()),
            n_cart(shells[ql].l()),
        );
        let (na2,) = (n_cart(shells[qi].l()),);
        assert_eq!(block.len(), na2 * nb * nc * nd);
        for a in 0..na2 {
            for b in 0..nb {
                for c in 0..nc {
                    for d in 0..nd {
                        let src = ((a * nb + b) * nc + c) * nd + d;
                        let (i, j, k, l) = (offs[qi] + a, offs[qj] + b, offs[qk] + c, offs[ql] + d);
                        let dval = dense[((i * n + j) * n + k) * n + l];
                        assert!(
                            (block[src] - dval).abs() <= 1e-12 * dval.abs().max(1e-12),
                            "layout mismatch at block {src} ({i}{j}|{k}{l}): {} vs {}",
                            block[src],
                            dval
                        );
                    }
                }
            }
        }
    }
}
