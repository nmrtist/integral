//! The MAX_L = 6 angular-momentum boundary.
//! Confirms l = 6 is accepted and computes, and l = 7 fails *cleanly* with a
//! descriptive error rather than panicking or producing silent garbage.

use integral::{Basis, IntegralError, Shell};

#[test]
fn l6_is_accepted_and_computes() {
    // i shell (l = 6): construction succeeds and overlap runs without panic.
    let sh = Shell::new(6, [0.0, 0.0, 0.0], vec![1.0], vec![1.0]).unwrap();
    assert_eq!(sh.n_cart(), 28); // (6+1)(6+2)/2
    let basis = Basis::new(vec![sh]);
    let s = basis.overlap();
    assert_eq!(s.len(), 28 * 28);

    // The stretched component x^6 = (6,0,0) is first in the canonical order and is
    // the one integral normalizes to unit self-overlap.
    assert!((s[0] - 1.0).abs() < 1e-12, "diag[0] (x^6) = {}", s[0]);
    // All diagonal self-overlaps are finite and positive (no garbage/NaN).
    for i in 0..28 {
        let d = s[i * 28 + i];
        assert!(d.is_finite() && d > 0.0, "diag[{i}] = {d}");
    }
}

#[test]
fn l7_fails_cleanly() {
    let err = Shell::new(7, [0.0, 0.0, 0.0], vec![1.0], vec![1.0]).unwrap_err();
    assert_eq!(err, IntegralError::AngularMomentumTooHigh { l: 7, max: 6 });
    // Error message is descriptive.
    let msg = err.to_string();
    assert!(
        msg.contains("angular momentum") && msg.contains('7') && msg.contains('6'),
        "{msg}"
    );
}
