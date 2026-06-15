//! A small 3-D complex FFT built on `rustfft`.
//!
//! The density is real, but v1 uses a full complex transform (real promoted to complex,
//! imaginary part discarded at the end of the Poisson solve) — the simplest correct route.
//! Switching the contiguous axis to a real-to-complex (`realfft`) transform is a deferred
//! memory/throughput optimization (it does not change results); see the plan's decision log.
//!
//! `rustfft` transforms operate on a contiguous buffer, so a transform along a non-contiguous
//! axis gathers each line into a scratch buffer, transforms it, and scatters it back.
//! Forward and inverse are both **unnormalized** (`inverse ∘ forward = N · identity`); the
//! Poisson solver folds the `1/N` into its reciprocal-space coefficients, so it is not
//! applied here.

use rustfft::num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// A cached set of per-axis FFT plans for a fixed `n1 × n2 × n3` grid.
pub(crate) struct Fft3d {
    n: [usize; 3],
    fwd: [Arc<dyn Fft<f64>>; 3],
    inv: [Arc<dyn Fft<f64>>; 3],
}

impl Fft3d {
    /// Build plans for a grid of dimensions `n = [n1, n2, n3]`.
    pub(crate) fn new(n: [usize; 3]) -> Self {
        let mut planner = FftPlanner::new();
        let fwd = [
            planner.plan_fft_forward(n[0]),
            planner.plan_fft_forward(n[1]),
            planner.plan_fft_forward(n[2]),
        ];
        let inv = [
            planner.plan_fft_inverse(n[0]),
            planner.plan_fft_inverse(n[1]),
            planner.plan_fft_inverse(n[2]),
        ];
        Self { n, fwd, inv }
    }

    /// Forward transform in place: `out[G] = Σ_r data[r] · e^{-iG·r}` (unnormalized).
    pub(crate) fn forward(&self, data: &mut [Complex64]) {
        for axis in 0..3 {
            self.transform_axis(data, axis, self.fwd[axis].as_ref());
        }
    }

    /// Inverse transform in place: `out[r] = Σ_G data[G] · e^{+iG·r}` (unnormalized).
    pub(crate) fn inverse(&self, data: &mut [Complex64]) {
        for axis in 0..3 {
            self.transform_axis(data, axis, self.inv[axis].as_ref());
        }
    }

    fn transform_axis(&self, data: &mut [Complex64], axis: usize, plan: &dyn Fft<f64>) {
        let [n0, n1, n2] = self.n;
        debug_assert_eq!(data.len(), n0 * n1 * n2);
        let len = self.n[axis];
        // Stride of the transform axis, and (extent, stride) of the two remaining axes.
        let (stride, others): (usize, [(usize, usize); 2]) = match axis {
            0 => (n1 * n2, [(n1, n2), (n2, 1)]),
            1 => (n2, [(n0, n1 * n2), (n2, 1)]),
            2 => (1, [(n0, n1 * n2), (n1, n2)]),
            _ => unreachable!(),
        };
        let zero = Complex64::new(0.0, 0.0);
        let mut scratch = vec![zero; plan.get_inplace_scratch_len()];
        let mut line = vec![zero; len];
        let (ea, sa) = others[0];
        let (eb, sb) = others[1];
        for a in 0..ea {
            for b in 0..eb {
                let start = a * sa + b * sb;
                for (t, l) in line.iter_mut().enumerate() {
                    *l = data[start + t * stride];
                }
                plan.process_with_scratch(&mut line, &mut scratch);
                for (t, l) in line.iter().enumerate() {
                    data[start + t * stride] = *l;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: &[Complex64], b: &[Complex64], tol: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).norm() < tol)
    }

    #[test]
    fn round_trip_recovers_input() {
        let n = [4usize, 5, 6];
        let npts = n[0] * n[1] * n[2];
        // A deterministic, non-symmetric complex input.
        let input: Vec<Complex64> = (0..npts)
            .map(|i| Complex64::new((i as f64 * 0.37).sin(), (i as f64 * 0.11 + 1.0).cos()))
            .collect();
        let fft = Fft3d::new(n);
        let mut data = input.clone();
        fft.forward(&mut data);
        fft.inverse(&mut data);
        // inverse ∘ forward = N · identity.
        let nfac = npts as f64;
        for d in &mut data {
            *d /= nfac;
        }
        assert!(approx_eq(&data, &input, 1e-12));
    }

    #[test]
    fn forward_of_constant_is_a_single_peak() {
        let n = [3usize, 3, 3];
        let npts = n[0] * n[1] * n[2];
        let mut data = vec![Complex64::new(2.0, 0.0); npts];
        Fft3d::new(n).forward(&mut data);
        // DC bin = Σ data = npts · 2; all others ≈ 0.
        assert!((data[0] - Complex64::new(2.0 * npts as f64, 0.0)).norm() < 1e-10);
        for d in &data[1..] {
            assert!(d.norm() < 1e-10);
        }
    }
}
