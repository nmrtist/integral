//! The uniform real-space grid over a periodic cell, and its reciprocal (FFT) vectors.

use latx::Cell;

/// A uniform real-space grid of `n1 × n2 × n3` points spanning one [`Cell`].
///
/// Grid index `[i, j, k]` (with `0 ≤ i < n1`, etc.) sits at fractional coordinates
/// `[i/n1, j/n2, k/n3]`. Field arrays (density, potential) are row-major with `k` fastest;
/// use [`RealSpaceGrid::linear_index`] for the flat offset. The volume element is the
/// constant `dV = Ω / (n1 n2 n3)`.
#[derive(Debug, Clone)]
pub struct RealSpaceGrid {
    cell: Cell,
    n: [usize; 3],
    dv: f64,
}

impl RealSpaceGrid {
    /// A grid of explicit dimensions `n = [n1, n2, n3]` over `cell`.
    ///
    /// # Panics
    /// Panics if any dimension is zero.
    #[must_use]
    pub fn new(cell: Cell, n: [usize; 3]) -> Self {
        assert!(
            n.iter().all(|&x| x > 0),
            "grid dimensions must be > 0, got {n:?}"
        );
        let dv = cell.volume() / (n[0] * n[1] * n[2]) as f64;
        Self { cell, n, dv }
    }

    /// A grid sized from a **density** plane-wave cutoff `e_cut` (hartree).
    ///
    /// The largest representable density wavevector is `G_max = √(2·e_cut)`; along lattice
    /// direction `i` the grid must satisfy `Nᵢ ≥ G_max·dᵢ/π = 2·G_max/|bᵢ|`, where
    /// `dᵢ = 2π/|bᵢ|` is the interplanar spacing. Each `Nᵢ` is rounded up to the next
    /// 7-smooth integer (a fast FFT size).
    ///
    /// # Panics
    /// Panics if `e_cut` is not positive.
    #[must_use]
    pub fn from_cutoff(cell: Cell, e_cut: f64) -> Self {
        assert!(e_cut > 0.0, "density cutoff must be > 0, got {e_cut}");
        let gmax = (2.0 * e_cut).sqrt();
        let (b0, b1, b2) = cell.reciprocal().vectors();
        let need = |b: [f64; 3]| {
            let nb = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
            next_smooth(((2.0 * gmax / nb).ceil() as usize).max(1))
        };
        Self::new(cell, [need(b0), need(b1), need(b2)])
    }

    /// The underlying cell.
    #[must_use]
    pub fn cell(&self) -> &Cell {
        &self.cell
    }

    /// The grid dimensions `[n1, n2, n3]`.
    #[must_use]
    pub fn n(&self) -> [usize; 3] {
        self.n
    }

    /// The total number of grid points `n1·n2·n3`.
    #[must_use]
    pub fn n_points(&self) -> usize {
        self.n[0] * self.n[1] * self.n[2]
    }

    /// The (constant) real-space volume element `dV = Ω / N`.
    #[must_use]
    pub fn dv(&self) -> f64 {
        self.dv
    }

    /// The flat row-major offset of grid index `[i, j, k]` (`k` fastest).
    #[must_use]
    pub fn linear_index(&self, idx: [usize; 3]) -> usize {
        (idx[0] * self.n[1] + idx[1]) * self.n[2] + idx[2]
    }

    /// The Cartesian position (bohr) of grid index `[i, j, k]`.
    #[must_use]
    pub fn point(&self, idx: [usize; 3]) -> [f64; 3] {
        let f = [
            idx[0] as f64 / self.n[0] as f64,
            idx[1] as f64 / self.n[1] as f64,
            idx[2] as f64 / self.n[2] as f64,
        ];
        self.cell.frac_to_cart(f)
    }

    /// All grid-point Cartesian positions (bohr), row-major (`k` fastest).
    #[must_use]
    pub fn points(&self) -> Vec<[f64; 3]> {
        let mut pts = Vec::with_capacity(self.n_points());
        for i in 0..self.n[0] {
            for j in 0..self.n[1] {
                for k in 0..self.n[2] {
                    pts.push(self.point([i, j, k]));
                }
            }
        }
        pts
    }

    /// The reciprocal-space vector `G` (bohr⁻¹) for every FFT bin, row-major in the same
    /// order as the density/potential field arrays. Used by the Poisson solver.
    pub(crate) fn g_vectors(&self) -> Vec<[f64; 3]> {
        let recip = self.cell.reciprocal();
        let mut g = Vec::with_capacity(self.n_points());
        for i in 0..self.n[0] {
            let fi = signed_freq(i, self.n[0]);
            for j in 0..self.n[1] {
                let fj = signed_freq(j, self.n[1]);
                for k in 0..self.n[2] {
                    let fk = signed_freq(k, self.n[2]);
                    g.push(recip.g_vector([fi, fj, fk]));
                }
            }
        }
        g
    }
}

/// The signed DFT frequency for bin `i` of a length-`n` axis: `0, 1, …, ⌊n/2⌋, …, -1`.
fn signed_freq(i: usize, n: usize) -> i32 {
    if i <= n / 2 {
        i as i32
    } else {
        i as i32 - n as i32
    }
}

/// The smallest 7-smooth integer `≥ n` (only prime factors 2, 3, 5, 7) — a fast FFT length.
fn next_smooth(mut n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    loop {
        let mut m = n;
        for p in [2usize, 3, 5, 7] {
            while m % p == 0 {
                m /= p;
            }
        }
        if m == 1 {
            return n;
        }
        n += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_count_and_volume_element() {
        let grid = RealSpaceGrid::new(Cell::cubic(4.0).unwrap(), [8, 8, 8]);
        assert_eq!(grid.n_points(), 512);
        // dV · N = Ω.
        assert!((grid.dv() * grid.n_points() as f64 - 64.0).abs() < 1e-10);
    }

    #[test]
    fn grid_points_lie_in_cell() {
        let grid = RealSpaceGrid::new(Cell::cubic(2.0).unwrap(), [4, 4, 4]);
        // Corner [0,0,0] is the origin; [2,2,2] is the cell centre.
        assert_eq!(grid.point([0, 0, 0]), [0.0, 0.0, 0.0]);
        let mid = grid.point([2, 2, 2]);
        for c in mid {
            assert!((c - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn next_smooth_rounds_up() {
        assert_eq!(next_smooth(1), 1);
        assert_eq!(next_smooth(2), 2);
        assert_eq!(next_smooth(11), 12); // 11 is prime → 12 = 2²·3
        assert_eq!(next_smooth(13), 14); // 14 = 2·7
        assert_eq!(next_smooth(625), 625); // 5⁴
    }

    #[test]
    fn from_cutoff_grows_with_cutoff() {
        let cell = Cell::cubic(10.0).unwrap();
        let coarse = RealSpaceGrid::from_cutoff(cell, 20.0);
        let fine = RealSpaceGrid::from_cutoff(cell, 80.0);
        assert!(fine.n()[0] >= coarse.n()[0]);
        // Each dimension is 7-smooth.
        for &nd in &fine.n() {
            assert_eq!(next_smooth(nd), nd);
        }
    }

    #[test]
    fn gamma_g_vector_is_zero() {
        let grid = RealSpaceGrid::new(Cell::cubic(5.0).unwrap(), [4, 4, 4]);
        let g = grid.g_vectors();
        assert_eq!(g[0], [0.0, 0.0, 0.0]); // bin [0,0,0] is G = 0
    }
}
