//! Periodic / GPW machinery (Gaussian and Plane Waves), behind the `periodic` feature.
//!
//! This module holds the "density-matrix ↔ grid ↔ potential-matrix" infrastructure for
//! periodic Kohn–Sham DFT: a uniform real-space grid over a [`latx::Cell`], a 3-D FFT, and
//! the reciprocal-space Poisson solve for the Hartree potential. Collocation
//! (`P → n(r)`), grid→matrix integration (`V(r) → V_μν`), GTH nonlocal projectors, and
//! Bloch/lattice sums are layered on in later milestones.
//!
//! Everything here is gated by the `periodic` feature so the default `integral` build stays
//! dependency-free; see the crate-level docs for the public overview.
//!
//! ## Conventions
//!
//! - **Units:** atomic units (bohr); the Hartree potential is in hartree.
//! - **Grid layout:** a point with grid index `[i, j, k]` sits at fractional coordinates
//!   `[i/n1, j/n2, k/n3]`; arrays are row-major with `k` fastest
//!   ([`RealSpaceGrid::linear_index`]), matching the FFT's index order.

pub mod bloch;
pub mod collocate;
mod fft;
pub mod grid;
pub mod lattice;
pub mod multigrid;
pub mod poisson;
pub mod projector;

pub use bloch::{
    bloch_kinetic, bloch_kinetic_grad_contract, bloch_kinetic_stress_contract, bloch_overlap,
    bloch_overlap_grad_contract, bloch_overlap_stress_contract,
};
pub use collocate::{collocate_density, integrate_potential, project_function};
pub use grid::RealSpaceGrid;
pub use lattice::{BlochPhases, ChiCache, ImageBlocks, LatticeCollocator};
pub use multigrid::{MultiBlochPhases, MultiGridCollocator};
pub use poisson::{hartree, hartree_reciprocal_stress};
pub use projector::{
    bloch_projector_overlaps, bloch_projector_overlaps_grad, bloch_projector_overlaps_strain,
    projector_overlaps, ProjectorChannel,
};
