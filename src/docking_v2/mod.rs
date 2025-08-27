//! A new approach, leveraging our molecular dynamics state and processes.

use lin_alg::f64::Vec3;

pub struct DockingPose {
    lig_atom_posits: Vec<Vec3>,
    potential_energy: f64,
}

#[derive(Debug, Default)]
pub struct DockingStateV2 {}
