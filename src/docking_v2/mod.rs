//! A new approach, leveraging our molecular dynamics state and processes.

use bincode::{Decode, Encode};
use dynamics::{MdConfig, ParamError};
use lin_alg::f64::Vec3;

use crate::{
    State,
    md::{build_dynamics, reassign_snapshot_indices, run_dynamics},
};

#[derive(Clone, Debug, Default)]
/// Bonds that are marked as flexible, using a semi-rigid conformation.
pub struct Torsion {
    pub bond: usize, // Index.
    pub dihedral_angle: f32,
}

#[derive(Debug, Clone, Encode, Decode)]
/// Area IVO the docking site.
pub struct DockingSite {
    pub site_center: Vec3,
    pub site_radius: f64,
}

impl Default for DockingSite {
    fn default() -> Self {
        Self {
            site_center: Vec3::new_zero(),
            site_radius: 8.,
        }
    }
}

// // todo: Rem if not used.
// #[derive(Clone, Debug, Default)]
// pub enum ConformationType {
//     #[default]
//     /// Don't reposition atoms based on the pose. This is what we use when assigning each atom
//     /// a position using molecular dynamics.
//     AbsolutePosits,
//     // Rigid,
//     /// Certain bonds are marked as flexible, with rotation allowed around them.
//     AssignedTorsions { torsions: Vec<Torsion> },
// }

// todo: Rem if not used.
#[derive(Clone, Debug, Default)]
pub struct Pose {
    // pub conformation_type: ConformationType,
    // /// The offset of the ligand's anchor atom from the docking center.
    // /// Only for rigid and torsion-set-based conformations.
    // /// todo: Consider normalizing positions to be around the origin, for numerical precision issues.
    // pub anchor_posit: Vec3,
    // /// Only for rigid and torsion-set-based conformations.
    // pub orientation: Quaternion,
    pub posits: Vec<Vec3>,
}

pub struct DockingPose {
    lig_atom_posits: Vec<Vec3>,
    potential_energy: f64,
}

#[derive(Debug, Default)]
pub struct DockingState {}

pub fn dock(state: &mut State, mol_i: usize) -> Result<(), ParamError> {
    let num_steps = 100;

    let peptide = state.peptide.as_ref().unwrap(); // ?
    let mol = &mut state.ligands[mol_i];
    // Move the ligand away from the docking site prior to vectoring it towards it.

    let start_dist = 10.;
    let speed = 1_000.; // Å/ps

    let docking_site = mol.common.centroid(); // for now

    // let dir = (mol.common.centroid() - state.volatile.docking_site_center).to_normalized();
    let dir = (peptide.common.centroid() - docking_site).to_normalized();

    let starting_posit = docking_site + dir * start_dist;
    let starting_vel = dir * speed;

    mol.common.move_to(starting_posit);

    let ligs = vec![mol];

    let cfg = MdConfig {
        zero_com_drift: false, // May already be false.
        ..state.to_save.md_config.clone()
    };

    let mut md_state = build_dynamics(
        &state.dev,
        &ligs,
        &Vec::new(),
        Some(peptide),
        &state.ff_param_set,
        &state.lig_specific_params,
        &cfg,
        true,
        true,
        &mut state.volatile.md_peptide_selected,
    )?;

    let dt = 0.002;
    let n_steps = 100;
    run_dynamics(&mut md_state, &state.dev, dt, n_steps);

    reassign_snapshot_indices(
        peptide,
        &ligs,
        &Vec::new(),
        &mut md_state.snapshots,
        &state.volatile.md_peptide_selected,
    );

    state.mol_dynamics = Some(md_state);

    Ok(())
}
