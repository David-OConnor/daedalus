//! A new approach, leveraging our molecular dynamics state and processes.

use std::collections::{HashMap, HashSet};

use bincode::{Decode, Encode};
use bio_files::{create_bonds, md_params::ForceFieldParams};
use dynamics::{
    ComputationDevice, FfMolType, MdConfig, MdState, MolDynamics, ParamError, params::FfParamSet,
};
use lin_alg::{f32::Vec3 as Vec3F32, f64::Vec3};

use crate::{
    State,
    lipid::MoleculeLipid,
    md::{build_dynamics, filter_peptide_atoms, reassign_snapshot_indices, run_dynamics},
    mol_lig::MoleculeSmall,
    molecule::MoleculePeptide,
};

#[derive(Clone, Debug, Default)]
/// Bonds that are marked as flexible, using a semi-rigid conformation.
pub struct Torsion {
    pub bond: usize, // Index.
    pub dihedral_angle: f32,
}

#[derive(Debug, Clone, PartialEq, Encode, Decode)]
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
    let peptide = state.peptide.as_mut().unwrap(); // ?
    let mol = &mut state.ligands[mol_i];
    // Move the ligand away from the docking site prior to vectoring it towards it.

    peptide.common.selected_for_md = true; // Required to properly re-assign snapshot indices.
    mol.common.selected_for_md = true; // Required to not get filtered out in `build_dynamics`.

    let start_dist = 10.;
    let speed = 60.; // Å/ps

    let docking_site = mol.common.centroid(); // for now

    let dir = (docking_site - peptide.common.centroid()).to_normalized();

    let starting_posit = docking_site + dir * start_dist;
    let starting_vel = -dir * speed;

    mol.common.move_to(starting_posit);

    let cfg = MdConfig {
        zero_com_drift: false, // May already be false.
        // Currently we must skip this for Velocity to not be 0ed. This is fixable.
        max_init_relaxation_iters: None,
        ..state.to_save.md_config.clone()
    };

    // todo: Examine and revamp which peptide atoms are included in the sim.

    let mut md_state = build_dynamics_docking(
        &state.dev,
        &mol,
        Some(peptide),
        starting_vel.into(),
        &state.ff_param_set,
        &state.lig_specific_params,
        &cfg,
        true,
        true,
        &mut state.volatile.md_peptide_selected,
    )?;

    // todo: We may opt for a higher-than-normal DT here.
    let dt = 0.002;
    let n_steps = 1_000;

    // todo: We may need to interrupt periodically e.g. to relax once close.

    // todo: You need a binding energy computation each step.

    run_dynamics(&mut md_state, &state.dev, dt, n_steps);

    reassign_snapshot_indices(
        peptide,
        &[mol],
        &Vec::new(),
        &mut md_state.snapshots,
        &state.volatile.md_peptide_selected,
    );

    state.mol_dynamics = Some(md_state);

    Ok(())
}

// todo: DRy with the primary MD setup fn.
fn build_dynamics_docking(
    dev: &ComputationDevice,
    mol: &MoleculeSmall,
    peptide: Option<&MoleculePeptide>,
    starting_vel: Vec3F32,
    param_set: &FfParamSet,
    mol_specific_params: &HashMap<String, ForceFieldParams>,
    cfg: &MdConfig,
    mut static_peptide: bool,
    peptide_only_near_lig: bool,
    pep_atom_set: &mut HashSet<(usize, usize)>,
) -> Result<MdState, ParamError> {
    println!("Setting up docking dynamics...");

    let mut mols = Vec::new();

    let atoms_gen: Vec<_> = mol.common.atoms.iter().map(|a| a.to_generic()).collect();
    let bonds_gen: Vec<_> = mol.common.bonds.iter().map(|b| b.to_generic()).collect();

    let Some(msp) = mol_specific_params.get(&mol.common.ident) else {
        return Err(ParamError::new(&format!(
            "Missing molecule-specific parameters for  {}",
            mol.common.ident
        )));
    };

    let atom_initial_velocities = vec![starting_vel; mol.common.atoms.len()];

    mols.push(MolDynamics {
        ff_mol_type: FfMolType::SmallOrganic,
        atoms: atoms_gen,
        atom_posits: Some(mol.common.atom_posits.clone()),
        atom_init_velocities: Some(atom_initial_velocities),
        bonds: bonds_gen,
        adjacency_list: Some(mol.common.adjacency_list.clone()),
        static_: false,
        bonded_only: false,
        mol_specific_params: Some(msp.clone()),
    });

    if let Some(p) = peptide {
        // todo: Make sure you're filtering nearby based on the docking config; not hte initial one
        // tood if moving towards it
        // We assume hetero atoms are ligands, water etc, and are not part of the protein.
        let atoms = filter_peptide_atoms(pep_atom_set, p, &[mol], peptide_only_near_lig);
        println!("Peptide atom count: {}", atoms.len());

        let bonds = create_bonds(&atoms);

        mols.push(MolDynamics {
            ff_mol_type: FfMolType::Peptide,
            atoms,
            atom_posits: None,
            atom_init_velocities: None,
            bonds,
            adjacency_list: None,
            static_: static_peptide,
            bonded_only: false,
            mol_specific_params: None,
        });
    }

    //
    println!("Initializing docking MD state...");
    let md_state = MdState::new(dev, &cfg, &mols, param_set)?;
    println!("Done.");

    Ok(md_state)
}
