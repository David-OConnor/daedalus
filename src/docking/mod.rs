//! A new approach, leveraging our molecular dynamics state and processes.

use std::{
    collections::{HashMap, HashSet},
    time::Instant,
};

use bincode::{Decode, Encode};
use bio_files::{create_bonds, md_params::ForceFieldParams};
use dynamics::{
    ComputationDevice, FfMolType, HydrogenConstraint, MdConfig, MdState, MolDynamics, ParamError,
    params::FfParamSet,
};
use graphics::{EngineUpdates, Scene};
use lin_alg::{f32::Vec3 as Vec3F32, f64::Vec3};

use crate::{
    State,
    md::{filter_peptide_atoms, post_run_cleanup, reassign_snapshot_indices, run_dynamics},
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

pub fn dock(
    state: &mut State,
    mol_i: usize,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
) -> Result<(), ParamError> {
    let Some(pep) = state.peptide.as_mut() else {
        return Err(ParamError::new("No peptide; can't dock."));
    };
    let mol = &mut state.ligands[mol_i];
    // Move the ligand away from the docking site prior to vectoring it towards it.

    // todo: QC if you need these.
    pep.common.selected_for_md = true; // Required to properly re-assign snapshot indices.
    mol.common.selected_for_md = true; // Required to not get filtered out in `build_dynamics`.

    let start_dist = 10.;
    let speed = 60.; // Å/ps

    let docking_site = mol.common.centroid(); // for now

    let dir = (docking_site - pep.common.centroid()).to_normalized();

    let starting_posit = docking_site + dir * start_dist;
    let starting_vel = -dir * speed;

    mol.common.move_to(starting_posit);

    let cfg = MdConfig {
        zero_com_drift: false, // May already be false.
        // todo: A/R. Have to relax proteins currently, or hydrogens are likely to end up
        // todo too close to one another.
        max_init_relaxation_iters: Some(300),
        // For now at least. Constrained seems to be blowing up proteins in general, not just
        // for docking.
        hydrogen_constraint: HydrogenConstraint::Flexible,
        ..state.to_save.md_config.clone()
    };

    // todo: Examine and revamp which peptide atoms are included in the sim.

    let mut md_state = build_dynamics_docking(
        &state.dev,
        &mol,
        Some(pep),
        starting_vel.into(),
        &state.ff_param_set,
        &state.lig_specific_params,
        &cfg,
        &mut state.volatile.md_peptide_selected,
    )?;

    state.volatile.md_local.start = Some(Instant::now());
    state.volatile.md_local.running = true;

    // todo: We may opt for a higher-than-normal DT here.
    let dt = 0.002;
    let n_steps = 600;

    // todo: We may need to interrupt periodically e.g. to relax once close.

    // todo: You need a binding energy computation each step.

    // Blocking for now.
    run_dynamics(&mut md_state, &state.dev, dt, n_steps);
    post_run_cleanup(state, scene, engine_updates);

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

    // todo: Let's try: Use all protein atoms, but make all but the ones near the docking site
    // todo both static, and bonded only. This should perhaps anchor the docking site ones in position,
    // todo while allowing their outside area to move?
    //
    // todo: Perhaps mod Dynamics for this case specifically: Make sure all forces
    // todo are skipped for atoms marked as both static and bonded only, except for the bonded
    // todo forces between them and non-static atoms.

    // todo: Looks like with your current dynamics setup,

    let Some(pep) = peptide else {
        return Err(ParamError::new("No peptide; can't dock."));
    };

    // todo: Make sure you're filtering nearby based on the docking config; not hte initial one
    // tood if moving towards it
    // We assume hetero atoms are ligands, water etc, and are not part of the protein.

    // let atoms = filter_peptide_atoms(pep_atom_set, p, &[mol], peptide_only_near_lig);
    // println!("Peptide atom count: {}", atoms.len());
    // let bonds = create_bonds(&atoms);

    // Filter out hetero atoms.
    let pep_atoms = filter_peptide_atoms(pep_atom_set, pep, &[], None);

    // todo: Let's try using all peptide atoms, but assigning certain
    // todo AtomsDynamics to be static and bonded only.
    let bonds = create_bonds(&pep_atoms);

    // todo: Now: How to mark certain *atoms* vs molecules as bonded nly and static.

    mols.push(MolDynamics {
        ff_mol_type: FfMolType::Peptide,
        atoms: pep_atoms,
        atom_posits: None,
        atom_init_velocities: None,
        bonds,
        adjacency_list: None,
        static_: false,
        bonded_only: false,
        mol_specific_params: None,
    });

    // All peptide atoms are included, for the purposes of un-flattening snapshot atoms.
    for i in 0..pep.common.atoms.len() {
        pep_atom_set.insert((0, i));
    }

    //
    println!("Initializing docking MD state...");
    let mut md_state = MdState::new(dev, &cfg, &mols, param_set)?;
    println!("MD init done.");

    // todo: This is a bit awkward location-wise, but ok for now. Consider adding this directly to Dynamics,
    // todo if it makes sense

    // Mark atoms not near the ligand as static and bonded-forces only. These anchor
    // the non-static ones. Bonded force computations are (unnecessarily) run on them, but this is cheap,
    // and scales linearly with atom count.
    let mut pep_set_near = HashSet::new();

    let near_lig_thresh: f64 = 20.; // todo: Experiment
    let _ = filter_peptide_atoms(&mut pep_set_near, pep, &[mol], Some(near_lig_thresh));

    let pep_start_i = mol.common.atoms.len();
    for (i, atom) in md_state.atoms.iter_mut().enumerate() {
        if i < pep_start_i {
            continue;
        }

        let i_pep = i - pep_start_i;
        if !pep_set_near.contains(&(0, i_pep)) {
            atom.bonded_only = true;
            atom.static_ = true;
        }
    }

    Ok(md_state)
}
