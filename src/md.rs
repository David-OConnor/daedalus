//! An interface to dynamics library.

use std::{
    collections::HashMap,
    sync::{
        Arc,
        mpsc::{self, Receiver},
    },
    thread,
    time::Instant,
};

use bio_files::{create_bonds, md_params::ForceFieldParams};
#[cfg(feature = "cuda")]
use cudarc::driver::CudaModule;
use dynamics::{
    ComputationDevice, FfMolType, MdConfig, MdState, MolDynamics, ParamError, params::FfParamSet,
    snapshot::Snapshot,
};

use crate::{mol_lig::MoleculeSmall, molecule::MoleculePeptide};

// Å. Static atoms must be at least this close to a dynamic atom at the start of MD to be counted.
// Set this wide to take into account motion.
const STATIC_ATOM_DIST_THRESH: f64 = 8.; // todo: Increase (?) A/R.

/// Perform MD on the ligand, with nearby protein (receptor) atoms, from the docking setup as static
/// non-bonded contributors. (Vdw and coulomb)
pub fn build_dynamics(
    #[cfg(feature = "cuda")] dev: &(ComputationDevice, Option<Arc<CudaModule>>),
    #[cfg(not(feature = "cuda"))] dev: &ComputationDevice,
    ligs: Vec<&mut MoleculeSmall>,
    peptide: Option<&MoleculePeptide>,
    param_set: &FfParamSet,
    mol_specific_params: &HashMap<String, ForceFieldParams>,
    cfg: &MdConfig,
    n_steps: u32,
    static_peptide: bool,
    peptide_only_near_lig: bool,
    dt: f32,
) -> Result<MdState, ParamError> {
    println!("Setting up dynamics...");

    let mut mols = Vec::new();

    for lig in &ligs {
        let atoms_gen: Vec<_> = lig.common.atoms.iter().map(|a| a.to_generic()).collect();
        let bonds_gen: Vec<_> = lig.common.bonds.iter().map(|b| b.to_generic()).collect();

        let Some(msp) = mol_specific_params.get(&lig.common.ident) else {
            return Err(ParamError::new(&format!(
                "Missing molecule-specific parameters for  {}",
                lig.common.ident
            )));
        };

        mols.push(MolDynamics {
            ff_mol_type: FfMolType::SmallOrganic,
            atoms: atoms_gen,
            atom_posits: Some(lig.common.atom_posits.clone()),
            bonds: bonds_gen,
            adjacency_list: Some(lig.common.adjacency_list.clone()),
            static_: false,
            mol_specific_params: Some(msp.clone()),
        })
    }

    if let Some(p) = peptide {
        // We assume hetero atoms are ligands, water etc, and are not part of the protein.
        let atoms: Vec<_> = p
            .common
            .atoms
            .iter()
            .filter(|a| {
                if !peptide_only_near_lig {
                    return !a.hetero;
                }

                let mut closest_dist = f64::MAX;
                for lig in &ligs {
                    // todo: Use protein atom.posits A/R.
                    for p in &lig.common.atom_posits {
                        let dist = (*p - a.posit).magnitude();
                        if dist < closest_dist {
                            closest_dist = dist;
                        }
                    }
                }

                !a.hetero && closest_dist < STATIC_ATOM_DIST_THRESH
            })
            .map(|a| a.to_generic())
            .collect();

        let bonds = create_bonds(&atoms);

        mols.push(MolDynamics {
            ff_mol_type: FfMolType::Peptide,
            atoms,
            // todo: A/R if you allow moving the peptide.
            atom_posits: None,
            bonds,
            adjacency_list: None,
            static_: static_peptide,
            mol_specific_params: None,
        })
    }

    println!("Initializing MD state...");
    let mut md_state = MdState::new(&dev.0, cfg, &mols, param_set)?;
    println!("Done.");

    let start = Instant::now();

    for _ in 0..n_steps {
        md_state.step(&dev.0, dt);
    }

    let elapsed = start.elapsed();
    println!(
        "MD complete in {:.2} s",
        elapsed.as_millis() as f32 / 1_000.
    );

    println!(
        "Neighbor rebuild count: {} time: {}ms",
        md_state.neighbor_rebuild_count,
        md_state.neighbor_rebuild_us / 1_000
    );

    change_snapshot(ligs, &md_state.snapshots[0]);

    Ok(md_state)
}

/// Set atom positions for molecules involve in dynamics to that of a snapshot.
pub fn change_snapshot_form2(ligs: &mut [MoleculeSmall], snapshot: &Snapshot) {
    // todo: Handle peptide too!

    // todo: QC this logic.

    // Unflatten.
    let mut start_i_this_mol = 0;
    for lig in ligs {
        for (i_snap, posit) in snapshot.atom_posits.iter().enumerate() {
            if i_snap < start_i_this_mol
                || i_snap >= lig.common.atom_posits.len() + start_i_this_mol
            {
                continue;
            }
            lig.common.atom_posits[i_snap - start_i_this_mol] = (*posit).into();
        }

        start_i_this_mol += lig.common.atom_posits.len();
    }
}

// todo: This is so annoying. &[T] vs [&T].
/// Set atom positions for molecules involve in dynamics to that of a snapshot.
pub fn change_snapshot(ligs: Vec<&mut MoleculeSmall>, snapshot: &Snapshot) {
    // todo: Handle peptide too!

    // todo: QC this logic.

    // Unflatten.
    let mut start_i_this_mol = 0;
    for lig in ligs {
        for (i, posit) in snapshot.atom_posits.iter().enumerate() {
            if i < start_i_this_mol || i >= lig.common.atom_posits.len() + start_i_this_mol {
                continue;
            }
            lig.common.atom_posits[i - start_i_this_mol] = (*posit).into();
        }

        start_i_this_mol += lig.common.atom_posits.len();
    }
}
