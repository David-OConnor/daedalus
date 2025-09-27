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
use lin_alg::f64::Vec3;

use crate::{
    lipid::MoleculeLipid,
    mol_lig::MoleculeSmall,
    molecule::{MoleculeCommon, MoleculePeptide},
};

// Å. Static atoms must be at least this close to a dynamic atom at the start of MD to be counted.
// Set this wide to take into account motion.
const STATIC_ATOM_DIST_THRESH: f64 = 8.; // todo: Increase (?) A/R.

/// Perform MD on selected molecules.
pub fn build_dynamics(
    #[cfg(feature = "cuda")] dev: &(ComputationDevice, Option<Arc<CudaModule>>),
    #[cfg(not(feature = "cuda"))] dev: &ComputationDevice,
    ligs: Vec<&mut MoleculeSmall>,
    lipids: Vec<&mut MoleculeLipid>,
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

    for mol in &ligs {
        if !mol.common.selected_for_md {
            continue;
        }
        let atoms_gen: Vec<_> = mol.common.atoms.iter().map(|a| a.to_generic()).collect();
        let bonds_gen: Vec<_> = mol.common.bonds.iter().map(|b| b.to_generic()).collect();

        let Some(msp) = mol_specific_params.get(&mol.common.ident) else {
            return Err(ParamError::new(&format!(
                "Missing molecule-specific parameters for  {}",
                mol.common.ident
            )));
        };

        mols.push(MolDynamics {
            ff_mol_type: FfMolType::SmallOrganic,
            atoms: atoms_gen,
            atom_posits: Some(mol.common.atom_posits.clone()),
            bonds: bonds_gen,
            adjacency_list: Some(mol.common.adjacency_list.clone()),
            static_: false,
            bonded_only: false,
            mol_specific_params: Some(msp.clone()),
        })
    }

    // todo: DRY
    for mol in &ligs {
        if !mol.common.selected_for_md {
            continue;
        }
        let atoms_gen: Vec<_> = mol.common.atoms.iter().map(|a| a.to_generic()).collect();
        let bonds_gen: Vec<_> = mol.common.bonds.iter().map(|b| b.to_generic()).collect();

        let Some(msp) = mol_specific_params.get(&mol.common.ident) else {
            return Err(ParamError::new(&format!(
                "Missing molecule-specific parameters for  {}",
                mol.common.ident
            )));
        };

        mols.push(MolDynamics {
            ff_mol_type: FfMolType::Lipid,
            atoms: atoms_gen,
            atom_posits: Some(mol.common.atom_posits.clone()),
            bonds: bonds_gen,
            adjacency_list: Some(mol.common.adjacency_list.clone()),
            static_: false,
            bonded_only: false,
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
            bonded_only: false,
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

    // We filtered peptide hetero atoms out above. Adjust snapshot indices and atom positions so they
    // are properly synchronized.
    // todo: Delegat this to its own fn A/R.
    {
        println!("Re-assigning snapshot indices to match atoms excluded for MD...");
        if let Some(p) = peptide {
            // Which peptide atoms were included in MD (same logic as when building `atoms` above)
            let included: Vec<bool> = p
                .common
                .atoms
                .iter()
                .map(|a| {
                    if !peptide_only_near_lig {
                        !a.hetero
                    } else {
                        let mut closest_dist = f64::MAX;
                        for lig in &ligs {
                            for lp in &lig.common.atom_posits {
                                let d = (*lp - a.posit).magnitude();
                                if d < closest_dist {
                                    closest_dist = d;
                                }
                            }
                        }
                        !a.hetero && closest_dist < STATIC_ATOM_DIST_THRESH
                    }
                })
                .collect();

            let n_included = included.iter().filter(|&&b| b).count();

            // Count how many ligand atoms precede the peptide in the snapshot ordering.
            let lig_atom_count: usize = ligs
                .iter()
                .filter(|l| l.common.selected_for_md)
                .map(|l| l.common.atoms.len())
                .sum();

            let lipid_atom_count: usize = lipids
                .iter()
                .filter(|l| l.common.selected_for_md)
                .map(|l| l.common.atoms.len())
                .sum();

            let pep_start_i = lig_atom_count + lipid_atom_count;

            // Rebuild each snapshot's atom_posits: [ligands as-is] + [full peptide with holes filled]
            for snap in &mut md_state.snapshots {
                // Iterator over the peptide positions that actually participated in MD
                let mut pept_md_posits = snap.atom_posits[pep_start_i..pep_start_i + n_included]
                    .iter()
                    .cloned();

                let mut pept_md_vels = snap.atom_velocities[pep_start_i..pep_start_i + n_included]
                    .iter()
                    .cloned();

                let mut new_posits = Vec::with_capacity(pep_start_i + p.common.atoms.len());
                let mut new_vels = Vec::with_capacity(pep_start_i + p.common.atoms.len());

                // Keep ligand portion unchanged
                new_posits.extend_from_slice(&snap.atom_posits[..pep_start_i]);
                new_vels.extend_from_slice(&snap.atom_velocities[..pep_start_i]);

                // Insert peptide atoms in original index order; use MD-updated posits if included,
                // otherwise the original (excluded) atom position.
                // todo: Do velocities too A/R.
                for (a, inc) in p.common.atoms.iter().zip(included.iter()) {
                    if *inc {
                        new_posits
                            .push(pept_md_posits.next().expect("peptide MD posits exhausted"));
                        new_vels.push(
                            pept_md_vels
                                .next()
                                .expect("peptide MD velocities exhausted"),
                        );
                    } else {
                        new_posits.push(a.posit.into());
                        new_vels.push(lin_alg::f32::Vec3::new_zero());
                    }
                }

                // Replace the snapshot's positions with the reindexed set
                snap.atom_posits = new_posits;
                snap.atom_velocities = new_vels;
            }
        }
        println!("Done.");
    }

    // change_snapshot(peptide, ligs, &md_state.snapshots[0]);

    Ok(md_state)
}

fn change_snapshot_helper(posits: &mut [Vec3], start_i_this_mol: &mut usize, snapshot: &Snapshot) {
    // Unflatten.
    for (i_snap, posit) in snapshot.atom_posits.iter().enumerate() {
        if i_snap < *start_i_this_mol || i_snap >= posits.len() + *start_i_this_mol {
            continue;
        }
        posits[i_snap - *start_i_this_mol] = (*posit).into();
    }

    *start_i_this_mol += posits.len();
}

/// Set atom positions for molecules involve in dynamics to that of a snapshot. Ligs and lipids are only ones included
/// in dynamics.
pub fn change_snapshot(
    peptide: Option<&mut MoleculePeptide>,
    ligs: Vec<&mut MoleculeSmall>,
    lipids: Vec<&mut MoleculeLipid>,
    snapshot: &Snapshot,
) {
    let mut start_i_this_mol = 0;

    for mol in ligs {
        change_snapshot_helper(&mut mol.common.atom_posits, &mut start_i_this_mol, snapshot);
    }

    for mol in lipids {
        change_snapshot_helper(&mut mol.common.atom_posits, &mut start_i_this_mol, snapshot);
    }

    if let Some(mol) = peptide {
        change_snapshot_helper(&mut mol.common.atom_posits, &mut start_i_this_mol, snapshot);
    }
}
