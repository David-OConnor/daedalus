//! An interface to dynamics library.

use std::{collections::HashMap, sync::Arc, time::Instant};
use std::collections::HashSet;
use bio_files::{create_bonds, md_params::ForceFieldParams, AtomGeneric};
#[cfg(feature = "cuda")]
use cudarc::driver::CudaModule;
use dynamics::{ComputationDevice, FfMolType, MdConfig, MdState, MolDynamics, ParamError, params::FfParamSet, snapshot::Snapshot, AtomDynamics};
use lin_alg::f64::Vec3;

use crate::{lipid::MoleculeLipid, mol_lig::MoleculeSmall, molecule::MoleculePeptide};
use crate::mol_lig::Ligand;

// Å. Static atoms must be at least this close to a dynamic atom at the start of MD to be counted.
// Set this wide to take into account motion.
pub const STATIC_ATOM_DIST_THRESH: f64 = 14.;

pub fn build_and_run_dynamics(
    dev: &ComputationDevice,
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
    pep_atom_set: &mut HashSet<(usize, usize)>,
) -> Result<MdState, ParamError> {
    let mut md_state = build_dynamics(
        dev,
        &ligs,
        &lipids,
        peptide,
        param_set,
        mol_specific_params,
        cfg,
        static_peptide,
        peptide_only_near_lig,
        pep_atom_set,
    )?;

    run_dynamics(&mut md_state, dev, dt, n_steps as usize);

    if let Some(p) = peptide {
        reassign_snapshot_indices(
            p,
            &ligs,
            &lipids,
            &mut md_state.snapshots,
            pep_atom_set,
        );
    }

    Ok(md_state)
}

fn filter_peptide_atoms(set: &mut HashSet<(usize, usize)>, pep: &MoleculePeptide, ligs: &[&mut MoleculeSmall], only_near_lig: bool) -> Vec<AtomGeneric> {
    *set = HashSet::new();

    pep
        .common
        .atoms
        .iter()
        .enumerate()
        .filter_map(|(i, a)| {
            let pass = if !only_near_lig {
                !a.hetero
            } else {
                let mut closest_dist = f64::MAX;
                for lig in ligs {
                    for p in &lig.common.atom_posits {
                        let dist = (*p - pep.common.atom_posits[i]).magnitude();
                        if dist < closest_dist {
                            closest_dist = dist;
                        }
                    }
                }
                !a.hetero && closest_dist < STATIC_ATOM_DIST_THRESH
            };

            if pass {
                set.insert((0, i));
                Some(a.to_generic())
            } else {
                None
            }
        })
        .collect()
}

/// Set up MD for selected molecules.
pub fn build_dynamics(
    dev: &ComputationDevice,
    ligs: &[&mut MoleculeSmall],
    lipids: &[&mut MoleculeLipid],
    peptide: Option<&MoleculePeptide>,
    param_set: &FfParamSet,
    mol_specific_params: &HashMap<String, ForceFieldParams>,
    cfg: &MdConfig,
    static_peptide: bool,
    peptide_only_near_lig: bool,
    pep_atom_set: &mut HashSet<(usize, usize)>,
) -> Result<MdState, ParamError> {
    println!("Setting up dynamics...");

    let mut mols = Vec::new();

    for mol in ligs {
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
    for mol in lipids {
        if !mol.common.selected_for_md {
            continue;
        }
        let atoms_gen: Vec<_> = mol.common.atoms.iter().map(|a| a.to_generic()).collect();
        let bonds_gen: Vec<_> = mol.common.bonds.iter().map(|b| b.to_generic()).collect();

        mols.push(MolDynamics {
            ff_mol_type: FfMolType::Lipid,
            atoms: atoms_gen,
            atom_posits: Some(mol.common.atom_posits.clone()),
            bonds: bonds_gen,
            adjacency_list: Some(mol.common.adjacency_list.clone()),
            static_: false,
            bonded_only: false,
            mol_specific_params: None,
        })
    }

    if let Some(p) = peptide {
        // We assume hetero atoms are ligands, water etc, and are not part of the protein.
        let atoms = filter_peptide_atoms(pep_atom_set, p, ligs, peptide_only_near_lig);
        println!("Peptide atom count: {}", atoms.len());

        let bonds = create_bonds(&atoms);

        println!("STATIC: {}", static_peptide);
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
    let md_state = MdState::new(dev, cfg, &mols, param_set)?;
    println!("Done.");

    Ok(md_state)
}

pub fn run_dynamics(md_state: &mut MdState, dev: &ComputationDevice, dt: f32, n_steps: usize) {
    let start = Instant::now();

    let i_20_pc = n_steps / 5;
    let mut disp_count = 0;
    for i in 0..n_steps {
        if i.is_multiple_of(i_20_pc) {
            println!("{}% Complete", disp_count * 20);
            disp_count += 1;
        }

        md_state.step(dev, dt);
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
}

/// We filter peptide hetero atoms out of the MD workflow. Adjust snapshot indices and atom positions so they
/// are properly synchronized. This also handles the case of resassigning due to peptide atoms near the ligand.
pub fn reassign_snapshot_indices(
    pep: &MoleculePeptide,
    ligs: &[&mut MoleculeSmall],
    lipids: &[&mut MoleculeLipid],
    snapshots: &mut [Snapshot],
    included_set: &HashSet<(usize, usize)>
) {
    println!("Re-assigning snapshot indices to match atoms excluded for MD...");

    let n_included = included_set.len();

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
    for snap in snapshots {
        // Iterator over the peptide positions that actually participated in MD
        let mut pept_md_posits = snap.atom_posits[pep_start_i..pep_start_i + n_included]
            .iter()
            .cloned();

        let mut pept_md_vels = snap.atom_velocities[pep_start_i..pep_start_i + n_included]
            .iter()
            .cloned();

        let mut new_posits = Vec::with_capacity(pep_start_i + pep.common.atoms.len());
        let mut new_vels = Vec::with_capacity(pep_start_i + pep.common.atoms.len());

        // Keep ligand portion unchanged
        new_posits.extend_from_slice(&snap.atom_posits[..pep_start_i]);
        new_vels.extend_from_slice(&snap.atom_velocities[..pep_start_i]);

        // Reinsert peptide atoms in their original order
        for (i, atom) in pep.common.atoms.iter().enumerate() {
            let is_included = included_set.contains(&(0, i));

            if is_included {
                new_posits.push(
                    pept_md_posits
                        .next()
                        .expect("Ran out of peptide MD positions"),
                );
                new_vels.push(
                    pept_md_vels
                        .next()
                        .expect("Ran out of peptide MD velocities"),
                );
            } else {
                // Non-MD atom: use its original static position
                new_posits.push(atom.posit.into());
                new_vels.push(lin_alg::f32::Vec3::new_zero());
            }
        }

        // Replace the snapshot's positions with the reindexed set
        snap.atom_posits = new_posits;
        snap.atom_velocities = new_vels;
    }
    println!("Done.");
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
