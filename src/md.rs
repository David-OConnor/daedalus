//! An interface to dynamics library.

use std::{
    collections::{HashMap, HashSet},
    time::Instant,
};

use bio_files::{AtomGeneric, create_bonds, md_params::ForceFieldParams};
#[cfg(feature = "cuda")]
use cudarc::driver::HostSlice;
use dynamics::{
    ComputationDevice, FfMolType, HydrogenConstraint, MdConfig, MdOverrides, MdState, MolDynamics,
    ParamError, params::FfParamSet, snapshot::Snapshot,
};
use graphics::{EngineUpdates, EntityUpdate, Scene};
use lin_alg::f64::Vec3;

use crate::ui::cam::move_cam_to_active_mol;
use crate::util::handle_err;
use crate::{
    MdStateLocal, State,
    drawing::{draw_peptide, draw_water},
    lipid::MoleculeLipid,
    mol_lig::MoleculeSmall,
    molecule::{MolType, MoleculeCommon, MoleculePeptide},
    nucleic_acid::MoleculeNucleicAcid,
    util::handle_success,
};

// Å. Static atoms must be at least this close to a dynamic atom at the start of MD to be counted.
// Set this wide to take into account motion.
pub const STATIC_ATOM_DIST_THRESH: f64 = 14.;

// Run this many MD steps per frame. This is used to balance MD time with the rest of the application.
// A higher value will reduce the overall MD computation time,
// but make the UI laggier. If this is >= the number of steps, the whole MD operation will block
// the rest of the program. For initial test, a value above ~10 doesn't seem to
// noticeably increase total computation time. e.g. the frame time is small compared to this many
// MD steps for a small molecule + water sim.
const MD_STEPS_PER_APPLICATION_FRAME: usize = 10;

pub fn post_run_cleanup(state: &mut State, scene: &mut Scene, engine_updates: &mut EngineUpdates) {
    if state.mol_dynamics.is_none() {
        eprintln!("Can't run MD cleanup; MD state is None");
        return;
    }

    state.volatile.md_local.running = false;
    state.volatile.md_local.start = None;

    if let Some(p) = &state.peptide {
        let ligs: Vec<_> = state
            .ligands
            .iter_mut()
            .filter(|l| l.common.selected_for_md)
            .collect();

        let lipids: Vec<_> = state
            .lipids
            .iter_mut()
            .filter(|l| l.common.selected_for_md)
            .collect();

        let nucleic_acids: Vec<_> = state
            .nucleic_acids
            .iter_mut()
            .filter(|l| l.common.selected_for_md)
            .collect();

        let md = state.mol_dynamics.as_mut().unwrap();
        reassign_snapshot_indices(
            p,
            &ligs,
            &lipids,
            &nucleic_acids,
            &mut md.snapshots,
            &state.volatile.md_peptide_selected,
        );
    }

    handle_success(&mut state.ui, "MD complete".to_string());

    // Tricky behavior here to prevent a dbl-borrow.
    {
        let md = state.mol_dynamics.as_ref().unwrap();
        let snap = &md.snapshots[0];

        draw_water(
            scene,
            &snap.water_o_posits,
            &snap.water_h0_posits,
            &snap.water_h1_posits,
            state.ui.visibility.hide_water,
        );
    }
    draw_peptide(state, scene);

    state.ui.current_snapshot = 0;

    engine_updates.entities = EntityUpdate::All;
}
pub fn build_and_run_dynamics(
    dev: &ComputationDevice,
    mols: &[(FfMolType, &MoleculeCommon)],
    peptide: Option<&MoleculePeptide>,
    param_set: &FfParamSet,
    mol_specific_params: &HashMap<String, ForceFieldParams>,
    cfg: &MdConfig,
    static_peptide: bool,
    peptide_only_near_lig: Option<f64>,
    pep_atom_set: &mut HashSet<(usize, usize)>,
    md_local: &mut MdStateLocal,
) -> Result<MdState, ParamError> {
    let md_state = build_dynamics(
        dev,
        mols,
        peptide,
        param_set,
        mol_specific_params,
        cfg,
        static_peptide,
        peptide_only_near_lig,
        pep_atom_set,
    )?;

    md_local.start = Some(Instant::now());
    md_local.running = true;

    Ok(md_state)
}

/// Filter out hetero atoms, and if necessary, atoms not close to a ligand.
pub fn filter_peptide_atoms(
    set: &mut HashSet<(usize, usize)>,
    pep: &MoleculePeptide,
    mols_non_pep: &[(FfMolType, &MoleculeCommon)],
    only_near_lig: Option<f64>,
) -> Vec<AtomGeneric> {
    *set = HashSet::new();

    pep.common
        .atoms
        .iter()
        .enumerate()
        .filter_map(|(i, a)| {
            let pass = if let Some(thresh) = only_near_lig {
                let mut closest_dist = f64::MAX;
                for lig in mols_non_pep {
                    for p in &lig.1.atom_posits {
                        let dist = (*p - pep.common.atom_posits[i]).magnitude();
                        if dist < closest_dist {
                            closest_dist = dist;
                        }
                    }
                }
                !a.hetero && closest_dist < thresh
            } else {
                !a.hetero
            };

            if pass {
                // The initial 0 is for the peptide mol number; we may support multiple
                // peptides in the future.
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
    mols_in: &[(FfMolType, &MoleculeCommon)],
    peptide: Option<&MoleculePeptide>,
    param_set: &FfParamSet,
    mol_specific_params: &HashMap<String, ForceFieldParams>,
    cfg: &MdConfig,
    mut static_peptide: bool,
    mut peptide_only_near_lig: Option<f64>,
    pep_atom_set: &mut HashSet<(usize, usize)>,
) -> Result<MdState, ParamError> {
    println!("Setting up dynamics...");

    // if ligs.is_empty() && lipids.is_empty() {
    if mols_in.is_empty() {
        static_peptide = false;
        peptide_only_near_lig = None;
    }

    let mut mols = Vec::new();

    for (ff_mol_type, mol) in mols_in {
        if !mol.selected_for_md {
            continue;
        }
        let atoms_gen: Vec<_> = mol.atoms.iter().map(|a| a.to_generic()).collect();
        let bonds_gen: Vec<_> = mol.bonds.iter().map(|b| b.to_generic()).collect();

        let msp = match mol_specific_params.get(&mol.ident) {
            Some(v) => Some(v.clone()),
            None => {
                if *ff_mol_type == FfMolType::SmallOrganic {
                    return Err(ParamError::new(&format!(
                        "Missing molecule-specific parameters for  {}",
                        mol.ident
                    )));
                }
                None
            }
        };

        mols.push(MolDynamics {
            ff_mol_type: *ff_mol_type,
            atoms: atoms_gen,
            atom_posits: Some(mol.atom_posits.clone()),
            atom_init_velocities: None,
            bonds: bonds_gen,
            adjacency_list: Some(mol.adjacency_list.clone()),
            static_: false,
            bonded_only: false,
            mol_specific_params: msp,
        });
    }

    if let Some(p) = peptide {
        // We assume hetero atoms are ligands, water etc, and are not part of the protein.
        // let atoms = filter_peptide_atoms(pep_atom_set, p, ligs, peptide_only_near_lig);
        let atoms = filter_peptide_atoms(pep_atom_set, p, mols_in, peptide_only_near_lig);
        println!(
            "Peptide atom count: {}. Set count: {}",
            atoms.len(),
            pep_atom_set.len()
        );

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

    // Uncomment as required for validating individual processes.
    // let cfg = MdConfig {
    //     overrides: MdOverrides {
    //         allow_missing_dihedral_params: true,
    //         skip_water: true,
    //         bonded_disabled: true,
    //         coulomb_disabled: false,
    //         lj_disabled: true,
    //         long_range_recip_disabled: false,
    //         thermo_disabled: true,
    //         baro_disabled: true,
    //     },
    //     max_init_relaxation_iters: None,
    //     ..cfg.clone()
    // };

    // todo temp
    let cfg = MdConfig {
        overrides: MdOverrides {
            // coulomb_disabled: false,
            // long_range_recip_disabled: false,
            // lj_disabled: false,
            // skip_water: true,
            ..Default::default()
        },
        // max_init_relaxation_iters: None,
        ..cfg.clone()
    };

    println!("Initializing MD state...");
    let md_state = MdState::new(dev, &cfg, &mols, param_set)?;
    println!("MD init done.");

    Ok(md_state)
}

/// Run the dynamics in one go. Blocking.
pub fn run_dynamics(md_state: &mut MdState, dev: &ComputationDevice, dt: f32, n_steps: usize) {
    if n_steps == 0 {
        return;
    }

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
    println!(
        "\nMD computation time: {}",
        md_state.computation_time().unwrap()
    );

    let elapsed = start.elapsed();
    println!(
        "MD complete in {:.1} s",
        elapsed.as_millis() as f32 / 1_000.
    );
}

/// We filter peptide hetero atoms out of the MD workflow. Adjust snapshot indices and atom positions so they
/// are properly synchronized. This also handles the case of resassigning due to peptide atoms near the ligand.
pub fn reassign_snapshot_indices(
    pep: &MoleculePeptide,
    ligs: &[&mut MoleculeSmall],
    lipids: &[&mut MoleculeLipid],
    nucleic_acids: &[&mut MoleculeNucleicAcid],
    snapshots: &mut [Snapshot],
    pep_atom_set: &HashSet<(usize, usize)>,
) {
    println!("Re-assigning snapshot indices to match atoms excluded for MD...");

    let pep_count = pep_atom_set.len();

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

    let na_atom_count: usize = nucleic_acids
        .iter()
        .filter(|l| l.common.selected_for_md)
        .map(|l| l.common.atoms.len())
        .sum();

    let pep_start_i = lig_atom_count + lipid_atom_count + na_atom_count;

    // Rebuild each snapshot's atom_posits: [ligands as-is] + [full peptide with holes filled]
    for snap in snapshots {
        if pep_start_i + pep_count > snap.atom_posits.len() {
            eprintln!(
                "Error: Invalid index when reassigning snapshot posits. \
            Snap atom count: {}, lig count {lig_atom_count} Pep start: {pep_start_i}, Pep count: {pep_count}",
                snap.atom_posits.len()
            );
            continue;
        }

        // Iterator over the peptide positions that actually participated in MD
        let mut pept_md_posits = snap.atom_posits[pep_start_i..pep_start_i + pep_count]
            .iter()
            .cloned();

        let mut pept_md_vels = snap.atom_velocities[pep_start_i..pep_start_i + pep_count]
            .iter()
            .cloned();

        let mut new_posits = Vec::with_capacity(pep_start_i + pep.common.atoms.len());
        let mut new_vels = Vec::with_capacity(pep_start_i + pep.common.atoms.len());

        // Keep ligand portion unchanged
        new_posits.extend_from_slice(&snap.atom_posits[..pep_start_i]);
        new_vels.extend_from_slice(&snap.atom_velocities[..pep_start_i]);

        // Reinsert peptide atoms in their original order
        for (i, atom) in pep.common.atoms.iter().enumerate() {
            let is_included = pep_atom_set.contains(&(0, i));

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

pub fn change_snapshot_helper(
    posits: &mut [Vec3],
    start_i_this_mol: &mut usize,
    snapshot: &Snapshot,
) {
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
    nucleic_acids: Vec<&mut MoleculeNucleicAcid>,
    snapshot: &Snapshot,
) {
    let mut start_i_this_mol = 0;

    for mol in ligs {
        change_snapshot_helper(&mut mol.common.atom_posits, &mut start_i_this_mol, snapshot);
    }

    for mol in lipids {
        change_snapshot_helper(&mut mol.common.atom_posits, &mut start_i_this_mol, snapshot);
    }

    for mol in nucleic_acids {
        change_snapshot_helper(&mut mol.common.atom_posits, &mut start_i_this_mol, snapshot);
    }

    if let Some(mol) = peptide {
        change_snapshot_helper(&mut mol.common.atom_posits, &mut start_i_this_mol, snapshot);
    }
}

impl State {
    /// Run MD for a single step if ready, and update atom positions immediately after. Blocks for
    /// a fixed number of steps only; intended to be run each frame until complete.
    pub fn md_step(&mut self, scene: &mut Scene, engine_updates: &mut EngineUpdates) {
        if !self.volatile.md_local.running {
            return;
        }
        let Some(md) = &mut self.mol_dynamics else {
            return;
        };

        for _ in 0..MD_STEPS_PER_APPLICATION_FRAME {
            if md.step_count >= self.to_save.num_md_steps as usize {
                println!(
                    "\nMD computation time: {} \n Total run time: {} ms",
                    md.computation_time().unwrap(),
                    self.volatile.md_local.start.unwrap().elapsed().as_millis()
                );

                post_run_cleanup(self, scene, engine_updates);
                break;
            }
            md.step(&self.dev, self.to_save.md_dt);
        }
    }
}

// Callced directly from the UI;

pub fn launch_md(state: &mut State) {
    // Filter molecules for docking by if they're selected.
    // mut so we can move their posits in the initial snapshot change.
    let ligs: Vec<_> = state
        .ligands
        .iter()
        .filter(|l| l.common.selected_for_md)
        .collect();
    let lipids: Vec<_> = state
        .lipids
        .iter()
        .filter(|l| l.common.selected_for_md)
        .collect();
    let nucleic_acids: Vec<_> = state
        .nucleic_acids
        .iter()
        .filter(|l| l.common.selected_for_md)
        .collect();

    let mut mols = Vec::new();
    for m in &ligs {
        mols.push((FfMolType::SmallOrganic, &m.common));
    }
    for m in &lipids {
        mols.push((FfMolType::Lipid, &m.common));
    }
    // todo: You must specify DNA or RNA here!
    for m in &nucleic_acids {
        mols.push((FfMolType::Dna, &m.common));
    }

    let mol = match &state.peptide {
        Some(m) => {
            if m.common.selected_for_md {
                Some(m)
            } else {
                None
            }
        }
        None => None,
    };

    let near_lig_thresh = if state.ui.md.peptide_only_near_ligs {
        Some(STATIC_ATOM_DIST_THRESH)
    } else {
        None
    };

    match build_and_run_dynamics(
        &state.dev,
        &mols,
        mol,
        &state.ff_param_set,
        &state.lig_specific_params,
        &state.to_save.md_config,
        state.ui.md.peptide_static,
        near_lig_thresh,
        &mut state.volatile.md_peptide_selected,
        &mut state.volatile.md_local,
    ) {
        Ok(md) => {
            state.mol_dynamics = Some(md);
        }
        Err(e) => handle_err(&mut state.ui, e.descrip),
    }
}
