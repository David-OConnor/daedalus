//! An interface to MD code. Wraps the `dynamics` crate.

use std::time::Instant;

use bio_files::{ResidueEnd, ResidueType};
use dynamics::{
    AtomDynamics, ComputationDevice, FfParamSet, ForceFieldParamsIndexed, HydrogenMdType, MdMode,
    MdState, ParamError, ProtFFTypeChargeMap, SnapshotDynamics,
};
use lin_alg::f64::Vec3;
use na_seq::{AminoAcid, AminoAcidGeneral, AminoAcidProtenationVariant, AtomTypeInRes};

use crate::{
    docking_v2::ConformationType,
    mol_lig::MoleculeSmall,
    molecule::{Atom, Bond, MoleculePeptide, Residue, build_adjacency_list},
};

// Å. Static atoms must be at least this close to a dynamic atom at the start of MD to count.
// Set this wide to take into account motion.
const STATIC_ATOM_DIST_THRESH: f64 = 8.; // todo: Increase (?) A/R.

/// Perform MD on the peptide (protein) only. Can be very computationally intensive due to the large
/// number of atoms.
pub fn build_dynamics_peptide(
    dev: &ComputationDevice,
    mol: &mut MoleculePeptide,
    ff_params: &FfParamSet,
    temp_target: f64,
    pressure_target: f64,
    n_steps: u32,
    dt: f64,
) -> Result<MdState, ParamError> {
    println!("Building peptide dynamics...");

    let posits: Vec<_> = mol.common.atoms.iter().map(|a| a.posit).collect();

    let mut md_state = new_md_peptide(
        &mol.common.atoms,
        &posits,
        &mol.common.bonds,
        ff_params,
        temp_target,
        pressure_target,
    )?;

    let start = Instant::now();

    for _ in 0..n_steps {
        md_state.step(dev, dt)
    }

    let elapsed = start.elapsed();
    println!("MD complete in {:.2} s", elapsed.as_secs());

    change_snapshot_peptide(mol, &md_state.atoms, &md_state.snapshots[0]);

    Ok(md_state)
}

/// Perform MD on the ligand, with nearby protein (receptor) atoms, from the docking setup as static
/// non-bonded contributors. (Vdw and coulomb)
pub fn build_dynamics_docking(
    dev: &ComputationDevice,
    // lig: &mut MoleculeSmall,
    // This approach avoids a dbl-borrow
    ligs: &mut [MoleculeSmall],
    lig_i: usize,
    mol: &MoleculePeptide,
    // setup: &DockingSetup,
    ff_params: &FfParamSet,
    temp_target: f64,
    pressure_target: f64,
    n_steps: u32,
    dt: f64,
) -> Result<MdState, ParamError> {
    println!("Building docking dyanmics...");

    let lig = &mut ligs[lig_i];

    if let Some(data) = &mut lig.lig_data {
        data.pose.conformation_type = ConformationType::AbsolutePosits;
    }

    let mut md_state = new_md_docking(
        &lig.common.atoms,
        &lig.common.atom_posits,
        &lig.common.adjacency_list,
        &lig.common.bonds,
        &mol.common.atoms,
        ff_params,
        temp_target,
        pressure_target,
        &lig.common.ident,
    )?;

    let start = Instant::now();

    for _ in 0..n_steps {
        md_state.step(dev, dt);
    }

    let elapsed = start.elapsed();
    println!("MD complete in {:.2} s", elapsed.as_secs());

    for (i, atom) in md_state.atoms.iter().enumerate() {
        lig.common.atom_posits[i] = atom.posit;
    }
    // change_snapshot_docking(lig, &md_state.snapshots[0], &mut None);
    change_snapshot_docking(lig, &md_state.snapshots[0]);

    Ok(md_state)
}

/// Set ligand atom positions to that of a snapshot. We assume a rigid receptor.
/// Body masses are separate from the snapshot, since it's invariant.
pub fn change_snapshot_docking(
    lig: &mut MoleculeSmall,
    snapshot: &SnapshotDynamics,
    // energy_disp: &mut Option<BindingEnergy>,
) {
    let Some(data) = &mut lig.lig_data else {
        return;
    };

    data.pose.conformation_type = ConformationType::AbsolutePosits;
    lig.common.atom_posits = snapshot.atom_posits.iter().map(|p| (*p).into()).collect();
    // *energy_disp = snapshot.energy.clone();
}

pub fn change_snapshot_peptide(
    mol: &mut MoleculePeptide,
    atoms_dy: &[AtomDynamics],
    snapshot: &SnapshotDynamics,
) {
    let mut posits = Vec::with_capacity(mol.common.atoms.len());

    // todo: This is slow. Use a predefined mapping; much faster.
    // If the atom's SN is present in the snap, use it; otherwise, use the original posit (e.g. hetero)
    for atom in &mol.common.atoms {
        let mut found = false;
        for (i_dy, atom_dy) in atoms_dy.iter().enumerate() {
            if atom_dy.serial_number == atom.serial_number {
                posits.push(snapshot.atom_posits[i_dy]);
                found = true;
                break;
            }
        }
        if !found {
            posits.push(atom.posit); // Fallback to the orig.
        }
    }

    mol.common.atom_posits = posits;
}

/// For a dynamic ligand, and static (set of a) peptide.
pub fn new_md_docking(
    atoms: &[Atom],
    atom_posits: &[Vec3],
    adjacency_list: &[Vec<usize>],
    bonds: &[Bond],
    // This is the whole set; not just nearby. E.g. all protein atoms.
    atoms_static_all: &[Atom],
    ff_params: &FfParamSet,
    temp_target: f64,
    pressure_target: f64, // Bar
    lig_ident: &str,
    // todo: Temperature/thermostat.
) -> Result<MdState, ParamError> {
    let mut hydrogen_md_type = HydrogenMdType::Fixed(Vec::new());

    let Some(ff_params_lig_keyed) = &ff_params.lig_general else {
        return Err(ParamError::new("MD failure: Missing lig general params"));
    };
    let Some(ff_params_prot_keyed) = &ff_params.prot_general else {
        return Err(ParamError::new(
            "MD failure: Missing prot params general params",
        ));
    };

    let mut atoms_static_near = Vec::new();
    for atom_st in atoms_static_all {
        if atom_st.hetero {
            continue;
        }
        let mut closest_dist = 99999.;

        for i in 0..atoms.len() {
            let dist = (atom_posits[i] - atom_st.posit).magnitude();
            if dist < closest_dist {
                closest_dist = dist;
            }
        }

        if closest_dist < STATIC_ATOM_DIST_THRESH {
            atoms_static_near.push(atom_st.clone());
        }
    }

    // Assign FF type and charge to protein atoms; FF type must be assigned prior to initializing `ForceFieldParamsIndexed`.
    // (Ligand atoms will already have FF type assigned).

    let ff_params_keyed_lig_specific = match ff_params.lig_specific.get(lig_ident) {
        Some(l) => l,
        None => {
            return Err(ParamError::new(&format!(
                "Missing lig-specific (FRCMOD) parameters for {lig_ident}"
            )));
        }
    };

    // Convert FF params from keyed to index-based.
    println!("\nBuilding FF params indexed ligand for docking...");
    let atoms_gen: Vec<_> = atoms.iter().map(|a| a.to_generic()).collect();
    let bonds_gen: Vec<_> = bonds.iter().map(|b| b.to_generic()).collect();

    let ff_params_non_static = ForceFieldParamsIndexed::new(
        ff_params_lig_keyed,
        Some(ff_params_keyed_lig_specific),
        &atoms_gen,
        &bonds_gen,
        adjacency_list,
        &mut hydrogen_md_type,
    )?;

    // This assumes nonbonded interactions only with external atoms; this is fine for
    // rigid protein models, and is how this is currently structured.
    let bonds_static = Vec::new();
    let adj_list_static = Vec::new();

    let atoms_static_near_gen: Vec<_> = atoms_static_near.iter().map(|a| a.to_generic()).collect();

    println!("\nBuilding FF params indexed static for docking...");
    let ff_params_static = ForceFieldParamsIndexed::new(
        ff_params_prot_keyed,
        None,
        &atoms_static_near_gen,
        &bonds_static,
        &adj_list_static,
        &mut hydrogen_md_type,
    )?;

    // We are using this approach instead of `.into`, so we can use the atom_posits from
    // the positioned ligand. (its atom coords are relative; we need absolute)
    let mut atoms_dy = Vec::with_capacity(atoms.len());
    for (i, atom) in atoms.iter().enumerate() {
        atoms_dy.push(AtomDynamics::new(
            &atom.to_generic(),
            atom_posits,
            &ff_params_non_static,
            i,
        )?);
    }

    let mut atoms_dy_static = Vec::with_capacity(atoms_static_near.len());
    let atom_posits_static: Vec<_> = atoms_static_near.iter().map(|a| a.posit).collect();

    for (i, atom) in atoms_static_near.iter().enumerate() {
        atoms_dy_static.push(AtomDynamics::new(
            &atom.to_generic(),
            &atom_posits_static,
            &ff_params_static,
            i,
        )?);
    }

    Ok(MdState::new(
        MdMode::Docking,
        atoms_dy,
        atoms_dy_static,
        ff_params_non_static,
        temp_target,
        pressure_target,
        hydrogen_md_type,
        adjacency_list.to_vec(),
    ))
}

/// For a dynamic peptide, and no ligand. There is no need to filter by hetero only
/// atoms upstream.
pub fn new_md_peptide(
    atoms: &[Atom],
    atom_posits: &[Vec3],
    bonds: &[Bond],
    ff_params: &FfParamSet,
    temp_target: f64,
    pressure_target: f64,
    // todo: Thermostat.
) -> Result<MdState, ParamError> {
    let mut hydrogen_md_type = HydrogenMdType::Fixed(Vec::new());

    let Some(ff_params_prot_keyed) = &ff_params.prot_general else {
        return Err(ParamError::new(
            "MD failure: Missing prot params general params",
        ));
    };

    // Assign FF type and charge to protein atoms; FF type must be assigned prior to initializing `ForceFieldParamsIndexed`.
    // (Ligand atoms will already have FF type assigned).

    let atoms: Vec<_> = atoms.iter().filter(|a| !a.hetero).cloned().collect();

    // Re-assign bond indices. The original indices no longer work due to the filter above, but we
    // can still use serial numbers to reassign.
    let mut bonds_filtered = Vec::new();
    for bond in bonds {
        let mut atom_0 = None;
        let mut atom_1 = None;
        for (i, atom) in atoms.iter().enumerate() {
            if bond.atom_0_sn == atom.serial_number {
                atom_0 = Some(i);
            } else if bond.atom_1_sn == atom.serial_number {
                atom_1 = Some(i);
            }
        }

        if atom_0.is_some() && atom_1.is_some() {
            bonds_filtered.push(Bond {
                atom_0: atom_0.unwrap(),
                atom_1: atom_1.unwrap(),
                ..bond.clone()
            })
        } else {
            return Err(ParamError::new(
                "Problem remapping bonds to filtered atoms.",
            ));
        }
    }

    let adjacency_list = build_adjacency_list(&bonds_filtered, atoms.len());

    // Convert FF params from keyed to index-based.
    println!("\nBuilding FF params indexed for peptide...");
    let atoms_gen: Vec<_> = atoms.iter().map(|a| a.to_generic()).collect();
    let bonds_gen: Vec<_> = bonds_filtered.iter().map(|b| b.to_generic()).collect();

    let ff_params_non_static = ForceFieldParamsIndexed::new(
        ff_params_prot_keyed,
        None,
        &atoms_gen,
        &bonds_gen,
        &adjacency_list,
        &mut hydrogen_md_type,
    )?;

    let mut atoms_dy = Vec::with_capacity(atoms.len());
    for (i, atom) in atoms.iter().enumerate() {
        atoms_dy.push(AtomDynamics::new(
            &atom.to_generic(),
            atom_posits,
            &ff_params_non_static,
            i,
        )?);
    }

    Ok(MdState::new(
        MdMode::Peptide,
        atoms_dy,
        Vec::new(),
        ff_params_non_static,
        temp_target,
        pressure_target,
        hydrogen_md_type,
        adjacency_list.to_vec(),
    ))
}

/// Populate forcefield type, and partial charge.
/// `residues` must be the full set; this is relevant to how we index it.
pub fn populate_ff_and_q(
    atoms: &mut [Atom],
    residues: &[Residue],
    ff_type_charge: &ProtFFTypeChargeMap,
) -> Result<(), ParamError> {
    for atom in atoms {
        if atom.hetero {
            continue;
        }

        let Some(res_i) = atom.residue else {
            return Err(ParamError::new(&format!(
                "MD failure: Missing residue when populating ff name and q: {atom}"
            )));
        };

        let Some(type_in_res) = &atom.type_in_res else {
            return Err(ParamError::new(&format!(
                "MD failure: Missing type in residue for atom: {atom}"
            )));
        };

        let atom_res_type = &residues[res_i].res_type;

        let ResidueType::AminoAcid(aa) = atom_res_type else {
            // e.g. water or other hetero atoms; skip.
            continue;
        };

        // todo: Eventually, determine how to load non-standard AA variants from files; set up your
        // todo state to use those labels. They are available in the params.
        let aa_gen = AminoAcidGeneral::Standard(*aa);

        let charge_map = match residues[res_i].end {
            ResidueEnd::Internal => &ff_type_charge.internal,
            ResidueEnd::NTerminus => &ff_type_charge.n_terminus,
            ResidueEnd::CTerminus => &ff_type_charge.c_terminus,
            ResidueEnd::Hetero => {
                return Err(ParamError::new(&format!(
                    "Error: Encountered hetero atom when parsing amino acid FF types: {atom}"
                )));
            }
        };

        let charges = match charge_map.get(&aa_gen) {
            Some(c) => c,
            // A specific workaround to plain "HIS" being absent from amino19.lib (2025.
            // Choose one of "HID", "HIE", "HIP arbitrarily.
            // todo: Re-evaluate this, e.g. which one of the three to load.
            None if aa_gen == AminoAcidGeneral::Standard(AminoAcid::His) => charge_map
                .get(&AminoAcidGeneral::Variant(AminoAcidProtenationVariant::Hid))
                .ok_or_else(|| ParamError::new("Unable to find AA mapping"))?,
            None => return Err(ParamError::new("Unable to find AA mapping")),
        };

        let mut found = false;

        for charge in charges {
            // todo: Note that we have multiple branches in some case, due to Amber names like
            // todo: "HYP" for variants on AAs for different protenation states. Handle this.
            if &charge.type_in_res == type_in_res {
                atom.force_field_type = Some(charge.ff_type.clone());
                atom.partial_charge = Some(charge.charge);

                found = true;
                break;
            }
        }

        // Code below is mainly for the case of missing data; otherwise, the logic for this operation
        // is complete.

        if !found {
            match type_in_res {
                // todo: This is a workaround for having trouble with H types. LIkely
                // todo when we create them. For now, this meets the intent.
                AtomTypeInRes::H(_) => {
                    // Note: We've witnessed this due to errors in the mmCIF file, e.g. on ASP #88 on 9GLS.
                    eprintln!(
                        "Error assigning FF type and q based on atom type in res: Failed to match H type. #{}, {type_in_res}, {aa_gen:?}. \
                         Falling back to a generic H",
                        &residues[res_i].serial_number
                    );

                    for charge in charges {
                        if &charge.type_in_res == &AtomTypeInRes::H("H".to_string())
                            || &charge.type_in_res == &AtomTypeInRes::H("HA".to_string())
                        {
                            atom.force_field_type = Some("HB2".to_string());
                            atom.partial_charge = Some(charge.charge);

                            found = true;
                            break;
                        }
                    }
                }
                // // This is an N-terminal oxygen of a C-terminal carboxyl group.
                // // todo: You should parse `aminoct12.lib`, and `aminont12.lib`, then delete this.
                // AtomTypeInRes::OXT => {
                //     match atom_res_type {
                //         // todo: QC that it's the N-terminal Met too, or return an error.
                //         ResidueType::AminoAcid(AminoAcid::Met) => {
                //             atom.force_field_type = Some("O2".to_owned());
                //             // Fm amino12ct.lib
                //             atom.partial_charge = Some(-0.804100);
                //             found = true;
                //         }
                //         _ => return Err(ParamError::new("Error populating FF type: OXT atom-in-res type,\
                //         not at the C terminal")),
                //     }
                // }
                _ => (),
            }

            // i.e. if still not found after our specific workarounds above.
            if !found {
                return Err(ParamError::new(&format!(
                    "Error assigning FF type and q based on atom type in res: {atom}",
                )));
            }
        }
    }

    Ok(())
}
