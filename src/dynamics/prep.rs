//! Contains setup code, including applying forcefield data to our specific
//! atoms.

// Notes to square away the 3 "atom name" / "Amber atom type" / "force field type" keys.
// This guide shows Type 1. https://emleddin.github.io/comp-chem-website/AMBERguide-AMBER-atom-types.html,
//
// Update: "Type 1" = "type_in_res" in our code now. "Type 2" = "ff_type" for AAs, and "Type 3" = "ff_type" for small mols.
//
// Type 1 Examples: "CA", "HA", "CZ", "HB3", "HH22", HZ2", "N", "H", "HG3", "O", "CD", "C", "HG23", "CG", "CB", "CG1", "HE2", "HB3",
// Type 1 Sources: `amino19.lib`, col 0. mmCIF atom coordinate files.
//
// Type 2 Examples:  "HC", "C8", "HC", "H"(both), "XC", "N"(both), "H"(both), "H1", "CT", "OH", "HO", "2C",
// Type 2 Sources: `amino19.lib` (AA/protein partial charges), col 1. `frcmod.ff19SB`. (AA/protein params)
//
// Small Mol/lig:
// Type 3 Examples: "oh", "h1", "ca", "o", "os", "c6", "n3", "c3"
// Type 3 Sources: `.mol2` generated by Amber. (Small mol coordinates and partial charges) `gaff2.dat` (Small molg params)
//
// MOl2 for ligands also have "C10", "O7" etc, which is ambiguous here, and not required, as their params
// use Type 3, which is present. It's clear what to do for ligand
//
// Best guess: Type 1 identifies labels within the residue only. Type 2 (AA) and Type 3 (small mol) are the FF types.

use std::{collections::HashSet, time::Instant};

use bio_files::{
    ResidueType,
    amber_params::{
        AngleBendingParams, BondStretchingParams, ForceFieldParamsKeyed, MassParams, VdwParams,
    },
};
use cudarc::driver::HostSlice;
use itertools::Itertools;
use lin_alg::f64::Vec3;
use na_seq::{AminoAcid, AminoAcidGeneral, AminoAcidProtenationVariant, AtomTypeInRes, Element};

use crate::{
    ComputationDevice, FfParamSet, ProtFFTypeChargeMap,
    docking::{BindingEnergy, ConformationType, prep::DockingSetup},
    dynamics::{
        AtomDynamics, ForceFieldParamsIndexed, MdMode, MdState, ParamError, SKIN, SnapshotDynamics,
        ambient::SimBox, non_bonded::CUTOFF_VDW, water_opc::make_water_mols,
    },
    molecule::{Atom, Bond, Ligand, Molecule, Residue, ResidueEnd, build_adjacency_list},
};

// Todo: QC this.
const TEMP_TGT_DEFAULT: f64 = 310.; // Kelvin.

/// Build a single lookup table in which ligand-specific parameters
/// (when given) replace or add to the generic ones.
pub fn merge_params(
    generic: &ForceFieldParamsKeyed,
    lig_specific: Option<&ForceFieldParamsKeyed>,
) -> ForceFieldParamsKeyed {
    // Start with a deep copy of the generic parameters.
    let mut merged = generic.clone();

    if let Some(lig) = lig_specific {
        merged.mass.extend(lig.mass.clone());
        // merged.partial_charges.extend(lig.partial_charges.clone());
        merged.van_der_waals.extend(lig.van_der_waals.clone());

        merged.bond.extend(lig.bond.clone());
        merged.angle.extend(lig.angle.clone());
        merged.dihedral.extend(lig.dihedral.clone());
        merged
            .dihedral_improper
            .extend(lig.dihedral_improper.clone());
    }

    merged
}

/// Helper that reduces repetition. Used for populating all bonded parameters by index.
fn ff_type_from_idx<'a>(
    atoms: &'a [Atom],
    idx: usize,
    descriptor: &str,
) -> Result<&'a String, ParamError> {
    let atom = &atoms[idx];

    atom.force_field_type.as_ref().ok_or_else(|| {
        ParamError::new(&format!(
            "MD failure: Atom missing FF type on {descriptor}: {atom}"
        ))
    })
}

/// Associate loaded Force field data (e.g. from Amber) into the atom indices used in a specific
/// dynamics sim. This handles combining general and molecule-specific parameter sets, and converting
/// between atom name, and the specific indices of the atoms we're using.
///
/// This code is straightforward if params are available; much of the logic here is related to handling
/// missing parameters.
impl ForceFieldParamsIndexed {
    pub fn new(
        params_general: &ForceFieldParamsKeyed,
        params_specific: Option<&ForceFieldParamsKeyed>,
        atoms: &[Atom],
        bonds: &[Bond],
        adjacency_list: &[Vec<usize>],
    ) -> Result<Self, ParamError> {
        let mut result = Self::default();

        // Combine the two force field sets. When a value is present in both, refer the lig-specific
        // one.
        let params = merge_params(params_general, params_specific);

        for (i, atom) in atoms.iter().enumerate() {
            let ff_type = match &atom.force_field_type {
                Some(ff_t) => ff_t,
                None => {
                    eprintln!("Atom missing FF type: {atom}");
                    match atom.element {
                        Element::Carbon => {
                            eprintln!(
                                "Indexing: Atom missing FF type: {atom}; Falling back to generic C"
                            );
                            "C"
                        }
                        Element::Nitrogen => {
                            eprintln!(
                                "Indexing: Atom missing FF type: {atom}; Falling back to generic N"
                            );
                            "N"
                        }
                        Element::Oxygen => {
                            eprintln!(
                                "Indexing: Atom missing FF type: {atom}; Falling back to generic O"
                            );
                            "O"
                        }
                        Element::Hydrogen => {
                            eprintln!(
                                "Indexing: Atom missing FF type: {atom}; Falling back to generic H"
                            );
                            "H"
                        }
                        _ => {
                            return Err(ParamError::new(&format!(
                                "MD failure: Atom missing FF type: {atom}"
                            )));
                        }
                    }
                }
            };

            // Mass
            if let Some(mass) = params.mass.get(ff_type) {
                result.mass.insert(i, mass.clone());
            } else {
                if ff_type.starts_with("C") {
                    match params.mass.get("C") {
                        Some(m) => {
                            result.mass.insert(i, m.clone());
                            println!("Using C fallback mass for {ff_type}");
                        }
                        None => {
                            return Err(ParamError::new(&format!(
                                "MD failure: Missing mass params for {ff_type}"
                            )));
                        }
                    }
                } else if ff_type.starts_with("N") {
                    println!("TS. {atom}");
                    match params.mass.get("N") {
                        Some(m) => {
                            result.mass.insert(i, m.clone());
                            println!("Using N fallback mass for {ff_type}");
                        }
                        None => {
                            return Err(ParamError::new(&format!(
                                "MD failure: Missing mass params for {ff_type}"
                            )));
                        }
                    }
                } else if ff_type.starts_with("O") {
                    match params.mass.get("O") {
                        Some(m) => {
                            result.mass.insert(i, m.clone());
                            println!("Using O fallback mass for {ff_type}");
                        }
                        None => {
                            return Err(ParamError::new(&format!(
                                "MD failure: Missing mass params for {ff_type}"
                            )));
                        }
                    }
                } else {
                    result.mass.insert(
                        i,
                        MassParams {
                            atom_type: "".to_string(),
                            mass: atom.element.atomic_weight(),
                            comment: None,
                        },
                    );

                    println!("Missing mass params on {atom}; using element default.");

                    // return Err(ParamError::new(&format!(
                    //     "MD failure: Missing mass params for {ff_type}"
                    // )));
                }
            }

            // Lennard-Jones / van der Waals
            if let Some(vdw) = params.van_der_waals.get(ff_type) {
                result.van_der_waals.insert(i, vdw.clone());
                // If the key is missing for the given FF type in our loaded data, check for certain
                // special cases.
            } else {
                // ChatGpt seems to think this is the move. I only asked it about "2C", and it inferred
                // I should also map 3C and C8 to this, which is a good sign. Note that the mass values
                // for all 4 of these are present in frcmod.ff19sb.
                if ff_type == "2C" || ff_type == "3C" || ff_type == "C8" {
                    result
                        .van_der_waals
                        .insert(i, params.van_der_waals.get("CT").unwrap().clone());
                } else if ff_type == "CO" {
                    result
                        .van_der_waals
                        .insert(i, params.van_der_waals.get("C").unwrap().clone());
                } else if ff_type == "OXT" {
                    result
                        .van_der_waals
                        .insert(i, params.van_der_waals.get("O2").unwrap().clone());
                } else if ff_type.starts_with("N") {
                    result
                        .van_der_waals
                        .insert(i, params.van_der_waals.get("N").unwrap().clone());
                    println!("Using N fallback VdW for {atom}");
                } else if ff_type.starts_with("O") {
                    result
                        .van_der_waals
                        .insert(i, params.van_der_waals.get("O").unwrap().clone());
                    println!("Using O fallback VdW for {atom}");
                } else {
                    println!("Missing Vdw params for {atom}; setting to 0.");
                    // 0. no interaction.
                    // todo: If this is "CG" etc, fall back to other carbon params instead.
                    result.van_der_waals.insert(
                        i,
                        VdwParams {
                            atom_type: "".to_string(),
                            sigma: 0.,
                            eps: 0.,
                        },
                    );
                }

                // return Err(ParamError::new(&format!(
                //     "MD failure: Missing Van der Waals params for {ff_type}"
                // )));
            }
        }

        // Bond lengths.
        for bond in bonds {
            let (i0, i1) = (bond.atom_0, bond.atom_1);
            let type_0 = ff_type_from_idx(atoms, i0, "Bond")?;
            let type_1 = ff_type_from_idx(atoms, i1, "Bond")?;

            let data = params
                .bond
                .get(&(type_0.clone(), type_1.clone()))
                .or_else(|| params.bond.get(&(type_1.clone(), type_0.clone())))
                .cloned();

            let Some(data) = data else {
                // todo: We get this sometimes with glitched mmCIF files that have duplicate atoms
                // todo in slightly different positions.
                eprintln!(
                    "Missing bond parameters for {type_0}-{type_1} on {} - {}. Using a safe default.",
                    atoms[i0], atoms[i1]
                );
                result.bond_stretching.insert(
                    (i0.min(i1), i0.max(i1)),
                    BondStretchingParams {
                        atom_types: (String::new(), String::new()),
                        k_b: 300.,
                        r_0: (atoms[i0].posit - atoms[i1].posit).magnitude() as f32,
                        comment: None,
                    },
                );
                continue;
            };

            result
                .bond_stretching
                .insert((i0.min(i1), i0.max(i1)), data);
        }

        // Valence angles: Every connection between 3 atoms bonded linearly.
        for (ctr, neighbors) in adjacency_list.iter().enumerate() {
            if neighbors.len() < 2 {
                continue;
            }
            for (&n0, &n1) in neighbors.iter().tuple_combinations() {
                let type_n0 = ff_type_from_idx(atoms, n0, "Angle")?;
                let type_ctr = ff_type_from_idx(atoms, ctr, "Angle")?;
                let type_n1 = ff_type_from_idx(atoms, n1, "Angle")?;

                let data = match params.angle.get(&(
                    type_n0.clone(),
                    type_ctr.clone(),
                    type_n1.clone(),
                )) {
                    Some(param) => param.clone(),
                    // Try the other atom order.
                    None => {
                        match params.angle.get(&(
                            type_n1.clone(),
                            type_ctr.clone(),
                            type_n0.clone(),
                        )) {
                            Some(param) => param.clone(),
                            None => {
                                // todo: Get to the bottom of this.
                                // todo: In at least some cases, it's caused by duplicate atoms in the MMCIf file. Consider
                                // todo: sanitizing it on load.
                                println!(
                                    "Missing valence angle params {type_n0}-{type_ctr}-{type_n1} on {} - {} - {}. Using a safe default.",
                                    atoms[n0], atoms[ctr], atoms[n1]
                                );
                                // parm19.dat, HC-CT-HC
                                AngleBendingParams {
                                    atom_types: (String::new(), String::new(), String::new()),
                                    k: 35.,
                                    theta_0: 1.91113,
                                    comment: None,
                                }
                            }
                        }
                    }
                };

                result.angle.insert((n0, ctr, n1), data);
            }
        }

        // Proper and improper dihedral angles.
        let mut seen = HashSet::<(usize, usize, usize, usize)>::new();

        // Proper dihedrals: Atoms 1-2-3-4 bonded linearly
        for (i1, nbr_j) in adjacency_list.iter().enumerate() {
            for &i2 in nbr_j {
                if i1 >= i2 {
                    continue;
                } // handle each j-k bond once

                for &i0 in adjacency_list[i1].iter().filter(|&&x| x != i2) {
                    for &i3 in adjacency_list[i2].iter().filter(|&&x| x != i1) {
                        if i0 == i3 {
                            continue;
                        }

                        // Canonicalise so (i1, i2) is always (min, max)
                        let idx_key = if i1 < i2 {
                            (i0, i1, i2, i3)
                        } else {
                            (i3, i2, i1, i0)
                        };
                        if !seen.insert(idx_key) {
                            continue;
                        }

                        let type_0 = ff_type_from_idx(atoms, i0, "Dihedral")?;
                        let type_1 = ff_type_from_idx(atoms, i1, "Dihedral")?;
                        let type_2 = ff_type_from_idx(atoms, i2, "Dihedral")?;
                        let type_3 = ff_type_from_idx(atoms, i3, "Dihedral")?;

                        if let Some(dihe) = params.get_dihedral(
                            &(
                                type_0.clone(),
                                type_1.clone(),
                                type_2.clone(),
                                type_3.clone(),
                            ),
                            true,
                        ) {
                            let mut dihe = dihe.clone();
                            // Divide here; then don't do it during the dyamics run.
                            dihe.barrier_height /= dihe.divider as f32;
                            dihe.divider = 1;
                            result.dihedral.insert(idx_key, dihe);
                        } else {
                            return Err(ParamError::new(&format!(
                                "MD failure: Missing dihedral params for {type_0}-{type_1}-{type_2}-{type_3}"
                            )));
                        }
                    }
                }
            }
        }

        // Improper dihedrals 2-1-3-4. Atom 3 is the hub, with the other 3 atoms bonded to it.
        // The order of the others in the angle calculation affects the sign of the result.
        for (ctr, satellites) in adjacency_list.iter().enumerate() {
            if satellites.len() < 3 {
                continue;
            }

            // Unique unordered triples of neighbours
            for a in 0..satellites.len() - 2 {
                for b in a + 1..satellites.len() - 1 {
                    for d in b + 1..satellites.len() {
                        let (sat0, sat1, sat2) = (satellites[a], satellites[b], satellites[d]);
                        let idx_key = (sat0, sat1, ctr, sat2); // order is fixed → no swap
                        if !seen.insert(idx_key) {
                            continue;
                        }

                        // todo this! I believe Amber assumes the third one is the center, and you have it as the second ?!
                        let t0 = ff_type_from_idx(atoms, sat0, "Improper dihedral")?;
                        let t1 = ff_type_from_idx(atoms, sat1, "Improper dihedral")?;
                        let t_ctr = ff_type_from_idx(atoms, ctr, "Improper dihedral")?;
                        let t2 = ff_type_from_idx(atoms, sat2, "Improper dihedral")?;

                        if let Some(dihe) = params.get_dihedral(
                            &(t0.clone(), t1.clone(), t_ctr.clone(), t2.clone()),
                            false,
                        ) {
                            let mut dihe = dihe.clone();
                            // Generally, there is no divisor for impropers, but set it up here
                            // to be more general.
                            dihe.barrier_height /= dihe.divider as f32;
                            dihe.divider = 1;
                            result.improper.insert(idx_key, dihe);
                        } else {
                            return Err(ParamError::new(&format!(
                                "MD failure: Missing improper parameters for {t0}-{t1}-{t_ctr}-{t2}"
                            )));
                        }
                    }
                }
            }
        }

        Ok(result)
    }
}

impl MdState {
    /// For a dynamic ligand, and static (set of a) peptide.
    pub fn new_docking(
        atoms: &[Atom],
        atom_posits: &[Vec3],
        adjacency_list: &[Vec<usize>],
        bonds: &[Bond],
        atoms_static: &[Atom],
        ff_params: &FfParamSet,
        temp_target: f64,
        // todo: Temperature/thermostat.
    ) -> Result<Self, ParamError> {
        let Some(ff_params_lig_keyed) = &ff_params.lig_general else {
            return Err(ParamError::new("MD failure: Missing lig general params"));
        };
        let Some(ff_params_prot_keyed) = &ff_params.prot_general else {
            return Err(ParamError::new(
                "MD failure: Missing prot params general params",
            ));
        };

        // Assign FF type and charge to protein atoms; FF type must be assigned prior to initializing `ForceFieldParamsIndexed`.
        // (Ligand atoms will already have FF type assigned).

        // todo temp!
        let ff_params_keyed_lig_specific = ff_params.lig_specific.get("CPB");

        // Convert FF params from keyed to index-based.
        println!("Building FF params indexed ligand for docking...");
        let ff_params_non_static = ForceFieldParamsIndexed::new(
            ff_params_lig_keyed,
            ff_params_keyed_lig_specific,
            atoms,
            bonds,
            adjacency_list,
        )?;

        // This assumes nonbonded interactions only with external atoms; this is fine for
        // rigid protein models, and is how this is currently structured.
        let bonds_static = Vec::new();
        let adj_list_static = Vec::new();

        println!("Building FF params indexed static for docking...");
        let ff_params_static = ForceFieldParamsIndexed::new(
            ff_params_prot_keyed,
            None,
            atoms_static,
            &bonds_static,
            &adj_list_static,
        )?;

        // We are using this approach instead of `.into`, so we can use the atom_posits from
        // the positioned ligand. (its atom coords are relative; we need absolute)
        let mut atoms_dy = Vec::with_capacity(atoms.len());
        for (i, atom) in atoms.iter().enumerate() {
            atoms_dy.push(AtomDynamics::new(
                atom,
                atom_posits,
                &ff_params_non_static,
                i,
            )?);
        }

        let mut atoms_dy_static = Vec::with_capacity(atoms_static.len());
        let atom_posits_static: Vec<_> = atoms_static.iter().map(|a| a.posit).collect();

        // for (i, atom) in atoms_external.iter().enumerate() {
        for (i, atom) in atoms_static.iter().enumerate() {
            atoms_dy_static.push(AtomDynamics::new(
                atom,
                &atom_posits_static,
                &ff_params_static,
                i,
            )?);
        }

        let cell = {
            let (mut min, mut max) = (Vec3::splat(f64::INFINITY), Vec3::splat(f64::NEG_INFINITY));
            for a in &atoms_dy {
                min = min.min(a.posit);
                max = max.max(a.posit);
            }
            let pad = 15.0; // Å
            let lo = min - Vec3::splat(pad);
            let hi = max + Vec3::splat(pad);

            println!("Initizing sim box. L: {lo} H: {hi}");

            SimBox {
                bounds_low: lo,
                bounds_high: hi,
            }
        };

        // todo temp!
        let atoms_dy_static: Vec<AtomDynamics> = Vec::new();

        let mut result = Self {
            mode: MdMode::Docking,
            atoms: atoms_dy,
            adjacency_list: adjacency_list.to_vec(),
            atoms_static: atoms_dy_static,
            cell,
            nonbonded_exclusions: HashSet::new(),
            nonbonded_scaled: HashSet::new(),
            force_field_params: ff_params_non_static,
            temp_target,
            ..Default::default()
        };

        // todo temp rm
        // result.water = make_water_mols(&cell, result.temp_target);

        result.setup_nonbonded_exclusion_scale_flags();
        result.build_neighbours();

        Ok(result)
    }

    /// For a dynamic peptide, and no ligand. There is no need to filter by hetero only
    /// atoms upstream.
    pub fn new_peptide(
        atoms: &[Atom],
        atom_posits: &[Vec3],
        bonds: &[Bond],
        ff_params: &FfParamSet,
        temp_target: f64,
        // todo: Thermostat.
    ) -> Result<Self, ParamError> {
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
        println!("Building FF params indexed for peptide...");
        let ff_params_non_static = ForceFieldParamsIndexed::new(
            ff_params_prot_keyed,
            None,
            &atoms,
            &bonds_filtered,
            &adjacency_list,
        )?;

        let mut atoms_dy = Vec::with_capacity(atoms.len());
        for (i, atom) in atoms.iter().enumerate() {
            atoms_dy.push(AtomDynamics::new(
                atom,
                atom_posits,
                &ff_params_non_static,
                i,
            )?);
        }

        let cell = {
            let (mut min, mut max) = (Vec3::splat(f64::INFINITY), Vec3::splat(f64::NEG_INFINITY));
            for a in &atoms_dy {
                min = min.min(a.posit);
                max = max.max(a.posit);
            }
            let pad = 15.0; // Å
            let lo = min - Vec3::splat(pad);
            let hi = max + Vec3::splat(pad);

            println!("Initizing sim box. L: {lo} H: {hi}");

            SimBox {
                bounds_low: lo,
                bounds_high: hi,
            }
        };

        let mut result = Self {
            mode: MdMode::Peptide,
            atoms: atoms_dy,
            adjacency_list: adjacency_list.to_vec(),
            atoms_static: Vec::new(),
            cell,
            nonbonded_exclusions: HashSet::new(),
            nonbonded_scaled: HashSet::new(),
            force_field_params: ff_params_non_static,
            temp_target,
            ..Default::default()
        };

        result.water = make_water_mols(&cell, result.temp_target);

        result.setup_nonbonded_exclusion_scale_flags();
        result.build_neighbours();

        Ok(result)
    }

    /// We use this to set up optimizations defined in the Amber reference manual. `excluded` deals
    /// with sections were we skip coulomb and Vdw interactions for atoms separated by 1 or 2 bonds. `scaled14` applies a force
    /// scaler for these interactions, when separated by 3 bonds.
    fn setup_nonbonded_exclusion_scale_flags(&mut self) {
        // Helper to store pairs in canonical (low,high) order
        let push = |set: &mut HashSet<(usize, usize)>, i: usize, j: usize| {
            if i < j {
                set.insert((i, j));
            } else {
                set.insert((j, i));
            }
        };

        // 1-2
        for (indices, _) in &self.force_field_params.bond_stretching {
            push(&mut self.nonbonded_exclusions, indices.0, indices.1);
        }

        // 1-3
        for (indices, _) in &self.force_field_params.angle {
            push(&mut self.nonbonded_exclusions, indices.0, indices.2);
        }

        // 1-4. We do not count improper dihedrals here.
        for (indices, _) in &self.force_field_params.dihedral {
            push(&mut self.nonbonded_scaled, indices.0, indices.3);
        }

        // Make sure no 1-4 pair is also in the excluded set
        for p in &self.nonbonded_scaled {
            self.nonbonded_exclusions.remove(p);
        }
    }

    /// Build / rebuild Verlet list
    pub fn build_neighbours(&mut self) {
        let cutoff_sq = (CUTOFF_VDW + SKIN).powi(2);

        self.neighbour = vec![Vec::new(); self.atoms.len()];
        for i in 0..self.atoms.len() - 1 {
            for j in i + 1..self.atoms.len() {
                let dv = self
                    .cell
                    .min_image(self.atoms[j].posit - self.atoms[i].posit);

                if dv.magnitude_squared() < cutoff_sq {
                    self.neighbour[i].push(j);
                    self.neighbour[j].push(i);
                }
            }
        }
        // reset displacement tracker
        for a in &mut self.atoms {
            a.vel;
        }
        self.max_disp_sq = 0.0;
    }
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

/// Perform MD on the ligand, with nearby protein (receptor) atoms, from the docking setup as static
/// non-bonded contributors. (Vdw and coulomb)
pub fn build_dynamics_docking(
    dev: &ComputationDevice,
    lig: &mut Ligand,
    setup: &DockingSetup,
    ff_params: &FfParamSet,
    n_steps: u32,
    dt: f64,
) -> Result<MdState, ParamError> {
    println!("Building docking dyanmics...");
    let start = Instant::now();

    lig.pose.conformation_type = ConformationType::AbsolutePosits;

    let mut md_state = MdState::new_docking(
        &lig.molecule.atoms,
        &lig.atom_posits,
        &lig.molecule.adjacency_list,
        &lig.molecule.bonds,
        &setup.rec_atoms_near_site,
        ff_params,
        TEMP_TGT_DEFAULT,
    )?;

    for _ in 0..n_steps {
        md_state.step(dt)
    }

    for (i, atom) in md_state.atoms.iter().enumerate() {
        lig.atom_posits[i] = atom.posit;
    }
    change_snapshot_docking(lig, &md_state.snapshots[0], &mut None);

    Ok(md_state)
}

/// Perform MD on the peptide (protein) only. Can be very computationally intensive due to the large
/// number of atoms.
pub fn build_dynamics_peptide(
    dev: &ComputationDevice,
    mol: &mut Molecule,
    ff_params: &FfParamSet,
    n_steps: u32,
    dt: f64,
) -> Result<MdState, ParamError> {
    println!("Building peptide dynamics...");
    let start = Instant::now();

    let posits: Vec<_> = mol.atoms.iter().map(|a| a.posit).collect();

    let mut md_state =
        MdState::new_peptide(&mol.atoms, &posits, &mol.bonds, ff_params, TEMP_TGT_DEFAULT)?;

    for _ in 0..n_steps {
        md_state.step(dt)
    }

    change_snapshot_peptide(mol, &md_state.atoms, &md_state.snapshots[0]);

    Ok(md_state)
}

/// Set ligand atom positions to that of a snapshot. We assume a rigid receptor.
/// Body masses are separate from the snapshot, since it's invariant.
pub fn change_snapshot_docking(
    lig: &mut Ligand,
    snapshot: &SnapshotDynamics,
    energy_disp: &mut Option<BindingEnergy>,
) {
    lig.pose.conformation_type = ConformationType::AbsolutePosits;
    lig.atom_posits = snapshot.atom_posits.iter().map(|p| (*p).into()).collect();
    // *energy_disp = snapshot.energy.clone();
}

pub fn change_snapshot_peptide(
    mol: &mut Molecule,
    atoms_dy: &[AtomDynamics],
    snapshot: &SnapshotDynamics,
) {
    let mut posits = Vec::with_capacity(mol.atoms.len());

    // todo: This is slow. Use a predefined mapping; much faster.
    // If the atom's SN is present in the snap, use it; otherwise, use the original posit (e.g. hetero)
    for atom in &mol.atoms {
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

    mol.atom_posits = Some(posits);
}
