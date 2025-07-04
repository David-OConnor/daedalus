//! Contains setup code, including applying forcefield data to our specific
//! atoms.

use std::collections::HashSet;

use bio_files::frcmod::DihedralData;
use itertools::Itertools;
use lin_alg::f64::Vec3;
use na_seq::element::LjTable;

use crate::{
    dynamics::{
        AtomDynamics, CUTOFF, ForceFieldParamsIndexed, ForceFieldParamsKeyed, MdState, ParamError,
        SKIN, ambient::SimBox,
    },
    molecule::{Atom, Bond},
};

/// Build a single lookup table in which ligand-specific parameters
/// (when given) replace or add to the generic ones.
fn merge_params(
    generic: &ForceFieldParamsKeyed,
    lig_specific: Option<&ForceFieldParamsKeyed>,
) -> ForceFieldParamsKeyed {
    // Start with a deep copy of the generic parameters.
    let mut merged = generic.clone();

    if let Some(lig) = lig_specific {
        merged.mass.extend(lig.mass.clone());
        merged.partial_charges.extend(lig.partial_charges.clone());
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

/// Associate loaded Force field data (e.g. from Amber) into the atom indices used in a specific
/// dynamics sim.
impl ForceFieldParamsIndexed {
    pub fn new(
        params_general: &ForceFieldParamsKeyed,
        params_lig_specific: Option<&ForceFieldParamsKeyed>,
        atoms: &[Atom],
        bonds: &[Bond],
        adjacency_list: &[Vec<usize>],
    ) -> Result<Self, ParamError> {
        let mut result = Self::default();

        // Combine the two force field sets. When a value is present in both, refer the lig-specific
        // one.
        let params = merge_params(params_general, params_lig_specific);

        /* ---------- per–atom tables --------------------------------------------------------- */
        for (idx, atom) in atoms.iter().enumerate() {
            let ff_type = atom
                .force_field_type
                .as_ref()
                .ok_or_else(|| ParamError::new("Atom missing FF type"))?;

            // todo: A/R: Is this much different from element.atomic_weight()?
            // // Mass
            // let mass = params
            //     .mass
            //     .get(name)
            //     .ok_or_else(|| ParamError::new(&format!("No mass entry for '{name}'")))?;
            //
            // if result.mass.len() <= idx {
            //     result.mass.resize(idx + 1, mass.clone());
            // } else {
            //     result.mass[idx] = mass.clone();
            // }

            // Partial charge (optional -- leave 0.0 if not given)
            // todo: Add.
            // if let Some(q) = params.partial_charges.get(name) {
            //     if result.partial_charges.len() <= idx {
            //         result.partial_charges.resize(idx + 1, *q);
            //     } else {
            //         result.partial_charges[idx] = *q;
            //     }
            // }

            // Lennard-Jones / van der Waals
            // todo: Add.
            // if let Some(vdw) = params.van_der_waals.get(name) {
            //     if result.van_der_waals.len() <= idx {
            //         result.van_der_waals.resize(idx + 1, vdw.clone());
            //     } else {
            //         result.van_der_waals[idx] = vdw.clone();
            //     }
            // }
        }

        // Bonds
        for bond in bonds {
            let (i, j) = (bond.atom_0, bond.atom_1);
            let (type_i, type_j) = (
                atoms[i]
                    .force_field_type
                    .as_ref()
                    .ok_or_else(|| ParamError::new("Atom missing FF type"))?,
                atoms[j]
                    .force_field_type
                    .as_ref()
                    .ok_or_else(|| ParamError::new("Atom missing FF type"))?,
            );

            let data = params
                .bond
                .get(&(type_i.clone(), type_j.clone()))
                .or_else(|| params.bond.get(&(type_j.clone(), type_i.clone())))
                .cloned()
                .ok_or_else(|| {
                    ParamError::new(&format!("Missing bond parameters for {type_i}-{type_j}"))
                })?;

            result.bond.insert((i.min(j), i.max(j)), data);
        }

        // Angles. (Between 3 atoms)
        for (center, neigh) in adjacency_list.iter().enumerate() {
            if neigh.len() < 2 {
                continue;
            }
            for (&i, &k) in neigh.iter().tuple_combinations() {
                let (type_i, type_j, type_k) = (
                    atoms[i]
                        .force_field_type
                        .as_ref()
                        .ok_or_else(|| ParamError::new("Atom missing FF type"))?,
                    atoms[center]
                        .force_field_type
                        .as_ref()
                        .ok_or_else(|| ParamError::new("Atom missing FF type"))?,
                    atoms[k]
                        .force_field_type
                        .as_ref()
                        .ok_or_else(|| ParamError::new("Atom missing FF type"))?,
                );

                let data = params
                    .angle
                    .get(&(type_i.clone(), type_j.clone(), type_k.clone()))
                    .or_else(|| {
                        params
                            .angle
                            .get(&(type_k.clone(), type_j.clone(), type_i.clone()))
                    })
                    .cloned()
                    .ok_or_else(|| {
                        ParamError::new(&format!(
                            "No ANGLE parameters for {type_i}-{type_j}-{type_k}"
                        ))
                    })?;

                result.angle.insert((i, center, k), data);
            }
        }

        // Proper dihedral angles.
        let mut seen = HashSet::<(usize, usize, usize, usize)>::new();

        for (j, nbr_j) in adjacency_list.iter().enumerate() {
            for &k in nbr_j {
                if j >= k {
                    continue; // treat each central bond j-k once
                }
                for &i in &adjacency_list[j] {
                    if i == k {
                        continue;
                    }
                    for &l in &adjacency_list[k] {
                        if l == j {
                            continue;
                        }

                        let idx_key = (i, j, k, l);
                        if !seen.insert(idx_key) {
                            continue; // already handled through another path
                        }

                        let (type_i, type_j, type_k, type_l) = (
                            atoms[i]
                                .force_field_type
                                .as_ref()
                                .ok_or_else(|| ParamError::new("Atom missing FF type"))?,
                            atoms[j]
                                .force_field_type
                                .as_ref()
                                .ok_or_else(|| ParamError::new("Atom missing FF type"))?,
                            atoms[k]
                                .force_field_type
                                .as_ref()
                                .ok_or_else(|| ParamError::new("Atom missing FF type"))?,
                            atoms[l]
                                .force_field_type
                                .as_ref()
                                .ok_or_else(|| ParamError::new("Atom missing FF type"))?,
                        );

                        let mut data = params.dihedral.get(&(
                            type_i.clone(),
                            type_j.clone(),
                            type_k.clone(),
                            type_l.clone(),
                        ));

                        if data.is_none() {
                            data = params.dihedral.get(&(
                                type_l.clone(),
                                type_k.clone(),
                                type_j.clone(),
                                type_i.clone(),
                            ));
                        }

                        match data {
                            Some(d) => {
                                result.dihedral.insert(idx_key, d.clone());
                            }
                            None => {
                                eprintln!(
                                    "No dihedral parameters for \
                                     {type_i}-{type_j}-{type_k}-{type_l}"
                                );
                                result.dihedral.insert(idx_key, Default::default());

                                // return Err(ParamError::new(&format!(
                                //     "No dihedral parameters for \
                                //      {type_i}-{type_j}-{type_k}-{type_l}"
                                // )));
                            }
                        }
                    }
                }
            }
        }

        // todo: Handle improper. A lot of the missing params you see for dihedral are impropers.

        // println!("\n\nFF for this ligand: {:?}", result);

        Ok(result)
    }
}

// impl ForceFieldParamsIndexed {
//     pub fn new(
//         params: &ForceFieldParamsKeyed,
//         atoms: &[Atom],
//         bonds: &[Bond],
//         adjacency_list: &[Vec<usize>],
//     ) -> Result<Self, ParamError> {
//         let mut result = Self::default();
//
//         // todo: Mainly evaluating if we've properly loaded atom names here.
//         println!("Getting field params for: {:?}", atoms);
//
//         // Load mass and partial charge data per atom.
//         for (i, atom) in atoms.iter().enumerate() {}
//
//         // Match bond data to indices.
//         for bond in bonds {
//             // todo: Error handling on index range?
//             let atom_0 = &atoms[bond.atom_0];
//             let atom_1 = &atoms[bond.atom_1];
//
//             let Some((atom_name_0, atom_name_1)) = (atom_0.name.as_ref(), atom_1.name.as_ref())
//             else {
//                 return Err(ParamError::new(
//                     "Missing an atom name when loading bond params.",
//                 ));
//             };
//
//             // todo: Use the key?
//             for data in &params.bond.values() {
//                 let mut found = false;
//
//                 if (data.atom_names.0 == atom_name_0 && data.atom_names.1 == atom_name_1)
//                     || (data.atom_names.1 == atom_name_0 && data.atom_names.0 == atom_name_1)
//                 {
//                     result.bond.insert((bond.atom_0, bond.atom_1), data.clone());
//                     found = true;
//                 }
//
//                 if !found {
//                     return Err(ParamError::new(&format!(
//                         "Unable to find bond data for {atom_name_0}-{atom_name_1}"
//                     )));
//                 }
//             }
//         }
//
//         // Set up angles of 3 atoms.
//         // for (i, atom) in atoms.iter().enumerate() {
//         for i_outer in 0..atoms.len() {
//             if adjacency_list[i_outer].len() < 3 {
//                 continue;
//             }
//             // todo: Fill this out properly to get all unique combinations.
//             for (i, j, k) in adjacency_list[i_outer].combinations() {
//                 let atom_0 = &atoms[i];
//                 let atom_1 = &atoms[j];
//                 let atom_2 = &atoms[k];
//
//                 let Some((atom_name_0, atom_name_1, atom_name_2)) = (
//                     atom_0.name.as_ref(),
//                     atom_1.name.as_ref(),
//                     atom_2.name.as_ref(),
//                 ) else {
//                     return Err(ParamError::new(
//                         "Missing an atom name when loading angle params.",
//                     ));
//                 };
//
//                 // todo: Use the key?
//                 for data in &params.angle.values() {
//                     let mut found = false;
//
//                     // todo: Fix this logic; n eeds to be order-invariant to all (6?) combinations.
//                     if (data.atom_names.0 == atom_name_0
//                         && data.atom_names.1 == atom_name_1
//                         && data.atom_names.2 == atom_name_2)
//                         || (data.atom_names.1 == atom_name_0 && data.atom_names.0 == atom_name_1)
//                     {
//                         result.angle.insert((i, j, k), data.clone());
//                         found = true;
//                     }
//
//                     if !found {
//                         return Err(ParamError::new(&format!(
//                             "Unable to find angle data for {atom_name_0}-{atom_name_1}-{atom_name_2}"
//                         )));
//                     }
//                 }
//
//                 // Set up dihedral angles.
//
//                 // Generate every i–j–k–l path (three consecutive bonds)
//                 let mut seen = HashSet::<(usize, usize, usize, usize)>::new();
//
//                 for (j, adjacency_list_j) in adjacency_list.iter().enumerate() {
//                     for &k in adjacency_list_j {
//                         if j >= k {
//                             continue;
//                         } // treat each central bond once
//                         for &i in &adjacency_list[j] {
//                             if i == k {
//                                 continue;
//                             } // same bond
//                             for &l in &adjacency_list[k] {
//                                 if l == j {
//                                     continue;
//                                 }
//                                 // canonical ordering keeps i-j-k-l as unique key
//                                 let key = (i, j, k, l);
//                                 if !seen.insert(key) {
//                                     continue;
//                                 }
//
//                                 let atom_0 = &atoms[i];
//                                 let atom_1 = &atoms[j];
//                                 let atom_2 = &atoms[k];
//                                 let atom_3 = &atoms[l];
//
//                                 let Some((atom_name_0, atom_name_1, atom_name_2, atom_name_3)) = (
//                                     atom_0.name.as_ref(),
//                                     atom_1.name.as_ref(),
//                                     atom_2.name.as_ref(),
//                                     atom_3.name.as_ref(),
//                                 ) else {
//                                     return Err(ParamError::new(
//                                         "Missing an atom name when loading dihedral params.",
//                                     ));
//                                 };
//
//                                 // todo: Use the key?
//                                 for data in &params.angle.values() {
//                                     let mut found = false;
//
//                                     // todo: Fix this logic; n eeds to be order-invariant to all (6?) combinations.
//                                     if (data.atom_names.0 == atom_name_0
//                                         && data.atom_names.1 == atom_name_1
//                                         && data.atom_names.2 == atom_name_2)
//                                         || (data.atom_names.1 == atom_name_0 && data.atom_names.0 == atom_name_1)
//                                     {
//                                         result.dihedral.insert((i, j, k, l), data.clone());
//                                         found = true;
//                                     }
//
//                                     if !found {
//                                         return Err(ParamError::new(&format!(
//                                             "Unable to find dihedral data for {atom_name_0}-{atom_name_1}-{atom_name_2}-{atom_name_3}"
//                                         )));
//                                     }
//                                 }
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//
//         Ok(result)
//     }
// }

impl MdState {
    pub fn new(
        atoms: &[Atom],
        atom_posits: &[Vec3],
        adjacency_list: &[Vec<usize>],
        bonds: &[Bond],
        atoms_external: &[Atom],
        lj_table: &LjTable,
        ff_params_keyed: &ForceFieldParamsKeyed,
        ff_params_keyed_lig_specific: Option<&ForceFieldParamsKeyed>,
    ) -> Result<Self, ParamError> {
        // Convert FF params from keyed to index-based.
        let force_field_params = ForceFieldParamsIndexed::new(
            ff_params_keyed,
            ff_params_keyed_lig_specific,
            atoms,
            bonds,
            adjacency_list,
        )?;

        // We are using this approach instead of `.into`, so we can use the atom_posits from
        // the positioned ligand. (its atom coords are relative; we need absolute)
        let mut atoms_dy = Vec::with_capacity(atoms.len());
        for (i, atom) in atoms.iter().enumerate() {
            atoms_dy.push(AtomDynamics {
                element: atom.element,
                name: atom.name.clone().unwrap_or_default(),
                posit: atom_posits[i],
                vel: Vec3::new_zero(),
                accel: Vec3::new_zero(),
                mass: atom.element.atomic_weight() as f64,
                partial_charge: atom.partial_charge.unwrap_or_default() as f64,
                force_field_type: Some(atom.force_field_type.clone().unwrap_or_default()),
            });
        }

        // let atoms_dy = atoms.iter().map(|a| a.into()).collect();
        // let bonds_dy = bonds.iter().map(|b| b.into()).collect();

        // // todo: Temp on bonds this way until we know how to init r0.
        // let bonds_dy = bonds
        //     .iter()
        //     .map(|b| BondDynamics::from_bond(b, atoms))
        //     .collect();

        let atoms_dy_external: Vec<_> = atoms_external.iter().map(|a| a.into()).collect();

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

            SimBox { lo, hi }
        };

        let mut result = Self {
            atoms: atoms_dy,
            // bonds: bonds_dy,
            adjacency_list: adjacency_list.to_vec(),
            atoms_external: atoms_dy_external,
            // lj_lut: lj_table.clone(),
            cell,
            excluded_pairs: HashSet::new(),
            scaled14_pairs: HashSet::new(),
            force_field_params,
            ..Default::default()
        };

        result.build_masks();
        result.build_neighbours();

        Ok(result)
    }

    // todo: Evaluate whtaq this does, and if you keep it, document.
    fn build_masks(&mut self) {
        // Helper to store pairs in canonical (low,high) order
        let mut push = |set: &mut HashSet<(usize, usize)>, i: usize, j: usize| {
            if i < j {
                set.insert((i, j));
            } else {
                set.insert((j, i));
            }
        };

        // 1-2
        for (indices, _) in &self.force_field_params.bond {
            push(&mut self.excluded_pairs, indices.0, indices.1);
        }

        // 1-3
        for (indices, _) in &self.force_field_params.angle {
            push(&mut self.excluded_pairs, indices.0, indices.2);
        }

        // 1-4
        for (indices, _) in &self.force_field_params.dihedral {
            push(&mut self.scaled14_pairs, indices.0, indices.3);
        }

        // Make sure no 1-4 pair is also in the excluded set
        for p in &self.scaled14_pairs {
            self.excluded_pairs.remove(p);
        }
    }

    /// Build / rebuild Verlet list
    pub fn build_neighbours(&mut self) {
        let cutoff2 = (CUTOFF + SKIN).powi(2);
        self.neighbour = vec![Vec::new(); self.atoms.len()];
        for i in 0..self.atoms.len() - 1 {
            for j in i + 1..self.atoms.len() {
                let dv = self
                    .cell
                    .min_image(self.atoms[j].posit - self.atoms[i].posit);
                if dv.magnitude_squared() < cutoff2 {
                    self.neighbour[i].push(j);
                    self.neighbour[j].push(i);
                }
            }
        }
        // reset displacement tracker
        for a in &mut self.atoms {
            a.vel /* nothing */;
        }
        self.max_disp_sq = 0.0;
    }
}
