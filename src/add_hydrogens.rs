use std::f64::consts::TAU;

use lin_alg::f64::Vec3;
use na_seq::AminoAcid;
use crate::{
    Element,
    file_io::pdbqt::DockType,
    molecule::{Atom, AtomRole, Bond, BondCount, BondType, Molecule},
};
use crate::aa_coords::aa_data_from_coords;
use crate::molecule::ResidueType;

/// A simple enum for guessable hybridization
#[derive(Clone, Copy, PartialEq, Debug)]
enum Hybridization {
    Sp,
    Sp2,
    Sp3,
}

impl Molecule {
    /// Sum up effective bond order for a given atom index
    fn sum_bond_order_for_atom(&self, atom_index: usize) -> f64 {
        self.bonds
            .iter()
            .filter_map(|bond| {
                if bond.atom_0 == atom_index || bond.atom_1 == atom_index {
                    match bond.bond_type {
                        BondType::Covalent { count } => Some(count.value()),
                        _ => None,
                    }
                } else {
                    None
                }
            })
            .sum()
    }

    /// Attempt to guess hybridization from total bond order:
    /// - sp if we see 2.9+ (e.g. triple bond) => linear
    /// - sp2 if we see ~2 => trig planar
    /// - otherwise sp3 => tetrahedral
    ///
    /// *Naive approach*, ignoring lone pairs and other edge cases.
    fn guess_hybridization(total_bond_order: f64, needed_h: usize) -> Hybridization {
        // A quick-and-dirty approach:
        if total_bond_order >= 2.9 {
            // e.g., triple bond or near
            Hybridization::Sp
        } else if (total_bond_order >= 1.9 && total_bond_order < 2.9)
            // e.g. a double bond or hybrid
            // Also check if we only need 1 H or 0 H => typical sp2 scenario
            || (needed_h == 1 && total_bond_order > 1.0)
        {
            Hybridization::Sp2
        } else {
            // default
            Hybridization::Sp3
        }
    }

    /// Return a set of directions for the missing H’s given the guessed hybridization.
    /// The directions are unit vectors from the heavy atom center.
    /// For sp2, we make a planar arrangement. For sp, we do linear.
    /// For sp3, we do tetrahedral.
    ///
    /// Note: If e.g. an sp2 atom needs 2 hydrogens, we assume an arrangement with ~120° separation.
    fn hydrogen_directions_sp(n: usize) -> Vec<Vec3> {
        // For sp, everything is linear. If we need 1 H, place along +Z; if we need 2, +/-Z, etc.
        match n {
            1 => vec![Vec3::new(0.0, 0.0, 1.0)],
            2 => vec![Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, -1.0)],
            // More than 2 on sp is unusual, but let's place them in a line too
            _ => {
                let mut dirs = Vec::new();
                let step = 2.0 / (n as f64 - 1.0);
                for i in 0..n {
                    let z = -1.0 + step * i as f64; // from -1 to 1
                    dirs.push(Vec3::new(0.0, 0.0, z).to_normalized());
                }
                dirs
            }
        }
    }

    fn hydrogen_directions_sp2(n: usize) -> Vec<Vec3> {
        // For sp2, a simple approach is to place all H in a plane (XY) separated by 120°, 180°, etc.
        if n == 0 {
            return vec![];
        }
        let mut dirs = Vec::new();
        let n_f64 = n as f64;
        for i in 0..n {
            let angle = TAU * (i as f64) / n_f64;
            dirs.push(Vec3::new(angle.cos(), angle.sin(), 0.0));
        }
        dirs
    }

    fn hydrogen_directions_sp3(n: usize) -> Vec<Vec3> {
        const TAU_DIV_6: f64 = TAU / 6.;

        // For sp3, we try naive tetrahedral or placeholders:
        match n {
            1 => vec![Vec3::new(0.0, 0.0, 1.0)],
            2 => vec![Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, -1.0)],
            3 => {
                // "Trigonal pyramid" arrangement
                // We'll place them 120° apart in XY plane + 1 above or below, etc.
                vec![
                    Vec3::new(1.0, 0.0, 0.0),
                    Vec3::new(-0.5, 0.866, 0.0),
                    Vec3::new(-0.5, -0.866, 0.0),
                ]
            }
            4 => {
                // A more accurate tetrahedral arrangement
                // Using ~109.5° separation:
                let sqrt2over3 = (2.0_f64 / 3.0).sqrt();
                vec![
                    Vec3::new(0.0, 0.0, 1.0),
                    Vec3::new(2.0 * (TAU_DIV_6).cos(), 0.0, -0.3333),
                    Vec3::new(-(TAU_DIV_6).cos(), (TAU_DIV_6).sin() * sqrt2over3, -0.3333),
                    Vec3::new(-(TAU_DIV_6).cos(), -(TAU_DIV_6).sin() * sqrt2over3, -0.3333),
                ]
            }
            // If more than 4 needed (unusual, but can happen with e.g. hypervalent P),
            // we just place them in a circle in XY plane plus one above if needed:
            n => {
                let mut dirs = Vec::new();
                let n_f64 = n as f64;
                for i in 0..n {
                    let angle = TAU * (i as f64) / n_f64;
                    dirs.push(Vec3::new(angle.cos(), angle.sin(), 0.0));
                }
                dirs
            }
        }
            .into_iter()
            .map(|v| v.to_normalized())
            .collect()
    }

    /// Crystolography files often omit hydrogens; add them programmatically. Required for molecular
    /// dynamics.
    // todo: Some API asym between this and adding H bonds, which is a standalone fn.
    pub fn populate_hydrogens_(&mut self) {
        // Collect new atoms/bonds in vectors to be appended later
        let mut new_atoms = Vec::new();
        let mut new_bonds = Vec::new();

        // Track largest serial_number so we can assign new ones
        let mut next_serial = self
            .atoms
            .iter()
            .map(|a| a.serial_number)
            .max()
            .unwrap_or(0)
            + 1;

        // For each atom, see if it needs hydrogens
        for (idx, atom) in self.atoms.iter().enumerate() {
            // skip if it is hydrogen or metals
            if atom.element == Element::Hydrogen {
                continue;
            }
            let typical_val = atom.element.valence_typical();
            if typical_val == 0 {
                continue; // e.g. metals or unknown in this naive approach
            }

            // sum up bond order
            let current_bond_order = self.sum_bond_order_for_atom(idx);
            let needed_float = typical_val as f64 - current_bond_order;
            // if negative or near 0, do nothing
            if needed_float <= 0.0 {
                continue;
            }

            // Round up. Real chemistry can get tricky with partial bonds,
            // but let's do a naive integer rounding of needed hydrogens:
            let needed_h = needed_float.round() as usize;
            if needed_h == 0 {
                continue;
            }

            // Guess the hybridization
            let hybrid = Self::guess_hybridization(current_bond_order, needed_h);

            // Get directions for the needed H in 3D
            let directions = match hybrid {
                Hybridization::Sp => Self::hydrogen_directions_sp(needed_h),
                Hybridization::Sp2 => Self::hydrogen_directions_sp2(needed_h),
                Hybridization::Sp3 => Self::hydrogen_directions_sp3(needed_h),
            };

            // For each direction, create an H atom + bond
            for dir in directions {
                // typical bond length (C-H ~1.09, N-H ~1.0, etc.).
                // We'll pick something around 1.0 Å for demonstration.
                let bond_len = 1.0;
                let h_pos = atom.posit + dir * bond_len;

                let new_atom = Atom {
                    serial_number: next_serial,
                    posit: h_pos,
                    element: Element::Hydrogen,
                    name: "H".to_owned(), // todo: Is this right?
                    // Decide if it's backbone or sidechain, etc.
                    // For a real system, you'd check if the heavy atom is a backbone atom, etc.
                    role: Some(AtomRole::H_Backbone),
                    residue_type: atom.residue_type.clone(),
                    hetero: false,
                    partial_charge: None,
                    dock_type: None,
                    occupancy: None,
                    temperature_factor: None,
                };
                let new_atom_index = self.atoms.len() + new_atoms.len();
                new_atoms.push(new_atom);

                new_bonds.push(Bond {
                    bond_type: BondType::Covalent {
                        count: BondCount::Single,
                    },
                    atom_0: idx,
                    atom_1: new_atom_index,
                    is_backbone: false, // or match the parent's `is_backbone` if you want
                });

                next_serial += 1;
            }
        }

        // Append the new atoms and bonds
        self.atoms.extend(new_atoms);
        self.bonds.extend(new_bonds);
    }


    // todo: DIff approach below:
    pub fn populate_hydrogens(&mut self) {
        // todo: Move this fn to this module? Split this and its diehdral component, or not?

        let mut prev_cp_pos = Vec3::new_zero();

        for res in &self.residues {
            let atoms: Vec<&Atom> = res.atoms.iter().map(|i| &self.atoms[*i]).collect();

            if let ResidueType::AminoAcid(aa) = &res.res_type {
                let data = aa_data_from_coords(
                    &atoms,
                    *aa,
                    // todo: Skip the first one; not set up yet.
                    prev_cp_pos,
                );

                match data {
                    Ok((dihedral_angles, hydrogens, cp_pos)) => {
                        self.atoms.extend(hydrogens);
                        prev_cp_pos = cp_pos;
                    }
                    Err(_) => {
                        eprintln!("Problem making hydrogens");
                    }
                }

            }
        }

    }
}