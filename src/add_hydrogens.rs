use std::f64::consts::TAU;

use lin_alg::f64::Vec3;
use na_seq::AminoAcid;

use crate::{
    Element,
    aa_coords::aa_data_from_coords,
    bond_inference::{create_bonds, make_hydrogen_bonds},
    file_io::pdbqt::DockType,
    molecule::{Atom, AtomRole, Bond, BondCount, BondType, Molecule, ResidueType},
};

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

    /// Adds hydrogens, and populdates residue dihedral angles.
    pub fn populate_hydrogens(&mut self) {
        // todo: Move this fn to this module? Split this and its diehdral component, or not?

        let mut prev_cp_pos = Vec3::new_zero();
        let mut prev_ca_pos = Vec3::new_zero();

        let res_len = self.residues.len();

        let res_clone = self.residues.clone(); // todo: Come back to /avoid if possible.
        // todo: Avoids a double-borrow error below.

        for (i, res) in self.residues.iter_mut().enumerate() {
            let atoms: Vec<&Atom> = res.atoms.iter().map(|i| &self.atoms[*i]).collect();

            let mut n_next_pos = Vec3::new_zero();
            // todo: Messy DRY from the aa_data_from_coords fn.
            if i < res_len - 1 {
                let res_next = &res_clone[i + 1];
                let n_next = res_next.atoms.iter().find(|i| {
                    if let Some(role) = &self.atoms[**i].role {
                        *role == AtomRole::N_Backbone
                    } else {
                        false
                    }
                });
                if let Some(n_next) = n_next {
                    n_next_pos = self.atoms[*n_next].posit;
                }
            }

            if let ResidueType::AminoAcid(aa) = &res.res_type {
                let data = aa_data_from_coords(
                    &atoms,
                    *aa,
                    // todo: Skip the first one; not set up yet.
                    prev_cp_pos,
                    prev_ca_pos,
                    n_next_pos,
                );

                match data {
                    Ok((dihedral, hydrogens, cp_pos, ca_pos)) => {
                        for h in hydrogens {
                            self.atoms.push(h);
                            res.atoms.push(self.atoms.len() - 1);
                        }
                        prev_cp_pos = cp_pos;
                        prev_ca_pos = ca_pos;

                        res.dihedral = Some(dihedral);
                    }
                    Err(_) => {
                        eprintln!("Problem making hydrogens");
                    }
                }
            }
        }
    }
}
