use na_seq::{AtomTypeInRes, Element, Element::*};

use crate::{
    aa_coords::aa_data_from_coords,
    molecule::{Atom, AtomRole, Molecule},
};

#[derive(Clone, Copy, PartialEq)]
pub enum BondGeometry {
    Planar,
    Linear,
    Tetrahedral,
    Other,
}

/// Return the AMBER/GAFF hydrogen atom-type for a H we are adding.
/// `neighbors` is atoms bonded to the atom the H is bonded to ?
///
/// H or hn: Aimde or imino H
/// HO: On hydroxyl oxygen
/// OS: On sulfur
/// HP: On Phosphorus
/// HZ: On sp-carbon
/// HA: On Aromatic carbon
/// H4: On aromatic carbon with 1 electronegative neighbor
/// H5: On aromatic carbon with 2 electronegative neighbors
/// HC: On aliphatic carbon
/// H1: On aliphatic carbon with 1 EWD group
/// H2: On aliphatic carbon with 2 EWD groups
/// H3: On aliphatic carbon with 3 EWD groups
///
/// See [this unofficial page](https://emleddin.github.io/comp-chem-website/AMBERguide-AMBER-atom-types.html)
/// todo: We apply the residue atom type (e.g. "HB2", "HD23", "HA" etc, and
/// use the amber params to load ff type. So, not described as above.
pub fn h_at_type_in_res(
    element: Element,
    geometry: BondGeometry,
    neighbor_count: usize,
) -> AtomTypeInRes {
    // `neighbor_count` is # of` atoms bound to the *parent* carbon
    // This is equivalent to "electron-withdrawing-group". (EWG)

    return AtomTypeInRes::H("HA".to_string()); // todo temp!
    //
    // // todo: QC these bindings.
    // match element {
    //     Nitrogen => "H",
    //     Oxygen => "HO",
    //     Sulfur => "HS",
    //     Phosphorus => "HP",
    //     Carbon => match geometry {
    //         BondGeometry::Planar => match neighbor_count {
    //             0 => "HA", // Aromatic
    //             1 => "H4", // Aliphatic with 4 EWG
    //             _ => "H5",
    //         },
    //         BondGeometry::Linear => "HZ",
    //         _ => match neighbor_count {
    //             // Aliphatic.
    //             0 => "HC",
    //             1 => "H1", // 1 EWG
    //             2 => "H2", // 2 EWG etc
    //             _ => "H3",
    //         },
    //     },
    //     _ => "H", // Default.
    // }
    // .to_string()
}

/// Helper? todo: Figure out this thing's deal...
pub fn bonded_heavy_atoms<'a>(atoms_bonded: &'a [(usize, &'a Atom)]) -> Vec<&'a Atom> {
    atoms_bonded.iter().map(|(_, a)| *a).collect()
}

impl Molecule {
    /// Adds hydrogens, and populdates residue dihedral angles.
    pub fn populate_hydrogens_angles(&mut self) {
        // todo: Move this fn to this module? Split this and its diehdral component, or not?

        let mut prev_cp_ca = None;

        let res_len = self.residues.len();

        // todo: The Clone avoids a double-borrow error below. Come back to /avoid if possible.
        let res_clone = self.residues.clone();

        for (res_i, res) in self.residues.iter_mut().enumerate() {
            let atoms: Vec<&Atom> = res.atoms.iter().map(|i| &self.atoms[*i]).collect();

            let mut n_next_pos = None;
            // todo: Messy DRY from the aa_data_from_coords fn.
            if res_i < res_len - 1 {
                let res_next = &res_clone[res_i + 1];
                let n_next = res_next.atoms.iter().find(|i| {
                    if let Some(role) = &self.atoms[**i].role {
                        *role == AtomRole::N_Backbone
                    } else {
                        false
                    }
                });
                if let Some(n_next) = n_next {
                    n_next_pos = Some(self.atoms[*n_next].posit);
                }
            }

            // Get the first atom's chain; probably OK for assigning a chain to H.
            let chain_i = if !atoms.is_empty() {
                atoms[0].chain.unwrap_or_default()
            } else {
                0
            };
            let (dihedral, hydrogens, this_cp_ca) = aa_data_from_coords(
                &atoms,
                &res.res_type,
                res_i,
                chain_i,
                prev_cp_ca,
                n_next_pos,
            );

            for h in hydrogens {
                self.atoms.push(h);
                res.atoms.push(self.atoms.len() - 1);

                // todo: Add to the chains
            }

            prev_cp_ca = this_cp_ca;
            res.dihedral = Some(dihedral);
        }
    }
}
