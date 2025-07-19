use na_seq::{AtomTypeInRes, Element, Element::*};

use crate::{
    aa_coords::aa_data_from_coords,
    molecule::{Atom, AtomRole, Molecule},
};
use crate::dynamics::ParamError;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum BondGeometry {
    Planar,
    Linear,
    Tetrahedral,
    Other,
}

/// Assign atom-type-in-res for hydrogen atoms in polypeptides. This is not for small molecules,
/// which use GAFF types, nor generally required for them: Files for those tend to include H atoms,
/// while mmCIF and PDF files for proteins generally don't.
///
/// This function is for sidechain only; Backbone H are always "H" for on N, and "HA", "HA2", or "HA3"
/// for on Cα (The latter two for the case of Glycine only, which has no sidechain).
///
/// `neighbors` is atoms bonded to the atom the H is bonded to ?
/// Reference `amino19.lib`, which shows which atom-in-res types we should expect (including)
/// for these H atoms.
///
/// We need to correctly populate these atom-in-res types, to properly assign Amber FF type, and
/// partial charge downstream.
///
/// Example. For Asp, we should have one each of "H", "HA", "HB2", and "HB3".
// pub fn h_at_type_in_res_sidechain(parent_name: &str, ordinal: usize, total_h: usize) -> Result<AtomTypeInRes, ParamError> {
pub fn h_at_type_in_res_sidechain(element: Element, geom: BondGeometry, neighbor_count: usize) -> Result<AtomTypeInRes, ParamError> {
    let result = String::from("H");

    // if total_h < 1 || total_h > 3 {
    //     return Err(ParamError::new("Invalid total H count when assigning atom type to H"));
    // }
    //
    // let rest = &parent_name[1..];                 // strip the leading C/N/O/S/P
    // let first_char = parent_name.chars().next().unwrap();
    //
    // let result = match first_char {
    //     // Backbone
    //     'N' if parent_name == "N" => {
    //         if total_h == 1 {
    //             "H".into()
    //         } else {
    //             format!("H{}", ordinal + 1)       // H1 / H2 / H3
    //         }
    //     }
    //     'C' if parent_name == "CA" => {
    //         if total_h == 1 {
    //             "HA".into()
    //         } else {
    //             // Gly: ordinal 0→HA2  , 1→HA3
    //             format!("HA{}", ordinal + 2)
    //         }
    //     }
    //
    //     // Side-chain & hetero atoms ─────────────────────────────────────────────
    //     'C' | 'N' | 'O' | 'S' | 'P' => {
    //         let mut name = String::from("H");
    //         name.push_str(rest);
    //
    //         match total_h {
    //             1 => name,                        // HB, HG, HD1, HG2, HH, …
    //             2 => {
    //                 // CB → HB2/HB3; CG1 → HG12/HG13
    //                 name.push(char::from(b'2' + ordinal as u8));
    //                 name
    //             }
    //             3 => {
    //                 // methyl: HG21/HG22/HG23, HD11/HD12/HD13 …
    //                 name.push(char::from(b'1' + ordinal as u8));
    //                 name
    //             }
    //             _ => unreachable!(),
    //         }
    //     }
    //     _ => {
    //         return Err(ParamError::new(&format!("Unsupported parent atom when finding H atom type: {parent_name}")));
    //     }
    // };

    Ok(AtomTypeInRes::H(result))
}

/// Helper? todo: Figure out this thing's deal...
pub fn bonded_heavy_atoms<'a>(atoms_bonded: &'a [(usize, &'a Atom)]) -> Vec<&'a Atom> {
    atoms_bonded.iter().map(|(_, a)| *a).collect()
}

impl Molecule {
    /// Adds hydrogens, and populdates residue dihedral angles.
    pub fn populate_hydrogens_angles(&mut self) -> Result<(), ParamError> {
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
            )?;

            for h in hydrogens {
                self.atoms.push(h);
                res.atoms.push(self.atoms.len() - 1);

                // todo: Add to the chains
            }

            prev_cp_ca = this_cp_ca;
            res.dihedral = Some(dihedral);
        }

        Ok(())
    }
}
