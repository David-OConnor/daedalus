use std::collections::HashMap;

use bio_files::amber_params::ChargeParams;
use na_seq::{AminoAcid, AminoAcidGeneral, AtomTypeInRes, Element, Element::*};

use crate::{
    ProtFfMap,
    aa_coords::aa_data_from_coords,
    dynamics::ParamError,
    molecule::{Atom, AtomRole, Molecule},
};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum BondGeometry {
    Planar,
    Linear,
    Tetrahedral,
    Other,
}

/// We use this to validate H atom type assignments. We derive this directly from `amino19.lib` (Amber)
/// Returns `true` if valid.
/// Note that this does not ensure completeness of the H set for a given AA; only if a given
/// value is valid for that AA.
/// h_num=0 means it's just "HE" or similar.
fn validate_h_atom_type(
    tir: &AtomTypeInRes,
    aa: AminoAcid,
    ff_map: &ProtFfMap,
) -> Result<bool, ParamError> {
    // Our protein files only contains "standard" AA data. I.e. "HIS" vice "HIE".
    let data = ff_map.get(&AminoAcidGeneral::Standard(aa)).ok_or_else(|| {
        ParamError::new(&format!(
            "No parm19_data entry for amino acid {:?}",
            AminoAcidGeneral::Standard(aa)
        ))
    })?;

    for cp in data {
        if &cp.type_in_res == tir {
            return Ok(true);
        }
    }

    Ok(false)
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
///
/// `h_num_this_parent` is 1, 2, or sometimes 3. Increments for a given parent that has multiple H.
/// Assigns the numerical value in the result, e.g. the "2" in "NE2". `parent_depth` provides the letter
/// e.g. the "D" in "HD1". (WHere "H" means Hydrogen, and "1" means the first hydrogen attached to this parent.
pub fn h_type_in_res_sidechain(
    h_num_this_parent: usize,
    parent_depth: usize,
    aa: AminoAcid,
    ff_map: &ProtFfMap,
) -> Result<AtomTypeInRes, ParamError> {
    let depths = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];

    if parent_depth >= depths.len() {
        return Err(ParamError::new(&format!(
            "Invalid parent depth on H assignment: {:?}",
            parent_depth
        )));
    }

    // todo: Consider adding a completeness validator for the AA, ensuring all expected
    // Hs are present.

    let result = AtomTypeInRes::H(format!("H{}{h_num_this_parent}", depths[parent_depth]));

    if !validate_h_atom_type(&result, aa, ff_map)? {
        return Err(ParamError::new(&format!(
            "Invalid H type: {result} for {aa}"
        )));
    }

    Ok(result)
}

/// Helper? todo: Figure out this thing's deal...
pub fn bonded_heavy_atoms<'a>(atoms_bonded: &'a [(usize, &'a Atom)]) -> Vec<&'a Atom> {
    atoms_bonded.iter().map(|(_, a)| *a).collect()
}

impl Molecule {
    /// Adds hydrogens, and populdates residue dihedral angles.
    pub fn populate_hydrogens_angles(&mut self, ff_map: &ProtFfMap) -> Result<(), ParamError> {
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
                &res_clone,
                ff_map,
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
