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

pub type DigitMap = HashMap<AminoAcid, HashMap<char, Vec<u8>>>;

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

// todo: Include N and C terminus maps A/R.
/// Helper to get the digit part of the H from what's expected in Amber's naming conventions.
/// E.g. this might map an incrementing `0` and `1` to `2` and `3` for HE2 and HE3.
pub fn make_h_digit_map(ff_map: &ProtFfMap) -> DigitMap {
    let mut result: DigitMap = HashMap::new();

    // ff_map is assumed to be something like
    // HashMap<AminoAcid, Vec<AtomData>>
    for (&aa_gen, params) in ff_map {
        let mut per_heavy: HashMap<char, Vec<u8>> = HashMap::new();

        for cp in params {
            let tir = &cp.type_in_res; // adjust accessor as needed

            match tir {
                AtomTypeInRes::H(name) => {
                    // No room for designator or digit – skip "H" or "HA" etc.
                    if name.len() < 3 {
                        continue;
                    }

                    // Split:  H  <designator-char>  <digits...>
                    let mut chars = name.chars();
                    chars.next(); // discard the leading 'H'

                    // Heavy-atom designator is always a single alphabetic char
                    let designator = match chars.next() {
                        Some(c) if c.is_ascii_alphabetic() => c,
                        _ => continue, // malformed – ignore
                    };

                    // Collect *all* trailing digits (handles "11", "21", ...)
                    let digits: String = chars.filter(|c| c.is_ascii_digit()).collect();
                    if digits.is_empty() {
                        continue;
                    }

                    // Safe because Amber never goes beyond two digits
                    let num: u8 = digits.parse().unwrap();

                    per_heavy.entry(designator).or_default().push(num);
                }
                // We only care about hydrogens that *do* carry a numeric suffix
                _ => (),
            }
        }

        let aa = match aa_gen {
            AminoAcidGeneral::Standard(a) => a,
            _ => continue,
        };

        // Make the relationship deterministic (ordinal 0 → smallest digit, …)
        for v in per_heavy.values_mut() {
            v.sort_unstable();
        }

        if !per_heavy.is_empty() {
            result.insert(aa, per_heavy);
        }
    }

    result
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
/// `h_num_this_parent` increments from 0. We use a table to map these to digits, e.g. 0 and 1 might mean the
/// `2` and `3` in "HB2" and "HB3". Increments for a given parent that has multiple H.
/// Assigns the numerical value in the result, e.g. the "2" in "NE2". `parent_depth` provides the letter
/// e.g. the "D" in "HD1". (WHere "H" means Hydrogen, and "1" means the first hydrogen attached to this parent.
pub fn h_type_in_res_sidechain(
    h_num_this_parent: usize,
    parent_tir: &AtomTypeInRes,
    aa: AminoAcid,
    ff_map: &ProtFfMap,
    h_digit_map: &DigitMap,
) -> Result<AtomTypeInRes, ParamError> {
    // todo: Assign the number based on parent type as well??
    let depth = match parent_tir {
        AtomTypeInRes::CB => 'B',
        AtomTypeInRes::CD | AtomTypeInRes::CD1 | AtomTypeInRes::CD2 => 'D',
        AtomTypeInRes::CE | AtomTypeInRes::CE1 | AtomTypeInRes::CE2 | AtomTypeInRes::CE3 => 'E',
        AtomTypeInRes::CG | AtomTypeInRes::CG1 | AtomTypeInRes::CG2 => 'G',
        AtomTypeInRes::CH2 | AtomTypeInRes::CH3 => 'H',
        AtomTypeInRes::CZ | AtomTypeInRes::CZ1 | AtomTypeInRes::CZ2 | AtomTypeInRes::CZ3 => 'Z',
        AtomTypeInRes::OD1 | AtomTypeInRes::OD2 => 'D',
        AtomTypeInRes::OG | AtomTypeInRes::OG1 | AtomTypeInRes::OG2 => 'G',
        AtomTypeInRes::OE1 | AtomTypeInRes::OE2 => 'E',
        AtomTypeInRes::ND1 | AtomTypeInRes::ND2 => 'D',
        AtomTypeInRes::NH1 | AtomTypeInRes::NH2 => 'H',
        AtomTypeInRes::NE | AtomTypeInRes::NE1 | AtomTypeInRes::NE2 => 'E',
        AtomTypeInRes::SE => 'E',
        AtomTypeInRes::SG => 'G',
        _ => {
            return Err(ParamError::new(&format!(
                "Invalid parent type in res on H assignment: {parent_tir:?}",
            )));
        }
    };

    // todo: Don't unwrap.
    let digits = h_digit_map.get(&aa).unwrap().get(&depth).unwrap();

    let digit = match digits.get(h_num_this_parent) {
        Some(d) => d,
        None => {
            return Err(ParamError::new(&format!(
                "H Digit out of range. Digit: {h_num_this_parent} not in {digits:?} - {parent_tir:?}",
            )));
        }
    };

    // todo: Handle the N term and C term cases; pass those params in.

    // todo: Consider adding a completeness validator for the AA, ensuring all expected
    // todo: Hs are present.

    let result = AtomTypeInRes::H(format!("H{depth}{digit}"));

    if !validate_h_atom_type(&result, aa, ff_map)? {
        return Err(ParamError::new(&format!(
            "Invalid H type: {result} on {aa}. Parent: {parent_tir}"
        )));
    }

    Ok(result)
}

impl Molecule {
    /// Adds hydrogens, and populdates residue dihedral angles.
    pub fn populate_hydrogens_angles(&mut self, ff_map: &ProtFfMap) -> Result<(), ParamError> {
        // todo: Move this fn to this module? Split this and its diehdral component, or not?

        let mut prev_cp_ca = None;

        let res_len = self.residues.len();

        // todo: The Clone avoids a double-borrow error below. Come back to /avoid if possible.
        let res_clone = self.residues.clone();

        let digit_map = make_h_digit_map(ff_map);

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

            // todo: Handle the N term and C term cases; pass those params in.
            let (dihedral, hydrogens, this_cp_ca) = aa_data_from_coords(
                &atoms,
                &res.res_type,
                res_i,
                chain_i,
                prev_cp_ca,
                n_next_pos,
                &res_clone,
                ff_map,
                &digit_map,
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
