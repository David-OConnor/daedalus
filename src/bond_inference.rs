//! This module creates bonds between protein components. Most macromolecule PDB/CIF files don't include
//! explicit bond information, and the `pdbtbx` library doesn't handle this. Infer bond lengths
//! by comparing each interactomic bond distance, and matching against known amino acid bond lengths.
//!
//! Some info here: https://www.ruppweb.org/Xray/tutorial/protein_structure.htm
//! https://itp.uni-frankfurt.de/~engel/amino.html
//! These lengths are in angstrom.

use rayon::iter::ParallelIterator;

use crate::{Atom, Bond, BondType};

// Peptide
// Double bond len of C' to N.
const LEN_CP_N: f64 = 1.33;
const LEN_N_CALPHA: f64 = 1.46;
const LEN_CALPHA_CP: f64 = 1.53;

// Single bond
const LEN_C_C: f64 = 1.54;
const LEN_C_N: f64 = 1.48;
const LEN_C_O: f64 = 1.43;

// Hydrogen
const LEN_OH_OH: f64 = 2.8;
const LEN_NH_OC: f64 = 2.9;
const LEN_OH_OC: f64 = 2.8;

// Bonds to H. Mostly ~1
const LEN_N_H: f64 = 1.00;
const LEN_C_H: f64 = 1.10;
const LEN_O_H: f64 = 1.0; // In water molecules. What is it in proteins?

// If interatomic distance is within this distance of one of our known bond lenghts, consider it to be a bond.
const BOND_LEN_THRESH: f64 = 0.02; // todo: Adjust A/R based on performance.

/// Infer bonds from atoms. Slow: O(n^2).
pub fn create_bonds(atoms: &[Atom]) -> Vec<Bond> {
    // todo: Paralllize?
    let mut result = Vec::new();

    for atom_0 in atoms {
        for atom_1 in atoms {
            let dist = (atom_0.posit - atom_1.posit).magnitude();

            // todo: Other bond types
            for bond_len in [
                LEN_CP_N,
                LEN_N_CALPHA,
                LEN_CALPHA_CP,
                LEN_C_C,
                LEN_C_N,
                LEN_C_O,
                LEN_N_H,
                LEN_C_H,
                LEN_O_H,
            ] {
                if (dist - bond_len).abs() < BOND_LEN_THRESH {
                    result.push(Bond {
                        bond_type: BondType::Covalent,
                        posit_0: atom_0.posit,
                        posit_1: atom_1.posit,
                    });
                }
            }

            for bond_len in [LEN_OH_OH, LEN_NH_OC, LEN_OH_OC] {
                if (dist - bond_len).abs() < BOND_LEN_THRESH {
                    result.push(Bond {
                        bond_type: BondType::Hydrogen,
                        posit_0: atom_0.posit,
                        posit_1: atom_1.posit,
                    });
                }
            }
        }
    }

    result
}
