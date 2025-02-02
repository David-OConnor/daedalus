//! This module creates bonds between protein components. Most macromolecule PDB/CIF files don't include
//! explicit bond information, and the `pdbtbx` library doesn't handle this. Infer bond lengths
//! by comparing each interactomic bond distance, and matching against known amino acid bond lengths.
//!
//! Some info here: https://www.ruppweb.org/Xray/tutorial/protein_structure.htm
//! https://itp.uni-frankfurt.de/~engel/amino.html
//! These lengths are in angstrom.

use std::collections::HashMap;

use crate::{
    molecule::{Bond, BondType},
    Atom,
};

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
const LEN_O_H: f64 = 1.0;

// If interatomic distance is within this distance of one of our known bond lenghts, consider it to be a bond.
const BOND_LEN_THRESH: f64 = 0.04; // todo: Adjust A/R based on performance.
const GRID_SIZE: f64 = 3.0; // Slightly larger than the largest bond threshold

/// Infer bonds from atom distances. Uses spacial partitioning for efficiency.
/// We Check pairs only within nearby bins
pub fn create_bonds(atoms: &[Atom]) -> Vec<Bond> {
    let lens_covalent = vec![
        LEN_CP_N,
        LEN_N_CALPHA,
        LEN_CALPHA_CP,
        LEN_C_C,
        LEN_C_N,
        LEN_C_O,
        LEN_N_H,
        LEN_C_H,
        LEN_O_H,
    ];

    let lens_hydrogen = vec![LEN_OH_OH, LEN_NH_OC, LEN_OH_OC];

    // todo: Paralllize?
    let mut result = Vec::new();

    // We use spacial partitioning, so as not to copmare every pair of atoms.
    let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();

    for (i, atom) in atoms.iter().enumerate() {
        let grid_pos = (
            (atom.posit.x / GRID_SIZE).floor() as i32,
            (atom.posit.y / GRID_SIZE).floor() as i32,
            (atom.posit.z / GRID_SIZE).floor() as i32,
        );
        grid.entry(grid_pos).or_default().push(i);
    }

    let neighbor_offsets = [
        (0, 0, 0),
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
        (1, 1, 0),
        (-1, -1, 0),
        (1, 0, 1),
        (-1, 0, -1),
        (0, 1, 1),
        (0, -1, -1),
        (1, 1, 1),
        (-1, -1, -1),
    ];

    for (&cell, atom_indices) in &grid {
        for offset in &neighbor_offsets {
            let neighbor_cell = (cell.0 + offset.0, cell.1 + offset.1, cell.2 + offset.2);
            if let Some(neighbor_indices) = grid.get(&neighbor_cell) {
                for &i in atom_indices {
                    for &j in neighbor_indices {
                        if i >= j {
                            continue;
                        }

                        let atom_0 = &atoms[i];
                        let atom_1 = &atoms[j];
                        let dist = (atom_0.posit - atom_1.posit).magnitude();

                        for (lens, bond_type) in [
                            (&lens_covalent, BondType::Covalent),
                            (&lens_hydrogen, BondType::Hydrogen),
                        ] {
                            for &bond_len in lens {
                                if (dist - bond_len).abs() < BOND_LEN_THRESH {
                                    result.push(Bond {
                                        bond_type,
                                        posit_0: atom_0.posit,
                                        posit_1: atom_1.posit,
                                    });
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    result
}
