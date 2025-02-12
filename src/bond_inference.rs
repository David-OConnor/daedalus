//! This module creates bonds between protein components. Most macromolecule PDB/CIF files don't include
//! explicit bond information, and the `pdbtbx` library doesn't handle this. Infer bond lengths
//! by comparing each interactomic bond distance, and matching against known amino acid bond lengths.
//!
//! Some info here: https://www.ruppweb.org/Xray/tutorial/protein_structure.htm
//! https://itp.uni-frankfurt.de/~engel/amino.html
//!
//! All lengths are in angstrom.

use std::collections::HashMap;

use crate::{
    molecule::{
        Atom, Bond,
        BondCount::{self, *},
        BondType::{self, *},
    },
    Element,
    Element::{Carbon, Hydrogen, Nitrogen, Oxygen, Sulfur},
};

struct BondSpecs {
    len: f64,
    elements: (Element, Element),
    count: BondCount,
    bond_type: BondType,
}

impl BondSpecs {
    pub fn new(
        len: f64,
        elements: (Element, Element),
        count: BondCount,
        bond_type: BondType,
    ) -> Self {
        Self {
            len,
            elements,
            count,
            bond_type,
        }
    }
}

// If interatomic distance is within this distance of one of our known bond lenghts, consider it to be a bond.
const BOND_LEN_THRESH: f64 = 0.04; // todo: Adjust A/R based on performance.
const GRID_SIZE: f64 = 1.6; // Slightly larger than the largest bond distancen + thresh.

#[rustfmt::skip]
fn get_specs() -> Vec<BondSpecs> {
    vec![
        // --------------------
        // Carbon–Carbon Bonds
        // --------------------

        // C–C single bond
        // The most frequently encountered bond length for saturated, sp³-hybridized carbons (e.g., in alkanes).
        BondSpecs::new(1.54, (Carbon, Carbon), Single, Covalent),

        // Cα–C′: ~1.50 - 1.52 Å
        BondSpecs::new(1.51, (Carbon, Carbon), Single, Covalent),

        // C–C sp²–sp³ single bond, e.g. connecting Phe's ring to the rest of the atom.
        BondSpecs::new(1.50, (Carbon, Carbon), SingleDoubleHybrid, Covalent),

        // C-C phenyl (aromatic) ring bond, or benzene ring.
        // Found in alkynes, where carbons are sp-hybridized (linear). ~1.37-1.40 Å
        BondSpecs::new(1.39, (Carbon, Carbon), SingleDoubleHybrid, Covalent),

        // C-C Seems to be required for one fo the Trp rings?
        BondSpecs::new(1.36, (Carbon, Carbon), SingleDoubleHybrid, Covalent),

        // C=C double bond
        // Common in alkenes (sp²-hybridized). Range: ~1.33–1.34 Å
        BondSpecs::new(1.33, (Carbon, Carbon), Double, Covalent),

        // C≡C triple bond
        // Found in alkynes, where carbons are sp-hybridized (linear). ~1.20 Å
        BondSpecs::new(1.20, (Carbon, Carbon), Triple, Covalent),

        // --------------------
        // Carbon–Nitrogen Bonds
        // --------------------

        // C–N single bond
        // Typical for amines or alkyl–amine bonds. ~1.45-1.47 Å
        // Also covers Amide Nitrogen to C-alpha bond in protein backbones.
        BondSpecs::new(1.46, (Carbon, Nitrogen), Single, Covalent),

        // C-N Indole N in 5-member aromatic ring, e.g. Trp. 1.36-1.39
        BondSpecs::new(1.37, (Carbon, Nitrogen), SingleDoubleHybrid, Covalent),

        // C-N (amide). Partial double-bond character due to resonance in the amide.
        BondSpecs::new(1.33, (Carbon, Nitrogen), SingleDoubleHybrid, Covalent),

        // C=N double bond
        // Typical for imines (Schiff bases). ~1.28 Å
        BondSpecs::new(1.28, (Carbon, Nitrogen), Double, Covalent),

        // C≡N triple bond
        // Typical of nitriles (–C≡N). ~1.16 Å
        BondSpecs::new(1.16, (Carbon, Nitrogen), Triple, Covalent),
        // NOTE:
        // In proteins, the amide (peptide) bond between C=O and N has partial double-bond character,
        // and the C–N bond length in an amide is around 1.32–1.33 Å.

        // --------------------
        // Carbon–Oxygen Bonds
        // --------------------

        // C–O single bond
        // Found in alcohols, ethers (sp³–O). ~1.43 Å
        BondSpecs::new(1.43, (Carbon, Oxygen), Single, Covalent),

        // C(phenyl)–O. Phenolic C–O bond often shorter than a typical aliphatic C–O. 1.36-1.38 Å
        BondSpecs::new(1.37, (Carbon, Oxygen), Single, Covalent),

        // C′–O (in –COO⁻). 1.25-1.27 Å
        BondSpecs::new(1.26, (Carbon, Oxygen), Single, Covalent),

        // C=O double bond
        // Typical for carbonyl groups (aldehydes, ketones, carboxylic acids, amides). ~1.21–1.23 Å
        BondSpecs::new(1.22, (Carbon, Oxygen), Double, Covalent),

        // --------------------
        // Carbon–Hydrogen Bonds
        // --------------------
        // todo: Expand this section.

        BondSpecs::new(1.09, (Carbon, Hydrogen), Single, Covalent),

        // 1.01–1.02 Å
        BondSpecs::new(1.01, (Nitrogen, Hydrogen), Single, Covalent),

        // 0.96 – 0.98 Å
        BondSpecs::new(1.01, (Oxygen, Hydrogen), Single, Covalent),


        // Non-protein-backbond bond lengths.

        // 1.34 - 1.35. Example: Cys.
        BondSpecs::new(1.34, (Sulfur, Hydrogen), Single, Covalent),

        // 1.81 - 1.82. Example: Cys.
        BondSpecs::new(1.81, (Sulfur, Carbon), Single, Covalent),
    ]
}

/// This is the business logic of evaluating bond lengths. For a single atom pair.
fn eval_lens(bonds: &mut Vec<Bond>, atoms: &[Atom], i: usize, j: usize, specs: &[BondSpecs]) {
    let atom_0 = &atoms[i];
    let atom_1 = &atoms[j];

    let dist = (atom_0.posit - atom_1.posit).magnitude();

    for spec in specs {
        // This directionality ensures only one bond per atom pair. Otherwise, we'd add two identical
        // ones with swapped atom positions.
        if !(atom_0.element == spec.elements.0 && atom_1.element == spec.elements.1) {
            continue;
        }

        if (dist - spec.len).abs() < BOND_LEN_THRESH {
            bonds.push(Bond {
                bond_type: spec.bond_type,
                bond_count: spec.count,
                atom_0: i,
                atom_1: j,
                is_backbone: atom_0.is_backbone() && atom_1.is_backbone(),
            });
            break;
        }
    }
}

/// Infer bonds from atom distances. Uses spacial partitioning for efficiency.
/// We Check pairs only within nearby bins.
pub fn create_bonds(atoms: &[Atom]) -> Vec<Bond> {
    // todo: Paralllize?

    let mut result = Vec::new();

    let specs = get_specs();

    // for (i, atom) in atoms.iter().enumerate() {
    //     for (j, atom) in atoms.iter().enumerate() {
    //         if i == j {
    //             continue
    //         }
    //         eval_lens(&mut result, atoms, i, j, &specs);
    //     }
    // }

    // todo: Something with your cell code is broken! Get it working. For now, we're skipping it.
    // return result;

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

    for (&cell, atom_indices) in &grid {
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let neighbor_cell = (cell.0 + dx, cell.1 + dy, cell.2 + dz);
                    if let Some(neighbor_indices) = grid.get(&neighbor_cell) {
                        for &i in atom_indices {
                            for &j in neighbor_indices {
                                if i >= j {  // or i < j, depending on how you skip duplicates
                                    continue;
                                }
                                eval_lens(&mut result, atoms, i, j, &specs);
                            }
                        }
                    }
                }
            }
        }
    }

    result
}
