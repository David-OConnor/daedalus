//! This module creates bonds between protein components. Most macromolecule PDB/CIF files don't include
//! explicit bond information, and the `pdbtbx` library doesn't handle this. Infer bond lengths
//! by comparing each interactomic bond distance, and matching against known amino acid bond lengths.
//!
//! Some info here: https://www.ruppweb.org/Xray/tutorial/protein_structure.htm
//! https://itp.uni-frankfurt.de/~engel/amino.html
//!
//! All lengths are in angstrom.

use crate::{
    element::{
        Element,
        Element::{Carbon, Hydrogen, Nitrogen, Oxygen, Sulfur},
    },
    molecule::{
        Atom, Bond,
        BondCount::*,
        BondType::{self, *},
    },
    util::setup_neighbor_pairs,
};

struct BondSpecs {
    len: f64,
    elements: (Element, Element),
    bond_type: BondType,
}

impl BondSpecs {
    pub fn new(len: f64, elements: (Element, Element), bond_type: BondType) -> Self {
        Self {
            len,
            elements,
            bond_type,
        }
    }
}

// If interatomic distance is within this distance of one of our known bond lenghts, consider it to be a bond.
// Relevant to this is both bond variability under various conditions, and measurement precision.
const COV_BOND_LEN_THRESH: f64 = 0.04; // todo: Adjust A/R based on performance.
const COV_DIST_GRID: f64 = 1.6; // Slightly larger than the largest bond distance + thresh.

const H_BOND_DIST_THRESH: f64 = 3.5; // Angstrom.
const H_BOND_DIST_GRID: f64 = 3.6; // Angstrom.

#[rustfmt::skip]
fn get_specs() -> Vec<BondSpecs> {
    // Code shorteners
    let single = Covalent { count: Single };
    let hybrid = Covalent { count: SingleDoubleHybrid };
    let double = Covalent { count: Double };
    let triple = Covalent { count: Triple };

    vec![
        // --------------------
        // Carbon–Carbon Bonds
        // --------------------

        // C–C single bond
        // The most frequently encountered bond length for saturated, sp³-hybridized carbons (e.g., in alkanes).
        BondSpecs::new(1.54, (Carbon, Carbon), single),

        // Cα–C′: ~1.50 - 1.52 Å
        BondSpecs::new(1.51, (Carbon, Carbon), single),

        // C–C sp²–sp³ single bond, e.g. connecting Phe's ring to the rest of the atom.
        BondSpecs::new(1.50, (Carbon, Carbon), hybrid),

        // Workaround for Phe's ring in some cases.
        BondSpecs::new(1.47, (Carbon, Carbon), hybrid),
        BondSpecs::new(1.44, (Carbon, Carbon), hybrid),
        BondSpecs::new(1.41, (Carbon, Carbon), hybrid),

        // C-C phenyl (aromatic) ring bond, or benzene ring.
        // Found in alkynes, where carbons are sp-hybridized (linear). ~1.37-1.40 Å
        BondSpecs::new(1.39, (Carbon, Carbon), hybrid),

        // C-C Seems to be required for one fo the Trp rings?
        BondSpecs::new(1.36, (Carbon, Carbon), hybrid),

        // C=C double bond
        // Common in alkenes (sp²-hybridized). Range: ~1.33–1.34 Å
        BondSpecs::new(1.33, (Carbon, Carbon), double),

        // C≡C triple bond
        // Found in alkynes, where carbons are sp-hybridized (linear). ~1.20 Å
        BondSpecs::new(1.20, (Carbon, Carbon), triple),

        // --------------------
        // Carbon–Nitrogen Bonds
        // --------------------

        // C–N single bond
        // Typical for amines or alkyl–amine bonds. ~1.45-1.47 Å
        // Also covers Amide Nitrogen to C-alpha bond in protein backbones.
        BondSpecs::new(1.46, (Carbon, Nitrogen), single),

        // C-N Indole N in 5-member aromatic ring, e.g. Trp. 1.36-1.39
        // BondSpecs::new(1.37, (Carbon, Nitrogen), type_hybrid),
        BondSpecs::new(1.37, (Carbon, Nitrogen), single),

        // todo: Some adjustments here may be required regarding single vs hybrid N-C bonds.

        // C-N (amide). Partial double-bond character due to resonance in the amide.
        // BondSpecs::new(1.33, (Carbon, Nitrogen), type_hybrid),
        BondSpecs::new(1.33, (Carbon, Nitrogen), single),

        // C=N double bond
        // Typical for imines (Schiff bases). ~1.28 Å
        BondSpecs::new(1.28, (Carbon, Nitrogen), double),

        // C≡N triple bond
        // Typical of nitriles (–C≡N). ~1.16 Å
        BondSpecs::new(1.16, (Carbon, Nitrogen), triple),
        // NOTE:
        // In proteins, the amide (peptide) bond between C=O and N has partial double-bond character,
        // and the C–N bond length in an amide is around 1.32–1.33 Å.

        // --------------------
        // Carbon–Oxygen Bonds
        // --------------------

        // C–O single bond
        // Found in alcohols, ethers (sp³–O). ~1.43 Å
        BondSpecs::new(1.43, (Carbon, Oxygen), single),

        // C(phenyl)–O. Phenolic C–O bond often shorter than a typical aliphatic C–O. 1.36-1.38 Å
        BondSpecs::new(1.37, (Carbon, Oxygen), single),

        // C′–O (in –COO⁻). 1.25-1.27 Å
        // BondSpecs::new(1.26, (Carbon, Oxygen), type_singl),
        BondSpecs::new(1.26, (Carbon, Oxygen), double),

        // C=O double bond
        // Typical for carbonyl groups (aldehydes, ketones, carboxylic acids, amides). ~1.21–1.23 Å
        BondSpecs::new(1.22, (Carbon, Oxygen), double),

        // --------------------
        // Carbon–Hydrogen Bonds
        // --------------------
        // todo: Expand this section.

        BondSpecs::new(1.09, (Hydrogen, Carbon), single),

        // 1.01–1.02 Å
        BondSpecs::new(1.01, (Hydrogen, Nitrogen), single),

        // 0.96 – 0.98 Å
        BondSpecs::new(1.01, (Hydrogen, Oxygen), single),
        // BondSpecs::new(1.01, (Hydrogen, Oxygen), single),
        BondSpecs::new(0.95, (Hydrogen, Oxygen), single),


        // Non-protein-backbond bond lengths.

        // 1.34 - 1.35. Example: Cys.
        BondSpecs::new(1.34, (Sulfur, Hydrogen), single),

        // 1.81 - 1.82. Example: Cys.
        BondSpecs::new(1.81, (Sulfur, Carbon), single),
    ]
}

/// Infer bonds from atom distances. Uses spacial partitioning for efficiency.
/// We Check pairs only within nearby bins.
pub fn create_bonds(atoms: &[Atom]) -> Vec<Bond> {
    // todo: Paralllize?
    println!("Starting bond creation...");

    let mut result = Vec::new();

    let specs = get_specs();

    // We use spacial partitioning, so as not to copmare every pair of atoms.
    let posits: Vec<_> = atoms.iter().map(|a| &a.posit).collect();
    let neighbor_pairs = setup_neighbor_pairs(&posits, COV_DIST_GRID);

    // todo: Should we create an Vec of neighbors for each atom. (Maybe storeed in a hashmap etc)
    // todo, then iterate over that for neighbors in the j loop? WOuld be more generalizable/extract
    // todo it out from the bus logic.

    for &(i, j) in &neighbor_pairs {
        let atom_0 = &atoms[i];
        let atom_1 = &atoms[j];

        let dist = (atom_0.posit - atom_1.posit).magnitude();

        for spec in &specs {
            if !((atom_0.element == spec.elements.0 && atom_1.element == spec.elements.1)
                || (atom_0.element == spec.elements.1 && atom_1.element == spec.elements.0))
            {
                continue;
            }

            if (dist - spec.len).abs() < COV_BOND_LEN_THRESH {
                result.push(Bond {
                    bond_type: spec.bond_type,
                    atom_0: i,
                    atom_1: j,
                    is_backbone: atom_0.is_backbone() && atom_1.is_backbone(),
                });
                break;
            }
        }
    }

    println!("Bond creation complete.");

    result
}

/// Infer hydrogen bonds from a list of atoms.
/// This simple implementation iterates over all pairs of atoms.
/// It considers only atoms that are candidates (N or O)
/// and adds a bond if their distance is less than the cutoff.
/// todo: UPdate this approach to be more robust. Use bond agnles etc. And maybe integrate
/// todo with populated h ydrogens.
pub fn create_hydrogen_bonds(atoms: &[Atom]) -> Vec<Bond> {
    let mut result = Vec::new();

    let posits: Vec<_> = atoms.iter().map(|a| &a.posit).collect();
    let neighbor_pairs = setup_neighbor_pairs(&posits, H_BOND_DIST_GRID);

    // todo: This approach is very crude.

    for &(i, j) in &neighbor_pairs {
        let atom0 = &atoms[i];
        let atom1 = &atoms[j];

        // todo: QC this logic with rules of H bonding.
        if !matches!(atom0.element, Nitrogen | Oxygen)
            || !matches!(atom1.element, Nitrogen | Oxygen)
        {
            continue;
        }

        if (atom0.posit - atom1.posit).magnitude() < H_BOND_DIST_THRESH {
            result.push(Bond {
                bond_type: BondType::Hydrogen,
                // todo: Set it up so atom_0 is always the donor!
                atom_0: i,
                atom_1: j,
                is_backbone: false,
            });
        }
    }

    result
}
