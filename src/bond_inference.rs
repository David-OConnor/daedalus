//! This module creates bonds between protein components. Most macromolecule PDB/CIF files don't include
//! explicit bond information, and the `pdbtbx` library doesn't handle this. Infer bond lengths
//! by comparing each interactomic bond distance, and matching against known amino acid bond lengths.
//!
//! Some info here: https://www.ruppweb.org/Xray/tutorial/protein_structure.htm
//! https://itp.uni-frankfurt.de/~engel/amino.html
//!
//! All lengths are in angstrom (Å)

use std::f64::consts::TAU;

use na_seq::{
    Element,
    Element::{Carbon, Fluorine, Hydrogen, Nitrogen, Oxygen, Sulfur},
};
use rayon::prelude::*;

use crate::{
    molecule::{
        Atom, Bond,
        BondCount::*,
        BondType::{self, *},
        HydrogenBond,
    },
    util::{find_atom, setup_neighbor_pairs},
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
const COV_BOND_LEN_THRESH: f64 = 0.04; // todo: Adjust A/R based on performannce.
const COV_DIST_GRID: f64 = 1.6; // Slightly larger than the largest bond distance + thresh.

// Note: Chimera shows H bonds as ranging generally from 2.8 to 3.3.
// Note: These values all depend on which is the donor. Your code doesn't take this into account.
const H_BOND_O_O_DIST: f64 = 2.7;
const H_BOND_N_N_DIST: f64 = 3.05;
const H_BOND_O_N_DIST: f64 = 2.9;

const H_BOND_N_F_DIST: f64 = 2.75;
const H_BOND_N_S_DIST: f64 = 3.35;
const H_BOND_S_O_DIST: f64 = 3.35;
const H_BOND_S_F_DIST: f64 = 3.2; // rarate

const H_BOND_DIST_THRESH: f64 = 0.3;
const H_BOND_DIST_GRID: f64 = 3.6;

const H_BOND_ANGLE_THRESH: f64 = TAU / 3.;

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
    let specs = get_specs();

    // We use spacial partitioning, so as not to copmare every pair of atoms.
    let posits: Vec<_> = atoms.iter().map(|a| &a.posit).collect();
    // Indices are all values here.
    let indices: Vec<_> = (0..posits.len()).collect();
    let neighbor_pairs = setup_neighbor_pairs(&posits, &indices, COV_DIST_GRID);

    // todo: Should we create an Vec of neighbors for each atom. (Maybe storeed in a hashmap etc)
    // todo, then iterate over that for neighbors in the j loop? WOuld be more generalizable/extract
    // todo it out from the bus logic.

    neighbor_pairs
        .par_iter()
        .filter_map(|(i, j)| {
            let atom_0 = &atoms[*i];
            let atom_1 = &atoms[*j];
            let dist = (atom_0.posit - atom_1.posit).magnitude();

            specs.iter().find_map(|spec| {
                let matches_elements = (atom_0.element == spec.elements.0
                    && atom_1.element == spec.elements.1)
                    || (atom_0.element == spec.elements.1 && atom_1.element == spec.elements.0);

                // If both the element match and distance-threshold check pass,
                // we create a Bond and stop searching any further specs.
                if matches_elements && (dist - spec.len).abs() < COV_BOND_LEN_THRESH {
                    Some(Bond {
                        bond_type: spec.bond_type,
                        atom_0: *i,
                        atom_1: *j,
                        is_backbone: atom_0.is_backbone() && atom_1.is_backbone(),
                    })
                } else {
                    None
                }
            })
        })
        .collect()
}

/// Helper
fn h_bond_candidate_el(atom: &Atom) -> bool {
    matches!(atom.element, Nitrogen | Oxygen | Sulfur | Fluorine)
}

fn hydrogen_bond_inner(
    bonds: &mut Vec<HydrogenBond>,
    donor_heavy: &Atom,
    donor_h: &Atom,
    acc_candidate: &Atom,
    donor_heavy_i: usize,
    donor_h_i: usize,
    acc_i: usize,
    relaxed_dist_thresh: bool,
) {
    let d_e = donor_heavy.element; // Cleans up the verbose code below.
    let a_e = acc_candidate.element;
    // todo: Take into account typical lenghs of donor and receptor; here your order isn't used.
    let dist_thresh = if d_e == Oxygen && a_e == Oxygen {
        H_BOND_O_O_DIST
    } else if d_e == Nitrogen && a_e == Nitrogen {
        H_BOND_N_N_DIST
    } else if (d_e == Oxygen && a_e == Nitrogen) || (d_e == Nitrogen && a_e == Oxygen) {
        H_BOND_O_N_DIST
    } else if (d_e == Fluorine && a_e == Nitrogen) || (d_e == Nitrogen && a_e == Fluorine) {
        H_BOND_N_F_DIST
    } else {
        H_BOND_N_S_DIST // Good enough for other combos involving S and F, for now.
    };

    let modifier = if relaxed_dist_thresh {
        H_BOND_DIST_THRESH * 2.
    } else {
        H_BOND_DIST_THRESH
    };

    let dist_thresh_min = dist_thresh - modifier;
    let dist_thresh_max = dist_thresh + modifier;

    let dist = (acc_candidate.posit - donor_heavy.posit).magnitude();
    if dist < dist_thresh_min || dist > dist_thresh_max {
        return;
    }

    let angle = {
        let donor_h = donor_h.posit - donor_heavy.posit;
        let donor_acceptor = donor_heavy.posit - acc_candidate.posit;

        donor_acceptor
            .to_normalized()
            .dot(donor_h.to_normalized())
            .acos()
    };

    if angle > H_BOND_ANGLE_THRESH {
        bonds.push(HydrogenBond {
            donor: donor_heavy_i,
            acceptor: acc_i,
            hydrogen: donor_h_i,
        });
    }
}

/// Create hydrogen bonds between all atomsm in a group. See `create_hydrogen_bonds_one_way` for the more
/// flexible fn it calls.
pub fn create_hydrogen_bonds(atoms: &[Atom], bonds: &[Bond]) -> Vec<HydrogenBond> {
    let indices: Vec<_> = (0..atoms.len()).collect();
    create_hydrogen_bonds_one_way(atoms, &indices, bonds, atoms, &indices, false)
}

/// Infer hydrogen bonds from a list of atoms. This takes into account bond distance between suitable
/// atom types (generally N and O; sometimes S and F), and geometry regarding the hydrogen covalently
/// bonded to the donor.
///
/// Separates donor from acceptor inputs, for use in cases like bonds between targets and ligands.
/// We indlude indices, in the case where atoms are subsets of molecules; this allows bonds indices
/// to be preserved.
pub fn create_hydrogen_bonds_one_way(
    atoms_donor: &[Atom],
    atoms_donor_i: &[usize],
    bonds_donor: &[Bond],
    atoms_acc: &[Atom],
    atoms_acc_i: &[usize],
    relaxed_dist_thresh: bool,
) -> Vec<HydrogenBond> {
    let mut result = Vec::new();

    // Bonds between donor and H.
    let potential_donor_bonds: Vec<&Bond> = bonds_donor
        .iter()
        .filter(|b| {
            // Maps from a global index, to a local atom from a subset.
            let atom_0 = find_atom(atoms_donor, atoms_donor_i, b.atom_0);
            let atom_1 = find_atom(atoms_donor, atoms_donor_i, b.atom_1);

            let (Some(atom_0), Some(atom_1)) = (atom_0, atom_1) else {
                eprintln!(
                    "Error! Can't find atoms from indices when making H bonds (Donor finding)"
                );
                return false;
            };

            let cfg_0_valid = h_bond_candidate_el(atom_0) && atom_1.element == Hydrogen;
            let cfg_1_valid = h_bond_candidate_el(atom_1) && atom_0.element == Hydrogen;

            cfg_0_valid || cfg_1_valid
        })
        .collect();

    let potential_acceptors: Vec<(usize, &Atom)> = atoms_acc
        .iter()
        .enumerate()
        .filter(|(i, a)| h_bond_candidate_el(a))
        .map(|(i, a)| (atoms_acc_i[i], a))
        .collect();

    for donor_bond in potential_donor_bonds {
        let donor_0 = find_atom(atoms_donor, atoms_donor_i, donor_bond.atom_0);
        let donor_1 = find_atom(atoms_donor, atoms_donor_i, donor_bond.atom_1);

        let (Some(donor_0), Some(donor_1)) = (donor_0, donor_1) else {
            eprintln!("Error! Can't find atoms from indices when making H bonds");
            continue;
        };

        let (donor_heavy, donor_h, donor_heavy_i, donor_h_i) = if donor_0.element == Hydrogen {
            (donor_1, donor_0, donor_bond.atom_1, donor_bond.atom_0)
        } else {
            (donor_0, donor_1, donor_bond.atom_0, donor_bond.atom_1)
        };

        for (acc_i, acc_candidate) in &potential_acceptors {
            hydrogen_bond_inner(
                &mut result,
                donor_heavy,
                donor_h,
                acc_candidate,
                donor_heavy_i,
                donor_h_i,
                *acc_i,
                relaxed_dist_thresh,
            );
        }
    }

    result
}
