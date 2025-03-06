//! This module prepares molecules for docking, generating PDBQT files and equivalents for
//! targets and equivalent. Adds hydrogens and charges for targets, and specifies rotatable bonds
//! for ligands.
//!
//! See Meeko (Python package), Open Babel[Open Babel](http://openbabel.org) (GUI + CLI program), ADT etc for examples.
//!
//! [Meeko](https://meeko.readthedocs.io/en/release-doc/) use. Install: `pip install meeko`.
//!
//! Can use as a python library, or as a CLI application using scripts it includes.
//! Ligand prep: `mk_prepare_ligand.py -i molecule.sdf -o molecule.pdbqt`
//! Target prep: `mk_prepare_receptor.py -i nucleic_acid.cif -o my_receptor -j -p -f A:42`
//! Converting docking results back to PDB and SDF: `mk_export.py vina_results.pdbqt -j my_receptor.json -s lig_docked.sdf -p rec_docked.pdb`
//!
//!
//! MGLTools CLI flow:
//! `pythonsh prepare_receptor4.py -r myprotein.pdb -o myprotein.pdbqt -A checkhydrogens` (tgt) or
//! `pythonsh prepare_receptor.py -r myprotein.pdb -o myprotein.pdbqt` (tgt)
//! `pythonsh prepare_ligand4.py -l myligand.pdb -o myligand.pdbqt` (ligand)
//!
//! ADFR: https://ccsb.scripps.edu/adfr/downloads/ : another one. Having errors parsing OpenBabel's output.
//!
//! What we will use to start: the OpenBabel CLI program.

use std::fmt::Display;

use lin_alg::f64::Vec3;

use crate::{
    element::Element,
    molecule::{Atom, Bond, Molecule},
    util::setup_neighbor_pairs,
};

const GRID_SIZE: f64 = 1.6; // Slightly larger than the largest... todo: What?

#[derive(Debug)]
pub struct PartialCharge {
    pub posit: Vec3,
    pub charge: f32,
}

/// Create a set of partial charges around atoms. Rough simulation of electronic density imbalances
/// in charges molecules, and/or at short distances.
///
/// `charge_density` is a general standin. Higher means more charges; more accurate, and more computationally
/// intense.
fn create_partial_charges(atoms: &[Atom], charge_density: f32) -> Vec<PartialCharge> {
    let mut result = Vec::new();

    // to

    result
}

/// Used to determine if a gasteiger charge is a donar (bonded to at least one H), or accepter (not
/// bonded to any H).
fn bonded_to_h(bonds: &[Bond], atoms: &[Atom]) -> bool {
    for bond in bonds {
        let atom_0 = &atoms[bond.atom_0];
        let atom_1 = &atoms[bond.atom_1];
        if atom_0.element == Element::Hydrogen || atom_1.element == Element::Hydrogen {
            return true;
        }
    }
    false
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum DockType {
    // Standard AutoDock4/Vina atom types
    A,  // Aromatic carbon
    C,  // Aliphatic carbon
    N,  // Nitrogen
    Na, // Nitrogen (acceptor)
    // O,  // Oxygen
    Oh,
    Oa, // Oxygen (acceptor)
    S,  // Sulfur
    Sa, // Sulfur (acceptor)
    P,  // Phosphorus
    F,  // Fluorine
    Cl, // Chlorine
    K,
    Al,
    Pb,
    Au,
    Ag,
    Br, // Bromine
    I,  // Iodine
    Zn,
    Fe,
    Mg,
    Ca,
    Mn,
    Cu,
    Hd,    // Polar hydrogen (hydrogen donor)
    Other, // Fallback for unknown types
}

impl DockType {
    /// Simple guess
    /// For N if it is bonded to at least one hydrogen, treat it as a standard N
    /// (likely acting as a donor), otherwise mark it as NA (acceptor).
    /// For an oxygen atom, if it’s bound to a hydrogen (for example, in a hydroxyl group), you might
    /// assign it as Oh; if not, then as Oa.
    pub fn infer(atom: &Atom, bonds: &[Bond], atoms: &[Atom]) -> Self {
        match atom.element {
            Element::Carbon => {
                // todo:  Hande Aromatic case A/R.
                Self::C
            }
            Element::Nitrogen => {
                if bonded_to_h(bonds, atoms) {
                    Self::N
                } else {
                    Self::Na
                }
            }
            Element::Oxygen => {
                if bonded_to_h(bonds, atoms) {
                    Self::Oh
                } else {
                    Self::Oa
                }
                // todo: When to make it Oh?
            }
            Element::Sulfur => {
                if bonded_to_h(bonds, atoms) {
                    Self::S
                } else {
                    Self::Sa
                }
            }
            Element::Phosphorus => Self::P,
            Element::Fluorine => Self::F,
            Element::Chlorine => Self::Cl,
            Element::Bromine => Self::Br,
            Element::Iodine => Self::I,
            Element::Zinc => Self::Zn,
            Element::Iron => Self::Fe,
            Element::Magnesium => Self::Mg,
            Element::Calcium => Self::Ca,
            Element::Potassium => Self::Other,
            Element::Aluminum => Self::Other,
            Element::Mercury => Self::Other,   // todo
            Element::Tin => Self::Other,       // todo
            Element::Tungsten => Self::Other,  // todo
            Element::Tellurium => Self::Other, // todo
            Element::Selenium => Self::Other,  // todo
            Element::Lead => Self::Other,
            Element::Gold => Self::Other,
            Element::Silver => Self::Other,
            Element::Manganese => Self::Mn,
            Element::Copper => Self::Cu,
            Element::Hydrogen => Self::Hd,
            Element::Other => Self::Other,
        }
    }

    pub fn from_str(s: &str) -> Self {
        let s = s.to_uppercase();
        match s.as_ref() {
            "A" => Self::A,
            "C" => Self::C,
            "N" => Self::N,
            "NA" => Self::Na,
            // "O" => Self::O,
            "OA" => Self::Oa,
            "OH" => Self::Oh,
            "S" => Self::S,
            "SA" => Self::Sa,
            "P" => Self::P,
            "F" => Self::F,
            "CL" => Self::Cl,
            "K" => Self::K,
            "AL" => Self::Al,
            "PB" => Self::Pb,
            "AU" => Self::Au,
            "AG" => Self::Ag,
            "BR" => Self::Br,
            "I" => Self::I,
            "ZN" => Self::Zn,
            "FE" => Self::Fe,
            "MG" => Self::Mg,
            "CA" => Self::Ca,
            "MN" => Self::Mn,
            "CU" => Self::Cu,
            "HD" => Self::Hd,
            _ => {
                if s.starts_with("C") {
                    Self::C
                } else if s.starts_with("N") {
                    Self::N
                } else if s.starts_with("O") {
                    Self::Oh
                } else if s.starts_with("S") {
                    Self::S
                } else {
                    eprintln!("Unknown dock type: {}", s);
                    Self::Other
                }
            }
        }
    }

    pub fn to_str(&self) -> String {
        match self {
            Self::A => "A",
            Self::C => "C",
            Self::N => "N",
            Self::Na => "NA",
            // Self::O => "O",
            Self::Oa => "OA",
            Self::Oh => "O",
            Self::S => "S",
            Self::Sa => "SA",
            Self::P => "P",
            Self::F => "F",
            Self::Cl => "CL",
            Self::K => "K",
            Self::Al => "AL",
            Self::Pb => "PB",
            Self::Au => "AU",
            Self::Ag => "AG",
            Self::Br => "BR",
            Self::I => "I",
            Self::Zn => "", // Appears to be the correct response. Likely for other types here too.
            Self::Fe => "FE",
            Self::Mg => "MG",
            Self::Ca => "CA",
            Self::Mn => "MN",
            Self::Cu => "CU",
            Self::Hd => "HD",
            Self::Other => "--",
        }
        .to_string()
    }

    pub fn gasteiger_electronegativity(&self) -> f32 {
        match self {
            Self::A => 2.50,
            Self::C => 2.55,
            Self::N => 3.04,
            Self::Na => 3.10,
            // Self::O => 3.44,
            Self::Oh => 3.44,
            Self::Oa => 3.50,
            Self::S => 2.50,
            Self::Sa => 2.60,
            Self::P => 2.19,
            Self::F => 3.98,
            Self::Cl => 3.16,
            // todo: These 5: Need to look up.
            Self::K => 2.84,
            Self::Al => 2.65,
            Self::Pb => 2.10,
            Self::Au => 2.10,
            Self::Ag => 2.10,
            Self::Br => 2.96,
            Self::I => 2.66,
            Self::Zn => 1.65,
            Self::Fe => 1.83,
            Self::Mg => 1.31,
            Self::Ca => 1.00,
            Self::Mn => 1.55,
            Self::Cu => 1.90,
            Self::Hd => 2.20,
            Self::Other => 2.50, // Fallback default value
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TorsionStatus {
    Active,
    Inactive,
}

impl Display for TorsionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match self {
            Self::Active => "A".to_string(),
            Self::Inactive => "I".to_string(),
        };
        write!(f, "{}", str)
    }
}

#[derive(Debug, Default)]
pub struct UnitCellDims {
    /// Lengths in Angstroms.
    pub a: f32,
    pub b: f32,
    pub c: f32,
    /// Angles in degrees.
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,
}

#[derive(Debug)]
pub struct Torsion {
    // todo: Initial hack; add/fix A/R.
    pub status: TorsionStatus,
    pub atom_0: String, // todo: Something else, like index.
    pub atom_1: String, // todo: Something else, like index.
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PartialChargeType {
    Gasteiger,
    Kollman,
}

/// Note: Hydrogens must already be added prior to adding charges.
pub fn setup_partial_charges(atoms: &mut [Atom], charge_type: PartialChargeType) {
    if charge_type == PartialChargeType::Kollman {
        unimplemented!()
    }

    // We use spacial partitioning, so as not to copmare every pair of atoms.
    let posits: Vec<_> = atoms.iter().map(|a| &a.posit).collect();
    let indices: Vec<_> = (0..atoms.len()).collect();
    let neighbor_pairs = setup_neighbor_pairs(&posits, &indices, GRID_SIZE);

    // Run the iterative charge update over all candidate pairs.
    const ITERATIONS: usize = 6; // More iterations may be needed in practice.
    for _ in 0..ITERATIONS {
        let mut charge_updates = vec![0.0; atoms.len()];

        for &(i, j) in &neighbor_pairs {
            if atoms[i].dock_type.is_none() || atoms[j].dock_type.is_none() {
                continue;
            }
            let en_i = atoms[i].dock_type.unwrap().gasteiger_electronegativity();
            let en_j = atoms[j].dock_type.unwrap().gasteiger_electronegativity();
            // Compute a simple difference-based transfer.
            let delta = 0.1 * (en_i - en_j);
            // Transfer charge from atom i to atom j if en_i > en_j.
            charge_updates[i] -= delta;
            charge_updates[j] += delta;
        }

        // Apply the computed updates simultaneously.
        for (atom, delta) in atoms.iter_mut().zip(charge_updates.iter()) {
            match &mut atom.partial_charge {
                Some(c) => *c += delta,
                None => atom.partial_charge = Some(*delta),
            }
        }
    }
}

impl Molecule {
    /// Adds hydrogens, assigns partial charges etc.
    pub fn prep_target(&mut self) {}

    /// Indicates which atoms and bonds are flexible, etc.
    pub fn prep_ligand(&mut self) {}
}

/// todo: Output string if it's a text-based format.
pub fn export_pdbqt(target: &Molecule, ligand: &Molecule) -> Vec<u8> {
    Vec::new()
}

// // todo: C+P from ChatGPT. I'm low-confidence on it
// /// This helper function infers the DockType of an atom based on its element
// /// and its bonding environment. The `atoms` slice is used to examine neighbors.
// pub fn infer_dock_type(atom: &Atom, atoms: &[Atom]) -> DockType {
//     match atom.element {
//         Element::Nitrogen => {
//             // If any neighbor is hydrogen, assume it’s a donor (N);
//             // otherwise, treat it as an acceptor (NA).
//             let has_hydrogen = atom.neighbors.iter()
//                 .any(|&i| atoms[i].element == Element::Hydrogen);
//             if has_hydrogen {
//                 DockType::N
//             } else {
//                 DockType::Na
//             }
//         }
//         Element::Oxygen => {
//             // If oxygen is bonded to a hydrogen, assign it as hydroxyl (Oh);
//             // otherwise, assign it as acceptor (Oa).
//             let has_hydrogen = atom.neighbors.iter()
//                 .any(|&i| atoms[i].element == Element::Hydrogen);
//             if has_hydrogen {
//                 DockType::Oh
//             } else {
//                 DockType::Oa
//             }
//         }
//         Element::Carbon => {
//             // Here you could add aromaticity detection.
//             // For now, we assume a default of aliphatic carbon.
//             DockType::C
//         }
//         Element::Hydrogen => {
//             // Polar hydrogen donors are typically assigned Hd.
//             DockType::Hd
//         }
//         _ => DockType::Other,
//     }
// }
