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

use std::{collections::HashMap, fmt::Display};

use barnes_hut::{BhConfig, Cube, Tree};
use graphics::Mesh;

use crate::{
    docking::{
        DockingSite, partial_charge::PartialCharge, setup_eem_charges,
    },
    element::Element,
    molecule::{Atom, Bond, BondCount, BondType, Ligand, Molecule},
};
use crate::docking::ATOM_NEAR_SITE_DIST_THRESH;

/// This can be used across multiple poses, but is specific to a receptor, ligand, and docking site.
pub struct DockingSetup {
    pub rec_atoms_near_site: Vec<Atom>,
    pub rec_indices: Vec<usize>,
    /// We omit partial ligand charges, since these include position.
    pub charges_rec: Vec<PartialCharge>,
    pub rec_bonds_near_site: Vec<Bond>,
    pub lj_pairs: Vec<(usize, usize, f32, f32)>,
    pub charge_tree: Tree,
    pub bh_config: BhConfig,
}

impl DockingSetup {
    pub fn new(
        receptor: &Molecule,
        ligand: &mut Ligand,
        lj_lut: &HashMap<(Element, Element), (f32, f32)>,
        bh_config: &BhConfig,
    ) -> Self {
        let (mut rec_atoms_near_site, rec_indices) =
            find_rec_atoms_near_site(receptor, &ligand.docking_site);

        // Bonds here is used for identifying donor heavy and H pairs for hydrogen bonds.
        let rec_bonds_near_site: Vec<_> = receptor
            .bonds
            .iter()
            // Don't use ||; all atom indices in these bonds must be present in `tgt_atoms_near_site`.
            .filter(|b| {
                rec_indices.contains(&b.atom_0) && rec_indices.contains(&b.atom_1)
            })
            .map(|b| b.clone()) // todo: don't like the clone
            .collect();

        let (partial_charges_rec, lj_pairs) = setup_eem_charges(
            receptor,
            ligand,
            &mut rec_atoms_near_site,
            &rec_indices,
            lj_lut,
        );

        // This tree is over the target (receptor) charges. This may be more efficient
        // than over the ligand, as we expect the receptor nearby atoms to be more numerous.
        // We can set up our tree once for the whole simulation, as for the case of fixed
        // receptor, one side of the charge pairs doesn't change.
        let charge_tree = {
            // For the Barnes Hut electrostatics tree.
            let bh_bounding_box = Cube::from_bodies(&partial_charges_rec, 0., true);

            Tree::new(&partial_charges_rec, &bh_bounding_box.unwrap(), &bh_config)
        };

        Self {
            rec_atoms_near_site,
            rec_indices,
            charges_rec: partial_charges_rec,
            rec_bonds_near_site,
            lj_pairs,
            charge_tree,
            bh_config: bh_config.clone(),
        }
    }
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

#[derive(Clone, Debug)]
/// Bonds that are marked as flexible.
pub struct Torsion {
    // pub status: TorsionStatus,
    pub bond: usize, // Index.
    pub dihedral_angle: f32,
}

/// Counts the number of bonds connected to a given atom. Used for flexibility computation.
fn count_bonds(atom_index: usize, mol: &Molecule) -> usize {
    mol.bonds
        .iter()
        .filter(|bond| bond.atom_0 == atom_index || bond.atom_1 == atom_index)
        .count()
}

// Set up which atoms in a ligand are flexible.
pub fn setup_flexibility(mol: &mut Molecule) -> Vec<usize> {
    let mut flexible_bonds = Vec::new();

    for (i, bond) in mol.bonds.iter().enumerate() {
        // Only consider single bonds.
        let bond_count = match bond.bond_type {
            BondType::Covalent { count, .. } => count,
            _ => BondCount::Triple,
        };
        if bond_count != BondCount::Single {
            continue;
        }

        // Exclude bonds that are part of a ring.
        if is_bond_in_ring(bond, mol) {
            continue;
        }

        // Retrieve atoms at each end.
        let atom0 = &mol.atoms[bond.atom_0];
        let atom1 = &mol.atoms[bond.atom_1];

        // Check if both atoms are carbon.
        if atom0.element != Element::Carbon || atom1.element != Element::Carbon {
            continue;
        }

        // Exclude terminal bonds (e.g., bonds where an atom only has one connection).
        if count_bonds(bond.atom_0, mol) <= 1 || count_bonds(bond.atom_1, mol) <= 1 {
            continue;
        }

        // Additional heuristics (e.g. hybridization or sterics) could be added here.

        flexible_bonds.push(i);
    }

    flexible_bonds
}

/// Returns the list of neighboring atom indices for a given atom.
fn get_neighbors(atom_index: usize, mol: &Molecule) -> Vec<usize> {
    let mut neighbors = Vec::new();
    for bond in &mol.bonds {
        if bond.atom_0 == atom_index {
            neighbors.push(bond.atom_1);
        } else if bond.atom_1 == atom_index {
            neighbors.push(bond.atom_0);
        }
    }
    neighbors
}

/// Checks if a bond is part of a ring.
/// This function performs a DFS from one atom to see if the other can be reached without using the bond itself.
fn is_bond_in_ring(bond: &Bond, mol: &Molecule) -> bool {
    let start = bond.atom_0;
    let target = bond.atom_1;
    let mut visited = vec![false; mol.atoms.len()];
    let mut stack = vec![start];

    while let Some(current) = stack.pop() {
        if current == target {
            return true;
        }
        visited[current] = true;
        for neighbor in get_neighbors(current, mol) {
            // Skip the edge corresponding to the bond in question.
            if (current == bond.atom_0 && neighbor == bond.atom_1)
                || (current == bond.atom_1 && neighbor == bond.atom_0)
            {
                continue;
            }
            if !visited[neighbor] {
                stack.push(neighbor);
            }
        }
    }
    false
}


/// Find the subet of receptor atoms near a docking site. Only perform force calculations
/// between this set and the ligand, to keep computational complexity under control.
pub fn find_rec_atoms_near_site(
    receptor: &Molecule,
    site: &DockingSite,
) -> (Vec<Atom>, Vec<usize>) {
    let dist_thresh = ATOM_NEAR_SITE_DIST_THRESH * site.site_box_size;
    println!("Dist thresh: {:?}", dist_thresh);

    let mut indices = Vec::new();

    let atoms = receptor
        .atoms
        .iter()
        .enumerate()
        .filter(|(i, a)| {
            let r = (a.posit - site.site_center).magnitude() < dist_thresh && !a.hetero;
            if r {
                indices.push(*i);
            }
            r
        })
        .map(|(i, a)| a.clone()) // todo: Don't like the clone;
        .collect();

    (atoms, indices)
}