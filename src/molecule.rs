//! Contains data structures and related code for molecules, atoms, residues, chains, etc.
use std::{fmt, str::FromStr};

use lin_alg::{f32::Vec3 as Vec3F32, f64::Vec3};
use na_seq::AminoAcid;
use pdbtbx::SecondaryStructure;

use crate::{
    Selection,
    aa_coords::Dihedral,
    docking::{
        ConformationType, DockingInit, Pose,
        docking_prep::{DockType, Torsion, UnitCellDims},
    },
    element::Element,
    rcsb_api::PdbMetaData,
};

pub const ATOM_NEIGHBOR_DIST_THRESH: f64 = 5.; // todo: Adjust A/R.

#[derive(Debug, Default, Clone)]
pub struct Molecule {
    pub ident: String,
    pub atoms: Vec<Atom>,
    pub bonds: Vec<Bond>,
    pub bonds_hydrogen: Vec<HydrogenBond>,
    pub chains: Vec<Chain>,
    pub residues: Vec<Residue>,
    pub metadata: Option<PdbMetaData>,
    /// Solvent-accessible surface. Details may evolve.
    /// Current structure is a Vec of rings.
    /// Initializes to empty; updated A/R when the appropriate view is selected.
    pub sa_surface_pts: Option<Vec<Vec<Vec3F32>>>,
    /// Stored in scene meshes; this variable keeps track if that's populated.
    pub mesh_created: bool,
    pub secondary_structure: Vec<SecondaryStructure>,
    /// Center and size are used for lighting, and for rotating ligands.
    pub center: Vec3,
    pub size: f32,
    pub pubchem_cid: Option<u32>,
    pub drugbank_id: Option<String>,
}

impl Molecule {
    /// If residue, get an arbitrary atom. (todo: Get c alpha always).
    pub fn get_sel_atom(&self, sel: Selection) -> Option<&Atom> {
        match sel {
            Selection::Atom(i) => self.atoms.get(i),
            Selection::Residue(i) => {
                let res = &self.residues[i];
                if !res.atoms.is_empty() {
                    Some(&self.atoms[res.atoms[0]])
                } else {
                    None
                }
            }
            Selection::None => None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum AtomRole {
    C_Alpha,
    C_Prime,
    N_Backbone,
    O_Backbone,
    H_Backbone,
    Sidechain,
    H_Sidechain,
    Water,
    Other,
}

impl AtomRole {
    pub fn from_name(name: &str) -> Self {
        match name {
            "CA" => Self::C_Alpha,
            "C" => Self::C_Prime,
            "N" => Self::N_Backbone,
            "O" => Self::O_Backbone,
            "H" | "H1" | "H2" | "H3" | "HA" | "HA2" | "HA3" => Self::H_Backbone,
            _ => Self::Sidechain,
        }
    }
}

impl fmt::Display for AtomRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AtomRole::C_Alpha => write!(f, "Cα"),
            AtomRole::C_Prime => write!(f, "C'"),
            AtomRole::N_Backbone => write!(f, "N (bb)"),
            AtomRole::O_Backbone => write!(f, "O (bb)"),
            AtomRole::H_Backbone => write!(f, "H (bb)"),
            AtomRole::Sidechain => write!(f, "Sidechain"),
            AtomRole::H_Sidechain => write!(f, "H SC"),
            AtomRole::Water => write!(f, "Water"),
            AtomRole::Other => write!(f, "Other"),
        }
    }
}

#[derive(Debug, Default)]
pub struct Ligand {
    pub molecule: Molecule,
    // pub offset: Vec3,
    pub pose: Pose,
    pub docking_init: DockingInit,
    // pub orientation: Quaternion, // Assumes rigid.
    pub torsions: Vec<Torsion>,
    pub unit_cell_dims: UnitCellDims,
}

impl Ligand {
    pub fn new(molecule: Molecule) -> Self {
        let docking_init = DockingInit {
            // site_center: Vec3::new(-18.955, -5.188, 8.617),
            site_center: Vec3::new(38.699, 36.415, 30.815),
            site_box_size: 10.,
        };

        let mut result = Self {
            molecule,
            docking_init,
            ..Default::default()
        };
        result.set_anchor();
        result
    }

    /// Separate from constructor; run when the pose changes, for now.
    pub fn set_anchor(&mut self) {
        let mut center = Vec3::new_zero();
        for atom in &self.molecule.atoms {
            center += atom.posit;
        }
        center /= self.molecule.atoms.len() as f64;

        let mut anchor_atom = 0;
        let mut best_dist = 999999.;

        for (i, atom) in self.molecule.atoms.iter().enumerate() {
            let dist = (atom.posit - center).magnitude();
            if dist < best_dist {
                best_dist = dist;
                anchor_atom = i;
            }
        }

        self.pose.anchor_atom = anchor_atom;
    }

    pub fn position_atom(&self, atom_i: usize, pose: Option<&Pose>) -> Vec3 {
        let atom = self.molecule.atoms[atom_i].posit;

        let pose_ = match pose {
            Some(p) => p,
            None => &self.pose,
        };

        match &pose_.conformation_type {
            ConformationType::Rigid { orientation } => {
                let anchor = self.molecule.atoms[pose_.anchor_atom].posit;
                // Rotate around the anchor atom.
                let posit_rel = atom - anchor;
                pose_.anchor_posit + orientation.rotate_vec(posit_rel)
                // self.pose.anchor_posit + posit_rel
            }
            ConformationType::Flexible { dihedral_angles } => {
                unimplemented!()
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum BondType {
    // C+P from pdbtbx for now
    Covalent {
        count: BondCount,
    },
    /// Donor is always `atom0`.`
    Hydrogen,
    Disulfide,
    MetalCoordination,
    MisMatchedBasePairs,
    SaltBridge,
    CovalentModificationResidue,
    CovalentModificationNucleotideBase,
    CovalentModificationNucleotideSugar,
    CovalentModificationNucleotidePhosphate,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum BondCount {
    Single,
    SingleDoubleHybrid,
    Double,
    Triple,
}

impl BondCount {
    pub fn value(&self) -> f64 {
        match self {
            Self::Single => 1.0,
            Self::SingleDoubleHybrid => 1.5,
            Self::Double => 2.0,
            Self::Triple => 3.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Bond {
    pub bond_type: BondType,
    /// Index
    pub atom_0: usize,
    /// Index
    pub atom_1: usize,
    pub is_backbone: bool,
}

#[derive(Debug, Clone)]
pub struct HydrogenBond {
    /// All three atoms are indexes.
    pub donor: usize,
    pub acceptor: usize,
    pub hydrogen: usize,
}

#[derive(Debug, Clone)]
pub struct Chain {
    pub id: String,
    // todo: Do we want both residues and atoms stored here? It's an overconstraint.
    pub residues: Vec<usize>,
    /// Indexes
    pub atoms: Vec<usize>,
    // todo: Perhaps vis would make more sense in a separate UI-related place.
    pub visible: bool,
}

#[derive(Debug, Clone)]
pub enum ResidueType {
    AminoAcid(AminoAcid),
    Water,
    Other(String),
}

impl ResidueType {
    /// Parses from the "name" field in common text-based formats lik CIF, PDB, and PDBQT.
    pub fn from_str(name: &str) -> Self {
        if name.to_uppercase() == "HOH" {
            ResidueType::Water
        } else {
            match AminoAcid::from_str(name) {
                Ok(aa) => ResidueType::AminoAcid(aa),
                Err(_) => ResidueType::Other(name.to_owned()),
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Residue {
    /// We use serial number of display, search etc, and array index to select. Residue serial number is not
    /// unique in the molecule; only in the chain.
    pub serial_number: isize, // pdbtbx uses isize. Negative allowed?
    pub res_type: ResidueType,
    pub atoms: Vec<usize>, // Atom index
    pub dihedral: Option<Dihedral>,
}

impl Residue {
    pub fn descrip(&self) -> String {
        let name = match &self.res_type {
            ResidueType::AminoAcid(aa) => aa.to_string(),
            ResidueType::Water => "Water".to_owned(),
            ResidueType::Other(name) => name.clone(),
        };

        let mut result = format!("Res: {}: {name}", self.serial_number);
        if let Some(dihedral) = &self.dihedral {
            result += &format!("   {dihedral}");
        }
        result
    }
}

#[derive(Debug, Clone)]
pub struct Atom {
    pub serial_number: usize,
    pub posit: Vec3,
    pub element: Element,
    pub name: String,
    pub role: Option<AtomRole>,
    pub residue_type: ResidueType, // todo: Duplicate with the residue association.
    pub hetero: bool,
    /// For docking.
    /// // todo: Consider a substruct for docking fields.
    pub dock_type: Option<DockType>,
    pub occupancy: Option<f32>,
    pub partial_charge: Option<f32>,
    pub temperature_factor: Option<f32>,
    // todo: Impl this, for various calculations
    // /// Atoms relatively close to this; simplifies  certain calculations.
    // pub neighbors: Vec<usize>,
}

impl Atom {
    /// Note: This doesn't include backbone O etc; just the 3 main ones.
    pub fn is_backbone(&self) -> bool {
        match self.role {
            Some(r) => [
                AtomRole::C_Alpha,
                AtomRole::N_Backbone,
                AtomRole::C_Prime,
                AtomRole::O_Backbone,
            ]
            .contains(&r),
            None => false,
        }
    }
}

/// Can't find a PyMol equiv. Experimenting
pub const fn aa_color(aa: AminoAcid) -> (f32, f32, f32) {
    match aa {
        AminoAcid::Arg => (0.7, 0.2, 0.9),
        AminoAcid::His => (0.2, 1., 0.2),
        AminoAcid::Lys => (1., 0.3, 0.3),
        AminoAcid::Asp => (0.2, 0.2, 1.0),
        AminoAcid::Glu => (0.701, 0.7, 0.2),
        AminoAcid::Ser => (177. / 255., 187. / 255., 161. / 255.),
        AminoAcid::Thr => (1.0, 0.502, 0.),
        AminoAcid::Asn => (0.878, 0.4, 0.2),
        AminoAcid::Gln => (0.784, 0.502, 0.2),
        AminoAcid::Cys => (0.239, 1.0, 0.),
        AminoAcid::Sec => (0.561, 0.251, 0.831),
        AminoAcid::Gly => (0.749, 0.651, 0.651),
        AminoAcid::Pro => (0.341, 0.349, 0.380),
        AminoAcid::Ala => (1., 0.820, 0.137),
        AminoAcid::Val => (0.753, 0.753, 0.753),
        AminoAcid::Ile => (0.322, 0.722, 0.916),
        AminoAcid::Leu => (0.4, 0.502, 0.502),
        AminoAcid::Met => (0.490, 0.502, 0.690),
        AminoAcid::Phe => (0.580, 0., 0.580),
        AminoAcid::Tyr => (0.541, 1., 0.),
        AminoAcid::Trp => (0.121, 0.941, 0.121),
    }
}

// todo: A/R.

// #[derive(Debug, Clone, PartialEq)]
// /// http://www.bmsc.washington.edu/CrystaLinks/man/pdb/part_42.html
// pub enum HelixClass {
//     Right-handed alpha (default)                1
// Right-handed omega                          2
// Right-handed pi                             3
// Right-handed gamma                          4
// Right-handed 310                            5
// Left-handed alpha                           6
// Left-handed omega                           7
// Left-handed gamma                           8
// 27 ribbon/helix                             9
// Polyproline                                10
// }
//
// impl HelixClass {
//     pub fn from
// }
