#![allow(non_camel_case_types)]

//! Contains data structures and related code for molecules, atoms, residues, chains, etc.
use std::{
    collections::HashMap,
    fmt,
    str::FromStr,
    sync::mpsc::{self, Receiver},
    thread,
};

use bio_apis::{
    ReqError, rcsb,
    rcsb::{DataAvailable, PdbMetaData},
};
use lin_alg::{
    f32::Vec3 as Vec3F32,
    f64::{Quaternion, Vec3},
};
use na_seq::AminoAcid;
use rayon::prelude::*;

// use pdbtbx::SecondaryStructure;
use crate::{
    Selection,
    aa_coords::Dihedral,
    bond_inference::{create_bonds, create_hydrogen_bonds},
    docking::{
        ConformationType, DockingSite, Pose,
        prep::{DockType, Torsion, UnitCellDims, setup_flexibility},
    },
    element::Element,
    util::mol_center_size,
};
use crate::{prefs::PerMolToSave, reflection::ReflectionsData};

pub const ATOM_NEIGHBOR_DIST_THRESH: f64 = 5.; // todo: Adjust A/R.

#[derive(Debug, Default, Clone)]
pub struct Molecule {
    pub ident: String,
    pub atoms: Vec<Atom>,
    pub bonds: Vec<Bond>,
    /// Relating covalent bonds. For each atom, a list of atoms bonded to it.
    pub adjacency_list: Vec<Vec<usize>>,
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
    pub eem_charges_assigned: bool,
    // pub secondary_structure: Vec<SecondaryStructure>,
    /// Center and size are used for lighting, and for rotating ligands.
    pub center: Vec3,
    pub size: f32,
    pub pubchem_cid: Option<u32>,
    pub drugbank_id: Option<String>,
    /// We currently use this for aligning ligands to CIF etc data, where they may already be included
    /// in a protein/ligand complex as hetero atoms.
    pub het_residues: Vec<Residue>,
    pub rcsb_data_avail: Option<DataAvailable>,
    pub reflections_data: Option<ReflectionsData>,
}

impl Molecule {
    /// This constructor handles assumes details are ingested into a common format upstream. It adds
    /// them to the resulting structure, and augments it with bonds, hydrogen positions, and other things A/R.
    pub fn new(
        ident: String,
        atoms: Vec<Atom>,
        chains: Vec<Chain>,
        residues: Vec<Residue>,
        // secondary_structure: Vec<SecondaryStructure>,
        pubchem_cid: Option<u32>,
        drugbank_id: Option<String>,
    ) -> Self {
        let (center, size) = mol_center_size(&atoms);

        let mut result = Self {
            ident,
            atoms,
            bonds: Vec::new(),
            chains,
            residues,
            // secondary_structure,
            center,
            size,
            pubchem_cid,
            drugbank_id,
            ..Default::default()
        };

        // todo: Perhaps you still want to calculate dihedral angles if hydrogens are populated already.
        // todo; For now, you are skipping both. Example when this comes up: Ligands.
        // Attempt to only populate Hydrogens if there aren't many.
        if result
            .atoms
            .iter()
            .filter(|a| a.element == Element::Hydrogen)
            .count()
            < 4
        {
            result.populate_hydrogens_angles();
        }

        result.bonds = create_bonds(&result.atoms);
        result.bonds_hydrogen = create_hydrogen_bonds(&result.atoms, &result.bonds);
        result.adjacency_list = result.build_adjacency_list();

        for res in &result.residues {
            if let ResidueType::Other(_) = &res.res_type {
                if res.atoms.len() >= 10 {
                    result.het_residues.push(res.clone());
                }
            }
        }

        // todo: Don't like this clone.
        let atoms_clone = result.atoms.clone();
        for atom in &mut result.atoms {
            atom.dock_type = Some(DockType::infer(atom, &result.bonds, &atoms_clone));
        }

        result
    }

    /// We use this as part of our flexible-bond conformation algorithm.
    fn build_adjacency_list(&self) -> Vec<Vec<usize>> {
        let n_atoms = self.atoms.len();
        // Start with empty neighbors for each atom
        let mut adjacency_list = vec![Vec::new(); n_atoms];

        // For each bond, record its atoms as neighbors of each other
        for bond in &self.bonds {
            adjacency_list[bond.atom_0].push(bond.atom_1);
            adjacency_list[bond.atom_1].push(bond.atom_0);
        }

        adjacency_list
    }

    /// If a residue, get the alpha C.
    pub fn get_sel_atom(&self, sel: Selection) -> Option<&Atom> {
        match sel {
            Selection::Atom(i) => self.atoms.get(i),
            Selection::Residue(i) => {
                let res = &self.residues[i];
                if !res.atoms.is_empty() {
                    for atom_i in &res.atoms {
                        let atom = &self.atoms[*atom_i];
                        if let Some(role) = atom.role {
                            if role == AtomRole::C_Alpha {
                                return Some(atom);
                            }
                        }
                    }

                    // If we can't find  C alpha, default to the first atom.
                    Some(&self.atoms[res.atoms[0]])
                } else {
                    None
                }
            }
            Selection::None => None,
        }
    }

    /// Load which (beyond coordinates) data items are available from the PDB. We do this
    /// in a new thread, to prevent blocking the UI, or delaying a molecule's loading.
    pub fn update_data_avail(
        &mut self,
        pending_data_avail: &mut Option<Receiver<Result<DataAvailable, ReqError>>>,
    ) {
        println!("Updating data avail...");

        // todo
        if self.rcsb_data_avail.is_some() || pending_data_avail.is_some() {
            return;
        }

        println!(
            "Spawning worker thread to fetch data-avail for {}",
            self.ident
        );

        let ident = self.ident.clone(); // data the worker needs
        let (tx, rx) = mpsc::channel(); // one-shot channel

        thread::spawn(move || {
            let res = rcsb::get_data_avail(&ident);
            // it’s fine if the send fails (e.g. the app closed)
            let _ = tx.send(res);
        });

        *pending_data_avail = Some(rx);

        if self.rcsb_data_avail.is_none() {
            println!("Getting web data avail for {:?}", self.ident);
            match rcsb::get_data_avail(&self.ident) {
                Ok(d) => {
                    println!("Data available loaded: {:?}", d);
                    self.rcsb_data_avail = Some(d);
                }
                Err(_) => eprintln!("Error getting RCSB data availability for {}", self.ident),
            }
        } else {
            // todo temp
            println!(
                "Already have data available: {:?}",
                &self.rcsb_data_avail.as_ref().unwrap()
            );
        }
    }

    /// Call this periodically from the UI/event loop; it’s non-blocking.
    pub fn poll_data_avail(
        &mut self,
        pending_data_avail: &mut Option<Receiver<Result<DataAvailable, ReqError>>>,
    ) {
        if let Some(rx) = pending_data_avail {
            // `try_recv` returns immediately
            match rx.try_recv() {
                Ok(Ok(d)) => {
                    println!("Data-avail ready for {}: {:?}", self.ident, d);
                    self.rcsb_data_avail = Some(d);
                    *pending_data_avail = None; // finished
                }
                Ok(Err(e)) => {
                    eprintln!("Failed to fetch data-avail for {}: {e:?}", self.ident);
                    *pending_data_avail = None;
                }
                Err(mpsc::TryRecvError::Empty) => {
                    // Still working – do nothing this frame
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    eprintln!("Worker thread died before sending result");
                    *pending_data_avail = None;
                }
            }
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

#[derive(Debug, Clone, Default)]
pub struct Ligand {
    /// Molecule atom positions remain relative.
    pub molecule: Molecule,
    /// These positions are derivative of the pose, in conjunction with the molecule atoms' [relative]
    /// positions.
    pub atom_posits: Vec<Vec3>,
    // pub offset: Vec3,
    pub anchor_atom: usize,         // Index.
    pub flexible_bonds: Vec<usize>, // Index
    pub pose: Pose,
    pub docking_site: DockingSite,
    pub unit_cell_dims: UnitCellDims, // todo: Unused
}

impl Ligand {
    pub fn new(molecule: Molecule) -> Self {
        let mut result = Self {
            molecule,
            ..Default::default()
        };

        result.set_anchor();
        result.flexible_bonds = setup_flexibility(&result.molecule);

        result.pose.conformation_type = ConformationType::Flexible {
            torsions: result
                .flexible_bonds
                .iter()
                .map(|b| Torsion {
                    bond: *b,
                    dihedral_angle: 0.,
                })
                .collect(),
        };

        println!("Torsions: {:?}", result.pose.conformation_type);
        // todo: Temp for testing.
        // {
        //     result.docking_site = DockingSite {
        //         site_center: Vec3::new(40.6807, 36.2017, 28.5526),
        //         site_radius: 10.,
        //     };
        //     result.pose.anchor_posit = result.docking_site.site_center;
        //     result.pose.orientation = Quaternion::new(0.1156, -0.7155, 0.4165, 0.5488);
        //
        //     if let ConformationType::Flexible { torsions } = &mut result.pose.conformation_type {
        //         // torsions[1].dihedral_angle = 0.884;
        //         // torsions[0].dihedral_angle = 2.553;
        //         torsions[0].dihedral_angle = 0.884;
        //         torsions[1].dihedral_angle = 2.553;
        //     }
        // }

        result.atom_posits = result.position_atoms(None);
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

        self.anchor_atom = anchor_atom;
    }

    /// Creates global positions for all atoms. This takes into account position, orientation, and if applicable,
    /// torsion angles from flexible bonds. Each pivot rotation rotates the side of the flexible bond that
    /// has fewer atoms; the intent is to minimize the overall position changes for these flexible bond angle
    /// changes.
    pub fn position_atoms(&self, pose: Option<&Pose>) -> Vec<Vec3> {
        let anchor = self.molecule.atoms[self.anchor_atom].posit;

        let pose_ = match pose {
            Some(p) => p,
            None => &self.pose,
        };

        let mut result: Vec<_> = self
            .molecule
            .atoms
            .par_iter()
            .map(|atom| {
                let posit_rel = atom.posit - anchor;
                pose_.anchor_posit + pose_.orientation.rotate_vec(posit_rel)
            })
            .collect();

        if let ConformationType::Flexible { torsions } = &pose_.conformation_type {
            // Second pass: Rotations. For each flexible bond, divide all atoms into two groups:
            // those upstream of this bond, and those downstream. For all downstream atoms, rotate
            // by `torsions[i]`: The dihedral angle along this bond. If there are ambiguities in this
            // process, it may mean the bond should not have been marked as flexible.
            for torsion in torsions {
                let bond = &self.molecule.bonds[torsion.bond];

                // -- Step 1: measure how many atoms would be "downstream" from each side
                let side0_downstream = self.find_downstream_atoms(bond.atom_1, bond.atom_0);
                let side1_downstream = self.find_downstream_atoms(bond.atom_0, bond.atom_1);

                // -- Step 2: pick the pivot as the side with a larger subtree
                let (pivot_idx, side_idx, downstream_atom_indices) =
                    if side0_downstream.len() > side1_downstream.len() {
                        // side0_downstream means "downstream from atom_1 ignoring bond to atom_0"
                        // => so pivot is atom_0, side is atom_1
                        (bond.atom_0, bond.atom_1, side1_downstream)
                    } else {
                        // side1_downstream has equal or more
                        (bond.atom_1, bond.atom_0, side0_downstream)
                    };

                // pivot and side positions
                let pivot_pos = result[pivot_idx];
                let side_pos = result[side_idx];
                let axis_vec = (side_pos - pivot_pos).to_normalized();

                // Build the Quaternion for this rotation
                let rotator = Quaternion::from_axis_angle(axis_vec, torsion.dihedral_angle as f64);

                // Now apply the rotation to each downstream atom:
                for &atom_idx in &downstream_atom_indices {
                    let old_pos = result[atom_idx];
                    let relative = old_pos - pivot_pos;
                    let new_pos = pivot_pos + rotator.rotate_vec(relative);
                    result[atom_idx] = new_pos;
                }
            }
        }

        result
    }

    /// We use this to rotate flexible molecules around torsion (e.g. dihedral) angles.
    /// `pivot` and `side` are atom indices in the molecule.
    pub fn find_downstream_atoms(&self, pivot: usize, side: usize) -> Vec<usize> {
        // adjacency_list[atom] -> list of neighbors
        // We want all atoms reachable from `side` when we remove the edge (side->pivot).
        let mut visited = vec![false; self.molecule.atoms.len()];
        let mut stack = vec![side];
        let mut result = vec![];

        visited[side] = true;

        while let Some(current) = stack.pop() {
            result.push(current);

            for &nbr in &self.molecule.adjacency_list[current] {
                // skip the pivot to avoid going back across the chosen bond
                if nbr == pivot {
                    continue;
                }
                if !visited[nbr] {
                    visited[nbr] = true;
                    stack.push(nbr);
                }
            }
        }

        result
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

impl Default for ResidueType {
    fn default() -> Self {
        Self::Other(String::new())
    }
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

#[derive(Debug, Clone, Default)]
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
