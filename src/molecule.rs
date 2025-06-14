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
    rcsb::{FilesAvailable, PdbDataResults, PdbMetaData},
};
use bio_files::{
    AtomGeneric, BondGeneric, Chain, ChargeType, DensityMap, MapHeader, Mol2, MolType,
    ResidueGeneric, ResidueType, sdf::Sdf,
};
use lin_alg::{
    f32::Vec3 as Vec3F32,
    f64::{Quaternion, Vec3},
};
use na_seq::{AminoAcid, Element};
use rayon::prelude::*;

use crate::{
    Selection,
    aa_coords::Dihedral,
    bond_inference::{create_bonds, create_hydrogen_bonds},
    cartoon_mesh::BackboneSS,
    docking::{
        ConformationType, DockingSite, Pose,
        prep::{DockType, Torsion, UnitCellDims, setup_flexibility},
    },
    reflection::{DensityRect, ElectronDensity, ReflectionsData},
    util::mol_center_size,
};

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
    pub secondary_structure: Vec<BackboneSS>,
    /// Center and size are used for lighting, and for rotating ligands.
    pub center: Vec3,
    pub size: f32,
    pub pubchem_cid: Option<u32>,
    pub drugbank_id: Option<String>,
    /// We currently use this for aligning ligands to CIF etc data, where they may already be included
    /// in a protein/ligand complex as hetero atoms.
    pub het_residues: Vec<Residue>,
    /// The full (Or partial while WIP) results from the RCSB data api.
    pub rcsb_data: Option<PdbDataResults>,
    pub rcsb_files_avail: Option<FilesAvailable>,
    pub reflections_data: Option<ReflectionsData>,
    /// E.g. from a MAP file, or 2fo-fc header.
    /// From reflections
    pub elec_density: Option<Vec<ElectronDensity>>,
    pub density_map: Option<DensityMap>,
    pub density_rect: Option<DensityRect>,
    pub aa_seq: Vec<AminoAcid>,
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

        result.aa_seq = result.get_seq();

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

        let bonds = create_bonds(&result.atoms);
        result.bonds = bonds;

        result.bonds_hydrogen = create_hydrogen_bonds(&result.atoms, &result.bonds);
        println!("H bond count: {:?}", result.bonds_hydrogen.len());

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
    pub fn build_adjacency_list(&self) -> Vec<Vec<usize>> {
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

    /// If a residue, get the alpha C. If multiple, get an arbtirary one.
    pub fn get_sel_atom(&self, sel: &Selection) -> Option<&Atom> {
        match sel {
            Selection::Atom(i) => self.atoms.get(*i),
            Selection::Residue(i) => {
                let res = &self.residues[*i];
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
            Selection::Atoms(is) => {
                // todo temp?
                self.atoms.get(is[0])
            }
            Selection::None => None,
        }
    }

    /// Load RCSB data, and the list of (non-coordinate) files available from the PDB. We do this
    /// in a new thread, to prevent blocking the UI, or delaying a molecule's loading.
    pub fn updates_rcsb_data(
        &mut self,
        pending_data: &mut Option<
            Receiver<(
                Result<PdbDataResults, ReqError>,
                Result<FilesAvailable, ReqError>,
            )>,
        >,
    ) {
        if (self.rcsb_files_avail.is_some() && self.rcsb_data.is_some()) || pending_data.is_some() {
            return;
        }

        println!("Existing data: {:?}", self.rcsb_data);
        println!("Existing files: {:?}", self.rcsb_files_avail);

        let ident = self.ident.clone(); // data the worker needs
        let (tx, rx) = mpsc::channel(); // one-shot channel

        println!("Getting RCSB data...");

        thread::spawn(move || {
            let data = rcsb::get_all_data(&ident);
            let files_data = rcsb::get_files_avail(&ident);

            // it’s fine if the send fails (e.g. the app closed)
            let _ = tx.send((data, files_data));
        });

        *pending_data = Some(rx);
    }

    /// Call this periodically from the UI/event loop; it’s non-blocking.
    /// Returns if it updated, e.g. so we can update prefs.
    pub fn poll_data_avail(
        &mut self,
        pending_data_avail: &mut Option<
            Receiver<(
                Result<PdbDataResults, ReqError>,
                Result<FilesAvailable, ReqError>,
            )>,
        >,
    ) -> bool {
        if let Some(rx) = pending_data_avail {
            // `try_recv` returns immediately
            match rx.try_recv() {
                // both fetches succeeded:
                Ok((Ok(pdb_data), Ok(files_avail))) => {
                    println!("RCSB data ready for {}", self.ident);
                    self.rcsb_data = Some(pdb_data);
                    self.rcsb_files_avail = Some(files_avail);

                    // todo: Save state here, but need to get a proper signal with &mut State available.

                    *pending_data_avail = None;
                    return true;
                }

                // PdbDataResults failed, but FilesAvailable might not have been sent:
                Ok((Err(e), _)) => {
                    eprintln!("Failed to fetch PDB data for {}: {e:?}", self.ident);
                    *pending_data_avail = None;
                }

                // FilesAvailable failed (even if PdbDataResults succeeded):
                Ok((_, Err(e))) => {
                    eprintln!("Failed to fetch file‐list for {}: {e:?}", self.ident);
                    *pending_data_avail = None;
                }

                // the worker hasn’t sent anything yet:
                Err(mpsc::TryRecvError::Empty) => {
                    // still pending; do nothing this frame
                }

                // the sender hung up before sending:
                Err(mpsc::TryRecvError::Disconnected) => {
                    eprintln!("Worker thread died before sending result");
                    *pending_data_avail = None;
                }
            }
        }
        false
    }

    /// Get the amino acid sequence from the currently opened molecule, if applicable.
    fn get_seq(&self) -> Vec<AminoAcid> {
        // todo: If not a polypeptide, should we return an error, or empty vec?
        let mut result = Vec::new();

        // todo This is fragile, I believe.
        for res in &self.residues {
            if let ResidueType::AminoAcid(aa) = res.res_type {
                result.push(aa);
            }
        }

        result
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
        if self.anchor_atom >= self.molecule.atoms.len() {
            eprintln!(
                "Error positioning ligand atoms: Anchor outside atom count. Atom cound: {:?}",
                self.molecule.atoms.len()
            );
            return Vec::new();
        }
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

#[allow(unused)]
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

#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum BondCount {
    #[default]
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

    pub fn _from_count(count: u8) -> Self {
        match count {
            1 => Self::Single,
            2 => Self::Double,
            3 => Self::Triple,
            _ => {
                eprintln!("Error: Invalid count value: {}", count);
                Self::Single
            }
        }
    }

    /// E.g. the Mol2 format.
    pub fn from_str(val: &str) -> Self {
        // 1 = single
        // 2 = double
        // 3 = triple
        // am = amide
        // ar = aromatic
        // du = dummy
        // un = unknown (cannot be determined from the parameter tables)
        // nc = not connected
        match val {
            "1" => Self::Single,
            "2" => Self::Double,
            "3" => Self::Triple,
            // todo: How should we handle these? New types in the enum?
            "am" => Self::SingleDoubleHybrid,
            "ar" => Self::Triple,
            "du" => Self::Single,
            "un" => Self::Single,
            "nc" => Self::Single,
            _ => {
                eprintln!("Error: Invalid count value: {}", val);
                Self::Single
            }
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

impl Bond {
    pub fn to_generic(&self) -> BondGeneric {
        BondGeneric {
            bond_type: "1".to_owned(), // todo!
            // todo: Map serial num to index incase these don't ascend by one.
            atom_0: self.atom_0 + 1,
            atom_1: self.atom_1 + 1,
        }
    }
}

impl From<&BondGeneric> for Bond {
    fn from(bond: &BondGeneric) -> Self {
        Self {
            bond_type: BondType::Covalent {
                count: BondCount::from_str(&bond.bond_type),
            },
            // Our bonds are by index; these are by serial number. This should align them in most cases.
            // todo: Map serial num to index incase these don't ascend by one.
            atom_0: bond.atom_0 - 1,
            atom_1: bond.atom_1 - 1,
            is_backbone: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HydrogenBond {
    /// All three atoms are indexes.
    pub donor: usize,
    pub acceptor: usize,
    pub hydrogen: usize,
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
    pub fn to_generic(&self) -> ResidueGeneric {
        ResidueGeneric {
            serial_number: self.serial_number,
            res_type: self.res_type.clone(),
            atoms: self.atoms.clone(),
        }
    }
}

impl From<&ResidueGeneric> for Residue {
    fn from(res: &ResidueGeneric) -> Self {
        Self {
            serial_number: res.serial_number,
            res_type: res.res_type.clone(),
            atoms: res.atoms.clone(),
            dihedral: None,
        }
    }
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
    // todo: We should have a residue *pointer* etc to speed up computations;
    // todo: We shouldn't have to iterate through residues checking for atom membership.
    /// We include this reference to the residue for speed; iterating through residues to check for
    /// atom membership is slow.
    pub residue: Option<usize>,
    // pub residue_type: ResidueType, // todo: Duplicate with the residue association.
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

    pub fn to_generic(&self) -> AtomGeneric {
        AtomGeneric {
            serial_number: self.serial_number,
            posit: self.posit,
            element: self.element,
            // name: String::new(),
            partial_charge: self.partial_charge,
            ..Default::default()
        }
    }
}

impl From<&AtomGeneric> for Atom {
    fn from(atom: &AtomGeneric) -> Self {
        Self {
            serial_number: atom.serial_number,
            posit: atom.posit,
            element: atom.element,
            name: String::new(),
            partial_charge: atom.partial_charge,
            ..Default::default()
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

impl From<Mol2> for Molecule {
    fn from(m: Mol2) -> Self {
        let atoms = m.atoms.iter().map(|a| a.into()).collect();

        let mut result = Self::new(m.ident, atoms, Vec::new(), Vec::new(), None, None);

        let bonds = m.bonds.iter().map(|b| b.into()).collect();

        // This replaces the built-in bond computation with our own. Ideally, we don't even calculate
        // those for performance reasons.
        result.bonds = bonds;
        result.bonds_hydrogen = Vec::new();
        result.adjacency_list = result.build_adjacency_list();

        result
    }
}

impl From<Sdf> for Molecule {
    fn from(m: Sdf) -> Self {
        let atoms = m.atoms.iter().map(|a| a.into()).collect();
        let residues = m.residues.iter().map(|r| r.into()).collect();

        let mut result = Self::new(m.ident, atoms, m.chains.clone(), residues, None, None);

        let bonds = m.bonds.iter().map(|b| b.into()).collect();

        // See note in Mol2's method.
        result.bonds = bonds;
        result.bonds_hydrogen = Vec::new();
        result.adjacency_list = result.build_adjacency_list();

        result
    }
}

impl Molecule {
    pub fn to_mol2(&self) -> Mol2 {
        let atoms = self.atoms.iter().map(|a| a.to_generic()).collect();
        let bonds = self.bonds.iter().map(|b| b.to_generic()).collect();

        Mol2 {
            ident: self.ident.clone(),
            mol_type: MolType::Small,
            charge_type: ChargeType::None,
            comment: None,
            atoms,
            bonds,
        }
    }

    // todo: DRY!
    pub fn to_sdf(&self) -> Sdf {
        let atoms = self.atoms.iter().map(|a| a.to_generic()).collect();
        let bonds = self.bonds.iter().map(|b| b.to_generic()).collect();
        let residues = self.residues.iter().map(|r| r.to_generic()).collect();

        Sdf {
            ident: self.ident.clone(),
            atoms,
            bonds,
            chains: self.chains.clone(),
            residues,
            metadata: HashMap::new(), // todo?
            pubchem_cid: self.pubchem_cid,
            drugbank_id: self.drugbank_id.clone(),
        }
    }
}
