#![allow(non_camel_case_types)]

//! Contains data structures and related code for molecules, atoms, residues, chains, etc.
use std::{
    collections::HashMap,
    fmt,
    fmt::{Display, Formatter},
    io,
    io::ErrorKind,
    str::FromStr,
    sync::mpsc::{self, Receiver},
    thread,
};

use bio_apis::{
    ReqError, rcsb,
    rcsb::{FilesAvailable, PdbDataResults, PdbMetaData},
};
use bio_files::{
    AtomGeneric, BackboneSS, BondGeneric, ChainGeneric, ChargeType, DensityMap, ExperimentalMethod,
    MmCif, Mol2, MolType, ResidueGeneric, ResidueType, Sdf,
};
use lin_alg::{
    f32::Vec3 as Vec3F32,
    f64::{Quaternion, Vec3},
};
use na_seq::{AminoAcid, AtomTypeInRes, Element};
use rayon::prelude::*;

use crate::{
    ProtFfMap, Selection,
    aa_coords::Dihedral,
    bond_inference::{create_bonds, create_hydrogen_bonds},
    docking::{
        ConformationType, DockingSite, Pose,
        prep::{DockType, Torsion, UnitCellDims, setup_flexibility},
    },
    dynamics::ForceFieldParamsIndexed,
    reflection::{DensityRect, ElectronDensity, ReflectionsData},
    util::mol_center_size,
};

pub const ATOM_NEIGHBOR_DIST_THRESH: f64 = 5.; // todo: Adjust A/R.

#[derive(Debug, Default, Clone)]
pub struct Molecule {
    pub ident: String,
    pub atoms: Vec<Atom>,
    /// Similar to for the ligand. Used for molecular dynamics position, and could be applied
    /// to having multiple peptides open.
    pub atom_posits: Option<Vec<Vec3>>,
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
    pub experimental_method: Option<ExperimentalMethod>,
    pub ff_params: Option<ForceFieldParamsIndexed>,
}

impl Molecule {
    /// This constructor handles assumes details are ingested into a common format upstream. It adds
    /// them to the resulting structure, and augments it with bonds, hydrogen positions, and other things A/R.
    pub fn new(
        ident: String,
        atoms: Vec<Atom>,
        chains: Vec<Chain>,
        residues: Vec<Residue>,
        pubchem_cid: Option<u32>,
        drugbank_id: Option<String>,
        // Populate this with the Amino19.lib map, if we wish to add Hydrogens. (e.g. for mmCif protein
        // data, but not for small molecules).
        add_hydrogens: Option<&ProtFfMap>,
    ) -> Self {
        let (center, size) = mol_center_size(&atoms);

        println!("Loading atoms into mol...");

        let mut result = Self {
            ident,
            atoms,
            bonds: Vec::new(),
            chains,
            residues,
            center,
            size,
            pubchem_cid,
            drugbank_id,
            ..Default::default()
        };

        result.aa_seq = result.get_seq();

        if let Some(ff_map) = add_hydrogens {
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
                if let Err(e) = result.populate_hydrogens_angles(ff_map) {
                    eprintln!("Unable to populate Hydrogens and residue dihedral angles: {e:?}");
                };
            }
        }

        let bonds = create_bonds(&result.atoms);
        result.bonds = bonds;

        result.bonds_hydrogen = create_hydrogen_bonds(&result.atoms, &result.bonds);

        result.adjacency_list = build_adjacency_list(&result.bonds, result.atoms.len());

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

    /// If a residue, get the alpha C. If multiple, get an arbtirary one.
    pub fn get_sel_atom(&self, sel: &Selection) -> Option<&Atom> {
        match sel {
            Selection::Atom(i) | Selection::AtomLigand(i) => self.atoms.get(*i),
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
}

impl AtomRole {
    pub fn from_type_in_res(tir: &AtomTypeInRes) -> Self {
        match tir {
            AtomTypeInRes::CA => AtomRole::C_Alpha,
            AtomTypeInRes::C => AtomRole::C_Prime,
            AtomTypeInRes::N => AtomRole::N_Backbone,
            AtomTypeInRes::O => AtomRole::O_Backbone,
            AtomTypeInRes::H(h_type) => match h_type.as_ref() {
                "H" | "H1" | "H2" | "H3" | "HA" | "HA2" | "HA3" | "HN" | "HT1" | "HT2" | "HT3" => {
                    Self::H_Backbone
                }
                _ => Self::Sidechain,
            },
            _ => Self::Sidechain,
        }
    }
}

impl Display for AtomRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AtomRole::C_Alpha => write!(f, "Cα"),
            AtomRole::C_Prime => write!(f, "C'"),
            AtomRole::N_Backbone => write!(f, "N (bb)"),
            AtomRole::O_Backbone => write!(f, "O (bb)"),
            AtomRole::H_Backbone => write!(f, "H (bb)"),
            AtomRole::Sidechain => write!(f, "Side"),
            AtomRole::H_Sidechain => write!(f, "H SC"),
            AtomRole::Water => write!(f, "Water"),
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

        result.position_atoms(None);
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
    ///
    /// If we return None, use the existing atom_posits data; it has presumably been already set.
    pub fn position_atoms(&mut self, pose: Option<&Pose>) {
        let pose_ = match pose {
            Some(p) => p,
            None => &self.pose,
        };

        match &pose_.conformation_type {
            ConformationType::Flexible { torsions } => {
                if self.anchor_atom >= self.molecule.atoms.len() {
                    eprintln!(
                        "Error positioning ligand atoms: Anchor outside atom count. Atom cound: {:?}",
                        self.molecule.atoms.len()
                    );
                    return;
                }
                let anchor = self.molecule.atoms[self.anchor_atom].posit;

                let mut result: Vec<_> = self
                    .molecule
                    .atoms
                    .par_iter()
                    .map(|atom| {
                        let posit_rel = atom.posit - anchor;
                        pose_.anchor_posit + pose_.orientation.rotate_vec(posit_rel)
                    })
                    .collect();
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
                    let rotator =
                        Quaternion::from_axis_angle(axis_vec, torsion.dihedral_angle as f64);

                    // Now apply the rotation to each downstream atom:
                    for &atom_idx in &downstream_atom_indices {
                        let old_pos = result[atom_idx];
                        let relative = old_pos - pivot_pos;
                        let new_pos = pivot_pos + rotator.rotate_vec(relative);
                        result[atom_idx] = new_pos;
                    }
                }
                self.atom_posits = result;
            }
            ConformationType::AbsolutePosits => {
                // take no action; we are using atom_posits.
            }
        }
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
    pub atom_0_sn: u32,
    pub atom_1_sn: u32,
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
            atom_0_sn: self.atom_0_sn,
            atom_1_sn: self.atom_1_sn,
        }
    }
}

// impl From<&BondGeneric> for Bond {
impl Bond {
    // fn from(bond: &BondGeneric) -> Self {
    fn from_generic(bond: &BondGeneric, atom_set: &[Atom]) -> io::Result<Self> {
        let mut atom_0 = 0;
        let mut atom_1 = 0;

        match atom_sns_to_indices(bond.atom_0_sn, atom_set) {
            Some(i) => {
                atom_0 = i;
            }
            None => {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    "Unable to find atom SN when loading from generic res",
                ));
            }
        }
        // todo DRY
        match atom_sns_to_indices(bond.atom_1_sn, atom_set) {
            Some(i) => {
                atom_1 = i;
            }
            None => {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    "Unable to find atom SN when loading from generic res",
                ));
            }
        }

        Ok(Self {
            bond_type: BondType::Covalent {
                count: BondCount::from_str(&bond.bond_type),
            },
            atom_0_sn: bond.atom_0_sn,
            atom_1_sn: bond.atom_1_sn,
            atom_0,
            atom_1,
            is_backbone: false,
        })
    }
}

#[derive(Debug, Clone)]
pub struct HydrogenBond {
    /// All three atoms are indexes.
    pub donor: usize,
    pub acceptor: usize,
    pub hydrogen: usize,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ResidueEnd {
    Internal,
    NTerminus,
    CTerminus,
    /// Not part of a protein/polypeptide.
    Hetero,
}

#[derive(Debug, Clone)]
pub struct Residue {
    /// We use serial number of display, search etc, and array index to select. Residue serial number is not
    /// unique in the molecule; only in the chain.
    pub serial_number: u32,
    pub res_type: ResidueType,
    /// Serial number
    pub atom_sns: Vec<u32>,
    pub atoms: Vec<usize>, // Atom index
    pub dihedral: Option<Dihedral>,
    pub end: ResidueEnd,
}

impl Residue {
    pub fn to_generic(&self) -> ResidueGeneric {
        ResidueGeneric {
            serial_number: self.serial_number,
            res_type: self.res_type.clone(),
            atom_sns: self.atom_sns.clone(),
        }
    }
}

impl Residue {
    fn from_generic(res: &ResidueGeneric, atom_set: &[Atom], end: ResidueEnd) -> io::Result<Self> {
        let mut atoms = Vec::with_capacity(res.atom_sns.len());

        for sn in &res.atom_sns {
            match atom_sns_to_indices(*sn, atom_set) {
                Some(i) => atoms.push(i),
                None => {
                    return Err(io::Error::new(
                        ErrorKind::InvalidData,
                        "Unable to find atom SN when loading from generic res",
                    ));
                }
            }
        }

        Ok(Self {
            serial_number: res.serial_number,
            res_type: res.res_type.clone(),
            atom_sns: res.atom_sns.clone(),
            atoms,
            dihedral: None,
            end,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Chain {
    pub id: String,
    // todo: Do we want both residues and atoms stored here? It's an overconstraint.
    /// Serial number
    pub residue_sns: Vec<u32>,
    pub residues: Vec<usize>,
    /// Serial number
    pub atom_sns: Vec<u32>,
    pub atoms: Vec<usize>,
    pub visible: bool,
}

impl Chain {
    fn from_generic(
        chain: &ChainGeneric,
        atom_set: &[Atom],
        res_set: &[Residue],
    ) -> io::Result<Self> {
        // todo: DRY with res code above.
        let mut atoms = Vec::with_capacity(chain.atom_sns.len());
        let mut residues = Vec::with_capacity(chain.residue_sns.len());

        for sn in &chain.atom_sns {
            match atom_sns_to_indices(*sn, atom_set) {
                Some(i) => atoms.push(i),
                None => {
                    return Err(io::Error::new(
                        ErrorKind::InvalidData,
                        "Unable to find atom SN when loading from generic res",
                    ));
                }
            }
        }

        for sn in &chain.residue_sns {
            match res_sns_to_indices(*sn, res_set) {
                Some(i) => residues.push(i),
                None => {
                    return Err(io::Error::new(
                        ErrorKind::InvalidData,
                        "Unable to find res SN when loading from generic res",
                    ));
                }
            }
        }

        Ok(Self {
            id: chain.id.clone(),
            residue_sns: chain.residue_sns.clone(),
            residues,
            atom_sns: chain.atom_sns.clone(),
            atoms,
            visible: true,
        })
    }

    pub fn to_generic(&self) -> ChainGeneric {
        ChainGeneric {
            id: self.id.clone(),
            residue_sns: self.residue_sns.clone(),
            atom_sns: self.atom_sns.clone(),
        }
    }
}

/// Helper
fn atom_sns_to_indices(sn_tgt: u32, atom_set: &[Atom]) -> Option<usize> {
    for (i, atom) in atom_set.iter().enumerate() {
        if atom.serial_number == sn_tgt {
            return Some(i);
        }
    }

    None
}

/// Helper, and dry
fn res_sns_to_indices(sn_tgt: u32, res_set: &[Residue]) -> Option<usize> {
    for (i, atom) in res_set.iter().enumerate() {
        if atom.serial_number == sn_tgt {
            return Some(i);
        }
    }

    None
}

impl fmt::Display for Residue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match &self.res_type {
            ResidueType::AminoAcid(aa) => aa.to_string(),
            ResidueType::Water => "Water".to_owned(),
            ResidueType::Other(name) => name.clone(),
        };

        write!(f, "#{}: {}", self.serial_number, name)?;

        if let Some(dihedral) = &self.dihedral {
            write!(f, "   {}", dihedral)?;
        }

        match self.end {
            ResidueEnd::CTerminus => write!(f, " C-term")?,
            ResidueEnd::NTerminus => write!(f, " N-term")?,
            _ => (),
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub struct Atom {
    pub serial_number: u32,
    pub posit: Vec3,
    pub element: Element,
    /// e.g. "HA", "C", "N", "HB3" etc.
    pub type_in_res: Option<AtomTypeInRes>,
    /// "Type 2" for proteins/AA. For ligands and small molecules, this
    /// is a "Type 3".
    /// E.g. "c6", "ca", "n3", "ha", "h0" etc, as seen in Mol2 files from AMBER.
    pub force_field_type: Option<String>,
    // todo: Review what DockType does.
    /// todo: Consider a substruct for docking fields.
    pub dock_type: Option<DockType>,
    // todo: Note that `role` is a subset of `type_in_res`.
    pub role: Option<AtomRole>,
    /// We include these references to the residue and chain indices for speed; iterating through
    /// residues (or chains) to check for atom membership is slow.
    pub residue: Option<usize>,
    pub chain: Option<usize>,
    pub hetero: bool,
    /// For docking.
    pub occupancy: Option<f32>,
    pub partial_charge: Option<f32>,
    pub temperature_factor: Option<f32>,
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
            type_in_res: self.type_in_res.clone(),
            posit: self.posit,
            element: self.element,
            partial_charge: self.partial_charge,
            force_field_type: self.force_field_type.clone(),
            ..Default::default()
        }
    }
}

impl From<&AtomGeneric> for Atom {
    fn from(atom: &AtomGeneric) -> Self {
        let role = match &atom.type_in_res {
            Some(tir) => Some(AtomRole::from_type_in_res(tir)),
            None => None,
        };

        // We will fill out chain and residue later, after chains and residue are loaded.

        Self {
            serial_number: atom.serial_number,
            posit: atom.posit,
            element: atom.element,
            type_in_res: atom.type_in_res.clone(),
            role,
            partial_charge: atom.partial_charge,
            force_field_type: atom.force_field_type.clone(),
            hetero: atom.hetero,
            ..Default::default()
        }
    }
}

impl Display for Atom {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let ff_type = match &self.force_field_type {
            Some(f) => f,
            None => "None",
        };

        let q = match &self.partial_charge {
            Some(q_) => format!("{q_:.3}"),
            None => "None".to_string(),
        };

        write!(
            f,
            "Atom {}: {}, {}. {:?}, ff: {ff_type}, q: {q}",
            self.serial_number,
            self.element.to_letter(),
            self.posit,
            self.type_in_res,
        )?;

        if self.hetero {
            write!(f, ", Het");
        }

        Ok(())
    }
}

/// Can't find a PyMol equiv. Experimenting
pub const fn aa_color(aa: AminoAcid) -> (f32, f32, f32) {
    match aa {
        AminoAcid::Arg => (0.7, 0.2, 0.9),
        AminoAcid::His => (0.2, 1., 0.2),
        AminoAcid::Lys => (1., 0.3, 0.3),
        AminoAcid::Asp => (0.4, 0.4, 1.0),
        AminoAcid::Glu => (0.701, 0.7, 0.2),
        AminoAcid::Ser => (177. / 255., 187. / 255., 161. / 255.),
        AminoAcid::Thr => (1.0, 0.502, 0.),
        AminoAcid::Asn => (0.878, 0.4, 0.2),
        AminoAcid::Gln => (0.784, 0.502, 0.2),
        AminoAcid::Cys => (0.239, 1.0, 0.),
        AminoAcid::Sec => (0.561, 0.251, 0.831),
        AminoAcid::Gly => (0.749, 0.651, 0.651),
        AminoAcid::Pro => (0.74, 0.6, 0.4),
        AminoAcid::Ala => (1., 0.820, 0.137),
        AminoAcid::Val => (0.753, 0.753, 0.753),
        AminoAcid::Ile => (0.322, 0.722, 0.716),
        AminoAcid::Leu => (0.5, 0.702, 0.602),
        AminoAcid::Met => (0.490, 0.502, 0.690),
        AminoAcid::Phe => (0.780, 0., 0.780),
        AminoAcid::Tyr => (0.541, 1., 0.),
        AminoAcid::Trp => (0.121, 0.941, 0.121),
    }
}

impl Molecule {
    // impl TryFrom<MmCif> for Molecule {
    //     type Error = io::Error;

    pub fn from_mmcif(m: MmCif, ff_map: &ProtFfMap) -> Result<Self, io::Error> {
        // fn try_from(m: MmCif) -> Result<Self, Self::Error> {
        let mut atoms: Vec<_> = m.atoms.iter().map(|a| a.into()).collect();

        // todo: Crude logic for finding the C terminus. Relies on atom position,
        // todo, and dodens't take chains into account. (Which we may need to?)
        // todo: Also assumes all non-het are listed prior to het.
        let mut last_non_het = 0;
        for (i, res) in m.residues.iter().enumerate() {
            match res.res_type {
                ResidueType::AminoAcid(_) => last_non_het = i,
                _ => break,
            }
        }

        // todo: Check out the below logic; RustRover is greying out the last_non_het arm,
        // todo and saing the _ arm is unreachable.

        let mut residues = Vec::with_capacity(m.residues.len());
        for (i, res) in m.residues.iter().enumerate() {
            let mut end = ResidueEnd::Internal;

            // Match arm won't work due to non-constant arms, e.g. non_hetero?
            if i == 0 {
                end = ResidueEnd::NTerminus;
            } else if i == last_non_het {
                end = ResidueEnd::CTerminus;
            }

            match res.res_type {
                ResidueType::AminoAcid(_) => (),
                _ => end = ResidueEnd::Hetero,
            }

            residues.push(Residue::from_generic(res, &atoms, end)?);
        }

        // Populate the residue end.

        let mut chains = Vec::with_capacity(m.chains.len());
        for c in &m.chains {
            chains.push(Chain::from_generic(c, &atoms, &residues)?);
        }

        // Now that chains and residues are loaded, update atoms with their back-ref index.
        for atom in &mut atoms {
            for (i, res) in residues.iter().enumerate() {
                if res.atom_sns.contains(&atom.serial_number) {
                    atom.residue = Some(i);
                    break;
                }
            }

            for (i, chain) in chains.iter().enumerate() {
                if chain.atom_sns.contains(&atom.serial_number) {
                    atom.chain = Some(i);
                    break;
                }
            }
        }

        let mut result = Self::new(
            m.ident.clone(),
            atoms,
            chains,
            residues,
            None,
            None,
            Some(ff_map),
        );

        result.experimental_method = m.experimental_method.clone();
        result.secondary_structure = m.secondary_structure.clone();

        Ok(result)
    }
}

impl TryFrom<Mol2> for Molecule {
    type Error = io::Error;
    fn try_from(m: Mol2) -> Result<Self, Self::Error> {
        let atoms: Vec<_> = m.atoms.iter().map(|a| a.into()).collect();

        let bonds: Vec<Bond> = m
            .bonds
            .iter()
            .map(|b| Bond::from_generic(b, &atoms))
            .collect::<Result<_, _>>()?;

        let mut result = Self::new(m.ident, atoms, Vec::new(), Vec::new(), None, None, None);

        // This replaces the built-in bond computation with our own. Ideally, we don't even calculate
        // those for performance reasons.
        result.bonds = bonds;
        result.bonds_hydrogen = Vec::new();
        result.adjacency_list = build_adjacency_list(&result.bonds, result.atoms.len());

        Ok(result)
    }
}

impl TryFrom<Sdf> for Molecule {
    type Error = io::Error;
    fn try_from(m: Sdf) -> Result<Self, Self::Error> {
        let atoms: Vec<_> = m.atoms.iter().map(|a| a.into()).collect();

        let mut residues = Vec::with_capacity(m.residues.len());
        for res in &m.residues {
            residues.push(Residue::from_generic(res, &atoms, ResidueEnd::Hetero)?);
        }

        let mut chains = Vec::with_capacity(m.chains.len());
        for c in &m.chains {
            chains.push(Chain::from_generic(c, &atoms, &residues)?);
        }

        let bonds: Vec<Bond> = m
            .bonds
            .iter()
            .map(|b| Bond::from_generic(b, &atoms))
            .collect::<Result<_, _>>()?;

        let mut result = Self::new(m.ident, atoms, chains, residues, None, None, None);

        // See note in Mol2's method.
        result.bonds = bonds;
        result.bonds_hydrogen = Vec::new();
        result.adjacency_list = build_adjacency_list(&result.bonds, result.atoms.len());

        Ok(result)
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
        let chains = self.chains.iter().map(|c| c.to_generic()).collect();

        Sdf {
            ident: self.ident.clone(),
            atoms,
            bonds,
            chains,
            residues,
            metadata: HashMap::new(), // todo?
            pubchem_cid: self.pubchem_cid,
            drugbank_id: self.drugbank_id.clone(),
        }
    }
}

/// Build a list of, for each atom, all atoms bonded to it.
/// We use this as part of our flexible-bond conformation algorithm, and in setting up
/// angles and dihedrals for molecular docking.
pub fn build_adjacency_list(bonds: &[Bond], atoms_len: usize) -> Vec<Vec<usize>> {
    let mut result = vec![Vec::new(); atoms_len];

    // For each bond, record its atoms as neighbors of each other
    for bond in bonds {
        result[bond.atom_0].push(bond.atom_1);
        result[bond.atom_1].push(bond.atom_0);
    }

    result
}

#[derive(Clone, Copy, PartialEq, Default)]
pub enum PeptideAtomPosits {
    #[default]
    /// E.g. as imported from a mmCIF file, from experimental data
    Original,
    /// As calculated in a snapshot from a MD sim
    Dynamics,
}

impl Display for PeptideAtomPosits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let val = match self {
            Self::Original => "Original",
            Self::Dynamics => "Dynamics",
        };
        write!(f, "{val}")
    }
}

impl Molecule {
    /// For example, this can be used to create a ligand from a residue that was loaded with a mmCIF
    /// file from RCSB. It can then be used for docking, or saving to a Mol2 or SDF file.
    ///
    /// `atoms` here should be the full set, as indexed by `res`, unless `use_sns` is true.
    /// `use_sns` = false is faster.
    ///
    /// We assume the residue is already populate with hydrogens.
    pub fn from_res(res: &Residue, atoms: &[Atom], use_sns: bool) -> Self {
        let atoms_this = if use_sns {
            unimplemented!()
        } else {
            res.atoms.iter().map(|i| atoms[*i].clone()).collect()
        };

        Self::new(
            res.res_type.to_string(),
            atoms_this,
            // No chains, residues, or pubchem/drugbank identifiers.
            Vec::new(),
            Vec::new(),
            None,
            None,
            None,
        )
    }
}
