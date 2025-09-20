#![allow(non_camel_case_types)]

//! Contains data structures and related code for molecules, atoms, residues, chains, etc.
use std::{
    collections::HashMap,
    fmt::{self, Display, Formatter},
    io,
    io::ErrorKind,
    path::PathBuf,
    sync::mpsc::{self, Receiver},
    thread,
};

use bio_apis::{
    ReqError, rcsb,
    rcsb::{FilesAvailable, PdbDataResults},
};
use bio_files::{
    AtomGeneric, BackboneSS, BondGeneric, BondType, ChainGeneric, DensityMap, ExperimentalMethod,
    MmCif, ResidueEnd, ResidueGeneric, ResidueType,
};
use dynamics::{
    Dihedral, ParamError, ProtFFTypeChargeMap,
    params::{populate_peptide_ff_and_q, prepare_peptide_mmcif},
    populate_hydrogens_dihedrals,
};
use lin_alg::{f32::Vec3 as Vec3F32, f64::Vec3};
use na_seq::{AminoAcid, AminoAcidGeneral, AminoAcidProtenationVariant, AtomTypeInRes, Element};
use rayon::prelude::*;

use crate::{
    Selection,
    bond_inference::create_hydrogen_bonds,
    mol_lig::MoleculeSmall,
    nucleic_acid::MoleculeNucleicAcid,
    reflection::{DensityRect, ElectronDensity, ReflectionsData},
    util::{handle_err, mol_center_size},
};

/// Contains fields shared by all molecule types.
#[derive(Debug, Clone)]
pub struct MoleculeCommon {
    pub ident: String,
    pub atoms: Vec<Atom>,
    pub bonds: Vec<Bond>,
    /// Relating covalent bonds. For each atom, a list of atoms bonded to it.
    pub adjacency_list: Vec<Vec<usize>>,
    /// For repositioning atoms, e.g. from dynamics or absolute positioning.
    ///
    /// Absolute atom positions. For absolute conformation type[s], these positions are set and accessed directly, e.g., by MD
    /// simulations. We leave the molecule atom positions as ingested directly from data files. (e.g., relative positions).
    /// For rigid and semi-rigid conformations, these are derivative of the pose, in conjunction with
    /// the molecule atoms' (relative) positions.
    pub atom_posits: Vec<Vec3>,
    pub metadata: HashMap<String, String>,
    /// This is a bit different, as it's for our UI only. Doesn't fit with the others,
    /// but is safer and easier than trying to sync Vec indices.
    pub visible: bool,
    pub path: Option<PathBuf>,
    pub selected_for_md: bool,
}

impl Default for MoleculeCommon {
    /// Only so we can set visible: true.
    fn default() -> Self {
        Self {
            ident: String::new(),
            metadata: HashMap::new(),
            atoms: Vec::new(),
            bonds: Vec::new(),
            adjacency_list: Vec::new(),
            atom_posits: Vec::new(),
            visible: true,
            path: None,
            selected_for_md: false,
        }
    }
}

impl MoleculeCommon {
    /// If `bonds` is none, create it based on atom distances. Useful in the case of mmCIF files,
    /// which usually lack bond information.
    ///
    /// Hydrogens should have been added to the atom set, if required, prior to running this,
    /// so bonds are created.
    pub fn new(
        ident: String,
        atoms: Vec<Atom>,
        bonds: Vec<Bond>,
        metadata: HashMap<String, String>,
        path: Option<PathBuf>,
    ) -> Self {
        let atom_posits = atoms.iter().map(|a| a.posit).collect();

        let mut result = Self {
            ident,
            metadata,
            atoms,
            bonds,
            atom_posits,
            path,
            ..Self::default()
        };

        result.build_adjacency_list();
        result
    }

    /// Build a list of, for each atom, all atoms bonded to it.
    /// We use this as part of our flexible-bond conformation algorithm, and in setting up
    /// angles and dihedrals for molecular docking.
    ///
    /// Run this after populate hydrogens.
    pub fn build_adjacency_list(&mut self) {
        self.adjacency_list = build_adjacency_list(&self.bonds, self.atoms.len());
    }
}

#[derive(Debug)]
pub enum MoleculeGeneric {
    Peptide(MoleculePeptide),
    Ligand(MoleculeSmall),
    NucleicAcid(MoleculeNucleicAcid),
}

impl MoleculeGeneric {
    pub fn _common(&self) -> &MoleculeCommon {
        match self {
            Self::Peptide(m) => &m.common,
            Self::Ligand(m) => &m.common,
            Self::NucleicAcid(m) => &m.common,
        }
    }

    pub fn common_mut(&mut self) -> &mut MoleculeCommon {
        match self {
            Self::Peptide(m) => &mut m.common,
            Self::Ligand(m) => &mut m.common,
            Self::NucleicAcid(m) => &mut m.common,
        }
    }
}

/// We currently use this for mol description.
#[derive(Debug)]
pub enum MoleculeGenericRef<'a> {
    Peptide(&'a MoleculePeptide),
    Ligand(&'a MoleculeSmall),
    NucleicAcid(&'a MoleculeNucleicAcid),
}

impl<'a> MoleculeGenericRef<'a> {
    pub fn common(&self) -> &MoleculeCommon {
        match self {
            Self::Peptide(m) => &m.common,
            Self::Ligand(m) => &m.common,
            Self::NucleicAcid(m) => &m.common,
        }
    }
}

/// A polypeptide molecule, e.g. a protein.
#[derive(Debug, Default, Clone)]
pub struct MoleculePeptide {
    pub common: MoleculeCommon,
    pub bonds_hydrogen: Vec<HydrogenBond>,
    pub chains: Vec<Chain>,
    pub residues: Vec<Residue>,
    /// We currently use this for aligning ligands to CIF etc data, where they may already be included
    /// in a protein/ligand complex as hetero atoms.
    pub het_residues: Vec<Residue>,
    /// Solvent-accessible surface. Used as one of our visualization methods.
    /// Current structure is a Vec of rings.
    /// Initializes to empty; updated A/R when the appropriate view is selected.
    pub sa_surface_pts: Option<Vec<Vec<Vec3F32>>>,
    pub secondary_structure: Vec<BackboneSS>,
    /// Center and size are used for lighting, and for rotating ligands.
    pub center: Vec3,
    pub size: f32,
    /// The full (Or partial while WIP) results from the RCSB data api.
    pub rcsb_data: Option<PdbDataResults>,
    pub rcsb_files_avail: Option<FilesAvailable>,
    pub reflections_data: Option<ReflectionsData>,
    /// E.g. from a MAP file, MTX, or 2fo-fc header.
    pub elec_density: Option<Vec<ElectronDensity>>,
    pub density_map: Option<DensityMap>,
    pub density_rect: Option<DensityRect>,
    pub aa_seq: Vec<AminoAcid>,
    pub experimental_method: Option<ExperimentalMethod>,
    // pub ff_params: Option<ForceFieldParamsIndexed>,
}

impl MoleculePeptide {
    /// This constructor handles assumes details are ingested into a common format upstream. It adds
    /// them to the resulting structure, and augments it with bonds, hydrogen positions, and other things A/R.
    pub fn new(
        ident: String,
        atoms: Vec<Atom>,
        bonds: Vec<Bond>,
        chains: Vec<Chain>,
        residues: Vec<Residue>,
        metadata: HashMap<String, String>,
        path: Option<PathBuf>,
    ) -> Self {
        let (center, size) = mol_center_size(&atoms);

        println!("Loading atoms into mol...");

        let mut result = Self {
            // We create bonds only after
            common: MoleculeCommon::new(ident, atoms, bonds, metadata, path),
            chains,
            residues,
            center,
            size,
            ..Default::default()
        };

        result.aa_seq = result.get_seq();
        result.bonds_hydrogen = create_hydrogen_bonds(&result.common.atoms, &result.common.bonds);

        // Override the one set in Common::new(), now that we've added hydrogens.
        result.common.build_adjacency_list();

        for res in &result.residues {
            if let ResidueType::Other(_) = &res.res_type {
                if res.atoms.len() >= 10 {
                    result.het_residues.push(res.clone());
                }
            }
        }

        for atom in &mut result.common.atoms {
            // println!("{:?}", atom.role);
            //     // This is redundant; but can serve as a cache.
            //     if let Some(role) = &atom.role {
            //         if matches!(role, AtomRole::C_Alpha | AtomRole::C_Prime | AtomRole::N_Backbone | AtomRole::O_Backbone) {
            //             println!("HET");
            //             atom.is = true;
            //         }
            //     }
        }

        result
    }

    /// If a residue, get the alpha C. If multiple, get an arbitrary one.
    pub fn get_sel_atom(&self, sel: &Selection) -> Option<&Atom> {
        match sel {
            Selection::Atom(i) => self.common.atoms.get(*i),
            Selection::AtomLig((lig_i, atom_i)) => None,
            Selection::Residue(i) => {
                let res = &self.residues[*i];
                if !res.atoms.is_empty() {
                    for atom_i in &res.atoms {
                        let atom = &self.common.atoms[*atom_i];
                        if let Some(role) = atom.role {
                            if role == AtomRole::C_Alpha {
                                return Some(atom);
                            }
                        }
                    }

                    // If we can't find  C alpha, default to the first atom.
                    Some(&self.common.atoms[res.atoms[0]])
                } else {
                    None
                }
            }
            Selection::Atoms(is) => {
                // todo temp?
                self.common.atoms.get(is[0])
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

        let ident = self.common.ident.clone(); // data the worker needs
        let (tx, rx) = mpsc::channel(); // one-shot channel

        println!("Getting RCSB data...");

        thread::spawn(move || {
            let data = rcsb::get_all_data(&ident);
            let files_data = rcsb::get_files_avail(&ident);

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
            match rx.try_recv() {
                Ok((Ok(pdb_data), Ok(files_avail))) => {
                    println!("RCSB data ready for {}", self.common.ident);
                    self.rcsb_data = Some(pdb_data);
                    self.rcsb_files_avail = Some(files_avail);

                    // todo: Save state here, but need to get a proper signal with &mut State available.

                    *pending_data_avail = None;
                    return true;
                }

                // PdbDataResults failed, but FilesAvailable might not have been sent:
                Ok((Err(e), _)) => {
                    eprintln!("Failed to fetch PDB data for {}: {e:?}", self.common.ident);
                    *pending_data_avail = None;
                }

                // FilesAvailable failed (even if PdbDataResults succeeded):
                Ok((_, Err(e))) => {
                    eprintln!("Failed to fetch file‐list for {}: {e:?}", self.common.ident);
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
            bond_type: self.bond_type,
            atom_0_sn: self.atom_0_sn,
            atom_1_sn: self.atom_1_sn,
        }
    }
}

impl Bond {
    pub fn from_generic(bond: &BondGeneric, atom_set: &[Atom]) -> io::Result<Self> {
        let atom_0 = match atom_sns_to_indices(bond.atom_0_sn, atom_set) {
            Some(i) => i,
            None => {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    "Unable to find atom SN when loading from generic res",
                ));
            }
        };

        // todo DRY
        let atom_1 = match atom_sns_to_indices(bond.atom_1_sn, atom_set) {
            Some(i) => i,
            None => {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    "Unable to find atom SN when loading from generic res",
                ));
            }
        };

        Ok(Self {
            bond_type: bond.bond_type,
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
            end: self.end,
        }
    }
}

impl Residue {
    pub fn from_generic(res: &ResidueGeneric, atom_set: &[Atom]) -> io::Result<Self> {
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
            end: res.end,
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
    pub fn from_generic(
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

        // We get this empty set with some small (or old?) residues from PDB, e.g. 1BOM.
        if !res_set.is_empty() {
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

impl Display for Residue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}: {}", self.serial_number, self.res_type)?;

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
    // pub dock_type: Option<DockType>,
    // todo: Note that `role` is a subset of `type_in_res`.
    pub role: Option<AtomRole>,
    /// We include these references to the residue and chain indices for speed; iterating through
    /// residues (or chains) to check for atom membership is slow.
    pub residue: Option<usize>,
    pub chain: Option<usize>,
    pub hetero: bool,
    /// For docking.
    pub occupancy: Option<f32>,
    /// Elementary charge. (Charge of a proton)
    pub partial_charge: Option<f32>,
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
            write!(f, ", Het")?;
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

impl MoleculePeptide {
    pub fn from_mmcif(
        mut m: MmCif,
        ff_map: &ProtFFTypeChargeMap,
        path: Option<PathBuf>,
        ph: f32,
    ) -> Result<Self, io::Error> {
        // todo: Perhaps you still want to calculate dihedral angles if hydrogens are populated already.
        // todo; For now, you are skipping both. Example when this comes up: Ligands.
        // Attempt to only populate Hydrogens if there aren't many.

        // Add hydrogens, FF types, partial charge, and bonds.
        let (bonds_, dihedrals) = prepare_peptide_mmcif(&mut m, ff_map, ph)
            .map_err(|e| io::Error::new(ErrorKind::InvalidData, e.descrip))?;

        let mut atoms: Vec<_> = m.atoms.iter().map(|a| a.into()).collect();

        let mut bonds = Vec::with_capacity(bonds_.len());
        for bond in &bonds_ {
            bonds.push(Bond::from_generic(bond, &atoms)?);
        }

        let mut residues = Vec::with_capacity(m.residues.len());

        if dihedrals.len() == m.residues.len() {
            for (i, res) in m.residues.iter().enumerate() {
                let mut res = Residue::from_generic(res, &atoms)?;
                res.dihedral = Some(dihedrals[i].clone());
                residues.push(res);
            }
        } else {
            eprintln!("Error: Problem generating dihedrals.");
        }

        let mut chains = Vec::with_capacity(m.chains.len());
        for c in &m.chains {
            chains.push(Chain::from_generic(c, &atoms, &residues)?);
        }

        // Now that chains and residues are loaded, update atoms with their back-ref index.
        for atom in &mut atoms {
            for (i, res) in residues.iter().enumerate() {
                if res.atom_sns.contains(&atom.serial_number) {
                    atom.residue = Some(i);

                    // Update which atoms are waters.
                    if residues[i].res_type == ResidueType::Water {
                        atom.role = Some(AtomRole::Water);
                    }

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
            bonds,
            chains,
            residues,
            m.metadata,
            path,
        );

        result.experimental_method = m.experimental_method.clone();
        result.secondary_structure = m.secondary_structure.clone();

        for bond in &mut result.common.bonds {
            if result.common.atoms[bond.atom_0].is_backbone()
                && result.common.atoms[bond.atom_1].is_backbone()
            {
                bond.is_backbone = true;
            }
        }

        Ok(result)
    }
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

pub fn build_adjacency_list(bonds: &Vec<Bond>, atoms_len: usize) -> Vec<Vec<usize>> {
    let mut result = vec![Vec::new(); atoms_len];

    // For each bond, record its atoms as neighbors of each other
    for bond in bonds {
        result[bond.atom_0].push(bond.atom_1);
        result[bond.atom_1].push(bond.atom_0);
    }

    result
}

// todo! This is a C+P from that in dynamics, but using our native types. Sort this, and how
// todo you populate H, FF type, and Q in ggeneral between here and the lib.
/// Populate forcefield type, and partial charge on atoms. This should be run on mmCIF
/// files prior to running molecular dynamics on them. These files from RCSB PDB do not
/// natively have this data.
///
/// `residues` must be the full set; this is relevant to how we index it.
/// Populate forcefield type, and partial charge.
/// `residues` must be the full set; this is relevant to how we index it.
pub fn __populate_ff_and_q(
    atoms: &mut [Atom],
    residues: &[Residue],
    ff_type_charge: &ProtFFTypeChargeMap,
) -> Result<(), ParamError> {
    for atom in atoms {
        if atom.hetero {
            continue;
        }

        let Some(res_i) = atom.residue else {
            return Err(ParamError::new(&format!(
                "MD failure: Missing residue when populating ff name and q: {atom}"
            )));
        };

        let Some(type_in_res) = &atom.type_in_res else {
            return Err(ParamError::new(&format!(
                "MD failure: Missing type in residue for atom: {atom}"
            )));
        };

        let atom_res_type = &residues[res_i].res_type;

        let ResidueType::AminoAcid(aa) = atom_res_type else {
            // e.g. water or other hetero atoms; skip.
            continue;
        };

        // todo: Eventually, determine how to load non-standard AA variants from files; set up your
        // todo state to use those labels. They are available in the params.
        let aa_gen = AminoAcidGeneral::Standard(*aa);

        let charge_map = match residues[res_i].end {
            ResidueEnd::Internal => &ff_type_charge.internal,
            ResidueEnd::NTerminus => &ff_type_charge.n_terminus,
            ResidueEnd::CTerminus => &ff_type_charge.c_terminus,
            ResidueEnd::Hetero => {
                return Err(ParamError::new(&format!(
                    "Error: Encountered hetero atom when parsing amino acid FF types: {atom}"
                )));
            }
        };

        let charges = match charge_map.get(&aa_gen) {
            Some(c) => c,
            // A specific workaround to plain "HIS" being absent from amino19.lib (2025.
            // Choose one of "HID", "HIE", "HIP arbitrarily.
            // todo: Re-evaluate this, e.g. which one of the three to load.
            None if aa_gen == AminoAcidGeneral::Standard(AminoAcid::His) => charge_map
                .get(&AminoAcidGeneral::Variant(AminoAcidProtenationVariant::Hid))
                .ok_or_else(|| ParamError::new("Unable to find AA mapping"))?,
            None => return Err(ParamError::new("Unable to find AA mapping")),
        };

        let mut found = false;

        for charge in charges {
            // todo: Note that we have multiple branches in some case, due to Amber names like
            // todo: "HYP" for variants on AAs for different protenation states. Handle this.
            if &charge.type_in_res == type_in_res {
                atom.force_field_type = Some(charge.ff_type.clone());
                atom.partial_charge = Some(charge.charge);

                found = true;
                break;
            }
        }

        // Code below is mainly for the case of missing data; otherwise, the logic for this operation
        // is complete.

        if !found {
            match type_in_res {
                // todo: This is a workaround for having trouble with H types. LIkely
                // todo when we create them. For now, this meets the intent.
                AtomTypeInRes::H(_) => {
                    // Note: We've witnessed this due to errors in the mmCIF file, e.g. on ASP #88 on 9GLS.
                    eprintln!(
                        "Error assigning FF type and q based on atom type in res: Failed to match H type. #{}, {type_in_res}, {aa_gen:?}. \
                         Falling back to a generic H",
                        &residues[res_i].serial_number
                    );

                    for charge in charges {
                        if &charge.type_in_res == &AtomTypeInRes::H("H".to_string())
                            || &charge.type_in_res == &AtomTypeInRes::H("HA".to_string())
                        {
                            atom.force_field_type = Some("HB2".to_string());
                            atom.partial_charge = Some(charge.charge);

                            found = true;
                            break;
                        }
                    }
                }
                // // This is an N-terminal oxygen of a C-terminal carboxyl group.
                // // todo: You should parse `aminoct12.lib`, and `aminont12.lib`, then delete this.
                // AtomTypeInRes::OXT => {
                //     match atom_res_type {
                //         // todo: QC that it's the N-terminal Met too, or return an error.
                //         ResidueType::AminoAcid(AminoAcid::Met) => {
                //             atom.force_field_type = Some("O2".to_owned());
                //             // Fm amino12ct.lib
                //             atom.partial_charge = Some(-0.804100);
                //             found = true;
                //         }
                //         _ => return Err(ParamError::new("Error populating FF type: OXT atom-in-res type,\
                //         not at the C terminal")),
                //     }
                // }
                _ => (),
            }

            // i.e. if still not found after our specific workarounds above.
            if !found {
                return Err(ParamError::new(&format!(
                    "Error assigning FF type and q based on atom type in res: {atom}",
                )));
            }
        }
    }

    Ok(())
}
