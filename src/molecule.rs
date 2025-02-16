/// Contains data structures and related code for molecules, atoms, residues, chains, etc.
use std::str::FromStr;
use std::{collections::HashMap, fmt};

use lin_alg::f64::Vec3;
use na_seq::AminoAcid;
use pdbtbx::PDB;
use rayon::prelude::*;

use crate::{bond_inference::create_bonds, rcsb_api::PdbMetaData, Element, Selection};

#[derive(Debug)]
// todo: This, or a PDB-specific format?
pub struct Molecule {
    pub ident: String,
    pub atoms: Vec<Atom>,
    pub bonds: Vec<Bond>,
    pub chains: Vec<Chain>,
    pub residues: Vec<Residue>,
    pub metadata: Option<PdbMetaData>,
}

impl Molecule {
    pub fn from_pdb(pdb: &PDB) -> Self {
        // todo: Maybe return the PDB type here, and store that. Also have a way to
        // todo get molecules from it

        // todo: Pdbtbx doesn't implm this yet for CIF.
        for remark in pdb.remarks() {
            // println!("Remark: {remark:?}");
        }

        println!("Loading atoms...");
        let atoms_pdb: Vec<&pdbtbx::Atom> = pdb.par_atoms().collect();

        println!("Gather residues...");
        let res_pdb: Vec<&pdbtbx::Residue> = pdb.par_residues().collect();

        let mut residues: Vec<Residue> = pdb
            .par_residues()
            .map(|res| Residue::from_pdb(res, &atoms_pdb))
            .collect();

        residues.sort_by_key(|r| r.serial_number);

        println!("Setting up chains...");

        let mut chains = Vec::with_capacity(pdb.chain_count());
        for chain_pdb in pdb.chains() {
            // println!("Chain: {chain_pdb:?}");

            let mut chain = Chain {
                id: chain_pdb.id().to_owned(),
                atoms: Vec::with_capacity(chain_pdb.atom_count()),
                residues: Vec::with_capacity(chain_pdb.residue_count()),
                visible: true,
            };

            for atom_c in chain_pdb.atoms() {
                let atom_pdb = atoms_pdb
                    .iter()
                    .enumerate()
                    .find(|(i, a)| a.serial_number() == atom_c.serial_number());
                if let Some((i, _atom)) = atom_pdb {
                    chain.atoms.push(i);
                }
            }

            // We don't have a way to, using serial numbers alone, using PDBTBX, find which residues are associated with
            // which chain. This method is a bit more indirect, using both serial number, and atom indexes.
            for res_c in chain_pdb.residues() {
                for (i, res) in residues.iter().enumerate() {
                    if res.serial_number == res_c.serial_number() {
                        let atom_sns_chain: Vec<usize> =
                            res_c.atoms().map(|a| a.serial_number()).collect();
                        // let atom_sns_res: Vec<usize> = res.atoms.iter().map(|a| a.serial_number).collect();
                        let mut atom_sns_res = Vec::with_capacity(res.atoms.len());
                        for atom_i in &res.atoms {
                            atom_sns_res.push(atoms_pdb[*atom_i].serial_number());
                        }

                        // println!("Atoms 1: {:?}", atom_sns_chain);
                        // println!("Atoms 2: {:?}\n", atom_sns_res);

                        if atom_sns_chain == atom_sns_res {
                            chain.residues.push(i);
                        }
                    }
                }

                // let res = residues
                //     .iter()
                //     .enumerate()
                //     .find(|(i, r)| r.serial_number == res_c.serial_number() && r.atoms );
                // if let Some((i, _res)) = res {
                //     chain.residues.push(i);
                // }
            }

            // println!("Chain: {}, {:?}", chain.id, chain.residues);

            chains.push(chain);
        }

        println!("Atoms final...");

        // This pre-computation of the AA map is more efficient;
        let mut aa_map = HashMap::new();
        for res in &residues {
            for atom_i in &res.atoms {
                aa_map.insert(*atom_i, res.res_type.clone());
            }
        }

        // todo: This is taking a while.
        let atoms: Vec<Atom> = atoms_pdb
            .into_iter()
            .enumerate()
            .map(|(i, atom)| Atom::from_pdb(atom, i, &aa_map))
            .collect();

        println!("Complete.");

        // todo: We use our own bond inference, since most PDBs seem to lack bond information.
        // let mut bonds = Vec::new();
        // for (a0, a1, bond) in pdb.bonds() {
        //     bonds.push((Atom::from_pdb(a0), Atom::from_pdb(a1), bond));
        // }

        let bonds = create_bonds(&atoms);

        Molecule {
            ident: pdb.identifier.clone().unwrap_or_default(),
            atoms,
            bonds,
            chains,
            residues,
            metadata: None,
        }
    }

    /// If residue, get an arbitrary atom. (todo: Get c alpha always).
    pub fn get_sel_atom(&self, sel: Selection) -> Option<&Atom> {
        match sel {
            Selection::Atom(i) => {
                if i < self.atoms.len() {
                    Some(&self.atoms[i])
                } else {
                    None
                }
            }
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
    Other,
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
            AtomRole::Other => write!(f, "Other"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum BondType {
    // C+P from pdbtbx for now
    Covalent,
    Disulfide,
    Hydrogen,
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
    Double,
    Triple,
    SingleDoubleHybrid,
}

#[derive(Debug)]
pub struct Bond {
    pub bond_type: BondType,
    pub bond_count: BondCount,
    /// Index
    pub atom_0: usize,
    /// Index
    pub atom_1: usize,
    pub is_backbone: bool,
}

pub struct Ligand {}

#[derive(Debug)]
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

#[derive(Debug)]
pub struct Residue {
    /// We use serial number of display, search etc, and array index to select. Residue serial number is not
    /// unique in the molecule; only in the chain.
    pub serial_number: isize, // pdbtbx uses isize. Negative allowed?
    pub res_type: ResidueType,
    pub atoms: Vec<usize>, // Atom index
}

impl Residue {
    pub fn from_pdb(res_pdb: &pdbtbx::Residue, atoms_pdb: &[&pdbtbx::Atom]) -> Self {
        let res_name = res_pdb.name().unwrap_or_default();

        let res_type = if res_name.to_uppercase() == "HOH" {
            ResidueType::Water
        } else {
            match AminoAcid::from_str(res_name) {
                Ok(aa) => ResidueType::AminoAcid(aa),
                Err(_) => ResidueType::Other(res_pdb.name().unwrap_or_default().to_owned()),
            }
        };

        let mut res = Residue {
            serial_number: res_pdb.serial_number(),
            res_type,
            atoms: Vec::new(),
        };

        for atom_c in res_pdb.atoms() {
            let atom_pdb = atoms_pdb
                .iter()
                .enumerate()
                .find(|(i, a)| a.serial_number() == atom_c.serial_number());
            if let Some((i, _atom)) = atom_pdb {
                res.atoms.push(i);
            }
        }

        res
    }
}

#[derive(Debug)]
pub struct Atom {
    pub serial_number: usize,
    pub posit: Vec3,
    pub element: Element,
    pub role: Option<AtomRole>,
    pub amino_acid: Option<AminoAcid>, // todo: Duplicate with storing atom IDs with residues.
}

impl Atom {
    pub fn from_pdb(
        atom_pdb: &pdbtbx::Atom,
        atom_i: usize,
        aa_map: &HashMap<usize, ResidueType>,
    ) -> Self {
        // println!("Data: {:?}", atom_pdb.name());

        // Find the amino acid type this atom is part of, if applicable.
        let amino_acid = match aa_map.get(&atom_i) {
            Some(res_type) => match res_type {
                ResidueType::AminoAcid(aa) => Some(aa),
                _ => None,
            },
            None => None,
        };

        // todo: This may be fragile.
        // todo: I don't fully understand how these are annotated in files; apeing pdbtbx's approach for now.
        let role = match amino_acid {
            Some(_) => match atom_pdb.name() {
                "CA" => Some(AtomRole::C_Alpha),
                "C" => Some(AtomRole::C_Prime),
                "N" => Some(AtomRole::N_Backbone),
                "O" => Some(AtomRole::O_Backbone),
                "H" | "H1" | "H2" | "H3" | "HA" | "HA2" | "HA3" => Some(AtomRole::H_Backbone),
                _ => Some(AtomRole::Sidechain),
            },
            None => None,
        };

        Self {
            serial_number: atom_pdb.serial_number(),
            posit: Vec3::new(atom_pdb.x(), atom_pdb.y(), atom_pdb.z()),
            element: Element::from_pdb(atom_pdb.element()),
            // amino_acid: AminoAcid::from_pdb(pdb.r)
            role,
            amino_acid: amino_acid.copied(),
        }
    }

    /// Note: This doesn't include backbone O etc; just the 3 main ones.
    pub fn is_backbone(&self) -> bool {
        match self.role {
            Some(r) => [AtomRole::C_Alpha, AtomRole::N_Backbone, AtomRole::C_Prime].contains(&r),
            None => false,
        }
    }
}

/// Can't find a PyMol equiv. Experimenting
pub fn aa_color(aa: AminoAcid) -> (f32, f32, f32) {
    match aa {
        AminoAcid::Arg => (0.7, 0.2, 0.9),
        AminoAcid::His => (0.2, 1., 0.2),
        AminoAcid::Lys => (1., 0.3, 0.3),
        AminoAcid::Asp => (0.2, 0.2, 1.0),
        AminoAcid::Glu => (0.701, 0.7, 0.2),
        AminoAcid::Ser => (0.9, 0.775, 0.25),
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
