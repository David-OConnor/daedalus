/// Contains data structures and related code for molecules, atoms, residues, chains, etc.
use std::str::FromStr;

use lin_alg::f64::Vec3;
use na_seq::AminoAcid;
use pdbtbx::PDB;
use rayon::prelude::*;

use crate::{bond_inference::create_bonds, Element, Selection};

#[derive(Debug)]
// todo: This, or a PDB-specific format?
pub struct Molecule {
    pub ident: String,
    pub atoms: Vec<Atom>,
    pub bonds: Vec<Bond>,
    pub chains: Vec<Chain>,
    pub residues: Vec<Residue>,
}

impl Molecule {
    pub fn from_pdb(pdb: &PDB) -> Self {
        // todo: Maybe return the PDB type here, and store that. Also have a way to
        // todo get molecules from it

        let atoms_pdb: Vec<&pdbtbx::Atom> = pdb.par_atoms().collect();

        let res_pdb: Vec<&pdbtbx::Residue> = pdb.residues().collect();

        let mut residues: Vec<Residue> = pdb.residues()
            .into_iter()
            .map(|res| Residue::from_pdb(res, &atoms_pdb))
            .collect();

        residues.sort_by_key(|r| r.serial_number);

        let mut chains = Vec::with_capacity(pdb.chain_count());
        for chain_pdb in pdb.chains() {
            // println!("Chain: {chain_pdb:?}");

            let mut chain = Chain {
                id: chain_pdb.id().to_owned(),
                atoms: Vec::new(),
                residues: Vec::new(),
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

            // Using our residues due to the sort; need this as long as we select etc based on index.
            // todo: Consider selecting based on SN!
            for res_c in chain_pdb.residues() {
                let res = residues
                    .iter()
                    .enumerate()
                    .find(|(i, r)| r.serial_number == res_c.serial_number());
                if let Some((i, _res)) = res {
                    chain.residues.push(i);
                }
            }

            chains.push(chain);
        }

        let atoms: Vec<Atom> = atoms_pdb
            .into_iter()
            .map(|atom| Atom::from_pdb(atom, &res_pdb))
            .collect();

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
pub enum AaRole {
    C_Alpha,
    C_Prime,
    C_N,
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

#[derive(Debug)]
pub enum ResidueType {
    AminoAcid(AminoAcid),
    Other(String),
}

#[derive(Debug)]
pub struct Residue {
    /// We currently use serial number of display, search etc, and arra index to select.
    // todo: Residue type that includes water etc in addition to AAs. Enum that wraps AminoAcid, for example.
    pub serial_number: isize, // pdbtbx uses isize. Negative allowed?
    pub aa: Option<AminoAcid>,
    pub atoms: Vec<usize>, // Atom index
}

impl Residue {
    pub fn from_pdb(res_pdb: &pdbtbx::Residue, atoms_pdb: &[&pdbtbx::Atom]) -> Self {
        let aa = AminoAcid::from_str(res_pdb.name().unwrap_or_default()).ok();
        let mut res = Residue {
            serial_number: res_pdb.serial_number(),
            aa,
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
    pub role: Option<AaRole>,
    pub amino_acid: Option<AminoAcid>, // todo: Duplicate with storing atom IDs with residues.
}

impl Atom {
    pub fn from_pdb(atom_pdb: &pdbtbx::Atom, residues: &[&pdbtbx::Residue]) -> Self {
        let mut amino_acid = None;
        // println!("Data: {:?}", pdb.type_id());

        for res in residues {
            let res_atoms: Vec<&pdbtbx::Atom> = res.atoms().collect();

            for atom in &res_atoms {
                if atom.serial_number() == atom_pdb.serial_number() {
                    let aa = AminoAcid::from_str(res.name().unwrap_or_default());

                    if let Ok(a) = aa {
                        amino_acid = Some(a);
                    }
                }
            }
        }

        Self {
            serial_number: atom_pdb.serial_number(),
            posit: Vec3::new(atom_pdb.x(), atom_pdb.y(), atom_pdb.z()),
            element: Element::from_pdb(atom_pdb.element()),
            // amino_acid: AminoAcid::from_pdb(pdb.r)
            // todo
            role: None,
            amino_acid,
            // is_backbone: pdb.is_backbone(),
        }
    }

    pub fn is_backbone(&self) -> bool {
        match self.role {
            Some(r) => [AaRole::C_Alpha, AaRole::C_N, AaRole::C_Prime].contains(&r),
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
