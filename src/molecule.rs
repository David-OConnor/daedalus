use lin_alg::f64::Vec3;
use na_seq::AminoAcid;
use pdbtbx::PDB;
use rayon::prelude::*;

use crate::{bond_inference::create_bonds, Element};

#[derive(Debug)]
pub struct Chain {}

#[derive(Debug)]
// todo: This, or a PDB-specific format?
pub struct Molecule {
    pub atoms: Vec<Atom>,
    /// todo: For now, as returned by pdbtbx. Adjust A/R. (Refs to atoms etc)
    // pub bonds: Vec<(Atom, Atom, pdb::Bond)>,
    pub bonds: Vec<Bond>,
    pub chains: Vec<Chain>,
}

impl Molecule {
    pub fn from_pdb(pdb: &PDB) -> Self {
        // todo: Maybe return the PDB type here, and store that. Also have a way to
        // todo get molecules from it

        for res in pdb.residues() {
            println!("Res: {:?}", res);
        }

        // for conf in pdb.conformers() {
        // println!("Conf: {:?}", conf);
        // }

        // for atom in pdb.atoms() {
        // for atom in pdb.par_atoms() {
        let atoms: Vec<Atom> = pdb.par_atoms().map(|atom| Atom::from_pdb(atom)).collect();

        // let mut bonds = Vec::new();
        // /// todo: Adjust etc so you're not adding so many new atoms to state.
        // for (a0, a1, bond) in pdb.bonds() {
        //     bonds.push((Atom::from_pdb(a0), Atom::from_pdb(a1), bond));
        // }

        let bonds = create_bonds(&atoms);

        let chains = Vec::new();

        Molecule {
            atoms,
            bonds,
            chains,
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
}

#[derive(Debug)]
pub struct Bond {
    pub bond_type: BondType,
    pub bond_count: BondCount,
    // todo: Refs to Atom etc A/R.
    pub posit_0: Vec3,
    pub posit_1: Vec3,
    pub is_backbone: bool,
}

pub struct Ligand {}

pub struct Residue {}

#[derive(Debug)]
pub struct Atom {
    pub posit: Vec3,
    pub element: Element,
    pub role: Option<AaRole>,
    pub amino_acid: Option<AminoAcid>,
    // pub is_backbone: bool,
}

impl Atom {
    pub fn from_pdb(pdb: &pdbtbx::Atom) -> Self {
        Self {
            posit: Vec3::new(pdb.x(), pdb.y(), pdb.z()),
            element: Element::from_pdb(pdb.element()),
            // amino_acid: AminoAcid::from_pdb(pdb.r)
            // todo
            role: None,
            amino_acid: None,
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
