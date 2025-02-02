use lin_alg::f64::Vec3;
use pdbtbx::PDB;
use rayon::prelude::*;

use crate::{bond_inference::create_bonds, Atom};

#[derive(Debug)]
// todo: This, or a PDB-specific format?
pub struct Molecule {
    pub atoms: Vec<Atom>,
    /// todo: For now, as returned by pdbtbx. Adjust A/R. (Refs to atoms etc)
    // pub bonds: Vec<(Atom, Atom, pdb::Bond)>,
    pub bonds: Vec<Bond>,
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

        // todo: Chains

        Molecule { atoms, bonds }
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

#[derive(Debug)]
pub struct Bond {
    pub bond_type: BondType,
    // todo: Refs to Atom etc A/R.
    pub posit_0: Vec3,
    pub posit_1: Vec3,
}

pub struct Ligand {}

pub struct Residue {}
