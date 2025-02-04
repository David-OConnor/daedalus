use std::any::Any;
use std::str::FromStr;
use lin_alg::f64::Vec3;
use na_seq::AminoAcid;
use pdbtbx::PDB;
use rayon::prelude::*;

use crate::{bond_inference::create_bonds, Element};

#[derive(Debug)]
pub struct Chain {}

#[derive(Debug)]
// todo: This, or a PDB-specific format?
pub struct Molecule<> {
    pub atoms: Vec<Atom>,
    /// todo: For now, as returned by pdbtbx. Adjust A/R. (Refs to atoms etc)
    // pub bonds: Vec<(Atom, Atom, pdb::Bond)>,
    pub bonds: Vec<Bond>,
    pub chains: Vec<Chain>,
    pub residues: Vec<Residue>
}

impl Molecule {
    pub fn from_pdb(pdb: &PDB) -> Self {
        // todo: Maybe return the PDB type here, and store that. Also have a way to
        // todo get molecules from it

        let atoms_pdb: Vec<&pdbtbx::Atom> = pdb.par_atoms().collect();

        // for atom in &atoms_pdb {
        //     println!("Atom: {:?}", atom);
        // }

        let mut residues = Vec::new();

        for res_pdb in pdb.residues() {
            // println!("\nRes: {res_pdb:?}");
            // println!("\nConfs:");
            for conf in res_pdb.conformers() {

                let aa = AminoAcid::from_str("arg").ok();
                // let aa: AminoAcid = conf.name().into();
                // let aa = AminoAcid::from_str(conf.name()).ok();
                let mut res_us = Residue {
                    aa,
                    atoms: Vec::new()
                };

                for atom_c in conf.atoms() {
                    let atom_main = atoms_pdb.iter().enumerate().find(|(i, a)| a.serial_number() == atom_c.serial_number());
                    if let Some((i, _atom)) = atom_main {
                        res_us.atoms.push(i);
                    }
                }

                residues.push(res_us);
            }
        }

        // for conf in pdb.conformers() {
        // println!("Conf: {:?}", conf);
        // }

        // for atom in pdb.atoms() {
        // for atom in pdb.par_atoms() {
        let atoms: Vec<Atom> = atoms_pdb.into_iter().map(|atom| Atom::from_pdb(atom)).collect();

        // let mut bonds = Vec::new();
        // /// todo: Adjust etc so you're not adding so many new atoms to state.
        // for (a0, a1, bond) in pdb.bonds() {
        //     bonds.push((Atom::from_pdb(a0), Atom::from_pdb(a1), bond));
        // }

        let bonds = create_bonds(&atoms);

        let chains = Vec::with_capacity(pdb.chain_count());
        for chain in pdb.chains() {
            // println!("Chain: {chain:?}");
        }

        println!("Residues: {:?}", residues);

        Molecule {
            atoms,
            bonds,
            chains,
            residues,
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

#[derive(Debug)]
pub struct Residue {
    // todo: Residue type that includes water etc in addition to AAs. Enum that wraps AminoAcid, for example.
    aa: Option<AminoAcid>,
    atoms: Vec<usize> // Atom index
}

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
        let mut amino_acid = None;
        // println!("Data: {:?}", pdb.type_id());

        Self {
            posit: Vec3::new(pdb.x(), pdb.y(), pdb.z()),
            element: Element::from_pdb(pdb.element()),
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
