//! This module prepares molecules for docking, generating PDBQT files and equivalents for
//! targets and equivalent. Adds hydrogens and charges for targets, and specifies rotatable bonds
//! for ligands.
//!
//! See Meeko (Python package), Open Babel[Open Babel](http://openbabel.org) (GUI + CLI program), ADT etc for examples.
//!
//! [Meeko](https://meeko.readthedocs.io/en/release-doc/) use. Install: `pip install meeko`.
//!
//! Can use as a python library, or as a CLI application using scripts it includes.
//! Ligand prep: `mk_prepare_ligand.py -i molecule.sdf -o molecule.pdbqt`
//! Target prep: `mk_prepare_receptor.py -i nucleic_acid.cif -o my_receptor -j -p -f A:42`
//! Converting docking results back to PDB and SDF: `mk_export.py vina_results.pdbqt -j my_receptor.json -s lig_docked.sdf -p rec_docked.pdb`
//!
//!
//! MGLTools CLI flow:
//! `pythonsh prepare_receptor4.py -r myprotein.pdb -o myprotein.pdbqt -A checkhydrogens` (tgt) or
//! `pythonsh prepare_receptor.py -r myprotein.pdb -o myprotein.pdbqt` (tgt)
//! `pythonsh prepare_ligand4.py -l myligand.pdb -o myligand.pdbqt` (ligand)
//!
//! What we will use to start: the OpenBabel CLI program.

use crate::molecule::Molecule;

/// Adds hydrogens and charge data.
pub fn prep_target(mol: &mut Molecule) {}

pub fn prep_ligand(mol: &mut Molecule) {}

/// todo: Output string if it's a text-based format.
pub fn export_pdbqt(target: &Molecule, ligand: &Molecule) -> Vec<u8> {
    Vec::new()
}
