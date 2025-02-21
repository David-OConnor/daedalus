//! This module prepares molecules for docking, generating PDBQT files and equivalents for
//! targets and equivalent. Adds hydrogens and charges for targets, and specifies rotatable bonds
//! for ligands.
//!
//! See Meeko (Python package), Open Babel (CLI program), ADT etc for examples.
//!
//! MGLTools CLI flow:
//! `pythonsh prepare_receptor4.py -r myprotein.pdb -o myprotein.pdbqt -A checkhydrogens` (tgt) or
//! `pythonsh prepare_receptor.py -r myprotein.pdb -o myprotein.pdbqt` (tgt)
//! `pythonsh prepare_ligand4.py -l myligand.pdb -o myligand.pdbqt` (ligand)

use crate::molecule::Molecule;

/// Adds hydrogens and charge data.
pub fn prep_target(mol: &mut Molecule) {}

pub fn prep_ligand(mol: &mut Molecule) {}

/// todo: Output string if it's a text-based format.
pub fn export_pdbqt(target: &Molecule, ligand: &Molecule) -> Vec<u8> {
    Vec::new()
}
