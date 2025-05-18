//! Allows downloading PDB files from various APIs.

use bio_apis::{ReqError, drugbank, pubchem, rcsb};
use pdbtbx::PDB;

use crate::{file_io::pdb::read_pdb, molecule::Molecule};

/// Download a CIF file from the RSCB, and parse as PDB.
pub fn load_cif_rcsb(ident: &str) -> Result<PDB, ReqError> {
    let cif_data = rcsb::load_cif(ident)?;

    read_pdb(&cif_data).map_err(|e| {
        eprintln!("Error parsing mmCIF file: {e}");
        ReqError {}
    })
}

/// Download an SDF file from DrugBank, and parse as a molecule.
pub fn load_sdf_drugbank(ident: &str) -> Result<Molecule, ReqError> {
    let sdf_data = drugbank::load_sdf(ident)?;

    Molecule::from_sdf(&sdf_data).map_err(|e| ReqError {})
}

/// Download an SDF file from DrugBank, and parse as a molecule.
pub fn load_sdf_pubchem(ident: &str) -> Result<Molecule, ReqError> {
    let sdf_data = pubchem::load_sdf(ident)?;

    Molecule::from_sdf(&sdf_data).map_err(|_e| ReqError {})
}
