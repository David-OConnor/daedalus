//! Allows downloading PDB files from various APIs.

use bio_apis::{ReqError, drugbank, pubchem, rcsb};
use bio_files::Mol2;
use pdbtbx::PDB;

use crate::{file_io::cif_pdb::read_pdb, molecule::Molecule};

/// Download a CIF file from the RSCB, and parse as PDB.
pub fn load_cif_rcsb(ident: &str) -> Result<(PDB, String), ReqError> {
    let cif_data = rcsb::load_cif(ident)?;

    let pdb = read_pdb(&cif_data).map_err(|e| {
        eprintln!("Error parsing mmCIF file: {e}");
        e
    });

    Ok((pdb?, cif_data))
}

/// Download an SDF file from DrugBank, and parse as a molecule.
pub fn load_sdf_drugbank(ident: &str) -> Result<Molecule, ReqError> {
    let sdf_data = drugbank::load_sdf(ident)?;

    match Mol2::new(&sdf_data) {
        Ok(m) => Ok(m.into()),
        Err(e) => Err(ReqError::Http),
    }
}

/// Download an SDF file from DrugBank, and parse as a molecule.
pub fn load_sdf_pubchem(ident: &str) -> Result<Molecule, ReqError> {
    let sdf_data = pubchem::load_sdf(ident)?;

    match Mol2::new(&sdf_data) {
        Ok(m) => Ok(m.into()),
        Err(e) => Err(ReqError::Http),
    }
}
