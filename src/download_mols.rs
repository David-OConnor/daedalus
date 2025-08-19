//! Allows downloading PDB files from various APIs.

use bio_apis::{ReqError, drugbank, pubchem, rcsb};
use bio_files::{MmCif, Mol2};

use crate::molecule::Molecule;

/// Download mmCIF file from the RSCB, parse into a struct.
pub fn load_cif_rcsb(ident: &str) -> Result<(MmCif, String), ReqError> {
    let cif_text = rcsb::load_cif(ident)?;

    let mmcif = MmCif::new(&cif_text).map_err(|e| {
        eprintln!("Error parsing mmCIF file: {e}");
        e
    });

    Ok((mmcif?, cif_text))
}

/// Download an SDF file from DrugBank, and parse as a molecule.
pub fn load_sdf_drugbank(ident: &str) -> Result<Molecule, ReqError> {
    let sdf_data = drugbank::load_sdf(ident)?;

    match Mol2::new(&sdf_data) {
        Ok(m) => Ok(m.try_into().map_err(|e| ReqError::from(e))?),
        Err(e) => Err(ReqError::Http),
    }
}

/// Download an SDF file from DrugBank, and parse as a molecule.
pub fn load_sdf_pubchem(ident: &str) -> Result<Molecule, ReqError> {
    let sdf_data = pubchem::load_sdf(ident)?;

    match Mol2::new(&sdf_data) {
        Ok(m) => Ok(m.try_into().map_err(|e| ReqError::from(e))?),
        Err(_) => Err(ReqError::Http),
    }
}
