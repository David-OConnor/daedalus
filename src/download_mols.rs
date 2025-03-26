//! Allows downloading PDB files from the [RCSB Protein Data Bank](https://www.rcsb.org/)

use std::{io, time::Duration};

use pdbtbx::PDB;
use ureq::{self, Agent};

use crate::{file_io::pdb::read_pdb, molecule::Molecule};

const PDB_BASE_URL: &str = "https://www.rcsb.org/structure";
// const PDB_3D_VIEW_URL: &str = "https://www.rcsb.org/3d-view";
// const PDB_STRUCTURE_FILE_URL: &str = "https://files.rcsb.org/view";

// const PDB_SEARCH_API_URL: &str = "https://search.rcsb.org/rcsbsearch/v2/query";
// const PDB_DATA_API_URL: &str = "https://data.rcsb.org/rest/v1/core/entry";

const HTTP_TIMEOUT: u64 = 4; // In seconds

// Workraound for not being able to construct ureq's errors in a way I've found.
pub struct ReqError {}

fn make_agent() -> Agent {
    let config = Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(HTTP_TIMEOUT)))
        .build();

    config.into()
}

impl From<ureq::Error> for ReqError {
    fn from(_err: ureq::Error) -> Self {
        Self {}
    }
}

impl From<io::Error> for ReqError {
    fn from(_err: io::Error) -> Self {
        Self {}
    }
}

fn cif_url_rscb(ident: &str) -> String {
    format!(
        "https://files.rcsb.org/download/{}.cif",
        ident.to_uppercase()
    )
}

fn sdf_url_drugbank(ident: &str) -> String {
    format!(
        "https://go.drugbank.com/structures/small_molecule_drugs/{}.sdf?type=3d",
        ident.to_uppercase()
    )
}

fn sdf_url_pubchem(ident: &str) -> String {
    // todo: LIkely wrong.
    format!(
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/conformers/0000FE0400000001/SDF?response_type=\
        save&response_basename=Conformer3D_COMPOUND_CID_{}",
        ident.to_uppercase()
    )
}

/// Download a CIF file from the RSCB, and parse as PDB.
pub fn load_cif_rcsb(ident: &str) -> Result<PDB, ReqError> {
    let agent = make_agent();

    let resp = agent
        .get(cif_url_rscb(ident))
        .call()?
        .body_mut()
        .read_to_string()?;

    read_pdb(&resp).map_err(|e| ReqError {})
}

/// Download an SDF file from DrugBank, and parse as a molecule.
pub fn load_sdf_drugbank(ident: &str) -> Result<Molecule, ReqError> {
    let agent = make_agent();

    let resp = agent
        .get(sdf_url_drugbank(ident))
        .call()?
        .body_mut()
        .read_to_string()?;

    Molecule::from_sdf(&resp).map_err(|e| ReqError {})
}

/// Download an SDF file from DrugBank, and parse as a molecule.
pub fn load_sdf_pubchem(ident: &str) -> Result<Molecule, ReqError> {
    let agent = make_agent();

    let resp = agent
        .get(sdf_url_pubchem(ident))
        .call()?
        .body_mut()
        .read_to_string()?;

    Molecule::from_sdf(&resp).map_err(|_e| ReqError {})
}
