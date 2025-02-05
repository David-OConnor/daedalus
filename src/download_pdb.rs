//! Allows downloading PDB files from the [RCSB Protein Data Bank](https://www.rcsb.org/)

use std::{io, time::Duration};

use pdbtbx::PDB;
use ureq::{self, Agent};

use crate::{molecule::Molecule, pdb::read_pdb};

const PDB_BASE_URL: &str = "https://www.rcsb.org/structure";
// const PDB_3D_VIEW_URL: &str = "https://www.rcsb.org/3d-view";
// const PDB_STRUCTURE_FILE_URL: &str = "https://files.rcsb.org/view";

// const PDB_SEARCH_API_URL: &str = "https://search.rcsb.org/rcsbsearch/v2/query";
// const PDB_DATA_API_URL: &str = "https://data.rcsb.org/rest/v1/core/entry";

const HTTP_TIMEOUT: u64 = 4; // In seconds

// Workraound for not being able to construct ureq's errors in a way I've found.
pub struct ReqError {}

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

fn cif_url(ident: &str) -> String {
    format!(
        "https://files.rcsb.org/download/{}.cif",
        ident.to_uppercase()
    )
}

pub fn load_rcsb(ident: &str) -> Result<PDB, ReqError> {
    let config = Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(HTTP_TIMEOUT)))
        .build();

    let agent: Agent = config.into();

    let resp = agent
        .get(cif_url(ident))
        .call()?
        .body_mut()
        .read_to_string()?;

    // todo: Don't unwrap.
    read_pdb(&resp).map_err(|e| ReqError {})
}
