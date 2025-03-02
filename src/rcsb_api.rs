//! For loading data from the RSCB website's API.
//! C+P from PlasCAD

//! For opening the browser to NCBI BLAST, PDB etc.
//!
//! PDB Search API: https://search.rcsb.org/#search-api
//! PDB Data API: https://data.rcsb.org/#data-api

use std::{io, io::read_to_string, time::Duration};

use bincode::{Decode, Encode};
use na_seq::{Nucleotide, seq_aa_to_str, seq_to_str_lower};
use serde::{Deserialize, Serialize};
use serde_json::{self};
use ureq::{self, Agent};
use url::Url;

use crate::{Selection, State};

const PDB_BASE_URL: &str = "https://www.rcsb.org/structure";
const DRUGBANK_BASE_URL: &str = "https://go.drugbank.com/drugs";
const PUBCHEM_BASE_URL: &str = "https://pubchem.ncbi.nlm.nih.gov/compound";
const PDB_3D_VIEW_URL: &str = "https://www.rcsb.org/3d-view";
const PDB_STRUCTURE_FILE_URL: &str = "https://files.rcsb.org/view";

const PDB_SEARCH_API_URL: &str = "https://search.rcsb.org/rcsbsearch/v2/query";
const PDB_DATA_API_URL: &str = "https://data.rcsb.org/rest/v1/core/entry";

// An arbitrary limit to prevent excessive queries to the PDB data api,
// and to simplify display code.
const MAX_PDB_RESULTS: usize = 8;

const HTTP_TIMEOUT: u64 = 4; // In seconds

#[derive(Default, Serialize)]
struct PdbSearchParams {
    value: String,
    sequence_type: String,
    evalue_cutoff: u8,
    identity_cutoff: f32,
}

#[derive(Default, Serialize)]
struct PdbSearchQuery {
    #[serde(rename = "type")]
    type_: String,
    service: String,
    parameters: PdbSearchParams,
}

#[derive(Default, Serialize)]
struct SearchRequestOptions {
    scoring_strategy: String,
}

#[derive(Default, Serialize)]
struct PdbPayloadSearch {
    return_type: String,
    query: PdbSearchQuery,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_options: Option<SearchRequestOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_info: Option<String>,
}

#[derive(Default, Debug, Encode, Decode, Deserialize)]
pub struct PdbSearchResult {
    pub identifier: String,
    pub score: f32,
}

#[derive(Encode, Decode, Clone, Debug)]
pub struct PdbMetaData {
    // todo: A/R
    pub prim_cit_title: String,
}

#[derive(Default, Debug, Deserialize)]
struct PdbSearchResults {
    query_id: String,
    result_type: String,
    total_count: u32,
    result_set: Vec<PdbSearchResult>,
}

#[derive(Default, Debug, Deserialize)]
struct PdbStruct {
    title: String,
}

#[derive(Default, Debug, Deserialize)]
struct PdbDataResults {
    #[serde(rename = "struct")]
    struct_: PdbStruct,
}

#[derive(Default, Debug, Deserialize)]
struct PrimaryCitation {
    title: String,
}

#[derive(Default, Debug, Deserialize)]
struct PdbMetaDataResults {
    rcsb_primary_citation: PrimaryCitation,
}

// /// This doesn't deserialize directly; it's the format we use internally.
#[derive(Encode, Decode)]
pub struct PdbData {
    pub rcsb_id: String,
    pub title: String,
}

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

/// Load PDB data using [its API](https://search.rcsRb.org/#search-api)
/// Returns the set of PDB ID matches, with scores.
pub fn load_pdb_data() -> Result<Vec<PdbData>, ReqError> {
    let payload_search = PdbPayloadSearch {
        return_type: "entry".to_string(),
        query: PdbSearchQuery {
            type_: "terminal".to_owned(),
            service: "sequence".to_owned(),
            parameters: PdbSearchParams {
                // value: seq_aa_to_str(&protein.aa_seq),
                value: "".to_string(),
                sequence_type: "protein".to_owned(),
                evalue_cutoff: 1,
                identity_cutoff: 0.9,
            },
        },
        request_options: Some(SearchRequestOptions {
            scoring_strategy: "sequence".to_owned(),
        }),

        // return_type: "assembly".to_string(), // todo: Experiment.
        ..Default::default()
    };

    // todo: Limit the query to our result cap, instead of indexing after?

    let payload_json = serde_json::to_string(&payload_search).unwrap();

    let config = Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(HTTP_TIMEOUT)))
        .build();

    let agent: Agent = config.into();

    let resp: String = agent
        .post(PDB_SEARCH_API_URL)
        .header("Content-Type", "application/json")
        .send(&payload_json)?
        .body_mut()
        .read_to_string()?;

    let search_data: PdbSearchResults = serde_json::from_str(&resp).map_err(|_| ReqError {})?;

    let mut result_search = Vec::new();
    for (i, r) in search_data.result_set.into_iter().enumerate() {
        if i < MAX_PDB_RESULTS {
            result_search.push(r);
        }
    }

    let mut result = Vec::with_capacity(result_search.len());
    for r in result_search {
        let resp = agent
            .get(&format!("{PDB_DATA_API_URL}/{}", r.identifier))
            .call()?
            .body_mut()
            .read_to_string()?;

        let data: PdbDataResults = serde_json::from_str(&resp).map_err(|_| ReqError {})?;

        result.push(PdbData {
            rcsb_id: r.identifier,
            title: data.struct_.title,
        })
    }

    Ok(result)
}

/// Open a PDB search for this protein's sequence, given a PDB ID, which we load from the API.
pub fn open_pdb(pdb_id: &str) {
    if let Err(e) = webbrowser::open(&format!("{PDB_BASE_URL}/{pdb_id}")) {
        eprintln!("Failed to open the web browser: {:?}", e);
    }
}

/// Open a PDB search for this protein's sequence, given a PDB ID, which we load from the API.
pub fn open_pdb_3d_view(pdb_id: &str) {
    if let Err(e) = webbrowser::open(&format!("{PDB_3D_VIEW_URL}/{pdb_id}")) {
        eprintln!("Failed to open the web browser: {:?}", e);
    }
}

pub fn open_drugbank(id: &str) {
    if let Err(e) = webbrowser::open(&format!("{DRUGBANK_BASE_URL}/{id}")) {
        eprintln!("Failed to open the web browser: {:?}", e);
    }
}

pub fn open_pubchem(id: u32) {
    if let Err(e) = webbrowser::open(&format!("{PUBCHEM_BASE_URL}/{id}")) {
        eprintln!("Failed to open the web browser: {:?}", e);
    }
}

/// Load PDB structure data in the PDBx/mmCIF format. This is a modern, text-based format.
/// It avoids the XML, and limitations of the other two available formats.
/// todo: When to use wwpdb vs rscb?
pub fn load_pdb_structure(pdb_id: &str) {
    let pdb_id = pdb_id.to_owned().to_lowercase();

    // todo: Use EGUI_file to save the file and reqwest to load it.

    // let url_pdb_format = format!("https://files.wwpdb.org/pub/pdb/data/structures/divided/pdb/zg/pdb{pdb_id}.ent.gz");
    // let url_pdbx_format = format!("https://files.wwpdb.org/pub/pdb/data/structures/divided/mmCIF/zg/{pdb_id}.cif.gz");

    let url = format!("{PDB_STRUCTURE_FILE_URL}/{pdb_id}.cif");

    if let Err(e) = webbrowser::open(&url) {
        eprintln!("Failed to open the web browser: {:?}", e);
    }
}

pub fn load_pdb_metadata(pdb_id: &str) -> Result<PdbMetaData, ReqError> {
    // let pdb_id = pdb_id.to_owned().to_lowercase();

    let config = Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(HTTP_TIMEOUT)))
        .build();

    let agent: Agent = config.into();

    let resp = agent
        .get(&format!("{PDB_DATA_API_URL}/{}", pdb_id))
        .call()?
        .body_mut()
        .read_to_string()?;

    let data: PdbMetaDataResults = serde_json::from_str(&resp).map_err(|_| ReqError {})?;

    Ok(PdbMetaData {
        prim_cit_title: data.rcsb_primary_citation.title,
    })
}
