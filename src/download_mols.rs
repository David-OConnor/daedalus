//! Allows downloading PDB files from various APIs.

use std::time::Instant;

use bio_apis::{ReqError, amber_geostd, amber_geostd::GeostdItem, drugbank, pubchem, rcsb};
use bio_files::{MmCif, Sdf};

use crate::{StateUi, mol_lig::MoleculeSmall, util::handle_err};

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
pub fn load_sdf_drugbank(ident: &str) -> Result<MoleculeSmall, ReqError> {
    let sdf_data = drugbank::load_sdf(ident)?;

    match Sdf::new(&sdf_data) {
        Ok(m) => Ok(m.try_into().map_err(|e| ReqError::from(e))?),
        Err(_) => Err(ReqError::Http),
    }
}

/// Download an SDF file from DrugBank, and parse as a molecule.
pub fn load_sdf_pubchem(ident: &str) -> Result<MoleculeSmall, ReqError> {
    let sdf_data = pubchem::load_sdf(ident)?;

    match Sdf::new(&sdf_data) {
        Ok(m) => Ok(m.try_into().map_err(|e| ReqError::from(e))?),
        Err(_) => Err(ReqError::Http),
    }
}

pub fn load_geostd(ident: &str, load_data: &mut Option<GeostdItem>, state_ui: &mut StateUi) {
    println!("Loading Amber Geostd data...");
    let start = Instant::now();

    match amber_geostd::find_mols(&ident) {
        Ok(data) => match data.len() {
            0 => handle_err(
                state_ui,
                "Unable to find an Amber molecule for this residue".to_string(),
            ),
            1 => {
                *load_data = Some(data[0].clone());
            }
            _ => {
                *load_data = Some(data[0].clone());
                eprintln!("More than 1 geostd items available");
            }
        },
        Err(e) => handle_err(state_ui, format!("Problem loading mol data online: {e:?}")),
    }

    let elapsed = start.elapsed().as_millis();
    println!("Loaded Amber Geostd in {elapsed:.1}ms");
}
