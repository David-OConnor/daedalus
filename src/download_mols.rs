//! Allows downloading PDB files from various APIs.

use std::time::Instant;

use bio_apis::{ReqError, amber_geostd, amber_geostd::GeostdItem, rcsb};
use bio_files::{MmCif, Mol2, Sdf, md_params::ForceFieldParams};
use graphics::{ControlScheme, EngineUpdates, Scene};
use na_seq::AaIdent;

use crate::{
    State, StateUi,
    mol_lig::MoleculeSmall,
    molecule::{MoGenericRefMut, MolIdent, MolType, MoleculeGeneric, MoleculePeptide},
    render::set_flashlight,
    util::handle_err,
};

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
    match Sdf::load_drugbank(ident) {
        Ok(m) => Ok(m.try_into().map_err(|e| ReqError::from(e))?),
        Err(_) => Err(ReqError::Http),
    }
}

/// Download an SDF file from PubChem, and parse as a molecule.
pub fn load_sdf_pubchem(cid: u32) -> Result<MoleculeSmall, ReqError> {
    match Sdf::load_pubchem(cid) {
        Ok(m) => Ok(m.try_into().map_err(|e| ReqError::from(e))?),
        Err(_) => Err(ReqError::Http),
    }
}

// todo: Diff between this and the 2 variant?
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

pub fn load_atom_coords_rcsb(
    ident: &str,
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    redraw: &mut bool,
    reset_cam: &mut bool,
) {
    println!("Loading atom data from RCSB...");
    let start = Instant::now();

    match load_cif_rcsb(ident) {
        // todo: For organization purposes, move this code out of the UI.
        Ok((cif, cif_text)) => {
            let Some(ff_map) = &state.ff_param_set.peptide_ff_q_map else {
                handle_err(
                    &mut state.ui,
                    "Unable to find the peptide FF Q map in parameters; can't load the molecule"
                        .to_owned(),
                );
                return;
            };

            let mut mol: MoleculePeptide =
                match MoleculePeptide::from_mmcif(cif, ff_map, None, state.to_save.ph) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("Problem parsing mmCif data into molecule: {e:?}");
                        return;
                    }
                };

            state.volatile.aa_seq_text = String::with_capacity(mol.common.atoms.len());
            for aa in &mol.aa_seq {
                state
                    .volatile
                    .aa_seq_text
                    .push_str(&aa.to_str(AaIdent::OneLetter));
            }

            // todo: DRY from `open_molecule`. Refactor into shared code?

            state.volatile.aa_seq_text = String::with_capacity(mol.common.atoms.len());
            for aa in &mol.aa_seq {
                state
                    .volatile
                    .aa_seq_text
                    .push_str(&aa.to_str(AaIdent::OneLetter));
            }

            state.volatile.orbit_center = Some((MolType::Peptide, 0));
            if let ControlScheme::Arc { center } = &mut scene.input_settings.control_scheme {
                *center = mol.center.into();
            }

            state.volatile.flags.ss_mesh_created = false;
            state.volatile.flags.sas_mesh_created = false;
            state.volatile.flags.clear_density_drawing = true;
            state.peptide = Some(mol);
            state.cif_pdb_raw = Some(cif_text);
        }
        Err(e) => {
            handle_err(
                &mut state.ui,
                format!("Problem loading molecule from CIF: {e:?}"),
            );
            return;
        }
    }
    let elapsed = start.elapsed().as_millis();
    println!("Loading complete in {elapsed:.1}ms");

    state.update_from_prefs();

    *redraw = true;
    *reset_cam = true;
    set_flashlight(scene);
    engine_updates.lighting = true;

    // todo: async
    // Only after updating from prefs (to prevent unecesasary loading) do we update data avail.
    state
        .peptide
        .as_mut()
        .unwrap()
        .updates_rcsb_data(&mut state.volatile.mol_pending_data_avail);
}

// todo: DIff between this and the non-2 variant?
pub fn load_geostd2(
    state: &mut State,
    scene: &mut Scene,
    ident: &str,
    load_mol2: bool,
    load_frcmod: bool,
    engine_updates: &mut EngineUpdates,
) {
    match amber_geostd::load_mol_files(ident) {
        Ok(data) => {
            // Load FRCmod first, then the Ligand constructor will populate that it loaded.
            if load_frcmod {
                if let Some(frcmod) = data.frcmod {
                    match ForceFieldParams::from_frcmod(&frcmod) {
                        Ok(v) => {
                            state.lig_specific_params.insert(ident.to_uppercase(), v);
                        }
                        Err(e) => {
                            handle_err(&mut state.ui, format!("FRCmod empty from geostd: {e:?}"));
                        }
                    }
                    if let Some(lig) = state.active_mol_mut() {
                        if let MoGenericRefMut::Ligand(l) = lig {
                            l.frcmod_loaded = true;
                        }
                    }
                }
            }

            if let Some(_lib) = data.lib {
                println!("todo: Lib data available from geostd; download?");
            }

            if load_mol2 {
                match Mol2::new(&data.mol2) {
                    Ok(mol2) => {
                        let mut mol: MoleculeSmall = mol2.try_into().unwrap();
                        mol.idents.push(MolIdent::PdbeAmber(ident.to_owned()));
                        if let Some(cid) = data.pubchem_cid {
                            mol.idents.push(MolIdent::PubChem(cid));
                        }

                        state.load_mol_to_state(
                            MoleculeGeneric::Ligand(mol),
                            Some(scene),
                            engine_updates,
                            None,
                        );
                    }
                    Err(e) => handle_err(
                        &mut state.ui,
                        format!("Unable to make a Mol2 from Geostd data: {:?}", e),
                    ),
                }
            }
        }
        Err(_) => handle_err(
            &mut state.ui,
            format!("Unable to load Amber Geostd data (Server or internet problem?)"),
        ),
    }
}
