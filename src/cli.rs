//! Our CLI system. Apes PyMol's syntax. We don't introduce our own commands, as this functionality
//! is primarily for PyMol users who are comfortable with this workflow.

use std::{io, path::PathBuf, str::FromStr};

use graphics::{EngineUpdates, Scene};
use regex::Regex;

use crate::{State, file_io::pdb::save_pdb, mol_drawing::MoleculeView, render::set_flashlight, ui::load_file, util, CamSnapshot};

/// Process a raw CLI command from the user. Return the CLI output from the entered command.
pub fn handle_cmd(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    redraw: &mut bool,
    reset_cam: &mut bool,
) -> io::Result<String> {
    let input = state.ui.cmd_line_input.trim().to_string();


    let re_help = Regex::new(r"(?i)^help$").unwrap();
    let re_fetch = Regex::new(r"(?i)^fetch\s+([A-Za-z0-9]{4})$").unwrap();
    let re_save = Regex::new(r"(?i)^save\s+([A-Za-z0-9./]+)$").unwrap();
    let re_load = Regex::new(r"(?i)^load\s+([A-Za-z0-9./]+)$").unwrap();
    let re_show = Regex::new(r"(?i)^(show|show_as)\s+([A-Za-z0-9./]+)$").unwrap();
    let re_view = Regex::new(r"(?i)^view\s+([^,\s]+)(?:\s*,\s*(store|recall))?\s*$")
        .unwrap();

    if let Some(_caps) = re_help.captures(&input) {
        // todo: Multiline, once you set that up.
        return Ok(String::from(
            "The following commands are available: fetch, save, load, show, view",
        ));
    }

    if let Some(caps) = re_fetch.captures(&input) {
        let ident = &caps[1];
        util::query_rcsb(ident, state, scene, engine_updates, redraw, reset_cam);
    }

    // todo: Save and load: Limited functionalitiy, and DRY with ui.

    // todo: Load ligands and other types of file.
    if let Some(caps) = re_save.captures(&input) {
        if let Some(mol) = &state.molecule {
            if let Some(pdb) = &mut state.pdb {
                let filename = &caps[1];
                let path = PathBuf::from_str(filename).unwrap();

                if let Err(e) = save_pdb(pdb, &path) {
                    eprintln!("Error saving pdb: {}", e);
                } else {
                    state.to_save.last_opened = Some(path.to_owned());
                    state.update_save_prefs()
                }
            }
        }
    }

    // todo: Load other types of file, e.g. map and mtz.
    if let Some(caps) = re_load.captures(&input) {
        let filename = &caps[1];
        let path = PathBuf::from_str(filename).unwrap();

        let ligand_load = matches!(
            path.extension()
                .unwrap_or_default()
                .to_ascii_lowercase()
                .to_str()
                .unwrap_or_default(),
            "sdf" | "mol2"
        );

        load_file(&path, state, redraw, reset_cam, engine_updates, ligand_load);
        set_flashlight(scene);
        engine_updates.lighting = true;
    }

    // Note: We don't have show and hide for the varous display items; this sets the display.
    if let Some(caps) = re_show.captures(&input) {
        let mode = &caps[1].trim();

        state.ui.mol_view = mode.parse()?;
        *redraw = true;
    }

    if let Some(caps) = re_view.captures(&input) {
        let name = &caps[1];

        let mut recall = false;
        match caps.get(2) {
            Some(action)  => {
                if action.as_str().eq_ignore_ascii_case("store") {
                    util::save_snap(state, &scene.camera, &name);
                } else {
                    recall = true;
                }
            }
            None => recall = true,
        }

        if recall {
            let mut found = false; // Avoids borrow error.
            for (i, snap) in state.cam_snapshots.iter().enumerate() {
                if snap.name.to_lowercase().trim() == name.to_lowercase().trim() {
                    state.ui.cam_snapshot = Some(i);
                    found = true;
                    break;
                }
            }
            if found {
                util::load_snap(state, scene, engine_updates);
            }
        }
    }

    Ok(String::from("Command succeeded"))
}
