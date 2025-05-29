//! Our CLI system. Apes PyMol's syntax. We don't introduce our own commands, as this functionality
//! is primarily for PyMol users who are comfortable with this workflow.

use std::{io, path::PathBuf, str::FromStr};

use graphics::{EngineUpdates, Scene};
use regex::Regex;

use crate::{State, file_io::pdb::save_pdb, render::set_flashlight, ui::load_file, util};

/// Process a raw CLI command from the user. Return the CLI output from the entered command.
pub fn handle_cmd(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    redraw: &mut bool,
    reset_cam: &mut bool,
) -> io::Result<String> {
    let input = state.ui.cmd_line_input.trim().to_string();

    let re_fetch = Regex::new(r"(?i)^fetch\s+([A-Za-z0-9]{4})$").expect("invalid regex");
    let re_save = Regex::new(r"(?i)^save\s+([A-Za-z0-9./]+)$").expect("invalid regex");
    let re_load = Regex::new(r"(?i)^load\s+([A-Za-z0-9./]+)$").expect("invalid regex");

    if let Some(caps) = re_fetch.captures(&input) {
        let ident = &caps[1];
        util::query_rcsb(ident, state, scene, engine_updates, redraw, reset_cam);
    }

    // todo: Save and load: Limited functionalitiy, and DRY with ui.
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

    if let Some(caps) = re_load.captures(&input) {
        let filename = &caps[1];
        let path = PathBuf::from_str(filename).unwrap();

        load_file(&path, state, redraw, reset_cam, engine_updates, false);
        set_flashlight(scene);
        engine_updates.lighting = true;
    }

    Ok(String::from("Command succeeded"))
}
