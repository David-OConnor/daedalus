//! Our CLI system. Apes PyMol's syntax. We don't introduce our own commands, as this functionality
//! is primarily for PyMol users who are comfortable with this workflow.

use std::{io, path::PathBuf, str::FromStr};

use graphics::{EngineUpdates, Scene};
use regex::Regex;

use crate::{
    State, file_io::pdb::save_pdb, mol_drawing::MoleculeView, render::set_flashlight,
    ui::load_file, util,
};

/// Process a raw CLI command from the user. Return the CLI output from the entered command.
pub fn handle_cmd(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    redraw: &mut bool,
    reset_cam: &mut bool,
) -> io::Result<String> {
    let input = state.ui.cmd_line_input.trim().to_string();

    let re_help = Regex::new(r"(?i)^help$").expect("invalid regex");
    let re_fetch = Regex::new(r"(?i)^fetch\s+([A-Za-z0-9]{4})$").expect("invalid regex");
    let re_save = Regex::new(r"(?i)^save\s+([A-Za-z0-9./]+)$").expect("invalid regex");
    let re_load = Regex::new(r"(?i)^load\s+([A-Za-z0-9./]+)$").expect("invalid regex");
    let re_show = Regex::new(r"(?i)^(show|show_as)\s+([A-Za-z0-9./]+)$").expect("invalid regex");

    if let Some(_caps) = re_help.captures(&input) {
        // todo: Multiline, once you set that up.
        return Ok(String::from(
            "The following commands are available: fetch, save, load, show",
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

    Ok(String::from("Command succeeded"))
}
