//! Our CLI system. Apes PyMol's syntax. We don't introduce our own commands, as this functionality
//! is primarily for PyMol users who are comfortable with this workflow.

use std::{f32::consts::TAU, io, io::ErrorKind, path::PathBuf, str::FromStr};

use graphics::{EngineUpdates, Scene};
use lin_alg::f32::{Quaternion, Vec3};
use regex::Regex;

use crate::{
    CamSnapshot, State,
    element::Element,
    file_io::pdb::save_pdb,
    mol_drawing::MoleculeView,
    molecule::AtomRole,
    render::set_flashlight,
    ui::load_file,
    util,
    util::{cam_look_at, reset_camera},
};

// We use this for autocomplete.
pub const CLI_CMDS: [&str; 14] = [
    "help", "fetch", "save", "load", "show", "show_as", "view", "hide", "remove", "orient", "roll",
    "turn", "move", "reset"
];

/// Process a raw CLI command from the user. Return the CLI output from the entered command.
pub fn handle_cmd(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    redraw: &mut bool,
    reset_cam: &mut bool,
) -> io::Result<String> {
    let input = state.ui.cmd_line_input.trim().to_string();

    // todo: Helpers to reduce regex DRY.
    let re_help = Regex::new(r"(?i)^help$").unwrap();
    let re_fetch = Regex::new(r"(?i)^fetch\s+([a-z0-9]{4})$").unwrap();
    let re_save = Regex::new(r"(?i)^save\s+([a-z0-9./]+)$").unwrap();
    let re_load = Regex::new(r"(?i)^load\s+([a-z0-9./]+)$").unwrap();
    let re_show = Regex::new(r"(?i)^(?:show|show_as)\s+([a-z0-9./]+)$").unwrap();
    // todo: Shoudl this be get_view and set_view? Have seen both.
    let re_view = Regex::new(r"(?i)^view\s+([^,\s]+)(?:\s*,\s*(store|recall))?\s*$").unwrap();

    let re_hide = Regex::new(r"(?i)^hide\s+([a-z0-9\s]+)$").unwrap();
    let re_remove = Regex::new(r"(?i)^remove\s+([a-z0-9\s]+)$").unwrap();
    let re_orient = Regex::new(r"(?i)^orient\s*(?:sel)?$").unwrap();
    let re_turn = Regex::new(r"(?i)^turn\s+([xyz]),\s*(-*\d{1,4})$").unwrap();
    let re_roll = Regex::new(r"(?i)^roll\s+([xyz]),\s*(-*\d{1,4})$").unwrap();
    let re_move = Regex::new(r"(?i)^move\s+([xyz]),\s*(-*\d{1,4})$").unwrap();
    let re_zoom = Regex::new(r"(?i)^zoom\s+([a-z0-9\s]+)$").unwrap();
    let re_reset = Regex::new(r"(?i)^reset\s*$").unwrap();

    if let Some(_caps) = re_help.captures(&input) {
        // todo: Multiline, once you set that up.
        return Ok(String::from(
            "The following commands are available: fetch, save, load, show, view, hide, remove, orient,\
             roll, turn, move, reset",
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
        let mode = &caps[1];

        state.ui.mol_view = mode.parse()?;
        *redraw = true;
    }

    if let Some(caps) = re_view.captures(&input) {
        let name = &caps[1];

        let mut recall = false;
        match caps.get(2) {
            Some(action) => {
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

    if let Some(caps) = re_hide.captures(&input) {
        let item = &caps[1].to_lowercase();

        // todo: To match PyMol, this should be much more robust. Hiding chains, residues etc.
        // todo: A lot of these just hide, not remove... Should remove probably.
        match item.as_ref() {
            "solvents" => {
                state.ui.visibility.hide_hetero = true;
            }
            "hetatm" => {
                state.ui.visibility.hide_hetero = true;
                state.ui.visibility.hide_ligand = true;
            }
            "resn hoh" => {
                // todo: The space won't work in the regex.
                state.ui.visibility.hide_water = true;
            }
            "hydro" => {
                // todo: The space won't work in the regex.
                state.ui.visibility.hide_hydrogen = true;
            }
            _ => (),
        }

        *redraw = true;
    }

    if let Some(caps) = re_remove.captures(&input) {
        let item = &caps[1].to_lowercase();

        // todo: To match PyMol, this should be much more robust. Removing chains, residues etc.
        // todo: A lot of these just hide, not remove... Should remove probably.

        return Ok("Remove is temproarily disabled".to_owned());

        // todo uhoh: When you remove atoms, their indices in the vec get screwed up! You may need to use
        // todo a unique id!
        if let Some(mol) = &mut state.molecule {
            match item.as_ref() {
                "solvents" => {
                    // todo: Remove residues as well?
                    mol.atoms.retain(|a| {
                        if let Some(role) = a.role {
                            role != AtomRole::Water
                        } else {
                            true
                        }
                    });
                }
                "hetatm" => {
                    mol.atoms.retain(|a| !a.hetero);
                }
                "resn hoh" => {
                    // todo: Remove residues as well?
                    mol.atoms.retain(|a| {
                        if let Some(role) = a.role {
                            role != AtomRole::Water
                        } else {
                            true
                        }
                    });
                }
                "hydro" => {
                    mol.atoms.retain(|a| a.element != Element::Hydrogen);
                    state.ui.visibility.hide_hydrogen = true;
                }
                _ => (),
            }
        }

        //
        // match item.as_ref() {
        //     "" => {}
        //     _ => (),
        // }

        *redraw = true;
    }

    if let Some(caps) = re_orient.captures(&input) {
        engine_updates.camera = true;
    }

    if let Some(caps) = re_roll.captures(&input) {
        let axis = match caps[1].to_lowercase().as_ref() {
            "x" => Vec3::new(1., 0., 0.),
            "y" => Vec3::new(0., 1., 0.),
            "z" => Vec3::new(1., 0., 1.),
            _ => unreachable!(),
        };
        let amt: f32 = caps[2]
            .parse()
            .map_err(|_| io::Error::new(ErrorKind::InvalidData, "Invalid angle."))?;

        let rotation = Quaternion::from_axis_angle(axis, amt * TAU / 360.);
        scene.camera.orientation = rotation * scene.camera.orientation;
        engine_updates.camera = true;
    }

    // todo: DRY with roll
    if let Some(caps) = re_turn.captures(&input) {
        // todo: I'm not sure how this works. How should it be different from roll?
        let axis = match caps[1].to_lowercase().as_ref() {
            "x" => Vec3::new(1., 0., 0.),
            "y" => Vec3::new(0., 1., 0.),
            "z" => Vec3::new(1., 0., 1.),
            _ => unreachable!(),
        };
        let amt: f32 = caps[2]
            .parse()
            .map_err(|_| io::Error::new(ErrorKind::InvalidData, "Invalid angle."))?;

        let rotation = Quaternion::from_axis_angle(axis, amt * TAU / 360.);
        scene.camera.orientation = rotation * scene.camera.orientation;
        engine_updates.camera = true;
    }

    if let Some(caps) = re_move.captures(&input) {
        let axis = match caps[1].to_lowercase().as_ref() {
            "x" => Vec3::new(1., 0., 0.),
            "y" => Vec3::new(0., 1., 0.),
            "z" => Vec3::new(1., 0., 1.),
            _ => unreachable!(),
        };
        let amt: f32 = caps[2]
            .parse()
            .map_err(|_| io::Error::new(ErrorKind::InvalidData, "Invalid move amount."))?;

        let movement = axis * amt;
        scene.camera.position += movement;
        engine_updates.camera = true;
    }

    if let Some(caps) = re_orient.captures(&input) {
        if let Some(mol) = &state.molecule {
            let atom_sel = mol.get_sel_atom(state.selection);

            if let Some(atom) = atom_sel {
                cam_look_at(&mut scene.camera, atom.posit);
                engine_updates.camera = true;
                state.ui.cam_snapshot = None;
            }
        }
    }

    if let Some(_) = re_reset.captures(&input) {
        if let Some(mol) = &state.molecule {
            reset_camera(&mut scene.camera, &mut state.ui.view_depth, mol);
            engine_updates.camera = true;
        }
    }

    Ok(String::from("Command succeeded"))
}

/// Simple autocomplete.
pub fn autocomplete_cli(input: &mut String) {
    // todo: Try to guess arguments; not just the params.
    let trimmed = input.trim().to_string();
    for cmd in CLI_CMDS {
        if cmd.starts_with(&trimmed) {
            // Start with just the command.
            if trimmed.len() < cmd.len() {
                *input = cmd.to_owned() + " ";
                // todo: Make it so it auto-positiosn the cursor at the end.
                // edit_resp.surrender_focus();
                // edit_resp.request_focus();
            } else {
                // todo: This needs work to work.
                // The command has been entered; supply arguments/actions.
                if cmd == "view" {}
            }
        }
    }
}
