use std::{
    f32::consts::TAU,
    path::{Path, PathBuf},
    str::FromStr,
    time::Instant,
};

use egui::{Color32, ComboBox, Context, RichText, Slider, TextEdit, TopBottomPanel, Ui};
use graphics::{Camera, EngineUpdates, RIGHT_VEC, Scene, UP_VEC};
use lin_alg::f32::{Quaternion, Vec3};
use na_seq::{AaIdent, ligation::ligate};

use crate::{
    CamSnapshot, Selection, State, ViewSelLevel,
    docking::{
        check_adv_avail,
        docking_prep_external::{prepare_ligand, prepare_target},
        find_optimal_pose, run_adv,
    },
    download_pdb::load_rcsb,
    mol_drawing::{MoleculeView, draw_ligand, draw_molecule},
    molecule::{BondType, Molecule, ResidueType},
    rcsb_api::{open_drugbank, open_pdb, open_pubchem},
    render::{CAM_INIT_OFFSET, RENDER_DIST},
    util::{cam_look_at, check_prefs_save, cycle_res_selected, select_from_search},
};

pub const ROW_SPACING: f32 = 10.;
pub const COL_SPACING: f32 = 30.;

const VIEW_DEPTH_MIN: u16 = 10;
pub const VIEW_DEPTH_MAX: u16 = 50;

const NEARBY_THRESH_MIN: u16 = 5;
const NEARBY_THRESH_MAX: u16 = 60;

// todo: Teese aren't reacting correctly; too slow for the values set.
// const CAM_BUTTON_POS_STEP: f32 = 30.;
// const CAM_BUTTON_ROT_STEP: f32 = TAU / 3.;
const CAM_BUTTON_POS_STEP: f32 = 30. * 3.;
const CAM_BUTTON_ROT_STEP: f32 = TAU / 3. * 3.;

const COLOR_INACTIVE: Color32 = Color32::GRAY;
const COLOR_ACTIVE: Color32 = Color32::LIGHT_GREEN;

const MAX_TITLE_LEN: usize = 120; // Number of characters to display.

fn active_color(val: bool) -> Color32 {
    if val { COLOR_ACTIVE } else { COLOR_INACTIVE }
}

/// Update the tilebar to reflect the current molecule
fn set_window_title(title: &str, scene: &mut Scene) {
    scene.window_title = title.to_owned();
    // ui.ctx().send_viewport_cmd(ViewportCommand::Title(title.to_string()));
}

fn load_file(
    path: &Path,
    state: &mut State,
    redraw: &mut bool,
    reset_cam: &mut bool,
    engine_updates: &mut EngineUpdates,
    ligand_load: bool,
) {
    state.open_molecule(path, ligand_load);

    // todo: These only if successful.
    *redraw = true;
    *reset_cam = true;
    engine_updates.entities = true;

    if ligand_load {
        state.to_save.last_ligand_opened = Some(path.to_owned());
    } else {
        state.to_save.last_opened = Some(path.to_owned());
    }

    state.update_save_prefs()
}

fn _int_field(val: &mut usize, label: &str, redraw: &mut bool, ui: &mut Ui) {
    ui.label(label);
    ui.label(label);
    let mut val_str = val.to_string();
    if ui
        .add_sized(
            [60., Ui::available_height(ui)],
            TextEdit::singleline(&mut val_str),
        )
        .changed()
    {
        if let Ok(v) = val_str.parse::<usize>() {
            *val = v;
            *redraw = true;
        }
    }
}

/// Handles keyboard and mouse input not associated with a widget.
pub fn handle_input(
    state: &mut State,
    ui: &mut Ui,
    redraw: &mut bool,
    reset_cam: &mut bool,
    engine_updates: &mut EngineUpdates,
) {
    ui.ctx().input(|ip| {
        // Check for file drop
        if let Some(dropped_files) = ip.raw.dropped_files.first() {
            if let Some(path) = &dropped_files.path {
                load_file(path, state, redraw, reset_cam, engine_updates, false);
            }
        }
    });
}

fn get_snap_name(snap: Option<usize>, snaps: &[CamSnapshot]) -> String {
    match snap {
        Some(i) => snaps[i].name.clone(),
        None => "None".to_owned(),
    }
}

fn cam_snapshots(
    cam: &mut Camera,
    state: &mut State,
    engine_updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    ui.heading("Scenes:");
    ui.label("Label:");
    ui.add(TextEdit::singleline(&mut state.ui.cam_snapshot_name).desired_width(60.));
    if ui.button("Save").clicked() {
        let name = if !state.ui.cam_snapshot_name.is_empty() {
            state.ui.cam_snapshot_name.clone()
        } else {
            format!("Scene {}", state.cam_snapshots.len() + 1)
        };

        state.cam_snapshots.push(CamSnapshot::from_cam(cam, name));
        state.ui.cam_snapshot_name = String::new();

        state.ui.cam_snapshot = Some(state.cam_snapshots.len() - 1);

        state.update_save_prefs();
    }

    let prev_snap = state.ui.cam_snapshot;
    let snap_name = get_snap_name(prev_snap, &state.cam_snapshots);

    ComboBox::from_id_salt(2)
        .width(80.)
        .selected_text(snap_name)
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut state.ui.cam_snapshot, None, "(None)");
            for (i, _snap) in state.cam_snapshots.iter().enumerate() {
                ui.selectable_value(
                    &mut state.ui.cam_snapshot,
                    Some(i),
                    get_snap_name(Some(i), &state.cam_snapshots),
                );
            }
        });

    if let Some(i) = state.ui.cam_snapshot {
        // ui.add_space(COL_SPACING);
        if ui.button(RichText::new("❌").color(Color32::RED)).clicked() {
            if i < state.cam_snapshots.len() {
                state.cam_snapshots.remove(i);
            }
            state.ui.cam_snapshot = None;
            state.update_save_prefs();
        }
    }

    if state.ui.cam_snapshot != prev_snap {
        if let Some(snap_i) = state.ui.cam_snapshot {
            let snap = &state.cam_snapshots[snap_i];
            cam.position = snap.position;
            cam.orientation = snap.orientation;
            cam.far = snap.far;

            cam.update_proj_mat(); // In case `far` etc changed.
            engine_updates.camera = true;
        }
    }
}

fn cam_controls(
    cam: &mut Camera,
    state: &mut State,
    engine_updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    // todo: Here and at init, set the camera dist dynamically based on mol size.
    // todo: Set the position not relative to 0, but  relative to the center of the atoms.

    let mut changed = false;

    ui.horizontal(|ui| {
        ui.label("Camera:");

        // Preset buttons

        if ui.button("Front").clicked() {
            if let Some(mol) = &state.molecule {
                let center: Vec3 = mol.center.into();
                cam.position =
                    Vec3::new(center.x, center.y, center.z - (mol.size + CAM_INIT_OFFSET));
                cam.orientation = Quaternion::new_identity();

                changed = true;
            }
        }

        if ui.button("Top").clicked() {
            if let Some(mol) = &state.molecule {
                let center: Vec3 = mol.center.into();
                cam.position =
                    Vec3::new(center.x, center.y + (mol.size + CAM_INIT_OFFSET), center.z);
                cam.orientation = Quaternion::from_axis_angle(RIGHT_VEC, TAU / 4.);

                changed = true;
            }
        }

        if ui.button("Left").clicked() {
            if let Some(mol) = &state.molecule {
                let center: Vec3 = mol.center.into();
                cam.position =
                    Vec3::new(center.x - (mol.size + CAM_INIT_OFFSET), center.y, center.z);
                cam.orientation = Quaternion::from_axis_angle(UP_VEC, TAU / 4.);

                changed = true;
            }
        }

        ui.add_space(COL_SPACING);

        // todo: Grey-out, instead of setting render dist. (e.g. fog)
        ui.label("Depth:");
        let depth_prev = state.ui.view_depth;
        ui.add(Slider::new(
            &mut state.ui.view_depth,
            VIEW_DEPTH_MIN..=VIEW_DEPTH_MAX,
        ));

        if state.ui.view_depth != depth_prev {
            // Interpret the slider being at max position to mean (effectively) unlimited.
            cam.far = if state.ui.view_depth == VIEW_DEPTH_MAX {
                RENDER_DIST
            } else {
                state.ui.view_depth as f32
            };
            cam.update_proj_mat();
            changed = true;
        }

        ui.add_space(COL_SPACING);

        // let mut movement_vec = None;
        // let mut rotation = None;

        // state.ui.inputs_commanded = Default::default();

        // Workaround for .is_pointer_button_down() stopping after 1s.
        // let pointer_down = {
        //     let input = ui.input(|i| i.clone());
        //     println!("Pointer: {:?}", input.pointer);
        //     input.pointer.button_down(PointerButton::Primary)
        // };

        //     if ui
        //         .button("⬅")
        //         .on_hover_text("Hotkey: A")
        //         // `is_pointer_button_down_on()` is ideal, but stops after ~1s. Using hover +
        //         // pointer.button_down check fails too. (Bug in EGUI?)
        //         .is_pointer_button_down_on()
        //     {
        //         // state.ui.inputs_commanded.left = true;
        //         movement_vec = Some(Vec3::new(-CAM_BUTTON_POS_STEP * state.ui.dt, 0., 0.));
        //     }
        //     if ui
        //         .button("➡")
        //         .on_hover_text("Hotkey: D")
        //         .is_pointer_button_down_on()
        //     {
        //         // state.ui.inputs_commanded.right = true;
        //         movement_vec = Some(Vec3::new(CAM_BUTTON_POS_STEP * state.ui.dt, 0., 0.));
        //     }
        //     if ui
        //         .button("⬇")
        //         .on_hover_text("Hotkey: C")
        //         .is_pointer_button_down_on()
        //     {
        //         // state.ui.inputs_commanded.down = true;
        //         movement_vec = Some(Vec3::new(0., -CAM_BUTTON_POS_STEP * state.ui.dt, 0.));
        //     }
        //     if ui
        //         .button("⬆")
        //         .on_hover_text("Hotkey: Space")
        //         .is_pointer_button_down_on()
        //     {
        //         // state.ui.inputs_commanded.up = true;
        //         movement_vec = Some(Vec3::new(0., CAM_BUTTON_POS_STEP * state.ui.dt, 0.));
        //     }
        //     if ui
        //         .button("⬋")
        //         .on_hover_text("Hotkey: S")
        //         .is_pointer_button_down_on()
        //     {
        //         // state.ui.inputs_commanded.back = true;
        //         movement_vec = Some(Vec3::new(0., 0., -CAM_BUTTON_POS_STEP * state.ui.dt));
        //     }
        //
        //     if ui
        //         .button("⬈")
        //         .on_hover_text("Hotkey: W")
        //         .is_pointer_button_down_on()
        //     {
        //         // println!("Flats: {:?}", fwd_btn.flags);
        //         // if fwd_btn.is_pointer_button_down_on() {
        //         // if fwd_btn.hovered() && pointer_down {
        //         // if fwd_btn.contains_pointer() && fwd_btn.clicked() {
        //         // state.ui.inputs_commanded.fwd = true;
        //         movement_vec = Some(Vec3::new(0., 0., CAM_BUTTON_POS_STEP * state.ui.dt));
        //     }
        //
        //     // Rotation (Alternative to keyboard)
        //     if ui
        //         .button("⟲")
        //         .on_hover_text("Hotkey: Q")
        //         .is_pointer_button_down_on()
        //     {
        //         // state.ui.inputs_commanded.roll_ccw = true;
        //         let fwd = cam.orientation.rotate_vec(FWD_VEC);
        //         rotation = Some(Quaternion::from_axis_angle(
        //             fwd,
        //             CAM_BUTTON_ROT_STEP * state.ui.dt,
        //         ));
        //     }
        //     if ui
        //         .button("⟳")
        //         .on_hover_text("Hotkey: R")
        //         .is_pointer_button_down_on()
        //     {
        //         // state.ui.inputs_commanded.roll_ccw = true;
        //         let fwd = cam.orientation.rotate_vec(FWD_VEC);
        //         rotation = Some(Quaternion::from_axis_angle(
        //             fwd,
        //             -CAM_BUTTON_ROT_STEP * state.ui.dt,
        //         ));
        //     }
        //
        //     if let Some(m) = movement_vec {
        //         cam.position += cam.orientation.rotate_vec(m);
        //         changed = true;
        //     }
        //
        //     if let Some(r) = rotation {
        //         cam.orientation = r * cam.orientation;
        //         changed = true;
        //     }
    });

    if changed {
        engine_updates.camera = true;
        state.ui.cam_snapshot = None;
    }
}

/// Display text of the selected atom
fn selected_data(mol: &Molecule, selection: Selection, ui: &mut Ui) {
    match selection {
        Selection::Atom(sel_i) => {
            if sel_i >= mol.atoms.len() {
                return;
            }

            let atom = &mol.atoms[sel_i];

            let aa = match atom.residue_type {
                ResidueType::AminoAcid(a) => format!("AA: {}", a.to_str(AaIdent::OneLetter)),
                _ => String::new(),
            };

            let role = match atom.role {
                Some(r) => format!("Role: {r}"),
                None => String::new(),
            };

            let mut text = format!(
                "{}  {}  El: {:?}  {aa}  {role}",
                atom.posit, atom.serial_number, atom.element
            );

            // todo: Helper fn for this, and find otehr places in the code where we need it.
            let res = &mol.residues.iter().find(|r| r.atoms.contains(&sel_i));
            if let Some(r) = res {
                text += &format!("  {}", r.descrip());
            }

            ui.label(RichText::new(text).color(Color32::GOLD));
        }
        Selection::Residue(sel_i) => {
            if sel_i >= mol.residues.len() {
                return;
            }

            let res = &mol.residues[sel_i];
            ui.label(RichText::new(res.descrip()).color(Color32::GOLD));
        }
        Selection::None => (),
    }
}

fn residue_selector(state: &mut State, redraw: &mut bool, ui: &mut Ui) {
    // This is a bit fuzzy, as the size varies by residue name (Not always 1 for non-AAs), and index digits.
    if let Some(mol) = &state.molecule {
        if let Some(chain_i) = state.ui.chain_to_pick_res {
            if chain_i >= mol.chains.len() {
                return;
            }
            let chain = &mol.chains[chain_i];

            ui.add_space(ROW_SPACING);
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 8.0;

                for (i, res) in mol.residues.iter().enumerate() {
                    // For now, peptide residues only.
                    if let ResidueType::Water = res.res_type {
                        continue;
                    }

                    // Only let the user select residue from the selected chain. This should keep
                    // it more organized, and keep UI space used down.
                    if !chain.residues.contains(&i) {
                        continue;
                    }

                    let name = match &res.res_type {
                        ResidueType::AminoAcid(aa) => aa.to_str(AaIdent::OneLetter),
                        ResidueType::Water => "Water".to_owned(),
                        ResidueType::Other(name) => name.clone(),
                    };

                    let mut color = Color32::GRAY;
                    if let Selection::Residue(sel_i) = state.selection {
                        if sel_i == i {
                            color = COLOR_ACTIVE;
                        }
                    }
                    if ui
                        .button(
                            RichText::new(format!("{} {name}", res.serial_number))
                                .size(10.)
                                .color(color),
                        )
                        .clicked()
                    {
                        state.ui.view_sel_level = ViewSelLevel::Residue;
                        state.selection = Selection::Residue(i);

                        *redraw = true;
                    }
                }
            });
        }
    }
}

/// Toggles chain visibility
fn chain_selector(state: &mut State, redraw: &mut bool, ui: &mut Ui) {
    // todo: For now, DRY with res selec
    ui.horizontal(|ui| {
        ui.label("Chain visibility:");
        if let Some(mol) = &mut state.molecule {
            for chain in &mut mol.chains {
                let color = active_color(chain.visible);

                if ui
                    .button(RichText::new(chain.id.clone()).color(color))
                    .clicked()
                {
                    chain.visible = !chain.visible;
                    *redraw = true;
                }
            }

            ui.add_space(COL_SPACING);

            ui.label("Select residues from:");

            for (i, chain) in mol.chains.iter().enumerate() {
                let mut color = Color32::GRAY;
                if let Some(i_sel) = state.ui.chain_to_pick_res {
                    if i == i_sel {
                        color = COLOR_ACTIVE
                    }
                }
                if ui
                    .button(RichText::new(chain.id.clone()).color(color))
                    .clicked()
                {
                    // Toggle behavior.
                    if let Some(sel_i) = state.ui.chain_to_pick_res {
                        if i == sel_i {
                            state.ui.chain_to_pick_res = None;
                        } else {
                            state.ui.chain_to_pick_res = Some(i);
                        }
                    } else {
                        state.ui.chain_to_pick_res = Some(i);
                    }
                    state.volatile.ui_height = ui.ctx().used_size().y;
                }
            }

            if state.ui.chain_to_pick_res.is_some() {
                if ui.button("(None)").clicked() {
                    state.ui.chain_to_pick_res = None;
                    state.volatile.ui_height = ui.ctx().used_size().y;
                }
            }
        }
    });
}

fn residue_search(state: &mut State, redraw: &mut bool, ui: &mut Ui) {
    ui.horizontal(|ui| {
        let sel_prev = state.selection;
        ui.label("Find residue:");
        if ui
            .add(TextEdit::singleline(&mut state.ui.residue_search).desired_width(60.))
            .changed()
        {
            select_from_search(state);
        }

        if sel_prev != state.selection {
            *redraw = true;
        }

        // todo: for UI adccessbility
        // if key_up_is_down {
        //     up_button.highlight();
        // }
        if state.molecule.is_some() {
            if ui
                .button("Prev AA")
                .on_hover_text("Hotkey: Left arrow")
                .clicked()
            {
                cycle_res_selected(state, true);
                *redraw = true;
            }
            // todo: DRY
            if ui
                .button("Next AA")
                .on_hover_text("Hotkey: Right arrow")
                .clicked()
            {
                cycle_res_selected(state, false);
                *redraw = true;
            }
        }

        ui.add_space(COL_SPACING);

        // todo: Consider removing, and doing automatically on load.
        // if let Some(mol) = &mut state.molecule {
        //     if ui.button(RichText::new("Add H")).clicked() {
        //         mol.populate_hydrogens();
        //         *redraw = true;
        //     }
        // }

        // todo: Delegate to a docking fn.
        if state.molecule.is_some() && state.ligand.is_some() {
            let ligand = &mut state.ligand.as_mut().unwrap();

            if state.babel_avail {
                if ui.button("Prepare (Babel)").clicked() {
                    let mut success_tgt = false;
                    let mut success_ligand = false;
                    // todo: We may need to save path with the molecule and ligand.
                    if let Some(path) = &state.to_save.last_opened {
                        if let Err(e) =
                            prepare_target(path, &state.molecule.as_ref().unwrap().ident)
                        {
                            eprintln!("Error: Unable to process target molecule: {e:?}");
                        } else {
                            success_tgt = true;
                        }
                    }

                    if let Some(path) = &state.to_save.last_ligand_opened {
                        if let Err(e) = prepare_ligand(path, &ligand.molecule.ident, false) {
                            eprintln!("Error: Unable to process ligand molecule: {e:?}");
                        } else {
                            success_ligand = true;
                        }
                    }

                    // This is a loose proxy.
                    if success_tgt && success_ligand {
                        state.docking_ready = true;
                    }
                }
            }

            if ui.button("Save PDBQT").clicked() {
                state.volatile.save_pdbqt_dialog.pick_directory();
            }

            // if state.docking_ready {
            if ui.button("Dock").clicked() {
                // let tgt = state.molecule.as_ref().unwrap();
                let mol = state.molecule.as_ref().unwrap();
                // find_optimal_pose(tgt, ligand, &Default::default());

                // Allow the user to select the autodock executable.
                if state.to_save.autodock_vina_path.is_none() {
                    state.volatile.autodock_path_dialog.pick_file();
                }

                if let Some(vina_path) = &state.to_save.autodock_vina_path {
                    match run_adv(
                        &ligand.docking_init,
                        vina_path,
                        &PathBuf::from_str(&format!("{}_target.pdbqt", mol.ident)).unwrap(),
                        &PathBuf::from_str(&format!("{}_ligand.pdbqt", ligand.molecule.ident))
                            .unwrap(),
                        // &PathBuf::from_str("target_prepped.pdbqt").unwrap(),
                        // &PathBuf::from_str("ligand_prepped.pdbqt").unwrap(),
                    ) {
                        Ok(r) => println!("Docking successful"),
                        Err(e) => eprintln!("Docking failed: {e:?}"),
                    }
                } else {
                    eprintln!("No Autodock Vina install located yet.");
                }
            }

            ui.label("Docking site setup:");
            ui.label("Center:");

            if ui
                .add(TextEdit::singleline(&mut state.ui.docking_site_x).desired_width(30.))
                .changed()
            {
                if let Ok(v) = state.ui.docking_site_x.parse::<f64>() {
                    ligand.docking_init.site_center.x = v;
                    *redraw = true;
                }
            }
            if ui
                .add(TextEdit::singleline(&mut state.ui.docking_site_y).desired_width(30.))
                .changed()
            {
                if let Ok(v) = state.ui.docking_site_y.parse::<f64>() {
                    ligand.docking_init.site_center.y = v;
                    *redraw = true;
                }
            }
            if ui
                .add(TextEdit::singleline(&mut state.ui.docking_site_z).desired_width(30.))
                .changed()
            {
                if let Ok(v) = state.ui.docking_site_z.parse::<f64>() {
                    ligand.docking_init.site_center.z = v;
                    *redraw = true;
                }
            }

            ui.label("Size:");
            if ui
                .add(TextEdit::singleline(&mut state.ui.docking_site_size).desired_width(30.))
                .changed()
            {
                if let Ok(v) = state.ui.docking_site_size.parse::<f64>() {
                    ligand.docking_init.site_box_size = v;
                    *redraw = true;
                }
            }
        }
        // }

        ui.add_space(COL_SPACING);

        ui.label(RichText::new("🔘AV").color(active_color(state.ui.autodock_path_valid)))
            .on_hover_text("Autodock Vina available (Docking)");

        ui.label(RichText::new("🔘OB").color(active_color(state.babel_avail)))
            .on_hover_text("Open Babel available (Docking prep)");
    });
}

fn selection_section(
    state: &mut State,
    scene: &mut Scene,
    redraw: &mut bool,
    engine_updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    // todo: DRY with view.
    ui.horizontal(|ui| {
        ui.label("View/Select:");
        let prev_view = state.ui.view_sel_level;
        ComboBox::from_id_salt(1)
            .width(80.)
            .selected_text(state.ui.view_sel_level.to_string())
            .show_ui(ui, |ui| {
                for view in &[ViewSelLevel::Atom, ViewSelLevel::Residue] {
                    ui.selectable_value(&mut state.ui.view_sel_level, *view, view.to_string());
                }
            });

        if state.ui.view_sel_level != prev_view {
            *redraw = true;
            // Kludge to prevent surprising behavior.
            state.selection = Selection::None;
        }

        ui.add_space(COL_SPACING);

        ui.label("Filter nearby:");
        if ui.checkbox(&mut state.ui.show_nearby_only, "").changed() {
            *redraw = true;
        }

        if state.ui.show_nearby_only {
            ui.label("Dist:");
            let dist_prev = state.ui.nearby_dist_thresh;
            ui.add(Slider::new(
                &mut state.ui.nearby_dist_thresh,
                NEARBY_THRESH_MIN..=NEARBY_THRESH_MAX,
            ));

            if state.ui.nearby_dist_thresh != dist_prev {
                *redraw = true;
            }
        }

        ui.add_space(COL_SPACING);

        if let Some(mol) = &state.molecule {
            if state.selection != Selection::None {
                selected_data(mol, state.selection, ui);

                ui.add_space(COL_SPACING / 2.);
                if ui.button("Move cam to").clicked() {
                    let atom_sel = mol.get_sel_atom(state.selection);

                    if let Some(atom) = atom_sel {
                        cam_look_at(&mut scene.camera, atom.posit);
                        engine_updates.camera = true;
                        state.ui.cam_snapshot = None;
                    }
                }
            }
        }
    });
}

fn mol_descrip(mol: &Molecule, ui: &mut Ui) {
    ui.heading(RichText::new(mol.ident.clone()).color(Color32::GOLD));

    if let Some(metadata) = &mol.metadata {
        // Limit size to prevent UI problems.
        let mut title: String = metadata
            .prim_cit_title
            .chars()
            .take(MAX_TITLE_LEN)
            .collect();
        if title.len() != metadata.prim_cit_title.len() {
            title += "...";
        }
        ui.label(RichText::new(title).color(Color32::WHITE).size(12.));
    }

    if mol.ident.len() <= 5 {
        // todo: You likely need a better approach.
        if ui.button("View on RCSB").clicked() {
            open_pdb(&mol.ident);
        }
    }

    if let Some(id) = &mol.drugbank_id {
        if ui.button("View on Drugbank").clicked() {
            open_drugbank(id);
        }
    }

    if let Some(id) = mol.pubchem_cid {
        if ui.button("View on PubChem").clicked() {
            open_pubchem(id);
        }
    }
}

/// This function draws the (immediate-mode) GUI.
/// [UI items](https://docs.rs/egui/latest/egui/struct.Ui.html)
pub fn ui_handler(state: &mut State, ctx: &Context, scene: &mut Scene) -> EngineUpdates {
    let mut engine_updates = EngineUpdates::default();

    check_prefs_save(state);

    // return  engine_updates;
    let mut redraw = false;
    let mut reset_cam = false;

    // For getting DT for certain buttons when held. Does not seem to be the same as the 3D render DT.
    let start = Instant::now();

    TopBottomPanel::top("0").show(ctx, |ui| {
        ui.spacing_mut().slider_width = 120.;

        handle_input(state, ui, &mut redraw, &mut reset_cam, &mut engine_updates);

        ui.horizontal(|ui| {
            let mut metadata_loaded = false; // avoids borrow error.
            if let Some(mol) = &mut state.molecule {
                mol_descrip(mol, ui);
                ui.add_space(COL_SPACING);
            }

            if ui.button("Open").clicked() {
                state.volatile.load_dialog.pick_file();
            }

            if ui.button("Open ligand").clicked() {
                state.volatile.load_ligand_dialog.pick_file();
            }

            // if ui.button("Get RCSB").clicked() {
            //     match load_pdb_metadata(&mol.ident) {
            //         Ok(d) => {
            //             println!("Metadata loaded: {d:?}");
            //             mol.metadata = Some(d);
            //             metadata_loaded = true;
            //         },
            //         Err(_) => eprintln!("Error getting PDB metadata"),
            //     }
            // }

            if metadata_loaded {
                state.update_save_prefs();
            }

            ui.add_space(COL_SPACING);

            ui.add_space(COL_SPACING);
            ui.label("Query RCSB. Ident:");
            ui.add(TextEdit::singleline(&mut state.ui.rcsb_input).desired_width(40.));

            if !state.ui.rcsb_input.is_empty() {
                if ui.button("Download from RCSB").clicked() {
                    match load_rcsb(&state.ui.rcsb_input) {
                        Ok(pdb) => {
                            state.pdb = Some(pdb);
                            state.molecule = Some(Molecule::from_pdb(state.pdb.as_ref().unwrap()));
                            state.update_from_prefs();

                            redraw = true;
                            reset_cam = true;
                        }
                        Err(_e) => {
                            eprintln!("Error loading PDB file");
                        }
                    }
                }
            }

            ui.add_space(COL_SPACING);

            ui.label("View:");
            let prev_view = state.ui.mol_view;
            ComboBox::from_id_salt(0)
                .width(80.)
                .selected_text(state.ui.mol_view.to_string())
                .show_ui(ui, |ui| {
                    for view in &[
                        MoleculeView::Sticks,
                        MoleculeView::Backbone,
                        MoleculeView::BallAndStick,
                        // MoleculeView::Cartoon,
                        MoleculeView::SpaceFill,
                        // MoleculeView::Surface, // Partially-implemented, but broken/crashes.
                        // MoleculeView::Mesh,
                        MoleculeView::Dots,
                    ] {
                        ui.selectable_value(&mut state.ui.mol_view, *view, view.to_string());
                    }
                });

            if state.ui.mol_view != prev_view {
                redraw = true;
            }
        });

        ui.add_space(ROW_SPACING);
        if let Some(ligand) = &mut state.ligand {
            ui.horizontal(|ui| {
                mol_descrip(&ligand.molecule, ui);
            });
        }

        ui.add_space(ROW_SPACING);
        selection_section(state, scene, &mut redraw, &mut engine_updates, ui);

        ui.add_space(ROW_SPACING);

        ui.horizontal(|ui| {
            cam_controls(&mut scene.camera, state, &mut engine_updates, ui);
            // ui.add_space(COL_SPACING * 2.);
            cam_snapshots(&mut scene.camera, state, &mut engine_updates, ui);
        });

        ui.add_space(ROW_SPACING);

        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    chain_selector(state, &mut redraw, ui);

                    ui.add_space(COL_SPACING);

                    ui.label("Vis:");

                    let color = active_color(!state.ui.visibility.hide_non_hetero);
                    if ui.button(RichText::new("Peptide").color(color)).clicked() {
                        state.ui.visibility.hide_non_hetero = !state.ui.visibility.hide_non_hetero;
                        redraw = true;
                    }

                    let color = active_color(!state.ui.visibility.hide_hetero);
                    if ui.button(RichText::new("Hetero").color(color)).clicked() {
                        state.ui.visibility.hide_hetero = !state.ui.visibility.hide_hetero;
                        redraw = true;
                    }

                    ui.add_space(COL_SPACING / 2.);

                    if !state.ui.visibility.hide_non_hetero {
                        // Subset of peptide.
                        let color = active_color(!state.ui.visibility.hide_sidechains);
                        if ui
                            .button(RichText::new("Sidechains").color(color))
                            .clicked()
                        {
                            state.ui.visibility.hide_sidechains =
                                !state.ui.visibility.hide_sidechains;
                            redraw = true;
                        }

                        let color = active_color(!state.ui.visibility.hide_hydrogen);
                        if ui.button(RichText::new("H").color(color)).clicked() {
                            state.ui.visibility.hide_hydrogen = !state.ui.visibility.hide_hydrogen;
                            redraw = true;
                        }
                    }

                    if !state.ui.visibility.hide_hetero {
                        // Subset of hetero.
                        let color = active_color(!state.ui.visibility.hide_water);
                        if ui.button(RichText::new("Water").color(color)).clicked() {
                            state.ui.visibility.hide_water = !state.ui.visibility.hide_water;
                            redraw = true;
                        }
                    }

                    if state.ligand.is_some() {
                        let color = active_color(!state.ui.visibility.hide_ligand);
                        if ui.button(RichText::new("Ligand").color(color)).clicked() {
                            state.ui.visibility.hide_ligand = !state.ui.visibility.hide_ligand;
                            redraw = true;
                        }
                    }

                    let color = active_color(!state.ui.visibility.hide_h_bonds);
                    if ui.button(RichText::new("H bonds").color(color)).clicked() {
                        state.ui.visibility.hide_h_bonds = !state.ui.visibility.hide_h_bonds;
                        redraw = true;
                    }
                });

                // todo: Show hide based on AaCategory? i.e. residue.amino_acid.category(). Hydrophilic, acidic etc.

                residue_selector(state, &mut redraw, ui);
            });
        });

        ui.add_space(ROW_SPACING);

        residue_search(state, &mut redraw, ui);

        // todo: Allow switching between chains and secondary-structure features here.

        ui.add_space(ROW_SPACING / 2.);

        if let Some(path) = &state.volatile.load_dialog.take_picked() {
            load_file(
                path,
                state,
                &mut redraw,
                &mut reset_cam,
                &mut engine_updates,
                false,
            );
        }

        if let Some(path) = &state.volatile.load_ligand_dialog.take_picked() {
            load_file(
                path,
                state,
                &mut redraw,
                &mut reset_cam,
                &mut engine_updates,
                true,
            );
        }

        if let Some(path) = &state.volatile.autodock_path_dialog.take_picked() {
            state.ui.autodock_path_valid = check_adv_avail(path);
            if state.ui.autodock_path_valid {
                state.to_save.autodock_vina_path = Some(path.to_owned());
                state.update_save_prefs();
            }
        }

        if let Some(path_dir) = &state.volatile.save_pdbqt_dialog.take_picked() {
            if let Some(mol) = &state.molecule {
                let filename = format!("{}_target.pdbqt", mol.ident);
                let path = Path::new(path_dir).join(filename);
                if mol.save_pdbqt(&path, None).is_err() {
                    eprintln!("Error saving PDBQT target");
                }
            }

            if let Some(lig) = &state.ligand {
                let filename = format!("{}_ligand.pdbqt", lig.molecule.ident);
                let path = Path::new(path_dir).join(filename);
                if lig.molecule.save_pdbqt(&path, None).is_err() {
                    eprintln!("Error saving PDBQT ligand");
                }
            }
        }

        if redraw {
            draw_molecule(state, scene, reset_cam);
            draw_ligand(state, scene, reset_cam);

            if let Some(mol) = &state.molecule {
                set_window_title(&mol.ident, scene);
            }

            engine_updates.entities = true;
        }
    });

    // At init only.
    if state.volatile.ui_height < f32::EPSILON {
        state.volatile.ui_height = ctx.used_size().y;
    }

    state.volatile.load_dialog.update(ctx);
    state.volatile.load_ligand_dialog.update(ctx);
    state.volatile.autodock_path_dialog.update(ctx);
    state.volatile.save_pdbqt_dialog.update(ctx);

    state.ui.dt = start.elapsed().as_secs_f32();

    engine_updates
}
