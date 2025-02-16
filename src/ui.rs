use std::{f32::consts::TAU, path::Path, time::Instant};

use egui::{
    Color32, ComboBox, Context, RichText, ScrollArea, Slider, TextEdit, TopBottomPanel, Ui,
};
use graphics::{Camera, EngineUpdates, Scene, FWD_VEC, RIGHT_VEC, UP_VEC};
use lin_alg::f32::{Quaternion, Vec3};
use na_seq::AaIdent;

use crate::{
    download_pdb::load_rcsb,
    molecule::{Molecule, ResidueType},
    pdb::load_pdb,
    rcsb_api::open_pdb,
    render::{draw_molecule, MoleculeView, CAM_INIT_OFFSET, RENDER_DIST},
    util::{cam_look_at, check_prefs_save, cycle_res_selected, select_from_search},
    CamSnapshot, Selection, State, StateUi, StateVolatile, ViewSelLevel,
};

pub const ROW_SPACING: f32 = 10.;
pub const COL_SPACING: f32 = 30.;

const VIEW_DEPTH_MIN: u16 = 10;
pub const VIEW_DEPTH_MAX: u16 = 200;

const NEARBY_THRESH_MIN: u16 = 5;
const NEARBY_THRESH_MAX: u16 = 60;

const CAM_BUTTON_POS_STEP: f32 = 30.;
const CAM_BUTTON_ROT_STEP: f32 = TAU / 3.;

const COLOR_ACTIVE: Color32 = Color32::LIGHT_GREEN;

const MAX_TITLE_LEN: usize = 120; // Number of characters to display.

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
) {
    let pdb = load_pdb(&path);

    if let Ok(p) = pdb {
        state.pdb = Some(p);
        state.molecule = Some(Molecule::from_pdb(state.pdb.as_ref().unwrap()));
        state.update_from_prefs();

        *redraw = true;
        *reset_cam = true;
        engine_updates.entities = true;
    } else {
        eprintln!("Error loading PDB file");
    }
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
                load_file(&path, state, redraw, reset_cam, engine_updates)
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
    ui.heading("Views:");
    ui.label("Label:");
    ui.add(TextEdit::singleline(&mut state.ui.cam_snapshot_name).desired_width(60.));
    if ui.button("Save").clicked() {
        let name = if !state.ui.cam_snapshot_name.is_empty() {
            state.ui.cam_snapshot_name.clone()
        } else {
            format!("Snap {}", state.cam_snapshots.len() + 1)
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
    state_ui: &mut StateUi,
    volatile: &StateVolatile,
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
            cam.position = Vec3::new(
                volatile.mol_center.x,
                volatile.mol_center.y,
                volatile.mol_center.z - (volatile.mol_size + CAM_INIT_OFFSET),
            );
            cam.orientation = Quaternion::new_identity();

            changed = true;
        }

        if ui.button("Top").clicked() {
            cam.position = Vec3::new(
                volatile.mol_center.x,
                volatile.mol_center.y + (volatile.mol_size + CAM_INIT_OFFSET),
                volatile.mol_center.z,
            );
            cam.orientation = Quaternion::from_axis_angle(RIGHT_VEC, TAU / 4.);

            changed = true;
        }

        if ui.button("Left").clicked() {
            cam.position = Vec3::new(
                volatile.mol_center.x - (volatile.mol_size + CAM_INIT_OFFSET),
                volatile.mol_center.y,
                volatile.mol_center.z,
            );
            cam.orientation = Quaternion::from_axis_angle(UP_VEC, TAU / 4.);

            changed = true;
        }

        ui.add_space(COL_SPACING);

        // todo: Grey-out, instead of setting render dist. (e.g. fog)
        ui.label("Depth:");
        let depth_prev = state_ui.view_depth;
        ui.add(Slider::new(
            &mut state_ui.view_depth,
            VIEW_DEPTH_MIN..=VIEW_DEPTH_MAX,
        ));

        if state_ui.view_depth != depth_prev {
            // Interpret the slider being at max position to mean (effectively) unlimited.
            cam.far = if state_ui.view_depth == VIEW_DEPTH_MAX {
                RENDER_DIST
            } else {
                state_ui.view_depth as f32
            };
            cam.update_proj_mat();
            changed = true;
        }

        ui.add_space(COL_SPACING);

        let mut movement_vec = None;
        let mut rotation = None;

        state_ui.inputs_commanded = Default::default();

        if ui
            .button("⬅")
            .on_hover_text("Hotkey: A")
            // `is_pointer_button_down_on()` is ideal, but stops after ~1s. Using hover +
            // pointer.button_down check fails too. (Bug in EGUI?)
            // .is_pointer_button_down_on()
            .hovered()
        {
            state_ui.inputs_commanded.left = true;
            movement_vec = Some(Vec3::new(-CAM_BUTTON_POS_STEP * state_ui.dt, 0., 0.));
        }
        if ui.button("➡").on_hover_text("Hotkey: D").hovered() {
            state_ui.inputs_commanded.right = true;
            movement_vec = Some(Vec3::new(CAM_BUTTON_POS_STEP * state_ui.dt, 0., 0.));
        }
        if ui.button("⬇").on_hover_text("Hotkey: C").hovered() {
            state_ui.inputs_commanded.down = true;
            movement_vec = Some(Vec3::new(0., -CAM_BUTTON_POS_STEP * state_ui.dt, 0.));
        }
        if ui.button("⬆").on_hover_text("Hotkey: Space").hovered() {
            state_ui.inputs_commanded.up = true;
            movement_vec = Some(Vec3::new(0., CAM_BUTTON_POS_STEP * state_ui.dt, 0.));
        }
        if ui.button("⬋").on_hover_text("Hotkey: S").hovered() {
            state_ui.inputs_commanded.back = true;
            movement_vec = Some(Vec3::new(0., 0., -CAM_BUTTON_POS_STEP * state_ui.dt));
        }
        if ui.button("⬈").on_hover_text("Hotkey: W").hovered() {
            state_ui.inputs_commanded.fwd = true;
            movement_vec = Some(Vec3::new(0., 0., CAM_BUTTON_POS_STEP * state_ui.dt));
        }

        // Rotation (Alternative to keyboard)
        if ui.button("⟲").on_hover_text("Hotkey: Q").hovered() {
            state_ui.inputs_commanded.roll_ccw = true;
            let fwd = cam.orientation.rotate_vec(FWD_VEC);
            rotation = Some(Quaternion::from_axis_angle(
                fwd,
                CAM_BUTTON_ROT_STEP * state_ui.dt,
            ));
        }
        if ui.button("⟳").on_hover_text("Hotkey: R").hovered() {
            state_ui.inputs_commanded.roll_ccw = true;
            let fwd = cam.orientation.rotate_vec(FWD_VEC);
            rotation = Some(Quaternion::from_axis_angle(
                fwd,
                -CAM_BUTTON_ROT_STEP * state_ui.dt,
            ));
        }

        if let Some(m) = movement_vec {
            cam.position += cam.orientation.rotate_vec(m);
            changed = true;
        }

        if let Some(r) = rotation {
            cam.orientation = r * cam.orientation;
            changed = true;
        }
    });

    if changed {
        engine_updates.camera = true;
        state_ui.cam_snapshot = None;
    }
}

/// Display text of the selected atom
fn selected_data(mol: &Molecule, selection: Selection, ui: &mut Ui) {
    match selection {
        Selection::Atom(sel) => {
            let atom = &mol.atoms[sel];

            let aa = match atom.amino_acid {
                Some(a) => format!("AA: {}", a.to_str(AaIdent::OneLetter)),
                None => String::new(),
            };

            let role = match atom.role {
                Some(r) => format!("Role: {r}"),
                None => String::new(),
            };

            ui.label(
                RichText::new(format!(
                    // todo: Coorsd are temp
                    "{}, {} El: {:?}, {aa} {role}",
                    atom.posit, atom.serial_number, atom.element
                ))
                .color(Color32::GOLD),
            );
        }
        Selection::Residue(sel_i) => {
            let res = &mol.residues[sel_i];
            let name = match &res.res_type {
                ResidueType::AminoAcid(aa) => aa.to_str(AaIdent::ThreeLetters),
                ResidueType::Water => "Water".to_owned(),
                ResidueType::Other(name) => name.clone(),
            };

            // todo: Sequesnce number etc.
            ui.label(
                RichText::new(format!("Res: {}: {name}", res.serial_number)).color(Color32::GOLD),
            );
        }
        Selection::None => (),
    }
}

fn residue_selector(state: &mut State, redraw: &mut bool, ui: &mut Ui) {
    // This is a bit fuzzy, as the size varies by residue name (Not always 1 for non-AAs), and index digits.
    const BUTTON_WIDTH: f32 = 16.;

    if let Some(mol) = &state.molecule {
        if let Some(chain_i) = state.ui.chain_to_pick_res {
            let chain = &mol.chains[chain_i];

            ui.add_space(ROW_SPACING);
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 8.0;

                for (i, res) in mol.residues.iter().enumerate() {
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
                let color = if chain.visible {
                    COLOR_ACTIVE
                } else {
                    Color32::GRAY
                };
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
    });
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

                if ui.button("Open").clicked() {
                    state.volatile.load_dialog.pick_file();
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

                if ui.button("Open RCSB").clicked() {
                    open_pdb(&mol.ident);
                }
            }
            if metadata_loaded {
                state.update_save_prefs();
            }

            ui.add_space(COL_SPACING);

            ui.add_space(COL_SPACING);
            ui.label("RCSB ident:");
            ui.add(TextEdit::singleline(&mut state.ui.rcsb_input).desired_width(60.));

            if ui.button("Download mol from RCSB").clicked() {
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

            ui.add_space(COL_SPACING);

            ui.label("View:");
            let prev_view = state.ui.mol_view;
            ComboBox::from_id_salt(0)
                .width(80.)
                .selected_text(state.ui.mol_view.to_string())
                .show_ui(ui, |ui| {
                    for view in &[
                        MoleculeView::Sticks,
                        MoleculeView::Ribbon,
                        MoleculeView::BallAndStick,
                        MoleculeView::Cartoon,
                        MoleculeView::Spheres,
                        MoleculeView::Surface,
                        MoleculeView::Mesh,
                        MoleculeView::Dots,
                    ] {
                        ui.selectable_value(&mut state.ui.mol_view, *view, view.to_string());
                    }
                });

            if state.ui.mol_view != prev_view {
                redraw = true;
            }

            ui.add_space(COL_SPACING);

            // todo: DRY with view.
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
                redraw = true;
                // Kludge to prevent surprising behavior.
                state.selection = Selection::None;
            }

            ui.add_space(COL_SPACING);

            ui.label("Filter nearby:");
            if ui.checkbox(&mut state.ui.show_nearby_only, "").changed() {
                redraw = true;
            }

            ui.add_space(COL_SPACING);

            ui.label("Nearby filter:");
            let dist_prev = state.ui.nearby_dist_thresh;
            ui.add(Slider::new(
                &mut state.ui.nearby_dist_thresh,
                NEARBY_THRESH_MIN..=NEARBY_THRESH_MAX,
            ));

            if state.ui.nearby_dist_thresh != dist_prev {
                redraw = true;
            }

            ui.add_space(COL_SPACING);

            if let Some(mol) = &state.molecule {
                if state.selection != Selection::None {
                    selected_data(mol, state.selection, ui);

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

        ui.add_space(ROW_SPACING);

        ui.horizontal(|ui| {
            cam_controls(
                &mut scene.camera,
                &mut state.ui,
                &state.volatile,
                &mut engine_updates,
                ui,
            );
            ui.add_space(COL_SPACING * 2.);
            cam_snapshots(&mut scene.camera, state, &mut engine_updates, ui);
        });

        ui.add_space(ROW_SPACING);

        ui.horizontal(|ui| {
            chain_selector(state, &mut redraw, ui);
            residue_selector(state, &mut redraw, ui);
            ui.add_space(COL_SPACING);
            if ui
                .checkbox(&mut state.ui.hide_sidechains, "Hide sidechains")
                .changed()
            {
                redraw = true;
            }
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
                // ui,
            );
        }

        if redraw {
            draw_molecule(state, scene, reset_cam);

            if let Some(mol) = &state.molecule {
                set_window_title(&mol.ident, scene);
            }

            engine_updates.entities = true;
        }
    });

    // At init only.
    if state.volatile.ui_height < f32::EPSILON {
        println!("Setting UI height: {:?}", ctx.used_size().y);
        state.volatile.ui_height = ctx.used_size().y;
    }

    state.volatile.load_dialog.update(ctx);

    state.ui.dt = start.elapsed().as_secs_f32();

    engine_updates
}
