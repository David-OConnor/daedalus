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
    render::{draw_molecule, MoleculeView, CAM_INIT_OFFSET, RENDER_DIST},
    util::{cam_look_at, cycle_res_selected, select_from_search},
    CamSnapshot, Selection, State, StateUi, StateVolatile, ViewSelLevel,
};

pub const ROW_SPACING: f32 = 10.;
pub const COL_SPACING: f32 = 30.;

const VIEW_DEPTH_MIN: u16 = 10;
pub const VIEW_DEPTH_MAX: u16 = 200;

const NEARBY_THRESH_MIN: u16 = 5;
const NEARBY_THRESH_MAX: u16 = 60;

const CAM_BUTTON_POS_STEP: f32 = 20.;
const CAM_BUTTON_ROT_STEP: f32 = TAU / 4.;

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
        Some(i) => {
            let snap = &snaps[i];
            match &snap.name {
                Some(name) => name.to_owned(),
                None => i.to_string(),
            }
        }
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
    ui.add(TextEdit::singleline(&mut state.ui.cam_snapshot_name).desired_width(80.));
    if ui.button("Save").clicked() {
        state
            .cam_snapshots
            .push(CamSnapshot::from_cam(cam, &state.ui.cam_snapshot_name));
        state.ui.cam_snapshot_name = String::new();
    }

    let prev_snap = state.ui.cam_snapshot;
    let snap_name = get_snap_name(prev_snap, &state.cam_snapshots);

    ComboBox::from_id_salt(2)
        .width(80.)
        .selected_text(snap_name)
        .show_ui(ui, |ui| {
            for (i, _snap) in state.cam_snapshots.iter().enumerate() {
                ui.selectable_value(
                    &mut state.ui.cam_snapshot,
                    Some(i),
                    get_snap_name(Some(i), &state.cam_snapshots),
                );
            }
        });

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

        // todo: for UI adccessbility
        // if key_up_is_down {
        //     up_button.highlight();
        // }
        // Movement (Alternative to keyboard)
        if ui
            .button("⬅")
            .on_hover_text("Hotkey: A")
            .is_pointer_button_down_on()
        {
            movement_vec = Some(Vec3::new(-CAM_BUTTON_POS_STEP * state_ui.dt, 0., 0.));
        }
        if ui
            .button("➡")
            .on_hover_text("Hotkey: D")
            .is_pointer_button_down_on()
        {
            movement_vec = Some(Vec3::new(CAM_BUTTON_POS_STEP * state_ui.dt, 0., 0.));
        }
        if ui
            .button("⬇")
            .on_hover_text("Hotkey: C")
            .is_pointer_button_down_on()
        {
            movement_vec = Some(Vec3::new(0., -CAM_BUTTON_POS_STEP * state_ui.dt, 0.));
        }
        if ui
            .button("⬆")
            .on_hover_text("Hotkey: Space")
            .is_pointer_button_down_on()
        {
            movement_vec = Some(Vec3::new(0., CAM_BUTTON_POS_STEP * state_ui.dt, 0.));
        }
        if ui
            .button("⬋")
            .on_hover_text("Hotkey: S")
            .is_pointer_button_down_on()
        {
            movement_vec = Some(Vec3::new(0., 0., -CAM_BUTTON_POS_STEP * state_ui.dt));
        }
        if ui
            .button("⬈")
            .on_hover_text("Hotkey: W")
            .is_pointer_button_down_on()
        {
            movement_vec = Some(Vec3::new(0., 0., CAM_BUTTON_POS_STEP * state_ui.dt));
        }

        // Rotation (Alternative to keyboard)
        if ui
            .button("⟲")
            .on_hover_text("Hotkey: Q")
            .is_pointer_button_down_on()
        {
            let fwd = cam.orientation.rotate_vec(FWD_VEC);
            rotation = Some(Quaternion::from_axis_angle(
                fwd,
                CAM_BUTTON_ROT_STEP * state_ui.dt,
            ));
        }
        if ui
            .button("⟳")
            .on_hover_text("Hotkey: R")
            .is_pointer_button_down_on()
        {
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
    }
}

/// Display text of the selected atom
fn selected_data(mol: &Molecule, selection: Selection, ui: &mut Ui) {
    match selection {
        Selection::Atom(sel) => {
            let atom = &mol.atoms[sel];
            ui.label(
                RichText::new(format!(
                    // todo: Coorsd are temp
                    "{}, {} El: {:?}, AA: {:?}, Role: {:?}",
                    atom.posit, atom.serial_number, atom.element, atom.amino_acid, atom.role
                ))
                .color(Color32::GOLD),
            );
        }
        Selection::Residue(sel_i) => {
            let res = &mol.residues[sel_i];
            let name = match &res.res_type {
                ResidueType::AminoAcid(aa) => aa.to_str(AaIdent::ThreeLetters),
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
    ui.horizontal(|ui| {
        if let Some(mol) = &state.molecule {
            if let Some(chain_i) = state.ui.chain_to_pick_res {
                let chain = &mol.chains[chain_i];

                for (i, res) in mol.residues.iter().enumerate() {
                    // Only let the user select residue from the selected chain. This should keep
                    // it more organized, and keep UI space used down.

                    if !chain.residues.contains(&i) {
                        continue;
                    }

                    let name = match &res.res_type {
                        ResidueType::AminoAcid(aa) => aa.to_str(AaIdent::OneLetter),
                        ResidueType::Other(name) => name.clone(),
                    };

                    if ui
                        .button(format!("{}: {name}", res.serial_number))
                        .clicked()
                    {
                        state.ui.view_sel_level = ViewSelLevel::Residue;
                        state.selection = Selection::Residue(i);

                        // let res = &mol.residues[i];
                        // if !res.atoms.is_empty() {
                        //     let atom = &mol.atoms[res.atoms[0]];
                        //     cam_look_at(cam, atom.posit);
                        //     engine_updates.camera = true;
                        // }

                        *redraw = true;
                    }
                }
            }
        }
    });
}

/// Toggles chain visibility
fn chain_selector(state: &mut State, redraw: &mut bool, ui: &mut Ui) {
    // todo: For now, DRY with res selec
    ui.horizontal(|ui| {
        ui.label("Chain visibility:");
        if let Some(mol) = &mut state.molecule {
            for chain in &mut mol.chains {
                let color = if chain.visible {
                    Color32::LIGHT_BLUE
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
                        color = Color32::LIGHT_BLUE
                    }
                }
                if ui
                    .button(RichText::new(chain.id.clone()).color(color))
                    .clicked()
                {
                    state.ui.chain_to_pick_res = Some(i);
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

    // return  engine_updates;
    let mut redraw = false;
    let mut reset_cam = false;

    // For getting DT for certain buttons when held. Does not seem to be the same as the 3D render DT.
    let start = Instant::now();

    TopBottomPanel::top("0").show(ctx, |ui| {
        ui.spacing_mut().slider_width = 120.;

        handle_input(state, ui, &mut redraw, &mut reset_cam, &mut engine_updates);

        ui.horizontal(|ui| {
            if let Some(mol) = &state.molecule {
                ui.heading(RichText::new(mol.ident.clone()).color(Color32::GOLD));
            }

            ui.add_space(COL_SPACING);

            if ui.button("Open").clicked() {
                state.volatile.load_dialog.pick_file();
            }

            ui.add_space(COL_SPACING);
            ui.label("RCSB ident:");
            ui.add(TextEdit::singleline(&mut state.ui.rcsb_input).desired_width(60.));

            if ui.button("Load RCSB").clicked() {
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

            state.volatile.load_dialog.update(ctx);

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

        chain_selector(state, &mut redraw, ui);

        residue_selector(state, &mut redraw, ui);

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

    // todo: You must set the UI height when performing actions that change it! Or selection will be wonky.
    if state.volatile.ui_height < f32::EPSILON {
        state.volatile.ui_height = ctx.used_size().y;
    }

    state.volatile.load_dialog.update(ctx);

    state.ui.dt = start.elapsed().as_secs_f32();

    engine_updates
}
