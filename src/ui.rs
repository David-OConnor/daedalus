use std::{f32::consts::TAU, path::Path};

use egui::{
    Color32, ComboBox, Context, RichText, Slider, TextEdit, TopBottomPanel, Ui, ViewportCommand,
};
use graphics::{Camera, EngineUpdates, Scene, FWD_VEC, RIGHT_VEC, UP_VEC};
use lin_alg::f32::{Quaternion, Vec3};

use crate::{
    download_pdb::load_rcsb,
    molecule::Molecule,
    pdb::load_pdb,
    render::{draw_molecule, MoleculeView, CAM_INIT_OFFSET, RENDER_DIST},
    Selection, State, StateUi, ViewSelLevel,
};

pub const ROW_SPACING: f32 = 10.;
pub const COL_SPACING: f32 = 30.;

const VIEW_DEPTH_MIN: u16 = 10;
const VIEW_DEPTH_MAX: u16 = 200;

const NEARBY_THRESH_MIN: u16 = 5;
const NEARBY_THRESH_MAX: u16 = 60;

const CAM_BUTTON_POS_STEP: f32 = 5.;
const CAM_BUTTON_ROT_STEP: f32 = TAU / 24.;

/// Update the tilebar to reflect the current molecule
fn set_window_title(title: &str, ui: &mut Ui) {
    // todo: Not working. Maybe need a new way when using WGPU.
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

fn cam_controls(
    cam: &mut Camera,
    state_ui: &mut StateUi,
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
                state_ui.mol_center.x,
                state_ui.mol_center.y,
                state_ui.mol_center.z - (state_ui.mol_size + CAM_INIT_OFFSET),
            );
            cam.orientation = Quaternion::new_identity();

            changed = true;
        }

        if ui.button("Top").clicked() {
            cam.position = Vec3::new(
                state_ui.mol_center.x,
                state_ui.mol_center.y + (state_ui.mol_size + CAM_INIT_OFFSET),
                state_ui.mol_center.z,
            );
            cam.orientation = Quaternion::from_axis_angle(RIGHT_VEC, TAU / 4.);

            changed = true;
        }

        if ui.button("Left").clicked() {
            cam.position = Vec3::new(
                state_ui.mol_center.x - (state_ui.mol_size + CAM_INIT_OFFSET),
                state_ui.mol_center.y,
                state_ui.mol_center.z,
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

        // Movement (Alternative to keyboard)
        if ui.button("⬅").on_hover_text("Hotkey: A").clicked() {
            movement_vec = Some(Vec3::new(-CAM_BUTTON_POS_STEP, 0., 0.));
        }
        if ui.button("➡").on_hover_text("Hotkey: D").clicked() {
            movement_vec = Some(Vec3::new(CAM_BUTTON_POS_STEP, 0., 0.));
        }
        if ui.button("⬇").on_hover_text("Hotkey: C").clicked() {
            movement_vec = Some(Vec3::new(0., -CAM_BUTTON_POS_STEP, 0.));
        }
        if ui.button("⬆").on_hover_text("Hotkey: Space").clicked() {
            movement_vec = Some(Vec3::new(0., CAM_BUTTON_POS_STEP, 0.));
        }
        if ui.button("⬋").on_hover_text("Hotkey: S").clicked() {
            movement_vec = Some(Vec3::new(0., 0., -CAM_BUTTON_POS_STEP));
        }
        if ui.button("⬈").on_hover_text("Hotkey: W").clicked() {
            movement_vec = Some(Vec3::new(0., 0., CAM_BUTTON_POS_STEP));
        }

        // Rotation (Alternative to keyboard)
        if ui.button("⟲").on_hover_text("Hotkey: Q").clicked() {
            let fwd = cam.orientation.rotate_vec(FWD_VEC);
            rotation = Some(Quaternion::from_axis_angle(fwd, CAM_BUTTON_ROT_STEP));
        }
        if ui.button("⟳").on_hover_text("Hotkey: R").clicked() {
            let fwd = cam.orientation.rotate_vec(FWD_VEC);
            rotation = Some(Quaternion::from_axis_angle(fwd, -CAM_BUTTON_ROT_STEP));
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
            ui.label(format!(
                "El: {:?}, AA: {:?}, Role: {:?}",
                atom.element, atom.amino_acid, atom.role
            ));
        }
        Selection::Residue(sel) => {
            let res = &mol.residues[sel];
            let name = if let Some(aa) = res.aa {
                aa.to_string()
            } else {
                "-".to_owned() // todo temp
            };

            // todo: Sequesnce number etc.
            ui.label(format!("Res: {:?}", name,));
        }
        Selection::None => (),
    }
}

/// This function draws the (immediate-mode) GUI.
/// [UI items](https://docs.rs/egui/latest/egui/struct.Ui.html)
pub fn ui_handler(state: &mut State, ctx: &Context, scene: &mut Scene) -> EngineUpdates {
    let mut engine_updates = EngineUpdates::default();
    let mut redraw = false;
    let mut reset_cam = false;

    // let mut reset_window_title = false; // This setup avoids borrow errors.

    TopBottomPanel::top("0").show(ctx, |ui| {
        ui.spacing_mut().slider_width = 120.;

        handle_input(state, ui, &mut redraw, &mut reset_cam, &mut engine_updates);

        ui.horizontal(|ui| {
            if let Some(mol) = &state.molecule {
                ui.heading(RichText::new(mol.ident.clone()).color(Color32::GOLD));
            }

            ui.add_space(COL_SPACING);

            if ui.button("Open").clicked() {
                state.ui.load_dialog.pick_file();
            }

            ui.add_space(COL_SPACING);
            ui.label("RCSB ident:");
            ui.add(TextEdit::singleline(&mut state.ui.rcsb_input).desired_width(60.));

            if ui.button("Load RCSB").clicked() {
                match load_rcsb(&state.ui.rcsb_input) {
                    Ok(pdb) => {
                        state.pdb = Some(pdb);
                        state.molecule = Some(Molecule::from_pdb(state.pdb.as_ref().unwrap()));

                        redraw = true;
                        reset_cam = true;
                    }
                    Err(_e) => {
                        eprintln!("Error loading PDB file");
                    }
                }
            }

            ui.add_space(COL_SPACING);

            state.ui.load_dialog.update(ctx);

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
                selected_data(mol, state.selection, ui);
            }
        });

        ui.add_space(ROW_SPACING);

        cam_controls(&mut scene.camera, &mut state.ui, &mut engine_updates, ui);

        ui.add_space(ROW_SPACING / 2.);

        if let Some(path) = &state.ui.load_dialog.take_picked() {
            load_file(
                path,
                state,
                &mut redraw,
                &mut reset_cam,
                &mut engine_updates,
            );
        }

        if redraw {
            if let Some(molecule) = &state.molecule {
                draw_molecule(scene, &mut state.ui, molecule, state.selection, reset_cam);

                set_window_title(&molecule.ident, ui);
                engine_updates.entities = true;
            }
        }
    });

    if state.ui.ui_height < f32::EPSILON {
        println!("Setting height: {:?}", ctx.used_size().y);
        state.ui.ui_height = ctx.used_size().y;
    }

    state.ui.load_dialog.update(ctx);
    engine_updates
}
