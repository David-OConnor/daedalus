use std::path::Path;

use egui::{Context, TopBottomPanel, Ui};
use graphics::{EngineUpdates, Scene};
use lin_alg::f32::{Mat4, Vec3};

use crate::{molecule::Molecule, pdb::load_pdb, render::draw_molecule, State};

pub const ROW_SPACING: f32 = 10.;
pub const COL_SPACING: f32 = 30.;

fn load_file(
    path: &Path,
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
) {
    let pdb = load_pdb(&path);

    if let Ok(p) = pdb {
        state.pdb = Some(p);
        state.molecule = Some(Molecule::from_pdb(state.pdb.as_ref().unwrap()));

        draw_molecule(
            &mut scene.entities,
            &state.molecule.as_ref().unwrap(),
            state.ui.mol_view,
        );

        engine_updates.entities = true;
    } else {
        eprintln!("Error loading PDB file");
    }
}

fn int_field(val: &mut usize, label: &str, redraw_bodies: &mut bool, ui: &mut Ui) {
    ui.label(label);
    let mut val_str = val.to_string();
    if ui
        .add_sized(
            [60., Ui::available_height(ui)],
            egui::TextEdit::singleline(&mut val_str),
        )
        .changed()
    {
        if let Ok(v) = val_str.parse::<usize>() {
            *val = v;
            *redraw_bodies = true;
        }
    }
}

/// Handles keyboard and mouse input not associated with a widget.
pub fn handle_input(
    state: &mut State,
    ui: &mut Ui,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
) {
    let mut reset_window_title = false; // This setup avoids borrow errors.

    ui.ctx().input(|ip| {
        // Check for file drop
        if let Some(dropped_files) = ip.raw.dropped_files.first() {
            if let Some(path) = &dropped_files.path {
                load_file(&path, state, scene, engine_updates)
            }
        }
    });
}

/// This function draws the (immediate-mode) GUI.
/// [UI items](https://docs.rs/egui/latest/egui/struct.Ui.html)
pub fn ui_handler(state: &mut State, ctx: &Context, scene: &mut Scene) -> EngineUpdates {
    let mut engine_updates = EngineUpdates::default();
    TopBottomPanel::top("0").show(ctx, |ui| {
        handle_input(state, ui, scene, &mut engine_updates);

        if ui.button("Open").clicked() {
            state.ui.load_dialog.pick_file();
        }

        state.ui.load_dialog.update(ctx);
    });

    if let Some(path) = &state.ui.load_dialog.take_picked() {
        load_file(path, state, scene, &mut engine_updates);
    }

    state.ui.load_dialog.update(ctx);
    engine_updates
}

// todo: Move this to the Graphics lib, the UI page, the Render page, etc
/// Find a vector that passes through all 3d points in render space, at a given 2d screen location.
/// z_limits determines the ends of the result.
pub fn screen_to_render(screen_pos: (f32, f32), z_limits: (f32, f32), proj: Mat4, ui: &Ui) -> Vec3 {
    let proj_in = proj.inverse();

    Vec3::new_zero()
}
