use std::{path::PathBuf, str::FromStr};

use egui::{Color32, ComboBox, Context, RichText, Slider, TopBottomPanel, Ui};
use egui_file_dialog::FileDialog;
use graphics::{EngineUpdates, Entity, Scene};
use crate::pdb::load_pdb;
use crate::{Molecule, State};
use crate::render::draw_molecule;

pub const ROW_SPACING: f32 = 10.;
pub const COL_SPACING: f32 = 30.;

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
pub fn handle_input(state: &mut State, ui: &mut Ui, scene: &mut Scene, engine_updates: &mut EngineUpdates) {
    let mut reset_window_title = false; // This setup avoids borrow errors.

    ui.ctx().input(|ip| {
        // Check for file drop
        if let Some(dropped_files) = ip.raw.dropped_files.first() {
            println!("Drop 0");
            return;
            if let Some(path) = &dropped_files.path {
                // todo: This loading code is DRY. make a fn
                println!("Drop a");
                state.pdb = Some(load_pdb(&path).unwrap());
                println!("Drop b");
                state.molecule = Some(Molecule::from_pdb(&state.pdb.as_ref().unwrap()));

                println!("Drop C");
                draw_molecule(&mut scene.entities, &state.molecule.as_ref().unwrap());
                engine_updates.entities = true;
                println!("Drop D");
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
        //     state.ui.load_dialog.pick_file();
        }

        // state.ui.load_dialog.update(ctx);

    });

    if let Some(path) = state.ui.load_dialog.take_picked() {
        let pdb = load_pdb(&path).unwrap();
        let molecule = Molecule::from_pdb(&pdb);

        state.pdb = Some(pdb);
        state.molecule = Some(molecule);

        draw_molecule(&mut scene.entities, &state.molecule.as_ref().unwrap());

        engine_updates.entities = true;
    }

    state.ui.load_dialog.update(ctx);
    engine_updates
}
