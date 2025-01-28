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

/// This function draws the (immediate-mode) GUI.
/// [UI items](https://docs.rs/egui/latest/egui/struct.Ui.html)
pub fn ui_handler(state: &mut State, ctx: &Context, scene: &mut Scene) -> EngineUpdates {
    let mut engine_updates = EngineUpdates::default();

    TopBottomPanel::top("0").show(ctx, |ui| {
        if ui.button("Open").clicked() {
            state.ui.load_dialog.pick_file();
        }

        state.ui.load_dialog.update(ctx);

        if let Some(path) = state.ui.load_dialog.take_picked() {
            // let pdb = load_pdb(&path).unwrap();
            // let molecule = Molecule::from_pdb(&pdb);
            //
            // state.pdb = Some(pdb);
            // state.molecule = Some(molecule);
            //
            // draw_molecule(&mut scene.entities, &state.molecule.as_ref().unwrap());
            //
            // engine_updates.entities = true;
        }
    });

    engine_updates
}
