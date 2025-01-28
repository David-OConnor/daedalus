use std::{path::PathBuf, str::FromStr};

use egui::{Color32, ComboBox, Context, RichText, Slider, TopBottomPanel, Ui};
use graphics::{EngineUpdates, Entity, Scene};
use lin_alg::f32::{Quaternion, Vec3};

use crate::State;

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

    TopBottomPanel::top("0").show(ctx, |ui| {});

    engine_updates
}
