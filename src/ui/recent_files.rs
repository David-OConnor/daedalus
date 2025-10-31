use std::path::Path;

use chrono::Utc;
use egui::{Color32, Popup, PopupAnchor, Pos2, RectAlign, RichText, Ui};
use graphics::{EngineUpdates, Scene};

use crate::{
    State,
    molecule::MolType,
    prefs::OpenType,
    ui::{COL_SPACING, ROW_SPACING},
    util::handle_err,
};

const NUM_TO_SHOW: usize = 30;

// todo: Return a color too
fn recentness_descrip(age_min: i64) -> (String, Color32) {
    match age_min {
        0..=30 => ("Active".to_string(), Color32::GREEN),
        31..=480 => ("Today".to_string(), Color32::LIGHT_GREEN),
        481..=10_080 => ("Recent".to_string(), Color32::YELLOW),
        _ => ("Older".to_string(), Color32::GRAY),
    }
}

pub(super) fn recent_files(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    engine_updates: &mut EngineUpdates,
) {
    let popup_id = ui.make_persistent_id("settings_popup");

    Popup::new(
        popup_id,
        ui.ctx().clone(), // todo clone???
        PopupAnchor::Position(Pos2::new(60., 60.)),
        ui.layer_id(), // draw on top of the current layer
    )
    // .align(RectAlign::TOP)
    .align(RectAlign::BOTTOM_START)
    .open(true)
    .gap(4.0)
    .show(|ui| {
        let mut open = None;
        let now = Utc::now();

        // todo: Take only recent ones
        for file in state.to_save.open_history.iter().rev().take(NUM_TO_SHOW) {
            ui.horizontal(|ui| {
                let filename = Path::new(&file.path)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap();

                // todo: Make the whole row clickable?
                let (r, g, b) = match file.type_ {
                    OpenType::Peptide => MolType::Peptide.color(),
                    OpenType::Ligand => MolType::Ligand.color(),
                    OpenType::NucleicAcid => MolType::NucleicAcid.color(),
                    OpenType::Lipid => MolType::Lipid.color(),
                    _ => (255, 255, 255),
                };
                let color = Color32::from_rgb(r, g, b);
                if ui.button(RichText::new(filename).color(color)).clicked() {
                    open = Some(file.path.clone());
                }

                ui.add_space(COL_SPACING);
                ui.label(RichText::new(file.type_.to_string()));

                let elapsed_minutes = (now - file.timestamp).num_minutes();

                let (descrip, color) = recentness_descrip(elapsed_minutes);
                ui.label(RichText::new(descrip).color(color));
            });
        }

        if let Some(path) = open {
            if state.open_file(&path, Some(scene), engine_updates).is_err() {
                handle_err(&mut state.ui, format!("Problem opening file {:?}", path));
            }
            state.ui.popup.recent_files = false;
        }

        ui.add_space(ROW_SPACING);
        if ui
            .button(RichText::new("Close").color(Color32::LIGHT_RED))
            .clicked()
        {
            state.ui.popup.recent_files = false;
        }
    });
}
