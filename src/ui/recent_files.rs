use egui::{Color32, Popup, PopupAnchor, Pos2, RectAlign, RichText, Ui};
use graphics::Scene;

use crate::State;

pub(super) fn recent_files(state: &mut State, scene: &mut Scene, ui: &mut Ui) {
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
        if ui
            .button(RichText::new("Close").color(Color32::LIGHT_RED))
            .clicked()
        {
            state.ui.popup.recent_files = false;
        }
    });
}
