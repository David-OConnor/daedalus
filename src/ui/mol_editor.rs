use egui::{Color32, RichText, Ui};
use graphics::{EngineUpdates, Scene};
use lin_alg::f64::Vec3;

use crate::{
    State,
    mol_editor::{MolEditorState, exit_edit_mode, templates},
    ui::COL_SPACING,
};
// todo: Check DBs (with a button maybe?) to see if the molecule exists in a DB already, or if
// todo a similar one does.

pub fn editor(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    // todo: New state for the WIp molecule. New struct for it.

    ui.horizontal(|ui| {
        if ui.button("C").clicked() {}

        if ui.button("O").clicked() {}

        if ui.button("N").clicked() {}

        ui.add_space(COL_SPACING);

        if ui.button("−OH").clicked() {}

        if ui.button("−COOH").clicked() {
            let anchor = Vec3::new_zero();
            let atoms = templates::cooh_group(anchor, 0);
        }

        if ui.button("−NH₂").clicked() {}

        if ui.button("Ring").clicked() {
            let anchor = Vec3::new_zero();
            let atoms = templates::benzene_ring(anchor, 0);
        }

        ui.add_space(COL_SPACING);

        if ui
            .button(RichText::new("Exit editor").color(Color32::LIGHT_RED))
            .clicked()
        {
            exit_edit_mode(state, scene, engine_updates);
        }
    });
}
