use egui::{Color32, RichText, Ui};
use graphics::{EngineUpdates, EntityUpdate, Scene};
use lin_alg::{f32::Quaternion, f64::Vec3};

use crate::{
    Selection, State, mol_editor,
    mol_editor::{INIT_CAM_DIST, MolEditorState, exit_edit_mode, templates},
    ui::{COL_SPACING, misc::section_box},
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
        section_box().show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Reset cam").clicked() {
                    scene.camera.position = lin_alg::f32::Vec3::new(0., 0., -INIT_CAM_DIST);
                    scene.camera.orientation = Quaternion::new_identity();
                }
            });

            ui.add_space(COL_SPACING);

            section_box().show(ui, |ui| {
                if ui.button("C").on_hover_text("Add a Carbon atom").clicked() {}

                if ui.button("O").on_hover_text("Add an Oxygen atom").clicked() {}

                if ui
                    .button("N")
                    .on_hover_text("Add an Nitrogen atom")
                    .clicked()
                {}
            });

            ui.add_space(COL_SPACING);

            section_box().show(ui, |ui| {
                if ui
                    .button("−OH")
                    .on_hover_text("Add a hydroxyl functional group")
                    .clicked()
                {}

                if ui
                    .button("−COOH")
                    .on_hover_text("Add a carboxylic acid functional group")
                    .clicked()
                {
                    let anchor = Vec3::new_zero();
                    let atoms = templates::cooh_group(anchor, 0);
                }

                if ui
                    .button("−NH₂")
                    .on_hover_text("Add an admide functional group")
                    .clicked()
                {}

                if ui
                    .button("Ring")
                    .on_hover_text("Add a benzene ring")
                    .clicked()
                {
                    let anchor = Vec3::new_zero();
                    let atoms = templates::benzene_ring(anchor, 0);
                }
            });
        });

        ui.add_space(COL_SPACING);

        match state.ui.selection {
            Selection::AtomLig((_, i)) => {
                if ui
                    .button(RichText::new("Delete atom").color(Color32::LIGHT_RED))
                    .on_hover_text("Delete the selected atom")
                    .clicked()
                {
                    if state.mol_editor.delete_atom(i).is_err() {
                        eprintln!("Error deleting atom");
                    };
                    mol_editor::redraw(&mut scene.entities, &state.mol_editor.mol, &state.ui);
                    engine_updates.entities = EntityUpdate::All;
                }
            }
            _ => (),
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
