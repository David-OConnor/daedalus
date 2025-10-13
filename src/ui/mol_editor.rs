use bio_files::{BondType, Sdf};
use egui::{Color32, RichText, Ui};
use graphics::{EngineUpdates, Entity, EntityUpdate, Scene};
use lin_alg::{f32::Quaternion, f64::Vec3};
use na_seq::{
    Element,
    Element::{Carbon, Nitrogen, Oxygen},
};

use crate::{
    Selection, State, StateUi, ViewSelLevel, mol_editor,
    mol_editor::{INIT_CAM_DIST, MolEditorState, add_atom, exit_edit_mode, templates},
    mol_lig::MoleculeSmall,
    molecule::{Atom, Bond, MolGenericRef},
    ui::{
        COL_SPACING, COLOR_ACTIVE, COLOR_INACTIVE, cam::cam_reset_controls, misc::section_box,
        mol_data::selected_data,
    },
    util::handle_err,
};
// todo: Check DBs (with a button maybe?) to see if the molecule exists in a DB already, or if
// todo a similar one does.

pub fn editor(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    let mut redraw = false;

    ui.horizontal(|ui| {
        section_box().show(ui, |ui| {
            ui.horizontal(|ui| {
                let mut cam_changed = false;

                // todo: The distances this function resets to may not be ideal for our use case
                // todo here. Adjust A/R.
                cam_reset_controls(state, scene, ui, engine_updates, &mut cam_changed);
                // if ui.button("Reset cam").clicked() {
                //     scene.camera.position = lin_alg::f32::Vec3::new(0., 0., -INIT_CAM_DIST);
                //     scene.camera.orientation = Quaternion::new_identity();
                // }

                ui.add_space(COL_SPACING / 2.);

                // todo: This is a C+P from the main editor
                let color = if state.ui.atom_color_by_charge {
                    COLOR_ACTIVE
                } else {
                    COLOR_INACTIVE
                };
                if ui
                    .button(RichText::new("Color by q").color(color))
                    .on_hover_text(
                        "Color the atom by partial charge, instead of element-specific colors",
                    )
                    .clicked()
                {
                    state.ui.atom_color_by_charge = !state.ui.atom_color_by_charge;
                    state.ui.view_sel_level = ViewSelLevel::Atom;

                    redraw = true;
                }
            });
        });

        ui.add_space(COL_SPACING / 2.);

        section_box().show(ui, |ui| {
            if ui.button("C").on_hover_text("Add a Carbon atom").clicked() {
                add_atom(
                    &mut scene.entities,
                    &mut state.mol_editor.mol,
                    Carbon,
                    &mut state.ui,
                    engine_updates,
                );
            }

            if ui.button("O").on_hover_text("Add an Oxygen atom").clicked() {
                add_atom(
                    &mut scene.entities,
                    &mut state.mol_editor.mol,
                    Oxygen,
                    &mut state.ui,
                    engine_updates,
                );
            }

            if ui
                .button("N")
                .on_hover_text("Add an Nitrogen atom")
                .clicked()
            {
                add_atom(
                    &mut scene.entities,
                    &mut state.mol_editor.mol,
                    Nitrogen,
                    &mut state.ui,
                    engine_updates,
                );
            }
        });

        ui.add_space(COL_SPACING / 2.);

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

        ui.add_space(COL_SPACING);

        section_box().show(ui, |ui| {
            if ui
                .button(RichText::new("↔ Move atom").color(Color32::LIGHT_RED))
                .on_hover_text("(Hotkey: M) Delete the selected atom")
                .clicked()
            {
                // if state.mol_editor.delete_atom(i).is_err() {
                //     eprintln!("Error deleting atom");
                // };
                // redraw = true;
            }
        });

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
                    redraw = true;
                }
            }
            _ => (),
        }

        ui.add_space(COL_SPACING / 2.);
        // todo: implement
        if ui.button("Edit metadata").clicked() {}

        ui.add_space(COL_SPACING);
        if ui
            .button(RichText::new("Save"))
            .on_hover_text("Save to a Mol2, SDF, or PDBQT file")
            .clicked()
        {
            if state
                .mol_editor
                .mol
                .common
                .save(&mut state.volatile.dialogs.save)
                .is_err()
            {
                handle_err(&mut state.ui, "Problem saving this file".to_owned());
            }
        }
        if ui
            .button(RichText::new("Load"))
            .on_hover_text("Save to a Mol2 or SDF file")
            .clicked()
        {}

        ui.add_space(COL_SPACING);
        if ui
            .button(RichText::new("Exit editor").color(Color32::LIGHT_RED))
            .clicked()
        {
            exit_edit_mode(state, scene, engine_updates);
        }
    });

    // This trick prevents a clone.
    let mol = std::mem::take(&mut state.mol_editor.mol); // move out, leave default in place
    selected_data(
        state,
        std::slice::from_ref(&&mol),
        &[],
        &[],
        &state.ui.selection,
        ui,
    );
    state.mol_editor.mol = mol;

    // Prevents the UI from jumping when going between a selection and none.
    if state.ui.selection == Selection::None {
        ui.add_space(6.);
    }

    if redraw {
        mol_editor::redraw(&mut scene.entities, &state.mol_editor.mol, &state.ui);
        engine_updates.entities = EntityUpdate::All;
    }
}
