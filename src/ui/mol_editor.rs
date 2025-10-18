use bio_files::BondType;
use egui::{Color32, ComboBox, RichText, Slider, Ui};
use graphics::{EngineUpdates, EntityUpdate, Scene};
use lin_alg::f64::Vec3;
use na_seq::Element::{Carbon, Nitrogen, Oxygen};

use crate::{
    Selection, State, ViewSelLevel,
    drawing::MoleculeView,
    mol_editor,
    mol_editor::{add_atoms::add_atom, exit_edit_mode, templates},
    ui::{
        COL_SPACING, COLOR_ACTIVE, COLOR_INACTIVE, cam::cam_reset_controls, md::energy_disp, misc,
        misc::section_box, mol_data::selected_data, view_sel_selector,
    },
    util::handle_err,
};
// todo: Check DBs (with a button maybe?) to see if the molecule exists in a DB already, or if
// todo a similar one does.

// These are in ps.
const DT_MIN: f32 = 0.00005;
const DT_MAX: f32 = 0.0005; // No more than 0.002 for stability. currently 0.5fs.

// Higher may be relax more, but takes longer. We aim for a small time so it can be done without
// a noticeable lag.
const MAX_RELAX_ITERS: usize = 80;

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

        // C+P from main editor, with fewer options.
        // todo: If on one of the unavail-here view modes when entering editor,
        // todo: chagne it.
        section_box().show(ui, |ui| {
            ui.label("View:");
            let prev_view = state.ui.mol_view;
            ComboBox::from_id_salt(0)
                .width(80.)
                .selected_text(state.ui.mol_view.to_string())
                .show_ui(ui, |ui| {
                    for view in &[
                        MoleculeView::Sticks,
                        MoleculeView::BallAndStick,
                        MoleculeView::SpaceFill,
                    ] {
                        ui.selectable_value(&mut state.ui.mol_view, *view, view.to_string());
                    }
                });

            if state.ui.mol_view != prev_view {
                redraw = true;
            }

            view_sel_selector(state, &mut redraw, ui, false);
        });

        section_box().show(ui, |ui| {
            ui.label("Vis:");

            misc::toggle_btn(
                &mut state.ui.visibility.hide_hydrogen,
                "H",
                ui,
                &mut redraw,
            );
        });

        section_box().show(ui, |ui| {
            misc::toggle_btn_not_inv(
                &mut state.mol_editor.md_running,
                "MD running",
                ui,
                &mut redraw,
            );
        });

        ui.add_space(COL_SPACING);

        section_box().show(ui, |ui| {
            if ui
                .button(RichText::new("↔ Move atom"))
                .on_hover_text("(Hotkey: M) Move the selected atom")
                .clicked()
            {
                // if state.mol_editor.move_atom(i).is_err() {
                //     eprintln!("Error moving atom");
                // };
                // redraw = true;
            }
        });

        if let Selection::AtomLig((_, i)) = state.ui.selection {
            if ui
                .button(RichText::new("Del atom").color(Color32::LIGHT_RED))
                .on_hover_text("(Hotkey: Delete) Delete the selected atom")
                .clicked()
            {
                if state.mol_editor.delete_atom(i).is_err() {
                    eprintln!("Error deleting atom");
                };
                redraw = true;
            }
        }

        ui.add_space(COL_SPACING / 2.);
        // todo: implement
        if ui.button("Metadata")
            .on_hover_text("View and edit metadata for this molecule. This will be stored in the file when saved.")
            .clicked() {}

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
        {
            state.volatile.dialogs.load.pick_file();
        }

        if ui
            .button(RichText::new("Re-assign SNs"))
            .on_hover_text("Reset atom serial numbers to be sequential without gaps.")
            .clicked()
        {
            // todo: Be more clever about this.
            let mut updated_sns = Vec::with_capacity(state.mol_editor.mol.common.atoms.len());

            for (i, atom) in state.mol_editor.mol.common.atoms.iter_mut().enumerate() {
                // let sn_prev = atom.serial_number;
                let sn_new = i as u32 + 1;
                atom.serial_number = sn_new;
                updated_sns.push(sn_new);
            }

            for bond in &mut state.mol_editor.mol.common.bonds {
                bond.atom_0_sn = updated_sns[bond.atom_0];
                bond.atom_1_sn = updated_sns[bond.atom_1];
            }

        }

        if let Some(md) = &mut state.mol_editor.md_state {
            if ui.button("Relax")
                .on_hover_text("Relax geometry; adjust atom positions to mimimize energy.")
                .clicked() {
                md.minimize_energy(&state.dev, MAX_RELAX_ITERS); // todo: Iters A/R.

                state.mol_editor.load_atom_posits_from_md(&mut scene.entities, &state.ui, engine_updates);
            }
        }

        ui.add_space(COL_SPACING);
        if ui
            .button(RichText::new("Exit editor").color(Color32::LIGHT_RED))
            .clicked()
        {
            exit_edit_mode(state, scene, engine_updates);
        }
    });

    ui.horizontal(|ui| {
        edit_tools(state, scene, ui, engine_updates);

        if state.mol_editor.md_state.is_some() {
            section_box().show(ui, |ui| {
                ui.label("MD speed:");

                ui.spacing_mut().slider_width = 200.;
                ui.add(Slider::new(
                    &mut state.mol_editor.dt_md,
                    DT_MIN..=DT_MAX,
                ))
                    .on_hover_text("Set the simulation ratio compared to normal time.");

                ui.add_space(COL_SPACING);

                if let Some(snap) = &state.mol_editor.snap {
                    energy_disp(snap, ui);
                }
            });

            let ratio_help = "The ratio of simulation time to real time. A higher value means \
            the sim is running faster relative to reality. If 1×10-¹⁵, it means for every second viewing in real time,\
            the simulation runs 10¹⁵ of computed time.";

            // todo: Cache; don't compute each frame.
            // See the dt field doc comments for how we get this computation. We insert an additional
            // factor of 10e5 to make the value more readable.
            // 10e5: 10e3 for ms run interval. 10e3 to get between 10e-12 (input dt) and 10e-15 (displayed value)
            // todo: Still an unaccounted for factor of 10...
            let ratio = state.mol_editor.dt_md * 10e5 / state.mol_editor.time_between_md_runs;
            ui.label(format!("Ratio: {ratio:.1}×10-¹⁵")).on_hover_text(ratio_help);
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

fn edit_tools(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    engine_updates: &mut EngineUpdates,
) {
    section_box().show(ui, |ui| {
        if ui.button("C").on_hover_text("Add a Carbon atom").clicked() {
            let Selection::AtomLig((_, i)) = state.ui.selection else {
                eprintln!("Attempting to add an atom with no parent to add it to");
                return;
            };

            add_atom(
                &mut state.mol_editor,
                &mut scene.entities,
                i,
                Carbon,
                BondType::Single,
                Some("ca".to_owned()), // todo
                Some(1.4),             // todo
                0.13,                  // todo
                &mut state.ui,
                engine_updates,
            );
        }

        if ui.button("O").on_hover_text("Add an Oxygen atom").clicked() {
            let Selection::AtomLig((_, i)) = state.ui.selection else {
                eprintln!("Attempting to add an atom with no parent to add it to");
                return;
            };

            add_atom(
                &mut state.mol_editor,
                &mut scene.entities,
                i,
                Oxygen,
                BondType::Single,
                Some("oh".to_owned()), // todo
                Some(1.1377),          // todo
                -0.48,                 // todo
                &mut state.ui,
                engine_updates,
            );
        }

        if ui
            .button("O=")
            .on_hover_text("Add an Oxygen atom double-bonded")
            .clicked()
        {
            let Selection::AtomLig((_, i)) = state.ui.selection else {
                eprintln!("Attempting to add an atom with no parent to add it to");
                return;
            };

            add_atom(
                &mut state.mol_editor,
                &mut scene.entities,
                i,
                Oxygen,
                BondType::Double,
                Some("o".to_owned()),
                Some(1.1377), // todo
                -0.53,        // todo
                &mut state.ui,
                engine_updates,
            );
        }

        if ui
            .button("N")
            .on_hover_text("Add an Nitrogen atom")
            .clicked()
        {
            let Selection::AtomLig((_, i)) = state.ui.selection else {
                eprintln!("Attempting to add an atom with no parent to add it to");
                return;
            };

            add_atom(
                &mut state.mol_editor,
                &mut scene.entities,
                i,
                Nitrogen,
                BondType::Single,
                Some("n".to_owned()), // todo
                Some(1.4),            // todo
                -0.71,                // todo
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
}
