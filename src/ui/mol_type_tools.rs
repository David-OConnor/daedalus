//! Optional toolbars for nucleic acids, lipids etc.

use egui::{Color32, ComboBox, RichText, TextEdit, Ui};
use graphics::{EngineUpdates, EntityUpdate, FWD_VEC, Scene};
use na_seq::seq_from_str;

use crate::{
    State,
    docking::dock,
    drawing::EntityClass,
    drawing_wrappers::{draw_all_lipids, draw_all_nucleic_acids},
    lipid::{LipidShape, make_bacterial_lipids},
    molecule::MolGenericRef,
    nucleic_acid::{MoleculeNucleicAcid, NucleicAcidType, Strands},
    ui,
    ui::{COL_SPACING, COLOR_ACTION, misc::section_box},
    util::{clear_mol_entity_indices, handle_err},
};

pub(in crate::ui) fn mol_type_toolbars(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    ui.horizontal(|ui| {
        if state.ui.ui_vis.lipids {
            lipid_section(state, scene, engine_updates, ui);
        }
        if state.ui.ui_vis.nucleic_acids {
            na_section(state, scene, engine_updates, ui);
        }
        if state.ui.ui_vis.amino_acids {
            aa_section(state, scene, engine_updates, ui);
        }

        if let Some(mol) = &state.active_mol()
            && state.peptide.is_some()
        {
            if let MolGenericRef::Ligand(_) = mol {
                ui.add_space(COL_SPACING);
                ui.label("Docking:");

                if ui
                    .button(RichText::new("Dock").color(Color32::GOLD))
                    .clicked()
                {
                    // The other views make it tough to see the ligand rel the protein.
                    // if !matches!(state.ui.mol_view, MoleculeView::SpaceFill | MoleculeView::Surface) {
                    //     // todo: Dim peptide?
                    //     state.ui.mol_view = MoleculeView::Surface;
                    // }

                    if let Err(e) = dock(
                        state,
                        state.volatile.active_mol.unwrap().1,
                        scene,
                        engine_updates,
                    ) {
                        handle_err(&mut state.ui, format!("Problem setting up docking: {e:?}"));
                    }
                }
            }
        }
    });
}

/// Add and manage lipids
pub(in crate::ui) fn lipid_section(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    if state.ui.lipid_to_add >= state.templates.lipid.len() {
        eprintln!("Error: Not enough lipid templates");
        return;
    }

    section_box().show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label("Add lipids:");

            let add_standard_text = state.templates.lipid[state.ui.lipid_to_add]
                .common
                .ident
                .clone();

            ComboBox::from_id_salt(102)
                .width(90.)
                .selected_text(state.ui.lipid_shape.to_string())
                .show_ui(ui, |ui| {
                    for shape in [LipidShape::Free, LipidShape::Membrane, LipidShape::Lnp] {
                        ui.selectable_value(&mut state.ui.lipid_shape, shape, shape.to_string());
                    }
                })
                .response
                .on_hover_text("Add lipids in this pattern");

            if state.ui.lipid_shape == LipidShape::Free {
                ComboBox::from_id_salt(101)
                    .width(30.)
                    .selected_text(add_standard_text)
                    .show_ui(ui, |ui| {
                        for (i, mol) in state.templates.lipid.iter().enumerate() {
                            ui.selectable_value(&mut state.ui.lipid_to_add, i, &mol.common.ident);
                        }
                    })
                    .response
                    .on_hover_text("Add this lipid to the scene.");
            }

            ui::num_field(&mut state.ui.lipid_mol_count, "# mols", 36, ui);

            // todo: Multiple and sets once this is validated
            if ui.button("+").clicked() {
                // Place in front of the camera.
                let center = scene.camera.position
                    + scene.camera.orientation.rotate_vec(FWD_VEC)
                        * crate::cam_misc::MOVE_TO_CAM_DIST;

                state.lipids.extend(make_bacterial_lipids(
                    state.ui.lipid_mol_count as usize,
                    center.into(),
                    state.ui.lipid_shape,
                    &state.templates.lipid,
                ));
                //
                // let mut mol = state.templates.lipid[state.ui.lipid_to_add].clone();
                // for p in &mut mol.common.atom_posits {
                //     *p = *p + Vec3::new_zero();
                // }
                //
                // state.lipids.push(mol);

                draw_all_lipids(state, scene);
                engine_updates.entities = EntityUpdate::All;
            }

            if !state.lipids.is_empty() {
                if ui
                    .button(RichText::new("Close all lipids").color(Color32::LIGHT_RED))
                    .clicked()
                {
                    state.lipids = Vec::new();
                    scene
                        .entities
                        .retain(|e| e.class != EntityClass::Lipid as u32);
                    clear_mol_entity_indices(state, None);

                    engine_updates.entities = EntityUpdate::All;
                }
            }
        });
    });
}

/// Add and manage nucleic acids
pub(in crate::ui) fn na_section(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    section_box().show(ui, |ui| {
        let help_text = "Enter the nucleotide sequence of the DNA or RNA molecule to create";

        ui.label("Seq").on_hover_text(help_text);

        ui.add(TextEdit::singleline(&mut state.ui.na_seq_to_create).desired_width(60.))
            .on_hover_text(help_text);

        if ui
            .button(RichText::new("Create").color(COLOR_ACTION))
            .clicked()
        {
            // todo: Handle RNA U.
            let seq = seq_from_str(&state.ui.na_seq_to_create);

            let mol = match MoleculeNucleicAcid::from_seq(
                &seq,
                NucleicAcidType::Dna,
                Strands::Single,
                &state.templates.dna,
                &state.templates.rna,
            ) {
                Ok(v) => v,
                Err(e) => {
                    handle_err(
                        &mut state.ui,
                        format!("Problem making a Nucleic acid: {e:?}"),
                    );
                    return;
                }
            };

            state.nucleic_acids.push(mol);

            draw_all_nucleic_acids(state, scene);
            engine_updates.entities = EntityUpdate::Classes(vec![EntityClass::NucleicAcid as u32]);
        }
    });
}

/// Add and manage amino acids
pub(in crate::ui) fn aa_section(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    section_box().show(ui, |ui| {
        if state.ui.lipid_to_add >= state.templates.lipid.len() {
            eprintln!("Error: Not enough lipid templates");
            return;
        }
    });
}
