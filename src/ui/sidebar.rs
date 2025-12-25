use egui::{Color32, Context, RichText, Ui};
use graphics::{ControlScheme, EngineUpdates, EntityUpdate, Scene};

use crate::{
    State,
    molecule::{MolType, MoleculeCommon},
    ui::{COLOR_ACTION, COLOR_ACTIVE, COLOR_ACTIVE_RADIO, COLOR_INACTIVE},
    util::{close_mol, handle_err, orbit_center},
};

/// Abstracts over all molecule types. (Currently not protein though)
/// A single row for the molecule.
fn mol_picker_one(
    active_mol: &mut Option<(MolType, usize)>,
    orbit_center: &mut Option<(MolType, usize)>,
    i_mol: usize,
    mol: &mut MoleculeCommon,
    mol_type: MolType,
    ui: &mut Ui,
    engine_updates: &mut EngineUpdates,
    redraw: &mut bool,
    recenter_orbit: &mut bool,
    close: &mut Option<(MolType, usize)>,
) {
    let help_text = "Make this molecule the active / selected one. Middle click to close it.";

    let active = match active_mol {
        Some((_mol_type, i)) => *i == i_mol,
        _ => false,
    };

    let color = if active {
        COLOR_ACTIVE_RADIO
    } else {
        COLOR_INACTIVE
    };

    ui.horizontal(|ui| {
        let sel_btn = ui
            .button(RichText::new(&mol.ident).color(color))
            .on_hover_text(help_text);
        if sel_btn.clicked() {
            if active && active_mol.is_some() {
                *active_mol = None;
            } else {
                *active_mol = Some((mol_type, i_mol));
                *orbit_center = *active_mol;

                *recenter_orbit = true;
            }

            *redraw = true; // To reflect the change in thickness, color etc.
        }

        if sel_btn.middle_clicked() {
            *close = Some((mol_type, i_mol));
        }

        // Allocate the rest of the row and place buttons at the far right.
        let remaining_w = ui.available_width();
        let row_h = ui.spacing().interact_size.y;

        let color_vis = if mol.visible {
            COLOR_ACTIVE
        } else {
            COLOR_INACTIVE
        };

        // ui.allocate_ui_with_layout(
        //     egui::vec2(remaining_w, row_h),
        //     egui::Layout::right_to_left(egui::Align::Center),
        //     |ui| {
        if ui.button(RichText::new("👁").color(color_vis)).clicked() {
            mol.visible = !mol.visible;

            *redraw = true; // todo Overkill; only need to redraw (or even just clear) one.
            // todo: Generalize.
            engine_updates.entities = EntityUpdate::All;
            // engine_updates.entities.push_class(mol_type.entity_class() as u32);
        }

        if ui
            .button(RichText::new("❌").color(Color32::LIGHT_RED))
            .on_hover_text("(Hotkey: Delete) Close this molecule.")
            .clicked()
        {
            *close = Some((mol_type, i_mol));
        }
    });
    // });
}

/// Select, close, hide etc molecules from ones opened.
fn mol_picker(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    redraw_pep: &mut bool,
    redraw_lig: &mut bool,
    redraw_lipid: &mut bool,
    redraw_na: &mut bool,
    engine_updates: &mut EngineUpdates,
) {
    let help_text = "Make this molecule the active / selected one. Middle click to close it.";
    // todo: Make this support other types.
    let mut recenter_orbit = false;
    if let Some(mol) = &mut state.peptide {
        let i_mol = 0; // todo: A/R if you add more peptides.

        let active = match state.volatile.active_mol {
            Some((MolType::Peptide, i_)) => i_ == i_mol,
            _ => false,
        };

        let color = if active {
            COLOR_ACTIVE_RADIO
        } else {
            COLOR_INACTIVE
        };

        let sel_btn = ui
            .button(RichText::new(&mol.common.ident).color(color))
            .on_hover_text(help_text);
        if sel_btn.clicked() {
            if active && state.volatile.active_mol.is_some() {
                state.volatile.active_mol = None;
            } else {
                state.volatile.active_mol = Some((MolType::Peptide, i_mol));
                state.volatile.orbit_center = state.volatile.active_mol;

                recenter_orbit = true;
            }

            *redraw_pep = true; // To reflect the change in thickness, color etc.

            let color_vis = if mol.common.visible {
                COLOR_ACTIVE
            } else {
                COLOR_INACTIVE
            };

            if ui.button(RichText::new("👁").color(color_vis)).clicked() {
                mol.common.visible = !mol.common.visible;

                *redraw_lig = true; // todo Overkill; only need to redraw (or even just clear) one.
                // todo: Generalize.
                engine_updates.entities = EntityUpdate::All;
                // engine_updates.entities.push_class(EntityClass::Peptide as u32);
            }
        }

        if sel_btn.middle_clicked() {
            close_mol(MolType::Peptide, i_mol, state, scene, engine_updates);
        }
    }

    let mut close = None; // Avoids borrow error.

    for (i_mol, mol) in state.ligands.iter_mut().enumerate() {
        mol_picker_one(
            &mut state.volatile.active_mol,
            &mut state.volatile.orbit_center,
            i_mol,
            &mut mol.common,
            MolType::Ligand,
            ui,
            engine_updates,
            redraw_lig,
            &mut recenter_orbit,
            &mut close,
        );
    }

    for (i_mol, mol) in state.lipids.iter_mut().enumerate() {
        mol_picker_one(
            &mut state.volatile.active_mol,
            &mut state.volatile.orbit_center,
            i_mol,
            &mut mol.common,
            MolType::Lipid,
            ui,
            engine_updates,
            redraw_lipid,
            &mut recenter_orbit,
            &mut close,
        );
    }

    for (i_mol, mol) in state.nucleic_acids.iter_mut().enumerate() {
        mol_picker_one(
            &mut state.volatile.active_mol,
            &mut state.volatile.orbit_center,
            i_mol,
            &mut mol.common,
            MolType::NucleicAcid,
            ui,
            engine_updates,
            redraw_na,
            &mut recenter_orbit,
            &mut close,
        );
    }

    // todo: AAs here too?

    if let Some((mol_type, i_mol)) = close {
        close_mol(mol_type, i_mol, state, scene, engine_updates);
    }

    if recenter_orbit {
        if let ControlScheme::Arc { center } = &mut scene.input_settings.control_scheme {
            *center = orbit_center(state);
        }
    }
}

fn open_tools(state: &mut State, ui: &mut Ui) {
    let color_open_tools = if state.peptide.is_none() && state.ligands.is_empty() {
        COLOR_ACTION
    } else {
        COLOR_INACTIVE
    };

    if ui
        .button(RichText::new("Open").color(color_open_tools))
        .on_hover_text("Open a molecule, electron density, or other file from disk.")
        .clicked()
    {
        state.volatile.dialogs.load.pick_file();
    }

    if ui
        .button(RichText::new("Recent").color(color_open_tools))
        .on_hover_text("Select a recently-opened file to open")
        .clicked()
    {
        state.ui.popup.recent_files = !state.ui.popup.recent_files;
    }
}

pub(in crate::ui) fn sidebar(
    state: &mut State,
    scene: &mut Scene,
    redraw_pep: &mut bool,
    redraw_lig: &mut bool,
    redraw_lipid: &mut bool,
    redraw_na: &mut bool,
    engine_updates: &mut EngineUpdates,
    ctx: &Context,
) {
    // return;

    egui::SidePanel::left("sidebar")
        // .resizable(true) // let user drag the width
        // .default_width(200.0)
        // .width_range(160.0..=420.0)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Mols");

                let color_open_tools = if state.peptide.is_none() && state.ligands.is_empty() {
                    COLOR_ACTION
                } else {
                    COLOR_INACTIVE
                };

                if ui
                    .button(RichText::new("Open").color(color_open_tools))
                    .on_hover_text("Open a molecule, electron density, or other file from disk.")
                    .clicked()
                {
                    state.volatile.dialogs.load.pick_file();
                }

                if ui
                    .button(RichText::new("Recent").color(color_open_tools))
                    .on_hover_text("Select a recently-opened file to open")
                    .clicked()
                {
                    state.ui.popup.recent_files = !state.ui.popup.recent_files;
                    open_tools(state, ui);
                }

                if let Some(mol) = &state.peptide {
                    // let color = if state.to_save.last_peptide_opened.is_none() {
                    //     COLOR_ATTENTION
                    // } else {
                    //     Color32::GRAY
                    // };
                    // todo: Put a form of this back.
                    let color = Color32::GRAY;
                    if ui.button(RichText::new("Save").color(color)).clicked() {
                        let filename = {
                            let extension = "cif";

                            let name = if mol.common.ident.is_empty() {
                                "molecule".to_string()
                            } else {
                                mol.common.ident.clone()
                            };
                            format!("{name}.{extension}")
                        };

                        state.volatile.dialogs.save.config_mut().default_file_name =
                            filename.to_string();
                        state.volatile.dialogs.save.save_file();
                    }
                }
                let mut mol_to_save = None; // avoids dbl-borrow.
                if let Some(mol) = state.active_mol() {
                    // Highlight the button if we haven't saved this to file, e.g. if opened from online.
                    // let color = if state.to_save.last_ligand_opened.is_none() {
                    //     COLOR_ATTENTION
                    // } else {
                    //     Color32::GRAY
                    // };
                    // todo: Put a form of this back.
                    let color = Color32::GRAY;

                    if ui
                        .button(RichText::new("Save mol").color(color))
                        .on_hover_text(
                            "Save the active small molecule, nucleic acid, or lipid to a file.",
                        )
                        .clicked()
                    {
                        mol_to_save = Some(mol.common().clone());
                    }
                }
                if let Some(mol) = mol_to_save {
                    if mol.save(&mut state.volatile.dialogs.save).is_err() {
                        handle_err(&mut state.ui, "Problem saving this file".to_owned());
                    }
                }
            });

            ui.separator();

            // todo: Function or macro to reduce this DRY.
            mol_picker(
                state,
                scene,
                ui,
                redraw_pep,
                redraw_lig,
                redraw_lipid,
                redraw_na,
                engine_updates,
            );
        });
}
