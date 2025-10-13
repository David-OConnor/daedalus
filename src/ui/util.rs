use std::{io, path::Path};

use bio_apis::{amber_geostd, rcsb};
use egui::{Color32, Popup, PopupAnchor, Pos2, RectAlign, RichText, Ui};
use graphics::{EngineUpdates, EntityUpdate, Scene};

use crate::{
    State,
    download_mols::load_atom_coords_rcsb,
    molecule::MolGenericRef,
    render::set_flashlight,
    ui::{COL_SPACING, COLOR_HIGHLIGHT, ROW_SPACING},
    util::handle_err,
};

/// Run this each frame, after all UI elements that affect it are rendered.
pub fn update_file_dialogs(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    redraw_peptide: &mut bool,
    reset_cam: &mut bool,
    engine_updates: &mut EngineUpdates,
) {
    let ctx = ui.ctx();

    state.volatile.dialogs.load.update(ctx);
    state.volatile.dialogs.save.update(ctx);
    state.volatile.dialogs.autodock_path.update(ctx);

    if let Some(path) = &state.volatile.dialogs.load.take_picked() {
        if let Err(e) = load_file(
            path,
            state,
            redraw_peptide,
            reset_cam,
            engine_updates,
            scene,
        ) {
            handle_err(&mut state.ui, e.to_string());
        }

        set_flashlight(scene);
        engine_updates.lighting = true;
    }

    if let Some(path) = &state.volatile.dialogs.save.take_picked() {
        state.save(path).ok();
    }
}

pub fn load_popups(state: &mut State, scene: &mut Scene, ui: &mut Ui, redraw_lig: &mut bool) {
    if state.ui.popup.show_get_geostd {
        let popup_id = ui.make_persistent_id("no_ff_params_popup");

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
            let mut load_ff = false;
            let mut load_frcmod = false;
            // These vars avoid dbl borrow.
            // todo: Is this always the lig you want?
            if let Some(lig) = &state.active_mol() {
                if let MolGenericRef::Ligand(l) = lig {
                    load_ff = !l.ff_params_loaded;
                    load_frcmod = !l.frcmod_loaded;

                    // let Some(lig) = state.active_mol_mut() else {
                    //     return;
                    // };
                    let mut msg = String::from("Not ready for dynamics: ");

                    if !l.ff_params_loaded {
                        msg += "No FF params or partial charges are present on this ligand."
                    }

                    // if !lig.frcmod_loaded {
                    //     msg += "No FRCMOD parameters loaded for this ligand."
                    // }

                    ui.label(RichText::new(msg).color(Color32::LIGHT_RED));
                }

                ui.add_space(ROW_SPACING);

                // todo: What about cases where a SDF from pubchem or drugbank doesn't include teh name used by Amber?
                if ui.button("Check online").clicked() {
                    // let Some(lig) = state.ligand.as_mut() else {
                    //     return;
                    // };

                    match amber_geostd::find_mols(&lig.common().ident) {
                        Ok(data) => {
                            state.ui.popup.get_geostd_items = data;
                        }
                        Err(e) => handle_err(
                            &mut state.ui,
                            format!("Problem loading mol data online: {e:?}"),
                        ),
                    }
                }

                // This clone is annoying; db borrow.
                let items = state.ui.popup.get_geostd_items.clone();
                for mol_data in items {
                    if ui
                        .button(
                            RichText::new(format!("Load params for {}", mol_data.ident_pdbe))
                                .color(COLOR_HIGHLIGHT),
                        )
                        .clicked()
                    {
                        state.load_geostd_mol_data(
                            &mol_data.ident_pdbe,
                            load_ff,
                            load_frcmod,
                            redraw_lig,
                            &scene.camera,
                        );

                        state.ui.popup.show_get_geostd = false;
                    }
                }
            }

            ui.add_space(ROW_SPACING);

            if ui
                .button(RichText::new("Close").color(Color32::LIGHT_RED))
                .clicked()
            {
                state.ui.popup.show_get_geostd = false;
            }
        });
    }

    if state.ui.popup.show_associated_structures {
        let mut associated_structs = Vec::new();
        if let Some(lig) = state.active_mol() {
            if let MolGenericRef::Ligand(l) = lig {
                // todo: I don't like this clone, but not sure how else to do it.
                associated_structs = l.associated_structures.clone();
            }
        }

        if state.active_mol().is_some() {
            let popup_id = ui.make_persistent_id("associated_structs_popup");
            Popup::new(
                popup_id,
                ui.ctx().clone(),
                PopupAnchor::Position(Pos2::new(300., 60.)),
                ui.layer_id(), // draw on top of the current layer
            )
            .align(RectAlign::TOP)
            .open(true)
            .gap(4.0)
            .show(|ui| {
                for s in &associated_structs {
                    ui.horizontal(|ui| {
                        if ui
                            .button(RichText::new(format!("{}", s.pdb_id)).color(COLOR_HIGHLIGHT))
                            .clicked()
                        {
                            rcsb::open_overview(&s.pdb_id);
                        }
                        ui.add_space(COL_SPACING);

                        if ui
                            .button(
                                RichText::new(format!("Open this protein")).color(COLOR_HIGHLIGHT),
                            )
                            .clicked()
                        {
                            load_atom_coords_rcsb(
                                &s.pdb_id,
                                state,
                                scene,
                                &mut engine_updates,
                                &mut redraw_peptide,
                                &mut reset_cam,
                            );
                        }
                    });

                    ui.label(RichText::new(format!("{}", s.description)));

                    ui.add_space(ROW_SPACING);
                }

                ui.add_space(ROW_SPACING);
            });

            ui.add_space(ROW_SPACING);
            if ui
                .button(RichText::new("Close").color(Color32::LIGHT_RED))
                .clicked()
            {
                state.ui.popup.show_associated_structures = false;
            }
        }
    }
}

pub fn load_file(
    path: &Path,
    state: &mut State,
    redraw: &mut bool,
    reset_cam: &mut bool,
    engine_updates: &mut EngineUpdates,
    scene: &graphics::Scene,
) -> io::Result<()> {
    state.open(path, Some(scene))?;

    // Clear last map opened here, vice in `open_molecule`, to prevent it clearing the map
    // on init.

    *redraw = true;
    *reset_cam = true;
    // engine_updates.entities = true;
    // todo: Overkill.
    engine_updates.entities = EntityUpdate::All;

    Ok(())
}
