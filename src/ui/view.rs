use std::sync::atomic::Ordering;

use egui::{ComboBox, RichText, Slider, Ui};
use graphics::{EngineUpdates, EntityUpdate, Scene};

use crate::{
    State,
    drawing::{
        EntityClass, MoleculeView, draw_density_point_cloud, draw_density_surface, draw_water,
    },
    drawing_wrappers::{draw_all_ligs, draw_all_lipids, draw_all_nucleic_acids},
    molecule::MolType,
    ui::{
        COL_SPACING, DENS_ISO_MAX, DENS_ISO_MIN, UI_HEIGHT_CHANGED, misc,
        misc::{section_box, toggle_btn, toggle_btn_inv},
    },
    util::clear_mol_entity_indices,
};

pub fn view_settings(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    redraw_peptide: &mut bool,
    redraw_lig: &mut bool,
    redraw_na: &mut bool,
    redraw_lipid: &mut bool,
    ui: &mut Ui,
) {
    section_box().show(ui, |ui| {
        ui.horizontal(|ui| {
            let help_text = "(Hotkeys: square brackets [ ]). Change the way we display molecules";

            ui.label("View:").on_hover_text(help_text);
            let prev_view = state.ui.mol_view;
            ComboBox::from_id_salt(0)
                .width(80.)
                .selected_text(state.ui.mol_view.to_string())
                .show_ui(ui, |ui| {
                    for view in &[
                        MoleculeView::Backbone,
                        MoleculeView::Sticks,
                        MoleculeView::BallAndStick,
                        // MoleculeView::Cartoon,
                        MoleculeView::SpaceFill,
                        MoleculeView::Surface,
                        MoleculeView::Dots,
                    ] {
                        ui.selectable_value(&mut state.ui.mol_view, *view, view.to_string());
                    }
                })
                .response
                .on_hover_text(help_text);

            if state.ui.mol_view != prev_view {
                *redraw_peptide = true;
                *redraw_lig = true;
                *redraw_na = true;
                *redraw_lipid = true;
            }

            ui.add_space(COL_SPACING);

            ui.label("Vis:");

            if state.peptide.is_some() {
                toggle_btn_inv(
                    &mut state.ui.visibility.hide_protein,
                    "Peptide",
                    "Show or hide the protein/peptide",
                    ui,
                    redraw_peptide,
                );
                toggle_btn_inv(
                    &mut state.ui.visibility.hide_hetero,
                    "Hetero",
                    "Show or hide non-amino-acid atoms in the protein/peptide",
                    ui,
                    redraw_peptide,
                );

                ui.add_space(COL_SPACING / 2.);

                if !state.ui.visibility.hide_protein {
                    // Subset of peptide.
                    toggle_btn_inv(
                        &mut state.ui.visibility.hide_sidechains,
                        "Sidechains",
                        "Show or sidechains in the protein/peptide",
                        ui,
                        redraw_peptide,
                    );
                }
            }

            let hide_hydrogen_prev = state.ui.visibility.hide_hydrogen;
            toggle_btn_inv(
                &mut state.ui.visibility.hide_hydrogen,
                "H",
                "Show or hide non-amino-acid atoms in the protein/peptide",
                ui,
                redraw_peptide,
            );

            // Hanndle the other redraws.
            if state.ui.visibility.hide_hydrogen != hide_hydrogen_prev {
                // *redraw_peptide = true;
                *redraw_lig = true;
                *redraw_na = true;
                *redraw_lipid = true;
            }

            // We allow toggling water now regardless of hide hetero, as it's part of our MD sim.
            // if !state.ui.visibility.hide_hetero {
            // Subset of hetero.
            let water_prev = state.ui.visibility.hide_water;
            toggle_btn_inv(
                &mut state.ui.visibility.hide_water,
                "Water",
                "Show or hide water molecules",
                ui,
                redraw_peptide,
            );

            if !state.nucleic_acids.is_empty() {
                toggle_btn_inv(
                    &mut state.ui.visibility.hide_nucleic_acids,
                    "Nucleic acids",
                    "Show or hide nucleic acis",
                    ui,
                    redraw_peptide,
                );
            }
            // }

            if let Some(md) = &state.mol_dynamics {
                if state.ui.visibility.hide_water != water_prev {
                    let snap = &md.snapshots[0];

                    draw_water(
                        scene,
                        &snap.water_o_posits,
                        &snap.water_h0_posits,
                        &snap.water_h1_posits,
                        state.ui.visibility.hide_water,
                        // state,
                    );
                }
            }

            if !state.ligands.is_empty() {
                let color = misc::active_color(!state.ui.visibility.hide_ligand);
                if ui.button(RichText::new("Lig").color(color)).clicked() {
                    state.ui.visibility.hide_ligand = !state.ui.visibility.hide_ligand;

                    draw_all_ligs(state, scene);
                    engine_updates.entities = EntityUpdate::All;
                    engine_updates.lighting = true; // docking light.
                }
            }
            if !state.nucleic_acids.is_empty() {
                let color = misc::active_color(!state.ui.visibility.hide_ligand);
                if ui.button(RichText::new("NA").color(color)).clicked() {
                    state.ui.visibility.hide_ligand = !state.ui.visibility.hide_ligand;

                    draw_all_nucleic_acids(state, scene);
                    engine_updates.entities = EntityUpdate::All;
                }
            }
            if !state.lipids.is_empty() {
                let color = misc::active_color(!state.ui.visibility.hide_lipids);
                if ui.button(RichText::new("Lipid").color(color)).clicked() {
                    state.ui.visibility.hide_lipids = !state.ui.visibility.hide_lipids;

                    draw_all_lipids(state, scene);
                    engine_updates.entities = EntityUpdate::All;
                }
            }

            toggle_btn_inv(
                &mut state.ui.visibility.hide_h_bonds,
                "H bonds",
                "Showh or hide Hydrogen bonds",
                ui,
                redraw_peptide,
            );

            let prev = state.ui.visibility.labels_atom_sn;
            toggle_btn(
                &mut state.ui.visibility.labels_atom_sn,
                "Lbl",
                "Show or hide atom serial numbers overlaid on their positions",
                ui,
                redraw_peptide,
            );
            if state.ui.visibility.labels_atom_sn != prev {
                //todo: This isn't working; not redrawing?
                *redraw_lig = true;
                *redraw_na = true;
                *redraw_lipid = true;
            }

            // vis_check(&mut state.ui.visibility.dim_peptide, "Dim peptide", ui, redraw);

            if state.peptide.is_some() {
                ui.add_space(COL_SPACING / 2.);
                // Not using `vis_check` for this because its semantics are inverted.
                let color = misc::active_color(state.ui.visibility.dim_peptide);
                if ui
                    .button(RichText::new("Dim peptide").color(color))
                    .on_hover_text("Dim the peptide, so that it's easier to see small molecules.")
                    .clicked()
                {
                    state.ui.visibility.dim_peptide = !state.ui.visibility.dim_peptide;
                    *redraw_peptide = true;
                }
            }

            // ui.add_space(COL_SPACING);
            // // todo temp
            // if ui.button("Load DNA").clicked() {
            //     if let Some(mol) = &state.peptide {
            //         state.nucleic_acids = vec![MoleculeNucleicAcid::from_peptide(
            //             &mol,
            //             NucleicAcidType::Dna,
            //             Strands::Single,
            //         )];
            //     }
            //     draw_all_nucleic_acids(state, scene);
            //     engine_updates.entities = true;
            // }
            //
            // if ui.button("Load RNA").clicked() {
            //     if let Some(mol) = &state.peptide {
            //         state.nucleic_acids = vec![MoleculeNucleicAcid::from_peptide(
            //             &mol,
            //             NucleicAcidType::Rna,
            //             Strands::Single,
            //         )];
            //     }
            //     draw_all_nucleic_acids(state, scene);
            //     engine_updates.entities = true;
            // }

            if let Some(mol) = &state.peptide {
                if let Some(dens) = &mol.elec_density {
                    let mut redraw_dens = false;
                    toggle_btn_inv(
                        &mut state.ui.visibility.hide_density_point_cloud,
                        "Density",
                        "Show or hide the electron density point cloud visualization",
                        ui,
                        &mut redraw_dens,
                    );

                    if redraw_dens {
                        if state.ui.visibility.hide_density_point_cloud {
                            scene
                                .entities
                                .retain(|ent| ent.class != EntityClass::DensityPoint as u32);
                        } else {
                            draw_density_point_cloud(&mut scene.entities, dens);
                        }
                        clear_mol_entity_indices(state, None);
                        engine_updates.entities = EntityUpdate::All;
                        // engine_updates.entities.push_class(EntityClass::Peptide as u32);
                    }

                    let mut redraw_dens_surface = false;
                    toggle_btn_inv(
                        &mut state.ui.visibility.hide_density_surface,
                        "Density sfc",
                        "Show or hide the electron density isosurface visualization",
                        ui,
                        &mut redraw_dens_surface,
                    );

                    if !state.ui.visibility.hide_density_surface {
                        let iso_prev = state.ui.density_iso_level;

                        ui.spacing_mut().slider_width = 300.;
                        ui.add(Slider::new(
                            &mut state.ui.density_iso_level,
                            DENS_ISO_MIN..=DENS_ISO_MAX,
                        ))
                        .on_hover_text("The density value at which to draw the ISO surface");
                        if state.ui.density_iso_level != iso_prev {
                            state.volatile.flags.make_density_iso_mesh = true;
                        }
                    }

                    if redraw_dens_surface {
                        if state.ui.visibility.hide_density_surface {
                            let _ = &mut scene
                                .entities
                                .retain(|ent| ent.class != EntityClass::DensitySurface as u32);
                        } else {
                            draw_density_surface(&mut scene.entities, state);
                        }
                        engine_updates.entities = EntityUpdate::All;
                        // engine_updates
                        //     .entities
                        //     .push_class(EntityClass::DensitySurface as u32);
                    }
                }
            }
        });
    });
}

fn vis_helper(vis: &mut bool, name: &str, tooltip: &str, ui: &mut Ui) {
    let prev = *vis;
    toggle_btn(vis, name, tooltip, ui, &mut false);

    if *vis != prev {
        UI_HEIGHT_CHANGED.store(true, Ordering::Release);
    }
}

/// For toggling on and off UI sections.
pub fn ui_section_vis(state: &mut State, ui: &mut Ui) {
    if state.peptide.is_some() {
        let tooltip = "Show or hide the amino acid sequence of the currently opened protein \
                    as single-letter identifiers. When in this mode, click the AA letter to select its residue.";

        vis_helper(&mut state.ui.ui_vis.aa_seq, "Seq", tooltip, ui);
    }

    if let Some(mol) = &state.active_mol()
        && mol.mol_type() == MolType::Ligand
    {
        vis_helper(
            &mut state.ui.ui_vis.smiles,
            "SMILES",
            "Show or hide the SMILES text representation of the molecular formula",
            ui,
        );
    }

    let tooltip = "Show or hide tools for adding lipids";
    vis_helper(&mut state.ui.ui_vis.lipids, "Lipid tools", tooltip, ui);

    let tooltip = "Show or hide the molecular dynamics section.";
    vis_helper(&mut state.ui.ui_vis.dynamics, "Dynamics", tooltip, ui);

    // todo: Do we want this button available if ORCA is not on the PATH? Maybe check at runtime,
    // todo or check when clicking this button. Not sure yet.
    let tooltip =
        "Show or hide the ORCA (Quantum mechanics package) section. Has powerful, but slow tools";
    vis_helper(&mut state.ui.ui_vis.orca, "ORCA", tooltip, ui);
}
