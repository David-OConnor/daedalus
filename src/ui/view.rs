use egui::{ComboBox, RichText, Slider, Ui};
use graphics::{EngineUpdates, Scene};

use crate::{
    State,
    drawing::{
        EntityType, MoleculeView, draw_all_ligs, draw_all_nucleic_acids, draw_density_point_cloud,
        draw_density_surface, draw_water,
    },
    nucleic_acid::{MoleculeNucleicAcid, NucleicAcidType, Strands},
    ui::{COL_SPACING, DENS_ISO_MAX, DENS_ISO_MIN, misc, misc::section_box},
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
            ui.label("View:");
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
                });

            if state.ui.mol_view != prev_view {
                *redraw_peptide = true;
                *redraw_lig = true;
                *redraw_na = true;
                *redraw_lipid = true;
            }

            ui.add_space(COL_SPACING);

            ui.label("Vis:");

            misc::vis_check(
                &mut state.ui.visibility.hide_non_hetero,
                "Peptide",
                ui,
                redraw_peptide,
            );
            misc::vis_check(
                &mut state.ui.visibility.hide_hetero,
                "Hetero",
                ui,
                redraw_peptide,
            );

            ui.add_space(COL_SPACING / 2.);

            if !state.ui.visibility.hide_non_hetero {
                // Subset of peptide.
                misc::vis_check(
                    &mut state.ui.visibility.hide_sidechains,
                    "Sidechains",
                    ui,
                    redraw_peptide,
                );
            }

            let mut hide_hydrogen_prev = state.ui.visibility.hide_hydrogen;
            misc::vis_check(
                &mut state.ui.visibility.hide_hydrogen,
                "H",
                ui,
                redraw_peptide,
            );

            if state.ui.visibility.hide_hydrogen != hide_hydrogen_prev {
                *redraw_peptide = true;
                *redraw_lig = true;
                *redraw_na = true;
                *redraw_lipid = true;
            }

            // We allow toggling water now regardless of hide hetero, as it's part of our MD sim.
            // if !state.ui.visibility.hide_hetero {
            // Subset of hetero.
            let water_prev = state.ui.visibility.hide_water;
            misc::vis_check(
                &mut state.ui.visibility.hide_water,
                "Water",
                ui,
                redraw_peptide,
            );

            if !state.nucleic_acids.is_empty() {
                misc::vis_check(
                    &mut state.ui.visibility.hide_nucleic_acids,
                    "Nucleic acids",
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
                    );
                }
            }

            if state.active_mol().is_some() {
                let color = misc::active_color(!state.ui.visibility.hide_ligand);
                if ui.button(RichText::new("Lig").color(color)).clicked() {
                    state.ui.visibility.hide_ligand = !state.ui.visibility.hide_ligand;

                    if state.ui.visibility.hide_ligand {
                        scene.entities.retain(|ent| {
                            ent.class != EntityType::Ligand as u32
                                && ent.class != EntityType::DockingSite as u32
                        });
                    } else {
                        draw_all_ligs(state, scene);
                    }

                    engine_updates.entities = true;
                    engine_updates.lighting = true; // docking light.
                }
            }

            misc::vis_check(
                &mut state.ui.visibility.hide_h_bonds,
                "H bonds",
                ui,
                redraw_peptide,
            );
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

                ui.add_space(COL_SPACING / 2.);
                let seq_text = if state.ui.show_aa_seq {
                    "Hide seq"
                } else {
                    "Show seq"
                };

                if ui.button(RichText::new(seq_text)).clicked() {
                    state.ui.show_aa_seq = !state.ui.show_aa_seq;
                }
            }

            ui.add_space(COL_SPACING);
            // todo temp
            if ui.button("Load DNA").clicked() {
                if let Some(mol) = &state.peptide {
                    state.nucleic_acids = vec![MoleculeNucleicAcid::from_peptide(
                        &mol,
                        NucleicAcidType::Dna,
                        Strands::Single,
                    )];
                }
                draw_all_nucleic_acids(state, scene);
                engine_updates.entities = true;
            }

            if ui.button("Load RNA").clicked() {
                if let Some(mol) = &state.peptide {
                    state.nucleic_acids = vec![MoleculeNucleicAcid::from_peptide(
                        &mol,
                        NucleicAcidType::Rna,
                        Strands::Single,
                    )];
                }
                draw_all_nucleic_acids(state, scene);
                engine_updates.entities = true;
            }

            if let Some(mol) = &state.peptide {
                if let Some(dens) = &mol.elec_density {
                    let mut redraw_dens = false;
                    misc::vis_check(
                        &mut state.ui.visibility.hide_density_point_cloud,
                        "Density",
                        ui,
                        &mut redraw_dens,
                    );

                    if redraw_dens {
                        if state.ui.visibility.hide_density_point_cloud {
                            scene
                                .entities
                                .retain(|ent| ent.class != EntityType::DensityPoint as u32);
                        } else {
                            draw_density_point_cloud(&mut scene.entities, dens);
                        }
                        engine_updates.entities = true;
                    }

                    let mut redraw_dens_surface = false;
                    misc::vis_check(
                        &mut state.ui.visibility.hide_density_surface,
                        "Density sfc",
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
                                .retain(|ent| ent.class != EntityType::DensitySurface as u32);
                        } else {
                            draw_density_surface(&mut scene.entities);
                        }
                        engine_updates.entities = true;
                    }
                }
            }
        });
    });
}
