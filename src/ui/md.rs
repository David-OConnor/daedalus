use dynamics::{ComputationDevice, HydrogenConstraint, Integrator, SimBoxInit};
use egui::{Color32, ComboBox, RichText, TextEdit, Ui};
use graphics::{EngineUpdates, Scene};

use crate::{
    State,
    drawing::{draw_peptide, draw_water},
    md::{build_dynamics_docking, build_dynamics_peptide},
    ui::{COL_SPACING, cam::move_cam_to_lig, misc, misc::MdMode, num_field},
    util::handle_err,
};

pub fn md_setup(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    redraw_lig: &mut bool,
    ui: &mut Ui,
) {
    misc::section_box().show(ui, |ui| {
        ui.horizontal(|ui| {
            if ui
                .button(RichText::new("Run MD on peptide").color(Color32::GOLD))
                .clicked() {
                let mol = state.molecule.as_mut().unwrap();
                state.volatile.md_mode = MdMode::Peptide;

                match build_dynamics_peptide(
                    &state.dev,
                    mol,
                    &state.ff_param_set,
                    &state.to_save.md_config,
                    state.to_save.num_md_steps,
                    state.to_save.md_dt,
                ) {
                    Ok(md) => {
                        let snap = &md.snapshots[0];
                        draw_peptide(state, scene);

                        draw_water(
                            scene,
                            &snap.water_o_posits,
                            &snap.water_h0_posits,
                            &snap.water_h1_posits,
                            state.ui.visibility.hide_water
                        );

                        state.mol_dynamics = Some(md);
                        state.ui.current_snapshot = 0;
                    }
                    Err(e) => handle_err(&mut state.ui, e.descrip),
                }
            }

            if state.active_lig().is_some() {
                ui.add_space(COL_SPACING / 2.);

                let run_clicked = ui
                    .button(RichText::new("Run MD docking").color(Color32::GOLD))
                    .on_hover_text("Run a molecular dynamics simulation on the ligand. The peptide atoms apply\
            Coulomb and Van der Waals forces, but do not move themselves. This is intended to be run\
            with the ligand positioned near a receptor site.")
                    .clicked();

                state.volatile.md_mode = MdMode::Docking;

                let mut ready_to_run = true;

                if run_clicked {
                    if state.active_lig().is_none() {
                        return;
                    }

                    {
                        let lig = state.active_lig().unwrap();
                        if !lig.ff_params_loaded || !lig.frcmod_loaded {
                            state.ui.popup.show_get_geostd = true;
                            ready_to_run = false;
                        }
                    }

                    if ready_to_run {

                        // todo: Set a loading indicator, and trigger the build next GUI frame.
                        move_cam_to_lig(state, scene, state.molecule.as_ref().unwrap().center, engine_updates);

                        let lig_ident = &state.ligands[state.volatile.active_lig.unwrap()].common.ident;

                        let lig_specific_params = match state.lig_specific_params.get(lig_ident) {
                            Some(p) => p,
                            None => {
                                handle_err(&mut state.ui, "Missing ligand-specific docking parameters; aborting.".to_string());
                                return;
                            }
                        };

                        let mol = state.molecule.as_mut().unwrap();
                        match build_dynamics_docking(
                            &state.dev,
                            &mut state.ligands,
                            state.volatile.active_lig.unwrap(),
                            mol,
                            &state.ff_param_set,
                            lig_specific_params,
                            &state.to_save.md_config,
                            state.to_save.num_md_steps,
                            state.to_save.md_dt,
                        ) {
                            Ok(md) => {
                                let snap = &md.snapshots[0];

                                draw_peptide(state, scene);
                                draw_water(
                                    scene,
                                    &snap.water_o_posits,
                                    &snap.water_h0_posits,
                                    &snap.water_h1_posits,
                                    state.ui.visibility.hide_water
                                );

                                state.ui.current_snapshot = 0;
                                engine_updates.entities = true;
                                state.mol_dynamics = Some(md);
                            }
                            Err(e) => handle_err(&mut state.ui, e.descrip),
                        }
                    }
                }
            }

            match &state.dev {
                ComputationDevice::Cpu => {
                    ui.label(RichText::new("CPU"));
                }
                #[cfg(feature = "cuda")]
                ComputationDevice::Gpu(_) => {
                    ui.label(RichText::new("GPU").color(Color32::LIGHT_GREEN));
                }
            }

            let num_steps_prev = state.to_save.num_md_steps;
            num_field(&mut state.to_save.num_md_steps, "Steps:", 50, ui);

            if state.to_save.num_md_steps != num_steps_prev {
                state.volatile.md_runtime = state.to_save.num_md_steps as f64 * state.to_save.md_dt;
            }

            ui.label("dt (ps):");
            if ui
                .add_sized(
                    [46., Ui::available_height(ui)],
                    TextEdit::singleline(&mut state.ui.md_dt_input),
                )
                .changed()
            {
                if let Ok(v) = state.ui.md_dt_input.parse::<f64>() {
                    state.to_save.md_dt = v;
                    state.volatile.md_runtime = state.to_save.num_md_steps as f64 * v;
                }
            }
            ui.add_space(COL_SPACING/2.);

            {
                let help_text = "Set the integrator to use for molecular dynamics. Verlet Velocity is a good default.";
                ui.label("Integrator:").on_hover_text(help_text);
                ComboBox::from_id_salt(4)
                    .width(80.)
                    .selected_text(state.to_save.md_config.integrator.to_string())
                    .show_ui(ui, |ui| {
                        // todo: More A/R
                        for v in &[Integrator::VerletVelocity] {
                            ui.selectable_value(&mut state.to_save.md_config.integrator, *v, v.to_string());
                        }
                    })
                    .response
                    .on_hover_text(help_text);
            }
            ui.add_space(COL_SPACING/2.);

            // todo: A/R
            // ui.checkbox(&mut state.to_save.md_config.zero_com_drift, "Zero drift")
            //     .on_hover_text("Zero the center-of-mass of items in the simulation.");

            ui.label("Pres (kPa):");
            if ui
                .add_sized([30., Ui::available_height(ui)], TextEdit::singleline(&mut state.ui.md_pressure_input))
                .changed()
            {
                if let Ok(v) =&mut state.ui.md_pressure_input.parse::<f32>() {
                    state.to_save.md_config.pressure_target = *v;
                }
            }

            ui.label("Temp (K):");
            if ui
                .add_sized([30., Ui::available_height(ui)], TextEdit::singleline(&mut state.ui.md_temp_input))
                .changed()
            {
                if let Ok(v) =&mut state.ui.md_temp_input.parse::<f32>() {
                    state.to_save.md_config.temp_target = *v;
                }
            }

            {
                let help_text = "Set to Constrained to allow higher time steps; Flexible may more more accurate.";
                ui.label("H:").on_hover_text(help_text);
                ComboBox::from_id_salt(5)
                    .width(80.)
                    .selected_text(state.to_save.md_config.hydrogen_constraint.to_string())
                    .show_ui(ui, |ui| {
                        // todo: More A/R
                        for v in &[HydrogenConstraint::Constrained, HydrogenConstraint::Flexible] {
                            ui.selectable_value(&mut state.to_save.md_config.hydrogen_constraint, *v, v.to_string());
                        }
                    })
                    .response
                    .on_hover_text(help_text);
            }
            ui.add_space(COL_SPACING/2.);

            num_field(&mut state.to_save.md_config.snapshot_ratio_memory, "Snapshot ratio:", 22, ui);

            // int_field_usize(&mut state.to_save.md_config.snapshot_ratio_file, "Snapshot ratio:", ui);


            ui.label("Solvent pad (Å):");
            if ui
                .add_sized([22., Ui::available_height(ui)], TextEdit::singleline(&mut state.ui.md_simbox_pad_input))
                .changed()
            {
                if let Ok(v) = &mut state.ui.md_simbox_pad_input.parse::<f32>() {
                    state.to_save.md_config.sim_box = SimBoxInit::Pad(*v);
                }
            }


            ui.add_space(COL_SPACING);
            ui.label(format!("Runtime: {:.1} ps", state.volatile.md_runtime));

            if let Some(md) = &state.mol_dynamics {
                let snap = &md.snapshots[state.ui.current_snapshot];

                ui.add_space(COL_SPACING);
                ui.label("E (kcal/mol) KE: ");
                ui.label(RichText::new(format!("{:.1}", snap.energy_kinetic)).color(Color32::GOLD));

                ui.label("PE: ");
                ui.label(RichText::new(format!("{:.1}", snap.energy_potential)).color(Color32::GOLD));
            }

        });
    });

    misc::dynamics_player(state, scene, engine_updates, ui);
}
