use dynamics::{ComputationDevice, HydrogenConstraint, Integrator, SimBoxInit, snapshot::Snapshot};
use egui::{Color32, ComboBox, RichText, TextEdit, Ui};
use graphics::{EngineUpdates, Scene};
use lin_alg::f64::Vec3;

use crate::{
    State,
    md::build_and_run_dynamics,
    ui::{
        COL_SPACING, COLOR_ACTIVE, COLOR_HIGHLIGHT, COLOR_INACTIVE, cam::move_cam_to_active_mol,
        flag_btn, misc, num_field,
    },
    util::{clear_cli_out, handle_err},
};

pub fn md_setup(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    misc::section_box().show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label("MD:");
            if let Some(mol) = &mut state.peptide {
                flag_btn(&mut mol.common.selected_for_md, &mol.common.ident, "Toggle if we use this molecule for MD.", ui);

                let num_ligs = state.ligands.iter().filter(|l| l.common.selected_for_md).count();
                if mol.common.selected_for_md && num_ligs > 0 {
                    flag_btn(&mut state.ui.md.peptide_only_near_ligs, "Pep only near lig", "Only model the subset of peptide atoms near a small molecule", ui);
                    flag_btn(&mut state.ui.md.peptide_static, "Pep static", "Let peptide (protein) atoms affect other molecules, but they don't move themselves", ui);
                }
            }

            if !state.lipids.is_empty() {
                let prev_val = state.lipids[0].common.selected_for_md;
                let color = if prev_val { COLOR_ACTIVE } else { COLOR_INACTIVE };
                if ui
                    .button(RichText::new("All lipids").color(color))
                    .on_hover_text("Select all lipids for MD")
                    .clicked()
                {
                    for l in &mut state.lipids {
                        l.common.selected_for_md = !prev_val;
                    }
                }


            }

            for mol in &mut state.ligands {
                flag_btn(&mut mol.common.selected_for_md, &mol.common.ident, "Toggle if we use this molecule for MD.", ui);
            }

            ui.add_space(COL_SPACING / 2.);

            let run_clicked = ui
                .button(RichText::new("Run MD").color(Color32::GOLD))
                .on_hover_text("Run a molecular dynamics simulation on all molecules selected.")
                .clicked();

            if run_clicked {
                clear_cli_out(&mut state.ui); // todo: Not working; not loaded until next frame.
                let mut ready_to_run = true;

                // Check that we have FF params and mol-specific parameters.
                for lig in &state.ligands {
                    if !lig.common.selected_for_md {
                        continue
                    }
                    if !lig.ff_params_loaded || !lig.frcmod_loaded {
                        state.ui.popup.show_get_geostd = true;
                        ready_to_run = false;
                    }
                }

                if ready_to_run {
                    {
                        let center = match &state.peptide {
                            Some(m) => m.center,
                            None => Vec3::new(0., 0., 0.),
                        };
                        // todo: Set a loading indicator, and trigger the build next GUI frame.
                        move_cam_to_active_mol(state, scene, center, engine_updates);
                    }

                    // Filter molecules for docking by if they're selected.
                    // mut so we can move their posits in the initial snapshot change.
                    let ligs: Vec<_> = state.ligands.iter_mut().filter(|l| l.common.selected_for_md).collect();
                    let lipids: Vec<_> = state.lipids.iter_mut().filter(|l| l.common.selected_for_md).collect();

                    let mol = match &state.peptide {
                        Some(m) => if m.common.selected_for_md { Some(m) } else { None },
                        None => None,
                    };
                    match build_and_run_dynamics(
                        &state.dev,
                        ligs,
                        lipids,
                        mol,
                        &state.ff_param_set,
                        &state.lig_specific_params,
                        &state.to_save.md_config,
                        state.ui.md.peptide_static,
                        state.ui.md.peptide_only_near_ligs,
                        &mut state.volatile.md_peptide_selected,
                        &mut state.volatile.md_local,
                    ) {
                        Ok(md) => {
                            state.mol_dynamics = Some(md);
                        }
                        Err(e) => handle_err(&mut state.ui, e.descrip),
                    }
                }
            }

            if state.volatile.md_local.running {
                if let Some(md) = &state.mol_dynamics {
                    let count = (md.step_count / 100) * 100;
                    ui.label(RichText::new(format!("MD running. Step {} of {}", count, state.to_save.num_md_steps)).color(COLOR_HIGHLIGHT));
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
                state.volatile.md_runtime = state.to_save.num_md_steps as f32 * state.to_save.md_dt;
            }

            ui.label("dt (ps):");
            if ui
                .add_sized(
                    [46., Ui::available_height(ui)],
                    TextEdit::singleline(&mut state.ui.md.dt_input),
                )
                .changed()
            {
                if let Ok(v) = state.ui.md.dt_input.parse::<f32>() {
                    state.to_save.md_dt = v;
                    state.volatile.md_runtime = state.to_save.num_md_steps as f32 * v;
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
                        // todo: What should gamma be? And make it customizable in UI and state.
                        for v in &[Integrator::LangevinMiddle { gamma: 0. }, Integrator::VerletVelocity] {
                            ui.selectable_value(&mut state.to_save.md_config.integrator, v.clone(), v.to_string());
                        }
                    })
                    .response
                    .on_hover_text(help_text);
            }
            if matches!(state.to_save.md_config.integrator, | Integrator::LangevinMiddle { gamma: _ }) {
                ui.label("γ:");
                if ui
                    .add_sized([22., Ui::available_height(ui)], TextEdit::singleline(&mut state.ui.md.langevin_γ))
                    .changed()
                {
                    if let Ok(v) = &mut state.ui.md.langevin_γ.parse::<f32>() {
                        match state.to_save.md_config.integrator {
                            // Integrator::Langevin { gamma: _ } => state.to_save.md_config.integrator = Integrator::Langevin { gamma: *v},
                            Integrator::LangevinMiddle { gamma: _} => state.to_save.md_config.integrator = Integrator::Langevin { gamma: *v},
                            _ => ()
                        }
                    }
                }
            }

            ui.add_space(COL_SPACING/2.);

            // todo: A/R
            // ui.checkbox(&mut state.to_save.md_config.zero_com_drift, "Zero drift")
            //     .on_hover_text("Zero the center-of-mass of items in the simulation.");

            ui.label("Pres (kPa):");
            if ui
                .add_sized([30., Ui::available_height(ui)], TextEdit::singleline(&mut state.ui.md.pressure_input))
                .changed()
            {
                if let Ok(v) =&mut state.ui.md.pressure_input.parse::<f32>() {
                    state.to_save.md_config.pressure_target = *v;
                }
            }

            ui.label("Temp (K):");
            if ui
                .add_sized([30., Ui::available_height(ui)], TextEdit::singleline(&mut state.ui.md.temp_input))
                .changed()
            {
                if let Ok(v) =&mut state.ui.md.temp_input.parse::<f32>() {
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

            // todo: Add snapshot cfg
            // num_field(&mut state.to_save.md_config.snapshot_ratio_memory, "Snapshot ratio:", 22, ui);

            // int_field_usize(&mut state.to_save.md_config.snapshot_ratio_file, "Snapshot ratio:", ui);


            let hover_text = "Set the minimum distance to pad the molecule in water atoms. Large values \
            can be more realistic, but significantly increase computation time.";
            ui.label("Solvent pad (Å):").on_hover_text(hover_text);
            if ui
                .add_sized([22., Ui::available_height(ui)], TextEdit::singleline(&mut state.ui.md.simbox_pad_input))
                .on_hover_text(hover_text)
                .changed()
            {
                if let Ok(v) = &mut state.ui.md.simbox_pad_input.parse::<f32>() {
                    state.to_save.md_config.sim_box = SimBoxInit::Pad(*v);
                }
            }


            ui.add_space(COL_SPACING/2.);
            ui.label(format!("Runtime: {:.1} ps", state.volatile.md_runtime));

            if let Some(md) = &state.mol_dynamics {
                if state.ui.current_snapshot < md.snapshots.len() {
                    energy_disp(&md.snapshots[state.ui.current_snapshot], ui);
                }
            }

        });
    });

    misc::dynamics_player(state, scene, engine_updates, ui);
}

pub fn energy_disp(snap: &Snapshot, ui: &mut Ui) {
    ui.add_space(COL_SPACING / 2.);
    ui.label("E (kcal/mol) KE: ");
    ui.label(RichText::new(format!("{:.1}", snap.energy_kinetic as u32)).color(Color32::GOLD));

    ui.label("E / atom: ");
    // todo: Don't continuosly run this!
    let e_per_atom = snap.energy_kinetic
        / ((snap.water_o_posits.len() * 3) as f32 + snap.atom_posits.len() as f32);
    ui.label(RichText::new(format!("{:.1}", e_per_atom)).color(Color32::GOLD));

    ui.label("PE: ");
    ui.label(RichText::new(format!("{:.1}", snap.energy_potential as u32)).color(Color32::GOLD));
}
