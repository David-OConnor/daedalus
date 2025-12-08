use dynamics::{
    ComputationDevice, FfMolType, HydrogenConstraint, Integrator, MdConfig, SimBoxInit,
    snapshot::Snapshot,
};
use egui::{Color32, ComboBox, RichText, TextEdit, Ui};
use graphics::{EngineUpdates, EntityUpdate, Scene};
use lin_alg::f64::Vec3;

use crate::{
    State,
    drawing::EntityClass,
    label,
    md::{
        STATIC_ATOM_DIST_THRESH, build_and_run_dynamics, launch_md, post_run_cleanup,
        reassign_snapshot_indices,
    },
    ui::{
        COL_SPACING, COLOR_ACTION, COLOR_ACTIVE, COLOR_HIGHLIGHT, COLOR_INACTIVE,
        cam::move_cam_to_active_mol, flag_btn, misc, num_field,
    },
    util::{clear_cli_out, handle_success},
};

pub fn md_setup(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    // This sequencing code is above the UI code below, so it's deferred a frame after any actions.
    if state.volatile.md_local.launching {
        state.volatile.md_local.launching = false;
        launch_md(state);
    } else if state.volatile.md_local.running {
        // This is spammed each frame, so don't print, which handle_success does.
        state.ui.cmd_line_output = "MD Running...".to_string();
        state.ui.cmd_line_out_is_err = false;
    }

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

            for mol in &mut state.nucleic_acids {
                flag_btn(&mut mol.common.selected_for_md, &mol.common.ident, "Toggle if we use this molecule for MD.", ui);
            }

            ui.add_space(COL_SPACING / 2.);

            if let Some(md) = &state.mol_dynamics &&
                state.ui.current_snapshot < md.snapshots.len() &&
                !md.snapshots[state.ui.current_snapshot].water_o_posits.is_empty() {
                if ui
                    .button(RichText::new("Clear water"))
                    .on_hover_text("Clear all rendered water molecules from the display")
                    .clicked() {

                    scene
                        .entities
                        .retain(|ent| ent.class != EntityClass::WaterModel as u32);
                }



                // todo: Setting water class isn't working for this update.
                // engine_updates.entities = EntityUpdate::Classes(vec![EntityClass::WaterModel as u32]);
                engine_updates.entities = EntityUpdate::All;
            }

            if ui
                .button(RichText::new("Run MD").color(COLOR_ACTION))
                .on_hover_text("Run a molecular dynamics simulation on all molecules selected.")
                .clicked() {
                clear_cli_out(&mut state.ui); // todo: Not working; not loaded until next frame.
                let mut ready_to_run = true;

                // Check that we have FF params and mol-specific parameters.
                for lig in &state.ligands {
                    if !lig.common.selected_for_md {
                        continue;
                    }
                    if !lig.ff_params_loaded || !lig.frcmod_loaded {
                        state.ui.popup.show_get_geostd = true;
                        ready_to_run = false;
                    }
                }

                if ready_to_run {
                    let center = match &state.peptide {
                        Some(m) => m.center,
                        None => Vec3::new(0., 0., 0.),
                    };
                    // todo: Set a loading indicator, and trigger the build next GUI frame.
                    move_cam_to_active_mol(state, scene, center, engine_updates);

                    handle_success(&mut state.ui, "Running MD. Initializing water, and relaxing the molecules...".to_string());

                    // We will wait a frame so we can display the message above.
                    state.volatile.md_local.launching = true;
                }
            }

            if state.volatile.md_local.running {
                if ui
                    .button(RichText::new("Abort").color(Color32::LIGHT_RED))
                    .on_hover_text("Stop the in-progress simulation")
                    .clicked()
                {
                    post_run_cleanup(state, scene, engine_updates);
                }

                if let Some(md) = &state.mol_dynamics {
                    let count = (md.step_count / 100) * 100;

                    ui.label(
                        RichText::new(format!(
                            "MD running. Step {} of {}",
                            count, state.to_save.num_md_steps
                        ))
                            .color(COLOR_HIGHLIGHT),
                    );
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
            ui.add_space(COL_SPACING / 2.);

            {
                let help_text = "Set the integrator to use for molecular dynamics. Verlet Velocity is a good default.";
                ui.label("Integrator:").on_hover_text(help_text);

                let mut prev = state.to_save.md_config.integrator.clone();
                if ComboBox::from_id_salt(4)
                    .width(80.)
                    .selected_text(state.to_save.md_config.integrator.to_string())
                    .show_ui(ui, |ui| {
                        // todo: More A/R
                        // todo: What should gamma be? And make it customizable in UI and state.
                        // todo: Langevin mid thermostat is out of control, and I'm not sure how
                        // todo to fix it.
                        // todo: For tau and gamma, consider using defaults from dynamics.

                        // todo temp; allow for enabling/disabling therm.
                        for v in &[
                            Integrator::LangevinMiddle { gamma: 0.3 },
                            Integrator::VerletVelocity { thermostat: Some(1.0) },
                        ] {
                            ui.selectable_value(&mut state.to_save.md_config.integrator, v.clone(), v.to_string());
                        }
                    })
                    .response
                    .on_hover_text(help_text).changed() {
                    if let Integrator::LangevinMiddle { gamma } = state.to_save.md_config.integrator {
                        if !matches!(prev, Integrator::LangevinMiddle {gamma: _}) {
                            state.ui.md.langevin_γ = gamma.to_string();
                        }
                    }
                }
            }

            match &mut state.to_save.md_config.integrator {
                Integrator::LangevinMiddle { gamma } => {
                    ui.label("γ:");
                    if ui
                        .add_sized([22., Ui::available_height(ui)], TextEdit::singleline(&mut state.ui.md.langevin_γ))
                        .changed()
                    {
                        if let Ok(v) = &mut state.ui.md.langevin_γ.parse::<f32>() {
                            *gamma = *v;
                        }
                    }
                }
                Integrator::VerletVelocity { thermostat } => {
                    let help_text = "Enable or disable the thermostat";
                    ui.label("Therm:").on_hover_text(help_text);
                    let mut v = thermostat.is_some();
                    if ui.checkbox(&mut v, "").on_hover_text(help_text).changed() {
                        *thermostat = if v {
                            Some(1.0)
                        } else {
                            None
                        };
                    }
                }
            }

            if matches!(state.to_save.md_config.integrator, | Integrator::LangevinMiddle { gamma: _ }) {}

            ui.add_space(COL_SPACING / 2.);

            // todo: A/R
            // ui.checkbox(&mut state.to_save.md_config.zero_com_drift, "Zero drift")
            //     .on_hover_text("Zero the center-of-mass of items in the simulation.");

            ui.label("Pres (bar):");
            if ui
                .add_sized([30., Ui::available_height(ui)], TextEdit::singleline(&mut state.ui.md.pressure_input))
                .changed()
            {
                if let Ok(v) = &mut state.ui.md.pressure_input.parse::<f32>() {
                    state.to_save.md_config.pressure_target = *v;
                }
            }

            ui.label("Temp (K):");
            if ui
                .add_sized([30., Ui::available_height(ui)], TextEdit::singleline(&mut state.ui.md.temp_input))
                .changed()
            {
                if let Ok(v) = &mut state.ui.md.temp_input.parse::<f32>() {
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
            ui.add_space(COL_SPACING / 2.);

            // todo: Add snapshot cfg
            // num_field(&mut state.to_save.md_config.snapshot_ratio_memory, "Snapshot ratio:", 22, ui);

            // int_field_usize(&mut state.to_save.md_config.snapshot_ratio_file, "Snapshot ratio:", ui);

            let mut relax = state.to_save.md_config.max_init_relaxation_iters.is_some();
            let relax_prev = relax;
            ui.label("Relax:");
            ui.checkbox(&mut relax, "").on_hover_text("Perform an initial relaxation of atom positions prior to starting MD. \
            This minimizes energy, and can take some time.");
            if relax != relax_prev {
                if relax {
                    state.to_save.md_config.max_init_relaxation_iters = MdConfig::default().max_init_relaxation_iters;
                } else {
                    state.to_save.md_config.max_init_relaxation_iters = None;
                }
            }


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


            ui.add_space(COL_SPACING / 2.);
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

pub(in crate::ui) fn energy_disp(snap: &Snapshot, ui: &mut Ui) {
    ui.add_space(COL_SPACING / 2.);
    ui.label("E (kcal/mol) KE: ");
    ui.label(RichText::new(format!("{:.1}", snap.energy_kinetic)).color(Color32::GOLD));

    ui.label("KE/atom: ");
    // todo: Don't continuously run these computations!
    let atom_count = (snap.water_o_posits.len() * 3) as f32 + snap.atom_posits.len() as f32;
    let ke_per_atom = snap.energy_kinetic / atom_count;
    ui.label(RichText::new(format!("{:.2}", ke_per_atom)).color(Color32::GOLD));

    ui.label("PE: ");
    label!(ui, format!("{:.2}", snap.energy_potential), Color32::GOLD);

    ui.label("PE/atom: ");
    // todo: Don't continuously run this!
    let pe_per_atom = snap.energy_potential / atom_count;
    label!(ui, format!("{:.3}", pe_per_atom), Color32::GOLD);

    ui.label("E tot: ");
    // todo: Don't continuously run this!
    let e = snap.energy_potential + snap.energy_kinetic;
    label!(ui, format!("{:.3}", e), Color32::GOLD);

    ui.label("PE between mols:");
    // todo: One pair only for now
    if snap.energy_potential_between_mols.len() >= 2 {
        // todo: Which index?
        ui.label(
            RichText::new(format!("{:.2}", snap.energy_potential_between_mols[1]))
                .color(Color32::GOLD),
        );
    }

    ui.label("Temp: ");
    label!(ui, format!("{:.1} K", snap.temperature), Color32::GOLD);

    ui.label("P: ");
    label!(ui, format!("{:.1} bar", snap.pressure), Color32::GOLD);
}
