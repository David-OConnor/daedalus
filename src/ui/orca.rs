use std::{path::PathBuf, str::FromStr};

use bio_files::orca::{
    GeomOptThresh, Keyword, OrcaInput, OrcaOutput, Task,
    basis_sets::BasisSetCategory,
    dynamics::{Dynamics, Thermostat},
    method::Method,
};

use egui::{Color32, ComboBox, RichText, Ui};
use graphics::{EngineUpdates, Scene};

use crate::{
    State, label, orca,
    orca::TaskType,
    ui::{COL_SPACING, COLOR_ACTION, COLOR_ACTIVE, misc},
    util::{handle_err, handle_success},
};

fn keyword_toggle(
    input: &mut OrcaInput,
    keyword: Keyword,
    text: &str,
    help_text: &str,
    ui: &mut Ui,
) {
    let color = if input.keywords.contains(&keyword) {
        COLOR_ACTIVE
    } else {
        Color32::GRAY
    };

    if ui
        .button(RichText::new(text).color(color))
        .on_hover_text(help_text)
        .clicked()
    {
        if input.keywords.contains(&keyword) {
            input.keywords.retain(|&k| k != keyword);
        } else {
            input.keywords.push(keyword);
        }
    }
}

pub(in crate::ui) fn orca_input(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    redraw: &mut bool,
    ui: &mut Ui,
) {
    misc::section_box().show(ui, |ui| {
        ui.horizontal(|ui| {
            if state.volatile.orca_avail {
                label!(ui, "ORCA ready", Color32::LIGHT_GREEN)
                    .on_hover_text("ORCA is installed and available; ready to run");
            } else {
                label!(ui, "ORCA unavail", Color32::LIGHT_RED)
                    .on_hover_text("Can't find ORCA. Is it installed? Is it available on the system path?");
            }

            ui.add_space(COL_SPACING / 2.0);

            {
                let help_text = "Choose the task to perform using ORCA.";
                ui.label("Task:").on_hover_text(help_text);
                ComboBox::from_id_salt(698)
                    // todo: different repr a/r
                    .selected_text(state.orca.task_type.to_string())
                    .show_ui(ui, |ui| {
                        for task in [TaskType::SinglePoint, TaskType::GeometryOptimization, TaskType::MbisCharges, TaskType::MolDynamics] {
                            ui.selectable_value(&mut state.orca.task_type, task, task.to_string()).on_hover_text(task.to_string());
                        }
                    })
                    .response
                    .on_hover_text(help_text);
            }

            {
                ui.add_space(COL_SPACING / 2.);
                let help_text = "Choose the method / functional. Note that ones that end in 3c have included basis sets.";
                ui.label("Method:").on_hover_text(help_text);
                ComboBox::from_id_salt(699)
                    .height(400.)
                    // todo: different repr a/r
                    .selected_text(state.orca.input.method.keyword())
                    .show_ui(ui, |ui| {
                        for m in &[
                            Method::Hf_3c,
                            Method::r2SCAN_3c,
                            Method::B97_3c,
                            Method::PBEh_3c,
                            Method::B3LYP_3c,
                            Method::wB97x_3c,
                            Method::PBE0,
                            Method::HartreeFock,
                            Method::Dft,
                            Method::BLYP,
                            Method::B1LYP,
                            Method::B3LYP,
                            Method::BP86,
                            Method::Mp2Perturbation,
                            Method::SpinComponentScaledMp2,
                            Method::OrbitalOptimzedMp2,
                            Method::RegularlizedMp2,
                            Method::DoubleHybridDft,
                            Method::CoupledCluster,
                            Method::Xtb,
                            Method::SemiEmpericalMethods,
                            Method::B97M_V,
                            Method::R2SCAN,
                            Method::FractionalOccupationDensity,
                            Method::None
                        ] {
                            ui.selectable_value(&mut state.orca.input.method, *m, m.keyword());
                        }
                    })
                    .response
                    .on_hover_text(help_text);
            }

            if !state.orca.input.method.is_composite() {
                {
                    ui.add_space(COL_SPACING / 2.);
                    let help_text = "Choose the category of basis sets; this helps organize basis sets you select to the right.";
                    ui.label("Basis cat:").on_hover_text(help_text);
                    ComboBox::from_id_salt(700)
                        // todo: different repr a/r
                        .selected_text(state.orca.basis_set_cat.to_string())
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut state.orca.basis_set_cat, BasisSetCategory::Pople, BasisSetCategory::Pople.to_string());
                            ui.selectable_value(&mut state.orca.basis_set_cat, BasisSetCategory::Ahlrich, BasisSetCategory::Ahlrich.to_string());
                            ui.selectable_value(&mut state.orca.basis_set_cat, BasisSetCategory::KarlseruheDef2, BasisSetCategory::KarlseruheDef2.to_string());
                            ui.selectable_value(&mut state.orca.basis_set_cat, BasisSetCategory::KarlseruheDhf, BasisSetCategory::KarlseruheDhf.to_string());
                            ui.selectable_value(&mut state.orca.basis_set_cat, BasisSetCategory::CorrelationConsistent, BasisSetCategory::CorrelationConsistent.to_string());
                        })
                        .response
                        .on_hover_text(help_text);
                }

                {
                    let help_text = "Choose the basis set";
                    ui.label("Basis:").on_hover_text(help_text);
                    ComboBox::from_id_salt(701)
                        .height(600.)

                        // todo: different repr a/r
                        .selected_text(state.orca.input.basis_set.keyword())
                        .show_ui(ui, |ui| {
                            for b in &state.orca.basis_set_cat.get_sets() {
                                ui.selectable_value(&mut state.orca.input.basis_set, *b, b.keyword());
                            }
                        })
                        .response
                        .on_hover_text(help_text);
                }
            }

            // If applicable, set up per-task config.
            match state.orca.task_type {
                TaskType::GeometryOptimization => {
                    let help_text = "Choose Geometry optimization threshold. Defaults to Opt.";
                    ui.label("Opt:").on_hover_text(help_text);
                    ComboBox::from_id_salt(702)
                        .height(600.)
                        .width(60.)

                        .selected_text(state.orca.geom_opt_thresh.to_string())
                        .show_ui(ui, |ui| {
                            for thresh in [GeomOptThresh::Loose, GeomOptThresh::Opt, GeomOptThresh::Tight, GeomOptThresh::VeryTight] {
                                ui.selectable_value(&mut state.orca.geom_opt_thresh, thresh, thresh.to_string());
                            }
                        })
                        .response
                        .on_hover_text(help_text);

                    {
                        // todo?
                        // let help_text = "Set the SCF convergence tightness";
                        // ui.label("SCF tol:").on_hover_text(help_text);
                        //
                        // let text = match &state.orca.input.scf {
                        //     Some(s) => s.convergence_tolerance.keyword(),
                        //     None => "Disabled".to_string(),
                        // };
                    }

                    ui.add_space(COL_SPACING / 2.0);
                }
                TaskType::MolDynamics => {}
                _ => (),
            }

            // todo: Make this a default?
            if !state.orca.input.method.is_composite() {
                keyword_toggle(
                    &mut state.orca.input,
                    Keyword::D4Dispersion,
                    "D4",
                    "Compute a D4 dispersion correction. Recommended when not using a composite method.",
                    ui,
                );
            }

            // todo: Sort these out here, and in bio files. e.g. in a Task type A/R, and with parsed output.
            keyword_toggle(
                &mut state.orca.input,
                Keyword::AnFreq,
                "Freq (analytic)",
                "Compute the vibrational frequencies, using analytical methods",
                ui,
            );

            keyword_toggle(
                &mut state.orca.input,
                Keyword::NumFreq,
                "Freq (numeric)",
                "Compute the vibrational frequencies, using numeric methods",
                ui,
            );

            // Avoids borrow error
            // todo: QC if you still need this.
            let mut run = false;

            if state.active_mol().is_some() {
                // todo: Delegate to src/orca A/R.
                //     if ui
                //         .button(RichText::new("Find min energy"))
                //         .on_hover_text(
                //             "Uses Orca's conformer search to find the global minimum energy conformation \
                // of the active small organic molecule.",
                //         )
                //         .clicked()
                //     {
                //         let Some(mol) = state.active_mol() else {
                //             return;
                //         };
                //         let atoms: Vec<_> = mol.common().atoms.iter().map(|a| a.to_generic()).collect();
                //         let mut inp = OrcaInput {
                //             method: Method::Xtb,
                //             basis_set: BasisSet::None,
                //             atoms,
                //             ..Default::default()
                //         };
                //
                //         inp.keywords.push(Keyword::ConformerSearch);
                //         println!("\nRunning ORCA input: \n{}\n", inp.make_inp());
                //     }
                //
                //     ui.add_space(COL_SPACING);

                if state.volatile.orca_avail {
                    if ui
                        .button(RichText::new("Run").color(COLOR_ACTION))
                        .on_hover_text(
                            "Run ORCA using the settings here, on the active molecule.",
                        )
                        .clicked()
                    {
                        run = true;
                        let Some(mol) = state.active_mol() else {
                            return;
                        };
                        state.orca.input.atoms = mol.common().atoms.iter().map(|a| a.to_generic()).collect();
                    }
                }
            }

            if run {

                // Assign parameters to the input which weren't conducive to do directly in the UI.
                state.orca.input.task = match state.orca.task_type {
                    TaskType::SinglePoint => Task::SinglePoint,
                    TaskType::GeometryOptimization => Task::GeometryOptimization((state.orca.geom_opt_thresh, None)),
                    TaskType::MbisCharges => Task::MbisCharges,
                    TaskType::MolDynamics => Task::MolDynamics(Dynamics {
                        // Convert ps to fs.
                        timestep: state.to_save.md_dt * 1_000., // ps to fs.
                        init_vel: state.to_save.md_config.temp_target,
                        thermostat: Thermostat::Csvr,
                        thermostat_temp: state.to_save.md_config.temp_target,
                        thermostat_timecon: 10., // between 10 and 100 generally.
                        traj_out_dir: PathBuf::from_str("out_traj.xyz").unwrap(),
                        steps: state.to_save.num_md_steps,
                    })
                };

                println!("Running ORCA input:\n{}\n...", state.orca.input.make_inp());

                match state.orca.input.run() {
                    Ok(out) => {
                        let Some(mut mol) = state.active_mol_mut() else {
                            return;
                        };

                        // This should correspond to the task.
                        match out {
                            OrcaOutput::Text(t) => {
                                println!("ORCA run complete. Output: \n\n{t}");

                                handle_success(&mut state.ui, format!("ORCA run complete"));
                            }
                            OrcaOutput::Charges(o) => {
                                println!("Charge output: {:?}", o);
                                // handle_success(&mut state.ui, format!("MBIS charges assigned for {}", mol.common().ident));

                                if o.charges.len() != mol.common().atoms.len() {
                                    // todo: Borrow mut error.
                                    // handle_err(&mut state.ui, "Mismatch in len on MBIS charges".to_string());
                                    eprintln!("Mismatch in atom count on MBIS charges.");
                                } else {
                                    for (i, q) in o.charges.into_iter().enumerate() {
                                        mol.common_mut().atoms[i].partial_charge = Some(q.charge as f32);
                                    }
                                }
                                // Maybe only required if in color-by-charge mode.
                                *redraw = true;
                            }
                            OrcaOutput::Dynamics(o) => {
                                println!("\n\nMD Trajectory: \n\n{:?}", o.trajectory);
                                orca::update_snapshots(state, o);
                            }
                            OrcaOutput::Geometry(p) => {
                                if p.posits.len() != mol.common().atoms.len() {
                                    eprintln!("Error: mismatch in len on atom count on geometry optimization.")
                                } else {
                                    for (i, posit) in p.posits.iter().enumerate() {
                                        mol.common_mut().atom_posits[i] = *posit;
                                    }
                                }

                                println!("Updated Atom positions from ORCA:");
                                for (i, p) in p.posits.into_iter().enumerate() {
                                    println!("{}: {p}", i + 1);
                                }
                                *redraw = true;
                            }
                        }
                    }
                    Err(e) => {
                        handle_err(&mut state.ui, format!("Problem running ORCA: {e:?}"));
                    }
                }
            }
        });
    });
}
