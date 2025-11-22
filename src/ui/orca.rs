use bio_files::orca::{
    Keyword, OrcaInput,
    basis_sets::{BasisSet, BasisSetCategory},
    method::Method,
};
use dynamics::Integrator;
use egui::{Color32, ComboBox, RichText, Ui};
use graphics::{EngineUpdates, Scene};

use crate::{
    State, label,
    orca::StateOrca,
    ui::{COL_SPACING, COLOR_ACTIVE, COLOR_HIGHLIGHT, misc, misc::toggle_btn},
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

pub(super) fn orca_input(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
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
                let help_text = "Choose the method";
                ui.label("Method:").on_hover_text(help_text);
                ComboBox::from_id_salt(699)
                    // .width(80.)
                    .height(400.)
                    // todo: different repr a/r
                    .selected_text(state.orca.input.method.keyword())
                    .show_ui(ui, |ui| {
                        for m in &[
                            Method::HartreeFock,
                            Method::Dft,
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
                            // Method::Wb97XV,
                            Method::FractionalOccupationDensity,
                            Method::None
                        ] {
                            ui.selectable_value(&mut state.orca.input.method, *m, m.keyword());
                        }
                    })
                    .response
                    .on_hover_text(help_text);
            }

            {
                let help_text = "Choose the category of basis sets; this helps organize basis sets you select to the right.";
                ui.label("Basis cat:").on_hover_text(help_text);
                ComboBox::from_id_salt(700)
                    // .width(80.)
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
                    // .width(80.)

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

            {
                let help_text = "Set the SCF convergence tightness";
                ui.label("SCF tol:").on_hover_text(help_text);

                let text = match &state.orca.input.scf {
                    Some(s) => s.convergence_tolerance.keyword(),
                    None => "Disabled".to_string(),
                };

                // ComboBox::from_id_salt(702)
                //     .width(80.)
                //     // todo: different repr a/r
                //     .selected_text(text)
                //     .show_ui(ui, |ui| {
                //         for m in &[
                //             ScfConvergenceTolerance::None,
                //             ScfConvergenceTolerance::Sloppy,
                //             ScfConvergenceTolerance::Loose,
                //             ScfConvergenceTolerance::Medium,
                //             ScfConvergenceTolerance::Strong,
                //             ScfConvergenceTolerance::Tight,
                //             ScfConvergenceTolerance::VeryTight,
                //             ScfConvergenceTolerance::Extreme,
                //         ] {
                //             ui.selectable_value(&mut state.orca.input.basis_set, *m, m.keyword());
                //         }
                //     })
                //     .response
                //     .on_hover_text(help_text);
            }

            ui.add_space(COL_SPACING / 2.0);

            keyword_toggle(
                &mut state.orca.input,
                Keyword::OptimizeGeometry,
                "Opt geom",
                "Optimize atom geometry",
                ui,
            );

            keyword_toggle(
                &mut state.orca.input,
                Keyword::Mbis,
                "MBIS charge",
                "Apply the MBIS model to generate atom-centered s-tyhpe Slater functions. Can be \
                used for FF parameterization",
                ui,
            );

            // toggle_btn(
            //     &mut state.orca.input.optimize_geometry,
            //     "Opt geom",
            //     "Optimize atom geometry",
            //     ui,
            //     &mut false,
            // );

            // toggle_btn(
            //     &mut state.orca.input.optimize_hydrogens,
            //     "Opt H",
            //     "Optimize atom geometry for Hydrogens only",
            //     ui,
            //     &mut false,
            // );

            if let Some(mol) = state.active_mol() {
                // todo: Delegate to src/orca A/R.
                if ui
                    .button(RichText::new("Find min energy"))
                    .on_hover_text(
                        "Uses Orca's conformer search to find the global minimum energy conformation \
            of the active small organic molecule.",
                    )
                    .clicked()
                {
                    let atoms: Vec<_> = mol.common().atoms.iter().map(|a| a.to_generic()).collect();
                    let mut inp = OrcaInput {
                        method: Method::Xtb,
                        basis_set: BasisSet::None,
                        atoms,
                        ..Default::default()
                    };

                    inp.keywords.push(Keyword::ConformerSearch);
                    println!("\nRunning ORCA input: \n{}\n", inp.make_inp());
                }

                ui.add_space(COL_SPACING);

                if state.volatile.orca_avail {
                    if ui
                        .button(RichText::new("Run").color(Color32::GOLD))
                        .on_hover_text(
                            "Run ORCA using the settings here, on the active molecule.",
                        )
                        .clicked()
                    {
                        let atoms: Vec<_> = mol.common().atoms.iter().map(|a| a.to_generic()).collect();
                        state.orca.input.atoms = atoms;
                        println!("Running ORCA input: \n{}\n", state.orca.input.make_inp());
                    }
                }
            }
        });
    });
}
