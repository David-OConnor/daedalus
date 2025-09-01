//! Misc utility-related UI functionality.

use bio_apis::{amber_geostd, drugbank, pubchem};
use bio_files::ResidueType;
use egui::{
    Color32, ComboBox, CornerRadius, Frame, Margin, RichText, Slider, Stroke, TextEdit, Ui,
};
use graphics::{EngineUpdates, Scene};
use lin_alg::f64::Vec3;

const COLOR_SECTION_BOX: Color32 = Color32::from_rgb(100, 100, 140);

use crate::{
    ComputationDevice, Selection, State, drawing,
    drawing::{
        CHARGE_MAP_MAX, CHARGE_MAP_MIN, COLOR_AA_NON_RESIDUE_EGUI, draw_ligand, draw_peptide,
        draw_water,
    },
    dynamics::{
        MdMode,
        prep::{
            build_dynamics_docking, build_dynamics_peptide, change_snapshot_docking,
            change_snapshot_peptide,
        },
    },
    mol_lig::{Ligand, MoleculeSmall},
    molecule::{Atom, MoleculeGenericRef, MoleculePeptide, PeptideAtomPosits, Residue, aa_color},
    ui::{
        COL_SPACING, COLOR_ACTIVE, COLOR_ACTIVE_RADIO, COLOR_INACTIVE, ROW_SPACING,
        cam::move_cam_to_lig, int_field, int_field_u16, mol_descrip,
    },
    util::{handle_err, handle_success, make_egui_color, move_lig_to_res},
};

/// `posit_override` is for example, relative atom positions, such as a positioned ligand.
fn disp_atom_data(atom: &Atom, residues: &[Residue], posit_override: Option<Vec3>, ui: &mut Ui) {
    let role = match atom.role {
        Some(r) => format!("Role: {r}"),
        None => String::new(),
    };

    // Similar to `Vec3`'s format impl, but with fewer digits.
    let posit = match posit_override {
        Some(p) => p,
        None => atom.posit,
    };

    let posit_txt = format!("|{:.3}, {:.3}, {:.3}|", posit.x, posit.y, posit.z);

    let text_0 = format!("#{}", atom.serial_number);
    let text_b = atom.element.to_letter();

    ui.label(RichText::new(text_0).color(Color32::WHITE));

    ui.label(RichText::new(posit_txt).color(Color32::GOLD));

    let atom_color = make_egui_color(atom.element.color());
    ui.label(RichText::new(text_b).color(atom_color));

    if let Some(res_i) = atom.residue {
        // Placeholder for water etc.
        let mut res_color = COLOR_AA_NON_RESIDUE_EGUI;
        let res = &residues[res_i];
        let res_txt = &format!("  {res}");

        if let ResidueType::AminoAcid(aa) = res.res_type {
            res_color = make_egui_color(aa_color(aa));
        }

        ui.label(RichText::new(res_txt).color(res_color));
    }

    ui.label(RichText::new(role).color(Color32::LIGHT_GRAY));

    if let Some(tir) = &atom.type_in_res {
        ui.label(RichText::new(format!("{tir}")).color(Color32::LIGHT_YELLOW));
    }

    if let Some(ff) = &atom.force_field_type {
        ui.label(RichText::new(format!("FF: {ff}")).color(Color32::LIGHT_YELLOW));
    }

    if let Some(q) = &atom.partial_charge {
        let plus = if *q > 0. { "+" } else { "" };
        let color = make_egui_color(drawing::color_viridis_float(
            *q,
            CHARGE_MAP_MIN,
            CHARGE_MAP_MAX,
        ));

        ui.label(RichText::new(format!("{plus}q: {q:.2}")).color(color));
    }
}

/// Display text of the selected atom or residue.
pub fn selected_data(
    mol: &MoleculePeptide,
    // ligand: &Option<MoleculeSmall>,
    ligand: Option<&MoleculeSmall>,
    selection: &Selection,
    ui: &mut Ui,
) {
    ui.horizontal_wrapped(|ui| {
        match selection {
            Selection::Atom(sel_i) => {
                if *sel_i >= mol.common.atoms.len() {
                    return;
                }

                let atom = &mol.common.atoms[*sel_i];
                section_box().show(ui, |ui| {
                    disp_atom_data(atom, &mol.residues, None, ui);
                });
            }
            Selection::AtomLigand(sel_i) => {
                let Some(lig) = ligand else {
                    return;
                };
                if *sel_i >= lig.common.atoms.len() {
                    return;
                }

                let atom = &lig.common.atoms[*sel_i];
                let posit = lig.common.atom_posits[*sel_i];
                section_box().show(ui, |ui| {
                    disp_atom_data(atom, &[], Some(posit), ui);
                });
            }
            Selection::Residue(sel_i) => {
                if *sel_i >= mol.residues.len() {
                    return;
                }

                let res = &mol.residues[*sel_i];
                // todo: Color-coding by part like atom, to make easier to view.

                let mut res_color = COLOR_AA_NON_RESIDUE_EGUI;

                if let ResidueType::AminoAcid(aa) = res.res_type {
                    res_color = make_egui_color(aa_color(aa));
                }
                section_box().show(ui, |ui| {
                    ui.label(RichText::new(res.to_string()).color(res_color));
                });
            }
            Selection::Atoms(is) => {
                // todo: A/R
                section_box().show(ui, |ui| {
                    ui.label(RichText::new(format!("{} atoms", is.len())).color(Color32::GOLD));
                });
            }
            Selection::None => (),
        }
    });
}

/// A checkbox to show or hide a category.
pub fn vis_check(val: &mut bool, text: &str, ui: &mut Ui, redraw: &mut bool) {
    let color = active_color(!*val);
    if ui.button(RichText::new(text).color(color)).clicked() {
        *val = !*val;
        *redraw = true;
    }
}

pub fn active_color(val: bool) -> Color32 {
    if val { COLOR_ACTIVE } else { COLOR_INACTIVE }
}

/// Visually distinct; fore buttons that operate as radio buttons
pub fn active_color_sel(val: bool) -> Color32 {
    if val {
        COLOR_ACTIVE_RADIO
    } else {
        COLOR_INACTIVE
    }
}

pub fn dynamics_player(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    if state.mol_dynamics.is_none() {
        return;
    }

    ui.horizontal(|ui| {
        let prev = state.ui.peptide_atom_posits;

        let help_text = "Toggle between viewing the original (pre-dynamics) atom positions, and \
        ones at the selected dynamics snapshot.";
        ui.label("Show atoms:").on_hover_text(help_text);
        ComboBox::from_id_salt(3)
            .width(80.)
            .selected_text(state.ui.peptide_atom_posits.to_string())
            .show_ui(ui, |ui| {
                for view in &[PeptideAtomPosits::Original, PeptideAtomPosits::Dynamics] {
                    ui.selectable_value(&mut state.ui.peptide_atom_posits, *view, view.to_string());
                }
            })
            .response
            .on_hover_text(help_text);

        if state.ui.peptide_atom_posits != prev {
            draw_peptide(state, scene);
            engine_updates.entities = true;
        }

        let snapshot_prev = state.ui.current_snapshot;

        let mut changed = false;

        if let Some(md) = &state.mol_dynamics {
            if !md.snapshots.is_empty() {
                ui.add_space(ROW_SPACING);

                ui.spacing_mut().slider_width = ui.available_width() - 100.;
                ui.add(Slider::new(
                    &mut state.ui.current_snapshot,
                    0..=md.snapshots.len() - 1,
                ));
                ui.label(format!(
                    "{:.2} ps",
                    state.ui.current_snapshot as f64 * state.to_save.md_dt
                ));
            }

            if state.ui.current_snapshot != snapshot_prev {
                changed = true;
                let snap = &md.snapshots[state.ui.current_snapshot];

                match md.mode {
                    MdMode::Docking => {
                        if let Some(lig) = state.active_lig() {
                            // change_snapshot_docking(lig, snap, &mut state.ui.binding_energy_disp);
                        }

                        draw_ligand(state, scene);
                    }
                    MdMode::Peptide => {
                        let mol = state.molecule.as_mut().unwrap();

                        change_snapshot_peptide(mol, &md.atoms, snap);
                        draw_peptide(state, scene);
                    }
                }

                engine_updates.entities = true;
            }
        };

        // This approach avoids a double-borrow.
        if changed {
            if let Some(md) = &state.mol_dynamics {
                let snap = &md.snapshots[state.ui.current_snapshot];

                draw_water(
                    scene,
                    &snap.water_o_posits,
                    &snap.water_h0_posits,
                    &snap.water_h1_posits,
                    state.ui.visibility.hide_water,
                );
            }
        }
    });
}

pub fn md_setup(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    redraw_lig: &mut bool,
    ui: &mut Ui,
) {
    section_box().show(ui, |ui| {
        ui.horizontal(|ui| {
            if ui
                .button(RichText::new("Run MD on peptide").color(Color32::GOLD))
                .clicked() {
                let mol = state.molecule.as_mut().unwrap();

                match build_dynamics_peptide(
                    &state.dev,
                    mol,
                    &state.ff_params,
                    state.to_save.md_temperature as f64,
                    state.to_save.md_pressure as f64 / 100., // Convert kPa to bar.
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

            ui.add_space(COL_SPACING / 2.);

            if state.active_lig().is_some() {
                let run_clicked = ui
                    .button(RichText::new("Run MD docking").color(Color32::GOLD))
                    .on_hover_text("Run a molecular dynamics simulation on the ligand. The peptide atoms apply\
            Coulomb and Van der Waals forces, but do not move themselves. This is intended to be run\
            with the ligand positioned near a receptor site.")
                    .clicked();

                let mut ready_to_run = true;

                if run_clicked {
                    let Some(lig) = state.active_lig_mut() else {
                        return;
                    };

                    if !lig.ff_params_loaded || !lig.frcmod_loaded {
                        state.ui.popup.show_get_geostd = true;
                        ready_to_run = false;
                    }

                    if ready_to_run {
                        let mol = state.molecule.as_mut().unwrap();

                        // todo: Set a loading indicator, and trigger the build next GUI frame.
                        move_cam_to_lig(state, scene, mol.center, engine_updates);

                        match build_dynamics_docking(
                            &state.dev,
                            lig,
                            mol,
                            // state.volatile.docking_setup.as_ref().unwrap(),
                            &state.ff_params,
                            state.to_save.md_temperature as f64,
                            state.to_save.md_pressure as f64 / 100., // Convert kPa to bar.
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

            ui.add_space(COL_SPACING);
            let num_steps_prev = state.to_save.num_md_steps;
            int_field(&mut state.to_save.num_md_steps, "Steps:", &mut false, ui);
            if state.to_save.num_md_steps != num_steps_prev {
                state.volatile.md_runtime = state.to_save.num_md_steps as f64 * state.to_save.md_dt;
            }

            ui.label("dt (ps):");
            if ui
                .add_sized(
                    [60., Ui::available_height(ui)],
                    TextEdit::singleline(&mut state.ui.md_dt_input),
                )
                .changed()
            {
                if let Ok(v) = state.ui.md_dt_input.parse::<f64>() {
                    state.to_save.md_dt = v;
                    state.volatile.md_runtime = state.to_save.num_md_steps as f64 * v;
                }
            }

            // ui.label(format!("Temp: {:.1} K", state.to_save.md_pressure));
            int_field_u16(&mut state.to_save.md_temperature, "Temp (K):", &mut false, ui);
            int_field_u16(&mut state.to_save.md_pressure, "Pressure (kPa):", &mut false, ui);

            // ui.label(format!("Temp: {:.1} kPa", state.to_save.md_temperature));
            // int_field(&mut state.to_save.num_md_steps, "Steps:", &mut false, ui);

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

    dynamics_player(state, scene, engine_updates, ui);
}

// A container that highlights a section of UI code, to make it visually distinct from neighboring areas.
pub fn section_box() -> Frame {
    Frame::new()
        .stroke(Stroke::new(1.0, COLOR_SECTION_BOX))
        .corner_radius(CornerRadius::same(6))
        .inner_margin(Margin::symmetric(8, 2))
        .outer_margin(Margin::symmetric(0, 0))
}

pub fn lig_section(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    redraw_lig: &mut bool,
    close_lig: &mut bool,
    engine_updates: &mut EngineUpdates,
) {
    if let Some(lig) = state.active_lig_mut() {
        ui.horizontal(|ui| {
            mol_descrip(&MoleculeGenericRef::Ligand(&lig), ui);

            if ui.button("Close lig").clicked() {
                *close_lig = true;
            }

            ui.add_space(COL_SPACING);

            // todo status color helper?
            ui.label("Loaded:");
            let color = if lig.ff_params_loaded {
                Color32::LIGHT_GREEN
            } else {
                Color32::LIGHT_RED
            };
            ui.label(RichText::new("FF/q").color(color)).on_hover_text(
                "Green if force field names, and partial charges are assigned \
                    for all ligand atoms. Required for ligand moleculer dynamics and docking.",
            );

            ui.add_space(COL_SPACING / 4.);

            let color = if lig.frcmod_loaded {
                Color32::LIGHT_GREEN
            } else {
                Color32::LIGHT_RED
            };
            ui.label(RichText::new("Frcmod").color(color))
                .on_hover_text(
                    "Green if molecule-specific Amber force field parameters are \
                    loaded for this ligand. Required for ligand molecular dynamics and docking.",
                );

            if let Some(id) = &lig.drugbank_id {
                if ui.button("View on Drugbank").clicked() {
                    drugbank::open_overview(id);
                }
            }

            if let Some(id) = lig.pubchem_cid {
                if ui.button("View on PubChem").clicked() {
                    pubchem::open_overview(id);
                }
            }

            if let Some(cid) = lig.pubchem_cid {
                if ui.button("Find associated structs").clicked() {
                    // todo: Don't block.
                    if lig.associated_structures.is_empty() {
                        match pubchem::load_associated_structures(cid) {
                            Ok(data) => {
                                lig.associated_structures = data;
                                state.ui.popup.show_associated_structures = true;
                            }
                            Err(_) => handle_err(
                                &mut state.ui,
                                "Unable to find structures for this ligand".to_owned(),
                            ),
                        }
                    } else {
                        state.ui.popup.show_associated_structures = true;
                    }
                }
            }

            // ui.label("Rotate bonds:");
            // for i in 0..ligand.flexible_bonds.len() {
            //     if let ConformationType::Flexible { torsions } =
            //         &mut ligand.pose.conformation_type
            //     {
            //         if ui.button(format!("{i}")).clicked() {
            //             torsions[i].dihedral_angle =
            //                 (torsions[i].dihedral_angle + TAU / 64.) % TAU;
            //
            //             ligand.position_atoms(None);
            //
            //             redraw_mol = true;
            //         }
            //     }
            // }

            ui.add_space(COL_SPACING);

            // if let Some(energy) = &state.ui.binding_energy_disp {
            //     ui.label(format!("{:.2?}", energy)); // todo placeholder.
            // }

            if let Some(mol) = &state.molecule {
                let res_selected = match state.ui.selection {
                    Selection::Atom(sel_i) => {
                        let atom = &mol.common.atoms[sel_i];
                        if let Some(res_i) = &atom.residue {
                            Some(&mol.residues[*res_i])
                        } else {
                            None
                        }
                    }
                    Selection::Residue(sel_i) => Some(&mol.residues[sel_i]),
                    _ => None,
                };

                if let Some(res) = res_selected {
                    if ui
                        .button(
                            RichText::new(format!("Make lig from {}", res.res_type))
                                .color(Color32::GOLD),
                        )
                        .on_hover_text(
                            "Create a ligand from this residue on the peptide. This can be \
                    saved to a Mol2 or SDF file, and used as a ligand.",
                        )
                        .clicked()
                    {
                        let res_type = res.res_type.clone(); // Avoids dbl-borrow.

                        let mol_fm_res = MoleculeSmall::from_res(
                            res,
                            &mol.common.atoms,
                            &mol.common.bonds,
                            false,
                            // &state.ff_params.lig_specific,
                        );
                        // let mut lig_new = Ligand::new(mol_fm_res, &state.ff_params.lig_specific);
                        let lig_new = mol_fm_res;

                        state.mol_dynamics = None;

                        let docking_center = move_lig_to_res(&mut lig_new, mol, res);

                        // todo: Put this save back / fix dble-borrow?
                        // state.update_docking_site(docking_center);
                        // state.update_save_prefs(false);
                        // set_docking_light(scene, Some(&lig.docking_site));
                        // engine_updates.lighting = true;

                        *redraw_lig = true;

                        // If creating from an AA, move to the origin (Where we assigned its atom positions).
                        // If from a hetero atom, leave it in place.
                        match &res_type {
                            ResidueType::AminoAcid(_) => {
                                lig.reset_posits();
                            }
                            _ => {
                                state.ui.visibility.hide_hetero = true;
                            }
                        }
                        *lig = lig_new;

                        // Make it clear that we've added the ligand by showing it, and hiding hetero (if creating from Hetero)
                        state.ui.visibility.hide_ligand = false;
                    }
                }
            }
        });
    }

    // If no ligand, provide convenience functionality for loading one based on hetero residues
    // in the protein.
    if state.active_lig().is_some() {
        return;
    }

    let mut load_data = None; // Avoids dbl-borrow.

    if let Some(mol) = &mut state.molecule {
        let mut count_geostd_candidate = 0;
        for res in &mol.het_residues {
            if let ResidueType::Other(name) = &res.res_type {
                if name.len() == 3 {
                    count_geostd_candidate += 1;
                }
            }
        }

        if count_geostd_candidate > 0 {
            ui.horizontal(|ui| {
                ui.label("Load Amber Geostd lig from: ").on_hover_text(
                    "Attempt to load a ligand molecule and force field \
                            params from a hetero residue included in the protein file.",
                );

                for res in &mol.het_residues {
                    let name = match &res.res_type {
                        ResidueType::Other(name) => name,
                        _ => "hetero residue",
                    };
                    if name.len() == 3 {
                        if ui
                            .button(RichText::new(name).color(Color32::GOLD))
                            .clicked()
                        {
                            match amber_geostd::find_mols(&name) {
                                Ok(data) => match data.len() {
                                    0 => handle_err(
                                        &mut state.ui,
                                        "Unable to find an Amber molecule for this residue"
                                            .to_string(),
                                    ),
                                    1 => {
                                        load_data = Some(data[0].clone());
                                    }
                                    _ => {
                                        load_data = Some(data[0].clone());
                                        eprintln!("More than 1 geostd items available");
                                    }
                                },
                                Err(e) => handle_err(
                                    &mut state.ui,
                                    format!("Problem loading mol data online: {e:?}"),
                                ),
                            }
                        }
                    }
                }
            });
        }
    }

    // Avoids dbl-borrow
    if let Some(data) = load_data {
        handle_success(
            &mut state.ui,
            format!("Loaded {} from Amber Geostd", data.ident),
        );
        state.load_geostd_mol_data(&data.ident, true, data.frcmod_avail, redraw_lig);

        if let Some(lig) = state.active_lig_mut() {
            if let Some(mol) = &state.molecule {
                move_cam_to_lig(&mut state.ui, scene, lig, mol.center, engine_updates);
            }
        }
    }
}
