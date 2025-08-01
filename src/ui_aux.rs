//! Misc utility-related UI functionality.

use bio_files::ResidueType;
use egui::{Color32, ComboBox, RichText, Slider, TextEdit, Ui};
use graphics::{EngineUpdates, Scene};

use crate::{
    Selection, State,
    dynamics::{
        MdMode,
        prep::{
            build_dynamics_docking, build_dynamics_peptide, change_snapshot_docking,
            change_snapshot_peptide,
        },
    },
    mol_drawing,
    mol_drawing::{
        CHARGE_MAP_MAX, CHARGE_MAP_MIN, COLOR_AA_NON_RESIDUE_EGUI, draw_ligand, draw_molecule,
        draw_water,
    },
    molecule::{Atom, Ligand, Molecule, PeptideAtomPosits, Residue, aa_color},
    render::set_docking_light,
    ui::{
        COL_SPACING, COLOR_ACTIVE, COLOR_ACTIVE_RADIO, COLOR_HIGHLIGHT, COLOR_INACTIVE,
        ROW_SPACING, int_field,
    },
    util::{handle_err, make_egui_color, move_lig_to_res},
};

fn disp_atom_data(atom: &Atom, residues: &[Residue], ui: &mut Ui) {
    let role = match atom.role {
        Some(r) => format!("Role: {r}"),
        None => String::new(),
    };

    // Similar to `Vec3`'s format impl, but with fewer digits.
    let posit_txt = format!(
        "|{:.3}, {:.3}, {:.3}|",
        atom.posit.x, atom.posit.y, atom.posit.z
    );

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
        let color = make_egui_color(mol_drawing::color_viridis_float(
            *q,
            CHARGE_MAP_MIN,
            CHARGE_MAP_MAX,
        ));

        ui.label(RichText::new(format!("{plus}q: {q:.2}")).color(color));
    }
}

/// Display text of the selected atom
pub fn selected_data(mol: &Molecule, ligand: &Option<Ligand>, selection: &Selection, ui: &mut Ui) {
    match selection {
        Selection::Atom(sel_i) => {
            if *sel_i >= mol.atoms.len() {
                return;
            }

            let atom = &mol.atoms[*sel_i];
            disp_atom_data(atom, &mol.residues, ui);
        }
        Selection::AtomLigand(sel_i) => {
            let Some(lig) = ligand else {
                return;
            };
            if *sel_i >= lig.molecule.atoms.len() {
                return;
            }

            let atom = &lig.molecule.atoms[*sel_i];
            disp_atom_data(atom, &[], ui);
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
            ui.label(RichText::new(res.to_string()).color(res_color));
        }
        Selection::Atoms(is) => {
            // todo: A/R
            ui.label(RichText::new(format!("{} atoms", is.len())).color(Color32::GOLD));
        }
        Selection::None => (),
    }
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
    ui.horizontal(|ui| {
        let prev = state.ui.peptide_atom_posits;
        ui.label("Show atoms:");
        ComboBox::from_id_salt(3)
            .width(80.)
            .selected_text(state.ui.peptide_atom_posits.to_string())
            .show_ui(ui, |ui| {
                for view in &[PeptideAtomPosits::Original, PeptideAtomPosits::Dynamics] {
                    ui.selectable_value(&mut state.ui.peptide_atom_posits, *view, view.to_string());
                }
            });

        if state.ui.peptide_atom_posits != prev {
            draw_molecule(state, scene);
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
            }

            if state.ui.current_snapshot != snapshot_prev {
                changed = true;
                let snap = &md.snapshots[state.ui.current_snapshot];

                match md.mode {
                    MdMode::Docking => {
                        let lig = state.ligand.as_mut().unwrap();

                        change_snapshot_docking(
                            // &mut scene.entities,
                            lig,
                            // &Vec::new(),
                            snap,
                            &mut state.ui.binding_energy_disp,
                        );

                        draw_ligand(state, scene);
                    }
                    MdMode::Peptide => {
                        let mol = state.molecule.as_mut().unwrap();

                        change_snapshot_peptide(mol, &md.atoms, snap);
                        draw_molecule(state, scene);
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
    ui.horizontal(|ui| {
        // todo: Move A/R
        // Workaround for double-borrow.
        let mut run_clicked = false;

        run_clicked = ui
            .button(RichText::new("Run MD on peptide").color(Color32::GOLD))
            .clicked();
        if run_clicked {
            // If not already loaded from static string to state, do so now.
            // We load on demand to save computation.
            // state.load_ffs_general();
        }
        if run_clicked {
            let mol = state.molecule.as_mut().unwrap();

            match build_dynamics_peptide(
                &state.dev,
                mol,
                &state.ff_params,
                state.to_save.num_md_steps,
                state.to_save.md_dt,
            ) {
                Ok(md) => {
                    let snap = &md.snapshots[0];
                    change_snapshot_peptide(mol, &md.atoms, snap);
                    draw_molecule(state, scene);

                    draw_water(
                        scene,
                        &snap.water_o_posits,
                        &snap.water_h0_posits,
                        &snap.water_h1_posits,
                    );

                    state.mol_dynamics = Some(md);
                    state.ui.current_snapshot = 0;
                }
                Err(e) => handle_err(&mut state.ui, e.descrip),
            }
        }

        ui.add_space(COL_SPACING / 2.);

        let mut run_clicked = ui
            .button(RichText::new("Run MD docking").color(Color32::GOLD))
            .clicked();

        let mut ready_to_run = true;

        if run_clicked {
            let Some(lig) = state.ligand.as_mut() else {
                return;
            };

            if !lig.ff_params_loaded || !lig.frcmod_loaded {
                state.ui.show_get_geostd_popup = true;
                ready_to_run = false;
            }

            if ready_to_run {
                match build_dynamics_docking(
                    &state.dev,
                    lig,
                    state.volatile.docking_setup.as_ref().unwrap(),
                    &state.ff_params,
                    state.to_save.num_md_steps,
                    state.to_save.md_dt,
                ) {
                    Ok(md) => {
                        let snap = &md.snapshots[0];
                        change_snapshot_docking(lig, snap, &mut None);

                        draw_molecule(state, scene);
                        draw_water(
                            scene,
                            &snap.water_o_posits,
                            &snap.water_h0_posits,
                            &snap.water_h1_posits,
                        );

                        state.ui.current_snapshot = 0;
                        state.mol_dynamics = Some(md);
                    }
                    Err(e) => handle_err(&mut state.ui, e.descrip),
                }
            }
        }

        ui.add_space(COL_SPACING);
        int_field(&mut state.to_save.num_md_steps, "Steps:", &mut false, ui);

        ui.label("dt:");
        if ui
            .add_sized(
                [60., Ui::available_height(ui)],
                TextEdit::singleline(&mut state.ui.md_dt_input),
            )
            .changed()
        {
            if let Ok(v) = state.ui.md_dt_input.parse::<f64>() {
                state.to_save.md_dt = v;
            }
        }

        ui.add_space(COL_SPACING);

        if let Some(mol) = &state.molecule {
            let res_selected = match state.ui.selection {
                Selection::Atom(sel_i) => {
                    let atom = &mol.atoms[sel_i];
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
                            .color(COLOR_HIGHLIGHT),
                    )
                    .clicked()
                {
                    let mol_fm_res = Molecule::from_res(res, &mol.atoms, false);
                    let mut lig = Ligand::new(mol_fm_res, &state.ff_params.lig_specific);

                    let docking_center = move_lig_to_res(&mut lig, mol, res);
                    state.update_docking_site(docking_center);

                    state.update_save_prefs();
                    set_docking_light(scene, Some(&lig.docking_site));
                    engine_updates.lighting = true;
                    *redraw_lig = true;

                    state.ligand = Some(lig);

                    // Make it clear that we've added the ligand by showing it, and hiding hetero.
                    state.ui.visibility.hide_ligand = false;
                    state.ui.visibility.hide_hetero = true;
                }
            }
        }

        ui.add_space(COL_SPACING);
    });

    dynamics_player(state, scene, engine_updates, ui);
}
