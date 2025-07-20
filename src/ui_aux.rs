//! Misc utility-related UI functionality.

use bio_files::ResidueType;
use egui::{Color32, RichText, Slider, Ui};
use graphics::{EngineUpdates, Scene};
use na_seq::AaIdent;

use crate::{
    Selection, State,
    dynamics::prep::change_snapshot_md,
    mol_drawing,
    mol_drawing::{CHARGE_MAP_MAX, CHARGE_MAP_MIN, draw_ligand},
    molecule::{Atom, Ligand, Molecule, Residue},
    ui::{COLOR_ACTIVE, COLOR_ACTIVE_RADIO, COLOR_INACTIVE, ROW_SPACING},
};
use crate::mol_drawing::{COLOR_AA_NON_RESIDUE, COLOR_AA_NON_RESIDUE_EGUI};
use crate::molecule::aa_color;
use crate::util::make_egui_color;

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

    let mut text_c = String::new();
    // Placeholder for water etc.
    let mut res_color = COLOR_AA_NON_RESIDUE_EGUI;

    if let Some(res_i) = atom.residue {
        let res = &residues[res_i];
        text_c += &format!("  {res}");

        if let ResidueType::AminoAcid(aa) = res.res_type {
            res_color = make_egui_color(aa_color(aa));
        }
    }

    ui.label(RichText::new(text_0).color(Color32::WHITE));

    ui.label(RichText::new(posit_txt).color(Color32::GOLD));

    let atom_color = make_egui_color( atom.element.color());

    ui.label(RichText::new(text_b).color(atom_color));
    ui.label(RichText::new(text_c).color(res_color));
    ui.label(RichText::new(role).color(Color32::LIGHT_GRAY));

    if let Some(tir) = &atom.type_in_res {
        ui.label(RichText::new(format!("{tir}")).color(Color32::LIGHT_YELLOW));
    }

    if let Some(ff) = &atom.force_field_type {
        ui.label(RichText::new(format!("FF: {ff}")).color(Color32::LIGHT_YELLOW));
    }

    if let Some(q) = &atom.partial_charge {
        let plus = if *q > 0. { "+" } else { "" };
        let color = make_egui_color(mol_drawing::color_viridis_float(*q, CHARGE_MAP_MIN, CHARGE_MAP_MAX));

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
    if let Some(md) = &state.mol_dynamics {
        if !md.snapshots.is_empty() {
            // if !state.volatile.snapshots.is_empty() {
            ui.add_space(ROW_SPACING);

            let snapshot_prev = state.ui.current_snapshot;
            ui.spacing_mut().slider_width = ui.available_width() - 100.;
            ui.add(Slider::new(
                &mut state.ui.current_snapshot,
                // 0..=state.volatile.snapshots.len() - 1,
                0..=md.snapshots.len() - 1, // todo exper
            ));

            if state.ui.current_snapshot != snapshot_prev {
                // change_snapshot(
                //     &mut scene.entities,
                //     lig,
                //     &Vec::new(),
                //     &mut state.ui.binding_energy_disp,
                //     &state.volatile.snapshots[state.ui.current_snapshot],
                // );

                let lig = state.ligand.as_mut().unwrap();

                change_snapshot_md(
                    &mut scene.entities,
                    lig,
                    &Vec::new(),
                    &mut state.ui.binding_energy_disp,
                    &md.snapshots[state.ui.current_snapshot],
                );

                draw_ligand(state, scene);

                engine_updates.entities = true;
            }
        }
    }
}
