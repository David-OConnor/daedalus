//! Misc utility-related UI functionality.

use egui::{Color32, CornerRadius, Frame, Margin, RichText, Slider, Stroke, Ui};
use graphics::{EngineUpdates, Scene};
const COLOR_SECTION_BOX: Color32 = Color32::from_rgb(100, 100, 140);

use crate::{
    State,
    drawing::{draw_all_ligs, draw_all_lipids, draw_peptide, draw_water},
    md::change_snapshot,
    ui::{COLOR_ACTIVE, COLOR_ACTIVE_RADIO, COLOR_INACTIVE, ROW_SPACING},
};

/// A checkbox to show or hide a category.
pub fn vis_check(val: &mut bool, text: &str, ui: &mut Ui, redraw: &mut bool) {
    let color = active_color(!*val);
    if ui.button(RichText::new(text).color(color)).clicked() {
        *val = !*val;
        *redraw = true;
    }
}

// #[derive(Clone, Copy, PartialEq)]
// pub enum MdMode {
//     Docking,
//     Peptide,
// }

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
        // let prev = state.ui.peptide_atom_posits;

        let help_text = "Toggle between viewing the original (pre-dynamics) atom positions, and \
        ones at the selected dynamics snapshot.";
        ui.label("Show atoms:").on_hover_text(help_text);
        // ComboBox::from_id_salt(3)
        //     .width(80.)
        //     .selected_text(state.ui.peptide_atom_posits.to_string())
        //     .show_ui(ui, |ui| {
        //         for view in &[PeptideAtomPosits::Original, PeptideAtomPosits::Dynamics] {
        //             ui.selectable_value(&mut state.ui.peptide_atom_posits, *view, view.to_string());
        //         }
        //     })
        //     .response
        //     .on_hover_text(help_text);

        // if state.ui.peptide_atom_posits != prev {
        //     draw_peptide(state, scene);
        //     engine_updates.entities = true;
        // }

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
                    state.ui.current_snapshot as f32 * state.to_save.md_dt
                ));
            }

            if state.ui.current_snapshot != snapshot_prev {
                changed = true;
                let snap = &md.snapshots[state.ui.current_snapshot];

                // todo note: This will break if you change selected ligs prior to re-reunning docking.
                let ligs_md: Vec<_> = state
                    .ligands
                    .iter_mut()
                    .filter(|l| l.common.selected_for_md)
                    .collect();
                let ligs_len = ligs_md.len();

                let lipids_md: Vec<_> = state
                    .lipids
                    .iter_mut()
                    .filter(|l| l.common.selected_for_md)
                    .collect();
                let lipids_len = lipids_md.len();

                let peptide_md = match &mut state.peptide {
                    Some(m) => {
                        if m.common.selected_for_md {
                            Some(m)
                        } else {
                            None
                        }
                    }
                    None => None,
                };

                change_snapshot(peptide_md, ligs_md, lipids_md, snap);
                // todo: Only if at least one lig is involved.
                if ligs_len > 0 {
                    draw_all_ligs(state, scene);
                }

                if lipids_len > 0 {
                    draw_all_lipids(state, scene);
                }

                if let Some(mol) = &state.peptide {
                    // if mol.common.atoms.len() > 0 {
                    if mol.common.selected_for_md {
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

// A container that highlights a section of UI code, to make it visually distinct from neighboring areas.
pub fn section_box() -> Frame {
    Frame::new()
        .stroke(Stroke::new(1.0, COLOR_SECTION_BOX))
        .corner_radius(CornerRadius::same(6))
        .inner_margin(Margin::symmetric(8, 2))
        .outer_margin(Margin::symmetric(0, 0))
}
