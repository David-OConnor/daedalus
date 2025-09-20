//! Misc utility-related UI functionality.

use egui::{Color32, ComboBox, CornerRadius, Frame, Margin, RichText, Slider, Stroke, Ui};
use graphics::{EngineUpdates, Scene};
const COLOR_SECTION_BOX: Color32 = Color32::from_rgb(100, 100, 140);

use crate::{
    State,
    drawing::{draw_all_ligs, draw_peptide, draw_water},
    md::change_snapshot_form2,
    molecule::PeptideAtomPosits,
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

#[derive(Clone, Copy, PartialEq)]
pub enum MdMode {
    Docking,
    Peptide,
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
                    state.ui.current_snapshot as f32 * state.to_save.md_dt
                ));
            }

            if state.ui.current_snapshot != snapshot_prev {
                changed = true;
                let snap = &md.snapshots[state.ui.current_snapshot];

                change_snapshot_form2(&mut state.ligands, snap);
                // todo: Only if at least one lig is involved.
                draw_all_ligs(state, scene);

                if let Some(mol) = &state.molecule {
                    if mol.common.atoms.len() > 0 {
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

pub fn handle_docking(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    engine_updates: &mut EngineUpdates,
) {
    // // todo: Ideally move the camera to the docking site prior to docking. You could do this
    // // todo by deferring the docking below to the next frame.
    //
    // // let (pose, binding_energy) = find_optimal_pose(
    // //     &state.dev,
    // //     state.volatile.docking_setup.as_ref().unwrap(),
    // //     lig,
    // // );
    //
    // // lig_data.pose = pose;
    // {
    //     // lig.position_atoms(None);
    //     //
    //     // lig.position_atoms(None);
    //     // let lig_pos: Vec3 = lig.common.atom_posits[lig_data.anchor_atom].into();
    //     // let ctr: Vec3 = state.molecule.unwrap().center.into();
    //     //
    //     // cam_look_at_outside(&mut scene.camera, lig_pos, ctr);
    //     //
    //     // engine_updates.camera = true;
    //     // state.ui.cam_snapshot = None;
    // }
    //
    // // Allow the user to select the autodock executable.
    // // if state.to_save.autodock_vina_path.is_none() {
    // //     state.volatile.autodock_path_dialog.pick_file();
    // // }
    // // dock_with_vina(mol, ligand, &state.to_save.autodock_vina_path);
    // *redraw_lig = true;
    //
    // // if ui.button("Docking energy").clicked() {
    // //     let poses = vec![lig_data.pose.clone()];
    // //     let mut lig_posits: Vec<Vec3> = Vec::with_capacity(poses.len());
    // //     // let mut partial_charges_lig = Vec::with_capacity(poses.len());
    // //
    // //     for pose in poses {
    // //         lig.position_atoms(Some(&pose));
    // //
    // //         let posits_this_pose: Vec<_> =
    // //             lig.common.atom_posits.iter().map(|p| (*p).into()).collect();
    // //
    // //         // partial_charges_lig.push(create_partial_charges(
    // //         //     &ligand.molecule.atoms,
    // //         //     Some(&posits_this_pose),
    // //         // ));
    // //         lig_posits.push(posits_this_pose);
    // //     }
    // //
    // //     // state.ui.binding_energy_disp = calc_binding_energy(
    // //     //     state.volatile.docking_setup.as_ref().unwrap(),
    // //     //     lig,
    // //     //     &lig_posits[0],
    // //     // );
    // // }
    //
    // ui.add_space(COL_SPACING);
    //
    // // todo: Put back A/r.
    // // if ui.button("Site sfc").clicked() {
    // //     // let (mesh, edges) = find_docking_site_surface(mol, &ligand.docking_site);
    // //
    // //     // scene.meshes[MESH_DOCKING_SURFACE] = mesh;
    // //
    // //     // todo: You must remove prev entities of it too! Do you need an entity ID for this? Likely.
    // //     // todo: Move to the draw module A/R.
    // //     let mut entity = Entity::new(
    // //         MESH_DOCKING_SURFACE,
    // //         Vec3::new_zero(),
    // //         Quaternion::new_identity(),
    // //         1.,
    // //         COLOR_DOCKING_SITE_MESH,
    // //         0.5,
    // //     );
    // //     entity.opacity = 0.8;
    // //     entity.class = EntityType::DockingSite as u32;
    // //
    // //     scene.entities.push(entity);
    // //
    // //     engine_updates.meshes = true;
    // //     engine_updates.entities = true;
    // // }
    //
    // ui.add_space(COL_SPACING);
    //
    // // todo: Put back A/R
    // let mut docking_init_changed = false;
    // if false {
    //     ui.label("Docking site setup:");
    //     ui.label("Center:");
    //
    //     if ui
    //         .add(TextEdit::singleline(&mut state.ui.docking_site_x).desired_width(30.))
    //         .changed()
    //     {
    //         if let Ok(v) = state.ui.docking_site_x.parse::<f64>() {
    //             // lig_data.docking_site.site_center.x = v;
    //             docking_init_changed = true;
    //         }
    //     }
    //     if ui
    //         .add(TextEdit::singleline(&mut state.ui.docking_site_y).desired_width(30.))
    //         .changed()
    //     {
    //         if let Ok(v) = state.ui.docking_site_y.parse::<f64>() {
    //             // lig_data.docking_site.site_center.y = v;
    //             docking_init_changed = true;
    //         }
    //     }
    //     if ui
    //         .add(TextEdit::singleline(&mut state.ui.docking_site_z).desired_width(30.))
    //         .changed()
    //     {
    //         if let Ok(v) = state.ui.docking_site_z.parse::<f64>() {
    //             // lig_data.docking_site.site_center.z = v;
    //             docking_init_changed = true;
    //         }
    //     }
    //
    //     // todo: Consider a slider.
    //     ui.label("Size:");
    //     if ui
    //         .add(TextEdit::singleline(&mut state.ui.docking_site_size).desired_width(30.))
    //         .changed()
    //     {
    //         if let Ok(v) = state.ui.docking_site_size.parse::<f64>() {
    //             // lig_data.docking_site.site_radius = v;
    //             docking_init_changed = true;
    //         }
    //     }
    // }
    //
}
