/// Camera controls and related.
use std::f32::consts::TAU;

use egui::{
    Color32, ComboBox, CornerRadius, Frame, Margin, RichText, Rounding, Slider, Stroke, TextEdit,
    Ui,
};
use graphics::{ControlScheme, EngineUpdates, RIGHT_VEC, Scene, UP_VEC};
use lin_alg::f32::{Quaternion, Vec3};

use crate::{
    Selection, State, StateUi,
    molecule::Ligand,
    render::{CAM_INIT_OFFSET, RENDER_DIST_FAR, RENDER_DIST_NEAR, set_flashlight},
    ui::{
        COL_SPACING, COLOR_HIGHLIGHT, VIEW_DEPTH_FAR_MAX, VIEW_DEPTH_FAR_MIN, VIEW_DEPTH_NEAR_MAX,
        VIEW_DEPTH_NEAR_MIN, aux, aux::section_box, get_snap_name,
    },
    util,
    util::{cam_look_at, cam_look_at_outside, handle_err, orbit_center, reset_camera},
};

pub fn cam_controls(
    scene: &mut Scene,
    state: &mut State,
    engine_updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    // todo: Here and at init, set the camera dist dynamically based on mol size.
    // todo: Set the position not relative to 0, but  relative to the center of the atoms.

    let mut changed = false;

    // let cam = &mut scene.camera;

    // This frame allows for a border to visually section this off.

    section_box()
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Cam:");

                // Preset buttons
                if ui.button("Front")
                    .on_hover_text("Reset the camera to look at the \"front\" of the molecule. (Y axis)")
                    .clicked() {
                    if let Some(mol) = &state.molecule {
                        reset_camera(scene, &mut state.ui.view_depth, engine_updates, mol);
                        changed = true;
                    }
                }

                if ui.button("Top")
                    .on_hover_text("Reset the camera to look at the \"top\" of the molecule. (Z axis)")
                    .clicked() {
                    if let Some(mol) = &state.molecule {
                        let center: Vec3 = mol.center.into();
                        reset_camera(scene, &mut state.ui.view_depth, engine_updates, mol);
                        scene.camera.position =
                            Vec3::new(center.x, center.y + (mol.size + CAM_INIT_OFFSET), center.z);
                        scene.camera.orientation = Quaternion::from_axis_angle(RIGHT_VEC, TAU / 4.);

                        changed = true;
                    }
                }

                if ui.button("Left")
                    .on_hover_text("Reset the camera to look at the \"left\" of the molecule. (X axis)")
                    .clicked() {
                    if let Some(mol) = &state.molecule {
                        let center: Vec3 = mol.center.into();
                        reset_camera(scene, &mut state.ui.view_depth, engine_updates, mol);
                        scene.camera.position =
                            Vec3::new(center.x - (mol.size + CAM_INIT_OFFSET), center.y, center.z);
                        scene.camera.orientation = Quaternion::from_axis_angle(UP_VEC, TAU / 4.);

                        changed = true;
                    }
                }

                ui.add_space(COL_SPACING);

                let free_active = scene.input_settings.control_scheme == ControlScheme::FreeCamera;
                let arc_active = scene.input_settings.control_scheme != ControlScheme::FreeCamera;

                if ui
                    .button(RichText::new("Free").color(aux::active_color_sel(free_active)))
                    .on_hover_text("Set the camera is a first-person mode, where your controls move its position. Similar to video games.")
                    .clicked()
                {
                    scene.input_settings.control_scheme = ControlScheme::FreeCamera;
                    state.to_save.control_scheme = ControlScheme::FreeCamera;
                }

                if ui
                    .button(RichText::new("Arc").color(aux::active_color_sel(arc_active)))
                    .on_hover_text("Set the camera to orbit around a point: Either the center of the molecule, or the selection.")
                    .clicked()
                {
                    let center = match &state.molecule {
                        Some(mol) => mol.center.into(),
                        None => Vec3::new_zero(),
                    };
                    scene.input_settings.control_scheme = ControlScheme::Arc { center };
                    state.to_save.control_scheme = ControlScheme::Arc { center };
                }

                if arc_active {
                    if ui
                        .button(
                            RichText::new("Orbit sel")
                                .color(aux::active_color(state.ui.orbit_around_selection)),
                        )
                        .on_hover_text("Toggle whether the camera orbits around the selection, or the molecule center.")
                        .clicked()
                    {
                        state.ui.orbit_around_selection = !state.ui.orbit_around_selection;

                        let center = orbit_center(state);
                        scene.input_settings.control_scheme = ControlScheme::Arc { center };
                    }
                }

                if let Some(mol) = &state.molecule {
                    ui.add_space(COL_SPACING);

                    if state.ui.selection != Selection::None {
                        if ui
                            .button(RichText::new("Cam to sel").color(COLOR_HIGHLIGHT))
                            .on_hover_text("Move camera near the selected atom or residue, looking at it.")
                            .clicked()
                        {
                            if let Selection::AtomLigand(i) = &state.ui.selection {
                                if let Some(lig) = &state.ligand {
                                    cam_look_at(&mut scene.camera, lig.atom_posits[*i]);
                                    engine_updates.camera = true;
                                    state.ui.cam_snapshot = None;
                                }
                            } else {
                                let atom_sel = mol.get_sel_atom(&state.ui.selection);

                                if let Some(atom) = atom_sel {
                                    cam_look_at(&mut scene.camera, atom.posit);
                                    engine_updates.camera = true;
                                    state.ui.cam_snapshot = None;
                                }
                            }
                        }
                    }

                    if let Some(lig) = &mut state.ligand {
                        if ui
                            .button(RichText::new("Cam to lig").color(COLOR_HIGHLIGHT))
                            .on_hover_text("Move camera near the ligand, looking at it.")
                            .clicked()
                        {
                            move_cam_to_lig(&mut state.ui, scene, lig, mol.center, engine_updates)
                        }
                    }
                }

                ui.add_space(COL_SPACING);

                // todo: Grey-out, instead of setting render dist. (e.g. fog)
                let depth_prev = state.ui.view_depth;
                ui.spacing_mut().slider_width = 60.;

                ui.label("Depth. Near(x10):");
                ui.add(Slider::new(
                    &mut state.ui.view_depth.0,
                    VIEW_DEPTH_NEAR_MIN..=VIEW_DEPTH_NEAR_MAX,
                ));

                ui.label("Far:");
                ui.add(Slider::new(
                    &mut state.ui.view_depth.1,
                    VIEW_DEPTH_FAR_MIN..=VIEW_DEPTH_FAR_MAX,
                ));

                if state.ui.view_depth != depth_prev {
                    // Interpret the slider being at min or max position to mean (effectively) unlimited.
                    scene.camera.near = if state.ui.view_depth.0 == VIEW_DEPTH_NEAR_MIN {
                        RENDER_DIST_NEAR
                    } else {
                        state.ui.view_depth.0 as f32 / 10.
                    };

                    scene.camera.far = if state.ui.view_depth.1 == VIEW_DEPTH_FAR_MAX {
                        RENDER_DIST_FAR
                    } else {
                        state.ui.view_depth.1 as f32
                    };

                    scene.camera.update_proj_mat();
                    changed = true;
                }
            });
        });

    if changed {
        engine_updates.camera = true;

        set_flashlight(scene);
        engine_updates.lighting = true; // flashlight.

        state.ui.cam_snapshot = None;
    }
}

pub fn cam_snapshots(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    section_box().show(ui, |ui| {
        ui.label("Scenes");
        // ui.label("Label:");
        ui.add(TextEdit::singleline(&mut state.ui.cam_snapshot_name).desired_width(60.));

        if ui.button("Save").clicked() {
            let name = if !state.ui.cam_snapshot_name.is_empty() {
                state.ui.cam_snapshot_name.clone()
            } else {
                format!("Scene {}", state.cam_snapshots.len() + 1)
            };

            util::save_snap(state, &scene.camera, &name);
        }

        let prev_snap = state.ui.cam_snapshot;
        let snap_name = get_snap_name(prev_snap, &state.cam_snapshots);

        ComboBox::from_id_salt(2)
            .width(80.)
            .selected_text(snap_name)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut state.ui.cam_snapshot, None, "(None)");
                for (i, _snap) in state.cam_snapshots.iter().enumerate() {
                    ui.selectable_value(
                        &mut state.ui.cam_snapshot,
                        Some(i),
                        get_snap_name(Some(i), &state.cam_snapshots),
                    );
                }
            });

        if let Some(i) = state.ui.cam_snapshot {
            if ui.button(RichText::new("❌").color(Color32::RED)).clicked() {
                if i < state.cam_snapshots.len() {
                    state.cam_snapshots.remove(i);
                }
                state.ui.cam_snapshot = None;
                state.update_save_prefs(false);
            }
        }

        if state.ui.cam_snapshot != prev_snap {
            util::load_snap(state, scene, engine_updates);
        }
    });
}

pub fn move_cam_to_lig(
    state_ui: &mut StateUi,
    scene: &mut Scene,
    lig: &mut Ligand,
    mol_center: lin_alg::f64::Vec3,
    engine_updates: &mut EngineUpdates,
) {
    if lig.anchor_atom >= lig.molecule.atoms.len() {
        handle_err(
            state_ui,
            "Problem positioning ligand atoms. Len shorter than anchor.".to_owned(),
        );
    } else {
        lig.position_atoms(None);

        let lig_pos: lin_alg::f32::Vec3 = lig.atom_posits[lig.anchor_atom].into();
        let ctr: lin_alg::f32::Vec3 = mol_center.into();

        cam_look_at_outside(&mut scene.camera, lig_pos, ctr);

        engine_updates.camera = true;
        state_ui.cam_snapshot = None;
    }
}
