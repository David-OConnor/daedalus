use egui::{Color32, ComboBox, RichText, Slider, TextEdit, Ui};
use graphics::{Camera, ControlScheme, EngineUpdates, FWD_VEC, RIGHT_VEC, Scene, UP_VEC};
use lin_alg::f32::Vec3;

use crate::{
    Selection, State, cam_misc,
    cam_misc::{cam_look_at_outside, move_cam_to_sel, reset_camera},
    render::set_flashlight,
    ui::{
        COL_SPACING, COLOR_HIGHLIGHT, get_snap_name,
        misc::{self, section_box},
    },
    util,
    util::orbit_center,
};

// This control the clip planes in the camera frustum.
pub const RENDER_DIST_NEAR: f32 = 0.2;
pub const RENDER_DIST_FAR: f32 = 1_000.;

// These are Å multiplied by 10. Affects the user-setting near property.
// Near sets the camera frustum's near property.
pub const VIEW_DEPTH_NEAR_MIN: u16 = 2;
pub const VIEW_DEPTH_NEAR_MAX: u16 = 300;

// Distance between start and end of the fade. A smaller distance is a more aggressive fade.
pub const FOG_HALF_DEPTH: u16 = 40;

// The range to start fading distance objects, and when the fade is complete.
pub const FOG_DIST_DEFAULT: u16 = 70;

// Affects the user-setting far property.
// Sets the fog center point in its fade.
pub const FOG_DIST_MIN: u16 = 1;
pub const FOG_DIST_MAX: u16 = 120;

pub fn calc_fog_dists(dist: u16) -> (f32, f32) {
    // Clamp.
    let min = if dist > FOG_HALF_DEPTH {
        dist - FOG_HALF_DEPTH
    } else {
        0
    };

    (min as f32, (dist + FOG_HALF_DEPTH) as f32)
}

pub fn set_fog_dist(cam: &mut Camera, dist: u16) {
    let (fog_start, fog_end) = if dist == FOG_DIST_MAX {
        (0., 0.) // No fog will render.
    } else {
        let val = dist;
        calc_fog_dists(val)
    };

    cam.fog_start = fog_start;
    cam.fog_end = fog_end;
}

pub(in crate::ui) fn cam_reset_controls(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    engine_updates: &mut EngineUpdates,
    changed: &mut bool,
) {
    ui.label("Cam:");

    // Preset buttons
    if ui
        .button("Front")
        .on_hover_text("Reset the camera to look at the \"front\" of the molecule. (Y axis)")
        .clicked()
    {
        reset_camera(state, scene, engine_updates, FWD_VEC);
        *changed = true;
    }

    if ui
        .button("Top")
        .on_hover_text("Reset the camera to look at the \"top\" of the molecule. (Z axis)")
        .clicked()
    {
        reset_camera(state, scene, engine_updates, -UP_VEC);
        *changed = true;
    }

    if ui
        .button("Left")
        .on_hover_text("Reset the camera to look at the \"left\" of the molecule. (X axis)")
        .clicked()
    {
        reset_camera(state, scene, engine_updates, RIGHT_VEC);
        *changed = true;
    }
}

pub(in crate::ui) fn cam_controls(
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
                cam_reset_controls(state, scene,  ui, engine_updates, &mut changed);

                ui.add_space(COL_SPACING);

                let free_active = scene.input_settings.control_scheme == ControlScheme::FreeCamera;
                let arc_active = scene.input_settings.control_scheme != ControlScheme::FreeCamera;

                if ui
                    .button(RichText::new("Free").color(misc::active_color_sel(free_active)))
                    .on_hover_text("Set the camera is a first-person mode, where your controls move its position. Similar to video games.")
                    .clicked()
                {
                    scene.input_settings.control_scheme = ControlScheme::FreeCamera;
                    state.to_save.control_scheme = ControlScheme::FreeCamera;
                }

                if ui
                    .button(RichText::new("Arc").color(misc::active_color_sel(arc_active)))
                    .on_hover_text("Set the camera to orbit around a point: Either the center of the molecule, or the selection.")
                    .clicked()
                {
                    let center = match &state.peptide {
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
                                .color(misc::active_color(state.ui.orbit_selected_atom)),
                        )
                        .on_hover_text("Toggle whether the camera orbits around the selection, or the molecule center.")
                        .clicked()
                    {
                        state.ui.orbit_selected_atom = !state.ui.orbit_selected_atom;

                        let center = orbit_center(state);
                        scene.input_settings.control_scheme = ControlScheme::Arc { center };
                    }
                }

                ui.add_space(COL_SPACING);

                if state.ui.selection != Selection::None {
                    if ui
                        .button(RichText::new("Cam to sel").color(COLOR_HIGHLIGHT))
                        .on_hover_text("(Hotkey: Enter) Move camera near the selected atom or residue, looking at it.")
                        .clicked()
                    {
                        move_cam_to_sel(&mut state.ui, &state.peptide, &state.ligands, &state.nucleic_acids, &state.lipids, &mut scene.camera, engine_updates);
                    }
                }

                if state.volatile.active_mol.is_some() {
                    if ui
                        .button(RichText::new("Cam to mol").color(COLOR_HIGHLIGHT))
                        .on_hover_text("Move camera near active molecule, looking at it.")
                        .clicked()
                    {
                        let pep_center = match &state.peptide {
                            Some(mol) => mol.center,
                            None => lin_alg::f64::Vec3::new_zero(),
                        };
                        // Setting mol center to 0 if no mol.
                        move_cam_to_active_mol(state, scene, pep_center, engine_updates)
                    }
                }

                ui.add_space(COL_SPACING);

                // todo: Grey-out, instead of setting render dist. (e.g. fog)
                let depth_prev = state.ui.view_depth;
                ui.spacing_mut().slider_width = 60.;

                let hover_text = "Don't render objects closer to the camera than this distance, in Å.";
                ui.label("Depth. Near(×10):")
                    .on_hover_text(hover_text);

                ui.add(Slider::new(
                    &mut state.ui.view_depth.0,
                    VIEW_DEPTH_NEAR_MIN..=VIEW_DEPTH_NEAR_MAX,
                )).on_hover_text(hover_text);

                let hover_text = "(Hotkey: Ctrl + scroll) Fade distant objects. This may make it easier to see objects near the camera.";
                ui.label("Far:")
                    .on_hover_text(hover_text);

                ui.add(Slider::new(
                    &mut state.ui.view_depth.1,
                    FOG_DIST_MIN..=FOG_DIST_MAX,
                )).on_hover_text(hover_text)    ;

                if state.ui.view_depth != depth_prev {
                    // Interpret the slider being at min or max position to mean (effectively) unlimited.

                    scene.camera.near = if state.ui.view_depth.0 == VIEW_DEPTH_NEAR_MIN {
                        RENDER_DIST_NEAR
                    } else {
                        state.ui.view_depth.0 as f32 / 10.
                    };
                    // todo: Only if near changed.
                    scene.camera.update_proj_mat();

                    set_fog_dist(&mut scene.camera, state.ui.view_depth.1);

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

pub(in crate::ui) fn cam_snapshots(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    // todo: Wraping isn't working here.
    section_box().show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label("Scenes");

            ui.add(TextEdit::singleline(&mut state.ui.cam_snapshot_name).desired_width(60.))
                .on_hover_text("Choose a name to save this scene as.");

            if ui
                .button("Save")
                .on_hover_text("Save the current camera position and orientation to a scene.")
                .clicked()
            {
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
                })
                .response
                .on_hover_text("Set the camera to a previously-saved scene.");

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
    });
}

pub(in crate::ui) fn move_cam_to_active_mol(
    state: &mut State,
    scene: &mut Scene,
    look_to_beyond: lin_alg::f64::Vec3,
    engine_updates: &mut EngineUpdates,
) {
    let Some(mol) = &mut state.active_mol_mut() else {
        return;
    };

    // todo: Cache centroid.
    let mol_pos: Vec3 = mol.common().centroid().into();
    let ctr: Vec3 = look_to_beyond.into();

    cam_look_at_outside(
        &mut scene.camera,
        mol_pos,
        ctr,
        cam_misc::MOVE_CAM_TO_MOL_DIST,
    );

    engine_updates.camera = true;

    set_flashlight(scene);
    engine_updates.lighting = true;

    state.ui.cam_snapshot = None;
}

// /// DRY with above. Can be more amenable to the borrow checker in some cases.
// pub fn move_cam_to_lig2(
//     // mol: &MoleculeGenericRef,
//     // cam_snapshot: &mut Option<usize>,
//     state: &mut State,
//     scene: &mut Scene,
//     mol_center: lin_alg::f64::Vec3,
//     engine_updates: &mut EngineUpdates,
// ) {
//     let mol = &state.active_mol().unwrap();
//     let Some(mol) = &mut state.active_mol_mut() else {
//         return;
//     };
//
//     // todo: Cache centroid.
//     let lig_pos: Vec3 = mol.common().centroid().into();
//     let ctr: Vec3 = mol_center.into();
//
//     cam_look_at_outside(&mut scene.camera, lig_pos, ctr);
//
//     engine_updates.camera = true;
//
//     set_flashlight(scene);
//     engine_updates.lighting = true;
//
//     state.ui.cam_snapshot = None;
// }
