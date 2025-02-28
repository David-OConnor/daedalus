//! This module integraties this application with the graphics engine.

use std::f32::consts::TAU;

use graphics::{
    Camera, ControlScheme, DeviceEvent, ElementState, EngineUpdates, FWD_VEC, InputSettings,
    LightType, Lighting, Mesh, PointLight, RIGHT_VEC, Scene, UiLayout, UiSettings, WindowEvent,
    event::MouseScrollDelta,
    winit::keyboard::{KeyCode, PhysicalKey::Code},
};
use lin_alg::{
    f32::{Quaternion, Vec3},
    f64::Quaternion as QuaternionF64,
    map_linear,
};

use crate::{
    State, mol_drawing,
    mol_drawing::MoleculeView,
    ui::ui_handler,
    util::{cycle_res_selected, find_selected_atom, points_along_ray},
};

pub type Color = (f32, f32, f32);

const WINDOW_TITLE: &str = "Bio Chem View";
const WINDOW_SIZE_X: f32 = 1_400.;
const WINDOW_SIZE_Y: f32 = 1_000.;
const BACKGROUND_COLOR: Color = (0., 0., 0.);
pub const RENDER_DIST: f32 = 1_000.;

// todo: Shinyness broken?
pub const ATOM_SHINYNESS: f32 = 12.;
pub const BODY_SHINYNESS: f32 = 12.;

// Keep this in sync with mesh init.
pub const MESH_SPHERE: usize = 0;
pub const MESH_CUBE: usize = 1;
pub const MESH_BOND: usize = 2;
pub const MESH_SPHERE_LOWRES: usize = 3;
pub const MESH_SURFACE: usize = 4; // Van Der Waals surface.

const SELECTION_DIST_THRESH_SMALL: f32 = 0.7; // e.g. ball + stick
const SELECTION_DIST_THRESH_LARGE: f32 = 1.3; // e.g. VDW views.

pub const BALL_STICK_RADIUS: f32 = 0.3;
pub const BALL_STICK_RADIUS_H: f32 = 0.2;

// todo: By bond type etc
// const BOND_COLOR: Color = (0.2, 0.2, 0.2);
pub const BOND_RADIUS: f32 = 0.12;
// const BOND_CAP_RADIUS: f32 = 1./BOND_RADIUS;
pub const BOND_RADIUS_DOUBLE: f32 = 0.07;

pub const RADIUS_SFC_DOT: f32 = 0.05;
pub const COLOR_SFC_DOT: Color = (0.7, 0.7, 0.7);

pub const COLOR_SELECTED: Color = (1., 0., 0.);
pub const COLOR_H_BOND: Color = (0.2, 0.2, 1.);
pub const RADIUS_H_BOND: f32 = 0.2; // A scaler relative to covalent sticks.

pub const SHELL_OPACITY: f32 = 0.01;

// From the farthest molecule.
pub const CAM_INIT_OFFSET: f32 = 10.;
pub const OUTSIDE_LIGHTING_OFFSET: f32 = 600.;

pub const COLOR_AA_NON_RESIDUE: Color = (0., 0.8, 1.0);

const MOVEMENT_SENS: f32 = 12.;
const RUN_FACTOR: f32 = 6.; // i.e. shift key multiplier
const SCROLL_MOVE_AMT: f32 = 4.;

const SEL_NEAR_PAD: f32 = 4.;

/// Set lighting based on the center and size of the molecule.
pub fn set_lighting(center: Vec3, size: f32) -> Lighting {
    let white = [1., 1., 1., 1.];

    Lighting {
        ambient_color: white,
        ambient_intensity: 0.12,
        point_lights: vec![
            // PointLight {
            //     type_: LightType::Omnidirectional,
            //     position: center + Vec3::new(40., size + OUTSIDE_LIGHTING_OFFSET, 0.),
            //     diffuse_color: white,
            //     specular_color: white,
            //     diffuse_intensity: 6_000.,
            //     specular_intensity: 60_000.,
            // },
            PointLight {
                type_: LightType::Omnidirectional,
                position: center + Vec3::new(40., -size - OUTSIDE_LIGHTING_OFFSET, 0.),
                diffuse_color: white,
                specular_color: white,
                diffuse_intensity: 20_000.,
                specular_intensity: 70_000.,
            },
        ],
    }
}

fn event_dev_handler(
    state_: &mut State,
    event: DeviceEvent,
    scene: &mut Scene,
    dt: f32,
) -> EngineUpdates {
    let mut updates = EngineUpdates::default();

    let mut redraw = false;

    // todo: Move this logic to the engine (graphics lib)?
    if !state_.ui.mouse_in_window {
        return updates;
    }

    match event {
        // Move the camera forward and back on scroll.
        DeviceEvent::MouseWheel { delta } => match delta {
            MouseScrollDelta::PixelDelta(_) => (),
            MouseScrollDelta::LineDelta(_x, y) => {
                let mut movement_vec = Vec3::new(0., 0., SCROLL_MOVE_AMT);
                if y < 0. {
                    movement_vec *= -1.;
                }

                scene.camera.position += scene.camera.orientation.rotate_vec(movement_vec);
                updates.camera = true;
            }
        },
        DeviceEvent::Button { button, state } => {
            // Workaround for EGUI's built-in way of doing this being broken
            // todo: This workaround isn't working due to inputs being disabled if mouse is in the GUI.
            // if button == 0  {
            //     // todo: Use input settings from below
            //     println!("IP C: {:?}", &state_.ui.inputs_commanded);
            //     adjust_camera(&mut scene.camera, &state_.ui.inputs_commanded, &InputSettings::default(), dt);
            // }

            if button == 0 {
                // See note about camera movement resetting the snapshot. This impliles click + drag;
                // we should probalby only do this when mouse movement is present too.
                state_.ui.cam_snapshot = None;
            }
            if button == 1 {
                // Right click
                match state {
                    ElementState::Pressed => {
                        if let Some(mut cursor) = state_.ui.cursor_pos {
                            // Due to a quirk of some combination of our graphics engine and the egui
                            // integration lib in it, we need this vertical offset for the UI; otherwise,
                            // the higher up we click, the more the projected ray will be below the one
                            // indicated by the cursor. (Rays will only be accurate if clicked at the bottom of the screen).
                            // todo: It may be worth addressing upstream.
                            cursor.1 -= map_linear(
                                cursor.1,
                                (scene.window_size.1, state_.volatile.ui_height),
                                (0., state_.volatile.ui_height),
                            );

                            let mut selected_ray = scene.screen_to_render(cursor);

                            // Clip the near end of this to prevent false selections that seem to the user
                            // to be behind the camera.
                            let diff = selected_ray.1 - selected_ray.0;

                            selected_ray.0 += diff.to_normalized() * SEL_NEAR_PAD;

                            if let Some(mol) = &state_.molecule {
                                // If we don't scale the selection distance appropriately, an atom etc
                                // behind the desired one, but closer to the ray, may be selected; likely
                                // this is undesired.
                                let dist_thresh = match state_.ui.mol_view {
                                    MoleculeView::Mesh
                                    | MoleculeView::Dots
                                    | MoleculeView::SpaceFill => SELECTION_DIST_THRESH_LARGE,
                                    _ => SELECTION_DIST_THRESH_SMALL,
                                };
                                let atoms_along_ray =
                                    points_along_ray(selected_ray, &mol.atoms, dist_thresh);

                                state_.selection = find_selected_atom(
                                    &atoms_along_ray,
                                    &mol.atoms,
                                    &mol.residues,
                                    &selected_ray,
                                    &state_.ui,
                                    &mol.chains,
                                );

                                // todo: Debug code to draw teh ray on screen, so we can see why the selection is off.
                                // {
                                //     let center = (selected_ray.0 + selected_ray.1) / 2.;
                                //
                                //     let diff = selected_ray.0 - selected_ray.1;
                                //     let diff_unit = diff.to_normalized();
                                //     let orientation = Quaternion::from_unit_vecs(UP_VEC, diff_unit);
                                //
                                //     let scale = Some(Vec3::new(0.3, diff.magnitude(), 0.3));
                                //
                                //     let mut ent = Entity::new(
                                //         MESH_BOND,
                                //         center,
                                //         orientation,
                                //         1.,
                                //         (1., 0., 1.),
                                //         BODY_SHINYNESS,
                                //     );
                                //     ent.scale_partial = scale;
                                //
                                //     scene.entities.push(ent);
                                // updates.entities = true;
                                // }
                                redraw = true;
                            }
                        }
                    }
                    ElementState::Released => (),
                }
            }
            if button == 2 {
                // Allow mouse movement to move the camera on middle click.
                state_.ui.middle_click_down = match state {
                    ElementState::Pressed => true,
                    ElementState::Released => false,
                }
            }
        }
        DeviceEvent::Key(key) => match key.state {
            ElementState::Pressed => match key.physical_key {
                Code(KeyCode::ArrowLeft) => {
                    cycle_res_selected(state_, true);
                    redraw = true;
                }
                Code(KeyCode::ArrowRight) => {
                    cycle_res_selected(state_, false);
                    redraw = true;
                }
                // Check the cases for the engine's built-in movement commands, to set the current-snapshot to None.
                // C+P partially, from `graphics`.
                // todo:  You need to check mouse movement too.
                Code(KeyCode::KeyW) => {
                    state_.ui.cam_snapshot = None;
                }
                Code(KeyCode::KeyS) => {
                    state_.ui.cam_snapshot = None;
                }
                Code(KeyCode::KeyA) => {
                    state_.ui.cam_snapshot = None;
                }
                Code(KeyCode::KeyD) => {
                    state_.ui.cam_snapshot = None;
                }
                Code(KeyCode::Space) => {
                    state_.ui.cam_snapshot = None;
                }
                Code(KeyCode::KeyC) => {
                    state_.ui.cam_snapshot = None;
                }
                Code(KeyCode::KeyQ) => {
                    state_.ui.cam_snapshot = None;
                }
                Code(KeyCode::KeyE) => {
                    state_.ui.cam_snapshot = None;
                }
                // todo: Temp to test Ligand rotation
                Code(KeyCode::BracketLeft) => {
                    if let Some(lig) = &mut state_.ligand {
                        let rotation: QuaternionF64 =
                            Quaternion::from_axis_angle(FWD_VEC, -10. * dt).into();
                        lig.orientation = rotation * lig.orientation;

                        // to clear entries; fine for this hack.
                        mol_drawing::draw_molecule(state_, scene, false);
                        mol_drawing::draw_ligand(state_, scene, false);
                        updates.entities = true;
                    }
                }
                // todo: Temp to test Ligand rotation
                Code(KeyCode::BracketRight) => {
                    if let Some(lig) = &mut state_.ligand {
                        let rotation: QuaternionF64 =
                            Quaternion::from_axis_angle(FWD_VEC, 10. * dt).into();
                        lig.orientation = rotation * lig.orientation;

                        // to clear entries; fine for this hack.
                        mol_drawing::draw_molecule(state_, scene, false);
                        mol_drawing::draw_ligand(state_, scene, false);
                        updates.entities = true;
                    }
                }
                _ => (),
            },
            ElementState::Released => (),
        },
        DeviceEvent::MouseMotion { delta } => {
            // Free look handled by the engine; handle middle-click-move here.
            if state_.ui.middle_click_down {
                // The same movement sensitivity scaler we use for the (1x effective multiplier)
                // on keyboard movement seems to work well enough here.
                let movement_vec = Vec3::new(
                    delta.0 as f32 * MOVEMENT_SENS * dt,
                    -delta.1 as f32 * MOVEMENT_SENS * dt,
                    0.,
                );

                scene.camera.position += scene.camera.orientation.rotate_vec(movement_vec);
                updates.camera = true;
            }
        }
        _ => (),
    }

    if redraw {
        // todo:This is overkill for certain keys. Just change the color of the one[s] in question, and set update.entities = true.
        mol_drawing::draw_molecule(state_, scene, false);
        mol_drawing::draw_ligand(state_, scene, false);
        updates.entities = true;
    }

    updates
}

fn event_win_handler(
    state: &mut State,
    event: WindowEvent,
    _scene: &mut Scene,
    _dt: f32,
) -> EngineUpdates {
    match event {
        WindowEvent::CursorMoved {
            device_id: _,
            position,
        } => {
            // state.ui.cursor_pos = Some((position.x as f32, position.y as f32 + state.ui.ui_height))
            state.ui.cursor_pos = Some((position.x as f32, position.y as f32))
        }
        WindowEvent::CursorEntered { device_id: _ } => {
            state.ui.mouse_in_window = true;
        }
        WindowEvent::CursorLeft { device_id: _ } => {
            state.ui.mouse_in_window = false;
        }
        WindowEvent::Resized(_) => {
            state.ui.mouse_in_window = true;
        }
        _ => (),
    }
    EngineUpdates::default() // todo: A/R.
}

/// This runs each frame. Currently, no updates.
fn render_handler(_state: &mut State, _scene: &mut Scene, _dt: f32) -> EngineUpdates {
    EngineUpdates::default()
}

/// Entry point to our render and event loop.
pub fn render(mut state: State) {
    let mut scene = Scene {
        meshes: vec![
            Mesh::new_sphere(1., 14, 28),
            // Mesh::from_obj_file("sphere.obj"),
            Mesh::new_box(1., 1., 1.),
            Mesh::new_cylinder(1., BOND_RADIUS, 20),
            Mesh::new_sphere(1., 8, 8), // low-res sphere
            Mesh::new_box(1., 1., 1.),  // Placeholder for a VDW surface; populated later.
        ],
        entities: Vec::new(),
        camera: Camera {
            fov_y: TAU / 8.,
            position: Vec3::new(0., 0., -60.),
            far: RENDER_DIST,
            near: 0.2, // todo: Adjust A/R
            // orientation: Quaternion::from_axis_angle(Vec3::new(1., 0., 0.), TAU / 16.),
            orientation: Quaternion::from_axis_angle(RIGHT_VEC, 0.),
            ..Default::default()
        },
        // Lighting is set when drawing molecules; placeholder here.
        lighting: Default::default(),
        background_color: BACKGROUND_COLOR,
        window_size: (WINDOW_SIZE_X, WINDOW_SIZE_Y),
        window_title: WINDOW_TITLE.to_owned(),
    };

    let input_settings = InputSettings {
        initial_controls: ControlScheme::FreeCamera,
        // initial_controls: ControlScheme::Arc{ center: Vec3::new_zero()},
        move_sens: MOVEMENT_SENS,
        run_factor: RUN_FACTOR,
        ..Default::default()
    };
    let ui_settings = UiSettings {
        layout: UiLayout::Top,
        icon_path: Some("resources/icon.png".to_owned()),
    };

    mol_drawing::draw_molecule(&mut state, &mut scene, true);
    mol_drawing::draw_ligand(&mut state, &mut scene, true);

    graphics::run(
        state,
        scene,
        input_settings,
        ui_settings,
        Default::default(),
        render_handler,
        event_dev_handler,
        event_win_handler,
        ui_handler,
    );
}
