//! Handles user inputs, e.g. from keyboard and mouse.

use graphics::{
    ControlScheme, DeviceEvent, ElementState, EngineUpdates, FWD_VEC, Scene, WindowEvent,
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
    render::set_flashlight,
    util::{cycle_res_selected, find_selected_atom, orbit_center, points_along_ray},
};

pub const MOVEMENT_SENS: f32 = 12.;
pub const RUN_FACTOR: f32 = 6.; // i.e. shift key multiplier

pub const SCROLL_MOVE_AMT: f32 = 4.;
pub const SCROLL_ROTATE_AMT: f32 = 12.;

const SELECTION_DIST_THRESH_SMALL: f32 = 0.7; // e.g. ball + stick
const SELECTION_DIST_THRESH_LARGE: f32 = 1.3; // e.g. VDW views.

const SEL_NEAR_PAD: f32 = 4.;

// todo: Consider moving the selection code from util to here.

pub fn event_dev_handler(
    state_: &mut State,
    event: DeviceEvent,
    scene: &mut Scene,
    _engine_inputs: bool,
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
        DeviceEvent::MouseWheel { delta: _ } => {
            set_flashlight(scene);
            updates.lighting = true;
        }
        DeviceEvent::Button { button, state } => {
            // Workaround for EGUI's built-in way of doing this being broken
            // todo: This workaround isn't working due to inputs being disabled if mouse is in the GUI.
            // if button == 0  {
            //     // todo: Use input settings from below
            //     println!("IP C: {:?}", &state_.ui.inputs_commanded);
            //     adjust_camera(&mut scene.camera, &state_.ui.inputs_commanded, &InputSettings::default(), dt);
            // }

            if button == 0 {
                state_.ui.left_click_down = match state {
                    ElementState::Pressed => true,
                    ElementState::Released => false,
                }
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

                                if let ControlScheme::Arc { center } =
                                    &mut scene.input_settings.control_scheme
                                {
                                    *center = orbit_center(state_);
                                }

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
        DeviceEvent::Key(key) => {
            match key.state {
                ElementState::Pressed => match key.physical_key {
                    Code(KeyCode::ArrowLeft) => {
                        cycle_res_selected(state_, scene, true);
                        redraw = true;
                    }
                    Code(KeyCode::ArrowRight) => {
                        cycle_res_selected(state_, scene, false);
                        redraw = true;
                    }
                    // todo: Temp to test Ligand rotation
                    Code(KeyCode::BracketLeft) => {
                        if let Some(lig) = &mut state_.ligand {
                            // let rotation: QuaternionF64 =
                            //     Quaternion::from_axis_angle(FWD_VEC, -10. * dt).into();
                            // lig.orientation = rotation * lig.orientation;

                            // to clear entries; fine for this hack.
                            mol_drawing::draw_molecule(state_, scene, false);
                            mol_drawing::draw_ligand(state_, scene, false);
                            updates.entities = true;
                        }
                    }
                    // todo: Temp to test Ligand rotation
                    Code(KeyCode::BracketRight) => {
                        if let Some(lig) = &mut state_.ligand {
                            // let rotation: QuaternionF64 =
                            //     Quaternion::from_axis_angle(FWD_VEC, 10. * dt).into();
                            // lig.orientation = rotation * lig.orientation;

                            // to clear entries; fine for this hack.
                            mol_drawing::draw_molecule(state_, scene, false);
                            mol_drawing::draw_ligand(state_, scene, false);
                            updates.entities = true;
                        }
                    }
                    _ => (),
                },
                ElementState::Released => (),
            }

            // todo: If you enable a direction-dependent flashlight, you will need to modify the mouse movement
            // todo state too.
            // Similar code to the engine. Update state so we can update the flashlight while a key is held,
            // and reset snapshots.
            if key.state == ElementState::Pressed {
                match key.physical_key {
                    // Check the cases for the engine's built-in movement commands, to set the current-snapshot to None.
                    // C+P partially, from `graphics`. These are for press and release.
                    // todo:  You need to check mouse movement too.
                    Code(KeyCode::KeyW) => {
                        state_.volatile.inputs_commanded.fwd = true;
                    }
                    Code(KeyCode::KeyS) => {
                        state_.volatile.inputs_commanded.back = true;
                    }
                    Code(KeyCode::KeyA) => {
                        state_.volatile.inputs_commanded.left = true;
                    }
                    Code(KeyCode::KeyD) => {
                        state_.volatile.inputs_commanded.right = true;
                    }
                    Code(KeyCode::Space) => {
                        state_.volatile.inputs_commanded.up = true;
                    }
                    Code(KeyCode::KeyC) => {
                        state_.volatile.inputs_commanded.down = true;
                    }
                    Code(KeyCode::KeyQ) => {
                        state_.volatile.inputs_commanded.roll_ccw = true;
                    }
                    Code(KeyCode::KeyE) => {
                        state_.volatile.inputs_commanded.roll_cw = true;
                    }
                    _ => (),
                }
            } else if key.state == ElementState::Released {
                match key.physical_key {
                    Code(KeyCode::KeyW) => {
                        state_.volatile.inputs_commanded.fwd = false;
                    }
                    Code(KeyCode::KeyS) => {
                        state_.volatile.inputs_commanded.back = false;
                    }
                    Code(KeyCode::KeyA) => {
                        state_.volatile.inputs_commanded.left = false;
                    }
                    Code(KeyCode::KeyD) => {
                        state_.volatile.inputs_commanded.right = false;
                    }
                    Code(KeyCode::Space) => {
                        state_.volatile.inputs_commanded.up = false;
                    }
                    Code(KeyCode::KeyC) => {
                        state_.volatile.inputs_commanded.down = false;
                    }
                    Code(KeyCode::KeyQ) => {
                        state_.volatile.inputs_commanded.roll_ccw = false;
                    }
                    Code(KeyCode::KeyE) => {
                        state_.volatile.inputs_commanded.roll_cw = false;
                    }
                    _ => (),
                }
            }
        }
        DeviceEvent::MouseMotion { delta } => {
            // Free look handled by the engine; handle middle-click-move here.
            if state_.ui.middle_click_down {
                // The same movement sensitivity scaler we use for the (1x effective multiplier)
                // on keyboard movement seems to work well enough here.
                let mut movement_vec = Vec3::new(
                    delta.0 as f32 * MOVEMENT_SENS * dt,
                    -delta.1 as f32 * MOVEMENT_SENS * dt,
                    0.,
                );

                if scene.input_settings.control_scheme != ControlScheme::FreeCamera {
                    movement_vec *= -1.;
                }

                scene.camera.position += scene.camera.orientation.rotate_vec(movement_vec);
                updates.camera = true;

                set_flashlight(scene);
                updates.lighting = true;
            }

            if state_.ui.left_click_down {
                set_flashlight(scene);
                updates.lighting = true;
            }
        }
        _ => (),
    }

    if redraw {
        // todo:This is overkill for certain keys. Just change the color of the one[s] in question, and set update.entities = true.
        mol_drawing::draw_molecule(state_, scene, false);
        mol_drawing::draw_ligand(state_, scene, false);
        updates.lighting = true; // Ligand docking light. // todo: Not always necessary.
        updates.entities = true;
    }

    // We handle the flashlight elsewhere, as this event handler only fires upon events; not while
    // a key is held.
    if state_.volatile.inputs_commanded.inputs_present() {
        state_.ui.cam_snapshot = None;
    }

    updates
}

pub fn event_win_handler(
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
