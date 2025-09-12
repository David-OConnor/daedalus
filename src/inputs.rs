//! Handles user inputs, e.g. from keyboard and mouse.

use graphics::{
    ControlScheme, DeviceEvent, ElementState, EngineUpdates, FWD_VEC, RIGHT_VEC, Scene, UP_VEC,
    WindowEvent,
    winit::keyboard::{KeyCode, PhysicalKey::Code},
};
use lin_alg::{
    f32::{Quaternion, Vec3},
    f64::{Quaternion as QuaternionF64, Vec3 as Vec3F64},
    map_linear,
};

use crate::{
    Selection, State, drawing,
    drawing::MoleculeView,
    molecule::Atom,
    render::set_flashlight,
    selection::{find_selected_atom, points_along_ray},
    util::{cycle_selected, orbit_center},
};
use crate::ui::COLOR_ACTIVE;

// These are defaults; overridden by the user A/R, and saved to prefs.
pub const MOVEMENT_SENS: f32 = 12.;
pub const ROTATE_SENS: f32 = 0.45;
pub const RUN_FACTOR: f32 = 6.; // i.e. shift key multiplier

pub const SCROLL_MOVE_AMT: f32 = 4.;
pub const SCROLL_ROTATE_AMT: f32 = 12.;

const SELECTION_DIST_THRESH_SMALL: f32 = 0.7; // e.g. ball + stick, or stick.
// Setting this high rel to `THRESH_SMALL` will cause more accidental selections of nearby atoms that
// the cursor is closer to the center of, but are behind the desired one.
// Setting it too low will cause the selector to "miss", even though the cursor is on an atom visual.
const SELECTION_DIST_THRESH_LARGE: f32 = 1.1; // e.g. VDW views like spheres.

const SEL_NEAR_PAD: f32 = 4.;

pub fn event_dev_handler(
    state_: &mut State,
    event: DeviceEvent,
    scene: &mut Scene,
    _engine_inputs: bool,
    dt: f32,
) -> EngineUpdates {
    let lig_move_amt = 0.05;
    let lig_rotate_amt = 2.;

    let mut updates = EngineUpdates::default();

    let mut redraw_protein = false;
    let mut redraw_lig = false;

    let mut lig_move_dir = None;
    let mut lig_rot_dir = None;

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
            #[cfg(target_os = "linux")]
            let (left_click, right_click) = (1, 3);
            #[cfg(not(target_os = "linux"))]
            let (left_click, right_click) = (0, 1);

            if button == left_click {
                state_.ui.left_click_down = match state {
                    ElementState::Pressed => true,
                    ElementState::Released => {
                        // Part of our move logic.
                        state_.volatile.drag_pivot0 = None;
                        false
                    },
                }
            }
            if button == right_click {
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
                                    MoleculeView::SpaceFill => SELECTION_DIST_THRESH_LARGE,
                                    _ => SELECTION_DIST_THRESH_SMALL,
                                };

                                let mut lig_atoms = Vec::new();
                                for lig in &state_.ligands {
                                    // todo: I don't like cloning all the atoms here. Not sure how else to do this for now.
                                    // todo not too many for ligs, but still.
                                    let atoms_this: Vec<_> = lig
                                        .common
                                        .atoms
                                        .iter()
                                        .enumerate()
                                        .map(|(i, a)| Atom {
                                            posit: lig.common.atom_posits[i],
                                            element: a.element,
                                            ..Default::default()
                                        })
                                        .collect();
                                    lig_atoms.push(atoms_this);
                                }

                                let (atoms_along_ray, atoms_along_ray_lig) = points_along_ray(
                                    selected_ray,
                                    &mol.common.atoms,
                                    &lig_atoms,
                                    dist_thresh,
                                );

                                let selection = find_selected_atom(
                                    &atoms_along_ray,
                                    &atoms_along_ray_lig,
                                    &mol.common.atoms,
                                    &mol.residues,
                                    &lig_atoms,
                                    &selected_ray,
                                    &state_.ui,
                                    &mol.chains,
                                );

                                // Change the active molecule to the one of the selected atom.
                                if let Selection::AtomLig((mol_i, _)) = selection {
                                    state_.volatile.active_lig = Some(mol_i);
                                }

                                if selection == state_.ui.selection {
                                    // Toggle.
                                    state_.ui.selection = Selection::None;
                                } else {
                                    state_.ui.selection = selection;
                                }

                                if let ControlScheme::Arc { center } =
                                    &mut scene.input_settings.control_scheme
                                {
                                    *center = orbit_center(state_);
                                }

                                redraw_protein = true;
                                redraw_lig = true;
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
                        cycle_selected(state_, scene, true);
                        if matches!(state_.ui.selection, Selection::Atom(_)) {
                            redraw_protein = true;
                        }
                        if matches!(state_.ui.selection, Selection::AtomLig(_)) {
                            redraw_lig = true;
                        }
                    }
                    Code(KeyCode::ArrowRight) => {
                        cycle_selected(state_, scene, false);
                        if matches!(state_.ui.selection, Selection::Atom(_)) {
                            redraw_protein = true;
                        }
                        if matches!(state_.ui.selection, Selection::AtomLig(_)) {
                            redraw_lig = true;
                        }
                    }
                    Code(KeyCode::Escape) => {
                        state_.ui.selection = Selection::None;
                        state_.volatile.move_mol = None;
                        state_.volatile.drag_pivot0 = None;
                        scene.input_settings.control_scheme = state_.volatile.control_scheme_prev;

                        redraw_protein = true;
                        redraw_lig = true;
                    }
                    // These lig rotations are temporary.
                    Code(KeyCode::KeyU) => {
                        lig_rot_dir = Some(FWD_VEC);
                    }
                    Code(KeyCode::KeyO) => {
                        lig_rot_dir = Some(-FWD_VEC);
                    }
                    Code(KeyCode::BracketLeft) => {
                        state_.ui.mol_view = state_.ui.mol_view.prev();
                        redraw_protein = true;
                        // lig_rot_dir = Some(-RIGHT_VEC);
                    }
                    Code(KeyCode::BracketRight) => {
                        state_.ui.mol_view = state_.ui.mol_view.next();
                        redraw_protein = true;
                        // lig_rot_dir = Some(RIGHT_VEC);
                    }
                    Code(KeyCode::Semicolon) => {
                        // lig_rot_dir = Some(UP_VEC);
                    }
                    Code(KeyCode::Quote) => {
                        // lig_rot_dir = Some(-UP_VEC);
                    }
                    Code(KeyCode::KeyI) => {
                        // lig_move_dir = Some(FWD_VEC);
                    }
                    // Ad-hoc ligand movement. For now, moves in discrete chunks, i.e. doens't respect key
                    // holding directly. Moves releative to the camera.
                    Code(KeyCode::KeyK) => {
                        lig_move_dir = Some(-FWD_VEC);
                    }
                    Code(KeyCode::KeyJ) => {
                        lig_move_dir = Some(-RIGHT_VEC);
                    }
                    Code(KeyCode::KeyL) => {
                        lig_move_dir = Some(RIGHT_VEC);
                    }
                    Code(KeyCode::Period) => {
                        lig_move_dir = Some(-UP_VEC);
                    }
                    Code(KeyCode::AltRight) => {
                        lig_move_dir = Some(UP_VEC);
                    }
                    Code(KeyCode::KeyM) => {
                        if let Some(i) = state_.volatile.active_lig {
                            let mut move_active = false;
                            if let Some(m) = state_.volatile.move_mol {
                                if m == i {
                                    move_active = true;
                                }
                                // DRY with the UI button.
                                state_.volatile.move_mol = if move_active {
                                    scene.input_settings.control_scheme = state_.volatile.control_scheme_prev;
                                    None
                                } else {
                                    state_.volatile.control_scheme_prev = scene.input_settings.control_scheme;
                                    scene.input_settings.control_scheme = ControlScheme::None;
                                    Some(i)
                                };
                            }
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
                if let Some(mol_i) = state_.volatile.move_mol {
                    let mol = &mut state_.ligands[mol_i];

                    if let Some(mut cursor) = state_.ui.cursor_pos {
                        // your existing UI offset fix
                        cursor.1 -= map_linear(
                            cursor.1,
                            (scene.window_size.1, state_.volatile.ui_height),
                            (0., state_.volatile.ui_height),
                        );

                        if state_.volatile.drag_pivot0.is_none() {
                            // let pivot0: Vec3 = mol.common.atom_posits[0].into();

                            // Use the centroid as the pivot.
                            let pivot0: Vec3F64 = {
                                let n = mol.common.atom_posits.len() as f64;
                                let sum = mol.common.atom_posits.iter().copied().fold(Vec3F64::new_zero(), |a, b| a + b);
                                sum / n
                            };

                            let n = scene.camera
                                .orientation
                                .rotate_vec(FWD_VEC)         // e.g. FWD_VEC = Vec3::new(0., 0., -1.)
                                .to_normalized().into();

                            state_.volatile.drag_pivot0 = Some(pivot0);
                            state_.volatile.drag_norm   = Some(n);
                            state_.volatile.drag_offset = Vec3F64::new_zero();
                        }

                        // Cached at drag start
                        let pivot0 = state_.volatile.drag_pivot0.unwrap();
                        let n= state_.volatile.drag_norm.unwrap();            // normalized
                        let prev_offset = state_.volatile.drag_offset;      // starts at ZERO

                        // Ray from screen
                        let (ray_origin, ray_point) = scene.screen_to_render(cursor);
                        let rd: Vec3F64 = (ray_point - ray_origin).to_normalized().into();

                        let ray_origin: Vec3F64 = ray_origin.into();

                        let denom = rd.dot(n);
                        if denom.abs() > 1e-6 {
                            // Fixed plane: n · (X - pivot0) = 0
                            let t = n.dot(pivot0 - ray_origin) / denom;
                            let hit = ray_origin + rd * t;

                            let offset = hit - pivot0;                 // desired total offset this frame
                            let delta = offset - prev_offset;           // move only by the change

                            // Apply delta (convert types if needed)
                            let movement_vec: Vec3F64 = delta.into();
                            for p in &mut mol.common.atom_posits {
                                *p += movement_vec;
                            }

                            state_.volatile.drag_offset = offset;       // remember for next frame

                            // todo experimenting
                            static mut I: u16 = 0;
                            unsafe {
                                I += 1;
                                // todo hacky. Get the mol drawing code faster.
                                if I % 40 == 0 {
                                    redraw_lig = true;
                                }
                            }
                        }
                    }
                }
            }

            if state_.ui.left_click_down {
                set_flashlight(scene);
                updates.lighting = true;
            }
        }
        _ => (),
    }

    if redraw_protein {
        // todo:This is overkill for certain keys. Just change the color of the one[s] in question, and set update.entities = true.
        drawing::draw_peptide(state_, scene);
        updates.entities = true;
    }

    // todo: Note that lig movements etc don't currently stack.
    if let Some(dir_) = lig_move_dir {
        if let Some(lig) = state_.active_lig_mut() {
            let dir = scene.camera.orientation.rotate_vec(dir_);
            let move_amt: Vec3F64 = (dir * lig_move_amt).into();

            if let Some(data) = &mut lig.lig_data {
                data.pose.anchor_posit += move_amt;
            }

            redraw_lig = true;
        }
    }

    if let Some(dir_) = lig_rot_dir {
        if let Some(lig) = state_.active_lig_mut() {
            let dir = scene.camera.orientation.rotate_vec(dir_);

            if let Some(data) = &mut lig.lig_data {
                let rotation: QuaternionF64 =
                    Quaternion::from_axis_angle(dir, lig_rotate_amt * dt).into();
                data.pose.orientation = rotation * data.pose.orientation;
            }

            redraw_lig = true;
        }
    }

    if redraw_lig {
        if let Some(lig) = &mut state_.active_lig_mut() {
            lig.position_atoms(None);
        }

        drawing::draw_all_ligs(state_, scene);
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
        WindowEvent::Focused(val) => {
            state.ui.mouse_in_window = val;
        }
        _ => (),
    }
    EngineUpdates::default()
}

#[allow(unused)]
/// Debug code to draw teh ray on screen, so we can see why the selection is off.
fn plot_ray() {
    // let center = (selected_ray.0 + selected_ray.1) / 2.;
    //
    // let diff = selected_ray.0 - selected_ray.1;
    // let diff_unit = diff.to_normalized();
    // let orientation = Quaternion::from_unit_vecs(UP_VEC, diff_unit);
    //
    // let scale = Some(Vec3::new(0.3, diff.magnitude(), 0.3));
    //
    // let mut ent = Entity::new(
    //     MESH_BOND,
    //     center,
    //     orientation,
    //     1.,
    //     (1., 0., 1.),
    //     BODY_SHINYNESS,
    // );
    // ent.scale_partial = scale;
    //
    // scene.entities.push(ent);
    // updates.entities = true;
}
