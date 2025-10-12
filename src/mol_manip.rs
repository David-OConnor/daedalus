//! Handles logic for moving and rotating molecules from user inputs.

use cudarc::driver::sys::cudaError_enum;
use graphics::{
    ControlScheme, EngineUpdates, FWD_VEC, RIGHT_VEC, Scene, UP_VEC, event::MouseScrollDelta,
};
use lin_alg::{
    f32::{Quaternion, Vec3},
    f64::Vec3 as Vec3F64,
    map_linear,
};

use crate::{
    ManipMode, State, StateVolatile,
    inputs::{SCROLL_MOVE_AMT, SENS_MOL_MOVE_SCROLL, SENS_MOL_ROT_MOUSE, SENS_MOL_ROT_SCROLL},
    molecule::MolType,
};

/// Blender-style mouse dragging of the molecule. For movement, creates a plane of the camera view,
/// at the molecules depth. The mouse cursor projects to this plane, moving the molecule
/// along it. (Movement). Handles rotation in a straightforward manner. This is for motion relative
/// to the 2D screen, e.g. from mouse movement.
pub fn handle_mol_manip_in_plane(
    state: &mut State,
    scene: &Scene,
    delta: (f64, f64),
    redraw_lig: &mut bool,
    redraw_na: &mut bool,
    redraw_lipid: &mut bool,
) {
    // We skip renders, as they are relatively slow. This produces choppy dragging,
    // but I don't have a better plan yet.
    static mut I: u16 = 0;

    let Some(mut cursor) = state.ui.cursor_pos else {
        return;
    };

    // your existing UI offset fix
    cursor.1 -= map_linear(
        cursor.1,
        (scene.window_size.1, state.volatile.ui_height),
        (0., state.volatile.ui_height),
    );

    match state.volatile.mol_manip.mol {
        ManipMode::Move((mol_type, mol_i)) => {
            let mol = match mol_type {
                MolType::Ligand => &mut state.ligands[mol_i].common,
                MolType::NucleicAcid => &mut state.nucleic_acids[mol_i].common,
                MolType::Lipid => &mut state.lipids[mol_i].common,
                _ => unimplemented!(),
            };

            if state.volatile.mol_manip.pivot.is_none() {
                let pivot: Vec3 = mol.centroid().into();

                let n = scene
                    .camera
                    .orientation
                    .rotate_vec(FWD_VEC)
                    .to_normalized()
                    .into();

                state.volatile.mol_manip.pivot = Some(pivot);
                state.volatile.mol_manip.view_dir = Some(n);
                state.volatile.mol_manip.offset = Vec3::new_zero();
            }

            // Cached at drag start
            let pivot = state.volatile.mol_manip.pivot.unwrap();
            let pivot_norm = state.volatile.mol_manip.view_dir.unwrap();
            let prev_offset = state.volatile.mol_manip.offset;

            // Ray from screen
            let (ray_origin, ray_point) = scene.screen_to_render(cursor);
            let ray_dir = (ray_point - ray_origin).to_normalized();

            let denom = ray_dir.dot(pivot_norm);
            if denom.abs() > 1e-6 {
                // Fixed plane: n · (X - pivot) = 0
                let t = pivot_norm.dot(pivot - ray_origin) / denom;

                let hit = ray_origin + ray_dir * t;

                let offset = hit - pivot;
                let delta_ = offset - prev_offset;

                // Apply delta (convert types if needed)
                let movement_vec: Vec3F64 = delta_.into();
                for p in &mut mol.atom_posits {
                    *p += movement_vec;
                }

                state.volatile.mol_manip.offset = offset;

                let ratio = 8;
                unsafe {
                    I += 1;
                    if I.is_multiple_of(ratio) {
                        match mol_type {
                            MolType::Ligand => *redraw_lig = true,
                            MolType::NucleicAcid => *redraw_na = true,
                            MolType::Lipid => *redraw_lipid = true,
                            _ => unimplemented!(),
                        };
                    }
                }
            }
        }
        ManipMode::Rotate((mol_type, mol_i)) => {
            let mol = match mol_type {
                MolType::Ligand => &mut state.ligands[mol_i].common,
                MolType::NucleicAcid => &mut state.nucleic_acids[mol_i].common,
                MolType::Lipid => &mut state.lipids[mol_i].common,
                _ => unimplemented!(),
            };

            // We handle rotation around the fwd/z axis using the scroll wheel.
            // let fwd = scene.camera.orientation.rotate_vec(FWD_VEC);
            let right = scene
                .camera
                .orientation
                .rotate_vec(RIGHT_VEC)
                .to_normalized();
            let up = scene.camera.orientation.rotate_vec(UP_VEC).to_normalized();

            let rot_x = Quaternion::from_axis_angle(right, -delta.1 as f32 * SENS_MOL_ROT_MOUSE);
            let rot_y = Quaternion::from_axis_angle(up, -delta.0 as f32 * SENS_MOL_ROT_MOUSE);

            let rot = rot_y * rot_x; // Note: Can swap the order for a slightly different effect.
            mol.rotate(rot.into(), None);

            let ratio = 8;
            unsafe {
                I += 1;
                if I.is_multiple_of(ratio) {
                    match mol_type {
                        MolType::Ligand => *redraw_lig = true,
                        MolType::NucleicAcid => *redraw_na = true,
                        MolType::Lipid => *redraw_lipid = true,
                        _ => unimplemented!(),
                    };
                }
            }
        }
        ManipMode::None => (),
    }
}

/// Forward and backwards, with 0 line of sight to the camera. E.g. with the scroll wheel.
pub fn handle_mol_manip_in_out(
    state: &mut State,
    scene: &mut Scene,
    delta: MouseScrollDelta,
    redraw_lig: &mut bool,
    redraw_na: &mut bool,
    redraw_lipid: &mut bool,
    // updates: &mut EngineUpdates,
) {
    let mut counter_movement = false;

    // Move the molecule forward and backwards relative to the camera on scroll.
    match state.volatile.mol_manip.mol {
        ManipMode::Move((mol_type, mol_i)) => {
            let mol = match mol_type {
                MolType::Ligand => &mut state.ligands[mol_i].common,
                MolType::NucleicAcid => &mut state.nucleic_acids[mol_i].common,
                MolType::Lipid => &mut state.lipids[mol_i].common,
                _ => unimplemented!(),
            };

            let scroll: f32 = match delta {
                MouseScrollDelta::LineDelta(_, y) => y,
                MouseScrollDelta::PixelDelta(p) => p.y as f32 / 120.0,
            };
            if scroll == 0.0 {
                return;
            }

            // todo: DRY.
            // Note: This mol manip state is relevant when scrolling with the mouse button down.
            // If not also supporting zooming in and out, we'd cache these values at the drag start.
            // If we do that, moving the mol and and out would be wonky mid-drag.
            {
                let pivot: Vec3 = mol.centroid().into();

                let cam_pos32: Vec3 = scene.camera.position.into();
                let view_dir = (pivot - cam_pos32).to_normalized();

                state.volatile.mol_manip.pivot = Some(pivot);
                state.volatile.mol_manip.view_dir = Some(view_dir);
                state.volatile.mol_manip.offset = Vec3::new_zero();
            }

            if let (Some(pivot), _) = (
                state.volatile.mol_manip.pivot,
                state.volatile.mol_manip.view_dir,
            ) {
                let cam_pos32: Vec3 = scene.camera.position.into();

                let view_dir = (pivot - cam_pos32).to_normalized();
                state.volatile.mol_manip.view_dir = Some(view_dir);

                let dist = (pivot - cam_pos32).magnitude();
                let step = state.to_save.mol_move_sens as f32 / 1_000. * dist;

                // let dv = pivot_norm * (scroll * step);
                let dv = view_dir * (scroll * step);

                {
                    let dv64: Vec3F64 = dv.into();
                    for p in &mut mol.atom_posits {
                        *p += dv64;
                    }
                }

                let new_pivot = pivot + dv;
                state.volatile.mol_manip.pivot = Some(new_pivot);
                state.volatile.mol_manip.depth_bias += scroll * step;

                // todo: QC if you need/want this.
                // recompute intersection on the shifted plane to maintain stable drag
                if let Some(mut cursor) = state.ui.cursor_pos {
                    cursor.1 -= map_linear(
                        cursor.1,
                        (scene.window_size.1, state.volatile.ui_height),
                        (0., state.volatile.ui_height),
                    );

                    let (ray_origin, ray_point) = scene.screen_to_render(cursor);
                    let rd = (ray_point - ray_origin).to_normalized();

                    let denom = rd.dot(view_dir);
                    if denom.abs() > 1e-6 {
                        let t = view_dir.dot(new_pivot - ray_origin) / denom;
                        if t > 0.0 {
                            let hit = ray_origin + rd * t;
                            state.volatile.mol_manip.offset = hit - new_pivot;
                        }
                    }
                }
            }
            match mol_type {
                MolType::Ligand => *redraw_lig = true,
                MolType::NucleicAcid => *redraw_na = true,
                MolType::Lipid => *redraw_lipid = true,
                _ => unimplemented!(),
            };
            counter_movement = true;
        }
        ManipMode::Rotate((mol_type, mol_i)) => {
            let scroll: f32 = match delta {
                MouseScrollDelta::LineDelta(_, y) => y,
                MouseScrollDelta::PixelDelta(p) => p.y as f32 / 120.0,
            };
            if scroll == 0.0 {
                return;
            }

            // todo: C+P with slight changes from the mouse-move variant.
            let mol = match mol_type {
                MolType::Ligand => &mut state.ligands[mol_i].common,
                MolType::NucleicAcid => &mut state.nucleic_acids[mol_i].common,
                MolType::Lipid => &mut state.lipids[mol_i].common,
                _ => unimplemented!(),
            };

            let fwd = scene.camera.orientation.rotate_vec(FWD_VEC).to_normalized();

            let rot = Quaternion::from_axis_angle(fwd, scroll * SENS_MOL_ROT_SCROLL);
            mol.rotate(rot.into(), None);

            match mol_type {
                MolType::Ligand => *redraw_lig = true,
                MolType::NucleicAcid => *redraw_na = true,
                MolType::Lipid => *redraw_lipid = true,
                _ => unimplemented!(),
            };
            counter_movement = true;
        }
        ManipMode::None => (),
    }
}

/// Sets the manipulation mode, and adjusts camera controls A/R. Called from inputs, or the UI.
pub fn set_manip(
    vol: &mut StateVolatile,
    scene: &mut Scene,
    redraw_lig: &mut bool,
    redraw_na: &mut bool,
    redraw_lipid: &mut bool,
    mode: ManipMode,
) {
    if let Some((mol_type_active, i_active)) = vol.active_mol {
        let mut move_active = false;
        let mut rotate_active = false;

        match vol.mol_manip.mol {
            ManipMode::None => (),
            ManipMode::Move((mol_type, mol_i)) => {
                if mol_type == mol_type_active && mol_i == i_active {
                    move_active = true;
                }
            }
            ManipMode::Rotate((mol_type, mol_i)) => {
                if mol_type == mol_type_active && mol_i == i_active {
                    rotate_active = true;
                }
            }
        }

        match mode {
            ManipMode::Move(_) => {
                if move_active {
                    scene.input_settings.control_scheme = vol.control_scheme_prev;
                    vol.mol_manip.mol = ManipMode::None;
                    vol.mol_manip.pivot = None;
                } else if rotate_active {
                    vol.mol_manip.mol = ManipMode::Move((mol_type_active, i_active));
                } else {
                    if scene.input_settings.control_scheme != ControlScheme::None {
                        vol.control_scheme_prev = scene.input_settings.control_scheme;
                    }
                    scene.input_settings.control_scheme = ControlScheme::None;
                    vol.mol_manip.mol = ManipMode::Move((mol_type_active, i_active));
                };
            }
            ManipMode::Rotate(_) => {
                if rotate_active {
                    scene.input_settings.control_scheme = vol.control_scheme_prev;
                    vol.mol_manip.mol = ManipMode::None;
                    vol.mol_manip.pivot = None;
                } else if move_active {
                    vol.mol_manip.mol = ManipMode::Rotate((mol_type_active, i_active));
                } else {
                    if scene.input_settings.control_scheme != ControlScheme::None {
                        vol.control_scheme_prev = scene.input_settings.control_scheme;
                    }
                    scene.input_settings.control_scheme = ControlScheme::None;
                    vol.mol_manip.mol = ManipMode::Rotate((mol_type_active, i_active));
                };
            }
            ManipMode::None => unreachable!(),
        }

        match mol_type_active {
            MolType::Ligand => *redraw_lig = true,
            MolType::NucleicAcid => *redraw_na = true,
            MolType::Lipid => *redraw_lipid = true,
            _ => (),
        }
    }
}
