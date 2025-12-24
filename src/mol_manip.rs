//! Handles logic for moving and rotating molecules from user inputs.
//! This is for both primary mode, and the mol editor. In the latter, it
//! can move individual atoms, and rotate parts of molecules around bonds.

use graphics::{Camera, ControlScheme, FWD_VEC, RIGHT_VEC, Scene, UP_VEC, event::MouseScrollDelta};
use lin_alg::{
    f32::{Quaternion, Vec3},
    f64::Vec3 as Vec3F64,
    map_linear,
};
use na_seq::Element;

use crate::{
    OperatingMode, Selection, State, StateVolatile,
    inputs::{SENS_MOL_ROT_MOUSE, SENS_MOL_ROT_SCROLL},
    molecule::{MolType, MoleculeCommon},
};
use crate::mol_editor::rotate_around_bond;

/// Blender-style mouse dragging of the molecule. For movement, creates a plane of the camera view,
/// at the molecule's depth. The mouse cursor projects to this plane, moving the molecule
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
    if state.volatile.mol_manip.mode == ManipMode::None {
        return;
    }

    // We skip renders, as they are relatively slow. This produces choppy dragging,
    // but I don't have a better plan yet.
    static mut I: u16 = 0;

    let Some(mut cursor) = state.ui.cursor_pos else {
        return;
    };

    // Offset for the UI, as we use elsewhere.
    cursor.1 -= map_linear(
        cursor.1,
        (scene.window_size.1, state.volatile.ui_height),
        (0., state.volatile.ui_height),
    );

    match state.volatile.mol_manip.mode {
        ManipMode::Move((mol_type, mol_i)) => {
            let mol = match state.volatile.operating_mode {
                OperatingMode::Primary => match mol_type {
                    MolType::Ligand => &mut state.ligands[mol_i].common,
                    MolType::NucleicAcid => &mut state.nucleic_acids[mol_i].common,
                    MolType::Lipid => &mut state.lipids[mol_i].common,
                    _ => unimplemented!(),
                },
                OperatingMode::MolEditor => &mut state.mol_editor.mol.common,
            };

            // Ray from screen
            let (ray_origin, ray_point) = scene.screen_to_render(cursor);
            let ray_dir = (ray_point - ray_origin).to_normalized();

            state.volatile.mol_manip.enter_movement(
                &scene.camera,
                mol,
                ray_origin,
                ray_dir,
                state.volatile.operating_mode,
                mol_i,
            );

            // Cached at drag start
            let pivot = state.volatile.mol_manip.pivot.unwrap();
            let pivot_norm = state.volatile.mol_manip.view_dir.unwrap();
            let prev_offset = state.volatile.mol_manip.offset;

            let denom = ray_dir.dot(pivot_norm);
            if denom.abs() > 1e-6 {
                // Fixed plane: n · (X - pivot) = 0
                let t = pivot_norm.dot(pivot - ray_origin) / denom;

                let hit = ray_origin + ray_dir * t;

                let offset = hit - pivot;
                let delta_ = offset - prev_offset;

                // Apply delta (convert types if needed)
                let movement_vec: Vec3F64 = delta_.into();

                match state.volatile.operating_mode {
                    OperatingMode::Primary => {
                        for p in &mut mol.atom_posits {
                            *p += movement_vec;
                        }
                    }
                    OperatingMode::MolEditor => {
                        // `mol_i` = atom_i here.
                        mol.atom_posits[mol_i] += movement_vec;
                        mol.atoms[mol_i].posit = mol.atom_posits[mol_i];

                        // Move all hydrogens bonded to the atom too.
                        for i in &mol.adjacency_list[mol_i] {
                            if mol.atoms[*i].element == Element::Hydrogen {
                                mol.atom_posits[*i] += movement_vec;
                            }
                        }
                    }
                }

                state.volatile.mol_manip.offset = offset;

                let ratio = 8;
                unsafe {
                    I += 1;
                    if I.is_multiple_of(ratio)
                        || state.volatile.operating_mode == OperatingMode::MolEditor
                    {
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
            // todo: DRY with above
            let mol = match state.volatile.operating_mode {
                OperatingMode::Primary => match mol_type {
                    MolType::Ligand => &mut state.ligands[mol_i].common,
                    MolType::NucleicAcid => &mut state.nucleic_acids[mol_i].common,
                    MolType::Lipid => &mut state.lipids[mol_i].common,
                    _ => unimplemented!(),
                },
                OperatingMode::MolEditor => &mut state.mol_editor.mol.common,
            };

            match state.volatile.operating_mode {
                OperatingMode::Primary => {

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
                }
                OperatingMode::MolEditor => {
                    // todo: X vs Y?

                    const ROT_FACTOR: f64 = 0.008;
                    rotate_around_bond(mol, mol_i, ROT_FACTOR * delta.0);
                }
            }

            let ratio = 8;
            unsafe {
                I += 1;
                if I.is_multiple_of(ratio) || state.volatile.operating_mode == OperatingMode::MolEditor {
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
) {
    // Move the molecule forward and backwards relative to the camera on scroll.
    match state.volatile.mol_manip.mode {
        ManipMode::Move((mol_type, mol_i)) => {
            let mol = match state.volatile.operating_mode {
                OperatingMode::Primary => match mol_type {
                    MolType::Ligand => {
                        if mol_i >= state.ligands.len() {
                            println!("Error: Index out of bounds on ligand for mol manip");
                            return;
                        }
                        &mut state.ligands[mol_i].common
                    },
                    MolType::NucleicAcid => &mut state.nucleic_acids[mol_i].common,
                    MolType::Lipid => &mut state.lipids[mol_i].common,
                    _ => unimplemented!(),
                },
                OperatingMode::MolEditor => &mut state.mol_editor.mol.common,
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
                let pivot: Vec3 = match state.volatile.operating_mode {
                    OperatingMode::Primary => mol.centroid().into(),
                    OperatingMode::MolEditor => mol.atom_posits[mol_i].into(), // actually atom i.
                };

                let cam_pos32: Vec3 = scene.camera.position.into();
                let view_dir = (pivot - cam_pos32).to_normalized();

                state.volatile.mol_manip.pivot = Some(pivot);
                state.volatile.mol_manip.view_dir = Some(view_dir);
                state.volatile.mol_manip.offset = Vec3::new_zero();
            }
            // state
            //     .volatile
            //     .mol_manip
            //     .enter_movement(&scene.camera, mol, ray_origin, ray_dir);

            if let (Some(pivot), _) = (
                state.volatile.mol_manip.pivot,
                state.volatile.mol_manip.view_dir,
            ) {
                let view_dir = (pivot - scene.camera.position).to_normalized();
                state.volatile.mol_manip.view_dir = Some(view_dir);

                let dist = (pivot - scene.camera.position).magnitude();
                let step = state.to_save.mol_move_sens as f32 / 1_000. * dist;

                let dv = view_dir * (scroll * step);
                let movement_vec: Vec3F64 = dv.into();

                match state.volatile.operating_mode {
                    OperatingMode::Primary => {
                        for p in &mut mol.atom_posits {
                            *p += movement_vec;
                        }
                    }
                    OperatingMode::MolEditor => {
                        mol.atom_posits[mol_i] += movement_vec;
                        mol.atoms[mol_i].posit = mol.atom_posits[mol_i];

                        // Move all hydrogens bonded to the atom too.
                        for i in &mol.adjacency_list[mol_i] {
                            if mol.atoms[*i].element == Element::Hydrogen {
                                mol.atom_posits[*i] += movement_vec;
                            }
                        }
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
                MolType::Ligand => {
                    if mol_i >= state.ligands.len() {
                        println!("Error: Index out of bounds on ligand for mol manip");
                        return;
                    }

                    &mut state.ligands[mol_i].common
                },
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
        }
        ManipMode::None => (),
    }
}

/// Sets the manipulation mode, and adjusts camera controls A/R. Called from inputs, or the UI.
pub fn set_manip(
    vol: &mut StateVolatile,
    save_flag: &mut bool,
    scene: &mut Scene,
    redraw_lig: &mut bool,
    redraw_na: &mut bool,
    redraw_lipid: &mut bool,
    rebuild_md_editor: &mut bool,
    // Note: The mol itself is overwritten but this sets move/rotate,
    mode: ManipMode,
    sel: &Selection,
) {
    let (mol_type_active, i_active) = match vol.operating_mode {
        OperatingMode::Primary => match vol.active_mol {
            Some(v) => v,
            None => return,
        },
        // In the editor mode, select the selected atom as the one to move.
        OperatingMode::MolEditor => match sel {
            Selection::AtomLig((_, i)) => (MolType::Ligand, *i),
            Selection::AtomsLig((_, i)) => {
                // todo: How should we handle this?
                (MolType::Ligand, i[0])
            }
            // For rotating.
            Selection::BondLig((_, i)) => (MolType::Ligand, *i),
            _ => return,
        },
    };

    let (move_active, rotate_active) = {
        let mut move_ = false;
        let mut rotate = false;

        match vol.mol_manip.mode {
            ManipMode::None => (),
            ManipMode::Move((mol_type, mol_i)) => {
                if mol_type == mol_type_active && mol_i == i_active {
                    move_ = true;
                }
            }
            ManipMode::Rotate((mol_type, mol_i)) => {
                if mol_type == mol_type_active && mol_i == i_active {
                    rotate = true;
                }
            }
        }
        (move_, rotate)
    };

    match mode {
        ManipMode::Move(_) => {
            if move_active {
                // Exiting a move
                scene.input_settings.control_scheme = vol.control_scheme_prev;
                vol.mol_manip.mode = ManipMode::None;
                vol.mol_manip.pivot = None;

                if vol.operating_mode == OperatingMode::MolEditor {
                    *rebuild_md_editor = true;
                }
            } else if rotate_active {
                // Entering a move from rotation
                vol.mol_manip.mode = ManipMode::Move((mol_type_active, i_active));
            } else {
                // Entering a move from no manip prior.
                if scene.input_settings.control_scheme != ControlScheme::None {
                    vol.control_scheme_prev = scene.input_settings.control_scheme;
                }
                scene.input_settings.control_scheme = ControlScheme::None;
                vol.mol_manip.mode = ManipMode::Move((mol_type_active, i_active));
            };
        }
        ManipMode::Rotate(_) => {
            if rotate_active {
                scene.input_settings.control_scheme = vol.control_scheme_prev;
                vol.mol_manip.mode = ManipMode::None;
                vol.mol_manip.pivot = None;

                if vol.operating_mode == OperatingMode::MolEditor {
                    *rebuild_md_editor = true;
                }
            } else if move_active {
                vol.mol_manip.mode = ManipMode::Rotate((mol_type_active, i_active));
            } else {
                if scene.input_settings.control_scheme != ControlScheme::None {
                    vol.control_scheme_prev = scene.input_settings.control_scheme;
                }
                scene.input_settings.control_scheme = ControlScheme::None;
                vol.mol_manip.mode = ManipMode::Rotate((mol_type_active, i_active));
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

/// Chooses the atom closest to the cursor ray as the pivot.
/// This helps ensure, while drag-moving, that the molecule stays anchored to the cursor in
/// a way that feels intuitive; the part moving to the cursor's position is the part originally
/// under the cursor at drag start.
fn pick_movement_pivot(mol: &MoleculeCommon, ray_origin: Vec3, ray_dir: Vec3) -> Vec3 {
    const PICK_RADIUS: f64 = 4.0; // Å-ish // A/R
    let pick_r2 = PICK_RADIUS * PICK_RADIUS;

    let ro: Vec3F64 = ray_origin.into();
    let rd: Vec3F64 = ray_dir.into();

    let mut best_d2 = f64::INFINITY;
    let mut best_p: Option<Vec3F64> = None;

    for p in &mol.atom_posits {
        let w = *p - ro;
        let t = w.dot(rd); // rd assumed normalized
        if t < 0.0 {
            continue;
        }
        let closest = ro + rd * t;
        let d2 = (*p - closest).magnitude_squared();
        if d2 < best_d2 {
            best_d2 = d2;
            best_p = Some(*p);
        }
    }

    if best_d2 <= pick_r2 {
        best_p.unwrap().into()
    } else {
        mol.centroid().into()
    }
}

#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub enum ManipMode {
    #[default]
    None,
    Move((MolType, usize)), // Index of mol
    Rotate((MolType, usize)),
}

/// State for dragging and rotating molecules.
#[derive(Default, Debug)]
pub struct MolManip {
    /// Allows the user to move a molecule around with mouse or keyboard.
    pub mode: ManipMode,
    /// For maintaining the screen plane when dragging the mol.
    pub pivot: Option<Vec3>,
    pub view_dir: Option<Vec3>,
    pub offset: Vec3,
    pub depth_bias: f32,
}

impl MolManip {
    pub fn enter_movement(
        &mut self,
        cam: &Camera,
        mol: &MoleculeCommon,
        ray_origin: Vec3,
        ray_dir: Vec3,
        mode: OperatingMode,
        atom_i: usize, // For edit mode, of the atom being moved.
    ) {
        if self.pivot.is_none() {
            // We set the pivot to be the coordinates of the molecule (e.g. nearest atom)
            // to the cursor. We're moving this pivot with the mouse cursor, so we take
            // this approach to prevent the mol jumping; this part *snaps* to the cursor.
            let pivot: Vec3 = match mode {
                OperatingMode::Primary => pick_movement_pivot(mol, ray_origin, ray_dir),
                OperatingMode::MolEditor => mol.atom_posits[atom_i].into(),
            };

            let n = cam.orientation.rotate_vec(FWD_VEC).to_normalized();

            self.pivot = Some(pivot);
            self.view_dir = Some(n);
            self.offset = Vec3::new_zero();
        }
    }
}
