//! Handles user inputs, e.g. from keyboard and mouse.

use graphics::{
    ControlScheme, DeviceEvent, ElementState, EngineUpdates, EntityUpdate, FWD_VEC, RIGHT_VEC,
    Scene, UP_VEC, WindowEvent,
    winit::keyboard::{KeyCode, PhysicalKey::Code},
};
use lin_alg::{f32::Vec3, f64::Vec3 as Vec3F64, map_linear};

use crate::{
    ManipMode, Selection, State, drawing,
    drawing::{EntityClass, MoleculeView},
    mol_manip,
    molecule::{Atom, MolType, MoleculeCommon},
    render::set_flashlight,
    selection::{find_selected_atom, points_along_ray},
    util::{cycle_selected, move_cam_to_sel, orbit_center},
};

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

// Sensitives for mol manip.
pub const SENS_MOL_MOVE_SCROLL: f32 = 1.5e-2;
pub const SENS_MOL_ROT_SCROLL: f32 = 5e-2;
pub const SENS_MOL_ROT_MOUSE: f32 = 5e-3;

pub fn event_dev_handler(
    state_: &mut State,
    event: DeviceEvent,
    scene: &mut Scene,
    _engine_inputs: bool,
    dt: f32,
) -> EngineUpdates {
    let mut updates = EngineUpdates::default();

    let mut redraw_protein = false;
    let mut redraw_lig = false;
    let mut redraw_na = false;
    let mut redraw_lipid = false;

    let mut redraw_ligs_inplace = false;
    let mut redraw_na_inplace = false;
    let mut redraw_lipid_inplace = false;

    // todo: Move this logic to the engine (graphics lib)?
    if !state_.ui.mouse_in_window {
        return updates;
    }

    match event {
        // Move the camera forward and back on scroll; handled by Graphics cam controls.
        DeviceEvent::MouseWheel { delta } => {
            set_flashlight(scene);
            updates.lighting = true;

            mol_manip::handle_mol_manip_in_out(
                state_,
                scene,
                delta,
                &mut redraw_ligs_inplace,
                &mut redraw_na_inplace,
                &mut redraw_lipid_inplace,
            );
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
                        state_.volatile.mol_manip.pivot = None;
                        false
                    }
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

                            // If we don't scale the selection distance appropriately, an atom etc
                            // behind the desired one, but closer to the ray, may be selected; likely
                            // this is undesired.
                            let dist_thresh = match state_.ui.mol_view {
                                MoleculeView::SpaceFill => SELECTION_DIST_THRESH_LARGE,
                                _ => SELECTION_DIST_THRESH_SMALL,
                            };

                            // todo: Lots of DRY here!

                            fn get_atoms(mol: &MoleculeCommon) -> Vec<Atom> {
                                // todo: I don't like this clone!
                                mol.atoms
                                    .iter()
                                    .enumerate()
                                    .map(|(i, a)| Atom {
                                        posit: mol.atom_posits[i],
                                        element: a.element,
                                        ..Default::default()
                                    })
                                    .collect()
                            }

                            let mut lig_atoms = Vec::new();
                            for mol in &state_.ligands {
                                lig_atoms.push(get_atoms(&mol.common));
                            }

                            let mut na_atoms = Vec::new();
                            for mol in &state_.nucleic_acids {
                                na_atoms.push(get_atoms(&mol.common));
                            }
                            let mut lipid_atoms = Vec::new();
                            for mol in &state_.lipids {
                                lipid_atoms.push(get_atoms(&mol.common));
                            }

                            let selection = match &state_.peptide {
                                Some(mol) => {
                                    let (
                                        atoms_along_ray,
                                        atoms_along_ray_lig,
                                        atoms_along_ray_na,
                                        atoms_along_ray_lipid,
                                    ) = points_along_ray(
                                        selected_ray,
                                        &mol.common.atoms,
                                        &lig_atoms,
                                        &na_atoms,
                                        &lipid_atoms,
                                        dist_thresh,
                                    );

                                    find_selected_atom(
                                        &atoms_along_ray,
                                        &atoms_along_ray_lig,
                                        &atoms_along_ray_na,
                                        &atoms_along_ray_lipid,
                                        &mol.common.atoms,
                                        &mol.residues,
                                        &lig_atoms,
                                        &na_atoms,
                                        &lipid_atoms,
                                        &selected_ray,
                                        &state_.ui,
                                        &mol.chains,
                                    )
                                }
                                None => {
                                    let (
                                        atoms_along_ray,
                                        atoms_along_ray_lig,
                                        atoms_along_ray_na,
                                        atoms_along_ray_lipid,
                                    ) = points_along_ray(
                                        selected_ray,
                                        &Vec::new(),
                                        &lig_atoms,
                                        &na_atoms,
                                        &lipid_atoms,
                                        dist_thresh,
                                    );

                                    find_selected_atom(
                                        &atoms_along_ray,
                                        &atoms_along_ray_lig,
                                        &atoms_along_ray_na,
                                        &atoms_along_ray_lipid,
                                        &Vec::new(),
                                        &Vec::new(),
                                        &lig_atoms,
                                        &na_atoms,
                                        &lipid_atoms,
                                        &selected_ray,
                                        &state_.ui,
                                        &Vec::new(),
                                    )
                                }
                            };

                            match selection {
                                Selection::AtomLig((mol_i, _)) => {
                                    state_.volatile.active_mol = Some((MolType::Ligand, mol_i));
                                }
                                Selection::AtomNucleicAcid((mol_i, _)) => {
                                    state_.volatile.active_mol =
                                        Some((MolType::NucleicAcid, mol_i));
                                }
                                Selection::AtomLipid((mol_i, _)) => {
                                    state_.volatile.active_mol = Some((MolType::Lipid, mol_i));
                                }
                                _ => (),
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
                            redraw_na = true;
                            redraw_lipid = true;
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

                        match state_.ui.selection {
                            Selection::AtomPeptide(_) | Selection::Residue(_) => {
                                redraw_protein = true
                            }
                            Selection::AtomLig(_) => redraw_lig = true,
                            Selection::AtomNucleicAcid(_) => redraw_na = true,
                            Selection::AtomLipid(_) => redraw_lipid = true,
                            _ => (),
                        }
                    }
                    Code(KeyCode::ArrowRight) => {
                        cycle_selected(state_, scene, false);

                        match state_.ui.selection {
                            Selection::AtomPeptide(_) | Selection::Residue(_) => {
                                redraw_protein = true
                            }
                            Selection::AtomLig(_) => redraw_lig = true,
                            Selection::AtomNucleicAcid(_) => redraw_na = true,
                            Selection::AtomLipid(_) => redraw_lipid = true,
                            _ => (),
                        }
                    }
                    Code(KeyCode::Escape) => {
                        state_.ui.selection = Selection::None;
                        state_.volatile.active_mol = None;
                        state_.volatile.mol_manip.mol = ManipMode::None;
                        state_.volatile.mol_manip.pivot = None;
                        scene.input_settings.control_scheme = state_.volatile.control_scheme_prev;

                        redraw_protein = true;
                        redraw_lig = true;
                        redraw_na = true;
                        redraw_lipid = true;
                    }
                    Code(KeyCode::Enter) => {
                        move_cam_to_sel(
                            &mut state_.ui,
                            &state_.peptide,
                            &state_.ligands,
                            &state_.nucleic_acids,
                            &state_.lipids,
                            &mut scene.camera,
                            &mut updates,
                        );
                    }
                    Code(KeyCode::BracketLeft) => {
                        state_.ui.mol_view = state_.ui.mol_view.prev();
                        redraw_protein = true;
                        redraw_lig = true;
                        redraw_na = true;
                        redraw_lipid = true;
                    }
                    Code(KeyCode::BracketRight) => {
                        state_.ui.mol_view = state_.ui.mol_view.next();
                        redraw_protein = true;
                        redraw_lig = true;
                        redraw_na = true;
                        redraw_lipid = true;
                    }

                    Code(KeyCode::KeyM) => {
                        let mol_type = match state_.active_mol() {
                            Some(m) => m.mol_type(),
                            None => return updates,
                        };

                        mol_manip::set_manip(
                            &mut state_.volatile,
                            scene,
                            &mut redraw_ligs_inplace,
                            &mut redraw_na_inplace,
                            &mut redraw_lipid_inplace,
                            ManipMode::Move((mol_type, 0)),
                        );
                    }
                    Code(KeyCode::KeyR) => {
                        let mol_type = match state_.active_mol() {
                            Some(m) => m.mol_type(),
                            None => return updates,
                        };

                        mol_manip::set_manip(
                            &mut state_.volatile,
                            scene,
                            &mut redraw_ligs_inplace,
                            &mut redraw_na_inplace,
                            &mut redraw_lipid_inplace,
                            ManipMode::Rotate((mol_type, 0)),
                        );
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
                mol_manip::handle_mol_manip_in_plane(
                    state_,
                    scene,
                    delta,
                    &mut redraw_ligs_inplace,
                    &mut redraw_na_inplace,
                    &mut redraw_lipid_inplace,
                );
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
        updates.entities = EntityUpdate::All;
        // updates.entities.push_class(EntityClass::Peptide as u32);
    }

    if redraw_lig {
        // if let Some(lig) = &mut state_.active_lig_mut() {
        //     lig.position_atoms(None);
        // }

        drawing::draw_all_ligs(state_, scene);
        updates.entities = EntityUpdate::All;
    }

    if redraw_na {
        drawing::draw_all_nucleic_acids(state_, scene);
        updates.entities = EntityUpdate::All;
    }

    if redraw_lipid {
        drawing::draw_all_lipids(state_, scene);
        updates.entities = EntityUpdate::All;
    }

    // if redraw_ligs_inplace {
    //     drawing::update_all_ligs_inplace(state_, scene);
    //     updates.entities.push_class(EntityClass::Ligand as u32);
    // }
    //
    // if redraw_na_inplace {
    //     drawing::update_all_na_inplace(state_, scene);
    //     updates.entities.push_class(EntityClass::NucleicAcid as u32);
    // }

    if redraw_ligs_inplace {
        // todo: Fragile? Error handle.
        let mol_i = state_.volatile.active_mol.unwrap().1;

        if mol_i >= state_.ligands.len() {
            eprintln!("Uhoh: Index error on in-place redraw");
            drawing::draw_all_ligs(state_, scene);
            return updates;
        }

        drawing::update_single_ligand_inplace(mol_i, state_, scene);

        let mol = &mut state_.ligands[mol_i];

        updates.entities = match mol.common.entity_i_range {
            Some(range) => EntityUpdate::Indexes(range),
            None => {
                eprintln!("Error: Missing entity index range.");
                EntityUpdate::All
            }
        };
    }

    if redraw_na_inplace {
        // todo: Fragile? Error handle.
        let mol_i = state_.volatile.active_mol.unwrap().1;

        if mol_i >= state_.nucleic_acids.len() {
            eprintln!("Uhoh: Index error on in-place redraw");
            drawing::draw_all_nucleic_acids(state_, scene);
            return updates;
        }

        drawing::update_single_nucleic_acid_inplace(mol_i, state_, scene);

        let mol = &mut state_.nucleic_acids[mol_i];
        updates.entities = match mol.common.entity_i_range {
            Some(range) => EntityUpdate::Indexes(range),
            None => {
                eprintln!("Error: Missing entity index range.");
                EntityUpdate::All
            }
        };
    }

    if redraw_lipid_inplace {
        // todo: Fragile? Error handle.
        let mol_i = state_.volatile.active_mol.unwrap().1;

        if mol_i >= state_.lipids.len() {
            eprintln!("Uhoh: Index error on in-place redraw");
            drawing::draw_all_lipids(state_, scene);
            return updates;
        }

        drawing::update_single_lipid_inplace(mol_i, state_, scene);

        let mol = &mut state_.lipids[mol_i];
        updates.entities = match mol.common.entity_i_range {
            Some(range) => EntityUpdate::Indexes(range),
            None => {
                eprintln!("Error: Missing entity index range.");
                EntityUpdate::All
            }
        };
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
        } => state.ui.cursor_pos = Some((position.x as f32, position.y as f32)),
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
