//! Handles user inputs, e.g. from keyboard and mouse.

use bio_files::BondType;
use graphics::{
    ControlScheme, DeviceEvent, ElementState, EngineUpdates, EntityUpdate, FWD_VEC, Scene,
    WindowEvent,
    event::MouseScrollDelta,
    winit::keyboard::{KeyCode, PhysicalKey::Code},
};
use lin_alg::f32::Vec3;
use na_seq::Element::Carbon;

use crate::{
    OperatingMode, Selection, State,
    cam_misc::move_cam_to_sel,
    drawing,
    drawing::EntityClass,
    drawing_wrappers, mol_editor,
    mol_editor::add_atoms::add_atom,
    mol_manip,
    mol_manip::ManipMode,
    molecule::MolType,
    render::set_flashlight,
    selection,
    ui::cam::{FOG_DIST_MIN, set_fog_dist},
    util::{close_mol, cycle_selected},
};

// These are defaults; overridden by the user A/R, and saved to prefs.
pub const MOVEMENT_SENS: f32 = 12.;
pub const ROTATE_SENS: f32 = 0.45;
pub const RUN_FACTOR: f32 = 6.; // i.e. shift key multiplier

pub const SCROLL_MOVE_AMT: f32 = 4.;
pub const SCROLL_ROTATE_AMT: f32 = 12.;

// Sensitives for mol manip.
pub const SENS_MOL_MOVE_SCROLL: f32 = 2.5e-2;
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

    // todo: DRY with graphics.
    // This affects our app-specific commands, vs engine built-in ones. For example, hot keys
    // to change various modes.

    let mut redraw_protein = false;
    let mut redraw_lig = false;
    let mut redraw_na = false;
    let mut redraw_lipid = false;

    let mut redraw_mol_editor = false;

    let mut redraw_ligs_inplace = false;
    let mut redraw_na_inplace = false;
    let mut redraw_lipid_inplace = false;

    if !state_.ui.mouse_in_window {
        return updates;
    }

    match event {
        // Move the camera forward and back on scroll; handled by Graphics cam controls.
        DeviceEvent::MouseWheel { delta } => {
            if state_.volatile.key_modifiers.state().control_key() {
                let scroll: f32 = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 120.0,
                };

                state_.ui.view_depth.1 =
                    (state_.ui.view_depth.1 as i16 + (scroll * 5.) as i16) as u16;

                // Overflowed from subtraction.
                if state_.ui.view_depth.1 > 2_000 {
                    state_.ui.view_depth.1 = FOG_DIST_MIN;
                }

                set_fog_dist(&mut scene.camera, state_.ui.view_depth.1);

                // Counteract the engine's default free look behavior. This is indirect, but good
                // enough for now.
                if matches!(
                    scene.input_settings.control_scheme,
                    ControlScheme::FreeCamera | ControlScheme::Arc { center: _ }
                ) {
                    let fwd = scene.camera.orientation.rotate_vec(FWD_VEC);
                    scene.camera.position += fwd * -scroll * SCROLL_MOVE_AMT;
                    updates.camera = true;
                }

                return updates;
            }

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
                    ElementState::Pressed => match state_.volatile.operating_mode {
                        OperatingMode::Primary => {
                            selection::handle_selection_attempt(
                                state_,
                                scene,
                                &mut redraw_protein,
                                &mut redraw_lig,
                                &mut redraw_lipid,
                                &mut redraw_na,
                            );
                        }
                        OperatingMode::MolEditor => {
                            selection::handle_selection_attempt_mol_editor(
                                state_,
                                scene,
                                &mut redraw_mol_editor,
                            );
                        }
                    },
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

                        match state_.volatile.operating_mode {
                            OperatingMode::Primary => match state_.ui.selection {
                                Selection::AtomPeptide(_)
                                | Selection::Residue(_)
                                | Selection::BondPeptide(_) => redraw_protein = true,
                                Selection::AtomLig(_) | Selection::BondLig(_) => redraw_lig = true,
                                Selection::AtomNucleicAcid(_) | Selection::BondNucleicAcid(_) => {
                                    redraw_na = true
                                }
                                Selection::AtomLipid(_) | Selection::BondLipid(_) => {
                                    redraw_lipid = true
                                }
                                _ => (),
                            },
                            OperatingMode::MolEditor => redraw_mol_editor = true,
                        }
                    }
                    Code(KeyCode::ArrowRight) => {
                        cycle_selected(state_, scene, false);

                        match state_.volatile.operating_mode {
                            OperatingMode::Primary => match state_.ui.selection {
                                Selection::AtomPeptide(_)
                                | Selection::Residue(_)
                                | Selection::BondPeptide(_) => redraw_protein = true,
                                Selection::AtomLig(_) | Selection::BondLig(_) => redraw_lig = true,
                                Selection::AtomNucleicAcid(_) | Selection::BondNucleicAcid(_) => {
                                    redraw_na = true
                                }
                                Selection::AtomLipid(_) | Selection::BondLipid(_) => {
                                    redraw_lipid = true
                                }
                                _ => (),
                            },
                            OperatingMode::MolEditor => redraw_mol_editor = true,
                        }
                    }
                    Code(KeyCode::Escape) => {
                        // If in manip mode, exit that, but don't remove selections.
                        if matches!(
                            state_.volatile.mol_manip.mode,
                            ManipMode::Move(_) | ManipMode::Rotate(_)
                        ) {
                            state_.volatile.mol_manip.mode = ManipMode::None;
                            state_.volatile.mol_manip.pivot = None;
                            scene.input_settings.control_scheme =
                                state_.volatile.control_scheme_prev;
                        } else {
                            state_.ui.selection = Selection::None;
                            state_.volatile.active_mol = None;
                        }

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
                    Code(KeyCode::BracketLeft) => match state_.volatile.operating_mode {
                        OperatingMode::Primary => {
                            state_.ui.mol_view = state_.ui.mol_view.prev();

                            redraw_protein = true;
                            redraw_lig = true;
                            redraw_na = true;
                            redraw_lipid = true;
                        }
                        OperatingMode::MolEditor => {
                            state_.ui.mol_view = state_.ui.mol_view.prev_editor();
                            redraw_mol_editor = true;
                        }
                    },
                    Code(KeyCode::BracketRight) => match state_.volatile.operating_mode {
                        OperatingMode::Primary => {
                            state_.ui.mol_view = state_.ui.mol_view.next();

                            redraw_protein = true;
                            redraw_lig = true;
                            redraw_na = true;
                            redraw_lipid = true;
                        }
                        OperatingMode::MolEditor => {
                            state_.ui.mol_view = state_.ui.mol_view.next_editor();
                            redraw_mol_editor = true;
                        }
                    },
                    Code(KeyCode::Semicolon) => match state_.volatile.operating_mode {
                        OperatingMode::Primary => {
                            state_.ui.view_sel_level = state_.ui.view_sel_level.prev();

                            redraw_protein = true;
                            redraw_lig = true;
                            redraw_na = true;
                            redraw_lipid = true;
                        }
                        OperatingMode::MolEditor => {
                            state_.ui.view_sel_level = state_.ui.view_sel_level.prev();
                            redraw_mol_editor = true;
                        }
                    },
                    Code(KeyCode::Quote) => match state_.volatile.operating_mode {
                        OperatingMode::Primary => {
                            state_.ui.view_sel_level = state_.ui.view_sel_level.next();

                            redraw_protein = true;
                            redraw_lig = true;
                            redraw_na = true;
                            redraw_lipid = true;
                        }
                        OperatingMode::MolEditor => {
                            state_.ui.view_sel_level = state_.ui.view_sel_level.next();
                            redraw_mol_editor = true;
                        }
                    },
                    Code(KeyCode::KeyM) => {
                        // let (mol_type, mol_i) = match state_.volatile.operating_mode {
                        //     OperatingMode::Primary => match state_.active_mol() {
                        //         Some(m) => (
                        //             m.mol_type(),
                        //             state_.volatile.active_mol.unwrap_or_default().1,
                        //         ),
                        //         None => return updates,
                        //     },
                        //     OperatingMode::MolEditor => {
                        //         // todo: DRY with mol editor button.
                        //         let (mol_i, atom_sel_i) = match &state_.ui.selection {
                        //             Selection::AtomLig((mol_i, i)) => (*mol_i, *i),
                        //             Selection::AtomsLig((mol_i, i)) => {
                        //                 // todo: How should we handle this?
                        //                 (*mol_i, i[0])
                        //             }
                        //             _ => return updates,
                        //         };
                        //
                        //         (MolType::Ligand, atom_sel_i)
                        //     }
                        // };

                        mol_manip::set_manip(
                            &mut state_.volatile,
                            &mut state_.to_save.save_flag,
                            scene,
                            &mut redraw_ligs_inplace,
                            &mut redraw_na_inplace,
                            &mut redraw_lipid_inplace,
                            ManipMode::Move((MolType::Ligand, 0)),
                            &state_.ui.selection,
                        );
                    }
                    Code(KeyCode::KeyR) => {
                        let mol_type = match state_.active_mol() {
                            Some(m) => m.mol_type(),
                            None => return updates,
                        };

                        mol_manip::set_manip(
                            &mut state_.volatile,
                            &mut state_.to_save.save_flag,
                            scene,
                            &mut redraw_ligs_inplace,
                            &mut redraw_na_inplace,
                            &mut redraw_lipid_inplace,
                            ManipMode::Rotate((mol_type, 0)),
                            &state_.ui.selection,
                        );
                    }
                    Code(KeyCode::Delete) => {
                        match state_.volatile.operating_mode {
                            OperatingMode::Primary => {
                                // Close the active mol?
                                if let Some((mol_type, i)) = state_.volatile.active_mol {
                                    close_mol(mol_type, i, state_, scene, &mut updates);
                                }
                            }
                            OperatingMode::MolEditor => {
                                // Delete the selected atom.

                                if let Selection::AtomLig((_, i)) = state_.ui.selection {
                                    state_.mol_editor.remove_atom(i);
                                    redraw_mol_editor = true;
                                }
                            }
                        }
                    }
                    Code(KeyCode::ShiftLeft | KeyCode::ShiftRight) => {
                        state_.volatile.inputs_commanded.run = true;
                    }
                    Code(KeyCode::Tab) => {
                        // todo: This is DRY/mostly C+P from the add atom button.
                        if state_.volatile.operating_mode == OperatingMode::MolEditor {
                            let (mol_i, atom_sel_i) = match &state_.ui.selection {
                                Selection::AtomLig((mol_i, i)) => (*mol_i, *i),
                                Selection::AtomsLig((mol_i, i)) => {
                                    // todo: How should we handle this?
                                    (*mol_i, i[0])
                                }
                                _ => return updates,
                            };

                            // todo: DRY here in some of the params with the button
                            add_atom(
                                &mut state_.mol_editor.mol.common,
                                &mut scene.entities,
                                atom_sel_i,
                                Carbon,
                                BondType::Single,
                                Some("c".to_owned()), // todo
                                Some(1.4),            // todo
                                0.13,                 // todo
                                &mut state_.ui,
                                &mut updates,
                                &mut scene.input_settings.control_scheme,
                                state_.volatile.mol_manip.mode,
                            );
                            // todo: Rebuild md here.

                            if state_.mol_editor.md_running {
                                // todo: Ideally don't rebuild the whole dynamics, for performance reasons.
                                match mol_editor::build_dynamics(
                                    &state_.dev,
                                    &mut state_.mol_editor.mol,
                                    &state_.ff_param_set,
                                    &mut state_.mol_editor.mol_specific_params,
                                    &state_.to_save.md_config,
                                ) {
                                    Ok(d) => state_.mol_editor.md_state = Some(d),
                                    Err(e) => eprintln!(
                                        "Problem setting up dynamics for the editor: {e:?}"
                                    ),
                                }
                            } else {
                                // Will be triggered next time MD is started.
                                state_.mol_editor.md_rebuild_required = true;
                            }
                        }
                    }
                    _ => (),
                },
                ElementState::Released => match key.physical_key {
                    Code(KeyCode::ShiftLeft | KeyCode::ShiftRight) => {
                        state_.volatile.inputs_commanded.run = false;
                    }
                    _ => (),
                },
            }

            // todo: If you enable a direction-dependent flashlight, you will need to modify the mouse movement
            // todo state too.
            // Similar code to the engine. Update state so we can update the flashlight while a key is held,
            // and reset snapshots.
            if key.state == ElementState::Pressed {
                match key.physical_key {
                    // Check the cases for the engine's built-in movement commands, to set the current-snapshot to None.
                    // C+P partially, from `graphics`. These are for press and release.
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

                set_flashlight(scene);
                updates.lighting = true;

                // todo: Experimenting
                // unsafe {
                //     I_FIND_NEAREST += 1;
                //     if I_FIND_NEAREST.is_multiple_of(RATIO_FIND_NEAREST) {
                //         state_.volatile.nearest_mol_dist_to_cam =
                //             find_nearest_mol_dist_to_cam(state_, &scene.camera);
                //
                //         if let Some(dist) = state_.volatile.nearest_mol_dist_to_cam {
                //
                //             let dist = max(dist as u16 + FOG_HALF_DEPTH/2 - 5, 5);
                //             set_fog_dist(&mut scene.camera, dist);
                //         }
                //     }
                // }
            }
        }
        _ => (),
    }

    if redraw_protein && state_.volatile.operating_mode == OperatingMode::Primary {
        // todo:This is overkill for certain keys. Just change the color of the one[s] in question, and set update.entities = true.
        drawing::draw_peptide(state_, scene);
        updates.entities = EntityUpdate::All;
        // updates.entities.push_class(EntityClass::Peptide as u32);
    }

    if redraw_lig {
        match state_.volatile.operating_mode {
            OperatingMode::Primary => drawing_wrappers::draw_all_ligs(state_, scene),

            OperatingMode::MolEditor => mol_editor::redraw(
                &mut scene.entities,
                &state_.mol_editor.mol,
                &state_.ui,
                state_.volatile.mol_manip.mode,
            ),
        }
        updates.entities = EntityUpdate::All;
    }

    if redraw_na && state_.volatile.operating_mode == OperatingMode::Primary {
        drawing_wrappers::draw_all_nucleic_acids(state_, scene);
        updates.entities = EntityUpdate::All;
    }

    if redraw_lipid && state_.volatile.operating_mode == OperatingMode::Primary {
        drawing_wrappers::draw_all_lipids(state_, scene);
        updates.entities = EntityUpdate::All;
    }

    if redraw_ligs_inplace {
        match state_.volatile.operating_mode {
            OperatingMode::Primary => {
                redraw_inplace_helper(MolType::Ligand, state_, scene, &mut updates);
            }
            OperatingMode::MolEditor => {
                mol_editor::redraw(
                    &mut scene.entities,
                    &state_.mol_editor.mol,
                    &state_.ui,
                    state_.volatile.mol_manip.mode,
                );
                updates.entities = EntityUpdate::Classes(vec![EntityClass::Ligand as u32]);
            }
        }
    }

    if redraw_na_inplace && state_.volatile.operating_mode == OperatingMode::Primary {
        redraw_inplace_helper(MolType::NucleicAcid, state_, scene, &mut updates);
    }

    if redraw_lipid_inplace && state_.volatile.operating_mode == OperatingMode::Primary {
        redraw_inplace_helper(MolType::Lipid, state_, scene, &mut updates);
    }

    if redraw_mol_editor {
        mol_editor::redraw(
            &mut scene.entities,
            &state_.mol_editor.mol,
            &state_.ui,
            state_.volatile.mol_manip.mode,
        );
        updates.entities = EntityUpdate::All;
    }

    // We handle the flashlight elsewhere, as this event handler only fires upon events; not while
    // a key is held.
    if state_.volatile.inputs_commanded.inputs_present() {
        state_.ui.cam_snapshot = None;

        // todo: Experimenting
        // unsafe {
        //     I_FIND_NEAREST += 1;
        //     if I_FIND_NEAREST.is_multiple_of(RATIO_FIND_NEAREST) {
        //         state_.volatile.nearest_mol_dist_to_cam =
        //             find_nearest_mol_dist_to_cam(state_, &scene.camera);
        //
        //         if let Some(dist) = state_.volatile.nearest_mol_dist_to_cam {
        //             let dist = max(dist as u16 + FOG_HALF_DEPTH/2 - 5, 5);
        //             set_fog_dist(&mut scene.camera, dist);
        //         }
        //     }
        // }
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
        WindowEvent::ModifiersChanged(val) => {
            state.volatile.key_modifiers = val;
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

//         // let mol_i = state_.volatile.active_mol.unwrap().1;
//         //
//         // if mol_i >= state_.lipids.len() {
//         //     eprintln!("Uhoh: Index error on in-place redraw");
//         //     drawing_wrappers::draw_all_lipids(state_, scene);
//         //     return updates;
//         // }
//         //
//         // drawing_wrappers::update_single_lipid_inplace(mol_i, state_, scene);
//         //
//         // let mol = &mut state_.lipids[mol_i];
//         // updates.entities = match mol.common.entity_i_range {
//         //     Some(range) => EntityUpdate::Indexes(range),
//         //     None => {
//         //         eprintln!("Error: Missing entity index range.");
//         //         EntityUpdate::All
//         //     }
//         // };

/// Abstracts over mol types.
fn redraw_inplace_helper(
    mol_type: MolType,
    state: &mut State,
    scene: &mut Scene,
    updates: &mut EngineUpdates,
) {
    // todo: Fragile? Error handle.
    let mol_i = state.volatile.active_mol.unwrap().1;

    let err = "Uhoh: Index error on in-place redraw";

    let mol = match mol_type {
        MolType::Ligand => {
            drawing_wrappers::update_single_ligand_inplace(mol_i, state, scene);

            if mol_i >= state.ligands.len() {
                eprintln!("{err}");
                drawing_wrappers::draw_all_ligs(state, scene);
                return;
            }

            &mut state.ligands[mol_i].common
        }
        MolType::NucleicAcid => {
            drawing_wrappers::update_single_nucleic_acid_inplace(mol_i, state, scene);

            if mol_i >= state.nucleic_acids.len() {
                eprintln!("{err}");
                drawing_wrappers::draw_all_nucleic_acids(state, scene);
                return;
            }

            &mut state.nucleic_acids[mol_i].common
        }
        MolType::Lipid => {
            drawing_wrappers::update_single_lipid_inplace(mol_i, state, scene);

            if mol_i >= state.lipids.len() {
                eprintln!("{err}");
                drawing_wrappers::draw_all_lipids(state, scene);
                return;
            }

            &mut state.lipids[mol_i].common
        }
        _ => unreachable!(),
    };

    updates.entities = match mol.entity_i_range {
        Some(range) => EntityUpdate::Indexes(range),
        None => {
            eprintln!("Error: Missing entity index range.");
            EntityUpdate::All
        }
    };
}
