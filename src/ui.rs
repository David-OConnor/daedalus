use std::{
    f32::consts::TAU,
    io,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

use bio_apis::{drugbank, pubchem, rcsb};
use egui::{Color32, ComboBox, Context, Key, RichText, Slider, TextEdit, TopBottomPanel, Ui};
use graphics::{
    Camera, ControlScheme, EngineUpdates, Entity, Mesh, RIGHT_VEC, Scene, UP_VEC, Vertex,
};
use lin_alg::f32::{Quaternion, Vec3};
use mcubes::{MarchingCubes, MeshSide};
use na_seq::{AaIdent, AminoAcid};

static INIT_COMPLETE: AtomicBool = AtomicBool::new(false);

use bio_files::{DensityMap, density_from_rcsb_gemmi, density_from_rcsb_gemmi2, ResidueType};

use crate::{
    CamSnapshot, MsaaSetting, Selection, State, ViewSelLevel, cli,
    cli::autocomplete_cli,
    docking::{
        ConformationType, calc_binding_energy,
        dynamics_playback::{build_vdw_dynamics, change_snapshot},
        external::check_adv_avail,
        find_optimal_pose,
        find_sites::find_docking_sites,
    },
    download_mols::{load_sdf_drugbank, load_sdf_pubchem},
    inputs::{MOVEMENT_SENS, ROTATE_SENS},
    mol_drawing::{
        COLOR_DOCKING_SITE_MESH, EntityType, MoleculeView, draw_density, draw_density_surface,
        draw_ligand, draw_molecule,
    },
    molecule::{Ligand, Molecule},
    reflection::ElectronDensity,
    render::{
        CAM_INIT_OFFSET, MESH_DENSITY_SURFACE, MESH_DOCKING_SURFACE,
        RENDER_DIST_FAR, RENDER_DIST_NEAR, set_docking_light, set_flashlight, set_static_light,
    },
    util,
    util::{
        cam_look_at, cam_look_at_outside, check_prefs_save, close_lig, close_mol,
        cycle_res_selected, handle_err, orbit_center, query_rcsb, reset_camera, select_from_search,
    },
};

pub const ROW_SPACING: f32 = 10.;
pub const COL_SPACING: f32 = 30.;

// These are divided by 10.
pub const VIEW_DEPTH_NEAR_MIN: u16 = 2;
pub const VIEW_DEPTH_NEAR_MAX: u16 = 300;

pub const VIEW_DEPTH_FAR_MIN: u16 = 10;
pub const VIEW_DEPTH_FAR_MAX: u16 = 60;

const NEARBY_THRESH_MIN: u16 = 5;
const NEARBY_THRESH_MAX: u16 = 60;

// todo: Teese aren't reacting correctly; too slow for the values set.
// const CAM_BUTTON_POS_STEP: f32 = 30.;
// const CAM_BUTTON_ROT_STEP: f32 = TAU / 3.;
// const CAM_BUTTON_POS_STEP: f32 = 30. * 3.;
// const CAM_BUTTON_ROT_STEP: f32 = TAU / 3. * 3.;

const COLOR_INACTIVE: Color32 = Color32::GRAY;
const COLOR_ACTIVE: Color32 = Color32::LIGHT_GREEN;
const COLOR_HIGHLIGHT: Color32 = Color32::LIGHT_BLUE;
const COLOR_ACTIVE_RADIO: Color32 = Color32::LIGHT_BLUE;
const COLOR_OUT_ERROR: Color32 = Color32::LIGHT_RED;
const COLOR_OUT_NORMAL: Color32 = Color32::WHITE;
const COLOR_OUT_SUCCESS: Color32 = Color32::LIGHT_GREEN; // Unused for now

const MAX_TITLE_LEN: usize = 120; // Number of characters to display.

fn active_color(val: bool) -> Color32 {
    if val { COLOR_ACTIVE } else { COLOR_INACTIVE }
}

/// Visually distinct; fore buttons that operate as radio buttons
fn active_color_sel(val: bool) -> Color32 {
    if val {
        COLOR_ACTIVE_RADIO
    } else {
        COLOR_INACTIVE
    }
}

/// A checkbox to show or hide a category.
fn vis_check(val: &mut bool, text: &str, ui: &mut Ui, redraw: &mut bool) {
    let color = active_color(!*val);
    if ui.button(RichText::new(text).color(color)).clicked() {
        *val = !*val;
        *redraw = true;
    }
}

/// Update the tilebar to reflect the current molecule
fn set_window_title(title: &str, scene: &mut Scene) {
    scene.window_title = title.to_owned();
    // ui.ctx().send_viewport_cmd(ViewportCommand::Title(title.to_string()));
}

pub fn load_file(
    path: &Path,
    state: &mut State,
    redraw: &mut bool,
    reset_cam: &mut bool,
    engine_updates: &mut EngineUpdates,
) -> io::Result<()> {
    state.open(path)?;

    *redraw = true;
    *reset_cam = true;
    engine_updates.entities = true;

    Ok(())
}

fn _int_field(val: &mut usize, label: &str, redraw: &mut bool, ui: &mut Ui) {
    ui.label(label);
    ui.label(label);
    let mut val_str = val.to_string();
    if ui
        .add_sized(
            [60., Ui::available_height(ui)],
            TextEdit::singleline(&mut val_str),
        )
        .changed()
    {
        if let Ok(v) = val_str.parse::<usize>() {
            *val = v;
            *redraw = true;
        }
    }
}

/// Handles keyboard and mouse input not associated with a widget.
pub fn handle_input(
    state: &mut State,
    ui: &mut Ui,
    redraw: &mut bool,
    reset_cam: &mut bool,
    engine_updates: &mut EngineUpdates,
) {
    ui.ctx().input(|ip| {
        // Check for file drop
        if let Some(dropped_files) = ip.raw.dropped_files.first() {
            if let Some(path) = &dropped_files.path {
                if let Err(e) = load_file(path, state, redraw, reset_cam, engine_updates) {
                    handle_err(&mut state.ui, e.to_string());
                }
            }
        }
    });
}

fn get_snap_name(snap: Option<usize>, snaps: &[CamSnapshot]) -> String {
    match snap {
        Some(i) => snaps[i].name.clone(),
        None => "None".to_owned(),
    }
}

fn cam_snapshots(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    ui.heading("Scenes:");
    ui.label("Label:");
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
            state.update_save_prefs();
        }
    }

    if state.ui.cam_snapshot != prev_snap {
        util::load_snap(state, scene, engine_updates);
    }
}

fn cam_controls(
    scene: &mut Scene,
    state: &mut State,
    engine_updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    // todo: Here and at init, set the camera dist dynamically based on mol size.
    // todo: Set the position not relative to 0, but  relative to the center of the atoms.

    let mut changed = false;

    // let cam = &mut scene.camera;

    ui.horizontal(|ui| {
        ui.label("Cam:");

        // Preset buttons
        if ui.button("Front").clicked() {
            if let Some(mol) = &state.molecule {
                reset_camera(scene, &mut state.ui.view_depth, engine_updates, mol);
                changed = true;
            }
        }

        if ui.button("Top").clicked() {
            if let Some(mol) = &state.molecule {
                let center: Vec3 = mol.center.into();
                reset_camera(scene, &mut state.ui.view_depth, engine_updates, mol);
                scene.camera.position =
                    Vec3::new(center.x, center.y + (mol.size + CAM_INIT_OFFSET), center.z);
                scene.camera.orientation = Quaternion::from_axis_angle(RIGHT_VEC, TAU / 4.);

                changed = true;
            }
        }

        if ui.button("Left").clicked() {
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
            .button(RichText::new("Free").color(active_color_sel(free_active)))
            .clicked()
        {
            scene.input_settings.control_scheme = ControlScheme::FreeCamera;
            state.to_save.control_scheme = ControlScheme::FreeCamera;
        }

        if ui
            .button(RichText::new("Arc").color(active_color_sel(arc_active)))
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
                    RichText::new("Orbit sel").color(active_color(state.ui.orbit_around_selection)),
                )
                .clicked()
            {
                state.ui.orbit_around_selection = !state.ui.orbit_around_selection;

                let center = orbit_center(state);
                scene.input_settings.control_scheme = ControlScheme::Arc { center };
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

        ui.add_space(COL_SPACING);
    });

    if changed {
        engine_updates.camera = true;

        set_flashlight(scene);
        engine_updates.lighting = true; // flashlight.

        state.ui.cam_snapshot = None;
    }
}

/// Display text of the selected atom
fn selected_data(mol: &Molecule, selection: &Selection, ui: &mut Ui) {
    match selection {
        Selection::Atom(sel_i) => {
            if *sel_i >= mol.atoms.len() {
                return;
            }

            let atom = &mol.atoms[*sel_i];

            let mut aa = String::new();
            if let Some(res_i) = atom.residue {
                let res = &mol.residues[res_i];
                aa = match res.res_type {
                    ResidueType::AminoAcid(a) => format!("AA: {}", a.to_str(AaIdent::OneLetter)),
                    _ => String::new(),
                };
            }

            let role = match atom.role {
                Some(r) => format!("Role: {r}"),
                None => String::new(),
            };

            // Similar to `Vec3`'s format impl, but with fewer digits.
            let posit_txt = format!(
                "|{:.3}, {:.3}, {:.3}|",
                atom.posit.x, atom.posit.y, atom.posit.z
            );

            // Split so we can color-code by element.
            let text_a = format!("{}  {}  El:", posit_txt, atom.serial_number);

            let text_b = atom.element.to_letter();

            let mut text_c = format!("{aa}  {role}",);

            if let Some(res_i) = atom.residue {
                let res = &mol.residues[res_i];
                text_c += &format!("  {}", res.descrip());
            }

            ui.label(RichText::new(text_a).color(Color32::GOLD));
            let (r, g, b) = atom.element.color();
            ui.label(RichText::new(text_b).color(Color32::from_rgb(
                (r * 255.) as u8,
                (g * 255.) as u8,
                (b * 255.) as u8,
            )));
            ui.label(RichText::new(text_c).color(Color32::GOLD));
        }
        Selection::Residue(sel_i) => {
            if *sel_i >= mol.residues.len() {
                return;
            }

            let res = &mol.residues[*sel_i];
            ui.label(RichText::new(res.descrip()).color(Color32::GOLD));
        }
        Selection::Atoms(is) => {
            // todo: A/R
            ui.label(RichText::new(format!("{} atoms", is.len())).color(Color32::GOLD));
        }
        Selection::None => (),
    }
}

fn residue_selector(state: &mut State, scene: &mut Scene, redraw: &mut bool, ui: &mut Ui) {
    // This is a bit fuzzy, as the size varies by residue name (Not always 1 for non-AAs), and index digits.

    let mut update_arc_center = false;

    if let Some(mol) = &state.molecule {
        if let Some(chain_i) = state.ui.chain_to_pick_res {
            if chain_i >= mol.chains.len() {
                return;
            }
            let chain = &mol.chains[chain_i];

            ui.add_space(ROW_SPACING);
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 8.0;

                for (i, res) in mol.residues.iter().enumerate() {
                    // For now, peptide residues only.
                    if let ResidueType::Water = res.res_type {
                        continue;
                    }

                    // Only let the user select residue from the selected chain. This should keep
                    // it more organized, and keep UI space used down.
                    if !chain.residues.contains(&i) {
                        continue;
                    }

                    let name = match &res.res_type {
                        ResidueType::AminoAcid(aa) => aa.to_str(AaIdent::OneLetter),
                        ResidueType::Water => "Water".to_owned(),
                        ResidueType::Other(name) => name.clone(),
                    };

                    let mut color = Color32::GRAY;
                    if let Selection::Residue(sel_i) = state.selection {
                        if sel_i == i {
                            color = COLOR_ACTIVE;
                        }
                    }
                    if ui
                        .button(
                            RichText::new(format!("{} {name}", res.serial_number))
                                .size(10.)
                                .color(color),
                        )
                        .clicked()
                    {
                        state.ui.view_sel_level = ViewSelLevel::Residue;
                        state.selection = Selection::Residue(i);

                        update_arc_center = true; // Avoids borrow error.

                        *redraw = true;
                    }
                }
            });
        }
    }

    if update_arc_center {
        if let ControlScheme::Arc { center } = &mut scene.input_settings.control_scheme {
            *center = orbit_center(state);
        }
    }
}

/// Toggles chain visibility
fn chain_selector(state: &mut State, redraw: &mut bool, ui: &mut Ui) {
    // todo: For now, DRY with res selec
    ui.horizontal(|ui| {
        ui.label("Chain vis:");
        if let Some(mol) = &mut state.molecule {
            for chain in &mut mol.chains {
                let color = active_color(chain.visible);

                if ui
                    .button(RichText::new(chain.id.clone()).color(color))
                    .clicked()
                {
                    chain.visible = !chain.visible;
                    *redraw = true;
                }
            }

            ui.add_space(COL_SPACING);

            ui.label("Select residues from:");

            for (i, chain) in mol.chains.iter().enumerate() {
                let mut color = Color32::GRAY;
                if let Some(i_sel) = state.ui.chain_to_pick_res {
                    if i == i_sel {
                        color = COLOR_ACTIVE
                    }
                }
                if ui
                    .button(RichText::new(chain.id.clone()).color(color))
                    .clicked()
                {
                    // Toggle behavior.
                    if let Some(sel_i) = state.ui.chain_to_pick_res {
                        if i == sel_i {
                            state.ui.chain_to_pick_res = None;
                        } else {
                            state.ui.chain_to_pick_res = Some(i);
                        }
                    } else {
                        state.ui.chain_to_pick_res = Some(i);
                    }
                    state.volatile.ui_height = ui.ctx().used_size().y;
                }
            }

            if state.ui.chain_to_pick_res.is_some() {
                if ui.button("(None)").clicked() {
                    state.ui.chain_to_pick_res = None;
                    state.volatile.ui_height = ui.ctx().used_size().y;
                }
            }
        }
    });
}

// todo: Update params A/R
fn draw_cli(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    redraw: &mut bool,
    reset_cam: &mut bool,
    ui: &mut Ui,
) {
    ui.horizontal(|ui| {
        ui.label("Out: ");
        let color = if state.ui.cmd_line_out_is_err {
            COLOR_OUT_ERROR
        } else {
            COLOR_OUT_NORMAL
        };
        ui.label(RichText::new(&state.ui.cmd_line_output).color(color));
    });

    ui.horizontal(|ui| {
        ui.label("In: ");
        let edit_resp = ui.add(
            TextEdit::singleline(&mut state.ui.cmd_line_input)
                .desired_width(200.)
                // Prevent losing focus on Tab;
                // .cursor_at_end(true)
                .lock_focus(true),
        );

        if edit_resp.changed() {
            // todo: Validate input and color-code?
        }

        ui.add_space(COL_SPACING / 2.);

        let button_clicked = ui.button(RichText::new("Submit")).clicked();
        // This  behavior of lost and has are due to the default EGUI beavhior of those keys.
        // We can use the `lock_focus(true)` method to prevent these, but we generally like them.
        let enter_pressed = edit_resp.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter));
        let tab_pressed = edit_resp.has_focus() && ui.input(|i| i.key_pressed(Key::Tab));
        let up_pressed = edit_resp.has_focus() && ui.input(|i| i.key_pressed(Key::ArrowUp));
        let dn_pressed = edit_resp.has_focus() && ui.input(|i| i.key_pressed(Key::ArrowDown));

        if tab_pressed && !state.ui.cmd_line_input.is_empty() {
            autocomplete_cli(&mut state.ui.cmd_line_input);

            // edit_resp.surrender_focus();
            // state.ui.cmd_line_input.push(' ');
            // state.ui.cmd_line_input.pop();
            // edit_resp.request_focus();
        }

        if up_pressed {
            if state.volatile.cli_input_selected != 0 {
                state.volatile.cli_input_selected -= 1;
            }
            if state.volatile.cli_input_history.len() > state.volatile.cli_input_selected {
                state.ui.cmd_line_input =
                    state.volatile.cli_input_history[state.volatile.cli_input_selected].clone();
            }
        }

        if dn_pressed {
            if state.volatile.cli_input_selected < state.volatile.cli_input_history.len() - 1 {
                state.volatile.cli_input_selected += 1;
            }
            if state.volatile.cli_input_history.len() > state.volatile.cli_input_selected {
                state.ui.cmd_line_input =
                    state.volatile.cli_input_history[state.volatile.cli_input_selected].clone();
            }
        }

        if (button_clicked || enter_pressed) && state.ui.cmd_line_input.len() >= 2 {
            // todo: Error color
            state.ui.cmd_line_output =
                match cli::handle_cmd(state, scene, engine_updates, redraw, reset_cam) {
                    Ok(out) => {
                        state.ui.cmd_line_out_is_err = false;
                        out
                    }
                    Err(e) => {
                        eprintln!("Error processing command");
                        state.ui.cmd_line_out_is_err = true;
                        e.to_string()
                    }
                };

            state.ui.cmd_line_input = String::new();
            // Compensates for the default lose focus behavior; we still want the cursor to remain here.
            edit_resp.request_focus();
        }
    });
}

fn docking(
    state: &mut State,
    scene: &mut Scene,
    redraw: &mut bool,
    reset_cam: bool,
    engine_updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    let (Some(mol), Some(lig)) = (&state.molecule, &mut state.ligand) else {
        return;
    };

    let mut docking_posit_update = None;

    ui.horizontal(|ui| {
        if ui.button("Find sites").clicked() {
            let sites = find_docking_sites(mol);
            for site in sites {
                println!("Docking site: {:?}", site);
            }
        }

        if ui.button("Dock").clicked() {
            // todo: Ideally move the camera to the docking site prior to docking. You could do this
            // todo by deferring the docking below to the next frame.

            let (pose, binding_energy) = find_optimal_pose(
                &state.dev,
                state.volatile.docking_setup.as_ref().unwrap(),
                lig,
            );

            lig.pose = pose;
            lig.atom_posits = lig.position_atoms(None);

            {
                let lig_pos: Vec3 = lig.position_atoms(None)[lig.anchor_atom].into();
                let ctr: Vec3 = mol.center.into();

                cam_look_at_outside(&mut scene.camera, lig_pos, ctr);

                engine_updates.camera = true;
                state.ui.cam_snapshot = None;
            }

            // Allow the user to select the autodock executable.
            // if state.to_save.autodock_vina_path.is_none() {
            //     state.volatile.autodock_path_dialog.pick_file();
            // }
            // dock_with_vina(mol, ligand, &state.to_save.autodock_vina_path);
            *redraw = true;
        }

        if ui.button("Docking energy").clicked() {
            let poses = vec![lig.pose.clone()];
            let mut lig_posits = Vec::with_capacity(poses.len());
            // let mut partial_charges_lig = Vec::with_capacity(poses.len());

            for pose in poses {
                let posits_this_pose: Vec<_> = lig
                    .position_atoms(Some(&pose))
                    .iter()
                    .map(|p| (*p).into())
                    .collect();

                // partial_charges_lig.push(create_partial_charges(
                //     &ligand.molecule.atoms,
                //     Some(&posits_this_pose),
                // ));
                lig_posits.push(posits_this_pose);
            }

            state.ui.binding_energy_disp = calc_binding_energy(
                state.volatile.docking_setup.as_ref().unwrap(),
                lig,
                &lig_posits[0],
            );
        }

        ui.add_space(COL_SPACING);

        // if ui.button("Save PDBQT").clicked() {
        //     state.volatile.dialogs.save_pdbqt.pick_directory();
        // }

        // if ui.button("Dock (Vina)").clicked() {
        //     let tgt = state.molecule.as_ref().unwrap();
        //     // Allow the user to select the autodock executable.
        //     if state.to_save.autodock_vina_path.is_none() {
        //         state.volatile.dialogs.autodock_path.pick_file();
        //     }
        //     dock_with_vina(tgt, ligand, &state.to_save.autodock_vina_path);
        //     *redraw = true;
        // }

        // todo: Make this automatic A/R. For not a button
        if ui.button("Site mesh").clicked() {
            // let (mesh, edges) = find_docking_site_surface(mol, &ligand.docking_site);

            // scene.meshes[MESH_DOCKING_SURFACE] = mesh;

            // todo: You must remove prev entities of it too! Do you need an entity ID for this? Likely.
            // todo: Move to the draw module A/R.
            let mut entity = Entity::new(
                MESH_DOCKING_SURFACE,
                Vec3::new_zero(),
                Quaternion::new_identity(),
                1.,
                COLOR_DOCKING_SITE_MESH,
                0.5,
            );
            entity.opacity = 0.8;
            scene.entities.push(entity);

            engine_updates.meshes = true;
            engine_updates.entities = true;
        }

        ui.add_space(COL_SPACING);

        ui.label("Docking site setup:");
        ui.label("Center:");

        let mut docking_init_changed = false;

        if ui
            .add(TextEdit::singleline(&mut state.ui.docking_site_x).desired_width(30.))
            .changed()
        {
            if let Ok(v) = state.ui.docking_site_x.parse::<f64>() {
                lig.docking_site.site_center.x = v;
                docking_init_changed = true;
            }
        }
        if ui
            .add(TextEdit::singleline(&mut state.ui.docking_site_y).desired_width(30.))
            .changed()
        {
            if let Ok(v) = state.ui.docking_site_y.parse::<f64>() {
                lig.docking_site.site_center.y = v;
                docking_init_changed = true;
            }
        }
        if ui
            .add(TextEdit::singleline(&mut state.ui.docking_site_z).desired_width(30.))
            .changed()
        {
            if let Ok(v) = state.ui.docking_site_z.parse::<f64>() {
                lig.docking_site.site_center.z = v;
                docking_init_changed = true;
            }
        }

        // todo: Consider a slider.
        ui.label("Size:");
        if ui
            .add(TextEdit::singleline(&mut state.ui.docking_site_size).desired_width(30.))
            .changed()
        {
            if let Ok(v) = state.ui.docking_site_size.parse::<f64>() {
                lig.docking_site.site_radius = v;
                docking_init_changed = true;
            }
        }

        if let Some(mol) = &state.molecule {
            for res in &mol.het_residues {
                // Note: This is crude.
                // todo: You have this very wide thresh because you are missing H on hetero; add those!
                // todo: Match the orientation.
                // if (res.atoms.len() - lig.molecule.atoms.len()) < 5 {
                if (res.atoms.len() as i16 - lig.molecule.atoms.len() as i16).abs() < 22 {
                    // todo: Don't list multiple; pick teh closest, at least in len.
                    let name = match &res.res_type {
                        ResidueType::Other(name) => name,
                        _ => "hetero residue",
                    };
                    ui.add_space(COL_SPACING / 2.);

                    if ui
                        .button(RichText::new(format!("Center on {name}")).color(COLOR_HIGHLIGHT))
                        .clicked()
                    {
                        // todo: Pick center-of-mass atom, or better yet, match it to the anchor atom.
                        let posit = mol.atoms[res.atoms[0]].posit;
                        docking_posit_update = Some(posit);
                        docking_init_changed = true;
                    }
                }
            }
        }

        if state.selection != Selection::None {
            ui.add_space(COL_SPACING / 2.);

            if ui
                .button(RichText::new("Center on sel").color(COLOR_HIGHLIGHT))
                .clicked()
            {
                let atom_sel = mol.get_sel_atom(&state.selection);

                if let Some(atom) = atom_sel {
                    docking_posit_update = Some(atom.posit);
                    docking_init_changed = true;
                }
            }
        }
        if docking_init_changed {
            *redraw = true;
            set_docking_light(scene, Some(&lig.docking_site));
            // todo: Hardcoded as some.
            engine_updates.lighting = true;
        }

        // ui.add_space(COL_SPACING);

        // ui.label(RichText::new("🔘AV").color(active_color(state.ui.autodock_path_valid)))
        //     .on_hover_text("Autodock Vina available (Docking)");
    });

    if let Some(posit) = docking_posit_update {
        state.update_docking_site(posit);
        state.update_save_prefs();
    }

    ui.horizontal(|ui| {
        if let Some(lig) = &mut state.ligand {
            if ui.button("Build VDW sim").clicked() {
                state.volatile.snapshots = build_vdw_dynamics(
                    &state.dev,
                    lig,
                    state.volatile.docking_setup.as_ref().unwrap(),
                    false,
                    1_500,
                );

                state.ui.current_snapshot = 0;
            }

            if !state.volatile.snapshots.is_empty() {
                ui.add_space(ROW_SPACING);

                let snapshot_prev = state.ui.current_snapshot;
                ui.spacing_mut().slider_width = ui.available_width() - 100.;
                ui.add(Slider::new(
                    &mut state.ui.current_snapshot,
                    0..=state.volatile.snapshots.len() - 1,
                ));

                if state.ui.current_snapshot != snapshot_prev {
                    change_snapshot(
                        &mut scene.entities,
                        lig,
                        &Vec::new(),
                        &mut state.ui.binding_energy_disp,
                        &state.volatile.snapshots[state.ui.current_snapshot],
                    );

                    engine_updates.entities = true;
                }
            }
        }
    });
}

fn residue_search(state: &mut State, scene: &mut Scene, redraw: &mut bool, ui: &mut Ui) {
    ui.horizontal(|ui| {
        // let sel_prev = &state.selection;
        ui.label("Find residue:");
        if ui
            .add(TextEdit::singleline(&mut state.ui.residue_search).desired_width(60.))
            .changed()
        {
            select_from_search(state);
            *redraw = true;
        }

        // if sel_prev != &state.selection {
        //     *redraw = true;
        // }

        // todo: for UI adccessbility
        // if key_up_is_down {
        //     up_button.highlight();
        // }
        if state.molecule.is_some() {
            if ui
                .button("Prev AA")
                .on_hover_text("Hotkey: Left arrow")
                .clicked()
            {
                cycle_res_selected(state, scene, true);
                *redraw = true;
            }
            // todo: DRY
            if ui
                .button("Next AA")
                .on_hover_text("Hotkey: Right arrow")
                .clicked()
            {
                cycle_res_selected(state, scene, false);
                *redraw = true;
            }

            ui.add_space(COL_SPACING * 2.);

            let dock_tools_text = if state.ui.show_docking_tools {
                "Hide docking tools"
            } else {
                "Show docking tools (Broken/WIP)"
            };

            if ui.button(RichText::new(dock_tools_text)).clicked() {
                state.ui.show_docking_tools = !state.ui.show_docking_tools;
            }

            ui.add_space(COL_SPACING / 2.);

            let dock_seq_text = if state.ui.show_aa_seq {
                "Hide seq"
            } else {
                "Show seq"
            };

            if ui.button(RichText::new(dock_seq_text)).clicked() {
                state.ui.show_aa_seq = !state.ui.show_aa_seq;
            }
        }
    });
}

fn add_aa_seq(seq_text: &str, ui: &mut Ui) {
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(seq_text).color(Color32::LIGHT_BLUE));
    });
}

fn selection_section(
    state: &mut State,
    scene: &mut Scene,
    redraw: &mut bool,
    engine_updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    // todo: DRY with view.
    ui.horizontal(|ui| {
        ui.label("View/Select:");
        let prev_view = state.ui.view_sel_level;
        ComboBox::from_id_salt(1)
            .width(80.)
            .selected_text(state.ui.view_sel_level.to_string())
            .show_ui(ui, |ui| {
                for view in &[ViewSelLevel::Atom, ViewSelLevel::Residue] {
                    ui.selectable_value(&mut state.ui.view_sel_level, *view, view.to_string());
                }
            });

        if state.ui.view_sel_level != prev_view {
            *redraw = true;
            // Kludge to prevent surprising behavior.
            state.selection = Selection::None;
        }

        if state.ui.view_sel_level == ViewSelLevel::Residue {
            ui.add_space(COL_SPACING / 2.);
            let dock_seq_text = if state.ui.res_color_by_index {
                "Color by res AA"
            } else {
                "Color by res #"
            };

            if ui.button(RichText::new(dock_seq_text)).clicked() {
                state.ui.res_color_by_index = !state.ui.res_color_by_index;
                state.ui.view_sel_level = ViewSelLevel::Residue;
                *redraw = true;
            }
        }

        ui.add_space(COL_SPACING);

        ui.label("Nearby sel only:");
        if ui.checkbox(&mut state.ui.show_near_sel_only, "").changed() {
            *redraw = true;

            // todo: For now, only allow one of near sel/lig
            if state.ui.show_near_sel_only {
                state.ui.show_near_lig_only = false
            }
        }

        if state.ligand.is_some() {
            ui.label("Nearby lig only:");
            if ui.checkbox(&mut state.ui.show_near_lig_only, "").changed() {
                *redraw = true;

                // todo: For now, only allow one of near sel/lig
                if state.ui.show_near_lig_only {
                    state.ui.show_near_sel_only = false
                }
            }
        }

        if state.ui.show_near_sel_only || state.ui.show_near_lig_only {
            ui.label("Dist:");
            let dist_prev = state.ui.nearby_dist_thresh;
            ui.add(Slider::new(
                &mut state.ui.nearby_dist_thresh,
                NEARBY_THRESH_MIN..=NEARBY_THRESH_MAX,
            ));

            if state.ui.nearby_dist_thresh != dist_prev {
                *redraw = true;
            }
        }

        if let Some(mol) = &state.molecule {
            ui.add_space(COL_SPACING);

            if state.selection != Selection::None {
                if ui
                    .button(RichText::new("Move cam to sel").color(COLOR_HIGHLIGHT))
                    .clicked()
                {
                    let atom_sel = mol.get_sel_atom(&state.selection);

                    if let Some(atom) = atom_sel {
                        cam_look_at(&mut scene.camera, atom.posit);
                        engine_updates.camera = true;
                        state.ui.cam_snapshot = None;
                    }
                }
            }

            if let Some(lig) = &state.ligand {
                ui.add_space(COL_SPACING / 2.);
                if ui
                    .button(RichText::new("Move cam to lig").color(COLOR_HIGHLIGHT))
                    .clicked()
                {
                    if lig.anchor_atom >= lig.molecule.atoms.len() {
                        handle_err(
                            &mut state.ui,
                            "Problem positioning ligand atoms. Len shorter than anchor.".to_owned(),
                        );
                    } else {
                        let lig_pos: Vec3 = lig.position_atoms(None)[lig.anchor_atom].into();
                        let ctr: Vec3 = mol.center.into();

                        cam_look_at_outside(&mut scene.camera, lig_pos, ctr);

                        engine_updates.camera = true;
                        state.ui.cam_snapshot = None;
                    }
                }
            }

            ui.add_space(COL_SPACING / 2.);
            selected_data(mol, &state.selection, ui);
        }
    });
}

fn mol_descrip(mol: &Molecule, ui: &mut Ui) {
    ui.heading(RichText::new(mol.ident.clone()).color(Color32::GOLD));

    ui.label(format!("{} atoms", mol.atoms.len()));

    if let Some(metadata) = &mol.metadata {
        // Limit size to prevent UI problems.
        let mut title: String = metadata
            .prim_cit_title
            .chars()
            .take(MAX_TITLE_LEN)
            .collect();
        if title.len() != metadata.prim_cit_title.len() {
            title += "...";
        }
        ui.label(RichText::new(title).color(Color32::WHITE).size(12.));
    }

    if mol.ident.len() <= 5 {
        // todo: You likely need a better approach.
        if ui.button("View on RCSB").clicked() {
            rcsb::open_overview(&mol.ident);
        }
    }

    if let Some(id) = &mol.drugbank_id {
        if ui.button("View on Drugbank").clicked() {
            drugbank::open_overview(id);
        }
    }

    if let Some(id) = mol.pubchem_cid {
        if ui.button("View on PubChem").clicked() {
            pubchem::open_overview(id);
        }
    }
}

fn view_settings(
    state: &mut State,
    entities: &mut Vec<Entity>,
    engine_updates: &mut EngineUpdates,
    redraw: &mut bool,
    ui: &mut Ui,
) {
    ui.horizontal(|ui| {
        ui.label("View:");
        let prev_view = state.ui.mol_view;
        ComboBox::from_id_salt(0)
            .width(80.)
            .selected_text(state.ui.mol_view.to_string())
            .show_ui(ui, |ui| {
                for view in &[
                    MoleculeView::Backbone,
                    MoleculeView::Sticks,
                    MoleculeView::BallAndStick,
                    // MoleculeView::Cartoon,
                    MoleculeView::SpaceFill,
                    // MoleculeView::Surface, // Partially-implemented, but broken/crashes.
                    // MoleculeView::Mesh,
                    MoleculeView::Dots,
                ] {
                    ui.selectable_value(&mut state.ui.mol_view, *view, view.to_string());
                }
            });

        if state.ui.mol_view != prev_view {
            *redraw = true;
        }

        ui.add_space(COL_SPACING);

        ui.label("Vis:");

        vis_check(
            &mut state.ui.visibility.hide_non_hetero,
            "Peptide",
            ui,
            redraw,
        );
        vis_check(&mut state.ui.visibility.hide_hetero, "Hetero", ui, redraw);

        ui.add_space(COL_SPACING / 2.);

        if !state.ui.visibility.hide_non_hetero {
            // Subset of peptide.
            vis_check(
                &mut state.ui.visibility.hide_sidechains,
                "Sidechains",
                ui,
                redraw,
            );
        }

        vis_check(&mut state.ui.visibility.hide_hydrogen, "H", ui, redraw);

        if !state.ui.visibility.hide_hetero {
            // Subset of hetero.
            vis_check(&mut state.ui.visibility.hide_water, "Water", ui, redraw);
        }

        if state.ligand.is_some() {
            vis_check(&mut state.ui.visibility.hide_ligand, "Lig", ui, redraw);
        }

        vis_check(&mut state.ui.visibility.hide_h_bonds, "H bonds", ui, redraw);
        // vis_check(&mut state.ui.visibility.dim_peptide, "Dim peptide", ui, redraw);

        if state.ligand.is_some() {
            ui.add_space(COL_SPACING / 2.);
            // Not using `vis_check` for this because its semantics are inverted.
            let color = active_color(state.ui.visibility.dim_peptide);
            if ui
                .button(RichText::new("Dim peptide").color(color))
                .clicked()
            {
                state.ui.visibility.dim_peptide = !state.ui.visibility.dim_peptide;
                *redraw = true;
            }
        }

        if let Some(mol) = &state.molecule {
            if let Some(dens) = &mol.elec_density {
                let mut redraw_dens = false;
                vis_check(
                    &mut state.ui.visibility.hide_density,
                    "Density",
                    ui,
                    &mut redraw_dens,
                );

                if redraw_dens {
                    if state.ui.visibility.hide_density {
                        entities.retain(|ent| ent.class != EntityType::Density as u32);
                    } else {
                        draw_density(entities, dens);
                    }
                    engine_updates.entities = true;
                }

                let mut redraw_dens_surface = false;
                vis_check(
                    &mut state.ui.visibility.hide_density_surface,
                    "Density sfc",
                    ui,
                    &mut redraw_dens_surface,
                );

                if !state.ui.visibility.hide_density_surface {
                    let iso_prev = state.ui.density_iso_level;

                    ui.spacing_mut().slider_width = 200.;
                    ui.add(Slider::new(
                        &mut state.ui.density_iso_level,
                        // todo: Consts for these
                        -0.0..=1.0,
                    ));
                    if state.ui.density_iso_level != iso_prev {
                        state.volatile.make_density_mesh = true;
                    }
                }

                // todo
                if redraw_dens_surface {
                    if state.ui.visibility.hide_density_surface {
                        entities.retain(|ent| ent.class != EntityType::DensitySurface as u32);
                    } else {
                        draw_density_surface(entities);
                    }
                    engine_updates.entities = true;
                }
            }
        }
    });
}

fn settings(state: &mut State, scene: &mut Scene, ui: &mut Ui) {
    if state.ui.show_settings {
        ui.horizontal(|ui| {
            ui.heading("Settings");
            ui.add_space(COL_SPACING);
            // todo: Make this consistent with your other controls.
            ui.label("MSAA (Restart the program to take effect):");

            let msaa_prev = state.to_save.msaa;
            ComboBox::from_id_salt(10)
                .width(40.)
                .selected_text(state.to_save.msaa.to_str())
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut state.to_save.msaa,
                        MsaaSetting::None,
                        MsaaSetting::None.to_str(),
                    );
                    ui.selectable_value(
                        &mut state.to_save.msaa,
                        MsaaSetting::Four,
                        MsaaSetting::Four.to_str(),
                    );
                });

            if state.to_save.msaa != msaa_prev {
                state.update_save_prefs();
            }

            ui.add_space(COL_SPACING);
            ui.label("Movement speed:");
            if ui
                .add(TextEdit::singleline(&mut state.ui.movement_speed_input).desired_width(32.))
                .changed()
            {
                if let Ok(v) = &mut state.ui.movement_speed_input.parse::<u8>() {
                    state.to_save.movement_speed = *v;
                    scene.input_settings.move_sens = *v as f32;

                    state.update_save_prefs();
                } else {
                    // reset
                    state.ui.movement_speed_input = state.to_save.movement_speed.to_string();
                }
            }

            ui.add_space(COL_SPACING / 2.);
            ui.label("Rotation sensitivity:");
            if ui
                .add(TextEdit::singleline(&mut state.ui.rotation_sens_input).desired_width(32.))
                .changed()
            {
                if let Ok(v) = &mut state.ui.rotation_sens_input.parse::<u8>() {
                    state.to_save.rotation_sens = *v;
                    scene.input_settings.rotate_sens = *v as f32 / 100.;

                    state.update_save_prefs();
                } else {
                    // reset
                    state.ui.rotation_sens_input = state.to_save.rotation_sens.to_string();
                }
            }

            ui.add_space(COL_SPACING / 2.);
            if ui.button("Reset sensitivities").clicked() {
                state.to_save.movement_speed = MOVEMENT_SENS as u8;
                state.ui.movement_speed_input = state.to_save.movement_speed.to_string();
                scene.input_settings.move_sens = MOVEMENT_SENS;

                state.to_save.rotation_sens = (ROTATE_SENS * 100.) as u8;
                state.ui.rotation_sens_input = state.to_save.rotation_sens.to_string();
                scene.input_settings.rotate_sens = ROTATE_SENS;

                state.update_save_prefs();
            }
        });
        ui.add_space(ROW_SPACING * 2.);
    }
}

/// This function draws the (immediate-mode) GUI.
/// [UI items](https://docs.rs/egui/latest/egui/struct.Ui.html)
pub fn ui_handler(state: &mut State, ctx: &Context, scene: &mut Scene) -> EngineUpdates {
    let mut engine_updates = EngineUpdates::default();

    check_prefs_save(state);

    // return  engine_updates;
    let mut redraw = false;
    let mut reset_cam = false;

    // For getting DT for certain buttons when held. Does not seem to be the same as the 3D render DT.
    let start = Instant::now();

    TopBottomPanel::top("0").show(ctx, |ui| {
        ui.spacing_mut().slider_width = 120.;

        handle_input(state, ui, &mut redraw, &mut reset_cam, &mut engine_updates);

        settings(state, scene, ui);

        ui.horizontal_wrapped(|ui| {
            let color_settings = if state.ui.show_settings {
                Color32::LIGHT_RED
            } else {
                Color32::GRAY
            };
            if ui
                .button(RichText::new("⚙").color(color_settings))
                .clicked()
            {
                state.ui.show_settings = !state.ui.show_settings;
            }

            let metadata_loaded = false; // avoids borrow error.
            if let Some(mol) = &mut state.molecule {
                mol_descrip(mol, ui);

                if ui.button("Close").clicked() {
                    close_mol(state, scene, &mut engine_updates);
                }
                ui.add_space(COL_SPACING);
            }

            let color_open_tools = if state.molecule.is_none() {
                Color32::GOLD
            } else {
                COLOR_INACTIVE
            };
            if ui
                .button(RichText::new("Open").color(color_open_tools))
                .clicked()
            {
                state.volatile.dialogs.load.pick_file();
            }

            if let Some(mol) = &mut state.molecule {
                if state.pdb.is_some() {
                    if ui.button("Save").clicked() {
                        let extension = "cif";

                        let filename = {
                            let name = if mol.ident.is_empty() {
                                "molecule".to_string()
                            } else {
                                mol.ident.clone()
                            };
                            format!("{name}.{extension}")
                        };

                        state.volatile.dialogs.save.config_mut().default_file_name =
                            filename.to_string();
                        state.volatile.dialogs.save.save_file();
                    }
                }

                // todo: Move these A/R. LIkely in a sub menu.
                if let Some(data) = &mol.rcsb_files_avail {
                    if data.structure_factors {
                        if ui
                            .button(RichText::new("SF").color(COLOR_HIGHLIGHT))
                            .clicked()
                        {
                            match rcsb::load_structure_factors_cif(&mol.ident) {
                                Ok(data) => {
                                    // println!("SF data: {:?}", data);
                                }
                                Err(_) => {
                                    let msg = format!(
                                        "Error loading RCSB structure factors for {:?}",
                                        &mol.ident
                                    );

                                    handle_err(&mut state.ui, msg);
                                }
                            }
                        }
                    }
                    if data.validation_2fo_fc {
                        if ui
                            .button(RichText::new("2fo-fc").color(COLOR_HIGHLIGHT))
                            .clicked()
                        {
                            match rcsb::load_validation_2fo_fc_cif(&mol.ident) {
                                Ok(data) => {
                                    // println!("SF data: {:?}", data);
                                }
                                Err(_) => {
                                    let msg = format!(
                                        "Error loading RCSB 2fo-fc map for {:?}",
                                        &mol.ident
                                    );
                                    handle_err(&mut state.ui, msg);
                                }
                            }
                        }
                        // todo: Map download directly from RCSB if avail?

                        if ui
                            .button(RichText::new("Load map").color(COLOR_HIGHLIGHT))
                            .clicked()
                        {
                            // todo: For now, we rely on Gemmi being available on the Path.
                            // todo: We will eventually get our own reflections loader working.

                            match density_from_rcsb_gemmi(&mol.ident) {
                                Ok((hdr, dens)) => {
                                    println!(
                                        "Succsesfully loaded density data from RSCB using Gemmi."
                                    );

                                    mol.elec_density_header = Some(hdr);

                                    let elec_dens = dens
                                        .iter()
                                        .map(|d| ElectronDensity {
                                            coords: d.coords,
                                            density: d.density,
                                        })
                                        .collect();
                                    mol.elec_density = Some(elec_dens);

                                    let dm = density_from_rcsb_gemmi2(&mol.ident).unwrap(); // todo unwrap temp.
                                    mol.density_map = Some(dm);

                                    state.volatile.make_density_mesh = true;

                                    if !state.ui.visibility.hide_density {
                                        draw_density(
                                            &mut scene.entities,
                                            mol.elec_density.as_ref().unwrap(),
                                        );

                                        engine_updates.entities = true;
                                    }
                                }
                                Err(e) => {
                                    let msg = format!(
                                        "Error loading reflections and density map data.: {e:?}"
                                    );
                                    handle_err(&mut state.ui, msg);
                                }
                            }

                            // match ReflectionsData::load_from_rcsb(&mol.ident) {
                            //     Ok(d) => {
                            //         println!("Successfully loaded reflections and density map data");
                            //
                            //         let density = compute_density_grid(&d);
                            //
                            //         mol.reflections_data = Some(d);
                            //         mol.elec_density = Some(density);
                            //
                            //         // todo: Update A/R based on how we visualize this.
                            //         redraw = true;
                            //     }
                            //     Err(e) => {
                            //         eprintln!("Error loading reflections and density map data.: {e:?}");
                            //     }
                            // }

                            // match ReflectionsData::load_from_rcsb(&mol.ident) {
                            //     Ok(d) => {
                            //         println!("Successfully loaded reflections and density map data");
                            //
                            //         let density = compute_density_grid(&d);
                            //
                            //         mol.reflections_data = Some(d);
                            //         mol.elec_density = Some(density);
                            //
                            //         // todo: Update A/R based on how we visualize this.
                            //         redraw = true;
                            //     }
                            //     Err(e) => {
                            //         eprintln!("Error loading reflections and density map data.: {e:?}");
                            //     }
                            // }
                        }
                    }

                    if data.validation_fo_fc {
                        if ui
                            .button(RichText::new("fo-fc").color(COLOR_HIGHLIGHT))
                            .clicked()
                        {
                            match rcsb::load_validation_fo_fc_cif(&mol.ident) {
                                Ok(data) => {
                                    // println!("SF data: {:?}", data);
                                }
                                Err(_) => {
                                    let msg = format!(
                                        "Error loading RCSB fo-fc map for {:?}",
                                        &mol.ident
                                    );
                                    handle_err(&mut state.ui, msg);
                                }
                            }
                        }
                    }

                    if data.validation {
                        if ui
                            .button(RichText::new("Val").color(COLOR_HIGHLIGHT))
                            .clicked()
                        {
                            match rcsb::load_validation_cif(&mol.ident) {
                                Ok(data) => {
                                    // println!("VAL DATA: {:?}", data);
                                }
                                Err(_) => {
                                    let msg = format!(
                                        "Error loading RCSB validation for {:?}",
                                        &mol.ident
                                    );
                                    handle_err(&mut state.ui, msg);
                                }
                            }
                        }
                    }
                }
            }

            if let Some(lig) = &state.ligand {
                if ui.button("Save lig").clicked() {
                    // todo: Allow saving as SDF, PDBQT, or mol2 here
                    let extension = "sdf";

                    let filename = {
                        let name = if lig.molecule.ident.is_empty() {
                            "molecule".to_string()
                        } else {
                            lig.molecule.ident.clone()
                        };
                        format!("{name}.{extension}")
                    };

                    // state
                    //     .volatile
                    //     .dialogs
                    //     .save_ligand
                    //     .config_mut()
                    //     .default_file_name = filename.to_string();
                    // state.volatile.dialogs.save_ligand.save_file();

                    state.volatile.dialogs.save.config_mut().default_file_name =
                        filename.to_string();

                    state.volatile.dialogs.save.save_file();
                }
            }

            // if ui.button("Get RCSB").clicked() {
            //     match load_pdb_metadata(&mol.ident) {
            //         Ok(d) => {
            //             println!("Metadata loaded: {d:?}");
            //             mol.metadata = Some(d);
            //             metadata_loaded = true;
            //         },
            //         Err(_) => eprintln!("Error getting PDB metadata"),
            //     }
            // }

            if metadata_loaded {
                state.update_save_prefs();
            }

            ui.add_space(COL_SPACING);

            ui.add_space(COL_SPACING);
            ui.label(RichText::new("Query databases (ident):").color(color_open_tools));
            let edit_resp = ui.add(TextEdit::singleline(&mut state.ui.db_input).desired_width(40.));

            if state.ui.db_input.len() >= 4 {
                let enter_pressed =
                    edit_resp.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter));
                let button_clicked = ui.button("Download from RCSB").clicked();

                // if response.lost_focus() && (button_clicked || enter_pressed)
                if (button_clicked || enter_pressed) && state.ui.db_input.trim().len() == 4 {
                    let ident = state.ui.db_input.clone().trim().to_owned();
                    query_rcsb(
                        &ident,
                        state,
                        scene,
                        &mut engine_updates,
                        &mut redraw,
                        &mut reset_cam,
                    );
                }

                if state.ui.db_input.to_uppercase().starts_with("DB") {
                    if ui.button("Download from DrugBank").clicked() {
                        match load_sdf_drugbank(&state.ui.db_input) {
                            // todo: Load as ligand for now.
                            Ok(mol) => {
                                // state.pdb = None;

                                state.ligand = Some(Ligand::new(mol));

                                state.update_from_prefs();

                                redraw = true;
                                reset_cam = true;
                            }
                            Err(_e) => {
                                let msg = "Error loading SDF file".to_owned();
                                handle_err(&mut state.ui, msg);
                            }
                        }
                    }
                }

                if ui.button("Download from PubChem").clicked() {
                    match load_sdf_pubchem(&state.ui.db_input) {
                        // todo: Load as ligand for now.
                        Ok(mol) => {
                            // state.pdb = None;

                            state.ligand = Some(Ligand::new(mol));

                            state.update_from_prefs();

                            redraw = true;
                            reset_cam = true;
                        }
                        Err(_e) => {
                            let msg = "Error loading SDF file".to_owned();
                            handle_err(&mut state.ui, msg);
                        }
                    }
                }
            }

            if state.molecule.is_none() && state.ligand.is_none() {
                ui.add_space(COL_SPACING / 2.);
                if ui
                    .button(RichText::new("I'm feeling lucky 🍀").color(color_open_tools))
                    .clicked()
                {
                    if let Ok(ident) = rcsb::get_newly_released() {
                        query_rcsb(
                            &ident,
                            state,
                            scene,
                            &mut engine_updates,
                            &mut redraw,
                            &mut reset_cam,
                        );
                    }
                }
            }
        });

        ui.add_space(ROW_SPACING);
        let mut close_ligand = false; // to avoid borrow error.
        if let Some(ligand) = &mut state.ligand {
            ui.horizontal(|ui| {
                mol_descrip(&ligand.molecule, ui);

                if ui.button("Close lig").clicked() {
                    close_ligand = true;
                }

                ui.add_space(COL_SPACING);

                ui.label("Rotate bonds:");
                for i in 0..ligand.flexible_bonds.len() {
                    if let ConformationType::Flexible { torsions } =
                        &mut ligand.pose.conformation_type
                    {
                        if ui.button(format!("{i}")).clicked() {
                            torsions[i].dihedral_angle =
                                (torsions[i].dihedral_angle + TAU / 64.) % TAU;

                            ligand.atom_posits = ligand.position_atoms(None);
                            redraw = true;
                        }
                    }
                }

                ui.add_space(COL_SPACING);

                if let Some(energy) = &state.ui.binding_energy_disp {
                    ui.label(format!("{:.2?}", energy)); // todo placeholder.
                }

                // todo: temp, or at least temp here
                ui.label(format!("Lig pos: {}", ligand.pose.anchor_posit));
                ui.label(format!("Lig or: {}", ligand.pose.orientation));
                if let ConformationType::Flexible { torsions } = &ligand.pose.conformation_type {
                    for torsion in torsions {
                        ui.label(format!("T: {:.3}", torsion.dihedral_angle));
                    }
                }
            });
        }

        ui.add_space(ROW_SPACING);
        selection_section(state, scene, &mut redraw, &mut engine_updates, ui);

        ui.add_space(ROW_SPACING);

        ui.horizontal_wrapped(|ui| {
            cam_controls(scene, state, &mut engine_updates, ui);

            ui.add_space(COL_SPACING);

            cam_snapshots(state, scene, &mut engine_updates, ui);
        });

        ui.add_space(ROW_SPACING);

        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                view_settings(
                    state,
                    &mut scene.entities,
                    &mut engine_updates,
                    &mut redraw,
                    ui,
                );
                ui.add_space(ROW_SPACING);
                chain_selector(state, &mut redraw, ui);

                // todo: Show hide based on AaCategory? i.e. residue.amino_acid.category(). Hydrophilic, acidic etc.

                residue_selector(state, scene, &mut redraw, ui);
            });
        });

        ui.add_space(ROW_SPACING);

        residue_search(state, scene, &mut redraw, ui);

        if state.ui.show_docking_tools {
            ui.add_space(ROW_SPACING);

            docking(
                state,
                scene,
                &mut redraw,
                reset_cam,
                &mut engine_updates,
                ui,
            );
        }

        // todo: Allow switching between chains and secondary-structure features here.

        ui.add_space(ROW_SPACING / 2.);

        if state.ui.show_aa_seq {
            if state.molecule.is_some() {
                add_aa_seq(&state.volatile.aa_seq_text, ui);
            }
        }

        // todo: Move A/r.
        draw_cli(
            state,
            scene,
            &mut engine_updates,
            &mut redraw,
            &mut reset_cam,
            ui,
        );

        // -------UI above; clean-up items (based on flags) below

        if close_ligand {
            close_lig(state, scene, &mut engine_updates);
        }

        if let Some(path) = &state.volatile.dialogs.load.take_picked() {
            if let Err(e) = load_file(
                path,
                state,
                &mut redraw,
                &mut reset_cam,
                &mut engine_updates,
            ) {
                handle_err(&mut state.ui, e.to_string());
            }

            set_flashlight(scene);
            engine_updates.lighting = true;
        }

        if let Some(path) = &state.volatile.dialogs.save.take_picked() {
            state.save(path).ok();
        }

        if let Some(path) = &state.volatile.dialogs.autodock_path.take_picked() {
            state.ui.autodock_path_valid = check_adv_avail(path);
            if state.ui.autodock_path_valid {
                state.to_save.autodock_vina_path = Some(path.to_owned());
                state.update_save_prefs();
            }
        }

        // if let Some(path_dir) = &state.volatile.dialogs.save_pdbqt.take_picked() {
        //     if let Some(mol) = &mut state.molecule {
        //         let filename = format!("{}_target.pdbqt", mol.ident);
        //         let path = Path::new(path_dir).join(filename);
        //
        //         // todo: You will likely need to add charges earlier, so you can view their data in the UI.
        //         // setup_partial_charges(&mut mol.atoms, PartialChargeType::Gasteiger);
        //         // create_partial_charges(&mut mol.atoms);
        //
        //         if mol.save_pdbqt(&path, None).is_err() {
        //             eprintln!("Error saving PDBQT target");
        //         }
        //     }
        //
        //     if let Some(lig) = &mut state.ligand {
        //         let filename = format!("{}_ligand.pdbqt", lig.molecule.ident);
        //         let path = Path::new(path_dir).join(filename);
        //
        //         // create_partial_charges(&mut lig.molecule.atoms);
        //         // setup_partial_charges(&mut lig.molecule.atoms, PartialChargeType::Gasteiger);
        //
        //         if lig.molecule.save_pdbqt(&path, None).is_err() {
        //             eprintln!("Error saving PDBQT ligand");
        //         }
        //     }
        // }

        if redraw {
            draw_molecule(state, scene);
            draw_ligand(state, scene);

            if let Some(mol) = &state.molecule {
                set_window_title(&mol.ident, scene);
            }

            engine_updates.entities = true;

            // For docking light, but may be overkill here.
            if state.ligand.is_some() {
                engine_updates.lighting = true;
            }
        }

        // Perform cleanup.
        if reset_cam {
            if let Some(mol) = &state.molecule {
                reset_camera(scene, &mut state.ui.view_depth, &mut engine_updates, mol);
            }
        }
    });

    state.volatile.dialogs.load.update(ctx);
    // state.volatile.dialogs.load_ligand.update(ctx);
    state.volatile.dialogs.save.update(ctx);
    // state.volatile.dialogs.save_ligand.update(ctx);
    state.volatile.dialogs.autodock_path.update(ctx);
    // state.volatile.dialogs.save_pdbqt.update(ctx);

    // todo: Appropriate place for this?
    if state.volatile.inputs_commanded.inputs_present() {
        set_flashlight(scene);
        engine_updates.lighting = true;
    }

    state.ui.dt = start.elapsed().as_secs_f32();

    if !INIT_COMPLETE.swap(true, Ordering::AcqRel) {
        if state.volatile.ui_height < f32::EPSILON {
            state.volatile.ui_height = ctx.used_size().y;
        }

        // todo: Move to new_mol_loaded code block?
        if let Some(mol) = &state.molecule {
            if let Some(lig) = &state.ligand {
                if lig.anchor_atom >= lig.molecule.atoms.len() {
                    let msg = "Error positioning ligand atoms; anchor outside len".to_owned();
                    handle_err(&mut state.ui, msg);
                } else {
                    let lig_pos: Vec3 = lig.position_atoms(None)[lig.anchor_atom].into();
                    let ctr: Vec3 = mol.center.into();

                    cam_look_at_outside(&mut scene.camera, lig_pos, ctr);

                    engine_updates.camera = true;
                    state.ui.cam_snapshot = None;
                }
            }
            set_static_light(scene, mol.center.into(), mol.size);
        }
    }

    if state.ui.new_mol_loaded {
        state.ui.new_mol_loaded = false;

        if let Some(mol) = &state.molecule {
            reset_camera(scene, &mut state.ui.view_depth, &mut engine_updates, mol);
        }

        set_flashlight(scene);
        engine_updates.lighting = true;
    }

    if state.ui.new_density_loaded {
        state.ui.new_density_loaded = false;
        if let Some(mol) = &state.molecule {
            if !state.ui.visibility.hide_density {
                if let Some(density) = &mol.elec_density {
                    draw_density(&mut scene.entities, density);
                    engine_updates.entities = true;
                }
            }
        }
    }
    if state.volatile.make_density_mesh {
        if let Some(mol) = &state.molecule {
            // if let Some(dens) = &mol.elec_density {

                // todo: Adapt this to your new approach, if it works.
                // let hdr = mol.elec_density_header.as_ref().unwrap();
                // let rect = mol.density_rect.as_ref().unwrap();
            if let Some(rect) = &mol.density_rect {

                /////

                // 2) Grid parameters for MarchingCubes
                let dims = (rect.dims[0], rect.dims[1], rect.dims[2]);        // (nx,ny,nz)

                let size = (
                    (rect.step[0] * rect.dims[0] as f64) as f32,              // Δx * nx  (Å)
                    (rect.step[1] * rect.dims[1] as f64) as f32,
                    (rect.step[2] * rect.dims[2] as f64) as f32,
                );

                // “sampling interval” in the original code is really the number of
                // samples along each axis (= nx,ny,nz), so just cast dims to f32:
                let samples = (
                    rect.dims[0] as f32,
                    rect.dims[1] as f32,
                    rect.dims[2] as f32,
                );
                
                ////
                

                match MarchingCubes::from_gridpoints(
                    dims,
                    size,
                    samples,
                    rect.origin_cart.into(),
                    // (hdr.nx as usize, hdr.ny as usize, hdr.nz as usize),
                    // (hdr.cell[0], hdr.cell[1], hdr.cell[2]),
                    // (hdr.mx as f32, hdr.my as f32, hdr.mz as f32),
                    mol.elec_density.as_ref().unwrap(),
                    state.ui.density_iso_level,
                ) {
                    Ok(mc) => {
                        let mesh = mc.generate(MeshSide::Both);

                        // Convert from `mcubes::Mesh` to `graphics::Mesh`.
                        let vertices = mesh
                            .vertices
                            .iter()
                            .map(|v| Vertex::new(v.posit.to_arr(), v.normal))
                            .collect();

                        scene.meshes[MESH_DENSITY_SURFACE] = Mesh {
                            vertices,
                            indices: mesh.indices,
                            material: 0,
                        };

                        if !state.ui.visibility.hide_density_surface {
                            draw_density_surface(&mut scene.entities);
                        }

                        engine_updates.meshes = true;
                        engine_updates.entities = true;
                    }
                    Err(e) => handle_err(&mut state.ui, e.to_string()),
                }
            }
        }

        state.volatile.make_density_mesh = false;
    }

    if state.volatile.mol_pending_data_avail.is_some() {
        if let Some(mol) = &mut state.molecule {
            if mol.poll_data_avail(&mut state.volatile.mol_pending_data_avail) {
                state.update_save_prefs();
            }
        }
    }

    engine_updates
}
