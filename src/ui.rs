use std::{f32::consts::TAU, path::Path, time::Instant};

use egui::{Color32, ComboBox, Context, RichText, Slider, TextEdit, TopBottomPanel, Ui};
use graphics::{Camera, ControlScheme, EngineUpdates, Entity, RIGHT_VEC, Scene, UP_VEC};
use lin_alg::f32::{Quaternion, Vec3};
use na_seq::AaIdent;

use crate::{
    CamSnapshot, Selection, State, ViewSelLevel,
    docking::{
        ConformationType, calc_binding_energy,
        dynamics_playback::{build_vdw_dynamics, change_snapshot},
        external::check_adv_avail,
        find_optimal_pose,
        find_sites::find_docking_sites,
    },
    download_mols::{load_cif_rcsb, load_sdf_drugbank, load_sdf_pubchem},
    file_io::pdb::save_pdb,
    mol_drawing::{COLOR_DOCKING_SITE_MESH, MoleculeView, draw_ligand, draw_molecule},
    molecule::{Ligand, Molecule, ResidueType},
    rcsb_api::{open_drugbank, open_pdb, open_pubchem},
    render::{
        CAM_INIT_OFFSET, MESH_DOCKING_SURFACE, RENDER_DIST_FAR, RENDER_DIST_NEAR,
        set_docking_light, set_flashlight,
    },
    util::{
        cam_look_at, cam_look_at_outside, check_prefs_save, cycle_res_selected, orbit_center,
        select_from_search,
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

fn load_file(
    path: &Path,
    state: &mut State,
    redraw: &mut bool,
    reset_cam: &mut bool,
    engine_updates: &mut EngineUpdates,
    ligand_load: bool,
) {
    state.open_molecule(path, ligand_load);

    // todo: These only if successful.
    *redraw = true;
    *reset_cam = true;
    engine_updates.entities = true;

    if ligand_load {
        state.to_save.last_ligand_opened = Some(path.to_owned());
    } else {
        state.to_save.last_opened = Some(path.to_owned());
    }

    state.update_save_prefs()
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
                let ligand_load = path
                    .extension()
                    .unwrap_or_default()
                    .to_ascii_lowercase()
                    .to_str()
                    .unwrap_or_default()
                    == "sdf";

                load_file(path, state, redraw, reset_cam, engine_updates, ligand_load);
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

        state
            .cam_snapshots
            .push(CamSnapshot::from_cam(&scene.camera, name));
        state.ui.cam_snapshot_name = String::new();

        state.ui.cam_snapshot = Some(state.cam_snapshots.len() - 1);

        state.update_save_prefs();
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
        // ui.add_space(COL_SPACING);
        if ui.button(RichText::new("❌").color(Color32::RED)).clicked() {
            if i < state.cam_snapshots.len() {
                state.cam_snapshots.remove(i);
            }
            state.ui.cam_snapshot = None;
            state.update_save_prefs();
        }
    }

    if state.ui.cam_snapshot != prev_snap {
        if let Some(snap_i) = state.ui.cam_snapshot {
            match state.cam_snapshots.get(snap_i) {
                Some(snap) => {
                    scene.camera.position = snap.position;
                    scene.camera.orientation = snap.orientation;
                    scene.camera.far = snap.far;

                    scene.camera.update_proj_mat(); // In case `far` etc changed.
                    engine_updates.camera = true;

                    set_flashlight(scene);
                    engine_updates.lighting = true;
                }
                None => {
                    eprintln!("Error: Could not find snapshot {}", snap_i);
                }
            }
        }
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

    let cam = &mut scene.camera;

    ui.horizontal(|ui| {
        ui.label("Camera:");

        // Preset buttons
        if ui.button("Front").clicked() {
            if let Some(mol) = &state.molecule {
                let center: Vec3 = mol.center.into();
                cam.position =
                    Vec3::new(center.x, center.y, center.z - (mol.size + CAM_INIT_OFFSET));
                cam.orientation = Quaternion::new_identity();

                state.ui.view_depth = (VIEW_DEPTH_NEAR_MIN, VIEW_DEPTH_FAR_MAX);
                changed = true;
            }
        }

        if ui.button("Top").clicked() {
            if let Some(mol) = &state.molecule {
                let center: Vec3 = mol.center.into();
                cam.position =
                    Vec3::new(center.x, center.y + (mol.size + CAM_INIT_OFFSET), center.z);
                cam.orientation = Quaternion::from_axis_angle(RIGHT_VEC, TAU / 4.);

                state.ui.view_depth = (VIEW_DEPTH_NEAR_MIN, VIEW_DEPTH_FAR_MAX);
                changed = true;
            }
        }

        if ui.button("Left").clicked() {
            if let Some(mol) = &state.molecule {
                let center: Vec3 = mol.center.into();
                cam.position =
                    Vec3::new(center.x - (mol.size + CAM_INIT_OFFSET), center.y, center.z);
                cam.orientation = Quaternion::from_axis_angle(UP_VEC, TAU / 4.);

                state.ui.view_depth = (VIEW_DEPTH_NEAR_MIN, VIEW_DEPTH_FAR_MAX);
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
            cam.near = if state.ui.view_depth.0 == VIEW_DEPTH_NEAR_MIN {
                RENDER_DIST_NEAR
            } else {
                state.ui.view_depth.0 as f32 / 10.
            };

            cam.far = if state.ui.view_depth.1 == VIEW_DEPTH_FAR_MAX {
                RENDER_DIST_FAR
            } else {
                state.ui.view_depth.1 as f32
            };

            cam.update_proj_mat();
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
fn selected_data(mol: &Molecule, selection: Selection, ui: &mut Ui) {
    match selection {
        Selection::Atom(sel_i) => {
            if sel_i >= mol.atoms.len() {
                return;
            }

            let atom = &mol.atoms[sel_i];

            let aa = match atom.residue_type {
                ResidueType::AminoAcid(a) => format!("AA: {}", a.to_str(AaIdent::OneLetter)),
                _ => String::new(),
            };

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

            let text_b = format!("{}", atom.element.to_letter());

            let mut text_c = format!("{aa}  {role}",);

            // todo: Helper fn for this, and find otehr places in the code where we need it.
            let res = &mol.residues.iter().find(|r| r.atoms.contains(&sel_i));
            if let Some(r) = res {
                text_c += &format!("  {}", r.descrip());
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
            if sel_i >= mol.residues.len() {
                return;
            }

            let res = &mol.residues[sel_i];
            ui.label(RichText::new(res.descrip()).color(Color32::GOLD));
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

fn docking(
    state: &mut State,
    scene: &mut Scene,
    redraw: &mut bool,
    reset_cam: bool,
    engine_updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    if state.molecule.is_none() || state.ligand.is_none() {
        return;
    }

    let mut updated_docking_site = false; // To avoid a double-borrow error.
    ui.horizontal(|ui| {
        let lig = &mut state.ligand.as_mut().unwrap();

        if ui.button("Find sites").clicked() {
            let mol = state.molecule.as_ref().unwrap();
            let sites = find_docking_sites(mol);
            for site in sites {
                println!("Docking site: {:?}", site);
            }
        }

        // if state.docking_ready {
        if ui.button("Dock").clicked() {
            // todo: Ideally move the camera to the docking site prior to docking. You could do this
            // todo by deferring the docking below to the next frame.

            let (pose, binding_energy) =
                find_optimal_pose(&state.dev, state.volatile.docking_setup.as_ref().unwrap(), lig);

            lig.pose = pose;
            lig.atom_posits = lig.position_atoms(None);

            {
                let lig_pos: Vec3 = lig.position_atoms(None)[lig.anchor_atom].into();
                let ctr: Vec3 = state.molecule.as_ref().unwrap().center.into();

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
            let mol = state.molecule.as_ref().unwrap();
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

        if state.selection != Selection::None {
            ui.add_space(COL_SPACING);

            if ui
                .button(RichText::new("Center on sel").color(COLOR_HIGHLIGHT))
                .clicked()
            {
                let atom_sel = state
                    .molecule
                    .as_ref()
                    .unwrap()
                    .get_sel_atom(state.selection);

                if let Some(atom) = atom_sel {
                    lig.docking_site.site_center = atom.posit;
                    lig.pose.anchor_posit = lig.docking_site.site_center;
                    lig.atom_posits = lig.position_atoms(None);

                    state.ui.docking_site_x = lig.docking_site.site_center.x.to_string();
                    state.ui.docking_site_y = lig.docking_site.site_center.y.to_string();
                    state.ui.docking_site_z = lig.docking_site.site_center.z.to_string();

                    docking_init_changed = true;
                    updated_docking_site = true;
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

    if updated_docking_site {
        state.update_save_prefs();
    }

    ui.horizontal(|ui| {
        if let Some(lig) = &mut state.ligand {
            if ui.button("Build VDW sim").clicked() {
                state.volatile.snapshots = build_vdw_dynamics(
                    &state.dev,
                    lig,
                    &state.volatile.docking_setup.as_ref().unwrap(),
                    false,
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

                    // todo: Come back to your approach here; definitely don't redraw the main molecule atoms etc.
                    scene.entities = Vec::new();
                    draw_molecule(state, scene, reset_cam);
                    draw_ligand(state, scene);

                    engine_updates.entities = true;
                }
            }
        }
    });
}

fn residue_search(state: &mut State, scene: &mut Scene, redraw: &mut bool, ui: &mut Ui) {
    ui.horizontal(|ui| {
        let sel_prev = state.selection;
        ui.label("Find residue:");
        if ui
            .add(TextEdit::singleline(&mut state.ui.residue_search).desired_width(60.))
            .changed()
        {
            select_from_search(state);
        }

        if sel_prev != state.selection {
            *redraw = true;
        }

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
        }
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
                    let atom_sel = mol.get_sel_atom(state.selection);

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
                    let lig_pos: Vec3 = lig.position_atoms(None)[lig.anchor_atom].into();
                    let ctr: Vec3 = mol.center.into();

                    cam_look_at_outside(&mut scene.camera, lig_pos, ctr);

                    engine_updates.camera = true;
                    state.ui.cam_snapshot = None;
                }
            }

            ui.add_space(COL_SPACING / 2.);
            selected_data(mol, state.selection, ui);
        }
    });
}

fn mol_descrip(mol: &Molecule, ui: &mut Ui) {
    ui.heading(RichText::new(mol.ident.clone()).color(Color32::GOLD));

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
            open_pdb(&mol.ident);
        }
    }

    if let Some(id) = &mol.drugbank_id {
        if ui.button("View on Drugbank").clicked() {
            open_drugbank(id);
        }
    }

    if let Some(id) = mol.pubchem_cid {
        if ui.button("View on PubChem").clicked() {
            open_pubchem(id);
        }
    }
}

fn view_settings(state: &mut State, redraw: &mut bool, ui: &mut Ui) {
    ui.horizontal(|ui| {
        chain_selector(state, redraw, ui);

        ui.add_space(COL_SPACING);

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
    });
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

        ui.horizontal_wrapped(|ui| {
            let metadata_loaded = false; // avoids borrow error.
            if let Some(mol) = &mut state.molecule {
                mol_descrip(mol, ui);

                if ui.button("Close").clicked() {
                    state.molecule = None;
                    scene.entities = Vec::new();
                    redraw = true;

                    state.to_save.last_opened = None;
                    state.update_save_prefs();
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

            if let Some(mol) = &state.molecule {
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
            }

            if ui.button("Open lig").clicked() {
                state.volatile.dialogs.load_ligand.pick_file();
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

                    state
                        .volatile
                        .dialogs
                        .save_ligand
                        .config_mut()
                        .default_file_name = filename.to_string();
                    state.volatile.dialogs.save_ligand.save_file();
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
            ui.add(TextEdit::singleline(&mut state.ui.db_input).desired_width(40.));

            if !state.ui.db_input.is_empty() {
                if ui.button("Download from RCSB").clicked() {
                    match load_cif_rcsb(&state.ui.db_input) {
                        Ok(pdb) => {
                            state.pdb = Some(pdb);
                            state.molecule = Some(Molecule::from_pdb(state.pdb.as_ref().unwrap()));
                            state.update_from_prefs();

                            redraw = true;
                            reset_cam = true;
                            set_flashlight(scene);
                            engine_updates.lighting = true;
                        }
                        Err(_e) => {
                            eprintln!("Error loading CIF file");
                        }
                    }
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
                                eprintln!("Error loading SDF file");
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
                            eprintln!("Error loading SDF file");
                        }
                    }
                }
            }
        });

        ui.add_space(ROW_SPACING);
        let mut close_ligand = false; // to avoid borrow error.
        if let Some(ligand) = &mut state.ligand {
            ui.horizontal(|ui| {
                mol_descrip(&ligand.molecule, ui);

                if ui.button("Close").clicked() {
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

        if close_ligand {
            state.ligand = None;
            scene.entities = Vec::new();
            redraw = true;

            state.to_save.last_ligand_opened = None;
            state.update_save_prefs();
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
                view_settings(state, &mut redraw, ui);

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

        if let Some(path) = &state.volatile.dialogs.load.take_picked() {
            load_file(
                path,
                state,
                &mut redraw,
                &mut reset_cam,
                &mut engine_updates,
                false,
            );
            set_flashlight(scene);
            engine_updates.lighting = true;
        }

        if let Some(path) = &state.volatile.dialogs.save.take_picked() {
            if let Some(mol) = &state.molecule {
                if let Some(pdb) = &mut state.pdb {
                    if let Err(e) = save_pdb(mol, pdb, path) {
                        eprintln!("Error saving pdb: {}", e);
                    } else {
                        state.to_save.last_opened = Some(path.to_owned());
                        state.update_save_prefs()
                    }
                }
            }
        }

        if let Some(path) = &state.volatile.dialogs.save_ligand.take_picked() {
            if let Some(lig) = &state.ligand {
                // todo: Other formats A/R
                if let Err(e) = lig.molecule.save_sdf(path) {
                    eprintln!("Error saving SDF: {}", e);
                } else {
                    state.to_save.last_ligand_opened = Some(path.to_owned());
                    state.update_save_prefs()
                }
            }
        }

        if let Some(path) = &state.volatile.dialogs.load_ligand.take_picked() {
            load_file(
                path,
                state,
                &mut redraw,
                &mut reset_cam,
                &mut engine_updates,
                true,
            );
        }

        if let Some(path) = &state.volatile.dialogs.autodock_path.take_picked() {
            state.ui.autodock_path_valid = check_adv_avail(path);
            if state.ui.autodock_path_valid {
                state.to_save.autodock_vina_path = Some(path.to_owned());
                state.update_save_prefs();
            }
        }

        if let Some(path_dir) = &state.volatile.dialogs.save_pdbqt.take_picked() {
            if let Some(mol) = &mut state.molecule {
                let filename = format!("{}_target.pdbqt", mol.ident);
                let path = Path::new(path_dir).join(filename);

                // todo: You will likely need to add charges earlier, so you can view their data in the UI.
                // setup_partial_charges(&mut mol.atoms, PartialChargeType::Gasteiger);
                // create_partial_charges(&mut mol.atoms);

                if mol.save_pdbqt(&path, None).is_err() {
                    eprintln!("Error saving PDBQT target");
                }
            }

            if let Some(lig) = &mut state.ligand {
                let filename = format!("{}_ligand.pdbqt", lig.molecule.ident);
                let path = Path::new(path_dir).join(filename);

                // create_partial_charges(&mut lig.molecule.atoms);
                // setup_partial_charges(&mut lig.molecule.atoms, PartialChargeType::Gasteiger);

                if lig.molecule.save_pdbqt(&path, None).is_err() {
                    eprintln!("Error saving PDBQT ligand");
                }
            }
        }

        if redraw {
            scene.entities = Vec::new();
            draw_molecule(state, scene, reset_cam);
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
    });

    // At init only.
    if state.volatile.ui_height < f32::EPSILON {
        state.volatile.ui_height = ctx.used_size().y;
    }

    state.volatile.dialogs.load.update(ctx);
    state.volatile.dialogs.load_ligand.update(ctx);
    state.volatile.dialogs.save.update(ctx);
    state.volatile.dialogs.save_ligand.update(ctx);
    state.volatile.dialogs.autodock_path.update(ctx);
    state.volatile.dialogs.save_pdbqt.update(ctx);

    // todo: Appropriate place for this?
    if state.volatile.inputs_commanded.inputs_present() {
        set_flashlight(scene);
        engine_updates.lighting = true;
    }

    state.ui.dt = start.elapsed().as_secs_f32();

    static mut INIT_COMPLETE: bool = false;

    unsafe {
        if !INIT_COMPLETE {
            if let Some(mol) = &state.molecule {
                if let Some(lig) = &state.ligand {
                    let lig_pos: Vec3 = lig.position_atoms(None)[lig.anchor_atom].into();
                    let ctr: Vec3 = mol.center.into();

                    cam_look_at_outside(&mut scene.camera, lig_pos, ctr);

                    engine_updates.camera = true;
                    state.ui.cam_snapshot = None;
                }
            }

            INIT_COMPLETE = true;
        }
    }

    if state.ui.new_mol_loaded {
        state.ui.new_mol_loaded = false;

        set_flashlight(scene);
        engine_updates.lighting = true;
    }

    engine_updates
}
