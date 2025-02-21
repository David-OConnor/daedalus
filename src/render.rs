//! This module integraties this application with the graphics engine.

use std::{f32::consts::TAU, fmt};

use bincode::{Decode, Encode};
use graphics::{
    Camera, ControlScheme, DeviceEvent, ElementState, EngineUpdates, Entity, FWD_VEC,
    InputSettings, LightType, Lighting, Mesh, PointLight, RIGHT_VEC, Scene, UP_VEC, UiLayout,
    UiSettings, WindowEvent,
    event::MouseScrollDelta,
    winit::keyboard::{KeyCode, PhysicalKey::Code},
};
use lin_alg::{
    f32::{Quaternion, Vec3},
    f64::Quaternion as QuaternionF64,
    map_linear,
};

use crate::{
    Selection, State, ViewSelLevel,
    asa::{get_mesh_points, mesh_from_sas_points},
    molecule::{Atom, AtomRole, BondCount, Chain, Residue, aa_color},
    ui::ui_handler,
    util::{cycle_res_selected, find_selected_atom, mol_center_size, points_along_ray},
};

type Color = (f32, f32, f32);

const WINDOW_TITLE: &str = "Bio Chem View";
const WINDOW_SIZE_X: f32 = 1_400.;
const WINDOW_SIZE_Y: f32 = 1_000.;
const BACKGROUND_COLOR: Color = (0., 0., 0.);
pub const RENDER_DIST: f32 = 1_000.;

// todo: Shinyness broken?
pub const ATOM_SHINYNESS: f32 = 12.;
pub const BODY_SHINYNESS: f32 = 12.;

// Keep this in sync with mesh init.
const MESH_SPHERE: usize = 0;
const MESH_CUBE: usize = 1;
const MESH_BOND: usize = 2;
const MESH_SPHERE_LOWRES: usize = 3;
const MESH_SURFACE: usize = 4; // Van Der Waals surface.

const SELECTION_DIST_THRESH_SMALL: f32 = 0.7; // e.g. ball + stick
const SELECTION_DIST_THRESH_LARGE: f32 = 1.3; // e.g. VDW views.

const BALL_STICK_RADIUS: f32 = 0.3;

// todo: By bond type etc
// const BOND_COLOR: Color = (0.2, 0.2, 0.2);
const BOND_RADIUS: f32 = 0.12;
// const BOND_CAP_RADIUS: f32 = 1./BOND_RADIUS;
const BOND_RADIUS_DOUBLE: f32 = 0.07;

const RADIUS_SFC_DOT: f32 = 0.05;
const COLOR_SFC_DOT: Color = (0.7, 0.7, 0.7);

pub const COLOR_SELECTED: Color = (1., 0., 0.);

pub const SHELL_OPACITY: f32 = 0.01;

// From the farthest molecule.
pub const CAM_INIT_OFFSET: f32 = 10.;
pub const OUTSIDE_LIGHTING_OFFSET: f32 = 300.;

pub const COLOR_AA_NON_RESIDUE: Color = (0., 0.8, 1.0);

const MOVEMENT_SENS: f32 = 12.;
const RUN_FACTOR: f32 = 6.; // i.e. shift key multiplier
const SCROLL_MOVE_AMT: f32 = 4.;

#[derive(Clone, Copy, PartialEq, Debug, Default, Encode, Decode)]
pub enum MoleculeView {
    Sticks,
    Backbone,
    #[default]
    BallAndStick,
    /// i.e. Van der Waals radius, or CPK.
    SpaceFill,
    Cartoon,
    Surface,
    Mesh,
    Dots,
}

impl fmt::Display for MoleculeView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let val = match self {
            Self::Backbone => "Backbone",
            Self::Sticks => "Sticks",
            Self::BallAndStick => "Ball and stick",
            Self::Cartoon => "Cartoon",
            Self::SpaceFill => "Spacefill (Van der Waals / CPK)",
            Self::Surface => "Surface (Van der Waals)",
            Self::Mesh => "Mesh (Van der Waals)",
            Self::Dots => "Dots (Van der Waals)",
        };

        write!(f, "{val}")
    }
}

/// Set lighting based on the center and size of the molecule.
fn set_lighting(center: Vec3, size: f32) -> Lighting {
    let white = [1., 1., 1., 1.];

    Lighting {
        ambient_color: white,
        ambient_intensity: 0.12,
        point_lights: vec![PointLight {
            type_: LightType::Omnidirectional,
            position: center + Vec3::new(40., size + OUTSIDE_LIGHTING_OFFSET, 0.),
            diffuse_color: white,
            specular_color: white,
            diffuse_intensity: 10_000.,
            specular_intensity: 60_000.,
        }],
    }
}

fn atom_color(
    atom: &Atom,
    i: usize,
    residues: &[Residue],
    selection: Selection,
    view_sel_level: ViewSelLevel,
) -> Color {
    let mut result = match view_sel_level {
        ViewSelLevel::Atom => atom.element.color(),
        ViewSelLevel::Residue => {
            let c = match atom.amino_acid {
                Some(aa) => aa_color(aa),
                None => COLOR_AA_NON_RESIDUE,
            };
            // Below is currently equivalent:
            // for res in &mol.residues {
            //     if res.atoms.contains(&i) {
            //         if let ResidueType::AminoAcid(aa) = res.res_type {
            //             c = aa_color(aa);
            //         }
            //     }
            // }
            c
        }
    };

    // If selected, the selected color overrides the element or residue color.
    match selection {
        Selection::Atom(sel_i) => {
            if sel_i == i {
                result = COLOR_SELECTED;
            }
        }
        Selection::Residue(sel_i) => {
            if residues[sel_i].atoms.contains(&i) {
                result = COLOR_SELECTED;
            }
        }
        Selection::None => (),
    }

    result
}

/// Adds a cylindrical bond. This is divided into two halves, so they can be color-coded by their side's
/// atom. Adds optional rounding. `thickness` is relative to BOND_RADIUS.
fn add_bond(
    entities: &mut Vec<Entity>,
    posit_0: Vec3,
    posit_1: Vec3,
    center: Vec3,
    color_0: Color,
    color_1: Color,
    orientation: Quaternion,
    dist_half: f32,
    caps: bool,
    thickness: f32,
) {
    // Split the bond into two entities, so you can color-code them separately based
    // on which atom the half is closer to.
    let center_0 = (posit_0 + center) / 2.;
    let center_1 = (posit_1 + center) / 2.;

    let mut entity_0 = Entity::new(
        MESH_BOND,
        center_0,
        orientation,
        1.,
        color_0,
        BODY_SHINYNESS,
    );

    let mut entity_1 = Entity::new(
        MESH_BOND,
        center_1,
        orientation,
        1.,
        color_1,
        BODY_SHINYNESS,
    );

    if caps {
        // These spheres are to put a rounded cap on each bond.
        // todo: You only need a dome; performance implications.
        let cap_0 = Entity::new(
            MESH_SPHERE,
            posit_0,
            Quaternion::new_identity(),
            BOND_RADIUS * thickness,
            color_0,
            BODY_SHINYNESS,
        );
        let cap_1 = Entity::new(
            MESH_SPHERE,
            posit_1,
            Quaternion::new_identity(),
            BOND_RADIUS * thickness,
            color_1,
            BODY_SHINYNESS,
        );

        entities.push(cap_0);
        entities.push(cap_1);
    }

    let scale = Some(Vec3::new(thickness, dist_half, thickness));
    entity_0.scale_partial = scale;
    entity_1.scale_partial = scale;

    entities.push(entity_0);
    entities.push(entity_1);
}

fn bond_entities(
    entities: &mut Vec<Entity>,
    posit_0: Vec3,
    posit_1: Vec3,
    color_0: Color,
    color_1: Color,
    bond_count: BondCount,
) {
    // todo: You probably need to update this to display double bonds correctly.

    // todo: YOur multibond plane logic is off.

    let center: Vec3 = (posit_0 + posit_1) / 2.;

    let diff = posit_0 - posit_1;
    let diff_unit = diff.to_normalized();
    let orientation = Quaternion::from_unit_vecs(UP_VEC, diff_unit);
    let dist_half = diff.magnitude() / 2.;

    let caps = true; // todo: Remove caps if ball+ stick

    // todo: Put this multibond code back.
    // todo: Lots of DRY!
    match bond_count {
        BondCount::Single => {
            add_bond(
                entities,
                posit_0,
                posit_1,
                center,
                color_0,
                color_1,
                orientation,
                dist_half,
                caps,
                1.,
            );
        }
        BondCount::SingleDoubleHybrid => {
            // Draw two offset bond cylinders.
            let rot_ortho = Quaternion::from_unit_vecs(FWD_VEC, UP_VEC);
            let rotator = rot_ortho * orientation;

            let offset_a = rotator.rotate_vec(Vec3::new(0.2, 0., 0.));
            let offset_b = rotator.rotate_vec(Vec3::new(-0.2, 0., 0.));

            // todo: Make this one better

            add_bond(
                entities,
                posit_0 + offset_a,
                posit_1 + offset_a,
                center + offset_a,
                color_0,
                color_1,
                orientation,
                dist_half,
                caps,
                0.7,
            );
            add_bond(
                entities,
                posit_0 + offset_b,
                posit_1 + offset_b,
                center + offset_b,
                color_0,
                color_1,
                orientation,
                dist_half,
                caps,
                0.4,
            );
        }
        BondCount::Double => {
            // Draw two offset bond cylinders.
            let rot_ortho = Quaternion::from_unit_vecs(FWD_VEC, UP_VEC);
            let rotator = rot_ortho * orientation;

            let offset_a = rotator.rotate_vec(Vec3::new(0.2, 0., 0.));
            let offset_b = rotator.rotate_vec(Vec3::new(-0.2, 0., 0.));

            add_bond(
                entities,
                posit_0 + offset_a,
                posit_1 + offset_a,
                center + offset_a,
                color_0,
                color_1,
                orientation,
                dist_half,
                caps,
                0.5,
            );
            add_bond(
                entities,
                posit_0 + offset_b,
                posit_1 + offset_b,
                center + offset_b,
                color_0,
                color_1,
                orientation,
                dist_half,
                caps,
                0.5,
            );
        }
        BondCount::Triple => {
            //         // Draw two offset bond cylinders.
            let rot_ortho = Quaternion::from_unit_vecs(FWD_VEC, UP_VEC);
            let rotator = rot_ortho * orientation;

            let offset_a = rotator.rotate_vec(Vec3::new(0.25, 0., 0.));
            let offset_b = rotator.rotate_vec(Vec3::new(-0.25, 0., 0.));

            add_bond(
                entities,
                posit_0,
                posit_1,
                center,
                color_0,
                color_1,
                orientation,
                dist_half,
                caps,
                0.4,
            );
            add_bond(
                entities,
                posit_0 + offset_a,
                posit_1 + offset_a,
                center + offset_a,
                color_0,
                color_1,
                orientation,
                dist_half,
                caps,
                0.4,
            );
            add_bond(
                entities,
                posit_0 + offset_b,
                posit_1 + offset_b,
                center + offset_b,
                color_0,
                color_1,
                orientation,
                dist_half,
                caps,
                0.4,
            );
        }
    }
}

// todo: DRY with/subset of draw_molecule?
pub fn draw_ligand(state: &mut State, scene: &mut Scene, update_cam_lighting: bool) {
    // Hard-coded for sticks for now.

    if state.ligand.is_none() {
        return;
    }
    let ligand = state.ligand.as_ref().unwrap();
    let mol = &ligand.molecule;

    // todo: rotate using the orientation relative to the offset. Atoms and bonds.

    let mut atoms_rotated = mol.atoms.clone();

    // Rotate around the *molecule center* we calculated; this is invariant of the initial molecule coordinates.
    // todo: that algorithm may be too naive.
    for atom in &mut atoms_rotated {
        let posit_offset = atom.posit - mol.center;
        atom.posit = ligand.orientation.rotate_vec(posit_offset);
    }

    // for atom in &mol.atoms {
    //     scene.entities.push(Entity::new(
    //         MESH_SPHERE,
    //         (atom.posit + ligand.offset).into(),
    //         Quaternion::new_identity(),
    //         BALL_STICK_RADIUS,
    //         atom.element.color(),
    //         ATOM_SHINYNESS,
    //     ));
    // }

    // todo: C+P from draw_molecule. With some removed, but a lot of repeated.
    for bond in &mol.bonds {
        let atom_0 = &atoms_rotated[bond.atom_0];
        let atom_1 = &atoms_rotated[bond.atom_1];

        let posit_0: Vec3 = (atom_0.posit + ligand.docking_init.site_posit).into();
        let posit_1: Vec3 = (atom_1.posit + ligand.docking_init.site_posit).into();

        bond_entities(
            &mut scene.entities,
            posit_0,
            posit_1,
            atom_0.element.color(),
            atom_1.element.color(),
            bond.bond_count,
        );
    }
}

/// Refreshes entities with the model passed.
/// Sensitive to various view configuration parameters.
pub fn draw_molecule(state: &mut State, scene: &mut Scene, update_cam_lighting: bool) {
    if state.molecule.is_none() {
        return;
    }
    let mol = state.molecule.as_mut().unwrap();

    // todo: Update this capacity A/R as you flesh out your renders.
    // *entities = Vec::with_capacity(molecule.bonds.len());
    scene.entities = Vec::new();

    let ui = &state.ui;
    let volatile = &mut state.volatile;

    let chains_invis: Vec<&Chain> = mol.chains.iter().filter(|c| !c.visible).collect();

    // todo: Figure out how to handle the VDW models A/R.
    // todo: Mesh and/or Surface A/R.
    if ui.mol_view == MoleculeView::Dots {
        if mol.sa_surface_pts.is_none() {
            println!("Starting getting mesh pts...");
            mol.sa_surface_pts = Some(get_mesh_points(&mol.atoms));
            println!("Mesh pts complete.");
        }

        // let mut i = 0;
        for ring in mol.sa_surface_pts.as_ref().unwrap() {
            for sfc_pt in ring {
                scene.entities.push(Entity::new(
                    MESH_SPHERE_LOWRES,
                    *sfc_pt,
                    Quaternion::new_identity(),
                    RADIUS_SFC_DOT,
                    COLOR_SFC_DOT,
                    ATOM_SHINYNESS,
                ));
            }
            // i += 1;
            // if i > 100 {
            //     break;
            // }
        }
    }

    if ui.mol_view == MoleculeView::Surface {
        if mol.sa_surface_pts.is_none() {
            // todo: DRY with above.
            println!("Starting getting mesh pts...");
            mol.sa_surface_pts = Some(get_mesh_points(&mol.atoms));
            println!("Mesh pts complete.");
        }

        if !mol.mesh_created {
            println!("Building surface mesh...");
            scene.meshes[MESH_SURFACE] =
                mesh_from_sas_points(&mol.sa_surface_pts.as_ref().unwrap());
            mol.mesh_created = true;
            println!("Mesh complete");
        }

        scene.entities.push(Entity::new(
            MESH_SURFACE,
            Vec3::new_zero(),
            Quaternion::new_identity(),
            1.,
            COLOR_SFC_DOT,  // todo
            ATOM_SHINYNESS, // todo
        ));
    }

    // Draw atoms.
    if [MoleculeView::BallAndStick, MoleculeView::SpaceFill].contains(&ui.mol_view) {
        for (i, atom) in mol.atoms.iter().enumerate() {
            let mut chain_not_sel = false;
            for chain in &chains_invis {
                if chain.atoms.contains(&i) {
                    chain_not_sel = true;
                    break;
                }
            }
            if chain_not_sel {
                continue;
            }

            if let Some(role) = atom.role {
                if state.ui.hide_sidechains {
                    if role == AtomRole::Sidechain {
                        continue;
                    }
                }
                if state.ui.hide_water || ui.mol_view == MoleculeView::SpaceFill {
                    if role == AtomRole::Water {
                        continue;
                    }
                }
            }

            if state.ui.hide_hetero && atom.hetero {
                continue;
            } else if state.ui.hide_non_hetero && !atom.hetero {
                continue;
            }

            if ui.show_nearby_only {
                if ui.show_nearby_only {
                    let atom_sel = mol.get_sel_atom(state.selection);
                    if let Some(a) = atom_sel {
                        if (atom.posit - a.posit).magnitude() as f32 > ui.nearby_dist_thresh as f32
                        {
                            continue;
                        }
                    }
                }
            }

            let radius = match ui.mol_view {
                MoleculeView::SpaceFill => atom.element.vdw_radius(),
                _ => BALL_STICK_RADIUS,
            };

            let color_atom = atom_color(
                &atom,
                i,
                &mol.residues,
                state.selection,
                state.ui.view_sel_level,
            );

            scene.entities.push(Entity::new(
                MESH_SPHERE,
                atom.posit.into(),
                Quaternion::new_identity(),
                radius,
                color_atom,
                ATOM_SHINYNESS,
            ));
        }
    }

    // Draw bonds.
    if ![MoleculeView::SpaceFill].contains(&ui.mol_view) {
        for bond in &mol.bonds {
            let atom_0 = &mol.atoms[bond.atom_0];
            let atom_1 = &mol.atoms[bond.atom_1];

            if ui.mol_view == MoleculeView::Backbone && !bond.is_backbone {
                continue;
            }

            if ui.show_nearby_only {
                let atom_sel = mol.get_sel_atom(state.selection);
                if let Some(a) = atom_sel {
                    if (atom_0.posit - a.posit).magnitude() as f32 > ui.nearby_dist_thresh as f32 {
                        continue;
                    }
                }
            }

            let mut chain_not_sel = false;
            for chain in &chains_invis {
                if chain.atoms.contains(&bond.atom_0) {
                    chain_not_sel = true;
                    break;
                }
            }
            if chain_not_sel {
                continue;
            }

            // Assuming water won't be bonded to the main molecule.
            if state.ui.hide_sidechains {
                if let Some(role_0) = atom_0.role {
                    if let Some(role_1) = atom_1.role {
                        if role_0 == AtomRole::Sidechain || role_1 == AtomRole::Sidechain {
                            continue;
                        }
                    }
                }
            }

            if state.ui.hide_hetero && atom_0.hetero && atom_1.hetero {
                continue;
            } else if state.ui.hide_non_hetero && !atom_0.hetero && !atom_1.hetero {
                continue;
            }

            let posit_0: Vec3 = atom_0.posit.into();
            let posit_1: Vec3 = atom_1.posit.into();

            let color_0 = atom_color(
                &atom_0,
                bond.atom_0,
                &mol.residues,
                state.selection,
                state.ui.view_sel_level,
            );
            let color_1 = atom_color(
                &atom_1,
                bond.atom_1,
                &mol.residues,
                state.selection,
                state.ui.view_sel_level,
            );

            bond_entities(
                &mut scene.entities,
                posit_0,
                posit_1,
                color_0,
                color_1,
                bond.bond_count,
            );
        }
    }

    if update_cam_lighting {
        let center: Vec3 = mol.center.into();
        scene.camera.position =
            Vec3::new(center.x, center.y, center.z - (mol.size + CAM_INIT_OFFSET));
        scene.camera.orientation = Quaternion::from_axis_angle(RIGHT_VEC, 0.);
        scene.camera.far = RENDER_DIST;
        scene.camera.update_proj_mat();

        // Update lighting based on the new molecule center and dims.
        scene.lighting = set_lighting(center, mol.size);
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

                            let selected_ray = scene.screen_to_render(cursor);

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
                        draw_molecule(state_, scene, false);
                        draw_ligand(state_, scene, false);
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
                        draw_molecule(state_, scene, false);
                        draw_ligand(state_, scene, false);
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
        draw_molecule(state_, scene, false);
        draw_ligand(state_, scene, false);
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
            Mesh::new_sphere(1., 16, 16),
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
        move_sens: MOVEMENT_SENS,
        run_factor: RUN_FACTOR,
        ..Default::default()
    };
    let ui_settings = UiSettings {
        layout: UiLayout::Top,
        icon_path: Some("./resources/icon.png".to_owned()),
    };

    draw_molecule(&mut state, &mut scene, true);
    draw_ligand(&mut state, &mut scene, true);

    graphics::run(
        state,
        scene,
        input_settings,
        ui_settings,
        render_handler,
        event_dev_handler,
        event_win_handler,
        ui_handler,
    );
}
