//! This module integraties this application with the graphics engine.

use std::{f32::consts::TAU, fmt};

use bincode::{Decode, Encode};
use graphics::{
    event::{RawKeyEvent, WindowEvent},
    winit::keyboard::{KeyCode, PhysicalKey},
    Camera, ControlScheme, DeviceEvent, ElementState, EngineUpdates, Entity, InputSettings,
    LightType, Lighting, Mesh, PointLight, Scene, UiLayout, UiSettings, FWD_VEC, RIGHT_VEC, UP_VEC,
};
use lin_alg::{
    f32::{Quaternion, Vec3},
    map_linear,
};

use crate::{
    molecule::{aa_color, BondCount, Chain, Molecule},
    ui::ui_handler,
    util::{cycle_res_selected, find_selected_atom, mol_center_size, points_along_ray},
    Selection, State, StateUi, StateVolatile, ViewSelLevel,
};

type Color = (f32, f32, f32);

const WINDOW_TITLE: &str = "Bio Chem View";
const WINDOW_SIZE_X: f32 = 1_400.;
const WINDOW_SIZE_Y: f32 = 1_000.;
const BACKGROUND_COLOR: Color = (0., 0., 0.);
pub const RENDER_DIST: f32 = 1_000.;

pub const ATOM_SHINYNESS: f32 = 2.;
pub const BODY_SHINYNESS: f32 = 2.;

// Keep this in sync with mesh init.
pub const MESH_SPHERE: usize = 0;
pub const MESH_CUBE: usize = 1;
pub const MESH_BOND: usize = 2;
pub const MESH_SURFACE: usize = 3; // Van Der Waals surface.

const SELECTION_DIST_THRESH_SMALL: f32 = 0.7; // e.g. ball + stick
const SELECTION_DIST_THRESH_LARGE: f32 = 1.3; // e.g. VDW views.

// todo: By bond type etc
const BOND_COLOR: Color = (0.2, 0.2, 0.2);
const BOND_RADIUS: f32 = 0.10;
const BOND_RADIUS_DOUBLE: f32 = 0.07;

pub const COLOR_SELECTED: Color = (1., 0., 0.);

pub const SHELL_OPACITY: f32 = 0.01;

// From the farthest molecule.
pub const CAM_INIT_OFFSET: f32 = 10.;
pub const OUTSIDE_LIGHTING_OFFSET: f32 = 400.;

#[derive(Clone, Copy, PartialEq, Debug, Default, Encode, Decode)]
pub enum MoleculeView {
    Sticks,
    Ribbon,
    #[default]
    BallAndStick,
    /// i.e. Van der Waals radius, or CPK.
    Spheres,
    Cartoon,
    Surface,
    Mesh,
    Dots,
}

impl fmt::Display for MoleculeView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let val = match self {
            Self::Ribbon => "Ribbon",
            Self::Sticks => "Sticks",
            Self::BallAndStick => "Ball and stick",
            Self::Cartoon => "Cartoon",
            Self::Spheres => "Spheres (Van der Waals)",
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
        point_lights: vec![
            // Light in the middle
            // PointLight {
            //     type_: LightType::Omnidirectional,
            //     position: center,
            //     diffuse_color: white,
            //     specular_color: white,
            //     diffuse_intensity: 5.,
            //     specular_intensity: 50.,
            // },

            // // Light from the right
            // PointLight {
            //     type_: LightType::Omnidirectional,
            //     position: center + Vec3::new(0., 0., size + OUTSIDE_LIGHTING_OFFSET),
            //     diffuse_color: white,
            //     specular_color: white,
            //     diffuse_intensity: 4_000.,
            //     specular_intensity: 16_000.,
            // },
            // Light from above
            PointLight {
                type_: LightType::Omnidirectional,
                position: center + Vec3::new(100., size + OUTSIDE_LIGHTING_OFFSET, 0.),
                diffuse_color: white,
                specular_color: white,
                diffuse_intensity: 10_000.,
                specular_intensity: 20_000.,
            },
        ],
    }
}

/// Refreshes entities with the model passed.
/// Sensitive to various view configuration parameters.
pub fn draw_molecule(state: &mut State, scene: &mut Scene, update_cam_lighting: bool) {
    if state.molecule.is_none() {
        return;
    }
    let mol = state.molecule.as_ref().unwrap();

    // todo: Update this capacity A/R as you flesh out your renders.
    // *entities = Vec::with_capacity(molecule.bonds.len());
    scene.entities = Vec::new();

    let ui = &state.ui;
    let volatile = &mut state.volatile;

    let chains_invis: Vec<&Chain> = mol.chains.iter().filter(|c| !c.visible).collect();

    // Draw atoms.
    if [MoleculeView::BallAndStick, MoleculeView::Spheres].contains(&ui.mol_view) {
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

            let mut color = match ui.view_sel_level {
                ViewSelLevel::Atom => atom.element.color(),
                ViewSelLevel::Residue => {
                    let mut c = atom.element.color();

                    if let Some(aa) = atom.amino_acid {
                        c = aa_color(aa);
                    }
                    // Below is currently equivalent:

                    // for res in &molecule.residues {
                    //     if res.atoms.contains(&i) {
                    //         if let Some(aa) = res.aa {
                    //             c = aa_color(aa);
                    //         }
                    //     }
                    // }
                    c
                }
            };

            // If selected, the selected color overrides the element or residue color.
            match state.selection {
                Selection::Atom(sel_i) => {
                    if sel_i == i {
                        color = COLOR_SELECTED;
                    }
                }
                Selection::Residue(sel_i) => {
                    if mol.residues[sel_i].atoms.contains(&i) {
                        color = COLOR_SELECTED;
                    }
                }
                Selection::None => (),
            }

            let radius = match ui.mol_view {
                MoleculeView::Spheres => atom.element.vdw_radius(),
                _ => 0.3,
            };

            scene.entities.push(Entity::new(
                MESH_SPHERE,
                atom.posit.into(),
                Quaternion::new_identity(),
                radius,
                color,
                ATOM_SHINYNESS,
            ));
        }
    }

    let rot_ortho = Quaternion::from_unit_vecs(FWD_VEC, UP_VEC);

    // Draw bonds.
    if ![MoleculeView::Spheres].contains(&ui.mol_view) {
        for bond in &mol.bonds {
            let atom_0 = &mol.atoms[bond.atom_0];
            let atom_1 = &mol.atoms[bond.atom_1];

            if ui.mol_view == MoleculeView::Ribbon && !bond.is_backbone {
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

            let center: Vec3 = ((atom_0.posit + atom_1.posit) / 2.).into();

            let diff: Vec3 = (atom_0.posit - atom_1.posit).into();
            let diff_unit = diff.to_normalized();
            let orientation = Quaternion::from_unit_vecs(UP_VEC, diff_unit);

            let scale = Some(Vec3::new(1., diff.magnitude(), 1.));
            let scale_multibond = Some(Vec3::new(0.7, diff.magnitude(), 0.7));

            let color = if [MoleculeView::Sticks].contains(&ui.mol_view) {
                // todo: A/R between teh two bonds. May need two bond elements.
                atom_0.element.color()
            } else {
                BOND_COLOR
            };

            // todo: Lots of DRY!
            match bond.bond_count {
                BondCount::SingleDoubleHybrid => {
                    // Draw two offset bond cylinders.
                    let rotator = rot_ortho * orientation;

                    let offset_a = rotator.rotate_vec(Vec3::new(0.2, 0., 0.));
                    let offset_b = rotator.rotate_vec(Vec3::new(-0.2, 0., 0.));

                    let mut entity_0 = Entity::new(
                        MESH_BOND,
                        center + offset_a,
                        orientation,
                        1.,
                        color,
                        BODY_SHINYNESS,
                    );

                    let mut entity_1 = Entity::new(
                        MESH_BOND,
                        center + offset_b,
                        orientation,
                        1.,
                        color,
                        BODY_SHINYNESS,
                    );

                    entity_0.scale_partial = scale_multibond;
                    // Show only half len on one of the bonds as a visual differentiator.
                    entity_1.scale_partial = Some(Vec3::new(0.7, diff.magnitude() * 0.3, 0.7));

                    scene.entities.push(entity_0);
                    scene.entities.push(entity_1);
                }
                BondCount::Double => {
                    // Draw two offset bond cylinders.
                    let rotator = rot_ortho * orientation;

                    let offset_a = rotator.rotate_vec(Vec3::new(0.2, 0., 0.));
                    let offset_b = rotator.rotate_vec(Vec3::new(-0.2, 0., 0.));

                    let mut entity_0 = Entity::new(
                        MESH_BOND,
                        center + offset_a,
                        orientation,
                        1.,
                        color,
                        BODY_SHINYNESS,
                    );

                    let mut entity_1 = Entity::new(
                        MESH_BOND,
                        center + offset_b,
                        orientation,
                        1.,
                        color,
                        BODY_SHINYNESS,
                    );

                    entity_0.scale_partial = scale_multibond;
                    entity_1.scale_partial = scale_multibond;

                    scene.entities.push(entity_0);
                    scene.entities.push(entity_1);
                }
                BondCount::Triple => {
                    // Draw two offset bond cylinders.
                    let rotator = rot_ortho * orientation;

                    let offset_a = rotator.rotate_vec(Vec3::new(0.2, 0., 0.));
                    let offset_b = rotator.rotate_vec(Vec3::new(-0.2, 0., 0.));

                    let mut entity_0 =
                        Entity::new(MESH_BOND, center, orientation, 1., color, BODY_SHINYNESS);

                    let mut entity_1 = Entity::new(
                        MESH_BOND,
                        center + offset_a,
                        orientation,
                        1.,
                        color,
                        BODY_SHINYNESS,
                    );

                    let mut entity_2 = Entity::new(
                        MESH_BOND,
                        center + offset_b,
                        orientation,
                        1.,
                        color,
                        BODY_SHINYNESS,
                    );

                    entity_0.scale_partial = scale_multibond;
                    entity_1.scale_partial = scale_multibond;
                    entity_2.scale_partial = scale_multibond;

                    scene.entities.push(entity_0);
                    scene.entities.push(entity_1);
                    scene.entities.push(entity_2);
                }
                _ => {
                    let mut entity =
                        Entity::new(MESH_BOND, center, orientation, 1., color, BODY_SHINYNESS);

                    entity.scale_partial = scale;
                    scene.entities.push(entity);
                }
            }
        }
    }

    if update_cam_lighting {
        let (center, size) = mol_center_size(&mol.atoms);
        volatile.mol_center = center;
        volatile.mol_size = size;

        scene.camera.position = Vec3::new(
            volatile.mol_center.x,
            volatile.mol_center.y,
            volatile.mol_center.z - (volatile.mol_size + CAM_INIT_OFFSET),
        );
        scene.camera.orientation = Quaternion::from_axis_angle(RIGHT_VEC, 0.);
        scene.camera.far = RENDER_DIST;
        scene.camera.update_proj_mat();

        // Update lighting based on the new molecule center and dims.
        scene.lighting = set_lighting(volatile.mol_center, volatile.mol_size);
    }

    // todo: Evaluate if this is a sufficiently-robust place to save prefs.
    state.update_save_prefs();
}

fn event_dev_handler(
    state_: &mut State,
    event: DeviceEvent,
    scene: &mut Scene,
    _dt: f32,
) -> EngineUpdates {
    let mut updates = EngineUpdates::default();

    let mut redraw = false;

    match event {
        DeviceEvent::Button { button, state } => {
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
                                    | MoleculeView::Spheres => SELECTION_DIST_THRESH_LARGE,
                                    _ => SELECTION_DIST_THRESH_SMALL,
                                };
                                let atoms_along_ray =
                                    points_along_ray(selected_ray, &mol.atoms, dist_thresh);

                                state_.selection = find_selected_atom(
                                    &atoms_along_ray,
                                    &mol.atoms,
                                    &mol.residues,
                                    &selected_ray,
                                    state_.ui.view_sel_level,
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
        }
        DeviceEvent::Key(key) => match key.state {
            ElementState::Pressed => match key.physical_key {
                PhysicalKey::Code(KeyCode::ArrowLeft) => {
                    cycle_res_selected(state_, true);
                    redraw = true;
                }
                PhysicalKey::Code(KeyCode::ArrowRight) => {
                    cycle_res_selected(state_, false);
                    redraw = true;
                }
                _ => (),
            },
            ElementState::Released => (),
        },
        _ => (),
    }

    if redraw {
        // todo:This is overkill for certain keys. Just change the color of the one[s] in question, and set update.entities = true.
        draw_molecule(state_, scene, false);
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
    let white = [1., 1., 1., 1.];

    let mut scene = Scene {
        meshes: vec![
            Mesh::new_sphere(1., 16, 16),
            // Mesh::from_obj_file("sphere.obj"),
            Mesh::new_box(1., 1., 1.),
            Mesh::new_cylinder(1., BOND_RADIUS, 6),
            Mesh::new_box(1., 1., 1.), // todo: Temp. For VDW surface.
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
        move_sens: 10.0,
        ..Default::default()
    };
    let ui_settings = UiSettings {
        layout: UiLayout::Top,
        icon_path: Some("./resources/icon.png".to_owned()),
    };

    draw_molecule(&mut state, &mut scene, true);

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
