//! This module integraties this application with the graphics engine.

use std::{f32::consts::TAU, fmt};

use graphics::{
    event::WindowEvent, Camera, ControlScheme, DeviceEvent, ElementState, EngineUpdates, Entity,
    InputSettings, LightType, Lighting, Mesh, PointLight, Scene, UiLayout, UiSettings,
};
use lin_alg::f32::{Quaternion, Vec3, FORWARD, UP};

use crate::{
    molecule::{BondCount, Molecule},
    ui::ui_handler,
    util::{find_selected_atom, points_along_ray, vec3_to_f32},
    State,
};

type Color = (f32, f32, f32);

const WINDOW_TITLE: &str = "Molecular docking";
const WINDOW_SIZE_X: f32 = 1_600.;
const WINDOW_SIZE_Y: f32 = 1_000.;
const BACKGROUND_COLOR: Color = (0., 0., 0.);
const RENDER_DIST: f32 = 1_000.;

pub const ATOM_SHINYNESS: f32 = 2.;
pub const BODY_SHINYNESS: f32 = 2.;

// Keep this in sync with mesh init.
pub const MESH_SPHERE: usize = 0;
pub const MESH_CUBE: usize = 1;
pub const MESH_ARROW: usize = 2;
pub const MESH_BOND: usize = 3;

// todo: By bond type etc
const BOND_COLOR: Color = (0.2, 0.2, 0.2);

pub const COLOR_SELECTED: Color = (1., 1., 1.);

pub const SHELL_OPACITY: f32 = 0.01;

#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum MoleculeView {
    Sticks,
    Ribbon,
    #[default]
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
            Self::Spheres => "Spheres",
            Self::Cartoon => "Cartoon",
            Self::Surface => "Surface",
            Self::Mesh => "Mesh",
            Self::Dots => "Dots",
        };

        write!(f, "{val}")
    }
}
/// Refreshes entities with the model passed.
pub fn draw_molecule(
    entities: &mut Vec<Entity>,
    molecule: &Molecule,
    view: MoleculeView,
    selected: Option<usize>,
) {
    // todo: Update this capacity A/R as you flesh out your renders.
    // *entities = Vec::with_capacity(molecule.bonds.len());
    *entities = Vec::new();

    if [MoleculeView::Spheres].contains(&view) {
        for (i, atom) in molecule.atoms.iter().enumerate() {
            let mut color = atom.element.color();
            if let Some(hl) = selected {
                if hl == i {
                    color = COLOR_SELECTED
                }
            }

            entities.push(Entity::new(
                MESH_SPHERE,
                vec3_to_f32(atom.posit),
                Quaternion::new_identity(),
                // 1.5,
                0.4, // todo: temp testing highlight
                color,
                ATOM_SHINYNESS,
            ));
        }
    }

    let rot_ortho = Quaternion::from_unit_vecs(UP, FORWARD);

    // for (atom0, atom1, bond) in &molecule.bonds {
    for bond in &molecule.bonds {
        if view == MoleculeView::Ribbon && !bond.is_backbone {
            continue;
        }

        // let center = (atom0.posit + atom1.posit) / 2.;
        let center = (bond.posit_0 + bond.posit_1) / 2.;

        let diff = vec3_to_f32(bond.posit_0 - bond.posit_1);
        let diff_unit = diff.to_normalized();
        let orientation = Quaternion::from_unit_vecs(FORWARD, diff_unit);

        match bond.bond_count {
            BondCount::Double => {
                // Draw two offset bond cylinders.
                // todo: QC this using quat logic. Seems to be working though.
                let rotator = rot_ortho * orientation;

                let offset_a = rotator.rotate_vec(Vec3::new(0.2, 0., 0.));
                let offset_b = rotator.rotate_vec(Vec3::new(-0.2, 0., 0.));

                entities.push(Entity::new(
                    MESH_BOND,
                    vec3_to_f32(center) + offset_a,
                    orientation,
                    1. * diff.magnitude(),
                    BOND_COLOR,
                    BODY_SHINYNESS,
                ));

                entities.push(Entity::new(
                    MESH_BOND,
                    vec3_to_f32(center) + offset_b,
                    orientation,
                    1. * diff.magnitude(),
                    BOND_COLOR,
                    BODY_SHINYNESS,
                ));
            }
            _ => {
                entities.push(Entity::new(
                    MESH_BOND,
                    vec3_to_f32(center),
                    orientation,
                    1. * diff.magnitude(),
                    BOND_COLOR,
                    BODY_SHINYNESS,
                ));
            }
        }
    }
}

fn event_dev_handler(
    state_: &mut State,
    event: DeviceEvent,
    scene: &mut Scene,
    _dt: f32,
) -> EngineUpdates {
    let mut updates = EngineUpdates::default();
    match event {
        DeviceEvent::Button { button, state } => {
            if button == 1 {
                // Right click
                match state {
                    ElementState::Pressed => {
                        if let Some(cursor) = state_.ui.cursor_pos {
                            let selected_ray = scene.screen_to_render(cursor);

                            if let Some(mol) = &state_.molecule {
                                let atoms_sel = points_along_ray(selected_ray, &mol.atoms, 0.6);

                                state_.atom_selected =
                                    find_selected_atom(&atoms_sel, &mol.atoms, &selected_ray);

                                draw_molecule(
                                    &mut scene.entities,
                                    mol,
                                    state_.ui.mol_view,
                                    state_.atom_selected,
                                );
                                updates.entities = true;
                            }
                        }
                    }
                    ElementState::Released => (),
                }
            }
        }
        _ => (),
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
            device_id,
            position,
        } => {
            // println!("Cursor: {:?}", position);
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
pub fn render(state: State) {
    let mut entities = Vec::new();
    if let Some(mol) = &state.molecule {
        draw_molecule(&mut entities, mol, state.ui.mol_view, state.atom_selected);
    }

    let white = [1., 1., 1., 1.];

    let scene = Scene {
        meshes: vec![
            Mesh::new_sphere(1., 12, 12),
            Mesh::new_box(1., 1., 1.),
            Mesh::new_arrow(1., 0.05, 8),
            Mesh::new_cylinder(1., 0.05, 6),
        ],
        entities,
        camera: Camera {
            fov_y: TAU / 8.,
            position: Vec3::new(0., 0., -60.),
            far: RENDER_DIST,
            near: 0.2, // todo: Adjust A/R
            // orientation: Quaternion::from_axis_angle(Vec3::new(1., 0., 0.), TAU / 16.),
            orientation: Quaternion::from_axis_angle(Vec3::new(1., 0., 0.), 0.),
            ..Default::default()
        },
        lighting: Lighting {
            ambient_color: white,
            ambient_intensity: 0.01,
            point_lights: vec![
                // Light from above
                PointLight {
                    type_: LightType::Omnidirectional,
                    position: Vec3::new(20., 20., 500.),
                    diffuse_color: white,
                    specular_color: white,
                    diffuse_intensity: 4_000.,
                    specular_intensity: 10_000.,
                },
                // Light from below
                PointLight {
                    type_: LightType::Omnidirectional,
                    position: Vec3::new(20., 20., -500.),
                    diffuse_color: white,
                    specular_color: white,
                    diffuse_intensity: 4_000.,
                    specular_intensity: 10_000.,
                },
            ],
        },
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

    // todo: Initialize entities here.

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
