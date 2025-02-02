//! This module integraties this application with the graphics engine.

use std::{f32::consts::TAU, fmt};

use graphics::{
    screen_to_render, Camera, ControlScheme, DeviceEvent, ElementState, EngineUpdates, Entity,
    InputSettings, LightType, Lighting, Mesh, PointLight, Scene, UiLayout, UiSettings,
};
use lin_alg::f32::{Quaternion, Vec3, FORWARD};

use crate::{molecule::Molecule, ui::ui_handler, util::vec3_to_f32, State};

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

pub const SHELL_OPACITY: f32 = 0.01;

#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum MoleculeView {
    #[default]
    Sticks,
    Ribbon,
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
pub fn draw_molecule(entities: &mut Vec<Entity>, molecule: &Molecule, view: MoleculeView) {
    // todo: Update this capacity A/R as you flesh out your renders.
    *entities = Vec::with_capacity(molecule.bonds.len());

    if [MoleculeView::Spheres].contains(&view) {
        for atom in &molecule.atoms {
            entities.push(Entity::new(
                MESH_SPHERE,
                vec3_to_f32(atom.posit),
                Quaternion::new_identity(),
                0.6,
                atom.element.color(),
                ATOM_SHINYNESS,
            ));
        }
    }

    // for (atom0, atom1, bond) in &molecule.bonds {
    for bond in &molecule.bonds {
        // let center = (atom0.posit + atom1.posit) / 2.;
        let center = (bond.posit_0 + bond.posit_1) / 2.;

        let diff = vec3_to_f32(bond.posit_0 - bond.posit_1);
        let orientation = Quaternion::from_unit_vecs(FORWARD, diff.to_normalized());

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

fn event_handler(
    state_: &mut State,
    event: DeviceEvent,
    scene: &mut Scene,
    _dt: f32,
) -> EngineUpdates {
    match event {
        DeviceEvent::Button { button, state } => {
            if button == 1 {
                // Right click
                match state {
                    ElementState::Pressed => {
                        if let Some(cursor) = state_.ui.cursor_pos {
                            let selected_ray = screen_to_render(cursor, &scene.camera);

                            println!("Sel ray: {:?}", selected_ray);
                        }
                    }
                    ElementState::Released => (),
                }
            }
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
        draw_molecule(&mut entities, mol, state.ui.mol_view);
    }

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
            ambient_color: [-1., 1., 1., 0.5],
            ambient_intensity: 0.05,
            point_lights: vec![
                // Light from above
                PointLight {
                    type_: LightType::Omnidirectional,
                    position: Vec3::new(20., 20., 500.),
                    diffuse_color: [0.3, 0.4, 0.4, 1.],
                    specular_color: [0.3, 0.4, 0.4, 1.],
                    diffuse_intensity: 5_000.,
                    specular_intensity: 8_000.,
                },
                // Light from below
                PointLight {
                    type_: LightType::Omnidirectional,
                    position: Vec3::new(-20., 20., -500.),
                    diffuse_color: [0.3, 0.4, 0.4, 1.],
                    specular_color: [0.3, 0.4, 0.4, 1.],
                    diffuse_intensity: 5_000.,
                    specular_intensity: 8_000.,
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
        event_handler,
        ui_handler,
    );
}
