//! This module integraties this application with the graphics engine.

use std::f32::consts::TAU;

use graphics::{Camera, ControlScheme, DeviceEvent, EngineUpdates, InputSettings, LightType, Lighting, Mesh, PointLight, Scene, UiLayout, UiSettings, Entity};
use lin_alg::f32::{Quaternion, Vec3};

use crate::{ui::ui_handler, State, Molecule};

type Color = (f32, f32, f32);

const WINDOW_TITLE: &str = "Molecular docking";
const WINDOW_SIZE_X: f32 = 1_600.;
const WINDOW_SIZE_Y: f32 = 1_000.;
const BACKGROUND_COLOR: Color = (0., 0., 0.);

const RENDER_DIST: f32 = 1_000.;

pub const BODY_COLOR: Color = (1.0, 0.4, 0.4);
pub const BODY_SHINYNESS: f32 = 2.;

// Keep this in sync with mesh init.
pub const MESH_SPHERE: usize = 0;
pub const MESH_CUBE: usize = 1;
pub const MESH_ARROW: usize = 2;

pub const SHELL_OPACITY: f32 = 0.01;


pub fn draw_molecule(entities: &mut Vec<Entity>, molecule: &Molecule) {
    *entities = Vec::with_capacity(molecule.atoms.len());

    for atom in &molecule.atoms {
        entities.push(Entity::new(
            MESH_SPHERE,
            Vec3::new(atom.posit.x as f32, atom.posit.y as f32, atom.posit.z as f32),
            Quaternion::new_identity(),
            0.5,
            BODY_COLOR,
            BODY_SHINYNESS,
        ));
    }

}

fn event_handler(
    _state: &mut State,
    _event: DeviceEvent,
    _scene: &mut Scene,
    _dt: f32,
) -> EngineUpdates {
    EngineUpdates::default()
}

/// This runs each frame. Currently, no updates.
fn render_handler(_state: &mut State, _scene: &mut Scene, _dt: f32) -> EngineUpdates {
    EngineUpdates::default()
}

/// Entry point to our render and event loop.
pub fn render(state: State) {
    let mut entities = Vec::new();
    if let Some(mol) = &state.molecule {
        draw_molecule(&mut entities, mol);
    } else {
        // todo: Our engine crashes if we don't have any entities.
        entities.push(Entity::new(
            MESH_SPHERE,
            Vec3::new_zero(),
            Quaternion::new_identity(),
            0.5,
            BODY_COLOR,
            BODY_SHINYNESS,
        ));
    }

    let scene = Scene {
        meshes: vec![
            Mesh::new_sphere(1., 12, 12),
            Mesh::new_box(1., 1., 1.),
            Mesh::new_arrow(1., 0.05, 8),
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
