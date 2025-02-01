//! This module integraties this application with the graphics engine.

use std::f32::consts::TAU;

use graphics::{
    Camera, ControlScheme, DeviceEvent, EngineUpdates, Entity, InputSettings, LightType, Lighting,
    Mesh, PointLight, Scene, UiLayout, UiSettings,
};
use lin_alg::f32::{Quaternion, Vec3, FORWARD, UP};

use crate::{ui::ui_handler, util::vec3_to_f32, Molecule, State};

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

#[derive(Clone, Copy, Debug, Default)]
pub enum MoleculeView {
    #[default]
    BallAndStick,
    SpaceFilling,
    Cartoon,
}

pub fn draw_molecule(entities: &mut Vec<Entity>, molecule: &Molecule) {
    *entities = Vec::with_capacity(molecule.atoms.len());

    for atom in &molecule.atoms {
        entities.push(Entity::new(
            MESH_SPHERE,
            vec3_to_f32(atom.posit),
            Quaternion::new_identity(),
            0.5,
            atom.element.color(),
            ATOM_SHINYNESS,
        ));
    }

    // for (atom0, atom1, bond) in &molecule.bonds {
    for bond in &molecule.bonds {
        // let center = (atom0.posit + atom1.posit) / 2.;
        let center = (bond.posit_0 + bond.posit_1) / 2.;

        let diff = vec3_to_f32(bond.posit_0 - bond.posit_1);
        // todo: FWD?
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
