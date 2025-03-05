//! This module integraties this application with the graphics engine.

use std::f32::consts::TAU;

use graphics::{
    Camera, ControlScheme, EngineUpdates, FWD_VEC, InputSettings, LightType, Lighting, Mesh,
    PointLight, RIGHT_VEC, Scene, ScrollBehavior, UiLayout, UiSettings,
};
use lin_alg::f32::{Quaternion, Vec3};

use crate::{
    State,
    docking::DockingInit,
    inputs,
    inputs::{MOVEMENT_SENS, RUN_FACTOR, SCROLL_MOVE_AMT, SCROLL_ROTATE_AMT},
    mol_drawing,
    ui::ui_handler,
};

pub type Color = (f32, f32, f32);

const WINDOW_TITLE: &str = "Bio Chem View";
const WINDOW_SIZE_X: f32 = 1_400.;
const WINDOW_SIZE_Y: f32 = 1_000.;
const BACKGROUND_COLOR: Color = (0., 0., 0.);
pub const RENDER_DIST: f32 = 1_000.;

// todo: Shinyness broken?
pub const ATOM_SHINYNESS: f32 = 0.9;
pub const BODY_SHINYNESS: f32 = 0.9;

// Keep this in sync with mesh init.
pub const MESH_SPHERE: usize = 0;
pub const MESH_CUBE: usize = 1;
pub const MESH_BOND: usize = 2;
pub const MESH_SPHERE_LOWRES: usize = 3;
pub const MESH_SURFACE: usize = 4; // Van Der Waals surface.
pub const MESH_BOX: usize = 5;

pub const BALL_STICK_RADIUS: f32 = 0.3;
pub const BALL_STICK_RADIUS_H: f32 = 0.2;

pub const BOND_RADIUS: f32 = 0.12;
// const BOND_CAP_RADIUS: f32 = 1./BOND_RADIUS;
pub const BOND_RADIUS_DOUBLE: f32 = 0.07;

pub const RADIUS_SFC_DOT: f32 = 0.05;
pub const COLOR_SFC_DOT: Color = (0.7, 0.7, 0.7);
pub const COLOR_DOCKING_BOX: Color = (0.3, 0.3, 0.9);

pub const COLOR_SELECTED: Color = (1., 0., 0.);
// pub const COLOR_H_BOND: Color = (0.2, 0.2, 1.);
pub const COLOR_H_BOND: Color = (1., 0.5, 0.1);
pub const RADIUS_H_BOND: f32 = 0.2; // A scaler relative to covalent sticks.

pub const SHELL_OPACITY: f32 = 0.01;

// From the farthest molecule.
pub const CAM_INIT_OFFSET: f32 = 10.;

pub const COLOR_AA_NON_RESIDUE: Color = (0., 0.8, 1.0);

// A higher value will result in a less-dramatic brightness change with distance.
const FLASHLIGHT_OFFSET: f32 = 10.;
const FLASHLIGHT_FOV: f32 = TAU / 16.;
pub const OUTSIDE_LIGHTING_OFFSET: f32 = 900.;
pub const DOCKING_LIGHT_INTENSITY: f32 = 0.3;

/// Set the flashlight to be a little bit behind the camera; prevents too dramatic of an intensity
/// scaling on the object looked at, WRT distance.
pub fn set_flashlight(scene: &mut Scene) {
    let light = &mut scene.lighting.point_lights[0];
    light.position = scene.camera.position
        + scene
            .camera
            .orientation
            .rotate_vec(Vec3::new(0., 0., -FLASHLIGHT_OFFSET));

    // todo: Put back. Problem with di
    // light.type_ = LightType::Directional {
    //     direction: scene.camera.orientation.rotate_vec(FWD_VEC),
    //     fov: FLASHLIGHT_FOV,
    // };
}

/// Set lighting based on the center and size of the molecule.
pub fn set_static_light(scene: &mut Scene, center: Vec3, size: f32) {
    scene.lighting.point_lights[1].position =
        center + Vec3::new(40., (size + OUTSIDE_LIGHTING_OFFSET), 0.);
}

/// Set lighting based on the docking location.
pub fn set_docking_light(scene: &mut Scene, docking_init: Option<&DockingInit>) {
    let mut light = &mut scene.lighting.point_lights[2];

    match docking_init {
        Some(docking_init) => {
            let intensity = DOCKING_LIGHT_INTENSITY * docking_init.site_box_size as f32;

            light.position = docking_init.site_center.into();
            light.diffuse_intensity = intensity;
            light.specular_intensity = intensity;
        }
        None => {
            light.diffuse_intensity = 0.;
            light.specular_intensity = 0.;
        }
    }
}

/// This runs each frame. Currently, no updates.
fn render_handler(_state: &mut State, _scene: &mut Scene, _dt: f32) -> EngineUpdates {
    EngineUpdates::default()
}

/// Entry point to our render and event loop.
pub fn render(mut state: State) {
    let white = [1., 1., 1., 0.5];
    let pink = [1., 0., 1., 1.];

    let mut scene = Scene {
        meshes: vec![
            Mesh::new_sphere(1., 3),
            Mesh::new_box(1., 1., 1.),
            Mesh::new_cylinder(1., BOND_RADIUS, 20),
            Mesh::new_sphere(1., 1),   // low-res sphere
            Mesh::new_box(1., 1., 1.), // Placeholder for a VDW surface; populated later.
            Mesh::new_box(1., 1., 1.),
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
        // Lighting is set when drawing mwolecules; placeholder here.
        lighting: Lighting {
            ambient_color: white,
            ambient_intensity: 0.005,
            point_lights: vec![
                // The camera-oriented *flashlight*. Moves with the camera.
                PointLight {
                    // todo: temp rm TS
                    // type_: LightType::Directional{ direction: Vec3::new_zero(), fov: FLASHLIGHT_FOV },
                    diffuse_intensity: 24.,
                    specular_intensity: 24.,
                    ..Default::default()
                },
                // A fixed light, from *above*
                PointLight {
                    diffuse_color: white,
                    specular_color: white,
                    diffuse_intensity: 30_000.,
                    specular_intensity: 30_000.,
                    ..Default::default()
                },
                // A light on the docking site, if applicable.
                PointLight {
                    diffuse_color: pink,
                    specular_color: pink,
                    diffuse_intensity: 0.,
                    specular_intensity: 0.,
                    ..Default::default()
                },
            ],
        },
        input_settings: InputSettings {
            // control_scheme: ControlScheme::FreeCamera,
            control_scheme: state.to_save.control_scheme,
            move_sens: MOVEMENT_SENS,
            run_factor: RUN_FACTOR,
            scroll_behavior: ScrollBehavior::MoveRoll {
                move_amt: SCROLL_MOVE_AMT,
                rotate_amt: SCROLL_ROTATE_AMT,
            },
            ..Default::default()
        },
        background_color: BACKGROUND_COLOR,
        window_size: (WINDOW_SIZE_X, WINDOW_SIZE_Y),
        window_title: WINDOW_TITLE.to_owned(),
    };

    let ui_settings = UiSettings {
        layout: UiLayout::Top,
        icon_path: Some("resources/icon.png".to_owned()),
    };

    mol_drawing::draw_molecule(&mut state, &mut scene, true);
    mol_drawing::draw_ligand(&mut state, &mut scene, true);

    set_flashlight(&mut scene);

    graphics::run(
        state,
        scene,
        ui_settings,
        Default::default(),
        render_handler,
        inputs::event_dev_handler,
        inputs::event_win_handler,
        ui_handler,
    );
}
