//! This module integraties this application with the graphics engine.

use std::f32::consts::TAU;

use graphics::{
    Camera, EngineUpdates, GraphicsSettings, InputSettings, Lighting, Mesh, PointLight, RIGHT_VEC,
    Scene, ScrollBehavior, UiLayout, UiSettings,
};
use lin_alg::f32::{Quaternion, Vec3};

use crate::{
    State,
    docking_v2::DockingSite,
    drawing,
    drawing::BOND_RADIUS,
    inputs,
    inputs::{RUN_FACTOR, SCROLL_MOVE_AMT, SCROLL_ROTATE_AMT},
    ui::{
        cam::{FOG_DIST_DEFAULT, RENDER_DIST_FAR, RENDER_DIST_NEAR, calc_fog_dists},
        ui_handler,
    },
};

pub type Color = (f32, f32, f32);

const WINDOW_TITLE: &str = "Daedalus";
const WINDOW_SIZE_X: f32 = 1_600.;
const WINDOW_SIZE_Y: f32 = 1_200.;
pub const BACKGROUND_COLOR: Color = (0., 0., 0.);

// todo: Shinyness broken?
pub const ATOM_SHININESS: f32 = 0.9;
pub const BODY_SHINYNESS: f32 = 0.9;

// Keep this in sync with mesh init.
pub const MESH_SPHERE_HIGHRES: usize = 0;
pub const MESH_CUBE: usize = 1;
pub const MESH_BOND: usize = 2;
pub const MESH_SPHERE_LOWRES: usize = 3;
pub const MESH_SPHERE_MEDRES: usize = 4;
pub const MESH_DOCKING_BOX: usize = 5;
pub const MESH_SOLVENT_SURFACE: usize = 6; // Van Der Waals surface.
pub const MESH_DOCKING_SURFACE: usize = 7;
pub const MESH_DENSITY_SURFACE: usize = 8;
pub const MESH_SECONDARY_STRUCTURE: usize = 9;

pub const BALL_STICK_RADIUS: f32 = 0.3;
pub const BALL_STICK_RADIUS_H: f32 = 0.2;

pub const BALL_RADIUS_WATER_O: f32 = 0.09;
pub const BALL_RADIUS_WATER_H: f32 = 0.06;
pub const WATER_BOND_THICKNESS: f32 = 0.1;
pub const WATER_OPACITY: f32 = 1.;

pub const SHELL_OPACITY: f32 = 0.01;

// From the farthest molecule.
pub const CAM_INIT_OFFSET: f32 = 10.;

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
        center + Vec3::new(40., size + OUTSIDE_LIGHTING_OFFSET, 0.);
}

/// Set lighting based on the docking location.
pub fn set_docking_light(scene: &mut Scene, docking_init: Option<&DockingSite>) {
    let light = &mut scene.lighting.point_lights[2];

    match docking_init {
        Some(docking_init) => {
            let intensity = DOCKING_LIGHT_INTENSITY * docking_init.site_radius as f32;

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

    let (fog_start, fog_end) = calc_fog_dists(FOG_DIST_DEFAULT);

    let camera = Camera {
        fov_y: TAU / 8.,
        position: Vec3::new(0., 0., -60.),
        far: RENDER_DIST_FAR,
        near: RENDER_DIST_NEAR,
        // orientation: Quaternion::from_axis_angle(Vec3::new(1., 0., 0.), TAU / 16.),
        orientation: Quaternion::from_axis_angle(RIGHT_VEC, 0.),
        // This affects how aggressive the fog is in fading objects to the background.
        // A lower value will leave objects dim, but visible.
        fog_density: 4.,
        fog_power: 6.,
        fog_start,
        fog_end,
        fog_color: [BACKGROUND_COLOR.0, BACKGROUND_COLOR.1, BACKGROUND_COLOR.2],
        ..Default::default()
    };

    let mut scene = Scene {
        meshes: vec![
            Mesh::new_sphere(1., 3),
            Mesh::new_box(1., 1., 1.),
            Mesh::new_cylinder(1., BOND_RADIUS, 14),
            Mesh::new_sphere(1., 1), // low-res sphere
            Mesh::new_sphere(1., 2), // med-res sphere
            Mesh::new_box(1., 1., 1.),
            Mesh::new_box(1., 1., 1.), // Placeholder for VDW/SA surface; populated later.
            Mesh::new_box(1., 1., 1.), // Placeholder for docking site sufrace; populated later.
            Mesh::new_box(1., 1., 1.), // Placeholder for density sufrace; populated later.
            Mesh::new_box(1., 1., 1.), // Placeholder for secondary structure surface; populated later.
        ],
        entities: Vec::new(),
        gaussians: Vec::new(),
        camera,
        // Lighting is set when drawing molecules; placeholder here.
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
            control_scheme: state.to_save.control_scheme,
            move_sens: state.to_save.movement_speed as f32,
            rotate_sens: (state.to_save.rotation_sens as f32) / 100.,
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

    drawing::draw_peptide(&mut state, &mut scene);
    drawing::draw_all_ligs(&mut state, &mut scene);

    set_flashlight(&mut scene);

    let msaa_samples = state.to_save.msaa as u8 as u32;

    graphics::run(
        state,
        scene,
        ui_settings,
        GraphicsSettings { msaa_samples },
        render_handler,
        inputs::event_dev_handler,
        inputs::event_win_handler,
        ui_handler,
    );
}
