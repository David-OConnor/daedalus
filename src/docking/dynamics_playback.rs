//! Experimental molecular dynamics, with a playback system. Starting with fixed-ligand position only,
//! referencing the anchor.

use barnes_hut::Cube;
use graphics::Entity;
use lin_alg::f32::Vec3;

use crate::{docking::BindingEnergy, molecule::Atom};

#[derive(Debug, Default)]
pub struct SnapShot {
    pub time: f32,
    pub lig_atom_posits: Vec<Vec3>,
    pub lig_atom_accs: Vec<Vec3>,
    // todo: Velocity?
    // pub dt: f32,
    pub energy: BindingEnergy,
}

/// Keeps orientation fixed and body rigid, for now.
pub fn run_dynamics(receptor_atoms: &[Atom], lig_atoms: &[Atom]) -> Vec<SnapShot> {
    let n_steps = 1_000;
    let dt = 0.001;
    let mut result = Vec::with_capacity(n_steps);

    result
}

/// Body masses are separate from the snapshot, since it's invariant.
pub fn change_snapshot(
    entities: &mut Vec<Entity>,
    lig_entity_ids: &[usize],
    snapshot: &SnapShot,
    body_masses: &[f32],
) {
    // todo: Shells, acc vecs A/R
    *entities = Vec::with_capacity(snapshot.lig_atom_posits.len());

    entities.retain(|e| !lig_entity_ids.contains(&e.id));

    //
    // for (i, posit) in snapshot.body_posits.iter().enumerate() {
    //     let entity_size = f32::clamp(
    //         BODY_SIZE_SCALER * body_masses[i],
    //         BODY_SIZE_MIN,
    //         BODY_SIZE_MAX,
    //     );
    //     entities.push(Entity::new(
    //         MESH_SPHERE,
    //         *posit,
    //         Quaternion::new_identity(),
    //         entity_size,
    //         BODY_COLOR,
    //         BODY_SHINYNESS,
    //     ));
    //
    //     // entities.push(Entity::new(
    //     //     MESH_ARROW,
    //     //     *posit,
    //     //     Quaternion::from_unit_vecs(UP_VEC, snapshot.body_accs[i].to_normalized()),
    //     //     snapshot.body_accs[i].magnitude() * 0.2,
    //     //     ARROW_COLOR,
    //     //     ARROW_SHINYNESS,
    //     // ));
    // }
    //
    // for cube in &snapshot.tree_cubes {
    //     entities.push(Entity::new(
    //         MESH_CUBE,
    //         Vec3f32::new(
    //             cube.center.x as f32,
    //             cube.center.y as f32,
    //             cube.center.z as f32,
    //         ),
    //         Quaternion::new_identity(),
    //         cube.width as f32 * TREE_CUBE_SCALE_FACTOR,
    //         TREE_COLOR,
    //         TREE_SHINYNESS,
    //     ));
    // }
}
