//! Experimental molecular dynamics, with a playback system. Starting with fixed-ligand position only,
//! referencing the anchor.

use std::collections::HashMap;

use barnes_hut::Cube;
use graphics::Entity;
use lin_alg::f32::Vec3;
use rayon::prelude::*;

use crate::{
    docking::{BindingEnergy, SOFTENING_FACTOR_SQ_ELECTROSTATIC},
    element::Element,
    forces::{coulomb_force, lj_force},
    molecule::{Atom, Molecule},
};

#[derive(Clone, Debug)]
struct BodyVdw {
    pub posit: Vec3,
    pub vel: Vec3,
    pub accel: Vec3,
    pub mass: f32,
    pub element: Element,
}

#[derive(Debug, Default)]
pub struct Snapshot {
    pub time: f32,
    pub lig_atom_posits: Vec<Vec3>,
    pub lig_atom_accs: Vec<Vec3>,
    // todo: Velocity?
    // pub dt: f32,
    pub energy: BindingEnergy,
}

/// Compute acceleration, position, and velocity, using RK4.
/// The acc fn: (id, target posit, target charge) -> Acceleration.
/// todo: C+P from causal grav.
pub fn integrate_rk4<F>(body_tgt: &mut BodyVdw, id_tgt: usize, acc: &F, dt: f32)
where
    F: Fn(usize, Vec3, Element, f32) -> Vec3,
{
    // Step 1: Calculate the k-values for position and velocity
    body_tgt.accel = acc(id_tgt, body_tgt.posit, body_tgt.element, body_tgt.mass);

    let k1_v = body_tgt.accel * dt;
    let k1_pos = body_tgt.vel * dt;

    let body_pos_k2 = body_tgt.posit + k1_pos * 0.5;
    let k2_v = acc(id_tgt, body_pos_k2, body_tgt.element, body_tgt.mass) * dt;
    let k2_pos = (body_tgt.vel + k1_v * 0.5) * dt;

    let body_pos_k3 = body_tgt.posit + k2_pos * 0.5;
    let k3_v = acc(id_tgt, body_pos_k3, body_tgt.element, body_tgt.mass) * dt;
    let k3_pos = (body_tgt.vel + k2_v * 0.5) * dt;

    let body_pos_k4 = body_tgt.posit + k3_pos;
    let k4_v = acc(id_tgt, body_pos_k4, body_tgt.element, body_tgt.mass) * dt;
    let k4_pos = (body_tgt.vel + k3_v) * dt;

    // Step 2: Update position and velocity using weighted average of k-values
    body_tgt.vel += (k1_v + k2_v * 2. + k3_v * 2. + k4_v) / 6.;
    body_tgt.posit += (k1_pos + k2_pos * 2. + k3_pos * 2. + k4_pos) / 6.;
}

/// Runs on a specific target, for all sources.
fn acc_coulomb(
    posit_target: Vec3,
    id_target: usize,
    el_tgt: Element,
    mass_tgt: f32,
    bodies_src: &[BodyVdw],
    lj_lut: &HashMap<(Element, Element), (f32, f32)>,
) -> Vec3 {
    // Compute the result in parallel and then sum the contributions.
    bodies_src
        .par_iter()
        .enumerate()
        .filter_map(|(i, body_source)| {
            if i == id_target {
                return None; // Skip self-interaction.
            }

            let posit_src = Vec3::new_zero();

            let diff = posit_src - posit_target;
            let dist = diff.magnitude();

            let dir = diff / dist; // Unit vec

            let (sigma, eps) = lj_lut.get(&(body_source.element, el_tgt)).unwrap();

            let f = lj_force(dir, dist, *sigma, *eps);

            Some(f / mass_tgt)
        })
        .reduce(Vec3::new_zero, |acc, elem| acc + elem) // Sum the contributions.
}

/// Keeps orientation fixed and body rigid, for now.
pub fn build_vdw_dynamics(
    receptor_atoms: &[Atom],
    lig_atoms: &[Atom],
    lj_lut: &HashMap<(Element, Element), (f32, f32)>,
) -> Vec<Snapshot> {
    let n_steps = 1_000;
    let dt = 0.001;
    let snapshot_ratio = 10;

    let mut result = Vec::with_capacity(n_steps);

    let mut time_elapsed = 0.;

    // todo: Rel unit system. proton mass = 1?
    // todo: Force constant?? Maybe keep at 1.

    // Static
    let bodies_rec: Vec<_> = receptor_atoms
        .iter()
        .map(|a| BodyVdw {
            posit: a.posit.into(),
            vel: Vec3::new_zero(),
            accel: Vec3::new_zero(),
            mass: a.element.atomic_number() as f32,
            element: a.element,
        })
        .collect();

    // These move.
    let mut bodies_lig: Vec<_> = lig_atoms
        .iter()
        .map(|a| BodyVdw {
            posit: a.posit.into(),
            vel: Vec3::new_zero(),
            accel: Vec3::new_zero(),
            mass: a.element.atomic_number() as f32,
            element: a.element,
        })
        .collect();

    let acc = |id_target, posit_target, el_target, mass_tgt| {
        acc_coulomb(
            posit_target,
            id_target,
            el_target,
            mass_tgt,
            &bodies_rec,
            &lj_lut,
        )
    };

    for t in 0..n_steps {
        bodies_lig
            .par_iter_mut()
            .enumerate()
            .for_each(|(id_lig, body_lig)| {
                integrate_rk4(body_lig, id_lig, &acc, dt);
            });

        time_elapsed += dt;

        // Save the current state to a snapshot, for later playback.
        if t % snapshot_ratio == 0 {
            result.push(Snapshot {
                time: time_elapsed,
                lig_atom_posits: bodies_lig.iter().map(|b| b.posit.into()).collect(),
                lig_atom_accs: bodies_lig.iter().map(|b| b.accel.into()).collect(),
                energy: BindingEnergy::default(), // todo
            });
        }
    }
    result
}

/// Body masses are separate from the snapshot, since it's invariant.
pub fn change_snapshot(
    entities: &mut Vec<Entity>,
    lig_mol: &mut Molecule,
    lig_entity_ids: &[usize],
    snapshot: &Snapshot,
) {
    // todo: Initial hack: Get working as individual particles. Then, try to incorporate
    // todo fixed rotation of the molecule, fixed movement, bond flexes etc.

    // todo: For now, redraw all entities; eventually, don't!
    // entities.retain(|e| !lig_entity_ids.contains(&e.id));
    *entities = Vec::with_capacity(snapshot.lig_atom_posits.len());

    // todo: You need to enforce rigidity.
    for (i, posit) in snapshot.lig_atom_posits.iter().enumerate() {
        lig_mol.atoms[i].posit = (*posit).into();
    }

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
