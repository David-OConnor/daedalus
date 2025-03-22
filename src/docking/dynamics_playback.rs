//! Experimental molecular dynamics, with a playback system. Starting with fixed-ligand position only,
//! referencing the anchor.

use std::collections::HashMap;

use graphics::Entity;
use lin_alg::f32::{Quaternion, Vec3, Vec3x8, f32x8};
use rayon::prelude::*;

use crate::{
    docking::{BindingEnergy, ConformationType, Pose, SOFTENING_FACTOR_SQ_ELECTROSTATIC},
    element::Element,
    forces::{coulomb_force, lj_force},
    molecule::{Atom, Ligand, Molecule},
};

#[derive(Clone, Debug)]
struct BodyVdw {
    pub posit: Vec3,
    pub vel: Vec3,
    pub accel: Vec3,
    pub mass: f32,
    pub element: Element,
}

impl BodyVdw {
    pub fn from_atom(atom: &Atom) -> Self {
        Self {
            posit: atom.posit.into(),
            vel: Default::default(),
            accel: Default::default(),
            mass: atom.element.atomic_number() as f32,
            element: atom.element,
        }
    }
}

#[derive(Clone, Debug)]
struct BodyVdwx8 {
    pub posit: Vec3x8,
    pub vel: Vec3x8,
    pub accel: Vec3x8,
    pub mass: f32x8,
    pub element: [Element; 8],
}

#[derive(Debug, Default)]
pub struct Snapshot {
    pub time: f32,
    pub lig_atom_posits: Vec<Vec3>,
    // pub lig_atom_accs: Vec<Vec3>,
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

pub fn integrate_rk4x8<F>(body_tgt: &mut BodyVdwx8, id_tgt: usize, acc: &F, dt: f32)
where
    F: Fn(usize, Vec3x8, [Element; 8], f32x8) -> Vec3x8,
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

fn force_lj(
    posit_target: Vec3,
    el_tgt: Element,
    bodies_src: &[BodyVdw],
    lj_lut: &HashMap<(Element, Element), (f32, f32)>,
) -> Vec3 {
    // Compute the result in parallel and then sum the contributions.
    bodies_src
        .par_iter()
        .enumerate()
        .filter_map(|(i, body_source)| {
            let posit_src = body_source.posit;

            let diff = posit_src - posit_target;
            let dist = diff.magnitude();

            let dir = diff / dist; // Unit vec

            let (sigma, eps) = lj_lut.get(&(body_source.element, el_tgt)).unwrap();

            Some(lj_force(dir, dist, *sigma, *eps))
        })
        .reduce(Vec3::new_zero, |acc, elem| acc + elem) // Sum the contributions.
}

/// Runs on a specific target, for all sources.
fn acc_lj(
    posit_target: Vec3,
    el_tgt: Element,
    mass_tgt: f32,
    bodies_src: &[BodyVdw],
    lj_lut: &HashMap<(Element, Element), (f32, f32)>,
) -> Vec3 {
    let f = force_lj(posit_target, el_tgt, bodies_src, lj_lut);
    f / mass_tgt
}

fn bodies_from_atoms(atoms: &[Atom]) -> Vec<BodyVdw> {
    atoms.iter().map(|a| BodyVdw::from_atom(a)).collect()
}

/// Keeps orientation fixed and body rigid, for now.
pub fn build_vdw_dynamics(
    receptor_atoms: &[Atom],
    lig: &Ligand,
    lj_lut: &HashMap<(Element, Element), (f32, f32)>,
) -> Vec<Snapshot> {
    println!("Starting vuilding VDW dyanmics...");

    // todo: You should possibly add your pre-computed LJ pairs, instead of looking up each time.
    // todo: See this code from docking.

    let n_steps = 40_000;
    // todo: Adaptive timestep.
    let dt = 0.00001;
    let snapshot_ratio = 1; // todo

    let mut result = Vec::with_capacity(n_steps + 1); // +1 for the initial snapshot.

    let mut time_elapsed = 0.;

    // Static
    let bodies_rec = bodies_from_atoms(receptor_atoms);
    // These move.
    let mut bodies_lig = bodies_from_atoms(&lig.molecule.atoms);

    let mut body_ligand_rigid = BodyVdw::from_atom(&lig.molecule.atoms[lig.anchor_atom]);
    let mut orientation = lig.pose.orientation;

    let mut ligand_mass = 0.;
    for atom in &lig.molecule.atoms {
        ligand_mass += atom.element.atomic_number() as f32; // Arbitrary mass scale for now.
    }

    // Initial snapshot
    result.push(Snapshot {
        time: time_elapsed,
        lig_atom_posits: bodies_lig.iter().map(|b| b.posit.into()).collect(),
        energy: BindingEnergy::default(), // todo
    });

    let acc_fn = |id_: usize, posit_target, el_target, mass_tgt| {
        acc_lj(posit_target, el_target, mass_tgt, &bodies_rec, &lj_lut)
    };

    for t in 0..n_steps {
        // These net values are for our rigid body model.
        // let mut acc_net = Vec3::new_zero();
        // bodies_lig
        //     .par_iter_mut()
        //     .for_each(|body_lig| {
        // // todo: You may need force instead of accel.
        //         integrate_rk4(body_lig, 0, &acc_fn, dt);
        //     });

        let mut vel_net = Vec3::new_zero();
        // let acc_net = bodies_lig
        let f_net = bodies_lig
            .par_iter_mut()
            .map(|body_lig| {
                force_lj(body_lig.posit, body_lig.element, &bodies_rec, &lj_lut)
            })
            .reduce(|| Vec3::new_zero(), |a, b| a + b);

        let acc_net = f_net / ligand_mass;

        vel_net += acc_net * dt;

        for body in &mut bodies_lig {
            body.posit += vel_net * dt;
        }

        // Keeps the same signature as individual, but doesn't use parts of it.
        let acc_fn_net = |id_, posit_target, el_target, mass_tgt| acc_net;
        integrate_rk4(&mut body_ligand_rigid, 0, &acc_fn_net, dt);
        //
        // let pose = Pose {
        //     anchor_posit: 0,
        //     orientation: 0,
        //     conformation_type: ConformationType::Flexible {torsions: Vec::new() }
        // };

        // let pos = lig.position_atoms(None);

        time_elapsed += dt;

        // Save the current state to a snapshot, for later playback.
        if t % snapshot_ratio == 0 {
            result.push(Snapshot {
                time: time_elapsed,
                lig_atom_posits: bodies_lig.iter().map(|b| b.posit.into()).collect(),
                energy: BindingEnergy::default(), // todo
            });
        }
    }

    println!("Complete.");
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
    // *entities = Vec::with_capacity(snapshot.lig_atom_posits.len());

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
