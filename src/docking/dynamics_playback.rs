//! Experimental molecular dynamics, with a playback system. Starting with fixed-ligand position only,
//! referencing the anchor.

use std::{collections::HashMap, time::Instant};

use graphics::Entity;
use lin_alg::f32::{Quaternion, Quaternionx8, Vec3, Vec3x8, f32x8, unpack_f32, unpack_vec3};
use rayon::prelude::*;

use crate::{
    docking::{BindingEnergy, ConformationType, Pose, SOFTENING_FACTOR_SQ_ELECTROSTATIC},
    element::Element,
    forces::{coulomb_force, lj_force, lj_force_x8},
    molecule::{Atom, Ligand, Molecule},
};

#[derive(Clone, Debug)]
struct BodyVdw {
    pub posit: Vec3,
    pub vel: Vec3,
    pub accel: Vec3,
    pub mass: f32,
    pub element: Element,
    /// Only relevant for the overall body; not the individuals.
    pub orientation: Quaternion,
}

impl BodyVdw {
    pub fn from_atom(atom: &Atom) -> Self {
        Self {
            posit: atom.posit.into(),
            vel: Default::default(),
            accel: Default::default(),
            mass: atom.element.atomic_number() as f32,
            element: atom.element,
            orientation: Default::default(),
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
    /// Only relevant for the overall body; not the individuals.
    pub orientation: Quaternionx8,
}

impl BodyVdwx8 {
    pub fn from_array(bodies: [BodyVdw; 8]) -> Self {
        let mut posits = [Vec3::new_zero(); 8];
        let mut vels = [Vec3::new_zero(); 8];
        let mut accels = [Vec3::new_zero(); 8];
        let mut masses = [0.0; 8];
        // Replace `Element::H` (for example) with some valid default for your `Element` type:
        let mut elements = [Element::Hydrogen; 8];
        let mut orients = [Quaternion::default(); 8];

        for (i, body) in bodies.into_iter().enumerate() {
            posits[i] = body.posit;
            vels[i] = body.vel;
            accels[i] = body.accel;
            masses[i] = body.mass;
            elements[i] = body.element;
            orients[i] = body.orientation;
        }

        Self {
            posit: Vec3x8::from_array(posits),
            vel: Vec3x8::from_array(vels),
            accel: Vec3x8::from_array(accels),
            mass: f32x8::from_array(masses),
            element: elements,
            orientation: Quaternionx8::from_array(orients),
        }
    }
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

/// Defaults to `Config::dt_integration`, but becomes more precise when
/// bodies are close. This is a global DT, vice local only for those bodies.
fn calc_dt_dynamic(
    bodies_src: &[BodyVdw],
    bodies_tgt: &[BodyVdw],
    dt_scaler: f32,
    dt_max: f32,
) -> f32 {
    let mut result = dt_max;

    // todo: Consider cacheing the distances, so this second iteration can be reused.
    for (id_tgt, body_tgt) in bodies_tgt.iter().enumerate() {
        for (i_src, body_src) in bodies_src.iter().enumerate() {
            // if i_src == id_tgt {
            //     continue; // self-interaction.
            // }

            let dist = (body_src.posit - body_tgt.posit).magnitude();
            let rel_velocity = (body_src.vel - body_tgt.vel).magnitude();
            let dt = dt_scaler * dist / rel_velocity;
            if dt < result {
                result = dt;
            }
        }
    }

    result
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

// todo: We may not use this in practice, as we compute this after the parallel computations are complete, I think.
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
    // distances: &[Vec<f32>],
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

fn force_lj_x8(
    posit_target: Vec3x8,
    el_tgt: [Element; 8],
    bodies_src: &[BodyVdwx8],
    // distances: &[Vec<f32x8>],
    lj_lut: &HashMap<(Element, Element), (f32, f32)>,
    chunks_src: usize,
    lanes_tgt: usize,
    valid_lanes_src_last: usize,
) -> Vec3x8 {
    // Compute the result in parallel and then sum the contributions.
    bodies_src
        .par_iter()
        .enumerate()
        .filter_map(|(i, body_source)| {
            let posit_src = body_source.posit;

            let diff = posit_src - posit_target;
            let dist = diff.magnitude();

            let dir = diff / dist; // Unit vec

            let mut sigmas = [0.; 8];
            let mut epss = [0.; 8];

            //todo: QC all this lane stuff.
            let lanes_src =  if i== chunks_src - 1 {
                valid_lanes_src_last
            } else {
                8
            };

            let valid_lanes = lanes_src.min(lanes_tgt);
            for lane in 0..valid_lanes {
                let (sigma, eps) = lj_lut.get(&(body_source.element[lane], el_tgt[lane]))
                    .unwrap();
                sigmas[lane] = *sigma;
                epss[lane] = *eps;
            }

            let sigma = f32x8::from_array(sigmas);
            let eps = f32x8::from_array(epss);

            Some(lj_force_x8(dir, dist, sigma, eps))
        })
        .reduce(Vec3x8::new_zero, |acc, elem| acc + elem) // Sum the contributions.
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

fn bodies_from_atomsx8(atoms: &[Atom]) -> Vec<BodyVdwx8> {
    let mut result = Vec::new();

    // DRY with `lin_alg::pack_vec3`:
    let remainder = atoms.len() % 8;
    let padding_needed = if remainder == 0 { 0 } else { 8 - remainder };

    let mut padded = Vec::with_capacity(atoms.len() + padding_needed);
    padded.extend_from_slice(atoms);
    padded.extend((0..padding_needed).map(|_| Atom::default()));

    // Now `padded.len()` is a multiple of 8, so chunks_exact(8) will consume it fully.
    let atoms_chunked: Vec<[Atom; 8]> = padded
        .chunks_exact(8)
        .map(|chunk| TryInto::<&[Atom; 8]>::try_into(chunk).unwrap().clone())
        .collect();

    for chunk in &atoms_chunked {
        let mut body = BodyVdwx8 {
            posit: Vec3x8::new_zero(),
            vel: Vec3x8::new_zero(),
            accel: Vec3x8::new_zero(),
            mass: f32x8::splat(0.),
            element: [Element::Carbon; 8],
            orientation: Quaternionx8::new_identity(),
        };

        let mut posits = Vec::with_capacity(8);
        let mut masses = Vec::with_capacity(8);
        for atom in chunk {
            posits.push(atom.posit.into());
            masses.push(atom.element.atomic_number() as f32);
        }
        body.posit = Vec3x8::from_slice(&posits);
        body.mass = f32x8::from_slice(&masses);

        result.push(body);
    }

    result
}

/// Keeps orientation fixed and body rigid, for now.
pub fn build_vdw_dynamics(
    receptor_atoms: &[Atom],
    lig: &Ligand,
    lj_lut: &HashMap<(Element, Element), (f32, f32)>,
    // ) -> (Vec<Snapshot>, Pose) {
) -> Vec<Snapshot> {
    println!("Starting vuilding VDW dyanmics...");
    let start = Instant::now();

    // todo: You should possibly add your pre-computed LJ pairs, instead of looking up each time.
    // todo: See this code from docking.

    // Keeps initial VDW moves from causing very high jumps. This should still accomodate an initial jump
    // away from a nearby atom.
    let vel_max = 300.;

    let n_steps = 1_000;
    // An adaptive timestep.
    let dt_max = 0.0001;

    let dt_dynamic_scaler = 100.;
    let dt_dynamic_scaler_x8 = f32x8::splat(dt_max);

    // todo: We're having trouble fitting dy namic DT into our functional code.
    // todo: One alternative is to pre-calculate it, but this has an additional distance step.

    let snapshot_ratio = 10; // todo

    let mut snapshots = Vec::with_capacity(n_steps + 1); // +1 for the initial snapshot.

    let mut time_elapsed = 0.;

    // Static
    let bodies_rec = bodies_from_atoms(receptor_atoms);
    // These move.
    let mut bodies_lig = bodies_from_atoms(&lig.molecule.atoms);
    let mut body_ligand_rigid = BodyVdw::from_atom(&lig.molecule.atoms[lig.anchor_atom]);

    // Static
    let bodies_rec_x8 = bodies_from_atomsx8(receptor_atoms);
    // These move.
    let mut bodies_lig_x8 = bodies_from_atomsx8(&lig.molecule.atoms);

    // How many 8-wide chunks we have
    let chunk_count_rec = bodies_rec_x8.len();
    let chunk_count_lig = bodies_lig_x8.len();

    // How many valid lanes there are in the last chunk
    let rem_rec = bodies_rec.len() % 8;
    let rem_lig = bodies_lig.len() % 8;

    let valid_lanes_rec_last = if rem_rec == 0 { 8 } else { rem_rec };
    let valid_lanes_lig_last = if rem_lig == 0 { 8 } else { rem_lig };

    let mut ligand_mass = 0.;
    for atom in &lig.molecule.atoms {
        ligand_mass += atom.element.atomic_number() as f32; // Arbitrary mass scale for now.
    }

    // Initial snapshot
    snapshots.push(Snapshot {
        time: time_elapsed,
        lig_atom_posits: bodies_lig.iter().map(|b| b.posit.into()).collect(),
        energy: BindingEnergy::default(), // todo
    });

    // let acc_fn = |id_: usize, posit_target, el_target, mass_tgt| {
    //     acc_lj(posit_target, el_target, mass_tgt, &bodies_rec, &lj_lut)
    // };

    // We use these to avoid performing computations on empty (0ed?) values on the final SIMD value.
    // This causes incorrect results.
    // Number of real bodies:

    for t in 0..n_steps {
        let (distances, dt) = {
            // todo: This is dramatically increasing computation time. Cache distances, and use in the LJ calc!
            let mut distances = Vec::new();
            let mut dt = dt_max;

            // Pre-compute distances, and calculate our dynamic DT.
            // for body_tgt in &bodies_rec {
            for (i_rec, body_rec) in bodies_rec_x8.iter().enumerate() {
                // Figure out how many lanes are valid in this rec chunk:
                // (If it's not the last chunk or if remainder is 0, that's 8. Otherwise it's `rec_rem`.)
                let lanes_rec = if i_rec == chunk_count_rec - 1 {
                    valid_lanes_rec_last
                } else {
                    8
                };

                let mut distances_tgt = Vec::new();
                // for body_src in &bodies_lig {
                for (i_lig, body_lig) in bodies_lig_x8.iter().enumerate() {
                    let lanes_lig = if i_lig == chunk_count_lig- 1 {
                        valid_lanes_lig_last
                    } else {
                        8
                    };

                    let dist = (body_lig.posit - body_rec.posit).magnitude();
                    distances_tgt.push(dist);

                    // Now compute dt_this (8-wide):
                    let rel_velocity = (body_lig.vel - body_rec.vel).magnitude();
                    let dt_this = dt_dynamic_scaler_x8 * dist / rel_velocity;

                    // Convert to an array so we can iterate lane by lane:
                    let dt_arr = dt_this.to_array();

                    // let dt_this = dt_dynamic_scaler * dist / rel_velocity;
                    // if dt_this < dt {
                    //     dt = dt_this;
                    // }

                    // Only the first `min(valid_rec_lanes, valid_lig_lanes)` lanes are real
                    let valid_lanes = lanes_rec.min(lanes_lig);
                    for lane in 0..valid_lanes {
                        let dt_lane = dt_arr[lane];
                        if dt_lane < dt {
                            dt = dt_lane;
                        }
                    }

                    //
                    // let rel_velocity = (body_lig.vel - body_rec.vel).magnitude();
                    //
                    // let dt_this = dt_dynamic_scaler_x8 * dist / rel_velocity;
                    // for dt_ in dt_this.to_array() {
                    //     if dt_ < dt {
                    //         dt = dt_;
                    //     }
                    // }
                }
                distances.push(distances_tgt);
            }

            (distances, dt)
        };

        // These net values are for our rigid body model.
        // let mut acc_net = Vec3::new_zero();
        // bodies_lig
        //     .par_iter_mut()
        //     .for_each(|body_lig| {
        // // todo: You may need force instead of accel.
        //         integrate_rk4(body_lig, 0, &acc_fn, dt);
        //     });

        // todo: Now, use these distances downstream.

        // let dt = calc_dt_dynamic(&bodies_rec, &bodies_lig, dt_dynamic_scaler, dt_max);

        let mut vel_net = Vec3::new_zero();

        // let f_net = bodies_lig
        let f_net = bodies_lig_x8
            .par_iter_mut()
            .enumerate()
            .filter_map(|(i_lig, body_lig)| {
                let lanes_lig = if i_lig == chunk_count_lig- 1 {
                    valid_lanes_lig_last
                } else {
                    8
                };

                // force_lj(body_lig.posit, body_lig.element, &bodies_rec, &lj_lut)
                Some(force_lj_x8(body_lig.posit, body_lig.element, &bodies_rec_x8, &lj_lut, chunk_count_rec, lanes_lig, valid_lanes_rec_last))
            })
            // .reduce(Vec3::new_zero, |a, b| a + b);
            .reduce(Vec3x8::new_zero, |a, b| a + b);

        // Now, unpack the SIMD-calculated acceleration into a single value.

        // Safely unpack: only use the valid lanes of the final sum.
        let mut f_net_unpacked = Vec3::new_zero();

        // Convert Vec3x8 into an array of Vec3 so we can iterate lane by lane:
        let lanes = f_net.to_array(); // [Vec3; 8]

        // For the last chunk, only the first valid_lanes_in_last_chunk are truly real
        for i in 0..valid_lanes_lig_last {
            f_net_unpacked += lanes[i];
        }

        // let acc_net = f_net / ligand_mass;
        let acc_net = f_net_unpacked / ligand_mass;

        vel_net += acc_net * dt;

        let vel_mag = vel_net.magnitude();

        if t % 500 == 0 {
            // println!("DT: {:?}", dt);
            // println!("VEL NET: {:?}", vel_mag);
        }

        if vel_mag > vel_max {
            vel_net = (vel_net / vel_mag) * vel_max;
        }

        // for (i, vel) in vel_mag.to_array().iter().enumerate() { {
        //     if *vel > vel_max {
        //         let mut vels = vel_net.to_array();
        //         vels[i] = (vels[i] / *vel) * vel_max;
        //         vel_net = Vec3x8::from_array(vels);
        //     }
        // }}

        // todo: You should use RK4 on the rigid body, somehow. Currently using Euler integration.

        // for body in &mut bodies_lig {
        for body in &mut bodies_lig_x8 {
            // body.vel += vel_net;
            body.vel += Vec3x8::splat(vel_net);
            // body.posit += vel_net * dt;
            body.posit += Vec3x8::splat(vel_net * dt);
        }

        // Keeps the same signature as individual, but doesn't use parts of it.
        // let acc_fn_net = |id_, posit_target, el_target, mass_tgt| acc_net;

        // integrate_rk4(&mut body_ligand_rigid, 0, &acc_fn_net, dt);

        // integrate_rk4x8(&mut body_ligand_rigid_x8, 0, &acc_fn_net, dt);
        //
        // todo: Return the pose of the final result.
        // let pose = Pose {
        //     anchor_posit: 0,
        //     orientation: 0,
        //     conformation_type: ConformationType::Flexible {torsions: Vec::new() }
        // };

        // let pos = lig.position_atoms(None);

        time_elapsed += dt;

        // Save the current state to a snapshot, for later playback.

        // Unpack bodies for the purposes of saving a snapshot.
        let bodies_lig: Vec<_> = bodies_lig_x8.iter().map(|b| b.posit).collect();
        let posits_unpacked = unpack_vec3(&bodies_lig);

        if t % snapshot_ratio == 0 {
            snapshots.push(Snapshot {
                time: time_elapsed,
                // lig_atom_posits: bodies_lig.iter().map(|b| b.posit.into()).collect(),
                lig_atom_posits: posits_unpacked,
                energy: BindingEnergy::default(), // todo
            });
        }
    }

    let elapsed = start.elapsed().as_millis();
    println!("Complete. Time: {elapsed}ms");
    // (snapshots, pose)
    snapshots
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
