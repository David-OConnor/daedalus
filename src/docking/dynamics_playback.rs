//! Experimental molecular dynamics, with a playback system. Starting with fixed-ligand position only,
//! referencing the anchor.

use std::{collections::HashMap, time::Instant};

use graphics::Entity;
use lin_alg::f32::{Mat3, Quaternion, Quaternionx8, Vec3, Vec3x8, f32x8, unpack_f32, unpack_vec3};
use rayon::prelude::*;

use crate::{
    docking::{BindingEnergy, ConformationType, Pose, SOFTENING_FACTOR_SQ_ELECTROSTATIC},
    element::Element,
    forces::{coulomb_force, lj_force, lj_force_x8},
    molecule::{Atom, Ligand, Molecule},
};

// This seems to be how we control rotation vice movement. A higher value means
// more movement, less rotation for a given dt.
const ROTATION_INERTIA: f32 = 1_000.;

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
/// We use this for integration.
struct BodyRigid {
    pub posit: Vec3,
    pub vel: Vec3,
    // pub accel: Vec3,
    pub orientation: Quaternion,
    pub ω: Vec3,
    pub mass: f32,
    /// Inertia tensor in the body frame (if it’s diagonal, can store as Vec3)
    pub inertia_body: Mat3,
    pub inertia_body_inv: Mat3,
}

impl BodyRigid {
    fn from_ligand(lig: &Ligand) -> Self {
        let mut mass = 0.;
        for atom in &lig.molecule.atoms {
            mass += atom.element.atomic_number() as f32; // Arbitrary mass scale for now.
        }

        let inertia_body = Mat3::new_identity() * ROTATION_INERTIA;
        let inertia_body_inv = inertia_body.inverse().unwrap();

        // todo: This assumes no initial bond flexes...?
        Self {
            posit: lig.pose.anchor_posit.into(),
            vel: Default::default(),
            orientation: lig.pose.orientation.into(),
            ω: Default::default(),
            mass,
            // todo: Set based on atom masses?
            inertia_body,
            inertia_body_inv,
        }
    }

    fn as_pose(&self) -> Pose {
        Pose {
            anchor_posit: self.posit.into(),
            orientation: self.orientation.into(),
            conformation_type: Default::default(),
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
    pub pose: Pose, // todo: Experimenting
    pub lig_atom_posits: Vec<Vec3>,
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

/// Integrates position and orientation of a rigid body using RK4.
pub fn integrate_rk4_rigid<F>(body: &mut BodyRigid, force_torque: &F, dt: f32)
where
    // force_torque(body) should return (net_force, net_torque) in *world* coordinates
    F: Fn(&BodyRigid) -> (Vec3, Vec3),
{
    // -- k1 --------------------------------------------------------------------
    let (f1, τ1) = force_torque(body);
    let acc_lin1 = f1 / body.mass;
    let acc_ω1 =
        body.inertia_body_inv.clone() * (τ1 - body.ω.cross(body.inertia_body.clone() * body.ω));

    let k1_v = acc_lin1 * dt; // change in velocity
    let k1_pos = body.vel * dt; // change in position
    let k1_w = acc_ω1 * dt; // change in angular velocity
    let k1_q = orientation_derivative(body.orientation, body.ω) * dt; // change in orientation

    // -- Prepare body_k2 (the state at t + dt/2) -------------------------------
    let mut body_k2 = body.clone();
    body_k2.vel = body.vel + k1_v * 0.5;
    body_k2.posit = body.posit + k1_pos * 0.5;
    body_k2.ω = body.ω + k1_w * 0.5;
    body_k2.orientation = (body.orientation + k1_q * 0.5).to_normalized();

    // -- k2 --------------------------------------------------------------------
    let (f2, τ2) = force_torque(&body_k2);
    let acc_lin2 = f2 / body_k2.mass;
    let acc_ω2 =
        body_k2.inertia_body_inv * (τ2 - body_k2.ω.cross(body_k2.inertia_body * body_k2.ω));

    let k2_v = acc_lin2 * dt;
    let k2_pos = body_k2.vel * dt;
    let k2_w = acc_ω2 * dt;
    let k2_q = orientation_derivative(body_k2.orientation, body_k2.ω) * dt;

    // -- Prepare body_k3 (the state at t + dt/2) -------------------------------
    let mut body_k3 = body.clone();
    body_k3.vel = body.vel + k2_v * 0.5;
    body_k3.posit = body.posit + k2_pos * 0.5;
    body_k3.ω = body.ω + k2_w * 0.5;
    body_k3.orientation = (body.orientation + k2_q * 0.5).to_normalized();

    // -- k3 --------------------------------------------------------------------
    let (f3, τ3) = force_torque(&body_k3);
    let acc_lin3 = f3 / body_k3.mass;
    let acc_ω3 =
        body_k3.inertia_body_inv * (τ3 - body_k3.ω.cross(body_k3.inertia_body * body_k3.ω));

    let k3_v = acc_lin3 * dt;
    let k3_pos = body_k3.vel * dt;
    let k3_w = acc_ω3 * dt;
    let k3_q = orientation_derivative(body_k3.orientation, body_k3.ω) * dt;

    // -- Prepare body_k4 (the state at t + dt) ---------------------------------
    let mut body_k4 = body.clone();
    body_k4.vel = body.vel + k3_v;
    body_k4.posit = body.posit + k3_pos;
    body_k4.ω = body.ω + k3_w;
    body_k4.orientation = (body.orientation + k3_q).to_normalized();

    // -- k4 --------------------------------------------------------------------
    let (f4, τ4) = force_torque(&body_k4);
    let acc_lin4 = f4 / body_k4.mass;
    let acc_ω4 =
        body_k4.inertia_body_inv * (τ4 - body_k4.ω.cross(body_k4.inertia_body * body_k4.ω));

    let k4_v = acc_lin4 * dt;
    let k4_pos = body_k4.vel * dt;
    let k4_w = acc_ω4 * dt;
    let k4_q = orientation_derivative(body_k4.orientation, body_k4.ω) * dt;

    // -- Combine k1..k4 to get final update -------------------------------------
    // Final: v(t+dt) = v(t) + 1/6 * (k1_v + 2k2_v + 2k3_v + k4_v)
    body.vel += (k1_v + k2_v * 2.0 + k3_v * 2.0 + k4_v) / 6.0;
    body.posit += (k1_pos + k2_pos * 2.0 + k3_pos * 2.0 + k4_pos) / 6.0;

    body.ω += (k1_w + k2_w * 2.0 + k3_w * 2.0 + k4_w) / 6.0;

    // For orientation, do the same weighted average of the "k" changes,
    // then normalize the result at the end.
    let orientation_update = (k1_q + k2_q * 2. + k3_q * 2.0 + k4_q) / 6.0;
    body.orientation = (body.orientation + orientation_update).to_normalized();
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

            let lanes_src = if i == chunks_src - 1 {
                valid_lanes_src_last
            } else {
                8
            };

            let valid_lanes = lanes_src.min(lanes_tgt);
            for lane in 0..valid_lanes {
                let (sigma, eps) = lj_lut
                    .get(&(body_source.element[lane], el_tgt[lane]))
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

// todo: QC this
fn orientation_derivative(orientation: Quaternion, ang_vel_world: Vec3) -> Quaternion {
    // Represent w in quaternion form: (0, wx, wy, wz)
    let w_quat = Quaternion::new(0.0, ang_vel_world.x, ang_vel_world.y, ang_vel_world.z);
    // dq/dt = 0.5 * w_quat * q
    w_quat * orientation * 0.5
}

/// Keeps orientation fixed and body rigid, for now.
pub fn build_vdw_dynamics(
    receptor_atoms: &[Atom],
    lig: &Ligand,
    lj_lut: &HashMap<(Element, Element), (f32, f32)>,
) -> Vec<Snapshot> {
    println!("Starting vuilding VDW dyanmics...");
    let start = Instant::now();

    // todo: You should possibly add your pre-computed LJ pairs, instead of looking up each time.
    // todo: See this code from docking.

    let n_steps = 500;
    // An adaptive timestep.
    // let dt_max = 0.00001;
    let dt_max = 0.00001;

    let dt_dynamic_scaler = 100.;
    let dt_dynamic_scaler_x8 = f32x8::splat(dt_dynamic_scaler);

    // todo: We're having trouble fitting dy namic DT into our functional code.
    // todo: One alternative is to pre-calculate it, but this has an additional distance step.

    let snapshot_ratio = 10;

    let mut snapshots = Vec::with_capacity(n_steps + 1); // +1 for the initial snapshot.

    let mut time_elapsed = 0.;

    // Static
    let bodies_rec_x8 = bodies_from_atomsx8(receptor_atoms);
    // These move.
    // let mut bodies_lig_x8 = bodies_from_atomsx8(&lig.molecule.atoms);

    // How many 8-wide chunks we have
    let chunk_count_rec = bodies_rec_x8.len();
    let chunk_count_lig = {
        // Overkill, but safe.
        let bodies = bodies_from_atomsx8(&lig.molecule.atoms);
        bodies.len()
    };

    // How many valid lanes there are in the last chunk
    let rem_rec = receptor_atoms.len() % 8;
    let rem_lig = lig.molecule.atoms.len() % 8;

    let valid_lanes_rec_last = if rem_rec == 0 { 8 } else { rem_rec };
    let valid_lanes_lig_last = if rem_lig == 0 { 8 } else { rem_lig };

    // End setup.

    // Assume atoms are positioned already?

    // let mut bodies_lig = bodies_from_atoms(&lig.molecule.atoms);
    let mut body_ligand_rigid = BodyRigid::from_ligand(lig);

    // Initial snapshot
    snapshots.push(Snapshot {
        time: time_elapsed,
        lig_atom_posits: lig.atom_posits.iter().map(|p| (*p).into()).collect(),
        pose: lig.pose.clone(),
        energy: BindingEnergy::default(), // todo
    });

    let mut dt = dt_max; // todo: Make adaptive again; you must pull that logic outside
    // todo of teh torque/force fn.

    let force_torque_fn = |body: &BodyRigid| {
        // Set up atom positions from the rigid body passed as a parameter.
        let atom_posits = lig.position_atoms(Some(&body.as_pose()));
        let mut atoms = lig.molecule.atoms.clone(); // todo: Not a fan of this clone, or these dummy atoms in general.
        for (i, posit) in atom_posits.iter().enumerate() {
            atoms[i].posit = *posit;
        }
        let bodies_lig_x8 = bodies_from_atomsx8(&atoms);

        let anchor_posit = Vec3x8::splat(body.posit);

        let (f_net, torque_net) = bodies_lig_x8
            // .par_iter_mut()
            .par_iter()
            .enumerate()
            .map(|(i_lig, body_lig)| {
                let lanes_lig = if i_lig == chunk_count_lig - 1 {
                    valid_lanes_lig_last
                } else {
                    8
                };

                let f = force_lj_x8(
                    body_lig.posit,
                    body_lig.element,
                    &bodies_rec_x8,
                    &lj_lut,
                    chunk_count_rec,
                    lanes_lig,
                    valid_lanes_rec_last,
                );

                // Torque = (r - R_cm) x F,
                // where R_cm is the center-of-mass position, and r is this atom's position.
                // But if you store each body_lig.posit already relative to the COM, then you can just use r x F
                // let diff = body_lig.posit - body_ligand_rigid.posit;

                let diff = body_lig.posit - anchor_posit;
                let torque = diff.cross(f);

                (f, torque)
            })
            .reduce(
                || (Vec3x8::new_zero(), Vec3x8::new_zero()),
                |a, b| (a.0 + b.0, a.1 + b.1),
            );

        // Now, unpack the SIMD-calculated acceleration into a single value.
        let f_net_unpacked: Vec3 = f_net.to_array().into_iter().sum();
        // let acc_net = f_net_unpacked / body_ligand_rigid.mass;

        let torque_net_unpacked: Vec3 = torque_net.to_array().into_iter().sum();
        (f_net_unpacked, torque_net_unpacked)
    };

    // We use these to avoid performing computations on empty (0ed?) values on the final SIMD value.
    // This causes incorrect results.
    // Number of real bodies:

    for t in 0..n_steps {
        // todo: This is dramatically increasing computation time. Cache distances, and use in the LJ calc!
        // let (distances, dt_) = {
        // let mut distances = Vec::new();
        // let mut dt_ = dt_max;

        // Pre-compute distances, and calculate our dynamic DT.
        //     for (i_rec, body_rec) in bodies_rec_x8.iter().enumerate() {
        //         // Figure out how many lanes are valid in this rec chunk:
        //         // (If it's not the last chunk or if remainder is 0, that's 8. Otherwise it's `rec_rem`.)
        //         let lanes_rec = if i_rec == chunk_count_rec - 1 {
        //             valid_lanes_rec_last
        //         } else {
        //             8
        //         };
        //
        //         let mut distances_tgt = Vec::new();
        //         for (i_lig, body_lig) in bodies_lig_x8.iter().enumerate() {
        //             let lanes_lig = if i_lig == chunk_count_lig - 1 {
        //                 valid_lanes_lig_last
        //             } else {
        //                 8
        //             };
        //
        //             let dist = (body_lig.posit - body_rec.posit).magnitude();
        //             distances_tgt.push(dist);
        //
        //             // Now compute dt_this (8-wide):
        //             let rel_velocity = (body_lig.vel - body_rec.vel).magnitude();
        //             let dt_this = dt_dynamic_scaler_x8 * dist / rel_velocity;
        //
        //             // Convert to an array so we can iterate lane by lane:
        //             let dt_arr = dt_this.to_array();
        //
        //             // Only the first `min(valid_rec_lanes, valid_lig_lanes)` lanes are real
        //             let valid_lanes = lanes_rec.min(lanes_lig);
        //             for lane in 0..valid_lanes {
        //                 let dt_lane = dt_arr[lane];
        //                 if dt_lane < dt_ {
        //                     dt_ = dt_lane;
        //                 }
        //             }
        //         }
        //         distances.push(distances_tgt);
        //     }
        //
        //     (distances, dt_)
        // };

        // dt = dt_;

        integrate_rk4_rigid(&mut body_ligand_rigid, &force_torque_fn, dt);

        time_elapsed += dt;

        // Save the current state to a snapshot, for later playback.

        // Unpack bodies for the purposes of saving a snapshot.
        // let bodies_lig: Vec<_> = bodies_lig_x8.iter().map(|b| b.posit).collect();
        // let posits_unpacked = unpack_vec3(&bodies_lig, lig.molecule.atoms.len());

        let pose = Pose {
            anchor_posit: body_ligand_rigid.posit.into(),
            orientation: body_ligand_rigid.orientation.into(),
            conformation_type: lig.pose.conformation_type.clone(), // todo Bond flexes, eventually.
        };

        let posits = lig
            .position_atoms(Some(&pose))
            .iter()
            .map(|p| (*p).into())
            .collect();

        if t % snapshot_ratio == 0 {
            snapshots.push(Snapshot {
                time: time_elapsed,
                lig_atom_posits: posits,
                pose,
                energy: BindingEnergy::default(), // todo
            });
        }
    }

    let elapsed = start.elapsed().as_millis();
    println!("Complete. Time: {elapsed}ms");
    snapshots
}

/// Body masses are separate from the snapshot, since it's invariant.
pub fn change_snapshot(
    entities: &mut Vec<Entity>,
    lig: &mut Ligand,
    lig_entity_ids: &[usize],
    snapshot: &Snapshot,
) {
    // todo: Initial hack: Get working as individual particles. Then, try to incorporate
    // todo fixed rotation of the molecule, fixed movement, bond flexes etc.

    // todo: For now, redraw all entities; eventually, don't!
    // entities.retain(|e| !lig_entity_ids.contains(&e.id));
    // *entities = Vec::with_capacity(snapshot.lig_atom_posits.len());

    lig.pose = snapshot.pose.clone();

    // Position atoms from pose  here? You could, but the snapshot has them pre-positioned.
    // This may make changing snapshots faster. But uses more memory from storing each

    lig.atom_posits = snapshot
        .lig_atom_posits
        .iter()
        .map(|p| (*p).into())
        .collect();

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
