#![allow(non_snake_case)]

//! Experimental molecular dynamics, with a playback system. Starting with fixed-ligand position only,
//! referencing the anchor.

use std::time::Instant;

use crate::ComputationDevice;

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use cudarc::driver::{CudaModule, CudaStream, LaunchConfig, PushKernelArg};
        use lin_alg::f32::{vec3s_from_dev, vec3s_to_dev};
    }
}

use graphics::Entity;
use lin_alg::{
    f32::{Mat3 as Mat3F32, Quaternion as QuaternionF32, Vec3 as Vec3F32},
    f64::{Mat3, Quaternion, Vec3},
};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use lin_alg::{
    // f32::{Vec3x8, f32x8, pack_slice, pack_vec3},
    f64::{Vec3x4, f64x4, pack_slice, pack_vec3},
};
use na_seq::Element;
use rayon::prelude::*;

#[cfg(feature = "cuda")]
use crate::forces::force_lj_gpu;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::forces::force_lj_x8;
use crate::{
    docking::{
        BindingEnergy, ConformationType, Pose, calc_binding_energy,
        prep::{DockingSetup, Torsion},
    },
    dynamics::{AtomDynamics, AtomDynamicsx4, MdState, SimBox},
    forces::{force_lj, force_lj_f32},
    integrate::{integrate_verlet, integrate_verlet_f64},
    molecule::{Atom, Ligand},
};
// This seems to be how we control rotation vice movement. A higher value means
// more movement, less rotation for a given dt.

// todo: A/R remove torque calc??
// const ROTATION_INERTIA: f32 = 500_000.;
const ROTATION_INERTIA: f64 = 500_000.;

// For calculating numerical derivatives.
// const DX: f32 = 0.1;
const DX: f64 = 0.1;

#[derive(Clone, Debug)]
/// We use this for integration.
pub struct BodyRigid {
    pub posit: Vec3,
    /// We track this for verlet integration.
    pub posit_prev: Vec3,
    pub vel: Vec3,
    // pub accel: Vec3,
    pub orientation: Quaternion,
    pub ω: Vec3,
    pub torsions: Vec<Torsion>,
    // pub mass: f32,
    pub mass: f64,
    /// Inertia tensor in the body frame (if it’s diagonal, can store as Vec3)
    pub inertia_body: Mat3,
    pub inertia_body_inv: Mat3,
}

impl BodyRigid {
    fn from_ligand(lig: &Ligand) -> Self {
        let mut mass = 0.;
        for atom in &lig.molecule.atoms {
            mass += atom.element.atomic_weight() as f64; // Arbitrary mass scale for now.
        }

        let inertia_body = Mat3::new_identity() * ROTATION_INERTIA;
        let inertia_body_inv = inertia_body.inverse().unwrap();

        let torsions = match &lig.pose.conformation_type {
            ConformationType::Flexible { torsions } => torsions.clone(),
            _ => Vec::new(),
        };

        Self {
            posit: lig.pose.anchor_posit.into(),
            posit_prev: lig.pose.anchor_posit.into(),
            vel: Default::default(),
            orientation: lig.pose.orientation.into(),
            torsions,
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
            conformation_type: ConformationType::Flexible {
                torsions: self.torsions.clone(),
            },
        }
    }
}

#[derive(Debug, Default)]
pub struct Snapshot {
    pub time: f64,
    pub pose: Pose, // todo: Experimenting
    pub lig_atom_posits: Vec<Vec3>,
    pub energy: Option<BindingEnergy>,
}

/// Defaults to `Config::dt_integration`, but becomes more precise when
/// bodies are close. This is a global DT, vice local only for those bodies.
fn calc_dt_dynamic(
    bodies_src: &[AtomDynamics],
    bodies_tgt: &[AtomDynamics],
    //     dt_scaler: f32,
    //     dt_max: f32,
    // ) -> f32 {
    dt_scaler: f64,
    dt_max: f64,
) -> f64 {
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

fn bodies_from_atoms(atoms: &[Atom]) -> Vec<AtomDynamics> {
    atoms.iter().map(|a| a.into()).collect()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
/// Also returns valid lanes in the last item.
fn bodies_from_atoms_x8(atoms: &[Atom]) -> (Vec<AtomDynamicsx4>, usize) {
    let mut posits: Vec<Vec3> = Vec::with_capacity(atoms.len());
    let mut els = Vec::with_capacity(atoms.len());

    for atom in atoms {
        posits.push(atom.posit.into());
        els.push(atom.element);
    }

    let (posits_x8, valid_lanes) = pack_vec3(&posits);

    // let (els_x8, _) = pack_slice::<_, 8>(&els);
    let (els_x4, _) = pack_slice::<_, 4>(&els);
    let mut result = Vec::with_capacity(posits_x8.len());

    for (i, posit) in posits_x8.iter().enumerate() {
        // let masses: Vec<_> = els_x4[i].iter().map(|el| el.atomic_weight()).collect();
        // let mass = f32x8::from_slice(&masses);
        let masses: Vec<_> = els_x4[i]
            .iter()
            .map(|el| el.atomic_weight() as f64)
            .collect();
        let mass = f64x4::from_slice(&masses);

        result.push(AtomDynamicsx4 {
            posit: *posit,
            vel: Vec3x4::new_zero(),
            accel: Vec3x4::new_zero(),
            mass,
            element: els_x4[i],
        })
    }

    (result, valid_lanes)
}

// todo: QC this
pub fn orientation_derivative(orientation: Quaternion, ang_vel_world: Vec3) -> Quaternion {
    // Represent w in quaternion form: (0, wx, wy, wz)
    let w_quat = Quaternion::new(0.0, ang_vel_world.x, ang_vel_world.y, ang_vel_world.z);
    // dq/dt = 0.5 * w_quat * q
    w_quat * orientation * 0.5
}

// todo: Params after net_V fn are temp.
/// Calculate the gradient vector, using a numerical first derivative, from potentials.
fn calc_gradient_posit<F>(body: &BodyRigid, net_V_fn: &F) -> Vec3
where
    // F: Fn(&BodyRigid) -> f32,
    F: Fn(&BodyRigid) -> f64,
{
    let body_x_prev = BodyRigid {
        posit: body.posit + Vec3::new(-DX, 0., 0.),
        ..body.clone()
    };
    let body_x_next = BodyRigid {
        posit: body.posit + Vec3::new(DX, 0., 0.),
        ..body.clone()
    };
    let body_y_prev = BodyRigid {
        posit: body.posit + Vec3::new(0., -DX, 0.),
        ..body.clone()
    };
    let body_y_next = BodyRigid {
        posit: body.posit + Vec3::new(0., DX, 0.),
        ..body.clone()
    };
    let body_z_prev = BodyRigid {
        posit: body.posit + Vec3::new(0., 0., -DX),
        ..body.clone()
    };
    let body_z_next = BodyRigid {
        posit: body.posit + Vec3::new(0., 0., DX),
        ..body.clone()
    };

    let V_x_prev = net_V_fn(&body_x_prev);
    let V_x_next = net_V_fn(&body_x_next);
    let V_y_prev = net_V_fn(&body_y_prev);
    let V_y_next = net_V_fn(&body_y_next);
    let V_z_prev = net_V_fn(&body_z_prev);
    let V_z_next = net_V_fn(&body_z_next);

    let dx2 = 2. * DX;
    Vec3::new(
        (V_x_next - V_x_prev) / dx2,
        (V_y_next - V_y_prev) / dx2,
        (V_z_next - V_z_prev) / dx2,
    )
}

// Prevents DRY between runtime and compile-time SIMD absent.
fn scalar_f_t(
    diffs: &[Vec3],
    setup: &DockingSetup,
    lig_posits_by_diff: &[Vec3],
    anchor_posit: Vec3,
) -> (Vec3, Vec3) {
    diffs
        .par_iter()
        .enumerate()
        .map(|(i, &diff)| {
            let r = diff.magnitude();
            let dir = diff / r;

            let sigma = setup.lj_sigma[i] as f64;
            let eps = setup.lj_eps[i] as f64;

            // let f = force_lj(dir, r, sigma, eps);
            let f = force_lj(dir, r, sigma, eps);

            // Torque = (r - R_cm) x F,
            // where R_cm is the center-of-mass position, and r is this atom's position.
            // But if you store each body_lig.posit already relative to the COM, then you can just use r x F

            let diff = lig_posits_by_diff[i] - anchor_posit;
            let torque = diff.cross(f);

            // todo: NaNs in some cases.
            // println!("F: {:?}", f);
            (f, torque)
        })
        .reduce(
            || (Vec3::new_zero(), Vec3::new_zero()),
            |a, b| (a.0 + b.0, a.1 + b.1),
        )
}

/// Keeps orientation fixed and body rigid, for now.
///
/// Observation: We can use analytic VDW force to position individual atoms, but once we treat
/// the molecule together, we seem to get bogus results using this approach. Instead, we use a numerical
/// derivative of the total VDW potential, and use gradient descent.
pub fn build_dock_dynamics(
    dev: &ComputationDevice,
    lig: &mut Ligand,
    setup: &DockingSetup,
    n_steps: usize,
) -> Vec<Snapshot> {
    println!("Building docking dyanmics...");
    let start = Instant::now();

    lig.pose.conformation_type = ConformationType::AbsolutePosits;

    // todo: Startign new approach
    {
        // todo: Use state dynamics state
        let mut md_state = MdState::new(
            &lig.molecule.atoms,
            &lig.molecule.bonds,
            &setup.rec_atoms_near_site,
            SimBox::default(),
            &setup.lj_lut,
        );

        let n_steps = 1_000;
        // In femtoseconds
        let dt = 1.;

        for _ in 0..n_steps {
            md_state.step(dt)
        }

        for (i, atom) in md_state.atoms.iter().enumerate() {
            if i < 30 {
                println!("Posit: {:?}", atom.posit);
            }

            lig.molecule.atoms[i].posit = atom.posit;
        }
    }

    // todo: You should possibly add your pre-computed LJ pairs, instead of looking up each time.
    // todo: See this code from docking.

    // An adaptive timestep.
    let dt_max = 0.0001;

    let dt_dynamic_scaler = 100.;

    let dt = dt_max;
    //             let rel_velocity = (body_lig.vel - body_rec.vel).magnitude();
    //             let dt_this = dt_dynamic_scaler_x8 * dist / rel_velocity;
    //
    //             // Convert to an array so we can iterate lane by lane:
    //             let dt_arr = dt_this.to_array();

    let snapshot_ratio = 10;
    let mut snapshots = Vec::with_capacity(n_steps + 1); // +1 for the initial snapshot.
    let mut time_elapsed = 0.;

    // todo: We're having trouble fitting dy namic DT into our functional code.
    // todo: One alternative is to pre-calculate it, but this has an additional distance step.
    // todo: Make adaptive again; you must pull that logic outside
    // todo of teh torque/force fn.

    // Static

    let mut body_ligand_rigid = BodyRigid::from_ligand(lig);

    // Initial snapshot
    snapshots.push(Snapshot {
        time: time_elapsed,
        lig_atom_posits: lig.atom_posits.iter().map(|p| (*p).into()).collect(),
        pose: lig.pose.clone(),
        energy: None, // todo: Initial energy?
    });

    let len_rec = setup.rec_atoms_near_site.len();
    let len_lig = lig.atom_posits.len();

    // todo: CUDA only. setup struct?
    let posits_rec: Vec<Vec3F32> = setup
        .rec_atoms_near_site
        .iter()
        .map(|r| r.posit.into())
        .collect();

    for t in 0..n_steps {
        lig.position_atoms(Some(&body_ligand_rigid.as_pose()));
        // todo: Split this into a separate fn A/R

        // Cache diffs.
        let mut diffs = Vec::with_capacity(len_rec * len_lig);
        // For torque calculations, compared to anchor.
        let mut lig_posits_by_diff = Vec::with_capacity(len_rec * len_lig);

        for i_rec in 0..len_rec {
            for i_lig in 0..len_lig {
                let posit_rec: Vec3 = setup.rec_atoms_near_site[i_rec].posit.into();
                let posit_lig: Vec3 = lig.atom_posits[i_lig].into();

                diffs.push(posit_rec - posit_lig);
                lig_posits_by_diff.push(posit_lig);
            }
        }

        let anchor_posit = body_ligand_rigid.posit;

        let start = Instant::now();

        let (force, torque) = match dev {
            #[cfg(feature = "cuda")]
            // todo: So... CUDA is taking about 1.5x the speed of CPU...
            ComputationDevice::Gpu((stream, module)) => {
                let lig_posits_f32: Vec<Vec3F32> =
                    lig.atom_posits.iter().map(|r| (*r).into()).collect();

                let f_lj_per_tgt = force_lj_gpu(
                    &stream,
                    module,
                    &lig_posits_f32,
                    &posits_rec,
                    &setup.lj_sigma,
                    &setup.lj_eps,
                );

                let mut f = Vec3F32::new_zero();
                for f_ in &f_lj_per_tgt {
                    f += *f_;
                }
                // let f = f_lj_per_tgt.iter().sum(); // todo: Impl sum.
                // todo: Torque: Need in kernel.
                let t = Vec3::new_zero(); // todo temp!
                // (f, t)
                (f.into(), t.into())
            }
            ComputationDevice::Cpu => {
                // todo: x8 isn't saving time here, and is causing invalid results. (not matching scalar)
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                // if !is_x86_feature_detected!("avx") {
                if true {
                    scalar_f_t(&diffs, setup, &lig_posits_by_diff, anchor_posit)
                } else {
                    // todo: Put this SIMD code back once you sort out f32 vs f64.
                    // let (diffs_x8, valid_lanes_last_diff) = pack_vec3(&diffs);
                    // let (lig_posits_by_diff_x8, _) = pack_vec3(&lig_posits_by_diff);
                    //
                    // let anchor_posit_x8 = Vec3x4::splat(anchor_posit);
                    //
                    // let (f, t) = diffs_x8
                    //     .par_iter()
                    //     .enumerate()
                    //     .map(|(i, &diff)| {
                    //         let r = diff.magnitude();
                    //         let dir = diff / r;
                    //         let sigma = setup.lj_sigma_x8[i];
                    //         let eps = setup.lj_eps_x8[i];
                    //
                    //         let f = force_lj_x8(dir, r, sigma, eps);
                    //
                    //         let diff = lig_posits_by_diff_x8[i] - anchor_posit_x8;
                    //         let torque = diff.cross(f);
                    //
                    //         (f, torque)
                    //     })
                    //     .reduce(
                    //         || (Vec3x4::new_zero(), Vec3x4::new_zero()),
                    //         |a, b| (a.0 + b.0, a.1 + b.1),
                    //     );
                    //
                    // // todo: Impl sum.
                    // let mut f_ = Vec3::new_zero();
                    // let mut t_ = Vec3::new_zero();
                    // let f_arr = f.to_array();
                    // let t_arr = t.to_array();
                    // for i in 0..8 {
                    //     f_ += f_arr[i];
                    //     t_ += t_arr[i];
                    // }
                    //
                    // (f_, t_)

                    (Vec3::new_zero(), Vec3::new_zero())
                }

                #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                scalar_f_t(&diffs, setup, &lig_posits_by_diff, anchor_posit)
            }
        };

        // todo: Switch to verlet, and use leap-frog between posit and orientation. (As you do)

        // let el = start.elapsed().as_micros();
        // println!("\nElapsed: {el}");

        // We use these to avoid performing computations on empty (0ed?) values on the final SIMD value.
        // This causes incorrect results.
        // Number of real bodies:

        // integrate_rk4_rigid(&mut body_ligand_rigid, &force_torque_fn, dt);

        // let (f, τ, atom_posits) = force_torque_fn(&body_ligand_rigid);

        // Skipping inertia/physics, to just nudge in the *right*? direction.
        // body_ligand_rigid.posit += f / body_ligand_rigid.mass * dt;

        // This approach of altering position and orientation seems to perform notably better than
        // combining the two.
        // todo: If this is how you do it, you can half computation time by alternating computation for torque
        // todo and force as well.

        // todo: RK4; once you are ready to make an acc fn again.
        if t % 2 == 0 {
            // Update position.
            let acc = force * dt;

            let posit_this = body_ligand_rigid.posit;
            // body_ligand_rigid.posit = integrate_verlet(
            body_ligand_rigid.posit = integrate_verlet_f64(
                body_ligand_rigid.posit,
                body_ligand_rigid.posit_prev,
                acc,
                dt,
            );
            body_ligand_rigid.posit_prev = posit_this;
        } else {
            // Update orientation.

            // todo: Update this using a verlet-adjacent approach?

            // let rotator = Quaternion::from_axis_angle(
            //     torque.to_normalized(),
            //     torque.magnitude() * dt * 0.00001,
            // );
            //
            // todo: QC this approach.
            body_ligand_rigid.ω += torque * dt;

            // let ω_mag = body_ligand_rigid.ω.magnitude();
            // if ω_mag > 0.0 {
            //     let axis = body_ligand_rigid.ω / ω_mag;
            //     let delta_q = Quaternion::from_axis_angle(axis, ω_mag * dt);
            //     // semi‐implicit Euler: apply rotation *after* updating ω
            //     body_ligand_rigid.orientation = (delta_q * body_ligand_rigid.orientation).to_normalized();
            // }
            let ω_q = Quaternion::new(
                0.0,
                body_ligand_rigid.ω.x,
                body_ligand_rigid.ω.y,
                body_ligand_rigid.ω.z,
            );
            let q_dot = ω_q * body_ligand_rigid.orientation * 0.5;
            body_ligand_rigid.orientation =
                (body_ligand_rigid.orientation + q_dot * dt).to_normalized();
        }

        // Experimenting with a drag term to prevent inertia from having too much influence.
        // body_ligand_rigid.vel *= 0.90;
        body_ligand_rigid.ω *= 0.90;

        time_elapsed += dt;

        if t % snapshot_ratio == 0 {
            let pose = body_ligand_rigid.as_pose();
            lig.position_atoms(Some(&pose));

            let posits: Vec<_> = lig.atom_posits.iter().map(|p| (*p).into()).collect();

            let energy = calc_binding_energy(setup, lig, &posits);

            snapshots.push(Snapshot {
                time: time_elapsed,
                // lig_atom_posits: posits,
                lig_atom_posits: lig.atom_posits.clone(),
                pose,
                energy,
            });
        }
    }

    // Final snapshot
    let pose = body_ligand_rigid.as_pose();
    lig.position_atoms(Some(&pose));

    let posits: Vec<_> = lig.atom_posits.iter().map(|p| (*p).into()).collect();

    let energy = calc_binding_energy(setup, lig, &posits);

    snapshots.push(Snapshot {
        time: time_elapsed,
        // lig_atom_posits: posits,
        lig_atom_posits: lig.atom_posits.clone(),
        pose,
        energy,
    });

    let elapsed = start.elapsed().as_millis();
    println!("Complete. Time: {elapsed}ms");
    snapshots
}

/// Body masses are separate from the snapshot, since it's invariant.
pub fn change_snapshot(
    entities: &mut [Entity],
    lig: &mut Ligand,
    lig_entity_ids: &[usize],
    energy_disp: &mut Option<BindingEnergy>,
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

    *energy_disp = snapshot.energy.clone();

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
