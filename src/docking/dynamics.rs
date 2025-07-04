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
    f32::Vec3 as Vec3F32,
    f64::{Mat3, Quaternion, Vec3},
};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use lin_alg::{
    // f32::{Vec3x8, f32x8, pack_slice, pack_vec3},
    f64::{Vec3x4, f64x4, pack_slice, pack_vec3},
};
use rayon::prelude::*;

#[cfg(feature = "cuda")]
use crate::forces::force_lj_gpu;
use crate::{
    docking::{
        BindingEnergy, ConformationType, Pose, calc_binding_energy,
        prep::{DockingSetup, Torsion},
    },
    dynamics::{
        AtomDynamics, AtomDynamicsx4, ForceFieldParamsIndexed, ForceFieldParamsKeyed, MdState,
        ParamError, SnapshotDynamics,
    },
    forces::force_lj,
    integrate::integrate_verlet_f64,
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
    ff_params: &ForceFieldParamsKeyed,
    ff_params_lig_specific: Option<&ForceFieldParamsKeyed>,
    n_steps: usize,
    // ) -> Vec<Snapshot> {
    // ) -> Vec<SnapshotDynamics> {
) -> Result<MdState, ParamError> {
    println!("Building docking dyanmics...");
    let start = Instant::now();

    lig.pose.conformation_type = ConformationType::AbsolutePosits;

    // todo: Startign new approach
    {
        // todo: Use state dynamics state
        let mut md_state = MdState::new(
            &lig.molecule.atoms,
            &lig.atom_posits,
            &lig.molecule.adjacency_list,
            &lig.molecule.bonds,
            &setup.rec_atoms_near_site,
            &setup.lj_lut,
            ff_params,
            ff_params_lig_specific,
        )?;

        let n_steps = 60_000;

        // let n_steps = 1; // todo temp, while we evaluate what's going wrong with our MD consts.
        // In femtoseconds
        let dt = 1.;

        for _ in 0..n_steps {
            md_state.step(dt)
        }

        for (i, atom) in md_state.atoms.iter().enumerate() {
            lig.molecule.atoms[i].posit = atom.posit;
        }

        Ok(md_state)
    }
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

    lig.pose = snapshot.pose.clone();

    // Position atoms from pose  here? You could, but the snapshot has them pre-positioned.
    // This may make changing snapshots faster. But uses more memory from storing each

    lig.atom_posits = snapshot
        .lig_atom_posits
        .iter()
        .map(|p| (*p).into())
        .collect();

    *energy_disp = snapshot.energy.clone();
}

/// Body masses are separate from the snapshot, since it's invariant.
pub fn change_snapshot_md(
    entities: &mut [Entity],
    lig: &mut Ligand,
    lig_entity_ids: &[usize],
    energy_disp: &mut Option<BindingEnergy>,
    snapshot: &SnapshotDynamics,
) {
    lig.pose.conformation_type = ConformationType::AbsolutePosits; // Should alreayd be set?

    // Position atoms from pose  here? You could, but the snapshot has them pre-positioned.
    // This may make changing snapshots faster. But uses more memory from storing each

    lig.atom_posits = snapshot.atom_posits.iter().map(|p| (*p).into()).collect();

    // *energy_disp = snapshot.energy.clone();
}
