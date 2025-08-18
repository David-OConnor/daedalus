//! Code for initializing water molecules, including assigning quantity, initial positions, and
//! velocities. Set up to meet density, pressure, and or temperature targets. Not specific to the
//! water model used.

use std::f64::consts::TAU;

use lin_alg::f64::{Mat3, Quaternion, Vec3};
use rand::{Rng, distr::Uniform};
use rand_distr::Distribution;

use crate::dynamics::{
    ACCEL_CONVERSION_INV, AtomDynamics, KB, ambient::SimBox, water_opc::WaterMol,
};

// 0.997 g cm⁻³ is a good default density. We use this for initializing and maintaining
// the water density and molecule count.
const WATER_DENSITY: f64 = 0.997;

// Don't generate water molecules that are too close to other atoms.
// Vdw contact distance between water molecules and organic molecules is roughly 3.5 Å.
const GENERATION_MIN_DIST: f64 = 3.75;
// A conservative water-water (Oxygen-Oxygen) minimum distance. 2.7 - 3.2 Å is suitable.
const MIN_OO_DIST: f64 = 2.8;

// This is similar to the Amber H and O masses we used summed, and could be explained
// by precision limits. We use it for generating atoms based on density.
const MASS_WATER: f64 = 18.015_28;

// Avogadro's constant. mol^-1.
const N_A: f64 = 6.022_140_76e23;

/// We pass atoms in so this doesn't generate water molecules that overlap with them.
pub fn make_water_mols(
    cell: &SimBox,
    t_target: f64,
    atoms_dy: &[AtomDynamics],
    atoms_static: &[AtomDynamics],
) -> Vec<WaterMol> {
    let vol = cell.volume();

    let n_float = WATER_DENSITY * vol * (N_A / (MASS_WATER * 1.0e24));
    let n_mols = n_float.round() as usize;

    let mut result = Vec::with_capacity(n_mols);
    let mut rng = rand::rng();

    let uni01 = Uniform::<f64>::new(0.0, 1.0).unwrap();

    let mut attempts = 0;
    let max_attempts = 50 * n_mols; // todo: Tune A/R.

    while result.len() < n_mols && attempts < max_attempts {
        attempts += 1;

        let posit = Vec3::new(
            rng.sample(uni01) * (cell.bounds_high.x - cell.bounds_low.x) + cell.bounds_low.x,
            rng.sample(uni01) * (cell.bounds_high.y - cell.bounds_low.y) + cell.bounds_low.y,
            rng.sample(uni01) * (cell.bounds_high.z - cell.bounds_low.z) + cell.bounds_low.z,
        );

        // Shoemake quaternion, then normalize (cheap & safe)
        let (u1, u2, u3) = (rng.sample(uni01), rng.sample(uni01), rng.sample(uni01));
        let sqrt1_minus_u1 = (1.0 - u1).sqrt();
        let sqrt_u1 = u1.sqrt();
        let (theta1, theta2) = (TAU * u2, TAU * u3);

        let q = Quaternion::new(
            sqrt1_minus_u1 * theta1.sin(),
            sqrt1_minus_u1 * theta1.cos(),
            sqrt_u1 * theta2.sin(),
            sqrt_u1 * theta2.cos(),
        )
        .to_normalized();

        if too_close_to_atoms(posit, atoms_dy, cell)
            || too_close_to_atoms(posit, atoms_static, cell)
            || too_close_to_waters(posit, &result, cell)
        {
            continue;
        }

        result.push(WaterMol::new(posit, Vec3::new_zero(), q));
    }

    if result.len() < n_mols {
        eprintln!(
            "Placed {} / {} waters; consider enlarging the box or loosening thresholds.",
            result.len(),
            n_mols
        );
    }

    init_velocities_rigid(&mut result, t_target, cell);
    result
}

fn init_velocities_rigid(mols: &mut [WaterMol], t_target: f64, _cell: &SimBox) {
    use rand_distr::Normal;

    let mut rng = rand::rng();
    let kT = KB * t_target; // in your internal units (kcal/mol if that’s what KE uses)
    for m in mols.iter_mut() {
        // COM & relative positions
        let (r_com, m_tot) = {
            let mut r = Vec3::new_zero();
            let mut m_tot = 0.0;
            for a in [&m.o, &m.h0, &m.h1] {
                r += a.posit * a.mass;
                m_tot += a.mass;
            }
            (r / m_tot, m_tot)
        };

        let rO = m.o.posit - r_com;
        let rH0 = m.h0.posit - r_com;
        let rH1 = m.h1.posit - r_com;

        // Sample COM velocity
        let sigma_v = (kT / m_tot).sqrt();
        let n = Normal::new(0.0, sigma_v).unwrap();
        let v_com = Vec3::new(n.sample(&mut rng), n.sample(&mut rng), n.sample(&mut rng));

        // Inertia tensor about COM (world frame)
        // Build as arrays (your code)
        let inertia = |r: Vec3, mass: f64| {
            let r2 = r.dot(r);
            [
                [
                    mass * (r2 - r.x * r.x),
                    -mass * r.x * r.y,
                    -mass * r.x * r.z,
                ],
                [
                    -mass * r.y * r.x,
                    mass * (r2 - r.y * r.y),
                    -mass * r.y * r.z,
                ],
                [
                    -mass * r.z * r.x,
                    -mass * r.z * r.y,
                    mass * (r2 - r.z * r.z),
                ],
            ]
        };
        let mut I_arr = inertia(rO, m.o.mass);
        let add_I = |I: &mut [[f64; 3]; 3], J: [[f64; 3]; 3]| {
            for i in 0..3 {
                for j in 0..3 {
                    I[i][j] += J[i][j];
                }
            }
        };
        add_I(&mut I_arr, inertia(rH0, m.h0.mass));
        add_I(&mut I_arr, inertia(rH1, m.h1.mass));

        // Convert to Mat3 once, then use
        let I = Mat3::from_arr(I_arr);

        // Diagonalize and solve with the Mat3 methods
        let (eigvecs, eigvals) = I.eigen_vecs_vals();
        let L_principal = Vec3::new(
            rand_distr::Normal::new(0.0, (kT * eigvals.x.max(0.0)).sqrt())
                .unwrap()
                .sample(&mut rng),
            rand_distr::Normal::new(0.0, (kT * eigvals.y.max(0.0)).sqrt())
                .unwrap()
                .sample(&mut rng),
            rand_distr::Normal::new(0.0, (kT * eigvals.z.max(0.0)).sqrt())
                .unwrap()
                .sample(&mut rng),
        );
        let L_world = eigvecs * L_principal; // assumes Mat3 * Vec3 is implemented
        let omega = I.solve_system(L_world); // ω = I^{-1} L

        // Set atomic velocities
        m.o.vel = v_com + omega.cross(rO);
        m.h0.vel = v_com + omega.cross(rH0);
        m.h1.vel = v_com + omega.cross(rH1);
    }

    // Remove global COM drift
    remove_com_velocity(mols);

    // Optional: compute KE (translation+rotation == sum ½ m v^2 now) and rescale to T_target
    let (ke_raw, dof) = kinetic_energy_and_dof(mols); // dof = 6*N - 3
    let lambda = (t_target / (2.0 * (ke_raw * ACCEL_CONVERSION_INV) / (dof as f64 * KB))).sqrt();
    for a in atoms_mut(mols) {
        if a.mass > 0.0 {
            a.vel *= lambda;
        }
    }
}

fn kinetic_energy_and_dof(mols: &[WaterMol]) -> (f64, usize) {
    let mut ke = 0.0;
    let mut dof = 0usize;
    for m in mols {
        for a in [&m.o, &m.h0, &m.h1] {
            ke += 0.5 * a.mass * a.vel.dot(a.vel);
            dof += 3;
        }
    }
    // remove 3 for total COM; remove constraints if you track them
    let n_constraints = 3 * mols.len();
    (ke, dof - 3 - n_constraints)
}

fn atoms_mut(mols: &mut [WaterMol]) -> impl Iterator<Item = &mut AtomDynamics> {
    mols.iter_mut()
        .flat_map(|m| [&mut m.o, &mut m.h0, &mut m.h1].into_iter())
}

/// Removes center-of-mass drift.
fn remove_com_velocity(mols: &mut [WaterMol]) {
    let mut p = Vec3::new_zero();
    let mut m_tot = 0.0;
    for a in atoms_mut(mols) {
        p += a.vel * a.mass;
        m_tot += a.mass;
    }

    let v_com = p / m_tot;
    for a in atoms_mut(mols) {
        a.vel -= v_com;
    }
}

fn too_close_to_atoms(p: Vec3, atoms: &[AtomDynamics], cell: &SimBox) -> bool {
    for a in atoms {
        let d = cell.min_image(a.posit - p).magnitude();
        if d < GENERATION_MIN_DIST {
            return true;
        }
    }
    false
}

fn too_close_to_waters(p: Vec3, waters: &[WaterMol], cell: &SimBox) -> bool {
    for w in waters {
        let d = cell.min_image(w.o.posit - p).magnitude();
        if d < MIN_OO_DIST {
            return true;
        }
    }
    false
}
