//! Code for initializing water molecules, including assigning initial positions and velocities.

use std::f64::consts::TAU;

use lin_alg::f64::{Quaternion, Vec3};
use na_seq::Element;
use rand::{Rng, distr::Uniform};
use rand_distr::{Distribution, StandardNormal};

use crate::dynamics::{
    ACCEL_CONVERSION_INV, AtomDynamics, KB, ambient::SimBox, water_opc::WaterMol,
};

// 0.997 g cm⁻³ is a good default density. We use this for initializing and maintaining
// the water density and molecule count.
const WATER_DENSITY: f64 = 0.997;

// Don't generate water molecules that are too close to other atoms.
// Vdw contact distance between water molecules and organic molecules is roughly 3.5 Å.
const GENERATION_MIN_DIST: f64 = 4.;

// This is similar to the Amber H and O masses we used summed, and could be explained
// by precision limits. We use it for generating atoms based on density.
const MASS_WATER: f64 = 18.015_28;

const NA: f64 = 6.022_140_76e23; // todo: What is this? Used in density calc. mol⁻¹

/// We pass atoms in so this doesn't generate water molecules that overlap with them.
pub fn make_water_mols(
    cell: &SimBox,
    t_target: f64,
    atoms_dy: &[AtomDynamics],
    atoms_static: &[AtomDynamics],
) -> Vec<WaterMol> {
    let vol = cell.volume();

    let n_float = WATER_DENSITY * vol * (NA / (MASS_WATER * 1.0e24));
    let n_mols = n_float.round() as usize;

    let mut result = Vec::with_capacity(n_mols);
    let mut rng = rand::rng();

    let uni01 = Uniform::<f64>::new(0.0, 1.0).unwrap();

    for _ in 0..n_mols {
        // Position (axis‑aligned box)
        let posit = Vec3::new(
            rng.sample(uni01) * (cell.bounds_high.x - cell.bounds_low.x) + cell.bounds_low.x,
            rng.sample(uni01) * (cell.bounds_high.y - cell.bounds_low.y) + cell.bounds_low.y,
            rng.sample(uni01) * (cell.bounds_high.z - cell.bounds_low.z) + cell.bounds_low.z,
        );

        // Orientation (uniform SO(3))
        // Shoemake (1992)
        let (u1, u2, u3) = (rng.sample(uni01), rng.sample(uni01), rng.sample(uni01));
        let q = {
            let sqrt1_minus_u1 = (1.0 - u1).sqrt();
            let sqrt_u1 = u1.sqrt();
            let (theta1, theta2) = (TAU * u2, TAU * u3);

            Quaternion::new(
                sqrt1_minus_u1 * theta1.sin(),
                sqrt1_minus_u1 * theta1.cos(),
                sqrt_u1 * theta2.sin(),
                sqrt_u1 * theta2.cos(),
            )
        };

        // todo: Min dist between water mols?
        // Don't place water molecules too close to other atoms.
        let mut skip = false;
        for atom_set in [atoms_dy, atoms_static] {
            for atom_non_water in atom_set {
                let dist = (atom_non_water.posit - posit).magnitude();
                if dist < GENERATION_MIN_DIST {
                    skip = true;
                    break;
                }
            }
            if skip {
                break;
            }
        }

        if skip {
            continue;
        }

        // We update velocities after
        result.push(WaterMol::new(posit, Vec3::new_zero(), q));
    }

    // Assign velocities based on temperature.
    init_velocities(&mut result, t_target);

    result
}

fn init_velocities(mols: &mut [WaterMol], t_target: f64) {
    let mut rng = rand::rng();

    // Gaussian draw
    for a in atoms_mut(mols) {
        if a.element == Element::Potassium {
            // M/EP
            continue;
        }

        let nx: f64 = StandardNormal.sample(&mut rng);
        let ny: f64 = StandardNormal.sample(&mut rng);
        let nz: f64 = StandardNormal.sample(&mut rng);

        // arbitrary sigma=1 for now; we'll rescale below
        a.vel = Vec3::new(nx, ny, nz);
    }

    // Remove center-of-mass drift
    remove_com_velocity(mols);

    // Compute instantaneous T
    let (ke_raw, dof) = kinetic_energy_and_dof(mols);
    let ke_kcal = ke_raw * ACCEL_CONVERSION_INV;
    let t_now = 2.0 * ke_kcal / (dof as f64 * KB);

    // Rescale to T_target
    let lambda = (t_target / t_now).sqrt();
    for a in atoms_mut(mols) {
        if a.mass == 0.0 {
            continue;
        }
        a.vel *= lambda;
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
