use std::f32::consts::TAU;

use barnes_hut::BodyModel;
use lin_alg::{f32::Vec3, f64::Vec3 as Vec3F64};
use rand::Rng;

use crate::{
    molecule::{Atom, Bond},
    util::setup_neighbor_pairs,
};

const GRID_SIZE: f64 = 1.6; // Slightly larger than the largest... todo: What?

#[derive(Debug)]
pub struct PartialCharge {
    pub posit: Vec3,
    pub charge: f32,
}

// todo: Ideally, you want f32 here, I believe. Need to modifify barnes_hut lib A/R to support.
impl BodyModel for PartialCharge {
    fn posit(&self) -> Vec3F64 {
        // fn posit(&self) -> Vec3 {
        self.posit.into()
    }

    fn mass(&self) -> f64 {
        // fn mass(&self) -> f32 {
        self.charge.into()
    }
}

/// Molecular Electrostatic potential
fn kollman_mep() {
    let layers = [1.4_f32, 1.6, 1.8, 2.0];

    let n_gridpoints_per_unit_area = 1;
    for layer_r in layers.iter() {
        let area = 2. * TAU * layer_r.powi(2);
        let n_grid = (n_gridpoints_per_unit_area as f32 * area) as usize;
        // let mut grid_points = Vec::with_capacity(n_grid);
    }

    // After evaluating the MEP at all valid grid points located on all four layers, atomic charges
    // are derived that reproduce the MEP as closely as possible."
}

/// Create a set of partial charges around atoms. Rough simulation of electronic density imbalances
/// in charges molecules, and/or at short distances. It places a single positive charge
/// at the center of each atom, representing its nucleus. It spreads out a number of smaller-charges
/// to represent electron density, around that nucleus.
///
/// `charge_density` is how many partial-charge points to generate per electron;
/// if 4.0, for instance, you’ll get ~4 negative “points” per electron.
pub fn create_partial_charges(atoms: &[Atom], charge_density: f32) -> Vec<PartialCharge> {
    let mut result = Vec::new();
    let mut rng = rand::rng();

    for atom in atoms {
        let z = atom.element.atomic_number();

        // If partial_charge is known, interpret it as net formal charge:
        // e.g., partial_charge = +1 => 1 fewer electron
        let net_charge = atom.partial_charge.unwrap_or(0.0).round() as i32; // e.g. +1, -1, 0, ...
        let electron_count = z as i32 - net_charge; // simplistic integer

        // 1) Add nucleus partial charge, +Z or adjusted by net charge if you like.
        //    For example, if net_charge=+1, you can either keep the nucleus as +Z
        //    or set it to + (z - net_charge). The usual naive approach is +Z at
        //    the nucleus, letting the negative electron distribution handle the difference.
        result.push(PartialCharge {
            posit: atom.posit.into(),
            charge: z as f32,
        });

        // 2) Distribute negative charges around the nucleus
        if electron_count > 0 {
            let count_neg_points = (electron_count as f32 * charge_density).round() as i32;
            if count_neg_points > 0 {
                let neg_charge_per_point = -(electron_count as f32) / (count_neg_points as f32);

                // Decide on a radius scale
                // For small molecules, you might pick something like 0.2–0.5 Å for “cloud radius”
                // or randomize in a band: [0, r_max].
                let r_max = 0.3;

                for _ in 0..count_neg_points {
                    // Sample a random direction
                    // We can do this by sampling spherical coordinates or
                    // sampling a random vector from a Gaussian, then normalizing
                    let theta = rng.random_range(0.0..TAU);
                    let u: f32 = rng.random_range(-1.0..1.0); // for cos(phi)

                    let sqrt_1_minus_u2 = (1.0 - u * u).sqrt();
                    let dx = sqrt_1_minus_u2 * theta.cos();
                    let dy = sqrt_1_minus_u2 * theta.sin();
                    let dz = u;

                    let dir = Vec3::new(dx, dy, dz);

                    // random radius in [0..r_max]
                    let radius = rng.random_range(0.0..r_max);
                    let offset = dir * radius;

                    let atom_posit: Vec3 = atom.posit.into();
                    result.push(PartialCharge {
                        posit: atom_posit + offset,
                        charge: neg_charge_per_point,
                    });
                }
            }
        }
    }

    result
}

/// Create a set of partial charges using a simplified Kollman approach.
/// This routine does two things:
///
/// 1. For each atom, it assigns a base partial charge from a lookup (mimicking Kollman charges).
///
/// 2. For bonds that are “polar” (i.e. where the electronegativity difference exceeds a threshold),
///    it adds a pair of extra point charges to mimic a dipole moment along the bond.
///    These extra charges sum to zero, so they do not change the net charge but add a dipole.
///    The `dipole_offset` controls how far (in Å) from the nucleus the extra points are placed.
pub fn create_kollman_partial_charges(
    atoms: &[Atom],
    bonds: &[Bond],
    dipole_offset: f32,
    polar_threshold: f32, // e.g. 0.4 on the Pauling scale
) -> Vec<PartialCharge> {
    let mut result = Vec::new();

    // 1. Assign base Kollman charges to each atom.
    // (These are assumed to be centered on the atomic nucleus.)
    for atom in atoms {
        let base_charge = atom.element.kollman_charge();
        result.push(PartialCharge {
            posit: atom.posit.into(),
            charge: base_charge as f32,
        });
    }

    // 2. For each bond, check if it is polar.
    // If the electronegativity difference is larger than a threshold,
    // add an extra pair of charges to mimic a dipole.
    for bond in bonds {
        let atom_i = &atoms[bond.atom_0];
        let atom_j = &atoms[bond.atom_1];

        let en_i = atom_i.element.electronegativity();
        let en_j = atom_j.element.electronegativity();
        let diff = (en_i - en_j).abs();

        if diff > polar_threshold {
            // Determine the bond vector (from i to j) and its unit vector.
            let bond_vec = atom_j.posit - atom_i.posit;
            let unit: Vec3 = bond_vec.to_normalized().into();

            // We'll create a small dipole by placing two extra charges,
            // one shifted by +dipole_offset and one by -dipole_offset from the midpoint of the bond.
            // The magnitude of these extra charges is an arbitrary parameter; here we use 0.1e.
            let extra_charge = 0.1;

            // Compute the midpoint of the bond.
            let mid: Vec3 = Vec3F64::new(
                (atom_i.posit.x + atom_j.posit.x) * 0.5,
                (atom_i.posit.y + atom_j.posit.y) * 0.5,
                (atom_i.posit.z + atom_j.posit.z) * 0.5,
            )
            .into();

            // Place the extra charges along the bond direction.
            let offset = unit * dipole_offset;
            result.push(PartialCharge {
                posit: mid + offset,
                charge: extra_charge,
            });
            result.push(PartialCharge {
                posit: mid - offset,
                charge: -extra_charge,
            });
        }
    }

    result
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PartialChargeType {
    Gasteiger,
    Kollman,
}

/// Note: Hydrogens must already be added prior to adding charges.
pub fn setup_partial_charges(atoms: &mut [Atom], charge_type: PartialChargeType) {
    if charge_type == PartialChargeType::Kollman {
        unimplemented!()
    }

    // We use spacial partitioning, so as not to copmare every pair of atoms.
    let posits: Vec<_> = atoms.iter().map(|a| &a.posit).collect();
    let indices: Vec<_> = (0..atoms.len()).collect();
    let neighbor_pairs = setup_neighbor_pairs(&posits, &indices, GRID_SIZE);

    // Run the iterative charge update over all candidate pairs.
    const ITERATIONS: usize = 6; // More iterations may be needed in practice.
    for _ in 0..ITERATIONS {
        let mut charge_updates = vec![0.0; atoms.len()];

        for &(i, j) in &neighbor_pairs {
            if atoms[i].dock_type.is_none() || atoms[j].dock_type.is_none() {
                continue;
            }
            let en_i = atoms[i].dock_type.unwrap().gasteiger_electronegativity();
            let en_j = atoms[j].dock_type.unwrap().gasteiger_electronegativity();
            // Compute a simple difference-based transfer.
            let delta = 0.1 * (en_i - en_j);
            // Transfer charge from atom i to atom j if en_i > en_j.
            charge_updates[i] -= delta;
            charge_updates[j] += delta;
        }

        // Apply the computed updates simultaneously.
        for (atom, delta) in atoms.iter_mut().zip(charge_updates.iter()) {
            match &mut atom.partial_charge {
                Some(c) => *c += delta,
                None => atom.partial_charge = Some(*delta),
            }
        }
    }
}
