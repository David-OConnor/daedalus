//! This module contains a traditional molecular dynamics approach
//!
//! [Good article](https://www.owlposting.com/p/a-primer-on-molecular-dynamics)
//! [A summary paper](https://arxiv.org/pdf/1401.1181)
//!
//! [Amber Force Fields reference](https://ambermd.org/AmberModels.php)
//! [Small molucules using GAFF2](https://ambermd.org/downloads/amber_geostd.tar.bz2)
//!
//! To download .dat files (GAFF2), download Amber source (Option 2) [here](https://ambermd.org/GetAmber.php#ambertools).
//! Files are in dat -> leap -> parm

//! We are using f64, and CPU-only for now, unless we confirm f32 will work.
//! Maybe a mixed approach: Coordinates, velocities, and forces in 32-bit; sensitive global
//! reductions (energy, virial, integration) in 64-bit.
//!
//! We use Verlet integration. todo: Velocity verlet? Other techniques that improve and build upon it?
//!
//! Amber: ff19SB for proteins, gaff2 for ligands. (Based on recs from https://ambermd.org/AmberModels.php).
//!
//!

// todo: Integration: Consider Verlet or Langevin

//  todo: Pressure and temp variables required. Perhaps related to ICs of water?

// todo: Long-term, you will need to figure out what to run as f32 vice f64, especially
// todo for being able to run on GPU.

// Note on timescale: Generally femtosecond (-15)

mod ambient;
mod prep;

use std::{
    collections::{HashMap, HashSet},
    string::ParseError,
};

use ambient::SimBox;
use bio_files::amber_params::{
    AngleData, BondData, DihedralData, ForceFieldParams, MassData, VdwData,
};
use lin_alg::f64::{Vec3, calc_dihedral_angle, calc_dihedral_angle_v2};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use lin_alg::f64::{Vec3x4, f64x4};
use na_seq::{Element, element::LjTable};
use rand_distr::Distribution;

use crate::{
    forces::{force_coulomb, force_lj},
    molecule::{Atom, Bond},
};

// vacuum permittivity constant   (k_e = 1/(4π ε0))
// SI, matched with charge in e₀ & Å → kcal mol // todo: QC
const ε_0: f64 = 8.8541878128e-1;
const k_e: f32 = 332.0636; // kcal·Å /(mol·e²)
// const ε_0: f64 = 1.;

// Verlet list parameters
const CUTOFF: f64 = 12.0; // Å
const SKIN: f64 = 2.0; // Å – rebuild list if an atom moved >½·SKIN
const M_O: f64 = 15.999; // Da
const M_H: f64 = 1.008; // Da
const R_OH: f64 = 0.9572; // Å
const ANG_HOH: f64 = 104.52_f64.to_radians();

// For exclusions.
const SCALE_LJ_14: f64 = 0.5; // AMBER default
const SCALE_COUL_14: f64 = 1.0 / 1.2; // 0.833̅

const SOFTENING_FACTOR_SQ: f64 = 1e-6;

// Conversion factor
// 2^(5/6); no powf in consts.
const SIGMA_FROM_RSTAR: f64 = 1.7817974362806785;

// todo: A/R
const SNAPSHOT_RATIO: usize = 10;

const EPS: f64 = 1.0e-8;

#[derive(Debug)]
pub struct ParamError {
    pub descrip: String,
}

impl ParamError {
    pub fn new(descrip: &str) -> Self {
        Self {
            descrip: descrip.to_owned(),
        }
    }
}

/// Todo: Experimenting with using indices. This is a derivative of the `Keyed` variant.
/// This variant of forcefield parameters offers the fastest lookups. Unlike the Vec and Hashmap
/// based parameter structs, this is specific to the atom in our docking setup: The incdices are provincial
/// to specific sets of atoms.
///
/// Note: The single-atom fields of `mass` and `partial_charges` are ommitted: They're part of our
/// `AtomDynamics` struct.`
#[derive(Debug, Default)]
pub struct ForceFieldParamsIndexed {
    pub mass: HashMap<usize, MassData>,
    pub bond: HashMap<(usize, usize), BondData>,
    pub angle: HashMap<(usize, usize, usize), AngleData>,
    /// This includes both normal, and improper dihedrals.
    pub dihedral: HashMap<(usize, usize, usize, usize), DihedralData>,
    // pub dihedral: HashMap<(usize, usize, usize, usize), Option<DihedralData>>,

    // Dihedrals are represented in Amber params as a fourier series; this Vec indlues all matches.
    // e.g. X-ca-ca-X may be present multiple times in gaff2.dat. (Although seems to be uncommon)
    //
    // X -nh-sx-X    4    3.000         0.000          -2.000
    // X -nh-sx-X    4    0.400       180.000           3.000

    // pub dihedral: HashMap<(usize, usize, usize, usize), Vec<DihedralData>>,
    // pub improper: HashMap<(usize, usize, usize, usize), DihedralData>,
    pub van_der_waals: HashMap<usize, VdwData>,
    pub partial_charge: HashMap<usize, f32>, // todo: A/r
}

#[derive(Debug, Default)]
pub struct SnapshotDynamics {
    pub time: f64,
    pub atom_posits: Vec<Vec3>,
    pub atom_velocities: Vec<Vec3>,
}

#[derive(Clone, Debug)]
/// A trimmed-down atom for use with molecular dynamics.
pub struct AtomDynamics {
    pub element: Element,
    pub name: String,
    pub posit: Vec3,
    pub vel: Vec3,
    pub accel: Vec3,
    /// Daltons
    /// todo: Move these 4 out of this to save memory; use from the params struct directly.
    pub mass: f64,
    pub partial_charge: f64,
    pub lj_r_star: f64,
    pub lj_eps: f64,
    pub force_field_type: Option<String>, // todo: Should this be an enum? // todo: Should it be required?
}

// impl AtomDynamics {
//     pub fn new(element: Element) -> Self {
//         Self {
//             element,
//             name: String::new(),
//             posit: Vec3::new_zero(),
//             vel: Vec3::new_zero(),
//             accel: Vec3::new_zero(),
//             mass: element.atomic_weight() as f64,
//             partial_charge: 0.0,
//             force_field_type: None,
//         }
//     }
// }

impl From<&Atom> for AtomDynamics {
    fn from(atom: &Atom) -> Self {
        Self {
            element: atom.element,
            name: atom.name.clone().unwrap_or_default(),
            posit: atom.posit.into(),
            vel: Vec3::new_zero(),
            accel: Vec3::new_zero(),
            mass: atom.element.atomic_weight() as f64,
            partial_charge: atom.partial_charge.unwrap_or_default() as f64,
            lj_r_star: 0.,
            lj_eps: 0.,
            force_field_type: atom.force_field_type.clone(),
        }
    }
}

impl AtomDynamics {
    fn new(
        atom: &Atom,
        atom_posits: &[Vec3],
        ff_params: &ForceFieldParamsIndexed,
        i: usize,
    ) -> Result<Self, ParamError> {
        let ff_type = match &atom.force_field_type {
            Some(ff_type) => ff_type.clone(),
            None => {
                return Err(ParamError::new(&format!(
                    "Atom missing FF type; can't run dynamics: {:?}",
                    atom
                )));
            }
        };

        Ok(Self {
            element: atom.element,
            name: atom.name.clone().unwrap_or_default(),
            posit: atom_posits[i],
            vel: Vec3::new_zero(),
            accel: Vec3::new_zero(),
            mass: ff_params.mass.get(&i).unwrap().mass as f64,
            // todo: A/R for partial charge.
            // partial_charge: atom.partial_charge.unwrap_or_default() as f64,
            partial_charge: *ff_params.partial_charge.get(&i).unwrap() as f64,
            lj_r_star: ff_params.van_der_waals.get(&i).unwrap().r_star as f64,
            lj_eps: ff_params.van_der_waals.get(&i).unwrap().eps as f64,
            force_field_type: Some(ff_type),
        })
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[derive(Clone, Debug)]
pub(crate) struct AtomDynamicsx4 {
    // pub posit: Vec3x8,
    // pub vel: Vec3x8,
    // pub accel: Vec3x8,
    // pub mass: f32x8,
    pub posit: Vec3x4,
    pub vel: Vec3x4,
    pub accel: Vec3x4,
    pub mass: f64x4,
    pub element: [Element; 4],
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl AtomDynamicsx4 {
    pub fn from_array(bodies: [AtomDynamics; 4]) -> Self {
        let mut posits = [Vec3::new_zero(); 4];
        let mut vels = [Vec3::new_zero(); 4];
        let mut accels = [Vec3::new_zero(); 4];
        let mut masses = [0.0; 4];
        // Replace `Element::H` (for example) with some valid default for your `Element` type:
        let mut elements = [Element::Hydrogen; 4];

        for (i, body) in bodies.into_iter().enumerate() {
            posits[i] = body.posit;
            vels[i] = body.vel;
            accels[i] = body.accel;
            masses[i] = body.mass;
            elements[i] = body.element;
        }

        Self {
            posit: Vec3x4::from_array(posits),
            vel: Vec3x4::from_array(vels),
            accel: Vec3x4::from_array(accels),
            mass: f64x4::from_array(masses),
            element: elements,
        }
    }
}

#[derive(Default)]
pub struct MdState {
    pub atoms: Vec<AtomDynamics>,
    pub adjacency_list: Vec<Vec<usize>>,
    /// Sources that affect atoms in the system, but are not themselves affected by it. E.g.
    /// in docking, this might be a rigid receptor. These are for *non-bonded* interactions (e.g. Coulomb
    /// and VDW) only.
    pub atoms_external: Vec<AtomDynamics>,
    pub force_field_params: ForceFieldParamsIndexed,
    /// `lj_lut`, `lj_sigma`, and `lj_eps` are Lennard Jones parameters. Flat here, with outer loop receptor.
    /// Flattened. Separate single-value array facilitate use in CUDA and SIMD, vice a tuple.
    /// todo: These are from our built-in table. Generalize, e.g. organize appropriately w/Amber.
    ///     pub lj_lut: LjTable,
    pub lj_sigma: Vec<f64>,
    pub lj_eps: Vec<f64>,
    // todo: Implment these SIMD variants A/R, bearing in mind the caveat about our built-in ones vs
    // todo ones loaded from [e.g. Amber] files.
    // #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    // pub lj_sigma_x8: Vec<f64x4>,
    // #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    // pub lj_eps_x8: Vec<f64x4>,
    /// In femtoseconds,
    pub time: f64,
    pub step_count: usize, // increments.
    pub snapshots: Vec<SnapshotDynamics>,
    pub cell: SimBox,
    neighbour: Vec<Vec<usize>>,    // Verlet list
    max_disp_sq: f64,              // track atom displacements²
    pub kb_berendsen: Option<f64>, // coupling constant (ps⁻¹) if you want a thermostat
    pub target_temp: f64,
    /// Exclusions / masks optimization.
    excluded_pairs: HashSet<(usize, usize)>, // 1-2 and 1-3
    scaled14_pairs: HashSet<(usize, usize)>, // 1-4
}

impl MdState {
    /// One **Velocity-Verlet** step (leap-frog style) of length `dt_fs` femtoseconds.
    pub fn step(&mut self, dt_fs: f64) {
        // todo: Is this OK?
        // let dt = dt_fs * 1.0e-15; // s
        let dt = dt_fs * 0.001;

        let dt_half = 0.5 * dt;

        // 1) First half-kick (v += a dt/2) and drift (x += v dt)
        // todo: Do we want traditional verlet instead?
        for a in &mut self.atoms {
            a.vel += a.accel * dt_half; // Half-kick
            a.posit += a.vel * dt; // Drift
            a.posit = self.cell.wrap(a.posit);

            // track the largest squared displacement to know when to rebuild the list
            self.max_disp_sq = self.max_disp_sq.max((a.vel * dt).magnitude_squared());
        }

        // Reset acceleration.
        for a in &mut self.atoms {
            a.accel = Vec3::new_zero();
        }

        self.apply_bond_forces();
        // self.apply_valence_angle_forces();
        self.apply_dihedral_forces();
        // self.apply_nonbonded_forces();

        // Second half-kick using new accelerations
        for a in &mut self.atoms {
            a.vel += a.accel * dt_half;
        }

        // Berendsen thermostat (T coupling to target every step)
        if let Some(tau_ps) = self.kb_berendsen {
            let tau = tau_ps * 1e-12;
            let curr_ke = self.current_kinetic_energy();
            let curr_t = 2.0 * curr_ke / (3.0 * self.atoms.len() as f64 * 0.0019872041); // k_B in kcal/mol
            let λ = (1.0 + dt / tau * (self.target_temp - curr_t) / curr_t).sqrt();
            for a in &mut self.atoms {
                a.vel *= λ;
            }
        }

        self.time += dt;
        self.step_count += 1;

        // Rebuild Verlet if needed
        if self.max_disp_sq > 0.25 * SKIN * SKIN {
            self.build_neighbours();
        }

        if self.step_count % SNAPSHOT_RATIO == 0 {
            self.take_snapshot();
        }
    }

    fn apply_bond_forces(&mut self) {
        for (indices, data) in &self.force_field_params.bond {
            let (a_0, a_1) = split2_mut(&mut self.atoms, indices.0, indices.1);

            let diff = a_1.posit - a_0.posit;
            let dist = diff.magnitude();

            let delta = dist - data.r_0 as f64;

            if delta > 0.1 {
                // println!("DIST: {:?}, {:?}, d: {:.04}, r0: {:.04}", &ai.force_field_type.as_ref().unwrap(),
                //          &aj.force_field_type.as_ref().unwrap(), dist, data.r_0);
            }

            // F = -dV/dr = -k 2Δ   (harmonic)
            // let f_mag = -2.0 * k * Δ / dist.max(1e-12);
            let f_mag = -data.k as f64 * delta / dist.max(1e-12);
            let f = diff * f_mag;
            a_0.accel -= f / a_0.mass;
            a_1.accel += f / a_1.mass;
        }
    }

    /// This maintains bond angles between sets of three atoms as they should be from hybridization.
    /// It reflects this hybridization, steric clashes, and partial double-bond character. This
    /// identifies deviations from the ideal angle, calculates restoring torque, and applies forces
    /// based on this to push the atoms back into their ideal positions in the molecule.
    ///
    /// Valence angles, which are the angle formed by two adjacent bonds ba et bc
    /// in a same molecule; a valence angle tends to maintain constant the anglê
    /// abc. A valence angle is thus concerned by the positions of three atoms.
    fn apply_valence_angle_forces(&mut self) {
        for (indices, data) in &self.force_field_params.angle {
            let (a_0, a_1, a_2) = split3_mut(&mut self.atoms, indices.0, indices.1, indices.2);

            // Bond vectors with atom 1 at the vertex.
            let bond_vec_01 = a_0.posit - a_1.posit;
            let bond_vec_21 = a_2.posit - a_1.posit;

            let b_vec_01_sq = bond_vec_01.magnitude_squared();
            let b_vec_21_sq = bond_vec_21.magnitude_squared();

            // Quit early if atoms are on top of each other
            if b_vec_01_sq < EPS || b_vec_21_sq < EPS {
                continue;
            }

            let b_vec_01_len = b_vec_01_sq.sqrt();
            let b_vec_21_len = b_vec_21_sq.sqrt();

            let inv_ab = 1.0 / (b_vec_01_len * b_vec_21_len);

            let cos_θ = (bond_vec_01.dot(bond_vec_21) * inv_ab).clamp(-1.0, 1.0);
            let sin_θ_sq = 1.0 - cos_θ * cos_θ;

            if sin_θ_sq < EPS {
                continue; // θ = 0 or τ; gradient ill-defined
            }

            // Measured angle, and its deviation from the parameter angle.
            let θ = cos_θ.acos();
            let Δθ = data.angle as f64 - θ;
            let dV_dθ = 2.0 * data.k as f64 * Δθ; // dV/dθ

            let c = bond_vec_01.cross(bond_vec_21); // n  ∝  r_ij × r_kj
            let c_len2 = c.magnitude_squared(); // |n|^2

            let geom_i = (c.cross(bond_vec_01) * b_vec_21_len) / c_len2;
            let geom_k = (bond_vec_21.cross(c) * b_vec_01_len) / c_len2;

            let f_0 = -geom_i * dV_dθ;
            let f_2 = -geom_k * dV_dθ;
            let f_1 = -(f_0 + f_2);

            a_0.accel += f_0 / a_0.mass;
            a_1.accel += f_1 / a_1.mass;
            a_2.accel += f_2 / a_2.mass;
        }
    }

    /// This maintains dihedral angles. (i.e. the angle between four atoms in a sequence). This models
    /// effects such as σ-bond overlap (e.g. staggered conformations), π-conjugation, which locks certain
    /// dihedrals near 0 or τ, and steric hindrance. (Bulky groups clashing).
    fn apply_dihedral_forces(&mut self) {
        for (indices, dihe) in &self.force_field_params.dihedral {
            // let Some(dihe) = dihe_ else { continue };

            // Split the four atoms mutably without aliasing
            let (a_0, a_1, a_2, a_3) =
                split4_mut(&mut self.atoms, indices.0, indices.1, indices.2, indices.3);

            // Convenience aliases for the positions
            let r_0 = a_0.posit;
            let r_1 = a_1.posit;
            let r_2 = a_2.posit;
            let r_3 = a_3.posit;

            // Bond vectors (see Allen & Tildesley, chap. 4)
            let b1 = r_0 - r_1; // r_ij
            let b2 = r_2 - r_1; // r_kj
            let b3 = r_3 - r_2; // r_lk

            // Normal vectors to the two planes
            let n1 = b1.cross(b2);
            let n2 = b3.cross(b2);

            let n1_sq = n1.magnitude_squared();
            let n2_sq = n2.magnitude_squared();
            let b2_len = b2.magnitude();

            // Bail out if the four atoms are (nearly) colinear
            if n1_sq < EPS || n2_sq < EPS || b2_len < EPS {
                continue;
            }

            let dihe_measured = calc_dihedral_angle_v2(&(r_0, r_1, r_2, r_3));

            if self.step_count == 0 {
                println!(
                    "{:?} - Ms: {:.2}, exp: {:.2}/{} sin: {:.2}",
                    &dihe.ff_types,
                    dihe_measured,
                    dihe.phase,
                    dihe.periodicity,
                    (dihe.periodicity as f64) * dihe_measured - dihe.phase as f64
                );
            }

            // dV/dφ
            let dV_dφ = -0.5
                // todo: Precompute this barrier height when loading to the indexed variant.
                // todo: Do that once this all owrks.
                * (dihe.barrier_height_vn as f64) / (dihe.integer_divisor as f64)
                * (dihe.periodicity as f64)
                * ((dihe.periodicity as f64) * dihe_measured - dihe.phase as f64).sin();

            // ∂φ/∂r   (see e.g. DOI 10.1016/S0021-9991(97)00040-8)
            let dφ_dr1 = n1 * (b2_len / n1_sq);
            let dφ_dr4 = -n2 * (b2_len / n2_sq);
            let dφ_dr2 =
                -n1 * (b1.dot(b2) / (b2_len * n1_sq)) + n2 * (b3.dot(b2) / (b2_len * n2_sq));
            let dφ_dr3 = -dφ_dr1 - dφ_dr2 - dφ_dr4; // Newton’s third law

            // F_i = −dV/dφ · ∂φ/∂r_i
            let f1 = -dφ_dr1 * dV_dφ;
            let f2 = -dφ_dr2 * dV_dφ;
            let f3 = -dφ_dr3 * dV_dφ;
            let f4 = -dφ_dr4 * dV_dφ;

            // Convert to accelerations
            a_0.accel += f1 / a_0.mass;
            a_1.accel += f2 / a_1.mass;
            a_2.accel += f3 / a_2.mass;
            a_3.accel += f4 / a_3.mass;
        }
    }

    /// Coulomb and Van der Waals.
    ///
    /// todo: If required, build a neighbors list for interactions with external atoms.
    fn apply_nonbonded_forces(&mut self) {
        let cutoff_sq = CUTOFF * CUTOFF;

        const EPS: f64 = 1e-6;

        for i in 0..self.atoms.len() {
            // todo: Can you unify this with your neighbor code used for bonds?
            for &j in &self.neighbour[i] {
                if j < i {
                    // Prevents duplication of the pair in the other order.
                    continue;
                }

                // Handle masks.
                let key = if i < j { (i, j) } else { (j, i) };

                if self.excluded_pairs.contains(&key) {
                    continue;
                }

                let scale14 = self.scaled14_pairs.contains(&key);

                let diff = self.atoms[j].posit - self.atoms[i].posit;

                let dv = self.cell.min_image(diff);
                let r_sq = dv.magnitude_squared();
                if r_sq > cutoff_sq {
                    continue;
                }

                // todo: Put back A/R, or load from amber.
                // let (σ, ϵ) = self
                //     .lj_lut
                //     .get(&(self.atoms[i].element, self.atoms[j].element))
                //     .unwrap();

                let dist = r_sq.sqrt();
                let dir = diff / dist;

                // if dist < EPS {
                //     continue;
                // }

                let σ = SIGMA_FROM_RSTAR * (self.atoms[i].lj_r_star + self.atoms[j].lj_r_star);
                let ε = 0.25 * (self.atoms[i].lj_eps * self.atoms[j].lj_eps).sqrt();

                let mut f_lj = force_lj(dir, dist, σ, ε);

                let mut f_coulomb = force_coulomb(
                    dir,
                    dist,
                    self.atoms[i].partial_charge,
                    self.atoms[j].partial_charge,
                    SOFTENING_FACTOR_SQ,
                );

                if scale14 {
                    f_lj *= SCALE_LJ_14;
                    f_coulomb *= SCALE_COUL_14;
                }

                let f = f_lj + f_coulomb;

                if self.atoms[i].mass < 0.0001 {
                    println!("Mass problem: {:?}", self.atoms[i]);
                }
                if self.atoms[j].mass < 0.0001 {
                    println!("Mass problem: {:?}", self.atoms[j]);
                }

                let accel_0 = f / self.atoms[i].mass;
                let accel_1 = f / self.atoms[j].mass;

                self.atoms[i].accel += accel_0;
                self.atoms[j].accel -= accel_1;
            }
        }

        // Second pass: External atoms.
        for ai in &mut self.atoms {
            for aj in &self.atoms_external {
                let dv = self.cell.min_image(aj.posit - ai.posit);

                // let dv = Vec3::new_zero();

                let r_sq = dv.magnitude_squared();
                if r_sq > cutoff_sq {
                    continue;
                }

                // let (σ, ϵ) = self.lj_lut.get(&(ai.element, aj.element)).unwrap();
                // todo: Update this for amber.
                let (σ, ϵ) = (0., 0.);

                // todo: Instead of your LUT above, once you figure out how to load these.
                // let (σ, ϵ, _) = force_field.lj[&ai.amber_type];   // (_,_,mass)
                // let (σ2,ϵ2, _) = force_field.lj[&aj.amber_type];
                // let (σ,ϵ) = mix_lorentz_berthelot(σ,σ2, ϵ,ϵ2);    // same rule as AMBER

                let dist = r_sq.sqrt();
                let dir = dv / dist;

                if dist < EPS {
                    continue;
                }

                let f = force_lj(dir, dist, σ as f64, ϵ as f64)
                    + force_coulomb(
                        dir,
                        dist,
                        ai.partial_charge,
                        aj.partial_charge,
                        SOFTENING_FACTOR_SQ,
                    );

                // todo: Experimenting with a scaler for docking trial+error.
                let scaler = 10.;

                ai.accel += f / ai.mass * scaler;
            }
        }
    }

    /// A helper for the thermostat
    #[inline]
    fn current_kinetic_energy(&self) -> f64 {
        self.atoms
            .iter()
            .map(|a| 0.5 * a.mass * a.vel.magnitude_squared())
            .sum()
    }

    pub fn take_snapshot(&mut self) {
        self.snapshots.push(SnapshotDynamics {
            time: self.time,
            atom_posits: self.atoms.iter().map(|a| a.posit).collect(),
            atom_velocities: self.atoms.iter().map(|a| a.vel).collect(),
        })
    }
}

#[inline]
/// Mutable aliasing helpers.
fn split2_mut<T>(v: &mut [T], i: usize, j: usize) -> (&mut T, &mut T) {
    assert!(i != j);
    let (low, high) = if i < j { (i, j) } else { (j, i) };
    let (left, right) = v.split_at_mut(high);
    (&mut left[low], &mut right[0])
}

#[inline]
fn split3_mut<T>(v: &mut [T], i: usize, j: usize, k: usize) -> (&mut T, &mut T, &mut T) {
    let len = v.len();
    assert!(i < len && j < len && k < len, "index out of bounds");
    assert!(i != j && i != k && j != k, "indices must be distinct");

    // SAFETY: we just asserted that 0 <= i,j,k < v.len() and that they're all different.
    let ptr = v.as_mut_ptr();
    unsafe {
        let a = &mut *ptr.add(i);
        let b = &mut *ptr.add(j);
        let c = &mut *ptr.add(k);
        (a, b, c)
    }
}

#[inline]
pub fn split4_mut<T>(
    slice: &mut [T],
    i0: usize,
    i1: usize,
    i2: usize,
    i3: usize,
) -> (&mut T, &mut T, &mut T, &mut T) {
    // ---------- safety gates ------------------------------------------------
    let len = slice.len();
    assert!(
        i0 < len && i1 < len && i2 < len && i3 < len,
        "index out of bounds"
    );
    assert!(
        i0 != i1 && i0 != i2 && i0 != i3 && i1 != i2 && i1 != i3 && i2 != i3,
        "indices must be pair-wise distinct"
    );
    // ------------------------------------------------------------------------

    // SAFETY:
    //  * The bounds checks above guarantee each offset is within the slice.
    //  * The pair-wise distinct check guarantees no two &mut point to
    //    the same element, so we uphold Rust’s aliasing rules.
    unsafe {
        let base = slice.as_mut_ptr();
        (
            &mut *base.add(i0),
            &mut *base.add(i1),
            &mut *base.add(i2),
            &mut *base.add(i3),
        )
    }
}
