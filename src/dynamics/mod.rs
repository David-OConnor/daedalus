//! This module contains a traditional molecular dynamics approach
//!
//! [Good article](https://www.owlposting.com/p/a-primer-on-molecular-dynamics)
//! [A summary paper](https://arxiv.org/pdf/1401.1181)
//!
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

mod amber;
mod ambient;
mod prep;

use std::collections::{HashMap, HashSet};

use ambient::SimBox;
use bio_files::frcmod::{
    AngleData, BondData, DihedralData, ForceFieldParams, ImproperDihedralData, MassData, VdwData,
};
use lin_alg::f64::Vec3;
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

/// Force field parameters, e.g. from Amber. Similar to that in `bio_files`, but
/// with Hashmap-based keys (of atom-name tuples) for fast look-ups.
///
/// For descriptions of each field and the units used, reference the structs in bio_files, of which
/// this uses internally.
#[derive(Debug, Default)]
pub struct ForceFieldParamsKeyed {
    pub mass: HashMap<String, MassData>,
    pub bond: HashMap<(String, String), BondData>,
    pub angle: HashMap<(String, String, String), AngleData>,
    pub dihedral: HashMap<(String, String, String, String), DihedralData>,
    pub improper: HashMap<(String, String, String, String), ImproperDihedralData>,
    pub van_der_waals: HashMap<String, VdwData>,
    // todo: Partial charges here A/R. Note that we also assign them to the atom directly.
    pub partial_charges: HashMap<String, f32>,
}

impl ForceFieldParamsKeyed {
    pub fn new(params: &ForceFieldParams) -> Self {
        let mut result = Self::default();

        for val in &params.mass {
            result.mass.insert(val.atom_type.clone(), val.clone());
        }

        for val in &params.bond {
            result.bond.insert(val.atom_names.clone(), val.clone());
        }

        for val in &params.angle {
            result.angle.insert(val.atom_names.clone(), val.clone());
        }

        for val in &params.dihedral {
            result.dihedral.insert(val.atom_names.clone(), val.clone());
        }

        for val in &params.improper {
            result.improper.insert(val.atom_names.clone(), val.clone());
        }

        for val in &params.van_der_waals {
            result
                .van_der_waals
                .insert(val.atom_name.clone(), val.clone());
        }

        result
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
    pub bond: HashMap<(usize, usize), BondData>,
    pub angle: HashMap<(usize, usize, usize), AngleData>,
    pub dihedral: HashMap<(usize, usize, usize, usize), DihedralData>,
    pub improper: HashMap<(usize, usize, usize, usize), ImproperDihedralData>,
    pub van_der_waals: HashMap<usize, VdwData>,
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
    pub mass: f64,
    pub partial_charge: f64,
    pub force_field_type: Option<String>, // todo: Should this be an enum?
}

impl AtomDynamics {
    pub fn new(element: Element) -> Self {
        Self {
            element,
            name: String::new(),
            posit: Vec3::new_zero(),
            vel: Vec3::new_zero(),
            accel: Vec3::new_zero(),
            mass: element.atomic_weight() as f64,
            partial_charge: 0.0,
            force_field_type: None,
        }
    }
}

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
            force_field_type: None,
        }
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
    // pub bonds: Vec<BondDynamics>,
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
            a.vel += a.accel * dt_half;
            a.posit += a.vel * dt;

            // todo; Put back, but getting NaNs.
            // todo: This isn't the only NaN source...

            a.posit = self.cell.wrap(a.posit);

            // track the largest squared displacement to know when to rebuild the list
            self.max_disp_sq = self.max_disp_sq.max((a.vel * dt).magnitude_squared());
        }

        for a in &mut self.atoms {
            a.accel = Vec3::new_zero();
            // println!("Acc: {}", a.accel);
        }

        self.apply_bond_forces();
        self.apply_valence_angle_forces();
        self.apply_torsion_forces();
        self.apply_nonbonded_forces();

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
            let (ai, aj) = split2_mut(&mut self.atoms, indices.0, indices.1);

            let diff = aj.posit - ai.posit;
            let dist = diff.magnitude();

            let Δ = dist - data.r_0 as f64;
            // F = -dV/dr = -k 2Δ   (harmonic)
            // let f_mag = -2.0 * k * Δ / dist.max(1e-12);
            let f_mag = -data.k as f64 * Δ / dist.max(1e-12);
            let f = diff * f_mag;
            ai.accel -= f / ai.mass;
            aj.accel += f / aj.mass;
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
            // r_i —— r_j —— r_k   (θ is ∠ijk)
            let (ai, aj, ak) = split3_mut(&mut self.atoms, indices.0, indices.1, indices.2);

            // Bond vectors with j at the vertex
            let a = ai.posit - aj.posit; // r_ij
            let b = ak.posit - aj.posit; // r_kj (note sign!)

            let a_sq = a.magnitude_squared();
            let b_sq = b.magnitude_squared();

            // Quit early if atoms are on top of each other
            if a_sq < EPS || b_sq < EPS {
                continue;
            }

            let a_len = a_sq.sqrt();
            let b_len = b_sq.sqrt();
            let inv_ab = 1.0 / (a_len * b_len);

            // cosθ and sinθ
            let cosθ = (a.dot(b) * inv_ab).clamp(-1.0, 1.0);
            let sinθ_sq = 1.0 - cosθ * cosθ;
            if sinθ_sq < EPS {
                continue; // θ ≈ 0° or 180°, gradient ill-defined
            }
            let sinθ = sinθ_sq.sqrt();

            // Actual angle and its deviation
            let θ = cosθ.acos();
            let Δθ = θ - data.angle as f64;
            let dV_dθ = 2.0 * data.k as f64 * Δθ; // ∂V/∂θ

            // Helpers reused in both force terms
            let coef = -dV_dθ / sinθ; // −∂V/∂θ  / sinθ
            let g1 = coef * inv_ab; // common factor

            // ∂θ/∂r_i  and  ∂θ/∂r_k  (Allen & Tildesley Eq. 4.82)
            let dθ_dri = (b / b_len - a * cosθ / a_len) * g1;
            let dθ_drk = (a / a_len - b * cosθ / b_len) * g1;

            // Forces
            let f_i = -dθ_dri * dV_dθ;
            let f_k = -dθ_drk * dV_dθ;
            let f_j = -(f_i + f_k); // Newton’s third law

            // Accelerations
            ai.accel += f_i / ai.mass;
            aj.accel += f_j / aj.mass;
            ak.accel += f_k / ak.mass;
        }
    }

    /// This maintains dihedral angles. (i.e. the angle between four atoms in a sequence). This models
    /// effects such as σ-bond overlap (e.g. staggered conformations), π-conjugation, which locks certain
    /// dihedrals near 0 or τ, and steric hindrance. (Bulky groups clashing).
    fn apply_torsion_forces(&mut self) {
        for (indices, dihe) in &self.force_field_params.dihedral {
            // Split the four atoms mutably without aliasing
            let (ai, aj, ak, al) =
                split4_mut(&mut self.atoms, indices.0, indices.1, indices.2, indices.3);

            // Convenience aliases for the positions
            let r1 = ai.posit;
            let r2 = aj.posit;
            let r3 = ak.posit;
            let r4 = al.posit;

            // Bond vectors (see Allen & Tildesley, chap. 4)
            let b1 = r1 - r2; // r_ij
            let b2 = r3 - r2; // r_kj  (note the sign!)
            let b3 = r4 - r3; // r_lk

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

            // Unit normals
            let n1_hat = n1 / n1_sq.sqrt();
            let n2_hat = n2 / n2_sq.sqrt();

            // m1 = n̂1 × (b̂2)
            let m1 = n1_hat.cross(b2 / b2_len);

            // Signed dihedral (−π … π)
            let x = n1_hat.dot(n2_hat);
            let y = m1.dot(n2_hat);
            let φ = y.atan2(x);

            // dV/dφ  (note the minus sign from ∂cos / ∂φ = −sin)
            let dV_dφ = -0.5
                * dihe.barrier_height_vn as f64
                * (dihe.periodicity as f64)
                * ((dihe.periodicity as f64) * φ - dihe.gamma as f64).sin();

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
            ai.accel += f1 / ai.mass;
            aj.accel += f2 / aj.mass;
            ak.accel += f3 / ak.mass;
            al.accel += f4 / al.mass;
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

                // println!("ATOMS: {:?} {:?}", self.atoms[i].element, self.atoms[j].element);

                // todo: Put back A/R, or load from amber.
                // let (σ, ϵ) = self
                //     .lj_lut
                //     .get(&(self.atoms[i].element, self.atoms[j].element))
                //     .unwrap();

                let (σ, ϵ) = (1., 1.); // todo temp!

                let dist = r_sq.sqrt();
                let dir = diff / dist;

                // if dist < EPS {
                //     continue;
                // }

                let mut f_lj = force_lj(dir, dist, σ as f64, ϵ as f64);

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
