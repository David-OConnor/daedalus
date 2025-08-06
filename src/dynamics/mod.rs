#![allow(non_snake_case)]

//! This module contains a traditional molecular dynamics approach
//!
//! [Good article](https://www.owlposting.com/p/a-primer-on-molecular-dynamics)
//! [A summary paper](https://arxiv.org/pdf/1401.1181)
//!
//! [Amber Force Fields reference](https://ambermd.org/AmberModels.php)
//! [Small molucules using GAFF2](https://ambermd.org/downloads/amber_geostd.tar.bz2)
//! [Amber RM 2025](https://ambermd.org/doc12/Amber25.pdf)
//!
//! To download .dat files (GAFF2), download Amber source (Option 2) [here](https://ambermd.org/GetAmber.php#ambertools).
//! Files are in dat -> leap -> parm
//!
//! Base units: Å, ps (10^-12),
//!
//! We are using f64, and CPU-only for now, unless we confirm f32 will work.
//! Maybe a mixed approach: Coordinates, velocities, and forces in 32-bit; sensitive global
//! reductions (energy, virial, integration) in 64-bit.
//!
//! We use Verlet integration. todo: Velocity verlet? Other techniques that improve and build upon it?
//!
//! Amber: ff19SB for proteins, gaff2 for ligands. (Based on recs from https://ambermd.org/AmberModels.php).
//!
//
//! A broad list of components of this simulation
//! - Thermostat/barostat, with a way to specifify temp, pressure, water density
//! - OPC water model
//! - Cell wrapping
//! - Verlet integration (Water and non-water?)
//! - Amber parameters for mass, partial charge, VdW (via LJ), dihedral/improper, angle, bond len
//! - Optimizations for Coulomb: Ewald/PME/SPME?
//! - Optimizations for LJ: Dist cutoff for now.
//! - Amber 1-2, 1-3 exclusions, and 1-4 scaling of covalently-bonded atoms.
//!
//! --------
//! A timing test, using bond-stretching forces between two atoms only. Measure the period
//! of oscillation for these atom combinations, e.g. using custom Mol2 files.
//! c6-c6: 35fs (correct).   os-os: 47fs        nc-nc: 34fs        hw-hw: 9fs
//! Our measurements, 2025-08-04
//! c6-c6: 35fs    os-os: 31fs        nc-nc: 34fs (Correct)       hw-hw: 6fs

// todo: Long-term, you will need to figure out what to run as f32 vice f64, especially
// todo for being able to run on GPU.

mod ambient;
mod non_bonded;
pub mod prep;
mod spme;
mod water_opc;

use std::collections::{HashMap, HashSet};

use ambient::SimBox;
use bio_files::amber_params::{
    AngleBendingParams, BondStretchingParams, DihedralParams, MassParams, VdwParams,
};
use lin_alg::f64::{Vec3, calc_dihedral_angle_v2};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use lin_alg::f64::{Vec3x4, f64x4};
use na_seq::Element;
use rand_distr::Distribution;
use rustfft::num_complex::ComplexFloat;

use crate::{
    dynamics::{ambient::BerendsenBarostat, non_bonded::CHARGE_UNIT_SCALER, water_opc::WaterMol},
    molecule::Atom,
};

// (indices), (sigma, eps)
pub type LjTable = HashMap<(usize, usize), (f64, f64)>;

// Verlet list parameters

const SKIN: f64 = 2.0; // Å – rebuild list if an atom moved >½·SKIN
const SKIN_SQ: f64 = SKIN * SKIN;

// Overrides for missing parameters.
const M_O: f64 = 15.999; // Da
const M_H: f64 = 1.008; // Da
const R_OH: f64 = 0.9572; // Å
const ANG_HOH: f64 = 104.52_f64.to_radians();

const SOFTENING_FACTOR_SQ: f64 = 1e-6;

const SNAPSHOT_RATIO: usize = 1;

const EPS: f64 = 1.0e-8;
/// Convert convert kcal mol⁻¹ Å⁻¹ (Values in the Amber parameter files) to amu Å ps⁻². Multiply all bonded
/// accelerations by this.
/// todo: Or is it 4.184e-4;?
const ACCEL_CONVERSION: f64 = 418.4;

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
#[derive(Clone, Debug, Default)]
pub struct ForceFieldParamsIndexed {
    pub mass: HashMap<usize, MassParams>,
    pub bond_stretching: HashMap<(usize, usize), BondStretchingParams>,
    pub angle: HashMap<(usize, usize, usize), AngleBendingParams>,
    pub dihedral: HashMap<(usize, usize, usize, usize), DihedralParams>,
    /// Generally only for planar hub and spoke arrangements, and always hold a planar dihedral shape.
    /// (e.g. τ/2 with symmetry 2)
    pub improper: HashMap<(usize, usize, usize, usize), DihedralParams>,

    // Dihedrals are represented in Amber params as a fourier series; this Vec indlues all matches.
    // e.g. X-ca-ca-X may be present multiple times in gaff2.dat. (Although seems to be uncommon)
    //
    // X -nh-sx-X    4    3.000         0.000          -2.000
    // X -nh-sx-X    4    0.400       180.000           3.000
    pub van_der_waals: HashMap<usize, VdwParams>,
    // pub partial_charge: HashMap<usize, f32>, // todo: A/r
}

#[derive(Debug, Default)]
pub struct SnapshotDynamics {
    pub time: f64,
    pub atom_posits: Vec<Vec3>,
    pub atom_velocities: Vec<Vec3>,
    pub water_o_posits: Vec<Vec3>,
    pub water_h0_posits: Vec<Vec3>,
    pub water_h1_posits: Vec<Vec3>,
    // For now, I believe velocities are unused, but tracked here for non-water atoms.
    // We can add water velocities if needed.
}

#[derive(Clone, Debug)]
/// A trimmed-down atom for use with molecular dynamics. Contains parameters for single-atom,
/// but we use ParametersIndex for multi-atom parameters.
pub struct AtomDynamics {
    pub serial_number: u32,
    pub force_field_type: String,
    pub element: Element,
    // pub name: String,
    pub posit: Vec3,
    /// Å / ps
    pub vel: Vec3,
    /// Å / ps²
    pub accel: Vec3,
    /// Daltons
    /// todo: Move these 4 out of this to save memory; use from the params struct directly.
    pub mass: f64,
    /// Amber charge units. This is not the elementary charge units found in amino19.lib and gaff2.dat;
    /// it's scaled by a constant.
    pub partial_charge: f64,
    /// Å
    pub lj_sigma: f64,
    /// kcal/mol
    pub lj_eps: f64,
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
            serial_number: atom.serial_number,
            element: atom.element,
            // name: atom.type_in_res.clone().unwrap_or_default(),
            posit: atom_posits[i],
            vel: Vec3::new_zero(),
            accel: Vec3::new_zero(),
            mass: ff_params.mass.get(&i).unwrap().mass as f64,
            // We get partial charge for ligands from (e.g. Amber-provided) Mol files, so we load it from the atom, vice
            // the loaded FF params. They are not in the dat or frcmod files that angle, bond-length etc params are from.
            partial_charge: CHARGE_UNIT_SCALER * atom.partial_charge.unwrap_or_default() as f64,
            lj_sigma: ff_params.van_der_waals.get(&i).unwrap().sigma as f64,
            lj_eps: ff_params.van_der_waals.get(&i).unwrap().eps as f64,
            force_field_type: ff_type,
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

#[derive(Clone, Copy, PartialEq, Default)]
pub enum MdMode {
    #[default]
    Docking,
    Peptide,
}

#[derive(Default)]
pub struct MdState {
    // todo: Update how we handle mode A/R.
    pub mode: MdMode,
    pub atoms: Vec<AtomDynamics>,
    pub adjacency_list: Vec<Vec<usize>>,
    /// Sources that affect atoms in the system, but are not themselves affected by it. E.g.
    /// in docking, this might be a rigid receptor. These are for *non-bonded* interactions (e.g. Coulomb
    /// and VDW) only.
    pub atoms_static: Vec<AtomDynamics>,
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
    /// In picoseconds.
    pub time: f64,
    pub step_count: usize, // increments.
    pub snapshots: Vec<SnapshotDynamics>,
    pub cell: SimBox,
    neighbour: Vec<Vec<usize>>, // Verlet list
    max_disp_sq: f64,           // track atom displacements²
    /// K
    temp_target: f64,
    barostat: BerendsenBarostat,
    /// Exclusions of non-bonded forces for atoms connected by 1, or 2 covalent bonds.
    /// I can't find this in the RM, but ChatGPT is confident of it, and references an Amber file
    /// called 'prmtop', which I can't find. Fishy, but we're going with it.
    nonbonded_exclusions: HashSet<(usize, usize)>,
    /// See Amber RM, sectcion 15, "1-4 Non-Bonded Interaction Scaling"
    /// These are indices of atoms separated by three consecutive bonds
    nonbonded_scaled: HashSet<(usize, usize)>,
    water: Vec<WaterMol>,
    /// We cache sigma and eps on the first step, then use it on the others. This increases
    /// memory use, and reduces CPU use.
    lj_table: LjTable,
    ///. E.g. between (dynamic atom, static receptor).
    lj_table_static: LjTable,
    /// Simpler than the other LJ table: no combinations needed, as the source is a single
    /// atom type: Water's O.
    /// todo: You could even use indices.
    lj_table_water: HashMap<usize, (f64, f64)>,
}

impl MdState {
    /// One **Velocity-Verlet** step (leap-frog style) of length `dt` is in picoseconds (10^-12),
    /// with typical values of 0.001, or 0.002ps (1 or 2fs)
    pub fn step(&mut self, dt: f64) {
        let dt_half = 0.5 * dt;

        // First half-kick (v += a dt/2) and drift (x += v dt)
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

        // Bonded forces
        self.apply_bond_stretching_forces();
        self.apply_angle_bending_forces();
        self.apply_dihedral_forces(false);
        self.apply_dihedral_forces(true);

        self.apply_nonbonded_forces();

        // Second half-kick using new accelerations, and update accelerations using the atom's mass;
        // up to this point, the accelerations have been missing that step; this is an optimization to
        // do it once at the end.
        for a in &mut self.atoms {
            // We divide by mass here, once accelerations have been computed in parts above; this
            // is an optimization to prevent dividing each accel component by it.
            a.accel = a.accel * ACCEL_CONVERSION / a.mass;
            a.vel += a.accel * dt_half;
        }

        {
            // We must include both the charge center M/EP, and the Vdw center O as a source.
            // We include the Hs as charge centers.
            let mut water_dyn = Vec::with_capacity(self.water.len() * 4);
            for water in &self.water {
                // todo: Evaluate if there's a way that avoids cloning.  Probably fine as it
                // todo doesn't scale as O^2. (Clones each water atom once per step)

                water_dyn.push(water.o.clone());
                water_dyn.push(water.m.clone());
                water_dyn.push(water.h0.clone());
                water_dyn.push(water.h1.clone());
            }

            // todo: Temporarily removed water-water interactions; getting a very slow simulation,
            // todo, and NaN propogation. Troubleshoot this later. Skipping this may be OK, compared
            // todo to not using water.
            let sources_on_water: Vec<AtomDynamics> =
                // [&self.atoms[..], &self.atoms_static[..], &water_dyn[..]].concat();
                [&self.atoms[..], &self.atoms_static[..]].concat();

            for water in &mut self.water {
                water.step(
                    dt,
                    &sources_on_water,
                    &self.cell,
                    &mut self.lj_table,
                    &mut self.lj_table_static,
                    &mut self.lj_table_water,
                );
            }
        }

        // todo: Apply the thermostat.

        // Berendsen thermostat (T coupling to target every step)
        // if let Some(tau_ps) = self.kb_berendsen {
        //     let tau = tau_ps * 1e-12;
        //     let curr_ke = self.current_kinetic_energy();
        //     let curr_t = 2.0 * curr_ke / (3.0 * self.atoms.len() as f64 * 0.0019872041); // k_B in kcal/mol
        //     let λ = (1.0 + dt / tau * (self.temp_target - curr_t) / curr_t).sqrt();
        //     for a in &mut self.atoms {
        //         a.vel *= λ;
        //     }
        // }

        self.time += dt;
        self.step_count += 1;

        // Rebuild Verlet if needed
        if self.max_disp_sq > 0.25 * SKIN_SQ {
            self.build_neighbours();
        }

        if self.step_count % SNAPSHOT_RATIO == 0 {
            self.take_snapshot();
        }
    }

    fn apply_bond_stretching_forces(&mut self) {
        for (indices, params) in &self.force_field_params.bond_stretching {
            let (a_0, a_1) = split2_mut(&mut self.atoms, indices.0, indices.1);

            let f = f_bond_stretching(a_0.posit, a_1.posit, params);

            // We divide by mass in `step`.
            a_0.accel += f;
            a_1.accel -= f;
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
    fn apply_angle_bending_forces(&mut self) {
        for (indices, params) in &self.force_field_params.angle {
            let (a_0, a_1, a_2) = split3_mut(&mut self.atoms, indices.0, indices.1, indices.2);

            let (f_0, f_1, f_2) = f_angle_bending(a_0.posit, a_1.posit, a_2.posit, params);

            // We divide by mass in `step`.
            a_0.accel += f_0;
            a_1.accel += f_1;
            a_2.accel += f_2;
        }
    }

    /// This maintains dihedral angles. (i.e. the angle between four atoms in a sequence). This models
    /// effects such as σ-bond overlap (e.g. staggered conformations), π-conjugation, which locks certain
    /// dihedrals near 0 or τ, and steric hindrance. (Bulky groups clashing).
    ///
    /// This applies both "proper" linear dihedral angles, and "improper", hub-and-spoke dihedrals. These
    /// two angles are calculated in the same way, but the covalent-bond arrangement of the 4 atoms differs.
    fn apply_dihedral_forces(&mut self, improper: bool) {
        let dihedrals = if improper {
            &self.force_field_params.improper
        } else {
            &self.force_field_params.dihedral
        };

        for (indices, dihe) in dihedrals {
            // Split the four atoms mutably without aliasing
            let (a_0, a_1, a_2, a_3) =
                split4_mut(&mut self.atoms, indices.0, indices.1, indices.2, indices.3);

            // Convenience aliases for the positions
            let r_0 = a_0.posit;
            let r_1 = a_1.posit;
            let r_2 = a_2.posit;
            let r_3 = a_3.posit;

            // Bond vectors (see Allen & Tildesley, chap. 4)
            let b1 = r_1 - r_0; // r_ij
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

            //
            // let t0_ctrl = Vec3::new(0., 0., 0.);
            // let t1_ctrl = Vec3::new(0., 1., 0.);
            // let t2_ctrl = Vec3::new(0., 1., 1.);
            // let t3_ctrl = Vec3::new(0., 0., 1.);
            //
            // let t0 = Vec3::new(43.0860, 40.1400, 24.3300);
            // let t1 = Vec3::new(43.7610, 40.2040, 25.5530);
            // let t2 = Vec3::new(42.9660, 40.0070, 26.6810);
            // let t3 = Vec3::new(41.5800, 39.7650, 26.6550);
            //
            // let test_di1 = calc_dihedral_angle_v2(&(t0, t1, t2, t3));
            // let test_di_ctrl = calc_dihedral_angle_v2(&(t0_ctrl, t1_ctrl, t2_ctrl, t3_ctrl));

            // Note: We have already divided barrier height by the integer divisor when setting up
            // the Indexed params.
            let k = dihe.barrier_height as f64;
            let per = dihe.periodicity as f64;

            let dV_dφ = if improper {
                2.0 * k * (dihe_measured - dihe.phase as f64)
            } else {
                let arg = per * dihe_measured - dihe.phase as f64;
                -k * per * arg.sin()
            };

            // if improper && a_2.force_field_type == "cc" && self.step_count < 3000 && self.step_count % 100 == 0 {
            //     let mut sats = [
            //         a_0.force_field_type.as_str(),
            //         a_1.force_field_type.as_str(),
            //         a_3.force_field_type.as_str(),    // NB: skip the hub (a_2)
            //     ];
            //     sats.sort_unstable();
            //     if sats == ["ca", "cd", "os"] {
            //     // if (a_0.force_field_type == "ca"
            //     //     && a_1.force_field_type == "cd"
            //     //     && a_2.force_field_type == "cc"
            //     //     && a_3.force_field_type == "os")
            //     //     || (a_0.force_field_type == "os"
            //     //     && a_1.force_field_type == "ca"
            //     //     && a_2.force_field_type == "cd"
            //     //     && a_3.force_field_type == "cc")
            //     // {
            //         println!(
            //             "\nPosits: {} {:.3}, {} {:.3}, {} {:.3}, {} {:.3}",
            //             a_0.force_field_type,
            //             r_0,
            //             a_1.force_field_type,
            //             r_1,
            //             a_2.force_field_type,
            //             r_2,
            //             a_3.force_field_type,
            //             r_3
            //         );
            //
            //         // println!("Test CA dihe: {test_di1:.3} ctrl: {test_di_ctrl:.3}");
            //         println!(
            //             // "{:?} - Ms raw: {dihe_measured_2:2} Ms: {:.2} exp: {:.2}/{} dV_dφ: {:.2}",
            //             "{:?} -  Ms: {:.2} exp: {:.2}/{} dV_dφ: {:.2} . Improper: {improper}",
            //             &dihe.atom_types,
            //             dihe_measured / TAU,
            //             dihe.phase / TAU as f32,
            //             dihe.periodicity,
            //             dV_dφ,
            //         );
            //     }
            // }

            // ∂φ/∂r   (see e.g. DOI 10.1016/S0021-9991(97)00040-8)
            let dφ_dr1 = -n1 * (b2_len / n1_sq);
            let dφ_dr4 = n2 * (b2_len / n2_sq);

            let dφ_dr2 =
                -n1 * (b1.dot(b2) / (b2_len * n1_sq)) + n2 * (b3.dot(b2) / (b2_len * n2_sq));

            let dφ_dr3 = -dφ_dr1 - dφ_dr2 - dφ_dr4; // Newton’s third law

            // F_i = −dV/dφ · ∂φ/∂r_i
            let f0 = -dφ_dr1 * dV_dφ;
            let f1 = -dφ_dr2 * dV_dφ;
            let f2 = -dφ_dr3 * dV_dφ;
            let f3 = -dφ_dr4 * dV_dφ;

            // todo from diagnostic: Can't find it in improper, although there's where the error is showing.
            if improper {
                // println!(
                //     "\nr0: {r_0} r1: {r_1} r2: {r_2} r3: {r_3} N1: {n1} N2: {n2} n1sq: {n1_sq} n2sq: {n2_sq} b2_len: {b2_len}"
                // );
                // println!(
                //     "DIHE: {:?}, dV_dφ: {}, k: {k}, phase: {}",
                //     dihe_measured, dV_dφ, dihe.phase as f64
                // );
                // println!(
                //     "B3dotB2: {:.3}, b1db2: {:.3}. 1: {}, 2: {}, 3: {}, 4: {}",
                //     b3.dot(b2),
                //     b1.dot(b2),
                //     dφ_dr1,
                //     dφ_dr2,
                //     dφ_dr3,
                //     dφ_dr4
                // );
            } else {
                // println!("\nNOT Improper");
                // println!("r0: {r_0} r1: {r_1} r2: {r_2} r3: {r_3} N1: {n1} N2: {n2} n1sq: {n1_sq} n2sq: {n2_sq} b2_len: {b2_len}");
                // println!("DIHE: {:?}, dV_dφ: {}, k: {k}, phase: {}", dihe_measured, dV_dφ, dihe.phase as f64);
                // println!("B3dotB2: {:.3}, b1db2: {:.3}. 1: {}, 2: {}, 3: {}, 4: {}", b3.dot(b2), b1.dot(b2), dφ_dr1, dφ_dr2, dφ_dr3, dφ_dr4);
                //
                // if r_0.x.is_nan() ||  r_1.x.is_nan() ||  r_2.x.is_nan() ||  r_3.x.is_nan() {
                //     panic!("NaN. a0: {a_0:?}, a1: {a_1:?}, a2: {a_2:?}, a3: {a_3:?}");
                // }
            }

            // We divide by mass in `step`.
            a_0.accel += f0;
            a_1.accel += f1;
            a_2.accel += f2;
            a_3.accel += f3;
        }
    }

    /// A helper for the thermostat
    fn current_kinetic_energy(&self) -> f64 {
        self.atoms
            .iter()
            .map(|a| 0.5 * a.mass * a.vel.magnitude_squared())
            .sum()
    }

    pub fn take_snapshot(&mut self) {
        let mut water_o_posits = Vec::with_capacity(self.water.len());
        let mut water_h0_posits = Vec::with_capacity(self.water.len());
        let mut water_h1_posits = Vec::with_capacity(self.water.len());

        for water in &self.water {
            water_o_posits.push(water.o.posit);
            water_h0_posits.push(water.h0.posit);
            water_h1_posits.push(water.h1.posit);
        }

        self.snapshots.push(SnapshotDynamics {
            time: self.time,
            atom_posits: self.atoms.iter().map(|a| a.posit).collect(),
            atom_velocities: self.atoms.iter().map(|a| a.vel).collect(),
            water_o_posits,
            water_h0_posits,
            water_h1_posits,
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
    // Safety gates
    let len = slice.len();
    assert!(
        i0 < len && i1 < len && i2 < len && i3 < len,
        "index out of bounds"
    );
    assert!(
        i0 != i1 && i0 != i2 && i0 != i3 && i1 != i2 && i1 != i3 && i2 != i3,
        "indices must be pair-wise distinct"
    );

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

/// Returns the force on the atom at position 0. Negate this for the force on posit 1.
pub fn f_bond_stretching(posit_0: Vec3, posit_1: Vec3, params: &BondStretchingParams) -> Vec3 {
    let diff = posit_1 - posit_0;
    let r_meas = diff.magnitude();

    let r_delta = r_meas - params.r_0 as f64;

    // Note: We include the factor of 2x k_b when setting up indexed parameters.
    // Unit check: kcal/mol/Å² * Å² = kcal/mol. (Energy).
    let f_mag = params.k_b as f64 * r_delta / r_meas.max(EPS);
    diff * f_mag
}

/// Valence angle; angle between 3 atoms.
pub fn f_angle_bending(
    posit_0: Vec3,
    posit_1: Vec3,
    posit_2: Vec3,
    params: &AngleBendingParams,
) -> (Vec3, Vec3, Vec3) {
    // Bond vectors with atom 1 at the vertex.
    let bond_vec_01 = posit_0 - posit_1;
    let bond_vec_21 = posit_2 - posit_1;

    let b_vec_01_sq = bond_vec_01.magnitude_squared();
    let b_vec_21_sq = bond_vec_21.magnitude_squared();

    // Quit early if atoms are on top of each other
    if b_vec_01_sq < EPS || b_vec_21_sq < EPS {
        return (Vec3::new_zero(), Vec3::new_zero(), Vec3::new_zero());
    }

    let b_vec_01_len = b_vec_01_sq.sqrt();
    let b_vec_21_len = b_vec_21_sq.sqrt();

    let inv_ab = 1.0 / (b_vec_01_len * b_vec_21_len);

    let cos_θ = (bond_vec_01.dot(bond_vec_21) * inv_ab).clamp(-1.0, 1.0);
    let sin_θ_sq = 1.0 - cos_θ * cos_θ;

    if sin_θ_sq < EPS {
        // θ = 0 or τ; gradient ill-defined
        return (Vec3::new_zero(), Vec3::new_zero(), Vec3::new_zero());
    }

    // Measured angle, and its deviation from the parameter angle.
    let θ = cos_θ.acos();
    let Δθ = params.theta_0 as f64 - θ;
    // Note: We include the factor of 2x k when setting up indexed parameters.
    let dV_dθ = params.k as f64 * Δθ; // dV/dθ

    let c = bond_vec_01.cross(bond_vec_21); // n  ∝  r_ij × r_kj
    let c_len2 = c.magnitude_squared(); // |n|^2

    let geom_i = (c.cross(bond_vec_01) * b_vec_21_len) / c_len2;
    let geom_k = (bond_vec_21.cross(c) * b_vec_01_len) / c_len2;

    let f_0 = -geom_i * dV_dθ;
    let f_2 = -geom_k * dV_dθ;
    let f_1 = -(f_0 + f_2);

    (f_0, f_1, f_2)
}
