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

//! We are using f64, and CPU-only for now, unless we confirm f32 will work.
//! Maybe a mixed approach: Coordinates, velocities, and forces in 32-bit; sensitive global
//! reductions (energy, virial, integration) in 64-bit.
//!
//! We use Verlet integration. todo: Velocity verlet? Other techniques that improve and build upon it?
//!
//! Amber: ff19SB for proteins, gaff2 for ligands. (Based on recs from https://ambermd.org/AmberModels.php).
//!
//!
//! A broad list of components of this simulation
//! - Thermostat/barostat, with a way to specifify temp, pressure, water density
//! - OPC water model
//! - Cell wrapping
//! - Verlet integration (Water and non-water?)
//! - Amber parameters for mass, partial charge, VdW (via LJ), dihedral/improper, angle, bond len
//! - Optimizations for Coulomb: Ewald/PME/SPME?
//! - Optimizations for LJ: Dist cutoff for now.
//! - Amber 1-2, 1-3 exclusions, and 1-4 scaling of covalently-bonded atoms.

// todo: Integration: Consider Verlet or Langevin

//  todo: Pressure and temp variables required. Perhaps related to ICs of water?

// todo: Long-term, you will need to figure out what to run as f32 vice f64, especially
// todo for being able to run on GPU.

// Note on timescale: Generally femtosecond (-15)

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
use lin_alg::f64::Vec3;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use lin_alg::f64::{Vec3x4, f64x4};
use na_seq::Element;
use rand_distr::Distribution;

use crate::{
    dynamics::{ambient::BerendsenBarostat, water_opc::WaterMol},
    molecule::Atom,
};
pub type LjTable = HashMap<(usize, usize), (f64, f64)>;

// vacuum permittivity constant   (k_e = 1/(4π ε0))
// SI, matched with charge in e₀ & Å → kcal mol // todo: QC
const ε_0: f64 = 8.8541878128e-1;
const k_e: f32 = 332.0636; // kcal·Å /(mol·e²)
// const ε_0: f64 = 1.;

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
    pub vel: Vec3,
    pub accel: Vec3,
    /// Daltons
    /// todo: Move these 4 out of this to save memory; use from the params struct directly.
    pub mass: f64,
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
            partial_charge: atom.partial_charge.unwrap_or_default() as f64,
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
    /// In femtoseconds,
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

/// Returns the force on the atom at position 0. Negate this for the force on posit 1.
pub fn f_bond_stretching(posit_0: Vec3, posit_1: Vec3, params: &BondStretchingParams) -> Vec3 {
    let diff = posit_1 - posit_0;
    let dist_measured = diff.magnitude();

    let r_delta = dist_measured - params.r_0 as f64;

    // todo: Do I need to multiply by 2?
    // Unit check: kcal/mol/Å² * Å² = kcal/mol. (Energy).
    let f_mag = params.k_b as f64 * r_delta / dist_measured.max(1e-12);
    diff * f_mag
}

/// Returns the forces.
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
    let dV_dθ = 2.0 * params.k as f64 * Δθ; // dV/dθ

    let c = bond_vec_01.cross(bond_vec_21); // n  ∝  r_ij × r_kj
    let c_len2 = c.magnitude_squared(); // |n|^2

    let geom_i = (c.cross(bond_vec_01) * b_vec_21_len) / c_len2;
    let geom_k = (bond_vec_21.cross(c) * b_vec_01_len) / c_len2;

    let f_0 = -geom_i * dV_dθ;
    let f_2 = -geom_k * dV_dθ;
    let f_1 = -(f_0 + f_2);

    (f_0, f_1, f_2)
}
