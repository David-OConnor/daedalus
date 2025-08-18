#![allow(non_upper_case_globals)]

//! We use the [OPC model](https://pubs.acs.org/doi/10.1021/jz501780a) for water.
//! See also, the Amber Reference Manual.
//!
//! This is a rigid model that includes an "EP" or "M" massless charge-only molecule (No LJ terms),
//! and no charge on the Oxygen. We integrate it using standard Amber-style forces.
//! Amber strongly recommends using this model when their ff19SB foces for proteins.
//!
//! Amber RM: "OPC is a non-polarizable, 4-point, 3-charge rigid water model. Geometrically, it
//! resembles TIP4P-like mod-
//! els, although the values of OPC point charges and charge-charge distances are quite different.
//! The model has a single VDW center on the oxygen nucleus."
//!
//! Note: The original paper uses the term "M" for the massless charge; Amber calls it "EP".
//!
//! We integrate the molecule's internal rigid geometry using the `SETTLE` algorithm. This is likely
//! to be cheaper, and more robust than Shake/Rattle. It's less general, but it works here.
//! Settle is specifically tailored for three-atom rigid bodies.
//!
//!
//! todo: H bond avg time: 1-20ps: Use this to validate your water model

use lin_alg::f64::{Quaternion, Vec3, X_VEC, Z_VEC};
use na_seq::Element;

use crate::dynamics::{
    ACCEL_CONVERSION, AtomDynamics, MdState, non_bonded::CHARGE_UNIT_SCALER,
    water_settle::settle_drift,
};

// Constant parameters below are for the OPC water (JPCL, 2014, 5 (21), pp 3863-3871)
// (Amber 2025, frcmod.opc) EP/M is the massless, 4th charge.
// These values are taken directly from `frcmod.opc`, in the Amber package. We have omitted
// values that are 0., or otherwise not relevant in this model. (e.g. EP mass, O charge, bonded params
// other than bond distances and the valence angle)
pub(crate) const O_MASS: f64 = 16.;
pub(crate) const H_MASS: f64 = 1.008;

// We have commented out flexible-bond parameters that are provided by Amber, but not
// used in this rigid model.

// Å; bond distance. (frcmod.opc, or Table 2.)
const O_EP_R_0: f64 = 0.159_398_33;
const O_H_R: f64 = 0.872_433_13;

// Angle bending angle, radians.
const H_O_H_θ: f64 = 1.808_161_105_066; // (103.6 degrees in frcmod.opc)
const H_O_H_θ_HALF: f64 = 0.5 * H_O_H_θ;

// For converting from R_star to eps.
const SIGMA_FACTOR: f64 = 1.122_462_048_309_373; // 2^(1/6)

// Van der Waals / JL params. Only O carries this.
const O_RSTAR: f64 = 1.777_167_268;
pub const O_SIGMA: f64 = 2.0 * O_RSTAR / SIGMA_FACTOR;
pub const O_EPS: f64 = 0.212_800_813_0;

// Partial charges. See the OPC paper, Table 2. None on O.
const Q_H: f64 = 0.6791 * CHARGE_UNIT_SCALER;
const Q_EP: f64 = -2. * Q_H;

/// Contains 4 atoms for each water molecules, at a given time step. Note that these
/// are not independent, but are useful in our general MD APIs, for compatibility with
/// non-water atoms.
///
/// Note: We currently don't use accel value on each atom directly, but use a `ForcesOnAtoms` abstraction.
///
/// Important: We repurpose the `accel` field of `AtomDynamics` to store forces instead. These differ
/// by a factor of mass.
/// todo: We may or may not change this A/R.
pub struct WaterMol {
    /// Chargeless; its charge is represented at the offset "M" or "EP".
    /// The only Lennard Jones/Vdw source. Has mass.
    pub o: AtomDynamics,
    /// Hydrogens: carries charge, but no VdW force; have mass.
    pub h0: AtomDynamics,
    pub h1: AtomDynamics,
    /// The massless, charged particle offset from O. Also known as EP.
    pub m: AtomDynamics,
}

impl WaterMol {
    pub fn new(o_pos: Vec3, vel: Vec3, orientation: Quaternion) -> Self {
        // Set up H and EP/M positions based on orientation.
        // Unit vectors defining the body frame
        let z_local = orientation.rotate_vec(Z_VEC);
        let e_local = orientation.rotate_vec(X_VEC);

        // Place Hs in the plane spanned by ex, ez with the right HOH angle.
        // Let the bisector be ez, and put the hydrogens symmetrically around it.

        let h0_dir = (z_local * H_O_H_θ_HALF.cos() + e_local * H_O_H_θ_HALF.sin()).to_normalized();
        let h1_dir = (z_local * H_O_H_θ_HALF.cos() - e_local * H_O_H_θ_HALF.sin()).to_normalized();

        let h0_pos = o_pos + h0_dir * O_H_R;
        let h1_pos = o_pos + h1_dir * O_H_R;

        // EP on the HOH bisector at fixed O–EP distance
        let ep_pos = o_pos + (h0_pos - o_pos + h1_pos - o_pos).to_normalized() * O_EP_R_0;

        let h0 = AtomDynamics {
            serial_number: 0,
            force_field_type: String::from("HW"),
            element: Element::Hydrogen,
            posit: h0_pos,
            vel,
            // This is actually force for our purposes, in the context of water molecules.
            accel: Vec3::new_zero(),
            mass: H_MASS,
            partial_charge: Q_H,
            lj_sigma: 0.,
            lj_eps: 0.,
        };

        Self {
            // Override LJ params, charge, and mass.
            o: AtomDynamics {
                force_field_type: String::from("OW"),
                posit: o_pos,
                element: Element::Oxygen,
                mass: O_MASS,
                partial_charge: 0.,
                lj_sigma: O_SIGMA,
                lj_eps: O_EPS,
                ..h0.clone()
            },
            h1: AtomDynamics {
                posit: h1_pos,
                ..h0.clone()
            },
            // Override charge and mass.
            m: AtomDynamics {
                force_field_type: String::from("EP"),
                posit: ep_pos,
                element: Element::Potassium, // Placeholder
                mass: 0.,
                partial_charge: Q_EP,
                ..h0.clone()
            },
            h0,
        }
    }

    /// Called twice each step, as part of the SETTLE algorithm, to update velocities. We don't apply velocity to M/EP,
    /// because it's massless; we rigidly place it each step based on geometry.
    /// For the second half-kick, the molecule's `accel` field must have been converted
    /// from force by dividing by mass, and contain the unit conversion from AMBER's, to our natural units.
    fn half_kick(&mut self, dt_half: f64) {
        self.o.vel += self.o.accel * dt_half;
        self.h0.vel += self.h0.accel * dt_half;
        self.h1.vel += self.h1.accel * dt_half;
    }

    /// Part of the OPC algorithm; EP/M doesn't move directly and is massless. We take into account
    /// the Coulomb force on it by applying it instead to O and H atoms.
    ///
    /// We use the accel field as a stand-in for force. This means that these values must actually
    /// be force (Not scaled by ACCEL_SCALER or mass) when this function is called.
    fn project_ep_force_to_real_sites(&mut self) {
        // Geometry in O-centered frame
        let r_O_H0 = self.h0.posit - self.o.posit;
        let r_O_H1 = self.h1.posit - self.o.posit;

        let s = r_O_H0 + r_O_H1;
        let s_norm = s.magnitude();

        if s_norm < 1e-12 {
            // Degenerate geometry: drop EP force this step
            self.m.accel = Vec3::new_zero();
            return;
        }

        // todo: If you use this approach, clarify what's force, and what's accel.

        let f_m = self.m.accel;

        // Unit bisector and projection operator P = (I - uu^T)/|s|
        let u = s / s_norm;
        let fm_parallel = u * f_m.dot(u);
        let fm_perp = f_m - fm_parallel; // (I - uu^T) f_m
        let scale = O_EP_R_0 / s_norm; // d / |s|

        // Chain rule: ∂rM/∂rO = I - 2 d P ;  ∂rM/∂rHk = d P
        // Because P is symmetric, (∂rM/∂ri)^T Fm == same expression with P acting on Fm.
        let fh = fm_perp * scale; // contribution that goes to each H
        let fo = f_m - fh * 2.0; // remaining force goes to O

        // Force on M/EP is now zero, and we've modified the forces on the other atoms from it.
        self.m.accel = Vec3::new_zero();
        self.o.accel += fo;
        self.h0.accel += fh;
        self.h1.accel += fh;
    }
}

impl MdState {
    /// Add reciprocal (PME) water-site forces into a per-water force array.
    /// Expects self.water_pme_sites_forces[iw] == [f_M, f_H0, f_H1] at current coords.
    ///
    /// Note that because this is for recip only, we don't apply force to O, because it's
    /// chargeless. (LJ only)
    // fn add_recip_to_water_forces(&self, fw: &mut [ForcesOnWaterMol]) {
    fn add_recip_to_water_forces(&mut self) {
        // todo: QC that this is set up and working.
        for (i, water_mol) in self.water.iter_mut().enumerate() {
            let [f_m, f_h0, f_h1] = self
                // todo: Hmmmm... do we want this pme_sites_forces at all?
                .water_pme_sites_forces
                .get(i)
                .copied()
                .unwrap_or([Vec3::new_zero(); 3]);

            water_mol.m.accel += f_m;
            water_mol.h0.accel += f_h0;
            water_mol.h1.accel += f_h1;
        }
    }

    /// Verlet velocity integration for water, part 1. Forces for this step must
    /// be pre-calculated. Accepts as mutable to allow projecting M/EP force onto the
    /// other atoms.
    ///
    /// In addition to the VV half-kick and drift, it handles force projection from M/EP,
    /// and applying SETTLE to main each molecul's rigid geometry.
    pub fn water_vv_first_half_and_drift(&mut self, dt: f64, dt_half: f64) {
        let cell = self.cell;

        for iw in 0..self.water.len() {
            let w = &mut self.water[iw];

            // Take the force on M/EP, and instead apply it to the other atoms. This leaves it at 0.

            // println!("Water accel: o: {} H0: {}, h1: {}", w.o.accel, w.h0.accel, w.h1.accel);

            // First half-kick. Don't apply conversions here, as they've already been applied in the
            // previous step.
            w.half_kick(dt_half);

            // Drift the rigid molecule with SETTLE
            settle_drift(
                &mut w.o,
                &mut w.h0,
                &mut w.h1,
                dt,
                &self.cell,
                &mut self.barostat.virial_pair_kcal,
            );

            // Place EP on the HOH bisector
            {
                let bisector = (w.h0.posit - w.o.posit) + (w.h1.posit - w.o.posit);
                w.m.posit = w.o.posit + bisector.to_normalized() * O_EP_R_0;
                w.m.vel = (w.h0.vel + w.h1.vel) * 0.5;
            }

            // Wrap molecule as a rigid unit (wrap O, translate H,H,EP)
            let new_o = cell.wrap(w.o.posit);
            let shift = new_o - w.o.posit;

            w.o.posit = new_o;
            w.h0.posit += shift;
            w.h1.posit += shift;
            w.m.posit += shift;
        }
    }

    /// Velocity-Verlet integration for water, part 2.
    /// Forces (as .accel) must be computed prior to this step.
    pub fn water_vv_second_half(&mut self, dt_half: f64) {
        for iw in 0..self.water.len() {
            let w = &mut self.water[iw];

            // Take the force on M/EP, and instead apply it to the other atoms. This leaves it at 0.
            // This is the only place where we need it, since we only apply forces once per step; this
            // will cover this step's second half-kick, and the first half-kick next step.
            w.project_ep_force_to_real_sites();

            // Convert forces to accel, in our native units.
            w.o.accel *= ACCEL_CONVERSION / w.o.mass;
            w.h0.accel *= ACCEL_CONVERSION / w.h0.mass;
            w.h1.accel *= ACCEL_CONVERSION / w.h1.mass;

            // Second half-kick. Apply unit and mass conversions here, as they've
            // been reset from the previous step, and re-calculated in this one.
            w.half_kick(dt_half);
        }
    }
}
