//! We use the [OPC model](https://pubs.acs.org/doi/10.1021/jz501780a) for water.
//! See also, the Amber Rerference Manual. (todo: Specific ref)
//!
//! This is a flexible-molecule model that includes a "EP" massless charge-only molecule,
//! and no charge on the Oxygen. We integrate it using standard Amber-style forces.
//! Amber strongly recommends using this model when their  ff19SB foces for proteins.
//!
//! Amber RM: "OPC is a non-polarizable, 4-point, 3-charge rigid water model. Geometrically, it resembles TIP4P-like mod-
//! els, although the values of OPC point charges and charge-charge distances are quite different.
//! The model has a single VDW center on the oxygen nucleus."
//!
//! Note: The original paper uses the term "M" for the massless charge; Amber calls it "EP".
//!
//! todo: Look into "SHAKE/RATTLE/SETTLE". I believe OPC is still rigid, and should
//! todo use this instead of your normal flexible MD setup?

use std::f64::consts::TAU;

use lin_alg::f64::{Quaternion, Vec3};
use na_seq::Element;
use rand::{Rng, distr::Uniform};

use crate::dynamics::{AtomDynamics, ambient::SimBox, f_nonbonded};

// Parameters for OPC water (JPCL, 2014, 5 (21), pp 3863-3871)
// (Amber 2025, frcmod.opc) EPC is presumably the massless, 4th charge.
// These values are taken directly from `frcmod.opc`, in the Amber package.
const O_MASS: f64 = 16.;
const H_MASS: f64 = 1.008;
const EP_MASS: f64 = 0.; // todo: Rem A/R.

// Bond stretching; Same K_b for all three bonds.
const K_B: f64 = 553.0; // kcal/mol/Å^2

// Å; bond distance. (frcmod.opc, or Table 2.)
const O_EP_R_0: f64 = 0.15939833;
const O_H_THETA_R_0: f64 = 0.87243313;
const H_H_THETA_R_0: f64 = 1.37120510;

// Angle Bending constant, kcal/mol/rad^2
const H_O_EP_K: f64 = 0.;
const H_O_H_K: f64 = 100.;
const H_H_O_K: f64 = 0.;

// Angle bending angle, radians.
const H_O_EP_θ0: f64 = 2.0943951023931953;
const H_O_H_θ0: f64 = 1.8081611050661253;
const H_H_O_θ0: f64 = 2.2294835864975564;

// Van der Waals / JL params. Note that only O carries a VdW force.
const O_RSTAR: f64 = 1.777167268;
const H_RSTAR: f64 = 0.;
const EP_RSTAR: f64 = 1.; // todo: Why is this 1 in the param file?

const O_SIGMA: f64 = 2.0 * O_RSTAR / SIGMA_FACTOR;
const H_SIGMA: f64 = 0.;
// Note: EP_RSTAR is 1. in the Amber param file, but I don't think this matters if ε is 0.
const EP_SIGMA: f64 = 0.;
// const H_SIGMA: f64 = 2.0 * H_RSTAR / SIGMA_FACTOR;
// const EP_SIGMA: f64 = 2.0 * EP_RSTAR / SIGMA_FACTOR;

const O_EPS: f64 = 0.2128008130;
const H_EPS: f64 = 0.;
const EP_EPS: f64 = 0.;

// For converting from R_star to eps.
const SIGMA_FACTOR: f64 = 1.122_462_048_309_373; // 2^(1/6)

// Partial charges. See the OPC paper, Table 2.
const Q_O: f64 = 0.;
const Q_H: f64 = 0.6791;
const Q_EP: f64 = -2. * Q_H;

const SHAKE_TOL2: f64 = 1e-10; // (Å²) |Δr²| convergence
const SHAKE_MAX_ITERS: usize = 20;

/// Contains absolute positions of each atom for a single molecule, at a given time step.
/// todo: Should we store as O position, and orientation quaternion instead?
/// todo: Should we just use `atom_dynamics` instead?
pub struct WaterMol {
    /// Chargeless; its charge is represented at the offset "M" or "EP".
    pub o: AtomDynamics,
    pub h0: AtomDynamics,
    pub h1: AtomDynamics,
    // todo: is this called "M", or "EP"? Have seen both.
    /// The massless, charged particle offset from O.
    pub m: AtomDynamics,
}

impl WaterMol {
    pub fn new(posit: Vec3, vel: Vec3, orientation: Quaternion) -> Self {
        // todo: Take orientation into account

        let base = AtomDynamics {
            serial_number: 0,
            force_field_type: String::new(),
            element: Element::Hydrogen,
            posit,
            vel,
            accel: Vec3::new_zero(),
            mass: H_MASS,
            partial_charge: Q_H,
            lj_sigma: H_SIGMA,
            lj_eps: H_EPS,
        };

        // todo: Make sure you're populating LJ sigma and eps.

        Self {
            o: AtomDynamics {
                force_field_type: String::from("OW"),
                element: Element::Oxygen,
                mass: O_MASS,
                partial_charge: Q_O, // 0
                lj_sigma: O_SIGMA,
                lj_eps: O_EPS,
                ..base.clone()
            },
            h0: AtomDynamics {
                force_field_type: String::from("HW"),
                ..base.clone()
            },
            h1: AtomDynamics { ..base.clone() },
            m: AtomDynamics {
                force_field_type: String::from("EP"),
                mass: 0.,
                partial_charge: Q_EP,
                lj_sigma: EP_SIGMA,
                lj_eps: EP_EPS,
                ..base.clone()
            },
        }
    }

    /// todo: Shake/rattle technique to update?
    /// Update dynamics based on own velocity, internal "forces" (?), and external coulomb (i.e.
    /// partial charge), and Van der Waals (LJ) forces from other molecules.
    /// One Velocity‑Verlet step with SHAKE/RATTLE constraints.
    ///
    /// `sources` includes both other water molecules, and non-waters.
    pub fn step(&mut self, dt: f64, sources: &[AtomDynamics]) {
        // External non‑bonded forces. Note that we don't track force
        // on M, as it has no mass.
        let mut f_o = Vec3::new_zero();
        let mut f_h0 = Vec3::new_zero();
        let mut f_h1 = Vec3::new_zero();

        for src in sources {
            let r_sq = (src.posit - self.o.posit).magnitude_squared();
            let diff = src.posit - self.o.posit;
            f_o += f_nonbonded(&self.o, src, r_sq, diff, false);

            let r_sq = (src.posit - self.h0.posit).magnitude_squared();
            let diff = src.posit - self.h0.posit;
            f_h0 += f_nonbonded(&self.h0, src, r_sq, diff, false);

            let r_sq = (src.posit - self.h1.posit).magnitude_squared();
            let diff = src.posit - self.h1.posit;
            f_h1 += f_nonbonded(&self.h1, src, r_sq, diff, false);
        }

        // Half‑kick -----------------------------------------------------
        self.o.vel += f_o * (0.5 * dt / self.o.mass);
        self.h0.vel += f_h0 * (0.5 * dt / self.h0.mass);
        self.h1.vel += f_h1 * (0.5 * dt / self.h1.mass);

        // Save old positions for RATTLE velocity update
        let o_old = self.o.posit;
        let h0_old = self.h0.posit;
        let h1_old = self.h1.posit;

        // Drift ---------------------------------------------------------
        self.o.posit += self.o.vel * dt;
        self.h0.posit += self.h0.vel * dt;
        self.h1.posit += self.h1.vel * dt;

        // SHAKE/SETTLE --------------------------------------------------
        self.settle();

        // --- 5. RATTLE: recompute constrained velocities ----------------------
        self.o.vel = (self.o.posit - o_old) / dt;
        self.h0.vel = (self.h0.posit - h0_old) / dt;
        self.h1.vel = (self.h1.posit - h1_old) / dt;

        // EP has no mass; velocity follows from position
        self.m.vel = (self.h0.vel + self.h1.vel) * 0.5;

        // Recompute forces & second half‑kick ---------------------------
        // todo: What?
        // (omit here if you reuse f_*; otherwise call force routine again)
        // self.o.vel  += f_o  * (0.5 * dt / self.o.mass);
        // self.h0.vel += f_h0 * (0.5 * dt / self.h0.mass);
        // self.h1.vel += f_h1 * (0.5 * dt / self.h1.mass);
    }

    fn settle(&mut self) {
        for _ in 0..SHAKE_MAX_ITERS {
            let mut max_err: f64 = 0.0;
            max_err =
                max_err.max(constrain_distance(&mut self.o, &mut self.h0, O_H_THETA_R_0).abs());
            max_err =
                max_err.max(constrain_distance(&mut self.o, &mut self.h1, O_H_THETA_R_0).abs());
            max_err =
                max_err.max(constrain_distance(&mut self.h0, &mut self.h1, H_H_THETA_R_0).abs());
            if max_err < SHAKE_TOL2 {
                break;
            }
        }

        // Place the EP point on the HOH bisector at the fixed O‑EP distance.
        let bis = (self.h0.posit - self.o.posit) + (self.h1.posit - self.o.posit);
        self.m.posit = self.o.posit + bis.to_normalized() * O_EP_R_0;
    }
}

pub fn make_water_mols(n_mols: usize, cell: &SimBox, max_vel: f64) -> Vec<WaterMol> {
    let mut result = Vec::with_capacity(n_mols);
    let mut rng = rand::rng();

    let uni01 = Uniform::<f64>::new(0.0, 1.0).unwrap();
    let uni11 = Uniform::<f64>::new(-1.0, 1.0).unwrap();

    for _ in 0..n_mols {
        /* ---------- position (axis‑aligned box) ---------- */
        let posit = Vec3::new(
            rng.sample(uni01) * (cell.bounds_high.x - cell.bounds_low.x) + cell.bounds_low.x,
            rng.sample(uni01) * (cell.bounds_high.y - cell.bounds_low.y) + cell.bounds_low.y,
            rng.sample(uni01) * (cell.bounds_high.z - cell.bounds_low.z) + cell.bounds_low.z,
        );

        /* ---------- velocity ---------- */
        // Direction: Marsaglia (1972) — rejection‑free, truly uniform on S²
        let (vx, vy, vz) = {
            let (r1, r2): (f64, f64) = (rng.sample(uni11), rng.sample(uni11));
            let s = r1 * r1 + r2 * r2;
            // Guaranteed s ∈ [0,2], so no loop needed.
            let factor = (1.0 - s / 2.0).sqrt();
            (
                2.0 * r1 * factor,
                2.0 * r2 * factor,
                1.0 - s, // z
            )
        };

        let speed = rng.sample(uni01) * max_vel;
        let vel = Vec3::new(vx, vy, vz) * speed;

        /* ---------- orientation (uniform SO(3)) ---------- */
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

        result.push(WaterMol::new(posit, vel, q));
    }

    result
}

fn constrain_distance(a: &mut AtomDynamics, b: &mut AtomDynamics, target: f64) -> f64 {
    let mut r = b.posit - a.posit;
    let dist2 = r.dot(r);
    let diff = dist2 - target * target;
    if diff.abs() < SHAKE_TOL2 {
        return diff;
    };

    let inv_ma = 1. / a.mass;
    let inv_mb = 1. / b.mass;

    let g = 1.0 / (inv_ma + inv_mb);
    // λ from SHAKE
    let lambda = -g * diff / r.dot(r);
    let corr = r * lambda;
    a.posit -= corr * inv_ma;
    b.posit += corr * inv_mb;
    diff
}
