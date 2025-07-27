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
//! We integrate the moceule's internal rigid geometry using the `SETTLE` algorithm. This is likely
//! to be cheaper, and more robust than Shake/Rattle. It's less general, but it works here.

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

// todo: SHould you just add up the atom masses from Amber?
const MASS_WATER: f64 = 18.015_28;
const NA: f64 = 6.022_140_76e23; // todo: What is this? Used in density calc. mol⁻¹

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

// 0.997 g cm⁻³ is a good default density.
const WATER_DENSITY: f64 = 0.997;

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
        // Set up H and EP/M positions based on orientation.
        // Unit vectors defining the body frame
        let ez = orientation.rotate_vec(Vec3::new(0.0, 0.0, 1.0));
        let ex = orientation.rotate_vec(Vec3::new(1.0, 0.0, 0.0));

        // Place Hs in the plane spanned by ex, ez with the right HOH angle.
        // Let the bisector be ez, and put the hydrogens symmetrically around it.
        let half_angle = 0.5 * H_O_H_θ0; // radians
        let oh = O_H_THETA_R_0; // Å
        let h_dir0 = (ez * half_angle.cos() + ex * half_angle.sin()).to_normalized();
        let h_dir1 = (ez * half_angle.cos() - ex * half_angle.sin()).to_normalized();

        let o_pos = posit;
        let h0_pos = o_pos + h_dir0 * oh;
        let h1_pos = o_pos + h_dir1 * oh;

        // EP on the HOH bisector at fixed O–EP distance
        let ep_pos = o_pos + (h0_pos - o_pos + h1_pos - o_pos).to_normalized() * O_EP_R_0;

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
                posit: h0_pos,
                ..base.clone()
            },
            h1: AtomDynamics {
                force_field_type: String::from("HW"),
                posit: h1_pos,
                ..base.clone()
            },
            m: AtomDynamics {
                force_field_type: String::from("EP"),
                posit: ep_pos,
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
        // ---------- 1. accumulate non‑bonded forces ----------
        let mut f_o = Vec3::new_zero();
        let mut f_h0 = Vec3::new_zero();
        let mut f_h1 = Vec3::new_zero();
        let mut f_ep = Vec3::new_zero(); // ← NEW

        for src in sources {
            // oxygen
            let r = src.posit - self.o.posit;
            f_o += f_nonbonded(&self.o, src, r.dot(r), r, false);

            // hydrogens
            let r = src.posit - self.h0.posit;
            f_h0 += f_nonbonded(&self.h0, src, r.dot(r), r, false);

            let r = src.posit - self.h1.posit;
            f_h1 += f_nonbonded(&self.h1, src, r.dot(r), r, false);

            // EP (virtual) – accumulate on f_ep; will project to O later
            let r = src.posit - self.m.posit;
            f_ep += f_nonbonded(&self.m, src, r.dot(r), r, false);
        }

        f_o += f_ep; // project EP force to oxygen

        // ---------- 2. half‑kick ----------
        self.o.vel += f_o * (0.5 * dt / self.o.mass);
        self.h0.vel += f_h0 * (0.5 * dt / self.h0.mass);
        self.h1.vel += f_h1 * (0.5 * dt / self.h1.mass);

        // ---------- 3. analytic SETTLE update ----------
        settle_opc(&mut self.o, &mut self.h0, &mut self.h1, dt);

        // ---------- 4. place EP rigidly ----------
        let bis = (self.h0.posit - self.o.posit) + (self.h1.posit - self.o.posit);
        self.m.posit = self.o.posit + bis.to_normalized() * O_EP_R_0;

        // EP velocity follows COM of hydrogens
        self.m.vel = (self.h0.vel + self.h1.vel) * 0.5;

        // ---------- 5. re‑compute forces ----------
        let mut f_o2 = Vec3::new_zero();
        let mut f_h02 = Vec3::new_zero();
        let mut f_h12 = Vec3::new_zero();
        let mut f_ep2 = Vec3::new_zero();

        for src in sources {
            let r = src.posit - self.o.posit;
            f_o2 += f_nonbonded(&self.o, src, r.dot(r), r, false);
            let r = src.posit - self.h0.posit;
            f_h02 += f_nonbonded(&self.h0, src, r.dot(r), r, false);
            let r = src.posit - self.h1.posit;
            f_h12 += f_nonbonded(&self.h1, src, r.dot(r), r, false);
            let r = src.posit - self.m.posit;
            f_ep2 += f_nonbonded(&self.m, src, r.dot(r), r, false);
        }
        f_o2 += f_ep2; // project EP again

        // ---------- 6. second half‑kick ----------
        self.o.vel += f_o2 * (0.5 * dt / self.o.mass);
        self.h0.vel += f_h02 * (0.5 * dt / self.h0.mass);
        self.h1.vel += f_h12 * (0.5 * dt / self.h1.mass);
    }
}

// todo: Should we pass density, vice n_mols?
pub fn make_water_mols(cell: &SimBox, max_vel: f64) -> Vec<WaterMol> {
// pub fn make_water_mols(n_mols: usize, cell: &SimBox, max_vel: f64) -> Vec<WaterMol> {
    let vol = cell.volume();

    let n_float = WATER_DENSITY * vol* (NA / (MASS_WATER * 1.0e24));
    let n_mols  = n_float.round() as usize;  // round to nearest integer

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

/// Analytic SETTLE for 3‑site rigid water (Miyamoto & Kollman, JCC 1992).
/// Works for any bond length / HOH angle, so it’s fine for OPC.
/// Uses O as the pivot.
///
/// All distances & masses are in MD internal units (Å, ps, amu, kcal/mol).
fn settle_opc(o: &mut AtomDynamics, h0: &mut AtomDynamics, h1: &mut AtomDynamics, dt: f64) {
    // Can't use cos in a const.
    const CSOHOH: f64 = -0.2351421131025898;
    // let COSHOH: f64 = (H_O_H_θ0 * 0.5).cos() * 2.0 * (H_O_H_θ0 * 0.5).cos() - 1.0; // cos(θ)

    // -- A. half‑step drift of the oxygen (center) --------------------
    o.posit += o.vel * dt; // translate O
    h0.posit += o.vel * dt; // same COM drift for H’s
    h1.posit += o.vel * dt;

    // -- B. rotate the OH pair analytically ---------------------------
    // work in the O‑centered frame
    let mut r0 = h0.posit - o.posit;
    let mut r1 = h1.posit - o.posit;
    let v0 = h0.vel - o.vel;
    let v1 = h1.vel - o.vel;

    // Translational angular momentum L = Σ m r×v (about O)
    let l = r0.cross(v0) * H_MASS + r1.cross(v1) * H_MASS;

    // inertia tensor (about O) for two equal masses at r0, r1
    let (ixx, iyy, izz, ixy, ixz, iyz) = {
        // I = Σ m (r² δij - r_i r_j)
        let m = O_MASS;
        let r2_0 = r0.dot(r0);
        let r2_1 = r1.dot(r1);
        let xx = m * (r2_0 + r2_1 - (r0.x * r0.x + r1.x * r1.x));
        let yy = m * (r2_0 + r2_1 - (r0.y * r0.y + r1.y * r1.y));
        let zz = m * (r2_0 + r2_1 - (r0.z * r0.z + r1.z * r1.z));
        let xy = -m * (r0.x * r0.y + r1.x * r1.y);
        let xz = -m * (r0.x * r0.z + r1.x * r1.z);
        let yz = -m * (r0.y * r0.z + r1.y * r1.z);
        (xx, yy, zz, xy, xz, yz)
    };

    // Solve ω from I·ω = L  (since I is symmetric 3×3, one can
    // invert analytically or use a small 3×3 solver)

    let omega = solve_symmetric3(ixx, iyy, izz, ixy, ixz, iyz, l);

    // rotate the H’s by Δq = ω × r dt
    r0 += omega.cross(r0) * dt;
    r1 += omega.cross(r1) * dt;

    // normalise back to exact geometry
    let r0n = r0.to_normalized() * O_H_THETA_R_0;
    let r1n = r1.to_normalized() * O_H_THETA_R_0;

    // rebuild exact positions
    h0.posit = o.posit + r0n;
    h1.posit = o.posit + r1n;

    // -- C. recompute H velocities from rigid‑body motion -------------
    // v = ω × r
    h0.vel = o.vel + omega.cross(r0n);
    h1.vel = o.vel + omega.cross(r1n);
}

/// Solve I · x = b for a 3×3 *symmetric* matrix I.
/// The six unique elements are
///     [ ixx  ixy  ixz ]
/// I = [ ixy  iyy  iyz ]
///     [ ixz  iyz  izz ]
///
/// Returns x as a Vec3.  Panics if det(I) ≃ 0.
fn solve_symmetric3(ixx: f64, iyy: f64, izz: f64, ixy: f64, ixz: f64, iyz: f64, b: Vec3) -> Vec3 {
    // --- determinant -------------------------------------------------
    let det = ixx * (iyy * izz - iyz * iyz) - ixy * (ixy * izz - iyz * ixz)
        + ixz * (ixy * iyz - iyy * ixz);

    const TOL: f64 = 1.0e-12;
    assert!(det.abs() > TOL, "singular 3×3 matrix in solve_symmetric3");

    let inv_det = 1.0 / det;

    // --- adjugate / inverse elements --------------------------------
    let inv00 = (iyy * izz - iyz * iyz) * inv_det;
    let inv01 = (ixz * iyz - ixy * izz) * inv_det;
    let inv02 = (ixy * iyz - ixz * iyy) * inv_det;
    let inv11 = (ixx * izz - ixz * ixz) * inv_det;
    let inv12 = (ixz * ixy - ixx * iyz) * inv_det;
    let inv22 = (ixx * iyy - ixy * ixy) * inv_det;

    // --- x = I⁻¹ · b -------------------------------------------------
    Vec3::new(
        inv00 * b.x + inv01 * b.y + inv02 * b.z,
        inv01 * b.x + inv11 * b.y + inv12 * b.z,
        inv02 * b.x + inv12 * b.y + inv22 * b.z,
    )
}
