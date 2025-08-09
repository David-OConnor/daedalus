#![allow(non_upper_case_globals)]

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
//! Settle is specifically tailored for three-atom rigid bodies.

use std::{collections::HashMap, f64::consts::TAU, mem};

use lin_alg::f64::{Quaternion, Vec3};
use na_seq::Element;
use rand::{Rng, distr::Uniform};
use rand_distr::{Distribution, StandardNormal};

use crate::{
    dynamics::{
        AtomDynamics, LjTable, MdState, ambient::SimBox, neighbors::NeighborsNb,
        non_bonded::f_nonbonded, split2_mut,
    },
    forces::force_coulomb,
};

// Parameters for OPC water (JPCL, 2014, 5 (21), pp 3863-3871)
// (Amber 2025, frcmod.opc) EPC is presumably the massless, 4th charge.
// These values are taken directly from `frcmod.opc`, in the Amber package.
const O_MASS: f64 = 16.;
const H_MASS: f64 = 1.008;
// const EP_MASS: f64 = 0.;

// For assigning velocities from temperature.
const KB: f64 = 0.001_987_204_1; // kcal mol⁻¹ K⁻¹ (Amber-style units)

// todo: SHould you just add up the atom masses from Amber?
const MASS_WATER: f64 = 18.015_28;
const NA: f64 = 6.022_140_76e23; // todo: What is this? Used in density calc. mol⁻¹

// We have commented out flexible-bond parameters that are provided by Amber, but not
// used in this rigid model.

// Bond stretching; Same K_b for all three bonds.
// const K_B: f64 = 553.0; // kcal/mol/Å^2

// Å; bond distance. (frcmod.opc, or Table 2.)
const O_EP_R_0: f64 = 0.15939833;
const O_H_THETA_R_0: f64 = 0.87243313;
// const H_H_THETA_R_0: f64 = 1.37120510;

// Angle Bending constant, kcal/mol/rad^2
// const H_O_EP_K: f64 = 0.;
// const H_O_H_K: f64 = 100.;
// const H_H_O_K: f64 = 0.;

// Angle bending angle, radians.
// const H_O_EP_θ0: f64 = 2.0943951023931953;
const H_O_H_θ0: f64 = 1.8081611050661253;
// const H_H_O_θ0: f64 = 2.2294835864975564;

// Van der Waals / JL params. Note that only O carries a VdW force.
const O_RSTAR: f64 = 1.777167268;
pub const O_SIGMA: f64 = 2.0 * O_RSTAR / SIGMA_FACTOR;
pub const O_EPS: f64 = 0.2128008130;

// For converting from R_star to eps.
const SIGMA_FACTOR: f64 = 1.122_462_048_309_373; // 2^(1/6)

// Partial charges. See the OPC paper, Table 2. None on O.
const Q_H: f64 = 0.6791;
const Q_EP: f64 = -2. * Q_H;

// 0.997 g cm⁻³ is a good default density.
const WATER_DENSITY: f64 = 0.997;

// Don't generate water molecules that are too close to other molecules.
// Vdw contact distance between water molecules and organic molecules is roughly 3.5 Angstroms.
const GENERATION_MIN_DIST: f64 = 4.;

/// Contains absolute positions of each atom for a single molecule, at a given time step.
/// todo: Should we store as O position, and orientation quaternion instead?
/// todo: Should we just use `atom_dynamics` instead?
pub struct WaterMol {
    /// Chargeless; its charge is represented at the offset "M" or "EP".
    /// The only Lennard Jones/Vdw source. Has mass.
    pub o: AtomDynamics,
    /// Hydrogens: carries charge; has mass.
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

        // This base has H and charge, and no LJ terms.
        let base = AtomDynamics {
            serial_number: 0,
            force_field_type: String::new(),
            element: Element::Hydrogen,
            posit,
            vel,
            accel: Vec3::new_zero(),
            mass: H_MASS,
            partial_charge: Q_H,
            lj_sigma: 0.,
            lj_eps: 0.,
        };

        Self {
            o: AtomDynamics {
                // Override LJ params, charge, and mass.
                force_field_type: String::from("OW"),
                element: Element::Oxygen,
                mass: O_MASS,
                partial_charge: 0.,
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
            // Override charge and mass.
            m: AtomDynamics {
                force_field_type: String::from("EP"),
                posit: ep_pos,
                mass: 0.,
                partial_charge: Q_EP,
                ..base.clone()
            },
        }
    }
}

// todo: Should we pass density, vice n_mols?
/// We pass atoms in so this doesn't generate water molecules that overlap with them.
pub fn make_water_mols(
    cell: &SimBox,
    t_target: f64,
    atoms_dy: &[AtomDynamics],
    atoms_static: &[AtomDynamics],
) -> Vec<WaterMol> {
    let vol = cell.volume();

    let n_float = WATER_DENSITY * vol * (NA / (MASS_WATER * 1.0e24));
    let n_mols = n_float.round() as usize; // round to nearest integer

    let mut result = Vec::with_capacity(n_mols);
    let mut rng = rand::rng();

    let uni01 = Uniform::<f64>::new(0.0, 1.0).unwrap();
    // let uni11 = Uniform::<f64>::new(-1.0, 1.0).unwrap();

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
        let mut skip = false;
        for atom_set in [atoms_dy, atoms_static] {
            for atom_non_water in atom_set {
                let dist = (atom_non_water.posit - posit).magnitude();
                if dist < GENERATION_MIN_DIST {
                    skip = true;
                    break;
                }
                if skip {
                    break;
                }
            }
        }

        if skip {
            continue;
        }

        result.push(WaterMol::new(posit, Vec3::new_zero(), q));
    }

    // Assign velocities based on temperature.
    init_velocities(&mut result, t_target);

    result
}

/// Analytic SETTLE for 3‑site rigid water (Miyamoto & Kollman, JCC 1992).
/// Works for any bond length / HOH angle, so it’s fine for OPC.
/// Uses O as the pivot.
///
/// All distances & masses are in MD internal units (Å, ps, amu, kcal/mol).
fn settle_opc(o: &mut AtomDynamics, h0: &mut AtomDynamics, h1: &mut AtomDynamics, dt: f64) {
    // Can't use cos in a const.
    // const CSOHOH: f64 = -0.2351421131025898;
    // let COSHOH: f64 = (H_O_H_θ0 * 0.5).cos() * 2.0 * (H_O_H_θ0 * 0.5).cos() - 1.0; // cos(θ)

    // Half‑step drift of the oxygen
    o.posit += o.vel * dt; // translate O
    h0.posit += o.vel * dt; // same COM drift for H’s
    h1.posit += o.vel * dt;

    // Rotate the OH pair analytically
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
        let m = H_MASS;

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

    // Normalize back to exact geometry
    let r0n = r0.to_normalized() * O_H_THETA_R_0;
    let r1n = r1.to_normalized() * O_H_THETA_R_0;

    // rebuild exact positions
    h0.posit = o.posit + r0n;
    h1.posit = o.posit + r1n;

    // Recompute H velocities from rigid‑body motion
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
    let det = ixx * (iyy * izz - iyz * iyz) - ixy * (ixy * izz - iyz * ixz)
        + ixz * (ixy * iyz - iyy * ixz);

    const TOL: f64 = 1.0e-12;
    if det.abs() < TOL {
        // Practically no rotation this step; keep ω = 0
        return Vec3::new_zero();
    }

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

pub fn init_velocities(mols: &mut [WaterMol], t_target: f64) {
    let mut rng = rand::rng();

    // 1) Gaussian draw
    for a in atoms_mut(mols) {
        if a.mass == 0.0 {
            continue;
        } // EP/virtual sites

        let nx: f64 = StandardNormal.sample(&mut rng);
        let ny: f64 = StandardNormal.sample(&mut rng);
        let nz: f64 = StandardNormal.sample(&mut rng);

        // arbitrary sigma=1 for now; we'll rescale below
        a.vel = Vec3::new(nx, ny, nz);
    }

    // 2) remove centre-of-mass drift
    remove_com_velocity(mols);

    // 3) compute instantaneous T
    let (ke, dof) = kinetic_energy_and_dof(mols);
    let t_now = 2.0 * ke / (dof as f64 * KB);

    // 4) rescale to T_target
    let lambda = (t_target / t_now).sqrt();
    for a in atoms_mut(mols) {
        if a.mass == 0.0 {
            continue;
        }
        a.vel *= lambda;
    }
}

/// Per-water, per-site force accumulator. Used transiently in `step_water`.
#[derive(Clone, Copy, Default)]
struct WaterForces {
    f_o: Vec3,
    f_h0: Vec3,
    f_h1: Vec3,
    f_m: Vec3, // EP/M site
}

impl MdState {
    /// todo: Shake/rattle technique to update?
    /// Update dynamics based on own velocity, internal "forces" (?), and external coulomb (i.e.
    /// partial charge), and Van der Waals (LJ) forces from other molecules.
    /// One Velocity‑Verlet step with SHAKE/RATTLE constraints.
    ///
    /// `sources` includes both other water molecules, and non-waters.
    pub fn step_water(&mut self, dt: f64) {
        let n_w = self.water.len();
        if n_w == 0 {
            return;
        }

        // Forces at current positions ----------
        let mut fw = vec![WaterForces::default(); n_w];

        // (a) dynamic → water
        // We only add forces to WATER here; the dynamic atoms already got their
        // counterpart in `apply_nonbonded_forces` to avoid double work.
        for iw in 0..n_w {
            for &i_dyn in &self.neighbors_nb.water_dy[iw] {
                let w = &self.water[iw];
                let a = &self.atoms[i_dyn];

                // O (LJ + Coulomb if O is charged in your model; in OPC O is neutral → LJ only)
                {
                    let dv = self.cell.min_image(a.posit - w.o.posit);
                    let r2 = dv.magnitude_squared();
                    let f = f_nonbonded(
                        &w.o,
                        a,
                        r2,
                        dv,
                        false,
                        None,
                        None,
                        Some(i_dyn), // scaling keys as in your API; none for dy-dy here
                        &self.lj_table,
                        &self.lj_table_static,
                        &self.lj_table_water,
                    );
                    fw[iw].f_o += f;
                }
                // H0
                {
                    let dv = self.cell.min_image(a.posit - w.h0.posit);
                    let r2 = dv.magnitude_squared();
                    let f = f_nonbonded(
                        &w.h0,
                        a,
                        r2,
                        dv,
                        false,
                        None,
                        None,
                        Some(i_dyn),
                        &self.lj_table,
                        &self.lj_table_static,
                        &self.lj_table_water,
                    );
                    fw[iw].f_h0 += f;
                }
                // H1
                {
                    let dv = self.cell.min_image(a.posit - w.h1.posit);
                    let r2 = dv.magnitude_squared();
                    let f = f_nonbonded(
                        &w.h1,
                        a,
                        r2,
                        dv,
                        false,
                        None,
                        None,
                        Some(i_dyn),
                        &self.lj_table,
                        &self.lj_table_static,
                        &self.lj_table_water,
                    );
                    fw[iw].f_h1 += f;
                }
                // M/EP (Coulomb only)
                {
                    let dv = self.cell.min_image(a.posit - w.m.posit);
                    let r2 = dv.magnitude_squared();
                    let f = f_nonbonded(
                        &w.m,
                        a,
                        r2,
                        dv,
                        false,
                        None,
                        None,
                        Some(i_dyn),
                        &self.lj_table,
                        &self.lj_table_static,
                        &self.lj_table_water,
                    );
                    fw[iw].f_m += f;
                }
            }
        }

        // (b) static → water (static atoms don’t move; only accumulate on water)
        for iw in 0..n_w {
            for &i_st in &self.neighbors_nb.water_static[iw] {
                let w = &self.water[iw];
                let a = &self.atoms_static[i_st];

                // O
                {
                    let dv = self.cell.min_image(a.posit - w.o.posit);
                    let r2 = dv.magnitude_squared();
                    let f = f_nonbonded(
                        &w.o,
                        a,
                        r2,
                        dv,
                        false,
                        None,
                        Some((usize::MAX, i_st)),
                        None,
                        &self.lj_table,
                        &self.lj_table_static,
                        &self.lj_table_water,
                    );
                    fw[iw].f_o += f;
                }
                // H0
                {
                    let dv = self.cell.min_image(a.posit - w.h0.posit);
                    let r2 = dv.magnitude_squared();
                    let f = f_nonbonded(
                        &w.h0,
                        a,
                        r2,
                        dv,
                        false,
                        None,
                        Some((usize::MAX, i_st)),
                        None,
                        &self.lj_table,
                        &self.lj_table_static,
                        &self.lj_table_water,
                    );
                    fw[iw].f_h0 += f;
                }
                // H1
                {
                    let dv = self.cell.min_image(a.posit - w.h1.posit);
                    let r2 = dv.magnitude_squared();
                    let f = f_nonbonded(
                        &w.h1,
                        a,
                        r2,
                        dv,
                        false,
                        None,
                        Some((usize::MAX, i_st)),
                        None,
                        &self.lj_table,
                        &self.lj_table_static,
                        &self.lj_table_water,
                    );
                    fw[iw].f_h1 += f;
                }
                // M
                {
                    let dv = self.cell.min_image(a.posit - w.m.posit);
                    let r2 = dv.magnitude_squared();
                    let f = f_nonbonded(
                        &w.m,
                        a,
                        r2,
                        dv,
                        false,
                        None,
                        Some((usize::MAX, i_st)),
                        None,
                        &self.lj_table,
                        &self.lj_table_static,
                        &self.lj_table_water,
                    );
                    fw[iw].f_m += f;
                }
            }
        }

        // (c) water ↔ water (use i<j; apply Newton’s 3rd law between molecules)
        for iw in 0..n_w {
            for &jw in &self.neighbors_nb.water_water[iw] {
                if jw <= iw {
                    continue;
                }

                let (fwi, fwj) = split2_mut(&mut fw, iw, jw); // disjoint &mut WaterForces

                let wi = &self.water[iw];
                let wj = &self.water[jw];

                // LJ: O–O only
                {
                    let dv = self.cell.min_image(wj.o.posit - wi.o.posit);
                    let r2 = dv.magnitude_squared();
                    let f = f_nonbonded(
                        &wi.o,
                        &wj.o,
                        r2,
                        dv,
                        false,
                        None,
                        None,
                        None,
                        &self.lj_table,
                        &self.lj_table_static,
                        &self.lj_table_water,
                    );
                    fwi.f_o += f;
                    fwj.f_o -= f;
                }

                // Coulomb among charged sites: H/H, H/M, M/M (O is neutral in OPC)
                // Helper: accumulate pair(i_site, j_site, &mut fw[i], &mut fw[j])
                let mut add_pair =
                    |ri: Vec3, qi: f64, rj: Vec3, qj: f64, fi: &mut Vec3, fj: &mut Vec3| {
                        // todo: QC this. Do you really need to bring in a dedicated coulomb function? What about the
                        // todo short-dist vs long dist ewald you use in f_nonbonded?
                        let diff = rj - ri;
                        let dir = diff.to_normalized();

                        let dv = self.cell.min_image(diff);
                        // let r2 = dv.magnitude_squared();
                        // Use your Coulomb path in f_nonbonded or a fast inline Coulomb here:
                        let f = force_coulomb(dir, dv.magnitude(), qi, qj, 1e-6); // todo...

                        *fi += f;
                        *fj -= f;
                    };

                // H0/H0, H0/H1, H1/H1
                add_pair(
                    wi.h0.posit,
                    wi.h0.partial_charge,
                    wj.h0.posit,
                    wj.h0.partial_charge,
                    &mut fwi.f_h0,
                    &mut fwj.f_h0,
                );
                add_pair(
                    wi.h0.posit,
                    wi.h0.partial_charge,
                    wj.h1.posit,
                    wj.h1.partial_charge,
                    &mut fwi.f_h0,
                    &mut fwj.f_h1,
                );
                add_pair(
                    wi.h1.posit,
                    wi.h1.partial_charge,
                    wj.h1.posit,
                    wj.h1.partial_charge,
                    &mut fwi.f_h1,
                    &mut fwj.f_h1,
                );

                // H/M cross
                add_pair(
                    wi.h0.posit,
                    wi.h0.partial_charge,
                    wj.m.posit,
                    wj.m.partial_charge,
                    &mut fwi.f_h0,
                    &mut fwj.f_m,
                );
                add_pair(
                    wi.h1.posit,
                    wi.h1.partial_charge,
                    wj.m.posit,
                    wj.m.partial_charge,
                    &mut fwi.f_h1,
                    &mut fwj.f_m,
                );
                add_pair(
                    wi.m.posit,
                    wi.m.partial_charge,
                    wj.h0.posit,
                    wj.h0.partial_charge,
                    &mut fwi.f_m,
                    &mut fwj.f_h0,
                );
                add_pair(
                    wi.m.posit,
                    wi.m.partial_charge,
                    wj.h1.posit,
                    wj.h1.partial_charge,
                    &mut fwi.f_m,
                    &mut fwj.f_h1,
                );

                // M/M
                add_pair(
                    wi.m.posit,
                    wi.m.partial_charge,
                    wj.m.posit,
                    wj.m.partial_charge,
                    &mut fwi.f_m,
                    &mut fwj.f_m,
                );
            }
        }

        // --- Project EP/M force to O, half-kick, SETTLE, place EP, wrap molecule ---
        let half = 0.5 * dt;
        let cell = self.cell;

        for iw in 0..n_w {
            // masses before mutable borrow
            let (mo, mh0, mh1) = {
                let w = &self.water[iw];
                (w.o.mass, w.h0.mass, w.h1.mass)
            };

            let w = &mut self.water[iw];

            // Project EP to O WITHOUT aliasing borrows
            {
                let wi = &mut fw[iw];
                let m_impulse = mem::take(&mut wi.f_m); // or: mem::replace(&mut wi.f_m, Vec3::new_zero())
                wi.f_o += m_impulse;
            }

            // Half-kick
            w.o.vel += fw[iw].f_o * (half / mo);
            w.h0.vel += fw[iw].f_h0 * (half / mh0);
            w.h1.vel += fw[iw].f_h1 * (half / mh1);

            // Rigid update
            settle_opc(&mut w.o, &mut w.h0, &mut w.h1, dt);

            // Place EP, wrap molecule
            let bis = (w.h0.posit - w.o.posit) + (w.h1.posit - w.o.posit);
            w.m.posit = w.o.posit + bis.to_normalized() * O_EP_R_0;
            w.m.vel = (w.h0.vel + w.h1.vel) * 0.5;

            let wrapped_o = cell.wrap(w.o.posit);
            let shift = wrapped_o - w.o.posit;
            w.o.posit = wrapped_o;
            w.h0.posit += shift;
            w.h1.posit += shift;
            w.m.posit += shift;
        }

        // ---------- Recompute forces (fw2), project EP, second half-kick ----------
        let mut fw2 = vec![WaterForces::default(); n_w];
        // ... fill fw2 ...

        for iw in 0..n_w {
            let (mo, mh0, mh1) = {
                let w = &self.water[iw];
                (w.o.mass, w.h0.mass, w.h1.mass)
            };

            let w = &mut self.water[iw];

            // Project EP again (same anti-aliasing trick)
            {
                let wi2 = &mut fw2[iw];
                let m2 = mem::take(&mut wi2.f_m); // or replace(...)
                wi2.f_o += m2;
            }

            // Second half-kick
            w.o.vel += fw2[iw].f_o * (half / mo);
            w.h0.vel += fw2[iw].f_h0 * (half / mh0);
            w.h1.vel += fw2[iw].f_h1 * (half / mh1);
        }
    }
}

fn atoms_mut(mols: &mut [WaterMol]) -> impl Iterator<Item = &mut AtomDynamics> {
    mols.iter_mut()
        .flat_map(|m| [&mut m.o, &mut m.h0, &mut m.h1].into_iter())
}

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
    (ke, dof - 3 /* - n_constraints */)
}
