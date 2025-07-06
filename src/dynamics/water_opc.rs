//! We use the [OPC model](https://pubs.acs.org/doi/10.1021/jz501780a) for water.
//! See also, the Amber Rerference Manual. (todo: Specific ref)

use bio_files::amber_params::ForceFieldParams;
use lin_alg::f64::Vec3;

use crate::dynamics::{AtomDynamics, f_bond_stretching};

// Parameters for OPC water (JPCL, 2014, 5 (21), pp 3863-3871)
// (Amber 2025, frcmod.opc) EPC is presumably the massless, 4th charge.
const O_MASS: f64 = 16.;
const H_MASS: f64 = 1.008;
const EP_MASS: f64 = 0.; // todo: Rem A/R.

// Bond stretching; Å. Same K_b for all three bonds.
const K_B: f64 = 553.0; // kcal/mol/Å^2

// Å
const O_EP_R_0: f64 = 0.15939833;
const O_H_THETA_R_0: f64 = 0.87243313;
const H_H_THETA_R_0: f64 = 1.37120510;

// Angle Bending. kcal/mol/rad^2
const H_O_EP_K: f64 = 0.;
const H_O_H_K: f64 = 100.;
const H_H_O_K: f64 = 0.;

// Rad
const H_O_EP_θ0: f64 = 2.0943951023931953;
const H_O_H_θ0: f64 = 1.8081611050661253;
const H_H_O_θ0: f64 = 2.2294835864975564;

// Van der Waals / JL params.
const O_RSTAR: f64 = 1.777167268;
const H_RSTAR: f64 = 0.;
const EP_RSTAR: f64 = 1.;

const O_EPS: f64 = 0.2128008130;
const H_EPS: f64 = 0.;
const EP_EPS: f64 = 0.;

// Partial charges
// todo: Fill out.
const charge_o: f64 = 1.;
const charge_h: f64 = 1.;
const charge_ep: f64 = 1.;

/// Amber RM: "OPC is a non-polarizable, 4-point, 3-charge rigid water model. Geometrically, it resembles TIP4P-like mod-
/// els, although the values of OPC point charges and charge-charge distances are quite different.
/// The model has a single VDW center on the oxygen nucleus."
pub struct WaterMol {
    /// Absolute positions // todo: Is this what we want, or keep track of O abs pos, and relative
    /// todo positions for the othres, and/or angles and orientations?
    // pub params: ForceFieldParams, // todo: A/R
    pub o: Vec3,
    pub h0: Vec3,
    pub h1: Vec3,
    pub m: Vec3,
}

impl WaterMol {
    pub fn step(&mut self, dt: f64, sources: &[AtomDynamics]) {
        // todo: Apply bonded and non-bonded calcs from normal dynamics, using shared code

        // let f_bond = f_bond_stretching(self.h0, self.h1, params);
    }

    /// Calculate the force from Coulomb and Van der Waals (Lennard Jones) forces on an
    /// atom. ("Non-bonded")
    pub fn force_on(tgt: &AtomDynamics) -> Vec3 {
        Vec3::new_zero()
    }
}

// todo: A/R
pub struct Water {
    // todo: Track a posit and velocit for the whole molecule, and model with an
    // todo orientation quaternion?
    // pub posit: Vec3,
    // pub orientation: Quaternion,
    pub o: AtomDynamics,
    pub h0: AtomDynamics,
    pub h1: AtomDynamics,
}

// /// Add water molecules
// fn hydrate(pressure: f64, temp: f64, bounds: (Vec3, Vec3), n_mols: usize) -> Vec<WaterOps> {
//     let mut result = Vec::with_capacity(n_mols);
//     // todo: TIP4P to start?
//     for _ in 0..n_mols {
//         // result.push(WaterOps {
//         //     o: AtomDynamics {
//         //         element: Element::Oxygen,
//         //         name: "wo".to_string(), // todo: Qc
//         //         // todo
//         //         posit: Vec3::new_zero(),
//         //         // todo: Init vel based on temp and pressure?
//         //         vel: Vec3::new_zero(),
//         //         accel: Vec3::new_zero(),
//         //         mass: 8.,
//         //         partial_charge: 0.,
//         //         lj_r_star: 0.,
//         //         lj_eps: 0.,
//         //         force_field_type: None,
//         //     },
//         //     h0: AtomDynamics {
//         //         // todo
//         //         element: Element::Hydrogen,
//         //         name: "wo".to_string(), // todo: Qc
//         //         posit: Vec3::new_zero(),
//         //         // todo: Init vel based on temp and pressure?
//         //         vel: Vec3::new_zero(),
//         //         accel: Vec3::new_zero(),
//         //         mass: 1.,
//         //         partial_charge: 0.,
//         //         lj_r_star: 0.,
//         //         lj_eps: 0.,
//         //         force_field_type: None,
//         //     },
//         //     h1: AtomDynamics {
//         //         element: Element::Hydrogen,
//         //         name: "wo".to_string(), // todo: Qc
//         //         // todo
//         //         posit: Vec3::new_zero(),
//         //         // todo: Init vel based on temp and pressure?
//         //         vel: Vec3::new_zero(),
//         //         accel: Vec3::new_zero(),
//         //         mass: 1.,
//         //         partial_charge: 0.,
//         //         lj_r_star: 0.,
//         //         lj_eps: 0.,
//         //         force_field_type: None,
//         //     },
//         // })
//     }
//
//     result
// }
