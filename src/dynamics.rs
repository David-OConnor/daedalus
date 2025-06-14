//! This module contains a traditional molecular dynamics approach
//!
//! [Good article](https://www.owlposting.com/p/a-primer-on-molecular-dynamics)

// todo: Integration: Consider Verlet or Langevin

//  todo: Pressure and temp variables required. Perhaps related to ICs of water?

// Note on timescale: Generally femtosecond (-15)

use lin_alg::f64::Vec3;

/// r_0 minimizes E.
fn f_bond_distance(r: f64, r_0: f64) -> Vec3 {
    let dist = (r - r_0).abs();
    Vec3::new_zero()
}

/// θ_0 minimizes E.
fn f_bond_angle(θ: f64, θ_0: f64) -> Vec3 {
    let dist = (θ - θ_0).abs(); // todo: QC wraparounds. I don't think this is right as-is.
    Vec3::new_zero()
}

/// Dihedral angle.
/// todo: How does this work? Single angle value like bond angle?
fn f_bond_torsion(θ: f64, θ_0: f64) -> Vec3 {
    Vec3::new_zero()
}

/// Electrostatic + VDW: "non-bonded forces" in trad MD.
fn f_electrostatic() -> Vec3 {
    Vec3::new_zero()
}

/// Electrostatic + VDW: "non-bonded forces" in trad MD.
fn f_vdw() -> Vec3 {
    Vec3::new_zero()
}


/// Add water molecules
fn hydrate(pressure: f64, temp: f64, bounds: (Vec3, Vec3)) {
// todo: TIP4P to start?
}

// todo: One more?