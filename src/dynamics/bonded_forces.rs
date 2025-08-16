use bio_files::amber_params::{AngleBendingParams, BondStretchingParams, DihedralParams};
use lin_alg::f64::{Vec3, calc_dihedral_angle_v2};

const EPS: f64 = 1e-10;

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

pub fn f_dihedral(
    posit_0: Vec3,
    posit_1: Vec3,
    posit_2: Vec3,
    posit_3: Vec3,
    params: &DihedralParams,
    improper: bool,
) -> (Vec3, Vec3, Vec3, Vec3) {
    // Bond vectors (see Allen & Tildesley, chap. 4)
    let b1 = posit_1 - posit_0; // r_ij
    let b2 = posit_2 - posit_1; // r_kj
    let b3 = posit_3 - posit_2; // r_lk

    // Normal vectors to the two planes
    let n1 = b1.cross(b2);
    let n2 = b3.cross(b2);

    let n1_sq = n1.magnitude_squared();
    let n2_sq = n2.magnitude_squared();
    let b2_len = b2.magnitude();

    // Bail out if the four atoms are (nearly) colinear
    if n1_sq < EPS || n2_sq < EPS || b2_len < EPS {
        return (
            Vec3::new_zero(),
            Vec3::new_zero(),
            Vec3::new_zero(),
            Vec3::new_zero(),
        );
    }

    let dihe_measured = calc_dihedral_angle_v2(&(posit_0, posit_1, posit_2, posit_3));

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
    let k = params.barrier_height as f64;
    let per = params.periodicity as f64;

    let dV_dφ = if improper {
        2.0 * k * (dihe_measured - params.phase as f64)
    } else {
        let arg = per * dihe_measured - params.phase as f64;
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
    //             &params.atom_types,
    //             dihe_measured / TAU,
    //             params.phase / TAU as f32,
    //             params.periodicity,
    //             dV_dφ,
    //         );
    //     }
    // }

    // ∂φ/∂r   (see e.g. DOI 10.1016/S0021-9991(97)00040-8)
    let dφ_dr1 = -n1 * (b2_len / n1_sq);
    let dφ_dr4 = n2 * (b2_len / n2_sq);

    let dφ_dr2 = -n1 * (b1.dot(b2) / (b2_len * n1_sq)) + n2 * (b3.dot(b2) / (b2_len * n2_sq));

    let dφ_dr3 = -dφ_dr1 - dφ_dr2 - dφ_dr4; // Newton’s third law

    // F_i = −dV/dφ · ∂φ/∂r_i
    let f_0 = -dφ_dr1 * dV_dφ;
    let f_1 = -dφ_dr2 * dV_dφ;
    let f_2 = -dφ_dr3 * dV_dφ;
    let f_3 = -dφ_dr4 * dV_dφ;

    // todo from diagnostic: Can't find it in improper, although there's where the error is showing.
    if improper {
        // println!(
        //     "\nr0: {r_0} r1: {r_1} r2: {r_2} r3: {r_3} N1: {n1} N2: {n2} n1sq: {n1_sq} n2sq: {n2_sq} b2_len: {b2_len}"
        // );
        // println!(
        //     "DIHE: {:?}, dV_dφ: {}, k: {k}, phase: {}",
        //     dihe_measured, dV_dφ, params.phase as f64
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
        // println!("DIHE: {:?}, dV_dφ: {}, k: {k}, phase: {}", dihe_measured, dV_dφ, params.phase as f64);
        // println!("B3dotB2: {:.3}, b1db2: {:.3}. 1: {}, 2: {}, 3: {}, 4: {}", b3.dot(b2), b1.dot(b2), dφ_dr1, dφ_dr2, dφ_dr3, dφ_dr4);
        //
        // if r_0.x.is_nan() ||  r_1.x.is_nan() ||  r_2.x.is_nan() ||  r_3.x.is_nan() {
        //     panic!("NaN. a0: {a_0:?}, a1: {a_1:?}, a2: {a_2:?}, a3: {a_3:?}");
        // }
    }

    (f_0, f_1, f_2, f_3)
}
