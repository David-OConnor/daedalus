use bio_files::amber_params::{AngleBendingParams, BondStretchingParams, DihedralParams};
use lin_alg::f64::{Vec3, calc_dihedral_angle_v2};

const EPS: f64 = 1e-10;

/// Returns the force on the atom at position 0. Negate this for the force on posit 1.
/// Also returns potential energy.
pub fn f_bond_stretching(
    posit_0: Vec3,
    posit_1: Vec3,
    params: &BondStretchingParams,
) -> (Vec3, f64) {
    let diff = posit_1 - posit_0;
    let r_meas = diff.magnitude();

    let r_delta = r_meas - params.r_0 as f64;

    let term_1 = params.k_b as f64 * r_delta; // Shared bewteen force and energy.
    // Note: We include the factor of 2x k_b when setting up indexed parameters.
    // Unit check: kcal/mol/Å² * Å² = kcal/mol. (Energy).
    let f_mag = term_1 / r_meas.max(EPS);

    // We divide by 2 due to the 2*kb factor stored in the indexed parameters.
    let energy = term_1 * r_delta * 0.5;

    (diff * f_mag, energy)
}

/// Valence angle; angle between 3 atoms.
/// Also returns potential energy.
pub fn f_angle_bending(
    posit_0: Vec3,
    posit_1: Vec3,
    posit_2: Vec3,
    params: &AngleBendingParams,
) -> ((Vec3, Vec3, Vec3), f64) {
    // Bond vectors with atom 1 at the vertex.
    let bond_vec_01 = posit_0 - posit_1;
    let bond_vec_21 = posit_2 - posit_1;

    let b_vec_01_sq = bond_vec_01.magnitude_squared();
    let b_vec_21_sq = bond_vec_21.magnitude_squared();

    // Quit early if atoms are on top of each other
    if b_vec_01_sq < EPS || b_vec_21_sq < EPS {
        return ((Vec3::new_zero(), Vec3::new_zero(), Vec3::new_zero()), 0.);
    }

    let b_vec_01_len = b_vec_01_sq.sqrt();
    let b_vec_21_len = b_vec_21_sq.sqrt();

    let inv_ab = 1.0 / (b_vec_01_len * b_vec_21_len);

    let cos_θ = (bond_vec_01.dot(bond_vec_21) * inv_ab).clamp(-1.0, 1.0);
    let sin_θ_sq = 1.0 - cos_θ * cos_θ;

    if sin_θ_sq < EPS {
        // θ = 0 or τ; gradient ill-defined
        return ((Vec3::new_zero(), Vec3::new_zero(), Vec3::new_zero()), 0.);
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

    let f = (f_0, f_1, f_2);
    // See note on bond len about division by 2.
    let energy = dV_dθ * Δθ * 0.5;

    (f, energy)
}

pub fn f_dihedral(
    posit_0: Vec3,
    posit_1: Vec3,
    posit_2: Vec3,
    posit_3: Vec3,
    params: &DihedralParams,
    improper: bool,
) -> ((Vec3, Vec3, Vec3, Vec3), f64) {
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
            (
                Vec3::new_zero(),
                Vec3::new_zero(),
                Vec3::new_zero(),
                Vec3::new_zero(),
            ),
            0.,
        );
    }

    let dihe_measured = calc_dihedral_angle_v2(&(posit_0, posit_1, posit_2, posit_3));

    // Note: We have already divided barrier height by the integer divisor when setting up
    // the Indexed params.
    let k = params.barrier_height as f64;
    let per = params.periodicity as f64;

    let (dV_dφ, dφ) = if improper {
        let dφ = dihe_measured - params.phase as f64;
        (2.0 * k * dφ, dφ)
    } else {
        let dφ = per * dihe_measured - params.phase as f64;
        (-k * per * dφ.sin(), dφ)
    };

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

    let energy = if improper {
        0.5 * dV_dφ * dφ // okay here, since V = k Δφ²
    } else {
        0.5 * k * (1.0 + (per * dihe_measured - params.phase as f64).cos())
    };

    ((f_0, f_1, f_2, f_3), energy)
}
