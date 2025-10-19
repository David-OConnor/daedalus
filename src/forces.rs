#![allow(non_snake_case)]
#![allow(unused)]

//! Force, acceleration, and related computations. There are general algorithms, and are called
//! by our more specific ones in the docking and molecular dynamics modules. Includes CPU, SIMD,
//! and CUDA where appropriate.
//!
//! Note: We don't use most (all?) of these: We use GPU kernels instead, or use a specific form
//! while not using the others. We keep them for reference.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use lin_alg::f32::{Vec3x8, f32x8};
use lin_alg::{f32::Vec3 as Vec3F32, f64::Vec3};

/// The most fundamental part of Newtonian acceleration calculation.
/// `acc_dir` is a unit vector.
pub fn force_coulomb_f32(
    dir: Vec3F32,
    dist: f32,
    q0: f32,
    q1: f32,
    softening_factor_sq: f32,
) -> Vec3F32 {
    dir * q0 * q1 / (dist.powi(2) + softening_factor_sq)
}

pub fn force_coulomb(dir: Vec3, dist: f64, q0: f64, q1: f64, softening_factor_sq: f64) -> Vec3 {
    dir * q0 * q1 / (dist.powi(2) + softening_factor_sq)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn force_coulomb_x8(
    dir: Vec3x8,
    dist: f32x8,
    q0: f32x8,
    q1: f32x8,
    softening_factor_sq: f32x8,
) -> Vec3x8 {
    dir * q0 * q1 / (dist.powi(2) + softening_factor_sq)
}

/// Calculate the Lennard-Jones potential between two atoms.
/// σ is in Å. ε is in kcal/mol.
///
/// σ_min (i, j) = 0.5(σ_min_i + σ_min_j)
/// ε(i, j) = sqrt(ε_i * ε_j)
pub fn V_lj(dist: f32, sigma: f32, eps: f32) -> f32 {
    if dist < f32::EPSILON {
        return 0.;
    }

    let sr = sigma / dist;
    let s_r_6 = sr.powi(6);
    let s_r_12 = s_r_6.powi(2);

    4. * eps * (s_r_12 - s_r_6)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
/// See notes on `V_lj()`.
pub fn V_lj_x8(dist: f32x8, sigma: f32x8, eps: f32x8) -> f32x8 {
    // if r < f32::EPSILON {
    //     return f32x8::splat(0.);
    // }

    let s_r = sigma / dist;
    let s_r_6 = s_r.powi(6);
    let s_r_12 = s_r_6.powi(2);

    f32x8::splat(4.) * eps * (s_r_12 - s_r_6)
}

/// See notes on `V_lj()`.
pub fn force_lj_f32(dir: Vec3F32, dist: f32, sigma: f32, eps: f32) -> Vec3F32 {
    let s_r = sigma / dist;
    let s_r_6 = s_r.powi(6);
    let s_r_12 = s_r_6.powi(2);

    // todo: ChatGPT is convinced I divide by r here, not r^2...
    let mag = 24. * eps * (2. * s_r_12 - s_r_6) / dist.powi(2);
    dir * mag
}

/// See notes on `V_lj()`. We set up the dist params we do to share computation
/// with Coulomb.
/// This assumes diff (and dir) is in order tgt - src.
pub fn force_lj(dir: Vec3, inv_dist: f64, sigma: f64, eps: f64) -> Vec3 {
    let sr = sigma * inv_dist;
    let sr6 = sr.powi(6);
    let sr12 = sr6.powi(2);

    let mag = 24. * eps * (2. * sr12 - sr6) * inv_dist;
    dir * mag
}

/// See notes on `V_lj()`. We set up the dist params we do to share computation
/// with Coulomb.
/// This assumes diff (and dir) is in order tgt - src.
/// This variant also computes energy.
pub fn force_e_lj(dir: Vec3, inv_dist: f64, sigma: f64, eps: f64) -> (Vec3, f64) {
    let sr = sigma * inv_dist;
    let sr6 = sr.powi(6);
    let sr12 = sr6.powi(2);

    let mag = 24. * eps * (2. * sr12 - sr6) * inv_dist;

    let energy = 4. * eps * (sr12 - sr6);
    (dir * mag, energy)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
/// See notes on `V_lj()`.
pub fn force_lj_x8(dir: Vec3x8, inv_dist: f32x8, sigma: f32x8, eps: f32x8) -> Vec3x8 {
    let sr = sigma * inv_dist;
    let sr6 = sr.powi(6);
    let sr12 = sr6.powi(2);

    let mag = f32x8::splat(24.) * eps * (f32x8::splat(2.) * sr12 - sr6) * inv_dist;
    dir * mag
}
