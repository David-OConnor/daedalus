//! For Smooth-Particle-Mesh_Ewald; a standard approximation for Coulomb forces in MD.

use std::f64::consts::{FRAC_2_SQRT_PI, PI, TAU};

// todo: This may be a good candidate for a standalone library.
use lin_alg::f64::Vec3;
#[cfg(target_arch = "x86_64")]
use lin_alg::f64::{Vec3x8, f64x8};
use rustfft::{FftDirection, FftPlanner, num_complex::Complex};
use statrs::function::erf::{erf, erfc};

use crate::dynamics::{AtomDynamics, MdState, ambient::SimBox, non_bonded::SCALE_COUL_14};

// Ewald SPME approximation for Coulomb force
pub const EWALD_ALPHA: f64 = 0.257_f64; // 1/Å – good default for ~10 Å cutoff
const SQRT_PI: f64 = 1.7724538509055159;
pub const PME_MESH_SPACING: f64 = 1.0;
// SPME order‑4 B‑spline interpolation
const SPLINE_ORDER: usize = 4;

/// We use this for short-range Coulomb forces, as part of SPME.
pub fn force_coulomb_ewald_real(dir: Vec3, r: f64, qi: f64, qj: f64, alpha: f64) -> Vec3 {
    // F = q_i q_j [ erfc(αr)/r² + 2α/√π · e^(−α²r²)/r ]  · 4πϵ0⁻¹  · r̂
    let qfac = qi * qj;
    let inv_r = 1.0 / r;
    let inv_r2 = inv_r * inv_r;

    let erfc_term = erfc(alpha * r);
    let exp_term = (-alpha * alpha * r * r).exp();
    let force_mag = qfac * (erfc_term * inv_r2 + 2.0 * alpha * exp_term / (SQRT_PI * r));

    dir * force_mag
}

pub fn force_coulomb_ewald_real_x8(
    dir: Vec3x8,
    r: f64x8,
    qi: f64x8,
    qj: f64x8,
    alpha: f64x8,
) -> Vec3x8 {
    // F = q_i q_j [ erfc(αr)/r² + 2α/√π · e^(−α²r²)/r ]  · 4πϵ0⁻¹  · r̂
    let qfac = qi * qj;
    let inv_r = f64x8::splat(1.) / r;
    let inv_r2 = inv_r * inv_r;

    // let erfc_term = erfc(alpha * r);
    let erfc_term = f64x8::splat(0.); // todo temp: Figure how how to do erfc with SIMD.

    // todo: Figure out how to do exp with SIMD. Probably need powf in lin_alg
    // let exp_term = (-alpha * alpha * r * r).exp();
    // let exp_term = f64x8::splat(E).pow(-alpha * alpha * r * r);
    let exp_term = f64x8::splat(1.); // todo temp

    let force_mag = qfac
        * (erfc_term * inv_r2 + f64x8::splat(2.) * alpha * exp_term / (f64x8::splat(SQRT_PI) * r));

    dir * force_mag
}
