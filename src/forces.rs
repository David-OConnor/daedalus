#![allow(non_snake_case)]

//! Force, acceleration, and related computations. There are general algorithms, and are called
//! by our more specific ones in the docking and molecular dynamics modules. Includes CPU, SIMD,
//! and CUDA where appropriate.

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use std::sync::Arc;
        use cudarc::driver::{CudaStream, CudaModule, LaunchConfig, PushKernelArg};
        use lin_alg::f32::{vec3s_to_dev, vec3s_from_dev};
    }
}

use lin_alg::{f32::Vec3 as Vec3F32, f64::Vec3};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use lin_alg::{
    f32::{Vec3x8, f32x8},
    f64::f64x4,
};

//
// /// Handles both LJ, and Coulomb (SPME short range) force.
// /// Inputs are structured differently here from our other one; uses pre-paired inputs and outputs, and
// /// a common index. Exclusions (e.g. Amber-style 1-2 adn 1-3) are handled upstream.
// ///
// /// Returns force, virial sum, and potential energy.
// ///
// /// todo: This class of function is just connecting, and I believe could be automated
// /// todo with a macro or code gen. look into how to do that. Start with your ideal API.
// #[cfg(feature = "cuda")]
// pub fn force_nonbonded_gpu(
//     stream: &Arc<CudaStream>,
//     module: &Arc<CudaModule>,
//     tgt_is: &[u32],
//     src_is: &[u32],
//     posits_tgt: &[Vec3F32],
//     posits_src: &[Vec3F32],
//     sigmas: &[f32],
//     epss: &[f32],
//     qs_tgt: &[f32],
//     qs_src: &[f32],
//     // We use these to determine which target array to accumulate
//     // force on.
//     atom_types_tgt: &[u8],  // 0 = dyn, 1 = water
//     water_types_tgt: &[u8], // See WaterSite repr
//     atom_types_src: &[u8],
//     water_types_src: &[u8],
//     scale_14: &[bool],
//     calc_ljs: &[bool],
//     calc_coulombs: &[bool],
//     symmetric: &[bool],
//     cutoff_ewald: f32,
//     alpha_ewald: f32,
//     cell_extent: Vec3F32,
//     n_dyn: usize,
//     n_water: usize,
// ) -> (Vec<Vec3F32>, Vec<ForcesOnWaterMol>, f64, f64) {
//     let n = posits_tgt.len();
//
//     assert_eq!(tgt_is.len(), n);
//     assert_eq!(src_is.len(), n);
//
//     assert_eq!(posits_src.len(), n);
//
//     assert_eq!(sigmas.len(), n);
//     assert_eq!(epss.len(), n);
//     assert_eq!(qs_tgt.len(), n);
//     assert_eq!(qs_src.len(), n);
//
//     assert_eq!(atom_types_tgt.len(), n);
//     assert_eq!(water_types_tgt.len(), n);
//     assert_eq!(atom_types_src.len(), n);
//     assert_eq!(water_types_src.len(), n);
//
//     assert_eq!(scale_14.len(), n);
//     assert_eq!(calc_ljs.len(), n);
//     assert_eq!(calc_coulombs.len(), n);
//     assert_eq!(symmetric.len(), n);
//
//     // May be safer for the GPU to pass u8 instead of bool??
//     let scale_14: Vec<_> = scale_14.iter().map(|v| *v as u8).collect();
//     let calc_ljs: Vec<_> = calc_ljs.iter().map(|v| *v as u8).collect();
//     let calc_coulombs: Vec<_> = calc_coulombs.iter().map(|v| *v as u8).collect();
//     let symmetric: Vec<_> = symmetric.iter().map(|v| *v as u8).collect();
//
//     // Set up empty device arrays the kernel will fill as output.
//     let mut forces_on_dyn = {
//         let v = vec![Vec3F32::new_zero(); n_dyn];
//         vec3s_to_dev(stream, &v)
//     };
//
//     let mut forces_on_water_o = {
//         let v = vec![Vec3F32::new_zero(); n_water];
//         vec3s_to_dev(stream, &v)
//     };
//     let mut forces_on_water_m = {
//         let v = vec![Vec3F32::new_zero(); n_water];
//         vec3s_to_dev(stream, &v)
//     };
//     let mut forces_on_water_h0 = {
//         let v = vec![Vec3F32::new_zero(); n_water];
//         vec3s_to_dev(stream, &v)
//     };
//     let mut forces_on_water_h1 = {
//         let v = vec![Vec3F32::new_zero(); n_water];
//         vec3s_to_dev(stream, &v)
//     };
//
//     let mut virial_gpu = stream.memcpy_stod(&[0.0f64]).unwrap();
//     let mut energy_gpu = stream.memcpy_stod(&[0.0f64]).unwrap();
//
//     // Store immutable input arrays to the device.
//
//     let tgt_is_gpu = stream.memcpy_stod(tgt_is).unwrap();
//     let src_is_gpu = stream.memcpy_stod(src_is).unwrap();
//
//     let posits_src_gpu = vec3s_to_dev(stream, posits_src);
//     let posits_tgt_gpu = vec3s_to_dev(stream, posits_tgt);
//
//     let sigmas_gpu = stream.memcpy_stod(sigmas).unwrap();
//     let epss_gpu = stream.memcpy_stod(epss).unwrap();
//
//     let qs_tgt_gpu = stream.memcpy_stod(qs_tgt).unwrap();
//     let qs_src_gpu = stream.memcpy_stod(qs_src).unwrap();
//
//     let atom_types_tgt_gpu = stream.memcpy_stod(atom_types_tgt).unwrap();
//     let water_types_tgt_gpu = stream.memcpy_stod(water_types_tgt).unwrap();
//     let atom_types_src_gpu = stream.memcpy_stod(atom_types_src).unwrap();
//     let water_types_src_gpu = stream.memcpy_stod(water_types_src).unwrap();
//
//     // For Amber-style 1-4 covalent bond scaling; not general LJ.
//     let scale_14_gpu = stream.memcpy_stod(&scale_14).unwrap();
//     let calc_ljs_gpu = stream.memcpy_stod(&calc_ljs).unwrap();
//     let calc_coulombs_gpu = stream.memcpy_stod(&calc_coulombs).unwrap();
//     let symmetric_gpu = stream.memcpy_stod(&symmetric).unwrap();
//
//     // todo: Likely load these functions (kernels) at init and pass as a param.
//     // todo: Seems to take only 4 μs (per time step), so should be fine here.
//     let kernel = module.load_function("nonbonded_force_kernel").unwrap();
//     let cfg = LaunchConfig::for_num_elems(n as u32);
//     let mut launch_args = stream.launch_builder(&kernel);
//
//     launch_args.arg(&mut forces_on_dyn);
//     launch_args.arg(&mut forces_on_water_o);
//     launch_args.arg(&mut forces_on_water_m);
//     launch_args.arg(&mut forces_on_water_h0);
//     launch_args.arg(&mut forces_on_water_h1);
//     launch_args.arg(&mut virial_gpu);
//     launch_args.arg(&mut energy_gpu);
//     //
//     launch_args.arg(&tgt_is_gpu);
//     launch_args.arg(&src_is_gpu);
//     launch_args.arg(&posits_tgt_gpu);
//     launch_args.arg(&posits_src_gpu);
//     launch_args.arg(&sigmas_gpu);
//     launch_args.arg(&epss_gpu);
//     launch_args.arg(&qs_tgt_gpu);
//     launch_args.arg(&qs_src_gpu);
//     launch_args.arg(&atom_types_tgt_gpu);
//     launch_args.arg(&water_types_tgt_gpu);
//     launch_args.arg(&atom_types_src_gpu);
//     launch_args.arg(&water_types_src_gpu);
//     launch_args.arg(&scale_14_gpu);
//     launch_args.arg(&calc_ljs_gpu);
//     launch_args.arg(&calc_coulombs_gpu);
//     launch_args.arg(&symmetric_gpu);
//     //
//     launch_args.arg(&cell_extent);
//     launch_args.arg(&cutoff_ewald);
//     launch_args.arg(&alpha_ewald);
//     launch_args.arg(&n);
//
//     unsafe { launch_args.launch(cfg) }.unwrap();
//
//     // todo: Consider dtoh; passing to an existing vec instead of re-allocating?
//     let forces_on_dyn = vec3s_from_dev(stream, &forces_on_dyn);
//
//     let forces_on_water_o = vec3s_from_dev(stream, &forces_on_water_o);
//     let forces_on_water_m = vec3s_from_dev(stream, &forces_on_water_m);
//     let forces_on_water_h0 = vec3s_from_dev(stream, &forces_on_water_h0);
//     let forces_on_water_h1 = vec3s_from_dev(stream, &forces_on_water_h1);
//
//     let mut forces_on_water = Vec::new();
//     for i in 0..n_water {
//         let f_o = forces_on_water_o[i].into();
//         let f_m = forces_on_water_m[i].into();
//         let f_h0 = forces_on_water_h0[i].into();
//         let f_h1 = forces_on_water_h1[i].into();
//
//         forces_on_water.push(ForcesOnWaterMol {
//             f_o,
//             f_m,
//             f_h0,
//             f_h1,
//         });
//     }
//
//     let virial = stream.memcpy_dtov(&virial_gpu).unwrap()[0];
//     let energy = stream.memcpy_dtov(&energy_gpu).unwrap()[0];
//
//     (forces_on_dyn, forces_on_water, virial, energy)
// }
//
// #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
// pub fn setup_sigma_eps_x8(
//     // todo: THis param list is onerous.
//     i_src: usize,
//     lj_lut: &LjTable,
//     chunks_src: usize,
//     lanes_tgt: usize,
//     valid_lanes_src_last: usize,
//     el_rec: &[Element],
//     body_source: &AtomDynamicsx4,
//     // ) -> (f32x8, f32x8) {
// ) -> (f64x4, f64x4) {
//     let lanes_src = if i_src == chunks_src - 1 {
//         valid_lanes_src_last
//     } else {
//         8
//     };
//
//     let valid_lanes = lanes_src.min(lanes_tgt);
//
//     // Setting sigma and eps to 0 for invalid lanes makes their contribution 0.
//     // let mut sigmas = [0.; 8];
//     // let mut epss = [0.; 8];
//     let mut sigmas = [0.; 4];
//     let mut epss = [0.; 4];
//
//     for lane in 0..valid_lanes {
//         let (sigma, eps) = lj_lut
//             .get(&(body_source.element[lane], el_rec[lane]))
//             .unwrap();
//         sigmas[lane] = *sigma as f64;
//         epss[lane] = *eps as f64;
//     }
//
//     // (f32x8::from_array(sigmas), f32x8::from_array(epss))
//     (f64x4::from_array(sigmas), f64x4::from_array(epss))
// }

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
