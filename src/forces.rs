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
use std::time::Instant;

use lin_alg::{f32::Vec3 as Vec3F32, f64::Vec3};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use lin_alg::{
    f32::{Vec3x8, f32x8},
    f64::f64x4,
};
use na_seq::{Element, element::LjTable};
use rayon::prelude::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::dynamics::AtomDynamicsx4;

// The rough Van der Waals (Lennard-Jones) minimum potential value, for two carbon atoms.
const LJ_MIN_R_CC: f32 = 3.82;

#[cfg(feature = "cuda")]
pub fn force_coulomb_gpu_outer(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    posits_src: &[Vec3F32],
    posits_tgt: &[Vec3F32],
    charges: &[f64], // Corresponds 1:1 with `posit_charges`.
) -> Vec<f64> {
    let start = Instant::now();

    // allocate buffers
    let n_sources = posits_src.len();
    let n_targets = posits_tgt.len();

    let posit_charges_gpus = vec3s_to_dev(stream, posits_src);
    let posits_sample_gpu = vec3s_to_dev(stream, posits_tgt);

    // Note: This step is not required when using f64ss.
    let charges: Vec<f32> = charges.iter().map(|c| *c as f32).collect();

    let mut charges_gpu = stream.alloc_zeros::<f32>(n_sources).unwrap();
    stream.memcpy_htod(&charges, &mut charges_gpu).unwrap();

    let mut V_per_sample = stream.alloc_zeros::<f32>(n_targets).unwrap();

    // todo: Likely load these functions (kernels) at init and pass as a param.
    let func_coulomb = module.load_function("coulomb_force_kernel").unwrap();
    // todo: Swap in when ready.
    // let func_coulomb = module.load_function("coulomb_force_spme_short_range_kernel").unwrap();

    let cfg = LaunchConfig::for_num_elems(n_targets as u32);

    // let cfg = {
    //     const NUM_THREADS: u32 = 1024;
    //     let num_blocks = (n_targets as u32).div_ceil(NUM_THREADS);
    //
    //     // Custom launch config for 2-dimensional data (?)
    //     LaunchConfig {
    //         grid_dim: (num_blocks, 1, 1),
    //         block_dim: (NUM_THREADS, 1, 1),
    //         shared_mem_bytes: 0,
    //     }
    // };

    let mut launch_args = stream.launch_builder(&func_coulomb);

    launch_args.arg(&mut V_per_sample);
    launch_args.arg(&posit_charges_gpus);
    launch_args.arg(&posits_sample_gpu);
    launch_args.arg(&charges_gpu);
    launch_args.arg(&n_sources);
    launch_args.arg(&n_targets);

    unsafe { launch_args.launch(cfg) }.unwrap();

    // todo: Consider dtoh; passing to an existing vec instead of re-allocating
    let result = stream.memcpy_dtov(&V_per_sample).unwrap();
    // stream.memcpy_dtoh(&V_per_sample, &mut result_buf).unwrap();

    let time_diff = Instant::now() - start;
    println!("GPU coulomb data collected. Time: {:?}", time_diff);

    // This step is not required when using f64.
    result.iter().map(|v| *v as f64).collect()
    // result
}

#[cfg(feature = "cuda")]
pub fn force_lj_gpu(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    posits_tgt: &[Vec3F32],
    posits_src: &[Vec3F32],
    sigmas: &[f32],
    epss: &[f32],
) -> Vec<Vec3F32> {
    // Out is per target.
    let start = Instant::now();

    // Allocate buffers
    let n_sources = posits_src.len();
    let n_targets = posits_tgt.len();

    let posits_src_gpu = vec3s_to_dev(stream, posits_src);
    let posits_tgt_gpu = vec3s_to_dev(stream, posits_tgt);

    let mut result_buf = {
        let v = vec![Vec3F32::new_zero(); n_targets];
        vec3s_to_dev(stream, &v)
    };

    let sigmas_gpu = stream.memcpy_stod(sigmas).unwrap();
    let epss_gpu = stream.memcpy_stod(epss).unwrap();

    // todo: Likely load these functions (kernels) at init and pass as a param.
    let func_lj_force = module.load_function("lj_force_kernel").unwrap();

    let cfg = LaunchConfig::for_num_elems(n_targets as u32);

    let mut launch_args = stream.launch_builder(&func_lj_force);

    launch_args.arg(&mut result_buf);
    launch_args.arg(&posits_src_gpu);
    launch_args.arg(&posits_tgt_gpu);
    launch_args.arg(&sigmas_gpu);
    launch_args.arg(&epss_gpu);
    launch_args.arg(&n_sources);
    launch_args.arg(&n_targets);

    unsafe { launch_args.launch(cfg) }.unwrap();

    // todo: Consider dtoh; passing to an existing vec instead of re-allocating
    let result = vec3s_from_dev(stream, &result_buf);

    // let time_diff = Instant::now() - start;
    // println!("GPU LJ force data collected. Time: {:?}", time_diff);

    result
}

/// Handles both LJ, and Coulomb (SPME short range) force.
/// Inputs are structured differently here from our other one; uses pre-paired inputs and outputs, and
/// a common index. Exclusions (e.g. Amber-style 1-2 adn 1-3) are handled upstream.
///
/// todo: Sort out how you handle the symmetric case here. Maybe return a separate "forces-on-sources"?
/// Returns (forces on targets, virial sum)
#[cfg(feature = "cuda")]
pub fn force_nonbonded_gpu(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    tgt_is: &[u32],
    src_is: &[u32],
    posits_tgt: &[Vec3F32],
    posits_src: &[Vec3F32],
    sigmas: &[f32],
    epss: &[f32],
    qs_tgt: &[f32],
    qs_src: &[f32],
    scale_14: &[bool],
    cutoff: f32,
    alpha: f32,
    symmetric: bool,
    n_targets: usize,
) -> (Vec<Vec3F32>, f32) {
    let start = Instant::now();

    let n = posits_tgt.len();

    assert_eq!(posits_src.len(), n);
    assert_eq!(sigmas.len(), n);
    assert_eq!(epss.len(), n);
    assert_eq!(qs_tgt.len(), n);
    assert_eq!(qs_src.len(), n);
    assert_eq!(scale_14.len(), n);
    assert_eq!(tgt_is.len(), n);
    assert_eq!(src_is.len(), n);

    // May be safer for the GPU?
    let scale_14: Vec<_> = scale_14.iter().map(|v| *v as u8).collect();

    let symmetric = symmetric as u8;

    // Allocate buffers
    let posits_src_gpu = vec3s_to_dev(stream, posits_src);
    let posits_tgt_gpu = vec3s_to_dev(stream, posits_tgt);

    let mut result_buf = {
        let v = vec![Vec3F32::new_zero(); n_targets];
        vec3s_to_dev(stream, &v)
    };

    let sigmas_gpu = stream.memcpy_stod(sigmas).unwrap();
    let epss_gpu = stream.memcpy_stod(epss).unwrap();

    let qs_tgt_gpu = stream.memcpy_stod(qs_tgt).unwrap();
    let qs_src_gpu = stream.memcpy_stod(qs_src).unwrap();

    let tgt_is_gpu = stream.memcpy_stod(tgt_is).unwrap();
    let src_is_gpu = stream.memcpy_stod(src_is).unwrap();

    // For Amber-style 1-4 covalent bond scaling; not general LJ.
    let scale_14_gpu = stream.memcpy_stod(&scale_14).unwrap();

    // todo: Likely load these functions (kernels) at init and pass as a param.
    // todo: Seems to take only 4 μs (per time step), so should be fine here.
    let func_force = module.load_function("nonbonded_force_kernel").unwrap();

    let cfg = LaunchConfig::for_num_elems(n as u32);

    let mut launch_args = stream.launch_builder(&func_force);

    // todo: Is there a better way to pass non-arrays?
    let mut virial_gpu = stream.memcpy_stod(&[0.0f32]).unwrap();

    launch_args.arg(&mut result_buf);
    launch_args.arg(&mut virial_gpu);
    launch_args.arg(&tgt_is_gpu);
    launch_args.arg(&src_is_gpu);
    launch_args.arg(&posits_tgt_gpu);
    launch_args.arg(&posits_src_gpu);
    launch_args.arg(&sigmas_gpu);
    launch_args.arg(&epss_gpu);
    launch_args.arg(&qs_tgt_gpu);
    launch_args.arg(&qs_src_gpu);
    launch_args.arg(&scale_14_gpu);
    launch_args.arg(&cutoff);
    launch_args.arg(&alpha);
    launch_args.arg(&symmetric);
    // todo: Cell A/R for wrapping water
    launch_args.arg(&n);

    unsafe { launch_args.launch(cfg) }.unwrap();

    // todo: Consider dtoh; passing to an existing vec instead of re-allocating?
    let forces = vec3s_from_dev(stream, &result_buf);
    // todo: QC this.
    let virial = stream.memcpy_dtov(&virial_gpu).unwrap()[0];

    // let time_diff = Instant::now() - start;
    // println!("GPU LJ force data collected. Time: {:?}", time_diff);

    (forces, virial)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn setup_sigma_eps_x8(
    // todo: THis param list is onerous.
    i_src: usize,
    lj_lut: &LjTable,
    chunks_src: usize,
    lanes_tgt: usize,
    valid_lanes_src_last: usize,
    el_rec: &[Element],
    body_source: &AtomDynamicsx4,
    // ) -> (f32x8, f32x8) {
) -> (f64x4, f64x4) {
    let lanes_src = if i_src == chunks_src - 1 {
        valid_lanes_src_last
    } else {
        8
    };

    let valid_lanes = lanes_src.min(lanes_tgt);

    // Setting sigma and eps to 0 for invalid lanes makes their contribution 0.
    // let mut sigmas = [0.; 8];
    // let mut epss = [0.; 8];
    let mut sigmas = [0.; 4];
    let mut epss = [0.; 4];

    for lane in 0..valid_lanes {
        let (sigma, eps) = lj_lut
            .get(&(body_source.element[lane], el_rec[lane]))
            .unwrap();
        sigmas[lane] = *sigma as f64;
        epss[lane] = *eps as f64;
    }

    // (f32x8::from_array(sigmas), f32x8::from_array(epss))
    (f64x4::from_array(sigmas), f64x4::from_array(epss))
}

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

    let mag = 24. * eps * (2. * s_r_12 - s_r_6) / dist.powi(2);
    -dir * mag
}

/// See notes on `V_lj()`.
pub fn force_lj(dir: Vec3, dist: f64, sigma: f64, eps: f64) -> Vec3 {
    let s_r = sigma / dist;
    let s_r_6 = s_r.powi(6);
    let s_r_12 = s_r_6.powi(2);

    let mag = 24. * eps * (2. * s_r_12 - s_r_6) / dist.powi(2);
    -dir * mag
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
/// See notes on `V_lj()`.
pub fn force_lj_x8(dir: Vec3x8, dist: f32x8, sigma: f32x8, eps: f32x8) -> Vec3x8 {
    let s_r = sigma / dist;
    let s_r_6 = s_r.powi(6);
    let s_r_12 = s_r_6.powi(2);

    let mag = f32x8::splat(24.) * eps * (f32x8::splat(2.) * s_r_12 - s_r_6) / dist.powi(2);

    -dir * mag
}
