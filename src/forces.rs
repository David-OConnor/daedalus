#![allow(non_snake_case)]

//! Force, acceleration, and related computations.

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use std::sync::Arc;
        use cudarc::driver::{CudaStream, CudaModule, LaunchConfig, PushKernelArg};
        use lin_alg::f32::{vec3s_to_dev, vec3s_from_dev};
    }
}
use std::{collections::HashMap, time::Instant};

use lin_alg::f32::{Vec3, Vec3x8, f32x8};
use rayon::iter::IntoParallelRefIterator;
use crate::{docking::dynamics_playback::BodyVdw, element::Element};
use crate::docking::dynamics_playback::BodyVdwx8;

// The rough Van der Waals (Lennard-Jones) minimum potential value, for two carbon atoms.
const LJ_MIN_R_CC: f32 = 3.82;

#[cfg(feature = "cuda")]
pub fn force_coulomb_gpu_outer(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    posits_src: &[Vec3],
    posits_tgt: &[Vec3],
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

    // Some profiling numbers for certain grid sizes.
    // 2D, f32: 99.144 ms
    // 3D, f32: 400.06 ms
    // 2D, f64: 1_658 ms
    // 3D, f64: 1_643 ms
    // 300 ms for both large and small sizes on f32 with std::sqrt???

    let time_diff = Instant::now() - start;
    println!("GPU coulomb data collected. Time: {:?}", time_diff);

    // This step is not required when using f64.
    result.iter().map(|v| *v as f64).collect()
    // result
}

#[cfg(feature = "cuda")]
pub fn force_lj_gpu_outer(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    posits_tgt: &[Vec3],
    els_tgt: &[Element],
    bodies_src: &[BodyVdw],
    lj_lut: &HashMap<(Element, Element), (f32, f32)>,
) -> Vec<Vec3> {
    // Out is per target.
    let start = Instant::now();

    let mut posits_src = Vec::with_capacity(bodies_src.len());
    for body in bodies_src {
        posits_src.push(body.posit);
    }

    // allocate buffers
    let n_sources = posits_src.len();
    let n_targets = posits_tgt.len();

    let posit_charges_gpus = vec3s_to_dev(stream, &posits_src);
    let posits_sample_gpu = vec3s_to_dev(stream, posits_tgt);

    let mut result_buf = {
        let v = vec![Vec3::new_zero(); n_targets];
        vec3s_to_dev(stream, &v)
    };

    // This loop order must match the kernels.
    let mut sigmas = Vec::new(); // todo: With cap
    let mut epss = Vec::new();

    // todo: QC this!
    for el_tgt in els_tgt {
        for body_src in bodies_src {
            let (sigma, eps) = lj_lut[&(body_src.element, *el_tgt)];
            sigmas.push(sigma);
            epss.push(eps);
        }
    }

    let sigmas_gpu = stream.memcpy_stod(&sigmas).unwrap();
    let epss_gpu = stream.memcpy_stod(&epss).unwrap();

    // todo: Likely load these functions (kernels) at init and pass as a param.
    let func_lj_force = module.load_function("lj_force_kernel").unwrap();

    let cfg = LaunchConfig::for_num_elems(n_targets as u32);

    let mut launch_args = stream.launch_builder(&func_lj_force);

    launch_args.arg(&mut result_buf);
    launch_args.arg(&posit_charges_gpus);
    launch_args.arg(&posits_sample_gpu);
    launch_args.arg(&sigmas_gpu);
    launch_args.arg(&epss_gpu);
    launch_args.arg(&n_sources);
    launch_args.arg(&n_targets);

    unsafe { launch_args.launch(cfg) }.unwrap();

    // todo: Consider dtoh; passing to an existing vec instead of re-allocating
    let result = vec3s_from_dev(stream, &mut result_buf);

    let time_diff = Instant::now() - start;
    println!("GPU LJ force data collected. Time: {:?}", time_diff);

    // This step is not required when using f64.
    result
}

pub fn force_lj_outer(
    posit_target: Vec3,
    el_tgt: Element,
    bodies_src: &[BodyVdw],
    lj_lut: &HashMap<(Element, Element), (f32, f32)>,
) -> Vec3 {
    bodies_src
        .par_iter()
        .enumerate()
        .filter_map(|(i, body_source)| {
            let posit_src = body_source.posit;

            let diff = posit_src - posit_target;

            let dist = diff.magnitude();

            let dir = diff / dist; // Unit vec

            let (sigma, eps) = lj_lut.get(&(body_source.element, el_tgt)).unwrap();

            Some(force_lj_outer(dir, dist, *sigma, *eps))
        })
        .reduce(Vec3::new_zero, |f, elem| f + elem) // Sum the contributions.
}

fn force_lj_x8_outer(
    posit_target: Vec3x8,
    el_tgt: [Element; 8],
    bodies_src: &[BodyVdwx8],
    // distances: &[Vec<f32x8>],
    lj_lut: &HashMap<(Element, Element), (f32, f32)>,
    chunks_src: usize,
    lanes_tgt: usize,
    valid_lanes_src_last: usize,
) -> Vec3x8 {
    bodies_src
        .par_iter()
        .enumerate()
        .filter_map(|(i, body_source)| {
            let posit_src = body_source.posit;

            let diff = posit_src - posit_target;

            let dist = diff.magnitude();

            let dir = diff / dist; // Unit vec

            let lanes_src = if i == chunks_src - 1 {
                valid_lanes_src_last
            } else {
                8
            };

            let valid_lanes = lanes_src.min(lanes_tgt);

            // Setting sigma and eps to 0 for invalid lanes makes their contribution 0.
            let mut sigmas = [0.; 8];
            let mut epss = [0.; 8];
            for lane in 0..valid_lanes {
                let (sigma, eps) = lj_lut
                    .get(&(body_source.element[lane], el_tgt[lane]))
                    .unwrap();
                sigmas[lane] = *sigma;
                epss[lane] = *eps;
            }

            let sigma = f32x8::from_array(sigmas);
            let eps = f32x8::from_array(epss);

            Some(force_lj_x8(dir, dist, sigma, eps))
        })
        .reduce(Vec3x8::new_zero, |f, elem| f + elem) // Sum the contributions.
}

/// The most fundamental part of Newtonian acceleration calculation.
/// `acc_dir` is a unit vector.
pub fn force_coulomb(dir: Vec3, dist: f32, q0: f32, q1: f32, softening_factor_sq: f32) -> Vec3 {
    // Assume the coulomb constant is 1.
    // println!("AD: {acc_dir}, src: {src_q} tgt: {tgt_q}  dist: {dist}");
    dir * q0 * q1 / (dist.powi(2) + softening_factor_sq)
}

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
///
/// \[ V_{LJ}(r) = 4 \epsilon \left[\left(\frac{\sigma}{r}\right)^{12}
///     - \left(\frac{\sigma}{r}\right)^{6}\right] \]
///
/// In a real system, you’d want to parameterize \(\sigma\) and \(\epsilon\)
/// based on the atom types (i.e. from a force field lookup). Here, we’ll
/// just demonstrate the structure of the calculation with made-up constants.
pub fn V_lj(r: f32, sigma: f32, eps: f32) -> f32 {
    if r < f32::EPSILON {
        return 0.;
    }

    let sr = sigma / r;
    let sr6 = sr.powi(6);
    let sr12 = sr6.powi(2);

    4. * eps * (sr12 - sr6)
}

pub fn V_lj_x8(r: f32x8, sigma: f32x8, eps: f32x8) -> f32x8 {
    // if r < f32::EPSILON {
    //     return f32x8::splat(0.);
    // }

    let sr = sigma / r;
    let sr6 = sr.powi(6);
    let sr12 = sr6.powi(2);

    f32x8::splat(4.) * eps * (sr12 - sr6)
}

/// Calculate the Lennard Jones force; a Newtonian force based on the LJ potential.
pub fn force_lj(dir: Vec3, r: f32, sigma: f32, eps: f32) -> Vec3 {
    let sr = sigma / r;
    let sr6 = sr.powi(6);
    let sr12 = sr6.powi(2);

    let mag = 24. * eps * (2. * sr12 - sr6) / r.powi(2);
    -dir * mag
}

/// Calculate the Lennard Jones force; a Newtonian force based on the LJ potential.
pub fn force_lj_x8(dir: Vec3x8, r: f32x8, sigma: f32x8, eps: f32x8) -> Vec3x8 {
    let sr = sigma / r;
    let sr6 = sr.powi(6);
    let sr12 = sr6.powi(2);

    let mag = f32x8::splat(24.) * eps * (f32x8::splat(2.) * sr12 - sr6) / r.powi(2);

    // println!("\n\nsr: {:?}", sr);
    // println!("r: {:?}", r);
    // println!("sig: {:?}", sigma);
    // println!("sr12: {:?}", sr12);
    //
    // println!("\nDIR: {:?}", dir);
    // println!("mag: {:?}", mag);
    // println!("d mag: {:?}", (-dir * mag));

    -dir * mag
}
