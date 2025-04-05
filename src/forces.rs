#![allow(non_snake_case)]

//! Force, acceleration, and related computations.

#[cfg(feature = "cuda")]
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use lin_alg::f32::{Vec3, Vec3x8, alloc_vec3s, f32x8};

// The rough Van der Waals (Lennard-Jones) minimum potential value, for two carbon atoms.
const LJ_MIN_R_CC: f32 = 3.82;

#[cfg(feature = "cuda")]
pub fn coulomb_force_gpu(
    dev: &Arc<CudaDevice>,
    posits_src: &[Vec3],
    posits_tgt: &[Vec3],
    charges: &[f64], // Corresponds 1:1 with `posit_charges`.
) -> Vec<f64> {
    let start = Instant::now();

    // allocate buffers
    let n_sources = posits_src.len();
    let n_targets = posits_tgt.len();

    let posit_charges_gpus = alloc_vec3s(dev, posits_src);
    let posits_sample_gpu = alloc_vec3s(dev, posits_tgt);

    // Note: This step is not required when using f64ss.
    let charges: Vec<f32> = charges.iter().map(|c| *c as f32).collect();

    let mut charges_gpu = dev.alloc_zeros::<f32>(n_sources).unwrap();
    dev.htod_sync_copy_into(&charges, &mut charges_gpu).unwrap();

    let mut V_per_sample = dev.alloc_zeros::<f32>(n_targets).unwrap();

    let kernel = dev.get_func("cuda", "coulomb_kernel").unwrap();

    // let cfg = LaunchConfig::for_num_elems(n_targets as u32);

    let cfg = {
        const NUM_THREADS: u32 = 1024;
        let num_blocks = (n_targets as u32).div_ceil(NUM_THREADS);

        // Custom launch config for 2-dimensional data (?)
        LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (NUM_THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    };

    unsafe {
        kernel.launch(
            cfg,
            (
                &mut V_per_sample,
                &posit_charges_gpus,
                &posits_sample_gpu,
                &charges_gpu,
                n_sources,
                n_targets,
            ),
        )
    }
    .unwrap();

    let result = dev.dtoh_sync_copy(&V_per_sample).unwrap();

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

/// The most fundamental part of Newtonian acceleration calculation.
/// `acc_dir` is a unit vector.
pub fn f_coulomb(
    dir: Vec3,
    dist: f32,
    q0: f32,
    q1: f32,
    softening_factor_sq: f32,
) -> Vec3 {
    // Assume the coulomb constant is 1.
    // println!("AD: {acc_dir}, src: {src_q} tgt: {tgt_q}  dist: {dist}");
    dir * q0 * q1 / (dist.powi(2) + softening_factor_sq)
}

pub fn f_coulomb_x8(
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
pub fn lj_potential(r: f32, sigma: f32, eps: f32) -> f32 {
    if r < f32::EPSILON {
        return 0.;
    }

    let sr = sigma / r;
    let sr6 = sr.powi(6);
    let sr12 = sr6.powi(2);

    4. * eps * (sr12 - sr6)
}

pub fn lj_potential_x8(r: f32x8, sigma: f32x8, eps: f32x8) -> f32x8 {
    // if r < f32::EPSILON {
    //     return f32x8::splat(0.);
    // }

    let sr = sigma / r;
    let sr6 = sr.powi(6);
    let sr12 = sr6.powi(2);

    f32x8::splat(4.) * eps * (sr12 - sr6)
}

/// Calculate the Lennard Jones force; a Newtonian force based on the LJ potential.
pub fn lj_force(dir: Vec3, r: f32, sigma: f32, eps: f32) -> Vec3 {
    let sr = sigma / r;
    let sr6 = sr.powi(6);
    let sr12 = sr6.powi(2);

    let mag = 24. * eps * (2. * sr12 - sr6) / r.powi(2);
    -dir * mag
}

/// Calculate the Lennard Jones force; a Newtonian force based on the LJ potential.
pub fn lj_force_x8(dir: Vec3x8, r: f32x8, sigma: f32x8, eps: f32x8) -> Vec3x8 {
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
