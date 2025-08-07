//! For Smooth-Particle-Mesh_Ewald; a standard approximation for Coulomb forces in MD.

use std::f64::consts::{E, TAU};

// todo: This may be a good candidate for a standalone library.
use itertools::iproduct;
use lin_alg::f64::Vec3;
use rustfft::{FftPlanner, num_complex::Complex};
use statrs::function::erf::erfc;

use crate::dynamics::{AtomDynamics, ambient::SimBox};

#[cfg(target_arch = "x86_64")]
use lin_alg::f64::{f64x8, Vec3x4, Vec3x8};

// Ewald SPME approximation for Coulomb force
pub const EWALD_ALPHA: f64 = 0.257_f64; // 1/Å – good default for ~10 Å cutoff
const SQRT_PI: f64 = 1.7724538509055159;
pub const PME_MESH_SPACING: f64 = 1.0;
// SPME order‑4 B‑spline interpolation
const SPLINE_ORDER: usize = 4;

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

pub fn force_coulomb_ewald_real_x8(dir: Vec3x8, r: f64x8, qi: f64x8, qj: f64x8, alpha: f64x8) -> Vec3x8 {
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

    let force_mag = qfac * (erfc_term * inv_r2 + f64x8::splat(2.) * alpha * exp_term / (f64x8::splat(SQRT_PI) * r));

    dir * force_mag
}

/// Put one mesh‑spaced value into a flat xyz‑indexed Vec
#[inline]
fn idx(i: usize, j: usize, k: usize, ny: usize, nz: usize) -> usize {
    (i * ny + j) * nz + k
}

/// Compute long‑range electrostatic forces by SPME.
/// Returns a Vec<Vec3> in the same order as `atoms`.
pub fn pme_long_range_forces(
    atoms: &[AtomDynamics],
    cell: &SimBox,
    alpha: f64,
    mesh_spacing: f64, // ≈1 Å is fine
) -> Vec<Vec3> {
    // Build mesh dimensions (next power of two for FFT efficiency)
    let ext = cell.extent(); // Å
    let mx = (ext.x / mesh_spacing).ceil() as usize;
    let my = (ext.y / mesh_spacing).ceil() as usize;
    let mz = (ext.z / mesh_spacing).ceil() as usize;
    let (nx, ny, nz) = (
        mx.next_power_of_two(),
        my.next_power_of_two(),
        mz.next_power_of_two(),
    );
    let vol = ext.x * ext.y * ext.z;

    // Spread charges to mesh with 4‑th order B‑splines
    let mut rho = vec![0.0_f64; nx * ny * nz];
    let order = SPLINE_ORDER as isize;

    for atom in atoms {
        // fractional coords in [0,1)
        let diff = atom.posit - cell.bounds_low;
        let frac = Vec3::new(diff.x / ext.x, diff.y / ext.y, diff.z / ext.z);

        let fx = frac.x * nx as f64;
        let fy = frac.y * ny as f64;
        let fz = frac.z * nz as f64;
        let ix0 = fx.floor() as isize;
        let iy0 = fy.floor() as isize;
        let iz0 = fz.floor() as isize;

        // pre‑compute B‑spline weights for each dimension
        let mut wx = [0.0; SPLINE_ORDER];
        let mut wy = [0.0; SPLINE_ORDER];
        let mut wz = [0.0; SPLINE_ORDER];

        bspline4_weights(fx - ix0 as f64, &mut wx);
        bspline4_weights(fy - iy0 as f64, &mut wy);
        bspline4_weights(fz - iz0 as f64, &mut wz);

        for (dx, &wxv) in wx.iter().enumerate() {
            let ix = (ix0 + dx as isize).rem_euclid(nx as isize) as usize;
            for (dy, &wyv) in wy.iter().enumerate() {
                let iy = (iy0 + dy as isize).rem_euclid(ny as isize) as usize;
                for (dz, &wzv) in wz.iter().enumerate() {
                    let iz = (iz0 + dz as isize).rem_euclid(nz as isize) as usize;
                    rho[idx(ix, iy, iz, ny, nz)] += atom.partial_charge * wxv * wyv * wzv;
                }
            }
        }
    }

    // Forward FFT  ρ(k)
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(nx * ny * nz);
    let mut rho_c: Vec<Complex<f64>> = rho
        .into_iter()
        .map(|r| Complex { re: r, im: 0.0 })
        .collect();
    fft.process(&mut rho_c);

    // Multiply by Green’s function  G(k) to obtain ϕ(k)
    for (n, (kx, ky, kz)) in iproduct!(0..nx, 0..ny, 0..nz).enumerate() {
        if kx == 0 && ky == 0 && kz == 0 {
            // k = 0 → net‑charge term; skip (or handle separately)
            rho_c[n] = Complex::ZERO;
            continue;
        }

        // reciprocal‑space vector (2π/L)·k
        let kvec = Vec3::new(kx as f64 / ext.x, ky as f64 / ext.y, kz as f64 / ext.z) * TAU;

        let k2 = kvec.dot(kvec);
        let c = (-k2 / (4.0 * alpha * alpha)).exp() / k2; // Smooth term
        let scale = 2. * TAU / vol;
        rho_c[n] *= scale * c;
    }

    // Inverse FFT ϕ(r)
    let ifft = planner.plan_fft_inverse(nx * ny * nz);
    ifft.process(&mut rho_c);

    // convert back to real grid, divide by total points (rustfft does not normalise)
    let mut phi = vec![0.0_f64; nx * ny * nz];
    let norm = 1.0 / (nx * ny * nz) as f64;
    for (i, c) in rho_c.into_iter().enumerate() {
        phi[i] = c.re * norm;
    }

    // Finite‑difference electric field  E = −∇ϕ  (central differences)
    let mut ex = vec![0.0; nx * ny * nz];
    let mut ey = vec![0.0; nx * ny * nz];
    let mut ez = vec![0.0; nx * ny * nz];
    for (i, j, k) in iproduct!(0..nx, 0..ny, 0..nz) {
        let ip1 = (i + 1) % nx;
        let im1 = (i + nx - 1) % nx;
        let jp1 = (j + 1) % ny;
        let jm1 = (j + ny - 1) % ny;
        let kp1 = (k + 1) % nz;
        let km1 = (k + nz - 1) % nz;

        let scale_x = (nx as f64) / ext.x;
        let scale_y = (ny as f64) / ext.y;
        let scale_z = (nz as f64) / ext.z;

        ex[idx(i, j, k, ny, nz)] =
            -(phi[idx(ip1, j, k, ny, nz)] - phi[idx(im1, j, k, ny, nz)]) * 0.5 * scale_x;
        ey[idx(i, j, k, ny, nz)] =
            -(phi[idx(i, jp1, k, ny, nz)] - phi[idx(i, jm1, k, ny, nz)]) * 0.5 * scale_y;
        ez[idx(i, j, k, ny, nz)] =
            -(phi[idx(i, j, kp1, ny, nz)] - phi[idx(i, j, km1, ny, nz)]) * 0.5 * scale_z;
    }

    // Gather forces back to atoms (same B‑spline weights)
    let mut forces = vec![Vec3::new_zero(); atoms.len()];

    for (a_idx, atom) in atoms.iter().enumerate() {
        let diff = atom.posit - cell.bounds_low;
        let frac = Vec3::new(diff.x / ext.x, diff.y / ext.y, diff.z / ext.z);

        let fx = frac.x * nx as f64;
        let fy = frac.y * ny as f64;
        let fz = frac.z * nz as f64;
        let ix0 = fx.floor() as isize;
        let iy0 = fy.floor() as isize;
        let iz0 = fz.floor() as isize;

        let mut wx = [0.0; SPLINE_ORDER];
        let mut wy = [0.0; SPLINE_ORDER];
        let mut wz = [0.0; SPLINE_ORDER];
        bspline4_weights(fx - ix0 as f64, &mut wx);
        bspline4_weights(fy - iy0 as f64, &mut wy);
        bspline4_weights(fz - iz0 as f64, &mut wz);

        let mut e = Vec3::new_zero();

        for (dx, &wxv) in wx.iter().enumerate() {
            let ix = (ix0 + dx as isize).rem_euclid(nx as isize) as usize;
            for (dy, &wyv) in wy.iter().enumerate() {
                let iy = (iy0 + dy as isize).rem_euclid(ny as isize) as usize;
                for (dz, &wzv) in wz.iter().enumerate() {
                    let iz = (iz0 + dz as isize).rem_euclid(nz as isize) as usize;
                    let weight = wxv * wyv * wzv;
                    let idx = idx(ix, iy, iz, ny, nz);
                    e.x += weight * ex[idx];
                    e.y += weight * ey[idx];
                    e.z += weight * ez[idx];
                }
            }
        }

        forces[a_idx] = e * atom.partial_charge; // F = qE
    }

    forces
}

/// 4‑th order cardinal B‑spline weights and their cumulative sums.
/// (Only the weights are needed for PME; derivative form not required here)
fn bspline4_weights(x: f64, w: &mut [f64; SPLINE_ORDER]) {
    // x ∈ [0,1)
    let xm1 = 1.0 - x;
    w[0] = (1.0 / 6.0) * xm1.powi(3);
    w[1] = (1.0 / 6.0) * (3.0 * x.powi(3) - 6.0 * x.powi(2) + 4.0);
    w[2] = (1.0 / 6.0) * (-3.0 * x.powi(3) + 3.0 * x.powi(2) + 3.0 * x + 1.0);
    w[3] = (1.0 / 6.0) * x.powi(3);
}
