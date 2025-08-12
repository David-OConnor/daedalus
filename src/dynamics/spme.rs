//! For Smooth-Particle-Mesh_Ewald; a standard approximation for Coulomb forces in MD.

use std::f64::consts::{FRAC_2_SQRT_PI, TAU};

// todo: This may be a good candidate for a standalone library.
use lin_alg::f64::Vec3;
#[cfg(target_arch = "x86_64")]
use lin_alg::f64::{Vec3x8, f64x8};
use rustfft::{FftDirection, FftPlanner, num_complex::Complex};
use statrs::function::erf::{erf, erfc};

use crate::dynamics::{AtomDynamics, MdState, ambient::SimBox};

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

/// Put one mesh‑spaced value into a flat xyz‑indexed Vec
#[inline]
fn idx(i: usize, j: usize, k: usize, ny: usize, nz: usize) -> usize {
    (i * ny + j) * nz + k
}

fn for_each_stencil<F: FnMut(usize, usize, usize, f64)>(
    atom: &AtomDynamics,
    cell: &SimBox,
    ext: Vec3,
    nx: usize,
    ny: usize,
    nz: usize,
    mut f: F,
) {
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

    let ix_base = ix0 - 1;
    let iy_base = iy0 - 1;
    let iz_base = iz0 - 1;

    for (dx, &wxv) in wx.iter().enumerate() {
        let ix = (ix_base + dx as isize).rem_euclid(nx as isize) as usize;
        for (dy, &wyv) in wy.iter().enumerate() {
            let iy = (iy_base + dy as isize).rem_euclid(ny as isize) as usize;
            for (dz, &wzv) in wz.iter().enumerate() {
                let iz = (iz_base + dz as isize).rem_euclid(nz as isize) as usize;
                f(ix, iy, iz, wxv * wyv * wzv);
            }
        }
    }
}

pub struct PmeLrForces {
    pub dyn_static: Vec<Vec3>,
    /// Packed in the same order as atoms_water input: M/EP, H0, H1 per water.
    pub water: Vec<Vec3>,
}

/// Compute long‑range electrostatic forces by SPME.
/// Returns a Vec<Vec3> in the same order as `atoms`.
pub fn pme_long_range_forces(
    atoms_static_dy: &[AtomDynamics],
    // Separate because of this type difference; &[T] vs &[&T].)
    atoms_water: &[&AtomDynamics],
    cell: &SimBox,
    alpha: f64,
    mesh_spacing: f64, // ≈1 Å is fine
) -> PmeLrForces {
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

    // Spread charges to mesh with 4‑th order B‑splines
    let mut rho = vec![0.0_f64; nx * ny * nz];

    for atom in atoms_static_dy {
        for_each_stencil(atom, cell, ext, nx, ny, nz, |ix, iy, iz, w| {
            rho[idx(ix, iy, iz, ny, nz)] += atom.partial_charge * w;
        });
    }

    for atom in atoms_water {
        for_each_stencil(atom, cell, ext, nx, ny, nz, |ix, iy, iz, w| {
            rho[idx(ix, iy, iz, ny, nz)] += atom.partial_charge * w;
        });
    }

    // todo: QC this placement.
    // Forward FFT  ρ(k)
    // build rho_c as Complex
    let mut rho_c: Vec<Complex<f64>> = rho
        .into_iter()
        .map(|r| Complex { re: r, im: 0.0 })
        .collect();

    // FORWARD 3D FFT
    fft3_inplace(
        &mut rho_c,
        nx,
        ny,
        nz,
        /*forward=*/ FftDirection::Forward,
    );

    // Multiply by influence function in k-space:
    // ϕ(k) = ρ(k) * 4π/k² * exp(-k²/(4α²)) / (|Bx|² |By|² |Bz|²) * (1/Vol)

    // --- Influence function & k-space gradient ---
    let mut ex_k = vec![Complex { re: 0.0, im: 0.0 }; nx * ny * nz];
    let mut ey_k = ex_k.clone();
    let mut ez_k = ex_k.clone();

    for kx in 0..nx {
        for ky in 0..ny {
            for kz in 0..nz {
                let n = idx(kx, ky, kz, ny, nz);
                if kx == 0 && ky == 0 && kz == 0 {
                    continue;
                }

                let kx_s = if kx <= nx / 2 {
                    kx as isize
                } else {
                    kx as isize - nx as isize
                };
                let ky_s = if ky <= ny / 2 {
                    ky as isize
                } else {
                    ky as isize - ny as isize
                };
                let kz_s = if kz <= nz / 2 {
                    kz as isize
                } else {
                    kz as isize - nz as isize
                };

                let k = Vec3::new(
                    (kx_s as f64) * TAU / ext.x,
                    (ky_s as f64) * TAU / ext.y,
                    (kz_s as f64) * TAU / ext.z,
                );
                let k2 = k.dot(k);

                // Gaussian Ewald factor
                let gk = (-k2 / (4.0 * alpha * alpha)).exp();

                // B-spline deconvolution (order=4)
                let bx = bspline_modulus(SPLINE_ORDER, kx, nx);
                let by = bspline_modulus(SPLINE_ORDER, ky, ny);
                let bz = bspline_modulus(SPLINE_ORDER, kz, nz);
                let mut b2 = (bx * bx) * (by * by) * (bz * bz);
                b2 = b2.max(1e-8); // simple stability clamp

                // Green’s function (no 1/Vol; we normalize after IFFT)
                let vol = ext.x * ext.y * ext.z; // compute once above the loops
                let green = (4.0 * std::f64::consts::PI) * gk / (k2 * b2 * vol);

                let phi_hat = rho_c[n] * green; // φ̂(k)

                // Ê = -i k φ̂
                ex_k[n] = Complex {
                    re: k.x * phi_hat.im,
                    im: -k.x * phi_hat.re,
                };
                ey_k[n] = Complex {
                    re: k.y * phi_hat.im,
                    im: -k.y * phi_hat.re,
                };
                ez_k[n] = Complex {
                    re: k.z * phi_hat.im,
                    im: -k.z * phi_hat.re,
                };
            }
        }
    }

    // IFFT each component and normalize
    fft3_inplace(&mut ex_k, nx, ny, nz, FftDirection::Inverse);
    fft3_inplace(&mut ey_k, nx, ny, nz, FftDirection::Inverse);
    fft3_inplace(&mut ez_k, nx, ny, nz, FftDirection::Inverse);

    let norm = 1.0 / (nx * ny * nz) as f64;
    let ex: Vec<f64> = ex_k.into_iter().map(|c| c.re * norm).collect();
    let ey: Vec<f64> = ey_k.into_iter().map(|c| c.re * norm).collect();
    let ez: Vec<f64> = ez_k.into_iter().map(|c| c.re * norm).collect();
    // --- end k-space path ---

    // Gather forces back to atoms (same B‑spline weights)
    let mut forces_dyn_static = vec![Vec3::new_zero(); atoms_static_dy.len()];
    for (i, atom) in atoms_static_dy.iter().enumerate() {
        let mut e = Vec3::new_zero();
        for_each_stencil(atom, cell, ext, nx, ny, nz, |ix, iy, iz, w| {
            let id = idx(ix, iy, iz, ny, nz);
            e.x += w * ex[id];
            e.y += w * ey[id];
            e.z += w * ez[id];
        });
        forces_dyn_static[i] = e * atom.partial_charge;
    }

    let mut forces_water = vec![Vec3::new_zero(); atoms_water.len()];
    for (i, atom) in atoms_water.iter().enumerate() {
        let mut e = Vec3::new_zero();
        for_each_stencil(atom, cell, ext, nx, ny, nz, |ix, iy, iz, w| {
            let id = idx(ix, iy, iz, ny, nz);
            e.x += w * ex[id];
            e.y += w * ey[id];
            e.z += w * ez[id];
        });
        forces_water[i] = e * atom.partial_charge;
    }

    PmeLrForces {
        dyn_static: forces_dyn_static,
        water: forces_water,
    }
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

fn bspline_modulus(order: usize, k: usize, n: usize) -> f64 {
    if k == 0 {
        return 1.0;
    }
    let x = std::f64::consts::PI * (k as f64) / (n as f64);
    let s = x.sin() / x; // sinc(πk/N)
    s.powi(order as i32)
}

fn fft3_inplace(a: &mut [Complex<f64>], nx: usize, ny: usize, nz: usize, fft_dir: FftDirection) {
    let mut planner = FftPlanner::new();
    let fft_z = planner.plan_fft(nz, fft_dir);
    let fft_y = planner.plan_fft(ny, fft_dir);
    let fft_x = planner.plan_fft(nx, fft_dir);

    // work buffers
    let mut line: Vec<Complex<f64>>;

    // Z-transform on each (x,y) line
    for x in 0..nx {
        for y in 0..ny {
            line = (0..nz).map(|z| a[idx(x, y, z, ny, nz)]).collect();
            fft_z.process(&mut line);
            for (z, v) in line.into_iter().enumerate() {
                a[idx(x, y, z, ny, nz)] = v;
            }
        }
    }

    // Y-transform on each (x,z) line
    for x in 0..nx {
        for z in 0..nz {
            line = (0..ny).map(|y| a[idx(x, y, z, ny, nz)]).collect();
            fft_y.process(&mut line);
            for (y, v) in line.into_iter().enumerate() {
                a[idx(x, y, z, ny, nz)] = v;
            }
        }
    }

    // X-transform on each (y,z) line
    for y in 0..ny {
        for z in 0..nz {
            line = (0..nx).map(|x| a[idx(x, y, z, ny, nz)]).collect();
            fft_x.process(&mut line);
            for (x, v) in line.into_iter().enumerate() {
                a[idx(x, y, z, ny, nz)] = v;
            }
        }
    }
}

pub fn force_coulomb_ewald_complement(
    dir: Vec3, // r̂
    r: f64,    // |r|
    qi: f64,
    qj: f64,
    alpha: f64,
) -> Vec3 {
    // F_comp = k * qi*qj * [ erf(αr)/r^2 - (2α/√π) e^{-(αr)^2}/r ] * r̂
    // With your charge scaling, k = 1 in internal units (as in your real-space fn).
    let ar = alpha * r;
    // todo: Cache this.
    let erfc_comp = FRAC_2_SQRT_PI * (-ar * ar).exp();

    let term = erf(ar) / (r * r) - erfc_comp * alpha / r;
    dir * (qi * qj * term)
}

impl MdState {
    pub fn apply_long_range_recip_forces(&mut self) {
        // Long‑range reciprocal‑space term (PME / SPME), both static and dynamic.
        // Build a temporary Vec with *all* charges so the mesh sees both
        // movable and rigid atoms.  We only add forces back to dynamic atoms. This section
        // does not use neighbor lists.
        let n_dynamic = self.atoms.len();
        let mut all_atoms = Vec::with_capacity(n_dynamic + self.atoms_static.len());

        all_atoms.extend(self.atoms.iter().cloned());
        all_atoms.extend(self.atoms_static.iter().cloned());

        // These are the water atoms that have Coulomb force; not O.
        // Separate from all_atoms because they make a [&AtomDynamics] instead of &[AtomDynamics].
        let mut atoms_water = Vec::with_capacity(self.water.len() * 3);
        for mol in &self.water {
            atoms_water.extend([&mol.m, &mol.h0, &mol.h1]);
        }

        let rec_forces = pme_long_range_forces(
            &all_atoms,   // dynamic+static as &[AtomDynamics]
            &atoms_water, // [&AtomDynamics] for (M,H0,H1) per water
            &self.cell,
            EWALD_ALPHA,
            PME_MESH_SPACING,
        );

        // (1) dynamic atoms
        for (atom, f_rec) in self
            .atoms
            .iter_mut()
            .zip(rec_forces.dyn_static.iter().take(n_dynamic))
        {
            atom.accel += *f_rec; // mass divide happens later in step()
        }

        if self.water_pme_sites_forces.len() != self.water.len() {
            self.water_pme_sites_forces
                .resize(self.water.len(), [Vec3::new_zero(); 3]);
        }

        for (iw, chunk) in rec_forces.water.chunks_exact(3).enumerate() {
            // order must match how you built atoms_water: [M, H0, H1]
            self.water_pme_sites_forces[iw] = [chunk[0], chunk[1], chunk[2]];
        }

        // === 1-4 reciprocal-space scaling via complement (pairwise) ==================
        // Goal: net Coulomb(1-4) = SCALE_COUL_14 * (real + reciprocal).
        // We already scaled the real-space piece pairwise. Here we add
        // ΔF = (SCALE_COUL_14 - 1) * F_recip(pair), and F_recip(pair) = F_comp(pair).
        let corr = crate::dynamics::non_bonded::SCALE_COUL_14 - 1.0; // e.g., 1/1.2 - 1 = -1/6

        for &(i, j) in &self.nonbonded_scaled {
            let rij = self
                .cell
                .min_image(self.atoms[i].posit - self.atoms[j].posit);

            let r2 = rij.magnitude_squared();
            if r2 < 1e-12 {
                continue;
            }
            let r = r2.sqrt();
            let dir = rij / r;

            let qi = self.atoms[i].partial_charge;
            let qj = self.atoms[j].partial_charge;

            let f_comp = force_coulomb_ewald_complement(dir, r, qi, qj, EWALD_ALPHA);
            let df = f_comp * corr;

            // Newton's third law
            self.atoms[i].accel += df;
            self.atoms[j].accel -= df;

            self.barostat.virial_pair_kcal += rij.dot(df);
        }
        // === end 1-4 PME correction ===================================================
    }
}
