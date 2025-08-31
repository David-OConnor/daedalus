// #include <math.h>
#include <initializer_list>

// todo: A/R
// #include <math.h>
#include <math_constants.h> // CUDART_PI_F

#include "util.cu"

// In this approach, we parallelize operations per sample, but run the
// charge computations in serial, due to the cumulative addition step. This appears
// to be much faster in practice, likely due to the addition being offloaded
// to the CPU in the other approach.
extern "C" __global__
void coulomb_force_kernel(
    float3* out,
    const float3* posits_src,
    const float3* posits_tgt,
    const float* charges,
    size_t N_srcs,
    size_t N_tgts
) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i_tgt = index; i_tgt < N_tgts; i_tgt += stride) {
        const float3 posit_tgt = posits_tgt[i_tgt];

        // Compute the sum serially, as it may not be possible to naively apply it in parallel,
        // and we may still be saturating GPU cores given the large number of targets.
        // todo: QC that.
        for (size_t i_src = 0; i_src < N_srcs; i_src++) {
            const float3 posit_src = posits_src[i_src];

            if (i_tgt < N_tgts) {
                // todo: Likely need two sets of charges too.
                out[i_tgt] = out[i_tgt] + coulomb_force(posit_src, posit_tgt, charges[i_src], charges[i_tgt]);
            }
        }
    }
}

// extern "C" __global__
// void coulomb_force_spme_short_range_kernel_pairwise(
//     float3* out,
//     const float3* posits_tgt,
//     const float3* posits_src,
//     const float* charges_tgt,
//     const float* charges_src,
//     size_t N,
//     float cutoff,
//     float alpha,
//     float3 cell     // {Lx, Ly, Lz}; set zeros to disable PBC
// ) {
//     // todo: Ensure you're handling periodic boundary condition correctly.
//     // DRY with the non-short range version for this block/thread/grid setup adn loop.
//
//     size_t index = blockIdx.x * blockDim.x + threadIdx.x;
//     size_t stride = blockDim.x * gridDim.x;
//
//     for (size_t i = index; i < N; i += stride) {
//         const float3 pt  = posits_tgt[i];
//         const float qt  = charges_tgt[i];
//
//         const float3 ps = posits_src[i];
//         const float qs  = charges_src[i];
//
//         const float3 force = coulomb_force_spme_short_range(ps, pt, qs, qt, alpha, cutoff, cell);
//
//         out[i] = out[i] + force;
//     }
// }

// Currently  unused
extern "C" __global__
void lj_V_kernel(
    float* out,
    const float3* posits_0,
    const float3* posits_1,
    const float* sigmas,
    const float* epsilons,
    size_t N_srcs,
    size_t N_tgts
) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i_tgt = index; i_tgt < N_tgts; i_tgt += stride) {
        // Compute the sum serially, as it may not be possible to naively apply it in parallel,
        // and we may still be saturating GPU cores given the large number of tgts.
        // todo: QC that.
        for (size_t i_src = 0; i_src < N_srcs; i_src++) {
            const float3 posit_0 = posits_0[i_src];
            const float3 posit_1 = posits_1[i_tgt];

            // todo: Sort out the index here.
            const float sigma = sigmas[0];
            const float eps = epsilons[0];

            if (i_tgt < N_tgts) {
                out[i_tgt] += lj_V(posit_0, posit_1, sigma, eps);
            }
        }
    }
}

// Currently unused
extern "C" __global__
void lj_force_kernel(
    float3* out,
    const float3* posits_src,
    const float3* posits_tgt,
    const float* sigmas,
    const float* epss,
    size_t N_srcs,
    size_t N_tgts
) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i_tgt = index; i_tgt < N_tgts; i_tgt += stride) {
        const float3 posit_tgt = posits_tgt[i_tgt];

        for (size_t i_src = 0; i_src < N_srcs; i_src++) {
            const float3 posit_src = posits_src[i_src];

            const size_t i_sig_eps = i_tgt * N_srcs + i_src;
            const float sigma = sigmas[i_sig_eps];
            const float eps = epss[i_sig_eps];

            if (i_tgt < N_tgts) {
                // Summing on GPU.
                out[i_tgt] = out[i_tgt] + lj_force(posit_tgt, posit_src, sigma, eps).force;
            }
        }
    }
}

// Handles LJ and Coulomb force, pairwise.
// This assumes inputs have already been organized and flattened. All inputs share the same index.
// Amber 1-2 and 1-3 exclusions are handled upstream.
extern "C" __global__
void nonbonded_force_kernel(
    // Out arrays and values
    float3* out_dyn,
    float3* out_water_o,
    float3* out_water_m,
    float3* out_water_h0,
    float3* out_water_h1,
    double* out_virial,  // Virial pair sum, used for the barostat.
    double* out_energy,
    // Pair-wise inputs
    const uint32_t* tgt_is,
    const uint32_t* src_is,
    const float3* posits_tgt,
    const float3* posits_src,
    const float* sigmas,
    const float* epss,
    const float* qs_tgt,
    const float* qs_src,
    // We use these two indices to know which output array to assign
    // forces to.
    const uint8_t* atom_types_tgt,
    const uint8_t* water_types_tgt,
    // For symmetric application
    const uint8_t* atom_types_src,
    const uint8_t* water_types_src,
    const uint8_t* scale_14s,
    const uint8_t* calc_ljs,
    const uint8_t* calc_coulombs,
    const uint8_t* symmetric,
    // Non-array inputs
    float3 cell_extent,
    float cutoff_ewald,
    float alpha_ewald,
    size_t N
) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // todo: When you apply this to water, you must use the unit cell
    // todo to take a min image of the diff, vice using it directly.

    for (size_t i = index; i < N; i += stride) {
        const float3 posit_tgt = posits_tgt[i];
        const float3 posit_src = posits_src[i];

        const float sigma = sigmas[i];
        const float eps = epss[i];

        const uint8_t scale_14 = scale_14s[i];

        float3 diff = posit_tgt - posit_src;
        diff = min_image(cell_extent, diff);

        // We set up r and its variants like this to share between the Coulomb and LJ
        // functions.
        const float r_sq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;

        // Protect against r ~ 0 (also skip exact self if arrays alias)
        if (r_sq < 1e-16f) {
            continue;
        }

        // `rsqrtf` is a fast/approximate CUDA function. Maybe worth revisiting if it introduces
        // errors.
        const float inv_r = rsqrtf(r_sq);
        const float r = r_sq * inv_r;

        const float3 dir = diff * inv_r;

        ForceEnergy f_lj;
        f_lj.force = make_float3(0.f, 0.f, 0.f);
        f_lj.energy = 0.f;

        if (calc_ljs[i]) {
            f_lj = lj_force_v2(diff, r, inv_r, dir, sigma, eps);
        }

        const float q_tgt = qs_tgt[i];
        const float q_src = qs_src[i];

        ForceEnergy f_coulomb;
        f_coulomb.force = make_float3(0.f, 0.f, 0.f);
        f_coulomb.energy = 0.f;

        if (calc_coulombs[i]) {
            f_coulomb = coulomb_force_spme_short_range(
                r,
                inv_r,
                dir,
                q_tgt,
                q_src,
                cutoff_ewald,
                alpha_ewald
            );
        }

        if (scale_14) {
            f_lj.force = f_lj.force * 0.5f;
            f_lj.energy = f_lj.energy * 0.5f;

            f_coulomb.force = f_coulomb.force * 0.833333333f;
            f_coulomb.energy = f_coulomb.energy * 0.833333333f;
        }

        const float3 f = f_lj.force + f_coulomb.force;
        const double e_pair = (double)f_lj.energy + (double)f_coulomb.energy;

        // Virial per pair · F
        double virial_pair = ((double)diff.x * (double)f.x + (double)diff.y * (double)f.y + (double)diff.z * (double)f.z);
        atomicAdd(out_virial, virial_pair);

        const uint32_t out_i = tgt_is[i];

        if (atom_types_tgt[i] == 0) {
            atomicAddFloat3(&out_dyn[out_i], f);
            // We don't currently track energy on water atoms. Keep this in sync
            // with application assumptions, and how you handle it on the CPU.
            atomicAdd(out_energy, e_pair);
        } else {
            if (water_types_tgt[i] == 1) {
                atomicAddFloat3(&out_water_o[out_i], f);
            } else if (water_types_tgt[i] == 2) {
                atomicAddFloat3(&out_water_m[out_i], f);
            } else if (water_types_tgt[i] == 3) {
                atomicAddFloat3(&out_water_h0[out_i], f);
            } else {
                atomicAddFloat3(&out_water_h1[out_i], f);
            }
        }

        if (symmetric[i]) {
            const uint32_t out_i_s = src_is[i];
            const float3 f_s = f * -1.0f;

            if (atom_types_src[i] == 0) {
                atomicAddFloat3(&out_dyn[out_i_s], f_s);
            } else {
                if (water_types_src[i] == 1) {
                    atomicAddFloat3(&out_water_o[out_i_s], f_s);
                } else if (water_types_src[i] == 2) {
                    atomicAddFloat3(&out_water_m[out_i_s], f_s);
                } else if (water_types_src[i] == 3) {
                    atomicAddFloat3(&out_water_h0[out_i_s], f_s);
                } else {
                    atomicAddFloat3(&out_water_h1[out_i_s], f_s);
                }
            }
        }
    }
}

// Perform the fourier transform required to compute electron density from reflection data.
// todo: This is currently unused.
extern "C" __global__
void reflection_transform_kernel(
    float* out,
    const float3* posits,
    const float* h,
    const float* k,
    const float* l,
    const float* phase,
    // pre-chosen amplitude (weighted or unweighted).
    const float* amp,
    size_t N
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

//      for (size_t i = index; i< N; i += stride) {
     for ( ; i < N; i += stride) {
         if (amp[i] == 0.0f) continue;

        //  2π(hx + ky + lz)  (negative sign because CCP4/Coot convention)
        const float arg = -TAU * (
            h[i] * posits[i].x +
            k[i] * posits[i].y +
            l[i] * posits[i].z
        );

        //  real part of  F · e^{iφ} · e^{iarg} = amp·cos(φ+arg)
        out[i] += amp[i]* cosf(phase[i] + arg);
    }
}

extern "C" __global__
void make_densities_kernel(
    float3* out_coords,
    float* out_densities,
    const uint3* triplets,
    const float3* atom_posits,
    const float* data,
    const float3 step_vec_0,
    const float3 step_vec_1,
    const float3 step_vec_2,
    const float3 origin,
    const float dist_thresh_sq,
    const size_t nx,
    const size_t ny,
    size_t N,
    size_t N_atom_posits
) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N; i += stride) {
        uint32_t kx = triplets[i].x;
        uint32_t ky = triplets[i].y;
        uint32_t kz = triplets[i].z;

        const size_t i_data = (kz * ny + ky) * nx + kx;
        float density = data[i_data];

        const float3 coords = origin + step_vec_0 * (float)kx + step_vec_1 * (float)ky + step_vec_2 * (float)kz;

        float nearest_dist_sq = 9999999.f;
        for (size_t j = 0; j < N_atom_posits; j++) {
            const float dx = atom_posits[j].x - coords.x;
            const float dy = atom_posits[j].y - coords.y;
            const float dz = atom_posits[j].z - coords.z;
            const float dist_sq = dx * dx + dy * dy + dz * dz;

            if (dist_sq < nearest_dist_sq) {
                nearest_dist_sq = dist_sq;
            }
        }

        if (nearest_dist_sq > dist_thresh_sq) {
            density = 0.0f;
        }

        out_coords[i] = coords;
        out_densities[i] = density;
    }
}