// #include <math.h>
// #include <initializer_list>

// todo: A/R
// #include <math.h>

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