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
    const float3* __restrict__ posits_src,
    const float3* __restrict__ posits_tgt,
    const float* __restrict__ charges,
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
//     float3* __restrict__ out,
//     const float3* __restrict__ posits_tgt,
//     const float3* __restrict__ posits_src,
//     const float* __restrict__ charges_tgt,
//     const float* __restrict__ charges_src,
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

extern "C" __global__
void lj_V_kernel(
    float* __restrict__ out,
    const float3* __restrict__ posits_0,
    const float3* __restrict__ posits_1,
    const float* __restrict__ sigmas,
    const float* __restrict__ epsilons,
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

extern "C" __global__
void lj_force_kernel(
    float3* __restrict__ out,
    const float3* __restrict__ posits_src,
    const float3* __restrict__ posits_tgt,
    const float* __restrict__ sigmas,
    const float* __restrict__ epss,
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
                out[i_tgt] = out[i_tgt] + lj_force(posit_tgt, posit_src, sigma, eps);
            }
        }
    }
}

// Handles LJ and Coulomb force, pairwise.
// Unlike some other , this assumes inputs have already been organized and flattened
// into target/source pairs. All inputs share the same index.
// Amber 1-2 and 1-3 exclusions are handled upstream.
extern "C" __global__
void nonbonded_force_kernel(
    float3* __restrict__ out,
    float* __restrict__ virial,  // Virial pair sum, used for the barostat.
    const float3* __restrict__ posits_tgt,
    const float3* __restrict__ posits_src,
    const float* __restrict__ sigmas,
    const float* __restrict__ epss,
    const float* __restrict__ qs_tgt,
    const float* __restrict__ qs_src,
    const uint8_t* __restrict__ scale_14s,
    float cutoff,
    float alpha,
    // todo: Cell A/R
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

        const float3 diff = posit_src - posit_tgt;
        const float r = std::sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
        const float3 dir = diff / r;

        float3 f_lj = lj_force_v2(diff, r, dir, sigma, eps);

        const float q_tgt = qs_tgt[i];
        const float q_src = qs_src[i];

        // todo: A/R
        const float3 cell = make_float3(0.f, 0.f, 0.f);

        float3 f_coulomb = coulomb_force_spme_short_range(
            diff,
            r,
            dir,
            q_tgt,
            q_src,
            cutoff,
            alpha,
            cell
        );

        if (scale_14) {
            f_lj = f_lj * 0.5f;
            f_coulomb = f_coulomb * 0.833333333f;
        }

        const float3 f = f_lj + f_coulomb;

        // Virial per pair (pair counted once): -0.5 * r · F
        // todo: Cell wrapping for water.
        float w_pair = -0.5f * (diff.x * f.x + diff.y * f.y + diff.z * f.z);
        atomicAdd(virial, w_pair);

        out[i] = out[i] + f;
    }
}

// Perform the fourier transform required to compute electron density from reflection data.
// todo: f32 ok?

extern "C" __global__
void reflection_transform_kernel(
    float* __restrict__ out,
    const float3* __restrict__ posits,
    const float* __restrict__ h,
    const float* __restrict__ k,
    const float* __restrict__ l,
    const float* __restrict__ phase,
    // pre-chosen amplitude (weighted or unweighted).
    const float* __restrict__ amp,
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