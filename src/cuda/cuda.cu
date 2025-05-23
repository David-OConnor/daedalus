// #include <math.h>
#include <initializer_list>

#include "util.cu"


// __device__
// void leaves(dtype3 posit_target,  )

// extern "C" __global__
// void acc_bh_kernel(
//     dtype *out,
//     dtype3 *nodes
//     dtype3 posit_target,
//     size_t id_target,
//     Vec3 *node, // todo temp
//     dtype theta,
//     size_t max_bodies_per_mode,
// ) {
// //     dtype3 acc_diff =
// //     dtype dist = calc_dist(); // todo: You are double-subtracting; don't do that.
// }


// In this approach, we parallelize operations per sample, but run the
// charge computations in serial, due to the cumulative addition step. This appears
// to be much faster in practice, likely due to the addition being offloaded
// to the CPU in the other approach.
extern "C" __global__
void coulomb_force_kernel(
    float3 *out,
    const float3 *posits_src,
    const float3 *posits_tgt,
    const float *charges,
    size_t N_srcs,
    size_t N_tgts
) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i_tgt = index; i_tgt < N_tgts; i_tgt += stride) {
        // Compute the sum serially, as it may not be possible to naively apply it in parallel,
        // and we may still be saturating GPU cores given the large number of targets.
        // todo: QC that.
        for (size_t i_src = 0; i_src < N_srcs; i_src++) {
            float3 posit_src = posits_src[i_src];
            float3 posit_tgt = posits_tgt[i_tgt];

            if (i_tgt < N_tgts) {
                // todo: Likely need two sets of charges too.
                out[i_tgt] = out[i_tgt] + coulomb_force(posit_src, posit_tgt, charges[i_src], charges[i_tgt]);
            }
        }
    }
}

extern "C" __global__
void lj_V_kernel(
    float *out,
    const float3 *posits_0,
    const float3 *posits_1,
    const float *sigmas,
    const float *epsilons,
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
            float3 posit_0 = posits_0[i_src];
            float3 posit_1 = posits_1[i_tgt];

            // todo: Sort out the index here.
            float sigma = sigmas[0];
            float eps = epsilons[0];

            if (i_tgt < N_tgts) {
                out[i_tgt] += lj_V(posit_0, posit_1, sigma, eps);
            }
        }
    }
}

extern "C" __global__
void lj_force_kernel(
    float3 *out,
    const float3 *posits_src,
    const float3 *posits_tgt,
    const float *sigmas,
    const float *epss,
    size_t N_srcs,
    size_t N_tgts
) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i_tgt = index; i_tgt < N_tgts; i_tgt += stride) {
        float3 posit_tgt = posits_tgt[i_tgt];

        for (size_t i_src = 0; i_src < N_srcs; i_src++) {
            float3 posit_src = posits_src[i_src];

            size_t i_sig_eps = i_tgt * N_srcs + i_src;
            float sigma = sigmas[i_sig_eps];
            float eps = epss[i_sig_eps];

            if (i_tgt < N_tgts) {
                // Summing on GPU.
                out[i_tgt] = out[i_tgt] + lj_force(posit_tgt, posit_src, sigma, eps);
            }
        }
    }
}

// Perform the fourier transform required to compute electron density from reflection data.
// todo: f32 ok?

extern "C" __global__
void reflection_transform_kernel(
    float *out,
    const float3 *posits,
    const float *h,
    const float *k,
    const float *l,
    const float *phase,
    // pre-chosen amplitude (weighted or unweighted).
    const float *amp,
    size_t N
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

//      for (size_t i = index; i< N; i += stride) {
     for ( ; i < N; i += stride) {
         if (amp[i] == 0.0f) continue;

        //  2π(hx + ky + lz)  (negative sign because CCP4/Coot convention)
        float arg = -TAU * (
            h[i] * posits[i].x +
            k[i] * posits[i].y +
            l[i] * posits[i].z
        );

        //  real part of  F · e^{iφ} · e^{iarg} = amp·cos(φ+arg)
        out[i] += amp[i]* cosf(phase[i] + arg);
    }
}