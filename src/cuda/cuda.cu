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
// C + P from `cuda.cu`
extern "C" __global__
void coulomb_kernel(
    dtype *out,
    dtype3 *posits_src,
    dtype3 *posits_tgt,
    dtype *charges,
    size_t N_srcs,
    size_t N_tgts
) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i_tgt = index; i_tgt < N_tgts; i_tgt += stride) {
        // Compute the sum serially, as it may not be possible to naively apply it in parallel,
        // and we may still be saturating GPU cores given the large number of samples.
        for (size_t i_src = 0; i_src < N_srcs; i_src++) {
            dtype3 posit_src = posits_src[i_src];
            dtype3 posit_tgt = posits_tgt[i_tgt];

            if (i_tgt < N_tgts) {
                out[i_tgt] += coulomb(posit_src, posit_tgt, charges[i_src]);
            }
        }
    }
}

extern "C" __global__
void lj_kernel(
    dtype *out,
    dtype3 *posits_0,
    dtype3 *posits_1,
    dtype *sigmas,
    dtype *epsilons,
    size_t N_srcs,
    size_t N_tgts
) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i_tgt = index; i_tgt < N_tgts; i_tgt += stride) {
        // Compute the sum serially, as it may not be possible to naively apply it in parallel,
        // and we may still be saturating GPU cores given the large number of tgts.
        for (size_t i_src = 0; i_src < N_srcs; i_src++) {
            dtype3 posit_0 = posits_0[i_src];
            dtype3 posit_1 = posits_1[i_tgt];

            // todo: Sort out the index here.
            dtype sigma = sigmas[0];
            dtype eps = epsilons[0];

            if (i_tgt < N_tgts) {
                out[i_tgt] += lj_potential(posit_0, posit_1, sigma, eps);
            }
        }
    }
}