#ifndef UTIL_H
#define UTIL_H

// #include <cstdint>

struct float3;
struct double3;

// Allows easy switching between float and double.
// #define dtype double
// #define dtype3 double3
#define dtype float
#define dtype3 float3

extern __device__ const dtype SOFTENING_FACTOR_SQ;


__device__ dtype3 f_coulomb(dtype3 acc_dir, dtype src_q, dst_q, dtype dist);
__device__ dtype lj_potential(dtype3 posit_0, dtype3 posit_1, dtype sigma, dtype epsilon);


#endif // UTIL_H
