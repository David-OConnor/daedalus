#ifndef UTIL_H
#define UTIL_H

// #include <cstdint>

// Forward declaration of CUDA types
struct float3;
struct double3;

// Allows easy switching between float and double.
// #define dtype double
// #define dtype3 double3
#define dtype float
#define dtype3 float3

// Declaration of constants
extern __device__ const dtype G;
extern __device__ const dtype SOFTENING_FACTOR_SQ;


// Function declarations
__device__ dtype3 acc_newton(dtype3 acc_dir, dtype src_mass, dtype dist);


#endif // UTIL_H
