// This module contains constants and utility functions related to the kernels we use.


// Allows easy switching between float and double.
using dtype = float;
using dtype3 = float3;


__device__
const dtype SOFTENING_FACTOR_SQ = 0.000001f;

// __device__
// const dtype EPS_DIV0 = 0.00000000001f;

// Vector operations for float3
__device__ inline float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 operator-(const float3 &a, const float3 &b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 operator/(const float3 &a, const float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

__device__ inline float3 operator*(const float3 &a, const float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__
dtype3 f_coulomb(dtype3 posit_src, dtype3 posit_tgt, dtype q_src, dtype q_tgt) {
    dtype3 diff = posit_tgt - posit_src; // todo: QC direction
    dtype dist = std::sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);

    dtype3 dir = diff / dist;

    dtype magnitude = q_src * q_tgt / (dist * dist + SOFTENING_FACTOR_SQ);

    return dir * magnitude;
}

__device__
dtype lj_potential(
    dtype3 posit_0,
    dtype3 posit_1,
    dtype sigma,
    dtype eps
) {
    dtype3 diff = posit_0 - posit_0;
    dtype r = std::sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);

    dtype sr = sigma / r;
    dtype sr6 = powf(sr, 6.);
    dtype sr12 = sr6 * sr6;

    return 4.0f * eps * (sr12 - sr6);
}