// This module contains constants and utility functions related to the kernels we use.


// Allows easy switching between float and double.
using dtype = float;
using dtype3 = float3;


__device__
const float SOFTENING_FACTOR_SQ = 0.000001f;

// __device__
// const float EPS_DIV0 = 0.00000000001f;

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
float3 coulomb_force(float3 posit_src, float3 posit_tgt, float q_src, float q_tgt) {
    float3 diff = posit_tgt - posit_src; // todo: QC direction
    float dist = std::sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);

    float3 dir = diff / dist;

    float mag = q_src * q_tgt / (dist * dist + SOFTENING_FACTOR_SQ);

    return dir * mag;
}

__device__
float lj_V(
    float3 posit_0,
    float3 posit_1,
    float sigma,
    float eps
) {
    float3 diff = posit_1 - posit_0;
    float r = std::sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);

    float sr = sigma / r;
    float sr6 = powf(sr, 6.);
    float sr12 = sr6 * sr6;

    return 4.0f * eps * (sr12 - sr6);
}

__device__
float3 lj_force(
    float3 posit_0,
    float3 posit_1,
    float sigma,
    float eps
) {
    float3 diff = posit_1 - posit_0;
    float r = std::sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);

    float3 dir = diff / r;

    float sr = sigma / r;
    float sr6 = powf(sr, 6.);
    float sr12 = sr6 * sr6;

    float mag = -24.0f * eps * (2. * sr12 - sr6) / (r * r);

    return dir * mag;
}