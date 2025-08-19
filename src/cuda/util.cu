// This module contains constants and utility functions related to the kernels we use.


// Allows easy switching between float and double.
using dtype = float;
using dtype3 = float3;


__device__
const float SOFTENING_FACTOR_SQ = 0.000001f;

__device__
const float TAU = 6.283185307179586f;

// 1/sqrt(pi)
__device__
// const float INV_SQRT_PI = 1.0f / sqrtf(CUDART_PI_F);
const float INV_SQRT_PI = 0.5641895835477563f;

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
    const float3 diff = posit_tgt - posit_src; // todo: QC direction
    const float dist = std::sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);

    const float3 dir = diff / dist;

    const float mag = q_src * q_tgt / (dist * dist + SOFTENING_FACTOR_SQ);

    return dir * mag;
}

// Minimum-image for orthorhombic box if box.{x,y,z} > 0
__device__ inline float3 min_image(float3 dv, float3 box) {
    if (box.x > 0.f && box.y > 0.f && box.z > 0.f) {
        dv.x -= rintf(dv.x / box.x) * box.x;
        dv.y -= rintf(dv.y / box.y) * box.y;
        dv.z -= rintf(dv.z / box.z) * box.z;
    }
    return dv;
}

__device__
float3 coulomb_force_spme_short_range(
    float3 diff,
    float r,
    float3 dir,
    float q_0,
    float q_1,
    float cutoff,
    float alpha,
    float3 cell     // {Lx, Ly, Lz}; set zeros to disable PBC
) {
    // Outside cutoff: no short-range contribution
    if (r >= cutoff) return make_float3(0.f, 0.f, 0.f);

    // Protect against r ~ 0 (also skip exact self if arrays alias)
    if (r < 1e-16f) return make_float3(0.f, 0.f, 0.f);

    const float inv_r = 1.0f / r;
    const float inv_r2 = inv_r * inv_r;

    const float alpha_r = alpha * r;
    const float erfc_term = erfcf(alpha_r);
    const float exp_term  = __expf(-(alpha_r * alpha_r));

    const float force_mag = q_0 * q_1 * (erfc_term * inv_r2 + 2.0f * alpha * exp_term * INV_SQRT_PI * inv_r);

    return dir * force_mag;
}

__device__
float lj_V(
    float3 posit_0,
    float3 posit_1,
    float sigma,
    float eps
) {
    const float3 diff = posit_1 - posit_0;
    const float r = std::sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);

    const float sr = sigma / r;
    const float sr6 = powf(sr, 6.);
    const float sr12 = sr6 * sr6;

    return 4.0f * eps * (sr12 - sr6);
}

__device__
float3 lj_force(
    float3 posit_tgt,
    float3 posit_src,
    float sigma,
    float eps
) {
    const float3 diff = posit_src - posit_tgt;
    const float r = std::sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);

    const float3 dir = diff / r;

    const float sr = sigma / r;
    const float sr6 = powf(sr, 6.);
    const float sr12 = sr6 * sr6;

    const float mag = -24.0f * eps * (2. * sr12 - sr6) / (r * r);

    return dir * mag;
}

// Different API.
__device__
float3 lj_force_v2(
    float3 diff,
    float r,
    float3 dir,
    float sigma,
    float eps
) {
    const float sr = sigma / r;
    const float sr6 = powf(sr, 6.);
    const float sr12 = sr6 * sr6;

    const float mag = -24.0f * eps * (2. * sr12 - sr6) / (r * r);

    return dir * mag;
}