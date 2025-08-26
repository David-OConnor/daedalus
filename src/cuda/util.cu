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

// For returning both from a function.
struct ForceEnergy {
    float3 force;
    float energy;
};

// Apparently normally adding to output can cause race conditions.
__device__ __forceinline__ void atomicAddFloat3(float3* addr, const float3 v) {
    atomicAdd(&addr->x, v.x);
    atomicAdd(&addr->y, v.y);
    atomicAdd(&addr->z, v.z);
}

__device__
float3 coulomb_force(float3 posit_src, float3 posit_tgt, float q_src, float q_tgt) {
    const float3 diff = posit_tgt - posit_src; // todo: QC direction
    const float dist = std::sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);

    const float3 dir = diff / dist;

    const float mag = q_src * q_tgt / (dist * dist + SOFTENING_FACTOR_SQ);

    return dir * mag;
}

__device__ inline float3 min_image(float3 ext, float3 dv) {
    dv.x -= rintf(dv.x / ext.x) * ext.x;
    dv.y -= rintf(dv.y / ext.y) * ext.y;
    dv.z -= rintf(dv.z / ext.z) * ext.z;

    return dv;
}

// These params includes inv_r due to it being shared with LJ.
__device__
ForceEnergy coulomb_force_spme_short_range(
    float r,
    float inv_r,
    float3 dir,
    float q_0,
    float q_1,
    float cutoff_dist,
    float alpha
) {
    ForceEnergy result;

    // Outside cutoff: no short-range contribution
    if (r >= cutoff_dist) {
        result.force  = make_float3(0.f, 0.f, 0.f);
        result.energy = 0.f;
        return result;
    }

    const float alpha_r = alpha * r;
    const float erfc_term = erfcf(alpha_r);
    const float charge_term = q_0 * q_1;

    const float exp_term  = __expf(-(alpha_r * alpha_r));

    const float force_mag = charge_term * (erfc_term * inv_r * inv_r + 2.0f * alpha * exp_term * INV_SQRT_PI * inv_r);

    result.force = dir * force_mag;
    result.energy = charge_term * inv_r * erfc_term;
    return result;
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

// This assumes diff (and dir) is in order tgt - src.
// Different API.
__device__
ForceEnergy lj_force_v2(
    float3 diff,
    float r,
    float inv_r,
    float3 dir,
    float sigma,
    float eps
) {
    const float sr = sigma * inv_r;
    const float sr6 = powf(sr, 6.);
    const float sr12 = sr6 * sr6;

    const float mag = 24.0f * eps * (2. * sr12 - sr6) * inv_r;

    ForceEnergy result;
    result.force = dir * mag;
    result.energy = 4. * eps * (sr12 - sr6);

    return result;
}

// This assumes diff (and dir) is in order tgt - src.
__device__
ForceEnergy lj_force(
    float3 posit_tgt,
    float3 posit_src,
    float sigma,
    float eps
) {
    const float3 diff = posit_tgt - posit_src;
    const float r_sq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
    const float r = std::sqrt(r_sq);
    const float inv_r = 1.0f / r;

    const float3 dir = diff * inv_r;

    return lj_force_v2(
        diff,
        r,
        inv_r,
        dir,
        sigma,
        eps
    );
}