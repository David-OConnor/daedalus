// This module contains constants and utility functions related to the kernels we use.


// Allows easy switching between float and double.
using dtype = float;
using dtype3 = float3;


__device__
const dtype SOFTENING_FACTOR_SQ = 0.000000000001f;

// __device__
// const dtype EPS_DIV0 = 0.00000000001f;


__device__
dtype calc_dist(dtype3 point0, dtype3 point1) {
    dtype3 diff;
    diff.x = point0.x - point1.x;
    diff.y = point0.y - point1.y;
    diff.z = point0.z - point1.z;

    return std::sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
}


__device__
dtype coulomb(dtype3 q0, dtype3 q1, dtype charge) {
    float r = calc_dist(q0, q1);

    return 1.f * charge / (r + SOFTENING_FACTOR_SQ);
}

__device__
dtype3 f_coulomb(dtype3 acc_dir, dtype src_q, dtype dst_q, dtype dist) {
    dtype acc_mag = src_q * dst_q / (dist * dist + SOFTENING_FACTOR_SQ);

    dtype3 result;
    result.x = acc_dir.x * acc_mag;
    result.y = acc_dir.y * acc_mag;
    result.z = acc_dir.z * acc_mag;

    return result;
}

__device__
dtype lj_potential(
    dtype3 posit_0,
    dtype3 posit_1,
    dtype sigma,
    dtype epsilon
) {
    dtype r = calc_dist(posit_0, posit_1);

    dtype sr = sigma / r;
    dtype sr6 = powf(sr, 6.);
    dtype sr12 = sr6 * sr6;

    return 4.0f * epsilon * (sr12 - sr6);
}