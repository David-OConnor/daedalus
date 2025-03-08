// This module contains constants and utility functions related to the kernels we use.


// Allows easy switching between float and double.
using dtype = float;
using dtype3 = float3;

__device__
const dtype G = 0.0000000000044984f; // todo: QC and keep in sync.
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
dtype3 acc_newton(dtype3 acc_dir, dtype src_mass, dtype dist) {
    dtype acc_mag = G * src_mass / (dist * dist + SOFTENING_FACTOR_SQ);

    dtype3 result;
    result.x = acc_dir.x * acc_mag;
    result.y = acc_dir.y * acc_mag;
    result.z = acc_dir.z * acc_mag;
    return result;
}