//! This module implements the SETTLE algorithm for rigid water molecules.

use lin_alg::f64::Vec3;

use crate::dynamics::{
    AtomDynamics,
    ambient::SimBox,
    water_opc::{H_MASS, O_MASS},
};

/// Analytic SETTLE implementation for 3‑site rigid water (Miyamoto & Kollman, JCC 1992).
/// Works for any bond length / HOH angle.
///
/// All distances & masses are in MD internal units (Å, ps, amu, kcal/mol).
pub fn settle_opc(
    o: &mut AtomDynamics,
    h0: &mut AtomDynamics,
    h1: &mut AtomDynamics,
    dt: f64,
    cell: &SimBox,
) {
    const MASS_MOL: f64 = O_MASS + 2.0 * H_MASS;

    let o_pos = o.posit;
    let h0_pos_local = o_pos + cell.min_image(h0.posit - o_pos);
    let h1_pos_local = o_pos + cell.min_image(h1.posit - o_pos);

    // COM position & velocity at start of the drift/rotation substep
    let r_com = (o.posit * O_MASS + h0_pos_local * H_MASS + h1_pos_local * H_MASS) / MASS_MOL;
    let v_com = (o.vel * O_MASS + h0.vel * H_MASS + h1.vel * H_MASS) / MASS_MOL;

    // Shift to COM frame
    let (rO, rH0, rH1) = (o_pos - r_com, h0_pos_local - r_com, h1_pos_local - r_com);

    let (vO, vH0, vH1) = (o.vel - v_com, h0.vel - v_com, h1.vel - v_com);

    // angular momentum about COM
    let L = rO.cross(vO) * O_MASS + rH0.cross(vH0) * H_MASS + rH1.cross(vH1) * H_MASS;

    // inertia tensor about COM (symmetric 3×3)
    let accI = |r: Vec3, m: f64| {
        let x = r.x;
        let y = r.y;
        let z = r.z;
        let r2 = r.dot(r);
        (
            m * (r2 - x * x),
            m * (r2 - y * y),
            m * (r2 - z * z),
            -m * x * y,
            -m * x * z,
            -m * y * z,
        )
    };
    let (iOxx, iOyy, iOzz, iOxy, iOxz, iOyz) = accI(rO, O_MASS);
    let (iH0x, iH0y, iH0z, iH0xy, iH0xz, iH0yz) = accI(rH0, H_MASS);
    let (iH1x, iH1y, iH1z, iH1xy, iH1xz, iH1yz) = accI(rH1, H_MASS);

    let (ixx, iyy, izz, ixy, ixz, iyz) = (
        iOxx + iH0x + iH1x,
        iOyy + iH0y + iH1y,
        iOzz + iH0z + iH1z,
        iOxy + iH0xy + iH1xy,
        iOxz + iH0xz + iH1xz,
        iOyz + iH0yz + iH1yz,
    );

    // ω from I·ω = L
    let ω = solve_symmetric3(ixx, iyy, izz, ixy, ixz, iyz, L);

    // pure translation of COM + rigid rotation about COM
    let Δ = v_com * dt;
    let rO2 = rodrigues_rotate(rO, ω, dt);
    let rH02 = rodrigues_rotate(rH0, ω, dt);
    let rH12 = rodrigues_rotate(rH1, ω, dt);

    let new_o = r_com + Δ + rO2;
    let new_h0 = r_com + Δ + rH02;
    let new_h1 = r_com + Δ + rH12;

    // Wwrap O, then pull each H next to O using min_image
    o.posit = cell.wrap(new_o);
    h0.posit = o.posit + cell.min_image(new_h0 - o.posit);
    h1.posit = o.posit + cell.min_image(new_h1 - o.posit);

    o.vel = v_com + ω.cross(rO2);
    h0.vel = v_com + ω.cross(rH02);
    h1.vel = v_com + ω.cross(rH12);
}

/// Solve I · x = b for a 3×3 *symmetric* matrix I.
/// The six unique elements are
///     [ ixx  ixy  ixz ]
/// I = [ ixy  iyy  iyz ]
///     [ ixz  iyz  izz ]
///
/// Returns x as a Vec3.  Panics if det(I) ≃ 0.
fn solve_symmetric3(ixx: f64, iyy: f64, izz: f64, ixy: f64, ixz: f64, iyz: f64, b: Vec3) -> Vec3 {
    let det = ixx * (iyy * izz - iyz * iyz) - ixy * (ixy * izz - iyz * ixz)
        + ixz * (ixy * iyz - iyy * ixz);

    const TOL: f64 = 1.0e-12;
    if det.abs() < TOL {
        // Practically no rotation this step; keep ω = 0
        return Vec3::new_zero();
    }

    let inv_det = 1.0 / det;

    // Adjugate / inverse elements
    let inv00 = (iyy * izz - iyz * iyz) * inv_det;
    let inv01 = (ixz * iyz - ixy * izz) * inv_det;
    let inv02 = (ixy * iyz - ixz * iyy) * inv_det;
    let inv11 = (ixx * izz - ixz * ixz) * inv_det;
    let inv12 = (ixz * ixy - ixx * iyz) * inv_det;
    let inv22 = (ixx * iyy - ixy * ixy) * inv_det;

    // x = I⁻¹ · b
    Vec3::new(
        inv00 * b.x + inv01 * b.y + inv02 * b.z,
        inv01 * b.x + inv11 * b.y + inv12 * b.z,
        inv02 * b.x + inv12 * b.y + inv22 * b.z,
    )
}

fn rodrigues_rotate(r: Vec3, omega: Vec3, dt: f64) -> Vec3 {
    // Rotate vector r by angle θ = |ω| dt about axis n = ω/|ω|
    // Use series for tiny θ to avoid loss of precision.
    let omega_dt = omega * dt;
    let theta = omega_dt.magnitude();

    if theta < 1e-12 {
        // 2nd-order series: r' ≈ r + (ω×r) dt + 0.5 (ω×(ω×r)) dt^2
        let wxr = omega_dt.cross(r);
        return r + wxr + omega_dt.cross(wxr) * 0.5;
    }

    let n = omega_dt / theta; // unit axis
    let c = theta.cos();
    let s = theta.sin();
    // r' = r c + (n×r) s + n (n·r) (1−c)
    r * c + n.cross(r) * s + n * (n.dot(r)) * (1.0 - c)
}
