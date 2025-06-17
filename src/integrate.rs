//! Numerical integration techniques

use lin_alg::{f32::Vec3 as Vec3F32, f64::Vec3};
use na_seq::Element;

use crate::dynamics::AtomDynamics;

/// Compute acceleration, position, and velocity, using RK4.
/// The acc fn: (id, target posit, target element, target charge) -> Acceleration.
/// todo: C+P from causal grav.
/// todo: We may not use rk4.
pub fn integrate_rk4<F>(body_tgt: &mut AtomDynamics, id_tgt: usize, acc: &F, dt: f64)
where
    F: Fn(usize, Vec3, Element, f64) -> Vec3,
{
    // Step 1: Calculate the k-values for position and velocity
    body_tgt.accel = acc(id_tgt, body_tgt.posit, body_tgt.element, body_tgt.mass);

    let k1_v = body_tgt.accel * dt;
    let k1_pos = body_tgt.vel * dt;

    let body_pos_k2 = body_tgt.posit + k1_pos * 0.5;
    let k2_v = acc(id_tgt, body_pos_k2, body_tgt.element, body_tgt.mass) * dt;
    let k2_pos = (body_tgt.vel + k1_v * 0.5) * dt;

    let body_pos_k3 = body_tgt.posit + k2_pos * 0.5;
    let k3_v = acc(id_tgt, body_pos_k3, body_tgt.element, body_tgt.mass) * dt;
    let k3_pos = (body_tgt.vel + k2_v * 0.5) * dt;

    let body_pos_k4 = body_tgt.posit + k3_pos;
    let k4_v = acc(id_tgt, body_pos_k4, body_tgt.element, body_tgt.mass) * dt;
    let k4_pos = (body_tgt.vel + k3_v) * dt;

    // Step 2: Update position and velocity using weighted average of k-values
    body_tgt.vel += (k1_v + k2_v * 2. + k3_v * 2. + k4_v) / 6.;
    body_tgt.posit += (k1_pos + k2_pos * 2. + k3_pos * 2. + k4_pos) / 6.;
}

/// Standard verlet integration. (Without velocity). Returns position. (i.e. position next).
pub fn integrate_verlet(p: Vec3F32, p_prev: Vec3F32, a: Vec3F32, dt: f32) -> Vec3F32 {
    p * 2. - p_prev + a * dt.powi(2)
}

/// Standard verlet integration. (Without velocity). Returns position. (i.e. position next).
pub fn integrate_verlet_f64(p: Vec3, p_prev: Vec3, a: Vec3, dt: f64) -> Vec3 {
    p * 2. - p_prev + a * dt.powi(2)
}

/// Velocity-Verlet integration step.
///
/// p_{n+1} = p_n + v_n*dt + ½ a_n * dt²
/// v_{n+1} = v_n + ½ (a_n + a_{n+1}) * dt
///
/// # Returns `(p_next, v_next)`
pub fn integrate_velocity_verlet(p: Vec3, v: Vec3, a: Vec3, a_next: Vec3, dt: f64) -> (Vec3, Vec3) {
    let p_next = p + v * dt + a * (0.5 * dt.powi(2));
    let v_next = v + (a + a_next) * (0.5 * dt);

    (p_next, v_next)
}
