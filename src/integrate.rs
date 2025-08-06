//! Numerical integration techniques

use lin_alg::{f32::Vec3 as Vec3F32, f64::Vec3};
use na_seq::Element;

use crate::dynamics::AtomDynamics;

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
