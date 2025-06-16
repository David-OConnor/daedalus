//! Numerical integration techniques

use lin_alg::{f32::Vec3, f64::Vec3 as Vec3F64};
use na_seq::Element;

use crate::docking::dynamics::BodyDockDynamics;

/// Compute acceleration, position, and velocity, using RK4.
/// The acc fn: (id, target posit, target element, target charge) -> Acceleration.
/// todo: C+P from causal grav.
pub fn integrate_rk4<F>(body_tgt: &mut BodyDockDynamics, id_tgt: usize, acc: &F, dt: f32)
where
    F: Fn(usize, Vec3, Element, f32) -> Vec3,
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

/// Integrates position and orientation of a rigid body using RK4.
pub fn integrate_rk4_rigid<F>(
    body: &mut crate::docking::dynamics::BodyRigid,
    force_torque: &F,
    dt: f32,
) where
    // force_torque(body) should return (net_force, net_torque) in *world* coordinates
    F: Fn(&crate::docking::dynamics::BodyRigid) -> (Vec3, Vec3),
{
    // -- k1 --------------------------------------------------------------------
    let (f1, τ1) = force_torque(body);
    let acc_lin1 = f1 / body.mass;
    let acc_ω1 =
        body.inertia_body_inv.clone() * (τ1 - body.ω.cross(body.inertia_body.clone() * body.ω));

    let k1_v = acc_lin1 * dt; // change in velocity
    let k1_pos = body.vel * dt; // change in position
    let k1_w = acc_ω1 * dt; // change in angular velocity
    let k1_q = crate::docking::dynamics::orientation_derivative(body.orientation, body.ω) * dt; // change in orientation

    // -- Prepare body_k2 (the state at t + dt/2) -------------------------------
    let mut body_k2 = body.clone();
    body_k2.vel = body.vel + k1_v * 0.5;
    body_k2.posit = body.posit + k1_pos * 0.5;
    body_k2.ω = body.ω + k1_w * 0.5;
    body_k2.orientation = (body.orientation + k1_q * 0.5).to_normalized();

    // -- k2 --------------------------------------------------------------------
    let (f2, τ2) = force_torque(&body_k2);
    let acc_lin2 = f2 / body_k2.mass;
    let acc_ω2 =
        body_k2.inertia_body_inv * (τ2 - body_k2.ω.cross(body_k2.inertia_body * body_k2.ω));

    let k2_v = acc_lin2 * dt;
    let k2_pos = body_k2.vel * dt;
    let k2_w = acc_ω2 * dt;
    let k2_q =
        crate::docking::dynamics::orientation_derivative(body_k2.orientation, body_k2.ω) * dt;

    // -- Prepare body_k3 (the state at t + dt/2) -------------------------------
    let mut body_k3 = body.clone();
    body_k3.vel = body.vel + k2_v * 0.5;
    body_k3.posit = body.posit + k2_pos * 0.5;
    body_k3.ω = body.ω + k2_w * 0.5;
    body_k3.orientation = (body.orientation + k2_q * 0.5).to_normalized();

    // -- k3 --------------------------------------------------------------------
    let (f3, τ3) = force_torque(&body_k3);
    let acc_lin3 = f3 / body_k3.mass;
    let acc_ω3 =
        body_k3.inertia_body_inv * (τ3 - body_k3.ω.cross(body_k3.inertia_body * body_k3.ω));

    let k3_v = acc_lin3 * dt;
    let k3_pos = body_k3.vel * dt;
    let k3_w = acc_ω3 * dt;
    let k3_q =
        crate::docking::dynamics::orientation_derivative(body_k3.orientation, body_k3.ω) * dt;

    // -- Prepare body_k4 (the state at t + dt) ---------------------------------
    let mut body_k4 = body.clone();
    body_k4.vel = body.vel + k3_v;
    body_k4.posit = body.posit + k3_pos;
    body_k4.ω = body.ω + k3_w;
    body_k4.orientation = (body.orientation + k3_q).to_normalized();

    // -- k4 --------------------------------------------------------------------
    let (f4, τ4) = force_torque(&body_k4);
    let acc_lin4 = f4 / body_k4.mass;
    let acc_ω4 =
        body_k4.inertia_body_inv * (τ4 - body_k4.ω.cross(body_k4.inertia_body * body_k4.ω));

    let k4_v = acc_lin4 * dt;
    let k4_pos = body_k4.vel * dt;
    let k4_w = acc_ω4 * dt;
    let k4_q =
        crate::docking::dynamics::orientation_derivative(body_k4.orientation, body_k4.ω) * dt;

    // -- Combine k1..k4 to get final update -------------------------------------
    // Final: v(t+dt) = v(t) + 1/6 * (k1_v + 2k2_v + 2k3_v + k4_v)
    body.vel += (k1_v + k2_v * 2.0 + k3_v * 2.0 + k4_v) / 6.0;
    body.posit += (k1_pos + k2_pos * 2.0 + k3_pos * 2.0 + k4_pos) / 6.0;

    body.ω += (k1_w + k2_w * 2.0 + k3_w * 2.0 + k4_w) / 6.0;

    // For orientation, do the same weighted average of the "k" changes,
    // then normalize the result at the end.
    let orientation_update = (k1_q + k2_q * 2. + k3_q * 2.0 + k4_q) / 6.0;
    body.orientation = (body.orientation + orientation_update).to_normalized();
}

/// Standard verlet integration. (Without velocity). Returns position. (i.e. position next).
pub fn integrate_verlet(p: Vec3, p_prev: Vec3, a: Vec3, dt: f32) -> Vec3 {
    p * 2. - p_prev + a * dt.powi(2)
}

/// Standard verlet integration. (Without velocity). Returns position. (i.e. position next).
pub fn integrate_verlet_f64(p: Vec3F64, p_prev: Vec3F64, a: Vec3F64, dt: f64) -> Vec3F64 {
    p * 2. - p_prev + a * dt.powi(2)
}

/// Velocity-Verlet integration step.
///
/// p_{n+1} = p_n + v_n*dt + ½ a_n * dt²
/// v_{n+1} = v_n + ½ (a_n + a_{n+1}) * dt
///
/// # Returns `(p_next, v_next)`
pub fn integrate_velocity_verlet(
    p: Vec3F64,
    v: Vec3F64,
    a: Vec3F64,
    a_next: Vec3F64,
    dt: f64,
) -> (Vec3F64, Vec3F64) {
    let p_next = p + v * dt + a * (0.5 * dt.powi(2));
    let v_next = v + (a + a_next) * (0.5 * dt);

    (p_next, v_next)
}
