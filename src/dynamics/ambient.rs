//! For water molecules, the sim box, thermostat etc.

use lin_alg::f64::{Quaternion, Vec3};
use na_seq::Element;

use crate::dynamics::{AtomDynamics, MdState};

/// Simulation cell (orthorhombic for now)
#[derive(Clone, Copy, Default)]
pub struct SimBox {
    pub bounds_low: Vec3,
    pub bounds_high: Vec3,
}

impl SimBox {
    #[inline]
    pub fn extent(&self) -> Vec3 {
        self.bounds_high - self.bounds_low
    }

    /// wrap an absolute coordinate back into the box (orthorhombic)
    #[inline]
    pub fn wrap(&self, p: Vec3) -> Vec3 {
        let ext = self.extent();

        assert!(
            ext.x > 0.0 && ext.y > 0.0 && ext.z > 0.0,
            "SimBox edges must be > 0 (lo={:?}, hi={:?})",
            self.bounds_low,
            self.bounds_high
        );

        // rem_euclid keeps the value in [0, ext)
        let wrapped = Vec3::new(
            (p.x - self.bounds_low.x).rem_euclid(ext.x) + self.bounds_low.x,
            (p.y - self.bounds_low.y).rem_euclid(ext.y) + self.bounds_low.y,
            (p.z - self.bounds_low.z).rem_euclid(ext.z) + self.bounds_low.z,
        );

        wrapped
    }

    /// minimum-image displacement vector (no √)
    #[inline]
    pub fn min_image(&self, dv: Vec3) -> Vec3 {
        let ext = self.extent();
        debug_assert!(ext.x > 0.0 && ext.y > 0.0 && ext.z > 0.0);

        Vec3::new(
            dv.x - (dv.x / ext.x).round() * ext.x,
            dv.y - (dv.y / ext.y).round() * ext.y,
            dv.z - (dv.z / ext.z).round() * ext.z,
        )
    }
}
