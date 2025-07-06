//! For water molecules, the sim box, thermostat etc.

use lin_alg::f64::{Quaternion, Vec3};
use na_seq::Element;

use crate::dynamics::{AtomDynamics, MdState};

/// Simulation cell (orthorhombic for now)
#[derive(Clone, Copy, Default)]
pub struct SimBox {
    pub lo: Vec3,
    pub hi: Vec3,
}

impl SimBox {
    #[inline]
    pub fn extent(&self) -> Vec3 {
        self.hi - self.lo
    }

    /// wrap an absolute coordinate back into the box (orthorhombic)
    #[inline]
    pub fn wrap(&self, p: Vec3) -> Vec3 {
        let ext = self.extent();

        assert!(
            ext.x > 0.0 && ext.y > 0.0 && ext.z > 0.0,
            "SimBox edges must be > 0 (lo={:?}, hi={:?})",
            self.lo,
            self.hi
        );

        // rem_euclid keeps the value in [0, ext)
        let wrapped = Vec3::new(
            (p.x - self.lo.x).rem_euclid(ext.x) + self.lo.x,
            (p.y - self.lo.y).rem_euclid(ext.y) + self.lo.y,
            (p.z - self.lo.z).rem_euclid(ext.z) + self.lo.z,
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
