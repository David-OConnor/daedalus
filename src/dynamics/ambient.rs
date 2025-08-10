//! For water molecules, the sim box, thermostat etc.

use lin_alg::f64::Vec3;

use crate::dynamics::water_opc::WaterMol;

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

    /// Minimum-image displacement vector (no √)
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

    pub fn volume(&self) -> f64 {
        (self.bounds_high.x - self.bounds_low.x).abs()
            * (self.bounds_high.y - self.bounds_low.y).abs()
            * (self.bounds_high.z - self.bounds_low.z).abs()
    }
}

/// Isotropic Berendsen barostat (τ=relaxation time, κT=isothermal compressibility)
#[derive(Clone, Copy)]
pub struct BerendsenBarostat {
    /// bar
    pub p_target: f64,
    /// picoseconds
    pub tau_p: f64,
    /// bar‑1   (≈4.5×10⁻⁵ for water at 300K, 1bar)
    pub kappa_t: f64,
}

impl Default for BerendsenBarostat {
    fn default() -> Self {
        Self {
            // Standard atmospheric pressure.
            p_target: 1.,
            // Relaxation time: 1 ps ⇒ gentle volume changes every few steps.
            tau_p: 1.,
            // Isothermal compressibility of water at 298 K.
            kappa_t: 4.5e-5,
        }
    }
}

impl BerendsenBarostat {
    #[inline]
    pub fn scale_factor(&self, p_inst: f64, dt: f64) -> f64 {
        // λ = [1 − dt/τ (P − P₀) κ_T]^{1/3}
        let arg = 1.0 - (dt / self.tau_p) * (p_inst - self.p_target) * self.kappa_t;
        arg.max(0.0).powf(1.0 / 3.0) // guard against negative round‑off
    }

    // todo: Apply beyond water.
    /// Apply volume/coordinate rescaling
    pub fn apply(&self, cell: &mut SimBox, mols: &mut [WaterMol], p_inst: f64, dt: f64) {
        let λ = self.scale_factor(p_inst, dt);

        // grow/shrink the box
        let centre = (cell.bounds_low + cell.bounds_high) * 0.5;
        cell.bounds_low = centre + (cell.bounds_low - centre) * λ;
        cell.bounds_high = centre + (cell.bounds_high - centre) * λ;

        // scale every atomic coordinate *about the same centre*
        for m in mols.iter_mut() {
            for a in [&mut m.o, &mut m.h0, &mut m.h1, &mut m.m] {
                a.posit = centre + (a.posit - centre) * λ;
                if a.mass.abs() > f64::EPSILON {
                    // skip EP and virtual sites
                    a.vel *= λ;
                }
            }
        }
    }
}
