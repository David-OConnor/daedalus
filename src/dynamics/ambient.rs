//! For water molecules, the sim box, thermostat etc.

use lin_alg::f64::Vec3;
use na_seq::Element;
use rand::{Rng, prelude::ThreadRng};

use crate::dynamics::{
    ACCEL_CONVERSION_INV, KB, MdState, prep::HydrogenMdType, water_opc::WaterMol,
};

const BAR_PER_KCAL_MOL_PER_A3: f64 = 69476.95457055373;

/// Simulation cell (orthorhombic for now)
#[derive(Clone, Copy, Default)]
pub struct SimBox {
    pub bounds_low: Vec3,
    pub bounds_high: Vec3,
}

impl SimBox {
    pub fn extent(&self) -> Vec3 {
        self.bounds_high - self.bounds_low
    }

    /// Wrap an absolute coordinate back into the box (orthorhombic)
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

    pub fn center(&self) -> Vec3 {
        (self.bounds_low + self.bounds_high) * 0.5
    }

    // For use with the thermo/barostat.
    pub fn scale_isotropic(&mut self, lambda: f64) {
        let c = self.center();
        self.bounds_low = c + (self.bounds_low - c) * lambda;
        self.bounds_high = c + (self.bounds_high - c) * lambda;
    }
}

/// Isotropic Berendsen barostat (τ=relaxation time, κT=isothermal compressibility)
pub struct BerendsenBarostat {
    /// bar
    pub p_target: f64,
    /// picoseconds
    pub tau_p: f64,
    /// bar‑1   (≈4.5×10⁻⁵ for water at 300K, 1bar)
    pub kappa_t: f64,
    pub virial_pair_kcal: f64,
    pub rng: ThreadRng,
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
            virial_pair_kcal: 0.0, // todo: What should this init to?
            rng: rand::rng(),
        }
    }
}

impl BerendsenBarostat {
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

impl MdState {
    fn kinetic_energy_kcal(&self) -> f64 {
        // dynamic atoms + waters (skip massless EP)
        let mut ke = 0.0;
        for a in &self.atoms {
            ke += 0.5 * a.mass * a.vel.magnitude_squared();
        }

        for w in &self.water {
            ke += 0.5 * w.o.mass * w.o.vel.magnitude_squared();
            ke += 0.5 * w.h0.mass * w.h0.vel.magnitude_squared();
            ke += 0.5 * w.h1.mass * w.h1.vel.magnitude_squared();
        }
        ke * ACCEL_CONVERSION_INV
    }

    fn num_constraints_estimate(&self) -> usize {
        let mut c = 0;

        // (1) Rigid waters (O,H0,H1 rigid; EP is massless/virtual)
        // 3 constraints per water triad.
        c += 3 * self.water.len();

        // (2) SHAKE/RATTLE on X–H bonds among *dynamic* atoms (not counting waters here)
        // If hydrogens are constrained (your code calls shake_hydrogens() when HydrogenMdType::Fixed),
        // count ≈ number of H atoms among self.atoms (each has one constrained bond).
        let hydrogens_constrained = matches!(self.hydrogen_md_type, HydrogenMdType::Fixed(_));
        if hydrogens_constrained {
            c += self
                .atoms
                .iter()
                .filter(|a| a.element == Element::Hydrogen)
                .count();
        }

        // (3) If you have any extra explicit constraints elsewhere, add them here.
        // e.g., c += self.extra_constraints.len();

        c
    }

    fn dof_for_thermo(&self) -> usize {
        // 3 per massive DoF – constraints – (optionally) 3 for COM removal
        let mut n = 3 * (self.atoms.len() + 3 * self.water.len()); // O,H0,H1 only
        n -= self.num_constraints_estimate(); // eg 1 per SHAKE bond; 3 per rigid water
        n.saturating_sub(3) // if you zero COM momentum
    }

    // --- Berendsen thermostat (deterministic; use for quick equilibration) ---
    pub fn apply_thermostat_berendsen(&mut self, dt_ps: f64, t_target_k: f64, tau_t_ps: f64) {
        let ke = self.kinetic_energy_kcal();
        let dof = self.dof_for_thermo().max(1) as f64;
        let t_inst = 2.0 * ke / (dof * KB);
        let lam = ((1.0 + (dt_ps / tau_t_ps) * ((t_target_k / t_inst) - 1.0)).max(0.0)).sqrt();

        for a in &mut self.atoms {
            a.vel *= lam;
        }
        for w in &mut self.water {
            w.o.vel *= lam;
            w.h0.vel *= lam;
            w.h1.vel *= lam;
        }
        // EP/virtual sites have zero mass → leave them alone
    }

    // --- CSVR (Bussi) thermostat: canonical velocity-rescale ---
    pub fn apply_thermostat_csvr(&mut self, dt: f64, t_target_k: f64, tau_t_ps: f64) {
        use rand_distr::{ChiSquared, Distribution, StandardNormal};

        let dof = self.dof_for_thermo().max(2) as f64;
        let ke = self.kinetic_energy_kcal();
        let ke_bar = 0.5 * dof * KB * t_target_k;

        let c = (-dt / tau_t_ps).exp();
        // Draw the two random variates used in the exact CSVR update:
        let r: f64 = StandardNormal.sample(&mut self.barostat.rng); // N(0,1)
        let chi = ChiSquared::new(dof - 1.0)
            .unwrap()
            .sample(&mut self.barostat.rng); // χ²_{dof-1}

        // Discrete-time exact solution for the OU process in K (from Bussi 2007):
        // K' = K*c + ke_bar*(1.0 - c) * [ (chi + r*r)/dof ] + 2.0*r*sqrt(c*(1.0-c)*K*ke_bar/dof)
        let kprime = ke * c
            + ke_bar * (1.0 - c) * ((chi + r * r) / dof)
            + 2.0 * r * ((c * (1.0 - c) * ke * ke_bar / dof).sqrt());

        let lam = (kprime / ke).sqrt();

        for a in &mut self.atoms {
            a.vel *= lam;
        }
        for w in &mut self.water {
            w.o.vel *= lam;
            w.h0.vel *= lam;
            w.h1.vel *= lam;
        }
    }

    /// Instantaneous pressure in **bar** (pair virial only).
    /// P = (2K + W) / (3V)
    pub fn instantaneous_pressure_bar(&self) -> f64 {
        let vol_a3 = self.cell.volume(); // Å^3
        if !(vol_a3 > 0.0) {
            return f64::NAN;
        }
        let k_kcal = self.kinetic_energy_kcal(); // kcal/mol
        let w_kcal = self.barostat.virial_pair_kcal; // kcal/mol (pairs included this step)
        let p_kcal_per_a3 = (2.0 * k_kcal + w_kcal) / (3.0 * vol_a3);
        p_kcal_per_a3 * BAR_PER_KCAL_MOL_PER_A3
    }

    // call each step (or every nstpcouple steps) after thermostat
    pub fn apply_barostat_berendsen(&mut self, dt: f64) {
        let p_inst_bar = self.instantaneous_pressure_bar();
        let arg = 1.0
            - (dt / self.barostat.tau_p)
                * (p_inst_bar - self.barostat.p_target)
                * self.barostat.kappa_t;
        let lambda = arg.max(0.0).powf(1.0 / 3.0);

        // 1) scale the cell
        let c = self.cell.center();
        self.cell.scale_isotropic(lambda);

        // 2) scale flexible atom coordinates about c; scale velocities
        for a in &mut self.atoms {
            a.posit = c + (a.posit - c) * lambda;
            a.vel *= lambda;
        }
        for a in &mut self.atoms_static {
            a.posit = c + (a.posit - c) * lambda; // keep frozen but move with box
        }

        // 3) translate rigid waters by COM only; scale COM velocity
        for w in &mut self.water {
            let m_tot = w.o.mass + w.h0.mass + w.h1.mass; // EP massless
            let com =
                (w.o.posit * w.o.mass + w.h0.posit * w.h0.mass + w.h1.posit * w.h1.mass) / m_tot;
            let com_v = (w.o.vel * w.o.mass + w.h0.vel * w.h0.mass + w.h1.vel * w.h1.mass) / m_tot;

            let com_new = c + (com - c) * lambda;
            let d = com_new - com;

            w.o.posit += d;
            w.h0.posit += d;
            w.h1.posit += d;
            w.m.posit += d;
            let com_v_new = com_v * lambda;
            let dv = com_v_new - com_v;
            w.o.vel += dv;
            w.h0.vel += dv;
            w.h1.vel += dv;
            // EP velocity unused
        }
    }
}
