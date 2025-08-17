//! This module deals with the sim box, thermostat, and barostat.
//!
//! We set up Sim box, or cell, which is a rectangular prism (cube currently) which wraps at each face,
//! indefinitely. Its purpose is to simulate an infinity of water molecules. This box covers the atoms of interest,
//! but atoms in the neighboring (tiled) boxes influence the system as well. We use the concept of
//! a "minimum image" to find the closest copy of an item to a given site, among all tiled boxes.

use lin_alg::f64::Vec3;
use na_seq::Element;
use rand::prelude::ThreadRng;

use crate::dynamics::{ACCEL_CONVERSION_INV, AtomDynamics, KB, MdState, prep::HydrogenMdType};

// If we are setting up as a pad around all relevant atoms
const SIMBOX_PAD: f64 = 7.0; // Å

// If we are setting up as a fixed size.

// "The ewald real-space cutoff (and neighbor-list radius) must be 1/2 of the smallest box edge length
const SIMBOX_WIDTH: f64 = 24.; // Å
const SIMBOX_WIDTH_DIV2: f64 = SIMBOX_WIDTH / 2.0;

const BAR_PER_KCAL_MOL_PER_A3: f64 = 69476.95457055373;

/// This bounds the area where atoms are wrapped. For now at least, it is only
/// used for water atoms. Its size and position should be such as to keep the system
/// solvated. We may move it around during the sim.
#[derive(Clone, Copy, Default)]
pub struct SimBox {
    pub bounds_low: Vec3,
    pub bounds_high: Vec3,
    extent: Vec3,
}

impl SimBox {
    /// Set up to surround all atoms, with a pad. `atoms` is whichever we use to center the bix.
    pub fn new_padded(atoms: &[AtomDynamics]) -> Self {
        let (mut min, mut max) = (Vec3::splat(f64::INFINITY), Vec3::splat(f64::NEG_INFINITY));
        for a in atoms {
            min = min.min(a.posit);
            max = max.max(a.posit);
        }

        let bounds_low = min - Vec3::splat(SIMBOX_PAD);
        let bounds_high = max + Vec3::splat(SIMBOX_PAD);

        Self {
            bounds_low,
            bounds_high,
            extent: bounds_high - bounds_low,
        }
    }

    /// It may be worth this calculation determining the center, as we run it no more than once a step, and
    /// doing so may allow a much bigger savings by reducing the simbox size required.
    pub fn new_fixed_size(atoms: &[AtomDynamics]) -> Self {
        let mut center = Vec3::new_zero();
        for atom in atoms {
            center += atom.posit;
        }
        center /= atoms.len() as f64;

        let bounds_low = center - Vec3::splat(SIMBOX_WIDTH_DIV2);
        let bounds_high = center + Vec3::splat(SIMBOX_WIDTH_DIV2);

        Self {
            bounds_low,
            bounds_high,
            extent: bounds_high - bounds_low,
        }
    }

    /// Wrap an absolute coordinate back into the unit cell. (orthorhombic). We use it to
    /// keep arbitrary coordinates inside it.
    pub fn wrap(&self, p: Vec3) -> Vec3 {
        let ext = &self.extent;

        assert!(
            ext.x > 0.0 && ext.y > 0.0 && ext.z > 0.0,
            "SimBox edges must be > 0 (lo={:?}, hi={:?})",
            self.bounds_low,
            self.bounds_high
        );

        // rem_euclid keeps the value in [0, ext)
        Vec3::new(
            (p.x - self.bounds_low.x).rem_euclid(ext.x) + self.bounds_low.x,
            (p.y - self.bounds_low.y).rem_euclid(ext.y) + self.bounds_low.y,
            (p.z - self.bounds_low.z).rem_euclid(ext.z) + self.bounds_low.z,
        )
    }

    /// Minimum-image displacement vector. Find the closest copy
    /// of an item to a given site, among all tiled boxes. Maps a displacement vector to the closest
    /// periodic image. Allows distance measurements to use the shortest separation.
    pub fn min_image(&self, dv: Vec3) -> Vec3 {
        let ext = &self.extent;
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
        // Treat non-finite or tiny λ as "no-op"
        let lam = if lambda.is_finite() && lambda.abs() > 1.0e-12 {
            lambda
        } else {
            1.0
        };

        let c = self.center();
        let lo = c + (self.bounds_low - c) * lam;
        let hi = c + (self.bounds_high - c) * lam;

        // Enforce low <= high per component
        self.bounds_low = Vec3::new(lo.x.min(hi.x), lo.y.min(hi.y), lo.z.min(hi.z));
        self.bounds_high = Vec3::new(lo.x.max(hi.x), lo.y.max(hi.y), lo.z.max(hi.z));
        self.extent = self.bounds_high - self.bounds_low;

        debug_assert!({
            let ext = &self.extent;
            ext.x > 0.0 && ext.y > 0.0 && ext.z > 0.0
        });
    }
}

/// Isotropic Berendsen barostat (τ=relaxation time, κT=isothermal compressibility)
pub struct BerendsenBarostat {
    /// bar
    pub p_target: f64,
    /// picoseconds
    pub tau_pressure: f64,
    pub tau_temp: f64,
    /// bar‑1 (≈4.5×10⁻⁵ for water at 300K, 1bar)
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
            tau_pressure: 1.,
            tau_temp: 1.,
            // Isothermal compressibility of water at 298 K.
            kappa_t: 4.5e-5,
            virial_pair_kcal: 0.0, // Inits to 0 here, and at the start of each integrator step.
            rng: rand::rng(),
        }
    }
}

impl BerendsenBarostat {
    pub fn scale_factor(&self, p_inst: f64, dt: f64) -> f64 {
        // Δln V = (κ_T/τ_p) (P - P0) dt
        let mut dlnv = (self.kappa_t / self.tau_pressure) * (p_inst - self.p_target) * dt;

        // Cap per-step volume change (e.g., ≤10%)
        const MAX_DLNV: f64 = 0.10;
        dlnv = dlnv.clamp(-MAX_DLNV, MAX_DLNV);

        // λ = exp(ΔlnV/3) — strictly positive and well-behaved
        (dlnv / 3.0).exp()
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

    // --- CSVR (Bussi) thermostat: canonical velocity-rescale ---
    pub fn apply_thermostat_csvr(&mut self, dt: f64, t_target_k: f64) {
        use rand_distr::{ChiSquared, Distribution, StandardNormal};

        let dof = self.dof_for_thermo().max(2) as f64;
        let ke = self.kinetic_energy_kcal();
        let ke_bar = 0.5 * dof * KB * t_target_k;

        let c = (-dt / self.barostat.tau_temp).exp();
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
        if !p_inst_bar.is_finite() {
            return; // don't touch the box if pressure is bad
        }

        let lambda = self.barostat.scale_factor(p_inst_bar, dt);

        // 1) scale the cell
        let c = self.cell.center();
        self.cell.scale_isotropic(lambda);

        // 2) scale flexible atom coordinates about c; scale velocities
        for a in &mut self.atoms {
            a.posit = c + (a.posit - c) * lambda;
            a.vel *= lambda;
        }
        for a in &mut self.atoms_static {
            a.posit = c + (a.posit - c) * lambda;
        }

        // 3) translate rigid waters by COM only; scale COM velocity
        for w in &mut self.water {
            let m_tot = w.o.mass + w.h0.mass + w.h1.mass;
            let com =
                (w.o.posit * w.o.mass + w.h0.posit * w.h0.mass + w.h1.posit * w.h1.mass) / m_tot;
            let com_v = (w.o.vel * w.o.mass + w.h0.vel * w.h0.mass + w.h1.vel * w.h1.mass) / m_tot;

            let com_new = c + (com - c) * lambda;
            let d = com_new - com;

            w.o.posit += d;
            w.h0.posit += d;
            w.h1.posit += d;
            w.m.posit += d;

            let dv = com_v * lambda - com_v;
            w.o.vel += dv;
            w.h0.vel += dv;
            w.h1.vel += dv;
        }
    }
}
