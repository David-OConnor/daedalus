//! This module contains a traditional molecular dynamics approach
//!
//! [Good article](https://www.owlposting.com/p/a-primer-on-molecular-dynamics)
//!
//! We are using f64, and CPU-only for now, unless we confirm f32 will work.
//! Maybe a mixed approach: Coordinates, velocities, and forces in 32-bit; sensitive global
//! reductions (energy, virial, integration) in 64-bit.
//!
//! We use Verlet integration. todo: Velocity verlet? Other techniques that improve and build upon it?

// todo: Integration: Consider Verlet or Langevin

//  todo: Pressure and temp variables required. Perhaps related to ICs of water?

// Note on timescale: Generally femtosecond (-15)

use std::collections::HashMap;

use lin_alg::f64::{Quaternion, Vec3};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use lin_alg::f64::{f64x4, f64x8, pack_slice};
use na_seq::{Element, element::LjTable};

use crate::{
    forces::{force_coulomb_f64, force_lj_f64},
    molecule::Atom,
};

// vacuum permittivity constant   (k_e = 1/(4π ε0))
// SI, matched with charge in e₀ & Å → kcal mol // todo: QC
const ε_0: f64 = 8.8541878128e-1;
const k_e: f32 = 332.0636; // kcal·Å /(mol·e²)
// const ε_0: f64 = 1.;

#[derive(Debug, Default)]
pub struct SnapshotDynamics {
    pub time: f64,
    // These are in a retained order so we don't need to store the Atom details in the snapshots.
    pub atom_posits: Vec<Vec3>,
    pub atom_velocities: Vec<Vec3>,
    // pub energy: Option<BindingEnergy>,
}

/// A trimmed-down atom for use with molecular dynamics.
struct AtomDynamics {
    pub posit: Vec3,
    pub vel: Vec3,
    pub accel: Vec3,
    pub mass: f64,        // Daltons
    pub element: Element, // e₀
    pub partial_charge: f64,
}

impl AtomDynamics {
    pub fn new(element: Element) -> Self {
        Self {
            posit: Vec3::new_zero(),
            vel: Vec3::new_zero(),
            accel: Vec3::new_zero(),
            mass: element.atomic_weight() as f64,
            partial_charge: 0.0,
            element,
        }
    }
}

impl From<&Atom> for AtomDynamics {
    fn from(atom: &Atom) -> Self {
        Self {
            posit: atom.posit.into(),
            vel: Vec3::new_zero(),
            accel: Vec3::new_zero(),
            // todo: Sort this out. What is the proton mass here.
            mass: atom.element.atomic_weight() as f64,
            element: atom.element,
            partial_charge: atom.partial_charge.unwrap() as f64,
        }
    }
}

/// Harmonic bond (AMBER-style).
#[derive(Debug, Clone, Copy)]
pub struct BondDynamics {
    pub atom_0: usize,
    pub atom_1: usize,
    pub k: f64,  // kcal mol⁻¹ Å⁻²
    pub r0: f64, // Å
}

/// Harmonic angle.
#[derive(Debug, Clone, Copy)]
pub struct Angle {
    pub i: usize,
    pub j: usize,
    pub k: usize,
    pub kθ: f64, // kcal mol⁻¹ rad⁻²
    pub θ0: f64, // rad
}

/// Periodic torsion: V(φ)=∑ (Vn/2)(1+cos(nφ−γ))
#[derive(Debug, Clone, Copy)]
pub struct Torsion {
    pub i: usize,
    pub j: usize,
    pub k: usize,
    pub l: usize,
    pub vn: f64, // kcal mol⁻¹
    pub n: u8,
    pub γ: f64, // rad
}

#[derive(Default)]
pub struct MdState {
    pub atoms: Vec<AtomDynamics>,
    pub bonds: Vec<BondDynamics>,
    pub angles: Vec<Angle>,
    pub torsions: Vec<Torsion>,
    pub lj_lut: LjTable,
    /// Sigmas and epsilons are Lennard Jones parameters. Flat here, with outer loop receptor.
    /// Flattened. Separate single-value array facilitate use in CUDA and SIMD, vice a tuple.
    pub lj_sigma: Vec<f64>,
    pub lj_eps: Vec<f64>,
    pub time: f64, // todo: is this dt?
    // #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    // pub lj_sigma_x8: Vec<f64x4>,
    // #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    // pub lj_eps_x8: Vec<f64x4>,
    pub snapshots: Vec<SnapshotDynamics>,
}

impl MdState {
    pub fn new(atoms: &[Atom], lj_table: &LjTable) -> Self {
        let atoms_dy = atoms.iter().map(|a| a.into()).collect();

        Self {
            atoms: atoms_dy,
            lj_lut: lj_table.clone(),
            ..Default::default()
        }
    }

    /// One **Velocity-Verlet** step (leap-frog style) of length `dt_fs` femtoseconds.
    pub fn step(&mut self, dt_fs: f64) {
        let dt = dt_fs * 1.0e-15; // s
        let dt_half = 0.5 * dt;

        // 1) First half-kick (v += a dt/2) and drift (x += v dt)

        // todo: Do we want traditional verlet instead?
        for a in &mut self.atoms {
            a.vel += a.accel * dt_half;
            a.posit += a.vel * dt;
        }

        // 2) Recalculate forces → update accelerations
        self.zero_acc();
        self.apply_bond_forces();
        self.apply_angle_forces();
        self.apply_torsion_forces();
        self.apply_nonbonded_forces();

        // 3) Second half-kick using new accelerations
        for a in &mut self.atoms {
            a.vel += a.accel * dt_half;
        }

        self.time += dt_fs;
    }

    fn zero_acc(&mut self) {
        for a in &mut self.atoms {
            a.accel = Vec3::new_zero();
        }
    }

    fn apply_bond_forces(&mut self) {
        for &BondDynamics {
            atom_0,
            atom_1,
            k,
            r0,
        } in &self.bonds
        {
            let (ai, aj) = split2_mut(&mut self.atoms, atom_0, atom_1);

            let diff = aj.posit - ai.posit;
            let dist = diff.magnitude();

            let Δ = dist - r0;
            // F = -dV/dr = -k 2Δ   (harmonic)
            let f_mag = -2.0 * k * Δ / dist.max(1e-12);
            let f = diff * f_mag;
            ai.accel -= f / ai.mass;
            aj.accel += f / aj.mass;
        }
    }

    fn apply_angle_forces(&mut self) {
        for &Angle { i, j, k, kθ, θ0 } in &self.angles {
            // todo: Use your existing dihedral angle code?
            let (ai, aj, ak) = split3_mut(&mut self.atoms, i, j, k);

            let dist_ij = ai.posit - aj.posit;
            let dist_kj = ak.posit - aj.posit;

            let cosθ =
                dist_ij.dot(dist_kj) / (dist_ij.magnitude() * dist_kj.magnitude()).max(1e-12);
            let θ = cosθ.acos();
            let Δ = θ - θ0;

            // simplified small-angle approximation: torque converted to force along bisectors
            let f_mag = -2.0 * kθ * Δ;
            let n = dist_ij.cross(dist_kj).to_normalized();
            let f_i = n.cross(dist_ij).to_normalized() * f_mag;
            let f_k = dist_kj.cross(n).to_normalized() * f_mag;

            ai.accel += f_i / ai.mass;
            ak.accel += f_k / ak.mass;
            aj.accel -= (f_i + f_k) / aj.mass;
        }
    }

    fn apply_torsion_forces(&mut self) {
        for &Torsion {
            i,
            j,
            k,
            l,
            vn,
            n,
            γ,
        } in &self.torsions
        {
            let (ai, aj, ak, al) = split4_mut(&mut self.atoms, i, j, k, l);

            let dist_ij = ai.posit - aj.posit;
            let dist_kj = ak.posit - aj.posit;
            let dist_kj2 = ak.posit - al.posit;

            let n1 = dist_ij.cross(dist_kj).to_normalized();
            let n2 = dist_kj.cross(dist_kj2).to_normalized();

            let φ = n1.dot(n2).clamp(-1.0, 1.0).acos();
            let dV_dφ = 0.5 * vn * (n as f64) * ((n as f64) * φ - γ).sin();

            // Approximate: project torque equally; full derivation omitted for brevity.
            let torque = dV_dφ;
            let f = n2 * torque;
            ai.accel += f / ai.mass;
            al.accel -= f / al.mass;
        }
    }

    fn apply_nonbonded_forces(&mut self) {
        let n = self.atoms.len();
        for i in 0..n - 1 {
            for j in i + 1..n {
                let (σ, ϵ) = self
                    .lj_lut
                    .get(&(self.atoms[i].element, self.atoms[j].element))
                    .unwrap();

                let diff = self.atoms[j].posit - self.atoms[i].posit;
                let dist = diff.magnitude();
                let dir = diff / dist;

                let f_lj = force_lj_f64(dir, dist, *σ as f64, *ϵ as f64);

                let f_coulomb = force_coulomb_f64(
                    dir,
                    dist,
                    self.atoms[i].partial_charge,
                    self.atoms[j].partial_charge,
                    0.0000001, // todo
                );

                let f = f_lj + f_coulomb;

                let accel_0 = f / self.atoms[i].mass;
                let accel_1 = f / self.atoms[j].mass;

                self.atoms[i].accel += accel_0;
                self.atoms[j].accel -= accel_1;
            }
        }
    }

    pub fn snapshot(&self) -> SnapshotDynamics {
        SnapshotDynamics {
            time: self.time,
            atom_posits: self.atoms.iter().map(|a| a.posit).collect(),
            atom_velocities: self.atoms.iter().map(|a| a.vel).collect(),
        }
    }
}

#[inline]
/// Mutable aliasing helpers.
fn split2_mut<T>(v: &mut [T], i: usize, j: usize) -> (&mut T, &mut T) {
    assert!(i != j);
    let (low, high) = if i < j { (i, j) } else { (j, i) };
    let (left, right) = v.split_at_mut(high);
    (&mut left[low], &mut right[0])
}

#[inline]
fn split3_mut<T>(v: &mut [T], i: usize, j: usize, k: usize) -> (&mut T, &mut T, &mut T) {
    let len = v.len();
    assert!(i < len && j < len && k < len, "index out of bounds");
    assert!(i != j && i != k && j != k, "indices must be distinct");

    // SAFETY: we just asserted that 0 <= i,j,k < v.len() and that they're all different.
    let ptr = v.as_mut_ptr();
    unsafe {
        let a = &mut *ptr.add(i);
        let b = &mut *ptr.add(j);
        let c = &mut *ptr.add(k);
        (a, b, c)
    }
}

#[inline]
pub fn split4_mut<T>(
    slice: &mut [T],
    i0: usize,
    i1: usize,
    i2: usize,
    i3: usize,
) -> (&mut T, &mut T, &mut T, &mut T) {
    // ---------- safety gates ------------------------------------------------
    let len = slice.len();
    assert!(
        i0 < len && i1 < len && i2 < len && i3 < len,
        "index out of bounds"
    );
    assert!(
        i0 != i1 && i0 != i2 && i0 != i3 && i1 != i2 && i1 != i3 && i2 != i3,
        "indices must be pair-wise distinct"
    );
    // ------------------------------------------------------------------------

    // SAFETY:
    //  * The bounds checks above guarantee each offset is within the slice.
    //  * The pair-wise distinct check guarantees no two &mut point to
    //    the same element, so we uphold Rust’s aliasing rules.
    unsafe {
        let base = slice.as_mut_ptr();
        (
            &mut *base.add(i0),
            &mut *base.add(i1),
            &mut *base.add(i2),
            &mut *base.add(i3),
        )
    }
}

// todo: A/R
pub struct Water {
    // todo: Track a posit and velocit for the whole molecule, and model with an
    // todo orientation quaternion?
    // pub posit: Vec3,
    // pub orientation: Quaternion,
    pub o: AtomDynamics,
    pub h0: AtomDynamics,
    pub h1: AtomDynamics,
}

/// Add water molecules
fn hydrate(pressure: f64, temp: f64, bounds: (Vec3, Vec3), n_mols: usize) -> Vec<Water> {
    let mut result = Vec::with_capacity(n_mols);
    // todo: TIP4P to start?
    for _ in 0..n_mols {
        result.push(Water {
            o: AtomDynamics {
                // todo
                posit: Vec3::new_zero(),
                // todo: Init vel based on temp and pressure?
                vel: Vec3::new_zero(),
                accel: Vec3::new_zero(),
                mass: 8.,
                element: Element::Oxygen,
                partial_charge: 0.,
            },
            h0: AtomDynamics {
                // todo
                posit: Vec3::new_zero(),
                // todo: Init vel based on temp and pressure?
                vel: Vec3::new_zero(),
                accel: Vec3::new_zero(),
                mass: 1.,
                element: Element::Hydrogen,
                partial_charge: 0.,
            },
            h1: AtomDynamics {
                // todo
                posit: Vec3::new_zero(),
                // todo: Init vel based on temp and pressure?
                vel: Vec3::new_zero(),
                accel: Vec3::new_zero(),
                mass: 1.,
                element: Element::Hydrogen,
                partial_charge: 0.,
            },
        })
    }

    result
}

// todo: One more?
