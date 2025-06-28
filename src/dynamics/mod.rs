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

// todo: Long-term, you will need to figure out what to run as f32 vice f64, especially
// todo for being able to run on GPU.

// Note on timescale: Generally femtosecond (-15)

use std::collections::{HashMap, HashSet};

use lin_alg::f64::{Quaternion, Vec3};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use lin_alg::f64::{Vec3x4, f64x4, f64x8, pack_slice};
use na_seq::{Element, element::LjTable};
use rand_distr::{Distribution, UnitSphere};

use crate::{
    docking::dynamics::Snapshot,
    forces::{force_coulomb, force_lj},
    molecule::{Atom, Bond},
};

// vacuum permittivity constant   (k_e = 1/(4π ε0))
// SI, matched with charge in e₀ & Å → kcal mol // todo: QC
const ε_0: f64 = 8.8541878128e-1;
const k_e: f32 = 332.0636; // kcal·Å /(mol·e²)
// const ε_0: f64 = 1.;

// Verlet list parameters
const CUTOFF: f64 = 12.0; // Å
const SKIN: f64 = 2.0; // Å – rebuild list if an atom moved >½·SKIN
const M_O: f64 = 15.999; // Da
const M_H: f64 = 1.008; // Da
const R_OH: f64 = 0.9572; // Å
const ANG_HOH: f64 = 104.52_f64.to_radians();

// For exclusions.
const SCALE_LJ_14: f64 = 0.5; // AMBER default
const SCALE_COUL_14: f64 = 1.0 / 1.2; // 0.833̅

const SOFTENING_FACTOR_SQ: f64 = 1e-6;

// todo: A/R
const SNAPSHOT_RATIO: usize = 10;

const EPS: f64 = 1.0e-8;

#[derive(Debug, Default)]
pub struct SnapshotDynamics {
    pub time: f64,
    pub atom_posits: Vec<Vec3>,
    pub atom_velocities: Vec<Vec3>,
}

#[derive(Clone, Debug)]
/// A trimmed-down atom for use with molecular dynamics.
pub struct AtomDynamics {
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
            mass: atom.element.atomic_weight() as f64,
            element: atom.element,
            partial_charge: atom.partial_charge.unwrap_or_default() as f64,
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[derive(Clone, Debug)]
pub(crate) struct AtomDynamicsx4 {
    // pub posit: Vec3x8,
    // pub vel: Vec3x8,
    // pub accel: Vec3x8,
    // pub mass: f32x8,
    pub posit: Vec3x4,
    pub vel: Vec3x4,
    pub accel: Vec3x4,
    pub mass: f64x4,
    pub element: [Element; 4],
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl AtomDynamicsx4 {
    pub fn from_array(bodies: [AtomDynamics; 4]) -> Self {
        let mut posits = [Vec3::new_zero(); 4];
        let mut vels = [Vec3::new_zero(); 4];
        let mut accels = [Vec3::new_zero(); 4];
        let mut masses = [0.0; 4];
        // Replace `Element::H` (for example) with some valid default for your `Element` type:
        let mut elements = [Element::Hydrogen; 4];

        for (i, body) in bodies.into_iter().enumerate() {
            posits[i] = body.posit;
            vels[i] = body.vel;
            accels[i] = body.accel;
            masses[i] = body.mass;
            elements[i] = body.element;
        }

        Self {
            posit: Vec3x4::from_array(posits),
            vel: Vec3x4::from_array(vels),
            accel: Vec3x4::from_array(accels),
            mass: f64x4::from_array(masses),
            element: elements,
        }
    }
}

/// Harmonic bond (AMBER-style).
#[derive(Debug, Clone, Copy)]
pub struct BondDynamics {
    pub atom_0: usize,
    pub atom_1: usize,
    /// Harmonic force const,  kcal mol⁻¹ Å⁻²
    /// Higher values mean less flexibility to bond length.
    pub k: f64,
    /// equilibrium bond length, Å
    pub r0: f64,
}

// impl From<&Bond> for BondDynamics {
//     fn from(bond: &Bond) -> Self {
//
//         Self {
//             atom_0: bond.atom_0,
//             atom_1: bond.atom_1,
//             k: 350., // todo temp. Find out how to get this.
//             r0:0., // todo: Figure out how to get this. Using `from_bonds` until then.
//         }
//     }
// }

impl BondDynamics {
    // fn from(bond: &Bond) -> Self {
    fn from_bond(bond: &Bond, atoms: &[Atom]) -> Self {
        // todo: Temp. Figure out how to get this.
        let init_bond_len = (atoms[bond.atom_0].posit - atoms[bond.atom_1].posit).magnitude();

        Self {
            atom_0: bond.atom_0,
            atom_1: bond.atom_1,
            // k: 350., // todo temp. Find out how to get this.
            k: 600., // todo temp. Find out how to get this.
            r0: init_bond_len,
        }
    }
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
    /// Sources that affect atoms in the system, but are not themselves affected by it. E.g.
    /// in docking, this might be a rigid receptor. These are for *non-bonded* interactions (e.g. Coulomb
    /// and VDW) only.
    pub atoms_external: Vec<AtomDynamics>,
    pub angles: Vec<Angle>,
    pub torsions: Vec<Torsion>,
    pub lj_lut: LjTable,
    /// Sigmas and epsilons are Lennard Jones parameters. Flat here, with outer loop receptor.
    /// Flattened. Separate single-value array facilitate use in CUDA and SIMD, vice a tuple.
    pub lj_sigma: Vec<f64>,
    pub lj_eps: Vec<f64>,
    /// In femtoseconds,
    pub time: f64,
    pub step_count: usize, // increments.
    // #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    // pub lj_sigma_x8: Vec<f64x4>,
    // #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    // pub lj_eps_x8: Vec<f64x4>,
    pub snapshots: Vec<SnapshotDynamics>,
    pub cell: SimBox,
    neighbour: Vec<Vec<usize>>,    // Verlet list
    max_disp_sq: f64,              // track atom displacements²
    pub kb_berendsen: Option<f64>, // coupling constant (ps⁻¹) if you want a thermostat
    pub target_temp: f64,
    /// Exclusions / masks optimization.
    excluded_pairs: HashSet<(usize, usize)>, // 1-2 and 1-3
    scaled14_pairs: HashSet<(usize, usize)>, // 1-4
}

impl MdState {
    pub fn new(
        atoms: &[Atom],
        atom_posits: &[Vec3],
        bonds: &[Bond],
        atoms_external: &[Atom],
        lj_table: &LjTable,
    ) -> Self {
        // We are using this approach instead of `.into`, so we can use the atom_posits from
        // the positioned ligand. (its atom coords are relative; we need absolute)
        let mut atoms_dy = Vec::with_capacity(atoms.len());
        for (i, atom) in atoms.iter().enumerate() {
            atoms_dy.push(AtomDynamics {
                posit: atom_posits[i],
                vel: Vec3::new_zero(),
                accel: Vec3::new_zero(),
                mass: atom.element.atomic_weight() as f64,
                element: atom.element,
                partial_charge: atom.partial_charge.unwrap_or_default() as f64,
            });
        }

        // let atoms_dy = atoms.iter().map(|a| a.into()).collect();
        // let bonds_dy = bonds.iter().map(|b| b.into()).collect();

        // todo: Temp on bonds this way until we know how to init r0.
        let bonds_dy = bonds.iter().map(|b| {
            BondDynamics::from_bond(b, atoms)
        }).collect();

        let atoms_dy_external: Vec<_> = atoms_external.iter().map(|a| a.into()).collect();

        let cell = {
            let (mut min, mut max) = (Vec3::splat(f64::INFINITY), Vec3::splat(f64::NEG_INFINITY));
            for a in &atoms_dy {
                min = min.min(a.posit);
                max = max.max(a.posit);
            }
            let pad = 15.0; // Å
            let lo = min - Vec3::splat(pad);
            let hi = max + Vec3::splat(pad);

            println!("Initizing sim box. L: {lo} H: {hi}");

            SimBox { lo, hi }
        };

        // 5-a) adjacency list from the bond table
        let mut nbrs: Vec<Vec<usize>> = vec![Vec::new(); atoms.len()];
        for b in bonds {
            nbrs[b.atom_0].push(b.atom_1);
            nbrs[b.atom_1].push(b.atom_0);
        }

        // 5-b) generate every i–j–k–l path (three consecutive bonds)
        let mut torsions = Vec::<Torsion>::new();
        let mut seen     = HashSet::<(usize, usize, usize, usize)>::new();

        for (j, nbrs_j) in nbrs.iter().enumerate() {
            for &k in nbrs_j {
                if j >= k { continue }                // treat each central bond once
                for &i in &nbrs[j] {
                    if i == k { continue }            // same bond
                    for &l in &nbrs[k] {
                        if l == j { continue }
                        // canonical ordering keeps i-j-k-l as unique key
                        let key = (i, j, k, l);
                        if !seen.insert(key) { continue }

                        // look-up (or stub-out) the torsion parameters
                        let (vn, n, gamma) =
                            lookup_torsion_params(&atoms[i], &atoms[j], &atoms[k], &atoms[l]);

                        torsions.push(Torsion { i, j, k, l, vn, n, γ: gamma });
                    }
                }
            }
        }

        let mut result = Self {
            atoms: atoms_dy,
            bonds: bonds_dy,
            torsions,
            atoms_external: atoms_dy_external,
            lj_lut: lj_table.clone(),
            cell,
            excluded_pairs: HashSet::new(),
            scaled14_pairs: HashSet::new(),
            ..Default::default()
        };

        result.build_masks();
        result.build_neighbour();

        result
    }

    fn build_masks(&mut self) {
        // helper to store pairs in canonical (low,high) order
        let mut push = |set: &mut HashSet<(usize, usize)>, i: usize, j: usize| {
            if i < j {
                set.insert((i, j));
            } else {
                set.insert((j, i));
            }
        };

        // 1-2 -----------------------------------------------------------
        for b in &self.bonds {
            push(&mut self.excluded_pairs, b.atom_0, b.atom_1);
        }

        // 1-3 -----------------------------------------------------------
        for a in &self.angles {
            push(&mut self.excluded_pairs, a.i, a.k);
        }

        // 1-4 -----------------------------------------------------------
        for t in &self.torsions {
            push(&mut self.scaled14_pairs, t.i, t.l);
        }

        // make sure no 1-4 pair is also in the excluded set
        for p in &self.scaled14_pairs {
            self.excluded_pairs.remove(p);
        }
    }

    /// Build / rebuild Verlet list
    fn build_neighbour(&mut self) {
        let cutoff2 = (CUTOFF + SKIN).powi(2);
        self.neighbour = vec![Vec::new(); self.atoms.len()];
        for i in 0..self.atoms.len() - 1 {
            for j in i + 1..self.atoms.len() {
                let dv = self
                    .cell
                    .min_image(self.atoms[j].posit - self.atoms[i].posit);
                if dv.magnitude_squared() < cutoff2 {
                    self.neighbour[i].push(j);
                    self.neighbour[j].push(i);
                }
            }
        }
        // reset displacement tracker
        for a in &mut self.atoms {
            a.vel /* nothing */;
        }
        self.max_disp_sq = 0.0;
    }

    /// One **Velocity-Verlet** step (leap-frog style) of length `dt_fs` femtoseconds.
    pub fn step(&mut self, dt_fs: f64) {
        // todo: Is this OK?
        // let dt = dt_fs * 1.0e-15; // s
        let dt = dt_fs * 0.0001;

        let dt_half = 0.5 * dt;

        // 1) First half-kick (v += a dt/2) and drift (x += v dt)
        // todo: Do we want traditional verlet instead?
        for a in &mut self.atoms {
            a.vel += a.accel * dt_half;
            a.posit += a.vel * dt;

            // todo; Put back, but getting NaNs.
            // todo: This isn't the only NaN source...

            a.posit = self.cell.wrap(a.posit);

            // track the largest squared displacement to know when to rebuild the list
            self.max_disp_sq = self.max_disp_sq.max((a.vel * dt).magnitude_squared());
        }

        for a in &mut self.atoms {
            a.accel = Vec3::new_zero();
            // println!("Acc: {}", a.accel);
        }

        self.apply_bond_forces();
        self.apply_angle_forces();
        self.apply_torsion_forces();
        self.apply_nonbonded_forces();

        // Second half-kick using new accelerations
        for a in &mut self.atoms {
            a.vel += a.accel * dt_half;
        }

        // Berendsen thermostat (T coupling to target every step)
        if let Some(tau_ps) = self.kb_berendsen {
            let tau = tau_ps * 1e-12;
            let curr_ke = self.current_kinetic_energy();
            let curr_t = 2.0 * curr_ke / (3.0 * self.atoms.len() as f64 * 0.0019872041); // k_B in kcal/mol
            let λ = (1.0 + dt / tau * (self.target_temp - curr_t) / curr_t).sqrt();
            for a in &mut self.atoms {
                a.vel *= λ;
            }
        }

        self.time += dt;
        self.step_count += 1;

        // Rebuild Verlet if needed
        if self.max_disp_sq > 0.25 * SKIN * SKIN {
            self.build_neighbour();
        }

        if self.step_count % SNAPSHOT_RATIO == 0 {
            self.take_snapshot();
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
            // let f_mag = -2.0 * k * Δ / dist.max(1e-12);
            let f_mag = -k * Δ / dist.max(1e-12);
            let f = diff * f_mag;
            ai.accel -= f / ai.mass;
            aj.accel += f / aj.mass;
        }
    }

    /// This maintains bond angles between sets of three atoms as they should be from hybridization.
    /// It reflects this hybridization, steric clashes, and partial double-bond character. This
    /// identifies deviations from the ideal angle, calculates restoring torque, and applies forces
    /// based on this to push the atoms back into their ideal positions in the molecule.
    fn apply_angle_forces(&mut self) {
        for &Angle { i, j, k, kθ, θ0 } in &self.angles {
            // r_i —— r_j —— r_k   (θ is ∠ijk)
            let (ai, aj, ak) = split3_mut(&mut self.atoms, i, j, k);

            // Bond vectors with j at the vertex
            let a = ai.posit - aj.posit;              // r_ij
            let b = ak.posit - aj.posit;              // r_kj (note sign!)

            let a_sq = a.magnitude_squared();
            let b_sq = b.magnitude_squared();

            // Quit early if atoms are on top of each other
            if a_sq < EPS || b_sq < EPS {
                continue;
            }

            let a_len = a_sq.sqrt();
            let b_len = b_sq.sqrt();
            let inv_ab = 1.0 / (a_len * b_len);

            // cosθ and sinθ
            let cosθ = (a.dot(b) * inv_ab).clamp(-1.0, 1.0);
            let sinθ_sq = 1.0 - cosθ * cosθ;
            if sinθ_sq < EPS {
                continue;                     // θ ≈ 0° or 180°, gradient ill-defined
            }
            let sinθ = sinθ_sq.sqrt();

            // Actual angle and its deviation
            let θ     = cosθ.acos();
            let Δθ    = θ - θ0;
            let dV_dθ = 2.0 * kθ * Δθ;        // ∂V/∂θ

            // Helpers reused in both force terms
            let coef  = -dV_dθ / sinθ;        // −∂V/∂θ  / sinθ
            let g1    = coef * inv_ab;        // common factor

            // ∂θ/∂r_i  and  ∂θ/∂r_k  (Allen & Tildesley Eq. 4.82)
            let dθ_dri =  g1 * ( b / b_len - cosθ * a / a_len );
            let dθ_drk =  g1 * ( a / a_len - cosθ * b / b_len );

            // Forces
            let f_i = -dV_dθ * dθ_dri;
            let f_k = -dV_dθ * dθ_drk;
            let f_j = -(f_i + f_k);           // Newton’s third law

            // Accelerations
            ai.accel += f_i / ai.mass;
            aj.accel += f_j / aj.mass;
            ak.accel += f_k / ak.mass;
        }
    }

    /// This maintains dihedral angles. (i.e. the angle between four atoms in a sequence). This models
    /// effects such as σ-bond overlap (e.g. staggered conformations), π-conjugation, which locks certain
    /// dihedrals near 0 or τ, and steric hindrance. (Bulky groups clashing).
    fn apply_torsion_forces(&mut self) {

        for &Torsion { i, j, k, l, vn, n, γ } in &self.torsions {
            // Split the four atoms mutably without aliasing
            let (ai, aj, ak, al) = split4_mut(&mut self.atoms, i, j, k, l);

            // Convenience aliases for the positions
            let r1 = ai.posit;
            let r2 = aj.posit;
            let r3 = ak.posit;
            let r4 = al.posit;

            // Bond vectors (see Allen & Tildesley, chap. 4)
            let b1 = r1 - r2;         // r_ij
            let b2 = r3 - r2;         // r_kj  (note the sign!)
            let b3 = r4 - r3;         // r_lk

            // Normal vectors to the two planes
            let n1 = b1.cross(b2);
            let n2 = b3.cross(b2);

            let n1_sq = n1.magnitude_squared();
            let n2_sq = n2.magnitude_squared();
            let b2_len = b2.magnitude();

            // Bail out if the four atoms are (nearly) colinear
            if n1_sq < EPS || n2_sq < EPS || b2_len < EPS {
                continue;
            }

            // Unit normals
            let n1_hat = n1 / n1_sq.sqrt();
            let n2_hat = n2 / n2_sq.sqrt();

            // m1 = n̂1 × (b̂2)
            let m1 = n1_hat.cross(b2 / b2_len);

            // Signed dihedral (−π … π)
            let x = n1_hat.dot(n2_hat);
            let y = m1.dot(n2_hat);
            let φ = y.atan2(x);

            // dV/dφ  (note the minus sign from ∂cos / ∂φ = −sin)
            let dV_dφ = -0.5 * vn * (n as f64) * ((n as f64) * φ - γ).sin();

            // ∂φ/∂r   (see e.g. DOI 10.1016/S0021-9991(97)00040-8)
            let dφ_dr1 =  n1 * ( b2_len / n1_sq);
            let dφ_dr4 = -n2 * ( b2_len / n2_sq);
            let dφ_dr2 = -n1 * (b1.dot(b2) / (b2_len * n1_sq))
                + n2 * (b3.dot(b2) / (b2_len * n2_sq));
            let dφ_dr3 = -dφ_dr1 - dφ_dr2 - dφ_dr4; // Newton’s third law

            // F_i = −dV/dφ · ∂φ/∂r_i
            let f1 = -dφ_dr1 * dV_dφ;
            let f2 = -dφ_dr2 * dV_dφ;
            let f3 = -dφ_dr3 * dV_dφ;
            let f4 = -dφ_dr4 * dV_dφ;

            // Convert to accelerations
            ai.accel += f1 / ai.mass;
            aj.accel += f2 / aj.mass;
            ak.accel += f3 / ak.mass;
            al.accel += f4 / al.mass;
        }
    }

    /// Coulomb and Van der Waals.
    ///
    /// todo: If required, build a neighbors list for interactions with external atoms.
    fn apply_nonbonded_forces(&mut self) {
        let cutoff_sq = CUTOFF * CUTOFF;

        const EPS: f64 = 1e-6;

        for i in 0..self.atoms.len() {
            // todo: Can you unify this with your neighbor code used for bonds?
            for &j in &self.neighbour[i] {
                if j < i {
                    // Prevents duplication of the pair in the other order.
                    continue;
                }

                // Handle masks.
                let key = if i < j { (i, j) } else { (j, i) };

                if self.excluded_pairs.contains(&key) {
                    continue;
                }

                let scale14 = self.scaled14_pairs.contains(&key);

                let diff = self.atoms[j].posit - self.atoms[i].posit;

                let dv = self.cell.min_image(diff);
                let r_sq = dv.magnitude_squared();
                if r_sq > cutoff_sq {
                    continue;
                }

                // println!("ATOMS: {:?} {:?}", self.atoms[i].element, self.atoms[j].element);
                let (σ, ϵ) = self
                    .lj_lut
                    .get(&(self.atoms[i].element, self.atoms[j].element))
                    .unwrap();

                let dist = r_sq.sqrt();
                let dir = diff / dist;

                // if dist < EPS {
                //     continue;
                // }

                let mut f_lj = force_lj(dir, dist, *σ as f64, *ϵ as f64);

                let mut f_coulomb = force_coulomb(
                    dir,
                    dist,
                    self.atoms[i].partial_charge,
                    self.atoms[j].partial_charge,
                    SOFTENING_FACTOR_SQ,
                );

                if scale14 {
                    f_lj *= SCALE_LJ_14;
                    f_coulomb *= SCALE_COUL_14;
                }

                let f = f_lj + f_coulomb;

                if self.atoms[i].mass < 0.0001 {
                    println!("Mass problem: {:?}", self.atoms[i]);
                }
                if self.atoms[j].mass < 0.0001 {
                    println!("Mass problem: {:?}", self.atoms[j]);
                }

                let accel_0 = f / self.atoms[i].mass;
                let accel_1 = f / self.atoms[j].mass;

                self.atoms[i].accel += accel_0;
                self.atoms[j].accel -= accel_1;
            }
        }

        // Second pass: External atoms.
        for ai in &mut self.atoms {
            for aj in &self.atoms_external {
                let dv = self.cell.min_image(aj.posit - ai.posit);

                // let dv = Vec3::new_zero();

                let r_sq = dv.magnitude_squared();
                if r_sq > cutoff_sq {
                    continue;
                }

                let (σ, ϵ) = self.lj_lut.get(&(ai.element, aj.element)).unwrap();
                let dist = r_sq.sqrt();
                let dir = dv / dist;

                if dist < EPS {
                    continue;
                }

                let f = force_lj(dir, dist, *σ as f64, *ϵ as f64)
                    + force_coulomb(
                    dir,
                    dist,
                    ai.partial_charge,
                    aj.partial_charge,
                    SOFTENING_FACTOR_SQ,
                );

                ai.accel += f / ai.mass;
            }
        }
    }

    // quick helper for thermostat
    #[inline]
    fn current_kinetic_energy(&self) -> f64 {
        self.atoms
            .iter()
            .map(|a| 0.5 * a.mass * a.vel.magnitude_squared())
            .sum()
    }

    pub fn take_snapshot(&mut self) {
        self.snapshots.push(SnapshotDynamics {
            time: self.time,
            atom_posits: self.atoms.iter().map(|a| a.posit).collect(),
            atom_velocities: self.atoms.iter().map(|a| a.vel).collect(),
        })
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
/// Add `n` TIP3P molecules uniformly in the box.
pub fn add_tip3p(state: &mut MdState, n: usize, rng: &mut impl rand::Rng) {
    use rand_distr::{Distribution, UnitSphere};
    for _ in 0..n {
        // random COM inside box
        let rand3 = Vec3::new(rng.random::<f64>(), rng.random(), rng.random());

        // todo: What should this be doing?
        let com = state.cell.lo + rand3.hadamard_product(state.cell.extent());

        // random orientation – unit vector + perpendicular
        let z = Vec3::from_slice(&UnitSphere.sample(rng)).unwrap();
        // todo: This is a good idea...
        let x = z.any_perpendicular().to_normalized();
        let y = z.cross(x);

        // two H positions in the HOH plane
        let d = R_OH * (ANG_HOH / 2.0).sin();
        let h0 = com + z * R_OH;
        let h1 = com + z * (-R_OH * ANG_HOH.cos()) + x * d * 2.;

        let mut make = |pos, mass, q, elem| AtomDynamics {
            posit: pos,
            vel: Vec3::new_zero(),
            accel: Vec3::new_zero(),
            mass,
            partial_charge: q,
            element: elem,
        };

        state.atoms.push(make(com, M_O, -0.834, Element::Oxygen));
        state.atoms.push(make(h0, M_H, 0.417, Element::Hydrogen));
        state.atoms.push(make(h1, M_H, 0.417, Element::Hydrogen));
    }
    state.build_neighbour(); // list is stale now
}

// todo: Populate. Look for amber .dat files, and *frcmod or parm10.dat.
fn lookup_torsion_params(
    _ai: &Atom,
    _aj: &Atom,
    _ak: &Atom,
    _al: &Atom,
) -> (f64, u8, f64) {
    // TODO: replace with a real force-field database lookup.
    // The constants below (Cn3, phase = 0) give a mild (~0.15 kcal mol⁻¹) barrier
    // with 3-fold periodicity, so nothing explodes while you wire in a true FF.
    let vn     = 0.15;      // kcal·mol⁻¹
    let n      = 3;         // periodicity
    let gamma  = 0.0;       // rad
    (vn, n, gamma)
}