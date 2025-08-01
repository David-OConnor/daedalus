//! For VDW and Coulomb forces

use std::f64::consts::TAU;

use lin_alg::f64::{Vec3, calc_dihedral_angle_v2};

use crate::{
    dynamics,
    dynamics::{
        AtomDynamics, EPS, LjTable, MdState, SKIN_SQ, SNAPSHOT_RATIO, SnapshotDynamics,
        spme::{EWALD_ALPHA, PME_MESH_SPACING, force_coulomb_ewald_real, pme_long_range_forces},
    },
    forces::force_lj,
};

// Å. 9-12 should be fine; there is very little VDW force > this range due to
// the ^-7 falloff.
pub const CUTOFF_VDW: f64 = 12.0;
const CUTOFF_VDW_SQ: f64 = CUTOFF_VDW * CUTOFF_VDW;

// See Amber RM, section 15, "1-4 Non-Bonded Interaction Scaling"
// "Non-bonded interactions between atoms separated by three consecutive bonds... require a special
// treatment in Amber force fields."
// "By default, vdW 1-4 interactions are divided (scaled down) by a factor of 2.0, electrostatic 1-4 terms by a factor
// of 1.2."
const SCALE_LJ_14: f64 = 0.5;
const SCALE_COUL_14: f64 = 1.0 / 1.2;

impl MdState {
    /// One **Velocity-Verlet** step (leap-frog style) of length `dt_fs` femtoseconds.
    pub fn step(&mut self, dt: f64) {
        let dt_half = 0.5 * dt;

        // First half-kick (v += a dt/2) and drift (x += v dt)
        // todo: Do we want traditional verlet instead?
        for a in &mut self.atoms {
            a.vel += a.accel * dt_half; // Half-kick
            a.posit += a.vel * dt; // Drift
            a.posit = self.cell.wrap(a.posit);

            // track the largest squared displacement to know when to rebuild the list
            self.max_disp_sq = self.max_disp_sq.max((a.vel * dt).magnitude_squared());
        }

        // Reset acceleration.
        for a in &mut self.atoms {
            a.accel = Vec3::new_zero();
        }

        // Bonded forces
        self.apply_bond_stretching_forces();
        self.apply_angle_bending_forces();
        // todo: Dihedral not working. Skipping for now. Our measured and expected angles aren't lining up.
        self.apply_dihedral_forces();

        self.apply_nonbonded_forces();

        // Second half-kick using new accelerations
        for a in &mut self.atoms {
            a.vel += a.accel * dt_half;
        }

        let sources: Vec<AtomDynamics> = [
            &self.atoms[..],
            &self.atoms_static[..],
            // todo: You must take each water atom into account.
            // &self.water[..],
        ]
        .concat();

        for water in &mut self.water {
            let sources = &sources;
            water.step(dt, &sources, &self.cell, &mut self.lj_table);
        }

        // todo: Apply the thermostat.

        // Berendsen thermostat (T coupling to target every step)
        // if let Some(tau_ps) = self.kb_berendsen {
        //     let tau = tau_ps * 1e-12;
        //     let curr_ke = self.current_kinetic_energy();
        //     let curr_t = 2.0 * curr_ke / (3.0 * self.atoms.len() as f64 * 0.0019872041); // k_B in kcal/mol
        //     let λ = (1.0 + dt / tau * (self.temp_target - curr_t) / curr_t).sqrt();
        //     for a in &mut self.atoms {
        //         a.vel *= λ;
        //     }
        // }

        self.time += dt;
        self.step_count += 1;

        // Rebuild Verlet if needed
        if self.max_disp_sq > 0.25 * SKIN_SQ {
            self.build_neighbours();
        }

        if self.step_count % SNAPSHOT_RATIO == 0 {
            self.take_snapshot();
        }
    }

    fn apply_bond_stretching_forces(&mut self) {
        for (indices, params) in &self.force_field_params.bond_stretching {
            let (a_0, a_1) = dynamics::split2_mut(&mut self.atoms, indices.0, indices.1);

            let f = dynamics::f_bond_stretching(a_0.posit, a_1.posit, params);

            const KCALMOL_A_TO_A_FS2_PER_AMU: f64 = 4.184e-4;
            // todo: Multiply accels by this?? Or are our units self-consistent.

            a_0.accel += f / a_0.mass;
            a_1.accel -= f / a_1.mass;
        }
    }

    /// This maintains bond angles between sets of three atoms as they should be from hybridization.
    /// It reflects this hybridization, steric clashes, and partial double-bond character. This
    /// identifies deviations from the ideal angle, calculates restoring torque, and applies forces
    /// based on this to push the atoms back into their ideal positions in the molecule.
    ///
    /// Valence angles, which are the angle formed by two adjacent bonds ba et bc
    /// in a same molecule; a valence angle tends to maintain constant the anglê
    /// abc. A valence angle is thus concerned by the positions of three atoms.
    fn apply_angle_bending_forces(&mut self) {
        for (indices, params) in &self.force_field_params.angle {
            let (a_0, a_1, a_2) =
                dynamics::split3_mut(&mut self.atoms, indices.0, indices.1, indices.2);

            let (f_0, f_1, f_2) =
                dynamics::f_angle_bending(a_0.posit, a_1.posit, a_2.posit, params);

            a_0.accel += f_0 / a_0.mass;
            a_1.accel += f_1 / a_1.mass;
            a_2.accel += f_2 / a_2.mass;
        }
    }

    /// This maintains dihedral angles. (i.e. the angle between four atoms in a sequence). This models
    /// effects such as σ-bond overlap (e.g. staggered conformations), π-conjugation, which locks certain
    /// dihedrals near 0 or τ, and steric hindrance. (Bulky groups clashing).
    ///
    /// This applies both "proper" linear dihedral angles, and "improper", hub-and-spoke dihedrals. These
    /// two angles are calculated in the same way, but the covalent-bond arrangement of the 4 atoms differs.
    fn apply_dihedral_forces(&mut self) {
        for (indices, dihe) in &self.force_field_params.dihedral {
            if &dihe.atom_types.0 == "X" || &dihe.atom_types.3 == "X" {
                continue; // todo temp, until we sum.
            }

            // Split the four atoms mutably without aliasing
            let (a_0, a_1, a_2, a_3) =
                dynamics::split4_mut(&mut self.atoms, indices.0, indices.1, indices.2, indices.3);

            // Convenience aliases for the positions
            let r_0 = a_0.posit;
            let r_1 = a_1.posit;
            let r_2 = a_2.posit;
            let r_3 = a_3.posit;

            // Bond vectors (see Allen & Tildesley, chap. 4)
            let b1 = r_1 - r_0; // r_ij
            let b2 = r_2 - r_1; // r_kj
            let b3 = r_3 - r_2; // r_lk

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

            let dihe_measured = calc_dihedral_angle_v2(&(r_0, r_1, r_2, r_3));

            // Note: We assume that vn has already been divided by the integer divisor.
            // let dV_dφ = -0.5
            // todo: Precompute this barrier height when loading to the indexed variant.
            // todo: Do that once this all owrks.
            // let dV_dφ =  -(dihe.barrier_height_vn as f64) / (dihe.integer_divisor as f64)

            let k = -dihe.barrier_height as f64;
            let per = dihe.periodicity as f64;
            let arg = per * dihe_measured - dihe.phase as f64;

            let dV_dφ = k * per * arg.sin();

            if self.step_count == 0 {
                println!(
                    "{:?} - Ms: {:.2}, exp: {:.2}/{} dV_dφ: {:.2}",
                    &dihe.atom_types,
                    dihe_measured / TAU,
                    dihe.phase / TAU as f32,
                    dihe.periodicity,
                    dV_dφ,
                );
            }

            // ∂φ/∂r   (see e.g. DOI 10.1016/S0021-9991(97)00040-8)
            let dφ_dr1 = -n1 * (b2_len / n1_sq);
            let dφ_dr4 = n2 * (b2_len / n2_sq);

            let dφ_dr2 =
                -n1 * (b1.dot(b2) / (b2_len * n1_sq)) + n2 * (b3.dot(b2) / (b2_len * n2_sq));
            let dφ_dr3 = -dφ_dr1 - dφ_dr2 - dφ_dr4; // Newton’s third law

            // F_i = −dV/dφ · ∂φ/∂r_i
            let f1 = -dφ_dr1 * dV_dφ;
            let f2 = -dφ_dr2 * dV_dφ;
            let f3 = -dφ_dr3 * dV_dφ;
            let f4 = -dφ_dr4 * dV_dφ;

            // Convert to accelerations
            a_0.accel += f1 / a_0.mass;
            a_1.accel += f2 / a_1.mass;
            a_2.accel += f3 / a_2.mass;
            a_3.accel += f4 / a_3.mass;
        }
    }

    /// Coulomb and Van der Waals. (Lennard-Jones). We use the MD-standard [S]PME approach
    /// to handle approximated Coulomb forces.
    ///
    /// We use a hard distance cutoff for Vdw, due to its  ^-7 falloff.
    ///todo: The PME reciprical case still contains 1-4 coulomb; fix A/R, and QC
    /// todo teh SPME's interaction with exclusions adn 1-4 scaling in general.
    ///
    /// todo: ChatGPT's take:
    /// "
    ///     1-2 / 1-3: fine—the real-space part is zero; the reciprocal part still adds a tiny force, but Amber accepts that because those atoms are seldom >½ box apart. If you want bit-exact Amber, subtract the same pair from rec_forces.
    ///
    ///     1-4: you do scale the real-space part, but the reciprocal part is still full strength, so the net Coulomb-14 ends up too large by 1 – 1/SCEE (≈ 17 % with the default 1.2).
    ///     Fix: after building nonbonded_scaled, loop over it again and apply a corrective force/energy equal to (1 – 1/SCEE) * q_i q_j f(r) (or simply compute a second short-range pass with that factor and subtract it).
    ///
    /// If you prefer to avoid the extra pass, an alternative is to put the charges of a 1-4 pair into different mesh charge groups and annul their contribution in reciprocal space, but that is more intrusive.
    /// "
    pub fn apply_nonbonded_forces(&mut self) {
        const EPS: f64 = 1e-6;

        // Apply the short range terms: LJ, and Ewald-screened Coulomb.
        for i in 0..self.atoms.len() {
            for &j in &self.neighbour[i] {
                if j < i {
                    // Prevents duplication of the pair in the other order.
                    continue;
                }

                let scale14 = {
                    let key = if i < j { (i, j) } else { (j, i) };

                    if self.nonbonded_exclusions.contains(&key) {
                        continue;
                    }

                    self.nonbonded_scaled.contains(&key)
                };

                let diff = self.atoms[j].posit - self.atoms[i].posit;
                let dv = self.cell.min_image(diff);
                let r_sq = dv.magnitude_squared();

                let f = f_nonbonded(
                    &self.atoms[i],
                    &self.atoms[j],
                    r_sq,
                    diff,
                    scale14,
                    Some((i, j)),
                    &mut self.lj_table,
                );

                let accel_0 = f / self.atoms[i].mass;
                let accel_1 = f / self.atoms[j].mass;

                self.atoms[i].accel += accel_0;
                self.atoms[j].accel -= accel_1;
            }
        }

        // Second pass: Static atoms. (Short-range)
        for a_lig in &mut self.atoms {
            for a_static in &self.atoms_static {
                let diff = a_static.posit - a_lig.posit;
                let dv = self.cell.min_image(diff);

                let r_sq = dv.magnitude_squared();

                // No LJ cacheing here for now.
                // todo: cacheing?
                let f = f_nonbonded(a_lig, a_static, r_sq, diff, false, None, &mut self.lj_table);
                a_lig.accel += f / a_lig.mass;
            }
        }

        // Long‑range reciprocal‑space term (PME / SPME)
        // Build a temporary Vec with *all* charges so the mesh sees both
        // movable and rigid atoms.  We only add forces back to movable atoms.
        let n_movable = self.atoms.len();
        let mut all_atoms = Vec::with_capacity(n_movable + self.atoms_static.len());
        all_atoms.extend(self.atoms.iter().cloned());
        all_atoms.extend(self.atoms_static.iter().cloned());

        let rec_forces =
            pme_long_range_forces(&all_atoms, &self.cell, EWALD_ALPHA, PME_MESH_SPACING);

        // add reciprocal forces to *movable* atoms only
        for (atom, f_rec) in self.atoms.iter_mut().zip(rec_forces.iter().take(n_movable)) {
            atom.accel += *f_rec / atom.mass; // convert to acceleration
        }
    }

    /// A helper for the thermostat
    fn current_kinetic_energy(&self) -> f64 {
        self.atoms
            .iter()
            .map(|a| 0.5 * a.mass * a.vel.magnitude_squared())
            .sum()
    }

    pub fn take_snapshot(&mut self) {
        let mut water_o_posits = Vec::with_capacity(self.water.len());
        let mut water_h0_posits = Vec::with_capacity(self.water.len());
        let mut water_h1_posits = Vec::with_capacity(self.water.len());

        for water in &self.water {
            water_o_posits.push(water.o.posit);
            water_h0_posits.push(water.h0.posit);
            water_h1_posits.push(water.h1.posit);
        }

        self.snapshots.push(SnapshotDynamics {
            time: self.time,
            atom_posits: self.atoms.iter().map(|a| a.posit).collect(),
            atom_velocities: self.atoms.iter().map(|a| a.vel).collect(),
            water_o_posits,
            water_h0_posits,
            water_h1_posits,
        })
    }
}

/// Vdw and Coulomb forces. Used by water and non-water.
/// We split out `r_sq` and `diff` for use while integrating a unit cell, if applicable.
/// todo: Optimize using neighbors, and/or PME/SPME.
pub fn f_nonbonded(
    tgt: &AtomDynamics,
    src: &AtomDynamics,
    r_sq: f64,
    diff: Vec3,
    scale14: bool, // See notes earlier in this module.
    // For now, this caching optimization only applies to non-static, non-water interactions.
    // If it doesn't apply, this field is None.
    atom_indices: Option<(usize, usize)>,
    lj_table: &mut LjTable,
) -> Vec3 {
    let dist = r_sq.sqrt();
    let dir = diff / dist;

    // todo: This check helps, but ideally we skip the distance computation too in these cases.
    let mut f_lj = if r_sq > CUTOFF_VDW_SQ {
        Vec3::new_zero()
    } else {
        let (σ, ε) = match atom_indices {
            Some(indices) => match lj_table.get(&indices) {
                Some(params) => *params,
                None => {
                    let (σ, ε) = combine_lj_params(tgt, src);
                    lj_table.insert(indices, (σ, ε));
                    (σ, ε)
                }
            },
            // i.e. water or static.
            None => combine_lj_params(tgt, src),
        };

        let mut f = force_lj(dir, dist, σ, ε);
        if scale14 {
            f *= SCALE_LJ_14;
        }
        f
    };

    // let mut f_coulomb = force_coulomb(
    //     dir,
    //     dist,
    //     tgt.partial_charge,
    //     src.partial_charge,
    //     SOFTENING_FACTOR_SQ,
    // );

    let mut f_coulomb = force_coulomb_ewald_real(
        dir,
        dist,
        tgt.partial_charge,
        src.partial_charge,
        EWALD_ALPHA,
    );

    // See Amber RM, sectcion 15, "1-4 Non-Bonded Interaction Scaling"
    if scale14 {
        f_coulomb *= SCALE_COUL_14;
    }

    f_lj + f_coulomb
}

/// Helper. Returns σ, ε between an atom pair.
fn combine_lj_params(atom_0: &AtomDynamics, atom_1: &AtomDynamics) -> (f64, f64) {
    let σ = 0.5 * (atom_0.lj_sigma + atom_1.lj_sigma);
    let ε = (atom_0.lj_eps * atom_1.lj_eps).sqrt();

    (σ, ε)
}
