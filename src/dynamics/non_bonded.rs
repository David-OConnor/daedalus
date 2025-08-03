//! For VDW and Coulomb forces

use lin_alg::f64::Vec3;

use crate::{
    dynamics::{
        AtomDynamics, LjTable, MdState,
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

                let diff = self.atoms[i].posit - self.atoms[j].posit;
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

                println!(
                    "Posit 0: {}, Posit 1: {}, {f}, q0: {:.2}, q1: {:.2} dist: {:.2}",
                    self.atoms[i].posit,
                    self.atoms[j].posit,
                    self.atoms[i].partial_charge,
                    self.atoms[j].partial_charge,
                    diff.magnitude()
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
                let diff = a_lig.posit - a_static.posit;
                let dv = self.cell.min_image(diff);

                let r_sq = dv.magnitude_squared();

                // No LJ cacheing here for now.
                // todo: cacheing?
                let f = f_nonbonded(a_lig, a_static, r_sq, diff, false, None, &mut self.lj_table);

                println!(
                    "Posit 0: {}, Posit 1: {}, {f}, q0: {:.2}, q1: {:.2}",
                    a_lig.posit, a_static.posit, a_lig.partial_charge, a_static.partial_charge
                );

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

    // todo: This distance cutoff helps, but ideally we skip the distance computation too in these cases.
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

        // Negative due to our mix of conventions; keep it consistent with coulomb, and net correct.
        let mut f = -force_lj(dir, dist, σ, ε);
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
    // todo temp to test
    // f_coulomb
    // f_lj
}

/// Helper. Returns σ, ε between an atom pair.
fn combine_lj_params(atom_0: &AtomDynamics, atom_1: &AtomDynamics) -> (f64, f64) {
    let σ = 0.5 * (atom_0.lj_sigma + atom_1.lj_sigma);
    let ε = (atom_0.lj_eps * atom_1.lj_eps).sqrt();

    (σ, ε)
}
