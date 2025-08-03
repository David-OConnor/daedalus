//! For VDW and Coulomb forces

use std::collections::HashMap;

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
    /// to handle approximated Coulomb forces. This function applies forces from dynamic, static,
    /// and water sources.
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

        // Force from dynamic atoms (on the target dynamic atoms): LJ, and Ewald-screened Coulomb.
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
                    None,
                    None,
                    &mut self.lj_table,
                    &mut self.lj_table_static,
                    &mut self.lj_table_water,
                );

                // println!(
                //     "Posit 0: {}, Posit 1: {}, {f}, q0: {:.2}, q1: {:.2} dist: {:.2}",
                //     self.atoms[i].posit,
                //     self.atoms[j].posit,
                //     self.atoms[i].partial_charge,
                //     self.atoms[j].partial_charge,
                //     diff.magnitude()
                // );

                // We divide by mass in `step`.
                self.atoms[i].accel += f;
                self.atoms[j].accel -= f;
            }
        }

        // Force from static atoms and water. (Short-range). We currently don't use the neighbors approach here, although
        // it is likely useful, as when docking, there will be many static atoms IVO the dynamic ones.
        for (i_lig, a_lig) in self.atoms.iter_mut().enumerate() {
            // Force from static atoms.
            for (i_static, a_static) in self.atoms_static.iter().enumerate() {
                let diff = a_lig.posit - a_static.posit;
                let dv = self.cell.min_image(diff);

                let r_sq = dv.magnitude_squared();

                let f = f_nonbonded(
                    a_lig,
                    a_static,
                    r_sq,
                    diff,
                    false,
                    None,
                    Some((i_lig, i_static)),
                    None,
                    &mut self.lj_table,
                    &mut self.lj_table_static,
                    &mut self.lj_table_water,
                );

                // println!(
                //     "Posit 0: {}, Posit 1: {}, {f}, q0: {:.2}, q1: {:.2}",
                //     a_lig.posit, a_static.posit, a_lig.partial_charge, a_static.partial_charge
                // );

                // We divide by mass in `step`.
                a_lig.accel += f;
            }

            // Force from water
            // todo: Consider an LJ table here too. Note that this is simple: It's one
            // per target atom, as the only LJ source for water is the (uniform) O.
            for (i_water, mol_water) in self.water.iter().enumerate() {
                for a_water_src in [&mol_water.o, &mol_water.m, &mol_water.h0, &mol_water.h1] {
                    let diff = a_lig.posit - a_water_src.posit;
                    let dv = self.cell.min_image(diff);

                    let r_sq = dv.magnitude_squared();

                    let f = f_nonbonded(
                        a_lig,
                        a_water_src,
                        r_sq,
                        diff,
                        false,
                        None,
                        None,
                        Some(i_lig),
                        &mut self.lj_table,
                        &mut self.lj_table_static,
                        &mut self.lj_table_water,
                    );
                    // We divide by mass in `step`.
                    a_lig.accel += f;
                }
            }
        }

        // Long‑range reciprocal‑space term (PME / SPME), both static and dynamic.
        // Build a temporary Vec with *all* charges so the mesh sees both
        // movable and rigid atoms.  We only add forces back to movable atoms.
        let n_dynamic = self.atoms.len();
        let mut all_atoms = Vec::with_capacity(n_dynamic + self.atoms_static.len());

        all_atoms.extend(self.atoms.iter().cloned());
        all_atoms.extend(self.atoms_static.iter().cloned());

        let rec_forces =
            pme_long_range_forces(&all_atoms, &self.cell, EWALD_ALPHA, PME_MESH_SPACING);

        // add reciprocal forces to *movable* atoms only
        for (atom, f_rec) in self.atoms.iter_mut().zip(rec_forces.iter().take(n_dynamic)) {
            // We divide by mass in `step`.
            atom.accel += *f_rec;
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
    // For now, this caching optimization only to non-water interactions.
    // If it doesn't apply, this field is None.
    atom_indices: Option<(usize, usize)>,
    // (dynamic i, static i)
    atom_indices_static: Option<(usize, usize)>,
    atom_indices_water: Option<usize>,
    lj_table: &mut LjTable,
    // (dynamic i, static i)
    lj_table_static: &mut LjTable,
    lj_table_water: &mut HashMap<usize, (f64, f64)>,
) -> Vec3 {
    let dist = r_sq.sqrt();
    let dir = diff / dist;

    // todo: This trip match is heinous; flatten it.

    // todo: This distance cutoff helps, but ideally we skip the distance computation too in these cases.
    let mut f_lj = if r_sq > CUTOFF_VDW_SQ {
        Vec3::new_zero()
    } else {
        let (σ, ε) = match atom_indices {
            // todo: Try either order? If not, we sill need two passes here, and will have twice the entries;
            // todo: One for each order.
            Some(indices) => match lj_table.get(&indices) {
                Some(params) => *params,
                None => {
                    let (σ, ε) = combine_lj_params(tgt, src);
                    lj_table.insert(indices, (σ, ε));
                    (σ, ε)
                }
            },
            // i.e. water or static, or step 0.
            None => {
                // This nest is a bit messy
                match atom_indices_static {
                    // Note: Index order matters here, and the caveat for dynamic-dynamic interactions
                    // doesn't.
                    Some(indices) => match lj_table_static.get(&indices) {
                        Some(params) => *params,
                        None => {
                            let (σ, ε) = combine_lj_params(tgt, src);
                            lj_table_static.insert(indices, (σ, ε));
                            (σ, ε)
                        }
                    },
                    // i.e. water
                    None => {
                        match atom_indices_water {
                            //Single index of the lig; the only water mol is the uniform O.
                            Some(i) => match lj_table_water.get(&i) {
                                Some(params) => *params,
                                None => {
                                    let (σ, ε) = combine_lj_params(tgt, src);
                                    lj_table_water.insert(i, (σ, ε));
                                    (σ, ε)
                                }
                            },
                            None => {
                                unreachable!(
                                    "Must pass one of dynamic, static, or water index set."
                                );
                            }
                        }
                    }
                };
                // Neither LUT table.
                combine_lj_params(tgt, src)
            }
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

/// Helper. Returns σ, ε between an atom pair. Atom order passed as params doesn't matter.
fn combine_lj_params(atom_0: &AtomDynamics, atom_1: &AtomDynamics) -> (f64, f64) {
    let σ = 0.5 * (atom_0.lj_sigma + atom_1.lj_sigma);
    let ε = (atom_0.lj_eps * atom_1.lj_eps).sqrt();

    (σ, ε)
}
