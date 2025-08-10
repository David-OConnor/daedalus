//! For VDW and Coulomb forces

use std::collections::HashMap;

use lin_alg::f64::Vec3;
use rayon::prelude::*;

use crate::{
    dynamics::{
        AtomDynamics, LjTable, MdState,
        ambient::SimBox,
        spme::{EWALD_ALPHA, PME_MESH_SPACING, force_coulomb_ewald_real, pme_long_range_forces},
        water_opc,
    },
    forces::force_lj,
};

// Å. 9-12 should be fine; there is very little VDW force > this range due to
// the ^-7 falloff.
pub const CUTOFF_VDW: f64 = 12.0;
const CUTOFF_VDW_SQ: f64 = CUTOFF_VDW * CUTOFF_VDW;

// This is relatively large, as it accomodates water that might be near ligand atoms
// other than the sample one.
// todo: Adjust this A/R.
const CUTOFF_WATER_FROM_LIGAND: f64 = 14.0;

// See Amber RM, section 15, "1-4 Non-Bonded Interaction Scaling"
// "Non-bonded interactions between atoms separated by three consecutive bonds... require a special
// treatment in Amber force fields."
// "By default, vdW 1-4 interactions are divided (scaled down) by a factor of 2.0, electrostatic 1-4 terms by a factor
// of 1.2."
const SCALE_LJ_14: f64 = 0.5;
const SCALE_COUL_14: f64 = 1.0 / 1.2;

// Multiply by this to convert partial charges from elementary charge (What we store in Atoms loaded from mol2
// files and amino19.lib.) to the self-consistent amber units required to calculate Coulomb force.
pub const CHARGE_UNIT_SCALER: f64 = 18.2223;

/// Run this once per MD run. Sets up LJ caches for each pair of atoms.
/// Mirrors the force fn's params.
pub fn setup_lj_cache(
    tgt: &AtomDynamics,
    src: &AtomDynamics,
    atom_indices: Option<(usize, usize)>,
    // (dynamic i, static i)
    atom_indices_static: Option<(usize, usize)>,
    atom_indices_water: Option<usize>,
    lj_table: &mut LjTable,
    // (dynamic i, static i)
    lj_table_static: &mut LjTable,
    lj_table_water: &mut HashMap<usize, (f64, f64)>,
) {
    let (σ, ε) = combine_lj_params(tgt, src);

    if let Some(indices) = atom_indices {
        // Dynamic-dynamic
        lj_table.insert(indices, (σ, ε));
    } else if let Some(indices) = atom_indices_static {
        // Note: Index order matters here, and the caveat for dynamic-dynamic interactions
        // doesn't.
        lj_table_static.insert(indices, (σ, ε));
    } else if let Some(i) = atom_indices_water {
        lj_table_water.insert(i, (σ, ε));
    }
}

impl MdState {
    /// Coulomb and Van der Waals (Lennard-Jones) forces. We use the MD-standard [S]PME approach
    /// to handle approximated Coulomb forces. This function applies forces from dynamic, static,
    /// and water sources.
    ///
    /// We use a hard distance cutoff for Vdw, due to its  ^-7 falloff.
    /// todo: The PME reciprocal case still contains 1-4 coulomb; fix A/R, and QC
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
        // An approximation to simplify which water molecules are close enough to interact
        // with the ligand: Each X steps, we measure the distance between each molecule,
        // and an arbitrary ligand atom; this takes advantage of the ligand being small.

        // todo: Organize this function better between the 3 variants, if possible.

        let n_dyn = self.atoms.len();

        // ------ Forces from other dynamic atoms on dynamic ones ------

        // Exclusions and scaling apply to dynamic-dynamic interactions only.
        let exclusions = &self.nonbonded_exclusions;
        let scaled_set = &self.nonbonded_scaled;

        {
            // Set up pairs ahead of time; conducive to parallel iteration.
            let pairs: Vec<(usize, usize, bool)> = (0..n_dyn)
                .flat_map(|i| {
                    self.neighbors_nb.dy_dy[i]
                        .iter()
                        .copied()
                        .filter(move |&j| j > i) // Ensure stable order
                        .filter_map(move |j| {
                            let key = (i, j);
                            if exclusions.contains(&key) {
                                return None;
                            }
                            let scale14 = scaled_set.contains(&key);
                            Some((i, j, scale14))
                        })
                })
                .collect();

            let per_atom_accum: Vec<Vec3> = pairs
                .par_iter()
                .fold(
                    || vec![Vec3::new_zero(); n_dyn],
                    |mut acc, &(i, j, scale14)| {
                        let f = f_nonbonded(
                            &self.atoms[i],
                            &self.atoms[j],
                            &self.cell,
                            scale14,
                            Some((i, j)),
                            None,
                            None,
                            &self.lj_table,
                            &self.lj_table_static,
                            &self.lj_table_water,
                            true,
                            true,
                        );

                        acc[i] += f;
                        acc[j] -= f;
                        acc
                    },
                )
                .reduce(
                    || vec![Vec3::new_zero(); n_dyn],
                    |mut a, b| {
                        // elementwise add
                        for k in 0..n_dyn {
                            a[k] += b[k];
                        }
                        a
                    },
                );

            for i in 0..n_dyn {
                self.atoms[i].accel += per_atom_accum[i];
            }
        }

        // ------ Forces from static atoms on dynamic ones ------
        {
            let pairs: Vec<(usize, usize)> = (0..n_dyn)
                .flat_map(|i_dyn| {
                    self.neighbors_nb.dy_static[i_dyn]
                        .iter()
                        .copied()
                        .map(move |j_st| (i_dyn, j_st))
                })
                .collect();

            let per_atom_accum: Vec<Vec3> = pairs
                .par_iter()
                .fold(
                    || vec![Vec3::new_zero(); n_dyn],
                    |mut acc, &(i_dyn, j_st)| {
                        let a_dyn = &self.atoms[i_dyn];
                        let a_static = &self.atoms_static[j_st];

                        let f = f_nonbonded(
                            a_dyn,
                            a_static,
                            &self.cell,
                            false,               // no 1-4 scaling with static
                            None,                // dy-dy key
                            Some((i_dyn, j_st)), // dy-static key
                            None,                // dy-water key
                            &self.lj_table,
                            &self.lj_table_static,
                            &self.lj_table_water,
                            true,
                            true,
                        );

                        acc[i_dyn] += f; // static atoms don't move
                        acc
                    },
                )
                .reduce(
                    || vec![Vec3::new_zero(); n_dyn],
                    |mut a, b| {
                        for k in 0..n_dyn {
                            a[k] += b[k];
                        }
                        a
                    },
                );

            for i in 0..n_dyn {
                self.atoms[i].accel += per_atom_accum[i];
            }
        }

        // ------ Forces from water molecules on dynamic atoms ------

        {
            let pairs: Vec<(usize, usize, u8)> = (0..n_dyn)
                .flat_map(|i_dyn| {
                    self.neighbors_nb.dy_water[i_dyn]
                        .iter()
                        .copied()
                        .flat_map(move |iw| {
                            [0_u8, 1, 2, 3]
                                .into_iter()
                                .map(move |site| (i_dyn, iw, site))
                        })
                })
                .collect();

            let per_atom_accum: Vec<Vec3> = pairs
                .par_iter()
                .fold(
                    || vec![Vec3::new_zero(); n_dyn],
                    |mut acc, &(i_dyn, i_water, which)| {
                        let a_dyn = &self.atoms[i_dyn];
                        let w = &self.water[i_water];
                        let a_water_src = match which {
                            0 => &w.o,
                            1 => &w.m,
                            2 => &w.h0,
                            _ => &w.h1,
                        };

                        let f = f_nonbonded(
                            a_dyn,
                            a_water_src,
                            &self.cell,
                            false,       // no 1-4 scaling with water
                            None,        // dy-dy key
                            None,        // dy-static key
                            Some(i_dyn), // dy-water key (matches your original call)
                            &self.lj_table,
                            &self.lj_table_static,
                            &self.lj_table_water,
                            true,
                            true,
                        );

                        acc[i_dyn] += f; // water is rigid/fixed here
                        acc
                    },
                )
                .reduce(
                    || vec![Vec3::new_zero(); n_dyn],
                    |mut a, b| {
                        for k in 0..n_dyn {
                            a[k] += b[k];
                        }
                        a
                    },
                );

            for i in 0..n_dyn {
                self.atoms[i].accel += per_atom_accum[i];
            }
        }

        {
            // Long‑range reciprocal‑space term (PME / SPME), both static and dynamic.
            // Build a temporary Vec with *all* charges so the mesh sees both
            // movable and rigid atoms.  We only add forces back to dynamic atoms. This section
            // does not use neighbor lists.
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
}

/// Vdw and Coulomb forces. Used by water and non-water.
/// We split out `r_sq` and `diff` for use while integrating a unit cell, if applicable.
pub fn f_nonbonded(
    tgt: &AtomDynamics,
    src: &AtomDynamics,
    cell: &SimBox,
    scale14: bool, // See notes earlier in this module.
    // For now, this caching optimization only to non-water interactions.
    // If it doesn't apply, this field is None.
    atom_indices: Option<(usize, usize)>,
    // (dynamic i, static i)
    atom_indices_static: Option<(usize, usize)>,
    atom_indices_water: Option<usize>,
    lj_table: &LjTable,
    // (dynamic i, static i)
    lj_table_static: &LjTable,
    lj_table_water: &HashMap<usize, (f64, f64)>,
    // These values are for use with water.
    calc_lj: bool,
    calc_coulomb: bool,
) -> Vec3 {
    let diff = tgt.posit - src.posit;
    let diff_wrapped = cell.min_image(diff);

    let dir = diff.to_normalized();
    let dist_sq = diff_wrapped.magnitude_squared();
    let dist = dist_sq.sqrt();

    // todo: This distance cutoff helps, but ideally we skip the distance computation too in these cases.
    let f_lj = if !calc_lj || dist_sq > CUTOFF_VDW_SQ {
        Vec3::new_zero()
    } else {
        let (σ, ε) = if let Some(indices) = atom_indices {
            // Dynamic-dynamic
            lj_table.get(&indices).unwrap()
        } else if let Some(indices) = atom_indices_static {
            // Note: Index order matters here, and the caveat for dynamic-dynamic interactions
            // doesn't.
            lj_table_static.get(&indices).unwrap()
        } else if let Some(i) = atom_indices_water {
            // Single index of the lig; the only water mol is the uniform O.
            lj_table_water.get(&i).unwrap()
        } else {
            // Water-water
            &(water_opc::O_SIGMA, water_opc::O_EPS)
        };

        // Negative due to our mix of conventions; keep it consistent with coulomb, and net correct.
        let mut f = -force_lj(dir, dist, *σ, *ε);
        if scale14 {
            f *= SCALE_LJ_14;
        }
        f
    };

    // We assume that in the AtomDynamics structs, charges are already scaled to Amber units.
    // (No longer in elementary charge)
    let mut f_coulomb = if !calc_coulomb {
        Vec3::new_zero()
    } else {
        force_coulomb_ewald_real(
            dir,
            dist,
            tgt.partial_charge,
            src.partial_charge,
            EWALD_ALPHA,
        )
    };

    // See Amber RM, section 15, "1-4 Non-Bonded Interaction Scaling"
    if scale14 {
        f_coulomb *= SCALE_COUL_14;
    }

    f_lj + f_coulomb
    // todo temp to test one or the other
    // f_coulomb
    // f_lj
}

/// Helper. Returns σ, ε between an atom pair. Atom order passed as params doesn't matter.
fn combine_lj_params(atom_0: &AtomDynamics, atom_1: &AtomDynamics) -> (f64, f64) {
    let σ = 0.5 * (atom_0.lj_sigma + atom_1.lj_sigma);
    let ε = (atom_0.lj_eps * atom_1.lj_eps).sqrt();

    (σ, ε)
}
