//! For VDW and Coulomb forces

use std::collections::HashMap;

use lin_alg::f64::Vec3;
use rayon::prelude::*;

use crate::{
    dynamics::{
        AtomDynamics, MdState,
        ambient::SimBox,
        spme::{EWALD_ALPHA, PME_MESH_SPACING, force_coulomb_ewald_real, pme_long_range_forces},
        water_opc,
    },
    forces::force_lj,
};
use crate::dynamics::spme::force_coulomb_ewald_complement;

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

// Multiply by this to convert partial charges from elementary charge (What we store in Atoms loaded from mol2
// files and amino19.lib.) to the self-consistent amber units required to calculate Coulomb force.
// We apply this to dynamic and static atoms when building Indexed params, and to water molecules
// on their construction.
pub const CHARGE_UNIT_SCALER: f64 = 18.2223;

// (indices), (sigma, eps)
pub type LjTable = HashMap<(usize, usize), (f64, f64)>;

/// We use this to load the correct data from LJ lookup tables. Since we use indices,
/// we must index correctly into the dynamic, or static tables. We have single-index lookups
/// for atoms acting on water, since there is only one O LJ type.
pub enum LjTableIndices {
    DynDyn((usize, usize)),
    DynStatic((usize, usize)),
    DynOnWater(usize),
    StaticOnWater(usize),
    /// One value, stored as a constant (Water O -> Water O)
    WaterOnWater,
}

/// We cache sigma and eps on the first step, then use it on the others. This increases
/// memory use, and reduces CPU use.
#[derive(Default)]
pub struct LjTables {
    /// Keys: (Dynamic, Dynamic). For acting on dynamic atoms.
    pub dynamic: LjTable,
    /// Keys: (Dynamic, Static). For acting on dynamic atoms.
    pub static_: LjTable,
    /// Keys: Dynamic. Water acting on water O.
    /// Water tables are simpler than ones on dynamic: no combinations needed, as the source is a single
    /// target atom type: O (water).
    pub water_dyn: HashMap<usize, (f64, f64)>,
    /// Keys: Static. For acting on water O.
    pub water_static: HashMap<usize, (f64, f64)>,
}

/// Run this once per MD run. Sets up LJ caches for each pair of atoms.
/// Mirrors the force fn's params.
pub fn setup_lj_cache(
    tgt: &AtomDynamics,
    src: &AtomDynamics,
    indices: LjTableIndices,
    tables: &mut LjTables,
) {
    let (σ, ε) = combine_lj_params(tgt, src);

    match indices {
        LjTableIndices::DynDyn(indices) => {
            tables.dynamic.insert(indices, (σ, ε));
        }
        LjTableIndices::DynStatic(indices) => {
            tables.static_.insert(indices, (σ, ε));
        }
        LjTableIndices::DynOnWater(i) => {
            tables.water_dyn.insert(i, (σ, ε));
        }
        LjTableIndices::StaticOnWater(i) => {
            tables.water_static.insert(i, (σ, ε));
        }
        LjTableIndices::WaterOnWater => () // None; a single const.
    }
}

impl MdState {
    /// Coulomb and Van der Waals (Lennard-Jones) forces on dynamic atoms. We use the MD-standard [S]PME approach
    /// to handle approximated Coulomb forces. This function applies forces from dynamic, static,
    /// and water sources.
    ///
    /// We use a hard distance cutoff for Vdw, due to its  ^-7 falloff.
    pub fn apply_nonbonded_forces(&mut self) {
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
                            LjTableIndices::DynDyn((i, j)),
                            &self.lj_tables,
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
                            false,  // No 1-4 scaling with static
                            LjTableIndices::DynStatic((i_dyn, j_st)),
                            &self.lj_tables,
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
                    |mut acc, &(i_dyn, i_water, water_atom_i)| {
                        let a_dyn = &self.atoms[i_dyn];
                        let w = &self.water[i_water];

                        let (a_water_src, calc_lj, calc_coulomb) = match water_atom_i {
                            0 => (&w.o, true, false),
                            1 => (&w.m, false, true),
                            2 => (&w.h0, false, true),
                            _ => (&w.h1, false, true),
                        };

                        let f = f_nonbonded(
                            a_dyn,
                            a_water_src,
                            &self.cell,
                            false, // No 1-4 scaling with water
                            LjTableIndices::DynOnWater(i_dyn),
                            &self.lj_tables,
                            calc_lj,
                            calc_coulomb,
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

            // These are teh water atoms that have Coulomb force; not O.
            // Separate from all_atoms because they make a [&AtomDynamics] instead of &[AtomDynamics].
            let mut atoms_water = Vec::with_capacity(self.water.len() * 3);
            for mol in &self.water {
                atoms_water.extend([&mol.m, &mol.h0, &mol.h1]);
            }

            let rec_forces = pme_long_range_forces(
                &all_atoms,   // dynamic+static as &[AtomDynamics]
                &atoms_water, // [&AtomDynamics] for (M,H0,H1) per water
                &self.cell,
                EWALD_ALPHA,
                PME_MESH_SPACING,
            );

            // (1) dynamic atoms
            for (atom, f_rec) in self
                .atoms
                .iter_mut()
                .zip(rec_forces.dyn_static.iter().take(n_dynamic))
            {
                atom.accel += *f_rec; // mass divide happens later in step()
            }

            // (2) waters → pack into per-water triples (M,H0,H1)
            debug_assert_eq!(rec_forces.water.len(), self.water.len() * 3);

            if self.water_pme_sites_forces.len() != self.water.len() {
                self.water_pme_sites_forces
                    .resize(self.water.len(), [Vec3::new_zero(); 3]);
            }

            for (iw, chunk) in rec_forces.water.chunks_exact(3).enumerate() {
                // order must match how you built atoms_water: [M, H0, H1]
                self.water_pme_sites_forces[iw] = [chunk[0], chunk[1], chunk[2]];
            }
        }

        // === 1-4 reciprocal-space scaling via complement (pairwise) ==================
        // Goal: net Coulomb(1-4) = SCALE_COUL_14 * (real + reciprocal).
        // We already scaled the real-space piece pairwise. Here we add
        // ΔF = (SCALE_COUL_14 - 1) * F_recip(pair), and F_recip(pair) = F_comp(pair).
        let corr = SCALE_COUL_14 - 1.0; // e.g., 1/1.2 - 1 = -1/6

        for &(i, j) in &self.nonbonded_scaled {
            let rij = self.cell.min_image(self.atoms[i].posit - self.atoms[j].posit);
            let r2 = rij.magnitude_squared();
            if r2 < 1e-12 { continue; }
            let r = r2.sqrt();
            let dir = rij / r;

            let qi = self.atoms[i].partial_charge;
            let qj = self.atoms[j].partial_charge;

            let f_comp = force_coulomb_ewald_complement(dir, r, qi, qj, EWALD_ALPHA);
            let df = f_comp * corr;

            // Newton's third law
            self.atoms[i].accel += df;
            self.atoms[j].accel -= df;
        }
        // === end 1-4 PME correction ===================================================
    }
}

/// Vdw and Coulomb forces. Used by water and non-water. For Coulomb, this is the short-range version.
/// We handle long-range SPME Coulomb force spearately.
/// We split out `r_sq` and `diff` for use while integrating a unit cell, if applicable.
pub fn f_nonbonded(
    tgt: &AtomDynamics,
    src: &AtomDynamics,
    cell: &SimBox,
    scale14: bool, // See notes earlier in this module.
    lj_indices: LjTableIndices,
    lj_tables: &LjTables,
    // These flags are for use with forces on water.
    calc_lj: bool,
    calc_coulomb: bool,
) -> Vec3 {
    let diff = tgt.posit - src.posit;
    let diff_wrapped = cell.min_image(diff);

    let dist_sq = diff_wrapped.magnitude_squared();
    let dir = diff_wrapped.to_normalized();

    if dist_sq.abs() < 1e-12 {
        return Vec3::new_zero();
    }

    let dist = dist_sq.sqrt();

    let f_lj = if !calc_lj || dist_sq > CUTOFF_VDW_SQ {
        Vec3::new_zero()
    } else {
        let (σ, ε) = match lj_indices {
            LjTableIndices::DynDyn(indices) => {
                *lj_tables.dynamic.get(&indices).unwrap()
            }
            LjTableIndices::DynStatic(indices) => {
                *lj_tables.static_.get(&indices).unwrap()
            }
            LjTableIndices::DynOnWater(i) => {
                *lj_tables.water_dyn.get(&i).unwrap()
            }
            LjTableIndices::StaticOnWater(i) => {
                *lj_tables.water_static.get(&i).unwrap()
            }
            LjTableIndices::WaterOnWater => (water_opc::O_SIGMA, water_opc::O_EPS)
        };


        // Negative due to our mix of conventions; keep it consistent with coulomb, and net correct.
        let mut f = -force_lj(dir, dist, σ, ε);
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
}

/// Helper. Returns σ, ε between an atom pair. Atom order passed as params doesn't matter.
fn combine_lj_params(atom_0: &AtomDynamics, atom_1: &AtomDynamics) -> (f64, f64) {
    let σ = 0.5 * (atom_0.lj_sigma + atom_1.lj_sigma);
    let ε = (atom_0.lj_eps * atom_1.lj_eps).sqrt();

    (σ, ε)
}
