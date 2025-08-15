//! For VDW and Coulomb forces

use std::collections::{HashMap, HashSet};

use lin_alg::f64::Vec3;
use rayon::prelude::*;

use crate::{
    dynamics::{
        AtomDynamics, MdState,
        ambient::SimBox,
        spme::{EWALD_ALPHA, PME_MESH_SPACING, force_coulomb_ewald_real},
        water_opc,
        water_opc::ForcesOnWaterMol,
    },
    forces::force_lj,
    molecule::Atom,
};

// Å. 9-12 should be fine; there is very little VDW force > this range due to
// the ^-7 falloff.
pub const CUTOFF_VDW: f64 = 12.0;
// const CUTOFF_VDW_SQ: f64 = CUTOFF_VDW * CUTOFF_VDW;

// See Amber RM, section 15, "1-4 Non-Bonded Interaction Scaling"
// "Non-bonded interactions between atoms separated by three consecutive bonds... require a special
// treatment in Amber force fields."
// "By default, vdW 1-4 interactions are divided (scaled down) by a factor of 2.0, electrostatic 1-4 terms by a factor
// of 1.2."
const SCALE_LJ_14: f64 = 0.5;
pub const SCALE_COUL_14: f64 = 1.0 / 1.2;

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
#[derive(Debug)]
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
        LjTableIndices::WaterOnWater => (), // None; a single const.
    }
}

/// Helper / intermediate
#[derive(Clone, Copy)]
enum LjIndexType {
    DynDyn,
    DynStatic,
}

/// A helper. Applies non-bonded force in parallel over a set of atoms, with indices assigned
/// upstream.
///
/// Currently only works with static on dynamic, and dynamic on dynamic.
///
/// Return W, the virial pair we accumulate. For use with the temp/barostat. (kcal/mol)
/// todo: Use CUDA and SIMD here.
fn apply_force(
    pairs: &[(usize, usize, bool)],
    atoms_tgt: &mut [AtomDynamics],
    // If None, src are also targets. This avoids a double-borrow.
    atoms_src: Option<&[AtomDynamics]>,
    n_dyn: usize,
    cell: &SimBox,
    lj_type: LjIndexType,
    lj_tables: &LjTables,
) -> f64 {
    let indices = |i| match lj_type {
        LjIndexType::DynDyn => LjTableIndices::DynDyn(i),
        LjIndexType::DynStatic => LjTableIndices::DynStatic(i),
    };

    let src = match atoms_src {
        Some(src) => src,
        None => atoms_tgt,
    };

    let (per_atom_accum, w_sum) = pairs
        .par_iter()
        .fold(
            || (vec![Vec3::new_zero(); n_dyn], 0.0_f64),
            |(mut f_on_tgt, mut w_virial_local), &(i_tgt, i_src, scale14)| {
                let f = f_nonbonded(
                    Some(&mut w_virial_local),
                    &atoms_tgt[i_tgt],
                    &src[i_src],
                    cell,
                    scale14,
                    indices((i_tgt, i_src)),
                    lj_tables,
                    true,
                    true,
                );

                f_on_tgt[i_tgt] += f;
                if matches!(lj_type, LjIndexType::DynDyn) {
                    f_on_tgt[i_src] -= f;
                }
                (f_on_tgt, w_virial_local)
            },
        )
        .reduce(
            || (vec![Vec3::new_zero(); n_dyn], 0.0_f64),
            |(mut f, w_a), (b, w_b)| {
                for k in 0..n_dyn {
                    f[k] += b[k];
                }
                (f, w_a + w_b)
            },
        );

    for i in 0..n_dyn {
        // We divide by mass in `step`.
        atoms_tgt[i].accel += per_atom_accum[i];
    }

    // todo: Do we need to double this for the dynamic-dynamic case, because we apply the accel to each
    // todo atom in the pair because they act on each on each other, or not?
    w_sum // kcal/mol
}

impl MdState {
    /// Applies Coulomb and Van der Waals (Lennard-Jones) forces on dynamic atoms, in place.
    /// We use the MD-standard [S]PME approach to handle approximated Coulomb forces. This function
    /// applies forces from dynamic, static, and water sources.
    pub fn apply_nonbonded_forces(&mut self) {
        let n_dyn = self.atoms.len();
        let n_water_mols = self.water.len();

        // ------ Forces from other dynamic atoms on dynamic ones ------

        {
            // Exclusions and scaling apply to dynamic-dynamic interactions only.
            let exclusions = &self.pairs_excluded_12_13;
            let scaled_set = &self.pairs_14_scaled;

            // Set up pairs ahead of time; conducive to parallel iteration. We skip excluded pairs,
            // and mark scaled ones.
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

            self.barostat.virial_pair_kcal += apply_force(
                &pairs,
                &mut self.atoms,
                None,
                n_dyn,
                &self.cell,
                LjIndexType::DynDyn,
                &self.lj_tables,
            );
        }

        // ------ Forces from static atoms on dynamic ones ------
        {
            let pairs: Vec<(usize, usize, bool)> = (0..n_dyn)
                .flat_map(|i_dyn| {
                    self.neighbors_nb.dy_static[i_dyn]
                        .iter()
                        .copied()
                        .map(move |j_st| (i_dyn, j_st, false)) // No scaling with static atoms.
                })
                .collect();

            self.barostat.virial_pair_kcal += apply_force(
                &pairs,
                &mut self.atoms,
                Some(&self.atoms_static),
                n_dyn,
                &self.cell,
                LjIndexType::DynStatic,
                &self.lj_tables,
            );
        }

        // ------ Forces from water molecules on dynamic atoms, and vice-versa ------
        {
            // (i_dyn, i_water_mol, water atom index 0 - 3)
            let pairs: Vec<(usize, usize, u8)> = (0..n_dyn)
                .flat_map(|i_dyn| {
                    self.neighbors_nb.dy_water[i_dyn]
                        .iter()
                        .copied()
                        .flat_map(move |i_water| {
                            [0_u8, 1, 2, 3]
                                .into_iter()
                                .map(move |site| (i_dyn, i_water, site))
                        })
                })
                .collect();

            let (per_atom_dyn_accum, per_mol_water_accum, w_sum) = pairs
                .par_iter()
                .fold(
                    || {
                        (
                            vec![Vec3::new_zero(); n_dyn],
                            vec![ForcesOnWaterMol::default(); n_water_mols],
                            0.0_f64,
                        )
                    },
                    |(mut f_on_dyn, mut f_on_water, mut w_virial_local),
                     &(i_dyn, i_water, water_atom_i)| {
                        let a_dyn = &self.atoms[i_dyn];
                        let w = &self.water[i_water];

                        let (a_water_src, f_water_field, calc_lj, calc_coulomb) = match water_atom_i
                        {
                            0 => (&w.o, &mut f_on_water[i_water].f_o, true, false),
                            1 => (&w.m, &mut f_on_water[i_water].f_m, false, true),
                            2 => (&w.h0, &mut f_on_water[i_water].f_h0, false, true),
                            _ => (&w.h1, &mut f_on_water[i_water].f_h1, false, true),
                        };

                        let f = f_nonbonded(
                            Some(&mut w_virial_local),
                            a_dyn,
                            a_water_src,
                            &self.cell,
                            false, // No 1-4 scaling with water
                            LjTableIndices::DynOnWater(i_dyn),
                            &self.lj_tables,
                            calc_lj,
                            calc_coulomb,
                        );

                        f_on_dyn[i_dyn] += f;
                        *f_water_field -= f;

                        (f_on_dyn, f_on_water, w_virial_local)
                    },
                )
                .reduce(
                    || {
                        (
                            vec![Vec3::new_zero(); n_dyn],
                            vec![ForcesOnWaterMol::default(); n_water_mols],
                            0.0_f64,
                        )
                    },
                    |(mut f_dyn, mut f_water, w_a), (b_dyn, b_water, w_b)| {
                        for k in 0..n_dyn {
                            f_dyn[k] += b_dyn[k];
                        }
                        for k in 0..n_water_mols {
                            f_water[k].f_o += b_water[k].f_o;
                            f_water[k].f_m += b_water[k].f_m;
                            f_water[k].f_h0 += b_water[k].f_h0;
                            f_water[k].f_h1 += b_water[k].f_h1;
                        }
                        (f_dyn, f_water, w_a + w_b)
                    },
                );
            self.barostat.virial_pair_kcal += w_sum;

            for i in 0..n_dyn {
                self.atoms[i].accel += per_atom_dyn_accum[i];
            }
        }

        // todo:  You must add virial contribution from water molecules on other water molecules to the barostat.

        // todo; Removed: Pausing on thsi; we need to get it workign or change
        // todo from SPME, but it's a time sink, and not making any progress.
        // self.apply_long_range_recip_forces()
    }
}

/// Vdw and (short-range) Coulomb forces. Used by water and non-water.
/// We run long-range SPME Coulomb force separately.
///
/// We use a hard distance cutoff for Vdw, enabled by its ^-7 falloff.
pub fn f_nonbonded(
    virial_w: Option<&mut f64>,
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

    let dist = diff_wrapped.magnitude();

    if dist.abs() < 1e-12 {
        return Vec3::new_zero();
    }

    let dir = diff_wrapped / dist;

    let f_lj = if !calc_lj || dist > CUTOFF_VDW {
        Vec3::new_zero()
    } else {
        let (σ, ε) = match lj_indices {
            LjTableIndices::DynDyn(indices) => *lj_tables.dynamic.get(&indices).unwrap(),
            LjTableIndices::DynStatic(indices) => *lj_tables.static_.get(&indices).unwrap(),
            LjTableIndices::DynOnWater(i) => *lj_tables.water_dyn.get(&i).unwrap(),
            LjTableIndices::StaticOnWater(i) => *lj_tables.water_static.get(&i).unwrap(),
            LjTableIndices::WaterOnWater => (water_opc::O_SIGMA, water_opc::O_EPS),
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

    let result = f_lj + f_coulomb;

    if let Some(w) = virial_w {
        // Virial: r_ij · F_ij (use minimum-image)
        *w += diff_wrapped.dot(result);
    }

    // f_lj
    // f_coulomb
    // Vec3::new_zero()

    return f_lj; // todo temp
    // return f_coulomb;

    result
}

/// Helper. Returns σ, ε between an atom pair. Atom order passed as params doesn't matter.
/// Note that this uses the traditional algorithm; not the Amber-specific version: We pre-set
/// atom-specific σ and ε to traditional versions on ingest, and when building water.
fn combine_lj_params(atom_0: &AtomDynamics, atom_1: &AtomDynamics) -> (f64, f64) {
    let σ = 0.5 * (atom_0.lj_sigma + atom_1.lj_sigma);
    let ε = (atom_0.lj_eps * atom_1.lj_eps).sqrt();

    (σ, ε)
}
