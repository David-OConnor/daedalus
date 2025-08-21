//! For VDW and Coulomb forces

use std::{collections::HashMap, ops::AddAssign, sync::Arc};

use cudarc::driver::{CudaModule, CudaStream};
use ewald::force_coulomb_short_range;
use lin_alg::{f32::Vec3 as Vec3F32, f64::Vec3};
use rayon::prelude::*;

use crate::{
    ComputationDevice,
    dynamics::{
        AtomDynamics, MdState,
        ambient::SimBox,
        water_opc,
        water_opc::{O_EPS, O_SIGMA, WaterMol, WaterSite},
    },
    forces::{force_lj, force_nonbonded_gpu},
};

// Å. 9-12 should be fine; there is very little VDW force > this range due to
// the ^-7 falloff.
pub const CUTOFF_VDW: f64 = 12.0;
// const CUTOFF_VDW_SQ: f64 = CUTOFF_VDW * CUTOFF_VDW;

// Ewald SPME approximation for Coulomb force

// Above this distance when calculating Coulomb forces, use the long-range Ewald approximation.
// const EWALD_CUTOFF: f64 = 10.0; // Å. 9-10 is common.
// const EWALD_CUTOFF_SQ: f64 = EWALD_CUTOFF * EWALD_CUTOFF;

// Instead of a hard cutoff between short and long-range forces, these
// parameters control a smooth taper.
// Our neighbor list must use the same cutoff as this, so we use it directly.

// We don't use a taper, for now.
// const LONG_RANGE_SWITCH_START: f64 = 8.0; // start switching (Å)
pub const LONG_RANGE_CUTOFF: f64 = 10.0;

// A bigger α means more damping, and a smaller real-space contribution. (Cheaper real), but larger
// reciprocal load.
// Common rule for α: erfc(α r_c) ≲ 10⁻⁴…10⁻⁵
pub const EWALD_ALPHA: f64 = 0.35; // Å^-1. 0.35 is good for cutoff = 10.
pub const PME_MESH_SPACING: f64 = 1.0;
// SPME order‑4 B‑spline interpolation
pub const SPME_N: usize = 64;

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
    /// (tgt, src)
    DynDyn((usize, usize)),
    /// (dyn tgt, static src)
    DynStatic((usize, usize)),
    /// (dyn tgt or src))
    DynWater(usize),
    /// Index is the static atom.
    StaticWater(usize),
    /// One value, stored as a constant (Water O -> Water O)
    WaterWater,
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

impl LjTables {
    /// Get (σ, ε)
    pub fn lookup(&self, i: &LjTableIndices) -> (f64, f64) {
        match i {
            LjTableIndices::DynDyn(indices) => *self.dynamic.get(&indices).unwrap(),
            LjTableIndices::DynStatic(indices) => *self.static_.get(&indices).unwrap(),
            LjTableIndices::DynWater(i) => *self.water_dyn.get(&i).unwrap(),
            LjTableIndices::StaticWater(i) => *self.water_static.get(&i).unwrap(),
            LjTableIndices::WaterWater => (O_SIGMA, O_EPS),
        }
    }
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
        LjTableIndices::DynWater(i) => {
            tables.water_dyn.insert(i, (σ, ε));
        }
        LjTableIndices::StaticWater(i) => {
            tables.water_static.insert(i, (σ, ε));
        }
        LjTableIndices::WaterWater => (), // None; a single const.
    }
}

/// Per-water, per-site force accumulator. Used transiently when applying nonbonded forces.
/// This is the force *on* each atom in the molecule.
#[derive(Clone, Copy, Default)]
pub struct ForcesOnWaterMol {
    pub f_o: Vec3,
    pub f_h0: Vec3,
    pub f_h1: Vec3,
    /// SETTLE/constraint will redistribute force on M/EP.
    pub f_m: Vec3,
}

impl AddAssign<Self> for ForcesOnWaterMol {
    fn add_assign(&mut self, rhs: Self) {
        self.f_o += rhs.f_o;
        self.f_h0 += rhs.f_h0;
        self.f_h1 += rhs.f_h1;
        self.f_m += rhs.f_m;
    }
}

#[derive(Copy, Clone)]
enum BodyRef {
    Dyn(usize),
    Static(usize),
    Water { mol: usize, site: WaterSite },
}

impl BodyRef {
    fn get<'a>(
        &self,
        dyns: &'a [AtomDynamics],
        statics: &'a [AtomDynamics],
        waters: &'a [WaterMol],
    ) -> &'a AtomDynamics {
        match *self {
            BodyRef::Dyn(i) => &dyns[i],
            BodyRef::Static(i) => &statics[i],
            BodyRef::Water { mol, site } => match site {
                WaterSite::O => &waters[mol].o,
                WaterSite::M => &waters[mol].m,
                WaterSite::H0 => &waters[mol].h0,
                WaterSite::H1 => &waters[mol].h1,
            },
        }
    }
}

struct NonBondedPair {
    tgt: BodyRef,
    src: BodyRef,
    scale_14: bool,
    lj_indices: LjTableIndices,
    calc_lj: bool,
    calc_coulomb: bool,
    symmetric: bool,
}

/// Add a force into the right accumulator (dyn or water). Static never accumulates.
fn add_to_sink(
    sink_dyn: &mut [Vec3],
    sink_wat: &mut [ForcesOnWaterMol],
    body_type: BodyRef,
    f: Vec3,
) {
    match body_type {
        BodyRef::Dyn(i) => sink_dyn[i] += f,
        BodyRef::Water { mol, site } => match site {
            WaterSite::O => sink_wat[mol].f_o += f,
            WaterSite::M => sink_wat[mol].f_m += f,
            WaterSite::H0 => sink_wat[mol].f_h0 += f,
            WaterSite::H1 => sink_wat[mol].f_h1 += f,
        },
        BodyRef::Static(_) => (),
    }
}

/// Applies non-bonded force in parallel (thread-pool) over a set of atoms, with indices assigned
/// upstream.
///
/// Return the virial pair component we accumulate. For use with the temp/barostat. (kcal/mol)
fn calc_force(
    pairs: &[NonBondedPair],
    atoms_dyn: &[AtomDynamics],
    atoms_static: &[AtomDynamics],
    water: &[WaterMol],
    cell: &SimBox,
    lj_tables: &LjTables,
) -> (Vec<Vec3>, Vec<ForcesOnWaterMol>, f64) {
    let n_dyn = atoms_dyn.len();
    let n_wat = water.len();

    pairs
        .par_iter()
        .fold(
            || {
                (
                    vec![Vec3::new_zero(); n_dyn],
                    vec![ForcesOnWaterMol::default(); n_wat],
                    0.0_f64,
                )
            },
            |(mut acc_d, mut acc_w, mut w_local), p| {
                let a_t = p.tgt.get(atoms_dyn, atoms_static, water);
                let a_s = p.src.get(atoms_dyn, atoms_static, water);

                let f = f_nonbonded(
                    Some(&mut w_local),
                    a_t,
                    a_s,
                    cell,
                    p.scale_14,
                    &p.lj_indices,
                    lj_tables,
                    p.calc_lj,
                    p.calc_coulomb,
                );

                add_to_sink(&mut acc_d, &mut acc_w, p.tgt, f);
                if p.symmetric {
                    add_to_sink(&mut acc_d, &mut acc_w, p.src, -f);
                }

                (acc_d, acc_w, w_local)
            },
        )
        .reduce(
            || {
                (
                    vec![Vec3::new_zero(); n_dyn],
                    vec![ForcesOnWaterMol::default(); n_wat],
                    0.0_f64,
                )
            },
            |(mut f_on_dyn, mut f_on_water, virial_a), (db, wb, virial_b)| {
                for i in 0..n_dyn {
                    f_on_dyn[i] += db[i];
                }
                for i in 0..n_wat {
                    f_on_water[i].f_o += wb[i].f_o;
                    f_on_water[i].f_m += wb[i].f_m;
                    f_on_water[i].f_h0 += wb[i].f_h0;
                    f_on_water[i].f_h1 += wb[i].f_h1;
                }
                (f_on_dyn, f_on_water, virial_a + virial_b)
            },
        )
}

/// Instead of thread pools, uses the GPU.
#[cfg(feature = "cuda")]
fn calc_force_cuda(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    pairs: &[NonBondedPair],
    atoms_dyn: &[AtomDynamics],
    atoms_static: &[AtomDynamics],
    water: &[WaterMol],
    cell: &SimBox,
    lj_tables: &LjTables,
    cutoff_ewald: f64,
    alpha_ewald: f64,
) -> (Vec<Vec3>, Vec<ForcesOnWaterMol>, f64) {
    let n_dyn = atoms_dyn.len();
    let n_water = water.len();

    let n = pairs.len();

    let mut posits_tgt: Vec<Vec3F32> = Vec::with_capacity(n);
    let mut posits_src: Vec<Vec3F32> = Vec::with_capacity(n);

    let mut sigmas = Vec::with_capacity(n);
    let mut epss = Vec::with_capacity(n);

    let mut qs_tgt = Vec::with_capacity(n);
    let mut qs_src = Vec::with_capacity(n);

    let mut scale_14s = Vec::with_capacity(n);

    let mut tgt_is: Vec<u32> = Vec::with_capacity(n);
    let mut src_is: Vec<u32> = Vec::with_capacity(n);

    let mut calc_ljs = Vec::with_capacity(n);
    let mut calc_coulombs = Vec::with_capacity(n);
    let mut symmetric = Vec::with_capacity(n);

    // Unpack BodyRef to fields. It doesn't map neatly to CUDA flattening primitives.

    // These atom and water types are so the Kernel can assign to the correct output arrays.
    // 0 means Dyn, 1 means Water.
    let mut atom_types_tgt = vec![0; n];
    // 0 for not-water or N/A. 1 = O, 2 = M, 3 = H0, 4 = H1.
    // Pre-allocated to 0, which we use for dyn atom targets.
    let mut water_types_tgt = vec![0; n];

    let mut atom_types_src = vec![0; n];
    let mut water_types_src = vec![0; n];

    for (i, pair) in pairs.iter().enumerate() {
        let atom_tgt = match pair.tgt {
            BodyRef::Dyn(j) => {
                tgt_is.push(j as u32);
                &atoms_dyn[j]
            }
            BodyRef::Water { mol: j, site } => {
                tgt_is.push(j as u32);

                // Mark so the kernel will use the water output.
                atom_types_tgt[i] = 1;
                water_types_tgt[i] = site as u8;

                match site {
                    WaterSite::O => &water[j].o,
                    WaterSite::M => &water[j].m,
                    WaterSite::H0 => &water[j].h0,
                    WaterSite::H1 => &water[j].h1,
                }
            }
            _ => unreachable!(),
        };

        let atom_src = match pair.src {
            BodyRef::Dyn(j) => {
                src_is.push(j as u32);
                &atoms_dyn[j]
            }
            BodyRef::Static(j) => {
                src_is.push(j as u32);
                &atoms_static[j]
            }
            BodyRef::Water { mol: j, site } => {
                src_is.push(j as u32);

                // Mark so the kernel will use the water output. (In case of dyn/water symmetric)
                atom_types_src[i] = 1;
                water_types_src[i] = site as u8;
                match site {
                    WaterSite::O => &water[j].o,
                    WaterSite::M => &water[j].m,
                    WaterSite::H0 => &water[j].h0,
                    WaterSite::H1 => &water[j].h1,
                }
            }
        };

        // `into()` converts Vec3s from f64 to f32.
        posits_tgt.push(atom_tgt.posit.into());
        posits_src.push(atom_src.posit.into());

        let (σ, ε) = lj_tables.lookup(&pair.lj_indices);

        sigmas.push(σ as f32);
        epss.push(ε as f32);

        qs_tgt.push(atom_tgt.partial_charge as f32);
        qs_src.push(atom_src.partial_charge as f32);

        scale_14s.push(pair.scale_14);

        calc_ljs.push(pair.calc_lj);
        calc_coulombs.push(pair.calc_coulomb);
        symmetric.push(pair.symmetric);
    }

    // 1-4 scaling, and the symmetric case handled in the kernel.

    let cell_extent: Vec3F32 = cell.extent.into();

    let (f_on_dyn_f32, f_on_water, virial) = force_nonbonded_gpu(
        stream,
        module,
        &tgt_is,
        &src_is,
        &posits_tgt,
        &posits_src,
        &sigmas,
        &epss,
        &qs_tgt,
        &qs_src,
        &atom_types_tgt,
        &water_types_tgt,
        &atom_types_src,
        &water_types_src,
        &scale_14s,
        &calc_ljs,
        &calc_coulombs,
        &symmetric,
        cutoff_ewald as f32,
        alpha_ewald as f32,
        cell_extent,
        n_dyn,
        n_water,
    );

    // Convert back to f64.
    let f_on_dyn = f_on_dyn_f32.into_iter().map(|f| f.into()).collect();

    (f_on_dyn, f_on_water, virial as f64)
}

impl MdState {
    /// Run the appropriate force-computation function to get force on dynamic atoms, force
    /// on water atoms, and virial sum for the barostat. Uses GPU if available.
    fn apply_force(&mut self, dev: &ComputationDevice, pairs: &[NonBondedPair]) {
        let (f_on_dyn, f_on_water, virial) = match dev {
            ComputationDevice::Cpu => calc_force(
                &pairs,
                &self.atoms,
                &self.atoms_static,
                &self.water,
                &self.cell,
                &self.lj_tables,
            ),
            #[cfg(feature = "cuda")]
            ComputationDevice::Gpu((stream, module)) => calc_force_cuda(
                stream,
                module,
                &pairs,
                &self.atoms,
                &self.atoms_static,
                &self.water,
                &self.cell,
                &self.lj_tables,
                LONG_RANGE_CUTOFF,
                EWALD_ALPHA,
            ),
        };

        for (i, tgt) in self.atoms.iter_mut().enumerate() {
            let f_f64: Vec3 = f_on_dyn[i];
            tgt.accel += f_f64;
        }

        for (i, tgt) in self.water.iter_mut().enumerate() {
            let f = f_on_water[i];
            let f_0: Vec3 = f.f_o.into();
            let f_m: Vec3 = f.f_m.into();
            let f_h0: Vec3 = f.f_h0.into();
            let f_h1: Vec3 = f.f_h1.into();

            tgt.o.accel += f_0;
            tgt.m.accel += f_m;
            tgt.h0.accel += f_h0;
            tgt.h1.accel += f_h1;
        }

        self.barostat.virial_pair_kcal += virial;
    }

    /// Applies Coulomb and Van der Waals (Lennard-Jones) forces on dynamic atoms, in place.
    /// We use the MD-standard [S]PME approach to handle approximated Coulomb forces. This function
    /// applies forces from dynamic, static, and water sources.
    pub fn apply_nonbonded_forces(&mut self, dev: &ComputationDevice) {
        let n_dyn = self.atoms.len();
        let n_water_mols = self.water.len();

        let sites = [WaterSite::O, WaterSite::M, WaterSite::H0, WaterSite::H1];

        // todo: You can probably consolidate even further. Instead of calling apply_force
        // todo per each category, you can assemble one big set of pairs, and call it once.
        // todo: This has performance and probably code organization benefits. Maybe try
        // todo after you get the intial version working. Will have to add symmetric to pairs.

        // ------ Forces from other dynamic atoms on dynamic ones ------

        // Exclusions and scaling apply to dynamic-dynamic interactions only.
        let exclusions = &self.pairs_excluded_12_13;
        let scaled_set = &self.pairs_14_scaled;

        // Set up pairs ahead of time; conducive to parallel iteration. We skip excluded pairs,
        // and mark scaled ones. These pairs, in symmetric cases (e.g. dynamic-dynamic), only
        let mut pairs_dyn_dyn: Vec<_> = (0..n_dyn)
            .flat_map(|i_tgt| {
                self.neighbors_nb.dy_dy[i_tgt]
                    .iter()
                    .copied()
                    .filter(move |&j| j > i_tgt) // Ensure stable order
                    .filter_map(move |i_src| {
                        let key = (i_tgt, i_src);
                        if exclusions.contains(&key) {
                            return None;
                        }
                        let scale_14 = scaled_set.contains(&key);

                        Some(NonBondedPair {
                            tgt: BodyRef::Dyn(i_tgt),
                            src: BodyRef::Dyn(i_src),
                            scale_14,
                            lj_indices: LjTableIndices::DynDyn(key),
                            calc_lj: true,
                            calc_coulomb: true,
                            symmetric: true,
                        })
                    })
            })
            .collect();

        // Forces from static atoms on dynamic ones
        let mut pairs_dyn_static: Vec<_> = (0..n_dyn)
            .flat_map(|i_dyn| {
                self.neighbors_nb.dy_static[i_dyn]
                    .iter()
                    .copied()
                    .map(move |i_st| NonBondedPair {
                        tgt: BodyRef::Dyn(i_dyn),
                        src: BodyRef::Static(i_st),
                        // No scaling with static atoms.
                        scale_14: false,
                        lj_indices: LjTableIndices::DynStatic((i_dyn, i_st)),
                        calc_lj: true,
                        calc_coulomb: true,
                        symmetric: false,
                    })
            })
            .collect();

        // Forces from water on dynamic atoms, and vice-versa
        let mut pairs_dyn_water: Vec<_> = (0..n_dyn)
            .flat_map(|i_dyn| {
                self.neighbors_nb.dy_water[i_dyn]
                    .iter()
                    .copied()
                    .flat_map(move |i_water| {
                        sites.into_iter().map(move |site| NonBondedPair {
                            tgt: BodyRef::Dyn(i_dyn),
                            src: BodyRef::Water { mol: i_water, site },
                            scale_14: false,
                            // todo: Ensure you reverse it.
                            lj_indices: LjTableIndices::DynWater(i_dyn),
                            calc_lj: site == WaterSite::O,
                            calc_coulomb: site != WaterSite::O,
                            symmetric: true,
                        })
                    })
            })
            .collect();

        // Forces from static atoms on water molecules
        let mut pairs_water_static: Vec<_> = (0..n_water_mols)
            .flat_map(|i_water| {
                self.neighbors_nb.water_static[i_water]
                    .iter()
                    .copied()
                    .flat_map(move |i_st| {
                        sites.into_iter().map(move |site| NonBondedPair {
                            tgt: BodyRef::Water { mol: i_water, site },
                            src: BodyRef::Static(i_st),
                            scale_14: false,
                            lj_indices: LjTableIndices::StaticWater(i_st),
                            calc_lj: site == WaterSite::O,
                            calc_coulomb: site != WaterSite::O,
                            symmetric: false,
                        })
                    })
            })
            .collect();

        // ------ Water on water ------
        let mut pairs_water_water = Vec::new();

        for i_0 in 0..n_water_mols {
            for &i_1 in &self.neighbors_nb.water_water[i_0] {
                if i_1 <= i_0 {
                    continue;
                } // unique (i0,i1)

                for &site_0 in &sites {
                    for &site_1 in &sites {
                        let calc_lj = site_0 == WaterSite::O && site_1 == WaterSite::O;
                        let calc_coulomb = site_0 != WaterSite::O && site_1 != WaterSite::O;

                        if !(calc_lj || calc_coulomb) {
                            continue;
                        }

                        pairs_water_water.push(NonBondedPair {
                            tgt: BodyRef::Water {
                                mol: i_0,
                                site: site_0,
                            },
                            src: BodyRef::Water {
                                mol: i_1,
                                site: site_1,
                            },
                            scale_14: false,
                            lj_indices: LjTableIndices::WaterWater,
                            calc_lj,
                            calc_coulomb,
                            symmetric: true,
                        });
                    }
                }
            }
        }

        // todo: Consider just removing the functional parts above, and add to `pairs` directly.
        // Combine pairs into a single set; we compute in one parallel pass.
        let len_added = pairs_dyn_static.len()
            + pairs_dyn_water.len()
            + pairs_water_static.len()
            + pairs_water_water.len();

        let mut pairs = pairs_dyn_dyn;
        pairs.reserve(len_added);

        pairs.append(&mut pairs_dyn_static);
        pairs.append(&mut pairs_dyn_water);
        pairs.append(&mut pairs_water_static);
        pairs.append(&mut pairs_water_water);

        self.apply_force(dev, &pairs);
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
    lj_indices: &LjTableIndices,
    lj_tables: &LjTables,
    // These flags are for use with forces on water.
    calc_lj: bool,
    calc_coulomb: bool,
) -> Vec3 {
    let diff = tgt.posit - src.posit;
    let diff = cell.min_image(diff);

    // We compute these dist-related values once, and share them between
    // LJ and Coulomb.
    let dist_sq = diff.magnitude_squared();

    if dist_sq < 1e-12 {
        return Vec3::new_zero();
    }

    let dist = dist_sq.sqrt();
    let inv_dist = 1.0 / dist;
    let inv_dist_sq = inv_dist * inv_dist;
    let dir = diff * inv_dist;

    let f_lj = if !calc_lj || dist > CUTOFF_VDW {
        Vec3::new_zero()
    } else {
        let (σ, ε) = lj_tables.lookup(lj_indices);

        // Negative due to our mix of conventions; keep it consistent with coulomb, and net correct.
        let mut f = -force_lj(dir, inv_dist, inv_dist_sq, σ, ε);
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
        force_coulomb_short_range(
            dir,
            dist,
            inv_dist,
            inv_dist_sq,
            tgt.partial_charge,
            src.partial_charge,
            // (LONG_RANGE_SWITCH_START, LONG_RANGE_CUTOFF),
            LONG_RANGE_CUTOFF,
            EWALD_ALPHA,
        )

        // force_coulomb(dir, dist, tgt.partial_charge, src.partial_charge, 1e-6)
    };

    // See Amber RM, section 15, "1-4 Non-Bonded Interaction Scaling"
    if scale14 {
        f_coulomb *= SCALE_COUL_14;
    }

    let result = f_lj + f_coulomb;

    if let Some(w) = virial_w {
        // Virial: r_ij · F_ij (use minimum-image)
        *w += diff.dot(result);
    }

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
