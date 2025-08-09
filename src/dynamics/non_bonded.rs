//! For VDW and Coulomb forces

use std::collections::{HashMap, HashSet};

use lin_alg::f64::Vec3;
use rayon::prelude::*;

use crate::{
    dynamics::{
        AtomDynamics, CUTOFF_NEIGHBORS, ForceFieldParamsIndexed, LjTable, MdMode, MdState, SKIN,
        ambient::SimBox,
        prep::HydrogenMdType,
        spme::{EWALD_ALPHA, PME_MESH_SPACING, force_coulomb_ewald_real, pme_long_range_forces},
        water_opc,
        water_opc::make_water_mols,
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

#[derive(Default)]
/// Non-bonded neighbors; an important optimization for Van der Waals and Coulomb interactions.
/// By index for fast lookups; separate fields, as these indices have different meanings for dynamic atoms,
/// static atoms, and water.
///
/// To understand how we've set up the fields, each of the three types of atoms interactions with the others,
/// but note that static atoms are sources only; they are not acted on.
///
/// Note: These historically called "Verlet lists", but we're not using that term, as we use "Verlet" to refer
/// to the integrator, which this has nothing to do with. They do have to do with their applicability to
/// non-bonded interactions, so we call them "Non-bonded neighbors".
pub struct NeighborsNb {
    // Neighbors acting on dynamic atoms:
    /// Symmetric dynamic-dynamic indices. Dynamic source and target.
    pub dy_dy: Vec<Vec<usize>>,
    /// Outer: Dynamic. Inner: static. Dynamic target, static source.
    pub dy_static: Vec<Vec<usize>>,
    /// Outer: Dynamic. Inner: water. Each is a source and target.
    pub dy_water: Vec<Vec<usize>>,
    // Neighbors acting on water: (Dynamic acting on water is handled with `dy_water` above.
    /// Symmetric water-water indices. Dynamic source and target.
    pub water_water: Vec<Vec<usize>>,
    /// Outer: Water. Inner: static. Water target, static source.
    pub water_static: Vec<Vec<usize>>,
    //
    //
    // todo: Experimenting. See if you want/need these below:
    pub ref_pos_dyn: Vec<Vec3>,
    pub ref_pos_static: Vec<Vec3>,
    pub ref_pos_water_o: Vec<Vec3>, // use O as proxy for the rigid water
}

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
            let pairs_dy_dy: Vec<(usize, usize, bool)> = (0..n_dyn)
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

            let per_atom_accum: Vec<Vec3> = pairs_dy_dy
                .par_iter()
                .fold(
                    || vec![Vec3::new_zero(); n_dyn],
                    |mut acc, &(i, j, scale14)| {
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
                            &self.lj_table,
                            &self.lj_table_static,
                            &self.lj_table_water,
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

            // Apply accumulated forces
            for i in 0..n_dyn {
                self.atoms[i].accel += per_atom_accum[i];
            }
        }

        // ------ Forces from static atoms on dynamic ones ------
        {
            let n_static = self.atoms_static.len();
            let pairs_dy_static: Vec<(usize, usize)> = (0..n_dyn)
                .flat_map(|i_lig| (0..n_static).map(move |j_st| (i_lig, j_st)))
                .collect();

            // todo: QC n_static vs n_dyn h ere.
            let per_atom_accum: Vec<Vec3> = pairs_dy_static
                .par_iter()
                .fold(
                    || vec![Vec3::new_zero(); n_dyn],
                    |mut acc, &(i_dyn, j_st)| {
                        let a_lig = &self.atoms[i_dyn];
                        let a_static = &self.atoms_static[j_st];

                        let diff = a_lig.posit - a_static.posit;
                        let dv = self.cell.min_image(diff);
                        let r_sq = dv.magnitude_squared();

                        let f = f_nonbonded(
                            a_lig,
                            a_static,
                            r_sq,
                            diff,
                            false,               // no 1-4 scaling with static
                            None,                // dy-dy key
                            Some((i_dyn, j_st)), // dy-static key
                            None,                // dy-water key
                            &self.lj_table,
                            &self.lj_table_static,
                            &self.lj_table_water,
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

            // Apply accumulated forces
            for i in 0..n_dyn {
                self.atoms[i].accel += per_atom_accum[i];
            }
        }

        // ------ Forces from water molecules on dynamic atoms ------

        {
            let n_water = self.water.len() * 4;
            let pairs_dy_water: Vec<(usize, usize)> = (0..n_dyn)
                .flat_map(|i_lig| (0..n_static).map(move |j_st| (i_lig, j_st)))
                .collect();

            let per_atom_accum: Vec<Vec3> = pairs_dy_water
                .par_iter()
                .fold(
                    || vec![Vec3::new_zero(); n_dyn],
                    |mut acc, &(i_lig, iw, which)| {
                        let a_lig = &self.atoms[i_lig];
                        let w = &self.water[iw];
                        let a_water_src = match which {
                            0 => &w.o,
                            1 => &w.m,
                            2 => &w.h0,
                            _ => &w.h1,
                        };

                        let diff = a_lig.posit - a_water_src.posit;
                        let dv = self.cell.min_image(diff);
                        let r_sq = dv.magnitude_squared();

                        let f = f_nonbonded(
                            a_lig,
                            a_water_src,
                            r_sq,
                            diff,
                            false,       // no 1-4 scaling with water
                            None,        // dy-dy key
                            None,        // dy-static key
                            Some(i_lig), // dy-water key (matches your original call)
                            &self.lj_table,
                            &self.lj_table_static,
                            &self.lj_table_water,
                        );

                        acc[i_lig] += f; // water is rigid/fixed here
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

            // Apply accumulated forces
            for i in 0..n_dyn {
                self.atoms[i].accel += per_atom_accum[i];
            }
        }

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
    lj_table: &LjTable,
    // (dynamic i, static i)
    lj_table_static: &LjTable,
    lj_table_water: &HashMap<usize, (f64, f64)>,
) -> Vec3 {
    let dist = r_sq.sqrt();
    let dir = diff / dist;

    // todo: This distance cutoff helps, but ideally we skip the distance computation too in these cases.
    let f_lj = if r_sq > CUTOFF_VDW_SQ {
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
    let mut f_coulomb = force_coulomb_ewald_real(
        dir,
        dist,
        tgt.partial_charge,
        src.partial_charge,
        EWALD_ALPHA,
    );

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

impl MdState {
    pub fn new(
        mode: MdMode,
        atoms_dy: Vec<AtomDynamics>,
        atoms_static: Vec<AtomDynamics>,
        ff_params_non_static: ForceFieldParamsIndexed,
        temp_target: f64,
        hydrogen_md_type: HydrogenMdType,
        adjacency_list: Vec<Vec<usize>>,
    ) -> Self {
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

            SimBox {
                bounds_low: lo,
                bounds_high: hi,
            }
        };

        let mut result = Self {
            mode,
            atoms: atoms_dy,
            adjacency_list: adjacency_list.to_vec(),
            atoms_static,
            cell,
            nonbonded_exclusions: HashSet::new(),
            nonbonded_scaled: HashSet::new(),
            force_field_params: ff_params_non_static,
            temp_target,
            hydrogen_md_type,
            ..Default::default()
        };

        result.water = make_water_mols(
            &cell,
            result.temp_target,
            &result.atoms,
            &result.atoms_static,
        );

        result.setup_nonbonded_exclusion_scale_flags();
        result.build_neighbors();

        // Set up our LJ cache.
        if result.step_count == 0 {
            for i in 0..result.atoms.len() {
                for &j in &result.neighbors_nb.dy_dy[i] {
                    if j < i {
                        // Prevents duplication of the pair in the other order.
                        continue;
                    }

                    setup_lj_cache(
                        &result.atoms[i],
                        &result.atoms[j],
                        Some((i, j)),
                        None,
                        None,
                        &mut result.lj_table,
                        &mut result.lj_table_static,
                        &mut result.lj_table_water,
                    );
                }
            }

            for (i_lig, a_lig) in result.atoms.iter_mut().enumerate() {
                // Force from static atoms.
                for (i_static, a_static) in result.atoms_static.iter().enumerate() {
                    setup_lj_cache(
                        a_lig,
                        a_static,
                        None,
                        Some((i_lig, i_static)),
                        None,
                        &mut result.lj_table,
                        &mut result.lj_table_static,
                        &mut result.lj_table_water,
                    );
                }

                // Force from water
                if !result.water.is_empty() {
                    // Each water is identical, so we only need to do this once.
                    for a_water_src in [
                        &result.water[0].o,
                        &result.water[0].m,
                        &result.water[0].h0,
                        &result.water[0].h1,
                    ] {
                        setup_lj_cache(
                            a_lig,
                            a_water_src,
                            None,
                            None,
                            Some(i_lig),
                            &mut result.lj_table,
                            &mut result.lj_table_static,
                            &mut result.lj_table_water,
                        );
                    }
                }
            }
        }

        result
    }

    /// We use this to set up optimizations defined in the Amber reference manual. `excluded` deals
    /// with sections were we skip coulomb and Vdw interactions for atoms separated by 1 or 2 bonds. `scaled14` applies a force
    /// scaler for these interactions, when separated by 3 bonds.
    fn setup_nonbonded_exclusion_scale_flags(&mut self) {
        // Helper to store pairs in canonical (low,high) order
        let push = |set: &mut HashSet<(usize, usize)>, i: usize, j: usize| {
            if i < j {
                set.insert((i, j));
            } else {
                set.insert((j, i));
            }
        };

        // 1-2
        for indices in &self.force_field_params.bonds_topology {
            push(&mut self.nonbonded_exclusions, indices.0, indices.1);
        }

        // 1-3
        for (indices, _) in &self.force_field_params.angle {
            push(&mut self.nonbonded_exclusions, indices.0, indices.2);
        }

        // 1-4. We do not count improper dihedrals here.
        for (indices, _) in &self.force_field_params.dihedral {
            push(&mut self.nonbonded_scaled, indices.0, indices.3);
        }

        // Make sure no 1-4 pair is also in the excluded set
        for p in &self.nonbonded_scaled {
            self.nonbonded_exclusions.remove(p);
        }
    }

    /// [Re]build the neighbour list, used for non-bonded interactions. Run this periodically.
    /// todo: Is this for verlet integration, or non-bonded interactions?
    pub fn build_neighbors(&mut self) {
        const CUTOFF_SKIN_SQ: f64 = (CUTOFF_NEIGHBORS + SKIN) * (CUTOFF_NEIGHBORS + SKIN);

        self.neighbors_nb.dy_dy.clear();
        self.neighbors_nb.dy_water.clear();

        self.neighbors_nb.dy_dy = vec![Vec::new(); self.atoms.len()];
        for i in 0..self.atoms.len() - 1 {
            for j in i + 1..self.atoms.len() {
                let dv = self
                    .cell
                    .min_image(self.atoms[j].posit - self.atoms[i].posit);

                if dv.magnitude_squared() < CUTOFF_SKIN_SQ {
                    self.neighbors_nb.dy_dy[i].push(j);
                    self.neighbors_nb.dy_dy[j].push(i);
                }
            }
        }
    }

    /// For use with our non-bonded neighbors construction.
    pub fn max_displacement_sq_since_build(&self) -> f64 {
        let mut max_sq: f64 = 0.0;

        for (i, a) in self.atoms.iter().enumerate() {
            let d = self
                .cell
                .min_image(a.posit - self.neighbors_nb.ref_pos_dyn[i]);
            max_sq = max_sq.max(d.magnitude_squared());
        }

        for (i, a) in self.atoms_static.iter().enumerate() {
            let d = self
                .cell
                .min_image(a.posit - self.neighbors_nb.ref_pos_static[i]);
            max_sq = max_sq.max(d.magnitude_squared());
        }

        for (k, w) in self.water.iter().enumerate() {
            let d = self
                .cell
                .min_image(w.o.posit - self.neighbors_nb.ref_pos_water_o[k]);
            max_sq = max_sq.max(d.magnitude_squared());
        }
        max_sq
    }
}
