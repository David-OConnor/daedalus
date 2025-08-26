//! This module contains code for maintaining non-bonded neighbor lists.
//! This is an optimization to determine which atoms we count in Lennard Jones (Van der Waals)
//!, and short-term Ewald Coulomb interactions.
//!
//! Note: GPU is probably not a good fit for rebuilding neighbor lists.

use std::time::Instant;

use lin_alg::f64::Vec3;
use rayon::prelude::*;

use crate::dynamics::{
    AtomDynamics, MdState, ambient::SimBox, non_bonded::LONG_RANGE_CUTOFF, water_opc::WaterMol,
};

// These are for non-bonded neighbor list construction.
const SKIN: f64 = 2.0; // Å – rebuild list if an atom moved >½·SKIN. ~2Å.
const SKIN_SQ: f64 = SKIN * SKIN;
const SKIN_SQ_DIV_4: f64 = SKIN_SQ / 4.;

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
    /// Symmetric water-water indices. Dynamic source and target.
    pub water_water: Vec<Vec<usize>>,
    /// Outer: Water. Inner: static. Water target, static source.
    pub water_static: Vec<Vec<usize>>,
    /// Outer: Water. Inner: dynamic. Water target, dynamic source.
    /// todo: This is a direct reverse of dy_water, but may be worth keeping in for indexing order.
    pub water_dy: Vec<Vec<usize>>,
    //
    // Reference positions used when rebuilding. Only for movable atoms.
    pub ref_pos_dyn: Vec<Vec3>,
    /// Doesn't change.
    pub ref_pos_static: Vec<Vec3>,
    pub ref_pos_water_o: Vec<Vec3>, // use O as proxy for the rigid water
    /// Used to determine when to rebuild neighbor lists. todo: Implement.
    pub max_displacement_sq: f64,
}

impl MdState {
    /// Call during each step; determines if we need to rebuild neighbors, and does so A/R.
    /// todo: Run on GPU?
    pub fn build_neighbors_if_needed(&mut self) {
        let start = Instant::now();

        // Current positions
        let dyn_pos_now = positions_of(&self.atoms);
        let water_o_pos_now = positions_of_water_o(&self.water);

        // Displacements
        let dyn_disp_sq = max_displacement_sq_since_build(
            &dyn_pos_now,
            &self.neighbors_nb.ref_pos_dyn,
            &self.cell,
        );
        let wat_disp_sq = max_displacement_sq_since_build(
            &water_o_pos_now,
            &self.neighbors_nb.ref_pos_water_o,
            &self.cell,
        );

        let mut rebuilt_dyn = false;
        let mut rebuilt_wat = false;

        if dyn_disp_sq > SKIN_SQ_DIV_4 {
            self.neighbors_nb.dy_dy = build_neighbors(
                &self.neighbors_nb.ref_pos_dyn,
                &self.neighbors_nb.ref_pos_dyn,
                &self.cell,
                true,
            );

            self.neighbors_nb.dy_static = build_neighbors(
                &self.neighbors_nb.ref_pos_dyn,
                &self.neighbors_nb.ref_pos_static,
                &self.cell,
                false,
            );

            self.neighbors_nb.dy_water = build_neighbors(
                &self.neighbors_nb.ref_pos_dyn,
                &self.neighbors_nb.ref_pos_water_o,
                &self.cell,
                false,
            );
            self.rebuild_dy_water_inv();

            rebuilt_dyn = true;
        }

        if wat_disp_sq > SKIN_SQ_DIV_4 {
            self.neighbors_nb.water_static = build_neighbors(
                &self.neighbors_nb.ref_pos_water_o,
                &self.neighbors_nb.ref_pos_static,
                &self.cell,
                false,
            );

            self.neighbors_nb.water_water = build_neighbors(
                &self.neighbors_nb.ref_pos_water_o,
                &self.neighbors_nb.ref_pos_water_o,
                &self.cell,
                true,
            );

            if !rebuilt_dyn {
                // Don't double-run this, but it's required for both paths.
                self.neighbors_nb.dy_water = build_neighbors(
                    &self.neighbors_nb.ref_pos_dyn,
                    &self.neighbors_nb.ref_pos_water_o,
                    &self.cell,
                    false,
                );
                self.rebuild_dy_water_inv();
            }

            rebuilt_wat = true;
        }

        // Rebuild reference position lists for next use, for use with determining when to rebuild the neighbor list.
        // (Static refs doesn't get rebuilt after init)
        if rebuilt_dyn {
            for (i, a) in self.atoms.iter().enumerate() {
                self.neighbors_nb.ref_pos_dyn[i] = a.posit;
            }
        }

        if rebuilt_wat {
            for (i, m) in self.water.iter().enumerate() {
                self.neighbors_nb.ref_pos_water_o[i] = m.o.posit;
            }
        }

        static mut PRINTED: bool = false;
        if rebuilt_dyn || rebuilt_wat {
            let elapsed = start.elapsed();
            if !unsafe { PRINTED } {
                println!("Neighbor build time: {:?} μs", elapsed.as_micros());
                unsafe {
                    PRINTED = true;
                }
            }
        } else {
            // println!("No rebuild needed.");
        }
    }

    /// This inverts our neighbor set between water and dynamic atoms.
    pub fn rebuild_dy_water_inv(&mut self) {
        let n_waters = self.neighbors_nb.ref_pos_water_o.len();
        self.neighbors_nb.water_dy.clear();
        self.neighbors_nb.water_dy.resize(n_waters, Vec::new());

        // Iterate over what we actually have, not over atoms.len()
        for (i_dyn, ws) in self.neighbors_nb.dy_water.iter().enumerate() {
            for &iw in ws {
                debug_assert!(
                    iw < n_waters,
                    "water index out of range: {iw} >= {n_waters}"
                );
                self.neighbors_nb.water_dy[iw].push(i_dyn);
            }
        }
    }
}

/// [Re]build a neighbor list, used for non-bonded interactions. Run this periodically.
pub fn build_neighbors(
    tgt_posits: &[Vec3],
    src_posits: &[Vec3],
    cell: &SimBox,
    symmetric: bool,
) -> Vec<Vec<usize>> {
    const CUTOFF_SKIN_SQ: f64 = (LONG_RANGE_CUTOFF + SKIN) * (LONG_RANGE_CUTOFF + SKIN);

    let tgt_len = tgt_posits.len();
    let src_len = src_posits.len();

    if symmetric {
        assert_eq!(src_len, tgt_len, "symmetric=true requires identical sets");
        let n = tgt_len;

        let half: Vec<Vec<usize>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut out = Vec::new();
                let pi = tgt_posits[i];
                for j in (i + 1)..n {
                    let d = cell.min_image(pi - src_posits[j]);
                    if d.magnitude_squared() < CUTOFF_SKIN_SQ {
                        out.push(j);
                    }
                }
                out
            })
            .collect();

        let mut full = vec![Vec::<usize>::new(); n];
        for i in 0..n {
            full[i].reserve(half[i].len());
            for &j in &half[i] {
                full[i].push(j);
                full[j].push(i);
            }
        }
        full
    } else {
        (0..tgt_len)
            .into_par_iter()
            .map(|i_tgt| {
                let mut out = Vec::with_capacity(src_len);
                let pt = tgt_posits[i_tgt];
                for i_src in 0..src_len {
                    let d = cell.min_image(pt - src_posits[i_src]);
                    if d.magnitude_squared() < CUTOFF_SKIN_SQ {
                        out.push(i_src);
                    }
                }
                out
            })
            .collect()
    }
}

/// For use with our non-bonded neighbors construction.
pub fn max_displacement_sq_since_build(
    targets: &[Vec3],
    neighbor_ref_posits: &[Vec3],
    cell: &SimBox,
) -> f64 {
    let mut result: f64 = 0.0;

    for (i, posit) in targets.iter().enumerate() {
        let diff_min_img = cell.min_image(*posit - neighbor_ref_posits[i]);
        result = result.max(diff_min_img.magnitude_squared());
    }
    result
}

/// Helper
fn positions_of(atoms: &[AtomDynamics]) -> Vec<Vec3> {
    atoms.iter().map(|a| a.posit).collect()
}

/// Helper
fn positions_of_water_o(waters: &[WaterMol]) -> Vec<Vec3> {
    waters.iter().map(|w| w.o.posit).collect()
}
