//! This module contains code for maintaining non-bonded neighbor lists.
//! This is an optimization to determine which atoms we count in Lennard Jones (Van der Waals)
//!, and short-term Ewald Coulomb interactions.

use std::time::Instant;

use lin_alg::f64::Vec3;

use crate::dynamics::{
    AtomDynamics, CUTOFF_NEIGHBORS, MdState, SKIN, SKIN_SQ_DIV_4, ambient::SimBox,
    water_opc::WaterMol,
};

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
    pub ref_pos_water_o: Vec<Vec3>, // use O as proxy for the rigid water
}

impl MdState {
    /// Call during each step; determines if we need to rebuild neighbors, and does so A/R.
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
            let mut water_atoms = Vec::with_capacity(self.water.len());
            // todo: Fix this clone.
            for mol in &self.water {
                water_atoms.push(mol.o.clone());
            }

            build_neighbors(
                &mut self.neighbors_nb.dy_dy,
                &self.atoms,
                &self.atoms,
                &self.cell,
                true,
            );

            build_neighbors(
                &mut self.neighbors_nb.dy_static,
                &self.atoms,
                &self.atoms_static,
                &self.cell,
                false,
            );

            build_neighbors(
                &mut self.neighbors_nb.dy_water,
                &self.atoms,
                &water_atoms,
                &self.cell,
                false,
            );
            // todo: Helper; DRY between this and init.
            // Now invert dy_water -> water_dy
            let n_water = water_atoms.len();
            self.neighbors_nb.water_dy = vec![Vec::new(); n_water];

            for (dyn_idx, waters) in self.neighbors_nb.dy_water.iter().enumerate() {
                for &water_idx in waters {
                    self.neighbors_nb.water_dy[water_idx].push(dyn_idx);
                }
            }

            rebuilt_dyn = true;
        }

        if wat_disp_sq > SKIN_SQ_DIV_4 {
            let mut water_atoms = Vec::with_capacity(self.water.len());
            // todo: Fix this clone.
            for mol in &self.water {
                water_atoms.push(mol.o.clone());
            }

            build_neighbors(
                &mut self.neighbors_nb.water_static,
                &water_atoms,
                &self.atoms_static,
                &self.cell,
                false,
            );

            build_neighbors(
                &mut self.neighbors_nb.water_water,
                &water_atoms,
                &water_atoms,
                &self.cell,
                true,
            );

            if !rebuilt_dyn {
                // Don't double-run this, but it's required for both paths.
                build_neighbors(
                    &mut self.neighbors_nb.dy_water,
                    &self.atoms,
                    &water_atoms,
                    &self.cell,
                    false,
                );
            }

            rebuilt_wat = true;
        }

        // Rebuild reference position lists for next use, for use with determining when to rebuild the neighbor list.
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

        if rebuilt_dyn || rebuilt_wat {
            let elapsed = start.elapsed();
            println!("Neighbor build time: {:?} μs", elapsed.as_micros());
        } else {
            println!("No rebuild needed.");
        }
    }
}

/// [Re]build a neighbor list, used for non-bonded interactions. Run this periodically.
pub fn build_neighbors(
    neighbors: &mut Vec<Vec<usize>>,
    // todo: Consider accepting target and source posits to avoid cloning water atoms.
    targets: &[AtomDynamics],
    sources: &[AtomDynamics],
    cell: &SimBox,
    symmetric: bool,
) {
    const CUTOFF_SKIN_SQ: f64 = (CUTOFF_NEIGHBORS + SKIN) * (CUTOFF_NEIGHBORS + SKIN);

    neighbors.clear();
    neighbors.resize(targets.len(), Vec::new());

    let src_len = sources.len();
    let tgt_len = targets.len();

    let mut inner = |i_src: usize, i_tgt: usize| {
        let dv = cell.min_image(targets[i_tgt].posit - sources[i_src].posit);
        if dv.magnitude_squared() < CUTOFF_SKIN_SQ {
            neighbors[i_tgt].push(i_src);

            if symmetric {
                neighbors[i_src].push(i_tgt);
            }
        }
    };

    if symmetric {
        assert_eq!(src_len, tgt_len);
        for i_tgt in 0..tgt_len {
            for i_src in i_tgt + 1..src_len {
                inner(i_src, i_tgt);
            }
        }
    } else {
        for i_tgt in 0..tgt_len {
            for i_src in 0..src_len {
                inner(i_src, i_tgt);
            }
        }
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
        let d = cell.min_image(*posit - neighbor_ref_posits[i]);
        result = result.max(d.magnitude_squared());
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
