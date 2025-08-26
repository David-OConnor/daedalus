use std::collections::VecDeque;

use lin_alg::f64::Vec3;

use crate::{
    docking::{DockingSite, GRID_SPACING_SITE_FINDING},
    molecule::MoleculePeptide,
};

/// Attempt to find docking sites, using cavity detection.
pub fn find_docking_sites(mol: &MoleculePeptide) -> Vec<DockingSite> {
    // todo: Super chatGPT rough!!

    let mut result = Vec::new();

    // 1. Determine bounding box of the molecule
    let (mut min_x, mut min_y, mut min_z) = (f64::MAX, f64::MAX, f64::MAX);
    let (mut max_x, mut max_y, mut max_z) = (f64::MIN, f64::MIN, f64::MIN);

    if mol.atoms.is_empty() {
        return result; // No atoms, no sites
    }

    for atom in &mol.atoms {
        let p = atom.posit;
        if p.x < min_x {
            min_x = p.x;
        }
        if p.y < min_y {
            min_y = p.y;
        }
        if p.z < min_z {
            min_z = p.z;
        }
        if p.x > max_x {
            max_x = p.x;
        }
        if p.y > max_y {
            max_y = p.y;
        }
        if p.z > max_z {
            max_z = p.z;
        }
    }

    // Pad the bounding box slightly, to ensure we capture surface
    let atom_radius_temp: f64 = 2.0; // todo??
    let pad = 2.0 * atom_radius_temp;
    min_x -= pad;
    min_y -= pad;
    min_z -= pad;
    max_x += pad;
    max_y += pad;
    max_z += pad;

    // Helper function to see if a point is "inside" (within any atom's radius).
    let is_inside_molecule = |x: f64, y: f64, z: f64| -> bool {
        let pt = Vec3 { x, y, z };
        for atom in &mol.atoms {
            let dx = pt.x - atom.posit.x;
            let dy = pt.y - atom.posit.y;
            let dz = pt.z - atom.posit.z;
            let dist2 = dx * dx + dy * dy + dz * dz;
            // Compare squared distances to avoid sqrt call
            if dist2 < atom.element.vdw_radius().powi(2) as f64 {
                // Clashes with an atom => "inside" the molecule volume
                return true;
            }
        }
        false
    };

    // 2. Discretize bounding box into 3D grid
    let nx = ((max_x - min_x) / GRID_SPACING_SITE_FINDING).ceil() as usize;
    let ny = ((max_y - min_y) / GRID_SPACING_SITE_FINDING).ceil() as usize;
    let nz = ((max_z - min_z) / GRID_SPACING_SITE_FINDING).ceil() as usize;

    // We'll store a 3D array of booleans:
    //   true  => inside the molecule
    //   false => empty space
    // For convenience, flatten it into 1D: index = (ix + nx * (iy + ny * iz)).
    let mut grid = vec![false; nx * ny * nz];

    // 3. Fill the grid
    let mut index = 0;
    for iz in 0..nz {
        let zc = min_z + (iz as f64) * GRID_SPACING_SITE_FINDING;
        for iy in 0..ny {
            let yc = min_y + (iy as f64) * GRID_SPACING_SITE_FINDING;
            for ix in 0..nx {
                let xc = min_x + (ix as f64) * GRID_SPACING_SITE_FINDING;
                grid[index] = is_inside_molecule(xc, yc, zc);
                index += 1;
            }
        }
    }

    // 4. We want to find "interior pockets" => empty regions not connected to "outside."
    //    We'll do a flood fill from the boundaries of the grid to mark externally connected empty space.
    let mut visited = vec![false; grid.len()];

    // 4a. Helper to convert (ix, iy, iz) -> linear index
    let to_index = |ix: usize, iy: usize, iz: usize| ix + nx * (iy + ny * iz);

    // BFS function to mark connected empties from a starting cell
    let mut queue = VecDeque::new();
    let neighbors = |ix: usize, iy: usize, iz: usize| -> Vec<(usize, usize, usize)> {
        let mut neighs = Vec::new();
        let ix_i = ix as isize;
        let iy_i = iy as isize;
        let iz_i = iz as isize;
        for (dx, dy, dz) in &[
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ] {
            let nx_i = ix_i + dx;
            let ny_i = iy_i + dy;
            let nz_i = iz_i + dz;
            if nx_i >= 0
                && (nx_i as usize) < nx
                && ny_i >= 0
                && (ny_i as usize) < ny
                && nz_i >= 0
                && (nz_i as usize) < nz
            {
                neighs.push((nx_i as usize, ny_i as usize, nz_i as usize));
            }
        }
        neighs
    };

    // Mark all external empty cells by BFS from the "outer walls"
    // Because the outer boundary is definitely "outside" (just empty space away from the molecule).
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in [0, nz - 1] {
                let idx = to_index(ix, iy, iz);
                if !grid[idx] && !visited[idx] {
                    visited[idx] = true;
                    queue.push_back((ix, iy, iz));
                }
            }
        }
    }
    for iy in 0..ny {
        for iz in 0..nz {
            for ix in [0, nx - 1] {
                let idx = to_index(ix, iy, iz);
                if !grid[idx] && !visited[idx] {
                    visited[idx] = true;
                    queue.push_back((ix, iy, iz));
                }
            }
        }
    }
    for iz in 0..nz {
        for ix in 0..nx {
            for iy in [0, ny - 1] {
                let idx = to_index(ix, iy, iz);
                if !grid[idx] && !visited[idx] {
                    visited[idx] = true;
                    queue.push_back((ix, iy, iz));
                }
            }
        }
    }

    // Flood-fill from these boundary empties to mark them visited
    while let Some((cx, cy, cz)) = queue.pop_front() {
        for (nx_, ny_, nz_) in neighbors(cx, cy, cz) {
            let nidx = to_index(nx_, ny_, nz_);
            if !grid[nidx] && !visited[nidx] {
                visited[nidx] = true;
                queue.push_back((nx_, ny_, nz_));
            }
        }
    }

    // 4b. Now, anything that remains "false" in 'visited' & also false in 'grid' is an unvisited empty cell
    // => a pocket cell. We'll do a BFS over those to find distinct pockets.
    let mut pocket_id = vec![None; grid.len()];
    let mut current_pocket_label = 0;
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let idx = to_index(ix, iy, iz);
                // If it's inside the molecule, or it's visited from outside, skip
                if grid[idx] || visited[idx] {
                    continue;
                }
                // We found an unvisited empty cell => new pocket
                current_pocket_label += 1;
                let label = current_pocket_label;
                // BFS from here to label the entire pocket
                let mut queue2 = VecDeque::new();
                queue2.push_back((ix, iy, iz));
                pocket_id[idx] = Some(label);

                while let Some((cx, cy, cz)) = queue2.pop_front() {
                    for (nx_, ny_, nz_) in neighbors(cx, cy, cz) {
                        let nidx = to_index(nx_, ny_, nz_);
                        if !grid[nidx] && !visited[nidx] && pocket_id[nidx].is_none() {
                            pocket_id[nidx] = Some(label);
                            queue2.push_back((nx_, ny_, nz_));
                        }
                    }
                }
            }
        }
    }

    // 5. Compute a centroid for each pocket, then push a `DockingInit`.
    if current_pocket_label == 0 {
        // No pockets
        return result;
    }

    // Collect all the points for each pocket
    let mut sums = vec![(0.0, 0.0, 0.0, 0usize); current_pocket_label + 1];
    // sums[label] = (sumX, sumY, sumZ, count)
    for iz in 0..nz {
        let zc = min_z + (iz as f64) * GRID_SPACING_SITE_FINDING;
        for iy in 0..ny {
            let yc = min_y + (iy as f64) * GRID_SPACING_SITE_FINDING;
            for ix in 0..nx {
                let xc = min_x + (ix as f64) * GRID_SPACING_SITE_FINDING;
                let idx = to_index(ix, iy, iz);
                if let Some(label) = pocket_id[idx] {
                    let (sx, sy, sz, c) = sums[label];
                    sums[label] = (sx + xc, sy + yc, sz + zc, c + 1);
                }
            }
        }
    }

    // For each pocket, compute centroid => push a DockingInit
    // We also do a very rough bounding size by scanning the extents
    // of the pocket's grid points.
    let mut min_xyzs = vec![(f64::MAX, f64::MAX, f64::MAX); current_pocket_label + 1];
    let mut max_xyzs = vec![(f64::MIN, f64::MIN, f64::MIN); current_pocket_label + 1];

    for iz in 0..nz {
        let zc = min_z + (iz as f64) * GRID_SPACING_SITE_FINDING;
        for iy in 0..ny {
            let yc = min_y + (iy as f64) * GRID_SPACING_SITE_FINDING;
            for ix in 0..nx {
                let xc = min_x + (ix as f64) * GRID_SPACING_SITE_FINDING;
                let idx = to_index(ix, iy, iz);
                if let Some(label) = pocket_id[idx] {
                    let (min_xv, min_yv, min_zv) = min_xyzs[label];
                    let (max_xv, max_yv, max_zv) = max_xyzs[label];
                    min_xyzs[label] = (min_xv.min(xc), min_yv.min(yc), min_zv.min(zc));
                    max_xyzs[label] = (max_xv.max(xc), max_yv.max(yc), max_zv.max(zc));
                }
            }
        }
    }

    for label in 1..=current_pocket_label {
        let (sx, sy, sz, c) = sums[label];
        if c == 0 {
            continue;
        }
        let center = Vec3 {
            x: sx / (c as f64),
            y: sy / (c as f64),
            z: sz / (c as f64),
        };
        let (mnx, mny, mnz) = min_xyzs[label];
        let (mxx, mxy, mxz) = max_xyzs[label];
        // just use the largest dimension as site_box_size
        let dx = mxx - mnx;
        let dy = mxy - mny;
        let dz = mxz - mnz;
        let max_dim = dx.max(dy).max(dz);

        result.push(DockingSite {
            site_center: center,
            site_radius: max_dim,
        });
    }

    result
}
