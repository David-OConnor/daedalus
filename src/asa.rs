//! For calculating the accessible surface area (ASA), a proxy for acccessibility of solvents
//! to a molecule. Used for drawing *surface*, *dots*, and related meshes. Related to the van der Waals
//! radius.
//!
//! Uses the Shrake-Rupley, or similar "rolling ball" methods.
//! [This Rust lib](https://github.com/maxall41/RustSASA) appearse to be unsuitable to our purpose;
//! it provides a single 'total SASA value', vice a set of points defining a surface.
//!
//! [Check out this ChatGPT chat](https://chatgpt.com/c/67b37438-877c-8007-9894-551b9d244096) for ideas
//! on improvements and optimizations.

use std::{collections::HashMap, f32::consts::TAU};

use graphics::{Mesh, Vertex};
use lin_alg::f32::Vec3;

use crate::molecule::Atom;

const PROBE_RADIUS: f32 = 1.4; // Å "typical".
const NUM_PROBE_ANGLES: usize = 24; // todo: A/R.
const NUM_POINTS: usize = 100; // todo?

// these structs and methods are for our grid-based (?) optimization:
/// Simple bounding-box calculation for the set of atoms.
fn compute_bounding_box(atoms: &[Atom]) -> (Vec3, Vec3) {
    let mut min_pt = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
    let mut max_pt = Vec3::new(f32::MIN, f32::MIN, f32::MIN);

    for atom in atoms {
        let center: Vec3 = atom.posit.into();
        min_pt.x = min_pt.x.min(center.x);
        min_pt.y = min_pt.y.min(center.y);
        min_pt.z = min_pt.z.min(center.z);

        max_pt.x = max_pt.x.max(center.x);
        max_pt.y = max_pt.y.max(center.y);
        max_pt.z = max_pt.z.max(center.z);
    }

    (min_pt, max_pt)
}

/// Convert a world-space position to an integer grid cell coordinate (x,y,z).
fn position_to_grid(pos: Vec3, min_bounds: Vec3, cell_size: f32) -> (i32, i32, i32) {
    // Shift pos by min_bounds so the grid starts at (0,0,0).
    let shifted = pos - min_bounds;
    let ix = (shifted.x / cell_size).floor() as i32;
    let iy = (shifted.y / cell_size).floor() as i32;
    let iz = (shifted.z / cell_size).floor() as i32;
    (ix, iy, iz)
}

/// Build a uniform grid that maps each cell -> list of atom indices whose centers fall in that cell.
/// Also store the bounding box and cell_size so we can do lookups later.
struct AtomGrid {
    cell_size: f32,
    min_bounds: Vec3,
    // For convenience, store a HashMap of (ix, iy, iz) -> Vec<atom_index>
    cells: HashMap<(i32, i32, i32), Vec<usize>>,
}

impl AtomGrid {
    fn new(atoms: &[Atom], cell_size: f32) -> Self {
        let (min_bounds, max_bounds) = compute_bounding_box(atoms);

        // todo: Replace this with your pairs fn A/R.

        // Build the hash map of cell -> atom indices.
        let mut cells: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
        for (i, atom) in atoms.iter().enumerate() {
            let center: Vec3 = atom.posit.into();
            let (ix, iy, iz) = position_to_grid(center, min_bounds, cell_size);

            // Insert this atom index into that cell (and possibly neighbors).
            // But at minimum, insert into the cell of the atom's center:
            cells.entry((ix, iy, iz)).or_default().push(i);

            // Optionally, if you want to handle "large" atoms that might straddle
            // multiple cells, insert them in the neighboring 26 cells:
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if dx == 0 && dy == 0 && dz == 0 {
                            continue;
                        }
                        let nx = ix + dx;
                        let ny = iy + dy;
                        let nz = iz + dz;
                        cells.entry((nx, ny, nz)).or_default().push(i);
                    }
                }
            }
        }

        Self {
            cell_size,
            min_bounds,
            cells,
        }
    }

    /// Return a list of candidate atom indices for a given point by
    /// looking up that point's cell plus its neighbors.
    fn candidates_for_point(&self, pos: Vec3) -> Vec<usize> {
        let (ix, iy, iz) = position_to_grid(pos, self.min_bounds, self.cell_size);
        let mut candidates = Vec::new();
        // Collect atoms from the cell + neighbors
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let key = (ix + dx, iy + dy, iz + dz);
                    if let Some(atom_indices) = self.cells.get(&key) {
                        candidates.extend_from_slice(atom_indices);
                    }
                }
            }
        }
        candidates
    }
}

/// Proceeds inbound on a line between the starting point, and the origin, until the probe
/// intersects the VDW radius of an atom.
/// Calculate a (very) approximate set of 3D points for the Solvent Accessible Surface,
/// using a naive "rolling ball" method:
/// 1. For each atom, inflate its radius by PROBE_RADIUS.
/// 2. Sample points on this inflated sphere.
/// 3. Exclude points that are inside (occluded by) any other atom’s inflated sphere.
pub fn get_mesh_points(atoms: &[Atom]) -> Vec<Vec<Vec3>> {
    // We’ll store rings of sample points; each ring is a Vec<Vec3>.
    let steps_theta = 12; // how many slices around the longitude
    let steps_phi = 8; // how many slices around the latitude

    // --- 1) Build a grid for spatial acceleration ---
    // A typical choice: cell size ~ 2*(max_atom_radius + PROBE_RADIUS).
    // For proteins with typical radii around 1–2 Å, 6–8 is a decent guess.
    let max_atom_radius = atoms
        .iter()
        .map(|a| a.element.vdw_radius())
        .fold(0.0_f32, |acc, r| acc.max(r));
    let cell_size = 2.0 * (max_atom_radius + PROBE_RADIUS);

    let atom_grid = AtomGrid::new(atoms, cell_size);

    let mut mesh_rings = Vec::new();

    // --- 2) For each atom, sample the inflated sphere surface ---
    for (i, atom_i) in atoms.iter().enumerate() {
        let center_i: Vec3 = atom_i.posit.into();
        let inflated_radius_i = atom_i.element.vdw_radius() + PROBE_RADIUS;

        // We'll create rings by sweeping over phi in [0, π],
        // each ring subdivided by theta in [0, 2π).
        for p_i in 0..steps_phi {
            let phi = (TAU / 2.0) * p_i as f32 / steps_phi as f32;

            let mut ring_points = Vec::new();
            for t_i in 0..steps_theta {
                let theta = TAU * t_i as f32 / steps_theta as f32;

                // Convert spherical coords to Cartesian (unit vector):
                let sin_phi = phi.sin();
                let x = sin_phi * theta.cos();
                let y = sin_phi * theta.sin();
                let z = phi.cos();

                // Scale by the inflated radius and offset by the atom's center:
                let sample_pt = center_i + Vec3::new(x, y, z) * inflated_radius_i;

                // --- 3) Check if sample_pt is occluded by any other atom’s inflated sphere ---
                // Instead of checking *all* atoms, we only check
                // the atoms in the same cell + neighbors.
                let candidate_atom_indices = atom_grid.candidates_for_point(sample_pt);

                let mut occluded = false;
                for &j in &candidate_atom_indices {
                    // If it's the same atom, skip
                    if i == j {
                        continue;
                    }
                    let atom_j = &atoms[j];
                    let center_j: Vec3 = atom_j.posit.into();
                    let inflated_radius_j = atom_j.element.vdw_radius() + PROBE_RADIUS;

                    let dist = (sample_pt - center_j).magnitude();
                    if dist < inflated_radius_j {
                        occluded = true;
                        break;
                    }
                }

                if !occluded {
                    ring_points.push(sample_pt);
                }
            }

            if !ring_points.is_empty() {
                mesh_rings.push(ring_points);
            }
        }
    }

    mesh_rings
}

/// Build a naive triangle mesh from rings of SAS points.
///
/// - `rings`: Each `rings[i]` is a “ring” of points (e.g., from `get_mesh_points`).
/// - Returns a `Mesh` with:
///   - `vertices`: Flattened list of ring points (converted to `Vertex`s).
///   - `indices`: Triangles built between adjacent rings (i, i+1),
///     only if they have the same number of points.
///   - `material`: Set to 0 by default.
///
/// ## Assumptions & Simplifications
/// - Adjacent rings must have the same number of points to form quads/triangles.
/// - No “closing” of each ring in theta-direction.
/// - Normals are averaged from adjacent faces; tangents/bitangents are naive placeholders.
///
/// todo: Not working: `Index 7638 extends beyond limit 2076. Did you bind the correct index buffer?`
pub fn mesh_from_sas_points(rings: &[Vec<Vec3>]) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // 1) Flatten all rings into a single vertex buffer, keeping track of offsets per ring.
    let mut ring_offsets = Vec::with_capacity(rings.len());
    let mut current_offset = 0;

    for ring in rings {
        ring_offsets.push(current_offset);
        current_offset += ring.len();

        for &pt in ring {
            vertices.push(Vertex {
                position: [pt.x, pt.y, pt.z],
                tex_coords: [0.0, 0.0], // Placeholder, no UVs assigned
                normal: Vec3::new(0.0, 0.0, 0.0), // Will fill in later
                tangent: Vec3::new(0.0, 0.0, 0.0), // Will fill in later
                bitangent: Vec3::new(0.0, 0.0, 0.0), // Will fill in later
            });
        }
    }

    // 2) Build indices by connecting ring i to ring i+1
    for i in 0..(rings.len().saturating_sub(1)) {
        let ring0 = &rings[i];
        let ring1 = &rings[i + 1];

        // Only form triangles if ring0.len() == ring1.len() and not empty
        if ring0.len() == ring1.len() && ring0.len() >= 2 {
            let r0_offset = ring_offsets[i];
            let r1_offset = ring_offsets[i + 1];

            // Triangulate each pair of adjacent points (j, j+1).
            for j in 0..(ring0.len() - 1) {
                let i0 = r0_offset + j;
                let i1 = r1_offset + j;
                let i2 = r1_offset + j + 1;
                let i3 = r0_offset + j + 1;

                // Triangle #1: (i0, i1, i2)
                indices.push(i0);
                indices.push(i1);
                indices.push(i2);

                // Triangle #2: (i0, i2, i3)
                indices.push(i0);
                indices.push(i2);
                indices.push(i3);
            }
        }
    }

    // 3) Compute per-vertex normals by averaging face normals
    let mut normal_accum = vec![Vec3::new(0.0, 0.0, 0.0); vertices.len()];
    let mut count_accum = vec![0u32; vertices.len()];

    // Each triplet in indices is one triangle
    for tri_idx in 0..(indices.len() / 3) {
        let base = tri_idx * 3;
        let i0 = indices[base + 0];
        let i1 = indices[base + 1];
        let i2 = indices[base + 2];

        let v0 = Vec3::new(
            vertices[i0].position[0],
            vertices[i0].position[1],
            vertices[i0].position[2],
        );
        let v1 = Vec3::new(
            vertices[i1].position[0],
            vertices[i1].position[1],
            vertices[i1].position[2],
        );
        let v2 = Vec3::new(
            vertices[i2].position[0],
            vertices[i2].position[1],
            vertices[i2].position[2],
        );

        // Face normal via cross product
        let e1 = v1 - v0;
        let e2 = v2 - v0;
        let face_normal = e1.cross(e2).to_normalized();

        // Accumulate this face normal for each vertex
        normal_accum[i0] += face_normal;
        normal_accum[i1] += face_normal;
        normal_accum[i2] += face_normal;

        count_accum[i0] += 1;
        count_accum[i1] += 1;
        count_accum[i2] += 1;
    }

    // 4) Finalize each vertex's normal and set a simple tangent/bitangent
    for (idx, vertex) in vertices.iter_mut().enumerate() {
        if count_accum[idx] > 0 {
            let avg_normal = (normal_accum[idx] / (count_accum[idx] as f32)).to_normalized();
            vertex.normal = avg_normal;

            // Naive tangent: pick something perpendicular to normal, e.g. (1,0,0) if possible
            // If the normal is near the X axis, pick (0,1,0), etc.
            let arbitrary = if avg_normal.x.abs() < 0.9 {
                Vec3::new(1.0, 0.0, 0.0)
            } else {
                Vec3::new(0.0, 1.0, 0.0)
            };
            let tangent = avg_normal.cross(arbitrary).to_normalized();
            let bitangent = avg_normal.cross(tangent).to_normalized();

            vertex.tangent = tangent;
            vertex.bitangent = bitangent;
        } else {
            // If no faces share this vertex (rare if ring is isolated), just default
            vertex.normal = Vec3::new(0.0, 0.0, 1.0);
            vertex.tangent = Vec3::new(1.0, 0.0, 0.0);
            vertex.bitangent = Vec3::new(0.0, 1.0, 0.0);
        }
    }

    // 5) Return the assembled mesh
    Mesh {
        vertices,
        indices,
        material: 0, // or some other material ID
    }
}

// pub fn get_mesh_points(atoms: &[Atom]) -> Vec<Vec<Vec3>> {
//     // The result is a collection of "rings" of points on the SAS,
//     // each ring stored as a Vec<Vec3>. (One ring per latitude, for example.)
//
//     // todo: Parallelize with rayon; a grid-based approach for nearby items to improve
//     // todo: efficiency, as with your bond-creation code.
//
//     // todo: This can be used to draw dots. Take another pass to turn it into mesh triangles etc.
//
//     // Tweak these sampling resolutions for speed vs. surface fidelity:
//     let steps_theta = 36; // how many slices around the longitude
//     let steps_phi = 18; // how many slices around the latitude
//
//     // todo: Low due to poor perf currently
//     let steps_theta = 12; // how many slices around the longitude
//     let steps_phi = 8; // how many slices around the latitude
//
//     let mut mesh_rings = Vec::new();
//
//     // For each atom, create a "probe-inflated" sphere and sample its surface.
//     for (i, atom_i) in atoms.iter().enumerate() {
//         let center_i: Vec3 = atom_i.posit.into();
//         let inflated_radius_i = atom_i.element.vdw_radius() + PROBE_RADIUS;
//
//         // We'll create rings by sweeping over phi in [0, tau],
//         // then each ring is subdivided by theta in [0, tau).
//         for p_i in 0..steps_phi {
//             // phi goes from 0 (north pole) to π (south pole).
//             let phi = TAU / 2. * p_i as f32 / steps_phi as f32;
//
//             let mut ring_points = Vec::new();
//             for t_i in 0..steps_theta {
//                 // theta goes around the equator 0..tau
//                 let theta = TAU * t_i as f32 / steps_theta as f32;
//
//                 // Convert spherical coords to Cartesian unit vector:
//                 let sin_phi = phi.sin();
//                 let x = sin_phi * theta.cos();
//                 let y = sin_phi * theta.sin();
//                 let z = phi.cos();
//
//                 // Scale by the inflated radius and offset by the atom's center:
//                 let sample_pt = center_i + Vec3::new(x, y, z) * inflated_radius_i;
//
//                 // Check if the sample point is occluded by any other atom's inflated sphere:
//                 let mut occluded = false;
//                 for (j, atom_j) in atoms.iter().enumerate() {
//                     if i == j {
//                         continue;
//                     }
//                     let center_j: Vec3 = atom_j.posit.into();
//                     let inflated_radius_j = atom_j.element.vdw_radius() + PROBE_RADIUS;
//
//                     // If distance < inflated_radius_j, point is inside that sphere => occluded
//                     let dist = (sample_pt - center_j).magnitude();
//                     if dist < inflated_radius_j {
//                         occluded = true;
//                         break;
//                     }
//                 }
//
//                 // If the sample point is not occluded, keep it:
//                 if !occluded {
//                     ring_points.push(sample_pt);
//                 }
//             }
//
//             // Only add the ring if it's not empty
//             if !ring_points.is_empty() {
//                 mesh_rings.push(ring_points);
//             }
//         }
//     }
//
//     mesh_rings
// }

// /// Calculate the 2D mesh of points defining the Solvent Accessible Surface, using the "rolling ball" method.
// pub fn get_mesh_points(atoms: &[Atom]) -> Vec<Vec<Vec3>> {
//     // Use a series of starting points around a circle; take the closest path using the
//     // probe along each of these.
//
//     let mut result = Vec::new();
//
//     // todo: Find an appropriate starting point?
//     let init_general = Vec3::new(100., 0., 0.);
//
//     // Looking along the Z axis, take slices along the X/Y plane.
//     // This choice of coordinates is arbitrary.
//     for angle_z in linspace(0., TAU, NUM_PROBE_ANGLES) {
//         let rotator = Quaternion::from_axis_angle(UP, angle_z);
//         let init = rotator.rotate_vec(init_general);
//
//         // Start
//         let starting_posit = Vec3::new(0., 0., 0.);
//     }
//
//     result
// }

//
// pub fn get_mesh_points(pdb: &PDB) -> Vec<Vec3> {
//     // todo: Chain level A/R.
//     let result = calculate_sasa(pdb, Some(PROBE_RADIUS), Some(NUM_POINTS), SASALevel::Protein);
//
//     match result {
//         Ok(SASAResult::Protein(r)) => {
//             Vec::new()
//         }
//         Err(e) => {
//             eprintln!("Error calculating molecular surface: {e:?}");
//             Vec::new()
//         }
//         _ => unreachable!("Invalid SASA result type.")
//     }
// }
