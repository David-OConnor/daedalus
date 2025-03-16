use std::collections::HashMap;
use graphics::{Mesh, Vertex};
use lin_alg::f64::Vec3;
use lin_alg::f32::Vec3 as Vec3F32;

use qhull::Qh;

use crate::{docking::DockingSite, molecule::Molecule};

// todo: This currently relies on qhull FFI bindings. Try to get rid of this if you have trouble compiling.


// Some algorithms for non-convex surfaces:
// - Poisson Surface Reconstruction (smooth surfaces)
// - Ball-pivoting algorithm
// - Delaunay triangulation and alpha shapes.


// todo: Native rust implementation eventually, probably. And, concave! Ball-pivotting or poisson.
// todo: Apply whatever you come up with to the ASA mesh as well.
/// Generate triangulated indices, using the QuickHull algorithm.
/// Ideally, we will use an algorithm capable of non-concave shapes; thsi is a placeholder.
/// F64  here; qhull operates on those.
fn quickhull(points: &[Vec3], molecule_center: Vec3) -> Vec<usize> {
    let qh = Qh::builder()
        .compute(true)
        .build_from_iter(points.iter().map(|p| p.to_arr())).unwrap();

    let mut result = Vec::new();

    // Get all simplices (faces) from Qhull.
    for (i, simplex) in qh.simplices().enumerate() {
        let indices = simplex
            .vertices()
            .unwrap()
            .iter()
            .map(|v| v.index(&qh).unwrap())
            .collect::<Vec<_>>();


        // We'll assume these are always triangles (3 vertices),
        // though Qhull can produce more complex facets in some modes.
        // You might want a check or handle polygons gracefully.
        if indices.len() == 3 {
            let v0 = indices[0];
            let v1 = indices[1];
            let v2 = indices[2];

            // Actual positions
            let p0 = points[v0];
            let p1 = points[v1];
            let p2 = points[v2];

            let normal = Vec3::from_slice(&simplex.normal().unwrap()).unwrap();

            // Face centroid
            let centroid = (p0 + p1 + p2) / 3.0;

            // Check orientation: if dot < 0, normal is pointing "inward" relative
            // to the center (depending on your coordinate assumptions).
            // This condition can be flipped based on your geometry conventions.
            let outward_dir = centroid - molecule_center;
            let mut dot_val = normal.dot(outward_dir);

            // If you only want faces whose normals face "out" from the molecule_center,
            // then skip (don’t add) any face with negative dot.
            if dot_val > 0.0 {
                // This face is "away" from center, so omit
                continue;
            }

            // At this point, either cull_far_side is false, or the face is front-facing.
            // Add the triangle with whichever ordering you need:
            result.extend_from_slice(&[v0, v1, v2]);

            // Also add the reversed ordering for a two-sided mesh:
            // (This addresses orientation flips and ensures the face is rendered both ways.)
            result.extend_from_slice(&[v0, v2, v1]);
        } else {
            println!("Vertex len != 3 on hull: {:?}", indices.len());
            // If Qhull returns a facet with more than 3 vertices, you could:
            //   1) fan triangulate in some way,
            //   2) or handle it however best fits your use-case.
            // For brevity, let's just handle them directly here:
            // (No culling logic in this simple pass, unless you do your own triangulation.)
            result.extend(&indices);
            // And possibly the reversed ordering:
            let mut rev = indices.clone();
            rev.reverse();
            result.extend(&rev);
        }
    }

    result
}

/// Return a mesh, and also edges. We may use the edges directly to compute if which side something is on,
/// and use the `Mesh` for display? The mesh vertices should, for now, correspond to receptor atom positions.
/// later, we may offset them some distance away from the receptor molecule.
/// Return a mesh (convex hull) of atoms near the docking site, and a list of edges
pub fn find_docking_site_surface(
    receptor: &Molecule,
    site: &DockingSite,
) -> (Mesh, Vec<(usize, usize)>) {
    // 1) Identify atoms near the site by bounding box
    const SITE_PAD: f64 = 1.6;

    let half_box = site.site_box_size * 0.5 * SITE_PAD;
    let site_min = site.site_center - Vec3::new(half_box, half_box, half_box);
    let site_max = site.site_center + Vec3::new(half_box, half_box, half_box);

    let mut relevant_atom_indices = Vec::new();
    for (i, atom) in receptor.atoms.iter().enumerate() {
        if atom.hetero {
            continue
        }

        let p = atom.posit;
        if p.x >= site_min.x && p.x <= site_max.x
            && p.y >= site_min.y && p.y <= site_max.y
            && p.z >= site_min.z && p.z <= site_max.z
        {
            relevant_atom_indices.push(i);
        }
    }

    // 2) Convert relevant atoms to a list of positions (f32) for hull calculation
    //    Also build a mapping: local_idx -> global atom index
    // let mut vertices = Vec::new();            // final mesh vertices
    let mut pos_array = Vec::new();           // positions for hull algorithm
    let mut local_to_global = Vec::new();     // map local -> global

    for &atom_idx in &relevant_atom_indices {
        let a = &receptor.atoms[atom_idx];
        let p = a.posit;
        // pos_array.push(Vec3F32::new(p.x as f32, p.y as f32, p.z as f32));
        pos_array.push(p);
        local_to_global.push(atom_idx);
    }

    // If we have fewer than 3 points, we can't form any faces
    if pos_array.len() < 3 {
        let mesh = Mesh {
            vertices: pos_array
                .iter()
                .map(|&p| Vertex {
                    position: [p.x as f32, p.y as f32, p.z as f32],
                    tex_coords: [0.0, 0.0],
                    normal: Vec3F32::new(0.0, 0.0, 1.0),
                    tangent: Vec3F32::new(1.0, 0.0, 0.0),
                    bitangent: Vec3F32::new(0.0, 1.0, 0.0),
                })
                .collect(),
            indices: Vec::new(),
            material: 0,
        };
        // build edges from adjacency
        let edges = Vec::new();
        return (mesh, edges);
    }

    // 3) Compute the convex hull (returns a list of triangular faces in terms of the points).
    let hull_indices = quickhull(&pos_array, receptor.center);

    // 4) Build the final Mesh:
    //    - The hull algorithm re-uses the same point array, so we can keep them as “mesh vertices”.
    //    - We set dummy normal/tangent/bitangent; you can compute actual face normals, etc.
    let mesh_vertices: Vec<Vertex> = pos_array
        .iter()
        .map(|&p| Vertex {
            position: [p.x as f32, p.y as f32, p.z as f32],
            tex_coords: [0.0, 0.0],
            normal: Vec3F32::new(0.0, 0.0, 1.0),
            tangent: Vec3F32::new(1.0, 0.0, 0.0),
            bitangent: Vec3F32::new(0.0, 1.0, 0.0),
        })
        .collect();

    let mesh = Mesh {
        vertices: mesh_vertices,
        indices: hull_indices.clone(),
        material: 0,
    };

    // 5) For convenience, we can still return an "edge list" by iterating triangles.
    //    Each face is (i, j, k). Add edges (i,j), (j,k), (k,i).
    //    We'll store them as (min, max) to avoid duplicates.
    let mut edge_set = HashMap::new();
    for face in hull_indices.chunks(3) {
        if face.len() < 3 { continue; }
        let (i, j, k) = (face[0], face[1], face[2]);
        for &(a, b) in &[(i, j), (j, k), (k, i)] {
            let (mn, mx) = if a < b { (a, b) } else { (b, a) };
            edge_set.insert((mn, mx), true);
        }
    }
    let edges = edge_set.into_iter().map(|(k, _)| k).collect::<Vec<_>>();

    (mesh, edges)
}
