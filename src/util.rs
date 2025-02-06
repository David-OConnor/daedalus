use lin_alg::{f32::Vec3 as Vec3F32, f64::Vec3};

use crate::{molecule::Atom, Selection, ViewSelLevel};

pub fn vec3_to_f32(v: Vec3) -> Vec3F32 {
    Vec3F32::new(v.x as f32, v.y as f32, v.z as f32)
}

/// Used for cursor selection.
// pub fn points_along_ray(ray: (Vec3F32, Vec3F32), atoms: &[Atom], dist_thresh: f32) -> Vec<&Atom> {
pub fn points_along_ray(ray: (Vec3F32, Vec3F32), atoms: &[Atom], dist_thresh: f32) -> Vec<usize> {
    let mut result = Vec::new();

    let (ray_origin, ray_dir) = ray;
    let ray_dir = ray_dir.to_normalized(); // Ensure the ray direction is a unit vector

    for (i, atom) in atoms.iter().enumerate() {
        let atom_pos = vec3_to_f32(atom.posit);

        // Compute the closest point on the ray to the atom position
        let to_atom = atom_pos - ray_origin;
        let t = to_atom.dot(ray_dir);
        let closest_point = ray_origin + ray_dir * t;

        // Compute the perpendicular distance to the ray
        let dist_to_ray = (atom_pos - closest_point).magnitude();

        if dist_to_ray < dist_thresh {
            // result.push(atom);
            result.push(i);
        }
    }

    result
}

/// From under the cursor; pick the one near the ray, closest to the camera.
pub fn find_selected_atom(
    sel: &[usize],
    atoms: &[Atom],
    ray: &(Vec3F32, Vec3F32),
    sel_level: ViewSelLevel,
) -> Selection {
    if !sel.is_empty() {
        // todo: Also consider togglign between ones under the cursor near the front,
        // todo and picking the one closest to the ray.

        let mut near_i = 0;
        let mut near_dist = 99999.;

        for atom_i in sel {
            let atom = &atoms[*atom_i];
            let dist = (vec3_to_f32(atom.posit) - ray.0).magnitude();
            if dist < near_dist {
                near_i = *atom_i;
                near_dist = dist;
            }
        }

        match sel_level {
            ViewSelLevel::Atom => {}
            ViewSelLevel::Residue => {}
        }

        Selection::Atom(near_i)
    } else {
        Selection::None
    }
}

pub fn mol_center_size(atoms: &[Atom]) -> (Vec3F32, f32) {
    let mut sum = Vec3::new_zero();
    let mut max_dim = 0.;

    for atom in atoms {
        sum += atom.posit;

        // Cheaper than calculating magnitude.
        if atom.posit.x.abs() > max_dim {
            max_dim = atom.posit.x.abs();
        }
        if atom.posit.y.abs() > max_dim {
            max_dim = atom.posit.y.abs();
        }
        if atom.posit.z.abs() > max_dim {
            max_dim = atom.posit.z.abs();
        }
    }

    (vec3_to_f32(sum) / atoms.len() as f32, max_dim as f32)
}
