use std::{
    fs::File,
    io,
    io::{ErrorKind, Read, Write},
    path::Path,
};

use bincode::{Decode, Encode};
use graphics::{Camera, FWD_VEC, UP_VEC};
use lin_alg::{
    f32::{Quaternion, Vec3 as Vec3F32},
    f64::Vec3,
};
use na_seq::AaIdent;

use crate::{
    molecule::{Atom, Molecule, Residue},
    Selection, State, ViewSelLevel,
};

const MOVE_TO_TARGET_DIST: f32 = 15.;

pub fn vec3_to_f32(v: Vec3) -> Vec3F32 {
    Vec3F32::new(v.x as f32, v.y as f32, v.z as f32)
}

/// Used for cursor selection.
pub fn points_along_ray(ray: (Vec3F32, Vec3F32), atoms: &[Atom], dist_thresh: f32) -> Vec<usize> {
    let mut result = Vec::new();

    // todo: Address this fn n ext in your selection fix quest.

    let ray_dir = ray.1.to_normalized();

    for (i, atom) in atoms.iter().enumerate() {
        let atom_pos = vec3_to_f32(atom.posit);

        // Compute the closest point on the ray to the atom position
        let to_atom = atom_pos - ray.0;
        let t = to_atom.dot(ray_dir);
        let closest_point = ray.0 + ray_dir * t;

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
    ress: &[Residue],
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
            ViewSelLevel::Atom => Selection::Atom(near_i),
            ViewSelLevel::Residue => {
                for (i_res, res) in ress.iter().enumerate() {
                    if res.atoms.contains(&near_i) {
                        return Selection::Residue(i_res);
                    }
                }
                Selection::None // Selected atom is not in a residue.
            }
        }
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

/// Move the camera to look at a point of interest. Takes the starting location into account.
/// todo: Smooth interpolated zoom.
pub fn cam_look_at(cam: &mut Camera, target: Vec3) {
    let diff = vec3_to_f32(target) - cam.position;
    let dir = diff.to_normalized();
    let dist = diff.magnitude();

    // Rotate the camera to look at the target.
    let cam_looking_at = cam.orientation.rotate_vec(FWD_VEC);
    let rotator = Quaternion::from_unit_vecs(cam_looking_at, dir);

    cam.orientation = rotator * cam.orientation;

    // Slide along the patah between cam and target until close to it.
    let move_dist = dist - MOVE_TO_TARGET_DIST;
    cam.position += dir * move_dist;
}

pub fn select_from_search(state: &mut State) {
    let query = &state.ui.residue_search.to_lowercase();

    if let Some(mol) = &state.molecule {
        for (i, res) in mol.residues.iter().enumerate() {
            if query.contains(&i.to_string()) {
                state.selection = Selection::Residue(i);
            }
            if let Some(aa) = res.aa {
                if query.contains(&aa.to_str(AaIdent::ThreeLetters).to_lowercase()) {
                    state.selection = Selection::Residue(i);
                }
            }
        }
    }
}

pub fn cycle_res_selected(state: &mut State, reverse: bool) {
    if let Some(mol) = &state.molecule {
        state.ui.view_sel_level = ViewSelLevel::Residue;
        match state.selection {
            Selection::Residue(res_i) => {
                if reverse {
                    if res_i != 0 {
                        state.selection = Selection::Residue(res_i - 1);
                    }
                } else {
                    if res_i != mol.residues.len() - 1 {
                        state.selection = Selection::Residue(res_i + 1);
                    }
                }
            }
            _ => {
                if !mol.residues.is_empty() {
                    state.selection = Selection::Residue(0);
                }
            }
        }
    }
}
