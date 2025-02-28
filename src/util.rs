use std::{collections::HashMap, f64::consts::TAU, path::PathBuf, time::Instant};

use graphics::{Camera, FWD_VEC};
use lin_alg::{
    f32::{Quaternion, Vec3 as Vec3F32},
    f64::Vec3,
};
use na_seq::AaIdent;

use crate::{
    PREFS_SAVE_INTERVAL, Selection, State, StateUi, ViewSelLevel,
    molecule::{Atom, AtomRole, Bond, Chain, Residue, ResidueType},
};

const MOVE_TO_TARGET_DIST: f32 = 15.;

/// Used for cursor selection.
pub fn points_along_ray(ray: (Vec3F32, Vec3F32), atoms: &[Atom], dist_thresh: f32) -> Vec<usize> {
    let mut result = Vec::new();

    let ray_dir = (ray.1 - ray.0).to_normalized();

    for (i, atom) in atoms.iter().enumerate() {
        let atom_pos: Vec3F32 = atom.posit.into();

        // Compute the closest point on the ray to the atom position
        let to_atom: Vec3F32 = atom_pos - ray.0;
        let t = to_atom.dot(ray_dir);
        let closest_point = ray.0 + ray_dir * t;

        // Compute the perpendicular distance to the ray
        let dist_to_ray = (atom_pos - closest_point).magnitude();

        if dist_to_ray < dist_thresh {
            result.push(i);
        }
    }

    result
}

/// From under the cursor; pick the one near the ray, closest to the camera.
pub fn find_selected_atom(
    atoms_along_ray: &[usize],
    atoms: &[Atom],
    ress: &[Residue],
    ray: &(Vec3F32, Vec3F32),
    ui: &StateUi,
    chains: &[Chain],
) -> Selection {
    if atoms_along_ray.is_empty() {
        return Selection::None;
    }

    // todo: Also consider togglign between ones under the cursor near the front,
    // todo and picking the one closest to the ray.

    let mut near_i = 0;
    let mut near_dist = 99_999.;

    for atom_i in atoms_along_ray {
        let chains_this_atom: Vec<&Chain> =
            chains.iter().filter(|c| c.atoms.contains(atom_i)).collect();
        let mut chain_hidden = false;
        for chain in &chains_this_atom {
            if !chain.visible {
                chain_hidden = true;
                break;
            }
        }
        if chain_hidden {
            continue;
        }

        let atom = &atoms[*atom_i];

        if ui.visibility.hide_sidechains {
            if let Some(role) = atom.role {
                if role == AtomRole::Sidechain {
                    continue;
                }
            }
        }

        if ui.visibility.hide_hetero && atom.hetero {
            continue;
        }

        if ui.visibility.hide_non_hetero && !atom.hetero {
            continue;
        }

        let posit: Vec3F32 = atom.posit.into();
        let dist = (posit - ray.0).magnitude();
        if dist < near_dist {
            near_i = *atom_i;
            near_dist = dist;
        }
    }

    // This is equivalent to our empty check above, but catches the case of the atom count being
    // empty due to hidden chains.
    if near_dist == 99_999. {
        return Selection::None;
    }

    match ui.view_sel_level {
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
}

pub fn mol_center_size(atoms: &[Atom]) -> (Vec3, f32) {
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

    (sum / (atoms.len() as f64), max_dim as f32)
}

/// Move the camera to look at a point of interest. Takes the starting location into account.
/// todo: Smooth interpolated zoom.
pub fn cam_look_at(cam: &mut Camera, target: Vec3) {
    let tgt: Vec3F32 = target.into();
    let diff = tgt - cam.position;
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
            if query.contains(&res.serial_number.to_string()) {
                state.selection = Selection::Residue(i);
            }
            match &res.res_type {
                ResidueType::AminoAcid(aa) => {
                    if query.contains(&aa.to_str(AaIdent::ThreeLetters).to_lowercase()) {
                        state.selection = Selection::Residue(i);
                    }
                }
                ResidueType::Water => {} // todo: Select all water with a new selection type
                ResidueType::Other(name) => {
                    if query.contains(&name.to_lowercase()) {
                        state.selection = Selection::Residue(i);
                    }
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
                // todo: Consider using the chain the current selection is on instead, if applicable.
                // todo: E.g. if a residue is selected, but "Select Residues From" is None.
                if let Some(ch_i) = state.ui.chain_to_pick_res {
                    // Only cycle to a residue in the selected chain.
                    let chain = &mol.chains[ch_i];
                    let mut new_res_i = res_i as isize;

                    let dir = if reverse { -1 } else { 1 };

                    while new_res_i < (mol.residues.len() as isize) - 1 && new_res_i >= 0 {
                        new_res_i += dir;
                        let nri = new_res_i as usize;
                        if chain.residues.contains(&nri) {
                            state.selection = Selection::Residue(nri);
                            break;
                        }
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

pub fn check_prefs_save(state: &mut State) {
    static mut LAST_PREF_SAVE: Option<Instant> = None;
    let now = Instant::now();

    unsafe {
        if let Some(last_save) = LAST_PREF_SAVE {
            if (now - last_save).as_secs() > PREFS_SAVE_INTERVAL {
                LAST_PREF_SAVE = Some(now);
                state.update_save_prefs()
            }
        } else {
            // Initialize LAST_PREF_SAVE the first time it's accessed
            LAST_PREF_SAVE = Some(now);
        }
    }
}

// todo: Update this A/R.

// todo: Also calculate the dihedral angle using 3 bonds. (4 atoms).
/// Bond_0 and bond_1 must share an atom. For now, we assume `bond_0`'s atom_1 is the same as
/// `bond_1`'s atom_0.
pub fn bond_angle(atoms: &[Atom], bond_0: &Bond, bond_1: &Bond) -> f64 {
    if bond_0.atom_1 != bond_1.atom_0 {
        eprintln!("Error: bonds do not share an atom.");
        return 0.;
    }

    let posit_0 = atoms[bond_0.atom_0].posit;
    let posit_1 = atoms[bond_0.atom_1].posit; // Same as bond_1.atom_0.
    let posit_2 = atoms[bond_1.atom_1].posit;

    // Vectors from the central atom
    let v1 = posit_0 - posit_1;
    let v2 = posit_2 - posit_1;

    let dot = v1.dot(v2);
    let mag = v1.magnitude() * v2.magnitude();

    if mag.abs() < 1e-9 {
        // Avoid division by zero in degenerate cases
        0.0
    } else {
        // Clamp to [-1.0, 1.0] to avoid numerical issues before acos
        let mut cos_angle = dot / mag;
        cos_angle = cos_angle.clamp(-1.0, 1.0);
        cos_angle.acos().to_degrees()
    }
}

// pub fn setup_neighbor_pairs(posits: &[&Vec3], grid_size: f64) -> HashMap<(i32, i32, i32), Vec<usize>>{
pub fn setup_neighbor_pairs(posits: &[&Vec3], grid_size: f64) -> Vec<(usize, usize)> {
    // Build a spatial grid for atom indices.
    let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();

    for (i, posit) in posits.iter().enumerate() {
        let grid_pos = (
            (posit.x / grid_size).floor() as i32,
            (posit.y / grid_size).floor() as i32,
            (posit.z / grid_size).floor() as i32,
        );
        grid.entry(grid_pos).or_default().push(i);
    }

    // grid
    // Collect candidate atom pairs based on neighboring grid cells.
    let mut result = Vec::new();
    for (&cell, atom_indices) in &grid {
        // Look at this cell and its neighbors.
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let neighbor_cell = (cell.0 + dx, cell.1 + dy, cell.2 + dz);
                    if let Some(neighbor_indices) = grid.get(&neighbor_cell) {
                        for &i in atom_indices {
                            for &j in neighbor_indices {
                                // if i ==j {
                                //     continue
                                // }

                                // This seems to prevent duplicates of opposite order.
                                // todo: QC
                                if i < j {
                                    result.push((i, j));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    result
}
