//! A collection of utility functions that don't neatly belong in other modules.
//! For example, we may call some of these from the GUI, but they won't have any EGUI-specific
//! logic in them.

use std::{collections::HashMap, io::Cursor, time::Instant};

use bio_files::{Chain, ResidueType};
use graphics::{Camera, ControlScheme, EngineUpdates, FWD_VEC, Mesh, Scene, Vertex};
use lin_alg::{
    f32::{Quaternion, Vec3 as Vec3F32},
    f64::Vec3,
};
use mcubes::{MarchingCubes, MeshSide};
use na_seq::{AaIdent, Element};

use crate::{
    CamSnapshot, PREFS_SAVE_INTERVAL, Selection, State, StateUi, ViewSelLevel,
    download_mols::load_cif_rcsb,
    mol_drawing::{EntityType, MoleculeView, draw_density, draw_density_surface, draw_molecule},
    molecule::{Atom, AtomRole, Bond, Molecule, Residue},
    render::{
        CAM_INIT_OFFSET, MESH_DENSITY_SURFACE, MESH_SECONDARY_STRUCTURE, MESH_SOLVENT_SURFACE,
        RENDER_DIST_FAR, RENDER_DIST_NEAR, set_flashlight, set_static_light,
    },
    ribbon_mesh::build_cartoon_mesh,
    sa_surface::make_sas_mesh,
    ui::{VIEW_DEPTH_FAR_MAX, VIEW_DEPTH_NEAR_MIN},
};

const MOVE_TO_TARGET_DIST: f32 = 15.;
const MOVE_CAM_TO_LIG_DIST: f32 = 30.;

/// Used for cursor selection.
pub fn points_along_ray(
    ray: (Vec3F32, Vec3F32),
    atoms: &[Atom],
    mut dist_thresh: f32,
) -> Vec<usize> {
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

        // todo: take atom radius into account. E.g. Hydrogens should required a smaller dist.
        // todo: This approach is a bit sloppy, but probably better than not including it.
        if atom.element == Element::Hydrogen {
            // todo: This seems to prevent selecting at all; not sure why.
            // dist_thresh *= 0.9;
        }
        if dist_to_ray < dist_thresh {
            result.push(i);
        }
    }

    result
}

/// From under the cursor; pick the one near the ray, closest to the camera. This function is
/// run after the ray geometry is calculated, and is responsible for determing which atoms, residues, etc
/// are available for selection. It takes into account graphical filters, so only visible items
/// are selected.
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
        let chain_hidden = {
            let chains_this_atom: Vec<&Chain> =
                chains.iter().filter(|c| c.atoms.contains(atom_i)).collect();

            let mut hidden = false;
            for chain in &chains_this_atom {
                if !chain.visible {
                    hidden = true;
                    break;
                }
            }
            hidden
        };

        if chain_hidden {
            continue;
        }

        let atom = &atoms[*atom_i];

        if ui.visibility.hide_sidechains
            || matches!(
                ui.mol_view,
                MoleculeView::SpaceFill | MoleculeView::Backbone
            )
        {
            if let Some(role) = atom.role {
                if role == AtomRole::Sidechain || role == AtomRole::H_Sidechain {
                    continue;
                }
            }
        }

        if let Some(role) = atom.role {
            if ui.visibility.hide_sidechains && role == AtomRole::Sidechain {
                continue;
            }
            if role == AtomRole::Water
                && (ui.visibility.hide_water
                    || matches!(
                        ui.mol_view,
                        MoleculeView::SpaceFill | MoleculeView::Backbone
                    ))
            {
                continue;
            }
        }

        if ui.visibility.hide_hydrogen && atom.element == Element::Hydrogen {
            continue;
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
                let atom_near = &atoms[near_i];
                if let Some(res_i) = atom_near.residue {
                    if res_i == i_res {
                        return Selection::Residue(i_res);
                    }
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

pub fn cam_look_at_outside(cam: &mut Camera, target: Vec3F32, mol_center: Vec3F32) {
    // Note: This is similar to `cam_look_at`, but we don't call that, as we're positioning
    // with an absolute orientation in mind, vice `cam_look_at`'s use of current cam LOS.

    // Look from the outside in, so our view is unobstructed by the protein. Do this after
    // the camera is positioned.
    let look_vec = (target - mol_center).to_normalized();

    cam.position = target + look_vec * MOVE_CAM_TO_LIG_DIST;
    cam.orientation = Quaternion::from_unit_vecs(FWD_VEC, -look_vec);
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

pub fn cycle_res_selected(state: &mut State, scene: &mut Scene, reverse: bool) {
    let Some(mol) = &state.molecule else { return };

    state.ui.view_sel_level = ViewSelLevel::Residue;

    match state.selection {
        Selection::Residue(res_i) => {
            for chain in &mol.chains {
                if chain.residues.contains(&res_i) {
                    // Pick a residue from the chain the current selection is on.
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
                    break;
                }
            }
        }
        _ => {
            if !mol.residues.is_empty() {
                state.selection = Selection::Residue(0);
            }
        }
    }

    if let ControlScheme::Arc { center } = &mut scene.input_settings.control_scheme {
        *center = orbit_center(state);
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

/// Creates pairs of all *nearby* positions. Much faster than comparing every combination, if only nearly
/// ones are relevant.
/// The separate `indexes` parameter allows `posits` to be a subset of the array we're indexing into,
/// e.g. a filtered set of atoms.
pub fn setup_neighbor_pairs(
    posits: &[&Vec3],
    indexes: &[usize],
    grid_size: f64,
) -> Vec<(usize, usize)> {
    // Build a spatial grid for atom indices.
    let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();

    for (i, posit) in posits.iter().enumerate() {
        let grid_pos = (
            (posit.x / grid_size).floor() as i32,
            (posit.y / grid_size).floor() as i32,
            (posit.z / grid_size).floor() as i32,
        );

        grid.entry(grid_pos).or_default().push(indexes[i]);
    }

    // Collect candidate atom pairs based on neighboring grid cells.
    let mut result = Vec::new();
    for (&cell, indices) in &grid {
        // Look at this cell and its neighbors.
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let neighbor_cell = (cell.0 + dx, cell.1 + dy, cell.2 + dz);
                    if let Some(neighbor_indices) = grid.get(&neighbor_cell) {
                        // Attempt to prevent duplicates as we iterate. Note working.
                        for &i in indices {
                            for &j in neighbor_indices {
                                // The ordering prevents duplicates.
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

/// Based on selection status and if a molecule is open, find the center for the orbit camera.
pub fn orbit_center(state: &State) -> Vec3F32 {
    if state.ui.orbit_around_selection {
        match &state.selection {
            Selection::Atom(i) => {
                if let Some(mol) = &state.molecule {
                    match mol.atoms.get(*i) {
                        Some(a) => a.posit.into(),
                        None => Vec3F32::new_zero(),
                    }
                } else {
                    Vec3F32::new_zero()
                }
            }
            Selection::Residue(i) => {
                if let Some(mol) = &state.molecule {
                    match mol.residues.get(*i) {
                        Some(res) => {
                            match mol.atoms.get(match res.atoms.first() {
                                Some(a) => *a,
                                None => return Vec3F32::new_zero(),
                            }) {
                                Some(a) => a.posit.into(),
                                None => Vec3F32::new_zero(),
                            }
                        }
                        None => Vec3F32::new_zero(),
                    }
                } else {
                    Vec3F32::new_zero()
                }
            }
            Selection::Atoms(is) => {
                if let Some(mol) = &state.molecule {
                    match mol.atoms.get(is[0]) {
                        Some(a) => a.posit.into(),
                        None => Vec3F32::new_zero(),
                    }
                } else {
                    Vec3F32::new_zero()
                }
            }
            Selection::None => {
                if let Some(mol) = &state.molecule {
                    mol.center.into()
                } else {
                    lin_alg::f32::Vec3::new_zero()
                }
            }
        }
    } else {
        if let Some(mol) = &state.molecule {
            mol.center.into()
        } else {
            Vec3F32::new_zero()
        }
    }
}

/// A helper fn. Maps from a global index, to a local atom from a subset.
pub fn find_atom<'a>(atoms: &'a [Atom], indices: &[usize], i_to_find: usize) -> Option<&'a Atom> {
    for (i_set, atom) in atoms.iter().enumerate() {
        if indices[i_set] == i_to_find {
            return Some(atom);
        }
    }

    None
}

pub fn load_atom_coords_rcsb(
    ident: &str,
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    redraw: &mut bool,
    reset_cam: &mut bool,
) {
    match load_cif_rcsb(ident) {
        // tood: For organization purposes, move thi scode out of the UI.
        Ok((pdb, cif_data)) => {
            let cursor = Cursor::new(&cif_data);

            match Molecule::from_cif_pdb(&pdb, cursor) {
                Ok(mol) => {
                    state.volatile.aa_seq_text = String::with_capacity(mol.atoms.len());
                    for aa in &mol.aa_seq {
                        state
                            .volatile
                            .aa_seq_text
                            .push_str(&aa.to_str(AaIdent::OneLetter));
                    }

                    // todo: DRY from `open_molecule`. Refactor into shared code?

                    state.volatile.aa_seq_text = String::with_capacity(mol.atoms.len());
                    for aa in &mol.aa_seq {
                        state
                            .volatile
                            .aa_seq_text
                            .push_str(&aa.to_str(AaIdent::OneLetter));
                    }

                    state.volatile.flags.ss_mesh_created = false;
                    state.volatile.flags.sas_mesh_created = false;
                    state.volatile.flags.clear_density_drawing = true;
                    state.molecule = Some(mol)
                }
                Err(e) => eprintln!("Problem loading molecule from CIF: {e:?}"),
            }

            state.pdb = Some(pdb);
            state.cif_pdb_raw = Some(cif_data);
            state.update_from_prefs();

            *redraw = true;
            *reset_cam = true;
            set_flashlight(scene);
            engine_updates.lighting = true;

            // todo: async
            // Only after updating from prefs (to prevent unecesasary loading) do we update data avail.
            state
                .molecule
                .as_mut()
                .unwrap()
                .updates_rcsb_data(&mut state.volatile.mol_pending_data_avail);
        }
        Err(_e) => {
            eprintln!("Error loading CIF file");
        }
    }
}

pub fn save_snap(state: &mut State, cam: &Camera, name: &str) {
    state
        .cam_snapshots
        .push(CamSnapshot::from_cam(cam, name.to_owned()));
    state.ui.cam_snapshot_name = String::new();

    state.ui.cam_snapshot = Some(state.cam_snapshots.len() - 1);

    state.update_save_prefs();
}

// The snap must be set in state.ui.cam_snapshot ahead of calling this.
pub fn load_snap(state: &mut State, scene: &mut Scene, engine_updates: &mut EngineUpdates) {
    if let Some(snap_i) = state.ui.cam_snapshot {
        match state.cam_snapshots.get(snap_i) {
            Some(snap) => {
                scene.camera.position = snap.position;
                scene.camera.orientation = snap.orientation;
                scene.camera.far = snap.far;

                scene.camera.update_proj_mat(); // In case `far` etc changed.
                engine_updates.camera = true;

                set_flashlight(scene);
                engine_updates.lighting = true;
            }
            None => {
                eprintln!("Error: Could not find snapshot {}", snap_i);
            }
        }
    }
}

/// Resets the camera to the *front* view, and related settings.
pub fn reset_camera(
    scene: &mut Scene,
    view_depth: &mut (u16, u16),
    engine_updates: &mut EngineUpdates,
    mol: &Molecule,
) {
    let center: lin_alg::f32::Vec3 = mol.center.into();
    scene.camera.position =
        lin_alg::f32::Vec3::new(center.x, center.y, center.z - (mol.size + CAM_INIT_OFFSET));
    scene.camera.orientation = Quaternion::new_identity();

    scene.camera.near = RENDER_DIST_NEAR;
    scene.camera.far = RENDER_DIST_FAR;
    scene.camera.update_proj_mat();

    set_static_light(scene, center, mol.size);
    set_flashlight(scene);

    engine_updates.camera = true;
    engine_updates.lighting = true;

    *view_depth = (VIEW_DEPTH_NEAR_MIN, VIEW_DEPTH_FAR_MAX);
}

/// Utility function that prints to stderr, and the CLI output. Sets the out flag.
pub fn handle_err(ui: &mut StateUi, msg: String) {
    eprintln!("{msg}");
    ui.cmd_line_output = msg;
    ui.cmd_line_out_is_err = true;
}

pub fn close_mol(state: &mut State, scene: &mut Scene, engine_updates: &mut EngineUpdates) {
    state.molecule = None;
    scene.entities.retain(|ent| {
        ent.class != EntityType::Protein as u32
            && ent.class != EntityType::Density as u32
            && ent.class != EntityType::DensitySurface as u32
            && ent.class != EntityType::SecondaryStructure as u32
            && ent.class != EntityType::SaSurface as u32
    });
    engine_updates.entities = true;

    state.to_save.last_opened = None;
    state.to_save.last_map_opened = None;
    state.volatile.aa_seq_text = String::new();

    state.update_save_prefs();
}

pub fn close_lig(state: &mut State, scene: &mut Scene, engine_updates: &mut EngineUpdates) {
    state.ligand = None;
    scene
        .entities
        .retain(|ent| ent.class != EntityType::Ligand as u32);

    engine_updates.entities = true;

    state.to_save.last_ligand_opened = None;
    state.update_save_prefs();
}

/// Populdate the electron-density mesh (isosurface). This assumes the density_rect is already set up.
pub fn make_density_mesh(state: &mut State, scene: &mut Scene, engine_updates: &mut EngineUpdates) {
    if let Some(mol) = &state.molecule {
        // todo: Adapt this to your new approach, if it works.
        if let Some(rect) = &mol.density_rect {
            let dims = (rect.dims[0], rect.dims[1], rect.dims[2]); // (nx,ny,nz)

            let size = (
                (rect.step[0] * rect.dims[0] as f64) as f32, // Δx * nx  (Å)
                (rect.step[1] * rect.dims[1] as f64) as f32,
                (rect.step[2] * rect.dims[2] as f64) as f32,
            );

            // “sampling interval” in the original code is really the number of
            // samples along each axis (= nx,ny,nz), so just cast dims to f32:
            let samples = (
                rect.dims[0] as f32,
                rect.dims[1] as f32,
                rect.dims[2] as f32,
            );

            match MarchingCubes::from_gridpoints(
                dims,
                size,
                samples,
                rect.origin_cart.into(),
                mol.elec_density.as_ref().unwrap(),
                state.ui.density_iso_level,
            ) {
                Ok(mc) => {
                    let mesh = mc.generate(MeshSide::Both);

                    // Convert from `mcubes::Mesh` to `graphics::Mesh`.
                    let vertices = mesh
                        .vertices
                        .iter()
                        .map(|v| Vertex::new(v.posit.to_arr(), v.normal))
                        .collect();

                    scene.meshes[MESH_DENSITY_SURFACE] = Mesh {
                        vertices,
                        indices: mesh.indices,
                        material: 0,
                    };

                    if !state.ui.visibility.hide_density_surface {
                        draw_density_surface(&mut scene.entities);
                    }

                    engine_updates.meshes = true;
                    engine_updates.entities = true;
                }
                Err(e) => handle_err(&mut state.ui, e.to_string()),
            }
        }
    }
}

/// Code here is ctivated by flags. It's organized here, where we have access to the Scene.
/// These flags are set in places that don't have access to the scene.
pub fn handle_scene_flags(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
) {
    if state.volatile.flags.new_mol_loaded {
        state.volatile.flags.new_mol_loaded = false;

        if let Some(mol) = &state.molecule {
            reset_camera(scene, &mut state.ui.view_depth, engine_updates, mol);
        }

        set_flashlight(scene);
        engine_updates.lighting = true;
    }

    if state.volatile.flags.new_density_loaded {
        state.volatile.flags.new_density_loaded = false;

        if let Some(mol) = &state.molecule {
            if !state.ui.visibility.hide_density {
                if let Some(density) = &mol.elec_density {
                    draw_density(&mut scene.entities, density);
                    engine_updates.entities = true;
                }
            }
        }
    }

    if state.volatile.flags.clear_density_drawing {
        state.volatile.flags.clear_density_drawing = false;

        scene.entities.retain(|ent| {
            ent.class != EntityType::Density as u32
                && ent.class != EntityType::DensitySurface as u32
        });
    }

    // todo: temp experiencing a crash from wgpu on vertex buffer
    if state.volatile.flags.make_density_mesh {
        state.volatile.flags.make_density_mesh = false;
        make_density_mesh(state, scene, engine_updates);
    }

    if state.volatile.flags.update_ss_mesh {
        state.volatile.flags.update_ss_mesh = false;
        state.volatile.flags.ss_mesh_created = true;

        if let Some(mol) = &state.molecule {
            scene.meshes[MESH_SECONDARY_STRUCTURE] =
                build_cartoon_mesh(&mol.secondary_structure, &mol.atoms);

            engine_updates.meshes = true;
        }
    }

    if state.volatile.flags.update_sas_mesh {
        state.volatile.flags.update_sas_mesh = false;
        state.volatile.flags.sas_mesh_created = true;

        if let Some(mol) = &state.molecule {
            let atoms: Vec<&_> = mol.atoms.iter().filter(|a| !a.hetero).collect();
            scene.meshes[MESH_SOLVENT_SURFACE] =
                make_sas_mesh(&atoms, state.to_save.sa_surface_precision);

            // We draw the molecule here
            if matches!(
                state.ui.mol_view,
                MoleculeView::Dots | MoleculeView::Surface
            ) {
                // The dots are drawn from the mesh vertices
                draw_molecule(state, scene);
                engine_updates.entities = true;
            }

            engine_updates.meshes = true;
        }
    }

    if state.volatile.mol_pending_data_avail.is_some() {
        if let Some(mol) = &mut state.molecule {
            if mol.poll_data_avail(&mut state.volatile.mol_pending_data_avail) {
                state.update_save_prefs();
            }
        }
    }
}
