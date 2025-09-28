//! A collection of utility functions that don't neatly belong in other modules.
//! For example, we may call some of these from the GUI, but they won't have any EGUI-specific
//! logic in them.

use std::time::Instant;

use bio_files::ResidueType;
use egui::Color32;
use graphics::{Camera, ControlScheme, EngineUpdates, FWD_VEC, Mesh, Scene, Vertex};
use lin_alg::{
    f32::{Quaternion, Vec3 as Vec3F32},
    f64::Vec3,
};
use mcubes::{MarchingCubes, MeshSide};
use na_seq::AaIdent;

use crate::{
    CamSnapshot, ManipMode, PREFS_SAVE_INTERVAL, Selection, State, StateUi, ViewSelLevel,
    download_mols::load_cif_rcsb,
    drawing::{
        EntityType, MoleculeView, draw_all_ligs, draw_all_lipids, draw_all_nucleic_acids,
        draw_density_point_cloud, draw_density_surface, draw_peptide,
    },
    lipid::MoleculeLipid,
    mol_lig::MoleculeSmall,
    molecule::{
        Atom, Bond, MolType, MoleculeCommon, MoleculeGenericRefMut, MoleculePeptide, Residue,
    },
    nucleic_acid::MoleculeNucleicAcid,
    prefs::OpenType,
    render::{
        CAM_INIT_OFFSET, Color, MESH_DENSITY_SURFACE, MESH_SECONDARY_STRUCTURE,
        MESH_SOLVENT_SURFACE, set_flashlight, set_static_light,
    },
    ribbon_mesh::build_cartoon_mesh,
    sa_surface::make_sas_mesh,
    ui::cam::{
        FOG_DIST_DEFAULT, RENDER_DIST_FAR, RENDER_DIST_NEAR, VIEW_DEPTH_NEAR_MIN, calc_fog_dists,
    },
};

const MOVE_TO_TARGET_DIST: f32 = 15.;
const MOVE_CAM_TO_LIG_DIST: f32 = 30.;

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
    let query = &state.ui.atom_res_search.to_lowercase();

    let Some(mol) = &state.peptide else {
        return;
    };

    match state.ui.view_sel_level {
        ViewSelLevel::Atom => {
            for (i, atom) in mol.common.atoms.iter().enumerate() {
                // if query.contains(&atom.serial_number.to_string()) {
                if query == &atom.serial_number.to_string() {
                    state.ui.selection = Selection::AtomPeptide(i);
                    return;
                }
            }
        }
        ViewSelLevel::Residue => {
            for (i, res) in mol.residues.iter().enumerate() {
                if query.contains(&res.serial_number.to_string()) {
                    state.ui.selection = Selection::Residue(i);
                    return;
                }
                match &res.res_type {
                    ResidueType::AminoAcid(aa) => {
                        if query.contains(&aa.to_str(AaIdent::ThreeLetters).to_lowercase()) {
                            state.ui.selection = Selection::Residue(i);
                            return;
                        }
                    }
                    ResidueType::Water => {}
                    ResidueType::Other(name) => {
                        if query.contains(&name.to_lowercase()) {
                            state.ui.selection = Selection::Residue(i);
                            return;
                        }
                    }
                }
            }
        }
    }
}

pub fn cycle_selected(state: &mut State, scene: &mut Scene, reverse: bool) {
    let dir = if reverse { -1 } else { 1 };

    // todo: DRY between atom and res.
    match state.ui.view_sel_level {
        ViewSelLevel::Atom => match state.ui.selection {
            Selection::AtomPeptide(atom_i) => {
                let Some(mol) = &state.peptide else { return };

                for chain in &mol.chains {
                    if chain.atoms.contains(&atom_i) {
                        let mut new_atom_i = atom_i as isize;

                        while new_atom_i < (mol.common.atoms.len() as isize) - 1 && new_atom_i >= 0
                        {
                            new_atom_i += dir;
                            let nri = new_atom_i as usize;
                            if chain.atoms.contains(&nri) {
                                state.ui.selection = Selection::AtomPeptide(nri);
                                break;
                            }
                        }
                        break;
                    }
                }
            }
            Selection::AtomLig((mol_i, atom_i)) => {
                let Some(mol) = state.active_mol() else {
                    return;
                };

                let new_atom_i = atom_i as isize + dir;
                if new_atom_i < mol.common().atoms.len() as isize  && new_atom_i >= 0 {
                    state.ui.selection = Selection::AtomLig((mol_i, new_atom_i as usize));
                }
            }
            // todo: DRY!
            Selection::AtomNucleicAcid((mol_i, atom_i)) => {
                let Some(mol) = state.active_mol() else {
                    return;
                };

                // todo: DRY with the above for peptide atoms.
                let new_atom_i = atom_i as isize + dir;
                if new_atom_i < mol.common().atoms.len() as isize  && new_atom_i >= 0 {
                    state.ui.selection = Selection::AtomNucleicAcid((mol_i, new_atom_i as usize));
                }
            }
            // todo DRY
            Selection::AtomLipid((mol_i, atom_i)) => {
                let Some(mol) = state.active_mol() else {
                    return;
                };

                // todo: DRY with the above for peptide atoms.
                let new_atom_i = atom_i as isize + dir;
                if new_atom_i < mol.common().atoms.len() as isize  && new_atom_i >= 0 {
                    state.ui.selection = Selection::AtomLipid((mol_i, new_atom_i as usize));
                }
            }
            _ => {
                // if !mol.common.atoms.is_empty() {
                //     state.ui.selection = Selection::AtomPeptide(0);
                // }
            }
        },
        ViewSelLevel::Residue => {
            let Some(mol) = &state.peptide else { return };

            match state.ui.selection {
                Selection::Residue(res_i) => {
                    for chain in &mol.chains {
                        if chain.residues.contains(&res_i) {
                            // Pick a residue from the chain the current selection is on.
                            let mut new_res_i = res_i as isize;

                            while new_res_i < (mol.residues.len() as isize) - 1 && new_res_i >= 0 {
                                new_res_i += dir;
                                let nri = new_res_i as usize;
                                if chain.residues.contains(&nri) {
                                    state.ui.selection = Selection::Residue(nri);
                                    break;
                                }
                            }
                            break;
                        }
                    }
                }
                _ => {
                    if !mol.residues.is_empty() {
                        state.ui.selection = Selection::Residue(0);
                    }
                }
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
                state.update_save_prefs(false)
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

/// Based on selection status and if a molecule is open, find the center for the orbit camera.
pub fn orbit_center(state: &State) -> Vec3F32 {
    if state.ui.orbit_around_selection {
        match &state.ui.selection {
            Selection::AtomPeptide(i) => {
                if let Some(mol) = &state.peptide {
                    match mol.common.atoms.get(*i) {
                        Some(a) => a.posit.into(),
                        None => Vec3F32::new_zero(),
                    }
                } else {
                    Vec3F32::new_zero()
                }
            }
            Selection::AtomLig((i_lig, i_atom)) => {
                state.ligands[*i_lig].common.atom_posits[*i_atom].into()
            }
            Selection::AtomNucleicAcid((i_lig, i_atom)) => {
                state.nucleic_acids[*i_lig].common.atom_posits[*i_atom].into()
            }
            Selection::AtomLipid((i_lig, i_atom)) => {
                state.lipids[*i_lig].common.atom_posits[*i_atom].into()
            }

            Selection::Residue(i) => {
                if let Some(mol) = &state.peptide {
                    match mol.residues.get(*i) {
                        Some(res) => {
                            match mol.common.atoms.get(match res.atoms.first() {
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
                if let Some(mol) = &state.peptide {
                    match mol.common.atoms.get(is[0]) {
                        Some(a) => a.posit.into(),
                        None => Vec3F32::new_zero(),
                    }
                } else {
                    Vec3F32::new_zero()
                }
            }
            Selection::None => {
                if let Some(mol) = &state.peptide {
                    mol.center.into()
                } else {
                    lin_alg::f32::Vec3::new_zero()
                }
            }
        }
    } else {
        if let Some(mol) = &state.peptide {
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
        // todo: For organization purposes, move this code out of the UI.
        Ok((cif, cif_text)) => {
            let Some(ff_map) = &state.ff_param_set.peptide_ff_q_map else {
                handle_err(
                    &mut state.ui,
                    "Unable to find the peptide FF Q map in parameters; can't load the molecule"
                        .to_owned(),
                );
                return;
            };

            let mut mol: MoleculePeptide =
                match MoleculePeptide::from_mmcif(cif, ff_map, None, state.to_save.ph) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("Problem parsing mmCif data into molecule: {e:?}");
                        return;
                    }
                };

            state.volatile.aa_seq_text = String::with_capacity(mol.common.atoms.len());
            for aa in &mol.aa_seq {
                state
                    .volatile
                    .aa_seq_text
                    .push_str(&aa.to_str(AaIdent::OneLetter));
            }

            // todo: DRY from `open_molecule`. Refactor into shared code?

            state.volatile.aa_seq_text = String::with_capacity(mol.common.atoms.len());
            for aa in &mol.aa_seq {
                state
                    .volatile
                    .aa_seq_text
                    .push_str(&aa.to_str(AaIdent::OneLetter));
            }

            state.volatile.flags.ss_mesh_created = false;
            state.volatile.flags.sas_mesh_created = false;
            state.volatile.flags.clear_density_drawing = true;
            state.peptide = Some(mol);
            state.cif_pdb_raw = Some(cif_text);
        }
        Err(e) => {
            handle_err(
                &mut state.ui,
                format!("Problem loading molecule from CIF: {e:?}"),
            );
            return;
        }
    }

    state.update_from_prefs();

    *redraw = true;
    *reset_cam = true;
    set_flashlight(scene);
    engine_updates.lighting = true;

    // todo: async
    // Only after updating from prefs (to prevent unecesasary loading) do we update data avail.
    state
        .peptide
        .as_mut()
        .unwrap()
        .updates_rcsb_data(&mut state.volatile.mol_pending_data_avail);
}

pub fn save_snap(state: &mut State, cam: &Camera, name: &str) {
    state
        .cam_snapshots
        .push(CamSnapshot::from_cam(cam, name.to_owned()));
    state.ui.cam_snapshot_name = String::new();

    state.ui.cam_snapshot = Some(state.cam_snapshots.len() - 1);

    state.update_save_prefs(false);
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
    mol: &MoleculePeptide,
) {
    let center: lin_alg::f32::Vec3 = mol.center.into();
    scene.camera.position =
        lin_alg::f32::Vec3::new(center.x, center.y, center.z - (mol.size + CAM_INIT_OFFSET));
    scene.camera.orientation = Quaternion::new_identity();

    set_static_light(scene, center, mol.size);
    set_flashlight(scene);

    engine_updates.camera = true;
    engine_updates.lighting = true;

    *view_depth = (VIEW_DEPTH_NEAR_MIN, FOG_DIST_DEFAULT);
}

/// Utility function that prints to stderr, and the CLI output. Sets the out flag.
pub fn handle_err(ui: &mut StateUi, msg: String) {
    eprintln!("{msg}");
    ui.cmd_line_output = msg;
    ui.cmd_line_out_is_err = true;
}

/// Utility function that prints to stdout, and the CLI output. Sets the out flag.
pub fn handle_success(ui: &mut StateUi, msg: String) {
    println!("{msg}");
    ui.cmd_line_output = msg;
    ui.cmd_line_out_is_err = false;
}

pub fn clear_cli_out(ui: &mut StateUi) {
    ui.cmd_line_output = String::new();
    ui.cmd_line_out_is_err = false;
}

pub fn close_peptide(state: &mut State, scene: &mut Scene, engine_updates: &mut EngineUpdates) {
    let path = match &state.peptide {
        Some(mol) => mol.common.path.clone(),
        None => None,
    };

    state.peptide = None;
    state.mol_dynamics = None;

    scene.entities.retain(|ent| {
        ent.class != EntityType::Protein as u32
            && ent.class != EntityType::DensityPoint as u32
            && ent.class != EntityType::DensitySurface as u32
            && ent.class != EntityType::SecondaryStructure as u32
            && ent.class != EntityType::SaSurface as u32
            && ent.class != EntityType::SaSurfaceDots as u32
    });

    state.volatile.aa_seq_text = String::new();

    if let Some(path) = path {
        for history in &mut state.to_save.open_history {
            if let OpenType::Peptide = history.type_ {
                if history.path == path {
                    history.last_session = false;
                }
            }
        }
    }

    state.update_save_prefs(false);

    engine_updates.entities = true;
}

pub fn close_mol(
    mol_type: MolType,
    i: usize,
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
) {
    state.volatile.mol_manip.mol = ManipMode::None;
    engine_updates.entities = true;

    match mol_type {
        MolType::Ligand => {
            if i >= state.ligands.len() {
                eprintln!("Error: Invalid lig index");
                return;
            }

            let path = state.ligands[i].common.path.clone();

            state.ligands.remove(i);

            if !state.ligands.is_empty() {
                state.volatile.active_mol = Some((MolType::Ligand, state.ligands.len() - 1));
            }

            draw_all_ligs(state, scene);

            if let Some(path) = path {
                for history in &mut state.to_save.open_history {
                    if let OpenType::Ligand = history.type_ {
                        if history.path == path {
                            history.last_session = false;
                        }
                    }
                }
            }

            state.update_save_prefs(false);
        }
        // todo: DRY
        MolType::NucleicAcid => {
            if i >= state.nucleic_acids.len() {
                eprintln!("Error: Invalid nucleic acid index");
                return;
            }

            state.nucleic_acids.remove(i);

            if !state.nucleic_acids.is_empty() {
                state.volatile.active_mol =
                    Some((MolType::NucleicAcid, state.nucleic_acids.len() - 1));
            }

            draw_all_nucleic_acids(state, scene);
        }
        MolType::Lipid => {
            if i >= state.lipids.len() {
                eprintln!("Error: Invalid lipid index");
                return;
            }

            state.lipids.remove(i);

            if !state.lipids.is_empty() {
                state.volatile.active_mol = Some((MolType::Lipid, state.lipids.len() - 1));
            }

            draw_all_lipids(state, scene);
        }
        _ => unimplemented!(),
    }
}

/// Populate the electron-density mesh (isosurface). This assumes the density_rect is already set up.
pub fn make_density_mesh(state: &mut State, scene: &mut Scene, engine_updates: &mut EngineUpdates) {
    let Some(mol) = &state.peptide else {
        return;
    };
    let Some(rect) = &mol.density_rect else {
        return;
    };
    let Some(density) = &mol.elec_density else {
        return;
    };

    let dims = (rect.dims[0], rect.dims[1], rect.dims[2]); // (nx, ny, nz)

    let size = (
        (rect.step[0] * rect.dims[0] as f64) as f32, // Δx * nx  (Å)
        (rect.step[1] * rect.dims[1] as f64) as f32,
        (rect.step[2] * rect.dims[2] as f64) as f32,
    );

    let sampling_interval = (
        rect.dims[0] as f32,
        rect.dims[1] as f32,
        rect.dims[2] as f32,
    );

    match MarchingCubes::from_gridpoints(
        dims,
        size,
        sampling_interval,
        rect.origin_cart.into(),
        density,
        state.ui.density_iso_level,
    ) {
        Ok(mc) => {
            let mesh = mc.generate(MeshSide::OutsideOnly);

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

/// Code here is ctivated by flags. It's organized here, where we have access to the Scene.
/// These flags are set in places that don't have access to the scene.
pub fn handle_scene_flags(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
) {
    if state.volatile.flags.new_mol_loaded {
        state.volatile.flags.new_mol_loaded = false;

        if let Some(mol) = &state.peptide {
            reset_camera(scene, &mut state.ui.view_depth, engine_updates, mol);
        }

        set_flashlight(scene);
        engine_updates.lighting = true;
    }

    if state.volatile.flags.new_density_loaded {
        state.volatile.flags.new_density_loaded = false;

        if let Some(mol) = &state.peptide {
            if !state.ui.visibility.hide_density_point_cloud {
                if let Some(density) = &mol.elec_density {
                    draw_density_point_cloud(&mut scene.entities, density);
                    engine_updates.entities = true;
                    return;
                }
            }
        }
    }

    if state.volatile.flags.clear_density_drawing {
        state.volatile.flags.clear_density_drawing = false;

        scene.entities.retain(|ent| {
            ent.class != EntityType::DensityPoint as u32
                && ent.class != EntityType::DensitySurface as u32
        });
    }

    if state.volatile.flags.make_density_iso_mesh {
        state.volatile.flags.make_density_iso_mesh = false;
        make_density_mesh(state, scene, engine_updates);
    }

    if state.volatile.flags.update_ss_mesh {
        state.volatile.flags.update_ss_mesh = false;
        state.volatile.flags.ss_mesh_created = true;

        if let Some(mol) = &state.peptide {
            scene.meshes[MESH_SECONDARY_STRUCTURE] =
                build_cartoon_mesh(&mol.secondary_structure, &mol.common.atoms);

            engine_updates.meshes = true;
        }
    }

    if state.volatile.flags.update_sas_mesh {
        state.volatile.flags.update_sas_mesh = false;
        state.volatile.flags.sas_mesh_created = true;

        if let Some(mol) = &state.peptide {
            let atoms: Vec<&_> = mol.common.atoms.iter().filter(|a| !a.hetero).collect();
            scene.meshes[MESH_SOLVENT_SURFACE] =
                make_sas_mesh(&atoms, state.to_save.sa_surface_precision);

            // We draw the molecule here
            if matches!(
                state.ui.mol_view,
                MoleculeView::Dots | MoleculeView::Surface
            ) {
                // The dots are drawn from the mesh vertices
                draw_peptide(state, scene);
                engine_updates.entities = true;
            }

            engine_updates.meshes = true;
        }
    }

    if state.volatile.mol_pending_data_avail.is_some() {
        if let Some(mol) = &mut state.peptide {
            if mol.poll_data_avail(&mut state.volatile.mol_pending_data_avail) {
                state.update_save_prefs(false);
            }
        }
    }
}

pub fn make_egui_color(color: Color) -> Color32 {
    Color32::from_rgb(
        (color.0 * 255.) as u8,
        (color.1 * 255.) as u8,
        (color.2 * 255.) as u8,
    )
}

/// Align the ligand to a residue on a molecule. This is generally a hetero copy of the ligand,
/// as part of the protein's coordinate file. Attempt to match atoms exactly; this requires the ligand's
/// atom labels to match that in the residue.. If not,
/// use a flexible conformation, or match partly.
///
/// Return a center suitable for docking.
// pub fn move_mol_to_res(lig: &mut MoleculeSmall, mol: &MoleculePeptide, res: &Residue) -> Vec3 {
pub fn move_mol_to_res(
    mol: &mut MoleculeGenericRefMut,
    peptide: &MoleculePeptide,
    res: &Residue,
) -> Vec3 {
    // todo: Pick center-of-mass atom, or better yet, match it to the anchor atom.
    let posit = peptide.common.atoms[res.atoms[0]].posit;

    // todo: YOu need to add hydrogens to hetero atoms.

    let mut all_found = false;
    for lig_i in 0..mol.common().atoms.len() {
        let lig_type_in_res = { &mol.common().atoms[lig_i].type_in_res };
        if lig_type_in_res.is_none() {
            continue;
        }
        let mut found = false;

        for i in &res.atoms {
            let atom_res = &peptide.common.atoms[*i];
            if atom_res.type_in_res.is_none() {
                continue;
            }

            if atom_res.type_in_res == *lig_type_in_res {
                mol.common_mut().atom_posits[lig_i] = atom_res.posit;
                found = true;
                break;
            }
        }
        if !found {
            // todo: If it's just a few, automatically position based on geometry to the positioned atoms.
            eprintln!("Unable to position a ligand atom based on the residue.");
            all_found = false;
            // todo: Temp break rm until we can add het Hydrogens or find a workaround.
            // break;
        }
    }

    // lig.pose.conformation_type = if all_found {
    //     println!("Found all atoms required to position ligand to residue.");
    //     ConformationType::AbsolutePosits
    // } else {
    //     // todo temp abs until we populate het Hydrogens or find a workaround
    //     ConformationType::AbsolutePosits
    //
    //     // ConformationType::Flexible {
    //     //     torsions: Vec::new(),
    //     // }
    // };

    posit
}

/// A helper used, for example, for orienting double bonds. Finds an arbitrary neighbor to the bond.
/// Returns neighbor's index. Return the index instead of posit for flexibility, e.g. with lig.common.atom_posits.
/// Returns (index, if index is from atom 1). This is important for knowing which side we're working with.
///
/// Note: We don't take Hydrogens into account, because they confuse the situation of aromatic rings.
pub fn find_neighbor_posit(
    mol: &MoleculeCommon,
    atom_0: usize,
    atom_1: usize,
    hydrogen_is: &[bool],
) -> Option<(usize, bool)> {
    let neighbors_0 = &mol.adjacency_list[atom_0];

    if neighbors_0.len() >= 2 {
        for neighbor in neighbors_0 {
            if !hydrogen_is[*neighbor] {}
            if *neighbor != atom_1 && !hydrogen_is[*neighbor] {
                return Some((*neighbor, false));
            }
        }
    }

    let neighbors_1 = &mol.adjacency_list[atom_1];

    if !neighbors_1.len() >= 2 {
        for neighbor in neighbors_1 {
            if *neighbor != atom_0 && !hydrogen_is[*neighbor] {
                return Some((*neighbor, true));
            }
        }
    }

    None
}

// /// We use this when moving molecules. We use the same movement logic as Blender, where moving an object
// /// in 2d with the cursor reflects a 3D movement dependent on camera position.
// pub fn transform_to_3d(pos_2d: (f32, f32), cam_posit: Vec3F32, cam_or: Quaternion) -> Vec3 {
//     // todo placeholder!
//     Vec3::new(motion.0 as f64, 0., motion.1 as f64)
// }

pub fn move_cam_to_sel(
    state_ui: &mut StateUi,
    mol_: &Option<MoleculePeptide>,
    ligs: &[MoleculeSmall],
    nucleic_acids: &[MoleculeNucleicAcid],
    lipids: &[MoleculeLipid],
    cam: &mut Camera,
    engine_updates: &mut EngineUpdates,
) {
    match &state_ui.selection {
        Selection::AtomPeptide(_i_atom) => {
            let Some(mol) = mol_ else {
                return;
            };
            let atom_sel = mol.get_sel_atom(&state_ui.selection);

            if let Some(atom) = atom_sel {
                cam_look_at(cam, atom.posit);
            }
        }
        Selection::AtomLig((i_mol, i_atom)) => {
            cam_look_at(cam, ligs[*i_mol].common.atom_posits[*i_atom]);
        }
        Selection::AtomNucleicAcid((i_mol, i_atom)) => {
            // if *i_mol >= nucleic_acids.len() {
            //     return;
            // }
            cam_look_at(cam, nucleic_acids[*i_mol].common.atom_posits[*i_atom]);
        }
        Selection::AtomLipid((i_mol, i_atom)) => {
            cam_look_at(cam, lipids[*i_mol].common.atom_posits[*i_atom]);
        }
        _ => unimplemented!(),
    }

    engine_updates.camera = true;
    state_ui.cam_snapshot = None;
}
