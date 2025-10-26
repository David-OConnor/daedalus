//! A collection of utility functions that don't neatly belong in other modules.
//! For example, we may call some of these from the GUI, but they won't have any EGUI-specific
//! logic in them.

use std::time::Instant;

use bio_files::ResidueType;
#[cfg(feature = "cudarc")]
use cudarc::{
    driver::{CudaContext, CudaFunction},
    nvrtc::Ptx,
};
#[cfg(feature = "cuda")]
use dynamics::ComputationDevice;
use egui::Color32;
use graphics::{Camera, ControlScheme, EngineUpdates, EntityUpdate, FWD_VEC, Scene};
use lin_alg::{f32::Vec3 as Vec3F32, f64::Vec3};
use na_seq::{AaIdent, Element};

use crate::{
    CamSnapshot, ManipMode, PREFS_SAVE_INTERVAL, Selection, State, StateUi, ViewSelLevel, cam_misc,
    drawing::{EntityClass, MoleculeView, draw_density_point_cloud, draw_peptide},
    drawing_wrappers::{draw_all_ligs, draw_all_lipids, draw_all_nucleic_acids},
    mol_lig::MoleculeSmall,
    molecule::{Atom, Bond, MoGenericRefMut, MolGenericRef, MolType, MoleculePeptide, Residue},
    prefs::OpenType,
    reflection,
    render::{Color, MESH_SECONDARY_STRUCTURE, MESH_SOLVENT_SURFACE, set_flashlight},
    ribbon_mesh::build_cartoon_mesh,
    sa_surface::make_sas_mesh,
};

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

pub fn select_from_search(state: &mut State) {
    let query = &state.ui.atom_res_search.to_lowercase();

    let Some(mol) = &state.peptide else {
        return;
    };

    match state.ui.view_sel_level {
        ViewSelLevel::Atom => {
            for (i, atom) in mol.common.atoms.iter().enumerate() {
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
        ViewSelLevel::Bond => {
            for (i, bond) in mol.common.bonds.iter().enumerate() {
                if query == &bond.atom_0_sn.to_string() || query == &bond.atom_1_sn.to_string() {
                    state.ui.selection = Selection::AtomPeptide(i);
                    return;
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
                if new_atom_i < mol.common().atoms.len() as isize && new_atom_i >= 0 {
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
                if new_atom_i < mol.common().atoms.len() as isize && new_atom_i >= 0 {
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
                if new_atom_i < mol.common().atoms.len() as isize && new_atom_i >= 0 {
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
        ViewSelLevel::Bond => match state.ui.selection {
            Selection::BondPeptide(bond_i) => {
                let Some(mol) = &state.peptide else {
                    return;
                };

                let new_bond_i = bond_i as isize + dir;
                if new_bond_i < mol.common.bonds.len() as isize && new_bond_i >= 0 {
                    state.ui.selection = Selection::BondPeptide(new_bond_i as usize);
                }
            }
            Selection::BondLig((mol_i, bond_i)) => {
                let Some(mol) = state.active_mol() else {
                    return;
                };

                let new_bond_i = bond_i as isize + dir;
                if new_bond_i < mol.common().bonds.len() as isize && new_bond_i >= 0 {
                    state.ui.selection = Selection::BondLig((mol_i, new_bond_i as usize));
                }
            }
            Selection::BondNucleicAcid((mol_i, bond_i)) => {
                let Some(mol) = state.active_mol() else {
                    return;
                };

                let new_bond_i = bond_i as isize + dir;
                if new_bond_i < mol.common().bonds.len() as isize && new_bond_i >= 0 {
                    state.ui.selection = Selection::BondNucleicAcid((mol_i, new_bond_i as usize));
                }
            }
            Selection::BondLipid((mol_i, bond_i)) => {
                let Some(mol) = state.active_mol() else {
                    return;
                };

                let new_bond_i = bond_i as isize + dir;
                if new_bond_i < mol.common().bonds.len() as isize && new_bond_i >= 0 {
                    state.ui.selection = Selection::BondLipid((mol_i, new_bond_i as usize));
                }
            }
            _ => (),
        },
    }

    if let ControlScheme::Arc { center } = &mut scene.input_settings.control_scheme {
        *center = orbit_center(state);
    }
}

pub fn check_prefs_save(state: &mut State) {
    static mut LAST_PREF_SAVE_CHECK: Option<Instant> = None;
    let now = Instant::now();

    unsafe {
        if let Some(last_save) = LAST_PREF_SAVE_CHECK {
            if (now - last_save).as_secs() > PREFS_SAVE_INTERVAL {
                LAST_PREF_SAVE_CHECK = Some(now);

                if state.to_save != state.to_save_prev {
                    state.update_save_prefs(false)
                }
            }
        } else {
            // Initialize LAST_PREF_SAVE the first time it's accessed
            LAST_PREF_SAVE_CHECK = Some(now);
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

/// Based on selection status and if a molecule is open, find the center for the orbit camera. This
/// is generally around a specific atom, or a molecule's centroid.
pub fn orbit_center(state: &State) -> Vec3F32 {
    if state.ui.orbit_selected_atom {
        match &state.ui.selection {
            Selection::AtomPeptide(i) => {
                if let Some(mol) = &state.peptide {
                    match mol.common.atoms.get(*i) {
                        Some(a) => return a.posit.into(),
                        None => (),
                    }
                }
            }
            Selection::AtomLig((i_mol, i_atom)) => {
                return state.ligands[*i_mol].common.atom_posits[*i_atom].into();
            }
            Selection::AtomNucleicAcid((i_mol, i_atom)) => {
                return state.nucleic_acids[*i_mol].common.atom_posits[*i_atom].into();
            }
            Selection::AtomLipid((i_mol, i_atom)) => {
                return state.lipids[*i_mol].common.atom_posits[*i_atom].into();
            }

            Selection::Residue(i) => {
                if let Some(mol) = &state.peptide {
                    match mol.residues.get(*i) {
                        Some(res) => {
                            match mol.common.atoms.get(match res.atoms.first() {
                                Some(a) => *a, // todo: What?
                                None => return Vec3F32::new_zero(),
                            }) {
                                Some(a) => return a.posit.into(),
                                None => return Vec3F32::new_zero(),
                            }
                        }
                        None => return Vec3F32::new_zero(),
                    }
                }
            }
            Selection::AtomsPeptide(is) => {
                if let Some(mol) = &state.peptide {
                    match mol.common.atoms.get(is[0]) {
                        Some(a) => return a.posit.into(),
                        None => (),
                    }
                }
            }
            Selection::BondPeptide(i_atom) => {
                if let Some(mol) = &state.peptide {
                    match mol.common.bonds.get(*i_atom) {
                        Some(bond) => {
                            return ((mol.common.atom_posits[bond.atom_0]
                                + mol.common.atom_posits[bond.atom_1])
                                / 2.)
                                .into();
                        }
                        None => (),
                    }
                }
            }
            Selection::BondLig((i_mol, i_bond)) => {
                let mol = &state.ligands[*i_mol];
                let bond = &mol.common.bonds[*i_bond];
                return ((mol.common.atom_posits[bond.atom_0]
                    + mol.common.atom_posits[bond.atom_1])
                    / 2.)
                    .into();
            }
            Selection::BondNucleicAcid((i_mol, i_bond)) => {
                let mol = &state.nucleic_acids[*i_mol];
                let bond = &mol.common.bonds[*i_bond];
                return ((mol.common.atom_posits[bond.atom_0]
                    + mol.common.atom_posits[bond.atom_1])
                    / 2.)
                    .into();
            }
            Selection::BondLipid((i_mol, i_bond)) => {
                let mol = &state.lipids[*i_mol];
                let bond = &mol.common.bonds[*i_bond];
                return ((mol.common.atom_posits[bond.atom_0]
                    + mol.common.atom_posits[bond.atom_1])
                    / 2.)
                    .into();
            }
            Selection::None => {
                if let Some(mol) = &state.peptide {
                    return mol.center.into();
                }
            }
        }
        // Orbit around the selected molecule's centroid. Failing that, the origin.
    } else {
        let Some((mol_type, i)) = &state.volatile.orbit_center else {
            return Vec3F32::new_zero();
        };
        let i = *i;

        match mol_type {
            MolType::Peptide => {
                if let Some(mol) = &state.peptide {
                    // Used the cached position, as computing centroid may be expensive
                    // for large proteins.
                    return mol.center.into();
                }
            }
            MolType::Ligand => {
                if i < state.ligands.len() {
                    return state.ligands[i].common.centroid().into();
                }
            }
            MolType::NucleicAcid => {
                if i < state.nucleic_acids.len() {
                    return state.nucleic_acids[i].common.centroid().into();
                }
            }
            MolType::Lipid => {
                if i < state.lipids.len() {
                    return state.lipids[i].common.centroid().into();
                }
            }
            MolType::Water => (),
        }
    }

    Vec3F32::new_zero()
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
        ent.class != EntityClass::Protein as u32
            && ent.class != EntityClass::DensityPoint as u32
            && ent.class != EntityClass::DensitySurface as u32
            && ent.class != EntityClass::SecondaryStructure as u32
            && ent.class != EntityClass::SaSurface as u32
            && ent.class != EntityClass::SaSurfaceDots as u32
    });
    clear_mol_entity_indices(state, None);

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

    engine_updates.entities = EntityUpdate::All;
    // engine_updates.entities.push_class(EntityClass::Peptide as u32);

    if let Some((orbit_mol_type, orbit_i)) = &state.volatile.orbit_center
        && (*orbit_mol_type, *orbit_i) == (MolType::Peptide, 0)
    {
        reset_orbit_center(state, scene);
    }
}

/// Close the active molecule.
pub fn close_mol(
    mol_type: MolType,
    i: usize,
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
) {
    state.volatile.mol_manip.mol = ManipMode::None;
    engine_updates.entities = EntityUpdate::All;

    match mol_type {
        MolType::Peptide => {
            close_peptide(state, scene, engine_updates);
        }
        MolType::Ligand => {
            if i >= state.ligands.len() {
                eprintln!("Error: Invalid lig index");
                return;
            }

            let path = state.ligands[i].common.path.clone();

            state.ligands.remove(i);

            if state.ligands.is_empty() {
                state.volatile.active_mol = None;
            } else {
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

            if state.nucleic_acids.is_empty() {
                state.volatile.active_mol = None;
            } else {
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

            if state.lipids.is_empty() {
                state.volatile.active_mol = None;
            } else {
                state.volatile.active_mol = Some((MolType::Lipid, state.lipids.len() - 1));
            }

            draw_all_lipids(state, scene);
        }
        MolType::Water => (),
    }

    if let Some((orbit_mol_type, orbit_i)) = &state.volatile.orbit_center
        && (*orbit_mol_type, *orbit_i) == (mol_type, i)
    {
        reset_orbit_center(state, scene);
    }
}

pub fn reset_orbit_center(state: &mut State, scene: &mut Scene) {
    // Reset the arc center, if in that camera mode, and molecule was the active one.

    if state.peptide.is_some() {
        state.volatile.orbit_center = Some((MolType::Peptide, 0));
    } else if !state.ligands.is_empty() {
        state.volatile.orbit_center = Some((MolType::Ligand, state.ligands.len() - 1));
    } else if !state.nucleic_acids.is_empty() {
        state.volatile.orbit_center = Some((MolType::NucleicAcid, state.nucleic_acids.len() - 1));
    } else if !state.lipids.is_empty() {
        state.volatile.orbit_center = Some((MolType::Lipid, state.lipids.len() - 1));
    }

    if let ControlScheme::Arc { center } = &mut scene.input_settings.control_scheme {
        *center = orbit_center(state);
    }
}

/// Code here is activated by flags. It's organized here, where we have access to the Scene.
/// These flags are set in places that don't have access to the scene.
pub fn handle_scene_flags(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
) {
    if state.volatile.flags.new_mol_loaded {
        state.volatile.flags.new_mol_loaded = false;

        cam_misc::reset_camera(state, scene, engine_updates, FWD_VEC);

        set_flashlight(scene);
        engine_updates.lighting = true;
    }

    if state.volatile.flags.new_density_loaded {
        state.volatile.flags.new_density_loaded = false;

        if let Some(mol) = &state.peptide
            && !state.ui.visibility.hide_density_point_cloud
        {
            if let Some(density) = &mol.elec_density {
                draw_density_point_cloud(&mut scene.entities, density);
                clear_mol_entity_indices(state, None);
                engine_updates.entities = EntityUpdate::All;
                // engine_updates
                //     .entities
                //     .push_class(EntityClass::DensityPoint as u32);
                return;
            }
        }
    }

    if state.volatile.flags.clear_density_drawing {
        state.volatile.flags.clear_density_drawing = false;

        scene.entities.retain(|ent| {
            ent.class != EntityClass::DensityPoint as u32
                && ent.class != EntityClass::DensitySurface as u32
        });
        clear_mol_entity_indices(state, None);
    }

    if state.volatile.flags.make_density_iso_mesh {
        state.volatile.flags.make_density_iso_mesh = false;
        reflection::make_density_mesh(state, scene, engine_updates);
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
                engine_updates.entities = EntityUpdate::All;
                // engine_updates.entities.push_class(EntityClass::SaSurface as u32);
                // engine_updates
                //     .entities
                //     .push_class(EntityClass::SaSurfaceDots as u32);
            }

            engine_updates.meshes = true;
        }
    }

    if state.volatile.mol_pending_data_avail.is_some()
        && let Some(mol) = &mut state.peptide
        && mol.poll_data_avail(&mut state.volatile.mol_pending_data_avail)
    {
        state.update_save_prefs(false);
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
    mol: &mut MoGenericRefMut,
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
    adj_list: &[Vec<usize>],
    atom_0: usize,
    atom_1: usize,
    hydrogen_is: &[bool],
) -> Option<(usize, bool)> {
    // let neighbors_0 = &mol.adjacency_list[atom_0];
    let neighbors_0 = &adj_list[atom_0];

    if neighbors_0.len() >= 2 {
        for neighbor in neighbors_0 {
            if *neighbor != atom_1 && !hydrogen_is[*neighbor] {
                return Some((*neighbor, false));
            }
        }
    }

    // let neighbors_1 = &mol.adjacency_list[atom_1];
    let neighbors_1 = &adj_list[atom_1];

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

// todo: Maybe only invalidate indices that come before?
/// We use this to invalidate indices when removing entities. Only run this when entities are removed.
pub fn clear_mol_entity_indices(state: &mut State, exempt: Option<MolType>) {
    println!("Clearing indices");
    if let Some(pep) = &mut state.peptide {
        let mut skip = false;
        if let Some(e) = exempt {
            if e == MolType::Ligand {
                skip = true;
            }
        }
        if !skip {
            pep.common.entity_i_range = None;
        }
    }
    for mol in &mut state.ligands {
        if let Some(e) = exempt {
            if e == MolType::Ligand {
                break;
            }
        }
        mol.common.entity_i_range = None;
    }
    for mol in &mut state.nucleic_acids {
        if let Some(e) = exempt {
            if e == MolType::NucleicAcid {
                break;
            }
        }
        mol.common.entity_i_range = None;
    }
    for mol in &mut state.lipids {
        if let Some(e) = exempt {
            if e == MolType::Lipid {
                break;
            }
        }
        mol.common.entity_i_range = None;
    }
}

// pub fn make_lig_from_res(state: &mut State, res: &Residue, redraw_lig: &mut bool, lig_to_cam: Option<&Camera>) {
pub fn make_lig_from_res(state: &mut State, res: &Residue, redraw_lig: &mut bool) {
    let mol = &state.peptide.as_ref().unwrap().common;
    let mut mol_fm_res = MoleculeSmall::from_res(res, &mol.atoms, &mol.bonds);

    mol_fm_res.update_aux(&state.volatile.active_mol, &mut state.lig_specific_params);

    state.ligands.push(mol_fm_res);

    *redraw_lig = true;
    state.mol_dynamics = None;

    // We leave the new ligand in place, overlapping the residue we created it from.

    state.volatile.active_mol = Some((MolType::Ligand, state.ligands.len() - 1));

    // Make it clear that we've added the ligand by showing it, and hiding hetero (if creating from Hetero)
    state.ui.visibility.hide_ligand = false;
}

fn find_nearest_mol_inner(mol: MolGenericRef<'_>, cam: &Camera) -> Option<f32> {
    let posit: Vec3F32 = mol.common().atom_posits[0].into();

    // todo: Base the offset on the molecule size, e.g. atom count.
    if cam.in_view(posit).0 {
        return Some((cam.position - posit).magnitude() - 4.);
    }

    None
}

/// todo: Experimental
/// Find the distance of the closest molecule to the camera, in front of it. Run this regularly upon
/// camera movement, e.g. every x steps where camera position or orientation changes.
///
/// Returns None if there are no molecules in the camera FOV.
pub fn find_nearest_mol_dist_to_cam(state: &State, cam: &Camera) -> Option<f32> {
    let mut nearest = f32::INFINITY;

    // For the protein, rely on cached distances along a collection of radials.
    if let Some(pep) = &state.peptide {
        // todo: Very slow approach for now to demonstrate concept. Change this to use a cache!!
        for (i, _atom) in pep
            .common
            .atoms
            .iter()
            .filter(|a| a.element == Element::Carbon)
            .enumerate()
        {
            if !i.is_multiple_of(20) {
                continue;
            }

            let posit: Vec3F32 = pep.common.atom_posits[i].into();
            if cam.in_view(posit.into()).0 {
                let dist = (cam.position - posit).magnitude() - 4.;
                if dist < nearest {
                    nearest = dist;
                }
            }
        }
    }

    for mol in &state.ligands {
        if let Some(v) = find_nearest_mol_inner(MolGenericRef::Ligand(mol), cam)
            && v < nearest
        {
            nearest = v;
        }
    }

    for mol in &state.nucleic_acids {
        if let Some(v) = find_nearest_mol_inner(MolGenericRef::NucleicAcid(mol), cam)
            && v < nearest
        {
            nearest = v;
        }
    }

    for mol in &state.lipids {
        if let Some(v) = find_nearest_mol_inner(MolGenericRef::Lipid(mol), cam)
            && v < nearest
        {
            nearest = v;
        }
    }

    if nearest != f32::INFINITY {
        return Some(nearest);
    }
    None
}

/// This enables GPU computation if the right compiler flag is set, and there aren't
/// errors setting up the Cuda stream. It also handles loading cuda kernels used directly
/// by this application. (Dynamics modules, for example, are handled by that library)
#[cfg(feature = "cuda")]
pub fn get_computation_device() -> (ComputationDevice, Option<CudaFunction>) {
    match cudarc::driver::result::init() {
        Ok(_) => {
            let ctx = CudaContext::new(0).unwrap();
            let stream = ctx.default_stream();

            let module_reflections = ctx.load_module(Ptx::from_src(crate::PTX));

            match module_reflections {
                Ok(m) => {
                    let function = m.load_function("make_densities_kernel").unwrap();
                    (ComputationDevice::Gpu(stream), Some(function))
                }
                Err(e) => {
                    eprintln!("Error loading CUDA module; not using CUDA. Error: {e}");
                    (ComputationDevice::Cpu, None)
                }
            }
        }
        Err(e) => {
            eprintln!("Unable to init Cuda module: {e:?}");
            (ComputationDevice::Cpu, None)
        }
    }
}
