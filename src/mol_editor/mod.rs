pub mod add_atoms;
pub mod templates;

use std::{
    collections::HashMap,
    io,
    io::ErrorKind,
    path::Path,
    sync::atomic::{AtomicU32, Ordering},
    time::Instant,
};

use bio_files::{BondType, Mol2, Pdbqt, Sdf, Xyz, md_params::ForceFieldParams};
use dynamics::{
    ComputationDevice, FfMolType, HydrogenConstraint, Integrator, MdConfig, MdOverrides, MdState,
    MolDynamics, ParamError, TAU_TEMP_DEFAULT, params::FfParamSet, snapshot::Snapshot,
};
use graphics::{ControlScheme, EngineUpdates, Entity, EntityUpdate, Scene};
use lin_alg::{
    f32::{Quaternion, Vec3 as Vec3F32},
    f64::Vec3,
};
use na_seq::{
    AtomTypeInRes,
    Element::{Carbon, Hydrogen},
};

use crate::mol_manip::MolManip;
use crate::{
    OperatingMode, Selection, State, StateUi, ViewSelLevel,
    drawing::{
        EntityClass, MESH_BALL_STICK_SPHERE, MESH_SPACEFILL_SPHERE, MoleculeView, atom_color,
        bond_entities, draw_mol, draw_peptide,
    },
    drawing_wrappers::{draw_all_ligs, draw_all_lipids, draw_all_nucleic_acids},
    md::change_snapshot_helper,
    mol_editor,
    mol_lig::MoleculeSmall,
    mol_manip::ManipMode,
    molecule::{Atom, Bond, MolGenericRef, MolType},
    render::{
        ATOM_SHININESS, BALL_STICK_RADIUS, BALL_STICK_RADIUS_H, set_flashlight, set_static_light,
    },
    ui::UI_HEIGHT_CHANGED,
    util::find_neighbor_posit,
};

pub const INIT_CAM_DIST: f32 = 20.;

// Set a higher value to place the light farther away. (More uniform, dimmer lighting)
pub const STATIC_LIGHT_MOL_SIZE: f32 = 500.;

pub static NEXT_ATOM_SN: AtomicU32 = AtomicU32::new(0);

const MOL_IDENT: &str = "editor_mol";

/// For editing small organic molecules.
pub struct MolEditorState {
    pub mol: MoleculeSmall,
    pub md_state: Option<MdState>,
    pub mol_specific_params: ForceFieldParams,
    /// Picoseconds. Combined with how often we run MD. 0.001 - 0.002 is good for preventing
    /// the simulation from blowing up, but we have other concerns re frame rate and desired ratio.
    /// todo: Make this customizable in the UI, and display the ratio.
    pub dt_md: f32,
    /// ms. Sim time:real time ratio = dt_md x s x 10^-12 / (time_between_runs x s x 10^-3) = dt_md / time_between_runs * 10^-9)
    // ~30fps updates. Conservative. Should vary this based on how fast it's taking for the total
    // frame, including MD.
    pub time_between_md_runs: f32,
    pub md_running: bool,
    pub md_skip_water: bool,
    pub snap: Option<Snapshot>,
    pub last_dt_run: Instant,
    pub md_rebuild_required: bool,
}

impl Default for MolEditorState {
    fn default() -> Self {
        Self {
            mol: Default::default(),
            md_state: Default::default(),
            md_skip_water: true,
            mol_specific_params: Default::default(),
            dt_md: 0.00001,
            time_between_md_runs: 33.333,
            md_running: Default::default(),
            last_dt_run: Instant::now(),
            snap: None,
            md_rebuild_required: false,
        }
    }
}

impl MolEditorState {
    /// For now, sets up a pair of single-bonded carbon atoms.
    pub fn clear_mol(&mut self) {
        // todo: Change this dist; rough start.
        const DIST: f64 = 1.3;

        let mol = &mut self.mol.common;

        mol.atoms = vec![
            Atom {
                serial_number: 1,
                posit: Vec3::new_zero(),
                element: Carbon,
                type_in_res: Some(AtomTypeInRes::C), // todo: no; fix this
                force_field_type: Some("ca".to_owned()), // todo: A/R
                partial_charge: Some(0.),            // todo: A/R,
                ..Default::default()
            },
            Atom {
                serial_number: 2,
                posit: Vec3::new(DIST, 0., 0.),
                element: Carbon,
                type_in_res: Some(AtomTypeInRes::C), // todo: no; fix this
                force_field_type: Some("ca".to_owned()), // todo: A/R
                partial_charge: Some(0.),            // todo: A/R,
                ..Default::default()
            },
        ];

        mol.bonds = vec![Bond {
            bond_type: BondType::Single,
            atom_0_sn: 1,
            atom_1_sn: 2,
            atom_0: 0,
            atom_1: 1,
            is_backbone: false,
        }];

        mol.reset_posits();
        mol.build_adjacency_list();
    }

    /// A simplified variant of our primary `open_molecule` function.
    pub fn open_molecule(
        &mut self,
        dev: &ComputationDevice,
        param_set: &FfParamSet,
        mol_specific_params: &HashMap<String, ForceFieldParams>,
        md_cfg: &MdConfig,
        path: &Path,
        scene: &mut Scene,
        engine_updates: &mut EngineUpdates,
        state_ui: &mut StateUi,
        manip_mode: ManipMode,
    ) -> io::Result<()> {
        let binding = path.extension().unwrap_or_default().to_ascii_lowercase();
        let extension = binding;

        let molecule = match extension.to_str().unwrap() {
            "sdf" => {
                let mut m: MoleculeSmall = Sdf::load(path)?.try_into()?;
                m.common.path = Some(path.to_owned());
                m
            }
            "mol2" => MoleculeSmall::from_xyz(Xyz::load(path)?, path)?,
            "xyz" => {
                let mut m: MoleculeSmall = Mol2::load(path)?.try_into()?;
                m.common.path = Some(path.to_owned());
                m
            }
            "pdbqt" => {
                let mut m: MoleculeSmall = Pdbqt::load(path)?.try_into()?;
                m.common.path = Some(path.to_owned());
                m
            }
            // "cif" => {
            //     // todo
            // }
            _ => {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    "Invalid file extension",
                ));
            }
        };

        self.load_mol(
            &molecule,
            // param_set,
            // mol_specific_params,
            scene,
            engine_updates,
            state_ui,
            manip_mode,
        );
        Ok(())
    }
    //
    // fn _remove_repopulate_h(
    //     &mut self,
    //     scene: &mut Scene,
    //     engine_updates: &mut EngineUpdates,
    //     state_ui: &mut StateUi,
    // ) {
    //     // We assign H dynamically; ignore present ones.
    //     self.mol.common.atoms = self
    //         .mol
    //         .common
    //         .atoms
    //         .iter()
    //         .filter(|a| a.element != Hydrogen)
    //         .map(|a| a.clone())
    //         .collect();
    //
    //     // Remove bonds to atoms that no longer exist, and change indices otherwise:
    //     // serial_number -> new index after filtering
    //     let sn2idx: HashMap<u32, usize> = self
    //         .mol
    //         .common
    //         .atoms
    //         .iter()
    //         .enumerate()
    //         .map(|(i, a)| (a.serial_number, i))
    //         .collect();
    //
    //     // Keep only bonds whose endpoints still exist; reindex to new atom indices
    //     self.mol.common.bonds = self
    //         .mol
    //         .common
    //         .bonds
    //         .iter()
    //         .filter_map(|b| {
    //             let i0 = sn2idx.get(&b.atom_0_sn)?;
    //             let i1 = sn2idx.get(&b.atom_1_sn)?;
    //             Some(Bond {
    //                 bond_type: b.bond_type,
    //                 atom_0_sn: b.atom_0_sn,
    //                 atom_1_sn: b.atom_1_sn,
    //                 atom_0: *i0,
    //                 atom_1: *i1,
    //                 is_backbone: b.is_backbone,
    //             })
    //         })
    //         .collect();
    //
    //     // Rebuild these based on the new filters.
    //     self.mol.common.reset_posits();
    //     self.mol.common.build_adjacency_list();
    //
    //     // Re-populate hydrogens algorithmically.
    //     for (i, atom) in self.mol.common.atoms.clone().iter().enumerate() {
    //         println!("Populating H for atom {atom}");
    //         populate_hydrogens_on_atom(&mut self.mol.common, i, atom.element, &atom.force_field_type, &mut scene.entities, state_ui, engine_updates);
    //     }
    // }

    pub fn load_mol(
        &mut self,
        mol: &MoleculeSmall,
        scene: &mut Scene,
        engine_updates: &mut EngineUpdates,
        state_ui: &mut StateUi,
        manip_mode: ManipMode,
    ) {
        self.mol = mol.clone();
        self.mol.common.ident = MOL_IDENT.to_owned();

        self.mol.common.reset_posits();

        let mut highest_sn = 0;
        for atom in &self.mol.common.atoms {
            if atom.serial_number > highest_sn {
                highest_sn = atom.serial_number;
            }
        }
        NEXT_ATOM_SN.store(highest_sn + 1, Ordering::Release);

        // Load the initial relaxation into atom positions.
        self.load_atom_posits_from_md(&mut scene.entities, state_ui, engine_updates, manip_mode);

        self.mol.smiles = Some(self.mol.common.to_smiles());

        self.move_to_origin();
        scene.input_settings.control_scheme = ControlScheme::Arc {
            center: Vec3F32::new_zero(),
        };

        // Clear all entities for non-editor molecules. And render the initial relaxation
        // from building dynamics.
        redraw(&mut scene.entities, &self.mol, state_ui, manip_mode);

        set_flashlight(scene);
        engine_updates.entities = EntityUpdate::All;
        engine_updates.lighting = true;
    }

    pub fn save_mol2(&self, path: &Path) -> io::Result<()> {
        Ok(())
    }

    pub fn save_sdf(&self, path: &Path) -> io::Result<()> {
        Ok(())
    }

    fn move_to_origin(&mut self) {
        {
            let centroid = self.mol.common.centroid();
            for posit in &mut self.mol.common.atom_posits {
                *posit -= centroid;
            }
            self.mol.common.reset_posits();
        }
    }

    /// Load the latest snapshot into atom positions, and update entities.
    pub fn load_atom_posits_from_snap(
        &mut self,
        entities: &mut Vec<Entity>,
        state_ui: &StateUi,
        engine_updates: &mut EngineUpdates,
        manip_mode: ManipMode,
    ) {
        let Some(snap) = self.md_state.as_ref().unwrap().snapshots.last() else {
            return;
        };

        let mol = &mut self.mol.common;

        change_snapshot_helper(&mut mol.atom_posits, &mut 0, snap);

        // Since we assume they're synced:
        for (i, posit) in mol.atom_posits.iter().enumerate() {
            mol.atoms[i].posit = *posit;
        }
        self.snap = Some(snap.clone());

        self.md_state.as_mut().unwrap().snapshots = Vec::new();

        redraw(entities, &self.mol, state_ui, manip_mode);
        engine_updates.entities = EntityUpdate::All;
    }

    /// Load the latest md_posits into atom positions, and update entities.
    pub fn load_atom_posits_from_md(
        &mut self,
        entities: &mut Vec<Entity>,
        state_ui: &StateUi,
        engine_updates: &mut EngineUpdates,
        manip_mode: ManipMode,
    ) {
        let Some(md) = &self.md_state else { return };
        for (i, atom) in md.atoms.iter().enumerate() {
            self.mol.common.atoms[i].posit = atom.posit.into();
            self.mol.common.atom_posits[i] = atom.posit.into();
        }

        redraw(entities, &self.mol, state_ui, manip_mode);
        engine_updates.entities = EntityUpdate::All;
    }

    /// Run MD for a single step if ready, and update atom positions immediately after.
    pub fn md_step(
        &mut self,
        dev: &ComputationDevice,
        entities: &mut Vec<Entity>,
        state_ui: &StateUi,
        engine_updates: &mut EngineUpdates,
        manip_mode: ManipMode,
    ) {
        static mut I: u32 = 0;

        if !self.md_running
            || self.last_dt_run.elapsed().as_micros() as f32 * 1_000. < self.time_between_md_runs
        {
            return;
        }

        let Some(md) = &mut self.md_state else { return };

        self.last_dt_run = Instant::now();

        md.step(dev, self.dt_md);

        // unsafe {
        //     I += 1;
        //     if I.is_multiple_of(100) {
        //         let elapsed = self.last_dt_run.elapsed().as_micros();
        //         println!("DT ran in: {:?}μs", elapsed);
        //     }
        // }

        // Load the snapshot taken into current atom posits, and redraw.
        // Remove the snap from memory to prevent them from accumulating.
        // todo: use our dynamics posit directly, and clear snapshots?
        self.load_atom_posits_from_snap(entities, state_ui, engine_updates, manip_mode);
    }

    /// Re-assigns FF type, partial charge, and mol-specific (e.g. dihedral) params. An interface to
    /// `dynamics`' general NFF param updator, with setup that makes it work
    /// in this context.
    pub fn rebuild_ff_related(&mut self, param_set: &FfParamSet) {
        // Typical serial number layouts are currently required to prevent crashes when assigning
        // partial charges using SNs.
        self.mol.common.reassign_sns();

        // Setting these to `None` on any atom triggers FF param and partial charge rebuilds.
        if !self.mol.common.atoms.is_empty() {
            self.mol.common.atoms[0].force_field_type = None;
            self.mol.common.atoms[0].partial_charge = None;
        }

        self.mol.frcmod_loaded = false;

        // if let Some(p) = &param_set.small_mol {
        //     editor.mol.update_ff_related(&mut msp, p);
        // } else {
        //     eprintln!("Error: Unable to update a molecule's params due to missing GAFF2.");
        // }

        let mut msp = HashMap::new();

        // Update this immediately, as we may take advantage of FF types when adjusting geometry,
        // and it may be useful to view them.
        if let Some(p) = &param_set.small_mol {
            self.mol.update_ff_related(&mut msp, p);
        } else {
            eprintln!("Error: Unable to update a molecule's params due to missing GAFF2.");
        }

        if let Some(v) = msp.get(MOL_IDENT) {
            self.mol_specific_params = v.clone();
        }
    }
}

// todo: Into a GUI util?
pub fn enter_edit_mode(state: &mut State, scene: &mut Scene, engine_updates: &mut EngineUpdates) {
    state.volatile.operating_mode = OperatingMode::MolEditor;
    UI_HEIGHT_CHANGED.store(true, Ordering::Release);

    // Rebuilt shortly.
    state.mol_editor.md_state = None;

    // This stays false under several conditions.
    let mut mol_loaded = false;

    let mut arc_center = Vec3F32::new_zero();

    if let Some((mol_type, i)) = state.volatile.active_mol
        && mol_type == MolType::Ligand
    {
        if i >= state.ligands.len() {
            eprintln!("Expected a ligand at this index, but out of bounds when entering edit mode");
        } else {
            state.mol_editor.load_mol(
                &state.ligands[i],
                scene,
                engine_updates,
                &mut state.ui,
                state.volatile.mol_manip.mode,
            );
            mol_loaded = true;
            arc_center = state.ligands[i].common.centroid().into();

            state.volatile.mol_editing = Some(i);
        }
    }

    if !mol_loaded {
        state.mol_editor.clear_mol();
    }

    state.volatile.control_scheme_prev = scene.input_settings.control_scheme;

    // Reset positions to be around the origin.
    state.mol_editor.move_to_origin();
    scene.input_settings.control_scheme = ControlScheme::Arc { center: arc_center };
    println!("MOVED TO ORIGIN"); // todo temp

    // Select the first atom.
    state.ui.selection = if state.mol_editor.mol.common.atoms.is_empty() {
        Selection::None
    } else {
        Selection::AtomLig((0, 0))
    };

    state.volatile.primary_mode_cam = scene.camera.clone();
    scene.camera.position = Vec3F32::new(0., 0., -INIT_CAM_DIST);
    scene.camera.orientation = Quaternion::new_identity();

    state.volatile.mol_manip.mode = ManipMode::None;

    // Set to a view supported by the editor.
    // todo: In this case, store the previous view, and re-set it upon exiting the editor.
    if !matches!(
        state.ui.mol_view,
        MoleculeView::Sticks | MoleculeView::BallAndStick | MoleculeView::SpaceFill
    ) {
        state.ui.mol_view = MoleculeView::BallAndStick
    }

    // Clear all entities for non-editor molecules.
    redraw(
        &mut scene.entities,
        &state.mol_editor.mol,
        &state.ui,
        state.volatile.mol_manip.mode,
    );

    set_static_light(scene, Vec3F32::new_zero(), STATIC_LIGHT_MOL_SIZE);
    set_flashlight(scene);
    engine_updates.entities = EntityUpdate::All;
    engine_updates.lighting = true;
}

// todo: Into a GUI util?
pub fn exit_edit_mode(state: &mut State, scene: &mut Scene, engine_updates: &mut EngineUpdates) {
    state.volatile.operating_mode = OperatingMode::Primary;
    UI_HEIGHT_CHANGED.store(true, Ordering::Release);

    state.mol_editor.md_state = None;
    state.mol_editor.md_running = false;
    state.volatile.mol_editing = None;
    state.volatile.mol_manip.mode = ManipMode::None;

    scene.input_settings.control_scheme = state.volatile.control_scheme_prev;
    scene.camera = state.volatile.primary_mode_cam.clone();

    // Load all primary molecules into the engine.
    draw_peptide(state, scene);
    draw_all_ligs(state, scene);
    draw_all_nucleic_acids(state, scene);
    draw_all_lipids(state, scene);

    set_flashlight(scene);

    engine_updates.entities = EntityUpdate::All;
    engine_updates.lighting = true;
}

// todo: Move to drawing_wrappers?
pub fn redraw(
    entities: &mut Vec<Entity>,
    mol: &MoleculeSmall,
    ui: &StateUi,
    manip_mode: ManipMode,
) {
    entities.clear();

    entities.extend(draw_mol(
        MolGenericRef::Ligand(mol),
        0,
        ui,
        &None,
        manip_mode,
        OperatingMode::MolEditor,
    ));
}

/// Tailored function to prevent having to redraw the whole mol.
fn draw_atom(entities: &mut Vec<Entity>, atom: &Atom, ui: &StateUi) {
    if matches!(ui.mol_view, MoleculeView::BallAndStick) {
        if ui.visibility.hide_hydrogen && atom.element == Hydrogen {
            return;
        }

        let color = atom_color(
            atom,
            0,
            99999,
            &[],
            0,
            &ui.selection,
            ViewSelLevel::Atom, // Always color lipids by atom.
            false,
            ui.res_coloring,
            ui.atom_color_by_charge,
            MolType::Ligand,
        );

        let (radius, mesh) = match ui.mol_view {
            MoleculeView::SpaceFill => (atom.element.vdw_radius(), MESH_SPACEFILL_SPHERE),
            _ => match atom.element {
                Hydrogen => (BALL_STICK_RADIUS_H, MESH_BALL_STICK_SPHERE),
                _ => (BALL_STICK_RADIUS, MESH_BALL_STICK_SPHERE),
            },
        };

        let mut entity = Entity::new(
            mesh,
            // We assume atom.posit is synced with atom_posits here. (Not true generally)
            atom.posit.into(),
            Quaternion::new_identity(),
            radius,
            color,
            ATOM_SHININESS,
        );

        entity.class = EntityClass::Ligand as u32;
        entities.push(entity);
    }
}

/// Tailored function to prevent having to draw the whole mol.
fn draw_bond(
    entities: &mut Vec<Entity>,
    bond: &Bond,
    atoms: &[Atom],
    adj_list: &[Vec<usize>],
    ui: &StateUi,
) {
    // todo: C+P from draw_molecule. With some removed, but much repeated.
    let atom_0 = &atoms[bond.atom_0];
    let atom_1 = &atoms[bond.atom_1];

    if ui.visibility.hide_hydrogen && (atom_0.element == Hydrogen || atom_1.element == Hydrogen) {
        return;
    }

    // We assume atom.posit is synced with atom_posits here. (Not true generally)
    let posit_0: Vec3F32 = atoms[bond.atom_0].posit.into();
    let posit_1: Vec3F32 = atoms[bond.atom_1].posit.into();

    // For determining how to orient multiple-bonds. Only run for relevant bonds to save
    // computation.
    let neighbor_posit = match bond.bond_type {
        BondType::Aromatic | BondType::Double | BondType::Triple => {
            let mut hydrogen_is = Vec::with_capacity(atoms.len());
            for atom in atoms {
                hydrogen_is.push(atom.element == Hydrogen);
            }

            let neighbor_i = find_neighbor_posit(adj_list, bond.atom_0, bond.atom_1, &hydrogen_is);
            match neighbor_i {
                Some((i, p1)) => (atoms[i].posit.into(), p1),
                None => (atoms[0].posit.into(), false),
            }
        }
        _ => (lin_alg::f32::Vec3::new_zero(), false),
    };

    let color_0 = atom_color(
        atom_0,
        0,
        bond.atom_0,
        &[],
        0,
        &ui.selection,
        ViewSelLevel::Atom, // Always color ligands by atom.
        false,
        ui.res_coloring,
        ui.atom_color_by_charge,
        MolType::Ligand,
    );

    let color_1 = atom_color(
        atom_1,
        0,
        bond.atom_1,
        &[],
        0,
        &ui.selection,
        ViewSelLevel::Atom, // Always color ligands by atom.
        false,
        ui.res_coloring,
        ui.atom_color_by_charge,
        MolType::Ligand,
    );

    let to_hydrogen = atom_0.element == Hydrogen || atom_1.element == Hydrogen;

    entities.extend(bond_entities(
        posit_0,
        posit_1,
        color_0,
        color_1,
        bond.bond_type,
        MolType::Ligand,
        true,
        neighbor_posit,
        false,
        to_hydrogen,
    ));
}

/// Save the editor's molecule to disk.
pub fn save(state: &mut State, path: &Path) -> io::Result<()> {
    let mol = MolGenericRef::Ligand(&state.mol_editor.mol);

    let binding = path.extension().unwrap_or_default().to_ascii_lowercase();
    let extension = binding;

    match extension.to_str().unwrap_or_default() {
        "sdf" => mol.to_sdf().save(path)?,
        "mol2" => mol.to_mol2().save(path)?,
        "xyz" => mol.to_xyz().save(path)?,
        "prmtop" => (), // todo
        "pdbqt" => mol.to_pdbqt().save(path)?,
        _ => unimplemented!(),
    }

    println!("Saving editor file!"); // todo tmep
    // todo: A/R
    // state.update_history(path, OpenType::Ligand);
    // // Save the open history.
    // state.update_save_prefs(false);

    Ok(())
}

// todo: I think this approach is wrong. You can add multiple of the same one...
/// This is built from Amber's gaff2.dat. Returns each H FF type that can be bound to a given atom
/// (by force field type), and the bond distance in Å.
/// todo: Can/should we get partial charges too
pub fn hydrogens_avail(ff_type: &Option<String>) -> Vec<(String, f64)> {
    let Some(f) = ff_type else { return Vec::new() };
    match f.as_ref() {
        // Water
        "ow" => vec![("hw".to_owned(), 0.9572)],
        "hw" => vec![("hw".to_owned(), 1.5136)],

        // Generic sp carbon (c )
        "c" => vec![
            ("h4".to_owned(), 1.1123),
            ("h5".to_owned(), 1.1053),
            ("ha".to_owned(), 1.1010),
        ],

        // sp2 carbon families
        "c1" => vec![("ha".to_owned(), 1.0666), ("hc".to_owned(), 1.0600)],
        "c2" => vec![
            ("h4".to_owned(), 1.0865),
            ("h5".to_owned(), 1.0908),
            ("ha".to_owned(), 1.0882),
            ("hc".to_owned(), 1.0870),
            ("hx".to_owned(), 1.0836),
        ],
        "c3" => vec![
            ("h1".to_owned(), 1.0969),
            ("h2".to_owned(), 1.0950),
            ("h3".to_owned(), 1.0938),
            ("hc".to_owned(), 1.0962),
            ("hx".to_owned(), 1.0911),
        ],
        "c5" => vec![
            ("h1".to_owned(), 1.0972),
            ("h2".to_owned(), 1.0955),
            ("h3".to_owned(), 1.0958),
            ("hc".to_owned(), 1.0954),
            ("hx".to_owned(), 1.0917),
        ],
        "c6" => vec![
            ("h1".to_owned(), 1.0984),
            ("h2".to_owned(), 1.0985),
            ("h3".to_owned(), 1.0958),
            ("hc".to_owned(), 1.0979),
            ("hx".to_owned(), 1.0931),
        ],

        // Aromatic/condensed ring carbons
        "ca" => vec![
            ("ha".to_owned(), 1.0860),
            ("h4".to_owned(), 1.0885),
            ("h5".to_owned(), 1.0880),
        ],
        "cc" => vec![
            ("h4".to_owned(), 1.0809),
            ("h5".to_owned(), 1.0820),
            ("ha".to_owned(), 1.0838),
            ("hx".to_owned(), 1.0827),
        ],
        "cd" => vec![
            ("h4".to_owned(), 1.0818),
            ("h5".to_owned(), 1.0821),
            ("ha".to_owned(), 1.0835),
            ("hx".to_owned(), 1.0801),
        ],
        "ce" => vec![
            ("h4".to_owned(), 1.0914),
            ("h5".to_owned(), 1.0895),
            ("ha".to_owned(), 1.0880),
        ],
        "cf" => vec![
            ("h4".to_owned(), 1.0942),
            ("ha".to_owned(), 1.0885),
            // table also lists h5-cf (reverse order) at 1.0890
            ("h5".to_owned(), 1.0890),
        ],
        "cg" => Vec::new(), // no H entries shown for cg in the provided snippet

        // Other carbon families frequently seen
        "cu" => vec![("ha".to_owned(), 1.0786)],
        "cv" => vec![("ha".to_owned(), 1.0878)],
        "cx" => vec![
            ("h1".to_owned(), 1.0888),
            ("h2".to_owned(), 1.0869),
            ("hc".to_owned(), 1.0865),
            ("hx".to_owned(), 1.0849),
        ],
        "cy" => vec![
            ("h1".to_owned(), 1.0946),
            ("h2".to_owned(), 1.0930),
            ("hc".to_owned(), 1.0947),
            ("hx".to_owned(), 1.0913),
        ],

        // Nitrogen families: protonated H type is "hn"
        "n1" => vec![("hn".to_owned(), 0.9860)],
        "n2" => vec![("hn".to_owned(), 1.0221)],
        "n3" => vec![("hn".to_owned(), 1.0190)],
        "n4" => vec![("hn".to_owned(), 1.0300)],
        "n" => vec![("hn".to_owned(), 1.0130)],
        "n5" => vec![("hn".to_owned(), 1.0211)],
        "n6" => vec![("hn".to_owned(), 1.0183)],
        "n7" => vec![("hn".to_owned(), 1.0195)],
        "n8" => vec![("hn".to_owned(), 1.0192)],
        "n9" => vec![("hn".to_owned(), 1.0192)],
        "na" => vec![("hn".to_owned(), 1.0095)],
        "nh" => vec![("hn".to_owned(), 1.0120)],
        "nj" => vec![("hn".to_owned(), 1.0130)],
        "nl" => vec![("hn".to_owned(), 1.0476)],
        "no" => vec![("hn".to_owned(), 1.0440)],
        "np" => vec![("hn".to_owned(), 1.0210)],
        "nq" => vec![("hn".to_owned(), 1.0180)],
        "ns" => vec![("hn".to_owned(), 1.0132)],
        "nt" => vec![("hn".to_owned(), 1.0105)],
        "nu" => vec![("hn".to_owned(), 1.0137)],
        "nv" => vec![("hn".to_owned(), 1.0114)],
        "nx" => vec![("hn".to_owned(), 1.0338)],
        "ny" => vec![("hn".to_owned(), 1.0339)],
        "nz" => vec![("hn".to_owned(), 1.0271)],

        // Oxygen families: hydroxyl H type is "ho"
        "o" => vec![("ho".to_owned(), 0.9810)],
        "oh" => vec![("ho".to_owned(), 0.9725)],

        // Sulfur families: thiol H type is "hs"
        "s" => vec![("hs".to_owned(), 1.3530)],
        "s4" => vec![("hs".to_owned(), 1.3928)],
        "s6" => vec![("hs".to_owned(), 1.3709)],
        "sh" => vec![("hs".to_owned(), 1.3503)],
        "sy" => vec![("hs".to_owned(), 1.3716)],

        // Phosphorus families: acidic phosphate H type is "hp"
        "p2" => vec![("hp".to_owned(), 1.4272)],
        "p3" => vec![("hp".to_owned(), 1.4256)],
        "p4" => vec![("hp".to_owned(), 1.4271)],
        "p5" => vec![("hp".to_owned(), 1.4205)],
        "py" => vec![("hp".to_owned(), 1.4150)],

        _ => Vec::new(),
    }
}

/// Set up MD for the editor's molecule.
pub(super) fn build_dynamics(
    dev: &ComputationDevice,
    editor: &mut MolEditorState,
    param_set: &FfParamSet,
    cfg: &MdConfig,
) -> Result<MdState, ParamError> {
    println!("Setting up dynamics for the mol editor...");

    editor.rebuild_ff_related(param_set);

    let atoms_gen: Vec<_> = editor
        .mol
        .common
        .atoms
        .iter()
        .map(|a| a.to_generic())
        .collect();

    let bonds_gen: Vec<_> = editor
        .mol
        .common
        .bonds
        .iter()
        .map(|b| b.to_generic())
        .collect();

    let mols = vec![MolDynamics {
        ff_mol_type: FfMolType::SmallOrganic,
        atoms: atoms_gen,
        atom_posits: Some(editor.mol.common.atom_posits.clone()),
        atom_init_velocities: None,
        bonds: bonds_gen,
        adjacency_list: Some(editor.mol.common.adjacency_list.clone()),
        static_: false,
        bonded_only: false,
        mol_specific_params: Some(editor.mol_specific_params.clone()),
    }];

    let mut cfg = MdConfig {
        max_init_relaxation_iters: Some(50), // todo A/R
        overrides: MdOverrides {
            // todo: Reduced number of water relax steps to make it faster?
            // Water relaxation is slow.
            // skip_water_relaxation: true,
            long_range_recip_disabled: true,
            ..Default::default()
        },
        // todo: Which one?
        integrator: Integrator::VerletVelocity {
            // todo: Experimenting/troubleshooting. Temp is out of control.
            thermostat: Some(TAU_TEMP_DEFAULT * 0.1),
        },
        // We run slower time steps than in typical MD steps here, so this is OK, and may
        // provide a more realistic visualization.
        hydrogen_constraint: HydrogenConstraint::Flexible,
        ..cfg.clone()
    };

    println!("Initializing MD state...");

    // Workaround to now have to regen water. I call this a workaround as there can be other
    // elegant approaches in the Dynamics lib, and or that handle vacancies better from changing
    // atom counts. Or that applies automatically. Pre-arranged water is a good way to start, e.g.
    // instead of in a lattice.
    // let mut water_prev = None;
    //
    // if let Some(md_prev) = &editor.md_state {
    //     if !md_prev.water.is_empty() {
    //         water_prev = Some(md_prev.water.clone());
    //         cfg.overrides.skip_water = true;
    //     }
    // }

    let md_state = MdState::new(dev, &cfg, &mols, param_set)?;

    // if let Some(w) = water_prev {
    //     println!("Using previous water molecules");
    //     // todo: Dangerous: Could cause an overlap and therefor blowup.
    //     // md_state.water = w;
    // }

    println!("MD init done.");

    Ok(md_state)
}

/// Used to share this between GUI and inputs.
pub fn sync_md(state: &mut State) {
    if state.mol_editor.md_running {
        // todo: Ideally don't rebuild the whole dynamics, for performance reasons.
        match build_dynamics(
            &state.dev,
            &mut state.mol_editor,
            &state.ff_param_set,
            &state.to_save.md_config,
        ) {
            Ok(d) => state.mol_editor.md_state = Some(d),
            Err(e) => eprintln!("Problem setting up dynamics for the editor: {e:?}"),
        }
    } else {
        // The MD build Will be triggered next time MD is started.
        state.mol_editor.md_rebuild_required = true;
        state.mol_editor.rebuild_ff_related(&state.ff_param_set);
    }
}
