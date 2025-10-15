pub mod add_atoms;

use std::{
    collections::HashMap,
    io,
    io::ErrorKind,
    path::Path,
    sync::atomic::{AtomicU32, Ordering},
};

use bio_files::{BondType, Mol2, Pdbqt, Sdf, md_params::ForceFieldParams};
use dynamics::{
    ComputationDevice, FfMolType, MdConfig, MdState, MolDynamics, ParamError, params::FfParamSet,
};
use graphics::{ControlScheme, EngineUpdates, Entity, EntityUpdate, Scene};
use lin_alg::{
    f32::{Quaternion, Vec3 as Vec3F32},
    f64::Vec3,
};
use na_seq::{
    AtomTypeInRes,
    Element::{Carbon, Hydrogen, Oxygen},
};

use crate::{
    ManipMode, OperatingMode, State, StateUi, ViewSelLevel,
    drawing::{
        EntityClass, MESH_BALL_STICK_SPHERE, MESH_SPACEFILL_SPHERE, MoleculeView, atom_color,
        bond_entities, draw_mol, draw_peptide,
    },
    drawing_wrappers::{draw_all_ligs, draw_all_lipids, draw_all_nucleic_acids},
    mol_lig::MoleculeSmall,
    molecule::{Atom, Bond, MolGenericRef, MolType, MoleculeCommon},
    render::{
        ATOM_SHININESS, BALL_STICK_RADIUS, BALL_STICK_RADIUS_H, set_flashlight, set_static_light,
    },
    ui::UI_HEIGHT_CHANGED,
    util::find_neighbor_posit,
};

pub const INIT_CAM_DIST: f32 = 20.;

// Set a higher value to place the light farther away. (More uniform, dimmer lighting)
pub const STATIC_LIGHT_MOL_SIZE: f32 = 500.;

static NEXT_ATOM_SN: AtomicU32 = AtomicU32::new(0);

/// For editing small organic molecules.
#[derive(Default)]
pub struct MolEditorState {
    pub mol: MoleculeSmall,
    pub md_state: Option<MdState>,
    pub dt: f32, // ps.
}

impl MolEditorState {
    /// For now, sets up a pair of single-bonded carbon atoms.
    pub fn clear_mol(
        &mut self,
        dev: &ComputationDevice,
        param_set: &FfParamSet,
        md_cfg: &MdConfig,
    ) {
        // todo: Change this dist; rough start.
        const DIST: f64 = 1.3;

        self.mol.common.atoms = vec![
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

        self.mol.common.bonds = vec![Bond {
            bond_type: BondType::Single,
            atom_0_sn: 1,
            atom_1_sn: 2,
            atom_0: 0,
            atom_1: 1,
            is_backbone: false,
        }];

        self.mol.common.atom_posits = self.mol.common.atoms.iter().map(|a| a.posit).collect();
        self.mol.common.build_adjacency_list();

        match build_dynamics(
            dev,
            &self.mol,
            param_set,
            &HashMap::new(), // todo: A/R
            md_cfg,
        ) {
            Ok(d) => self.md_state = Some(d),
            Err(e) => eprintln!("Problem setting up dynamics: {e:?}"),
        }
    }

    /// A simplified variant of our primary `open_molecule` function.
    pub fn open_molecule(
        &mut self,
        dev: &ComputationDevice,
        param_set: &FfParamSet,
        md_cfg: &MdConfig,
        path: &Path,
        scene: &mut Scene,
        engine_updates: &mut EngineUpdates,
        state_ui: &mut StateUi,
    ) -> io::Result<()> {
        let binding = path.extension().unwrap_or_default().to_ascii_lowercase();
        let extension = binding;

        let molecule = match extension.to_str().unwrap() {
            "sdf" => {
                let mut m: MoleculeSmall = Sdf::load(path)?.try_into()?;
                m.common.path = Some(path.to_owned());
                m
            }
            "mol2" => {
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
            dev,
            &molecule.common,
            param_set,
            md_cfg,
            scene,
            engine_updates,
            state_ui,
        );
        Ok(())
    }

    pub fn load_mol(
        &mut self,
        dev: &ComputationDevice,
        mol: &MoleculeCommon,
        param_set: &FfParamSet,
        md_cfg: &MdConfig,
        scene: &mut Scene,
        engine_updates: &mut EngineUpdates,
        state_ui: &mut StateUi,
    ) {
        self.mol.common = mol.clone();

        // We assign H dynamically; ignore present ones.
        self.mol.common.atoms = mol
            .atoms
            .iter()
            .filter(|a| a.element != Hydrogen)
            .map(|a| a.clone())
            .collect();

        // Remove bonds to atoms that no longer exist, and change indices otherwise:
        // serial_number -> new index after filtering
        let sn2idx: HashMap<u32, usize> = self
            .mol
            .common
            .atoms
            .iter()
            .enumerate()
            .map(|(i, a)| (a.serial_number, i))
            .collect();

        // Keep only bonds whose endpoints still exist; reindex to new atom indices
        self.mol.common.bonds = mol
            .bonds
            .iter()
            .filter_map(|b| {
                let i0 = sn2idx.get(&b.atom_0_sn)?;
                let i1 = sn2idx.get(&b.atom_1_sn)?;
                Some(Bond {
                    bond_type: b.bond_type,
                    atom_0_sn: b.atom_0_sn,
                    atom_1_sn: b.atom_1_sn,
                    atom_0: *i0,
                    atom_1: *i1,
                    is_backbone: b.is_backbone,
                })
            })
            .collect();

        // Rebuild these based on the new filters.
        self.mol.common.atom_posits = self.mol.common.atoms.iter().map(|a| a.posit).collect();
        self.mol.common.build_adjacency_list();

        // Re-populate hydrogens algorithmically. This assumes we trust our algorithm more than the
        // initial molecule, which may or may not be true.
        for (i, atom) in self.mol.common.atoms.clone().iter().enumerate() {
            // todo. Don't clone!!! Find a better way to fix the borrow error.

            let mut skip = false;
            for bonded_i in &self.mol.common.adjacency_list[i] {
                // Don't add H to oxygens double-bonded.
                if self.mol.common.atoms[i].element == Oxygen {
                    for bond in &self.mol.common.bonds {
                        if bond.atom_0 == i && bond.atom_1 == *bonded_i
                            || bond.atom_1 == i && bond.atom_0 == *bonded_i
                        {
                            if matches!(bond.bond_type, BondType::Double) {
                                println!("FOUND IT!: {:?}", i);
                                skip = true;
                                break;
                            }
                        }
                    }
                }
            }

            if !skip {
                for (ff_type, bond_len) in hydrogens_avail(&atom.force_field_type) {
                    add_atoms::add_atom(
                        self,
                        &mut scene.entities,
                        i,
                        Hydrogen,
                        BondType::Single,
                        Some(ff_type),
                        Some(bond_len),
                        state_ui,
                        engine_updates,
                    )
                }
            }
        }

        let mut highest_sn = 0;
        for atom in &self.mol.common.atoms {
            if atom.serial_number > highest_sn {
                highest_sn = atom.serial_number;
            }
        }
        NEXT_ATOM_SN.store(highest_sn + 1, Ordering::Release);

        // Clear all entities for non-editor molecules.
        redraw(&mut scene.entities, &self.mol, state_ui);

        set_flashlight(scene);
        engine_updates.entities = EntityUpdate::All;
        engine_updates.lighting = true;

        match build_dynamics(
            dev,
            &self.mol,
            param_set,
            &HashMap::new(), // todo: A/R
            md_cfg,
        ) {
            Ok(d) => self.md_state = Some(d),
            Err(e) => eprintln!("Problem setting up dynamics: {e:?}"),
        }
    }

    pub fn delete_atom(&mut self, i: usize) -> io::Result<()> {
        if i >= self.mol.common.atoms.len() {
            return Err(io::Error::new(ErrorKind::InvalidData, "Out of range"));
        }

        self.mol.common.atoms.remove(i);
        self.mol.common.atom_posits.remove(i);

        // Drop bonds that referenced the removed atom
        self.mol
            .common
            .bonds
            .retain(|b| b.atom_0 != i && b.atom_1 != i);

        // Reindex remaining bonds (atom indices shift down after removal)
        for b in &mut self.mol.common.bonds {
            if b.atom_0 > i {
                b.atom_0 -= 1;
            }
            if b.atom_1 > i {
                b.atom_1 -= 1;
            }
        }

        for adj in &mut self.mol.common.adjacency_list {
            adj.retain(|&j| j != i);

            for j in adj.iter_mut() {
                if *j > i {
                    *j -= 1;
                }
            }
        }

        Ok(())
    }

    pub fn save_mol2(&self, path: &Path) -> io::Result<()> {
        Ok(())
    }

    pub fn save_sdf(&self, path: &Path) -> io::Result<()> {
        Ok(())
    }
}

pub mod templates {
    use bio_files::BondType;
    use lin_alg::f64::Vec3;
    use na_seq::{
        AtomTypeInRes,
        Element::{self, Carbon, Hydrogen, Oxygen},
    };

    use crate::molecule::{Atom, Bond};

    // todo: What does posit anchor too? Center? An corner marked in a certain way?
    pub fn cooh_group(anchor: Vec3, starting_sn: u32) -> (Vec<Atom>, Vec<Bond>) {
        const POSITS: [Vec3; 3] = [
            Vec3::new(0.0000, 0.0000, 0.0), // C (carboxyl)
            Vec3::new(1.2290, 0.0000, 0.0), // O (carbonyl)
            Vec3::new(-0.6715, 1.1645, 0.0), // O (hydroxyl)
                                            // Vec3::new(-1.0286, 1.7826, 0.0), // H (hydroxyl)
        ];

        // todo: Skip the H.
        // const ELEMENTS: [Element; 4] = [Carbon, Oxygen, Oxygen, Hydrogen];
        const ELEMENTS: [Element; 4] = [Carbon, Oxygen, Oxygen, Hydrogen];
        const FF_TYPES: [&str; 4] = ["c", "o", "oh", "ho"]; // GAFF2-style
        const CHARGES: [f32; 4] = [0.70, -0.55, -0.61, 0.44]; // todo: A/R

        let posits = POSITS.iter().map(|p| *p + anchor);

        let mut atoms = Vec::with_capacity(3);
        let mut bonds = Vec::with_capacity(3);

        for (i, posit) in posits.enumerate() {
            let serial_number = starting_sn + i as u32;

            atoms.push(Atom {
                serial_number,
                posit,
                element: ELEMENTS[i],
                type_in_res: None, // todo: no; fix this
                force_field_type: Some(FF_TYPES[i].to_owned()), // todo: A/R
                partial_charge: Some(CHARGES[i]), // todo: A/R,
                ..Default::default()
            })
        }

        bonds.push(Bond {
            bond_type: BondType::Double,
            atom_0_sn: atoms[0].serial_number,
            atom_1_sn: atoms[1].serial_number,
            atom_0: 0,
            atom_1: 1,
            is_backbone: false,
        });
        bonds.push(Bond {
            bond_type: BondType::Single,
            atom_0_sn: atoms[1].serial_number,
            atom_1_sn: atoms[2].serial_number,
            atom_0: 1,
            atom_1: 2,
            is_backbone: false,
        });

        (atoms, bonds)
    }

    // todo: What does posit anchor too? Center? An corner marked in a certain way?
    pub fn benzene_ring(anchor: Vec3, starting_sn: u32) -> (Vec<Atom>, Vec<Bond>) {
        const POSITS: [Vec3; 6] = [
            Vec3::new(1.3970, 0.0000, 0.0),
            Vec3::new(0.6985, 1.2090, 0.0),
            Vec3::new(-0.6985, 1.2090, 0.0),
            Vec3::new(-1.3970, 0.0000, 0.0),
            Vec3::new(-0.6985, -1.2090, 0.0),
            Vec3::new(0.6985, -1.2090, 0.0),
        ];

        let posits = POSITS.iter().map(|p| *p + anchor);

        let mut atoms = Vec::with_capacity(6);
        let mut bonds = Vec::with_capacity(6);

        for (i, posit) in posits.enumerate() {
            let serial_number = starting_sn + i as u32;

            atoms.push(Atom {
                serial_number,
                posit,
                element: Carbon,
                type_in_res: Some(AtomTypeInRes::CA), // todo: A/R
                force_field_type: Some("ca".to_owned()), // todo: A/R
                partial_charge: Some(-0.115),         // tood: Ar. -0.06 - 0.012 etc.
                ..Default::default()
            })
        }

        for i in 0..6 {
            let i_next = i % 6; // Wrap 6 to 0.
            bonds.push(Bond {
                bond_type: BondType::Aromatic,
                atom_0_sn: atoms[i].serial_number,
                atom_1_sn: atoms[i_next].serial_number,
                atom_0: i,
                atom_1: i_next,
                is_backbone: false,
            });
        }

        (atoms, bonds)
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

    if let Some((mol_type, i)) = state.volatile.active_mol {
        if mol_type == MolType::Ligand {
            if i >= state.ligands.len() {
                eprintln!(
                    "Expected a ligand at this index, but out of bounds when entering edit mode"
                );
            } else {
                state.mol_editor.load_mol(
                    &state.dev,
                    &state.ligands[i].common,
                    &state.ff_param_set,
                    &state.to_save.md_config,
                    scene,
                    engine_updates,
                    &mut state.ui,
                );
                mol_loaded = true;
            }
        }
    }

    if !mol_loaded {
        state
            .mol_editor
            .clear_mol(&state.dev, &state.ff_param_set, &state.to_save.md_config);
    }

    state.volatile.control_scheme_prev = scene.input_settings.control_scheme;
    scene.input_settings.control_scheme = ControlScheme::Arc {
        center: Vec3F32::new_zero(),
    };

    state.volatile.primary_mode_cam = scene.camera.clone();
    scene.camera.position = Vec3F32::new(0., 0., -INIT_CAM_DIST);
    scene.camera.orientation = Quaternion::new_identity();

    // Set to a view supported by the editor.
    // todo: In this case, store the previous view, and re-set it upon exiting the editor.
    if !matches!(
        state.ui.mol_view,
        MoleculeView::Sticks | MoleculeView::BallAndStick | MoleculeView::SpaceFill
    ) {
        state.ui.mol_view = MoleculeView::BallAndStick
    }

    // Clear all entities for non-editor molecules.
    redraw(&mut scene.entities, &state.mol_editor.mol, &state.ui);

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

    // todo: Not necessarily zero!
    scene.input_settings.control_scheme = state.volatile.control_scheme_prev;

    // Load all primary molecules into the engine.
    draw_peptide(state, scene);
    draw_all_ligs(state, scene);
    draw_all_nucleic_acids(state, scene);
    draw_all_lipids(state, scene);

    scene.camera = state.volatile.primary_mode_cam.clone();

    set_flashlight(scene);
    engine_updates.entities = EntityUpdate::All;
    engine_updates.lighting = true;
}

// todo: Move to drawing_wrappers?
pub fn redraw(entities: &mut Vec<Entity>, mol: &MoleculeSmall, ui: &StateUi) {
    *entities = Vec::new();

    entities.extend(draw_mol(
        MolGenericRef::Ligand(mol),
        0,
        ui,
        &None,
        ManipMode::None,
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
            ui.res_color_by_index,
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
        ui.res_color_by_index,
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
        ui.res_color_by_index,
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
fn build_dynamics(
    dev: &ComputationDevice,
    mol: &MoleculeSmall,
    param_set: &FfParamSet,
    mol_specific_params: &HashMap<String, ForceFieldParams>,
    cfg: &MdConfig,
) -> Result<MdState, ParamError> {
    println!("Setting up dynamics for the mol editor...");

    let atoms_gen: Vec<_> = mol.common.atoms.iter().map(|a| a.to_generic()).collect();
    let bonds_gen: Vec<_> = mol.common.bonds.iter().map(|b| b.to_generic()).collect();

    let mols = vec![MolDynamics {
        ff_mol_type: FfMolType::SmallOrganic,
        atoms: atoms_gen,
        atom_posits: Some(mol.common.atom_posits.clone()),
        bonds: bonds_gen,
        adjacency_list: Some(mol.common.adjacency_list.clone()),
        static_: false,
        bonded_only: false,
        // mol_specific_params: Some(msp.clone()),
        mol_specific_params: None,
    }];

    let cfg = MdConfig {
        max_init_relaxation_iters: None,
        allow_missing_dihedral_params: true,
        ..cfg.clone()
    };

    println!("Initializing MD state...");
    let md_state = MdState::new(dev, &cfg, &mols, param_set)?;
    println!("Done.");

    Ok(md_state)
}
