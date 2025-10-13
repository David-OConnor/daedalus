use std::{collections::HashMap, io, io::ErrorKind, path::Path};
use std::sync::atomic::Ordering;
use bio_files::{BondType, create_bonds};
use dynamics::find_tetra_posits;
use graphics::{ControlScheme, EngineUpdates, Entity, EntityUpdate, Scene};
use lin_alg::{
    f32::{Quaternion, Vec3 as Vec3F32},
    f64::Vec3,
};
use na_seq::{
    AtomTypeInRes, Element,
    Element::{Carbon, Hydrogen},
};

use crate::{
    ManipMode, OperatingMode, Selection, State, StateUi, ViewSelLevel,
    drawing::{
        COLOR_SELECTED, EntityClass, MESH_BALL_STICK_SPHERE, MESH_SPACEFILL_SPHERE, MoleculeView,
        atom_color, bond_entities, draw_mol, draw_peptide,
    },
    drawing_wrappers::{draw_all_ligs, draw_all_lipids, draw_all_nucleic_acids},
    mol_lig::{Ligand, MoleculeSmall},
    molecule::{Atom, Bond, MolType, MoleculeCommon, MoleculeGenericRef},
    render::{ATOM_SHININESS, BALL_STICK_RADIUS, BALL_STICK_RADIUS_H, set_flashlight},
    util::find_neighbor_posit,
};
use crate::ui::UI_HEIGHT_CHANGED;

pub const INIT_CAM_DIST: f32 = 20.;

/// For editing small organic molecules.
#[derive(Debug, Default)]
pub struct MolEditorState {
    pub mol: MoleculeSmall,
    // atoms: Vec<Atom>,
}

impl MolEditorState {
    /// For now, sets up a pair of single-bonded carbon atoms.
    pub fn clear_mol(&mut self) {
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
    }

    pub fn load_mol(&mut self, mol: &MoleculeCommon) {
        self.mol.common = mol.clone();

        // We assign H dynamically; ignore present ones.
        self.mol.common.atoms = mol
            .atoms
            .iter()
            .filter(|a| a.element != Element::Hydrogen)
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

    match state.volatile.active_mol {
        Some((mol_type, i)) => {
            if mol_type == MolType::Ligand {
                state.mol_editor.load_mol(&state.ligands[i].common);
            } else {
                state.mol_editor.clear_mol();
            }
        }
        None => {
            println!("Clearing mol");
            state.mol_editor.clear_mol();
        }
    }

    state.volatile.control_scheme_prev = scene.input_settings.control_scheme;
    scene.input_settings.control_scheme = ControlScheme::Arc {
        center: Vec3F32::new_zero(),
    };

    state.volatile.primary_mode_cam = scene.camera.clone();
    scene.camera.position = Vec3F32::new(0., 0., -INIT_CAM_DIST);
    scene.camera.orientation = Quaternion::new_identity();

    // Clear all entities for non-editor molecules.
    redraw(&mut scene.entities, &state.mol_editor.mol, &state.ui);

    set_flashlight(scene);
    engine_updates.entities = EntityUpdate::All;
    engine_updates.lighting = true;
}

// todo: Into a GUI util?
pub fn exit_edit_mode(state: &mut State, scene: &mut Scene, engine_updates: &mut EngineUpdates) {
    state.volatile.operating_mode = OperatingMode::Primary;
    UI_HEIGHT_CHANGED.store(true, Ordering::Release);

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
        MoleculeGenericRef::Ligand(mol),
        0,
        ui,
        &None,
        ManipMode::None,
        OperatingMode::MolEditor,
    ));
}

pub fn add_atom(
    entities: &mut Vec<Entity>,
    mol: &mut MoleculeSmall,
    element: Element,
    ui: &mut StateUi,
    updates: &mut EngineUpdates,
) {
    let Selection::AtomLig((_, i)) = ui.selection else {
        eprintln!("Attempting to add an atom with no parent to add it to");
        return;
    };

    let posit_parent = &mol.common.atom_posits[i];

    let mut neighbor_count = 0;
    for j in &mol.common.adjacency_list[i] {
        if mol.common.atoms[*j].element != Hydrogen {
            neighbor_count += 1;
        }
    }

    // todo: for now hard-coding tetra
    let posit = match neighbor_count {
        0 => Vec3::new(1.3, 0., 0.),
        1 => {
            let adj = mol.common.adjacency_list[i][0];
            let neighbor = mol.common.atoms[adj].posit;
            find_tetra_posits(*posit_parent, neighbor, Vec3::new_zero())
        }
        2 => {
            // If the incoming angles are ~τ/3, add in a planar config.

            // todo: Hmm. Need a better tetra fn.
            let adj_0 = mol.common.adjacency_list[i][0];
            let neighbor_0 = mol.common.atoms[adj_0].posit;
            let adj_1 = mol.common.adjacency_list[i][1];
            let neighbor_1 = mol.common.atoms[adj_1].posit;
            find_tetra_posits(*posit_parent, neighbor_0, neighbor_1)
        }
        3 => {
            // todo: Hmm. Need a better tetra fn.
            let adj_0 = mol.common.adjacency_list[i][0];
            let neighbor_0 = mol.common.atoms[adj_0].posit;
            let adj_1 = mol.common.adjacency_list[i][1];
            let neighbor_1 = mol.common.atoms[adj_1].posit;
            find_tetra_posits(*posit_parent, neighbor_0, neighbor_1)
        }
        _ => {
            return;
        }
    };

    let new_sn = 0; // todo A/R
    let new_i = mol.common.atoms.len();

    mol.common.atoms.push(Atom {
        serial_number: new_sn,
        posit,
        element,
        type_in_res: None,
        force_field_type: Some("ca".to_owned()), // todo: A/R
        partial_charge: Some(0.),                // todo: A/R,
        ..Default::default()
    });

    mol.common.bonds.push(Bond {
        bond_type: BondType::Single,
        atom_0_sn: mol.common.atoms[i].serial_number,
        atom_1_sn: new_sn,
        atom_0: i,
        atom_1: new_i,
        is_backbone: false,
    });

    mol.common.atom_posits.push(posit);

    mol.common.adjacency_list[i].push(new_i);
    mol.common.adjacency_list.push(vec![i]);

    draw_atom(entities, &mol.common.atoms[mol.common.atoms.len() - 1], ui);
    draw_bond(
        entities,
        &mol.common.bonds[mol.common.bonds.len() - 1],
        &mol.common.atoms,
        &mol.common.adjacency_list,
        ui,
    );

    // todo: Ideally just add the single entity, and add it to the
    // index buffer.
    updates.entities = EntityUpdate::All;
}

/// Tailored function to prevent having to redraw the whole mol.
fn draw_atom(entities: &mut Vec<Entity>, atom: &Atom, ui: &StateUi) {
    if matches!(ui.mol_view, MoleculeView::BallAndStick) {
        if ui.visibility.hide_hydrogen && atom.element == Element::Hydrogen {
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
                Element::Hydrogen => (BALL_STICK_RADIUS_H, MESH_BALL_STICK_SPHERE),
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

    if ui.visibility.hide_hydrogen
        && (atom_0.element == Element::Hydrogen || atom_1.element == Element::Hydrogen)
    {
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
                hydrogen_is.push(atom.element == Element::Hydrogen);
            }

            let neighbor_i = find_neighbor_posit(adj_list, bond.atom_0, bond.atom_1, &hydrogen_is);
            match neighbor_i {
                Some((i, p1)) => (atoms[i].posit.into(), p1),
                None => (atoms[0].posit.into(), false),
            }
        }
        _ => (lin_alg::f32::Vec3::new_zero(), false),
    };

    let color_0 = crate::drawing::atom_color(
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

    let color_1 = crate::drawing::atom_color(
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

    let to_hydrogen = atom_0.element == Element::Hydrogen || atom_1.element == Element::Hydrogen;

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
