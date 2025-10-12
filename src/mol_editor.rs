use std::{collections::HashMap, io, io::ErrorKind, path::Path};

use bio_files::{BondType, create_bonds};
use graphics::{ControlScheme, EngineUpdates, Entity, EntityUpdate, Scene};
use lin_alg::{
    f32::{Quaternion, Vec3 as Vec3F32},
    f64::Vec3,
};
use na_seq::{AtomTypeInRes, Element, Element::Carbon};

use crate::{
    ManipMode, OperatingMode, Selection, State, StateUi,
    drawing::{EntityClass, draw_mol, draw_peptide},
    drawing_wrappers::{draw_all_ligs, draw_all_lipids, draw_all_nucleic_acids},
    mol_lig::{Ligand, MoleculeSmall},
    molecule::{Atom, Bond, MolType, MoleculeCommon, MoleculeGenericRef},
    render::set_flashlight,
};

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
    state.volatile.operating_mode_prev = OperatingMode::Primary;

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
    state.volatile.operating_mode_prev = OperatingMode::MolEditor;

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
    ));
}
