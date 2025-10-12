use std::path::Path;

use graphics::{ControlScheme, EngineUpdates, EntityUpdate, Scene};
use lin_alg::{
    f32::{Quaternion, Vec3 as Vec3F32},
    f64::Vec3,
};
use na_seq::{AtomTypeInRes, Element, Element::Carbon};

use crate::{
    ManipMode, OperatingMode, State, StateUi,
    drawing::{draw_mol, draw_peptide},
    drawing_wrappers::{draw_all_ligs, draw_all_lipids, draw_all_nucleic_acids},
    mol_lig::{Ligand, MoleculeSmall},
    molecule::{Atom, MolType, MoleculeCommon, MoleculeGenericRef},
};

/// For editing small organic molecules.
pub struct MolEditorState {
    mol: MoleculeSmall, // todo: A/R re this and atoms.
    atoms: Vec<Atom>,
}

impl MolEditorState {
    /// For now, sets up a pair of single-bonded carbon atoms.
    pub fn clear_mol(&mut self) {
        // todo: Change this dist; rough start.
        const DIST: f64 = 1.3;

        self.atoms = vec![
            Atom {
                serial_number: 0,
                posit: Vec3::new_zero(),
                element: Carbon,
                type_in_res: Some(AtomTypeInRes::C), // todo: no; fix this
                force_field_type: Some("ca".to_owned()), // todo: A/R
                partial_charge: Some(0.),            // todo: A/R,
                ..Default::default()
            },
            Atom {
                serial_number: 1,
                posit: Vec3::new(DIST, 0., 0.),
                element: Carbon,
                type_in_res: Some(AtomTypeInRes::C), // todo: no; fix this
                force_field_type: Some("ca".to_owned()), // todo: A/R
                partial_charge: Some(0.),            // todo: A/R,
                ..Default::default()
            },
        ];
    }

    pub fn load_mol(&mut self, mol: &MoleculeCommon) {
        // We assign H dynamically; ignore present ones.

        let atoms: Vec<_> = mol
            .atoms
            .iter()
            .filter(|a| a.element != Element::Hydrogen)
            .map(|a| a.clone())
            .collect();

        self.atoms = atoms;
    }

    pub fn save_mol2(&self, path: &Path) -> Result<(), std::io::Error> {
        Ok(())
    }

    pub fn save_sdf(&self, path: &Path) -> Result<(), std::io::Error> {
        Ok(())
    }
}

pub mod templates {
    use lin_alg::f64::Vec3;
    use na_seq::{
        AtomTypeInRes,
        Element::{self, Carbon, Oxygen},
    };

    use crate::molecule::Atom;

    // todo: What does posit anchor too? Center? An corner marked in a certain way?
    pub fn cooh_group(anchor: Vec3, starting_sn: u32) -> Vec<Atom> {
        const POSITS: [Vec3; 3] = [
            Vec3::new(0.0000, 0.0000, 0.0), // C (carboxyl)
            Vec3::new(1.2290, 0.0000, 0.0), // O (carbonyl)
            Vec3::new(-0.6715, 1.1645, 0.0), // O (hydroxyl)
                                            // Vec3::new(-1.0286, 1.7826, 0.0), // H (hydroxyl)
        ];

        // todo: Skip the H.
        // const ELEMENTS: [Element; 4] = [Carbon, Oxygen, Oxygen, Hydrogen];
        const ELEMENTS: [Element; 3] = [Carbon, Oxygen, Oxygen];
        const FF_TYPES: [&str; 4] = ["c", "o", "oh", "ho"]; // GAFF2-style
        const CHARGES: [f32; 4] = [0.70, -0.55, -0.61, 0.44]; // todo: A/R

        let posits = POSITS.iter().map(|p| *p + anchor);

        let mut result = Vec::new();

        for (i, posit) in posits.enumerate() {
            let serial_number = starting_sn + i as u32;

            result.push(Atom {
                serial_number,
                posit,
                element: ELEMENTS[i],
                type_in_res: Some(AtomTypeInRes::CA), // todo: no; fix this
                force_field_type: Some(FF_TYPES[i].to_owned()), // todo: A/R
                partial_charge: Some(CHARGES[i]),     // todo: A/R,
                ..Default::default()
            })
        }

        result
    }

    // todo: What does posit anchor too? Center? An corner marked in a certain way?
    pub fn benzene_ring(anchor: Vec3, starting_sn: u32) -> Vec<Atom> {
        const POSITS: [Vec3; 6] = [
            Vec3::new(1.3970, 0.0000, 0.0),
            Vec3::new(0.6985, 1.2090, 0.0),
            Vec3::new(-0.6985, 1.2090, 0.0),
            Vec3::new(-1.3970, 0.0000, 0.0),
            Vec3::new(-0.6985, -1.2090, 0.0),
            Vec3::new(0.6985, -1.2090, 0.0),
        ];

        let posits = POSITS.iter().map(|p| *p + anchor);

        let mut result = Vec::new();

        for (i, posit) in posits.enumerate() {
            let serial_number = starting_sn + i as u32;

            result.push(Atom {
                serial_number,
                posit,
                element: Carbon,
                type_in_res: Some(AtomTypeInRes::CA), // todo: A/R
                force_field_type: Some("ca".to_owned()), // todo: A/R
                partial_charge: Some(-0.115),         // todo: A/R,
                ..Default::default()
            })
        }

        result
    }
}

// todo: Into a GUI util?
pub fn enter_edit_mode(state: &mut State, scene: &mut Scene, engine_updates: &mut EngineUpdates) {
    state.volatile.operating_mode = OperatingMode::MolEditor;
    state.volatile.operating_mode_prev = OperatingMode::Primary;

    let cam_dist: f32 = 15.;

    match state.active_mol() {
        Some(mol) => {
            state.mol_editor.load_mol(mol.common());
        }
        None => {
            state.mol_editor.clear_mol();
        }
    }

    state.volatile.control_scheme_prev = scene.input_settings.control_scheme;
    scene.input_settings.control_scheme = ControlScheme::Arc {
        center: Vec3F32::new_zero(),
    };

    scene.camera.position = Vec3F32::new(0., 0., -cam_dist);
    scene.camera.orientation = Quaternion::new_identity(); // todo: COnfirm this is fwd.

    // Hide all of the non-editor molecules.
    draw_peptide(state, scene);
    draw_all_ligs(state, scene);
    draw_all_nucleic_acids(state, scene);
    draw_all_lipids(state, scene);

    draw_mol(
        MoleculeGenericRef::Ligand(&state.mol_editor.mol),
        0,
        &state.ui,
        &None,
        ManipMode::None,
    );

    engine_updates.entities = EntityUpdate::All;
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

    engine_updates.entities = EntityUpdate::All;
}
