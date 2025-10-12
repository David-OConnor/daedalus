use egui::{Color32, RichText, Ui};
use graphics::{EngineUpdates, Scene};
use na_seq::Element;
use crate::{State};
use crate::molecule::{Atom, MoleculeCommon};
use crate::ui::COL_SPACING;
use crate::util::exit_edit_mode;

//  todo: Move business logic to a non-ui mol editor panel once there is a bit of it
// todo: Check DBs (with a button maybe?) to see if the molecule exists in a DB already, or if
// todo a similar one does.

/// For editing small organic molecules.
pub struct MolEditorState {
    atoms: Vec<Atom>,
}

mod templates {

}

impl MolEditorState {
    pub fn from_mol(mol: &MoleculeCommon) -> Self {
        // We assign H dynamically; ignore present ones.

        let atoms: Vec<_> = mol.atoms.iter()
            .filter(|a| a.element != Element::Hydrogen)
            .map(|a| a.clone())
            .collect();

        Self {
            atoms
        }
    }
}

pub fn editor(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    // todo: New state for the WIp molecule. New struct for it.

    ui.horizontal(|ui| {
    if ui.button("C").clicked() {

    }

    if ui.button("O").clicked() {

    }

    if ui.button("N").clicked() {

    }

    ui.add_space(COL_SPACING);

    if ui.button("−OH").clicked() {

    }

    if ui.button("−COOH").clicked() {

    }

    if ui.button("−NH₂").clicked() {

    }

    if ui.button("Ring").clicked() {

    }

    ui.add_space(COL_SPACING);

    if ui.button(RichText::new("Exit editor").color(Color32::LIGHT_RED)).clicked() {
        exit_edit_mode(state, scene, engine_updates);
    }
    });
}