//! Code related to saving user *preferences*. e.g. opened molecules, view configuration etc.

use std::{collections::HashMap, path::PathBuf};

use bincode::{Decode, Encode};
use egui::epaint::tessellator::Path;
use graphics::app_utils::{load, save};

use crate::{
    rcsb_api::{load_pdb_metadata, PdbMetaData},
    render::MoleculeView,
    CamSnapshot, Selection, State, ViewSelLevel,
};

pub const DEFAULT_PREFS_FILE: &str = "bcv_prefs.bcv";

#[derive(Debug, Default, Encode, Decode)]
pub struct ToSave {
    pub per_mol: HashMap<String, PerMolToSave>,
    pub last_opened: Option<PathBuf>,
}

#[derive(Debug, Encode, Decode)]
pub struct PerMolToSave {
    selection: Selection,
    cam_snapshots: Vec<CamSnapshot>,
    mol_view: MoleculeView,
    view_sel_level: ViewSelLevel,
    show_nearby_only: bool,
    nearby_dist_thresh: u16,
    chain_vis: Vec<bool>,
    chain_to_pick_res: Option<usize>,
    metadata: Option<PdbMetaData>,
    hide_sidechains: bool,
    hide_water: bool,
}

impl PerMolToSave {
    pub fn from_state(state: &State) -> Self {
        let mut chain_vis = Vec::new();
        let mut metadata = None;

        if let Some(mol) = &state.molecule {
            chain_vis = mol.chains.iter().map(|c| c.visible).collect();
            metadata = mol.metadata.clone();
        }

        Self {
            selection: state.selection.clone(),
            cam_snapshots: state.cam_snapshots.clone(),
            mol_view: state.ui.mol_view,
            view_sel_level: state.ui.view_sel_level,
            show_nearby_only: state.ui.show_nearby_only,
            nearby_dist_thresh: state.ui.nearby_dist_thresh,
            chain_vis,
            chain_to_pick_res: state.ui.chain_to_pick_res,
            metadata,
            hide_sidechains: state.ui.hide_sidechains,
            hide_water: state.ui.hide_water,
        }
    }
}

impl State {
    /// Update when prefs change, periodically etc.
    pub fn update_save_prefs(&mut self) {
        if let Some(mol) = &self.molecule {
            let data = PerMolToSave::from_state(self);

            self.to_save.per_mol.insert(mol.ident.clone(), data);

            if let Err(e) = save(&PathBuf::from(DEFAULT_PREFS_FILE), &self.to_save) {
                eprintln!("Error saving state: {e:?}");
            }
        }
    }

    /// Run this when prefs, or a new molecule are loaded.
    pub fn update_from_prefs(&mut self) {
        self.reset_selections();

        if let Some(mol) = &mut self.molecule {
            if self.to_save.per_mol.contains_key(&mol.ident) {
                let data = &self.to_save.per_mol[&mol.ident];

                self.selection = data.selection;
                self.cam_snapshots = data.cam_snapshots.clone();
                self.ui.mol_view = data.mol_view;
                self.ui.view_sel_level = data.view_sel_level;
                self.ui.show_nearby_only = data.show_nearby_only;
                self.ui.nearby_dist_thresh = data.nearby_dist_thresh;
                self.ui.chain_to_pick_res = data.chain_to_pick_res;
                self.ui.hide_sidechains = data.hide_sidechains;
                self.ui.hide_water = data.hide_water;

                if let Some(md) = &data.metadata {
                    mol.metadata = Some(md.clone())
                }

                for (i, chain) in mol.chains.iter_mut().enumerate() {
                    if i < data.chain_vis.len() {
                        chain.visible = data.chain_vis[i];
                    }
                }
            }

            // If loaded from file or not.
            if mol.metadata.is_none() {
                println!("Getting MD");
                match load_pdb_metadata(&mol.ident) {
                    Ok(md) => mol.metadata = Some(md),
                    Err(_) => eprintln!("Error loading metadata for: {}", mol.ident),
                }
            }
        }
    }

    pub fn load_prefs(&mut self) {
        match load(&PathBuf::from(DEFAULT_PREFS_FILE)) {
            Ok(p) => self.to_save = p,
            Err(e) => eprintln!("Error loading preferences on init: {e:?}"),
        }

        self.update_from_prefs();
    }
}
