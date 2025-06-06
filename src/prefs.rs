//! Code related to saving user *preferences*. e.g. opened molecules, view configuration etc.

use std::{collections::HashMap, path::PathBuf};

use bincode::{Decode, Encode};
use bio_apis::rcsb::{FilesAvailable, PdbDataResults, PdbMetaData, load_metadata};
use graphics::{
    ControlScheme,
    app_utils::{load, save},
};

use crate::{
    CamSnapshot, MsaaSetting, Selection, State, ViewSelLevel, Visibility,
    docking::DockingSite,
    inputs::{MOVEMENT_SENS, ROTATE_SENS},
    mol_drawing::MoleculeView,
};

pub const DEFAULT_PREFS_FILE: &str = "daedalus_prefs.dae";

#[derive(Debug, Encode, Decode)]
pub struct ToSave {
    pub per_mol: HashMap<String, PerMolToSave>,
    pub last_opened: Option<PathBuf>,
    pub last_ligand_opened: Option<PathBuf>,
    pub last_map_opened: Option<PathBuf>,
    pub autodock_vina_path: Option<PathBuf>,
    pub control_scheme: ControlScheme,
    pub msaa: MsaaSetting,
    /// Direct conversion from engine standard
    pub movement_speed: u8,
    /// Divide this by 100 to get engine standard.
    pub rotation_sens: u8,
}

impl Default for ToSave {
    fn default() -> Self {
        Self {
            per_mol: Default::default(),
            last_opened: Default::default(),
            last_map_opened: Default::default(),
            last_ligand_opened: Default::default(),
            autodock_vina_path: Default::default(),
            control_scheme: Default::default(),
            msaa: Default::default(),
            movement_speed: MOVEMENT_SENS as u8,
            rotation_sens: (ROTATE_SENS * 100.) as u8,
        }
    }
}

#[derive(Debug, Encode, Decode)]
pub struct MolMetaData {
    prim_cit_title: String,
}

#[derive(Debug, Encode, Decode)]
pub struct PerMolToSave {
    selection: Selection,
    cam_snapshots: Vec<CamSnapshot>,
    mol_view: MoleculeView,
    view_sel_level: ViewSelLevel,
    near_sel_only: bool,
    near_lig_only: bool,
    nearby_dist_thresh: u16,
    chain_vis: Vec<bool>,
    chain_to_pick_res: Option<usize>,
    visibility: Visibility,
    metadata: Option<MolMetaData>,
    docking_site: DockingSite,
    show_docking_tools: bool,
    res_color_by_index: bool,
    show_aa_seq: bool,
    rcsb_data: Option<PdbDataResults>,
    rcsb_files_avail: Option<FilesAvailable>,
}

impl PerMolToSave {
    pub fn from_state(state: &State) -> Self {
        let mut chain_vis = Vec::new();
        let mut metadata = None;
        let mut rcsb_data = None;
        let mut rcsb_files_avail = None;

        if let Some(mol) = &state.molecule {
            chain_vis = mol.chains.iter().map(|c| c.visible).collect();

            if let Some(md) = &mol.metadata {
                metadata = Some(MolMetaData {
                    prim_cit_title: md.prim_cit_title.clone(),
                });
            }

            rcsb_data = mol.rcsb_data.clone();
            rcsb_files_avail = mol.rcsb_files_avail.clone();
        }

        let mut docking_site = Default::default();
        if let Some(lig) = &state.ligand {
            docking_site = lig.docking_site.clone();
        }

        Self {
            selection: state.selection.clone(),
            cam_snapshots: state.cam_snapshots.clone(),
            mol_view: state.ui.mol_view,
            view_sel_level: state.ui.view_sel_level,
            near_sel_only: state.ui.show_near_sel_only,
            near_lig_only: state.ui.show_near_lig_only,
            nearby_dist_thresh: state.ui.nearby_dist_thresh,
            chain_vis,
            chain_to_pick_res: state.ui.chain_to_pick_res,
            visibility: state.ui.visibility.clone(),
            metadata,
            docking_site,
            show_docking_tools: state.ui.show_docking_tools,
            res_color_by_index: state.ui.res_color_by_index,
            show_aa_seq: state.ui.show_aa_seq,
            rcsb_data,
            rcsb_files_avail,
        }
    }
}

impl State {
    /// Update when prefs change, periodically etc.
    pub fn update_save_prefs(&mut self) {
        if let Some(mol) = &self.molecule {
            let data = PerMolToSave::from_state(self);

            self.to_save.per_mol.insert(mol.ident.clone(), data);
        }

        if let Err(e) = save(
            &self.volatile.prefs_dir.join(DEFAULT_PREFS_FILE),
            &self.to_save,
        ) {
            eprintln!("Error saving state: {e:?}");
        }
    }

    /// Run this when prefs, or a new molecule are loaded.
    pub fn update_from_prefs(&mut self) {
        self.reset_selections();

        let mut center = lin_alg::f64::Vec3::new_zero();

        if let Some(mol) = &mut self.molecule {
            if self.to_save.per_mol.contains_key(&mol.ident) {
                let data = &self.to_save.per_mol[&mol.ident];

                self.selection = data.selection.clone();
                self.cam_snapshots = data.cam_snapshots.clone();
                self.ui.mol_view = data.mol_view;
                self.ui.view_sel_level = data.view_sel_level;
                self.ui.show_near_sel_only = data.near_sel_only;
                self.ui.show_near_lig_only = data.near_lig_only;
                self.ui.nearby_dist_thresh = data.nearby_dist_thresh;
                self.ui.chain_to_pick_res = data.chain_to_pick_res;
                self.ui.visibility = data.visibility.clone();
                self.ui.show_docking_tools = data.show_docking_tools;
                self.ui.res_color_by_index = data.res_color_by_index;
                self.ui.show_aa_seq = data.show_aa_seq;

                if let Some(md) = &data.metadata {
                    mol.metadata = Some(PdbMetaData {
                        prim_cit_title: md.prim_cit_title.clone(),
                    })
                }

                for (i, chain) in mol.chains.iter_mut().enumerate() {
                    if i < data.chain_vis.len() {
                        chain.visible = data.chain_vis[i];
                    }
                }

                center = data.docking_site.site_center;

                mol.rcsb_data = data.rcsb_data.clone();
                mol.rcsb_files_avail = data.rcsb_files_avail.clone();
            }

            // If loaded from file or not.
            if mol.metadata.is_none() {
                println!("Getting MD");
                match load_metadata(&mol.ident) {
                    Ok(md) => mol.metadata = Some(md),
                    Err(_) => eprintln!("Error loading metadata for: {}", mol.ident),
                }
            }
        }

        self.ui.movement_speed_input = self.to_save.movement_speed.to_string();
        self.ui.rotation_sens_input = self.to_save.rotation_sens.to_string();

        self.update_docking_site(center);
    }

    pub fn load_prefs(&mut self) {
        match load(&PathBuf::from(DEFAULT_PREFS_FILE)) {
            Ok(p) => self.to_save = p,
            Err(_) => eprintln!("Unable to load save file; possibly the first time running."),
        }

        self.update_from_prefs();
    }
}
