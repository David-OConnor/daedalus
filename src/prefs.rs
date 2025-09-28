//! Code related to saving user *preferences*. e.g. opened molecules, view configuration etc.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use bincode::{
    BorrowDecode, Decode, Encode,
    de::BorrowDecoder,
    error::{DecodeError, EncodeError},
};
use bio_apis::rcsb::{FilesAvailable, PdbDataResults};
use chrono::{DateTime, TimeZone, Utc};
use dynamics::MdConfig;
use graphics::{
    ControlScheme,
    app_utils::{load, save},
};
use lin_alg::f64::Vec3;

use crate::docking_v2::DockingSite;
use crate::{
    CamSnapshot,
    MsaaSetting,
    Selection,
    State,
    ViewSelLevel,
    Visibility,
    // docking::{ConformationType, DockingSite},
    drawing::MoleculeView,
    inputs::{MOVEMENT_SENS, ROTATE_SENS},
};

pub const DEFAULT_PREFS_FILE: &str = "daedalus_prefs.dae";

#[derive(Clone, Copy, PartialEq, Debug, Encode, Decode)]
pub enum OpenType {
    Peptide,
    Ligand,
    NucleicAcid,
    Map,
    Frcmod,
}

#[derive(Debug, Clone)]
pub struct OpenHistory {
    pub timestamp: DateTime<Utc>,
    pub path: PathBuf,
    pub type_: OpenType,
    // /// e.g. RCSB, pubchem etc ident. Maches moleculeCommon.
    // pub ident: String,
    /// For determining which to open at program start.
    pub last_session: bool,
}

// Manual bincode impls
impl Encode for OpenHistory {
    fn encode<E: bincode::enc::Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        // Store timestamp as i64 (seconds since Unix epoch)
        let ts = self.timestamp.timestamp();
        ts.encode(encoder)?;

        self.path.encode(encoder)?;
        self.type_.encode(encoder)?;
        // self.ident.encode(encoder)?;
        self.last_session.encode(encoder)?;

        Ok(())
    }
}

impl<T> Decode<T> for OpenHistory {
    fn decode<D: bincode::de::Decoder>(decoder: &mut D) -> Result<Self, DecodeError> {
        let ts = i64::decode(decoder)?;
        let path = PathBuf::decode(decoder)?;
        let type_ = OpenType::decode(decoder)?;
        let last_session = bool::decode(decoder)?;

        Ok(OpenHistory {
            timestamp: Utc
                .timestamp_opt(ts, 0)
                .single()
                .ok_or_else(|| DecodeError::OtherString("invalid timestamp".to_string()))?,
            path,
            type_,
            // ident,
            last_session,
        })
    }
}

impl<'de, C> BorrowDecode<'de, C> for OpenHistory {
    fn borrow_decode<D>(decoder: &mut D) -> Result<Self, DecodeError>
    where
        D: BorrowDecoder<'de, Context = C>,
    {
        Decode::decode(decoder)
    }
}

impl OpenHistory {
    pub fn new(path: &Path, type_: OpenType) -> Self {
        Self {
            timestamp: Utc::now(),
            path: path.to_owned(),
            type_,
            // ident,
            last_session: true,
        }
    }
}

/// We maintain some of the state that is saved in the preferences file here, to keep
/// the save/load state streamlined, instead of in an intermediate struct between main state, and
/// saving/loading.
#[derive(Debug, Encode, Decode)]
pub struct ToSave {
    pub per_mol: HashMap<String, PerMolToSave>,
    pub open_history: Vec<OpenHistory>,
    pub control_scheme: ControlScheme,
    pub msaa: MsaaSetting,
    /// Direct conversion from engine standard
    pub movement_speed: u8,
    /// Divide this by 100 to get engine standard.
    pub rotation_sens: u8,
    /// Solvent-accessible surface (and dots) precision. Lower is higher precision. A value of 0.5 - 0.6
    /// is a good default. Too low will cause crashes and very poor performance. Higher is too coarse.
    pub sa_surface_precision: f32,
    pub md_config: MdConfig,
    pub num_md_steps: u32,
    /// ps (10^-12). Typical values are 0.001 or 0.002.
    pub md_dt: f32,
    pub ph: f32,
}

impl Default for ToSave {
    fn default() -> Self {
        Self {
            per_mol: Default::default(),
            open_history: Default::default(),
            control_scheme: Default::default(),
            msaa: Default::default(),
            movement_speed: MOVEMENT_SENS as u8,
            rotation_sens: (ROTATE_SENS * 100.) as u8,
            sa_surface_precision: 0.55,
            md_config: Default::default(),
            num_md_steps: 100,
            md_dt: 0.002,
            ph: 7.4,
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
    // todo: A/R
    // metadata: Option<MolMetaData>,
    pub docking_site: DockingSite,
    show_docking_tools: bool,
    res_color_by_index: bool,
    aatom_color_by_charge: bool,
    show_aa_seq: bool,
    rcsb_data: Option<PdbDataResults>,
    rcsb_files_avail: Option<FilesAvailable>,
    docking_site_posit: Vec3,
    /// This is useful in the case of absolute positions. Ideally, this is per-ligand.
    lig_atom_positions: Vec<Vec3>,
}

impl PerMolToSave {
    pub fn from_state(state: &State, on_init: bool) -> Self {
        let mut chain_vis = Vec::new();
        // let mut metadata = None;
        let mut rcsb_data = None;
        let mut rcsb_files_avail = None;

        if let Some(mol) = &state.peptide {
            chain_vis = mol.chains.iter().map(|c| c.visible).collect();

            rcsb_data = mol.rcsb_data.clone();
            rcsb_files_avail = mol.rcsb_files_avail.clone();
        }

        let mut docking_site = Default::default();

        let mut lig_posit = Vec3::new_zero();
        let mut lig_atom_positions = Vec::new();

        if let Some(mol) = state.active_mol() {
            // docking_site = lig.docking_site.clone();
            // lig_posit = lig.pose.anchor_posit;

            // Don't save this if on init; the data in lig is the default,
            // and we haven't loaded the posits to it yet.
            // todo: If you find a more robust way to handle saving order-dependent data,
            // todo: remove this.
            if !on_init {
                lig_atom_positions = mol.common().atom_posits.clone();
                if !mol.common().atom_posits.is_empty() {
                    println!("\nSaving atom posits: {:?}", mol.common().atom_posits[0]); // todo temp
                }
            }
        }

        Self {
            selection: state.ui.selection.clone(),
            cam_snapshots: state.cam_snapshots.clone(),
            mol_view: state.ui.mol_view,
            view_sel_level: state.ui.view_sel_level,
            near_sel_only: state.ui.show_near_sel_only,
            near_lig_only: state.ui.show_near_lig_only,
            nearby_dist_thresh: state.ui.nearby_dist_thresh,
            chain_vis,
            chain_to_pick_res: state.ui.chain_to_pick_res,
            visibility: state.ui.visibility.clone(),
            // metadata,
            docking_site,
            show_docking_tools: state.ui.show_docking_tools,
            res_color_by_index: state.ui.res_color_by_index,
            aatom_color_by_charge: state.ui.atom_color_by_charge,
            show_aa_seq: state.ui.show_aa_seq,
            rcsb_data,
            rcsb_files_avail,
            docking_site_posit: lig_posit,
            lig_atom_positions,
        }
    }
}

impl State {
    /// We run this after loading a molecule.
    pub fn update_save_prefs_no_mol(&mut self) {
        if let Err(e) = save(
            &self.volatile.prefs_dir.join(DEFAULT_PREFS_FILE),
            &self.to_save,
        ) {
            eprintln!("Error saving state: {e:?}");
        }
    }

    /// Update when prefs change, periodically etc.
    /// todo: See the note in PerMolsave::from_state. Workaround for order-related bugs.
    pub fn update_save_prefs(&mut self, on_init: bool) {
        println!("Saving state to prefs file.");
        if let Some(mol) = &self.peptide {
            let data = PerMolToSave::from_state(self, on_init);

            self.to_save.per_mol.insert(mol.common.ident.clone(), data);
        }

        // println!("Saving history: {:?}", self.to_save.open_history);

        if let Err(e) = save(
            &self.volatile.prefs_dir.join(DEFAULT_PREFS_FILE),
            &self.to_save,
        ) {
            eprintln!("Error saving state: {e:?}");
        }
    }

    /// Run this when prefs, or a new molecule are loaded.
    pub fn update_from_prefs(&mut self) {
        println!("Updating state from prefs data");
        self.reset_selections();

        let mut center = Vec3::new_zero();

        if let Some(mol) = &mut self.peptide {
            if self.to_save.per_mol.contains_key(&mol.common.ident) {
                let data = &self.to_save.per_mol[&mol.common.ident];

                self.ui.selection = data.selection.clone();
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
                self.ui.atom_color_by_charge = data.aatom_color_by_charge;
                self.ui.show_aa_seq = data.show_aa_seq;

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
            // if mol.metadata.is_none() {
            //     println!("Getting MD");
            //     match load_metadata(&mol.common.ident) {
            //         Ok(md) => mol.metadata = Some(md),
            //         Err(_) => eprintln!("Error loading metadata for: {}", mol.common.ident),
            //     }
            // }
        }

        // if let Some(lig) = self.active_lig_mut() {
        //     // lig.docking_site.site_center = data.docking_site_posit; // todo: Or docking site?
        //
        //     // todo: This check is a workaround for overal problems related to how we store molecules
        //     // todo and ligands. Without it, we can desync the positions, and cause index-error crashes
        //     if data.lig_atom_positions.len() == lig.common.atom_posits.len() {
        //         lig.common.atom_posits = data.lig_atom_positions.clone();
        //     } else {
        //         eprintln!("Error loading ligand atom positions; look into this.")
        //     }
        //
        //     // lig.pose.conformation_type = ConformationType::AbsolutePosits;
        // }

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
