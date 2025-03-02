extern crate core;

mod aa_coords;
mod add_hydrogens;
mod amino_acid_coords;
mod asa;
mod bond_inference;
mod cartoon_mesh;
mod docking;
mod download_pdb;
// mod drug_like;
mod element;
mod file_io;
mod input;
mod mol_drawing;
mod molecule;
mod navigation;
mod prefs;
mod rcsb_api;
mod render;
mod save_load;
mod ui;
mod util;
mod vibrations;

use std::{
    io,
    io::{ErrorKind, Read},
    path::Path,
    str::FromStr,
    sync::Arc,
};

use bincode::{Decode, Encode};
use egui_file_dialog::{FileDialog, FileDialogConfig};
use file_io::{pdb::load_pdb, sdf::load_sdf};
use graphics::Camera;
use lin_alg::{
    f32::{Quaternion, Vec3},
    f64::{Quaternion as QuaternionF64, Vec3 as Vec3F64},
};
use mol_drawing::MoleculeView;
use molecule::Molecule;
use pdbtbx::{self, PDB};
use rayon::iter::ParallelIterator;

use crate::{
    aa_coords::bond_vecs::init_local_bond_vecs,
    docking::{DockingInit, check_adv_avail, docking_prep_external::check_babel_avail},
    file_io::pdbqt::load_pdbqt,
    molecule::Ligand,
    navigation::Tab,
    prefs::ToSave,
    render::render,
    ui::VIEW_DEPTH_MAX,
};

// todo: Eventually, implement a system that automatically checks for changes, and don't
// todo save to disk if there are no changes.
const PREFS_SAVE_INTERVAL: u64 = 60; // Save user preferences this often, in seconds.

#[derive(Debug, Clone, Default)]
pub enum ComputationDevice {
    #[default]
    Cpu,
    #[cfg(feature = "cuda")]
    Gpu(Arc<CudaDevice>),
}

#[derive(Clone, Copy, PartialEq, Debug, Default, Encode, Decode)]
pub enum ViewSelLevel {
    Atom,
    #[default]
    Residue,
}

impl ViewSelLevel {
    pub fn to_string(&self) -> String {
        match self {
            Self::Atom => "Atom",
            Self::Residue => "Residue",
        }
        .to_owned()
    }
}

/// Temprary, and generated state.
struct StateVolatile {
    load_dialog: FileDialog,
    load_ligand_dialog: FileDialog,
    autodock_path_dialog: FileDialog,
    save_pdbqt_dialog: FileDialog,
    /// We use this for offsetting our cursor selection.
    ui_height: f32,
    // /// Center and size are used for setting the camera. Dependent on the molecule atom positions.
    // mol_center: Vec3,
    // mol_size: f32, // Dimension-agnostic
}

impl Default for StateVolatile {
    fn default() -> Self {
        let cfg = FileDialogConfig {
            ..Default::default()
        }
        .add_file_filter(
            "PDB/CIF/SDF",
            Arc::new(|p| {
                let ext = p.extension().unwrap_or_default().to_ascii_lowercase();
                ext == "pdb" || ext == "cif" || ext == "sdf" || ext == "pdbqt"
            }),
        );

        let cfg_vina = FileDialogConfig {
            ..Default::default()
        }
        .add_file_filter(
            "Executables",
            Arc::new(|p| {
                let ext = p.extension().unwrap_or_default().to_ascii_lowercase();
                ext == "" || ext == "exe"
            }),
        );

        let cfg_save_pdbqt = FileDialogConfig {
            ..Default::default()
        };
        // .add_file_filter(
        //     "Executables",
        //     Arc::new(|p| {
        //         let ext = p.extension().unwrap_or_default().to_ascii_lowercase();
        //         ext == "" || ext == "exe"
        //     }),
        // );

        let load_dialog = FileDialog::with_config(cfg.clone()).default_file_filter("PDB/CIF/SDF");
        let load_ligand_dialog = FileDialog::with_config(cfg).default_file_filter("PDB/CIF/SDF");

        let autodock_path_dialog =
            FileDialog::with_config(cfg_vina).default_file_filter("Executables");

        let save_pdbqt_dialog = FileDialog::with_config(cfg_save_pdbqt);
        // .default_file_filter("Executables");

        Self {
            load_dialog,
            load_ligand_dialog,
            autodock_path_dialog,
            save_pdbqt_dialog,
            ui_height: 0.,
        }
    }
}

#[derive(Debug, Clone, Encode, Decode)]
struct Visibility {
    hide_sidechains: bool,
    hide_water: bool,
    /// Hide hetero atoms: i.e. ones not part of a polypeptide.
    hide_hetero: bool,
    hide_non_hetero: bool,
    hide_ligand: bool,
    hide_hydrogen: bool,
    hide_h_bonds: bool,
}

impl Default for Visibility {
    fn default() -> Self {
        Self {
            hide_sidechains: false,
            hide_water: false,
            hide_hetero: false,
            hide_non_hetero: false,
            hide_ligand: false,
            hide_hydrogen: false,
            hide_h_bonds: true,
        }
    }
}

/// Ui text fields and similar.
#[derive(Default)]
struct StateUi {
    mol_view: MoleculeView,
    view_sel_level: ViewSelLevel,
    /// Mouse cursor
    cursor_pos: Option<(f32, f32)>,
    rcsb_input: String,
    cam_snapshot_name: String,
    residue_search: String,
    /// Experimental.
    show_nearby_only: bool,
    /// Angstrom. For selections.
    nearby_dist_thresh: u16,
    view_depth: u16, // angstrom
    cam_snapshot: Option<usize>,
    dt: f32, // seconds.
    // For selecting residues from the GUI.
    chain_to_pick_res: Option<usize>,
    /// Workaround for a bug or limitation in EGUI's `is_pointer_button_down_on`.
    // inputs_commanded: InputsCommanded,
    visibility: Visibility,
    middle_click_down: bool,
    autodock_path_valid: bool,
    mouse_in_window: bool,
    docking_site_x: String,
    docking_site_y: String,
    docking_site_z: String,
    docking_site_size: String,
}

#[derive(Clone, Copy, PartialEq, Debug, Default, Encode, Decode)]
pub enum Selection {
    #[default]
    None,
    Atom(usize),
    Residue(usize),
}

#[derive(Clone, Debug, Encode, Decode)]
pub struct CamSnapshot {
    // We don't use camera directly, so we don't have to store the projection matrix, and so we can impl
    // Encode/Decode
    pub position: Vec3,
    pub orientation: Quaternion,
    pub far: f32,
    pub name: String,
}

impl CamSnapshot {
    pub fn from_cam(cam: &Camera, name: String) -> Self {
        Self {
            position: cam.position,
            orientation: cam.orientation,
            far: cam.far,
            name,
        }
    }
}

#[derive(Default)]
struct State {
    pub ui: StateUi,
    pub volatile: StateVolatile,
    pub pdb: Option<PDB>,
    pub molecule: Option<Molecule>,
    // pub ligand: Option<Molecule>,
    pub ligand: Option<Ligand>,
    // todo: Should selection-related go in StateUi?
    pub selection: Selection,
    pub cam_snapshots: Vec<CamSnapshot>,
    // This allows us to keep in-memory data for other molecules.
    // Key is PDB ident; value is per-item.
    // todo: Make a new struct if you add more non
    pub to_save: ToSave,
    pub tabs_open: Vec<Tab>,
    pub babel_avail: bool,
    pub docking_ready: bool,
}

impl State {
    /// E.g. when loading a new molecule.
    pub fn reset_selections(&mut self) {
        self.selection = Selection::None;
        self.cam_snapshots = Vec::new();
        self.ui.cam_snapshot = None;
        self.ui.chain_to_pick_res = None;
    }

    // todo: Consider how you handle loading and storing of ligands vs targets.
    pub fn open_molecule(&mut self, path: &Path, is_ligand: bool) {
        let mut ligand = None;
        let molecule = match path
            .extension()
            .unwrap_or_default()
            .to_ascii_lowercase()
            .to_str()
            .unwrap_or_default()
        {
            "sdf" => load_sdf(path),
            "pdbqt" => {
                load_pdbqt(path).map(|(molecule, mut ligand_)| {
                    if ligand_.is_some() {
                        ligand_.as_mut().unwrap().molecule = molecule.clone(); // sloppy
                    }
                    ligand = ligand_;
                    molecule
                })
            } // todo: Handle the ligand part.
            "pdb" | "cif" => {
                let pdb = load_pdb(path);
                match pdb {
                    Ok(p) => {
                        self.pdb = Some(p);
                        Ok(Molecule::from_pdb(self.pdb.as_ref().unwrap()))
                    }
                    Err(e) => Err(e),
                }
            }
            _ => Err(io::Error::new(
                ErrorKind::InvalidData,
                "Invalid file extension",
            )),
        };

        match molecule {
            Ok(mol) => {
                if is_ligand {
                    let docking_init = DockingInit {
                        site_center: Vec3F64::new(-18.955, -5.188, 8.617),
                        site_box_size: 10.,
                    };

                    self.ui.docking_site_x = docking_init.site_center.x.to_string();
                    self.ui.docking_site_y = docking_init.site_center.y.to_string();
                    self.ui.docking_site_z = docking_init.site_center.z.to_string();
                    self.ui.docking_site_size = docking_init.site_box_size.to_string();

                    self.ligand = Some(Ligand {
                        molecule: mol,
                        docking_init,
                        orientation: QuaternionF64::new_identity(),
                        torsions: Vec::new(),
                        unit_cell_dims: Default::default(),
                    });
                } else {
                    self.molecule = Some(mol);
                }

                self.update_from_prefs();
            }
            Err(e) => eprintln!("Error loading file at path {path:?}: {e:?}"),
        }
    }
}

fn main() {
    init_local_bond_vecs();

    let mut state = State::default();
    state.ui.view_depth = VIEW_DEPTH_MAX;

    state.load_prefs();

    let last_opened = state.to_save.last_opened.clone();
    if let Some(path) = &last_opened {
        state.open_molecule(path, false);
    }

    let last_ligand_opened = state.to_save.last_ligand_opened.clone();
    if let Some(path) = &last_ligand_opened {
        state.open_molecule(path, true);
    }

    if let Some(path) = &state.to_save.autodock_vina_path {
        state.ui.autodock_path_valid = check_adv_avail(path);

        // If the saved path fails our check, leave it blank so the user can re-attempt.
        if !state.ui.autodock_path_valid {
            state.to_save.autodock_vina_path = None;
            state.update_save_prefs();
        }
    }

    state.babel_avail = check_babel_avail();

    render(state);
}
