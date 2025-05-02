// #![cfg_attr(
//     all(not(debug_assertions), target_os = "windows"),
//     windows_subsystem = "windows"
// )]

#![allow(clippy::too_many_arguments)]

mod aa_coords;
mod add_hydrogens;
mod amino_acid_coords;
mod asa;
mod bond_inference;
mod cartoon_mesh;
mod docking;
mod download_mols;
mod drug_like;
mod element;
mod file_io;
mod forces;
mod inputs;
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

#[cfg(test)]
mod tests;

use std::{collections::HashMap, fmt, io, io::ErrorKind, path::Path, sync::Arc};

use barnes_hut::BhConfig;
use bincode::{Decode, Encode};
#[cfg(feature = "cuda")]
use cuda_setup::ComputationDevice;
#[cfg(feature = "cuda")]
use cudarc::{
    driver::{CudaContext, CudaModule, CudaStream},
    nvrtc::Ptx,
};
use egui_file_dialog::{FileDialog, FileDialogConfig};
use file_io::{pdb::load_pdb, sdf::load_sdf};
use graphics::{Camera, InputsCommanded};
use lin_alg::f32::{Quaternion, Vec3, f32x8};
use mol_drawing::MoleculeView;
use molecule::Molecule;
use pdbtbx::{self, PDB};

use crate::{
    aa_coords::bond_vecs::init_local_bond_vecs,
    docking::{
        BindingEnergy, THETA_BH, dynamics_playback::Snapshot, external::check_adv_avail,
        prep::DockingSetup,
    },
    element::{Element, init_lj_lut},
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

#[derive(Clone, Copy, PartialEq, Debug, Default, Encode, Decode)]
pub enum ViewSelLevel {
    Atom,
    #[default]
    Residue,
}

impl fmt::Display for ViewSelLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Atom => write!(f, "Atom"),
            Self::Residue => write!(f, "Residue"),
        }
    }
}

struct FileDialogs {
    load: FileDialog,
    load_ligand: FileDialog,
    save: FileDialog,
    save_ligand: FileDialog,
    autodock_path: FileDialog,
    save_pdbqt: FileDialog,
}

impl Default for FileDialogs {
    fn default() -> Self {
        let cfg_protein = FileDialogConfig {
            ..Default::default()
        }
        .add_file_filter(
            "PDB/CIF",
            Arc::new(|p| {
                let ext = p.extension().unwrap_or_default().to_ascii_lowercase();
                ext == "pdb" || ext == "cif"
            }),
        );

        let cfg_small_mol = FileDialogConfig {
            ..Default::default()
        }
        .add_file_filter(
            "SDF/MOL2/PDBQT",
            Arc::new(|p| {
                let ext = p.extension().unwrap_or_default().to_ascii_lowercase();
                ext == "sdf" || ext == "mol2" || ext == "pdbqt"
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

        let load = FileDialog::with_config(cfg_protein.clone()).default_file_filter("PDB/CIF");
        let load_ligand =
            FileDialog::with_config(cfg_small_mol.clone()).default_file_filter("SDF/MOL2/PDBQT");

        let save = FileDialog::with_config(cfg_protein).default_file_filter("PDB/CIF");
        let save_ligand =
            FileDialog::with_config(cfg_small_mol).default_file_filter("SDF/MOL2/PDBQT");

        let autodock_path = FileDialog::with_config(cfg_vina).default_file_filter("Executables");

        let save_pdbqt = FileDialog::with_config(cfg_save_pdbqt);

        Self {
            load,
            load_ligand,
            save,
            save_ligand,
            autodock_path,
            save_pdbqt,
        }
    }
}

/// Temprary, and generated state.
struct StateVolatile {
    dialogs: FileDialogs,
    /// We use this for offsetting our cursor selection.
    ui_height: f32,
    // /// Center and size are used for setting the camera. Dependent on the molecule atom positions.
    // mol_center: Vec3,
    // mol_size: f32, // Dimension-agnostic
    /// We Use this to keep track of key press state for the camera movement, so we can continuously
    /// update the flashlight when moving.
    inputs_commanded: InputsCommanded,
    /// (Sigma, Epsilon). Initialize once at startup. Not-quite-static.
    lj_lookup_table: HashMap<(Element, Element), (f32, f32)>,
    snapshots: Vec<Snapshot>,
    docking_setup: Option<DockingSetup>,
}

impl Default for StateVolatile {
    fn default() -> Self {
        Self {
            dialogs: Default::default(),
            ui_height: 0.,
            inputs_commanded: Default::default(),
            lj_lookup_table: init_lj_lut(),
            snapshots: Vec::new(),
            docking_setup: None,
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
            hide_hydrogen: true,
            hide_h_bonds: false,
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
    db_input: String,
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
    left_click_down: bool,
    middle_click_down: bool,
    autodock_path_valid: bool,
    mouse_in_window: bool,
    docking_site_x: String,
    docking_site_y: String,
    docking_site_z: String,
    docking_site_size: String,
    /// For the arc/orbit cam only.
    orbit_around_selection: bool,
    binding_energy_disp: Option<BindingEnergy>,
    current_snapshot: usize,
    /// A flag so we know to update the flashlight upon loading a new model; this should be done within
    /// a callback.
    new_mol_loaded: bool,
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
    pub to_save: ToSave,
    pub tabs_open: Vec<Tab>,
    pub babel_avail: bool,
    pub docking_ready: bool,
    pub bh_config: BhConfig,
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
                    let lig = Ligand::new(mol);

                    self.ui.docking_site_x = lig.docking_site.site_center.x.to_string();
                    self.ui.docking_site_y = lig.docking_site.site_center.y.to_string();
                    self.ui.docking_site_z = lig.docking_site.site_center.z.to_string();
                    self.ui.docking_site_size = lig.docking_site.site_box_size.to_string();

                    self.ligand = Some(lig);
                } else {
                    self.molecule = Some(mol);
                }

                if self.get_make_docking_setup().is_none() {
                    eprintln!("Problem making or getting docking setup.");
                }

                self.update_from_prefs();
                self.ui.new_mol_loaded = true;
            }
            Err(e) => eprintln!("Error loading file at path {path:?}: {e:?}"),
        }
    }

    /// Gets the docking setup, creating it if it doesn't exist. Returns `None` if molecule
    /// or ligand are absent.
    pub fn get_make_docking_setup(&mut self) -> Option<&DockingSetup> {
        if self.molecule.is_none() || self.ligand.is_none() {
            return None;
        }

        Some(self.volatile.docking_setup.get_or_insert_with(|| {
            DockingSetup::new(
                self.molecule.as_ref().unwrap(),
                self.ligand.as_mut().unwrap(),
                &self.volatile.lj_lookup_table,
                &self.bh_config,
            )
        }))
    }
}

fn main() {
    // #[cfg(feature = "cuda")]
    // let dev = {
    //     // This is compiled in `build_`.
    //     let ctx = CudaContext::new(0).unwrap();
    //     let stream = ctx.default_stream();
    //
    //     let module = ctx.load_module(Ptx::from_file("./cuda.ptx")).unwrap();
    //
    //     // todo: Store/cache these, likely.
    //     // let func_coulomb = module.load_function("coulomb_kernel").unwrap();
    //     // let func_lj_V = module.load_function("lj_V_kernel").unwrap();
    //     // let func_lj_force = module.load_function("lj_force_kernel").unwrap();
    //
    //     // println!("Using the GPU for computations.");
    //     ComputationDevice::Gpu((stream, module))
    // };

    // #[cfg(not(feature = "cuda"))]
    let dev = ComputationDevice::Cpu;

    // let runtime_v = cudarc::runtime::result::version::get_runtime_version();
    // let driver_v = cudarc::runtime::result::version::get_driver_version();
    // println!("CUDA runtime: {runtime_v:?}");
    // println!("CUDA driver: {driver_v:?}");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            println!("AVX-512 is available");
        } else if is_x86_feature_detected!("avx") {
            println!("AVX (256-bit) is available");
        } else {
            println!("AVX is not available.");
        }
    }

    // Sets up write-once static muts.
    init_local_bond_vecs();

    let mut state = State::default();

    state.bh_config.θ = THETA_BH;

    // todo: Consider a custom default impl.
    state.ui.view_depth = VIEW_DEPTH_MAX;
    state.ui.new_mol_loaded = true;

    state.load_prefs();

    let last_opened = state.to_save.last_opened.clone();
    if let Some(path) = &last_opened {
        state.open_molecule(path, false);
    }

    let last_ligand_opened = state.to_save.last_ligand_opened.clone();
    if let Some(path) = &last_ligand_opened {
        state.open_molecule(path, true);
    }

    // todo: Not the ideal place, but having double-borrow errors when doing it on-demand.
    if state.get_make_docking_setup().is_none() {
        eprintln!("Problem making or getting docking setup.");
    }

    if let Some(path) = &state.to_save.autodock_vina_path {
        state.ui.autodock_path_valid = check_adv_avail(path);

        // If the saved path fails our check, leave it blank so the user can re-attempt.
        if !state.ui.autodock_path_valid {
            state.to_save.autodock_vina_path = None;
            state.update_save_prefs();
        }
    }

    render(state);
}
