// #![cfg_attr(
//     all(not(debug_assertions), target_os = "windows"),
//     windows_subsystem = "windows"
// )]

#![allow(clippy::too_many_arguments)]

mod aa_coords;
mod add_hydrogens;
mod amino_acid_coords;
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
mod surface;
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
use egui::RichText;
use egui_file_dialog::{FileDialog, FileDialogConfig};
use file_io::{pdb::load_pdb, sdf::load_sdf};
use graphics::{Camera, InputsCommanded};
use lin_alg::{
    f32::{Quaternion, Vec3, f32x8},
    f64::Vec3 as Vec3F64,
};
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
    molecule::{Ligand, ResidueType},
    navigation::Tab,
    prefs::ToSave,
    render::render,
    ui::{COL_SPACING, VIEW_DEPTH_FAR_MAX, VIEW_DEPTH_NEAR_MIN},
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
    dim_peptide: bool,
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
            dim_peptide: false,
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
    /// To selection.
    show_near_sel_only: bool,
    show_near_lig_only: bool,
    /// Angstrom. For selections, or ligand.
    nearby_dist_thresh: u16,
    view_depth: (u16, u16), // angstrom. min, max.
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
    show_docking_tools: bool,
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
    pub dev: ComputationDevice,
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
                load_pdbqt(path).map(|(molecule, mut lig_loaded)| {
                    if lig_loaded.is_some() {
                        lig_loaded.as_mut().unwrap().molecule = molecule.clone(); // sloppy
                    }
                    ligand = lig_loaded;
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
                    let het_residues = mol.het_residues.clone();
                    let mol_atoms = mol.atoms.clone();
                    self.ligand = Some(Ligand::new(mol));

                    let mut init_posit = Vec3F64::new_zero();

                    // Align to a hetero residue in the open molecule, if there is a match.
                    // todo: Keep this in sync with the UI button-based code; this will have updated.
                    for res in het_residues {
                        if (res.atoms.len() as i16
                            - self.ligand.as_ref().unwrap().molecule.atoms.len() as i16)
                            .abs()
                            < 22
                        {
                            init_posit = mol_atoms[res.atoms[0]].posit;
                        }
                    }

                    self.update_docking_site(init_posit);
                } else {
                    self.molecule = Some(mol);
                }

                self.update_from_prefs();

                if self.get_make_docking_setup().is_none() {
                    eprintln!("Problem making or getting docking setup.");
                }

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

    pub fn update_docking_site(&mut self, posit: Vec3F64) {
        if let Some(lig) = &mut self.ligand {
            lig.docking_site.site_center = posit;
            lig.pose.anchor_posit = lig.docking_site.site_center;
            lig.atom_posits = lig.position_atoms(None);

            self.ui.docking_site_x = posit.x.to_string();
            self.ui.docking_site_y = posit.y.to_string();
            self.ui.docking_site_z = posit.z.to_string();
        }
    }
}

fn main() {
    #[cfg(feature = "cuda")]
    let dev = {
        let runtime_v = cudarc::runtime::result::version::get_runtime_version();
        let driver_v = cudarc::runtime::result::version::get_driver_version();
        println!("CUDA runtime: {runtime_v:?}");
        println!("CUDA driver: {driver_v:?}");

        if runtime_v.is_ok() && driver_v.is_ok() {
            // This is compiled in `build_`.
            let ctx = CudaContext::new(0).unwrap();
            let stream = ctx.default_stream();

            let ptx_file = "./cuda.ptx";
            let module = ctx.load_module(Ptx::from_file(ptx_file));

            match module {
                Ok(m) => {
                    // todo: Store/cache these, likely.
                    // let func_coulomb = module.load_function("coulomb_kernel").unwrap();
                    // let func_lj_V = module.load_function("lj_V_kernel").unwrap();
                    // let func_lj_force = module.load_function("lj_force_kernel").unwrap();

                    ComputationDevice::Gpu((stream, m))
                }
                Err(e) => {
                    eprintln!("Error loading CUDA module: {ptx_file}; not using CUDA. Error: {e}");
                    ComputationDevice::Cpu
                }
            }
        } else {
            ComputationDevice::Cpu
        }

        // println!("Using the GPU for computations.");
    };

    #[cfg(not(feature = "cuda"))]
    let dev = ComputationDevice::Cpu;

    // todo For now. GPU currently is going slower than CPU for VDW.
    let dev = ComputationDevice::Cpu;

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

    state.dev = dev;
    state.bh_config.θ = THETA_BH;

    // todo: Consider a custom default impl. This is a substitute.
    state.ui.view_depth = (VIEW_DEPTH_NEAR_MIN, VIEW_DEPTH_FAR_MAX);
    state.ui.new_mol_loaded = true;
    state.ui.nearby_dist_thresh = 15;

    state.load_prefs();

    let last_opened = state.to_save.last_opened.clone();
    if let Some(path) = &last_opened {
        state.open_molecule(path, false);
    }

    let last_ligand_opened = state.to_save.last_ligand_opened.clone();
    if let Some(path) = &last_ligand_opened {
        state.open_molecule(path, true);
    }

    // Update ligand positions, e.g. from the docking position site center loaded from prefs.
    if let Some(lig) = &mut state.ligand {
        lig.pose.anchor_posit = lig.docking_site.site_center;
        lig.atom_posits = lig.position_atoms(None);
    }

    render(state);
}
