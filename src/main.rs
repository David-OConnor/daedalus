// #![cfg_attr(
//     all(not(debug_assertions), target_os = "windows"),
//     windows_subsystem = "windows"
// )]

#![allow(clippy::too_many_arguments)]

// Note: To test if it compiles on ARM:
// `rustup target add aarch64-pc-windows-msvc`
// `cargo check --target aarch64-pc-windows-msvc`
// note: Currently getting Clang errors when I attempt htis.

// todo: Features to add:
// - qvina2/qvina-w/gpuvina implementations too along with stuff like haddock for affinity-based
// protein-protein.
// - CLI interface or scripting, like PyMol
// - mol2 support
// - Better color scheme for residues?

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
mod render;
mod save_load;
mod surface;
mod ui;
mod util;
mod vibrations;

mod cli;
mod reflection;
#[cfg(test)]
mod tests;
// mod isosurface;

use std::{
    collections::HashMap,
    fmt, io,
    io::ErrorKind,
    path::{Path, PathBuf},
    str::FromStr,
    sync::{Arc, mpsc::Receiver},
};

use barnes_hut::BhConfig;
use bincode::{Decode, Encode};
use bio_apis::{ReqError, rcsb::DataAvailable};
// #[cfg(feature = "cuda")]
// use cuda_setup::ComputationDevice;
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
    file_io::{mol2::load_mol2, mtz::load_mtz, pdb::save_pdb, pdbqt::load_pdbqt},
    molecule::{Ligand, ResidueType},
    navigation::Tab,
    prefs::ToSave,
    render::render,
    ui::{COL_SPACING, VIEW_DEPTH_FAR_MAX, VIEW_DEPTH_NEAR_MIN},
};

// todo: Eventually, implement a system that automatically checks for changes, and don't
// todo save to disk if there are no changes.
const PREFS_SAVE_INTERVAL: u64 = 60; // Save user preferences this often, in seconds.

#[derive(Debug, Clone, Default)]
pub enum ComputationDevice {
    #[default]
    Cpu,
    #[cfg(feature = "cuda")]
    Gpu((Arc<CudaStream>, Arc<CudaModule>)),
}

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
    // load_ligand: FileDialog,
    save: FileDialog,
    // save_ligand: FileDialog,
    autodock_path: FileDialog,
    // save_pdbqt: FileDialog,
    // load_mdx: FileDialog,
    // load_crystallography: FileDialog,
}

impl Default for FileDialogs {
    fn default() -> Self {
        let cfg_all = FileDialogConfig::default()
            .add_file_filter_extensions(
                "All",
                vec!["pdb", "cif", "sdf", "mol2", "pdbqt", "map", "mtz"],
            )
            .add_file_filter_extensions("Coords", vec!["pdb", "cif", "sdf", "mol2", "pdbqt"])
            .add_file_filter_extensions("Protein", vec!["pdb", "cif"])
            .add_file_filter_extensions("Small mol", vec!["sdf", "mol2", "pdbqt"])
            .add_file_filter_extensions("Xtal", vec!["map", "mtz", "cif"])
            .add_save_extension("CIF", "cif")
            .add_save_extension("SDF", "sdf")
            .add_save_extension("Mol2", "sdf")
            .add_save_extension("Pdbqt", "pdbqt1")
            .add_save_extension("Map", "map");

        let cfg_vina = FileDialogConfig {
            ..Default::default()
        }
        .add_file_filter_extensions("Executables", vec!["", "exe"]);

        //
        // let cfg_protein = FileDialogConfig {
        //     ..Default::default()
        // }
        // .add_file_filter_extensions("PDB/CIF", vec!["pdb", "cif"]);
        //
        // let cfg_small_mol = FileDialogConfig {
        //     ..Default::default()
        // }
        // .add_file_filter_extensions("SDF/MOL2/PDBQT", vec!["sdf", "mol2", "pdbqt"]);
        //
        // let cfg_save_small_mol = FileDialogConfig {
        //     ..Default::default()
        // }
        // .add_save_extension("SDF", "sdf")
        // .add_save_extension("Mol2", "mol2");
        //
        // let cfg_crystallography = FileDialogConfig {
        //     ..Default::default()
        // }
        // .add_file_filter_extensions("Map/MTZ", vec!["map", "mtz"]);
        //
        // let cfg_save_crystallography = FileDialogConfig {
        //     ..Default::default()
        // }
        // .add_save_extension("Map", "map")
        // .add_save_extension("MTZ", "mtz");

        //
        // let cfg_save_pdbqt = FileDialogConfig {
        //     ..Default::default()
        // }
        // .add_save_extension("PDBQT", "pdbqt");
        //
        // let cfg_load_mdx = FileDialogConfig {
        //     ..Default::default()
        // }
        // .add_file_filter_extensions("MDX", vec!["mdx"]);
        //
        // let load = FileDialog::with_config(cfg_protein.clone()).default_file_filter("PDB/CIF");
        //
        // let load_ligand =
        //     FileDialog::with_config(cfg_small_mol.clone()).default_file_filter("SDF/MOL2/PDBQT");
        //
        // let load_crystallography =
        //     FileDialog::with_config(cfg_crystallography).default_file_filter("Map/MTZ");
        //
        // let save = FileDialog::with_config(cfg_protein)
        //     .add_save_extension("CIF", "cif")
        //     .default_save_extension("CIF");
        //
        // let save_ligand = FileDialog::with_config(cfg_save_small_mol).default_save_extension("SDF");
        // let save_pdbqt = FileDialog::with_config(cfg_save_pdbqt).default_save_extension("PDBQT");
        //
        // // todo: What is this?
        // let load_mdx = FileDialog::with_config(cfg_load_mdx).default_file_filter("MDX");

        let autodock_path = FileDialog::with_config(cfg_vina).default_file_filter("Executables");

        let load = FileDialog::with_config(cfg_all.clone()).default_file_filter("All");

        let save = FileDialog::with_config(cfg_all).default_save_extension("Protein");

        Self {
            load,
            // load_ligand,
            save,
            // save_ligand,
            autodock_path,
            // save_pdbqt,
            // load_mdx,
            // load_crystallography,
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
    /// e.g. waiting for the data avail thread to return
    mol_pending_data_avail: Option<Receiver<Result<DataAvailable, ReqError>>>,
}

impl Default for StateVolatile {
    fn default() -> Self {
        Self {
            dialogs: Default::default(),
            ui_height: Default::default(),
            inputs_commanded: Default::default(),
            lj_lookup_table: init_lj_lut(),
            snapshots: Default::default(),
            docking_setup: Default::default(),
            mol_pending_data_avail: Default::default(),
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

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Default, Encode, Decode)]
enum MsaaSetting {
    None = 1,
    // Two = 2, // todo: Not supported on this depth texture, but we could switch to a different one.
    #[default]
    Four = 4,
}

impl MsaaSetting {
    pub fn to_str(&self) -> String {
        match self {
            Self::None => "None",
            // Self::Two => "2×",
            Self::Four => "4×",
        }
        .to_owned()
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
    show_settings: bool,
    movement_speed_input: String,
    rotation_sens_input: String,
    cmd_line_input: String,
    cmd_line_output: String,
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

    /// A single endpoint to open a number of file types
    pub fn open(&mut self, path: &Path) -> io::Result<()> {
        match path
            .extension()
            .unwrap_or_default()
            .to_ascii_lowercase()
            .to_str()
            .unwrap_or_default()
        {
            "sdf" | "mol2" | "pdbqt" | "pdb" | "cif" => self.open_molecule(path)?,
            _ => {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    "Unsupported file extension",
                ));
            }
        }

        Ok(())
    }

    pub fn open_molecule(&mut self, path: &Path) -> io::Result<()> {
        let binding = path.extension().unwrap_or_default().to_ascii_lowercase();
        let extension = binding;

        let is_ligand = matches!(extension.to_str().unwrap(), "sdf" | "mol2");

        let mut ligand = None;
        let molecule = match extension.to_str().unwrap() {
            "sdf" => load_sdf(path),
            "mol2" => load_mol2(path),
            "pdbqt" => {
                load_pdbqt(path).map(|(molecule, mut lig_loaded)| {
                    if lig_loaded.is_some() {
                        lig_loaded.as_mut().unwrap().molecule = molecule.clone(); // sloppy
                    }
                    ligand = lig_loaded;
                    molecule
                })
            }
            "pdb" | "cif" => {
                let pdb = load_pdb(path);
                match pdb {
                    Ok(p) => {
                        let mol = Molecule::from_pdb(&p);
                        self.pdb = Some(p);
                        Ok(mol)
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

                    let mut init_posit = Vec3F64::new_zero();

                    let lig = Ligand::new(mol);

                    // Align to a hetero residue in the open molecule, if there is a match.
                    // todo: Keep this in sync with the UI button-based code; this will have updated.
                    for res in het_residues {
                        if (res.atoms.len() as i16 - lig.molecule.atoms.len() as i16).abs() < 22 {
                            init_posit = mol_atoms[res.atoms[0]].posit;
                        }
                    }

                    self.ligand = Some(lig);
                    self.to_save.last_ligand_opened = Some(path.to_owned());

                    self.update_docking_site(init_posit);
                } else {
                    self.molecule = Some(mol);
                    self.to_save.last_opened = Some(path.to_owned());
                }

                self.update_from_prefs();

                if let Some(mol) = &mut self.molecule {
                    // Only after updating from prefs (to prevent unecesasary loading) do we update data avail.
                    mol.update_data_avail(&mut self.volatile.mol_pending_data_avail);
                }

                if self.get_make_docking_setup().is_none() {
                    eprintln!("Problem making or getting docking setup.");
                }

                self.ui.new_mol_loaded = true;
            }
            Err(e) => {
                return Err(e);
            }
        }

        Ok(())
    }

    /// A single endpoint to save a number of file types
    pub fn save(&mut self, path: &Path) -> io::Result<()> {
        let binding = path.extension().unwrap_or_default().to_ascii_lowercase();
        let extension = binding;

        match extension.to_str().unwrap_or_default() {
            "pdb" | "cif" => {
                if let Some(pdb) = &mut self.pdb {
                    save_pdb(pdb, path)?;
                    self.to_save.last_opened = Some(path.to_owned());
                    self.update_save_prefs()
                }
            }
            "sdf" => match &self.ligand {
                Some(lig) => {
                    lig.molecule.save_sdf(path);
                    self.to_save.last_ligand_opened = Some(path.to_owned());
                    self.update_save_prefs()
                }
                None => return Err(io::Error::new(ErrorKind::InvalidData, "No ligand to save")),
            },
            "mol2" => match &self.ligand {
                Some(lig) => {
                    lig.molecule.save_mol2(path)?;
                    self.to_save.last_ligand_opened = Some(path.to_owned());
                    self.update_save_prefs()
                }
                None => return Err(io::Error::new(ErrorKind::InvalidData, "No ligand to save")),
            },
            "pdbqt" => match &self.ligand {
                Some(lig) => {
                    lig.molecule.save_pdbqt(path, None)?;
                    self.to_save.last_ligand_opened = Some(path.to_owned());
                    self.update_save_prefs()
                }
                None => return Err(io::Error::new(ErrorKind::InvalidData, "No ligand to save")),
            },
            "map" => {}
            _ => {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    "Unsupported file extension",
                ));
            }
        }

        Ok(())
    }

    /// Gets the docking setup, creating it if it doesn't exist. Returns `None` if molecule
    /// or ligand are absent.
    pub fn get_make_docking_setup(&mut self) -> Option<&DockingSetup> {
        let (Some(mol), Some(lig)) = (&self.molecule, &mut self.ligand) else {
            return None;
        };

        Some(self.volatile.docking_setup.get_or_insert_with(|| {
            DockingSetup::new(mol, lig, &self.volatile.lj_lookup_table, &self.bh_config)
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
        state.open_molecule(path).ok();
    }

    let last_ligand_opened = state.to_save.last_ligand_opened.clone();
    if let Some(path) = &last_ligand_opened {
        state.open_molecule(path).ok();
    }

    // Update ligand positions, e.g. from the docking position site center loaded from prefs.
    if let Some(lig) = &mut state.ligand {
        lig.pose.anchor_posit = lig.docking_site.site_center;
        lig.atom_posits = lig.position_atoms(None);
    }

    // todo temp
    // let mtz = load_mtz(&PathBuf::from_str("../../../Desktop/1fat_2fo.mtz").unwrap());
    // println!("MTZ: {:?}", mtz);

    {
        let map_path = PathBuf::from_str("../../../Desktop/reflections/1fat_2fo.map").unwrap();
        // let map_path = PathBuf::from_str("../../../Desktop/reflections/2f67_2fo.map").unwrap();
        let (hdr, dens) = file_io::map::read_map_data(&map_path).unwrap();

        println!("Map header: {:#?}", hdr);

        // for pt in &dens[0..100] {
        //     println!("{:.2?}", pt);
        // }
        if let Some(mol) = &mut state.molecule {
            mol.elec_density = Some(dens);
        }
    }

    render(state);
}
