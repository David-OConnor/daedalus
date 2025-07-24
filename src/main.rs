// #![cfg_attr(
//     all(not(debug_assertions), target_os = "windows"),
//     windows_subsystem = "windows"
// )]

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

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
mod bond_inference;
mod docking;
mod download_mols;
mod drug_like;
mod file_io;
mod forces;
mod inputs;
mod mol_drawing;
mod molecule;
mod navigation;
mod prefs;
mod render;
mod ribbon_mesh;
mod sa_surface;
mod save_load;
mod ui;
mod util;

mod cli;
mod dynamics;
mod integrate;
mod reflection;

mod ui_aux;

#[cfg(test)]
mod tests;

use std::{
    collections::HashMap,
    env, fmt, io,
    io::ErrorKind,
    path::{Path, PathBuf},
    str::FromStr,
    sync::{Arc, mpsc::Receiver},
};

use barnes_hut::BhConfig;
use bincode::{Decode, Encode};
use bio_apis::{
    ReqError, rcsb,
    rcsb::{FilesAvailable, PdbDataResults},
};
use bio_files::amber_params::{ChargeParams, ForceFieldParams, ForceFieldParamsKeyed};
// #[cfg(feature = "cuda")]
// use cuda_setup::ComputationDevice;
#[cfg(feature = "cuda")]
use cudarc::{
    driver::{CudaContext, CudaModule, CudaStream},
    nvrtc::Ptx,
};
use egui::RichText;
use egui_file_dialog::{FileDialog, FileDialogConfig};
// use file_io::cif_pdb::load_cif_pdb;
use graphics::{Camera, InputsCommanded};
use lin_alg::{
    f32::{Quaternion, Vec3},
    f64::Vec3 as Vec3F64,
};
use mol_drawing::MoleculeView;
use molecule::Molecule;
use na_seq::{
    AminoAcid, AminoAcidGeneral, Element,
    element::{LjTable, init_lj_lut},
};

use crate::{
    aa_coords::bond_vecs::init_local_bond_vecs,
    docking::{
        BindingEnergy, ConformationType, THETA_BH, external::check_adv_avail, prep::DockingSetup,
    },
    dynamics::MdState,
    file_io::{mtz::load_mtz, pdbqt::load_pdbqt},
    molecule::{Ligand, PeptideAtomPosits},
    navigation::Tab,
    prefs::ToSave,
    render::render,
    ui::{COL_SPACING, VIEW_DEPTH_FAR_MAX, VIEW_DEPTH_NEAR_MIN},
    util::handle_err,
};
// Include general Amber forcefield params with our program. See the Reference Manual, section ]
// 3.1.1 for details on which we include. (The recommended ones for Proteins, and ligands).

// Proteins and amino acids:
const PARM_19: &str = include_str!("../resources/parm19.dat"); // Bonded, and Van der Waals.
const FRCMOD_FF19SB: &str = include_str!("../resources/frcmod.ff19SB"); // Bonded, and Van der Waals: overrides and new types
const AMINO_19: &str = include_str!("../resources/amino19.lib"); // Charge; internal residues
const AMINO_NT12: &str = include_str!("../resources/aminont12.lib"); // Charge; protonated N-terminus residues
const AMINO_CT12: &str = include_str!("../resources/aminoct12.lib"); // Charge; protonated C-terminus residues

// Ligands/small organic molecules: *General Amber Force Fields*.
const GAFF2: &str = include_str!("../resources/gaff2.dat");

// Note: Water parameters are concise; we store them directly.

// todo: Eventually, implement a system that automatically checks for changes, and don't
// todo save to disk if there are no changes.
const PREFS_SAVE_INTERVAL: u64 = 60; // Save user preferences this often, in seconds.

pub type ProtFfMap = HashMap<AminoAcidGeneral, Vec<ChargeParams>>;

#[derive(Debug, Clone, Default)]
pub enum ComputationDevice {
    #[default]
    Cpu,
    #[cfg(feature = "cuda")]
    Gpu((Arc<CudaStream>, Arc<CudaModule>)),
}

#[derive(Clone, Copy, PartialEq, Debug, Default, Encode, Decode)]
pub enum ViewSelLevel {
    #[default]
    Atom,
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
    save: FileDialog,
    autodock_path: FileDialog,
}

impl Default for FileDialogs {
    fn default() -> Self {
        let cfg_all = FileDialogConfig::default()
            .add_file_filter_extensions(
                "All",
                vec![
                    "cif", "mol2", "sdf", "pdbqt", "map", "mtz", "frcmod", "dat",
                ],
            )
            .add_file_filter_extensions("Molecule", vec!["cif", "mol2", "sdf", "pdbqt"])
            .add_file_filter_extensions("Protein", vec!["cif"])
            .add_file_filter_extensions("Small mol", vec!["mol2", "sdf", "pdbqt"])
            .add_file_filter_extensions("Density", vec!["map", "mtz", "cif"])
            .add_file_filter_extensions("Mol dynamics", vec!["frcmod", "dat"])
            .add_save_extension("CIF", "cif")
            .add_save_extension("Mol2", "mol2")
            .add_save_extension("SDF", "sdf")
            .add_save_extension("Pdbqt", "pdbqt")
            .add_save_extension("Map", "map");

        let cfg_vina = FileDialogConfig {
            ..Default::default()
        }
        .add_file_filter_extensions("Executables", vec!["", "exe"]);

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

/// Flags to accomplish things that must be done somewhere with access to `Scene`.
#[derive(Default)]
struct SceneFlags {
    /// Secondary structure
    pub update_ss_mesh: bool,
    /// Solvent-accessible surface.
    pub update_sas_mesh: bool,
    pub ss_mesh_created: bool,
    pub sas_mesh_created: bool,
    pub make_density_mesh: bool,
    pub clear_density_drawing: bool,
    pub new_density_loaded: bool,
    pub new_mol_loaded: bool,
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
    lj_lookup_table: LjTable,
    // snapshots: Vec<Snapshot>,
    docking_setup: Option<DockingSetup>,
    /// e.g. waiting for the data avail thread to return
    mol_pending_data_avail: Option<
        Receiver<(
            Result<PdbDataResults, ReqError>,
            Result<FilesAvailable, ReqError>,
        )>,
    >,
    /// We may change CWD during CLI navigation; keep prefs directory constant.
    prefs_dir: PathBuf,
    /// Entered by the user, for this session.
    cli_input_history: Vec<String>,
    cli_input_selected: usize,
    /// Pre-computed from the molecule
    aa_seq_text: String,
    flags: SceneFlags,
}

impl Default for StateVolatile {
    fn default() -> Self {
        Self {
            dialogs: Default::default(),
            ui_height: Default::default(),
            inputs_commanded: Default::default(),
            lj_lookup_table: init_lj_lut(),
            // snapshots: Default::default(),
            docking_setup: Default::default(),
            mol_pending_data_avail: Default::default(),
            prefs_dir: env::current_dir().unwrap(),
            cli_input_history: Default::default(),
            cli_input_selected: Default::default(),
            aa_seq_text: Default::default(),
            flags: Default::default(),
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
    hide_density: bool,
    hide_density_surface: bool,
    // todo: Seq here, or not?
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
            hide_density: false,
            hide_density_surface: false,
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
    pub fn to_str(self) -> String {
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
    atom_res_search: String,
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
    selection: Selection,
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
    show_docking_tools: bool,
    show_settings: bool,
    movement_speed_input: String,
    rotation_sens_input: String,
    cmd_line_input: String,
    cmd_line_output: String,
    /// Indicates CLI, or errors more broadly by changing its displayed color.
    cmd_line_out_is_err: bool,
    show_aa_seq: bool,
    /// Use a viridis or simialar colr scheme to color residues gradually based on their
    /// position in the sequence.
    res_color_by_index: bool,
    atom_color_by_charge: bool,
    /// Affects the electron density mesh.
    density_iso_level: f32,
    /// E.g. set to original for from the mmCIF file, or Dynamics to view it after MD.
    peptide_atom_posits: PeptideAtomPosits,
    num_md_steps: u32,
}

#[derive(Clone, PartialEq, Debug, Default, Encode, Decode)]
pub enum Selection {
    #[default]
    None,
    /// Of the protein
    Atom(usize),
    /// Of the protein
    Residue(usize),
    /// Of the protein
    Atoms(Vec<usize>),
    AtomLigand(usize),
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

/// Maps type-in-residue (found in, e.g. mmCIF and PDB files) to Amber FF type, and partial charge.
/// We assume that if one of these is loaded, so are the others. So, these aren't `Options`s, but
/// the field that holds this struct should be one.
pub struct ProtFFTypeChargeMap {
    pub internal: ProtFfMap,
    pub n_terminus: ProtFfMap,
    pub c_terminus: ProtFfMap,
}

#[derive(Default)]
/// Force field parameters (e.g. Amber) for molecular dynamics.
pub struct FfParamSet {
    /// E.g. parsed from Amber `gaff2.dat`.
    pub lig_general: Option<ForceFieldParamsKeyed>,
    /// E.g. ff19SB. Loaded at init.
    pub prot_general: Option<ForceFieldParamsKeyed>,
    /// In addition to charge, this also contains the mapping of res type to FF type; required to map
    /// other parameters to protein atoms. From `amino19.lib`, and its N and C-terminus variants.
    pub prot_ff_q_map: Option<ProtFFTypeChargeMap>,
    /// Key: A unique identifier for the molecule. (e.g. ligand)
    pub lig_specific: HashMap<String, ForceFieldParamsKeyed>,
}

#[derive(Default)]
struct State {
    pub ui: StateUi,
    pub volatile: StateVolatile,
    // pub pdb: Option<PDB>,
    pub cif_pdb_raw: Option<String>,
    pub molecule: Option<Molecule>,
    pub ligand: Option<Ligand>,
    pub cam_snapshots: Vec<CamSnapshot>,
    /// This allows us to keep in-memory data for other molecules.
    pub to_save: ToSave,
    pub tabs_open: Vec<Tab>,
    pub babel_avail: bool,
    pub docking_ready: bool,
    pub bh_config: BhConfig,
    pub dev: ComputationDevice,
    pub mol_dynamics: Option<MdState>,
    // todo: Combine these params in a single struct.
    pub ff_params: FfParamSet,
}

impl State {
    /// E.g. when loading a new molecule.
    pub fn reset_selections(&mut self) {
        self.ui.selection = Selection::None;
        self.cam_snapshots = Vec::new();
        self.ui.cam_snapshot = None;
        self.ui.chain_to_pick_res = None;
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
            lig.position_atoms(None);

            self.ui.docking_site_x = posit.x.to_string();
            self.ui.docking_site_y = posit.y.to_string();
            self.ui.docking_site_z = posit.z.to_string();

            // todo: Make sure this isn't too computationally intensive to put here.
            if let Some(mol) = &self.molecule {
                self.volatile.docking_setup = Some(DockingSetup::new(
                    mol,
                    lig,
                    &self.volatile.lj_lookup_table,
                    &self.bh_config,
                ));
            }
        }
    }
}

fn main() {
    #[cfg(feature = "cuda")]
    let _dev = {
        let runtime_v = cudarc::runtime::result::version::get_runtime_version();
        let driver_v = cudarc::runtime::result::version::get_driver_version();
        println!("CUDA runtime: {runtime_v:?}. Driver: {driver_v:?}");

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
    let _dev = ComputationDevice::Cpu;

    // todo For now. GPU currently is going slower than CPU for VDW.
    let dev = ComputationDevice::Cpu;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            println!("AVX-512 is available\n");
        } else if is_x86_feature_detected!("avx") {
            println!("AVX (256-bit) is available\n");
        } else {
            println!("AVX is not available.\n");
        }
    }

    // Sets up write-once static muts.
    init_local_bond_vecs();

    // todo: Consider a custom default impl. This is a substitute.
    let mut state = State {
        dev,
        bh_config: BhConfig {
            θ: THETA_BH,
            ..Default::default()
        },
        ui: StateUi {
            view_depth: (VIEW_DEPTH_NEAR_MIN, VIEW_DEPTH_FAR_MAX),
            nearby_dist_thresh: 15,
            density_iso_level: 1.8,
            num_md_steps: 10_000,
            ..Default::default()
        },
        ..Default::default()
    };

    // todo: Consider if you want this here. Currently required when adding H to a molecule.
    // In release mode, takes 20ms on a fast CPU. (todo: Test on a slow CPU.)
    state.load_ffs_general();
    state.load_prefs();

    let last_opened = state.to_save.last_opened.clone();
    if let Some(path) = &last_opened {
        if let Err(e) = state.open_molecule(path) {
            handle_err(&mut state.ui, e.to_string());
        }
    }

    // Load map after molecule, so it knows the coordinates.
    let last_map_opened = state.to_save.last_map_opened.clone();
    if let Some(path) = &last_map_opened {
        if let Err(e) = state.open(path) {
            handle_err(&mut state.ui, e.to_string());
        }
    }

    let last_ligand_opened = state.to_save.last_ligand_opened.clone();
    if let Some(path) = &last_ligand_opened {
        if let Err(e) = state.open_molecule(path) {
            handle_err(&mut state.ui, e.to_string());
        }
    }

    // Update ligand positions, e.g. from the docking position site center loaded from prefs.
    if let Some(lig) = &mut state.ligand {
        lig.pose.anchor_posit = lig.docking_site.site_center;
        lig.position_atoms(None);
    }

    if let Some(mol) = &state.molecule {
        let posit = state.to_save.per_mol[&mol.ident].docking_site.site_center;
        state.update_docking_site(posit);
    }

    if let Err(e) = state.load_aa_charges_ff() {
        handle_err(
            &mut state.ui,
            format!("Unable to load protein charges (static): {e}"),
        );
    }

    // todo temp
    // state
    //     .open(&PathBuf::from_str("molecules/CPB.frcmod").unwrap())
    //     .unwrap();

    // todo temp
    // let mtz = load_mtz(&PathBuf::from_str("../../../Desktop/1fat_2fo.mtz").unwrap());
    // println!("MTZ: {:?}", mtz);

    render(state);
}
