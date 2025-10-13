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

// todo: Consider APBS: https://github.com/Electrostatics/apbs (For a type of surface visulaization?)P

//! [S3 Gemmi link](https://daedalus-mols.s3.us-east-1.amazonaws.com/gemmi.exe)
//! [S3 Geostd link](https://daedalus-mols.s3.us-east-1.amazonaws.com/amber_geostd)

mod bond_inference;
// mod docking;
mod docking_v2;
mod download_mols;
mod drawing;
mod file_io;
mod forces;
mod inputs;
mod molecule;
mod prefs;
mod render;
mod ribbon_mesh;
mod sa_surface;
mod ui;
mod util;

mod cli;
mod reflection;

mod cam_misc;
mod drawing_wrappers;
mod lipid;
mod md;
mod mol_editor;
mod mol_lig;
mod mol_manip;
mod nucleic_acid;
mod selection;
#[cfg(test)]
mod tests;

#[cfg(feature = "cuda")]
use std::sync::Arc;
use std::{
    collections::{HashMap, HashSet},
    env, fmt,
    fmt::Display,
    path::PathBuf,
    sync::mpsc::Receiver,
    time::Instant,
};

use bincode::{Decode, Encode};
use bio_apis::{
    ReqError,
    amber_geostd::GeostdItem,
    rcsb::{FilesAvailable, PdbDataResults},
};
use bio_files::md_params::{ForceFieldParams, load_lipid_templates};
#[cfg(feature = "cuda")]
use cudarc::{
    driver::{CudaContext, CudaModule, CudaStream},
    nvrtc::Ptx,
};
use drawing::MoleculeView;
use dynamics::{
    ComputationDevice, Integrator, MdState, SimBoxInit,
    params::{FfParamSet, LIPID_21_LIB},
};
use egui_file_dialog::{FileDialog, FileDialogConfig};
use graphics::{Camera, ControlScheme, InputsCommanded, winit::event::Modifiers};
use lin_alg::{
    f32::{Quaternion, Vec3},
    f64::Vec3 as Vec3F64,
};
use mol_lig::{Ligand, MoleculeSmall};
use molecule::MoleculePeptide;

use crate::{
    lipid::{LipidShape, MoleculeLipid},
    mol_editor::MolEditorState,
    molecule::{Bond, MoGenericRefMut, MolGenericRef, MolType, MoleculeCommon},
    nucleic_acid::MoleculeNucleicAcid,
    prefs::ToSave,
    render::render,
    ui::cam::{FOG_DIST_DEFAULT, FOG_DIST_MAX, FOG_DIST_MIN, VIEW_DEPTH_NEAR_MIN},
    util::handle_err,
};
// ------Including files into the executable

// Note: If you haven't generated this file yet when compiling (e.g. from a freshly-cloned repo),
// make an edit to one of the CUDA files (e.g. add a newline), then run, to create this file.
#[cfg(feature = "cuda")]
const PTX: &str = include_str!("../daedalus.ptx");

// Note: Water parameters are concise; we store them directly.

// ------ End file includes.

// todo: Eventually, implement a system that automatically checks for changes, and don't
// todo save to disk if there are no changes.
const PREFS_SAVE_INTERVAL: u64 = 60; // Save user preferences this often, in seconds.

/// The MdModule is owned by `dynamics::ComputationDevice`.
#[cfg(feature = "cuda")]
struct CudaModules {
    /// For processing as part of loading electron density data
    pub reflections: Arc<CudaModule>,
}

// /// This wraps `dyanmics::ComputationDevice`. It's a bit awkard, but for now
// /// allows Dynamics to own ComputationDev with the MD model. We add our additional
// /// models here. Note: The CudaStream is owned by the inner `dynamics::ComputationDevice`.
// enum ComputationDevOuter {
//     Cpu,
//     #[cfg(feature = "cuda")]
//     Gpu((ComputationDevice, Arc<CudaModule>>)),
// }

#[derive(Clone, Copy, PartialEq, Debug, Default, Encode, Decode)]
pub enum ViewSelLevel {
    #[default]
    Atom,
    Residue,
}

impl Display for ViewSelLevel {
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
    // todo: Add these A/R.
    // load_editor: FileDialog,
    // save_editor: FileDialog,
}

impl Default for FileDialogs {
    fn default() -> Self {
        let cfg_all = FileDialogConfig::default()
            .add_file_filter_extensions(
                "All",
                vec![
                    "cif", "mol2", "sdf", "pdbqt", "map", "mtz", "frcmod", "dat", "prmtop",
                ],
            )
            .add_file_filter_extensions("Molecule (small)", vec!["mol2", "sdf", "pdbqt", "prmtop"])
            .add_file_filter_extensions("Protein (CIF)", vec!["cif"])
            .add_file_filter_extensions("Density", vec!["map", "mtz", "cif"])
            .add_file_filter_extensions("Mol dynamics", vec!["frcmod", "dat", "lib", "prmtop"])
            //
            .add_file_filter_extensions("Molecule (small)", vec!["mol2", "sdf", "pdbqt", "prmtop"])
            .add_save_extension("Protein (CIF)", "cif")
            .add_save_extension("Mol2", "mol2")
            .add_save_extension("SDF", "sdf")
            .add_save_extension("Pdbqt", "pdbqt")
            .add_save_extension("Map", "map")
            .add_save_extension("MTZ", "mtz")
            .add_save_extension("Prmtop", "prmtop");

        let cfg_vina = FileDialogConfig {
            ..Default::default()
        }
        .add_file_filter_extensions("Executables", vec!["", "exe"]);

        let autodock_path = FileDialog::with_config(cfg_vina).default_file_filter("Executables");

        let load = FileDialog::with_config(cfg_all.clone()).default_file_filter("All");
        let save = FileDialog::with_config(cfg_all).default_save_extension("Protein");

        Self {
            load,
            save,
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
    pub make_density_iso_mesh: bool,
    pub clear_density_drawing: bool,
    pub new_density_loaded: bool,
    pub new_mol_loaded: bool,
}

#[derive(Clone, Copy, Default, PartialEq)]
pub enum ManipMode {
    #[default]
    None,
    Move((MolType, usize)), // Index of mol
    Rotate((MolType, usize)),
}

/// State for dragging and rotating molecules.
#[derive(Default)]
struct MolManip {
    /// Allows the user to move a molecule around with mouse or keyboard.
    mol: ManipMode,
    /// For maintaining the screen plane when dragging the mol.
    pivot: Option<Vec3>,
    view_dir: Option<Vec3>,
    offset: Vec3,
    depth_bias: f32,
}

// todo: Rename A/R
#[derive(Clone, Copy, PartialEq, Default)]
pub enum OperatingMode {
    #[default]
    Primary,
    /// For editing small molecules
    MolEditor,
}

/// Temporary, and generated state.
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
    // todo: Replace with the V2 version A/R
    // docking_setup: Option<DockingSetup>,
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
    /// Cached so we don't compute each UI paint. Picoseconds.
    md_runtime: f32,
    active_mol: Option<(MolType, usize)>,
    mol_manip: MolManip,
    /// For restoring after temprarily disabling mouse look.
    control_scheme_prev: ControlScheme,
    /// todo: Experimenting with a dynamic fog distance based on the nearest object to the camera.
    /// Angstrom.
    nearest_mol_dist_to_cam: Option<f32>,
    /// We maintain a set of atom indices of peptides that are used in MD. This for example, might
    /// exclude hetero atoms and atoms not near a docking site. (mol i, atom i)
    md_peptide_selected: HashSet<(usize, usize)>,
    /// Ctrl, alt, shift etc.
    key_modifiers: Modifiers,
    operating_mode: OperatingMode,
    /// Allows restoring after entering the mol edit mode.
    primary_mode_cam: Camera,
}

impl Default for StateVolatile {
    fn default() -> Self {
        Self {
            dialogs: Default::default(),
            ui_height: Default::default(),
            inputs_commanded: Default::default(),
            mol_pending_data_avail: Default::default(),
            prefs_dir: env::current_dir().unwrap(),
            cli_input_history: Default::default(),
            cli_input_selected: Default::default(),
            aa_seq_text: Default::default(),
            flags: Default::default(),
            md_runtime: Default::default(),
            active_mol: Default::default(),
            mol_manip: Default::default(),
            control_scheme_prev: Default::default(),
            nearest_mol_dist_to_cam: Default::default(),
            md_peptide_selected: Default::default(),
            key_modifiers: Default::default(),
            operating_mode: Default::default(),
            primary_mode_cam: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Encode, Decode)]
struct Visibility {
    hide_sidechains: bool,
    hide_water: bool,
    /// Hide hetero atoms: i.e. ones not part of a polypeptide.
    hide_hetero: bool,
    hide_protein: bool,
    hide_ligand: bool,
    hide_nucleic_acids: bool,
    hide_lipids: bool,
    hide_hydrogen: bool,
    hide_h_bonds: bool,
    dim_peptide: bool,
    hide_density_point_cloud: bool,
    hide_density_surface: bool,
}

impl Default for Visibility {
    fn default() -> Self {
        Self {
            hide_sidechains: false,
            hide_water: false,
            hide_hetero: false,
            hide_protein: false,
            hide_ligand: false,
            hide_nucleic_acids: false,
            hide_lipids: false,
            hide_hydrogen: true,
            hide_h_bonds: false,
            dim_peptide: false,
            hide_density_point_cloud: false,
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

#[derive(Default)]
struct PopupState {
    show_get_geostd: bool,
    show_associated_structures: bool,
    show_settings: bool,
    get_geostd_items: Vec<GeostdItem>,
    residue_selector: bool,
    rama_plot: bool,
}

struct StateUiMd {
    /// The state we store for this is a float, so we need to store state text too.
    dt_input: String,
    temp_input: String,
    pressure_input: String,
    simbox_pad_input: String,
    langevin_γ: String,
    /// Only perform MD on peptide atoms near a ligand.
    peptide_only_near_ligs: bool,
    /// Peptide atoms don't move, but exert forces.
    peptide_static: bool,
}

impl Default for StateUiMd {
    fn default() -> Self {
        Self {
            dt_input: Default::default(),
            temp_input: Default::default(),
            pressure_input: Default::default(),
            simbox_pad_input: Default::default(),
            langevin_γ: Default::default(),
            peptide_only_near_ligs: true,
            peptide_static: true,
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
    atom_res_search: String,
    /// To selection.
    show_near_sel_only: bool,
    show_near_lig_only: bool,
    /// Angstrom. For selections, or ligand.
    nearby_dist_thresh: u16,
    view_depth: (u16, u16), // angstrom. min, max.
    cam_snapshot: Option<usize>,
    dt_render: f32, // Seconds
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
    // todo: Re-implement A/R
    // binding_energy_disp: Option<BindingEnergy>,
    current_snapshot: usize,
    /// A flag so we know to update the flashlight upon loading a new model; this should be done within
    /// a callback.
    show_docking_tools: bool,
    movement_speed_input: String,
    rotation_sens_input: String,
    mol_move_sens_input: String, // scroll
    cmd_line_input: String,
    cmd_line_output: String,
    /// Indicates CLI, or errors more broadly by changing its displayed color.
    cmd_line_out_is_err: bool,
    ui_vis: UiVisibility,
    /// Use a viridis or simialar colr scheme to color residues gradually based on their
    /// position in the sequence.
    res_color_by_index: bool,
    atom_color_by_charge: bool,
    /// Affects the electron density mesh.
    density_iso_level: f32,
    // /// E.g. set to original for from the mmCIF file, or Dynamics to view it after MD.
    // peptide_atom_posits: PeptideAtomPosits,
    popup: PopupState,
    md: StateUiMd,
    ph_input: String,
    /// For the combo box. Stays at 0 if none loaded.
    lipid_to_add: usize,
    lipid_shape: LipidShape,
    lipid_mol_count: u16,
}

/// For showing and hiding UI sections.
pub struct UiVisibility {
    aa_seq: bool,
    lipids: bool,
    dynamics: bool,
}

impl Default for UiVisibility {
    fn default() -> Self {
        Self {
            aa_seq: false,
            lipids: false,
            dynamics: true,
        }
    }
}

#[derive(Clone, PartialEq, Debug, Default, Encode, Decode)]
pub enum Selection {
    #[default]
    None,
    /// Of the protein
    AtomPeptide(usize),
    /// Of the protein
    Residue(usize),
    /// Of the protein
    Atoms(Vec<usize>),
    /// Molecule index, atom index
    AtomLig((usize, usize)),
    /// Molecule index, atom index
    AtomNucleicAcid((usize, usize)),
    /// Molecule index, atom index
    AtomLipid((usize, usize)),
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

struct State {
    pub ui: StateUi,
    pub volatile: StateVolatile,
    pub cif_pdb_raw: Option<String>,
    // todo: Allow multiple?
    pub peptide: Option<MoleculePeptide>,
    pub ligands: Vec<MoleculeSmall>,
    pub nucleic_acids: Vec<MoleculeNucleicAcid>,
    pub lipids: Vec<MoleculeLipid>,
    pub cam_snapshots: Vec<CamSnapshot>,
    /// This allows us to keep in-memory data for other molecules.
    pub to_save: ToSave,
    // #[cfg(feature = "cuda")]
    // pub dev: (ComputationDevice, Option<Arc<CudaModule>>),
    pub dev: ComputationDevice,
    /// This is None if Computation Device is CPU.
    #[cfg(feature = "cuda")]
    pub cuda_modules: Option<CudaModules>,
    pub mol_dynamics: Option<MdState>,
    // todo: Combine these params in a single struct.
    pub ff_param_set: FfParamSet,
    pub lig_specific_params: HashMap<String, ForceFieldParams>,
    /// Common lipid types, e.g. as derived from Amber's `lipids21.lib`, but perhaps not exclusively.
    /// These are loaded at init; there will be one of each type.
    pub lipid_templates: Vec<MoleculeLipid>,
    pub mol_editor: MolEditorState,
}

impl Default for State {
    fn default() -> Self {
        // Many other UI defaults are loaded after the initial prefs load in `main`.
        let ui = StateUi {
            view_depth: (VIEW_DEPTH_NEAR_MIN, FOG_DIST_DEFAULT),
            nearby_dist_thresh: 15,
            density_iso_level: 1.8,
            lipid_mol_count: 10,
            ..Default::default()
        };

        Self {
            ui,
            volatile: Default::default(),
            cif_pdb_raw: Default::default(),
            peptide: Default::default(),
            ligands: Default::default(),
            nucleic_acids: Default::default(),
            lipids: Default::default(),
            cam_snapshots: Default::default(),
            to_save: Default::default(),
            dev: Default::default(),
            #[cfg(feature = "cuda")]
            cuda_modules: None,
            mol_dynamics: Default::default(),
            ff_param_set: Default::default(),
            lig_specific_params: Default::default(),
            lipid_templates: Default::default(),
            mol_editor: Default::default(),
        }
    }
}

impl State {
    /// E.g. when loading a new molecule.
    pub fn reset_selections(&mut self) {
        self.ui.selection = Selection::None;
        self.cam_snapshots = Vec::new();
        self.ui.cam_snapshot = None;
        self.ui.chain_to_pick_res = None;
    }

    // todo: Re-implement with v2 A/R
    // /// Gets the docking setup, creating it if it doesn't exist. Returns `None` if molecule
    // /// or ligand are absent.
    // pub fn get_make_docking_setup(&mut self) -> Option<&DockingSetup> {
    //     None
    //     let (Some(mol), Some(lig)) = (&self.molecule, &mut self.ligand) else {
    //         return None;
    //     };
    //
    //     Some(self.volatile.docking_setup.get_or_insert_with(|| {
    //         DockingSetup::new(mol, lig, &self.volatile.lj_lookup_table, &self.bh_config)
    //     }))
    // }

    pub fn update_docking_site(&mut self, posit: Vec3F64) {
        // if let Some(lig) = &mut self.ligand {
        //     if let Some(data) = &mut lig.lig_data {
        //         data.docking_site.site_center = posit;
        //
        //         self.ui.docking_site_x = posit.x.to_string();
        //         self.ui.docking_site_y = posit.y.to_string();
        //         self.ui.docking_site_z = posit.z.to_string();
        //     }
        // }
    }

    /// Helper
    pub fn active_mol(&self) -> Option<MolGenericRef<'_>> {
        match self.volatile.active_mol {
            Some((mol_type, i)) => match mol_type {
                MolType::Peptide => None,
                MolType::Ligand => {
                    if i < self.ligands.len() {
                        Some(MolGenericRef::Ligand(&self.ligands[i]))
                    } else {
                        None
                    }
                }
                MolType::NucleicAcid => {
                    if i < self.nucleic_acids.len() {
                        Some(MolGenericRef::NucleicAcid(&self.nucleic_acids[i]))
                    } else {
                        None
                    }
                }
                MolType::Lipid => {
                    if i < self.lipids.len() {
                        Some(MolGenericRef::Lipid(&self.lipids[i]))
                    } else {
                        None
                    }
                }
                _ => None,
            },
            None => None,
        }
    }

    /// Helper
    /// todo: DRy with the non-mutable variant.
    pub fn active_mol_mut(&mut self) -> Option<MoGenericRefMut<'_>> {
        match self.volatile.active_mol {
            Some((mol_type, i)) => match mol_type {
                MolType::Peptide => None,
                MolType::Ligand => {
                    if i < self.ligands.len() {
                        Some(MoGenericRefMut::Ligand(&mut self.ligands[i]))
                    } else {
                        None
                    }
                }
                MolType::NucleicAcid => {
                    if i < self.nucleic_acids.len() {
                        Some(MoGenericRefMut::NucleicAcid(&mut self.nucleic_acids[i]))
                    } else {
                        None
                    }
                }
                MolType::Lipid => {
                    if i < self.lipids.len() {
                        Some(MoGenericRefMut::Lipid(&mut self.lipids[i]))
                    } else {
                        None
                    }
                }
                _ => None,
            },
            None => None,
        }
    }

    /// Create lipid molecules from Amber's Lipids21.lib, which is included in the binary.
    pub fn load_lipid_templates(&mut self) {
        println!("Loading lipid templates...");
        let start = Instant::now();
        match load_lipid_templates(LIPID_21_LIB) {
            Ok(l) => {
                self.lipid_templates = Vec::new();
                for (ident, (atoms, bonds)) in l {
                    // todo: Move this to molecule mod A/R, e.g. lipid mod.
                    let mut mol = MoleculeLipid {
                        // t
                        common: MoleculeCommon {
                            ident,
                            ..Default::default()
                        },
                        lmsd_id: String::new(),
                        hmdb_id: String::new(),
                        kegg_id: String::new(),
                        common_name: String::new(),
                        residues: Vec::new(),
                    };
                    for atom in atoms {
                        mol.common.atoms.push((&atom).try_into().unwrap());
                    }

                    for bond in bonds {
                        mol.common
                            .bonds
                            .push(Bond::from_generic(&bond, &mol.common.atoms).unwrap());
                    }

                    mol.common.build_adjacency_list();
                    mol.common.atom_posits = mol.common.atoms.iter().map(|a| a.posit).collect();

                    mol.populate_db_ids();

                    self.lipid_templates.push(mol);
                }

                self.lipid_templates
                    .sort_by_key(|mol| mol.common.ident.clone());
            }
            Err(e) => {
                handle_err(&mut self.ui, format!("Unable to load lipid templates: {e}"));
            }
        };

        let elapsed = start.elapsed().as_millis();
        println!("Loaded lipid templates in {elapsed:.1}ms");
    }
}

fn main() {
    #[cfg(feature = "cuda")]
    let mut module_reflections = None;
    #[cfg(feature = "cuda")]
    let dev = {
        if cudarc::driver::result::init().is_ok() {
            // This is compiled in `build_`.
            let ctx = CudaContext::new(0).unwrap();
            let stream = ctx.default_stream();

            // todo: Figure out how to handle multiple modules, given you've moved the ComputationDevice
            // todo struct to dynamics. Your reflections GPU code will be broken until this is solved.
            let module_dynamics = ctx.load_module(Ptx::from_src(dynamics::PTX));

            match module_dynamics {
                Ok(m) => {
                    module_reflections = Some(ctx.load_module(Ptx::from_src(PTX)).unwrap());
                    // todo: Store/cache these, likely.
                    // let func_coulomb = module.load_function("coulomb_kernel").unwrap();
                    // let func_lj_V = module.load_function("lj_V_kernel").unwrap();
                    // let func_lj_force = module.load_function("lj_force_kernel").unwrap();

                    // (ComputationDevice::Gpu((stream, m)), Some(module.unwrap()))
                    ComputationDevice::Gpu((stream, m))
                }
                Err(e) => {
                    eprintln!(
                        "Error loading CUDA module: {}; not using CUDA. Error: {e}",
                        dynamics::PTX
                    );
                    // (ComputationDevice::Cpu, None)
                    ComputationDevice::Cpu
                }
            }
        } else {
            // (ComputationDevice::Cpu, None)
            ComputationDevice::Cpu
        }
    };

    #[cfg(not(feature = "cuda"))]
    let dev = ComputationDevice::Cpu;
    // let dev = ComputationDevice::Cpu;

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

    println!("Using computing device: {:?}\n", dev);

    // todo: Consider a custom default impl. This is a substitute.
    let mut state = State {
        dev,
        ..Default::default()
    };

    #[cfg(feature = "cuda")]
    if let Some(m) = module_reflections {
        state.cuda_modules = Some(CudaModules { reflections: m });
    }

    // todo: Consider if you want this here. Currently required when adding H to a molecule.
    // In release mode, takes 20ms on a fast CPU. (todo: Test on a slow CPU.)
    println!("Loading force field data from Amber lib/dat/frcmod...");
    let start = Instant::now();
    match FfParamSet::new_amber() {
        Ok(f) => {
            state.ff_param_set = f;
            let elapsed = start.elapsed().as_millis();
            println!("Loaded data in {elapsed}ms");
        }
        Err(e) => {
            handle_err(
                &mut state.ui,
                format!("Unable to load Amber force field data: {e:?}"),
            );
        }
    }

    state.load_prefs();

    // Set these UI strings for numerical values up after loading prefs
    state.volatile.md_runtime = state.to_save.num_md_steps as f32 * state.to_save.md_dt;
    state.ui.ph_input = state.to_save.ph.to_string();
    state.ui.md.dt_input = state.to_save.md_dt.to_string();
    state.ui.md.pressure_input = (state.to_save.md_config.pressure_target as u16).to_string();
    state.ui.md.temp_input = (state.to_save.md_config.temp_target as u16).to_string();
    state.ui.md.simbox_pad_input = match state.to_save.md_config.sim_box {
        SimBoxInit::Pad(p) => (p as u16).to_string(),
        SimBoxInit::Fixed(_) => "0".to_string(), // We currently don't use this.
    };

    state.ui.md.langevin_γ = match state.to_save.md_config.integrator {
        Integrator::Langevin { gamma } | Integrator::LangevinMiddle { gamma } => gamma.to_string(),
        _ => "0.".to_string(),
    };

    // We must have loaded prefs prior to this, so we know which file to open.
    state.load_last_opened();

    // todo trouble: It's somewhere around here, saving the inited-from-load atom posits, overwriting
    // todo the previously-saved ones.

    // todo: Workaround to allow us to apply params to the ligand once it's loaded. Unfortunate we have
    // todo to double-load prefs.
    state.load_prefs();

    // Update ligand positions, e.g. from the docking position site center loaded from prefs.
    // if let Some(lig) = &mut state.ligand {
    //     if let Some(data) = &mut lig.lig_data {
    //         data.pose.anchor_posit = data.docking_site.site_center;
    //         lig.position_atoms(None);
    //     }
    // }

    if let Some(mol) = &state.peptide {
        let posit = state.to_save.per_mol[&mol.common.ident]
            .docking_site
            .site_center;
        state.update_docking_site(posit);
    }

    // todo: Consider if you want this default, and if you also want to add default Lipids etc.
    if !state.ligands.is_empty() {
        state.volatile.active_mol = Some((MolType::Ligand, 0));
    }

    // if let Err(e) = state.load_aa_charges_ff() {
    //     handle_err(
    //         &mut state.ui,
    //         format!("Unable to load protein charges (static): {e}"),
    //     );
    // }

    // todo temp
    // let (atoms, params) = bio_files::prmtop::load_prmtop(std::path::Path::new("./molecules/str1.prmtop")).unwrap();
    //
    // println!("Loaded atoms:");
    // for at in atoms {
    //     println!("{at}");
    // }
    //
    // println!("\n\n\n params: {:?}", params);

    state.load_lipid_templates();

    render(state);
}
