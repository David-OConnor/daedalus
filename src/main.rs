extern crate core;

mod add_hydrogens;
mod amino_acid_coords;
mod asa;
mod bond_inference;
mod cartoon_mesh;
mod docking;
mod download_pdb;
mod drug_like;
mod input;
mod mol_drawing;
mod molecule;
mod navigation;
mod pdb;
mod prefs;
mod rcsb_api;
mod render;
mod save_load;
mod sdf;
mod ui;
mod util;
mod vibrations;

use std::{
    io,
    io::{ErrorKind, Read},
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
};

use bincode::{Decode, Encode};
use egui_file_dialog::{FileDialog, FileDialogConfig};
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
    docking::{DockingInit, check_adv_avail, docking_prep_external::check_babel_avail},
    molecule::Ligand2,
    navigation::Tab,
    pdb::load_pdb,
    prefs::ToSave,
    render::render,
    sdf::load_sdf,
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

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Element {
    Hydrogen,
    Carbon,
    Oxygen,
    Nitrogen,
    Fluorine,
    Sulfur,
    Phosphorus,
    Iron,
    Copper,
    Calcium,
    Potassium,
    Aluminum,
    Lead,
    Gold,
    Silver,
    Mercury,
    Tin,
    Zinc,
    Magnesium,
    Iodine,
    Chlorine,
    Tungsten,
    Tellurium,
    Selenium,
    Other,
}

impl Element {
    pub fn valence_typical(&self) -> usize {
        match self {
            Self::Hydrogen => 1,
            Self::Carbon => 4,
            Self::Oxygen => 2,
            Self::Nitrogen => 3,
            Self::Fluorine => 1,
            Self::Sulfur => 2,     // can be 2, 4, or 6, but 2 is a common choice
            Self::Phosphorus => 5, // can be 3 or 5, here we pick 5
            Self::Iron => 2,       // Fe(II) is common (Fe(III) also common)
            Self::Copper => 2,     // Cu(I) and Cu(II) both occur, pick 2 as a naive default
            Self::Calcium => 2,
            Self::Potassium => 1,
            Self::Aluminum => 3,
            Self::Lead => 2,    // Pb(II) or Pb(IV), but Pb(II) is more common/stable
            Self::Gold => 3,    // Au(I) and Au(III) are common, pick 3
            Self::Silver => 1,  // Ag(I) is most common
            Self::Mercury => 2, // Hg(I) and Hg(II), pick 2
            Self::Tin => 4,     // Sn(II) or Sn(IV), pick 4
            Self::Zinc => 2,
            Self::Magnesium => 2,
            Self::Iodine => 1, // can have higher, but 1 is typical in many simple compounds
            Self::Chlorine => 1, // can also be 3,5,7, but 1 is the simplest (e.g., HCl)
            Self::Tungsten => 6, // W can have multiple but 6 is a common oxidation state
            Self::Tellurium => 2, // can also be 4 or 6, pick 2
            Self::Selenium => 2, // can also be 4 or 6, pick 2
            Self::Other => 0,  // default to 0 for unknown or unhandled elements
        }
    }
}

impl Element {
    pub fn from_pdb(el: Option<&pdbtbx::Element>) -> Self {
        if let Some(e) = el {
            match e {
                pdbtbx::Element::H => Self::Hydrogen,
                pdbtbx::Element::C => Self::Carbon,
                pdbtbx::Element::O => Self::Oxygen,
                pdbtbx::Element::N => Self::Nitrogen,
                pdbtbx::Element::F => Self::Fluorine,
                pdbtbx::Element::S => Self::Sulfur,
                pdbtbx::Element::P => Self::Phosphorus,
                pdbtbx::Element::Fe => Self::Iron,
                pdbtbx::Element::Cu => Self::Copper,
                pdbtbx::Element::Ca => Self::Calcium,
                pdbtbx::Element::K => Self::Potassium,
                pdbtbx::Element::Al => Self::Aluminum,
                pdbtbx::Element::Pb => Self::Lead,
                pdbtbx::Element::Au => Self::Gold,
                pdbtbx::Element::Ag => Self::Silver,
                pdbtbx::Element::Hg => Self::Mercury,
                pdbtbx::Element::Sn => Self::Tin,
                pdbtbx::Element::Zn => Self::Zinc,
                pdbtbx::Element::Mg => Self::Magnesium,
                pdbtbx::Element::I => Self::Iodine,
                pdbtbx::Element::Cl => Self::Chlorine,
                pdbtbx::Element::W => Self::Tungsten,
                pdbtbx::Element::Te => Self::Tellurium,
                pdbtbx::Element::Se => Self::Selenium,

                _ => {
                    eprintln!("Unknown element: {e:?}");
                    Self::Other
                }
            }
        } else {
            // todo?
            Self::Other
        }
    }

    pub fn from_letter(letter: &str) -> io::Result<Self> {
        match letter.to_uppercase().as_ref() {
            "H" => Ok(Self::Hydrogen),
            "C" => Ok(Self::Carbon),
            "O" => Ok(Self::Oxygen),
            "N" => Ok(Self::Nitrogen),
            "F" => Ok(Self::Fluorine),
            "S" => Ok(Self::Sulfur),
            "P" => Ok(Self::Phosphorus),
            "FE" => Ok(Self::Iron),
            "CU" => Ok(Self::Copper),
            "CA" => Ok(Self::Calcium),
            "K" => Ok(Self::Potassium),
            // todo: Fill in if you need, or remove this fn.
            _ => Err(io::Error::new(
                ErrorKind::InvalidData,
                "Invalid atom letter",
            )),
        }
    }

    /// From [PyMol](https://pymolwiki.org/index.php/Color_Values)
    pub fn color(&self) -> (f32, f32, f32) {
        match self {
            Self::Hydrogen => (0.9, 0.9, 0.9),
            Self::Carbon => (0.2, 1., 0.2),
            Self::Oxygen => (1., 0.3, 0.3),
            Self::Nitrogen => (0.2, 0.2, 1.0),
            Self::Fluorine => (0.701, 1.0, 1.0),
            Self::Sulfur => (0.9, 0.775, 0.25),
            Self::Phosphorus => (1.0, 0.502, 0.),
            Self::Iron => (0.878, 0.4, 0.2),
            Self::Copper => (0.784, 0.502, 0.2),
            Self::Calcium => (0.239, 1.0, 0.),
            Self::Potassium => (0.561, 0.251, 0.831),
            Self::Aluminum => (0.749, 0.651, 0.651),
            Self::Lead => (0.341, 0.349, 0.380),
            Self::Gold => (1., 0.820, 0.137),
            Self::Silver => (0.753, 0.753, 0.753),
            Self::Mercury => (0.722, 0.722, 0.816),
            Self::Tin => (0.4, 0.502, 0.502),
            Self::Zinc => (0.490, 0.502, 0.690),
            Self::Magnesium => (0.541, 1., 0.),
            Self::Iodine => (0.580, 0., 0.580),
            Self::Chlorine => (0.121, 0.941, 0.121),
            Self::Tungsten => (0.129, 0.580, 0.840),
            Self::Tellurium => (0.831, 0.478, 0.),
            Self::Selenium => (1.0, 0.631, 0.),
            Self::Other => (5., 5., 5.),
        }
    }

    #[rustfmt::skip]
    /// Covalent radius, in angstrom.
    /// https://github.com/openbabel/openbabel/blob/master/src/elementtable.h
    pub fn covalent_radius(self) -> f64 {
        match self {
            Element::Hydrogen   => 0.31,
            Element::Carbon     => 0.76,
            Element::Oxygen     => 0.66,
            Element::Nitrogen   => 0.71,
            Element::Fluorine   => 0.57,
            Element::Sulfur     => 1.05,
            Element::Phosphorus => 1.07,
            Element::Iron       => 1.32,
            Element::Copper     => 1.32,
            Element::Calcium    => 1.76,
            Element::Potassium  => 2.03,
            Element::Aluminum   => 1.21,
            Element::Lead       => 1.46,
            Element::Gold       => 1.36,
            Element::Silver     => 1.45,
            Element::Mercury    => 1.32,
            Element::Tin        => 1.39,
            Element::Zinc       => 1.22,
            Element::Magnesium  => 1.41,
            Element::Iodine     => 1.39,
            Element::Chlorine   => 1.02,
            Element::Tungsten   => 1.62,
            Element::Tellurium  => 1.38,
            Element::Selenium   => 1.20,
            Element::Other      => 0.00,
        }
    }

    #[rustfmt::skip]
    /// Van-der-wals radius, in angstrom.
    /// https://github.com/openbabel/openbabel/blob/master/src/elementtable.h
    pub const fn vdw_radius(&self) -> f32 {
        match self {
            Element::Hydrogen   => 1.10,
            Element::Carbon     => 1.70,
            Element::Oxygen     => 1.52,
            Element::Nitrogen   => 1.55,
            Element::Fluorine   => 1.47,
            Element::Sulfur     => 1.80,
            Element::Phosphorus => 1.80,
            Element::Iron       => 2.05,
            Element::Copper     => 2.00,
            Element::Calcium    => 2.31,
            Element::Potassium  => 2.75,
            Element::Aluminum   => 1.84,
            Element::Lead       => 2.02,
            Element::Gold       => 2.10,
            Element::Silver     => 2.10,
            Element::Mercury    => 2.05,
            Element::Tin        => 1.93,
            Element::Zinc       => 2.10,
            Element::Magnesium  => 1.73,
            Element::Iodine     => 1.98,
            Element::Chlorine   => 1.75,
            Element::Tungsten   => 2.10,
            Element::Tellurium  => 2.06,
            Element::Selenium   => 1.90,
            Element::Other      => 0.0,
        }
    }
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
                ext == "pdb" || ext == "cif" || ext == "sdf"
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

        let load_dialog = FileDialog::with_config(cfg.clone()).default_file_filter("PDB/CIF/SDF");
        let load_ligand_dialog = FileDialog::with_config(cfg).default_file_filter("PDB/CIF/SDF");

        let autodock_path_dialog =
            FileDialog::with_config(cfg_vina).default_file_filter("Executables");

        Self {
            load_dialog,
            load_ligand_dialog,
            autodock_path_dialog,
            // mol_center: Vec3::new_zero(),
            // mol_size: 80.,
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
    pub ligand: Option<Ligand2>,
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
    pub fn open_molecule(&mut self, path: &Path, ligand: bool) {
        match path
            .extension()
            .unwrap_or_default()
            .to_ascii_lowercase()
            .to_str()
            .unwrap_or_default()
        {
            "sdf" => {
                let sdf = load_sdf(path);

                if let Ok(s) = sdf {
                    let mol = Molecule::from_sdf(&s);

                    if ligand {
                        self.ligand = Some(Ligand2 {
                            molecule: mol,
                            docking_init: DockingInit {
                                site_posit: Vec3F64::new(-11.83, 15.49, 65.88),
                                site_box_size: 3.,
                            },
                            orientation: QuaternionF64::new_identity(),
                        });
                    } else {
                        self.molecule = Some(mol);
                    }

                    self.update_from_prefs();
                } else {
                    eprintln!("Error loading SDF file.");
                }
            }
            _ => {
                // e.g. cif, pdb
                let pdb = load_pdb(path);
                if let Ok(p) = pdb {
                    self.pdb = Some(p);
                    let mol = Molecule::from_pdb(self.pdb.as_ref().unwrap());

                    if ligand {
                        self.ligand = Some(Ligand2 {
                            molecule: mol,
                            docking_init: DockingInit {
                                site_posit: Vec3F64::new_zero(),
                                site_box_size: 3.,
                            },
                            orientation: QuaternionF64::new_identity(),
                        });
                    } else {
                        self.molecule = Some(mol);
                    }

                    self.update_from_prefs();
                } else {
                    eprintln!("Error loading PDB file.");
                }
            }
        }
    }
}

fn main() {
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
