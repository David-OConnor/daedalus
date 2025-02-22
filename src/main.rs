extern crate core;

mod amino_acid_coords;
mod asa;
mod bond_inference;
mod cartoon_mesh;
mod docking;
mod docking_prep;
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
    docking::DockingInit, molecule::Ligand2, navigation::Tab, pdb::load_pdb, prefs::ToSave,
    render::render, sdf::load_sdf, ui::VIEW_DEPTH_MAX,
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

    /// Van-der-wals radius, in angstrom.
    pub const fn vdw_radius(&self) -> f32 {
        match self {
            Self::Hydrogen => 1.20,
            Self::Carbon => 1.70,
            Self::Nitrogen => 1.55,
            Self::Oxygen => 1.52,
            Self::Fluorine => 1.47,
            Self::Sulfur => 1.80,
            Self::Phosphorus => 1.80,
            Self::Iron => 2.00, // Many references list ~1.56–1.63; 2.00 is a common PyMOL default.
            Self::Copper => 1.40,
            Self::Calcium => 2.31,
            Self::Potassium => 2.75,
            Self::Aluminum => 2.00,
            Self::Lead => 2.02,
            Self::Gold => 1.66,
            Self::Silver => 1.72,
            Self::Mercury => 1.55,
            Self::Tin => 2.17,
            Self::Zinc => 1.39,
            Self::Magnesium => 1.73,
            Self::Iodine => 1.98,
            Self::Chlorine => 1.75,
            Self::Tungsten => 2.10,
            Self::Tellurium => 2.06,
            Self::Selenium => 1.90,

            // Fallback for elements not explicitly listed
            Self::Other => 2.00,
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
        let load_dialog = FileDialog::with_config(cfg.clone()).default_file_filter("PDB/CIF/SDF");
        let load_ligand_dialog = FileDialog::with_config(cfg).default_file_filter("PDB/CIF/SDF");

        Self {
            load_dialog,
            load_ligand_dialog,
            // mol_center: Vec3::new_zero(),
            // mol_size: 80.,
            ui_height: 0.,
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
    hide_sidechains: bool,
    hide_water: bool,
    /// Hide hetero atoms: i.e. ones not part of a polypeptide.
    hide_hetero: bool,
    hide_non_hetero: bool,
    hide_ligand: bool,
    middle_click_down: bool,
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
    // todo: Instead of molecule and ligand fields, maybe `Vec<Molecule>`?
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
    pub autodock_vina_path: Option<PathBuf>,
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

    render(state);
}
