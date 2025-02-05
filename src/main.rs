extern crate core;

mod amino_acid_coords;
mod bond_inference;
mod download_pdb;
mod drug_like;
mod molecule;
mod pdb;
mod render;
mod save_load;
mod ui;
mod util;
mod vibrations;

use std::{any::Any, io, io::ErrorKind, path::PathBuf, str::FromStr, sync::Arc};

use egui_file_dialog::{FileDialog, FileDialogConfig};
use lin_alg::f64::Vec3;
use molecule::Molecule;
use pdbtbx::{self, PDB};
use rayon::iter::ParallelIterator;

use crate::{
    molecule::Atom,
    pdb::load_pdb,
    render::{render, MoleculeView},
};

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

    pub fn _from_letter(letter: &str) -> io::Result<Self> {
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
}

#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum AtomColorCode {
    Atom,
    #[default]
    Residue,
}

impl AtomColorCode {
    pub fn to_string(&self) -> String {
        match self {
            Self::Atom => "Atom",
            Self::Residue => "Residue",
        }
        .to_owned()
    }
}

struct StateUi {
    load_dialog: FileDialog,
    mol_view: MoleculeView,
    /// Mouse cursor
    cursor_pos: Option<(f32, f32)>,
    pub rcsb_input: String,
    atom_color_code: AtomColorCode,
}

impl Default for StateUi {
    fn default() -> Self {
        let cfg = FileDialogConfig {
            ..Default::default()
        }
        .add_file_filter(
            // Note: We experience glitches if this name is too long. (Window extends horizontally)
            "PDB/CIF",
            Arc::new(|p| {
                let ext = p.extension().unwrap_or_default().to_ascii_lowercase();
                ext == "pdb" || ext == "cif"
            }),
        );
        // .add_file_filter(
        //     "PDB",
        //     Arc::new(|p| p.extension().unwrap_or_default().to_ascii_lowercase() == "pdb"),
        // )
        // .add_file_filter(
        //     "CIF",
        //     Arc::new(|p| p.extension().unwrap_or_default().to_ascii_lowercase() == "cif"),
        // );
        //
        let load_dialog = FileDialog::with_config(cfg)
            .default_file_filter("PDB/CIF")
            .id("fd1");

        Self {
            load_dialog,
            mol_view: Default::default(),
            cursor_pos: None,
            rcsb_input: String::new(),
            atom_color_code: Default::default(),
        }
    }
}

#[derive(Default)]
struct State {
    pub ui: StateUi,
    pub pdb: Option<PDB>,
    pub molecule: Option<Molecule>,
    /// Index
    /// todo: This is likely a temporary implementation
    pub atom_selected: Option<usize>,
}

fn main() {
    let mut state = State::default();

    state.atom_selected = Some(0);

    let pdb = load_pdb(&PathBuf::from_str("1kmk.pdb").unwrap());
    if let Ok(p) = pdb {
        state.pdb = Some(p);
        state.molecule = Some(Molecule::from_pdb(state.pdb.as_ref().unwrap()));
    } else {
        eprintln!("Error loading PDB file at init.");
    }

    render(state);
}
