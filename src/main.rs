mod bond_inference;
mod pdb;
mod render;
mod ui;
mod util;

use std::{
    any::Any,
    io,
    io::ErrorKind,
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
};

use egui_file_dialog::{FileDialog, FileDialogConfig};
use lin_alg::f64::Vec3;
use pdbtbx::{self, PDB};
use rayon::iter::ParallelIterator;

use crate::{
    bond_inference::create_bonds,
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
                _ => Self::Other,
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
            Self::Other => (5., 5., 5.),
        }
    }
}

#[derive(Debug)]
pub struct Atom {
    pub posit: Vec3, // todo: f32 or f64?
    pub element: Element,
}

impl Atom {
    pub fn from_pdb(pdb: &pdbtbx::Atom) -> Self {
        Self {
            posit: Vec3::new(pdb.x(), pdb.y(), pdb.z()),
            element: Element::from_pdb(pdb.element()),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum BondType {
    // C+P from pdbtbx for now
    Covalent,
    Disulfide,
    Hydrogen,
    MetalCoordination,
    MisMatchedBasePairs,
    SaltBridge,
    CovalentModificationResidue,
    CovalentModificationNucleotideBase,
    CovalentModificationNucleotideSugar,
    CovalentModificationNucleotidePhosphate,
}

#[derive(Debug)]
pub struct Bond {
    pub bond_type: BondType,
    // todo: Refs to Atom etc A/R.
    pub posit_0: Vec3,
    pub posit_1: Vec3,
}

#[derive(Debug)]
// todo: This, or a PDB-specific format?
pub struct Molecule {
    pub atoms: Vec<Atom>,
    /// todo: For now, as returned by pdbtbx. Adjust A/R. (Refs to atoms etc)
    // pub bonds: Vec<(Atom, Atom, pdb::Bond)>,
    pub bonds: Vec<Bond>,
}

impl Molecule {
    pub fn from_pdb(pdb: &PDB) -> Self {
        // todo: Maybe return the PDB type here, and store that. Also have a way to
        // todo get molecules from it

        // for atom in pdb.atoms() {
        // for atom in pdb.par_atoms() {
        let atoms: Vec<Atom> = pdb.par_atoms().map(|atom| Atom::from_pdb(atom)).collect();

        // let mut bonds = Vec::new();
        // /// todo: Adjust etc so you're not adding so many new atoms to state.
        // for (a0, a1, bond) in pdb.bonds() {
        //     bonds.push((Atom::from_pdb(a0), Atom::from_pdb(a1), bond));
        // }

        let bonds = create_bonds(&atoms);

        // todo: Chains

        Molecule { atoms, bonds }
    }
}

struct StateUi {
    load_dialog: FileDialog,
    mol_view: MoleculeView,
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
        }
    }
}

#[derive(Default)]
struct State {
    pub ui: StateUi,
    pub pdb: Option<PDB>,
    pub molecule: Option<Molecule>,
}

fn main() {
    let mut state = State::default();

    let pdb = load_pdb(&PathBuf::from_str("1kmk.pdb").unwrap());
    if let Ok(p) = pdb {
        state.pdb = Some(p);
        state.molecule = Some(Molecule::from_pdb(state.pdb.as_ref().unwrap()));
    } else {
        eprintln!("Error loading PDB file at init.");
    }

    render(state);
}
