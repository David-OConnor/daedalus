mod pdb;
mod render;
mod ui;

use std::{
    any::Any,
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
};

use egui_file_dialog::{FileDialog, FileDialogConfig};
use graphics::Entity;
use lin_alg::f64::Vec3;
use pdbtbx::PDB;

use crate::{pdb::load_pdb, render::render};

#[derive(Debug, Clone, Default)]
pub enum ComputationDevice {
    #[default]
    Cpu,
    #[cfg(feature = "cuda")]
    Gpu(Arc<CudaDevice>),
}

#[derive(Clone, Copy, PartialEq)]
pub enum AtomType {
    Carbon,
    Hydrogen,
    Oxygen,
}

#[derive(Debug)]
pub struct Atom {
    pub posit: Vec3, // todo: f32 or f64?
    // pub atom_type: AtomType,
    pub atom_type: String, // todo temp
}

#[derive(Debug)]
// todo: This, or a PDB-specific format?
pub struct Molecule {
    pub atoms: Vec<Atom>,
}

impl Molecule {
    pub fn from_pdb(pdb: &PDB) -> Self {
        let mut atoms = Vec::new();

        // todo: Maybe return the PDB type here, and store that. Also have a way to
        // todo get molecules from it

        for atom in pdb.atoms() {
            atoms.push(Atom {
                posit: Vec3::new(atom.x(), atom.y(), atom.z()),
                atom_type: atom.name().to_owned(),
            })
        }

        Molecule { atoms }
    }
}

struct StateUi {
    load_dialog: FileDialog,
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
        )
        .add_file_filter(
            "PDB",
            Arc::new(|p| p.extension().unwrap_or_default().to_ascii_lowercase() == "pdb"),
        )
        .add_file_filter(
            "CIF",
            Arc::new(|p| p.extension().unwrap_or_default().to_ascii_lowercase() == "cif"),
        );

        let load_dialog = FileDialog::with_config(cfg)
            .default_file_filter("PDB/CIF")
            .id("fd1");

        Self { load_dialog }
    }
}

#[derive(Default)]
struct State {
    ui: StateUi,
    pub pdb: Option<PDB>,
    pub molecule: Option<Molecule>,
}

fn main() {
    let mut state = State::default();

    let pdb = load_pdb(&PathBuf::from_str("7m7f.pdb").unwrap());
    if let Ok(p) = pdb {
        state.pdb = Some(p);
        state.molecule = Some(Molecule::from_pdb(state.pdb.as_ref().unwrap()));
    } else {
        eprintln!("Error loading PDB file at init.");
    }

    render(state);
}
