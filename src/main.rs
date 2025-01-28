mod pdb;
mod render;
mod ui;

use std::{path::PathBuf, str::FromStr};
use std::any::Any;
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
                atom_type: atom.name().to_owned()
            })
        }

        Molecule { atoms }
    }
}



#[derive(Default)]
struct State {
    pub molecule: Option<Molecule>
}

fn main() {
    let mut state = State::default();

    // let pdb = load_pdb(&PathBuf::from_str("1yyf.pdb").unwrap()).unwrap();
    let pdb = load_pdb(&PathBuf::from_str("1ubq.cif").unwrap()).unwrap();

    let molecule = Molecule::from_pdb(&pdb);
    state.molecule = Some(molecule);


    render(state);
}
