//! For setting up and rendering nucleic acids: DNA and RNA.

use std::collections::HashMap;

use na_seq::Nucleotide;
use lin_alg::f64::Vec3;

use crate::molecule::{Atom, Bond, MoleculeGeneric, MoleculeCommon};

// todo: Load Amber FF params for nucleic acids.

/// Represents a nucleic acid as a collection of atoms and bonds. Omits mol-generic fields.
#[derive(Debug, Default, Clone)]
pub struct MoleculeNucleicAcid {
    pub common: MoleculeCommon,
    pub seq: Vec<Nucleotide>,
    // pub bonds_hydrogen: Vec<HydrogenBond>,
    pub features: Vec<String, (usize, usize)>, // todo: A/R
    // todo: A/R
    pub metadata: HashMap<String, String>,
    // todo: A/R.
    // pub ff_params: Option<ForceFieldParamsIndexed>,
}

impl MoleculeNucleicAcid {
    // todo: Methods on Molecule?
    /// Initializes a linear molecule.
    pub fn from_seq(seq: &[Nucleotide]) -> Self {

    }
}
