//! Adapted from `peptide`. This includes sub-modules.

use std::f64::consts::TAU;

use lin_alg::f64::Vec3;
use na_seq::AminoAcid;

use crate::{
    Element,
    aa_coords::{bond_vecs::init_local_bond_vecs, sidechain::Sidechain},
    molecule::{Atom, AtomRole},
};

pub mod bond_vecs;
pub mod sc_atom_placement;
pub mod sidechain;

struct PlacementError {}

/// An amino acid in a protein structure, including all dihedral angles required to determine
/// the conformation. Includes backbone and side chain dihedral angles. Doesn't store coordinates,
/// but coordinates can be generated using forward kinematics from the angles.
#[derive(Debug)]
pub struct ResidueFlex {
    /// Dihedral angle between C' and N
    /// Tor (Cα, C, N, Cα) is the ω torsion angle
    /// Assumed to be TAU/2 for most cases
    pub ω: f64,
    /// Dihedral angle between Cα and N.
    /// Tor (C, N, Cα, C) is the φ torsion angle
    pub φ: f64,
    /// Dihedral angle, between Cα and C'
    ///  Tor (N, Cα, C, N) is the ψ torsion angle
    pub ψ: f64,
    // /// Contains the χ angles that define t
    pub sidechain: Sidechain,
    // pub dipole: Vec3,
}

/// Calculate the dihedral angle between 4 atoms.
/// todo: How are these bonds represented as Vec3s? Maybe just subtract the atom posits.
pub fn calc_dihedral_angle(bond_middle: Vec3, bond_adjacent1: Vec3, bond_adjacent2: Vec3) -> f64 {
    // Project the next and previous bonds onto the plane that has this bond as its normal.
    // Re-normalize after projecting.
    let bond1_on_plane = bond_adjacent1.project_to_plane(bond_middle).to_normalized();
    let bond2_on_plane = bond_adjacent2.project_to_plane(bond_middle).to_normalized();

    // Not sure why we need to offset by 𝜏/2 here, but it seems to be the case
    let result = bond1_on_plane.dot(bond2_on_plane).acos() + TAU / 2.;

    // The dot product approach to angles between vectors only covers half of possible
    // rotations; use a determinant of the 3 vectors as matrix columns to determine if we
    // need to modify to be on the second half.
    let det = lin_alg::f64::det_from_cols(bond1_on_plane, bond2_on_plane, bond_middle);

    // todo: Exception if vecs are the same??
    if det < 0. { result } else { TAU - result }
}

/// todo: Rename, etc.
/// todo: Infer residue from coords instead of accepting as param?
/// Returns (dihedral angles, H atoms)
pub fn aa_data_from_coords(
    atoms: &[Atom],
    aa: AminoAcid,
) -> Result<(f32, Vec<Atom>), PlacementError> {
    // todo: With_capacity based on aa?

    // todo: Maybe split this into separate functions.

    // Initialized to default. Now, how to fill this out?
    let res = ResidueFlex {
        ω: 0.,
        φ: 0.,
        ψ: 0.,
        sidechain: Sidechain::from_aa_type(aa),
    };

    // todo: Populate sidechain and main angles now based on coords. (?)

    let dihedral_angles = 0.;
    let mut hydrogens = Vec::new();

    for atom in atoms {
        let role = AtomRole::Sidechain;
        match atom.element {
            Element::Carbon => {}
            Element::Nitrogen => {}
            _ => {}
        }
    }

    Ok((dihedral_angles, hydrogens))
}
