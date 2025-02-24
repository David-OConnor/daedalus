//! Adapted from `peptide`.

use std::f64::consts::TAU;

use lin_alg::f64::Vec3;
use na_seq::AminoAcid;

use crate::molecule::Atom;

pub mod bond_vecs;
pub mod sc_atom_placement;
pub mod sidechain;

/// C+P from peptide
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
pub fn aa_data_from_coords(atoms: &[Atom], aa: AminoAcid) -> (f32, Vec<Atom>) {
    // todo: With_capacity based on aa?
    let hydrogens = Vec::new();

    let dihedral_angles = 0.;

    (dihedral_angles, hydrogens)
}
