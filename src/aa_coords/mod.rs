//! Adapted from `peptide`. This includes sub-modules.

use std::{
    f64::consts::TAU,
    fmt,
    fmt::{Formatter, write},
};

use lin_alg::f64::{Quaternion, Vec3};
use na_seq::AminoAcid;

use crate::{
    Element,
    aa_coords::{
        bond_vecs::{
            LEN_C_H, LEN_CALPHA_H, LEN_N_H, TETRA_A, TETRA_ANGLE, TETRA_B, TETRA_C, TETRA_D,
        },
        sidechain::Sidechain,
    },
    molecule::{Atom, AtomRole, ResidueType},
};

pub mod bond_vecs;
pub mod sc_atom_placement;
pub mod sidechain;

pub struct PlacementError {}

/// An amino acid in a protein structure, including all dihedral angles required to determine
/// the conformation. Includes backbone and side chain dihedral angles. Doesn't store coordinates,
/// but coordinates can be generated using forward kinematics from the angles.
#[derive(Debug, Clone)]
pub struct Dihedral {
    /// Dihedral angle between C' and N
    /// Tor (Cα, C', N, Cα) is the ω torsion angle
    /// Assumed to be TAU/2 for most cases
    pub ω: f64,
    /// Dihedral angle between Cα and N.
    /// Tor (C', N, Cα, C') is the φ torsion angle
    pub φ: f64,
    /// Dihedral angle, between Cα and C'
    ///  Tor (N, Cα, C', N) is the ψ torsion angle
    pub ψ: f64,
    // /// Contains the χ angles that define t
    pub sidechain: Sidechain,
    // pub dipole: Vec3,
}

impl fmt::Display for Dihedral {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ω: {:.2}τ  φ: {:.2}τ  ψ: {:.2}τ",
            self.ω / TAU,
            self.φ / TAU,
            self.ψ / TAU
        )?;
        Ok(())
    }
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

/// Given three tetrahedron legs, find the final one.
pub fn tetra_legs(leg_a: Vec3, leg_b: Vec3, leg_c: Vec3) -> Vec3 {
    (-(leg_a + leg_b + leg_c)).to_normalized()
}

pub fn tetra_atoms(atom_center: Vec3, atom_a: Vec3, atom_b: Vec3, atom_c: Vec3) -> Vec3 {
    let mut avg = (atom_a + atom_b + atom_c) / 3.;
    (avg - atom_center).to_normalized()
}

/// Given the positions of two atoms of a tetrahedron, find the remaining two.
/// `len` is the length between the center, and each apex.
fn tetra_atoms_2(center: Vec3, atom_0: Vec3, atom_1: Vec3, len: f64) -> (Vec3, Vec3) {
    // todo: Not working.
    // Move from world-space to local.
    let bond_0 = (atom_0 - center).to_normalized();
    let bond_1 = (atom_1 - center).to_normalized();

    // Aligns the tetrahedron leg A to bond 0.
    let rotator_a = Quaternion::from_unit_vecs(TETRA_A, bond_0);

    // Once the TETRA_A is aligned to bond_0, rotate the tetrahedron around this until TETRA_B aligs
    // with bond_1. Then, the other two tetra parts will be where we place our hydrogens.
    let tetra_b_rotated = rotator_a.rotate_vec(unsafe { TETRA_B });
    let tetra_b_on_plane = tetra_b_rotated.project_to_plane(bond_0);
    let bond_1_on_plane = bond_1.project_to_plane(bond_0);
    let rot_amt = tetra_b_on_plane.dot(bond_1_on_plane).acos();

    // let rotator_b = Quaternion::from_axis_angle(bond_0, -rot_amt);
    let rotator_b = Quaternion::from_axis_angle(bond_0, -rot_amt);

    let rotator = rotator_b * rotator_a;

    unsafe {
        (
            center + rotator.rotate_vec(TETRA_C) * len,
            center + rotator.rotate_vec(TETRA_D) * len,
        )
    }
}

/// todo: Rename, etc.
/// todo: Infer residue from coords instead of accepting as param?
/// Returns (dihedral angles, H atoms, c'_pos, ca_pos). The parameter and output carbon positions
/// are for use in calculating dihedral angles associated with other  chains.
pub fn aa_data_from_coords(
    atoms: &[&Atom],
    aa: AminoAcid,
    prev_cp_pos: Vec3,
    prev_ca_pos: Vec3,
    next_n_pos: Vec3,
) -> Result<(Dihedral, Vec<Atom>, Vec3, Vec3), PlacementError> {
    // todo: With_capacity based on aa?

    // todo: Maybe split this into separate functions.

    let h_default = Atom {
        serial_number: 0,
        posit: Vec3::new_zero(),
        element: Element::Hydrogen,
        name: "H".to_string(),
        role: Some(AtomRole::H_Backbone),
        residue_type: ResidueType::AminoAcid(aa),
        hetero: false,
        dock_type: None,
        occupancy: None,
        partial_charge: None,
        temperature_factor: None,
    };

    // Initialized to default. Now, how to fill this out?
    let mut dihedral = Dihedral {
        ω: 0.,
        φ: 0.,
        ψ: 0.,
        sidechain: Sidechain::from_aa_type(aa),
    };

    // todo: Populate sidechain and main angles now based on coords. (?)

    let mut hydrogens = Vec::new();

    // Find the positions of the backbone atoms.
    let mut n_posit = Vec3::new_zero();
    let mut c_alpha_posit = Vec3::new_zero();
    let mut c_p_posit = Vec3::new_zero();
    let mut n_found = false;
    let mut c_alpha_found = false;
    let mut c_p_found = false;

    for atom in atoms {
        if atom.role.is_none() {
            continue;
        }
        match atom.role.as_ref().unwrap() {
            AtomRole::N_Backbone => {
                n_posit = atom.posit;
                n_found = true;
            }
            AtomRole::C_Alpha => {
                c_alpha_posit = atom.posit;
                c_alpha_found = true;
            }
            AtomRole::C_Prime => {
                c_p_posit = atom.posit;
                c_p_found = true;
            }
            _ => (),
        }
    }
    if !c_alpha_found || !c_p_found || !n_found {
        eprintln!("Error: Missing backbone atoms in coords.");
        return Err(PlacementError {});
    }

    // /// Dihedral angle between C' and N
    // /// Tor (Cα, C, N, Cα) is the ω torsion angle
    // /// Assumed to be TAU/2 for most cases
    // pub ω: f64,
    // /// Dihedral angle between Cα and N.
    // /// Tor (C, N, Cα, C) is the φ torsion angle
    // pub φ: f64,
    // /// Dihedral angle, between Cα and C'
    // ///  Tor (N, Cα, C, N) is the ψ torsion angle
    // pub ψ: f64,
    let bond_cp_prev_ca_prev = prev_cp_pos - prev_ca_pos;
    let bond_n_cp_prev = n_posit - prev_cp_pos;
    let bond_ca_n = c_alpha_posit - n_posit;
    let bond_cp_ca = c_p_posit - c_alpha_posit;
    let bond_n_next_cp = next_n_pos - c_p_posit;

    dihedral.ω = calc_dihedral_angle(bond_n_cp_prev, bond_cp_prev_ca_prev, bond_ca_n);
    dihedral.φ = calc_dihedral_angle(bond_ca_n, bond_n_cp_prev, bond_cp_ca);
    dihedral.ψ = calc_dihedral_angle(bond_cp_ca, bond_ca_n, bond_n_next_cp);

    // Add a H to the N atom. Planar.
    let n_plane_normal = bond_n_cp_prev.cross(bond_ca_n).to_normalized();
    let rotator = Quaternion::from_axis_angle(n_plane_normal, TAU / 3.);
    hydrogens.push(Atom {
        posit: n_posit + rotator.rotate_vec(-bond_n_cp_prev.to_normalized() * LEN_N_H),
        ..h_default.clone()
    });

    // Find the nearest sidechain atom

    // Add a H to the C alpha atom. Tetrahedral.
    // let ca_plane_normal = bond_ca_n.cross(bond_cp_ca).to_normalized();
    // todo: There are two possible settings available for the rotator; one will be taken up by
    // a sidechain carbon.
    // let rotator = Quaternion::from_axis_angle(ca_plane_normal, TETRA_ANGLE);
    // todo: Another step required using sidechain carbon?
    let mut posits_sc = Vec::new();
    for atom_sc in atoms {
        if atom_sc.role.is_none() {
            continue;
        }
        match atom_sc.role.as_ref().unwrap() {
            AtomRole::Sidechain => {
                if atom_sc.element == Element::Carbon {
                    posits_sc.push(atom_sc.posit);
                }
            }
            _ => (),
        }
    }

    if posits_sc.is_empty() {
        eprintln!("Error: Could not find sidechain atom.");
        return Err(PlacementError {});
    }

    let mut closest = (posits_sc[0] - c_alpha_posit).magnitude();
    let mut closest_sc = posits_sc[0];

    for pos in posits_sc {
        let dist = (pos - c_alpha_posit).magnitude();
        if dist < closest {
            closest = dist;
            closest_sc = pos;
        }
    }
    let bond_ca_sidechain = c_alpha_posit - closest_sc;

    hydrogens.push(Atom {
        posit: c_alpha_posit
            + tetra_legs(
                -bond_ca_n.to_normalized(),
                bond_cp_ca.to_normalized(),
                -bond_ca_sidechain.to_normalized(),
            ) * LEN_CALPHA_H,
        ..h_default.clone()
    });

    // Handle sidechains.
    // todo: If this algorithm proves general enough, perhaps apply it to the backbone as well (?)
    for (i, atom) in atoms.into_iter().enumerate() {
        // println!("Atom: {}, {:?}", atom.element.to_letter(), atom.role);

        if let Some(role) = atom.role {
            match role {
                AtomRole::Sidechain => {
                    match atom.element {
                        Element::Carbon => {
                            // todo: Function for this A/R.
                            let atoms_bonded: Vec<&Atom> = atoms
                                .into_iter()
                                .enumerate()
                                .filter(|(j, a)| {
                                    // todo: Handling only all-carbon bonds here for now.
                                    // i != *j && a.element == Element::Carbon && (a.posit - atom.posit).magnitude() < 1.55
                                    i != *j && (a.posit - atom.posit).magnitude() < 1.55
                                })
                                .map(|(_, a)| *a)
                                .collect();

                            // todo: Handle O bonded (doubleu bonds).
                            match atoms_bonded.len() {
                                1 => unsafe {
                                    // todo: For now, this H array has an arbitrary rotation. I think the move may be to lock
                                    // todo it to something that counter-aligns to the next hub's atoms.
                                    // Add 3 H
                                    let rotator = Quaternion::from_unit_vecs(
                                        TETRA_A,
                                        (atoms_bonded[0].posit - atom.posit).to_normalized(),
                                    );
                                    for tetra in [TETRA_B, TETRA_C, TETRA_D] {
                                        hydrogens.push(Atom {
                                            posit: atom.posit + rotator.rotate_vec(tetra) * LEN_C_H,
                                            ..h_default.clone()
                                        });
                                    }
                                },
                                2 => unsafe {
                                    // Add 2 H.
                                    let (h_0, h_1) = tetra_atoms_2(
                                        atom.posit,
                                        atoms_bonded[0].posit,
                                        atoms_bonded[1].posit,
                                        LEN_C_H,
                                    );

                                    for posit in [h_0, h_1] {
                                        hydrogens.push(Atom {
                                            posit,
                                            ..h_default.clone()
                                        });
                                    }
                                },
                                3 => {
                                    if atoms_bonded[0].element == Element::Oxygen
                                        || atoms_bonded[1].element == Element::Oxygen
                                        || atoms_bonded[2].element == Element::Oxygen
                                    {
                                        continue;
                                    }
                                    // Add 1 H.
                                    hydrogens.push(Atom {
                                        posit: atom.posit
                                            // + tetra_atoms(
                                            - tetra_atoms(
                                                atom.posit,
                                                atoms_bonded[0].posit,
                                                atoms_bonded[1].posit,
                                                atoms_bonded[2].posit,
                                                // -bond_ca_n.to_normalized(),
                                                // bond_cp_ca.to_normalized(),
                                                // -bond_ca_sidechain.to_normalized(),
                                            ) * LEN_CALPHA_H,
                                        ..h_default.clone()
                                    });
                                }
                                _ => (),
                            }
                        }
                        Element::Nitrogen => {}
                        _ => {}
                    }
                }
                _ => (),
            }
        }
    }

    Ok((dihedral, hydrogens, c_p_posit, c_alpha_posit))
}
