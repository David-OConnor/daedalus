//! Adapted from `peptide`. Operations related to the geometry of atomic coordinates.

use std::{f64::consts::TAU, fmt, fmt::Formatter};

use lin_alg::f64::{Quaternion, Vec3, det_from_cols};
use na_seq::AminoAcid;

use crate::{
    aa_coords::{
        bond_vecs::{
            LEN_C_H, LEN_CALPHA_H, LEN_N_H, LEN_O_H, PLANAR3_A, PLANAR3_B, PLANAR3_C, TETRA_A,
            TETRA_B, TETRA_C, TETRA_D,
        },
        sidechain::Sidechain,
    },
    element::{Element, Element::Hydrogen},
    molecule::{Atom, AtomRole, ResidueType},
};

pub mod bond_vecs;
pub mod sc_atom_placement;
pub mod sidechain;

pub struct PlacementError {}

// From Peptide. Radians.
pub const PHI_HELIX: f64 = -0.715584993317675;
pub const PSI_HELIX: f64 = -0.715584993317675;
pub const PHI_SHEET: f64 = -140. * TAU / 360.;
pub const PSI_SHEET: f64 = 135. * TAU / 360.;

// The dihedral angle must be within this of [0 | TAU | TAU/2] for atoms to be considered planar.
const PLANAR_DIHEDRAL_THRESH: f64 = 0.4;

// The angle between adjacent bonds must be greater than this for a bond to be considered triplanar,
// vice tetrahedral. Tetra ideal: 1.91. Planar idea: 2.094
// todo: This seems high, but produces better results from oneo data set.d
const PLANAR_ANGLE_THRESH: f64 = 2.00; // Higher means more likely to classify as tetrahedral.

const SP2_PLANAR_ANGLE: f64 = TAU / 3.;

struct BondError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Hybridization {
    /// Linear geometry. E.g. carbon bonded to 2 atoms.
    Sp,
    /// Planar geometry. E.g. carbon bonded to 3 atoms.
    Sp2,
    /// Tetrahedral geometry. E.g. carbon bonded to 4 atoms.
    Sp3,
}

/// An amino acid in a protein structure, including all dihedral angles required to determine
/// the conformation. Includes backbone and side chain dihedral angles. Doesn't store coordinates,
/// but coordinates can be generated using forward kinematics from the angles.
#[derive(Debug, Clone, Default)]
pub struct Dihedral {
    /// Dihedral angle between C' and N
    /// Tor (Cα, C', N, Cα) is the ω torsion angle. None if the starting residue on a chain.
    /// Assumed to be τ/2 for most cases
    pub ω: Option<f64>,
    /// Dihedral angle between Cα and N.
    /// Tor (C', N, Cα, C') is the φ torsion angle. None if the starting residue on a chain.
    pub φ: Option<f64>,
    /// Dihedral angle, between Cα and C'
    ///  Tor (N, Cα, C', N) is the ψ torsion angle. None if the final residue on a chain.
    pub ψ: Option<f64>,
    // /// Contains the χ angles that define t
    pub sidechain: Sidechain,
    // pub dipole: Vec3,
}

impl fmt::Display for Dihedral {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut result = String::new();

        // todo: Sort out the initial space on the first item.

        if let Some(ω) = self.ω {
            result = format!("  ω: {:.2}τ", ω / TAU) + " " + &result;
        }

        if let Some(φ) = self.φ {
            result += &format!("  φ: {:.2}τ", φ / TAU);
        }

        if let Some(ψ) = self.ψ {
            result += &format!("  ψ: {:.2}τ", ψ / TAU);
        }
        write!(f, "{result}")?;
        Ok(())
    }
}

/// Calculate the dihedral angle between 4 positions.
/// The `bonds` are one atom substracted from the next.
/// todo: Move to `lin_alg` lib?
pub fn calc_dihedral_angle(bond_middle: Vec3, bond_adjacent1: Vec3, bond_adjacent2: Vec3) -> f64 {
    // Project the next and previous bonds onto the plane that has this bond as its normal.
    // Re-normalize after projecting.
    let bond1_on_plane = bond_adjacent1.project_to_plane(bond_middle).to_normalized();
    let bond2_on_plane = bond_adjacent2.project_to_plane(bond_middle).to_normalized();

    // Not sure why we need to offset by 𝜏/2 here, but it seems to be the case
    let result = bond1_on_plane.dot(bond2_on_plane).acos() + TAU / 2.;

    if result.is_nan() {
        // return 0.; // todo: What causes this? todo: Confirm it shouldn't be tau/2.
    }

    // The dot product approach to angles between vectors only covers half of possible
    // rotations; use a determinant of the 3 vectors as matrix columns to determine if what we
    // need to modify is on the second half.
    let det = det_from_cols(bond1_on_plane, bond2_on_plane, bond_middle);

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
    // Move from world-space to local.
    let bond_0 = (atom_0 - center).to_normalized();
    let bond_1 = (center - atom_1).to_normalized();

    // Aligns the tetrahedron leg A to bond 0.
    let rotator_a = Quaternion::from_unit_vecs(TETRA_A, bond_0);

    // Once the TETRA_A is aligned to bond_0, rotate the tetrahedron around this until TETRA_B aligs
    // with bond_1. Then, the other two tetra parts will be where we place our hydrogens.
    let tetra_b_rotated = rotator_a.rotate_vec(unsafe { TETRA_B });

    let dihedral = calc_dihedral_angle(bond_0, tetra_b_rotated, bond_1);

    let rotator_b = Quaternion::from_axis_angle(bond_0, -dihedral);

    let rotator = rotator_b * rotator_a;

    unsafe {
        (
            center + rotator.rotate_vec(TETRA_C) * len,
            center + rotator.rotate_vec(TETRA_D) * len,
        )
    }
}

/// Find the position of the third planar (SP2) atom.
fn planar_posit(posit_center: Vec3, bond_0: Vec3, bond_1: Vec3, len: f64) -> Vec3 {
    let bond_0_unit = bond_0.to_normalized();
    let n_plane_normal = bond_0_unit.cross(bond_1).to_normalized();
    let rotator = Quaternion::from_axis_angle(n_plane_normal, SP2_PLANAR_ANGLE);

    posit_center + rotator.rotate_vec(-bond_0_unit) * len
}

/// Find atoms covalently bonded to a given atom. The set of `atoms` must be small, or performance
/// will suffer. If unable to pre-filter, use a grid-approach like we do for the general bonding algorith .
fn find_bonded_atoms<'a>(
    atom: &'a Atom,
    atoms: &[&'a Atom],
    atom_i: usize,
) -> Vec<(usize, &'a Atom)> {
    atoms
        .iter()
        .enumerate()
        .filter(|(j, a)| {
            // todo: Adj this len A/R, or calc it per-branch with a fn.
            atom_i != *j && (a.posit - atom.posit).magnitude() < 1.80 && a.element != Hydrogen
            // atom_i != *j && (a.posit - atom.posit).magnitude() < 1.40
        })
        .map(|(j, a)| (j, *a))
        .collect()
}

/// Find bonds from the next (or prev) to current, and an arbitrary 2 offset from the next. Useful for finding
/// dihedral angles on sidechains, etc.
fn get_prev_bonds(
    atom: &Atom,
    atoms: &[&Atom],
    atom_i: usize,
    atom_next: (usize, &Atom),
) -> Result<(Vec3, Vec3), BondError> {
    // Find atoms one step farther down the chain.
    let bonded_to_next: Vec<(usize, &Atom)> = find_bonded_atoms(atom_next.1, atoms, atom_next.0)
        .into_iter()
        // Don't include the original atom in this list.
        .filter(|a| a.0 != atom_i)
        .collect();

    if bonded_to_next.is_empty() {
        return Err(BondError {});
    }

    // Arbitrary one.
    let atom_2_after = bonded_to_next[0].1;

    let this_to_next = (atom_next.1.posit - atom.posit).to_normalized();
    let next_to_2after = (atom_next.1.posit - atom_2_after.posit).to_normalized();

    Ok((this_to_next, next_to_2after))
}

/// Add hydrogens for side chains; this is more general than the initial logic that takes
/// care of backbone hydrogens. It doesn't have to be used exclusively for sidechains.
fn add_h_sidechain(hydrogens: &mut Vec<Atom>, atoms: &[&Atom], h_default: &Atom) {
    // Handle sidechains.
    // todo: If this algorithm proves general enough, perhaps apply it to the backbone as well (?)

    let h_default_sc = Atom {
        role: Some(AtomRole::H_Sidechain),
        ..h_default.clone()
    };

    for (i, atom) in atoms.iter().enumerate() {
        if atom.role.is_none() {
            continue;
        }
        // todo: Experimenting with the first/last residues, to get
        // if role != AtomRole::Sidechain && prev_cp_ca.is_some() && next_n.is_some() {
        if atom.role.as_ref().unwrap() != &AtomRole::Sidechain {
            continue;
        }

        let atoms_bonded = find_bonded_atoms(atom, atoms, i);

        match atom.element {
            Element::Carbon => {
                // todo: Handle O bonded (double bonds).
                match atoms_bonded.len() {
                    1 => unsafe {
                        // Methyl.
                        // todo: DRY with your Amine code below
                        let (bond_prev, bond_back2) =
                            match get_prev_bonds(atom, atoms, i, atoms_bonded[0]) {
                                Ok(v) => v,
                                Err(_) => {
                                    eprintln!("Error: Could not find prev bonds on Methyl");
                                    continue;
                                }
                            };

                        // Initial rotator to align the tetrahedral geometry; positions almost correctly,
                        // but needs an additional rotation around the bond vec axis.
                        let rotator_a = Quaternion::from_unit_vecs(TETRA_A, bond_prev);

                        let tetra_rotated = rotator_a.rotate_vec(TETRA_B);
                        let dihedral = calc_dihedral_angle(bond_prev, tetra_rotated, bond_back2);

                        // Offset; don't align; avoids steric hindrence.
                        let rotator_b =
                            Quaternion::from_axis_angle(bond_prev, -dihedral + TAU / 6.);
                        let rotator = rotator_b * rotator_a;

                        for tetra_bond in [TETRA_B, TETRA_C, TETRA_D] {
                            hydrogens.push(Atom {
                                posit: atom.posit + rotator.rotate_vec(tetra_bond) * LEN_C_H,
                                ..h_default_sc.clone()
                            });
                        }
                    },
                    2 => unsafe {
                        let mut planar = false;
                        if atoms_bonded[0].1.element == Element::Nitrogen
                            && atoms_bonded[1].1.element == Element::Nitrogen
                        {
                            planar = true;
                        } else {
                            // Rings. Calculate dihedral angle to assess if a flat geometry.
                            // todo: C+P. DRY.

                            // todo: Our check using dihedral angles is having trouble. Try this: A simple
                            // check for a typical planar-arrangemenet angle.
                            // note: Next and prev here are arbitrary.
                            let bond_next = (atoms_bonded[1].1.posit - atom.posit).to_normalized();
                            let bond_prev = (atoms_bonded[0].1.posit - atom.posit).to_normalized();

                            let angle = (bond_next.dot(bond_prev)).acos();

                            if angle > PLANAR_ANGLE_THRESH {
                                planar = true;
                            }

                            // // Check the atoms in both directions for a flat dihedral angle.
                            // for bonded in [atoms_bonded[0], atoms_bonded[1]] {
                            //     // todo: Perhaps you have to try the opposit eorder
                            //     let (bond_prev, bond_back2) =
                            //         match get_prev_bonds(atom, atoms, i, bonded) {
                            //             Ok(v) => v,
                            //             Err(_) => {
                            //                 eprintln!("Error: Could not find prev bonds when examining ring config");
                            //                 continue; // todo: Don't continue, just assume tetra. (?)
                            //             }
                            //         };
                            //
                            //
                            //
                            //     let dihedral = calc_dihedral_angle(bond_prev, -bond_next, -bond_back2);
                            //
                            //     if let ResidueType::AminoAcid(aa) = atom.residue_type {
                            //         if aa == AminoAcid::Tyr || aa == AminoAcid::Phe {
                            //             println!("Dihedral: {:?}", dihedral);
                            //             println!("Angle: {:?}", angle);
                            //         }
                            //     }
                            //
                            //     // todo: Adjust this thresh A/R.
                            //     // todo: I'm not sure why we need to check for both tau/2 and tau.
                            //     //     if (dihedral - TAU/2.).abs() < PLANAR_ANGLE_THRESH || dihedral.abs() < PLANAR_ANGLE_THRESH ||(TAU - dihedral.abs()) < PLANAR_ANGLE_THRESH {
                            //     if dihedral < PLANAR_ANGLE_THRESH || (TAU - dihedral).abs() < PLANAR_ANGLE_THRESH {
                            //         planar_dist = Some(LEN_C_H);
                            //         println!("Planar due to ring");
                            //         if let ResidueType::AminoAcid(aa) = atom.residue_type {
                            //             println!("AA: {:?}", aa);
                            //         }
                            //         break;
                            //     }
                            // }
                        }

                        if planar {
                            let bond_0 = atom.posit - atoms_bonded[0].1.posit;
                            let bond_1 = atoms_bonded[1].1.posit - atom.posit;
                            // Add a single H in planar config.
                            hydrogens.push(Atom {
                                posit: planar_posit(atom.posit, bond_0, bond_1, LEN_C_H),
                                ..h_default_sc.clone()
                            });

                            continue;
                        }

                        // Add 2 H in a tetrahedral config.
                        let (h_0, h_1) = tetra_atoms_2(
                            atom.posit,
                            atoms_bonded[0].1.posit,
                            atoms_bonded[1].1.posit,
                            LEN_C_H,
                        );

                        for posit in [h_0, h_1] {
                            hydrogens.push(Atom {
                                posit,
                                ..h_default_sc.clone()
                            });
                        }
                    },
                    3 => {
                        if atoms_bonded[0].1.element == Element::Oxygen
                            || atoms_bonded[1].1.element == Element::Oxygen
                            || atoms_bonded[2].1.element == Element::Oxygen
                        {
                            continue;
                        }

                        // Planar N arrangement.
                        if atoms_bonded[0].1.element == Element::Nitrogen
                            && atoms_bonded[1].1.element == Element::Nitrogen
                            && atoms_bonded[2].1.element == Element::Nitrogen
                        {
                            continue;
                        }

                        // Planar C arrangement.
                        let bond_next = (atoms_bonded[1].1.posit - atom.posit).to_normalized();
                        let bond_prev = (atoms_bonded[0].1.posit - atom.posit).to_normalized();

                        let angle = (bond_next.dot(bond_prev)).acos();

                        if angle > PLANAR_ANGLE_THRESH {
                            continue;
                        }

                        // Add 1 H.
                        // todo: If planar geometry, don't add a H!
                        hydrogens.push(Atom {
                            posit: atom.posit
                                - tetra_atoms(
                                    atom.posit,
                                    atoms_bonded[0].1.posit,
                                    atoms_bonded[1].1.posit,
                                    atoms_bonded[2].1.posit,
                                ) * LEN_CALPHA_H,
                            ..h_default_sc.clone()
                        });
                    }
                    _ => (),
                }
            }
            Element::Nitrogen => {
                match atoms_bonded.len() {
                    1 => unsafe {
                        // Add 2 H. (Amine)
                        // todo: DRY with methyl code above
                        let (bond_prev, bond_back2) =
                            match get_prev_bonds(atom, atoms, i, atoms_bonded[0]) {
                                Ok(v) => v,
                                Err(_) => {
                                    eprintln!("Error: Could not find prev bonds on Amine");
                                    continue;
                                }
                            };

                        // Initial rotator to align the tetrahedral geometry; positions almost correctly,
                        // but needs an additional rotation around the bond vec axis.
                        let rotator_a = Quaternion::from_unit_vecs(PLANAR3_A, bond_prev);

                        let planar_3_rotated = rotator_a.rotate_vec(PLANAR3_B);
                        let dihedral = calc_dihedral_angle(bond_prev, planar_3_rotated, bond_back2);

                        let rotator_b = Quaternion::from_axis_angle(bond_prev, -dihedral);
                        let rotator = rotator_b * rotator_a;

                        for planar_bond in [PLANAR3_B, PLANAR3_C] {
                            hydrogens.push(Atom {
                                posit: atom.posit + rotator.rotate_vec(planar_bond) * LEN_N_H,
                                ..h_default_sc.clone()
                            });
                        }
                    },
                    2 => {
                        // Add 1 H.
                        let bond_0 = atom.posit - atoms_bonded[0].1.posit;
                        let bond_1 = atoms_bonded[1].1.posit - atom.posit;

                        hydrogens.push(Atom {
                            posit: planar_posit(atom.posit, bond_0, bond_1, LEN_N_H),
                            ..h_default_sc.clone()
                        });
                    }
                    _ => (),
                }
            }
            Element::Oxygen => {
                match atoms_bonded.len() {
                    1 => unsafe {
                        // Hydroxyl. Add a single H with tetrahedral geometry.
                        // todo: The bonds are coming out right; not sure why.
                        // todo: This segment is DRY with 2+ sections above.
                        let (bond_prev, bond_back2) =
                            match get_prev_bonds(atom, atoms, i, atoms_bonded[0]) {
                                Ok(v) => v,
                                Err(_) => {
                                    eprintln!("Error: Could not find prev bonds on Hydroxyl");
                                    continue;
                                }
                            };

                        let bond_prev_non_norm = atoms_bonded[0].1.posit - atom.posit;
                        // This crude check may force these to only be created on Hydroxyls (?)
                        // Looking for len characterisitic of a single bond vice double.
                        if bond_prev_non_norm.magnitude() < 1.30 {
                            continue;
                        }

                        let rotator_a = Quaternion::from_unit_vecs(TETRA_A, bond_prev);

                        let tetra_rotated = rotator_a.rotate_vec(TETRA_B);
                        let dihedral = calc_dihedral_angle(bond_prev, tetra_rotated, bond_back2);

                        // Offset; don't align; avoids steric hindrence.
                        let rotator_b =
                            Quaternion::from_axis_angle(bond_prev, -dihedral + TAU / 6.);
                        let rotator = rotator_b * rotator_a;

                        hydrogens.push(Atom {
                            posit: atom.posit + rotator.rotate_vec(TETRA_B) * LEN_O_H,
                            ..h_default_sc.clone()
                        });
                    },
                    _ => (),
                }
            }
            _ => {}
        }
    }
}

/// todo: Rename, etc.
/// todo: Infer residue from coords instead of accepting as param?
/// Returns (dihedral angles, H atoms, c'_pos, ca_pos). The parameter and output carbon positions
/// are for use in calculating dihedral angles associated with other  chains.
pub fn aa_data_from_coords(
    atoms: &[&Atom],
    aa: AminoAcid,
    prev_cp_ca: Option<(Vec3, Vec3)>,
    next_n: Option<Vec3>,
) -> (Dihedral, Vec<Atom>, Vec3, Vec3) {
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
    let mut dihedral = Dihedral::default();

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
    }

    // /// Dihedral angle between C' and N
    // /// Tor (Cα, C', N, Cα) is the ω torsion angle
    // /// Assumed to be TAU/2 for most cases
    // pub ω: f64,
    // /// Dihedral angle between Cα and N.
    // /// Tor (C', N, Cα, C') is the φ torsion angle
    // pub φ: f64,
    // /// Dihedral angle, between Cα and C'
    // ///  Tor (N, Cα, C', N) is the ψ torsion angle
    // pub ψ: f64,

    let bond_ca_n = c_alpha_posit - n_posit;
    let bond_cp_ca = c_p_posit - c_alpha_posit;

    // For residues after the first.
    if let Some((prev_cp, prev_ca)) = prev_cp_ca {
        let bond_cp_prev_ca_prev = prev_cp - prev_ca;
        let bond_n_cp_prev = n_posit - prev_cp;
        dihedral.φ = Some(calc_dihedral_angle(bond_ca_n, bond_n_cp_prev, bond_cp_ca));
        dihedral.ω = Some(calc_dihedral_angle(
            bond_n_cp_prev,
            bond_cp_prev_ca_prev,
            bond_ca_n,
        ));

        // todo temp: C' Gly #154 is showing as the coords for Calpha.
        if dihedral.ω.unwrap().is_nan() {
            // println!("\nomega NAN. n_cp: {bond_n_cp_prev}, cp_prev_ca_prev: {bond_cp_prev_ca_prev}, ca-n: {bond_ca_n}");
            println!("NAN: prev_cp: {prev_cp} prev_ca: {prev_ca}\n")
        }

        // Add a H to the backbone N. (Amine) Sp2/Planar.
        hydrogens.push(Atom {
            posit: planar_posit(n_posit, bond_n_cp_prev, bond_ca_n, LEN_N_H),
            ..h_default.clone()
        });
    }

    // For residues prior to the last.
    if let Some(next_n) = next_n {
        let bond_n_next_cp = next_n - c_p_posit;
        dihedral.ψ = Some(calc_dihedral_angle(bond_cp_ca, bond_ca_n, bond_n_next_cp));
    }

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
        if atom_sc.role.as_ref().unwrap() == &AtomRole::Sidechain {
            if atom_sc.element == Element::Carbon {
                posits_sc.push(atom_sc.posit);
            }
        }
    }

    if posits_sc.is_empty() {
        // This generally means the residue is Glycine, which doesn't have a sidechain.

        // Note: This will also populate hydrogens on first and last backbones, and potentially
        // on residues that don't have roles marked.
        add_h_sidechain(&mut hydrogens, atoms, &h_default);
        return (dihedral, hydrogens, c_p_posit, c_alpha_posit);
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

    add_h_sidechain(&mut hydrogens, atoms, &h_default);

    (dihedral, hydrogens, c_p_posit, c_alpha_posit)
}
