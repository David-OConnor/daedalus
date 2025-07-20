//! Adapted from `peptide`. Operations related to the geometry of atomic coordinates.

use std::{f64::consts::TAU, fmt, fmt::Formatter, str::FromStr};

use bio_files::{BondGeneric, ResidueType};
use lin_alg::f64::{Quaternion, Vec3, calc_dihedral_angle, calc_dihedral_angle_v2};
use na_seq::{
    AminoAcid, AtomTypeInRes, Element,
    Element::{Carbon, Hydrogen, Nitrogen, Oxygen},
};

use crate::{
    ProtFfMap,
    aa_coords::{
        bond_vecs::{
            LEN_C_H, LEN_CALPHA_H, LEN_N_H, LEN_O_H, PLANAR3_A, PLANAR3_B, PLANAR3_C, TETRA_A,
            TETRA_B, TETRA_C, TETRA_D,
        },
        sidechain::Sidechain,
    },
    add_hydrogens::{DigitMap, h_type_in_res_sidechain},
    dynamics::ParamError,
    molecule::{Atom, AtomRole, Residue},
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
    // todo: Adj this len A/R, or calc it per-branch with a fn.
    // todo 1.80 seems to work well, but causes, for example, the CG - S bond in Met to be missed.
    // todo: 1.85 works in that case.
    const BONDED_LEN_THRESH: f64 = 1.85;

    atoms
        .iter()
        .enumerate()
        .filter(|(j, a)| {

            atom_i != *j && (a.posit - atom.posit).magnitude() < BONDED_LEN_THRESH && a.element != Hydrogen
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
fn add_h_sidechain(
    hydrogens: &mut Vec<Atom>,
    atoms: &[&Atom],
    h_default: &Atom,
    residues: &[Residue],
    ff_map: &ProtFfMap,
    digit_map: &DigitMap,
) -> Result<(), ParamError> {
    let h_default_sc = Atom {
        role: Some(AtomRole::H_Sidechain),
        ..h_default.clone()
    };

    for (i, atom) in atoms.iter().enumerate() {
        let Some(role) = atom.role else { continue };

        // todo: Experimenting with the first/last residues, to get
        // if role != AtomRole::Sidechain && prev_cp_ca.is_some() && next_n.is_some() {
        if role != AtomRole::Sidechain {
            continue;
        }

        // todo: Don't unwrap! Map to an error.
        let aa = match &residues[*atom.residue.as_ref().unwrap()].res_type {
            ResidueType::AminoAcid(aa) => aa,
            _ => {
                return Err(ParamError::new(&format!(
                    "Attempting to add H to non-AA residue: {atom}"
                )));
            }
        };

        let atoms_bonded = find_bonded_atoms(atom, atoms, i);

        // todo: QC this; likely not correct.
        // let depth = bonded_heavy_atoms(&atoms_bonded).len();
        let Some(parent_tir) = atom.type_in_res.as_ref() else {
            return Err(ParamError::new(&format!(
                "Missing parent type in res when adding H: {atom}"
            )));
        };

        match atom.element {
            Carbon => {
                // todo: Handle O bonded (double bonds).
                match atoms_bonded.len() {
                    1 => unsafe {
                        // Methyl.
                        // todo: DRY with your Amine code below
                        let (bond_prev, bond_back2) =
                            match get_prev_bonds(atom, atoms, i, atoms_bonded[0]) {
                                Ok(v) => v,
                                Err(_) => {
                                    // eprintln!("Error: Could not find prev bonds on Methyl");
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

                        for (i, tetra_bond) in [TETRA_B, TETRA_C, TETRA_D].into_iter().enumerate() {
                            let at =
                                h_type_in_res_sidechain(i, parent_tir, *aa, ff_map, digit_map)?;
                            hydrogens.push(Atom {
                                posit: atom.posit + rotator.rotate_vec(tetra_bond) * LEN_C_H,
                                type_in_res: Some(at),
                                ..h_default_sc.clone()
                            });
                        }
                    },
                    2 => {
                        let mut planar = false;
                        if atoms_bonded[0].1.element == Nitrogen
                            && atoms_bonded[1].1.element == Nitrogen
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

                            let angle = bond_next.dot(bond_prev).acos();

                            if angle > PLANAR_ANGLE_THRESH {
                                planar = true;
                            }
                        }

                        if planar {
                            let bond_0 = atom.posit - atoms_bonded[0].1.posit;
                            let bond_1 = atoms_bonded[1].1.posit - atom.posit;

                            let at =
                                h_type_in_res_sidechain(0, parent_tir, *aa, ff_map, digit_map)?;
                            // Add a single H in planar config.
                            hydrogens.push(Atom {
                                posit: planar_posit(atom.posit, bond_0, bond_1, LEN_C_H),
                                type_in_res: Some(at),
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

                        for (i, posit) in [h_0, h_1].into_iter().enumerate() {
                            let at =
                                h_type_in_res_sidechain(i, parent_tir, *aa, ff_map, digit_map)?;
                            hydrogens.push(Atom {
                                posit,
                                type_in_res: Some(at),
                                ..h_default_sc.clone()
                            });
                        }
                    }
                    3 => {
                        if atoms_bonded[0].1.element == Oxygen
                            || atoms_bonded[1].1.element == Oxygen
                            || atoms_bonded[2].1.element == Oxygen
                        {
                            continue;
                        }

                        // Planar N arrangement.
                        if atoms_bonded[0].1.element == Nitrogen
                            && atoms_bonded[1].1.element == Nitrogen
                            && atoms_bonded[2].1.element == Nitrogen
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
                        let at = h_type_in_res_sidechain(0, parent_tir, *aa, ff_map, digit_map)?;

                        hydrogens.push(Atom {
                            posit: atom.posit
                                - tetra_atoms(
                                    atom.posit,
                                    atoms_bonded[0].1.posit,
                                    atoms_bonded[1].1.posit,
                                    atoms_bonded[2].1.posit,
                                ) * LEN_CALPHA_H,
                            // todo: QC the tetrahedral here.
                            type_in_res: Some(at),
                            ..h_default_sc.clone()
                        });
                    }
                    _ => (),
                }
            }
            Nitrogen => {
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

                        for (i, planar_bond) in [PLANAR3_B, PLANAR3_C].into_iter().enumerate() {
                            let at =
                                h_type_in_res_sidechain(i, parent_tir, *aa, ff_map, digit_map)?;
                            hydrogens.push(Atom {
                                posit: atom.posit + rotator.rotate_vec(planar_bond) * LEN_N_H,
                                type_in_res: Some(at),
                                ..h_default_sc.clone()
                            });
                        }
                    },
                    2 => {
                        // Add 1 H.
                        let bond_0 = atom.posit - atoms_bonded[0].1.posit;
                        let bond_1 = atoms_bonded[1].1.posit - atom.posit;

                        let at = h_type_in_res_sidechain(0, parent_tir, *aa, ff_map, digit_map)?;
                        hydrogens.push(Atom {
                            posit: planar_posit(atom.posit, bond_0, bond_1, LEN_N_H),
                            type_in_res: Some(at),
                            ..h_default_sc.clone()
                        });
                    }
                    _ => (),
                }
            }
            Oxygen => {
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

                        let at = h_type_in_res_sidechain(0, parent_tir, *aa, ff_map, digit_map)?;
                        hydrogens.push(Atom {
                            posit: atom.posit + rotator.rotate_vec(TETRA_B) * LEN_O_H,
                            type_in_res: Some(at),
                            ..h_default_sc.clone()
                        });
                    },
                    _ => (),
                }
            }
            _ => {}
        }
    }

    Ok(())
}

/// Add hydrogens to AA backbone atoms, and update the dihedral angles.
/// Returns (Dihedral, (this c', this ca).
fn handle_backbone(
    hydrogens: &mut Vec<Atom>,
    atoms: &[&Atom],
    posits_sc: &[Vec3],
    prev_cp_ca: Option<(Vec3, Vec3)>,
    next_n: Option<Vec3>,
    h_default: &Atom,
    aa: &AminoAcid,
) -> Result<(Dihedral, Option<(Vec3, Vec3)>), ParamError> {
    let mut dihedral = Dihedral::default();

    // Find the positions of the backbone atoms.
    let mut n_posit = None;
    let mut c_alpha_posit = None;
    let mut c_p_posit = None;

    for atom in atoms {
        let Some(role) = atom.role else { continue };

        match role {
            AtomRole::N_Backbone => {
                n_posit = Some(atom.posit);
            }
            AtomRole::C_Alpha => {
                c_alpha_posit = Some(atom.posit);
            }
            AtomRole::C_Prime => {
                c_p_posit = Some(atom.posit);
            }
            _ => (),
        }
    }

    let (Some(c_alpha_posit), Some(c_p_posit), Some(n_posit)) = (c_alpha_posit, c_p_posit, n_posit)
    else {
        eprintln!("Error: Missing backbone atoms in coords.");
        return Ok((dihedral, None));
    };

    let bond_ca_n = c_alpha_posit - n_posit;
    let bond_cp_ca = c_p_posit - c_alpha_posit;

    // Dihedral angle sequence: ca_prev - cp_prev - n - cα - cp - n_next

    // For residues after the first.
    if let Some((cp_prev, ca_prev)) = prev_cp_ca {
        let bond_n_cp_prev = n_posit - cp_prev;
        dihedral.φ = Some(calc_dihedral_angle_v2(&(
            cp_prev,
            n_posit,
            c_alpha_posit,
            c_p_posit,
        )));
        dihedral.ω = Some(calc_dihedral_angle_v2(&(
            ca_prev,
            cp_prev,
            n_posit,
            c_alpha_posit,
        )));

        if dihedral.ω.unwrap().is_nan() {
            println!("NAN: prev_cp: {cp_prev} prev_ca: {ca_prev}\n")
        }

        // Add a H to the backbone N. (Amine) Sp2/Planar.
        hydrogens.push(Atom {
            posit: planar_posit(n_posit, bond_n_cp_prev, bond_ca_n, LEN_N_H),
            // todo "H" for N backbone always, for now at least.
            type_in_res: Some(AtomTypeInRes::H("H".to_string())),
            ..h_default.clone()
        });
    }

    // For residues prior to the last.
    if let Some(n_next) = next_n {
        dihedral.ψ = Some(calc_dihedral_angle_v2(&(
            n_posit,
            c_alpha_posit,
            c_p_posit,
            n_next,
        )));
    }

    if posits_sc.is_empty() {
        // This generally means the residue is Glycine, which doesn't have a sidechain.
        // Glycine is unique, having 2 H atoms attached to its Cα. It has correspondingly
        // different HA labels. (HA2 and HA3, vice the plain HA).
        // Add 2 H in a tetrahedral config.
        let (h_0, h_1) = tetra_atoms_2(c_alpha_posit, c_p_posit, n_posit, LEN_CALPHA_H);

        hydrogens.push(Atom {
            posit: h_0,
            type_in_res: Some(AtomTypeInRes::H("HA2".to_string())),
            ..h_default.clone()
        });

        hydrogens.push(Atom {
            posit: h_1,
            type_in_res: Some(AtomTypeInRes::H("HA3".to_string())),
            ..h_default.clone()
        });

        return Ok((dihedral, Some((c_p_posit, c_alpha_posit))));
    }

    let mut closest = (posits_sc[0] - c_alpha_posit).magnitude();
    let mut closest_sc = posits_sc[0];

    for pos in posits_sc {
        let dist = (*pos - c_alpha_posit).magnitude();
        if dist < closest {
            closest = dist;
            closest_sc = *pos;
        }
    }
    let bond_ca_sidechain = c_alpha_posit - closest_sc;

    let posit_ha = c_alpha_posit
        + tetra_legs(
            -bond_ca_n.to_normalized(),
            bond_cp_ca.to_normalized(),
            -bond_ca_sidechain.to_normalized(),
        ) * LEN_CALPHA_H;

    // H attached to the α carbon.
    hydrogens.push(Atom {
        posit: posit_ha,
        type_in_res: Some(AtomTypeInRes::H("HA".to_string())),
        ..h_default.clone()
    });

    Ok((dihedral, Some((c_p_posit, c_alpha_posit))))
}

/// todo: Rename, etc.
/// todo: Infer residue from coords instead of accepting as param?
/// Returns (dihedral angles, H atoms, (c'_pos, ca_pos)). The parameter and output carbon positions
/// are for use in calculating dihedral angles associated with other  chains.
pub fn aa_data_from_coords(
    atoms: &[&Atom],
    residue_type: &ResidueType,
    res_i: usize,
    chain_i: usize,
    prev_cp_ca: Option<(Vec3, Vec3)>,
    next_n: Option<Vec3>,
    residues: &[Residue],
    ff_map: &ProtFfMap,
    digit_map: &DigitMap,
) -> Result<(Dihedral, Vec<Atom>, Option<(Vec3, Vec3)>), ParamError> {
    // todo: With_capacity based on aa?

    // todo: Maybe split this into separate functions.

    let h_default = Atom {
        serial_number: 0,
        posit: Vec3::new_zero(),
        element: Hydrogen,
        // We update type_in_res when overriding this default, and ff_type downstream based on it, in the
        // same way we populate FF type (and partial charge) for non-H atoms.
        type_in_res: None,
        force_field_type: None,
        role: Some(AtomRole::H_Backbone),
        residue: Some(res_i),
        chain: Some(chain_i),
        hetero: false,
        dock_type: None,
        occupancy: None,
        partial_charge: None,
        temperature_factor: None,
    };

    // todo: Populate sidechain and main angles now based on coords. (?)

    let mut hydrogens = Vec::new();

    let mut dihedral = Dihedral::default();
    let mut this_cp_ca = None;

    // Find the nearest sidechain atom

    // Add a H to the C alpha atom. Tetrahedral.
    // let ca_plane_normal = bond_ca_n.cross(bond_cp_ca).to_normalized();
    // todo: There are two possible settings available for the rotator; one will be taken up by
    // a sidechain carbon.
    // let rotator = Quaternion::from_axis_angle(ca_plane_normal, TETRA_ANGLE);
    // todo: Another step required using sidechain carbon?
    let mut posits_sc = Vec::new();
    for atom_sc in atoms {
        let Some(role) = atom_sc.role else {
            continue;
        };
        if role == AtomRole::Sidechain && atom_sc.element == Carbon {
            posits_sc.push(atom_sc.posit);
        }
    }

    if let ResidueType::AminoAcid(aa) = residue_type {
        (dihedral, this_cp_ca) = handle_backbone(
            &mut hydrogens,
            atoms,
            &posits_sc,
            prev_cp_ca,
            next_n,
            &h_default,
            aa,
        )?;
    }

    add_h_sidechain(
        &mut hydrogens,
        atoms,
        &h_default,
        residues,
        ff_map,
        digit_map,
    )?;

    Ok((dihedral, hydrogens, this_cp_ca))
}
