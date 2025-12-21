use std::f64::consts::TAU;

use bio_files::BondType;
use dynamics::find_planar_posit;
use lin_alg::f64::{Quaternion, Vec3, X_VEC, Z_VEC};
use na_seq::{
    AtomTypeInRes,
    Element::{self, Carbon, Hydrogen, Nitrogen, Oxygen},
};

use crate::molecule::{Atom, Bond};

// todo: Deprecate in place of algoirthmetc approach
const POSITS_AR_RING: [Vec3; 6] = [
    Vec3::new_zero(),
    Vec3::new(-0.6985, 1.2090, 0.0),
    Vec3::new(-2.0955, 1.2090, 0.0),
    Vec3::new(-2.7940, 0.0000, 0.0),
    Vec3::new(-2.0955, -1.2090, 0.0),
    Vec3::new(-0.6985, -1.2090, 0.0),
];

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Template {
    /// Carboxylic acid
    Cooh,
    Amide,
    AromaticRing,
    PentaRing,
}

impl Template {
    pub(in crate::mol_editor) fn atoms_bonds(
        &self,
        anchor: Vec3,
        r_aligner: Vec3,
        start_sn: u32,
        start_i: usize,
    ) -> (Vec<Atom>, Vec<Bond>) {
        match self {
            Self::Cooh => cooh_group(anchor, r_aligner, start_sn, start_i),
            Self::Amide => amide_group(anchor, r_aligner, start_sn, start_i),
            Self::AromaticRing => ring(anchor, r_aligner, start_sn, start_i, &POSITS_AR_RING),
            Self::PentaRing => ring(anchor, r_aligner, start_sn, start_i, &POSITS_AR_RING), // todo temp
            _ => Default::default(),
        }
    }
}

/// Atom 0 is placed on the anchor; the r_group is used to match a "previous"
/// atom in the mol we're adding this to.
///
/// Example: Acetic acid
fn cooh_group(
    anchor: Vec3,
    aligner: Vec3,
    start_sn: u32,
    start_i: usize,
) -> (Vec<Atom>, Vec<Bond>) {
    // Atom 0 is must be the 0 vec. (Or we will have to offset everything until it is)
    const LEN_HYDROXYL: f64 = 1.362;
    const LEN_CARBONYL: f64 = 1.227;
    let mut posits = vec![Vec3::new_zero(), Vec3::new(LEN_HYDROXYL, 0., 0.)];

    let rot_hydr = Quaternion::from_axis_angle(Z_VEC, TAU / 3.);
    posits.push(rot_hydr.rotate_vec(posits[1]).to_normalized() * LEN_CARBONYL);

    // Used to orient the molecule.
    let r_group_local = find_planar_posit(posits[0], posits[1], posits[2]);

    // Rotates the molecule to the correct global orientation.
    let rotator = Quaternion::from_unit_vecs(
        r_group_local.to_normalized(),
        (aligner - anchor).to_normalized(),
    );

    // Align and rotate the local atom positions to global ones.
    let posits: Vec<_> = posits
        .iter()
        .map(|p| rotator.rotate_vec(*p) + anchor)
        .collect();

    const ELEMENTS: [Element; 3] = [Carbon, Oxygen, Oxygen];
    const FF_TYPES: [&str; 3] = ["c", "oh", "o"];

    let mut atoms = Vec::with_capacity(3);
    let mut bonds = Vec::with_capacity(2);

    for (i, posit) in posits.into_iter().enumerate() {
        let serial_number = start_sn + i as u32;

        atoms.push(Atom {
            serial_number,
            posit,
            element: ELEMENTS[i],
            force_field_type: Some(String::from(FF_TYPES[i])),
            ..Default::default()
        })
    }

    bonds.push(Bond {
        bond_type: BondType::Single,
        atom_0_sn: atoms[0].serial_number,
        atom_1_sn: atoms[1].serial_number,
        atom_0: start_i,
        atom_1: start_i + 1,
        is_backbone: false,
    });

    bonds.push(Bond {
        bond_type: BondType::Double,
        atom_0_sn: atoms[0].serial_number,
        atom_1_sn: atoms[2].serial_number,
        atom_0: start_i,
        atom_1: start_i + 2,
        is_backbone: false,
    });

    (atoms, bonds)
}

/// See comments on `cooh_group`.
/// Ref example: Formaldehyde
fn amide_group(
    anchor: Vec3,
    aligner: Vec3,
    start_sn: u32,
    start_i: usize,
) -> (Vec<Atom>, Vec<Bond>) {
    const LEN: f64 = 1.37;
    let mut posits = vec![Vec3::new_zero(), Vec3::new(LEN, 0., 0.)];

    let r_group_local = -X_VEC;

    // Rotates the molecule to the correct global orientation.
    let rotator = Quaternion::from_unit_vecs(
        r_group_local.to_normalized(),
        (aligner - anchor).to_normalized(),
    );

    // Align and rotate the local atom positions to global ones.
    let posits: Vec<_> = posits
        .iter()
        .map(|p| rotator.rotate_vec(*p) + anchor)
        .collect();

    const ELEMENTS: [Element; 2] = [Carbon, Nitrogen];
    const FF_TYPES: [&str; 2] = ["c", "nt"];

    let mut atoms = Vec::with_capacity(2);
    let mut bonds = Vec::with_capacity(2);

    for (i, posit) in posits.into_iter().enumerate() {
        let serial_number = start_sn + i as u32;

        atoms.push(Atom {
            serial_number,
            posit,
            element: ELEMENTS[i],
            force_field_type: Some(String::from(FF_TYPES[i])),
            ..Default::default()
        })
    }

    bonds.push(Bond {
        bond_type: BondType::Single,
        atom_0_sn: atoms[0].serial_number,
        atom_1_sn: atoms[1].serial_number,
        atom_0: start_i,
        atom_1: start_i + 1,
        is_backbone: false,
    });

    (atoms, bonds)
}

// todo: C+P from nucleic acids
fn rotate_about_axis(posit: Vec3, pivot: Vec3, axis: Vec3, angle: f64) -> Vec3 {
    let q = Quaternion::from_axis_angle(axis, angle);
    pivot + q.rotate_vec(posit - pivot)
}

// todo: What does posit anchor too? Center? An corner marked in a certain way?
// fn ar_ring(anchor: Vec3, orientation: Quaternion, start_sn: u32, start_i: usize) -> (Vec<Atom>, Vec<Bond>) {
// fn ar_ring(anchor: Vec3, r_aligner: Vec3, start_sn: u32, start_i: usize) -> (Vec<Atom>, Vec<Bond>) {
fn ring(
    anchor_0: Vec3,
    anchor_1: Vec3,
    start_sn: u32,
    start_i: usize,
    posits: &[Vec3],
) -> (Vec<Atom>, Vec<Bond>) {
    // todo: Create this algorithmically from number of points and bond len?
    let n = posits.len();

    // Move the ring to the anchor. (Assumes point 0 is the 0 vec)
    let mut posits: Vec<_> = posits.iter().map(|p| *p + anchor_0).collect();

    // Rotate the ring so that point 1 is at anchor 1.
    let dir_0_to_1_local = (posits[1] - posits[0]).to_normalized();
    let dir_0_to_1_global = (anchor_1 - anchor_0).to_normalized();

    // let rotator = Quaternion::from_unit_vecs(dir_0_to_1_local, dir_0_to_1_global);
    let rotate_amt = dir_0_to_1_global.dot(dir_0_to_1_local).acos();

    let axis = Z_VEC;
    for posit in &mut posits {
        *posit = rotate_about_axis(*posit, anchor_0, axis, rotate_amt);
    }
    // Note that the ring should be mostly correct now, but adjust posits[1] to be exactly at anchor_1.
    posits[1] = anchor_1;

    let mut atoms = Vec::with_capacity(6);

    for (i, posit) in posits.into_iter().enumerate() {
        let serial_number = start_sn + i as u32;

        atoms.push(Atom {
            serial_number,
            posit,
            element: Carbon,
            type_in_res: Some(AtomTypeInRes::CA), // todo: A/R
            ..Default::default()
        })
    }

    let mut bonds = Vec::with_capacity(n);
    for i in 0..n {
        let i_next = i % n; // Wrap 6 to 0.

        let i_0_global = start_i + i;
        let i_1_global = start_i + (i + 1) % n;

        bonds.push(Bond {
            bond_type: BondType::Aromatic,
            atom_0_sn: atoms[i].serial_number,
            atom_1_sn: atoms[i_next].serial_number,
            atom_0: i_0_global,
            atom_1: i_1_global,
            is_backbone: false,
        });
    }

    (atoms, bonds)
}
