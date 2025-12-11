use std::f64::consts::TAU;

use bio_files::BondType;
use dynamics::find_planar_posit;
use lin_alg::f64::{Quaternion, Vec3, X_VEC, Z_VEC};
use na_seq::{
    AtomTypeInRes,
    Element::{self, Carbon, Hydrogen, Nitrogen, Oxygen},
};

use crate::molecule::{Atom, Bond};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Template {
    /// Carboxylic acid
    Cooh,
    Amide,
    AromaticRing,
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
            Self::AromaticRing => ar_ring(anchor, r_aligner, start_sn, start_i),
            _ => Default::default(),
        }
    }
}

/// Atom 0 is placed on the anchor; the r_group is used to match a "previous"
/// atom in the mol we're adding this to.
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

    let mut atoms = Vec::with_capacity(3);
    let mut bonds = Vec::with_capacity(2);

    for (i, posit) in posits.into_iter().enumerate() {
        let serial_number = start_sn + i as u32;

        atoms.push(Atom {
            serial_number,
            posit,
            element: ELEMENTS[i],
            ..Default::default()
        })
    }

    bonds.push(Bond {
        bond_type: BondType::Double,
        atom_0_sn: atoms[0].serial_number,
        atom_1_sn: atoms[1].serial_number,
        atom_0: start_i,
        atom_1: start_i + 1,
        is_backbone: false,
    });

    bonds.push(Bond {
        bond_type: BondType::Single,
        atom_0_sn: atoms[0].serial_number,
        atom_1_sn: atoms[2].serial_number,
        atom_0: start_i,
        atom_1: start_i + 2,
        is_backbone: false,
    });

    (atoms, bonds)
}

/// See comments on `cooh_group`.
fn amide_group(
    anchor: Vec3,
    aligner: Vec3,
    start_sn: u32,
    start_i: usize,
) -> (Vec<Atom>, Vec<Bond>) {
    const LEN: f64 = 1.33;
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

    let mut atoms = Vec::with_capacity(2);
    let mut bonds = Vec::with_capacity(2);

    for (i, posit) in posits.into_iter().enumerate() {
        let serial_number = start_sn + i as u32;

        atoms.push(Atom {
            serial_number,
            posit,
            element: ELEMENTS[i],
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

// todo: What does posit anchor too? Center? An corner marked in a certain way?
// fn ar_ring(anchor: Vec3, orientation: Quaternion, start_sn: u32, start_i: usize) -> (Vec<Atom>, Vec<Bond>) {
fn ar_ring(anchor: Vec3, r_aligner: Vec3, start_sn: u32, start_i: usize) -> (Vec<Atom>, Vec<Bond>) {
    const POSITS: [Vec3; 6] = [
        Vec3::new(1.3970, 0.0000, 0.0),
        Vec3::new(0.6985, 1.2090, 0.0),
        Vec3::new(-0.6985, 1.2090, 0.0),
        Vec3::new(-1.3970, 0.0000, 0.0),
        Vec3::new(-0.6985, -1.2090, 0.0),
        Vec3::new(0.6985, -1.2090, 0.0),
    ];

    // let posits = POSITS.iter().map(|p| *p + anchor);
    let orientation = Quaternion::new_identity(); // todo temp
    let posits = POSITS.iter().map(|p| (orientation.rotate_vec(*p)) + anchor);

    let mut atoms = Vec::with_capacity(6);

    for (i, posit) in posits.enumerate() {
        let serial_number = start_sn + i as u32;

        atoms.push(Atom {
            serial_number,
            posit,
            element: Carbon,
            type_in_res: Some(AtomTypeInRes::CA), // todo: A/R
            ..Default::default()
        })
    }

    let mut bonds = Vec::with_capacity(6);
    for i in 0..6 {
        let i_next = i % 6; // Wrap 6 to 0.

        let i_0_global = start_i + i;
        let i_1_global = start_i + (i + 1) % 6;

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
