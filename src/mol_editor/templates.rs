use bio_files::BondType;
use lin_alg::f64::Vec3;
use na_seq::{
    AtomTypeInRes,
    Element::{self, Carbon, Hydrogen, Oxygen},
};

use crate::molecule::{Atom, Bond};

// todo: What does posit anchor too? Center? An corner marked in a certain way?
pub fn cooh_group(anchor: Vec3, start_sn: u32, start_i: usize) -> (Vec<Atom>, Vec<Bond>) {
    const POSITS: [Vec3; 3] = [
        Vec3::new(0.0000, 0.0000, 0.0), // C (carboxyl)
        Vec3::new(1.2290, 0.0000, 0.0), // O (carbonyl)
        Vec3::new(-0.6715, 1.1645, 0.0), // O (hydroxyl)
                                        // Vec3::new(-1.0286, 1.7826, 0.0), // H (hydroxyl)
    ];

    // todo: Skip the H.
    const ELEMENTS: [Element; 4] = [Carbon, Oxygen, Oxygen, Hydrogen];
    const FF_TYPES: [&str; 4] = ["c", "o", "oh", "ho"]; // GAFF2-style
    const CHARGES: [f32; 4] = [0.70, -0.55, -0.61, 0.44]; // todo: A/R

    let posits = POSITS.iter().map(|p| *p + anchor);

    let mut atoms = Vec::with_capacity(3);
    let mut bonds = Vec::with_capacity(3);

    for (i, posit) in posits.enumerate() {
        let serial_number = start_sn + i as u32;

        atoms.push(Atom {
            serial_number,
            posit,
            element: ELEMENTS[i],
            type_in_res: None,                              // todo: no; fix this
            force_field_type: Some(FF_TYPES[i].to_owned()), // todo: A/R
            partial_charge: Some(CHARGES[i]),               // todo: A/R,
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
        atom_0_sn: atoms[1].serial_number,
        atom_1_sn: atoms[2].serial_number,
        atom_0: start_i + 1,
        atom_1: start_i + 2,
        is_backbone: false,
    });

    (atoms, bonds)
}

// todo: What does posit anchor too? Center? An corner marked in a certain way?
pub fn ar_ring(anchor: Vec3, start_sn: u32, start_i: usize) -> (Vec<Atom>, Vec<Bond>) {
    const POSITS: [Vec3; 6] = [
        Vec3::new(1.3970, 0.0000, 0.0),
        Vec3::new(0.6985, 1.2090, 0.0),
        Vec3::new(-0.6985, 1.2090, 0.0),
        Vec3::new(-1.3970, 0.0000, 0.0),
        Vec3::new(-0.6985, -1.2090, 0.0),
        Vec3::new(0.6985, -1.2090, 0.0),
    ];

    let posits = POSITS.iter().map(|p| *p + anchor);

    let mut atoms = Vec::with_capacity(6);

    for (i, posit) in posits.enumerate() {
        let serial_number = start_sn + i as u32;

        atoms.push(Atom {
            serial_number,
            posit,
            element: Carbon,
            type_in_res: Some(AtomTypeInRes::CA), // todo: A/R
            force_field_type: Some("ca".to_owned()),
            partial_charge: Some(-0.115), // tood: Ar. -0.06 - 0.012 etc.
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
