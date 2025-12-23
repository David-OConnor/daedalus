use std::f64::consts::TAU;

use bio_files::BondType;
use dynamics::find_planar_posit;
use lin_alg::f64::{Quaternion, Vec3, X_VEC, Z_VEC};
use na_seq::{
    AtomTypeInRes,
    Element::{self, Carbon, Hydrogen, Nitrogen, Oxygen},
};

use crate::molecule::{Atom, Bond};

// // todo: Deprecate in place of algoirthmetc approach
// const POSITS_AR_RING: [Vec3; 6] = [
//     Vec3::new_zero(),
//     Vec3::new(-0.6985, 1.2090, 0.0),
//     Vec3::new(-2.0955, 1.2090, 0.0),
//     Vec3::new(-2.7940, 0.0000, 0.0),
//     Vec3::new(-2.0955, -1.2090, 0.0),
//     Vec3::new(-0.6985, -1.2090, 0.0),
// ];
//
// // todo temp/placeholder
// const POSITS_PENT_RING: [Vec3; 5] = [
//     Vec3::new_zero(),
//     Vec3::new(-0.6985, 1.2090, 0.0),
//     Vec3::new(-2.0955, 1.2090, 0.0),
//     Vec3::new(-2.7940, 0.0000, 0.0),
//     Vec3::new(-2.0955, -1.2090, 0.0),
// ];

const BOND_LEN_AROMATIC: f64 = 1.39;
const BOND_LEN_PENT_SAT: f64 = 1.53; // C-C single bonds
const BOND_LEN_PENT_UNSAT: f64 = 1.53; // A C=C in the ring

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Template {
    /// Carboxylic acid
    Cooh,
    Amide,
    AromaticRing,
    /// 6 Carbon atoms single-bonded.
    Cyclohexane,
    PentaRing,
}

impl Template {
    pub(in crate::mol_editor) fn atoms_bonds(
        self,
        anchor_is: &[usize],
        anchor_sns: &[u32],
        anchors: &[Vec3], // Len = 0 or 1.
        r_aligners: &[Vec3],
        start_sn: u32,
        start_i: usize,
    ) -> (Vec<Atom>, Vec<Bond>) {
        match self {
            Self::Cooh => cooh_group(anchors[0], r_aligners[0], start_sn, start_i),
            Self::Amide => amide_group(anchors[0], r_aligners[0], start_sn, start_i),
            Self::AromaticRing | Self::Cyclohexane | Self::PentaRing => ring(
                self, anchor_is, anchor_sns, anchors, r_aligners, start_sn, start_i,
            ),
        }
    }

    pub(in crate::mol_editor) fn is_ring(self) -> bool {
        matches!(
            self,
            Self::AromaticRing | Self::Cyclohexane | Self::PentaRing
        )
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

/// Construct a ring of Carbons of len 5 or 6. The anchor[s] are atoms part of the structure we're
/// adding the ring to, which will be part of the ring. So for a 6-atom ring, with one anchor, we add 5
/// atoms to it, roughly in plane with any other atoms bonded to the anchor. With two anchors, we add 4 atoms
/// too it, and the bond between anchor 0 and anchor 1 is part of the ring.
fn ring(
    template: Template,
    anchor_is: &[usize],
    anchor_sns: &[u32],
    anchors: &[Vec3],
    r_aligners: &[Vec3],
    start_sn: u32,
    start_i: usize,
) -> (Vec<Atom>, Vec<Bond>) {
    let num_atoms = match template {
        Template::AromaticRing | Template::Cyclohexane => 6,
        Template::PentaRing => 5,
        _ => unreachable!(),
    };

    assert!(anchors.len() == 1 || anchors.len() == 2);
    assert_eq!(anchor_is.len(), anchors.len());
    assert_eq!(anchor_sns.len(), anchors.len());

    let n = num_atoms;
    let angle = TAU / (n as f64);
    let bond_len = BOND_LEN_AROMATIC;

    let eps = 1e-12;

    let dot = |a: Vec3, b: Vec3| a.x * b.x + a.y * b.y + a.z * b.z;
    let cross = |a: Vec3, b: Vec3| {
        Vec3::new(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x,
        )
    };
    let len2 = |v: Vec3| dot(v, v);
    let len = |v: Vec3| len2(v).sqrt();
    let normalize = |v: Vec3, fallback: Vec3| {
        let l = len(v);
        if l > eps { v / l } else { fallback }
    };

    let x_ref = Vec3::new(1.0, 0.0, 0.0);
    let y_ref = Vec3::new(0.0, 1.0, 0.0);
    let z_ref = Vec3::new(0.0, 0.0, 1.0);

    let sin_pi_over_n = (std::f64::consts::PI / (n as f64)).sin();

    let a0 = anchors[0];

    // Best-fit-ish plane normal from anchor0->(aligners and anchor1).
    let mut dirs: Vec<Vec3> =
        Vec::with_capacity(r_aligners.len() + if anchors.len() == 2 { 1 } else { 0 });
    for p in r_aligners {
        let d = *p - a0;
        if len2(d) > eps {
            dirs.push(normalize(d, x_ref));
        }
    }
    if anchors.len() == 2 {
        let d = anchors[1] - a0;
        if len2(d) > eps {
            dirs.push(normalize(d, y_ref));
        }
    }

    let mut nrm = Vec3::new(0.0, 0.0, 0.0);
    for i in 0..dirs.len() {
        for j in (i + 1)..dirs.len() {
            nrm = nrm + cross(dirs[i], dirs[j]);
        }
    }
    if len2(nrm) <= eps {
        if !dirs.is_empty() {
            let d0 = dirs[0];
            let pref = if dot(d0, z_ref).abs() < 0.9 {
                z_ref
            } else {
                y_ref
            };
            nrm = cross(d0, pref);
        } else {
            nrm = z_ref;
        }
    }
    nrm = normalize(nrm, z_ref);

    // Score: larger is better. Uses only NEW atoms (not anchors).
    let score_ring = |ring_posits: &[Vec3]| -> (f64, f64) {
        if r_aligners.is_empty() {
            return (0.0, 0.0);
        }

        let start_k = anchors.len();
        let mut min_d2 = f64::INFINITY;
        let mut sum_d2 = 0.0;

        for k in start_k..n {
            let p = ring_posits[k];
            let mut best = f64::INFINITY;
            for q in r_aligners {
                let d = p - *q;
                let d2 = dot(d, d);
                if d2 < best {
                    best = d2;
                }
            }
            if best < min_d2 {
                min_d2 = best;
            }
            sum_d2 += best;
        }

        (min_d2, sum_d2)
    };

    // Build ring vertices from (center,u,v,radius). Vertex 0 is at center + u*radius.
    let build_ring = |center: Vec3, u: Vec3, v: Vec3, radius: f64| -> Vec<Vec3> {
        let mut ring_posits = vec![Vec3::new(0.0, 0.0, 0.0); n];
        for k in 0..n {
            let t = (k as f64) * angle;
            ring_posits[k] = center + u * (radius * t.cos()) + v * (radius * t.sin());
        }
        ring_posits[0] = anchors[0];
        if anchors.len() == 2 {
            ring_posits[1] = anchors[1];
        }
        ring_posits
    };

    let ring_posits = if anchors.len() == 2 {
        let a1 = anchors[1];
        let chord = a1 - a0;
        let d = len(chord).max(eps);
        let radius = d / (2.0 * sin_pi_over_n);

        let e = normalize(chord, x_ref);
        let mid = (a0 + a1) * 0.5;

        let h2 = radius * radius - (d * 0.5) * (d * 0.5);
        let h = if h2 > 0.0 { h2.sqrt() } else { 0.0 };

        let perp = normalize(cross(nrm, e), y_ref);

        let c1 = mid + perp * h;
        let c2 = mid - perp * h;

        let mut cand = Vec::with_capacity(2);
        for center in [c1, c2] {
            let u = normalize(a0 - center, x_ref);

            let r1 = normalize(a1 - center, x_ref);
            let cos_t = angle.cos();
            let sin_t = angle.sin();
            let v = if sin_t.abs() > 1e-8 {
                normalize((r1 - u * cos_t) / sin_t, y_ref)
            } else {
                normalize(cross(nrm, u), y_ref)
            };

            let rp = build_ring(center, u, v, radius);
            cand.push((rp.clone(), score_ring(&rp)));
        }

        // Pick higher min_d2; tie-breaker higher sum_d2
        cand.sort_by(|a, b| {
            b.1.0
                .partial_cmp(&a.1.0)
                .unwrap()
                .then_with(|| b.1.1.partial_cmp(&a.1.1).unwrap())
        });

        cand.remove(0).0
    } else {
        let radius = bond_len / (2.0 * sin_pi_over_n);

        // Mean neighbor direction, projected into plane.
        let mut mean_dir = Vec3::new(0.0, 0.0, 0.0);
        for p in r_aligners {
            mean_dir = mean_dir + (*p - a0);
        }
        if len2(mean_dir) <= eps {
            // Stable in-plane fallback
            let pref = if dot(nrm, z_ref).abs() < 0.9 {
                z_ref
            } else {
                y_ref
            };
            mean_dir = cross(nrm, pref);
        }
        mean_dir = mean_dir - nrm * dot(mean_dir, nrm);
        mean_dir = normalize(mean_dir, x_ref);

        // Two candidates: center on either side (u vs -u).
        // u is center->anchor direction.
        let mut cand = Vec::with_capacity(2);
        for u in [mean_dir, -mean_dir] {
            let v = normalize(cross(nrm, u), y_ref);
            let center = a0 - u * radius;
            let rp = build_ring(center, u, v, radius);
            cand.push((rp.clone(), score_ring(&rp)));
        }

        cand.sort_by(|a, b| {
            b.1.0
                .partial_cmp(&a.1.0)
                .unwrap()
                .then_with(|| b.1.1.partial_cmp(&a.1.1).unwrap())
        });

        cand.remove(0).0
    };

    let atoms_to_add = n - anchors.len();

    let vertex_global_i = |k: usize| -> usize {
        if anchors.len() == 2 {
            if k == 0 {
                anchor_is[0]
            } else if k == 1 {
                anchor_is[1]
            } else {
                start_i + (k - 2)
            }
        } else {
            if k == 0 {
                anchor_is[0]
            } else {
                start_i + (k - 1)
            }
        }
    };

    let vertex_sn = |k: usize| -> u32 {
        if anchors.len() == 2 {
            if k == 0 {
                anchor_sns[0]
            } else if k == 1 {
                anchor_sns[1]
            } else {
                start_sn + (k as u32) - 2
            }
        } else {
            if k == 0 {
                anchor_sns[0]
            } else {
                start_sn + (k as u32) - 1
            }
        }
    };

    let (ff_type, bond_type) = match template {
        Template::PentaRing => (String::from("c5"), BondType::Single), // todo: Not sure on this one...
        Template::Cyclohexane => (String::from("c6"), BondType::Single), // todo: QC. Find an example
        Template::AromaticRing => (String::from("ca"), BondType::Aromatic),
        _ => unreachable!(),
    };

    let mut atoms = Vec::with_capacity(atoms_to_add);
    for k in anchors.len()..n {
        atoms.push(Atom {
            serial_number: vertex_sn(k),
            posit: ring_posits[k],
            element: Carbon,
            force_field_type: Some(ff_type.clone()),
            type_in_res: Some(AtomTypeInRes::CA),
            ..Default::default()
        });
    }

    let mut bonds = Vec::with_capacity(if anchors.len() == 2 { n - 1 } else { n });

    // todo: Rough placeholder

    for k in 0..n {
        let k_next = (k + 1) % n;

        if anchors.len() == 2 && k == 0 {
            continue; // skip existing (0-1)
        }

        bonds.push(Bond {
            bond_type,
            atom_0_sn: vertex_sn(k),
            atom_1_sn: vertex_sn(k_next),
            atom_0: vertex_global_i(k),
            atom_1: vertex_global_i(k_next),
            is_backbone: false,
        });
    }

    (atoms, bonds)
}
