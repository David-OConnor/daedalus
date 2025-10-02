//! Can create lipids: Common ones by name, and ones built to specification.
//!
//! [LIPID MAPS Structure Database (LMSD)](https://www.lipidmaps.org/databases/lmsd/overview?utm_source=chatgpt.com)

use std::{
    collections::HashMap,
    f64::consts::TAU,
    fmt::{Display, Formatter},
};

use bio_files::{
    BondType::{self, *},
    LipidStandard, ResidueEnd, ResidueType,
};
use dynamics::Dihedral;
use egui::Key::Q;
use lin_alg::f64::{Quaternion, Vec3, Y_VEC, Z_VEC};
use na_seq::Element::{self, *};
use rand::{Rng, distr::Uniform, rngs::ThreadRng};

use crate::{
    mol_lig::MoleculeSmall,
    molecule::{
        Atom, Bond, MolGenericTrait, MolType, MoleculeCommon, MoleculeGenericRef, Residue,
        build_adjacency_list,
    },
};

// todo: These are the fields after posit. Sometimes seem to be filled in. Should we use them?
// todo: Charge code and atom stero parity seem to be filled out.
// Mass difference (isotope delta; 0 = natural abundance)
// Charge code (0 = 0; 1 = +3; 2 = +2; 3 = +1; 4 = radical; 5 = −1; 6 = −2; 7 = −3)
// Atom stereo parity (0 none, 1/2 odd/even, 3 either)

// Example ratios:
// - E-coli membrane: PE:PG:CL: 70-80:20-25:5-10 (mol%)
// - Staphy aureus: PG:CL: 80:20
// Bacillus subtilis: PE:PG:CL: 40-60%, 5-40%, 8-18%

// todo: Fragile. Used to rotate lipids around the phosphate in the head.
const PHOSPHATE_I: usize = 12;

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum LipidShape {
    Free,
    #[default]
    Membrane,
    Lnp,
}

impl Display for LipidShape {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Free => "Free",
            Self::Membrane => "Membrane",
            Self::Lnp => "Lnp",
        };

        write!(f, "{}", s)
    }
}

/// Hard-coded for E. Coli membrane protein for now.
/// Note: Natural bacteria almost always use PGS (?)
pub fn get_mol_from_distro(
    pe: &MoleculeLipid,
    pg: &MoleculeLipid,
    rng: &mut ThreadRng,
    uni: &Uniform<f32>,
) -> MoleculeLipid {
    // todo: Instead of sampling each atom, you may wish to ensure
    // todo: teh result is balance.d

    let v = rng.sample(uni);
    let mut lipid_std = LipidStandard::Pe;

    if v > 0.75 && v < 0.93 {
        lipid_std = LipidStandard::Pgs; // For Pgs or Pgr.
    } else if v >= 0.93 {
        // todo: Placeholder; Amber doesn't have cardiolipin.
        // standard = LipidStandard::Cardiolipin;
        lipid_std = LipidStandard::Pe;
    }

    match lipid_std {
        LipidStandard::Pe => pe.clone(),
        LipidStandard::Pgs => pg.clone(),
        // todo: Cardiolipin too once you have a template.
        _ => unreachable!(),
    }
}

/// Identify an atom in a molecule from its type-in-res.
fn find_atom_by_tir(m: &MoleculeLipid, name: &str) -> usize {
    m.common
        .atoms
        .iter()
        .position(|a| a.type_in_res_lipid.as_deref() == Some(name))
        .expect(name)
}

/// Bonds a phospholipid head's O11 to Acyl C1 (Carbonyl carbon). Bonds Head's O21 to other acyl C1.
/// Moves all atoms so that the head's phosphorous is at the origin.
///
/// Note: This makes new names like "PE(16:0/18:1) — also written as PE(PA/OL) or 1-palmitoyl-2-oleoyl-PE."
/// todo: LMSD ID: LMGP02010009 ?
fn combine_head_tail(
    head: &mut MoleculeLipid,
    mut tail_0: MoleculeLipid,
    mut tail_1: MoleculeLipid,
) {
    // Head joining atoms.
    // todo: Rename everyone once you find teh actual idents.
    let head_p = find_atom_by_tir(&head, "P31");

    let head_anchor_0 = find_atom_by_tir(&head, "O12");
    let head_anchor_1 = find_atom_by_tir(&head, "O22");
    // todo: QC these names.
    let t0_anchor = find_atom_by_tir(&tail_0, "C12");
    let t1_anchor = find_atom_by_tir(&tail_1, "C12");

    let o11_sn = head.common.atoms[head_anchor_0].serial_number;
    let o21_sn = head.common.atoms[head_anchor_1].serial_number;

    let o11_posit = head.common.atoms[head_anchor_0].posit;
    let o21_posit = head.common.atoms[head_anchor_1].posit;
    let t0_c1_posit = tail_0.common.atoms[t0_anchor].posit;
    let t1_c1_posit = tail_1.common.atoms[t1_anchor].posit;

    // Todo: Consider adding these prepared combinations to templates instead
    // todo of creating them each time.

    {
        // Rotate the tails; they are initially reversed relative to the head.
        let rotator = Quaternion::from_axis_angle(Z_VEC, TAU / 2.);

        tail_0.common.rotate(rotator, Some(t0_anchor));
        tail_1.common.rotate(rotator, Some(t1_anchor));

        // We normally only rotate the atom positions; that's what these functions do.
        // Because we're modifying the initial "reset" positions of the atoms, load these back into
        // the Atom structs.
        for (i, atom) in tail_0.common.atoms.iter_mut().enumerate() {
            atom.posit = tail_0.common.atom_posits[i];
        }
        for (i, atom) in tail_1.common.atoms.iter_mut().enumerate() {
            atom.posit = tail_1.common.atom_posits[i];
        }
    }

    // Update bond indices and SNs, offsetting from ones that come before. Order is
    // Head, tail on head's O11, tail on head's O21.
    let head_len = head.common.atoms.len();
    let offset_t0 = head_len;
    let offset_t1 = head_len + tail_0.common.atoms.len();

    // These serial numbers will deconflict, and leave the original values transparent.
    // (Take off the first digit).
    let offset_t0_sn = 1_000;
    let offset_t1_sn = 2_000;

    // We re-anchor tail atoms the head.
    // todo: Find a more accurate way to offset the first chain atom from the head Oxygen
    // todo it's bound to.
    // todo: Ideally use the geometry, but I'm feeling lazy.
    let diff_tail_start_head_end_0 = Vec3::new(1., 1., 1.0);
    let diff_tail_start_head_end_1 = Vec3::new(1., 1., -1.0);

    let tail_0_offset = o11_posit - t0_c1_posit + diff_tail_start_head_end_0;
    let tail_1_offset = o21_posit - t1_c1_posit + diff_tail_start_head_end_1;

    for atom in &mut tail_0.common.atoms {
        atom.serial_number += offset_t0_sn;
        atom.posit += tail_0_offset;
    }

    for atom in &mut tail_1.common.atoms {
        atom.serial_number += offset_t1_sn;
        atom.posit += tail_1_offset;
    }

    // Set these after assigning new SNs to the tail
    let t0_c1_sn = tail_0.common.atoms[t0_anchor].serial_number;
    let t1_c1_sn = tail_1.common.atoms[t1_anchor].serial_number;

    for bond in &mut tail_0.common.bonds {
        bond.atom_0 += offset_t0;
        bond.atom_0_sn += offset_t0_sn;
        bond.atom_1 += offset_t0;
        bond.atom_1_sn += offset_t0_sn;
    }

    for bond in &mut tail_1.common.bonds {
        bond.atom_0 += offset_t1;
        bond.atom_0_sn += offset_t1_sn;
        bond.atom_1 += offset_t1;
        bond.atom_1_sn += offset_t1_sn;
    }

    for atom in tail_0.common.atoms {
        head.common.atoms.push(atom);
    }
    for atom in tail_1.common.atoms {
        head.common.atoms.push(atom);
    }
    for bond in tail_0.common.bonds {
        head.common.bonds.push(bond);
    }
    for bond in tail_1.common.bonds {
        head.common.bonds.push(bond);
    }

    // Create ester bonds: O11–C1 (sn-1), O21–C1 (sn-2)
    // Adjust this to your actual Bond struct if needed.
    head.common.bonds.push(Bond {
        atom_0: head_anchor_0,
        atom_0_sn: o11_sn,
        atom_1: t0_anchor + offset_t0,
        atom_1_sn: t0_c1_sn,
        bond_type: Single,
        is_backbone: false,
    });

    head.common.bonds.push(Bond {
        atom_0: head_anchor_1,
        atom_0_sn: o21_sn,
        atom_1: t1_anchor + offset_t1,
        atom_1_sn: t1_c1_sn,
        bond_type: Single,
        is_backbone: false,
    });

    // Move all atoms to place the P at the origin.
    let p_posit = head.common.atoms[head_p].posit;
    for atom in &mut head.common.atoms {
        atom.posit -= p_posit;
    }

    // todo: Instead of rebulding the adjacency list, you could update it procedurally. For now,
    // todo doing this as it's safer. That would be faster
    head.common.build_adjacency_list();
    head.common.atom_posits = head.common.atoms.iter().map(|a| a.posit).collect();

    let total_len = head.common.atoms.len();

    // Populate residues; one per components.
    {
        head.residues.push(Residue {
            serial_number: 0,
            res_type: ResidueType::Other("Head".to_string()),
            atom_sns: head.common.atoms[..head_len]
                .iter()
                .map(|a| a.serial_number)
                .collect(),
            atoms: (0..head_len).collect(),
            dihedral: None,
            end: ResidueEnd::Internal, // N/A
        });
        head.residues.push(Residue {
            serial_number: 1,
            // todo: Name it properly. E.g. PA, OL etc
            res_type: ResidueType::Other("Chain 1".to_string()),
            atom_sns: head.common.atoms[head_len..offset_t1]
                .iter()
                .map(|a| a.serial_number)
                .collect(),
            atoms: (head_len..offset_t1).collect(),
            dihedral: None,
            end: ResidueEnd::Internal,
        });
        head.residues.push(Residue {
            serial_number: 2,
            res_type: ResidueType::Other("Chain 2".to_string()),
            atom_sns: head.common.atoms[offset_t1..total_len]
                .iter()
                .map(|a| a.serial_number)
                .collect(),
            atoms: (offset_t1..total_len).collect(),
            dihedral: None,
            end: ResidueEnd::Internal,
        });

        for (i, atom) in head.common.atoms.iter_mut().enumerate() {
            if i < head_len {
                atom.residue = Some(0);
            } else if i < offset_t1 {
                atom.residue = Some(1);
            } else {
                atom.residue = Some(2);
            }
        }
    }

    {
        //todo: The results has duplicate tail data. Not sure why.
        head.common.ident = format!(
            "{}({}/{})",
            head.common.ident, tail_0.common.ident, tail_1.common.ident
        );
        head.lmsd_id = format!("{}({}/{})", head.lmsd_id, tail_0.lmsd_id, tail_1.lmsd_id);
        head.common_name = format!(
            "{}({}/{})",
            head.common_name, tail_0.common_name, tail_1.common_name
        );
        head.hmdb_id = format!("{}({}/{})", head.hmdb_id, tail_0.hmdb_id, tail_1.hmdb_id);
        head.kegg_id = format!("{}({}/{})", head.kegg_id, tail_0.kegg_id, tail_1.kegg_id);
    }
}

fn make_phospholipid(head_std: LipidStandard, templates: &[MoleculeLipid]) -> MoleculeLipid {
    // todo: Add to the templates.
    let mut head = templates[head_std as usize].clone();

    // For now, hardcoding PA + OL tails (POPE and POPG)
    let chain_0 = templates[LipidStandard::Pa as usize].clone();
    let chain_1 = templates[LipidStandard::Ol as usize].clone();

    combine_head_tail(&mut head, chain_0, chain_1);

    head
}

/// todo: Hard-coded for E. coli for now.
pub fn make_bacterial_lipids(
    n_mols: usize,
    center: Vec3,
    shape: LipidShape,
    templates: &[MoleculeLipid],
) -> Vec<MoleculeLipid> {
    let mut rng = rand::rng();
    let uni = Uniform::<f32>::new(0.0, 1.0).unwrap();
    let angle = Uniform::<f64>::new(0.0, TAU).unwrap();

    let mut result = Vec::new();

    let pe = make_phospholipid(LipidStandard::Pe, templates);

    // Natural bacterial almost always use PGS (?)
    let pg_r_variant = true;
    let pg = if pg_r_variant {
        make_phospholipid(LipidStandard::Pgr, templates)
    } else {
        make_phospholipid(LipidStandard::Pgs, templates)
    };

    match shape {
        LipidShape::Free => {
            for _ in 0..n_mols {
                let mut mol = get_mol_from_distro(&pe, &pg, &mut rng, &uni);

                let rot = {
                    let w: f64 = rng.random();
                    let x: f64 = rng.random();
                    let y: f64 = rng.random();
                    let z: f64 = rng.random();

                    Quaternion::new(w, x, y, z).to_normalized()
                };

                mol.common.rotate(rot, None);

                // todo temp: You have to prevent collisions.
                let offset_mag = 20.;
                let offset = {
                    let x: f64 = rng.random();
                    let y: f64 = rng.random();
                    let z: f64 = rng.random();

                    Vec3::new(x, y, z).to_normalized() * offset_mag
                };

                for posit in &mut mol.common.atom_posits {
                    *posit = *posit + offset;
                }

                result.push(mol);
            }
        }
        LipidShape::Membrane => {
            // These are head-to-head distances.
            // todo: Consts A/R
            // Note: Area per lipid (APL) is ~60–62 Å² per lipid at ~37 °C.
            const HEADGROUP_SPACING: f64 = 7.9; // 7.8-8.4Å
            const DIST_ACROSS_MEMBRANE: f64 = 38.; // 36-39Å phosphate-to-phosphate

            let n_rows = n_mols.isqrt();
            let n_cols = (n_mols + n_rows - 1) / n_rows; // ceil(n_mols / n_rows)

            // start in the top-left so the grid is centered on `center`
            let mut p = center
                - Vec3::new(
                    (n_cols as f64 - 1.0) * 0.5 * HEADGROUP_SPACING,
                    (n_rows as f64 - 1.0) * 0.5 * HEADGROUP_SPACING,
                    0.0,
                );
            let mut row_start = p;

            for i in 0..n_mols {
                // todo: DRy with above.
                let mut mol = get_mol_from_distro(&pe, &pg, &mut rng, &uni);

                // For now, hardcoding PA + OL tails (POPE and POPG)
                let chain_0 = templates[LipidStandard::Pa as usize].clone();
                let chain_1 = templates[LipidStandard::Ol as usize].clone();

                combine_head_tail(&mut mol, chain_0, chain_1);

                // We rotate based on the original amber orientation, to have tails up and down
                // along the Y axis.

                // todo: PE and PGS seem to need opposite rotations.
                let rotator = Quaternion::from_axis_angle(Z_VEC, -TAU / 4.);

                // Apply an arbitrary rotation along the head/tail axis.
                let rot_z = Quaternion::from_axis_angle(Y_VEC, rng.sample(angle));

                // Hard-coded phorphorous pivot. May not be correct...
                mol.common.rotate(rot_z * rotator, Some(PHOSPHATE_I));

                for posit in &mut mol.common.atom_posits {
                    *posit += p;
                }
                result.push(mol);

                // advance across the row (columns go along +Y per your original code)
                p.x += HEADGROUP_SPACING;

                // wrap to next row after filling `n_cols` entries
                if (i + 1) % n_cols == 0 {
                    row_start.z += HEADGROUP_SPACING;
                    p = Vec3::new(row_start.x, row_start.y, row_start.z);
                }
            }

            // Now make the opposite side of the membrane.
            let mut other_side = Vec::new();
            for mol in &result {
                let mut mirror = mol.clone();
                let rot = Quaternion::from_axis_angle(Z_VEC, TAU / 2.);
                mirror.common.rotate(rot, Some(PHOSPHATE_I));
                for p in &mut mirror.common.atom_posits {
                    p.y -= DIST_ACROSS_MEMBRANE;
                }

                other_side.push(mirror);
            }
            result.append(&mut other_side);
        }
        LipidShape::Lnp => {}
    }

    result
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LipidType {
    Phospholipid,
}

/// Used to create arbitrary lipids
#[derive(Clone, Debug)]
pub struct Lipid {
    pub chain_len: u16,
    pub type_: LipidType,
}

/// Cleans up syntax
fn new_atom(serial_number: u32, posit: Vec3, element: Element) -> Atom {
    Atom {
        serial_number,
        posit,
        element,
        ..Default::default()
    }
}

/// Cleans up syntax
fn new_bond(bond_type: BondType, atom_0: usize, atom_1: usize) -> Bond {
    Bond {
        bond_type,
        atom_0_sn: atom_0 as u32 + 1,
        atom_1_sn: atom_1 as u32 + 1,
        atom_0,
        atom_1,
        is_backbone: false,
    }
}

// pub fn make_membrane(templates: &[MoleculeLipid], ) -> Vec<MoleculeLipid> {
//
// }

// pub fn make(standard: LipidStandard) -> MoleculeLipid {
//     // match standard {
//     //     LipidStandard::Pc => MoleculeLipid::make_pc(),
//     //     LipidStandard::Pe => MoleculeLipid::make_pe(),
//     //     LipidStandard::Ps => MoleculeLipid::make_ps(),
//     //     LipidStandard::Pi => MoleculeLipid::make_pi(),
//     //     LipidStandard::Pgr => MoleculeLipid::make_pg(),
//     //     // todo: Diff between these?
//     //     LipidStandard::Pgs => MoleculeLipid::make_pg(),
//     //     LipidStandard::Cardiolipin => MoleculeLipid::make_cl(),
//     //     _ => unimplemented!(),
//     // }
// }

#[derive(Clone, Debug)]
/// Note: Ident under common name is the LMSD id".
pub struct MoleculeLipid {
    // todo: If we have no other fields, use MoleculeCommon only
    pub common: MoleculeCommon,
    pub lmsd_id: String,
    pub hmdb_id: String,
    pub kegg_id: String,
    // todo: CHEBI ID A/R.
    pub common_name: String,
    /// We use residues to denote headgroups and chains.
    pub residues: Vec<Residue>,
}

impl MolGenericTrait for MoleculeLipid {
    fn common(&self) -> &MoleculeCommon {
        &self.common
    }

    fn common_mut(&mut self) -> &mut MoleculeCommon {
        &mut self.common
    }

    fn to_ref(&self) -> MoleculeGenericRef<'_> {
        MoleculeGenericRef::Lipid(self)
    }

    fn mol_type(&self) -> MolType {
        MolType::Lipid
    }
}

// todo: Deprecate these A/R. We switched to amber templates.
// `ForceFieldParams` should be loaded from lipid21 or similar.
impl MoleculeLipid {
    // /// https://www.lipidmaps.org/databases/lmsd/LMGP01010000
    // pub fn make_pc() -> Self {
    //     let lmsd_id = "LMGP01010000".to_owned();
    //
    //     let atoms = vec![
    //         new_atom(1, Vec3::new(8.1477, 7.1972, 0.0000), Carbon),
    //         new_atom(2, Vec3::new(7.4380, 7.6058, 0.0000), Carbon),
    //         new_atom(3, Vec3::new(6.7285, 7.1972, 0.0000), Oxygen),
    //         new_atom(4, Vec3::new(6.0185, 7.6058, 0.0000), Carbon),
    //         new_atom(5, Vec3::new(6.0185, 8.4262, 0.0000), Oxygen),
    //         new_atom(6, Vec3::new(8.5578, 6.4876, 0.0000), Hydrogen),
    //         new_atom(7, Vec3::new(7.7375, 6.4876, 0.0000), Oxygen),
    //         new_atom(8, Vec3::new(8.8576, 7.6072, 0.0000), Carbon),
    //         new_atom(9, Vec3::new(9.5673, 7.1972, 0.0000), Oxygen),
    //         new_atom(10, Vec3::new(11.3522, 7.2141, 0.0000), Oxygen),
    //         new_atom(11, Vec3::new(12.0618, 6.8043, 0.0000), Carbon),
    //         new_atom(12, Vec3::new(12.7715, 7.2141, 0.0000), Carbon),
    //         new_atom(13, Vec3::new(13.4814, 6.8043, 0.0000), Nitrogen),
    //         new_atom(14, Vec3::new(14.1912, 7.2141, 0.0000), Carbon),
    //         new_atom(15, Vec3::new(13.4814, 5.9845, 0.0000), Carbon),
    //         new_atom(16, Vec3::new(14.1912, 6.3943, 0.0000), Carbon),
    //         new_atom(17, Vec3::new(10.6048, 7.5183, 0.0000), Phosphorus),
    //         new_atom(18, Vec3::new(10.2434, 6.8914, 0.0000), Oxygen),
    //         new_atom(19, Vec3::new(10.6048, 8.2654, 0.0000), Oxygen),
    //         new_atom(20, Vec3::new(6.9940, 6.0682, 0.0000), Carbon),
    //         new_atom(21, Vec3::new(6.9940, 5.2475, 0.0000), Oxygen),
    //         new_atom(22, Vec3::new(6.2844, 6.4780, 0.0000), Carbon),
    //         new_atom(23, Vec3::new(5.3094, 7.1962, 0.0000), Carbon),
    //     ];
    //
    //     let atom_posits = atoms.iter().map(|a| a.posit).collect();
    //
    //     let bonds = vec![
    //         new_bond(Single, 0, 1),
    //         new_bond(Single, 1, 2),
    //         new_bond(Single, 2, 3),
    //         new_bond(Double, 3, 4),
    //         new_bond(Single, 0, 6),
    //         new_bond(Single, 0, 5),
    //         new_bond(Single, 7, 0),
    //         new_bond(Single, 8, 7),
    //         new_bond(Single, 10, 9),
    //         new_bond(Single, 11, 10),
    //         new_bond(Single, 12, 11),
    //         new_bond(Single, 13, 12),
    //         new_bond(Single, 12, 14),
    //         new_bond(Single, 12, 15),
    //         new_bond(Single, 16, 9),
    //         new_bond(Single, 16, 17),
    //         new_bond(Double, 16, 18),
    //         new_bond(Double, 19, 20),
    //         new_bond(Single, 19, 21),
    //         new_bond(Single, 19, 6),
    //         new_bond(Single, 8, 16),
    //         new_bond(Single, 3, 22),
    //     ];
    //
    //     let adjacency_list = build_adjacency_list(&bonds, atoms.len());
    //
    //     let mut metadata = HashMap::new();
    //     metadata.insert(
    //         "SYSTEMATIC_NAME".to_owned(),
    //         "1,2-diacyl-sn-glycero-3-phosphocholine".to_owned(),
    //     );
    //     metadata.insert("M_CHG".to_owned(), "13:+1;18:-1".to_owned()); // from SDF M  CHG
    //     metadata.insert("R_GROUPS".to_owned(), "22:R2;23:R1".to_owned()); // from SDF M  RGP
    //
    //     let common = MoleculeCommon {
    //         ident: lmsd_id.clone(),
    //         atoms,
    //         bonds,
    //         adjacency_list,
    //         atom_posits,
    //         metadata,
    //         visible: true,
    //         path: None,
    //         selected_for_md: false,
    //     };
    //
    //     Self {
    //         common,
    //         lmsd_id,
    //         hmdb_id: "HMDB00564".to_owned(),
    //         kegg_id: "C00157".to_owned(),
    //         common_name: "PC".to_owned(),
    //     }
    // }
    //
    // /// https://www.lipidmaps.org/databases/lmsd/LMGP02010000
    // pub fn make_pe() -> Self {
    //     let lmsd_id = "LMGP02010000".to_owned();
    //
    //     let atoms = vec![
    //         new_atom(1, Vec3::new(8.1517, 7.1998, 0.0000), Carbon),
    //         new_atom(2, Vec3::new(7.4412, 7.6089, 0.0000), Carbon),
    //         new_atom(3, Vec3::new(6.7303, 7.1998, 0.0000), Oxygen),
    //         new_atom(4, Vec3::new(6.0199, 7.6089, 0.0000), Carbon),
    //         new_atom(5, Vec3::new(6.0199, 8.4302, 0.0000), Oxygen),
    //         new_atom(6, Vec3::new(8.5625, 6.4892, 0.0000), Hydrogen),
    //         new_atom(7, Vec3::new(7.7410, 6.4892, 0.0000), Oxygen),
    //         new_atom(8, Vec3::new(5.3094, 7.1998, 0.0000), Carbon), // R1
    //         new_atom(9, Vec3::new(8.8625, 7.6102, 0.0000), Carbon),
    //         new_atom(10, Vec3::new(9.5731, 7.1998, 0.0000), Oxygen),
    //         new_atom(11, Vec3::new(11.3326, 7.1827, 0.0000), Oxygen),
    //         new_atom(12, Vec3::new(12.0432, 6.7723, 0.0000), Carbon),
    //         new_atom(13, Vec3::new(12.7541, 7.1827, 0.0000), Carbon),
    //         new_atom(14, Vec3::new(13.4647, 6.7723, 0.0000), Nitrogen),
    //         new_atom(15, Vec3::new(10.5844, 7.4873, 0.0000), Phosphorus),
    //         new_atom(16, Vec3::new(10.2223, 6.8598, 0.0000), Oxygen),
    //         new_atom(17, Vec3::new(10.5844, 8.2356, 0.0000), Oxygen),
    //         new_atom(18, Vec3::new(6.9965, 6.0691, 0.0000), Carbon),
    //         new_atom(19, Vec3::new(6.9965, 5.2475, 0.0000), Oxygen),
    //         new_atom(20, Vec3::new(6.2860, 6.4798, 0.0000), Carbon), // R2
    //     ];
    //
    //     let atom_posits = atoms.iter().map(|a| a.posit).collect();
    //
    //     let bonds = vec![
    //         new_bond(Single, 0, 1),
    //         new_bond(Single, 1, 2),
    //         new_bond(Single, 2, 3),
    //         new_bond(Double, 3, 4),
    //         new_bond(Single, 3, 7),
    //         new_bond(Single, 0, 6),
    //         new_bond(Single, 0, 5),
    //         new_bond(Single, 8, 0),
    //         new_bond(Single, 9, 8),
    //         new_bond(Single, 11, 10),
    //         new_bond(Single, 12, 11),
    //         new_bond(Single, 13, 12),
    //         new_bond(Single, 14, 10),
    //         new_bond(Single, 14, 15),
    //         new_bond(Double, 14, 16),
    //         new_bond(Double, 17, 18),
    //         new_bond(Single, 17, 19),
    //         new_bond(Single, 17, 6),
    //         new_bond(Single, 14, 9),
    //     ];
    //
    //     let adjacency_list = build_adjacency_list(&bonds, atoms.len());
    //
    //     let mut metadata = HashMap::new();
    //     metadata.insert(
    //         "SYSTEMATIC_NAME".to_owned(),
    //         "1,2-diacyl-sn-glycero-3-phosphoethanolamine".to_owned(),
    //     );
    //     metadata.insert("R_LABELS".to_owned(), "8:R1;20:R2".to_owned());
    //
    //     let common = MoleculeCommon {
    //         ident: lmsd_id.clone(),
    //         atoms,
    //         bonds,
    //         adjacency_list,
    //         atom_posits,
    //         metadata,
    //         visible: true,
    //         path: None,
    //         selected_for_md: false,
    //     };
    //
    //     Self {
    //         common,
    //         lmsd_id,
    //         hmdb_id: "HMDB05779".to_owned(),
    //         kegg_id: "C00350".to_owned(),
    //         common_name: "PE".to_owned(),
    //     }
    // }
    //
    // /// https://www.lipidmaps.org/databases/lmsd/LMGP04010000
    // pub fn make_pg() -> Self {
    //     let lmsd_id = "LMGP04010000".to_owned();
    //
    //     let atoms = vec![
    //         new_atom(1, Vec3::new(13.3845, 7.8228, 0.0000), Hydrogen),
    //         new_atom(2, Vec3::new(8.1739, 7.2151, 0.0000), Carbon),
    //         new_atom(3, Vec3::new(7.4579, 7.6273, 0.0000), Carbon),
    //         new_atom(4, Vec3::new(6.7415, 7.2151, 0.0000), Oxygen),
    //         new_atom(5, Vec3::new(6.0255, 7.6273, 0.0000), Carbon),
    //         new_atom(6, Vec3::new(6.0255, 8.4550, 0.0000), Oxygen),
    //         new_atom(7, Vec3::new(8.5878, 6.4989, 0.0000), Hydrogen),
    //         new_atom(8, Vec3::new(7.7599, 6.4989, 0.0000), Oxygen),
    //         new_atom(9, Vec3::new(5.3094, 7.2151, 0.0000), Carbon), // R1
    //         new_atom(10, Vec3::new(8.8901, 7.6286, 0.0000), Carbon),
    //         new_atom(11, Vec3::new(9.6064, 7.2151, 0.0000), Oxygen),
    //         new_atom(12, Vec3::new(11.3794, 7.1978, 0.0000), Oxygen),
    //         new_atom(13, Vec3::new(12.0958, 6.7842, 0.0000), Carbon),
    //         new_atom(14, Vec3::new(12.8120, 7.1978, 0.0000), Carbon),
    //         new_atom(15, Vec3::new(13.5283, 6.7842, 0.0000), Carbon),
    //         new_atom(16, Vec3::new(10.6256, 7.5047, 0.0000), Phosphorus),
    //         new_atom(17, Vec3::new(10.2607, 6.8724, 0.0000), Oxygen),
    //         new_atom(18, Vec3::new(10.6256, 8.2589, 0.0000), Oxygen),
    //         new_atom(19, Vec3::new(7.0098, 6.0757, 0.0000), Carbon),
    //         new_atom(20, Vec3::new(7.0098, 5.2475, 0.0000), Oxygen),
    //         new_atom(21, Vec3::new(6.2936, 6.4894, 0.0000), Carbon), // R2
    //         new_atom(22, Vec3::new(12.4729, 7.7854, 0.0000), Oxygen),
    //         new_atom(23, Vec3::new(14.2447, 7.1977, 0.0000), Oxygen),
    //     ];
    //
    //     let atom_posits = atoms.iter().map(|a| a.posit).collect();
    //
    //     let bonds = vec![
    //         new_bond(Single, 1, 2),
    //         new_bond(Single, 2, 3),
    //         new_bond(Single, 3, 4),
    //         new_bond(Double, 4, 5),
    //         new_bond(Single, 4, 8),
    //         new_bond(Single, 1, 7),
    //         new_bond(Single, 1, 6),
    //         new_bond(Single, 9, 1),
    //         new_bond(Single, 10, 9),
    //         new_bond(Single, 12, 11),
    //         new_bond(Single, 13, 12),
    //         new_bond(Single, 14, 13),
    //         new_bond(Single, 15, 11),
    //         new_bond(Single, 15, 16),
    //         new_bond(Double, 15, 17),
    //         new_bond(Double, 18, 19),
    //         new_bond(Single, 18, 20),
    //         new_bond(Single, 18, 7),
    //         new_bond(Single, 15, 10),
    //         new_bond(Single, 13, 21),
    //         new_bond(Single, 13, 0),
    //         new_bond(Single, 14, 22),
    //     ];
    //
    //     let adjacency_list = build_adjacency_list(&bonds, atoms.len());
    //
    //     let mut metadata = HashMap::new();
    //     metadata.insert(
    //         "SYSTEMATIC_NAME".to_owned(),
    //         "1,2-diacyl-sn-glycero-3-phospho-(1'-sn-glycerol)".to_owned(),
    //     );
    //     metadata.insert("R_LABELS".to_owned(), "9:R1;21:R2".to_owned());
    //
    //     let common = MoleculeCommon {
    //         ident: lmsd_id.clone(),
    //         atoms,
    //         bonds,
    //         adjacency_list,
    //         atom_posits,
    //         metadata,
    //         visible: true,
    //         path: None,
    //         selected_for_md: false,
    //     };
    //
    //     Self {
    //         common,
    //         lmsd_id,
    //         hmdb_id: "".to_owned(),
    //         kegg_id: "C00344".to_owned(),
    //         common_name: "PG".to_owned(),
    //     }
    // }
    //
    // /// https://www.lipidmaps.org/databases/lmsd/LMGP06010000
    // pub fn make_pi() -> Self {
    //     let lmsd_id = "LMGP06010000".to_owned();
    //
    //     let atoms = vec![
    //         new_atom(1, Vec3::new(7.7658, 6.9347, 0.0000), Carbon),
    //         new_atom(2, Vec3::new(7.1518, 7.2883, 0.0000), Carbon),
    //         new_atom(3, Vec3::new(6.5375, 6.9347, 0.0000), Oxygen),
    //         new_atom(4, Vec3::new(5.9234, 7.2883, 0.0000), Carbon),
    //         new_atom(5, Vec3::new(5.9234, 7.9981, 0.0000), Oxygen),
    //         new_atom(6, Vec3::new(8.1207, 6.3206, 0.0000), Hydrogen),
    //         new_atom(7, Vec3::new(7.4109, 6.3206, 0.0000), Oxygen),
    //         new_atom(8, Vec3::new(5.3094, 6.9347, 0.0000), Carbon), // R1
    //         new_atom(9, Vec3::new(8.3801, 7.2894, 0.0000), Carbon),
    //         new_atom(10, Vec3::new(8.9943, 6.9347, 0.0000), Oxygen),
    //         new_atom(11, Vec3::new(10.6697, 6.9163, 0.0000), Oxygen),
    //         new_atom(12, Vec3::new(10.0233, 7.1796, 0.0000), Phosphorus),
    //         new_atom(13, Vec3::new(9.7104, 6.6373, 0.0000), Oxygen),
    //         new_atom(14, Vec3::new(10.0233, 7.8262, 0.0000), Oxygen),
    //         new_atom(15, Vec3::new(6.7676, 5.9576, 0.0000), Carbon),
    //         new_atom(16, Vec3::new(6.7676, 5.2475, 0.0000), Oxygen),
    //         new_atom(17, Vec3::new(6.1535, 6.3124, 0.0000), Carbon), // R2
    //         new_atom(18, Vec3::new(13.0570, 7.1949, 0.0000), Carbon),
    //         new_atom(19, Vec3::new(11.9299, 7.4959, 0.0000), Carbon),
    //         new_atom(20, Vec3::new(11.3444, 6.4771, 0.0000), Carbon),
    //         new_atom(21, Vec3::new(12.4730, 6.7811, 0.0000), Carbon),
    //         new_atom(22, Vec3::new(13.6031, 6.4771, 0.0000), Carbon),
    //         new_atom(23, Vec3::new(14.1857, 7.4959, 0.0000), Carbon),
    //         new_atom(24, Vec3::new(11.2448, 7.3124, 0.0000), Oxygen),
    //         new_atom(25, Vec3::new(12.4695, 7.6801, 0.0000), Oxygen),
    //         new_atom(26, Vec3::new(13.4707, 7.7080, 0.0000), Oxygen),
    //         new_atom(27, Vec3::new(14.3884, 6.7345, 0.0000), Oxygen),
    //         new_atom(28, Vec3::new(14.7778, 7.3464, 0.0000), Oxygen),
    //     ];
    //
    //     let atom_posits = atoms.iter().map(|a| a.posit).collect();
    //
    //     let bonds = vec![
    //         new_bond(Single, 0, 1),
    //         new_bond(Single, 1, 2),
    //         new_bond(Single, 2, 3),
    //         new_bond(Double, 3, 4),
    //         new_bond(Single, 3, 7),
    //         new_bond(Single, 8, 0),
    //         new_bond(Single, 9, 8),
    //         new_bond(Single, 11, 10),
    //         new_bond(Single, 11, 12),
    //         new_bond(Double, 11, 13),
    //         new_bond(Double, 14, 15),
    //         new_bond(Single, 14, 16),
    //         new_bond(Single, 14, 6),
    //         new_bond(Single, 17, 18),
    //         new_bond(Single, 17, 22),
    //         new_bond(Single, 18, 19),
    //         new_bond(Single, 21, 22),
    //         new_bond(Single, 21, 20),
    //         new_bond(Single, 19, 20),
    //         new_bond(Single, 18, 23),
    //         new_bond(Single, 10, 19),
    //         new_bond(Single, 20, 24),
    //         new_bond(Single, 17, 25),
    //         new_bond(Single, 21, 26),
    //         new_bond(Single, 22, 27),
    //         new_bond(Single, 9, 11),
    //         new_bond(Single, 0, 6),
    //         new_bond(Single, 0, 5),
    //     ];
    //
    //     let adjacency_list = build_adjacency_list(&bonds, atoms.len());
    //
    //     let mut metadata = HashMap::new();
    //     metadata.insert(
    //         "SYSTEMATIC_NAME".to_owned(),
    //         "1,2-diacyl-sn-glycero-3-phospho-(1'-myo-inositol)".to_owned(),
    //     );
    //     metadata.insert("R_LABELS".to_owned(), "8:R1;17:R2".to_owned());
    //
    //     let common = MoleculeCommon {
    //         ident: lmsd_id.clone(),
    //         atoms,
    //         bonds,
    //         adjacency_list,
    //         atom_posits,
    //         metadata,
    //         visible: true,
    //         path: None,
    //         selected_for_md: false,
    //     };
    //
    //     Self {
    //         common,
    //         lmsd_id,
    //         hmdb_id: "".to_owned(),
    //         kegg_id: "C01194".to_owned(),
    //         common_name: "PI".to_owned(),
    //     }
    // }
    //
    // /// https://www.lipidmaps.org/databases/lmsd/LMGP03010000
    // pub fn make_ps() -> Self {
    //     let lmsd_id = "LMGP03010000".to_owned();
    //
    //     let atoms = vec![
    //         new_atom(1, Vec3::new(13.4696, 7.8498, 0.0000), Carbon),
    //         new_atom(2, Vec3::new(8.2042, 7.2358, 0.0000), Carbon),
    //         new_atom(3, Vec3::new(7.4805, 7.6523, 0.0000), Carbon),
    //         new_atom(4, Vec3::new(6.7566, 7.2358, 0.0000), Oxygen),
    //         new_atom(5, Vec3::new(6.0330, 7.6523, 0.0000), Carbon),
    //         new_atom(6, Vec3::new(6.0330, 8.4888, 0.0000), Oxygen),
    //         new_atom(7, Vec3::new(8.6224, 6.5121, 0.0000), Hydrogen),
    //         new_atom(8, Vec3::new(7.7857, 6.5121, 0.0000), Oxygen),
    //         new_atom(9, Vec3::new(5.3094, 7.2358, 0.0000), Carbon), // R1
    //         new_atom(10, Vec3::new(8.9280, 7.6536, 0.0000), Carbon),
    //         new_atom(11, Vec3::new(9.6518, 7.2358, 0.0000), Oxygen),
    //         new_atom(12, Vec3::new(11.4435, 7.2184, 0.0000), Oxygen),
    //         new_atom(13, Vec3::new(12.1674, 6.8004, 0.0000), Carbon),
    //         new_atom(14, Vec3::new(12.8912, 7.2184, 0.0000), Carbon),
    //         new_atom(15, Vec3::new(13.6150, 6.8004, 0.0000), Nitrogen),
    //         new_atom(16, Vec3::new(10.6816, 7.5285, 0.0000), Phosphorus),
    //         new_atom(17, Vec3::new(10.3128, 6.8894, 0.0000), Oxygen),
    //         new_atom(18, Vec3::new(10.6816, 8.2905, 0.0000), Oxygen),
    //         new_atom(19, Vec3::new(7.0277, 6.0844, 0.0000), Carbon),
    //         new_atom(20, Vec3::new(7.0277, 5.2475, 0.0000), Oxygen),
    //         new_atom(21, Vec3::new(6.3040, 6.5024, 0.0000), Carbon), // R2
    //         new_atom(22, Vec3::new(12.5484, 7.8120, 0.0000), Hydrogen),
    //         new_atom(23, Vec3::new(13.4696, 8.5911, 0.0000), Oxygen),
    //         new_atom(24, Vec3::new(14.1004, 7.4856, 0.0000), Oxygen),
    //     ];
    //
    //     let atom_posits = atoms.iter().map(|a| a.posit).collect();
    //
    //     let bonds = vec![
    //         new_bond(Single, 1, 2),
    //         new_bond(Single, 2, 3),
    //         new_bond(Single, 3, 4),
    //         new_bond(Double, 4, 5),
    //         new_bond(Single, 4, 8),
    //         new_bond(Single, 1, 7),
    //         new_bond(Single, 1, 6),
    //         new_bond(Single, 9, 1),
    //         new_bond(Single, 10, 9),
    //         new_bond(Single, 12, 11),
    //         new_bond(Single, 13, 12),
    //         new_bond(Single, 14, 13),
    //         new_bond(Single, 15, 11),
    //         new_bond(Single, 15, 16),
    //         new_bond(Double, 15, 17),
    //         new_bond(Double, 18, 19),
    //         new_bond(Single, 18, 20),
    //         new_bond(Single, 18, 7),
    //         new_bond(Single, 15, 10),
    //         new_bond(Single, 13, 21),
    //         new_bond(Single, 13, 0),
    //         new_bond(Double, 0, 22),
    //         new_bond(Single, 0, 23),
    //     ];
    //
    //     let adjacency_list = build_adjacency_list(&bonds, atoms.len());
    //
    //     let mut metadata = HashMap::new();
    //     metadata.insert(
    //         "SYSTEMATIC_NAME".to_owned(),
    //         "1,2-diacyl-sn-glycero-3-phosphoserine".to_owned(),
    //     );
    //     metadata.insert("R_LABELS".to_owned(), "9:R1;21:R2".to_owned());
    //
    //     let common = MoleculeCommon {
    //         ident: lmsd_id.clone(),
    //         atoms,
    //         bonds,
    //         adjacency_list,
    //         atom_posits,
    //         metadata,
    //         visible: true,
    //         path: None,
    //         selected_for_md: false,
    //     };
    //
    //     Self {
    //         common,
    //         lmsd_id,
    //         hmdb_id: "HMDB00614".to_owned(),
    //         kegg_id: "C02737".to_owned(),
    //         common_name: "PS".to_owned(),
    //     }
    // }
    //
    // /// https://www.lipidmaps.org/databases/lmsd/LMGP12010000
    // pub fn make_cl() -> Self {
    //     let lmsd_id = "LMGP12010000".to_owned();
    //
    //     let atoms = vec![
    //         new_atom(1, Vec3::new(11.1934, 10.7168, 0.0000), Phosphorus),
    //         new_atom(2, Vec3::new(11.1934, 11.5221, 0.0000), Oxygen),
    //         new_atom(3, Vec3::new(10.9855, 9.9383, 0.0000), Oxygen),
    //         new_atom(4, Vec3::new(11.9716, 10.5089, 0.0000), Oxygen),
    //         new_atom(5, Vec3::new(11.5528, 9.3697, 0.0000), Carbon),
    //         new_atom(6, Vec3::new(11.3467, 8.5930, 0.0000), Carbon),
    //         new_atom(7, Vec3::new(11.7494, 7.8963, 0.0000), Carbon),
    //         new_atom(8, Vec3::new(6.3098, 6.4509, 0.0000), Carbon), // R4
    //         new_atom(9, Vec3::new(7.0081, 5.2475, 0.0000), Oxygen),
    //         new_atom(10, Vec3::new(7.0081, 6.0481, 0.0000), Carbon),
    //         new_atom(11, Vec3::new(7.7159, 6.4509, 0.0000), Oxygen),
    //         new_atom(12, Vec3::new(5.3094, 7.1476, 0.0000), Carbon), // R3
    //         new_atom(13, Vec3::new(6.0078, 8.3541, 0.0000), Oxygen),
    //         new_atom(14, Vec3::new(6.0078, 7.5471, 0.0000), Carbon),
    //         new_atom(15, Vec3::new(6.7252, 7.1476, 0.0000), Oxygen),
    //         new_atom(16, Vec3::new(8.1219, 7.1476, 0.0000), Carbon),
    //         new_atom(17, Vec3::new(7.4235, 7.5471, 0.0000), Carbon),
    //         new_atom(18, Vec3::new(8.8170, 7.5471, 0.0000), Carbon),
    //         new_atom(19, Vec3::new(8.5245, 6.4509, 0.0000), Hydrogen),
    //         new_atom(20, Vec3::new(6.8663, 9.8279, 0.0000), Carbon), // R2
    //         new_atom(21, Vec3::new(7.5742, 8.6245, 0.0000), Oxygen),
    //         new_atom(22, Vec3::new(7.5742, 9.4252, 0.0000), Carbon),
    //         new_atom(23, Vec3::new(8.2772, 9.8279, 0.0000), Oxygen),
    //         new_atom(24, Vec3::new(5.8658, 10.5247, 0.0000), Carbon), // R1
    //         new_atom(25, Vec3::new(6.5690, 11.7312, 0.0000), Oxygen),
    //         new_atom(26, Vec3::new(6.5690, 10.9274, 0.0000), Carbon),
    //         new_atom(27, Vec3::new(7.2818, 10.5247, 0.0000), Oxygen),
    //         new_atom(28, Vec3::new(8.6752, 10.5247, 0.0000), Carbon),
    //         new_atom(29, Vec3::new(7.9801, 10.9274, 0.0000), Carbon),
    //         new_atom(30, Vec3::new(9.0858, 9.8279, 0.0000), Hydrogen),
    //         new_atom(31, Vec3::new(12.1249, 8.8008, 0.0000), Hydrogen),
    //         new_atom(32, Vec3::new(10.5685, 8.8008, 0.0000), Oxygen),
    //         new_atom(33, Vec3::new(11.2661, 6.9930, 0.0000), Oxygen),
    //         new_atom(34, Vec3::new(10.5085, 7.3402, 0.0000), Phosphorus),
    //         new_atom(35, Vec3::new(10.5085, 8.1295, 0.0000), Oxygen),
    //         new_atom(36, Vec3::new(9.3618, 10.8815, 0.0000), Carbon),
    //         new_atom(37, Vec3::new(10.2244, 6.6141, 0.0000), Oxygen),
    //         new_atom(38, Vec3::new(10.0442, 10.4875, 0.0000), Oxygen),
    //         new_atom(39, Vec3::new(9.4294, 7.1936, 0.0000), Oxygen),
    //     ];
    //
    //     let atom_posits = atoms.iter().map(|a| a.posit).collect();
    //
    //     let bonds = vec![
    //         new_bond(Double, 0, 1),
    //         new_bond(Single, 0, 2),
    //         new_bond(Single, 0, 3),
    //         new_bond(Single, 2, 4),
    //         new_bond(Single, 4, 5),
    //         new_bond(Single, 5, 6),
    //         new_bond(Double, 8, 9),
    //         new_bond(Single, 7, 9),
    //         new_bond(Single, 9, 10),
    //         new_bond(Double, 12, 13),
    //         new_bond(Single, 11, 13),
    //         new_bond(Single, 13, 14),
    //         new_bond(Single, 14, 16),
    //         new_bond(Single, 15, 16),
    //         new_bond(Single, 15, 17),
    //         new_bond(Double, 20, 21),
    //         new_bond(Single, 19, 21),
    //         new_bond(Single, 21, 22),
    //         new_bond(Double, 24, 25),
    //         new_bond(Single, 23, 25),
    //         new_bond(Single, 25, 26),
    //         new_bond(Single, 26, 28),
    //         new_bond(Single, 27, 28),
    //         new_bond(Single, 5, 30),
    //         new_bond(Single, 5, 31),
    //         new_bond(Single, 6, 32),
    //         new_bond(Single, 32, 33),
    //         new_bond(Double, 33, 34),
    //         new_bond(Single, 33, 36),
    //         new_bond(Single, 35, 27),
    //         new_bond(Single, 35, 37),
    //         new_bond(Single, 17, 38),
    //         new_bond(Single, 37, 0),
    //         new_bond(Single, 38, 33),
    //         new_bond(Single, 27, 22),
    //         new_bond(Single, 27, 29),
    //         new_bond(Single, 15, 10),
    //         new_bond(Single, 15, 18),
    //     ];
    //
    //     let adjacency_list = build_adjacency_list(&bonds, atoms.len());
    //
    //     let mut metadata = HashMap::new();
    //     metadata.insert(
    //         "SYSTEMATIC_NAME".to_owned(),
    //         "1',3'-Bis-(1,2-diacyl-sn-glycero-3-phospho)-sn-glycerol".to_owned(),
    //     );
    //     metadata.insert("R_LABELS".to_owned(), "24:R1;20:R2;12:R3;8:R4".to_owned());
    //
    //     let common = MoleculeCommon {
    //         ident: lmsd_id.clone(),
    //         atoms,
    //         bonds,
    //         adjacency_list,
    //         atom_posits,
    //         metadata,
    //         visible: true,
    //         path: None,
    //         selected_for_md: false,
    //     };
    //
    //     Self {
    //         common,
    //         lmsd_id,
    //         hmdb_id: "".to_owned(),
    //         kegg_id: "C05980".to_owned(),
    //         common_name: "Cardiolipin (CL)".to_owned(),
    //     }
    // }

    /// We assume common.ident is the Amber lipid21 mol ID.
    pub fn populate_db_ids(&mut self) {
        // todo: QC this all.
        match self.common.ident.as_str() {
            // todo: Cardiolipin and PI missing from amber?
            "AR" => {
                self.lmsd_id = "".to_owned();
                self.hmdb_id = "".to_owned();
                self.kegg_id = "".to_owned();
                self.common_name = "".to_owned();
            }
            "CHL" => {
                self.lmsd_id = "".to_owned();
                self.hmdb_id = "".to_owned();
                self.kegg_id = "".to_owned();
                self.common_name = "".to_owned();
            }
            "DHA" => {
                self.lmsd_id = "".to_owned();
                self.hmdb_id = "".to_owned();
                self.kegg_id = "".to_owned();
                self.common_name = "".to_owned();
            }
            "LAL" => {
                self.lmsd_id = "".to_owned();
                self.hmdb_id = "".to_owned();
                self.kegg_id = "".to_owned();
                self.common_name = "".to_owned();
            }
            "MY" => {
                self.lmsd_id = "".to_owned();
                self.hmdb_id = "".to_owned();
                self.kegg_id = "".to_owned();
                self.common_name = "".to_owned();
            }
            "OL" => {
                self.lmsd_id = "".to_owned();
                self.hmdb_id = "".to_owned();
                self.kegg_id = "".to_owned();
                self.common_name = "".to_owned();
            }
            "PA" => {
                self.lmsd_id = "".to_owned();
                self.hmdb_id = "".to_owned();
                self.kegg_id = "".to_owned();
                self.common_name = "".to_owned();
            }
            "PC" => {
                self.lmsd_id = "LMGP01010000".to_owned();
                self.hmdb_id = "HMDB00564".to_owned();
                self.kegg_id = "C00157".to_owned();
                self.common_name = "PC".to_owned();
            }
            "PE" => {
                self.lmsd_id = "LMGP02010000".to_owned();
                self.hmdb_id = "HMDB05779".to_owned();
                self.kegg_id = "C00350".to_owned();
                self.common_name = "PE".to_owned();
            }
            "PGR" => {
                self.lmsd_id = "LMGP04010000".to_owned();
                self.hmdb_id = "".to_owned();
                self.kegg_id = "C00344".to_owned();
                self.common_name = "PG".to_owned();
            }
            "PGS" => {
                self.lmsd_id = "".to_owned();
                self.hmdb_id = "".to_owned();
                self.kegg_id = "".to_owned();
                self.common_name = "".to_owned();
                // todo: QC this: Amber has pgs adn pgr; we just have PG so far.
            }
            "PH-" => {
                self.lmsd_id = "".to_owned();
                self.hmdb_id = "".to_owned();
                self.kegg_id = "".to_owned();
                self.common_name = "".to_owned();
            }
            "PS" => {
                self.lmsd_id = "LMGP03010000".to_owned();
                self.hmdb_id = "HMDB00614".to_owned();
                self.kegg_id = "C02737".to_owned();
                self.common_name = "PS".to_owned();
            }
            "SA" => {
                self.lmsd_id = "".to_owned();
                self.hmdb_id = "".to_owned();
                self.kegg_id = "".to_owned();
                self.common_name = "".to_owned();
            }
            "SPM" => {
                self.lmsd_id = "".to_owned();
                self.hmdb_id = "".to_owned();
                self.kegg_id = "".to_owned();
                self.common_name = "".to_owned();
            }
            "ST" => {
                self.lmsd_id = "".to_owned();
                self.hmdb_id = "".to_owned();
                self.kegg_id = "".to_owned();
                self.common_name = "".to_owned();
            }
            _ => (),
        }
    }
}
