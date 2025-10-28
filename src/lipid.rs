//! Can create lipids: Common ones by name, and ones built to specification.
//!
//! [LIPID MAPS Structure Database (LMSD)](https://www.lipidmaps.org/databases/lmsd/overview?utm_source=chatgpt.com)
//!
//! Lipids-per-area in membrane starting point:
//! PC (POPC/DPPC, fluid): ~60–68 Å², PE (POPE): ~52–58 Å², DOPC: ~68–72 Å²
//!
//! // Example ratios:
//! - E-coli membrane: PE:PG:CL: 70-80:20-25:5-10 (mol%)
//! - Staphy aureus: PG:CL: 80:20
//! Bacillus subtilis: PE:PG:CL: 40-60%, 5-40%, 8-18%

// notes for when constructing liposomes and LNPs:
// "
// Classic liposomes are almost always built from “normal” phospholipids (PC, PE, PS, PG, sometimes PA),
// often with cholesterol, and (for “stealth” liposomes) a small % of a PEG-lipid such as DSPE-PEG2000.
// • mRNA LNPs are a different class: a 4-component mix — an ionizable cationic lipid
// (e.g., MC3/ALC-0315/SM-102), cholesterol, a helper phospholipid (usually DSPC, i.e., a PC),
// and a PEG-lipid — at ~50/40/10/1–2 mol% respectively.
//
// Neutral “liposome” (drug-delivery style): DSPC:CHOL:DSPE-PEG2000 = 55:40:5 (mol%) (or any 56:38:5 Doxil-like recipe).
// Use LIPID21 for DSPC/CHL; GAFF2 for PEG.
//"

use std::{
    f64::consts::TAU,
    fmt::{Display, Formatter},
};

use bio_files::{
    BondType::{self, *},
    LipidStandard, ResidueEnd, ResidueType,
};
use lin_alg::f64::{Quaternion, Vec3, Y_VEC, Z_VEC};
use na_seq::Element::{self, *};
use rand::{Rng, distr::Uniform, rngs::ThreadRng};

use crate::molecule::{
    Atom, Bond, MolGenericRef, MolGenericTrait, MolType, MoleculeCommon, Residue,
};

// From Amber Lipid21.lib. This joins the phospholipid head to the tail.
const BOND_LEN_C11_C12: f64 = 1.508; // From Amber params. cC-cD

// These are head-to-head distances.
// Note: Area per lipid (APL) is ~60–62 Å² per lipid at ~37 °C.
// 8.3-8.5 Angstrom. With a hex grid, works out to the spacing above.
// a = sqrt(2A/sqrt(3))
const HEADGROUP_SPACING: f64 = 8.4;
const HEADGROUP_SPACING_DIV2: f64 = HEADGROUP_SPACING / 2.0;
const HEADGROUP_SPACING_DIV4: f64 = HEADGROUP_SPACING / 4.0;

// Spacing between rows. 1.73... is sqrt(3).
const MEMBRANE_ROW_H: f64 = HEADGROUP_SPACING * 1.7320508075 * 0.5;

const DIST_ACROSS_MEMBRANE: f64 = 38.; // 36-39Å phosphate-to-phosphate

// todo: These are the fields after posit. Sometimes seem to be filled in. Should we use them?
// todo: Charge code and atom stero parity seem to be filled out.
// Mass difference (isotope delta; 0 = natural abundance)
// Charge code (0 = 0; 1 = +3; 2 = +2; 3 = +1; 4 = radical; 5 = −1; 6 = −2; 7 = −3)
// Atom stereo parity (0 none, 1/2 odd/even, 3 either)

// todo: Fragile. Used to rotate lipids around the phosphate in the head.
const PHOSPHATE_I: usize = 12;

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum LipidShape {
    Free,
    #[default]
    Membrane,
    Liposome,
    Lnp,
}

impl Display for LipidShape {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Free => "Free",
            Self::Membrane => "Membrane",
            Self::Liposome => "Liposome",
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
        lipid_std = LipidStandard::Pgs;
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

/// Uses plane geometry to position the first carbon atom in the tail, C12.
///  todo: Base on valence angle and dihedral params instead.
fn find_c12_pos(head: &MoleculeLipid, c11_posit: Vec3, o11_name: &str, o12_name: &str) -> Vec3 {
    let o11 = find_atom_by_tir(head, o11_name);
    let o12 = find_atom_by_tir(head, o12_name);

    let o11_posit = head.common.atoms[o11].posit;
    let o12_posit = head.common.atoms[o12].posit;

    let bond_0 = (c11_posit - o11_posit).to_normalized();
    let bond_1 = (c11_posit - o12_posit).to_normalized();

    let plane = bond_0.cross(bond_1).to_normalized();

    // Close to amber params: 111.960 and 123.110 degrees
    // todo: Refine to match Amber exactly?
    const TAU_DIV3: f64 = TAU / 3.;
    let rotator = Quaternion::from_axis_angle(plane, -TAU_DIV3);

    let bond_new = rotator.rotate_vec(-bond_0) * BOND_LEN_C11_C12;
    c11_posit + bond_new
}

/// Bonds a phospholipid head's C11 or C21 to Acyl C12 (Carbonyl carbon).
/// Moves all atoms so that the head's phosphorous is at the origin.
///
/// Note: This makes new names like "PE(16:0/18:1) — also written as PE(PA/OL) or 1-palmitoyl-2-oleoyl-PE."
fn combine_head_tail(
    head: &mut MoleculeLipid,
    mut tail_0: MoleculeLipid,
    mut tail_1: MoleculeLipid,
    chain_0_name: &str,
    chain_1_name: &str,
) {
    // Head joining atoms.
    // todo: Rename everyone once you find teh actual idents.
    let head_p = find_atom_by_tir(head, "P31");

    let head_anchor_0 = find_atom_by_tir(head, "C11");
    let head_anchor_1 = find_atom_by_tir(head, "C21");

    let t0_anchor = find_atom_by_tir(&tail_0, "C12");
    let t1_anchor = find_atom_by_tir(&tail_1, "C12");

    let head_anchor_0_sn = head.common.atoms[head_anchor_0].serial_number;
    let head_anchor_1_sn = head.common.atoms[head_anchor_1].serial_number;

    // Todo: Consider adding these prepared combinations to templates instead
    // todo of creating them each time.

    // Rotate the tails; they are initially reversed relative to the head.
    {
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
            // Convert C11 to C21 etc; this is consistent with the head naming of this chain.
            let mut tir = atom.type_in_res_lipid.clone().unwrap();
            if tir.starts_with('C') {
                tir.replace_range(1..2, "2"); // Instead of 1.
            }
            atom.type_in_res_lipid = Some(tir);
        }
        // PG has a (relative to PE) reversed head
        if &head.common_name == "PGS" || &head.common_name == "PGR" {
            head.common.rotate(rotator, Some(head_p));

            for (i, atom) in head.common.atoms.iter_mut().enumerate() {
                atom.posit = head.common.atom_posits[i];
            }
        }
    }

    // Positions of the first chain atoms, in the head's coordinates.
    // We set up equal-angle plane geometry.
    let c11_pos = head.common.atoms[head_anchor_0].posit;
    let c21_pos = head.common.atoms[head_anchor_1].posit;

    let posit_c12 = find_c12_pos(head, c11_pos, "O11", "O12");
    let posit_c22 = find_c12_pos(head, c21_pos, "O21", "O22");

    // todo: You likely still have an alignment to perform. Get the initial dihedral right, by
    // todo rotating the chain along the C12/C13 bond.

    let c12_orig = tail_0.common.atoms[t0_anchor].posit;
    let c22_orig = tail_1.common.atoms[t1_anchor].posit;

    let tail_0_offset = posit_c12 - c12_orig;
    let tail_1_offset = posit_c22 - c22_orig;

    // These serial numbers will deconflict, and leave the original values transparent.
    // (Take off the first digit).
    let offset_t0_sn = 1_000;
    let offset_t1_sn = 2_000;

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

    // Update bond indices and SNs, offsetting from ones that come before. Order is
    // Head, tail on head's O11, tail on head's O21.
    let head_len = head.common.atoms.len();
    let offset_t0 = head_len;
    let offset_t1 = head_len + tail_0.common.atoms.len();

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
        atom_0_sn: head_anchor_0_sn,
        atom_1: t0_anchor + offset_t0,
        atom_1_sn: t0_c1_sn,
        bond_type: Single,
        is_backbone: false,
    });

    head.common.bonds.push(Bond {
        atom_0: head_anchor_1,
        atom_0_sn: head_anchor_1_sn,
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

    // todo: Instead of rebuilding the adjacency list, you could update it procedurally. For now,
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
            res_type: ResidueType::Other(chain_0_name.to_owned()),
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
            res_type: ResidueType::Other(chain_1_name.to_owned()),
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

    combine_head_tail(
        &mut head,
        chain_0,
        chain_1,
        &LipidStandard::Pa.to_string(),
        &LipidStandard::Ol.to_string(),
    );

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

    let mut result = Vec::new();

    let pe = make_phospholipid(LipidStandard::Pe, templates);

    // Natural bacterial almost always use PGS.
    let pg_r_variant = false;
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
            result = make_membrane(n_mols, center, &pe, &pg, &mut rng, &uni);
        }
        LipidShape::Liposome => {}
        LipidShape::Lnp => {}
    }

    result
}

#[allow(unused)] // todo: As required later
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LipidType {
    Phospholipid,
}

#[allow(unused)] // todo: As required later
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

/// Creates a dual-layer phospholipid membrane. Initialize in a hexagonal grid with
/// realistic spacing. `n_mols` is per membrane half; actual quantity is 2x this.
/// todo: different spacings for diff membrane types. And support more than PE/PG
///
pub fn make_membrane(
    n_mols: usize,
    center: Vec3,
    pe: &MoleculeLipid,
    pg: &MoleculeLipid,
    rng: &mut ThreadRng,
    uni: &Uniform<f32>,
) -> Vec<MoleculeLipid> {
    let mut result = Vec::with_capacity(n_mols * 2);
    let angle = Uniform::<f64>::new(0.0, TAU).unwrap();

    let n_rows = n_mols.isqrt();
    let n_cols = (n_mols + n_rows - 1) / n_rows; // ceil(n_mols / n_rows)

    // start in the top-left so the grid is centered on `center`
    let mut p = center
        - Vec3::new(
            (n_cols as f64 - 1.0) * 0.5 * HEADGROUP_SPACING,
            0.,
            (n_rows as f64 - 1.0) * 0.5 * MEMBRANE_ROW_H,
        );

    let mut row_start = p;
    let initial_x = row_start.x - HEADGROUP_SPACING_DIV4;
    let mut row_i = 0;

    for i in 0..n_mols {
        // todo: DRy with above.
        let mut mol = get_mol_from_distro(pe, pg, rng, &uni);

        // We rotate based on the original amber orientation, to have tails up and down
        // along the Y axis.

        // todo: PE and PGS seem to need opposite rotations.
        let rotator = Quaternion::from_axis_angle(Z_VEC, -TAU / 4.);

        // Apply an arbitrary rotation along the head/tail axis.
        let rot_z = Quaternion::from_axis_angle(Y_VEC, rng.sample(angle));

        // Hard-coded phosphorous pivot. May not be correct...
        mol.common.rotate(rot_z * rotator, Some(PHOSPHATE_I));

        // Add a small blue-noise jitter to break the order; it's not a
        // perfect crystal, and has no long-distance structure.

        const JITTER_MUL: f64 = 0.4;
        const JITTER_SUB: f64 = 0.2;

        for posit in &mut mol.common.atom_posits {
            let jx = rng.sample(uni) as f64 * JITTER_MUL - JITTER_SUB;
            let jz = rng.sample(uni) as f64 * JITTER_MUL - JITTER_SUB;

            *posit += p + Vec3::new(jx, 0., jz);
        }
        result.push(mol);

        p.x += HEADGROUP_SPACING;

        // wrap to next row after filling `n_cols` entries
        if (i + 1) % n_cols == 0 {
            row_i += 1;
            row_start.z += MEMBRANE_ROW_H;
            // odd rows shifted by +half for hex packing
            row_start.x = initial_x
                + if row_i % 2 == 1 {
                    HEADGROUP_SPACING_DIV2
                } else {
                    0.0
                };
            p = Vec3::new(row_start.x, row_start.y, row_start.z);
        }
    }

    // Now make the opposite side of the membrane.
    let mut other_side = Vec::new();
    for mol in &result {
        let mut mirror = mol.clone();

        let rot_invert = Quaternion::from_axis_angle(Z_VEC, TAU / 2.);
        // AN attempt to deconflict chain atoms between top and bottom
        // todo: Not good enough. For now, manually separating at init.
        let rot_decon = Quaternion::from_axis_angle(Y_VEC, TAU / 4.);

        // I believe either order is fine.
        let rot = rot_decon * rot_invert;

        mirror.common.rotate(rot, Some(PHOSPHATE_I));

        for p in &mut mirror.common.atom_posits {
            p.y -= DIST_ACROSS_MEMBRANE;
            p.y -= 8.; // todo temp to prevent conflicct at init. not ideal!

            // Shift the mirror ~1/2 lateral head-head dist to prevent initial overlap of
            // atoms between halfs.
            // todo: Better way? Rotate chains? More robust way that ensures deconfliction?
            p.x += HEADGROUP_SPACING_DIV2;
            p.z += HEADGROUP_SPACING_DIV2;
        }

        other_side.push(mirror);
    }
    result.append(&mut other_side);

    result
}

pub fn make_liposome(
    center: Vec3,
    radius_outer: f32,
    pe: &MoleculeLipid,
    pg: &MoleculeLipid,
    rng: &mut ThreadRng,
    uni: &Uniform<f32>,
) -> Vec<MoleculeLipid> {
    let mut result = Vec::new();

    result
}

pub fn make_lnp(
    center: Vec3,
    radius_outer: f32,
    pe: &MoleculeLipid,
    pg: &MoleculeLipid,
    rng: &mut ThreadRng,
    uni: &Uniform<f32>,
) -> Vec<MoleculeLipid> {
    let mut result = Vec::new();

    result
}

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

    fn to_ref(&self) -> MolGenericRef<'_> {
        MolGenericRef::Lipid(self)
    }

    fn mol_type(&self) -> MolType {
        MolType::Lipid
    }
}

// `ForceFieldParams` should be loaded from lipid21 or similar.
impl MoleculeLipid {
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
                self.common_name = "PGR".to_owned();
            }
            "PGS" => {
                self.lmsd_id = "".to_owned();
                self.hmdb_id = "".to_owned();
                self.kegg_id = "".to_owned();
                self.common_name = "PGS".to_owned();
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
