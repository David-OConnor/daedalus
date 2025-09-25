//! Can create lipids: Common ones by name, and ones built to specification.
//!
//! [LIPID MAPS Structure Database (LMSD)](https://www.lipidmaps.org/databases/lmsd/overview?utm_source=chatgpt.com)

use std::collections::HashMap;

use bio_files::{
    BondType::{self, *},
    LipidStandard,
};
use lin_alg::f64::Vec3;
use na_seq::Element::{self, *};
use rand::{Rng, distr::Uniform};

use crate::molecule::{Atom, Bond, MoleculeCommon, build_adjacency_list};

// todo: These are the fields after posit. Sometimes seem to be filled in. Should we use them?
// todo: Charge code and atom stero parity seem to be filled out.
// Mass difference (isotope delta; 0 = natural abundance)
// Charge code (0 = 0; 1 = +3; 2 = +2; 3 = +1; 4 = radical; 5 = −1; 6 = −2; 7 = −3)
// Atom stereo parity (0 none, 1/2 odd/even, 3 either)

// Example ratios:
// - E-coli membrane: PE:PG:CL: 70-80:20-25:5-10 (mol%)
// - Staphy aureus: PG:CL: 80:20
// Bacillus subtilis: PE:PG:CL: 40-60%, 5-40%, 8-18%

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LipidShape {
    Free,
    Membrane,
    Lnp,
}

/// todo: Hard-coded for E. coli for now.
pub fn make_bacterial_lipids(n_mols: usize, center: Vec3, shape: LipidShape) -> Vec<MoleculeLipid> {
    let mut rng = rand::rng();
    let mut result = Vec::new();
    // todo: Do we need this?
    let uni = Uniform::<f32>::new(0.0, 1.0).unwrap();

    match shape {
        LipidShape::Free => {
            for _ in 0..n_mols {
                // todo: Instead of sampling each atom, you may wish to ensure
                // todo: teh result is balance.d
                let v = rng.sample(uni);
                let mut standard = LipidStandard::Pe;
                if v > 0.75 && v < 0.93 {
                    standard = LipidStandard::Pgs; // todo: Or pgr?
                } else if v < 0.93 {
                    standard = LipidStandard::Cardiolipin;
                }

                let mut mol = make(standard);
                // todo: Set position and orientation here.
                result.push(mol);
            }
        }
        LipidShape::Membrane => {
            unimplemented!()
        }
        LipidShape::Lnp => {
            unimplemented!()
        }
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

pub fn make(standard: LipidStandard) -> MoleculeLipid {
    match standard {
        LipidStandard::Pc => MoleculeLipid::make_pc(),
        LipidStandard::Pe => MoleculeLipid::make_pe(),
        LipidStandard::Ps => MoleculeLipid::make_ps(),
        LipidStandard::Pi => MoleculeLipid::make_pi(),
        LipidStandard::Pgr => MoleculeLipid::make_pg(),
        // todo: Diff between these?
        LipidStandard::Pgs => MoleculeLipid::make_pg(),
        LipidStandard::Cardiolipin => MoleculeLipid::make_cl(),
        _ => unimplemented!(),
    }
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
}

// todo: Deprecate these A/R. We switched to amber templates.
// `ForceFieldParams` should be loaded from lipid21 or similar.
impl MoleculeLipid {
    /// https://www.lipidmaps.org/databases/lmsd/LMGP01010000
    pub fn make_pc() -> Self {
        let lmsd_id = "LMGP01010000".to_owned();

        let atoms = vec![
            new_atom(1, Vec3::new(8.1477, 7.1972, 0.0000), Carbon),
            new_atom(2, Vec3::new(7.4380, 7.6058, 0.0000), Carbon),
            new_atom(3, Vec3::new(6.7285, 7.1972, 0.0000), Oxygen),
            new_atom(4, Vec3::new(6.0185, 7.6058, 0.0000), Carbon),
            new_atom(5, Vec3::new(6.0185, 8.4262, 0.0000), Oxygen),
            new_atom(6, Vec3::new(8.5578, 6.4876, 0.0000), Hydrogen),
            new_atom(7, Vec3::new(7.7375, 6.4876, 0.0000), Oxygen),
            new_atom(8, Vec3::new(8.8576, 7.6072, 0.0000), Carbon),
            new_atom(9, Vec3::new(9.5673, 7.1972, 0.0000), Oxygen),
            new_atom(10, Vec3::new(11.3522, 7.2141, 0.0000), Oxygen),
            new_atom(11, Vec3::new(12.0618, 6.8043, 0.0000), Carbon),
            new_atom(12, Vec3::new(12.7715, 7.2141, 0.0000), Carbon),
            new_atom(13, Vec3::new(13.4814, 6.8043, 0.0000), Nitrogen),
            new_atom(14, Vec3::new(14.1912, 7.2141, 0.0000), Carbon),
            new_atom(15, Vec3::new(13.4814, 5.9845, 0.0000), Carbon),
            new_atom(16, Vec3::new(14.1912, 6.3943, 0.0000), Carbon),
            new_atom(17, Vec3::new(10.6048, 7.5183, 0.0000), Phosphorus),
            new_atom(18, Vec3::new(10.2434, 6.8914, 0.0000), Oxygen),
            new_atom(19, Vec3::new(10.6048, 8.2654, 0.0000), Oxygen),
            new_atom(20, Vec3::new(6.9940, 6.0682, 0.0000), Carbon),
            new_atom(21, Vec3::new(6.9940, 5.2475, 0.0000), Oxygen),
            new_atom(22, Vec3::new(6.2844, 6.4780, 0.0000), Carbon),
            new_atom(23, Vec3::new(5.3094, 7.1962, 0.0000), Carbon),
        ];

        let atom_posits = atoms.iter().map(|a| a.posit).collect();

        let bonds = vec![
            new_bond(Single, 0, 1),
            new_bond(Single, 1, 2),
            new_bond(Single, 2, 3),
            new_bond(Double, 3, 4),
            new_bond(Single, 0, 6),
            new_bond(Single, 0, 5),
            new_bond(Single, 7, 0),
            new_bond(Single, 8, 7),
            new_bond(Single, 10, 9),
            new_bond(Single, 11, 10),
            new_bond(Single, 12, 11),
            new_bond(Single, 13, 12),
            new_bond(Single, 12, 14),
            new_bond(Single, 12, 15),
            new_bond(Single, 16, 9),
            new_bond(Single, 16, 17),
            new_bond(Double, 16, 18),
            new_bond(Double, 19, 20),
            new_bond(Single, 19, 21),
            new_bond(Single, 19, 6),
            new_bond(Single, 8, 16),
            new_bond(Single, 3, 22),
        ];

        let adjacency_list = build_adjacency_list(&bonds, atoms.len());

        let mut metadata = HashMap::new();
        metadata.insert(
            "SYSTEMATIC_NAME".to_owned(),
            "1,2-diacyl-sn-glycero-3-phosphocholine".to_owned(),
        );
        metadata.insert("M_CHG".to_owned(), "13:+1;18:-1".to_owned()); // from SDF M  CHG
        metadata.insert("R_GROUPS".to_owned(), "22:R2;23:R1".to_owned()); // from SDF M  RGP

        let common = MoleculeCommon {
            ident: lmsd_id.clone(),
            atoms,
            bonds,
            adjacency_list,
            atom_posits,
            metadata,
            visible: true,
            path: None,
            selected_for_md: false,
        };

        Self {
            common,
            lmsd_id,
            hmdb_id: "HMDB00564".to_owned(),
            kegg_id: "C00157".to_owned(),
            common_name: "PC".to_owned(),
        }
    }

    /// https://www.lipidmaps.org/databases/lmsd/LMGP02010000
    pub fn make_pe() -> Self {
        let lmsd_id = "LMGP02010000".to_owned();

        let atoms = vec![
            new_atom(1, Vec3::new(8.1517, 7.1998, 0.0000), Carbon),
            new_atom(2, Vec3::new(7.4412, 7.6089, 0.0000), Carbon),
            new_atom(3, Vec3::new(6.7303, 7.1998, 0.0000), Oxygen),
            new_atom(4, Vec3::new(6.0199, 7.6089, 0.0000), Carbon),
            new_atom(5, Vec3::new(6.0199, 8.4302, 0.0000), Oxygen),
            new_atom(6, Vec3::new(8.5625, 6.4892, 0.0000), Hydrogen),
            new_atom(7, Vec3::new(7.7410, 6.4892, 0.0000), Oxygen),
            new_atom(8, Vec3::new(5.3094, 7.1998, 0.0000), Carbon), // R1
            new_atom(9, Vec3::new(8.8625, 7.6102, 0.0000), Carbon),
            new_atom(10, Vec3::new(9.5731, 7.1998, 0.0000), Oxygen),
            new_atom(11, Vec3::new(11.3326, 7.1827, 0.0000), Oxygen),
            new_atom(12, Vec3::new(12.0432, 6.7723, 0.0000), Carbon),
            new_atom(13, Vec3::new(12.7541, 7.1827, 0.0000), Carbon),
            new_atom(14, Vec3::new(13.4647, 6.7723, 0.0000), Nitrogen),
            new_atom(15, Vec3::new(10.5844, 7.4873, 0.0000), Phosphorus),
            new_atom(16, Vec3::new(10.2223, 6.8598, 0.0000), Oxygen),
            new_atom(17, Vec3::new(10.5844, 8.2356, 0.0000), Oxygen),
            new_atom(18, Vec3::new(6.9965, 6.0691, 0.0000), Carbon),
            new_atom(19, Vec3::new(6.9965, 5.2475, 0.0000), Oxygen),
            new_atom(20, Vec3::new(6.2860, 6.4798, 0.0000), Carbon), // R2
        ];

        let atom_posits = atoms.iter().map(|a| a.posit).collect();

        let bonds = vec![
            new_bond(Single, 0, 1),
            new_bond(Single, 1, 2),
            new_bond(Single, 2, 3),
            new_bond(Double, 3, 4),
            new_bond(Single, 3, 7),
            new_bond(Single, 0, 6),
            new_bond(Single, 0, 5),
            new_bond(Single, 8, 0),
            new_bond(Single, 9, 8),
            new_bond(Single, 11, 10),
            new_bond(Single, 12, 11),
            new_bond(Single, 13, 12),
            new_bond(Single, 14, 10),
            new_bond(Single, 14, 15),
            new_bond(Double, 14, 16),
            new_bond(Double, 17, 18),
            new_bond(Single, 17, 19),
            new_bond(Single, 17, 6),
            new_bond(Single, 14, 9),
        ];

        let adjacency_list = build_adjacency_list(&bonds, atoms.len());

        let mut metadata = HashMap::new();
        metadata.insert(
            "SYSTEMATIC_NAME".to_owned(),
            "1,2-diacyl-sn-glycero-3-phosphoethanolamine".to_owned(),
        );
        metadata.insert("R_LABELS".to_owned(), "8:R1;20:R2".to_owned());

        let common = MoleculeCommon {
            ident: lmsd_id.clone(),
            atoms,
            bonds,
            adjacency_list,
            atom_posits,
            metadata,
            visible: true,
            path: None,
            selected_for_md: false,
        };

        Self {
            common,
            lmsd_id,
            hmdb_id: "HMDB05779".to_owned(),
            kegg_id: "C00350".to_owned(),
            common_name: "PE".to_owned(),
        }
    }

    /// https://www.lipidmaps.org/databases/lmsd/LMGP04010000
    pub fn make_pg() -> Self {
        let lmsd_id = "LMGP04010000".to_owned();

        let atoms = vec![
            new_atom(1, Vec3::new(13.3845, 7.8228, 0.0000), Hydrogen),
            new_atom(2, Vec3::new(8.1739, 7.2151, 0.0000), Carbon),
            new_atom(3, Vec3::new(7.4579, 7.6273, 0.0000), Carbon),
            new_atom(4, Vec3::new(6.7415, 7.2151, 0.0000), Oxygen),
            new_atom(5, Vec3::new(6.0255, 7.6273, 0.0000), Carbon),
            new_atom(6, Vec3::new(6.0255, 8.4550, 0.0000), Oxygen),
            new_atom(7, Vec3::new(8.5878, 6.4989, 0.0000), Hydrogen),
            new_atom(8, Vec3::new(7.7599, 6.4989, 0.0000), Oxygen),
            new_atom(9, Vec3::new(5.3094, 7.2151, 0.0000), Carbon), // R1
            new_atom(10, Vec3::new(8.8901, 7.6286, 0.0000), Carbon),
            new_atom(11, Vec3::new(9.6064, 7.2151, 0.0000), Oxygen),
            new_atom(12, Vec3::new(11.3794, 7.1978, 0.0000), Oxygen),
            new_atom(13, Vec3::new(12.0958, 6.7842, 0.0000), Carbon),
            new_atom(14, Vec3::new(12.8120, 7.1978, 0.0000), Carbon),
            new_atom(15, Vec3::new(13.5283, 6.7842, 0.0000), Carbon),
            new_atom(16, Vec3::new(10.6256, 7.5047, 0.0000), Phosphorus),
            new_atom(17, Vec3::new(10.2607, 6.8724, 0.0000), Oxygen),
            new_atom(18, Vec3::new(10.6256, 8.2589, 0.0000), Oxygen),
            new_atom(19, Vec3::new(7.0098, 6.0757, 0.0000), Carbon),
            new_atom(20, Vec3::new(7.0098, 5.2475, 0.0000), Oxygen),
            new_atom(21, Vec3::new(6.2936, 6.4894, 0.0000), Carbon), // R2
            new_atom(22, Vec3::new(12.4729, 7.7854, 0.0000), Oxygen),
            new_atom(23, Vec3::new(14.2447, 7.1977, 0.0000), Oxygen),
        ];

        let atom_posits = atoms.iter().map(|a| a.posit).collect();

        let bonds = vec![
            new_bond(Single, 1, 2),
            new_bond(Single, 2, 3),
            new_bond(Single, 3, 4),
            new_bond(Double, 4, 5),
            new_bond(Single, 4, 8),
            new_bond(Single, 1, 7),
            new_bond(Single, 1, 6),
            new_bond(Single, 9, 1),
            new_bond(Single, 10, 9),
            new_bond(Single, 12, 11),
            new_bond(Single, 13, 12),
            new_bond(Single, 14, 13),
            new_bond(Single, 15, 11),
            new_bond(Single, 15, 16),
            new_bond(Double, 15, 17),
            new_bond(Double, 18, 19),
            new_bond(Single, 18, 20),
            new_bond(Single, 18, 7),
            new_bond(Single, 15, 10),
            new_bond(Single, 13, 21),
            new_bond(Single, 13, 0),
            new_bond(Single, 14, 22),
        ];

        let adjacency_list = build_adjacency_list(&bonds, atoms.len());

        let mut metadata = HashMap::new();
        metadata.insert(
            "SYSTEMATIC_NAME".to_owned(),
            "1,2-diacyl-sn-glycero-3-phospho-(1'-sn-glycerol)".to_owned(),
        );
        metadata.insert("R_LABELS".to_owned(), "9:R1;21:R2".to_owned());

        let common = MoleculeCommon {
            ident: lmsd_id.clone(),
            atoms,
            bonds,
            adjacency_list,
            atom_posits,
            metadata,
            visible: true,
            path: None,
            selected_for_md: false,
        };

        Self {
            common,
            lmsd_id,
            hmdb_id: "".to_owned(),
            kegg_id: "C00344".to_owned(),
            common_name: "PG".to_owned(),
        }
    }

    /// https://www.lipidmaps.org/databases/lmsd/LMGP06010000
    pub fn make_pi() -> Self {
        let lmsd_id = "LMGP06010000".to_owned();

        let atoms = vec![
            new_atom(1, Vec3::new(7.7658, 6.9347, 0.0000), Carbon),
            new_atom(2, Vec3::new(7.1518, 7.2883, 0.0000), Carbon),
            new_atom(3, Vec3::new(6.5375, 6.9347, 0.0000), Oxygen),
            new_atom(4, Vec3::new(5.9234, 7.2883, 0.0000), Carbon),
            new_atom(5, Vec3::new(5.9234, 7.9981, 0.0000), Oxygen),
            new_atom(6, Vec3::new(8.1207, 6.3206, 0.0000), Hydrogen),
            new_atom(7, Vec3::new(7.4109, 6.3206, 0.0000), Oxygen),
            new_atom(8, Vec3::new(5.3094, 6.9347, 0.0000), Carbon), // R1
            new_atom(9, Vec3::new(8.3801, 7.2894, 0.0000), Carbon),
            new_atom(10, Vec3::new(8.9943, 6.9347, 0.0000), Oxygen),
            new_atom(11, Vec3::new(10.6697, 6.9163, 0.0000), Oxygen),
            new_atom(12, Vec3::new(10.0233, 7.1796, 0.0000), Phosphorus),
            new_atom(13, Vec3::new(9.7104, 6.6373, 0.0000), Oxygen),
            new_atom(14, Vec3::new(10.0233, 7.8262, 0.0000), Oxygen),
            new_atom(15, Vec3::new(6.7676, 5.9576, 0.0000), Carbon),
            new_atom(16, Vec3::new(6.7676, 5.2475, 0.0000), Oxygen),
            new_atom(17, Vec3::new(6.1535, 6.3124, 0.0000), Carbon), // R2
            new_atom(18, Vec3::new(13.0570, 7.1949, 0.0000), Carbon),
            new_atom(19, Vec3::new(11.9299, 7.4959, 0.0000), Carbon),
            new_atom(20, Vec3::new(11.3444, 6.4771, 0.0000), Carbon),
            new_atom(21, Vec3::new(12.4730, 6.7811, 0.0000), Carbon),
            new_atom(22, Vec3::new(13.6031, 6.4771, 0.0000), Carbon),
            new_atom(23, Vec3::new(14.1857, 7.4959, 0.0000), Carbon),
            new_atom(24, Vec3::new(11.2448, 7.3124, 0.0000), Oxygen),
            new_atom(25, Vec3::new(12.4695, 7.6801, 0.0000), Oxygen),
            new_atom(26, Vec3::new(13.4707, 7.7080, 0.0000), Oxygen),
            new_atom(27, Vec3::new(14.3884, 6.7345, 0.0000), Oxygen),
            new_atom(28, Vec3::new(14.7778, 7.3464, 0.0000), Oxygen),
        ];

        let atom_posits = atoms.iter().map(|a| a.posit).collect();

        let bonds = vec![
            new_bond(Single, 0, 1),
            new_bond(Single, 1, 2),
            new_bond(Single, 2, 3),
            new_bond(Double, 3, 4),
            new_bond(Single, 3, 7),
            new_bond(Single, 8, 0),
            new_bond(Single, 9, 8),
            new_bond(Single, 11, 10),
            new_bond(Single, 11, 12),
            new_bond(Double, 11, 13),
            new_bond(Double, 14, 15),
            new_bond(Single, 14, 16),
            new_bond(Single, 14, 6),
            new_bond(Single, 17, 18),
            new_bond(Single, 17, 22),
            new_bond(Single, 18, 19),
            new_bond(Single, 21, 22),
            new_bond(Single, 21, 20),
            new_bond(Single, 19, 20),
            new_bond(Single, 18, 23),
            new_bond(Single, 10, 19),
            new_bond(Single, 20, 24),
            new_bond(Single, 17, 25),
            new_bond(Single, 21, 26),
            new_bond(Single, 22, 27),
            new_bond(Single, 9, 11),
            new_bond(Single, 0, 6),
            new_bond(Single, 0, 5),
        ];

        let adjacency_list = build_adjacency_list(&bonds, atoms.len());

        let mut metadata = HashMap::new();
        metadata.insert(
            "SYSTEMATIC_NAME".to_owned(),
            "1,2-diacyl-sn-glycero-3-phospho-(1'-myo-inositol)".to_owned(),
        );
        metadata.insert("R_LABELS".to_owned(), "8:R1;17:R2".to_owned());

        let common = MoleculeCommon {
            ident: lmsd_id.clone(),
            atoms,
            bonds,
            adjacency_list,
            atom_posits,
            metadata,
            visible: true,
            path: None,
            selected_for_md: false,
        };

        Self {
            common,
            lmsd_id,
            hmdb_id: "".to_owned(),
            kegg_id: "C01194".to_owned(),
            common_name: "PI".to_owned(),
        }
    }

    /// https://www.lipidmaps.org/databases/lmsd/LMGP03010000
    pub fn make_ps() -> Self {
        let lmsd_id = "LMGP03010000".to_owned();

        let atoms = vec![
            new_atom(1, Vec3::new(13.4696, 7.8498, 0.0000), Carbon),
            new_atom(2, Vec3::new(8.2042, 7.2358, 0.0000), Carbon),
            new_atom(3, Vec3::new(7.4805, 7.6523, 0.0000), Carbon),
            new_atom(4, Vec3::new(6.7566, 7.2358, 0.0000), Oxygen),
            new_atom(5, Vec3::new(6.0330, 7.6523, 0.0000), Carbon),
            new_atom(6, Vec3::new(6.0330, 8.4888, 0.0000), Oxygen),
            new_atom(7, Vec3::new(8.6224, 6.5121, 0.0000), Hydrogen),
            new_atom(8, Vec3::new(7.7857, 6.5121, 0.0000), Oxygen),
            new_atom(9, Vec3::new(5.3094, 7.2358, 0.0000), Carbon), // R1
            new_atom(10, Vec3::new(8.9280, 7.6536, 0.0000), Carbon),
            new_atom(11, Vec3::new(9.6518, 7.2358, 0.0000), Oxygen),
            new_atom(12, Vec3::new(11.4435, 7.2184, 0.0000), Oxygen),
            new_atom(13, Vec3::new(12.1674, 6.8004, 0.0000), Carbon),
            new_atom(14, Vec3::new(12.8912, 7.2184, 0.0000), Carbon),
            new_atom(15, Vec3::new(13.6150, 6.8004, 0.0000), Nitrogen),
            new_atom(16, Vec3::new(10.6816, 7.5285, 0.0000), Phosphorus),
            new_atom(17, Vec3::new(10.3128, 6.8894, 0.0000), Oxygen),
            new_atom(18, Vec3::new(10.6816, 8.2905, 0.0000), Oxygen),
            new_atom(19, Vec3::new(7.0277, 6.0844, 0.0000), Carbon),
            new_atom(20, Vec3::new(7.0277, 5.2475, 0.0000), Oxygen),
            new_atom(21, Vec3::new(6.3040, 6.5024, 0.0000), Carbon), // R2
            new_atom(22, Vec3::new(12.5484, 7.8120, 0.0000), Hydrogen),
            new_atom(23, Vec3::new(13.4696, 8.5911, 0.0000), Oxygen),
            new_atom(24, Vec3::new(14.1004, 7.4856, 0.0000), Oxygen),
        ];

        let atom_posits = atoms.iter().map(|a| a.posit).collect();

        let bonds = vec![
            new_bond(Single, 1, 2),
            new_bond(Single, 2, 3),
            new_bond(Single, 3, 4),
            new_bond(Double, 4, 5),
            new_bond(Single, 4, 8),
            new_bond(Single, 1, 7),
            new_bond(Single, 1, 6),
            new_bond(Single, 9, 1),
            new_bond(Single, 10, 9),
            new_bond(Single, 12, 11),
            new_bond(Single, 13, 12),
            new_bond(Single, 14, 13),
            new_bond(Single, 15, 11),
            new_bond(Single, 15, 16),
            new_bond(Double, 15, 17),
            new_bond(Double, 18, 19),
            new_bond(Single, 18, 20),
            new_bond(Single, 18, 7),
            new_bond(Single, 15, 10),
            new_bond(Single, 13, 21),
            new_bond(Single, 13, 0),
            new_bond(Double, 0, 22),
            new_bond(Single, 0, 23),
        ];

        let adjacency_list = build_adjacency_list(&bonds, atoms.len());

        let mut metadata = HashMap::new();
        metadata.insert(
            "SYSTEMATIC_NAME".to_owned(),
            "1,2-diacyl-sn-glycero-3-phosphoserine".to_owned(),
        );
        metadata.insert("R_LABELS".to_owned(), "9:R1;21:R2".to_owned());

        let common = MoleculeCommon {
            ident: lmsd_id.clone(),
            atoms,
            bonds,
            adjacency_list,
            atom_posits,
            metadata,
            visible: true,
            path: None,
            selected_for_md: false,
        };

        Self {
            common,
            lmsd_id,
            hmdb_id: "HMDB00614".to_owned(),
            kegg_id: "C02737".to_owned(),
            common_name: "PS".to_owned(),
        }
    }

    /// https://www.lipidmaps.org/databases/lmsd/LMGP12010000
    pub fn make_cl() -> Self {
        let lmsd_id = "LMGP12010000".to_owned();

        let atoms = vec![
            new_atom(1, Vec3::new(11.1934, 10.7168, 0.0000), Phosphorus),
            new_atom(2, Vec3::new(11.1934, 11.5221, 0.0000), Oxygen),
            new_atom(3, Vec3::new(10.9855, 9.9383, 0.0000), Oxygen),
            new_atom(4, Vec3::new(11.9716, 10.5089, 0.0000), Oxygen),
            new_atom(5, Vec3::new(11.5528, 9.3697, 0.0000), Carbon),
            new_atom(6, Vec3::new(11.3467, 8.5930, 0.0000), Carbon),
            new_atom(7, Vec3::new(11.7494, 7.8963, 0.0000), Carbon),
            new_atom(8, Vec3::new(6.3098, 6.4509, 0.0000), Carbon), // R4
            new_atom(9, Vec3::new(7.0081, 5.2475, 0.0000), Oxygen),
            new_atom(10, Vec3::new(7.0081, 6.0481, 0.0000), Carbon),
            new_atom(11, Vec3::new(7.7159, 6.4509, 0.0000), Oxygen),
            new_atom(12, Vec3::new(5.3094, 7.1476, 0.0000), Carbon), // R3
            new_atom(13, Vec3::new(6.0078, 8.3541, 0.0000), Oxygen),
            new_atom(14, Vec3::new(6.0078, 7.5471, 0.0000), Carbon),
            new_atom(15, Vec3::new(6.7252, 7.1476, 0.0000), Oxygen),
            new_atom(16, Vec3::new(8.1219, 7.1476, 0.0000), Carbon),
            new_atom(17, Vec3::new(7.4235, 7.5471, 0.0000), Carbon),
            new_atom(18, Vec3::new(8.8170, 7.5471, 0.0000), Carbon),
            new_atom(19, Vec3::new(8.5245, 6.4509, 0.0000), Hydrogen),
            new_atom(20, Vec3::new(6.8663, 9.8279, 0.0000), Carbon), // R2
            new_atom(21, Vec3::new(7.5742, 8.6245, 0.0000), Oxygen),
            new_atom(22, Vec3::new(7.5742, 9.4252, 0.0000), Carbon),
            new_atom(23, Vec3::new(8.2772, 9.8279, 0.0000), Oxygen),
            new_atom(24, Vec3::new(5.8658, 10.5247, 0.0000), Carbon), // R1
            new_atom(25, Vec3::new(6.5690, 11.7312, 0.0000), Oxygen),
            new_atom(26, Vec3::new(6.5690, 10.9274, 0.0000), Carbon),
            new_atom(27, Vec3::new(7.2818, 10.5247, 0.0000), Oxygen),
            new_atom(28, Vec3::new(8.6752, 10.5247, 0.0000), Carbon),
            new_atom(29, Vec3::new(7.9801, 10.9274, 0.0000), Carbon),
            new_atom(30, Vec3::new(9.0858, 9.8279, 0.0000), Hydrogen),
            new_atom(31, Vec3::new(12.1249, 8.8008, 0.0000), Hydrogen),
            new_atom(32, Vec3::new(10.5685, 8.8008, 0.0000), Oxygen),
            new_atom(33, Vec3::new(11.2661, 6.9930, 0.0000), Oxygen),
            new_atom(34, Vec3::new(10.5085, 7.3402, 0.0000), Phosphorus),
            new_atom(35, Vec3::new(10.5085, 8.1295, 0.0000), Oxygen),
            new_atom(36, Vec3::new(9.3618, 10.8815, 0.0000), Carbon),
            new_atom(37, Vec3::new(10.2244, 6.6141, 0.0000), Oxygen),
            new_atom(38, Vec3::new(10.0442, 10.4875, 0.0000), Oxygen),
            new_atom(39, Vec3::new(9.4294, 7.1936, 0.0000), Oxygen),
        ];

        let atom_posits = atoms.iter().map(|a| a.posit).collect();

        let bonds = vec![
            new_bond(Double, 0, 1),
            new_bond(Single, 0, 2),
            new_bond(Single, 0, 3),
            new_bond(Single, 2, 4),
            new_bond(Single, 4, 5),
            new_bond(Single, 5, 6),
            new_bond(Double, 8, 9),
            new_bond(Single, 7, 9),
            new_bond(Single, 9, 10),
            new_bond(Double, 12, 13),
            new_bond(Single, 11, 13),
            new_bond(Single, 13, 14),
            new_bond(Single, 14, 16),
            new_bond(Single, 15, 16),
            new_bond(Single, 15, 17),
            new_bond(Double, 20, 21),
            new_bond(Single, 19, 21),
            new_bond(Single, 21, 22),
            new_bond(Double, 24, 25),
            new_bond(Single, 23, 25),
            new_bond(Single, 25, 26),
            new_bond(Single, 26, 28),
            new_bond(Single, 27, 28),
            new_bond(Single, 5, 30),
            new_bond(Single, 5, 31),
            new_bond(Single, 6, 32),
            new_bond(Single, 32, 33),
            new_bond(Double, 33, 34),
            new_bond(Single, 33, 36),
            new_bond(Single, 35, 27),
            new_bond(Single, 35, 37),
            new_bond(Single, 17, 38),
            new_bond(Single, 37, 0),
            new_bond(Single, 38, 33),
            new_bond(Single, 27, 22),
            new_bond(Single, 27, 29),
            new_bond(Single, 15, 10),
            new_bond(Single, 15, 18),
        ];

        let adjacency_list = build_adjacency_list(&bonds, atoms.len());

        let mut metadata = HashMap::new();
        metadata.insert(
            "SYSTEMATIC_NAME".to_owned(),
            "1',3'-Bis-(1,2-diacyl-sn-glycero-3-phospho)-sn-glycerol".to_owned(),
        );
        metadata.insert("R_LABELS".to_owned(), "24:R1;20:R2;12:R3;8:R4".to_owned());

        let common = MoleculeCommon {
            ident: lmsd_id.clone(),
            atoms,
            bonds,
            adjacency_list,
            atom_posits,
            metadata,
            visible: true,
            path: None,
            selected_for_md: false,
        };

        Self {
            common,
            lmsd_id,
            hmdb_id: "".to_owned(),
            kegg_id: "C05980".to_owned(),
            common_name: "Cardiolipin (CL)".to_owned(),
        }
    }

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
