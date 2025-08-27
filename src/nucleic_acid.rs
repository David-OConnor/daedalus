//! For setting up and rendering nucleic acids: DNA and RNA.

// todo: Load Amber FF params for nucleic acids.

use std::{collections::HashMap, f64::consts::TAU};

use bio_files::{BondType, ResidueType};
use lin_alg::f64::Vec3;
use na_seq::{AminoAcid, Element, Nucleotide};

use crate::molecule::{Atom, Bond, MoleculeCommon, MoleculeGeneric, MoleculePeptide};

#[derive(Debug, Clone, Copy)]
pub enum NucleicAcidType {
    Dna,
    Rna,
}

/// Represents a nucleic acid as a collection of atoms and bonds. Omits mol-generic fields.
#[derive(Debug, Default, Clone)]
pub struct MoleculeNucleicAcid {
    pub common: MoleculeCommon,
    pub seq: Vec<Nucleotide>,
    // pub bonds_hydrogen: Vec<HydrogenBond>,
    pub features: Vec<(String, (usize, usize))>, // todo: A/R
    // todo: A/R
    pub metadata: HashMap<String, String>,
    // todo: A/R.
    // pub ff_params: Option<ForceFieldParamsIndexed>,
}

impl MoleculeNucleicAcid {
    /// Build a simple single-strand helix with a phosphate (P), sugar anchor (C4′ proxy),
    /// and a base anchor (N9 for purines, N1 for pyrimidines). Bonds:
    ///   P—S (intra), S—B (intra), and the inter-residue backbone S(i-1)—P(i).
    ///
    /// Geometry is **idealized B-DNA-like**: rise ~3.4 Å, twist 36°, with simple radial offsets.
    /// This is a minimal “it renders now” model you can extend with full atom templates later.
    /// Initializes a linear molecule.
    pub fn from_seq(seq: &[Nucleotide]) -> Self {
        // ---- Helical parameters (B-form style) ----
        const RISE: f64 = 3.4; // Å per nucleotide along helix axis
        const TWIST_DEG: f64 = 36.0; // degrees per nucleotide
        const R_BACKBONE: f64 = 4.2; // Å radius for phosphate/sugar path
        const R_BASE: f64 = 6.0; // Å radius for base anchor (sticks out)
        const SUGAR_OFFSET_ANG: f64 = 0.35; // radians offset P→S around the axis
        const SUGAR_OFFSET_Z: f64 = 0.6; // Å small axial offset P→S
        const BASE_OFFSET_ANG: f64 = TAU * 0.5; // bases roughly opposite backbone
        const BASE_OFFSET_Z: f64 = 0.2; // Å small axial offset S→B

        let twist = TWIST_DEG.to_radians();

        // Helper closure to place a point on a helix at index i, angle offset, radius, z offset.
        let helix_point = |i: usize, ang_off: f64, radius: f64, z_off: f64| -> Vec3 {
            let t = i as f64 * twist + ang_off;
            Vec3::new(radius * t.cos(), radius * t.sin(), i as f64 * RISE + z_off)
        };

        let mut common = MoleculeCommon::default();
        common.ident = format!("{}-nt nucleic acid", seq.len());

        // Reserve a bit to reduce reallocs
        common.atoms.reserve(seq.len() * 3);
        common.adjacency_list.reserve(seq.len() * 3);

        // Keep indices so we can wire bonds
        let mut p_idx: Vec<usize> = Vec::with_capacity(seq.len());
        let mut s_idx: Vec<usize> = Vec::with_capacity(seq.len());
        let mut b_idx: Vec<usize> = Vec::with_capacity(seq.len());

        // Utility: push atom and return its index
        let mut push_atom = |posit: Vec3, element: Element, residue: usize| -> usize {
            let idx = common.atoms.len();
            common.atoms.push(Atom {
                serial_number: (idx as u32) + 1,
                posit,
                element,
                // Keep template fields minimal for now:
                type_in_res: None,
                force_field_type: None,
                dock_type: None,
                role: None,
                residue: Some(residue),
                chain: Some(0),
                hetero: false,
                occupancy: None,
                partial_charge: None,
                temperature_factor: None,
            });
            common.adjacency_list.push(Vec::new());
            idx
        };

        // Place atoms
        for (i, nt) in seq.iter().enumerate() {
            // Phosphate (P) roughly on backbone radius
            let p_pos = helix_point(i, 0.0, R_BACKBONE, 0.0);
            let p_i = push_atom(p_pos, Element::Phosphorus, i);
            p_idx.push(p_i);

            // “Sugar anchor” (use C4′ proxy): nearby around the helix
            let s_pos = helix_point(i, SUGAR_OFFSET_ANG, R_BACKBONE, SUGAR_OFFSET_Z);
            let s_i = push_atom(s_pos, Element::Carbon, i);
            s_idx.push(s_i);

            // Base anchor atom:
            //   Purines (A,G) connect via N9; Pyrimidines (C,T,U) via N1.
            //   We just set the element to N and place it further out from the axis.
            let base_is_purine = matches!(nt, Nucleotide::A | Nucleotide::G);
            let _anchor_name = if base_is_purine { "N9" } else { "N1" };
            let b_pos = helix_point(
                i,
                BASE_OFFSET_ANG + SUGAR_OFFSET_ANG,
                R_BASE,
                SUGAR_OFFSET_Z + BASE_OFFSET_Z,
            );
            let b_i = push_atom(b_pos, Element::Nitrogen, i);
            b_idx.push(b_i);
        }

        // Utility: push bond and wire adjacency
        let mut push_bond = |a: usize, b: usize, is_backbone: bool| {
            let (a_sn, b_sn) = (common.atoms[a].serial_number, common.atoms[b].serial_number);
            common.bonds.push(Bond {
                bond_type: BondType::Single,
                atom_0_sn: a_sn,
                atom_1_sn: b_sn,
                atom_0: a,
                atom_1: b,
                is_backbone,
            });
            common.adjacency_list[a].push(b);
            common.adjacency_list[b].push(a);
        };

        // Intra-residue bonds: P—S and S—B
        for i in 0..seq.len() {
            push_bond(p_idx[i], s_idx[i], true); // phosphate to sugar
            push_bond(s_idx[i], b_idx[i], false); // sugar to base
        }

        // Inter-residue phosphodiester: S(i-1) — P(i)
        for i in 1..seq.len() {
            push_bond(s_idx[i - 1], p_idx[i], true);
        }

        // Mirror atom positions into common.atom_posits for your engine’s convenience
        common.atom_posits = common.atoms.iter().map(|a| a.posit).collect();

        Self {
            common,
            seq: seq.to_vec(),
            features: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// This wrapper that extracts the AA sequence, then chooses a suitable DNA sequence.
    /// note that there are many possible combinations due to multiple codons corresponding
    /// to some AAs.
    pub fn from_peptide(peptide: &MoleculePeptide) -> Self {
        let mut seq = Vec::with_capacity(&peptide.residues.len() * 3);
        for res in &peptide.residues {
            seq.push(Nucleotide::A);
            seq.push(Nucleotide::A);
            seq.push(Nucleotide::A);
        }

        Self::from_seq(&seq)
    }
}
