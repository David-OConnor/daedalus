#![allow(unused)]

//! For setting up and rendering nucleic acids: DNA and RNA.

// todo: Load Amber FF params for nucleic acids.

use std::{collections::HashMap, f64::consts::TAU, fmt::Display, io, time::Instant};

use bio_files::{
    BondType, ResidueEnd, ResidueType,
    mol_templates::{TemplateData, load_templates},
};
use dynamics::{
    Dihedral, find_tetra_posit_final,
    params::{OL24_LIB, RNA_LIB},
};
use lin_alg::f64::{Quaternion, Vec3, Z_VEC};
use na_seq::{
    AminoAcid,
    Element::{self, *},
    Nucleotide::{self, *},
    seq_complement,
};

use crate::{
    State, Templates,
    lipid::MoleculeLipid,
    mol_lig::MoleculeSmall,
    molecule::{
        Atom, Bond, MolGenericRef, MolGenericTrait, MolType, MoleculeCommon, MoleculePeptide,
        Residue, build_adjacency_list,
    },
    util::handle_err,
};

// Simple backbone bond length used for translating residues to match up.
// This is the length of the bond which connects residues.
// todo: Once ready, qc this against the FF equilibrium len by selecting the bond in the UI
// const BACKBONE_BOND_LEN: f64 = 1.6;

const HELIX_TWIST: f64 = TAU / 10.0; // 36°
const HELIX_RISE: f64 = 3.4; // Å per base (visual B-DNA-ish)
const HELIX_RADIUS: f64 = 10.0; // Å (tweak to taste)

// Len between residues.
// const BACKBONE_BOND_LEN: f64 = 1.6; //

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum NucleicAcidType {
    #[default]
    Dna,
    Rna,
}

impl Display for NucleicAcidType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let v = match self {
            Self::Dna => "DNA",
            Self::Rna => "RNA",
        };
        write!(f, "{v}")?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Strands {
    Single,
    #[default]
    Double,
}

impl Display for Strands {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let v = match self {
            Self::Single => "Single",
            Self::Double => "Double",
        };
        write!(f, "{v}")?;
        Ok(())
    }
}

/// Returns an arbitrary nucleotide combination that codes for the AA in question.
fn nts_from_aa(aa: AminoAcid) -> [Nucleotide; 3] {
    match aa {
        AminoAcid::Arg => [A, G, G],
        AminoAcid::His => [C, A, C],
        AminoAcid::Lys => [A, A, G],
        AminoAcid::Asp => [G, A, C],
        AminoAcid::Glu => [G, A, G],
        AminoAcid::Ser => [T, C, A],
        AminoAcid::Thr => [A, C, A],
        AminoAcid::Asn => [A, A, C],
        AminoAcid::Gln => [C, A, G],
        AminoAcid::Cys => [T, G, C],
        AminoAcid::Sec => [A, A, A], // todo temp. Find it.,
        AminoAcid::Gly => [G, G, A],
        AminoAcid::Pro => [C, C, A],
        AminoAcid::Ala => [G, C, A],
        AminoAcid::Val => [G, T, A],
        AminoAcid::Ile => [A, T, A],
        AminoAcid::Leu => [C, T, A],
        AminoAcid::Met => [A, T, G],
        AminoAcid::Phe => [T, T, C],
        AminoAcid::Tyr => [T, A, C],
        AminoAcid::Trp => [T, G, G],
    }
}

/// Represents a nucleic acid as a collection of atoms and bonds. Omits mol-generic fields.
#[derive(Debug, Default, Clone)]
pub struct MoleculeNucleicAcid {
    pub common: MoleculeCommon,
    pub residues: Vec<Residue>,
    pub seq: Vec<Nucleotide>,
    // pub bonds_hydrogen: Vec<HydrogenBond>,
    /// This is in the same vein as used by nucleic acid sequencing tools
    /// for example, things like promoters, operators, RNA bind site, etc.
    /// todo: Use the same struct as PlasCAD, or similar?
    pub features: Vec<(String, (usize, usize))>,
    //     pub common_name: String,
    //     /// We use residues to denote headgroups and chains.
    //     pub residues: Vec<Residue>,
}

/// Search our library, and choose the correct template for a given nucleic acid
/// in the chain.
fn find_template(
    nt: Nucleotide,
    na_type: NucleicAcidType,
    is_first: bool,
    is_last: bool,
    templates: &HashMap<String, TemplateData>,
) -> io::Result<&TemplateData> {
    let ident = match na_type {
        NucleicAcidType::Dna => {
            let b = nt.to_str_upper();
            if is_first && !is_last {
                format!("D{b}5")
            } else if is_last && !is_first {
                format!("D{b}3")
            } else {
                format!("D{b}")
            }
        }
        NucleicAcidType::Rna => {
            let mut b = nt.to_str_upper();
            if b == "T" {
                b = "U".to_string();
            }

            if is_first && !is_last {
                format!("{b}5")
            } else if is_last && !is_first {
                format!("{b}3")
            } else {
                b
            }
        }
    };

    match templates.get(&ident) {
        Some(t) => Ok(t),
        None => Err(io::Error::other(format!(
            "Unable to find the template for ident {ident}"
        ))),
    }
}

fn rot_z(v: Vec3, theta: f64) -> Vec3 {
    let rotator = Quaternion::from_axis_angle(Z_VEC, theta);
    rotator.rotate_vec(v)
}

fn build_single_strand(
    seq: &[Nucleotide],
    na_type: NucleicAcidType,
    posit_5p: Vec3,
    templates: &HashMap<String, TemplateData>,
    helix_phase: f64,
    strand_label: &str,
) -> io::Result<(Vec<Atom>, Vec<Bond>, Vec<Residue>)> {
    let mut atoms_out = Vec::new();
    let mut bonds_out = Vec::new();
    let mut res_out = Vec::new();

    let mut atom_sn: u32 = 1;
    let mut res_i: usize = 0;

    // Used for the bond between residues.
    let mut prev_tail_sn: Option<u32> = None;

    // We use this for offsetting bond serial numbers.
    let mut atom_count_current: u32 = 0;

    let mut attach_pt = posit_5p;

    // // posit_5p is where strand_0 residue 0 head lands.
    // // Strand_1 will be opposite (phase = PI) automatically.
    // let global_offset = posit_5p - Vec3::new(HELIX_RADIUS, 0.0, 0.0);

    for (i, &nt) in seq.iter().enumerate() {
        let is_first = i == 0;
        let is_last = i + 1 == seq.len();

        let template = find_template(nt, na_type, is_first, is_last, templates)?;
        let (head_local_i, tail_local_i) = template_attach_points(template)?;

        // P is the first atom in the template for non 5' variants; so for non-5', head_local_i is `Some`,
        // and we use it. For 5' ones, we find the one with name "P", or equivalently, index 0.
        let head_i = match head_local_i {
            Some(i) => i,
            None => find_atom_local_idx_by_name(template, "P").unwrap_or(0),
        };

        let posit_head_local = template
            .atoms
            .get(head_i)
            .ok_or_else(|| io::Error::other("Head attach index out of range"))?
            .posit;

        let theta = (i as f64) * HELIX_TWIST + helix_phase;
        // let head_global = global_offset
        //     + Vec3::new(
        //         HELIX_RADIUS * theta.cos(),
        //         HELIX_RADIUS * theta.sin(),
        //         (i as f64) * HELIX_RISE,
        //     );

        let posit_head_global = if is_first {
            attach_pt
        } else {
            // Since we're attacking the central head atom to the tail oxygen of the previous
            // residue, find what position this H should be, then subtract to position the P. (And other 3 Os)
            // which are part of the res we're adding.

            // todo: Unwrap is not ideal, but working for the templates we're using.
            let o_0 = template.atoms[find_atom_local_idx_by_name(template, "OP1").unwrap()].posit;
            let o_1 = template.atoms[find_atom_local_idx_by_name(template, "OP2").unwrap()].posit;
            let o_2 = template.atoms[find_atom_local_idx_by_name(template, "O5'").unwrap()].posit;

            let o_3p_posit = find_tetra_posit_final(posit_head_local, o_0, o_1, o_2);
            // attach_pt is the global O3' position of the previous res.
            attach_pt + o_3p_posit
        };

        let translation = posit_head_global - posit_head_local;

        let end = if is_first {
            ResidueEnd::NTerminus
        } else if is_last {
            ResidueEnd::CTerminus
        } else {
            ResidueEnd::Internal
        };
        let mut res = Residue {
            serial_number: res_i as u32 + 1,
            res_type: ResidueType::Other(format!("Nucleotide: {nt}")),
            atom_sns: Vec::new(),
            atoms: Vec::new(),
            dihedral: None,
            end,
        };

        // Translate atom positions, and convert from `AtomGeneric` to `Atom`.
        let mut local_to_global_sn = HashMap::new();

        for atom_template in &template.atoms {
            let mut atom: Atom = atom_template.try_into().unwrap();

            let new_sn = (atoms_out.len() as u32) + 1;
            atom.serial_number = new_sn;
            atom.posit += translation;
            atom.residue = Some(res_i);

            res.atom_sns.push(new_sn);
            // todo: QC this atom index
            res.atoms.push(new_sn as usize - 1);

            local_to_global_sn.insert(atom_template.serial_number, new_sn);
            atoms_out.push(atom);
        }

        res_out.push(res);
        res_i += 1;

        // // Map serial numbers to indices. (Often index = sn - 1)
        // let mut atom_index_map = HashMap::new();
        // for (i, atom) in template.atoms.iter().enumerate() {
        //     atom_index_map.insert(atom.serial_number, i);
        // }

        for bond_template in &template.bonds {
            let atom_0_sn = *local_to_global_sn
                .get(&bond_template.atom_0_sn)
                .ok_or_else(|| io::Error::other("Bond atom_0_sn missing from local->global map"))?;

            let atom_1_sn = *local_to_global_sn
                .get(&bond_template.atom_1_sn)
                .ok_or_else(|| io::Error::other("Bond atom_1_sn missing from local->global map"))?;

            let mut bond = bond_template.clone();
            bond.atom_0_sn = atom_0_sn;
            bond.atom_1_sn = atom_1_sn;

            bonds_out.push(Bond::from_generic(&bond, &atoms_out)?);
        }

        // Add inter-residue backbone bond: prev_tail -- current_head
        if !is_first {
            let head_i =
                head_local_i.ok_or_else(|| io::Error::other("Missing head attach point"))?;
            let head_local_sn = template
                .atoms
                .get(head_i)
                .ok_or_else(|| io::Error::other("Head attach index out of range"))?
                .serial_number;
            let cur_head_sn = *local_to_global_sn.get(&head_local_sn).ok_or_else(|| {
                io::Error::other("Head attach serial missing from local->global map")
            })?;

            let prev_tail_sn =
                prev_tail_sn.ok_or_else(|| io::Error::other("Missing prev tail sn"))?;

            let bond = bio_files::BondGeneric {
                atom_0_sn: prev_tail_sn,
                atom_1_sn: cur_head_sn,
                bond_type: BondType::Single,
            };

            bonds_out.push(Bond::from_generic(&bond, &atoms_out)?);
        }

        // Update the attach point.
        if !is_last {
            let tail_i =
                tail_local_i.ok_or_else(|| io::Error::other("Missing tail attach point"))?;
            let tail_atom = template
                .atoms
                .get(tail_i)
                .ok_or_else(|| io::Error::other("Tail attach index out of range"))?;

            let cur_tail_sn = *local_to_global_sn
                .get(&tail_atom.serial_number)
                .ok_or_else(|| {
                    io::Error::other("Tail attach serial missing from local->global map")
                })?;

            attach_pt = tail_atom.posit + translation;
            prev_tail_sn = Some(cur_tail_sn);
        }

        atom_count_current += template.atoms.len() as u32;
    }

    Ok((atoms_out, bonds_out, res_out))
}

fn find_atom_local_idx_by_name(t: &TemplateData, want: &str) -> Option<usize> {
    t.atoms
        .iter()
        .enumerate()
        .find(|(_, a)| a.type_in_res_general.as_deref() == Some(want))
        .map(|(i, _)| i)
}

/// Find the indices of a template that join it to previous ones. None for terminal templates.
fn template_attach_points(template: &TemplateData) -> io::Result<(Option<usize>, Option<usize>)> {
    let Some(unit_connect) = template.unit_connect else {
        return Err(io::Error::other("Missing unit connect on template"));
    };

    let head_i = match unit_connect.head {
        Some(v) => Some(v as usize - 1),
        None => None,
    };
    let tail_i = match unit_connect.tail {
        Some(v) => Some(v as usize - 1),
        None => None,
    };

    Ok((head_i, tail_i))
}

impl MoleculeNucleicAcid {
    /// Build a simple single-strand helix with a phosphate (P), sugar anchor (C4′ proxy),
    /// and a base anchor (N9 for purines, N1 for pyrimidines). Bonds:
    ///   P—S (intra), S—B (intra), and the inter-residue backbone S(i-1)—P(i).
    ///
    /// Geometry is **idealized B-DNA-like**: rise ~3.4 Å, twist 36°, with simple radial offsets.
    /// This is a minimal “it renders now” model you can extend with full atom templates later.
    /// Initializes a linear molecule.
    pub fn from_seq(
        seq: &[Nucleotide],
        na_type: NucleicAcidType,
        strands: Strands,
        posit_5p: Vec3,
        templates_dna: &HashMap<String, TemplateData>,
        templates_rna: &HashMap<String, TemplateData>,
    ) -> io::Result<Self> {
        let templates = match na_type {
            NucleicAcidType::Dna => templates_dna,
            NucleicAcidType::Rna => templates_rna,
        };

        let (mut atoms, mut bonds, mut residues) =
            build_single_strand(seq, na_type, posit_5p, templates, 0., "strand_0")?;

        if strands == Strands::Double && !seq.is_empty() {
            let seq2 = seq_complement(seq);
            let (atoms2, bonds2, mut residues2) =
                build_single_strand(&seq2, na_type, posit_5p, templates, TAU / 2., "strand_1")?;

            let sn_offset = atoms.len() as u32;

            // Shift serial_numbers + bonds for second strand, then append.
            let mut atoms2_shifted = atoms2;
            for a in &mut atoms2_shifted {
                a.serial_number += sn_offset;
            }

            let mut bonds2_shifted = bonds2;
            for b in &mut bonds2_shifted {
                b.atom_0 += sn_offset as usize;
                b.atom_1 += sn_offset as usize;
                b.atom_0_sn += sn_offset;
                b.atom_1_sn += sn_offset;
            }

            let mut residues2_shifted = residues2;
            for res in &mut residues2_shifted {
                for atom in &mut res.atoms {
                    *atom += sn_offset as usize;
                }
            }

            atoms.extend(atoms2_shifted);
            bonds.extend(bonds2_shifted);
            residues.extend(residues2_shifted);
        }

        let adjacency_list = build_adjacency_list(&bonds, atoms.len());

        let mut metadata = HashMap::new();
        metadata.insert(
            "nucleic_acid_type".to_string(),
            match na_type {
                NucleicAcidType::Dna => "dna",
                NucleicAcidType::Rna => "rna",
            }
            .to_string(),
        );
        metadata.insert(
            "strands".to_string(),
            match strands {
                Strands::Single => "single",
                Strands::Double => "double",
            }
            .to_string(),
        );

        let atom_posits = atoms.iter().map(|a| a.posit).collect();

        let common = MoleculeCommon {
            ident: match (na_type, strands) {
                (NucleicAcidType::Dna, Strands::Single) => "DNA(ss)".to_string(),
                (NucleicAcidType::Dna, Strands::Double) => "DNA(ds)".to_string(),
                (NucleicAcidType::Rna, Strands::Single) => "RNA(ss)".to_string(),
                (NucleicAcidType::Rna, Strands::Double) => "RNA(ds)".to_string(),
            },
            atoms,
            bonds,
            adjacency_list,
            atom_posits,
            metadata,
            visible: true,
            path: None,
            selected_for_md: false,
            entity_i_range: None,
        };

        Ok(Self {
            common,
            residues,
            seq: seq.to_vec(),
            features: Vec::new(),
        })
    }

    /// This wrapper that extracts the AA sequence, then chooses a suitable DNA sequence.
    /// note that there are many possible combinations due to multiple codons corresponding
    /// to some AAs.
    pub fn from_peptide(
        peptide: &MoleculePeptide,
        na_type: NucleicAcidType,
        strands: Strands,
        posit_5p: Vec3,
        templates_dna: &HashMap<String, TemplateData>,
        templates_rna: &HashMap<String, TemplateData>,
    ) -> io::Result<Self> {
        let mut seq = Vec::with_capacity(&peptide.residues.len() * 3);
        for res in &peptide.residues {
            seq.push(A);
            seq.push(A);
            seq.push(A);
        }

        Self::from_seq(
            &seq,
            na_type,
            strands,
            posit_5p,
            templates_dna,
            templates_rna,
        )
    }
}

impl MolGenericTrait for MoleculeNucleicAcid {
    fn common(&self) -> &MoleculeCommon {
        &self.common
    }

    fn common_mut(&mut self) -> &mut MoleculeCommon {
        &mut self.common
    }

    fn to_ref(&self) -> MolGenericRef<'_> {
        MolGenericRef::NucleicAcid(self)
    }

    fn mol_type(&self) -> MolType {
        MolType::NucleicAcid
    }
}

/// Returns (DNA, RNA)
pub fn load_na_templates()
-> io::Result<(HashMap<String, TemplateData>, HashMap<String, TemplateData>)> {
    let start = Instant::now();

    println!("Loading nucleic acid templates...");

    let templates_dna = load_templates(OL24_LIB)?;
    let templates_rna = load_templates(RNA_LIB)?;

    let elapsed = start.elapsed().as_millis();
    println!("Loaded nucleic acid templates in {elapsed:.1}ms");

    Ok((templates_dna, templates_rna))
}
