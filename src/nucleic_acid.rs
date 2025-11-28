#![allow(unused)]

//! For setting up and rendering nucleic acids: DNA and RNA.

// todo: Load Amber FF params for nucleic acids.

use std::{collections::HashMap, f64::consts::TAU, io, time::Instant};

use bio_files::{
    BondType,
    mol_templates::{TemplateData, load_templates},
};
use dynamics::params::OL24_LIB;
use lin_alg::f64::Vec3;
use na_seq::{
    AminoAcid,
    Element::{self, *},
    Nucleotide::{self, *},
    seq_complement,
};

use crate::{
    State,
    lipid::MoleculeLipid,
    mol_lig::MoleculeSmall,
    molecule::{
        Atom, Bond, MolGenericRef, MolGenericTrait, MolType, MoleculeCommon, MoleculePeptide,
        Residue, build_adjacency_list,
    },
    util::handle_err,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NucleicAcidType {
    Dna,
    Rna,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Strands {
    Single,
    Double,
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

fn build_single_strand(
    seq: &[Nucleotide],
    na_type: NucleicAcidType,
    templates: &[MoleculeNucleicAcid],
    strand_label: &str,
) -> io::Result<(Vec<Atom>, Vec<Vec3>, Vec<Bond>)> {
    let mut atoms_out: Vec<Atom> = Vec::new();
    let mut posits_out: Vec<Vec3> = Vec::new();
    let mut bonds_out: Vec<Bond> = Vec::new();

    let mut next_sn: u32 = 1;

    let mut prev_tail_global_sn: Option<u32> = None;
    let mut prev_tail_posit: Option<Vec3> = None;

    // todo: Examine this.

    // Simple backbone bond length used for translating residues to match up.
    let backbone_bond_len = 1.6;

    for (i, &nt) in seq.iter().enumerate() {
        let is_first = i == 0;
        let is_last = i + 1 == seq.len();

        let template = match na_type {
            NucleicAcidType::Dna => {
                let b = nt.to_str_upper();
                let mid = format!("D{b}");
                let five = format!("D{b}5");
                let three = format!("D{b}3");
                let want: [&str; 2] = if is_first && !is_last {
                    [&five, &mid]
                } else if is_last && !is_first {
                    [&three, &mid]
                } else {
                    [&mid, &mid]
                };
                find_template(templates, &want)
            }
            NucleicAcidType::Rna => {
                let mut b = nt.to_str_upper();
                if b == "T" {
                    b == "U";
                }

                let mid = format!("R{b}");
                let five = format!("R{b}5");
                let three = format!("R{b}3");

                // Some Amber libs use RU, some older/custom setups might use RT; try both for U.
                if b == "U" {
                    let mid_alt = "RT".to_string();
                    let five_alt = "RT5".to_string();
                    let three_alt = "RT3".to_string();
                    let want: [&str; 4] = if is_first && !is_last {
                        [&five, &mid, &five_alt, &mid_alt]
                    } else if is_last && !is_first {
                        [&three, &mid, &three_alt, &mid_alt]
                    } else {
                        [&mid, &mid_alt, &mid, &mid_alt]
                    };
                    find_template(templates, &want)
                } else {
                    let want: [&str; 2] = if is_first && !is_last {
                        [&five, &mid]
                    } else if is_last && !is_first {
                        [&three, &mid]
                    } else {
                        [&mid, &mid]
                    };
                    find_template(templates, &want)
                }
            }
        };

        // todo: This map of indexes to bonds is repeated a few times.
        let mut atom_index_map = HashMap::new();
        for (i, atom) in template.common.atoms.iter().enumerate() {
            atom_index_map.insert(atom.serial_number, i);
        }

        let (head_local_i, tail_local_i) = template_attach_points(template);

        let Some(head_local_i) = head_local_i else {
            let msg = format!(
                "Template {} missing head attach point (.unit.connect or P atom)",
                template.common.ident
            );

            return Err(io::Error::other(msg));
        };
        let Some(tail_local_i) = tail_local_i else {
            let msg = format!(
                "Template {} missing tail attach point (.unit.connect or P atom)",
                template.common.ident
            );

            return Err(io::Error::other(msg));
        };

        let tpl_head_posit = template
            .common
            .atom_posits
            .get(head_local_i)
            .copied()
            .unwrap_or_else(|| Vec3::new(0.0, 0.0, 0.0));

        let translation = if let Some(prev_tail) = prev_tail_posit {
            (prev_tail + Vec3::new(0.0, 0.0, backbone_bond_len)) - tpl_head_posit
        } else {
            Vec3::new(0.0, 0.0, 0.0)
        };

        let residue_start_sn = next_sn;

        // Copy atoms + positioned coordinates
        for (local_i, tpl_atom) in template.common.atoms.iter().enumerate() {
            let mut a = tpl_atom.clone();
            a.serial_number = next_sn;

            let p = template
                .common
                .atom_posits
                .get(local_i)
                .copied()
                .unwrap_or_else(|| Vec3::new(0.0, 0.0, 0.0))
                + translation;

            a.posit = p;

            atoms_out.push(a);
            posits_out.push(p);
            next_sn += 1;
        }

        // Copy intra-residue bonds with serial offset
        let sn_offset = residue_start_sn - 1;
        for b in &template.common.bonds {
            let mut nb = b.clone();
            nb.atom_0_sn = b.atom_0_sn + sn_offset;
            nb.atom_1_sn = b.atom_1_sn + sn_offset;
            bonds_out.push(nb);
        }

        // Add inter-residue backbone bond: prev_tail -- current_head
        let cur_head_global_sn = sn_offset + (head_local_i as u32) + 1;
        let cur_tail_global_sn = sn_offset + (tail_local_i as u32) + 1;

        if let Some(prev_tail_sn) = prev_tail_global_sn {
            bonds_out.push(Bond {
                atom_0_sn: prev_tail_sn,
                atom_1_sn: cur_head_global_sn,
                atom_0: atom_index_map[&prev_tail_sn],
                atom_1: atom_index_map[&cur_head_global_sn],
                bond_type: BondType::Single,
                is_backbone: false,
            });
        }

        let cur_tail_posit = posits_out
            .get((cur_tail_global_sn as usize) - 1)
            .copied()
            .unwrap_or_else(|| Vec3::new(0.0, 0.0, 0.0));

        prev_tail_global_sn = Some(cur_tail_global_sn);
        prev_tail_posit = Some(cur_tail_posit);

        // Optionally tag strand in metadata per-atom later; for now just a hook:
        let _ = strand_label;
    }

    Ok((atoms_out, posits_out, bonds_out))
}

// todo: Review and clean this up A/R. Especially the joins. between template segments
fn find_template<'a>(
    templates: &'a [MoleculeNucleicAcid],
    names: &[&str],
) -> &'a MoleculeNucleicAcid {
    for &want in names {
        if let Some(t) = templates.iter().find(|t| t.common.ident == want) {
            return t;
        }
    }
    panic!("No matching nucleic acid template found. Wanted one of: {names:?}");
}

fn find_atom_local_idx_by_name(t: &MoleculeNucleicAcid, want: &str) -> Option<usize> {
    t.common
        .atoms
        .iter()
        .enumerate()
        .find(|(_, a)| a.type_in_res_general.as_deref() == Some(want))
        .map(|(i, _)| i)
}

fn parse_u32_meta(t: &MoleculeNucleicAcid, key: &str) -> Option<u32> {
    t.common
        .metadata
        .get(key)
        .and_then(|v| v.parse::<u32>().ok())
}

fn template_attach_points(t: &MoleculeNucleicAcid) -> (Option<usize>, Option<usize>) {
    // Prefer unit.connect head/tail if present.
    let head = parse_u32_meta(t, "unit_connect_head").unwrap_or(0);
    let tail = parse_u32_meta(t, "unit_connect_tail").unwrap_or(0);

    if head != 0 || tail != 0 {
        let head_i = if head == 0 {
            None
        } else {
            Some((head - 1) as usize)
        };
        let tail_i = if tail == 0 {
            None
        } else {
            Some((tail - 1) as usize)
        };
        return (head_i, tail_i);
    }

    // Fallback to residueconnect c1/c2 if present.
    let c1 = parse_u32_meta(t, "residueconnect_c1x").unwrap_or(0);
    let c2 = parse_u32_meta(t, "residueconnect_c2x").unwrap_or(0);
    if c1 != 0 || c2 != 0 {
        let head_i = if c1 == 0 {
            None
        } else {
            Some((c1 - 1) as usize)
        };
        let tail_i = if c2 == 0 {
            None
        } else {
            Some((c2 - 1) as usize)
        };
        return (head_i, tail_i);
    }

    // Final fallback: atom names (Amber DNA/RNA typically: P is head; O3' is tail).
    let head_i = find_atom_local_idx_by_name(t, "P");
    let tail_i =
        find_atom_local_idx_by_name(t, "O3'").or_else(|| find_atom_local_idx_by_name(t, "O3*"));
    (head_i, tail_i)
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
        templates_dna: &[MoleculeNucleicAcid],
        templates_rna: &[MoleculeNucleicAcid],
    ) -> io::Result<Self> {
        let templates = match na_type {
            NucleicAcidType::Dna => templates_dna,
            NucleicAcidType::Rna => templates_rna,
        };

        let (mut atoms, mut atom_posits, mut bonds) =
            build_single_strand(seq, na_type, templates, "strand_0")?;

        if strands == Strands::Double && !seq.is_empty() {
            let seq2 = seq_complement(seq);
            let (atoms2, pos2, bonds2) =
                build_single_strand(&seq2, na_type, templates, "strand_1")?;

            let sn_offset = atoms.len() as u32;

            // Shift serial_numbers + bonds for second strand, then append.
            let mut atoms2_shifted = atoms2;
            for a in &mut atoms2_shifted {
                a.serial_number += sn_offset;
            }

            let mut bonds2_shifted = bonds2;
            for b in &mut bonds2_shifted {
                b.atom_0_sn += sn_offset;
                b.atom_1_sn += sn_offset;
            }

            atoms.extend(atoms2_shifted);
            atom_posits.extend(pos2);
            bonds.extend(bonds2_shifted);
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
        templates_dna: &[MoleculeNucleicAcid],
        templates_rna: &[MoleculeNucleicAcid],
    ) -> io::Result<Self> {
        let mut seq = Vec::with_capacity(&peptide.residues.len() * 3);
        for res in &peptide.residues {
            seq.push(A);
            seq.push(A);
            seq.push(A);
        }

        Self::from_seq(&seq, na_type, strands, templates_dna, templates_rna)
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

#[derive(Clone, Copy)]
struct HelixGeom {
    rise: f64,
    twist: f64,
    r_backbone: f64,
    base_rad: f64,
    sugar_ang: f64,
    sugar_dz: f64,
    base_ang: f64,
    base_dz: f64,
}

fn infer_na(seq: &[Nucleotide]) -> NucleicAcidType {
    // todo
    // if seq.iter().any(|n| matches!(n, Nucleotide::U)) {
    //     NucleicAcidType::Rna
    // } else {
    NucleicAcidType::Dna
    // }
}

fn helix_point(i: usize, ang_off: f64, radius: f64, dz: f64, h: &HelixGeom) -> Vec3 {
    let t = i as f64 * h.twist + ang_off;
    Vec3::new(radius * t.cos(), radius * t.sin(), i as f64 * h.rise + dz)
}

fn rotate_xy(v: Vec3, angle: f64) -> Vec3 {
    let (c, s) = (angle.cos(), angle.sin());
    Vec3::new(c * v.x - s * v.y, s * v.x + c * v.y, v.z)
}

struct Frame {
    o: Vec3,
    ex: Vec3,
    ey: Vec3,
    ez: Vec3,
}
impl Frame {
    fn apply(&self, local: Vec3) -> Vec3 {
        self.o + self.ex * local.x + self.ey * local.y + self.ez * local.z
    }
}

// todo: What is this?? Don't like.
#[derive(Clone, Copy)]
struct TAtom {
    name: &'static str,
    element: Element,
    coord: Vec3,
}

fn push_atom(common: &mut MoleculeCommon, pos: Vec3, element: Element, residue: usize) -> usize {
    let idx = common.atoms.len();
    common.atoms.push(Atom {
        serial_number: (idx as u32) + 1,
        posit: pos,
        element,
        residue: Some(residue),
        chain: Some(0),
        hetero: false,
        ..Default::default()
    });
    idx
}

fn push_bond_indices(common: &mut MoleculeCommon, a: usize, b: usize, is_backbone: bool) {
    let (a_sn, b_sn) = (common.atoms[a].serial_number, common.atoms[b].serial_number);
    common.bonds.push(Bond {
        bond_type: BondType::Single,
        atom_0_sn: a_sn,
        atom_1_sn: b_sn,
        atom_0: a,
        atom_1: b,
        is_backbone,
    });
}

fn push_bond_by_name(
    common: &mut MoleculeCommon,
    name_to_idx: &HashMap<&'static str, usize>,
    a: &'static str,
    b: &'static str,
    is_backbone: bool,
) {
    let ia = *name_to_idx.get(a).expect("atom name missing");
    let ib = *name_to_idx.get(b).expect("atom name missing");
    push_bond_indices(common, ia, ib, is_backbone);
}

/* ------------------------------- Backbone ------------------------------- */

fn sugar_template(na_type: NucleicAcidType) -> Vec<TAtom> {
    // Local frame: origin at C1′.
    // Coordinates are approximate but chemically sensible; units Å.
    let mut v = vec![
        TAtom {
            name: "C1'",
            element: Carbon,
            coord: v(0.00, 0.00, 0.00),
        },
        TAtom {
            name: "O4'",
            element: Oxygen,
            coord: v(-1.40, 0.10, 0.05),
        },
        TAtom {
            name: "C4'",
            element: Carbon,
            coord: v(-2.05, -0.70, 0.00),
        },
        TAtom {
            name: "C3'",
            element: Carbon,
            coord: v(-1.60, -2.05, 0.08),
        },
        TAtom {
            name: "O3'",
            element: Oxygen,
            coord: v(-1.75, -3.25, 0.10),
        },
        TAtom {
            name: "C2'",
            element: Carbon,
            coord: v(-0.25, -2.20, -0.05),
        },
        TAtom {
            name: "C5'",
            element: Carbon,
            coord: v(-3.40, -0.25, 0.10),
        },
        TAtom {
            name: "O5'",
            element: Oxygen,
            coord: v(-3.85, 1.05, 0.20),
        },
    ];
    if na_type == NucleicAcidType::Rna {
        v.push(TAtom {
            name: "O2'",
            element: Oxygen,
            coord: Vec3::new(0.55, -3.35, -0.05),
        });
    }
    v
}

fn sugar_bonds(na_type: NucleicAcidType) -> &'static [(&'static str, &'static str)] {
    // Within residue sugar ring + exocyclic O5′
    match na_type {
        NucleicAcidType::Dna => &[
            ("C1'", "O4'"),
            ("O4'", "C4'"),
            ("C4'", "C3'"),
            ("C3'", "O3'"),
            ("C3'", "C2'"),
            ("C2'", "C1'"),
            ("C4'", "C5'"),
            ("C5'", "O5'"),
        ],
        NucleicAcidType::Rna => &[
            ("C1'", "O4'"),
            ("O4'", "C4'"),
            ("C4'", "C3'"),
            ("C3'", "O3'"),
            ("C3'", "C2'"),
            ("C2'", "C1'"),
            ("C2'", "O2'"),
            ("C4'", "C5'"),
            ("C5'", "O5'"),
        ],
    }
}

fn phosphate_template() -> Vec<TAtom> {
    vec![
        TAtom {
            name: "P",
            element: Phosphorus,
            coord: v(-5.10, 1.30, 0.20),
        },
        TAtom {
            name: "OP1",
            element: Oxygen,
            coord: v(-6.25, 2.20, 0.20),
        },
        TAtom {
            name: "OP2",
            element: Oxygen,
            coord: v(-5.60, 0.00, 1.45),
        },
        // O5′ is on sugar; the P—O5′ bond is added via phosphate_bonds()
    ]
}

fn phosphate_bonds() -> &'static [(&'static str, &'static str)] {
    &[("O5'", "P"), ("P", "OP1"), ("P", "OP2")]
}

/* -------------------------------- Bases -------------------------------- */

struct BaseTemplate {
    atoms: Vec<TAtom>,
    bonds: Vec<(&'static str, &'static str)>,
    anchor: &'static str, // "N9" (purines) or "N1" (pyrimidines)
    anchor_shift: f64,    // shift along +x(local) so base sits near helix shell
}

fn base_template(nt: Nucleotide) -> BaseTemplate {
    match nt {
        Nucleotide::A => base_purine_adenine(),
        Nucleotide::G => base_purine_guanine(),
        Nucleotide::C => base_pyrimidine_cytosine(),
        Nucleotide::T => base_pyrimidine_thymine(),
        // todo
        // Nucleotide::U => base_pyrimidine_uracil(),
    }
}

// Simple planar purine; anchor N9 at ~1.47 Å from C1′ along +x.
fn base_purine_adenine() -> BaseTemplate {
    let a = 1.47; // glycosidic C1′—N9
    let atoms = vec![
        t("N9", Nitrogen, a, 0.00, 0.00),
        t("C8", Carbon, a + 1.20, 0.80, 0.00),
        t("N7", Nitrogen, a + 0.70, 2.00, 0.00),
        t("C5", Carbon, a - 0.50, 1.95, 0.00),
        t("C6", Carbon, a - 1.05, 0.70, 0.00),
        t("N6", Nitrogen, a - 2.35, 0.70, 0.00), // exocyclic amino on C6
        t("N1", Nitrogen, a - 0.50, -0.55, 0.00),
        t("C2", Carbon, a + 0.70, -0.85, 0.00),
        t("N3", Nitrogen, a + 1.35, 0.30, 0.00),
        t("C4", Carbon, a + 0.55, 1.40, 0.00),
    ];
    let bonds = vec![
        ("N9", "C8"),
        ("C8", "N7"),
        ("N7", "C5"),
        ("C5", "C6"),
        ("C6", "N1"),
        ("N1", "C2"),
        ("C2", "N3"),
        ("N3", "C4"),
        ("C4", "C5"),
        ("C6", "N6"),
    ];
    BaseTemplate {
        atoms,
        bonds,
        anchor: "N9",
        anchor_shift: 4.8,
    }
}

fn base_purine_guanine() -> BaseTemplate {
    let a = 1.47;
    let atoms = vec![
        t("N9", Nitrogen, a, 0.00, 0.00),
        t("C8", Carbon, a + 1.20, 0.80, 0.00),
        t("N7", Nitrogen, a + 0.70, 2.00, 0.00),
        t("C5", Carbon, a - 0.50, 1.95, 0.00),
        t("C6", Carbon, a - 1.05, 0.70, 0.00),
        t("O6", Oxygen, a - 2.25, 0.70, 0.00), // carbonyl on C6
        t("N1", Nitrogen, a - 0.50, -0.55, 0.00),
        t("C2", Carbon, a + 0.70, -0.85, 0.00),
        t("N2", Nitrogen, a + 1.75, -1.75, 0.00), // exocyclic amino on C2
        t("N3", Nitrogen, a + 1.35, 0.30, 0.00),
        t("C4", Carbon, a + 0.55, 1.40, 0.00),
    ];
    let bonds = vec![
        ("N9", "C8"),
        ("C8", "N7"),
        ("N7", "C5"),
        ("C5", "C6"),
        ("C6", "N1"),
        ("N1", "C2"),
        ("C2", "N3"),
        ("N3", "C4"),
        ("C4", "C5"),
        ("C6", "O6"),
        ("C2", "N2"),
    ];
    BaseTemplate {
        atoms,
        bonds,
        anchor: "N9",
        anchor_shift: 4.8,
    }
}

// Pyrimidine templates — anchor is N1
fn base_pyrimidine_cytosine() -> BaseTemplate {
    let a = 1.47; // C1′—N1
    let atoms = vec![
        t("N1", Nitrogen, a, 0.00, 0.00),
        t("C2", Carbon, a + 1.30, 0.00, 0.00),
        t("O2", Oxygen, a + 2.45, 0.00, 0.00), // carbonyl at C2
        t("N3", Nitrogen, a + 1.25, 1.15, 0.00),
        t("C4", Carbon, a + 0.10, 1.85, 0.00),
        t("N4", Nitrogen, a - 0.05, 3.05, 0.00), // amino at C4
        t("C5", Carbon, a - 1.05, 1.15, 0.00),
        t("C6", Carbon, a - 1.00, 0.00, 0.00),
    ];
    let bonds = vec![
        ("N1", "C2"),
        ("C2", "N3"),
        ("N3", "C4"),
        ("C4", "C5"),
        ("C5", "C6"),
        ("C6", "N1"),
        ("C2", "O2"),
        ("C4", "N4"),
    ];
    BaseTemplate {
        atoms,
        bonds,
        anchor: "N1",
        anchor_shift: 4.6,
    }
}

fn base_pyrimidine_thymine() -> BaseTemplate {
    let a = 1.47;
    let atoms = vec![
        t("N1", Nitrogen, a, 0.00, 0.00),
        t("C2", Carbon, a + 1.30, 0.00, 0.00),
        t("O2", Oxygen, a + 2.45, 0.00, 0.00),
        t("N3", Nitrogen, a + 1.25, 1.15, 0.00),
        t("C4", Carbon, a + 0.10, 1.85, 0.00),
        t("O4", Oxygen, a + 0.10, 3.00, 0.00), // carbonyl at C4
        t("C5", Carbon, a - 1.05, 1.15, 0.00),
        t("C5M", Carbon, a - 1.95, 2.10, 0.00), // methyl (no Hs drawn)
        t("C6", Carbon, a - 1.00, 0.00, 0.00),
    ];
    let bonds = vec![
        ("N1", "C2"),
        ("C2", "N3"),
        ("N3", "C4"),
        ("C4", "C5"),
        ("C5", "C6"),
        ("C6", "N1"),
        ("C2", "O2"),
        ("C4", "O4"),
        ("C5", "C5M"),
    ];
    BaseTemplate {
        atoms,
        bonds,
        anchor: "N1",
        anchor_shift: 4.6,
    }
}

fn base_pyrimidine_uracil() -> BaseTemplate {
    let a = 1.47;
    let atoms = vec![
        t("N1", Nitrogen, a, 0.00, 0.00),
        t("C2", Carbon, a + 1.30, 0.00, 0.00),
        t("O2", Element::Oxygen, a + 2.45, 0.00, 0.00),
        t("N3", Nitrogen, a + 1.25, 1.15, 0.00),
        t("C4", Carbon, a + 0.10, 1.85, 0.00),
        t("O4", Oxygen, a + 0.10, 3.00, 0.00),
        t("C5", Carbon, a - 1.05, 1.15, 0.00),
        t("C6", Carbon, a - 1.00, 0.00, 0.00),
    ];
    let bonds = vec![
        ("N1", "C2"),
        ("C2", "N3"),
        ("N3", "C4"),
        ("C4", "C5"),
        ("C5", "C6"),
        ("C6", "N1"),
        ("C2", "O2"),
        ("C4", "O4"),
    ];
    BaseTemplate {
        atoms,
        bonds,
        anchor: "N1",
        anchor_shift: 4.6,
    }
}

/* ------------------------------- utilities ------------------------------ */

#[inline]
fn v(x: f64, y: f64, z: f64) -> Vec3 {
    Vec3::new(x, y, z)
}

#[inline]
fn t(name: &'static str, el: Element, x: f64, y: f64, z: f64) -> TAtom {
    TAtom {
        name,
        element: el,
        coord: v(x, y, z),
    }
}

/// Returns (DNA, RNA)
pub fn load_na_templates() -> io::Result<(Vec<MoleculeNucleicAcid>, Vec<MoleculeNucleicAcid>)> {
    println!("Loading Nucleic acid templates...");

    let mut dna = Vec::new();
    let mut rna = Vec::new();

    // todo: Both DNA and RNA.
    let start = Instant::now();
    let templates = load_templates(OL24_LIB)?;

    for (ident, template) in templates {
        // todo: Move this to molecule mod A/R,.
        let mut mol = MoleculeNucleicAcid {
            common: MoleculeCommon {
                ident,
                ..Default::default()
            },
            seq: vec![],
            features: Vec::new(),
            // residues: Vec::new(),
        };
        for atom in template.atoms {
            mol.common.atoms.push((&atom).into());
        }

        for bond in template.bonds {
            mol.common
                .bonds
                .push(Bond::from_generic(&bond, &mol.common.atoms).unwrap());
        }

        mol.common.build_adjacency_list();
        mol.common.atom_posits = mol.common.atoms.iter().map(|a| a.posit).collect();

        dna.push(mol);
    }

    dna.sort_by_key(|mol| mol.common.ident.clone());

    let elapsed = start.elapsed().as_millis();
    println!("Loaded lipid templates in {elapsed:.1}ms");

    Ok((dna, rna))
}
