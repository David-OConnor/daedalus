#![allow(unused)]

//! For setting up and rendering nucleic acids: DNA and RNA.

// todo: Load Amber FF params for nucleic acids.

use std::{collections::HashMap, f64::consts::TAU};

use bio_files::BondType;
use lin_alg::f64::Vec3;
use na_seq::{
    AminoAcid,
    Element::{self, *},
    Nucleotide::{self, *},
};

use crate::{
    mol_lig::MoleculeSmall,
    molecule::{
        Atom, Bond, MolGenericRef, MolGenericTrait, MolType, MoleculeCommon, MoleculePeptide,
    },
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
    /// todo: Use the same struct as PlasCAD, or similar?
    pub features: Vec<(String, (usize, usize))>,
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
    pub fn from_seq(seq: &[Nucleotide], na_type: NucleicAcidType, strands: Strands) -> Self {
        let helix = match na_type {
            NucleicAcidType::Rna => {
                // A-form-ish
                HelixGeom {
                    rise: 2.60,
                    twist: 33.0_f64.to_radians(),
                    r_backbone: 4.6,
                    base_rad: 6.2,
                    sugar_ang: 0.35,
                    sugar_dz: 0.6,
                    base_ang: TAU / 2., // bases roughly opposite backbone
                    base_dz: 0.25,
                }
            }
            NucleicAcidType::Dna => {
                // B-DNA-ish
                HelixGeom {
                    rise: 3.40,
                    twist: 36.0_f64.to_radians(),
                    r_backbone: 4.2,
                    base_rad: 6.0,
                    sugar_ang: 0.35,
                    sugar_dz: 0.6,
                    base_ang: TAU / 2.,
                    base_dz: 0.20,
                }
            }
        };

        let mut common = MoleculeCommon::default();

        common.ident = format!(
            "{}-nt {} strand",
            seq.len(),
            match na_type {
                NucleicAcidType::Rna => "RNA",
                NucleicAcidType::Dna => "DNA",
            }
        );

        // Reserve roughly (#atoms per residue ≈ 18 backbone + 10–12 base) × n
        common.atoms.reserve(seq.len() * 32);

        // Keep per-residue name→index for wiring bonds
        let mut idx_maps: Vec<HashMap<&'static str, usize>> = Vec::with_capacity(seq.len());

        // Build residues
        for (i, nt) in seq.iter().enumerate() {
            // Residue placement on helix
            let angle = i as f64 * helix.twist;
            let s_world = helix_point(i, 0.0, helix.r_backbone, helix.sugar_dz, &helix);
            let s_world = rotate_xy(s_world, angle);
            // Local frame at residue (C1′ origin):
            //   e_r: outward radial (base points roughly along +x_local = +e_r)
            //   e_t: tangential (around the helix)
            //   e_z: helix axis (z)
            let e_r = Vec3::new(angle.cos(), angle.sin(), 0.0);
            let e_t = Vec3::new(-angle.sin(), angle.cos(), 0.0);
            let e_z = Vec3::new(0.0, 0.0, 1.0);
            let frame = Frame {
                o: s_world,
                ex: e_r,
                ey: e_t,
                ez: e_z,
            };

            let mut name_to_idx: HashMap<&'static str, usize> = HashMap::new();

            // === Sugar + phosphate ===
            let sugar = sugar_template(na_type);
            let phos = phosphate_template();
            for a in sugar.iter().chain(phos.iter()) {
                let pos = frame.apply(a.coord);
                let idx = push_atom(&mut common, pos, a.element, i);
                name_to_idx.insert(a.name, idx);
            }

            // === Base (ring + exocyclics) ===
            let base = base_template(*nt);
            for a in &base.atoms {
                // Base is further out from axis; we offset its local x by base radial delta.
                // The coordinates below are already drawn in the base plane; we add a global outward
                // shift so the base sits ~base_rad from axis.
                let local = Vec3::new(a.coord.x + base.anchor_shift, a.coord.y, a.coord.z);
                let pos = frame.apply(local);
                let idx = push_atom(&mut common, pos, a.element, i);
                name_to_idx.insert(a.name, idx);
            }

            // === Bonds within residue ===
            // Sugar bonds
            for (a, b) in sugar_bonds(na_type) {
                push_bond_by_name(&mut common, &name_to_idx, a, b, true);
            }
            // Phosphate bonds (O5′—P and two non-bridging oxygens)
            for (a, b) in phosphate_bonds() {
                push_bond_by_name(&mut common, &name_to_idx, a, b, true);
            }
            // Glycosidic bond C1′—(N9 or N1)
            push_bond_by_name(&mut common, &name_to_idx, "C1'", base.anchor, false);
            // Base internal bonds
            for (a, b) in &base.bonds {
                push_bond_by_name(&mut common, &name_to_idx, a, b, false);
            }

            idx_maps.push(name_to_idx);
        }

        // Inter-residue phosphodiester: O3′(i) — P(i+1)
        for i in 0..seq.len().saturating_sub(1) {
            let i0 = idx_maps[i]["O3'"];
            let i1 = idx_maps[i + 1]["P"];
            push_bond_indices(&mut common, i0, i1, true);
        }

        // Positions mirror
        common.atom_posits = common.atoms.iter().map(|a| a.posit).collect();

        common.build_adjacency_list();

        Self {
            common,
            seq: seq.to_vec(),
            features: Vec::new(),
        }
    }

    /// This wrapper that extracts the AA sequence, then chooses a suitable DNA sequence.
    /// note that there are many possible combinations due to multiple codons corresponding
    /// to some AAs.
    pub fn from_peptide(
        peptide: &MoleculePeptide,
        na_type: NucleicAcidType,
        strands: Strands,
    ) -> Self {
        let mut seq = Vec::with_capacity(&peptide.residues.len() * 3);
        for res in &peptide.residues {
            seq.push(Nucleotide::A);
            seq.push(Nucleotide::A);
            seq.push(Nucleotide::A);
        }

        Self::from_seq(&seq, na_type, strands)
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
