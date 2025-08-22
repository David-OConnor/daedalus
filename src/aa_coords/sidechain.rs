#![allow(unused)]

//! This module contains info related to side chains, including their geometry

// Don't show warnings for un`
use std::{f64::consts::TAU, fmt, fmt::Display};

use lin_alg::f64::{Quaternion, Vec3};
use na_seq::AminoAcid;

pub const TAU_DIV2: f64 = TAU / 2.;

// todo: These are temp
pub const LEN_SC: f64 = 1.53;

pub const PRO_PHI_MIN: f64 = 4.83456;
pub const PRO_PHI_MAX: f64 = 5.53269;

// As a convention, we use generic tetrahedral and planar geometry in this module.
// This is a stopgap. Note that we are treating the generic-geometry `BOND_A` as to
// the previous atom in a chain, and `BOND_B` to the next one. For branching, we incorporate
// `BOND_C` and `BOND_D` as required. We use planar geometry as required vice tetrahedral
// when appropriate, eg for Nitrogen atoms. For bonds from hydrogens, we use the anchor bond directly.

// todo: Clean up tetra vs planar

// todo: Gauche+ and trans etc for beta C. EG opposite C' or opposite N?

// todo: You might need more xi angles. Eg methyl groups at the end of a hydrophobic chain can
// todo probably rotate! look this up.

// Notes: Check the Hs on N and O in sidechains. Some depictions I've found have them in various
// places while others do not. For example, most depictions of ASP and GLU have one H on one of
// the Oxygens, but Pymol doesn't.
// For example:

#[derive(Debug, PartialEq, Clone)]
pub enum Sidechain {
    Arg(Arg),
    His(His),
    Lys(Lys),
    Asp(Asp),
    Glu(Glu),
    Ser(Ser),
    Thr(Thr),
    Asn(Asn),
    Gln(Gln),
    Cys(Cys),
    Gly(Gly),
    Pro(Pro),
    Ala(Ala),
    Val(Val),
    Ile(Ile),
    Leu(Leu),
    Met(Met),
    Phe(Phe),
    Tyr(Tyr),
    Trp(Trp),
    /// Sec is not one of the most-common 20.
    Sec(Sec),
}

impl Default for Sidechain {
    fn default() -> Self {
        Self::Gly(Default::default())
    }
}

impl Display for Sidechain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Arg(aa) => {
                write!(
                    f,
                    "Arg (R)\nχ1: {:.2}τ χ2: {:.2}τ χ3: {:.2}τ χ4: {:.2}τ χ5: {:.2}τ",
                    aa.χ_1 / TAU,
                    aa.χ_2 / TAU,
                    aa.χ_3 / TAU,
                    aa.χ_4 / TAU,
                    aa.χ_5
                )
            }
            Self::His(aa) => {
                write!(
                    f,
                    "His (H)\nχ1: {:.2}τ χ2: {:.2}τ",
                    aa.χ_1 / TAU,
                    aa.χ_2 / TAU
                )
            }
            Self::Lys(aa) => {
                write!(
                    f,
                    "Lys (K)\nχ1: {:.2}τ χ2: {:.2}τ χ3: {:.2}τ χ4: {:.2}τ",
                    aa.χ_1 / TAU,
                    aa.χ_2 / TAU,
                    aa.χ_3 / TAU,
                    aa.χ_4 / TAU
                )
            }
            Self::Asp(aa) => {
                write!(
                    f,
                    "Asp (D)\nχ1: {:.2}τ χ2: {:.2}τ",
                    aa.χ_1 / TAU,
                    aa.χ_2 / TAU
                )
            }
            Self::Glu(aa) => {
                write!(
                    f,
                    "Glu\nχ1: {:.2}τ χ2: {:.2}τ χ3: {:.2}τ",
                    aa.χ_1 / TAU,
                    aa.χ_2 / TAU,
                    aa.χ_3 / TAU
                )
            }
            Self::Ser(aa) => {
                write!(f, "Ser(S)\nχ1: {:.2}τ", aa.χ_1 / TAU)
            }
            Self::Thr(aa) => {
                write!(f, "Thr (T)\nχ1: {:.2}τ", aa.χ_1 / TAU)
            }
            Self::Asn(aa) => {
                write!(
                    f,
                    "Asn (N)\nχ1: {:.2}τ χ2: {:.2}τ",
                    aa.χ_1 / TAU,
                    aa.χ_2 / TAU
                )
            }
            Self::Gln(aa) => {
                write!(
                    f,
                    "Gln (Q)\nχ1: {:.2}τ χ2: {:.2}τ χ3: {:.2}τ",
                    aa.χ_1 / TAU,
                    aa.χ_2 / TAU,
                    aa.χ_3 / TAU
                )
            }
            Self::Cys(aa) => {
                write!(f, "Cys (C)\nχ1: {:.2}τ", aa.χ_1 / TAU)
            }
            Self::Sec(aa) => {
                write!(f, "Sec (U)\nχ1: {:.2}τ", aa.χ_1 / TAU)
            }
            Self::Gly(_aa) => {
                write!(f, "Gly (G)")
            }
            Self::Pro(aa) => {
                write!(f, "Pro (P)")
            }
            Self::Ala(_aa) => {
                write!(f, "Ala (A)")
            }
            Self::Val(aa) => {
                write!(f, "Val (V)\nχ1: {:.2}τ", aa.χ_1 / TAU)
            }
            Self::Ile(aa) => {
                write!(
                    f,
                    "Ile (I)\nχ1: {:.2}τ χ2: {:.2}τ",
                    aa.χ_1 / TAU,
                    aa.χ_2 / TAU
                )
            }
            Self::Leu(aa) => {
                write!(
                    f,
                    "Leu (L)\nχ1: {:.2}τ χ2: {:.2}τ",
                    aa.χ_1 / TAU,
                    aa.χ_2 / TAU
                )
            }
            Self::Met(aa) => {
                write!(
                    f,
                    "Met (M)\nχ1: {:.2}τ χ2: {:.2}τ χ3: {:.2}τ",
                    aa.χ_1 / TAU,
                    aa.χ_2 / TAU,
                    aa.χ_3 / TAU
                )
            }
            Self::Phe(aa) => {
                write!(
                    f,
                    "Phe (F)\nχ1: {:.2}τ χ2: {:.2}τ",
                    aa.χ_1 / TAU,
                    aa.χ_2 / TAU
                )
            }
            Self::Tyr(aa) => {
                write!(
                    f,
                    "Tyr (Y)\nχ1: {:.2}τ χ2: {:.2}τ",
                    aa.χ_1 / TAU,
                    aa.χ_2 / TAU
                )
            }
            Self::Trp(aa) => {
                write!(
                    f,
                    "Trp (W)\nχ1: {:.2}τ χ2: {:.2}τ χ3: {:.2}τ",
                    aa.χ_1 / TAU,
                    aa.χ_2 / TAU,
                    aa.χ_3 / TAU
                )
            }
        }
    }
}

impl Sidechain {
    /// Construct an AA (with default dihedral angles) from an amino acid type.
    pub fn from_aa_type(aa_type: AminoAcid) -> Self {
        match aa_type {
            AminoAcid::Arg => Self::Arg(Default::default()),
            AminoAcid::His => Self::His(Default::default()),
            AminoAcid::Lys => Self::Lys(Default::default()),
            AminoAcid::Asp => Self::Asp(Default::default()),
            AminoAcid::Glu => Self::Glu(Default::default()),
            AminoAcid::Ser => Self::Ser(Default::default()),
            AminoAcid::Thr => Self::Thr(Default::default()),
            AminoAcid::Asn => Self::Asn(Default::default()),
            AminoAcid::Gln => Self::Gln(Default::default()),
            AminoAcid::Cys => Self::Cys(Default::default()),
            AminoAcid::Sec => Self::Sec(Default::default()),
            AminoAcid::Gly => Self::Gly(Default::default()),
            AminoAcid::Pro => Self::Pro(Default::default()),
            AminoAcid::Ala => Self::Ala(Default::default()),
            AminoAcid::Val => Self::Val(Default::default()),
            AminoAcid::Ile => Self::Ile(Default::default()),
            AminoAcid::Leu => Self::Leu(Default::default()),
            AminoAcid::Met => Self::Met(Default::default()),
            AminoAcid::Phe => Self::Phe(Default::default()),
            AminoAcid::Tyr => Self::Tyr(Default::default()),
            AminoAcid::Trp => Self::Trp(Default::default()),
        }
    }

    /// Construct an AA (with default dihedral angles) from a single-letter identifier.
    /// Returns `None` if an invalid letter is passed.
    pub fn from_ident_single_letter(ident: &str) -> Option<Self> {
        match ident {
            "R" => Some(Self::Arg(Default::default())),
            "H" => Some(Self::His(Default::default())),
            "K" => Some(Self::Lys(Default::default())),
            "D" => Some(Self::Asp(Default::default())),
            "E" => Some(Self::Glu(Default::default())),
            "S" => Some(Self::Ser(Default::default())),
            "T" => Some(Self::Thr(Default::default())),
            "N" => Some(Self::Asn(Default::default())),
            "Q" => Some(Self::Gln(Default::default())),
            "C" => Some(Self::Cys(Default::default())),
            "U" => Some(Self::Sec(Default::default())),
            "G" => Some(Self::Gly(Default::default())),
            "P" => Some(Self::Pro(Default::default())),
            "A" => Some(Self::Ala(Default::default())),
            "V" => Some(Self::Val(Default::default())),
            "I" => Some(Self::Ile(Default::default())),
            "L" => Some(Self::Leu(Default::default())),
            "M" => Some(Self::Met(Default::default())),
            "F" => Some(Self::Phe(Default::default())),
            "Y" => Some(Self::Tyr(Default::default())),
            "W" => Some(Self::Trp(Default::default())),
            _ => None,
        }
    }

    /// todo: Instead of this, many impl partial eq in a way that makes sense?
    pub fn aa_type(&self) -> AminoAcid {
        match self {
            Self::Arg(_) => AminoAcid::Arg,
            Self::His(_) => AminoAcid::His,
            Self::Lys(_) => AminoAcid::Lys,
            Self::Asp(_) => AminoAcid::Asp,
            Self::Glu(_) => AminoAcid::Glu,
            Self::Ser(_) => AminoAcid::Ser,
            Self::Thr(_) => AminoAcid::Thr,
            Self::Asn(_) => AminoAcid::Asn,
            Self::Gln(_) => AminoAcid::Gln,
            Self::Cys(_) => AminoAcid::Cys,
            Self::Sec(_) => AminoAcid::Sec,
            Self::Gly(_) => AminoAcid::Gly,
            Self::Pro(_) => AminoAcid::Pro,
            Self::Ala(_) => AminoAcid::Ala,
            Self::Val(_) => AminoAcid::Val,
            Self::Ile(_) => AminoAcid::Ile,
            Self::Leu(_) => AminoAcid::Leu,
            Self::Met(_) => AminoAcid::Met,
            Self::Phe(_) => AminoAcid::Phe,
            Self::Tyr(_) => AminoAcid::Tyr,
            Self::Trp(_) => AminoAcid::Trp,
        }
    }

    pub fn aa_name(&self) -> &str {
        match self {
            Self::Arg(_) => "Arg (R)",
            Self::His(_) => "His (H)",
            Self::Lys(_) => "Lys (K)",
            Self::Asp(_) => "Asp (D)",
            Self::Glu(_) => "Glu (E)",
            Self::Ser(_) => "Ser (S)",
            Self::Thr(_) => "Thr (T)",
            Self::Asn(_) => "Asn (N)",
            Self::Gln(_) => "Gln (Q)",
            Self::Cys(_) => "Cys (C)",
            Self::Sec(_) => "Sec (U)",
            Self::Gly(_) => "Gly (G)",
            Self::Pro(_) => "Pro (P)",
            Self::Ala(_) => "Ala (A)",
            Self::Val(_) => "Val (V)",
            Self::Ile(_) => "Ile (I)",
            Self::Leu(_) => "Leu (L)",
            Self::Met(_) => "Met (M)",
            Self::Phe(_) => "Phe (F)",
            Self::Tyr(_) => "Tyr (Y)",
            Self::Trp(_) => "Trp (W)",
        }
    }

    pub fn aa_ident_single_letter(&self) -> &str {
        match self {
            Self::Arg(_) => "R",
            Self::His(_) => "H",
            Self::Lys(_) => "K",
            Self::Asp(_) => "D",
            Self::Glu(_) => "E",
            Self::Ser(_) => "S",
            Self::Thr(_) => "T",
            Self::Asn(_) => "N",
            Self::Gln(_) => "Q",
            Self::Cys(_) => "C",
            Self::Sec(_) => "U",
            Self::Gly(_) => "G",
            Self::Pro(_) => "P",
            Self::Ala(_) => "A",
            Self::Val(_) => "V",
            Self::Ile(_) => "I",
            Self::Leu(_) => "L",
            Self::Met(_) => "M",
            Self::Phe(_) => "F",
            Self::Tyr(_) => "Y",
            Self::Trp(_) => "W",
        }
    }

    pub fn get_χ1(&self) -> Option<f64> {
        match self {
            Self::Arg(aa) => Some(aa.χ_1),
            Self::His(aa) => Some(aa.χ_1),
            Self::Lys(aa) => Some(aa.χ_1),
            Self::Asp(aa) => Some(aa.χ_1),
            Self::Glu(aa) => Some(aa.χ_1),
            Self::Ser(aa) => Some(aa.χ_1),
            Self::Thr(aa) => Some(aa.χ_1),
            Self::Asn(aa) => Some(aa.χ_1),
            Self::Gln(aa) => Some(aa.χ_1),
            Self::Cys(aa) => Some(aa.χ_1),
            Self::Sec(aa) => Some(aa.χ_1),
            Self::Val(aa) => Some(aa.χ_1),
            Self::Ile(aa) => Some(aa.χ_1),
            Self::Leu(aa) => Some(aa.χ_1),
            Self::Met(aa) => Some(aa.χ_1),
            Self::Phe(aa) => Some(aa.χ_1),
            Self::Tyr(aa) => Some(aa.χ_1),
            Self::Trp(aa) => Some(aa.χ_1),
            _ => None,
        }
    }

    pub fn get_χ2(&self) -> Option<f64> {
        match self {
            Self::Arg(aa) => Some(aa.χ_2),
            Self::His(aa) => Some(aa.χ_2),
            Self::Lys(aa) => Some(aa.χ_2),
            Self::Asp(aa) => Some(aa.χ_2),
            Self::Glu(aa) => Some(aa.χ_2),
            Self::Asn(aa) => Some(aa.χ_2),
            Self::Gln(aa) => Some(aa.χ_2),
            Self::Ile(aa) => Some(aa.χ_2),
            Self::Leu(aa) => Some(aa.χ_2),
            Self::Met(aa) => Some(aa.χ_2),
            Self::Phe(aa) => Some(aa.χ_2),
            Self::Tyr(aa) => Some(aa.χ_2),
            Self::Trp(aa) => Some(aa.χ_2),
            _ => None,
        }
    }
    pub fn get_χ3(&self) -> Option<f64> {
        match self {
            Self::Arg(aa) => Some(aa.χ_3),
            Self::Lys(aa) => Some(aa.χ_3),
            Self::Glu(aa) => Some(aa.χ_3),
            Self::Gln(aa) => Some(aa.χ_3),
            Self::Met(aa) => Some(aa.χ_3),
            _ => None,
        }
    }
    pub fn get_χ4(&self) -> Option<f64> {
        match self {
            Self::Arg(aa) => Some(aa.χ_4),
            Self::Lys(aa) => Some(aa.χ_4),
            _ => None,
        }
    }
    pub fn get_χ5(&self) -> Option<f64> {
        match self {
            Self::Arg(aa) => Some(aa.χ_5),
            _ => None,
        }
    }

    pub fn get_mut_χ1(&mut self) -> Option<&mut f64> {
        match self {
            Self::Arg(aa) => Some(&mut aa.χ_1),
            Self::His(aa) => Some(&mut aa.χ_1),
            Self::Lys(aa) => Some(&mut aa.χ_1),
            Self::Asp(aa) => Some(&mut aa.χ_1),
            Self::Glu(aa) => Some(&mut aa.χ_1),
            Self::Ser(aa) => Some(&mut aa.χ_1),
            Self::Thr(aa) => Some(&mut aa.χ_1),
            Self::Asn(aa) => Some(&mut aa.χ_1),
            Self::Gln(aa) => Some(&mut aa.χ_1),
            Self::Cys(aa) => Some(&mut aa.χ_1),
            Self::Sec(aa) => Some(&mut aa.χ_1),
            Self::Val(aa) => Some(&mut aa.χ_1),
            Self::Ile(aa) => Some(&mut aa.χ_1),
            Self::Leu(aa) => Some(&mut aa.χ_1),
            Self::Met(aa) => Some(&mut aa.χ_1),
            Self::Phe(aa) => Some(&mut aa.χ_1),
            Self::Tyr(aa) => Some(&mut aa.χ_1),
            Self::Trp(aa) => Some(&mut aa.χ_1),
            _ => None,
        }
    }

    pub fn get_mut_χ2(&mut self) -> Option<&mut f64> {
        match self {
            Self::Arg(aa) => Some(&mut aa.χ_2),
            Self::His(aa) => Some(&mut aa.χ_2),
            Self::Lys(aa) => Some(&mut aa.χ_2),
            Self::Asp(aa) => Some(&mut aa.χ_2),
            Self::Glu(aa) => Some(&mut aa.χ_2),
            Self::Asn(aa) => Some(&mut aa.χ_2),
            Self::Gln(aa) => Some(&mut aa.χ_2),
            Self::Ile(aa) => Some(&mut aa.χ_2),
            Self::Leu(aa) => Some(&mut aa.χ_2),
            Self::Met(aa) => Some(&mut aa.χ_2),
            Self::Phe(aa) => Some(&mut aa.χ_2),
            Self::Tyr(aa) => Some(&mut aa.χ_2),
            Self::Trp(aa) => Some(&mut aa.χ_2),
            _ => None,
        }
    }
    pub fn get_mut_χ3(&mut self) -> Option<&mut f64> {
        match self {
            Self::Arg(aa) => Some(&mut aa.χ_3),
            Self::Lys(aa) => Some(&mut aa.χ_3),
            Self::Glu(aa) => Some(&mut aa.χ_3),
            Self::Gln(aa) => Some(&mut aa.χ_3),
            Self::Met(aa) => Some(&mut aa.χ_3),
            _ => None,
        }
    }
    pub fn get_mut_χ4(&mut self) -> Option<&mut f64> {
        match self {
            Self::Arg(aa) => Some(&mut aa.χ_4),
            Self::Lys(aa) => Some(&mut aa.χ_4),
            _ => None,
        }
    }
    pub fn get_mut_χ5(&mut self) -> Option<&mut f64> {
        match self {
            Self::Arg(aa) => Some(&mut aa.χ_5),
            _ => None,
        }
    }

    pub fn add_to_χ1(&mut self, val: f64) {
        match self {
            Self::Arg(aa) => aa.χ_1 += val,
            Self::His(aa) => aa.χ_1 += val,
            Self::Lys(aa) => aa.χ_1 += val,
            Self::Asp(aa) => aa.χ_1 += val,
            Self::Glu(aa) => aa.χ_1 += val,
            Self::Ser(aa) => aa.χ_1 += val,
            Self::Thr(aa) => aa.χ_1 += val,
            Self::Asn(aa) => aa.χ_1 += val,
            Self::Gln(aa) => aa.χ_1 += val,
            Self::Cys(aa) => aa.χ_1 += val,
            Self::Sec(aa) => aa.χ_1 += val,
            Self::Val(aa) => aa.χ_1 += val,
            Self::Ile(aa) => aa.χ_1 += val,
            Self::Leu(aa) => aa.χ_1 += val,
            Self::Met(aa) => aa.χ_1 += val,
            Self::Phe(aa) => aa.χ_1 += val,
            Self::Tyr(aa) => aa.χ_1 += val,
            Self::Trp(aa) => aa.χ_1 += val,
            _ => (),
        }
    }

    pub fn add_to_χ2(&mut self, val: f64) {
        match self {
            Self::Arg(aa) => aa.χ_2 += val,
            Self::His(aa) => aa.χ_2 += val,
            Self::Lys(aa) => aa.χ_2 += val,
            Self::Asp(aa) => aa.χ_2 += val,
            Self::Glu(aa) => aa.χ_2 += val,
            Self::Asn(aa) => aa.χ_2 += val,
            Self::Gln(aa) => aa.χ_2 += val,
            Self::Ile(aa) => aa.χ_2 += val,
            Self::Leu(aa) => aa.χ_2 += val,
            Self::Met(aa) => aa.χ_2 += val,
            Self::Phe(aa) => aa.χ_2 += val,
            Self::Tyr(aa) => aa.χ_2 += val,
            Self::Trp(aa) => aa.χ_2 += val,
            _ => (),
        }
    }
    pub fn add_to_χ3(&mut self, val: f64) {
        match self {
            Self::Arg(aa) => aa.χ_3 += val,
            Self::Lys(aa) => aa.χ_3 += val,
            Self::Glu(aa) => aa.χ_3 += val,
            Self::Gln(aa) => aa.χ_3 += val,
            Self::Met(aa) => aa.χ_3 += val,
            _ => (),
        }
    }
    pub fn _add_to_χ4(&mut self, val: f64) {
        match self {
            Self::Arg(aa) => aa.χ_4 += val,
            Self::Lys(aa) => aa.χ_4 += val,
            _ => (),
        }
    }
    pub fn _add_to_χ5(&mut self, val: f64) {
        match self {
            Self::Arg(aa) => aa.χ_5 += val,
            _ => (),
        }
    }

    // pub fn set_χ1(&mut self, val: f64) {
    //     match self {
    //         Self::Arg(aa) => aa.χ_1 = val,
    //         Self::His(aa) => aa.χ_1 = val,
    //         Self::Lys(aa) => aa.χ_1 = val,
    //         Self::Asp(aa) => aa.χ_1 = val,
    //         Self::Glu(aa) => aa.χ_1 = val,
    //         Self::Ser(aa) => aa.χ_1 = val,
    //         Self::Thr(aa) => aa.χ_1 = val,
    //         Self::Asn(aa) => aa.χ_1 = val,
    //         Self::Gln(aa) => aa.χ_1 = val,
    //         Self::Cys(aa) => aa.χ_1 = val,
    //         Self::Sec(aa) => aa.χ_1 = val,
    //         Self::Val(aa) => aa.χ_1 = val,
    //         Self::Ile(aa) => aa.χ_1 = val,
    //         Self::Leu(aa) => aa.χ_1 = val,
    //         Self::Met(aa) => aa.χ_1 = val,
    //         Self::Phe(aa) => aa.χ_1 = val,
    //         Self::Tyr(aa) => aa.χ_1 = val,
    //         Self::Trp(aa) => aa.χ_1 = val,
    //         _ => (),
    //     }
    // }
    //
    // pub fn set_χ2(&mut self, val: f64) {
    //     match self {
    //         Self::Arg(aa) => aa.χ_2 = val,
    //         Self::His(aa) => aa.χ_2 = val,
    //         Self::Lys(aa) => aa.χ_2 = val,
    //         Self::Asp(aa) => aa.χ_2 = val,
    //         Self::Glu(aa) => aa.χ_2 = val,
    //         Self::Asn(aa) => aa.χ_2 = val,
    //         Self::Gln(aa) => aa.χ_2 = val,
    //         Self::Ile(aa) => aa.χ_2 = val,
    //         Self::Leu(aa) => aa.χ_2 = val,
    //         Self::Met(aa) => aa.χ_2 = val,
    //         Self::Phe(aa) => aa.χ_2 = val,
    //         Self::Tyr(aa) => aa.χ_2 = val,
    //         Self::Trp(aa) => aa.χ_2 = val,
    //         _ => (),
    //     }
    // }
    // pub fn set_χ3(&mut self, val: f64) {
    //     match self {
    //         Self::Arg(aa) => aa.χ_3 = val,
    //         Self::Lys(aa) => aa.χ_3 = val,
    //         Self::Glu(aa) => aa.χ_3 = val,
    //         Self::Asn(aa) => aa.χ_3 = val,
    //         Self::Gln(aa) => aa.χ_3 = val,
    //         Self::Met(aa) => aa.χ_3 = val,
    //         _ => (),
    //     }
    // }
    // pub fn set_χ4(&mut self, val: f64) {
    //     match self {
    //         Self::Arg(aa) => aa.χ_4 = val,
    //         Self::Lys(aa) => aa.χ_4 = val,
    //         Self::Gln(aa) => aa.χ_4 = val,
    //         _ => (),
    //     }
    // }
    // pub fn set_χ5(&mut self, val: f64) {
    //     match self {
    //         Self::Arg(aa) => aa.χ_5 = val,
    //         _ => (),
    //     }
    // }
    // pub fn set_χ6(&mut self, val: f64) {
    //     match self {
    //         Self::Arg(aa) => aa.χ_6 = val,
    //         _ => (),
    //     }
    // }
    // pub fn set_χ7(&mut self, val: f64) {
    //     match self {
    //         Self::Arg(aa) => aa.χ_7 = val,
    //         _ => (),
    //     }
    // }
}

/// These are global coordinates. Analog to `BackboneCoordsAa`
#[derive(Debug, Default)]
pub struct CoordsArg {
    pub c_beta: Vec3,
    pub c_gamma: Vec3,
    pub c_delta: Vec3,
    pub n_eps: Vec3,
    pub c_zeta: Vec3,
    pub n_eta1: Vec3,
    pub n_eta2: Vec3,
    pub h_n_eps: Vec3,
    pub h_n_eta1_a: Vec3,
    pub h_n_eta1_b: Vec3,
    pub h_n_eta2_a: Vec3,
    pub h_n_eta2_b: Vec3,
    pub h_c_beta_a: Vec3,
    pub h_c_beta_b: Vec3,
    pub h_c_gamma_a: Vec3,
    pub h_c_gamma_b: Vec3,
    pub h_c_delta_a: Vec3,
    pub h_c_delta_b: Vec3,

    pub c_beta_orientation: Quaternion,
    pub c_gamma_orientation: Quaternion,
    pub c_delta_orientation: Quaternion,
    pub n_eps_orientation: Quaternion,
    pub c_zeta_orientation: Quaternion,
    pub n_eta1_orientation: Quaternion,
    pub n_eta2_orientation: Quaternion,
}

#[derive(Debug, Default)]
pub struct CoordsHis {
    pub c_beta: Vec3,
    pub c_gamma: Vec3,
    pub c_delta1: Vec3,
    pub n_delta2: Vec3,
    pub n_eps1: Vec3,
    pub c_eps2: Vec3,
    pub h_n_delta: Vec3,
    // pub h_n_eps: Vec3,
    pub h_c_beta_a: Vec3,
    pub h_c_beta_b: Vec3,
    pub h_c_delta1: Vec3,
    pub h_c_eps2: Vec3,

    pub c_beta_orientation: Quaternion,
    pub c_gamma_orientation: Quaternion,
    pub c_delta1_orientation: Quaternion,
    pub n_delta2_orientation: Quaternion,
    pub n_eps1_orientation: Quaternion,
    pub c_eps2_orientation: Quaternion,
}

#[derive(Debug, Default)]
pub struct CoordsLys {
    pub c_beta: Vec3,
    pub c_gamma: Vec3,
    pub c_delta: Vec3,
    pub c_eps: Vec3,
    pub n_zeta: Vec3,
    pub h_n_zeta_a: Vec3,
    pub h_n_zeta_b: Vec3,
    // todo: Third N here?
    pub h_c_beta_a: Vec3,
    pub h_c_beta_b: Vec3,
    pub h_c_gamma_a: Vec3,
    pub h_c_gamma_b: Vec3,
    pub h_c_delta_a: Vec3,
    pub h_c_delta_b: Vec3,
    pub h_c_eps_a: Vec3,
    pub h_c_eps_b: Vec3,

    pub c_beta_orientation: Quaternion,
    pub c_gamma_orientation: Quaternion,
    pub c_delta_orientation: Quaternion,
    pub c_eps_orientation: Quaternion,
    pub n_zeta_orientation: Quaternion,
}

#[derive(Debug, Default)]
pub struct CoordsAsp {
    pub c_beta: Vec3,
    pub c_gamma: Vec3,
    pub o_delta1: Vec3,
    pub o_delta2: Vec3,
    pub h_c_beta_a: Vec3,
    pub h_c_beta_b: Vec3,

    pub c_beta_orientation: Quaternion,
    pub c_gamma_orientation: Quaternion,
    pub o_delta1_orientation: Quaternion,
    pub o_delta2_orientation: Quaternion,
}

#[derive(Debug, Default)]
pub struct CoordsGlu {
    pub c_beta: Vec3,
    pub c_gamma: Vec3,
    pub c_delta: Vec3,
    pub o_eps1: Vec3,
    pub o_eps2: Vec3,
    pub h_c_beta_a: Vec3,
    pub h_c_beta_b: Vec3,
    pub h_c_gamma_a: Vec3,
    pub h_c_gamma_b: Vec3,

    pub c_beta_orientation: Quaternion,
    pub c_gamma_orientation: Quaternion,
    pub c_delta_orientation: Quaternion,
    pub o_eps1_orientation: Quaternion,
    pub o_eps2_orientation: Quaternion,
}

#[derive(Debug, Default)]
pub struct CoordsSer {
    pub c_beta: Vec3,
    pub o_gamma: Vec3,
    pub h_c_beta_a: Vec3,
    pub h_c_beta_b: Vec3,
    pub h_o_gamma: Vec3,

    pub c_beta_orientation: Quaternion,
    pub o_gamma_orientation: Quaternion,
}

#[derive(Debug, Default)]
pub struct CoordsThr {
    pub c_beta: Vec3,
    pub o_gamma1: Vec3,
    pub c_gamma2: Vec3,
    pub h_c_beta: Vec3,
    pub h_o: Vec3,
    pub h_c_gamma1: Vec3,
    pub h_c_gamma2: Vec3,
    pub h_c_gamma3: Vec3,

    pub c_beta_orientation: Quaternion,
    pub o_gamma1_orientation: Quaternion,
    pub c_gamma2_orientation: Quaternion,
}

#[derive(Debug, Default)]
pub struct CoordsAsn {
    pub c_beta: Vec3,
    pub c_gamma: Vec3,
    pub o_delta1: Vec3,
    pub n_delta2: Vec3,
    pub h_n_delta_a: Vec3,
    pub h_n_delta_b: Vec3,
    pub h_c_beta_a: Vec3,
    pub h_c_beta_b: Vec3,

    pub c_beta_orientation: Quaternion,
    pub c_gamma_orientation: Quaternion,
    pub n_delta2_orientation: Quaternion,
}

#[derive(Debug, Default)]
pub struct CoordsGln {
    pub c_beta: Vec3,
    pub c_gamma: Vec3,
    pub c_delta: Vec3,
    pub o_eps1: Vec3,
    pub n_eps2: Vec3,
    pub h_n_eps_a: Vec3,
    pub h_n_eps_b: Vec3,
    pub h_c_beta_a: Vec3,
    pub h_c_beta_b: Vec3,
    pub h_c_gamma_a: Vec3,
    pub h_c_gamma_b: Vec3,

    pub c_beta_orientation: Quaternion,
    pub c_gamma_orientation: Quaternion,
    pub c_delta_orientation: Quaternion,
    pub n_eps2_orientation: Quaternion,
}

#[derive(Debug, Default)]
pub struct CoordsAla {
    pub c_beta: Vec3,
    pub h_c_beta_a: Vec3,
    pub h_c_beta_b: Vec3,
    pub h_c_beta_c: Vec3,

    pub c_beta_orientation: Quaternion,
}

#[derive(Debug, Default)]
pub struct CoordsVal {
    pub c_beta: Vec3,
    pub c_gamma1: Vec3,
    pub c_gamma2: Vec3,
    pub h_c_beta: Vec3,
    pub h_c_gamma1_a: Vec3,
    pub h_c_gamma1_b: Vec3,
    pub h_c_gamma1_c: Vec3,
    pub h_c_gamma2_a: Vec3,
    pub h_c_gamma2_b: Vec3,
    pub h_c_gamma2_c: Vec3,

    pub c_beta_orientation: Quaternion,
    pub c_gamma1_orientation: Quaternion,
    pub c_gamma2_orientation: Quaternion,
}

#[derive(Debug, Default)]
pub struct CoordsIle {
    pub c_beta: Vec3,
    pub c_gamma1: Vec3,
    pub c_gamma2: Vec3,
    pub c_delta: Vec3,
    pub h_c_beta: Vec3,
    pub h_c_gamma1_a: Vec3,
    pub h_c_gamma1_b: Vec3,
    pub h_c_gamma1_c: Vec3,
    pub h_c_gamma2_a: Vec3,
    pub h_c_gamma2_b: Vec3,
    pub h_c_delta_a: Vec3,
    pub h_c_delta_b: Vec3,
    pub h_c_delta_c: Vec3,

    pub c_beta_orientation: Quaternion,
    pub c_gamma1_orientation: Quaternion,
    pub c_gamma2_orientation: Quaternion,
    pub c_delta_orientation: Quaternion,
}

#[derive(Debug, Default)]
pub struct CoordsLeu {
    pub c_beta: Vec3,
    pub c_gamma: Vec3,
    pub c_delta1: Vec3,
    pub c_delta2: Vec3,
    pub h_c_beta_a: Vec3,
    pub h_c_beta_b: Vec3,
    pub h_c_gamma: Vec3,
    pub h_c_delta1_a: Vec3,
    pub h_c_delta1_b: Vec3,
    pub h_c_delta1_c: Vec3,
    pub h_c_delta2_a: Vec3,
    pub h_c_delta2_b: Vec3,
    pub h_c_delta2_c: Vec3,

    pub c_beta_orientation: Quaternion,
    pub c_gamma_orientation: Quaternion,
    pub c_delta1_orientation: Quaternion,
    pub c_delta2_orientation: Quaternion,
}

#[derive(Debug, Default)]
pub struct CoordsCys {
    pub c_beta: Vec3,
    pub s_gamma: Vec3,
    pub h_c_beta_a: Vec3,
    pub h_c_beta_b: Vec3,
    pub h_s_gamma: Vec3,

    pub c_beta_orientation: Quaternion,
    pub s_gamma_orientation: Quaternion,
}

#[derive(Debug, Default)]
pub struct CoordsSec {
    pub c_beta: Vec3,
    pub se_gamma: Vec3,
    pub h_c_beta_a: Vec3,
    pub h_c_beta_b: Vec3,

    pub c_beta_orientation: Quaternion,
    pub se_gamma_orientation: Quaternion,
}

pub struct CoordsMet {
    pub c_beta: Vec3,
    pub c_gamma: Vec3,
    pub s_delta: Vec3,
    pub c_eps: Vec3,
    pub h_c_beta_a: Vec3,
    pub h_c_beta_b: Vec3,
    pub h_c_gamma_a: Vec3,
    pub h_c_gamma_b: Vec3,
    pub h_c_eps_a: Vec3,
    pub h_c_eps_b: Vec3,
    pub h_c_eps_c: Vec3,

    pub c_beta_orientation: Quaternion,
    pub c_gamma_orientation: Quaternion,
    pub s_delta_orientation: Quaternion,
    pub c_eps_orientation: Quaternion,
}

#[derive(Debug, Default)]
pub struct CoordsGly {
    pub h: Vec3,
}

#[derive(Debug, Default)]
pub struct CoordsPro {
    pub c_beta: Vec3,
    pub c_gamma: Vec3,
    pub c_delta: Vec3,
    pub h_c_beta_a: Vec3,
    pub h_c_beta_b: Vec3,
    pub h_c_gamma_a: Vec3,
    pub h_c_gamma_b: Vec3,
    pub h_c_delta_a: Vec3,
    pub h_c_delta_b: Vec3,

    pub c_beta_orientation: Quaternion,
    pub c_gamma_orientation: Quaternion,
    pub c_delta_orientation: Quaternion,
}

#[derive(Debug, Default)]
pub struct CoordsPhe {
    pub c_beta: Vec3,
    pub c_gamma: Vec3,
    pub c_delta1: Vec3,
    pub c_delta2: Vec3,
    pub c_eps1: Vec3,
    pub c_eps2: Vec3,
    pub c_zeta: Vec3,
    pub h_c_beta_a: Vec3,
    pub h_c_beta_b: Vec3,
    pub h_c_delta1: Vec3,
    pub h_c_delta2: Vec3,
    pub h_c_eps1: Vec3,
    pub h_c_eps2: Vec3,
    pub h_c_zeta: Vec3,

    pub c_beta_orientation: Quaternion,
    pub c_gamma_orientation: Quaternion,
    pub c_delta1_orientation: Quaternion,
    pub c_delta2_orientation: Quaternion,
    pub c_eps1_orientation: Quaternion,
    pub c_eps2_orientation: Quaternion,
    pub c_zeta_orientation: Quaternion,
}

#[derive(Debug, Default)]
pub struct CoordsTyr {
    pub c_beta: Vec3,
    pub c_gamma: Vec3,
    pub c_delta1: Vec3,
    pub c_delta2: Vec3,
    pub c_eps1: Vec3,
    pub c_eps2: Vec3,
    pub c_zeta: Vec3,
    pub o_eta: Vec3,
    pub h_c_beta_a: Vec3,
    pub h_c_beta_b: Vec3,
    pub h_c_delta1: Vec3,
    pub h_c_delta2: Vec3,
    pub h_c_eps1: Vec3,
    pub h_c_eps2: Vec3,
    pub h_o_eta: Vec3,

    pub c_beta_orientation: Quaternion,
    pub c_gamma_orientation: Quaternion,
    pub c_delta1_orientation: Quaternion,
    pub c_delta2_orientation: Quaternion,
    pub c_eps1_orientation: Quaternion,
    pub c_eps2_orientation: Quaternion,
    pub c_zeta_orientation: Quaternion,
    pub o_eta_orientation: Quaternion,
}

#[derive(Debug, Default)]
/// We snake around in such a way that there's no branching, and each atom
/// is informationally-connected to the prev one.
pub struct CoordsTrp {
    pub c_beta: Vec3,
    pub c_gamma: Vec3,
    pub c_delta: Vec3,
    pub n_eps: Vec3,
    pub c_zeta: Vec3,
    pub c_eta: Vec3,
    pub c_theta: Vec3,
    pub c_iota: Vec3,
    pub c_kappa: Vec3,
    pub c_lambda: Vec3,
    pub h_c_beta_a: Vec3,
    pub h_c_beta_b: Vec3,
    pub h_c_delta: Vec3,
    pub h_n_eps: Vec3,
    pub h_c_theta: Vec3,
    pub h_c_iota: Vec3,
    pub h_c_kappa: Vec3,
    pub h_c_lambda: Vec3,

    pub c_beta_orientation: Quaternion,
    pub c_gamma_orientation: Quaternion,
    pub c_delta_orientation: Quaternion,
    pub n_eps_orientation: Quaternion,
    pub c_zeta_orientation: Quaternion,
    pub c_eta_orientation: Quaternion,
    pub c_theta_orientation: Quaternion,
    pub c_iota_orientation: Quaternion,
    pub c_kappa_orientation: Quaternion,
    pub c_lambda_orientation: Quaternion,
}

// todo: Coord structs for the remaining AAs.

// the AA-specific structs below specify dihedral angles for each AA instance
// `χ_1` for each is for the bond between the c_alpha, and the first atom in the
// sidechain (eg c_bravo)

#[derive(Debug, PartialEq, Clone)]
pub struct Arg {
    pub χ_1: f64,
    pub χ_2: f64,
    pub χ_3: f64,
    pub χ_4: f64,
    pub χ_5: f64,
}

impl Default for Arg {
    fn default() -> Self {
        Self {
            χ_1: TAU_DIV2,
            χ_2: TAU_DIV2,
            χ_3: TAU_DIV2,
            χ_4: TAU_DIV2,
            χ_5: TAU_DIV2,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct His {
    pub χ_1: f64,
    pub χ_2: f64,
}

impl Default for His {
    fn default() -> Self {
        Self {
            χ_1: TAU_DIV2,
            χ_2: TAU_DIV2,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Lys {
    pub χ_1: f64,
    pub χ_2: f64,
    pub χ_3: f64,
    pub χ_4: f64,
}

impl Default for Lys {
    fn default() -> Self {
        Self {
            χ_1: TAU_DIV2,
            χ_2: TAU_DIV2,
            χ_3: TAU_DIV2,
            χ_4: TAU_DIV2,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Asp {
    pub χ_1: f64,
    pub χ_2: f64,
}

impl Default for Asp {
    fn default() -> Self {
        Self {
            χ_1: TAU_DIV2,
            χ_2: TAU_DIV2,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Glu {
    pub χ_1: f64,
    pub χ_2: f64,
    pub χ_3: f64,
}

impl Default for Glu {
    fn default() -> Self {
        Self {
            χ_1: TAU_DIV2,
            χ_2: TAU_DIV2,
            χ_3: TAU_DIV2,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Ser {
    pub χ_1: f64,
}

impl Default for Ser {
    fn default() -> Self {
        Self { χ_1: TAU_DIV2 }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Thr {
    pub χ_1: f64,
}

impl Default for Thr {
    fn default() -> Self {
        Self { χ_1: TAU_DIV2 }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Asn {
    pub χ_1: f64,
    pub χ_2: f64,
}

impl Default for Asn {
    fn default() -> Self {
        Self {
            χ_1: TAU_DIV2,
            χ_2: TAU_DIV2,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Gln {
    pub χ_1: f64,
    pub χ_2: f64,
    pub χ_3: f64,
}

impl Default for Gln {
    fn default() -> Self {
        Self {
            χ_1: TAU_DIV2,
            χ_2: TAU_DIV2,
            χ_3: TAU_DIV2,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Cys {
    pub χ_1: f64,
}

impl Default for Cys {
    fn default() -> Self {
        Self { χ_1: TAU_DIV2 }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Sec {
    pub χ_1: f64,
}

impl Default for Sec {
    fn default() -> Self {
        Self { χ_1: TAU_DIV2 }
    }
}

#[derive(Debug, Default, PartialEq, Clone)]
pub struct Gly {}

#[derive(Debug, Default, PartialEq, Clone)]
pub struct Pro {}

#[derive(Debug, Default, PartialEq, Clone)]
pub struct Ala {}

#[derive(Debug, PartialEq, Clone)]
pub struct Val {
    pub χ_1: f64,
}

impl Default for Val {
    fn default() -> Self {
        Self { χ_1: TAU_DIV2 }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Ile {
    pub χ_1: f64,
    pub χ_2: f64,
}

impl Default for Ile {
    fn default() -> Self {
        Self {
            χ_1: TAU_DIV2,
            χ_2: TAU_DIV2,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Leu {
    pub χ_1: f64,
    pub χ_2: f64,
}

impl Default for Leu {
    fn default() -> Self {
        Self {
            χ_1: TAU_DIV2,
            χ_2: TAU_DIV2,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Met {
    pub χ_1: f64,
    pub χ_2: f64,
    pub χ_3: f64,
}

impl Default for Met {
    fn default() -> Self {
        Self {
            χ_1: TAU_DIV2,
            χ_2: TAU_DIV2,
            χ_3: TAU_DIV2,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Phe {
    pub χ_1: f64,
    pub χ_2: f64,
}

impl Default for Phe {
    fn default() -> Self {
        Self {
            χ_1: TAU_DIV2,
            χ_2: TAU_DIV2,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Tyr {
    pub χ_1: f64,
    pub χ_2: f64,
}

impl Default for Tyr {
    fn default() -> Self {
        Self {
            χ_1: TAU_DIV2,
            χ_2: TAU_DIV2,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Trp {
    pub χ_1: f64,
    pub χ_2: f64,
    /// χ_3 is the rotation angle between the 2 rings.
    pub χ_3: f64,
}

impl Default for Trp {
    fn default() -> Self {
        Self {
            χ_1: TAU_DIV2,
            χ_2: TAU_DIV2,
            // todo: Make sure this is anchored correctly; it may
            // todo behave differently from single-atom angles.
            χ_3: TAU_DIV2,
        }
    }
}
