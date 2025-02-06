//! This module contains coordinates for Amino Acids

use lin_alg::f64::Vec3;
use na_seq::AminoAcid;

use crate::molecule::{AaRole, Atom};

// / //// Perhaps consider a standard
// pub struct AtomAa {
//     /// Relative
//     pub coords: Vec3,
//     pub role: AaRole,
// }

/// Atom coords are relative.
/// Peerhaps, consider a standard where a certain role. (anchor C'??) is at the origin,
/// and the backbone atom next to it is always directly (up, fwd etc) from it.
// pub fn atoms(aa: &AminoAcid) -> Vec<AtomAa> {
pub fn atoms(aa: &AminoAcid) -> Vec<Atom> {
    match aa {
        AminoAcid::Arg => vec![],
        AminoAcid::His => vec![],
        AminoAcid::Lys => vec![],
        AminoAcid::Asp => vec![],
        AminoAcid::Glu => vec![],
        AminoAcid::Ser => vec![],
        AminoAcid::Thr => vec![],
        AminoAcid::Asn => vec![],
        AminoAcid::Gln => vec![],
        AminoAcid::Cys => vec![],
        AminoAcid::Sec => vec![],
        AminoAcid::Gly => vec![],
        AminoAcid::Pro => vec![],
        AminoAcid::Ala => vec![],
        AminoAcid::Val => vec![],
        AminoAcid::Ile => vec![],
        AminoAcid::Leu => vec![],
        AminoAcid::Met => vec![],
        AminoAcid::Phe => vec![],
        AminoAcid::Tyr => vec![],
        AminoAcid::Trp => vec![],
    }
}
