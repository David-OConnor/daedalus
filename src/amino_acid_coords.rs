//! This module contains coordinates for Amino Acids

use na_seq::AminoAcid;
use crate::molecule::Atom;

/// Atom coords are relative.
pub fn atoms(aa: &AminoAcid) -> Vec<Atom> {
    // todo: Move to specific branch with capacity.
    let mut result = Vec::new();

    match aa {
        AminoAcid::Arg => {

        }
        AminoAcid::His => {

        }
        AminoAcid::Lys => {

        }
        AminoAcid::Asp => {

        }
        AminoAcid::Glu => {

        }
        AminoAcid::Ser => {

        }
        AminoAcid::Thr => {

        }
        AminoAcid::Thr => {

        }
        AminoAcid::Asn => {

        }
        AminoAcid::Gln => {

        }
        AminoAcid::Cys => {

        }
        AminoAcid::Sec => {

        }

        // Gly,
        // Pro,
        // Ala,
        // Val,
        // Ile,
        // Leu,
        // Met,
        // Phe,
        // Tyr,
        // Trp,
        _ => ()
    }


    result
}