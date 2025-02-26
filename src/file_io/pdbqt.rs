//! For reading and writing PDBQT (Autodock) files.

use std::{
    fs::File,
    io,
    io::{ErrorKind, Read, Write},
    path::Path,
};

use lin_alg::f64::Vec3;
use na_seq::AaIdent;
use regex::Regex;

use crate::{
    Element,
    molecule::{Atom, Bond, Molecule},
};
use crate::molecule::ResidueType;
// #[derive(Debug, Default)]
// pub struct PdbQt {
//     pub atoms: Vec<Atom>,
//     pub bonds: Vec<Bond>,
// }

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum AutodockType {
    // Standard AutoDock4/Vina atom types
    A,  // Aromatic carbon
    C,  // Aliphatic carbon
    N,  // Nitrogen
    Na, // Nitrogen (acceptor)
    O,  // Oxygen
    Oa, // Oxygen (acceptor)
    S,  // Sulfur
    Sa, // Sulfur (acceptor)
    P,  // Phosphorus
    F,  // Fluorine
    Cl, // Chlorine
    Br, // Bromine
    I,  // Iodine
    Zn,
    Fe,
    Mg,
    Ca,
    Mn,
    Cu,
    Hd,    // Polar hydrogen (hydrogen donor)
    Other, // Fallback for unknown types
    Cb,
    Cd,
    Cd1,
    Cd2,
    Ce1,
    Ce2,
    Cg,
    OG1,
    Og2,
    Oe1,
    Ne2,
}

impl AutodockType {
    pub fn from_str(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "A" => Self::A,
            "C" => Self::C,
            "N" => Self::N,
            "NA" => Self::Na,
            "O" => Self::O,
            "OA" => Self::Oa,
            "S" => Self::S,
            "SA" => Self::Sa,
            "P" => Self::P,
            "F" => Self::F,
            "CL" => Self::Cl,
            "BR" => Self::Br,
            "I" => Self::I,
            "ZN" => Self::Zn,
            "FE" => Self::Fe,
            "MG" => Self::Mg,
            "CA" => Self::Ca,
            "MN" => Self::Mn,
            "CU" => Self::Cu,
            "HD" => Self::Hd,
            "CB" => Self::Cb,
            "CD" => Self::Cd,
            "CD1" => Self::Cd1,
            "CD2" => Self::Cd2,
            "CE1" => Self::Ce1,
            "CE2" => Self::Ce2,
            "CG" => Self::Cg,
            "OG1" => Self::OG1,
            "OG2" => Self::Og2,
            "OE1" => Self::Oe1,
            "NE2" => Self::Ne2,
            _ => Self::Other,
        }
    }

    pub fn to_str(&self) -> String {
        match self {
            Self::A => "A",
            Self::C => "C",
            Self::N => "N",
            Self::Na => "NA",
            Self::O => "O",
            Self::Oa => "OA",
            Self::S => "S",
            Self::Sa => "SA",
            Self::P => "P",
            Self::F => "F",
            Self::Cl => "CL",
            Self::Br => "BR",
            Self::I => "I",
            Self::Zn => "ZN",
            Self::Fe => "FE",
            Self::Mg => "MG",
            Self::Ca => "CA",
            Self::Mn => "MN",
            Self::Cu => "CU",
            Self::Hd => "HD",
            Self::Cb =>"CB",
            Self::Cd =>"CD",
            Self::Cd1 =>"CD1",
            Self::Cd2 =>"CD2",
            Self::Ce1 =>"CE1",
            Self::Ce2 =>"CE2" ,
            Self::Cg =>"CG" ,
            Self::OG1 =>"OG1",
            Self::Og2 =>"OG2",
            Self::Oe1 =>"OE1",
            Self::Ne2 =>"NE2" ,
            Self::Other => "Xx",
        }
            .to_string()
    }
}

/// Helpers for parsing
fn parse_usize(s: &str) -> io::Result<usize> {
    s.parse::<usize>()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid integer"))
}
fn parse_f64(s: &str) -> io::Result<f64> {
    s.parse::<f64>()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid float"))
}
fn parse_optional_f32(s: &str) -> io::Result<Option<f32>> {
    if s.is_empty() {
        Ok(None)
    } else {
        let val = s
            .parse::<f32>()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid float"))?;
        Ok(Some(val))
    }
}

fn guess_atom_name(atom: &Atom) -> String {
    if let Some(ref ad_type) = atom.autodock_type {
        ad_type.to_str()
    } else {
        atom.element.to_letter()
    }
}

impl Molecule {
    /// From PQBQT text, e.g. loaded from a file.
    pub fn from_pdbqt(pdb_text: &str) -> io::Result<Self> {
        let mut result = Self::default();
        let mut atoms = Vec::new();

        let re_name = Regex::new(r"Name\s*=\s*(\S+)").unwrap();

        for line in pdb_text.lines() {
            // pad or skip lines that are too short to safely slice
            if line.len() < 6 {
                continue;
            }

            if let Some(caps) = re_name.captures(line) {
                result.ident = caps[1].to_string();
                continue;
            }

            let record_type = &line[0..6];

            // handle ATOM or HETATM
            if record_type.trim() == "ATOM" || record_type.trim() == "HETATM" {
                // Safely parse fields if line long enough;
                // many PDBQT lines are at least ~80 chars, but always check length.
                let serial_number = parse_usize(&line.get(6..11).unwrap_or("").trim())?;
                let x = parse_f64(&line.get(30..38).unwrap_or("").trim())?;
                let y = parse_f64(&line.get(38..46).unwrap_or("").trim())?;
                let z = parse_f64(&line.get(46..54).unwrap_or("").trim())?;

                // Partial charge (cols 71–76 in many PDBQT variants)
                // The line might not be that long, so check length
                let partial_charge = if line.len() >= 76 {
                    parse_optional_f32(line.get(70..76).unwrap_or("").trim())?
                } else {
                    None
                };

                // AutoDock type (cols 77–78)
                let autodock_type = if line.len() >= 78 {
                    let raw_type = line.get(77..79).unwrap_or("").trim();
                    if raw_type.is_empty() {
                        None
                    } else {
                        Some(AutodockType::from_str(raw_type))
                    }
                } else {
                    None
                };

                // We can guess element if you like from the autodock_type or from the last columns.
                // Here, we default to 'Other' for simplicity.
                let element = Element::Other;

                // Let’s call it ATOM if record is "ATOM  ", else hetero = true
                let hetero = record_type.trim() == "HETATM";

                atoms.push(Atom {
                    serial_number,
                    posit: Vec3 { x, y, z },
                    element,
                    role: None,
                    amino_acid: None,
                    hetero,
                    partial_charge,
                    autodock_type,
                    occupancy: None,
                    temperature_factor: None,
                });
            } else {
                // handle other records if you like, e.g. REMARK, BRANCH, etc.
            }
        }

        // put the atoms in the result
        result.atoms = atoms;

        Ok(result)
    }

    pub fn save_pdbqt(&self, path: &Path, ligand: bool) -> io::Result<()> {
        let mut file = File::create(path)?;

        // Typically you'd end with "END" or "ENDMDL" or so, but not strictly required for many readers.
        if !self.ident.is_empty() {
            writeln!(file, "REMARK  Name = {}", self.ident)?;
        }

        writeln!(file, "REMARK                            x       y       z     vdW  Elec       q    Type")?;
        writeln!(file, "REMARK                         _______ _______ _______ _____ _____    ______ ____")?;

        // Optionally write remarks, ROOT/ENDROOT, etc. here if needed.
        // For each atom:
        for (i, atom) in self.atoms.iter().enumerate() {
            // We'll just do a minimal line. Fill in placeholders for
            // residue name, chain, etc. as you like.

            // Decide record type
            let record_name = if atom.hetero { "HETATM" } else { "ATOM" };

            // Typically we might put:
            // - columns 1..6:   record name
            // - columns 7..11:  serial number
            // - columns 13..16: atom name (we might guess from element or autodock_type)
            // - columns 17..20: residue name (e.g. "LIG" or "UNK")
            // - columns 31..38, 39..46, 47..54: coords
            // - columns 71..76: partial charge
            // - columns 77..78: autodock type

            // For demonstration, let's assume:
            // * "C" as the atom name if unknown
            // * "LIG" as the residue
            // * 'A' chain
            // * serial_number for residue sequence too
            // * occupancy and tempFactor set to 0.00

            let atom_name = guess_atom_name(atom);

            let mut res_num = 1;

            let residue_name = if ligand {
                "UNL".to_owned()
            } else {
                match self.residues.iter().find(|r| r.atoms.contains(&i)) {
                    Some(r) => match &r.res_type {
                        ResidueType::AminoAcid(aa) => {
                            res_num = r.serial_number;
                            aa.to_str(AaIdent::ThreeLetters).to_uppercase()
                        },
                        // todo: Limit to 3 chars?
                        ResidueType::Other(name) => name.clone(),
                        ResidueType::Water => "HOH".to_owned(),
                    }
                    None => "---".to_owned()
                }
            };

            let chain_id = match self.chains.iter().find(|c| c.atoms.contains(&i)) {
                Some(c) => c.id.to_uppercase().chars().next().unwrap(),
                None => 'A'
            };

            // autodock_type or fallback
            let ad_type = atom
                .autodock_type
                .as_ref()
                .unwrap_or(&AutodockType::C)
                .to_str();

            // We'll format columns carefully with fixed widths:
            // (This uses a typical PDB-like fixed column approach.)
            // The example below is one possible format spec. Tweak spacing as needed.
            // Indices are approximate; watch alignment carefully.

            writeln!(
                file,
                "{:<6}{:>5}  {:<2}  {:<3} {:>1}{:>4}    {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}    {:>+6.3} {:<2}",
                record_name,        // columns 1-6
                atom.serial_number, // columns 7-11
                atom_name,          // columns 13-14 or 13-16
                residue_name,       // columns 18-20
                chain_id,           // column 22
                res_num,        // columns 23-26
                atom.posit.x,       // columns 31-38
                atom.posit.y,       // columns 39-46
                atom.posit.z,       // columns 47-54
                atom.occupancy.unwrap_or_default(),          // columns 55-60
                atom.temperature_factor.unwrap_or_default(),        // columns 61-66
                atom.partial_charge.unwrap_or_default(),             // columns 71-76
                ad_type             // columns 77-78
            )?;
        }

        // If your PDBQT format typically has "ENDROOT" after the atoms:
        writeln!(file, "ENDROOT")?;

        writeln!(file, "END")?;

        Ok(())
    }
}

pub fn load_pdbqt(path: &Path) -> io::Result<Molecule> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let data_str: String = String::from_utf8(buffer)
        .map_err(|_| io::Error::new(ErrorKind::InvalidData, "Invalid UTF8"))?;

    Molecule::from_pdbqt(&data_str)
}
