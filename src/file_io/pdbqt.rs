//! For reading and writing PDBQT (Autodock) files.

use std::{
    fs::File,
    io,
    io::{ErrorKind, Read, Write},
    path::Path,
    str::FromStr,
};

use lin_alg::f64::Vec3;
use na_seq::{AaIdent, AminoAcid};
use regex::Regex;

use crate::{
    Element,
    bond_inference::{create_bonds, make_hydrogen_bonds},
    molecule::{Atom, AtomRole, Bond, Chain, Molecule, Residue, ResidueType},
    util::mol_center_size,
};
use crate::docking::docking_prep::{Torsion, TorsionStatus};
// #[derive(Debug, Default)]
// pub struct PdbQt {
//     pub atoms: Vec<Atom>,
//     pub bonds: Vec<Bond>,
// }

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum DockType {
    // Standard AutoDock4/Vina atom types
    A, // Aromatic carbon
    C, // Aliphatic carbon
    // Cb,
    // Cd,
    // Cd1,
    // Cd2,
    // Ce,
    // Ce1,
    // Ce2,
    // Cg,
    // Cg1,
    // Cg2,
    // Cz,
    N,  // Nitrogen
    Na, // Nitrogen (acceptor)
    O,  // Oxygen
    // OG,
    // OG1,
    // Og2,
    // Oe1,
    Oh,
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
    Hd, // Polar hydrogen (hydrogen donor)
    Other, // Fallback for unknown types
        // Ne2,
        // Sd,
}

impl DockType {
    pub fn from_str(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "A" => Self::A,
            "C" => Self::C,
            // "CB" => Self::Cb,
            // "CD" => Self::Cd,
            // "CD1" => Self::Cd1,
            // "CD2" => Self::Cd2,
            // "CE" => Self::Ce,
            // "CE1" => Self::Ce1,
            // "CE2" => Self::Ce2,
            // "CG" => Self::Cg,
            // "CG1" => Self::Cg1,
            // "CG2" => Self::Cg2,
            // "CZ" => Self::Cz,
            "N" => Self::N,
            "NA" => Self::Na,
            "O" => Self::O,
            "OA" => Self::Oa,
            // "OG" => Self::OG,
            // "OG1" => Self::OG1,
            // "OG2" => Self::Og2,
            "OH" => Self::Oh,
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
            // "OE1" => Self::Oe1,
            // "NE2" => Self::Ne2,
            // "SD" => Self::Sd,
            _ => Self::Other,
        }
    }

    pub fn to_str(&self) -> String {
        match self {
            Self::A => "A",
            Self::C => "C",
            // Self::Cb =>"CB",
            // Self::Cd =>"CD",
            // Self::Cd1 =>"CD1",
            // Self::Cd2 =>"CD2",
            // Self::Ce =>"CE",
            // Self::Ce1 =>"CE1",
            // Self::Ce2 =>"CE2" ,
            // Self::Cg =>"CG" ,
            // Self::Cg1 =>"CG1",
            // Self::Cg2 =>"CG2" ,
            // Self::Cz =>"CZ" ,
            Self::N => "N",
            Self::Na => "NA",
            Self::O => "O",
            Self::Oa => "OA",
            // Self::OG =>"OG",
            // Self::OG1 =>"OG1",
            // Self::Og2 =>"OG2",
            Self::Oh => "OH",
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
            // Self::Oe1 =>"OE1",
            // Self::Ne2 =>"NE2" ,
            // Self::Sd =>"SD" ,
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


impl Molecule {
    /// From PQBQT text, e.g. loaded from a file.
    pub fn from_pdbqt(pdb_text: &str) -> io::Result<Self> {
        let mut result = Self::default();
        let mut atoms = Vec::new();

        let re_ident = Regex::new(r"Name\s*=\s*(\S+)").unwrap();

        let mut chains: Vec<Chain> = Vec::new();
        let mut residues: Vec<Residue> = Vec::new();

        for line in pdb_text.lines() {
            if let Some(caps) = re_ident.captures(line) {
                result.ident = caps[1].to_string();
                continue;
            }

            let cols: Vec<String> = line.split_whitespace().map(|s| s.to_string()).collect();

            if cols.len() < 9 {
                continue;
            }

            let record_type = &cols[0];

            if record_type.trim() == "ATOM" || record_type.trim() == "HETATM" {
                // Safely parse fields if line long enough;
                // todo: This is probably fragile. Split by spaces (column) instead.

                let serial_number = parse_usize(&cols[1])?;
                let atom_id = atoms.len(); // index.

                let element = Element::from_letter(&cols[2][..1]).unwrap_or(Element::Carbon);

                let name = cols[2].clone();

                let res_name = cols[3].clone();
                let residue_type = ResidueType::from_str(&res_name);
                let mut role = None;

                role = match residue_type {
                    ResidueType::AminoAcid(_aa) => Some(AtomRole::from_name(&res_name)),
                    ResidueType::Water => Some(AtomRole::Water),
                    _ => None,
                };

                let chain_id = cols[4].clone();
                let mut chain_found = false;
                for chain in &mut chains {
                    if chain.id == chain_id {
                        chain.atoms.push(atom_id);
                        chain_found = true;
                        break;
                    }
                }
                if !chain_found {
                    chains.push(Chain {
                        id: chain_id,
                        residues: Vec::new(), // todo temp
                        atoms: vec![atom_id],
                        visible: true,
                    });
                }

                let res_id = parse_usize(&cols[5]).unwrap_or_default() as isize;
                let mut res_found = false;
                for res in &mut residues {
                    if res.serial_number == res_id {
                        res.atoms.push(atom_id);
                        res_found = true;
                        break;
                    }
                }
                if !res_found {
                    residues.push(Residue {
                        serial_number: 0,                       // todo temp
                        res_type: ResidueType::Other(res_name), // todo temp
                        atoms: vec![atom_id],
                    });
                }

                let x = parse_f64(&cols[6])?;
                let y = parse_f64(&cols[7])?;
                let z = parse_f64(&cols[8])?;

                let partial_charge = parse_optional_f32(&cols[11])?;

                let dock_type = if cols.len() >= 13 {
                    Some(DockType::from_str(&cols[12]))
                } else {
                    None
                };

                let hetero = record_type.trim() == "HETATM";

                atoms.push(Atom {
                    serial_number,
                    posit: Vec3 { x, y, z },
                    element,
                    name,
                    role,
                    residue_type,
                    hetero,
                    partial_charge,
                    dock_type,       // todo: col 9?
                    occupancy: None, // todo: Col 10?
                    temperature_factor: None,
                });
            } else {
                // handle other records if you like, e.g. REMARK, BRANCH, etc.
            }
        }

        let mut bonds = create_bonds(&atoms);
        bonds.extend(make_hydrogen_bonds(&atoms));
        let (center, size) = mol_center_size(&atoms);

        // put the atoms in the result
        result.atoms = atoms;
        result.chains = chains;
        result.residues = residues;
        result.bonds = bonds;
        result.center = center;
        result.size = size;

        Ok(result)
    }

    pub fn save_pdbqt(&self, path: &Path, ligand: bool) -> io::Result<()> {
        let mut file = File::create(path)?;

        // Typically you'd end with "END" or "ENDMDL" or so, but not strictly required for many readers.
        if !self.ident.is_empty() {
            writeln!(file, "REMARK  Name = {}", self.ident)?;
        }


        if ligand {
            // todo: Temp! Use ligand field.
            let mut torsions: Vec<Torsion> = vec![Torsion {
                status: TorsionStatus::Active,
                atom_0: "O_1".to_string(),
                atom_1: "C_18".to_string(),
            }];
            let tor_len = torsions.len();
            if tor_len > 0 {
                writeln!(file, "REMARK  {tor_len} active torsions:")?;
                writeln!(file, "REMARK  status: ('A' for Active; 'I' for Inactive)")?;

            }

            for (i, torsion) in torsions.iter().enumerate() {
                writeln!(file, "REMARK {:>4}  {:>1}    between atoms: {}  and  {}", i + 1, torsion.status, torsion.atom_0, torsion.atom_1)?;
            }
        }

        writeln!(
            file,
            "REMARK                            x       y       z     vdW  Elec       q    Type"
        )?;
        writeln!(
            file,
            "REMARK                         _______ _______ _______ _____ _____    ______ ____"
        )?;

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

            let mut res_num = 1;

            let residue_name = if ligand {
                "UNL".to_owned()
            } else {
                match self.residues.iter().find(|r| r.atoms.contains(&i)) {
                    Some(r) => match &r.res_type {
                        ResidueType::AminoAcid(aa) => {
                            res_num = r.serial_number;
                            aa.to_str(AaIdent::ThreeLetters).to_uppercase()
                        }
                        // todo: Limit to 3 chars?
                        ResidueType::Other(name) => name.clone(),
                        ResidueType::Water => "HOH".to_owned(),
                    },
                    None => "---".to_owned(),
                }
            };

            let chain_id = match self.chains.iter().find(|c| c.atoms.contains(&i)) {
                Some(c) => c.id.to_uppercase().chars().next().unwrap(),
                None => 'A',
            };

            // autodock_type or fallback
            let mut dock_type = String::new();
            if let Some(dt) = atom.dock_type {
                dock_type = dt.to_str();
            }

            // We'll format columns carefully with fixed widths:
            // (This uses a typical PDB-like fixed column approach.)
            // The example below is one possible format spec. Tweak spacing as needed.
            // Indices are approximate; watch alignment carefully.

            writeln!(
                file,
                "{:<6}{:>5}  {:<3} {:<3} {:>1}{:>4}    {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}    {:>+6.3} {:<2}",
                record_name,                                 // columns 1-6
                atom.serial_number,                          // columns 7-11
                atom.name,                                   // columns 13-14 or 13-16
                residue_name,                                // columns 18-20
                chain_id,                                    // column 22
                res_num,                                     // columns 23-26
                atom.posit.x,                                // columns 31-38
                atom.posit.y,                                // columns 39-46
                atom.posit.z,                                // columns 47-54
                atom.occupancy.unwrap_or_default(),          // columns 55-60
                atom.temperature_factor.unwrap_or_default(), // columns 61-66
                atom.partial_charge.unwrap_or_default(),     // columns 71-76
                dock_type                                    // columns 77-78
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
