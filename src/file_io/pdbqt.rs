//! For reading and writing PDBQT (Autodock) files.
//! [Unofficial, incomplete spec](https://userguide.mdanalysis.org/2.6.0/formats/reference/pdbqt.html)

use std::{
    fmt::Display,
    fs::File,
    io,
    io::{ErrorKind, Read, Write},
    path::Path,
    str::FromStr,
};

use bio_files::{AtomGeneric, BondGeneric, ChargeType, Mol2, MolType, ResidueType};
use lin_alg::f64::Vec3;
use na_seq::{AaIdent, AtomTypeInRes, Element};
use regex::Regex;

use crate::{
    docking::{
        ConformationType,
        prep::{DockType, UnitCellDims},
    },
    molecule::{Atom, AtomRole, Chain, Ligand, MoleculeSmall, Residue, ResidueEnd},
};

#[derive(Clone, Copy, Debug, PartialEq)]
enum TorsionStatus {
    Active,
    Inactive,
}

impl Display for TorsionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match self {
            Self::Active => "A".to_string(),
            Self::Inactive => "I".to_string(),
        };
        write!(f, "{}", str)
    }
}

/// Helpers for parsing
fn parse_usize(s: &str) -> io::Result<usize> {
    s.parse::<usize>()
        .map_err(|_| io::Error::new(ErrorKind::InvalidData, "Invalid integer"))
}
fn parse_u32(s: &str) -> io::Result<u32> {
    s.parse::<u32>()
        .map_err(|_| io::Error::new(ErrorKind::InvalidData, "Invalid integer"))
}
fn parse_f64(s: &str) -> io::Result<f64> {
    s.parse::<f64>()
        .map_err(|_| io::Error::new(ErrorKind::InvalidData, "Invalid float"))
}

fn parse_f32(s: &str) -> io::Result<f32> {
    s.parse::<f32>()
        .map_err(|_| io::Error::new(ErrorKind::InvalidData, "Invalid float"))
}

fn parse_optional_f32(s: &str) -> io::Result<Option<f32>> {
    if s.is_empty() {
        Ok(None)
    } else {
        let val = s
            .parse::<f32>()
            .map_err(|_| io::Error::new(ErrorKind::InvalidData, "Invalid float"))?;
        Ok(Some(val))
    }
}

/// From PQBQT text, e.g. loaded from a file.
/// We use the Mol2 structure here, as it's similar enough.
pub fn open_pdbqt(pdb_text: &str) -> io::Result<Mol2> {
    let mut atoms = Vec::new();

    let re_ident = Regex::new(r"Name\s*=\s*(\S+)").unwrap();

    let mut chains: Vec<Chain> = Vec::new();
    let mut residues: Vec<Residue> = Vec::new();

    let mut lig: Option<Ligand> = None;

    let mut ident = String::new();

    for line in pdb_text.lines() {
        if let Some(caps) = re_ident.captures(line) {
            ident = caps[1].to_string();
            continue;
        }

        if line.len() < 6 {
            continue;
        }

        let record_type = line[0..6].trim();

        // todo: Parse Ligand torsions.

        if record_type == "ATOM" || record_type == "HETATM" {
            let serial_number = parse_u32(line[6..11].trim())?;

            let atom_id = atoms.len(); // index for assigning residues and chains.

            let name = line[12..16].trim();

            let element = Element::from_letter(&name[..1]).unwrap_or(Element::Carbon);

            let res_name = line[17..21].trim();
            let residue_type = ResidueType::from_str(res_name);

            let type_in_res = AtomTypeInRes::from_str(name)?;

            let _role = match residue_type {
                ResidueType::AminoAcid(_aa) => Some(AtomRole::from_type_in_res(&type_in_res)),
                ResidueType::Water => Some(AtomRole::Water),
                _ => None,
            };

            let chain_id = line[21..22].trim();
            let mut chain_found = false;
            for chain in &mut chains {
                if chain.id == *chain_id {
                    chain.atoms.push(atom_id);
                    chain_found = true;
                    break;
                }
            }
            if !chain_found {
                chains.push(Chain {
                    id: chain_id.to_string(),
                    residue_sns: Vec::new(), // todo temp
                    residues: Vec::new(),    // todo temp
                    atom_sns: vec![atom_id as u32],
                    atoms: vec![atom_id],
                    visible: true,
                });
            }

            let serial_number = parse_usize(line[22..26].trim()).unwrap_or_default() as u32;
            let mut res_found = false;
            for res in &mut residues {
                if res.serial_number == serial_number {
                    res.atoms.push(atom_id);
                    res_found = true;
                    break;
                }
            }
            if !res_found {
                residues.push(Residue {
                    serial_number: 0, // todo temp
                    res_type: residue_type.clone(),
                    atom_sns: vec![atom_id as u32],
                    atoms: vec![atom_id],
                    dihedral: None,
                    end: ResidueEnd::Hetero,
                });
            }

            let x = parse_f64(line[30..38].trim())?;
            let y = parse_f64(line[38..46].trim())?;
            let z = parse_f64(line[46..54].trim())?;

            let occupancy = parse_optional_f32(line[54..60].trim())?;
            let temperature_factor = parse_optional_f32(line[60..66].trim())?;
            // Gasteiger PEOE partial charge q.
            let partial_charge = parse_optional_f32(line[66..76].trim())?;

            // todo: May need to take into account lines of len 78 and 79: Is this leaving out single-letter ones?
            let dock_type = if line.len() < 80 {
                None
            } else {
                Some(DockType::from_str(line[78..80].trim()))
            };

            let hetero = record_type == "HETATM";

            atoms.push(AtomGeneric {
                serial_number,
                posit: Vec3 { x, y, z },
                element,
                type_in_res: Some(type_in_res),
                // role,
                // residue: None,
                // chain: None,
                hetero,
                occupancy,
                // temperature_factor,
                partial_charge,
                force_field_type: None,
                // dock_type,
            });

            // atoms.push(Atom {
            //     serial_number,
            //     posit: Vec3 { x, y, z },
            //     element,
            //     type_in_res: Some(type_in_res),
            //     role,
            //     residue: None,
            //     chain: None,
            //     hetero,
            //     occupancy,
            //     temperature_factor,
            //     partial_charge,
            //     force_field_type: None,
            //     dock_type,
            // });
        } else if record_type == "CRYST1" {
            if lig.is_none() {
                lig = Some(Default::default());
            }
            // todo: What to do with this?
        } else {
            // handle other records if you like, e.g. REMARK, BRANCH, etc.
        }
    }

    // todo: Handle bonds. Are they in the file, or should we infer them?
    let bonds = Vec::new();

    Ok(Mol2 {
        ident,
        mol_type: MolType::Small,
        // todo
        charge_type: ChargeType::Amber, // todo
        comment: None,                  // todo
        atoms,
        bonds,
    })
}

pub fn save_pdbqt(mol: &Mol2, path: &Path, ligand: Option<&Ligand>) -> io::Result<()> {
    let mut file = File::create(path)?;

    // Typically you'd end with "END" or "ENDMDL" or so, but not strictly required for many readers.
    if !mol.ident.is_empty() {
        writeln!(file, "REMARK  Name = {}", mol.ident)?;
    }

    if let Some(lig) = ligand {
        if let ConformationType::AssignedTorsions { torsions } = &lig.pose.conformation_type {
            let tor_len = torsions.len();
            if tor_len > 0 {
                writeln!(file, "REMARK  {tor_len} active torsions:")?;
                writeln!(file, "REMARK  status: ('A' for Active; 'I' for Inactive)")?;
            }

            for (i, torsion) in torsions.iter().enumerate() {
                let atom_0_i = lig.common.bonds[torsion.bond].atom_0;
                let atom_1_i = lig.common.bonds[torsion.bond].atom_1;
                let atom_0 = &lig.common.atoms[atom_0_i];
                let atom_1 = &lig.common.atoms[atom_1_i];

                let atom_0_text = format!("{}_{}", atom_0.element.to_letter(), atom_0_i);
                let atom_1_text = format!("{}_{}", atom_1.element.to_letter(), atom_1_i);

                writeln!(
                    file,
                    "REMARK {:>4}  {:>1}    between atoms: {}  and  {}",
                    i + 1,
                    TorsionStatus::Active,
                    atom_0_text,
                    atom_1_text,
                )?;
            }
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
    for (i, atom) in mol.atoms.iter().enumerate() {
        // todo: A/R
        // if let Some(role) = atom.role {
        //     // if role == AtomRole::Water || atom.element == Element::Hydrogen {
        //     if role == AtomRole::Water {
        //         // Skipping water in the context of Docking prep, which is where we expect
        //         // PDBQT files to be used.
        //         continue;
        //     }
        // }

        // Decide record type
        let record_name = if atom.hetero { "HETATM" } else { "ATOM" };

        // - columns 1..6:   record name
        // - columns 7..11:  serial number
        // - columns 13..16: atom name (we might guess from element or autodock_type)
        // - columns 17..20: residue name (e.g. "LIG" or "UNK")
        // - columns 31..38, 39..46, 47..54: coords
        // - columns 71..76: partial charge
        // - columns 77..78: autodock type

        let mut res_num = 1;

        let residue_name = if ligand.is_some() {
            "UNL".to_owned()
        } else {
            let mut text = "---".to_owned();

            //todo: A/R
            // if let Some(res_i) = atom.residue {
            //     let res = &mol.residues[res_i];
            //
            //     text = match &res.res_type {
            //         ResidueType::AminoAcid(aa) => {
            //             res_num = res.serial_number;
            //             aa.to_str(AaIdent::ThreeLetters).to_uppercase()
            //         }
            //         // todo: Limit to 3 chars?
            //         ResidueType::Other(name) => name.clone(),
            //         ResidueType::Water => "HOH".to_owned(),
            //     };
            // }
            text
        };

        // todo: A/R
        // let chain_id = match mol.chains.iter().find(|c| c.atoms.contains(&i)) {
        //     Some(c) => c.id.to_uppercase().chars().next().unwrap(),
        //     None => 'A',
        // };

        //todo: A/R
        // let mut dock_type = String::new();
        // if let Some(dt) = atom.dock_type {
        //     dock_type = dt.to_str();
        // }

        let name = match &atom.type_in_res {
            Some(name) => name.to_string(),
            None => atom.element.to_letter(),
        };

        // todo temp
        let chain_id = "A".to_string();
        let dock_type = " ".to_string();
        let temperature_factor = " ".to_string();

        writeln!(
            file,
            "{:<6}{:>5}  {:<3} {:<3} {:>1}{:>4}    {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}    {:>+6.3} {:<2}",
            record_name,                             // columns 1-6
            atom.serial_number,                      // columns 7-11
            name,                                    // columns 13-14 or 13-16
            residue_name,                            // columns 18-20
            chain_id,                                // column 22
            res_num,                                 // columns 23-26
            atom.posit.x,                            // columns 31-38
            atom.posit.y,                            // columns 39-46
            atom.posit.z,                            // columns 47-54
            atom.occupancy.unwrap_or_default(),      // columns 55-60
            temperature_factor,                      // columns 61-66
            atom.partial_charge.unwrap_or_default(), // columns 71-76
            dock_type                                // columns 77-78
        )?;
    }

    // todo?
    // writeln!(file, "ENDROOT")?;
    // writeln!(file, "END")?;

    Ok(())
}

pub fn load_pdbqt(path: &Path) -> io::Result<Mol2> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let data_str: String = String::from_utf8(buffer)
        .map_err(|_| io::Error::new(ErrorKind::InvalidData, "Invalid UTF8"))?;

    open_pdbqt(&data_str)
}
