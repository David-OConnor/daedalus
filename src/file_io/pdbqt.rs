//! For reading and writing PDBQT (Autodock) files.
//! [Unofficial, incomplete spec](https://userguide.mdanalysis.org/2.6.0/formats/reference/pdbqt.html)

use std::{
    fs::File,
    io,
    io::{ErrorKind, Read, Write},
    path::Path,
    str::FromStr,
};

use lin_alg::f64::Vec3;
use na_seq::AaIdent;
use regex::Regex;

use crate::{
    bond_inference::{create_bonds, create_hydrogen_bonds},
    docking::docking_prep::{DockType, UnitCellDims},
    element::Element,
    molecule::{Atom, AtomRole, Chain, Ligand, Molecule, Residue, ResidueType},
    util::mol_center_size,
};
// #[derive(Debug, Default)]
// pub struct PdbQt {
//     pub atoms: Vec<Atom>,
//     pub bonds: Vec<Bond>,
// }

/// Helpers for parsing
fn parse_usize(s: &str) -> io::Result<usize> {
    s.parse::<usize>()
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

impl Molecule {
    /// From PQBQT text, e.g. loaded from a file.
    pub fn from_pdbqt(pdb_text: &str) -> io::Result<(Self, Option<Ligand>)> {
        let mut result = Self::default();
        let mut atoms = Vec::new();

        let re_ident = Regex::new(r"Name\s*=\s*(\S+)").unwrap();

        let mut chains: Vec<Chain> = Vec::new();
        let mut residues: Vec<Residue> = Vec::new();

        let mut ligand: Option<Ligand> = None;

        for line in pdb_text.lines() {
            if let Some(caps) = re_ident.captures(line) {
                result.ident = caps[1].to_string();
                continue;
            }

            if line.len() < 6 {
                continue;
            }

            let record_type = line[0..6].trim();

            // todo: Parse Ligand torsions.

            if record_type == "ATOM" || record_type == "HETATM" {
                let serial_number = parse_usize(line[6..11].trim())?;

                let atom_id = atoms.len(); // index for assigning residues and chains.

                let name = line[12..16].trim();

                let element = Element::from_letter(&name[..1]).unwrap_or(Element::Carbon);

                let res_name = line[17..21].trim();
                let residue_type = ResidueType::from_str(&res_name);
                let mut role = None;

                role = match residue_type {
                    ResidueType::AminoAcid(_aa) => Some(AtomRole::from_name(&res_name)),
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
                        residues: Vec::new(), // todo temp
                        atoms: vec![atom_id],
                        visible: true,
                    });
                }

                let res_id = parse_usize(line[22..26].trim()).unwrap_or_default() as isize;
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
                        serial_number: 0, // todo temp
                        res_type: residue_type.clone(),
                        atoms: vec![atom_id],
                        dihedral: None,
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

                atoms.push(Atom {
                    serial_number,
                    posit: Vec3 { x, y, z },
                    element,
                    name: name.to_owned(),
                    role,
                    residue_type,
                    hetero,
                    occupancy,
                    temperature_factor,
                    partial_charge,
                    dock_type,
                });
            } else if record_type == "CRYST1" {
                let unit_cell_dims = UnitCellDims {
                    a: parse_f32(line[6..15].trim()).unwrap_or_default(),
                    b: parse_f32(line[15..24].trim()).unwrap_or_default(),
                    c: parse_f32(line[24..33].trim()).unwrap_or_default(),
                    alpha: parse_f32(line[33..40].trim()).unwrap_or_default(),
                    beta: parse_f32(line[40..47].trim()).unwrap_or_default(),
                    gamma: parse_f32(line[47..54].trim()).unwrap_or_default(),
                };

                if ligand.is_none() {
                    ligand = Some(Default::default());
                }

                ligand.as_mut().unwrap().unit_cell_dims = unit_cell_dims;

                // todo: What to do with this?
            } else {
                // handle other records if you like, e.g. REMARK, BRANCH, etc.
            }
        }

        let (center, size) = mol_center_size(&atoms);

        // put the atoms in the result
        result.atoms = atoms;
        result.chains = chains;
        result.residues = residues;
        result.center = center;
        result.size = size;

        result.populate_hydrogens_angles();
        result.bonds = create_bonds(&result.atoms);
        result.bonds.extend(create_hydrogen_bonds(&result.atoms));

        // todo: ligand molecule??

        Ok((result, ligand))
    }

    pub fn save_pdbqt(&self, path: &Path, ligand: Option<&Ligand>) -> io::Result<()> {
        let mut file = File::create(path)?;

        // Typically you'd end with "END" or "ENDMDL" or so, but not strictly required for many readers.
        if !self.ident.is_empty() {
            writeln!(file, "REMARK  Name = {}", self.ident)?;
        }

        if ligand.is_some() {
            let torsions = &ligand.as_ref().unwrap().torsions;
            let tor_len = torsions.len();
            if tor_len > 0 {
                writeln!(file, "REMARK  {tor_len} active torsions:")?;
                writeln!(file, "REMARK  status: ('A' for Active; 'I' for Inactive)")?;
            }

            for (i, torsion) in torsions.iter().enumerate() {
                writeln!(
                    file,
                    "REMARK {:>4}  {:>1}    between atoms: {}  and  {}",
                    i + 1,
                    torsion.status,
                    torsion.atom_0,
                    torsion.atom_1
                )?;
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
            if let Some(role) = atom.role {
                if role == AtomRole::Water {
                    // Skipping water in the context of Docking prep, which is where we expect
                    // PDBQT files to be used.
                    continue;
                }
            }

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

            let residue_name = if ligand.is_some() {
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

pub fn load_pdbqt(path: &Path) -> io::Result<(Molecule, Option<Ligand>)> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let data_str: String = String::from_utf8(buffer)
        .map_err(|_| io::Error::new(ErrorKind::InvalidData, "Invalid UTF8"))?;

    Molecule::from_pdbqt(&data_str)
}
