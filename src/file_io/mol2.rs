//! For opening Mol2 files. These are common molecular descriptions for ligands.
//! [This unofficial resource](https://chemicbook.com/2021/02/20/mol2-file-format-explained-for-beginners-part-2.html)
//! descripts the format.

use std::{
    collections::HashMap,
    fs::File,
    io,
    io::{ErrorKind, Read, Write},
    path::Path,
};

use lin_alg::f64::Vec3;

use crate::{
    element::Element,
    molecule::{Atom, Bond, BondCount, BondType, Molecule, ResidueType},
};

#[derive(Clone, Copy, PartialEq)]
enum MolType {
    Small,
    Bipolymer,
    Protein,
    NucleicAcid,
    Saccharide,
}

impl MolType {
    pub fn to_str(&self) -> String {
        match self {
            Self::Small => "SMALL",
            Self::Bipolymer => "BIPOLYMER",
            Self::Protein => "PROTEIN",
            Self::NucleicAcid => "NUCLEIC_ACID",
            Self::Saccharide => "SACCHARIDE",
        }
        .to_owned()
    }
}

#[derive(Clone, Copy, PartialEq)]
enum ChargeType {
    None,
    DelRe,
    Gasteiger,
    GastHuck,
    Huckel,
    Pullman,
    Gauss80,
    Ampac,
    Mulliken,
    Dict,
    MmFf94,
    User,
}

impl ChargeType {
    pub fn to_str(&self) -> String {
        match self {
            Self::None => "NO_CHARGES",
            Self::DelRe => "DEL_RE",
            Self::Gasteiger => "GASTEIGER",
            Self::GastHuck => "GAST_HUCK",
            Self::Huckel => "HUCKEL",
            Self::Pullman => "PULLMAN",
            Self::Gauss80 => "GAUSS80_CHARGES",
            Self::Ampac => "AMPAC_CHARGES",
            Self::Mulliken => "MULLIKEN_CHARGES",
            Self::Dict => "DICT_CHARGES",
            Self::MmFf94 => "MMFF94_CHARGES",
            Self::User => "USER_CHARGES",
        }
        .to_owned()
    }
}

struct Mol2 {
    pub molecule: Molecule,
    /// These fields aren't universal to the format.
    pub metadata: HashMap<String, String>,
}

impl Molecule {
    /// From a string of a CIF or PDB text file.
    pub fn from_mol2(sdf_text: &str) -> io::Result<Self> {
        let lines: Vec<&str> = sdf_text.lines().collect();

        // Example Mol2 header:
        // "
        // @<TRIPOS>MOLECULE
        // 5287969
        // 48 51
        // SMALL
        // USER_CHARGES
        // ****
        // Charges calculated by ChargeFW2 0.1, method: SQE+qp
        // @<TRIPOS>ATOM
        // "

        if lines.len() < 5 {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "Not enough lines to parse a MOL2 header",
            ));
        }

        let mut atoms = Vec::new();
        let mut bonds = Vec::new();

        let mut in_atom_section = false;
        let mut in_bond_section = false;

        for line in &lines {
            let upper = line.to_uppercase();
            if upper.contains("<TRIPOS>ATOM") {
                in_atom_section = true;
                in_bond_section = false;
                continue;
            }

            if upper.contains("<TRIPOS>BOND") {
                in_atom_section = false;
                in_bond_section = true;
                continue;
            }

            if upper.contains("@<TRIPOS>SUBSTRUCTURE") {
                // todo: As required. Example:
                //    1 SER     2 RESIDUE           4 A     SER     1 ROOT
                //      2 VAL    13 RESIDUE           4 A     VAL     2
                //      3 PRO    29 RESIDUE           4 A     PRO     2
                in_atom_section = false;
                in_bond_section = false;
                continue;
            }

            if upper.contains("@<TRIPOS>SET") {
                // todo: As required. Example:
                // ANCHOR          STATIC     ATOMS    <user>   **** Anchor Atom Set
                // 63 127 1110 128 129 610 130 131 132 133 134 740 135 53 741 54 55 617 1482 612 57 58 1485 60 1487 1488 742 1489 743 59 614 1075 611 1486 1076 1481 1077 613 1078 1079 615 616 744 1081 56 618 61 745 1080 1483 738 1074 739 1103 746 1104 1484 1105 1106 1107 1108 1109 1102 1082
                // RIGID           STATIC     BONDS    <user>   **** Rigid Bond Set
                // 56 280 58 59 281 60 61 62 63 64 65 671 672 673 674 332 675 676 484 485 677 678 284 333 24 480 481 26 282 482 334 483 28 486 283 30 31 285 335 487 286 337 338 336 493 494 495 496 27 497 25 499 500 331 279 498 29
                in_atom_section = false;
                in_bond_section = false;
                continue;
            }

            // atom_id atom_name x y z atom_type [subst_id[subst_name [charge [status_bit]]]]
            // Where:
            //
            // atom_id (integer): The ID number of the atom at the time the file was created. This is provided for reference only and is not used when the .mol2 file is read into any mol2 parser software
            // atom_name (string): The name of the atom
            // x (real): The x coordinate of the atom
            // y (real): The y coordinate of the atom
            // z (real): The z coordinate of the atom
            // atom_type (string): The SYBYL atom type for the atom
            // subst_id (integer): The ID number of the substructure containing the atom
            // subst_name (string): The name of the substructure containing the atom
            // charge (real): The charge associated with the atom
            // status_bit (string): The internal SYBYL status bits associated with the atom. These should never be set by the user. Valid status bits are DSPMOD, TYPECOL, CAP, BACKBONE, DICT, ESSENTIAL, WATER, and DIRECT

            if in_atom_section {
                let cols: Vec<&str> = line.split_whitespace().collect();

                let serial_number = cols[0].parse::<usize>().map_err(|_| {
                    io::Error::new(ErrorKind::InvalidData, "Could not parse serial number")
                })?;

                // Col 1: e.g. "H", "HG22" etc. Col 5: "C.3", "N.p13" etc.
                let mut elem_txt = cols[5].to_owned();
                if let Some((before_dot, _after_dot)) = elem_txt.split_once('.') {
                    elem_txt = before_dot.to_string();
                }

                let element = Element::from_letter(&elem_txt)?;

                let x = cols[2].parse::<f64>().map_err(|_| {
                    io::Error::new(ErrorKind::InvalidData, "Could not parse X coordinate")
                })?;
                let y = cols[3].parse::<f64>().map_err(|_| {
                    io::Error::new(ErrorKind::InvalidData, "Could not parse Y coordinate")
                })?;
                let z = cols[4].parse::<f64>().map_err(|_| {
                    io::Error::new(ErrorKind::InvalidData, "Could not parse Z coordinate")
                })?;

                let charge = cols[8].parse::<f32>().unwrap_or_default();

                // todo: ALso, parse the charge type at line 4.
                let partial_charge = if charge.abs() < 0.000001 {
                    None
                } else {
                    Some(charge)
                };

                // todo: More columns, including partial charge.

                atoms.push(Atom {
                    serial_number,
                    posit: Vec3 { x, y, z }, // or however you store coordinates
                    element,
                    name: String::new(),
                    role: None,
                    residue_type: ResidueType::Other(String::new()), // Not available in SDF.
                    hetero: false,
                    occupancy: None,
                    temperature_factor: None,
                    partial_charge,
                    dock_type: None,
                });
            }

            if in_bond_section {
                let cols: Vec<&str> = line.split_whitespace().collect();

                let atom_0 = cols[1].parse::<usize>().map_err(|_| {
                    io::Error::new(
                        ErrorKind::InvalidData,
                        format!("Could not parse atom 0 in bond: {}", cols[1]),
                    )
                })?;

                let atom_1 = cols[2].parse::<usize>().map_err(|_| {
                    io::Error::new(
                        ErrorKind::InvalidData,
                        format!("Could not parse atom 1 in bond: {}", cols[2]),
                    )
                })?;

                // 1 = single
                // 2 = double
                // 3 = triple
                // am = amide
                // ar = aromatic
                // du = dummy
                // un = unknown (cannot be determined from the parameter tables)
                // nc = not connected
                bonds.push(Bond {
                    bond_type: BondType::Covalent {
                        count: BondCount::from_str(cols[3]),
                    },
                    // Our bonds are by index; these are by serial number. This should align them in most cases.
                    // todo: Map serial num to index incase these don't ascend by one.
                    atom_0: atom_0 - 1,
                    atom_1: atom_1 - 1,
                    is_backbone: false,
                });
            }
        }

        // Note: This may not be the identifier we think of.
        let ident = lines[1].to_owned();

        let mut result = Molecule::new(ident, atoms, Vec::new(), Vec::new(), None, None);

        // This replaces the built-in bond computation with our own. Ideally, we don't even calculate
        // those for performance reasons.
        result.bonds = bonds;
        result.bonds_hydrogen = Vec::new();
        result.adjacency_list = result.build_adjacency_list();

        Ok(result)
    }

    pub fn save_mol2(&self, path: &Path) -> io::Result<()> {
        //todo: Fix this so it outputs mol2 instead of sdf.
        let mut file = File::create(path)?;

        // There is a subtlety here. Add that to your parser as well. There are two values
        // todo in the files we have; this top ident is not the DB id.
        writeln!(file, "@<TRIPOS>MOLECULE")?;
        writeln!(file, "{}", self.ident)?;
        writeln!(file, "{} {}", self.atoms.len(), self.bonds.len())?;
        writeln!(file, "{}", MolType::Small.to_str())?;
        writeln!(file, "{}", ChargeType::None.to_str())?;

        // **** Means a non-optional field is empty.
        writeln!(file, "****")?;
        // Optional line (comments, molecule weight, etc.)

        writeln!(file, "@<TRIPOS>ATOM")?;
        for (i, atom) in self.atoms.iter().enumerate() {
            writeln!(
                file,
                "{:>5} {:<2} {:>12.3} {:>8.3} {:>8.3} {:<2} {:>6} {:<3} {:>6.3}",
                i + 1,
                atom.element.to_letter(),
                atom.posit.x,
                atom.posit.y,
                atom.posit.z,
                atom.element.to_letter(),
                0,
                "UNL",
                atom.partial_charge.unwrap_or_default()
            )?;
        }

        writeln!(file, "@<TRIPOS>BOND")?;
        for (i, bond) in self.bonds.iter().enumerate() {
            let start_idx = bond.atom_0 + 1; // 1-based indexing
            let end_idx = bond.atom_1 + 1;
            let bond_count = match bond.bond_type {
                BondType::Covalent { count } => count.value() as u8,
                _ => 0,
            };

            writeln!(
                file,
                "{:>5}{:>6}{:>6}{:>3}",
                i + 1,
                start_idx,
                end_idx,
                bond_count
            )?;
        }

        Ok(())
    }
}

pub fn load_mol2(path: &Path) -> io::Result<Molecule> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let data_str: String = String::from_utf8(buffer)
        .map_err(|_| io::Error::new(ErrorKind::InvalidData, "Invalid UTF8"))?;

    Molecule::from_mol2(&data_str)
}
