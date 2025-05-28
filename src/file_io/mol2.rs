//! For opening Mol2 files. These are common molecular descriptions for ligands.

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
    molecule::{Atom, Bond, BondCount, BondType, Chain, Molecule, Residue, ResidueType},
};

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

        if lines.len() < 4 {
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
            if line.to_uppercase().contains("<TRIPOS>ATOM") {
                in_atom_section = true;
                in_bond_section = false;
                continue;
            }

            if line.to_uppercase().contains("<TRIPOS>BOND") {
                in_atom_section = false;
                in_bond_section = true;
                continue;
            }

            if in_atom_section {
                let cols: Vec<&str> = line.split_whitespace().collect();

                let serial_number = cols[0].parse::<usize>().map_err(|_| {
                    io::Error::new(ErrorKind::InvalidData, "Could not parse serial number")
                })?;

                // todo: THis, or col 5?
                let element = Element::from_letter(cols[1])?;

                let x = cols[2].parse::<f64>().map_err(|_| {
                    io::Error::new(ErrorKind::InvalidData, "Could not parse X coordinate")
                })?;
                let y = cols[3].parse::<f64>().map_err(|_| {
                    io::Error::new(ErrorKind::InvalidData, "Could not parse Y coordinate")
                })?;
                let z = cols[4].parse::<f64>().map_err(|_| {
                    io::Error::new(ErrorKind::InvalidData, "Could not parse Z coordinate")
                })?;

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
                    partial_charge: None,
                    dock_type: None,
                });
            }

            if in_bond_section {
                let cols: Vec<&str> = line.split_whitespace().collect();

                let atom_0 = cols[1].parse::<usize>().map_err(|_| {
                    io::Error::new(ErrorKind::InvalidData, "Could not parse atom 0 in bond")
                })?;

                let atom_1 = cols[2].parse::<usize>().map_err(|_| {
                    io::Error::new(ErrorKind::InvalidData, "Could not parse atom 1 in bond")
                })?;

                let count_num = cols[3].parse::<u8>().map_err(|_| {
                    io::Error::new(ErrorKind::InvalidData, "Could not parse atom 1 in bond")
                })?;

                bonds.push(Bond {
                    bond_type: BondType::Covalent {
                        count: BondCount::from_count(count_num),
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
        let mut file = File::create(path)?;

        // 1) Title line (often the first line in SDF).
        //    We use the molecule's name/identifier here:
        // todo: There is a subtlety here. Add that to your parser as well. There are two values
        // todo in the files we have; this top ident is not the DB id.
        writeln!(file, "{}", self.ident)?;

        // 2) Write two blank lines:
        writeln!(file)?;
        writeln!(file)?;

        // 3) Counts line:
        //    Typically "  X  Y  0  0  0  0  0  0  0999 V2000"
        //    Where X = number of atoms, Y = number of bonds
        let natoms = self.atoms.len();
        let nbonds = self.bonds.len();

        // Format the counts line. We loosely mimic typical spacing,
        // though it's not strictly required to line up exactly.
        writeln!(
            file,
            "{:>3}{:>3}  0  0  0  0           0999 V2000",
            natoms, nbonds
        )?;

        // 4) Atom block: each line typically has
        //      X Y Z Element 0  0  0  0  0  0  0  0  0  0
        //    We'll just place a few zeros after the element for now.
        for atom in &self.atoms {
            let x = atom.posit.x;
            let y = atom.posit.y;
            let z = atom.posit.z;
            let symbol = atom.element.to_letter();

            // MDL v2000 format often uses fixed-width fields,
            // but for simplicity we use whitespace separation:
            writeln!(
                file,
                "{:>10.4}{:>10.4}{:>10.4} {:<2}  0  0  0  0  0  0  0  0  0  0",
                x, y, z, symbol
            )?;
        }

        // 5) Bond block: if your `Molecule` has bond info, loop it here:
        for bond in &self.bonds {
            let start_idx = bond.atom_0 + 1; // 1-based in SDF
            let end_idx = bond.atom_1 + 1;
            let bond_count = match bond.bond_type {
                BondType::Covalent { count } => count.value() as u8,
                _ => 0,
            };

            writeln!(
                file,
                "{:>3}{:>3}{:>3}  0  0  0  0",
                start_idx, end_idx, bond_count
            )?;
        }

        // 6) MDL “M  END” line:
        writeln!(file, "M  END")?;

        // 7) Metadata fields:
        //    If you have anything like PUBCHEM_COMPOUND_CID or DRUGBANK_ID,
        //    we can write it in the > <FIELD_NAME> format,
        //    then the value, then a blank line.
        //
        //    For example, if you have a pubchem_cid or drugbank_id in the molecule:
        if let Some(cid) = self.pubchem_cid {
            writeln!(file, "> <PUBCHEM_COMPOUND_CID>")?;
            writeln!(file, "{}", cid)?;
            writeln!(file)?; // blank line
        }
        if let Some(ref dbid) = self.drugbank_id {
            writeln!(file, "> <DATABASE_ID>")?;
            writeln!(file, "{}", dbid)?;
            writeln!(file)?; // blank line
            writeln!(file, "> <DATABASE_NAME>")?;
            writeln!(file, "drugbank")?;
            writeln!(file)?; // blank line
        }

        // If you have a general metadata HashMap, you could do:
        // for (key, value) in &self.metadata {
        //     writeln!(file, "> <{}>", key)?;
        //     writeln!(file, "{}", value)?;
        //     writeln!(file)?;
        // }

        // 8) End of this molecule record in SDF
        writeln!(file, "$$$$")?;

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
