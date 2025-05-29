//! For opening Structure Data Format (SDF) files. These are common molecular descriptions for ligands. It's a simpler format
//! than PDB.

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
    molecule::{Atom, BondType, Chain, Molecule, Residue, ResidueType},
};

struct Sdf {
    pub molecule: Molecule,
    /// These fields aren't universal to the format.
    pub metadata: HashMap<String, String>,
}

impl Molecule {
    /// From a string of a CIF or PDB text file.
    pub fn from_sdf(sdf_text: &str) -> io::Result<Self> {
        let lines: Vec<&str> = sdf_text.lines().collect();

        // SDF files typically have at least 4 lines before the atom block:
        //   1) A title or identifier
        //   2) Usually blank or comments
        //   3) Often blank or comments
        //   4) "counts" line: e.g. " 50  50  0  ..." for V2000
        if lines.len() < 4 {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "Not enough lines to parse an SDF header",
            ));
        }

        // todo: Incorporate more cols A/R.
        // After element:
        // Mass difference (0, unless an isotope)
        // Charge (+1 for cation etc)
        // Stereo, valence, other flags

        // todo: Do bonds too
        // first atom index
        // second atom index
        // 1 for single, 2 for double etc
        // 0 for no stereochemistry, 1=up, 6=down etc
        // Other properties: Bond topology, reaction center flags etc. Usually 0

        // This is the "counts" line, e.g. " 50 50  0  0  0  0  0  0  0999 V2000"
        let counts_line = lines[3];
        let counts_cols: Vec<&str> = counts_line.split_whitespace().collect();

        if counts_cols.len() < 2 {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "Counts line doesn't have enough fields",
            ));
        }

        // Typically, the first number is the number of atoms (natoms)
        // and the second number is the number of bonds (nbonds).
        let natoms = counts_cols[0].parse::<usize>().map_err(|_| {
            io::Error::new(ErrorKind::InvalidData, "Could not parse number of atoms")
        })?;
        let nbonds = counts_cols[1].parse::<usize>().map_err(|_| {
            io::Error::new(ErrorKind::InvalidData, "Could not parse number of bonds")
        })?;

        // Now read the next 'natoms' lines as the atom block.
        // Each line usually looks like:
        //   X Y Z Element ??? ??? ...
        //   e.g. "    1.4386   -0.8054   -0.4963 O   0  0  0  0  0  0  0  0  0  0  0  0"
        //
        // Make sure we have enough lines in the file:
        let first_atom_line = 4;
        let last_atom_line = first_atom_line + natoms;
        if lines.len() < last_atom_line {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "Not enough lines for the declared atom block",
            ));
        }

        let mut atoms = Vec::new();

        for i in first_atom_line..last_atom_line {
            let atom_line = lines[i];
            let cols: Vec<&str> = atom_line.split_whitespace().collect();

            if cols.len() < 4 {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    format!("Atom line {} does not have enough columns", i),
                ));
            }

            let x = cols[0].parse::<f64>().map_err(|_| {
                io::Error::new(ErrorKind::InvalidData, "Could not parse X coordinate")
            })?;
            let y = cols[1].parse::<f64>().map_err(|_| {
                io::Error::new(ErrorKind::InvalidData, "Could not parse Y coordinate")
            })?;
            let z = cols[2].parse::<f64>().map_err(|_| {
                io::Error::new(ErrorKind::InvalidData, "Could not parse Z coordinate")
            })?;
            let element = cols[3];

            atoms.push(Atom {
                serial_number: i - first_atom_line + 1,
                posit: Vec3 { x, y, z }, // or however you store coordinates
                element: Element::from_letter(element)?,
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

        // Look for a molecule identifier in the file. Check for either
        // "> <PUBCHEM_COMPOUND_CID>" or "> <DATABASE_ID>" and take the next nonempty line.
        let mut pubchem_cid = None;
        let mut drugbank_id = None;

        for (i, line) in lines.iter().enumerate() {
            if line.contains("> <PUBCHEM_COMPOUND_CID>") {
                if let Some(value_line) = lines.get(i + 1) {
                    let value = value_line.trim();
                    if let Ok(v) = value.parse::<u32>() {
                        pubchem_cid = Some(v);
                    }
                }
            }
            if line.contains("> <DATABASE_ID>") {
                if let Some(value_line) = lines.get(i + 1) {
                    let value = value_line.trim();
                    if !value.is_empty() {
                        drugbank_id = Some(value.to_string());
                    }
                }
            }
        }

        let ident = lines[0].trim().to_string();

        // We could now skip over the bond lines if we want:
        //   let first_bond_line = last_atom_ line;
        //   let last_bond_line = first_bond_line + nbonds;
        // etc.
        // Then we look for "M  END" or the data fields, etc.

        // For now, just return the Sdf with the atoms we parsed:

        let mut chains = Vec::new();
        let mut residues = Vec::new();

        let atom_indices: Vec<usize> = (0..atoms.len()).collect();

        residues.push(Residue {
            serial_number: 0,
            res_type: ResidueType::Other("Unknown".to_string()),
            atoms: atom_indices.clone(),
            dihedral: None,
        });

        chains.push(Chain {
            id: "A".to_string(),
            residues: vec![0],
            atoms: atom_indices,
            visible: true,
        });

        Ok(Molecule::new(
            ident,
            atoms,
            chains,
            residues,
            pubchem_cid,
            drugbank_id,
        ))
    }

    pub fn save_sdf(&self, path: &Path) -> io::Result<()> {
        let mut file = File::create(path)?;

        // 1) Title line (often the first line in SDF).
        //    We use the molecule's name/identifier here:
        // todo: There is a subtlety here. Add that to your parser as well. There are two values
        // todo in the files we have; this top ident is not the DB id.
        writeln!(file, "{}", self.ident)?;

        // 2) Write two blank lines:
        writeln!(file)?;
        writeln!(file)?;

        let natoms = self.atoms.len();
        let nbonds = self.bonds.len();

        // Format the counts line. We loosely mimic typical spacing,
        // though it's not strictly required to line up exactly.
        writeln!(
            file,
            "{:>3}{:>3}  0  0  0  0           0999 V2000",
            natoms, nbonds
        )?;

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

        writeln!(file, "M  END")?;

        // Metadata
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

pub fn load_sdf(path: &Path) -> io::Result<Molecule> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let data_str: String = String::from_utf8(buffer)
        .map_err(|_| io::Error::new(ErrorKind::InvalidData, "Invalid UTF8"))?;

    Molecule::from_mol2(&data_str)
}
