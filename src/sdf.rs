//! For opening Structure Data Format (SDF) files. These are common molecular descriptions for ligands. It's a simpler format
//! than PDB.

use std::{
    fs::File,
    io,
    io::{BufReader, Read},
    path::Path,
};

use lin_alg::f64::Vec3;

use crate::{molecule::Atom, Element};

#[derive(Debug)]
pub struct Sdf {
    pub atoms: Vec<Atom>,
}

/// From a string of a CIF or PDB text file.
pub fn read_sdf(pdb_text: &str) -> io::Result<Sdf> {
    let mut result = Sdf { atoms: Vec::new() };
    let lines: Vec<&str> = pdb_text.lines().collect();

    // SDF files typically have at least 4 lines before the atom block:
    //   1) A title or identifier
    //   2) Usually blank or comments
    //   3) Often blank or comments
    //   4) "counts" line: e.g. " 50  50  0  ..." for V2000
    if lines.len() < 4 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Not enough lines to parse an SDF header",
        ));
    }

    // This is the "counts" line, e.g. " 50 50  0  0  0  0  0  0  0999 V2000"
    let counts_line = lines[3];
    let counts_cols: Vec<&str> = counts_line.split_whitespace().collect();

    if counts_cols.len() < 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Counts line doesn't have enough fields",
        ));
    }

    // Typically, the first number is the number of atoms (natoms)
    // and the second number is the number of bonds (nbonds).
    let natoms = counts_cols[0].parse::<usize>().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "Could not parse number of atoms",
        )
    })?;
    let nbonds = counts_cols[1].parse::<usize>().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "Could not parse number of bonds",
        )
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
            io::ErrorKind::InvalidData,
            "Not enough lines for the declared atom block",
        ));
    }

    for i in first_atom_line..last_atom_line {
        let atom_line = lines[i];
        let cols: Vec<&str> = atom_line.split_whitespace().collect();

        if cols.len() < 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Atom line {} does not have enough columns", i),
            ));
        }

        let x = cols[0].parse::<f64>().map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "Could not parse X coordinate")
        })?;
        let y = cols[1].parse::<f64>().map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "Could not parse Y coordinate")
        })?;
        let z = cols[2].parse::<f64>().map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "Could not parse Z coordinate")
        })?;
        let element = cols[3];

        // Now build your Atom struct.
        // (This assumes you have something like `pub struct Atom { ... }`
        // that can hold these values.)
        //
        // For example:
        let atom = Atom {
            serial_number: i - first_atom_line + 1,
            posit: Vec3 { x, y, z }, // or however you store coordinates
            element: Element::from_letter(element)?,
            role: None,
            amino_acid: None,
            hetero: false,
        };

        result.atoms.push(atom);
    }

    // We could now skip over the bond lines if we want:
    //   let first_bond_line = last_atom_line;
    //   let last_bond_line = first_bond_line + nbonds;
    // etc.
    // Then we look for "M  END" or the data fields, etc.

    // For now, just return the Sdf with the atoms we parsed:
    Ok(result)
}

pub fn load_sdf(path: &Path) -> io::Result<Sdf> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let data_str: String = String::from_utf8(buffer).expect("Found invalid UTF-8");

    read_sdf(&data_str)
}
