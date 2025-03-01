//! For opening Structure Data Format (SDF) files. These are common molecular descriptions for ligands. It's a simpler format
//! than PDB.

use std::{
    fs::File,
    io,
    io::{BufReader, ErrorKind, Read},
    path::Path,
};

use lin_alg::f64::Vec3;

use crate::{
    Element,
    bond_inference::{create_bonds, create_hydrogen_bonds},
    molecule::{Atom, Chain, Molecule, Residue, ResidueType},
    util::mol_center_size,
};

impl Molecule {
    /// From a string of a CIF or PDB text file.
    pub fn from_sdf(pdb_text: &str) -> io::Result<Self> {
        let lines: Vec<&str> = pdb_text.lines().collect();

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

            // Now build your Atom struct.
            // (This assumes you have something like `pub struct Atom { ... }`
            // that can hold these values.)
            //
            // For example:
            let atom = Atom {
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
            };

            atoms.push(atom);
        }

        // Look for a molecule identifier in the file. Check for either
        // "> <PUBCHEM_COMPOUND_CID>" or "> <DATABASE_ID>" and take the next nonempty line.
        let mut pubchem = false;
        let mut drugbank = false;
        let mut pubchem_cid = None;
        let mut drugbank_id = None;

        let mut ident: Option<String> = None;
        for (i, line) in lines.iter().enumerate() {
            if line.contains("> <PUBCHEM_COMPOUND_CID>") {
                pubchem = true;
            }
            if line.contains("> <DATABASE_ID>") {
                drugbank = true;
            }

            if pubchem || drugbank {
                if let Some(value_line) = lines.get(i + 1) {
                    let value = value_line.trim();
                    if !value.is_empty() {
                        ident = Some(value.to_string());
                        break;
                    }
                }
            }
        }

        // Fallback: use the first line (often the title) if no identifier field was found.
        let ident = ident.unwrap_or_else(|| lines[0].trim().to_string());

        if pubchem {
            pubchem_cid = Some(ident.parse().unwrap_or_default());
        }
        if drugbank {
            drugbank_id = Some(ident.clone());
        }

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

        let (center, size) = mol_center_size(&atoms);

        let mut result = Self {
            ident,
            atoms,
            bonds: Vec::new(),
            chains,
            residues,
            metadata: None,
            sa_surface_pts: None,
            mesh_created: false,
            secondary_structure: Vec::new(),
            center,
            size,
            pubchem_cid,
            drugbank_id,
        };

        result.populate_hydrogens_angles();
        result.bonds = create_bonds(&result.atoms);
        result.bonds.extend(create_hydrogen_bonds(&result.atoms));

        Ok(result)
    }

    pub fn save_sdf(&self, path: &Path) -> io::Result<()> {
        Ok(())
    }
}

pub fn load_sdf(path: &Path) -> io::Result<Molecule> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let data_str: String = String::from_utf8(buffer)
        .map_err(|_| io::Error::new(ErrorKind::InvalidData, "Invalid UTF8"))?;

    Molecule::from_sdf(&data_str)
}
