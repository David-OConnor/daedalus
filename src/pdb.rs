use std::{io, io::ErrorKind, path::Path};

use pdbtbx::{Format, ReadOptions, StrictnessLevel, PDB};

pub fn load_pdb(path: &Path) -> io::Result<PDB> {
    let (pdb, _errors) = ReadOptions::default()
        // At the default strictness level of Medium, we fail to parse a number of files. Medium and Strict
        // ensure closer conformance to the PDB and CIF specs, but many files in the wild do not. Setting
        // loose is required for practical use cases.
        .set_level(StrictnessLevel::Loose)
        .read(path.to_str().unwrap())
        .map_err(|e| io::Error::new(ErrorKind::InvalidData, "Problem opening PDB file"))?;

    Ok(pdb)
}
