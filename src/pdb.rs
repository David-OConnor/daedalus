use std::{
    io,
    io::{BufReader, ErrorKind},
    path::Path,
};

use pdbtbx::{Format, ReadOptions, StrictnessLevel, PDB};

/// From a string of a CIF or PDB text file.
pub fn read_pdb(pdb_text: &str) -> io::Result<PDB> {
    let reader = BufReader::new(pdb_text.as_bytes());

    let (pdb, _errors) = ReadOptions::default()
        .set_level(StrictnessLevel::Loose)
        .set_format(Format::Mmcif) // Must be set explicitly if  using read_raw.
        .read_raw(reader)
        .map_err(|e| io::Error::new(ErrorKind::InvalidData, "Problem reading PDB text"))?;

    Ok(pdb)
}

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
