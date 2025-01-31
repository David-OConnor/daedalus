use std::{io, io::ErrorKind, path::Path};

use pdbtbx::{Format, ReadOptions, StrictnessLevel, PDB};

pub fn load_pdb(path: &Path) -> io::Result<PDB> {
    let (pdb, _errors) = ReadOptions::default()
        // Note: at the default strictness level of Medium, we fail to parse a number of files, but I don't
        // know what the triggers are at this time.
        .set_level(StrictnessLevel::Loose)
        .read(path.to_str().unwrap())
        .map_err(|e| io::Error::new(ErrorKind::InvalidData, "Problem opening PDB file"))?;

    Ok(pdb)
}
