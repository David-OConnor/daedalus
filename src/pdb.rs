use std::{io, io::ErrorKind, path::Path};

use pdbtbx::PDB;

pub fn load_pdb(path: &Path) -> io::Result<PDB> {
    // let (pdb, _errors) = ReadOptions::default()
    //     .set_level(StrictnessLevel::Medium)
    //     .set_format(Format::Pdb)
    //     .read(path.to_str().unwrap())
    //     .map_err(|e| io::Error::new(ErrorKind::InvalidData, "Problem opening PDB file"))?;
    println!("PDB path: {:?}", path);
    let (pdb, _errors) = pdbtbx::open(path.to_str().unwrap())
        .map_err(|e| io::Error::new(ErrorKind::InvalidData, "Problem opening PDB file"))?;

    Ok(pdb)
}
