//! For parsing mmCIF files for secondary structure. Easier to implement here than modifying PDBTBX.

use std::io::{self, Read, Seek};

use crate::cartoon_mesh::BackboneSS;

// todo: PDB support too?

pub fn load_secondary_structure<R: Read + Seek>(data: R) -> io::Result<Vec<BackboneSS>> {
    // pub fn load_secondary_structure<R: Read + Seek>(mut data: R) -> io::Result<Vec<BackboneSS>> {
    let mut result = Vec::new();

    Ok(result)
}
