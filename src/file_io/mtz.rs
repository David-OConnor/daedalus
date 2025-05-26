//! For interoperability between MTZ files, and our reflections structs

use std::{fs::File, io, io::Read, path::Path};

use crate::reflection::{Reflection, ReflectionsData};

pub fn parse_mtz(data: &[u8]) -> io::Result<ReflectionsData> {
    let mut result = Default::default();

    Ok(result)
}

pub fn load_mtz(path: &Path) -> io::Result<ReflectionsData> {
    let mut file = File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;

    parse_mtz(&buf)
}

pub fn save_mtz(data: &ReflectionsData) -> Vec<u8> {
    let mut result = Vec::new();

    result
}
