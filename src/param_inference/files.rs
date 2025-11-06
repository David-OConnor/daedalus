//! For managing inference-related files

use std::{
    fs, io,
    path::{Path, PathBuf},
};

pub const MODEL_PATH: &str = "geostd_model.safetensors";
pub const VOCAB_PATH: &str = "geostd_model.vocab";

pub const GEOSTD_PATH: &str = "C:/users/the_a/Desktop/bio_misc/amber_geostd";

/// Find Mol2 and FRCMOD paths. Assumes there are per-letter subfolders one-layer deep.
/// todo: FRCmod as well
pub fn find_paths(geostd_dir: &Path) -> io::Result<(Vec<PathBuf>, Vec<PathBuf>)> {
    let mut mol2_paths = Vec::new();
    let mut frcmod_paths = Vec::new();

    for entry in fs::read_dir(geostd_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            for subentry in fs::read_dir(&path)? {
                let subentry = subentry?;
                let subpath = subentry.path();

                if subpath
                    .extension()
                    .map(|e| e.to_string_lossy().to_lowercase())
                    == Some("mol2".to_string())
                {
                    mol2_paths.push(subpath);
                } else if subpath
                    .extension()
                    .map(|e| e.to_string_lossy().to_lowercase())
                    == Some("frcmod".to_string())
                {
                    frcmod_paths.push(subpath);
                }
            }
        }
    }

    Ok((mol2_paths, frcmod_paths))
}
