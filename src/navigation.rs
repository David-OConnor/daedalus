//! C+P partly, from PlasCAD
//! todo: Into gui mod?

use std::path::PathBuf;

use bincode::{Decode, Encode};

const DEFAULT_TAB_NAME: &str = "New tab";
// When abbreviating a path, show no more than this many characters.
const PATH_ABBREV_MAX_LEN: usize = 16;

#[derive(Encode, Decode, Clone, Default, Debug)]
pub struct Tab {
    pub path: Option<PathBuf>,
    pub ab1: bool, // todo: Enum if you add a third category.
}

/// Used in several GUI components to get data from open tabs.
/// Note: For name, we currently default to file name (with extension), then
/// plasmid name, then a default. See if you want to default to plasmid name.
///
/// Returns the name, and the tab index.
pub fn get_tab_names(
    tabs: &[Tab],
    plasmid_names: &[&str],
    abbrev_name: bool,
) -> Vec<(String, usize)> {
    let mut result = Vec::new();

    for (i, p) in tabs.iter().enumerate() {
        let name = name_from_path(&p.path, plasmid_names[i], abbrev_name);
        result.push((name, i));
    }

    result
}

/// A short, descriptive name for a given opened tab.
pub fn name_from_path(path: &Option<PathBuf>, plasmid_name: &str, abbrev_name: bool) -> String {
    let mut name = match path {
        Some(path) => path
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name_str| name_str.to_string())
            .unwrap(),
        None => {
            if !plasmid_name.is_empty() {
                plasmid_name.to_owned()
            } else {
                DEFAULT_TAB_NAME.to_owned()
            }
        }
    };

    if abbrev_name && name.len() > PATH_ABBREV_MAX_LEN {
        name = format!("{}...", &name[..PATH_ABBREV_MAX_LEN].to_string())
    }

    name
}
