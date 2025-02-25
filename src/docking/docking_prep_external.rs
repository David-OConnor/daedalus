//! Runs Open babel to prepare targets and ligands.
//!
//! Update: Using Open Babel. Assume an installation, with `obabel` available in the path.

use std::{
    io,
    path::Path,
    process::{Command, Stdio},
};

pub fn check_babel_avail() -> bool {
    let status = Command::new("obabel")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .args(["-V"])
        .status();

    status.is_ok()
}

// `mk_prepare_receptor.py -i nucleic_acid.cif -o my_receptor -j -p -f A:42`
// todo: What are the extra args?
/// todo: PDB only; no cif.
/// Also, SEL modules crash it?
/// http://openbabel.org/docs/Command-line_tools/babel.html#options
pub fn prepare_target(mol_path: &Path) -> io::Result<()> {
    let path = mol_path.to_str().unwrap_or_default();
    println!("Preparing target with Open Babel...");

    // This adds (missing) hydrogens, assigns Gasteiger partial charges, and removes water.
    // update: Seems to *add* water??
    // todo: Kolman charges vice gasteiger?
    // Todo: Rem water?
    Command::new("obabel")
        .args([
            path,
            "-O",
            "target_prepped.pdbqt",
            "-h",
            "--partialcharge gasteiger",
            // "--partialcharge kolman",
            // `-xr` seems to be required to prevent errors about `ROOT` lines, when Vina reads the PDBQT file.
            "-xr",
            // "--filter",
            // "\"not water\""
        ])
        .status()?;
    println!("Complete");

    Ok(())
}

/// http://openbabel.org/docs/Command-line_tools/babel.html#options
pub fn prepare_ligand(mol_path: &Path, ligand_is_2d: bool) -> io::Result<()> {
    let path = mol_path.to_str().unwrap_or_default();

    // Adds H and partial charges as for target. Also handles notatible rotatable bonds.
    //
    let mut args = vec![
        path,
        "-O",
        "ligand_prepped.pdbqt",
        "--gen3d",
        "-h",
        "--partialcharge gasteiger",
    ];
    // Generates 3D coordinates if the SDF is in 2D only, or lacks coordinates.

    // Notes: You may also use the commands        `--minimize --ff GAFF` as optimizations.
    if ligand_is_2d {
        args.push("--gen3d")
    }

    println!("Preparing ligand with Open Babel...");
    Command::new("obabel").args(&args).status()?;
    println!("Complete");

    Ok(())
}
