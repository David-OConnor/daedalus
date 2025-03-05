//! Todo: Remove this module A/R

use std::{
    io,
    io::ErrorKind,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    str::FromStr,
};

use crate::{
    docking::{DockingInit, Pose},
    molecule::{Ligand, Molecule},
};

pub fn check_adv_avail(vina_path: &Path) -> bool {
    let status = Command::new(vina_path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .args(["--version"])
        .status();

    status.is_ok()
}

/// Run Autodock Vina. `target_path` and `ligand_path` are to the prepared PDBQT files.
/// https://vina.scripps.edu/manual/#usage (Or run the program with `--help`.)
fn run_adv(
    init: &DockingInit,
    vina_path: &Path,
    target_path: &Path,
    ligand_path: &Path,
) -> io::Result<Pose> {
    println!("Running Autodock Vina...");

    let output_filename = "docking_result.pdbqt";

    let output_text = Command::new(vina_path.to_str().unwrap_or_default())
        .args([
            "--receptor",
            target_path.to_str().unwrap_or_default(),
            // "--flex" // Flexible side chains; optional.
            "--ligand",
            ligand_path.to_str().unwrap_or_default(),
            "--out",
            output_filename,
            "--center_x",
            &init.site_center.x.to_string(),
            "--center_y",
            &init.site_center.y.to_string(),
            "--center_z",
            &init.site_center.z.to_string(),
            "--size_x",
            &init.site_box_size.to_string(),
            "--size_y",
            &init.site_box_size.to_string(),
            "--size_z",
            &init.site_box_size.to_string(),
            // "--exhaustiveness", // Proportional to runtime. Higher is more accurate. Defaults to 8.
            // "num_modes",
            // "energy_range",
        ])
        // todo: Status now for a clean print
        // .output()?;
        .status()?;

    println!("Complete.");
    //
    // // todo: Create a post from output text.
    // println!("\n\nOutput text: {:?}\n\n", output_text);

    // todo: Parse the output file into a pose here A/R

    // todo: Output the pose.
    Err(io::Error::new(ErrorKind::Other, ""))
}

pub fn dock_with_vina(mol: &Molecule, ligand: &Ligand, vina_path: &Option<PathBuf>) {
    if let Some(vina_path) = vina_path {
        match run_adv(
            &ligand.docking_init,
            vina_path,
            &PathBuf::from_str(&format!("{}_target.pdbqt", mol.ident)).unwrap(),
            &PathBuf::from_str(&format!("{}_ligand.pdbqt", ligand.molecule.ident)).unwrap(),
        ) {
            Ok(r) => println!("Docking successful"),
            Err(e) => eprintln!("Docking failed: {e:?}"),
        }
    } else {
        eprintln!("No Autodock Vina install located yet.");
    }
}
