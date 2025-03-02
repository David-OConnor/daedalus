//! Example docking software:
//!
//! [Autodock Vina](https://vina.scripps.edu/): Popular CLI tool, integrated into several GUI-based applications.
//! An improvement over Autodock 4 in speed and quality. Probably the best free Docking program; comparable in capability to commercial
//! solutions. (But maybe not in user experience?) Autodock GPU is another modern imrpovement over Autodock 4.
//!
//!
//! Chimera, with Autodock Vina integeration. [Open source](https://www.cgl.ucsf.edu/chimera/docs/sourcecode.html)
//!
//! AutoDock (4? Racoon, and Vision?;  PyRx?)
//! Mgl tools: Python Molecular viewer (PMV), Auto dock Tools.
//!
//! [Gnina](https://github.com/gnina/gnina?tab=readme-ov-file#usage) is another tool that may have some promising
//! reasons to consider over Vina. Based on a fork of a fork of Vina. Uses ML.
//!
//! Schrodinger Maestro: The one to beat.
//!
//! BIOVIA Discovery Studio Visualizer: Free viewer from Dassault (SolidWorks maker): https://discover.3ds.com/discovery-studio-visualizer-download
//!
//! //! Haddock: Server: https://rascar.science.uu.nl/haddock2.4/
//!
//!
//! Molecule databases:
//! - [Drugbank](https://go.drugbank.com/)
//! - [Pubchem](https://pubchem.ncbi.nlm.nih.gov/)
//!
//!
//! Docking formats: Autodock Vina uses PDBQT files for both target and ligand. The targets, compared
//! to PDB and mmCIF files, have hydrogens added and charges computed. The ligands define which bonds
//! are rotatable. This format is specialized for docking operations.

use std::{
    io,
    io::ErrorKind,
    path::Path,
    process::{Command, Stdio},
};

use lin_alg::f64::{Quaternion, Vec3};
use rand::Rng;

use crate::molecule::{Ligand, Molecule};

pub mod docking_prep;
pub mod docking_prep_external;

#[derive(Clone, Copy, PartialEq)]
enum GaCrossoverMode {
    Twopt,
}

/// todo: Figure this out
/// taken from a screenshot of an application that uses TK. (One of the ones you DLed?)
pub struct GeneticAlgorithmParameters {
    pub num_runs: usize,
    pub population_size: u32, // todo: usize?
    pub max_num_evals: u32,
    pub max_num_gens: u32,
    /// Maximum number of top individuals that automatically survive.
    pub max_num_top_individuals: u16,
    pub rate_of_gene_mutation: f32,
    pub rate_of_crossover: f32,
    pub ga_crossover_mode: GaCrossoverMode,
    /// Mean of Cauchy distribution for gene mutation.
    pub cauchy_mean: f32,
    /// Variance of Cauchy distribution for gene mutation.
    pub cauchy_variance: f32,
    /// Number of generations for picking worst individual
    pub num_gens_worst: u16,
}

impl Default for GeneticAlgorithmParameters {
    fn default() -> Self {
        Self {
            num_runs: 10,
            population_size: 150,
            max_num_evals: 25_000_000,
            max_num_gens: 27_000,
            max_num_top_individuals: 1,
            rate_of_gene_mutation: 0.02,
            rate_of_crossover: 0.8,
            ga_crossover_mode: GaCrossoverMode::Twopt,
            cauchy_mean: 0.,
            cauchy_variance: 1.,
            num_gens_worst: 10,
        }
    }
}

#[derive(Debug, Default)]
/// Area IVO the docking site.
pub struct DockingInit {
    pub site_center: Vec3,
    pub site_box_size: f64, // Assume square. // todo: Allow diff dims
                            // todo: Num points in each dimension?
}

enum ConformationType {
    Rigid { orientation: Quaternion },
    Flexible { dihedral_angles: Vec<f32> },
}

pub struct Pose {
    // todo
    /// These offsets are relative to the molecule origins for target and ligand. E.g., they depend,
    /// on the specific coordinate system used from mmCIF file etc they were loaded from.
    /// We move both the ligand, and target, so that the docking site is near the origin, for
    /// numerical reasons.
    // pub target_offset: Vec3,
    pub ligand_offset: Vec3,
    pub conformation_type: ConformationType,
}

/// Calculate binding energy, in kcal/mol. The result will be negative. The maximum (negative) binding
/// energy may be the ideal conformation. This is used as a scoring metric.
pub fn binding_energy(target: &Molecule, ligand: &Molecule, pose: &Pose) -> f32 {
    0.
}

/// Brute-force, naive iteration of combinations. (For now)
fn make_posits_orientations(
    init: &DockingInit,
    num_posits: usize,
    num_orientations: usize,
) -> (Vec<Vec3>, Vec<Quaternion>) {
    // We'll break the box into 4 × 5 × 5 = 100 points
    // so that we fill out 'num_posits' exactly:
    let (nx, ny, nz) = (10, 10, 10);
    let num_posits = nx * ny * nz; // = 100 // todo: ...

    // The total extent along each axis is 2*site_box_size.
    // We'll divide that into nx, ny, nz intervals respectively.
    let dx = (2.0 * init.site_box_size) / nx as f64;
    let dy = (2.0 * init.site_box_size) / ny as f64;
    let dz = (2.0 * init.site_box_size) / nz as f64;

    // Pre-allocate for efficiency
    let mut ligand_posits = Vec::with_capacity(num_posits);

    // Create points on a regular 3D grid:
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let x = init.site_center.x - init.site_box_size + (i as f64 + 0.5) * dx;
                let y = init.site_center.y - init.site_box_size + (j as f64 + 0.5) * dy;
                let z = init.site_center.z - init.site_box_size + (k as f64 + 0.5) * dz;
                ligand_posits.push(Vec3::new(x, y, z));
            }
        }
    }

    // todo: Try without random?

    let mut rng = rand::thread_rng();
    let orientations: Vec<Quaternion> = (0..num_orientations)
        .map(|_| {
            let u1: f64 = rng.random();
            let u2: f64 = rng.random();
            let u3: f64 = rng.random();

            let sqrt1_minus_u1 = (1.0 - u1).sqrt();
            let sqrt_u1 = u1.sqrt();
            let theta1 = 2.0 * std::f64::consts::PI * u2;
            let theta2 = 2.0 * std::f64::consts::PI * u3;

            Quaternion::new(
                sqrt1_minus_u1 * theta1.sin(), // w
                sqrt1_minus_u1 * theta1.cos(), // x
                sqrt_u1 * theta2.sin(),        // y
                sqrt_u1 * theta2.cos(),        // z
            )
        })
        .collect();

    (ligand_posits, orientations)
}

/// Return best pose, and energy.
pub fn find_optimal_pose(
    target: &Molecule,
    ligand: &Ligand,
    params: &GeneticAlgorithmParameters,
) -> (Pose, f32) {
    // todo: Generic algorithm etc. Maybe that goes in the scoring fn?
    // for _ in 0..params.num_runs {
    // }

    let num_posits = 1_000;
    let num_orientations = 200;

    let (ligand_posits, orientations) =
        make_posits_orientations(&ligand.docking_init, num_posits, num_orientations);

    let mut best_energy = 0.;
    let mut best_pose = Pose {
        ligand_offset: Vec3::new_zero(),
        conformation_type: ConformationType::Rigid {
            orientation: Quaternion::new_identity(),
        },
    };

    for ligand_posit in &ligand_posits {
        for orientation in &orientations {
            let pose = Pose {
                ligand_offset: *ligand_posit,
                conformation_type: ConformationType::Rigid {
                    orientation: *orientation,
                },
            };

            let energy = binding_energy(target, &ligand.molecule, &pose);

            if energy < best_energy {
                best_energy = energy;
                best_pose = pose;
            }
        }
    }

    (best_pose, best_energy)
}

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
pub fn run_adv(
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

// Find hydrogen bond interaction, hydrophobic interactions between ligand and protein.
// Find the "perfect" "Het" or "lead" molecule that will act as drug receptor
