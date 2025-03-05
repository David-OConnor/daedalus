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

// 4MZI/160355 docking example: https://www.youtube.com/watch?v=vU2aNuP3Y8I

use lin_alg::f64::{Quaternion, Vec3};
use rand::Rng;
use rayon::prelude::*;

use crate::molecule::{Atom, Ligand, Molecule};

pub mod docking_external;
pub mod docking_prep;
pub mod docking_prep_external;
pub mod find_sites;

const GRID_SPACING_SITE_FINDING: f64 = 5.0;

const ATOM_NEAR_SITE_DIST_THRESH: f64 = 5.0; // todo: A.R.

#[derive(Clone, Copy, PartialEq)]
enum GaCrossoverMode {
    Twopt,
}

/// Calculate the Lennard-Jones potential between two atoms.
///
/// \[ V_{LJ}(r) = 4 \epsilon \left[\left(\frac{\sigma}{r}\right)^{12}
///     - \left(\frac{\sigma}{r}\right)^{6}\right] \]
///
/// In a real system, you’d want to parameterize \(\sigma\) and \(\epsilon\)
/// based on the atom types (i.e. from a force field lookup). Here, we’ll
/// just demonstrate the structure of the calculation with made-up constants.
pub fn lj_potential(atom_0: &Atom, atom_1: &Atom, ligand_atom_posit: Vec3) -> f32 {
    let r = (atom_0.posit - ligand_atom_posit).magnitude() as f32;

    // todo: Experiment
    const LJ_DIST_THRESH: f32 = 5.; // performance saver?
    if r > LJ_DIST_THRESH {
        return 0.;
    }

    // 1. Get *intrinsic* LJ parameters for each element

    // todo: Cache these in a table!
    let (sigma_a, epsilon_a) = atom_0.element.lj_params();
    let (sigma_b, epsilon_b) = atom_1.element.lj_params();

    // 2. Combine them for the pair (Lorentz-Berthelot)
    let sigma = 0.5 * (sigma_a + sigma_b);
    let epsilon = (epsilon_a * epsilon_b).sqrt();

    if r < f32::EPSILON {
        return 0.0;
    }

    // Note: Our sigma and eps values are very rough.
    let sr = sigma / r;
    let sr6 = sr.powi(6);
    let sr12 = sr6 * sr6;
    4.0 * epsilon * (sr12 - sr6)
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

#[derive(Clone, Debug)]
pub enum ConformationType {
    Rigid { orientation: Quaternion },
    Flexible { dihedral_angles: Vec<f32> },
}

impl Default for ConformationType {
    fn default() -> Self {
        Self::Rigid {
            orientation: Quaternion::new_identity(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Pose {
    pub anchor_atom: usize, // Index.
    /// The offset of the ligand's anchor atom from the docking center.
    /// todo: Consider normalizing positions to be around the origin, for numerical precision issues.
    pub anchor_posit: Vec3,
    pub conformation_type: ConformationType,
}

/// Calculate binding energy, in kcal/mol. The result will be negative. The maximum (negative) binding
/// energy may be the ideal conformation. This is used as a scoring metric.
pub fn binding_energy(target: &Molecule, ligand: &Ligand, pose: &Pose) -> f32 {
    // todo: Integrate CUDA.

    // Add this offset to each ligand atom to gets is position relative to this pose.
    // The anchor atom will be at the pose's anchor posit.
    let ligand_offset = pose.anchor_posit - ligand.molecule.atoms[pose.anchor_atom].posit;

    let mut vdw = 0.;

    match &pose.conformation_type {
        ConformationType::Rigid { orientation } => {}
        ConformationType::Flexible { dihedral_angles } => {
            unimplemented!()
        }
    }

    // todo: Use a neighbor grid or similar. Set it up so there are two separate sides?

    let tgt_atoms_near_site: Vec<&Atom> = target
        .atoms
        .iter()
        .filter(|a| {
            // todo: What other filters besides non-hetero?
            (a.posit - pose.anchor_posit).magnitude() < ATOM_NEAR_SITE_DIST_THRESH && !a.hetero
        })
        .collect();

    let mut atom_posits: Vec<Vec3> = ligand.molecule.atoms.iter().map(|a| a.posit).collect();
    for i in 0..atom_posits.len() {
        atom_posits[i] = ligand.position_atom(i, Some(pose));
    }

    // Note: This iterates in parallel over target, which has a much higher atom count.
    let lj: f32 = tgt_atoms_near_site
        .par_iter()
        // For each `atom_tgt`, compute the sum of LJ potentials with all ligand atoms
        .map(|atom_tgt| {
            ligand
                .molecule
                .atoms
                .iter()
                .enumerate()
                .map(|(i_lig, atom_ligand)| {
                    lj_potential(atom_tgt, atom_ligand, atom_posits[i_lig])
                })
                .sum::<f32>()
        })
        // Finally, sum up all partial sums into a single total.
        .sum();

    let vdw = lj; // todo?

    let mut h_bond = 0.;
    let mut hydrophobic = 0.;
    let mut electrostatic = 0.;

    vdw + h_bond + hydrophobic + electrostatic
}

/// Brute-force, naive iteration of combinations. (For now)
fn make_posits_orientations(
    init: &DockingInit,
    posit_val: usize,
    num_orientations: usize,
) -> (Vec<Vec3>, Vec<Quaternion>) {
    // We'll break the box into 4 × 5 × 5 = 100 points
    // so that we fill out 'num_posits' exactly:
    let (nx, ny, nz) = (posit_val, posit_val, posit_val);
    let num_posits = nx * ny * nz;

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
    ligand: &mut Ligand,
    params: &GeneticAlgorithmParameters,
) -> (Pose, f32) {
    // todo: Generic algorithm etc. Maybe that goes in the scoring fn?
    // for _ in 0..params.num_runs {
    // }

    println!("Finding optimal pose...");

    let num_posits: usize = 5; // Gets cubed.
    let num_orientations = 20;

    println!(
        "Conformation count: {:?}",
        num_posits.pow(3) * num_orientations
    );

    // These positions are of the ligand's anchor atom.
    let (anchor_posits, orientations) =
        make_posits_orientations(&ligand.docking_init, num_posits, num_orientations);

    let mut best_energy = 0.;
    let mut best_pose = Pose {
        anchor_atom: 0,
        anchor_posit: Vec3::new_zero(),
        conformation_type: ConformationType::Rigid {
            orientation: Quaternion::new_identity(),
        },
    };

    // todo: RAyon.
    for anchor_posit in &anchor_posits {
        for orientation in &orientations {
            let pose = Pose {
                anchor_atom: 0,
                anchor_posit: *anchor_posit,
                conformation_type: ConformationType::Rigid {
                    orientation: *orientation,
                },
            };

            let energy = binding_energy(target, &ligand, &pose);

            if energy < best_energy {
                best_energy = energy;
                best_pose = pose;
            }
        }
    }

    ligand.pose = best_pose.clone();
    ligand.set_anchor();

    println!("Complete. Best pose: {best_pose:?} E: {best_energy}");
    (best_pose, best_energy)
}

// Find hydrogen bond interaction, hydrophobic interactions between ligand and protein.
// Find the "perfect" "Het" or "lead" molecule that will act as drug receptor
