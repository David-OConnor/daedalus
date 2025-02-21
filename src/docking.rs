//! Example docking software:
//!
//! Autodock Vina: Popular CLI tool, integrated into several GUI-based softwares:
//! https://vina.scripps.edu/downloads/
//!
//! Chimera, with Autodock Vina integeration. [Open source](https://www.cgl.ucsf.edu/chimera/docs/sourcecode.html)
//!
//! AutoDock (4? Racoon, and Vision?;  PyRx?)
//! Mgl tools: Python Molecular viewer (PMV), Auto dock Tools.
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

use lin_alg::{f32::Quaternion, f64::Vec3};

use crate::molecule::Molecule;

#[derive(Clone, Copy, PartialEq)]
enum GaCrossoverMode {
    Twopt,
}

// pub struct Ligand {}

/// todo: Figure this out
/// taken from a screenshot of an application that uses TK. (One of the ones you DLed?)
struct GeneticAlgorithmParameters {
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
    pub target_offset: Vec3,
    pub ligand_offset: Vec3,
    pub conformation_type: ConformationType,
}

/// Calculate binding energy, in kcal/mol. The result will be negative. The maximum (negative) binding
/// energy may be the ideal conformation. This is used as a scoring metric.
pub fn binding_energy(target: &Molecule, ligand: &Molecule, pose: Pose) -> f32 {
    0.
}

pub fn find_optimal_pose(
    target: &Molecule,
    ligand: &Molecule,
    params: &GeneticAlgorithmParameters,
) -> Pose {
    for _ in 0..params.num_runs {}
    Pose {
        target_offset: Vec3::new_zero(),
        ligand_offset: Vec3::new_zero(),
        conformation_type: ConformationType::Flexible {
            dihedral_angles: Vec::new(),
        },
    }
}

// Find hydrogen bond interaction, hydrophobic interactions between ligand and protein.
// Find the "perfect" "Het" or "lead" molecule that will act as drug receptor
