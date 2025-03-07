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

use std::collections::HashMap;

use lin_alg::f64::{Quaternion, Vec3};
use rand::Rng;
use rayon::prelude::*;

use crate::{
    bond_inference::{create_hydrogen_bonds, create_hydrogen_bonds_one_way},
    element::{Element, get_lj_params},
    molecule::{Atom, Bond, HydrogenBond, Ligand, Molecule},
};

pub mod docking_external;
pub mod docking_prep;
pub mod docking_prep_external;
pub mod find_sites;

const GRID_SPACING_SITE_FINDING: f64 = 5.0;

// Don't take into account target atoms that are not within this distance of the docking site center x
// docking site size. Keep in mind, the difference between side len (e.g. of the box drawn), and dist
// from center.
const ATOM_NEAR_SITE_DIST_THRESH: f64 = 1.2;

const HYDROPHOBIC_CUTOFF: f32 = 4.25; // 3.5 - 5 angstrom?

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
pub fn lj_potential(
    atom_0: &Atom,
    atom_1: &Atom,
    atom_1_posit: Vec3,
    lj_lut: &HashMap<(Element, Element), (f32, f32)>,
) -> f32 {
    let r = (atom_0.posit - atom_1_posit).magnitude() as f32;

    if r < f32::EPSILON {
        return 0.;
    }

    let (sigma, epsilon) = get_lj_params(atom_0.element, atom_1.element, lj_lut);

    // Note: Our sigma and eps values are very rough.
    let sr = sigma / r;
    let sr6 = sr.powi(6);
    let sr12 = sr6.powi(2);
    4. * epsilon * (sr12 - sr6)
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
    // pub anchor_atom: usize, // Index.
    /// The offset of the ligand's anchor atom from the docking center.
    /// todo: Consider normalizing positions to be around the origin, for numerical precision issues.
    pub anchor_posit: Vec3,
    pub conformation_type: ConformationType,
}

#[derive(Debug, Default)]
pub struct BindingEnergy {
    vdw: f32,
    h_bond_count: usize,
    h_bond: f32, // score
    hydrophobic: f32,
    electrostatic: f32,
    score: f32,
}

impl BindingEnergy {
    pub fn new(vdw: f32, h_bond_count: usize, hydrophobic: f32, electrostatic: f32) -> Self {
        const E_PER_H_BOND: f32 = -1.; // todo A/R.

        let h_bond = h_bond_count as f32 * E_PER_H_BOND;
        let score = vdw + h_bond + hydrophobic + electrostatic;

        Self {
            vdw,
            h_bond_count,
            h_bond,
            hydrophobic,
            electrostatic,
            score,
        }
    }
}

/// todo: Improve this.
fn is_hydrophobic(atom: &Atom) -> bool {
    match atom.element {
        Element::Carbon => true,
        _ => false,
    }
}

/// Calculate binding energy, in kcal/mol. The result will be negative. The maximum (negative) binding
/// energy may be the ideal conformation. This is used as a scoring metric.
pub fn binding_energy(
    tgt_atoms_near_site: &[Atom],
    tgt_indices: &[usize],
    tgt_bonds_near_site: &[Bond],
    ligand: &Ligand,
    pose: &Pose,
    lj_lut: &HashMap<(Element, Element), (f32, f32)>,
) -> BindingEnergy {
    // todo: Integrate CUDA.

    match &pose.conformation_type {
        ConformationType::Rigid { orientation } => {}
        ConformationType::Flexible { dihedral_angles } => {
            unimplemented!()
        }
    }

    // todo: Use a neighbor grid or similar. Set it up so there are two separate sides?

    let mut lig_posits: Vec<Vec3> = ligand.molecule.atoms.iter().map(|a| a.posit).collect();
    for i in 0..lig_posits.len() {
        lig_posits[i] = ligand.position_atom(i, Some(pose));
    }

    // Note: This iterates in parallel over target, which has a much higher atom count.
    let vdw: f32 = tgt_atoms_near_site
        .par_iter()
        // For each `atom_tgt`, compute the sum of LJ potentials with all ligand atoms
        .map(|atom_tgt| {
            ligand
                .molecule
                .atoms
                .iter()
                .enumerate()
                .map(|(i_lig, atom_ligand)| {
                    lj_potential(&atom_tgt, atom_ligand, lig_posits[i_lig], lj_lut)
                })
                .sum::<f32>()
        })
        .sum();

    // Calculate Hydrophobic (solvation) interactions
    // -- HYDROPHOBIC INTERACTION TERM -- //
    // Example: We’ll collect a simple sum of energies for hydrophobic pairs.
    // For a more advanced approach, you might do a distance-based well function, etc.
    let hydrophobic_score: f32 = tgt_atoms_near_site
        .par_iter()
        .map(|atom_tgt| {
            ligand
                .molecule
                .atoms
                .iter()
                .enumerate()
                .filter_map(|(i_lig, atom_ligand)| {
                    // Check if both are hydrophobic
                    if is_hydrophobic(atom_tgt) && is_hydrophobic(atom_ligand) {
                        let r = (atom_tgt.posit - lig_posits[i_lig]).magnitude() as f32;

                        // If they are within a cutoff, add favorable energy.
                        // (Your real scoring might be more nuanced.)
                        if r < HYDROPHOBIC_CUTOFF {
                            // Simple approach: add a small negative (favorable) energy
                            // or some distance-dependent function:
                            // E = -k * (1 - r/CUTOFF) for example
                            let k = 0.2; // scale factor in kcal/mol
                            let scaled = 1.0 - (r / HYDROPHOBIC_CUTOFF);
                            return Some(-k * scaled.max(0.0));
                        }
                    }
                    None
                })
                .sum::<f32>()
        })
        .sum();

    let h_bond_count = {
        // Calculate hydrogen bonds
        let lig_indices: Vec<usize> = (0..ligand.molecule.atoms.len()).collect();

        // todo: Given you're using a relaxed distance thresh for H bonds, adjust the score
        // todo based on the actual distance of the bond.
        // We keep these separate, so the bond indices are meaningful.
        let h_bonds_tgt_donor = create_hydrogen_bonds_one_way(
            tgt_atoms_near_site,
            tgt_indices,
            tgt_bonds_near_site,
            &ligand.molecule.atoms,
            &lig_indices,
            true,
        );
        let h_bonds_lig_donor = create_hydrogen_bonds_one_way(
            &ligand.molecule.atoms,
            &lig_indices,
            &ligand.molecule.bonds,
            tgt_atoms_near_site,
            tgt_indices,
            true,
        );

        h_bonds_tgt_donor.len() + h_bonds_lig_donor.len()
    };

    if h_bond_count > 0 {
        println!("Num H bonds: {:?}", h_bond_count);
    }

    BindingEnergy::new(vdw, h_bond_count, hydrophobic_score, 0.)
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
    lj_lut: &HashMap<(Element, Element), (f32, f32)>,
) -> (Pose, BindingEnergy) {
    // todo: Generic algorithm etc. Maybe that goes in the scoring fn?
    // for _ in 0..params.num_runs {
    // }

    let dist_thresh = ATOM_NEAR_SITE_DIST_THRESH * ligand.docking_init.site_box_size;

    let num_posits: usize = 8; // Gets cubed.
    let num_orientations = 60;

    // For your test, see if you can get a version tha tis flexed correctly.
    // todo: And/or set up a mol angle editing feature, and use it.

    println!(
        "\nFinding optimal pose... Conformation count: {:?}",
        num_posits.pow(3) * num_orientations
    );

    // These positions are of the ligand's anchor atom.
    let (anchor_posits, orientations) =
        make_posits_orientations(&ligand.docking_init, num_posits, num_orientations);

    // Preliminary setup; computations that are not pose-specific, including optimizations.
    // let tgt_atoms_near_site: Vec<(usize, &_)> = target
    //     .atoms
    //     .iter()
    //     .enumerate()
    //     .filter(|(i, a)| {
    //         (a.posit - ligand.docking_init.site_center).magnitude() < dist_thresh && !a.hetero
    //     })
    //     .collect();

    println!("Dist thresh: {:?}", dist_thresh);
    let mut tgt_atom_indices = Vec::new();
    let tgt_atoms_near_site: Vec<_> = target
        .atoms
        .iter()
        .enumerate()
        .filter(|(i, a)| {
            let r =
                (a.posit - ligand.docking_init.site_center).magnitude() < dist_thresh && !a.hetero;
            if r {
                tgt_atom_indices.push(*i);
            }
            r
        })
        .map(|(i, a)| a.clone()) // todo: Don't like the clone;
        .collect();

    // Bonds here is used for identifying donor heavy and H pairs for hydrogen bonds.
    let tgt_bonds_near_site: Vec<_> = target
        .bonds
        .iter()
        // Don't use ||; all atom indices in these bonds must be present in `tgt_atoms_near_site`.
        .filter(|b| tgt_atom_indices.contains(&b.atom_0) && tgt_atom_indices.contains(&b.atom_1))
        .map(|b| b.clone()) // todo: don't like the clone
        .collect();

    // Build a list of all candidate poses
    let poses: Vec<_> = anchor_posits
        .iter()
        .flat_map(|anchor_posit| {
            orientations.iter().map(move |orientation| Pose {
                anchor_posit: *anchor_posit,
                conformation_type: ConformationType::Rigid {
                    orientation: *orientation,
                },
            })
        })
        .collect();

    // Now process them in parallel and reduce to the single best pose:
    let (best_energy, best_pose) = poses
        .par_iter()
        .map(|pose| {
            let energy = binding_energy(
                &tgt_atoms_near_site,
                &tgt_atom_indices,
                &tgt_bonds_near_site,
                &ligand,
                pose,
                lj_lut,
            );
            (energy, pose)
        })
        // Provide an identity value for reduction; here we use +∞ for energy:
        .reduce(
            || (BindingEnergy::default(), &poses[0]),
            |(energy_a, pose_a), (energy_b, pose_b)| {
                if energy_b.score < energy_a.score {
                    (energy_b, pose_b)
                } else {
                    (energy_a, pose_a)
                }
            },
        );

    ligand.pose = best_pose.clone();

    println!("Complete. Best pose: {best_pose:?} E: {best_energy:?}");
    (best_pose.clone(), best_energy)
}

// Find hydrogen bond interaction, hydrophobic interactions between ligand and protein.
// Find the "perfect" "Het" or "lead" molecule that will act as drug receptor
