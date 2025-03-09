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
//! - [Available Chemicals Database](https://www.psds.ac.uk/) // todo: Login required?
//! - Roche compound inventory?
//!
//!
//! Docking formats: Autodock Vina uses PDBQT files for both target and ligand. The targets, compared
//! to PDB and mmCIF files, have hydrogens added and charges computed. The ligands define which bonds
//! are rotatable. This format is specialized for docking operations.

// 4MZI/160355 docking example: https://www.youtube.com/watch?v=vU2aNuP3Y8I

// todo: Temp/feature-gate
use std::{arch::x86_64::*, collections::HashMap};

use barnes_hut::{BhConfig, Cube, Tree};
use lin_alg::{
    f32::{Vec3 as Vec3F32, Vec3S},
    f64::{Quaternion, Vec3},
};
use partial_charge::{PartialCharge, create_partial_charges};
use rand::Rng;
use rayon::prelude::*;

use crate::{
    bond_inference::create_hydrogen_bonds_one_way,
    docking::docking_prep::Torsion,
    element::{Element, get_lj_params},
    molecule::{Atom, Bond, Ligand, Molecule},
};

pub mod docking_external;
pub mod docking_prep;
pub mod docking_prep_external;
pub mod find_sites;
pub mod partial_charge;

const GRID_SPACING_SITE_FINDING: f64 = 5.0;

// Don't take into account target atoms that are not within this distance of the docking site center x
// docking site size. Keep in mind, the difference between side len (e.g. of the box drawn), and dist
// from center.
const ATOM_NEAR_SITE_DIST_THRESH: f64 = 1.2;

const HYDROPHOBIC_CUTOFF: f32 = 4.25; // 3.5 - 5 angstrom?

const THETA_BH: f64 = 0.5;

// const SOFTENING_FACTOR_SQ_ELECTROSTATIC: f32 = 1e-6;
const SOFTENING_FACTOR_SQ_ELECTROSTATIC: f64 = 1e-6;

const Q: f32 = 1.; // elementary charge.

#[derive(Clone, Copy, PartialEq)]
enum GaCrossoverMode {
    Twopt,
}

/// The most fundamental part of Newtonian acceleration calculation.
/// `acc_dir` is a unit vector.
// todo: F64 A/R.
// pub fn acc_elec(acc_dir: Vec3F32, src_q: f32, tgt_q: f32, dist: f32, softening_factor_sq: f32) -> Vec3F32 {
pub fn acc_elec(
    acc_dir: Vec3,
    src_q: f64,
    tgt_q: f64,
    dist: f64,
    softening_factor_sq: f64,
) -> Vec3 {
    // Assume the coulomb constant is 1.
    acc_dir * src_q * tgt_q / (dist.powi(2) + softening_factor_sq)
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

/// todo: Experimental
pub fn lj_potential_simd(
    atom_0_posit: Vec3S,
    atom_1_posit: Vec3S,
    atom_0_els: [Element; 8],
    atom_1_els: [Element; 8],
    lj_lut: &HashMap<(Element, Element), (f32, f32)>,
) -> __m256 {
    unsafe {
        let r = (atom_0_posit - atom_1_posit).magnitude();

        let mut sig = [0.0; 8];
        let mut eps = [0.0; 8];

        for i in 0..8 {
            (sig[i], eps[i]) = get_lj_params(atom_0_els[i], atom_1_els[i], lj_lut)
        }

        let sig_ = _mm256_loadu_ps(sig.as_ptr());
        let eps_ = _mm256_loadu_ps(eps.as_ptr());

        // Intermediate steps; no SIMD exponent.
        let sr = _mm256_div_ps(sig_, r);
        let sr2 = _mm256_mul_ps(sr, sr);
        let sr4 = _mm256_mul_ps(sr2, sr2);

        let sr6 = _mm256_mul_ps(sr4, sr2);
        let sr12 = _mm256_mul_ps(sr6, sr6);

        let four = _mm256_set1_ps(4.);
        _mm256_mul_ps(four, _mm256_mul_ps(eps_, _mm256_div_ps(sr12, sr6)))
    }
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
    Flexible { torsions: Vec<Torsion> },
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

        let weight_vdw = 1.;
        let weight_hydrophobic = 1.;
        let weight_electrostatic = 1.;

        let score = weight_vdw * vdw
            + h_bond
            + weight_hydrophobic * hydrophobic
            + weight_electrostatic * electrostatic;

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
    rec_atoms_near_site: &[Atom],
    rec_indices: &[usize],
    rec_bonds_near_site: &[Bond],
    ligand: &Ligand,
    // pose: &Pose,
    lig_posits: &[Vec3],
    lj_lut: &HashMap<(Element, Element), (f32, f32)>,
    partial_charges_rec: &[PartialCharge],
    partial_charges_ligand: &[PartialCharge],
    bh_bounding_box: &Option<Cube>,
) -> BindingEnergy {
    // todo: Integrate CUDA.

    // todo: Use a neighbor grid or similar? Set it up so there are two separate sides?
    let vdw: f32 = {
        // todo: Experimenting with a different apch.

        let pair_count = rec_atoms_near_site.len() * ligand.molecule.atoms.len();
        let mut atoms_rec_paired = Vec::with_capacity(pair_count);
        let mut atoms_lig_paired = Vec::with_capacity(pair_count);
        let mut i_lig_paired = Vec::with_capacity(pair_count);

        for atom_rec in rec_atoms_near_site {
            for (i_lig, atom_lig) in ligand.molecule.atoms.iter().enumerate() {
                atoms_rec_paired.push(atom_rec);
                atoms_lig_paired.push(atom_lig);
                i_lig_paired.push(i_lig);
            }
        }

        let pairs: Vec<(&Atom, &Atom, usize)> = rec_atoms_near_site
            .iter()
            .flat_map(|atom_rec| {
                ligand
                    .molecule
                    .atoms
                    .iter()
                    .enumerate()
                    .map(move |(i_lig, atom_ligand)| (atom_rec, atom_ligand, i_lig))
            })
            .collect();

        // {
        //     // this is, for now, just to get you familiar with how SIMD might work.
        //     // Prepare SIMD Vec3s.
        //
        //     // todo: Placeholder, while you have `pairs` set up as a tuple. Avoid the extra iteration
        //     // todo by directio byilding these vecs.
        //     let rec_posits_: Vec<Vec3F32> = pairs.iter().map(|a| a.0.posit.into()).collect();
        //     let lig_posits_: Vec<Vec3F32> = pairs.iter().map(|a| a.1.posit.into()).collect();
        //
        //     let rec_posits = vec3s_to_simd(&rec_posits_);
        // let lig_posits = vec3s_to_simd(&lig_posits_);
        //
        //     // todo: You'd use Rayon here too.
        //     // let mut sum = Vec::new();
        //     // for i in 0..pairs.len() {
        //         // sum += lj_potential_simd(rec_posits[i], lig_posits[i], rec_els, lig_els, &lj_lut);
        //     // }
        // }

        pairs
            .par_iter()
            .map(|(atom_rec, atom_ligand, lig_posit_i)| {
                lj_potential(atom_rec, atom_ligand, lig_posits[*lig_posit_i], lj_lut)
            })
            .sum()
    };

    let h_bond_count = {
        // Calculate hydrogen bonds
        let lig_indices: Vec<usize> = (0..ligand.molecule.atoms.len()).collect();

        // todo: Given you're using a relaxed distance thresh for H bonds, adjust the score
        // todo based on the actual distance of the bond.
        // We keep these separate, so the bond indices are meaningful.
        let h_bonds_rec_donor = create_hydrogen_bonds_one_way(
            rec_atoms_near_site,
            rec_indices,
            rec_bonds_near_site,
            &ligand.molecule.atoms,
            &lig_indices,
            true,
        );
        let h_bonds_lig_donor = create_hydrogen_bonds_one_way(
            &ligand.molecule.atoms,
            &lig_indices,
            &ligand.molecule.bonds,
            rec_atoms_near_site,
            rec_indices,
            true,
        );

        h_bonds_rec_donor.len() + h_bonds_lig_donor.len()
    };

    if h_bond_count > 0 {
        println!("Num H bonds: {:?}", h_bond_count);
    }

    // Calculate Hydrophobic (solvation) interactions
    // -- HYDROPHOBIC INTERACTION TERM -- //
    // Example: We’ll collect a simple sum of energies for hydrophobic pairs.
    // For a more advanced approach, you might do a distance-based well function, etc.
    let hydrophobic_score: f32 = rec_atoms_near_site
        .par_iter()
        .map(|atom_rec| {
            ligand
                .molecule
                .atoms
                .iter()
                .enumerate()
                .filter_map(|(i_lig, atom_ligand)| {
                    // Check if both are hydrophobic
                    if is_hydrophobic(atom_rec) && is_hydrophobic(atom_ligand) {
                        let r = (atom_rec.posit - lig_posits[i_lig]).magnitude() as f32;

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

    // Handle partial charges between target, and ligand. This is a standard electrostatics calculation.
    // We use Barnes Hut to provide an approximate solution at higher speed.
    // todo: Determine if you want to use Rayon, for the expected partial charge count.
    // todo: Rayon.

    // todo: Your barnes_hut lib is only set up up for a single set where everything interacts.
    // todo: how do you set things up so one set interacts with the other?

    // todo: Sort out f32 vs f64 for this.
    let electrostatic = {
        if let Some(bb) = bh_bounding_box {
            // todo: Move this struct to state.
            let bh_config = BhConfig {
                θ: THETA_BH,
                ..Default::default()
            };

            // This tree is over the target (receptor) charges. This may be more efficient
            // than over the ligand, as we expect the receptor nearby atoms to be more numerous.
            let tree = Tree::new(&partial_charges_rec, bb, &bh_config);

            let mut force = Vec3F32::new_zero();

            // In the barnes_hut etc nomenclature, we are iterating over *target* bodies. (Not associated
            // with target=protein=receptor; actually, the opposite!

            for q_lig in partial_charges_ligand {
                // let diff: Vec3F32 = (q_rec.posit - q_lig.posit).into(); // todo: QC dir.
                // let dist = diff.magnitude();

                let force_fn = |acc_dir, q_src, dist| {
                    acc_elec(
                        acc_dir,
                        q_src,
                        q_lig.charge.into(),
                        dist,
                        SOFTENING_FACTOR_SQ_ELECTROSTATIC,
                    )
                };

                force += barnes_hut::run_bh(
                    q_lig.posit.into(),
                    999_999, // N/A, since we're comparing separate sets.
                    &tree,
                    &bh_config,
                    &force_fn,
                )
                .into();
            }

            // Force magnitude. Closest to 0 is best, indicating stability?
            force.magnitude()
        } else {
            0.
        }
    };

    BindingEnergy::new(vdw, h_bond_count, hydrophobic_score, electrostatic)
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

    let mut rng = rand::rng();
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
///
/// Note: We use the term `receptor` here vice `target`, as `target` is also used in terms of
/// calculating forces between pairs. (These targets may or may not align!)
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

    // Preliminary setup; computations that are not pose-specific, including optimizations.
    // let rec_atoms_near_site: Vec<(usize, &_)> = target
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
    let poses: Vec<_> = {
        // These positions are of the ligand's anchor atom.
        let (anchor_posits, orientations) =
            make_posits_orientations(&ligand.docking_init, num_posits, num_orientations);

        anchor_posits
            .iter()
            .flat_map(|anchor_posit| {
                orientations.iter().map(move |orientation| Pose {
                    anchor_posit: *anchor_posit,
                    conformation_type: ConformationType::Rigid {
                        orientation: *orientation,
                    },
                })
            })
            .collect()
    };

    // todo: Currently an outer/inner dynamic.
    let lig_atom_count = ligand.molecule.atoms.len();
    // Set up the ligand atom positions for each pose; that's all that matters re the pose for now.
    let mut lig_posits = Vec::with_capacity(poses.len());
    for pose in &poses {
        let mut posits_this_pose = Vec::with_capacity(lig_atom_count);
        for i in 0..lig_atom_count {
            posits_this_pose.push(ligand.position_atom(i, Some(pose)));
        }
        lig_posits.push(posits_this_pose);
    }

    // Note: Splitting the partial charges between target and ligand (As opposed to analyzing every pair
    // combination) may give us more useful data, and is likely much more efficient, if one side has substantially
    // fewer charges than the other.
    let partial_charges_tgt = create_partial_charges(&tgt_atoms_near_site, 1.);
    let partial_charges_lig = create_partial_charges(&ligand.molecule.atoms, 1.);

    println!(
        "Partial charges rec: {} lig: {}",
        partial_charges_tgt.len(),
        partial_charges_lig.len()
    );

    // For the Barnes Hut electrostatics tree.
    let bh_bounding_box = Cube::from_bodies(&partial_charges_tgt, 0., true);

    // Now process them in parallel and reduce to the single best pose:
    let (best_energy, best_pose) = poses
        .par_iter()
        .enumerate()
        .map(|(i_pose, pose)| {
            let energy = binding_energy(
                &tgt_atoms_near_site,
                &tgt_atom_indices,
                &tgt_bonds_near_site,
                &ligand,
                &lig_posits[i_pose],
                lj_lut,
                &partial_charges_tgt,
                &partial_charges_lig,
                &bh_bounding_box,
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

    println!("Complete. \n\nBest pose: {best_pose:?} \n\nScores: {best_energy:.3?}\n\n");
    (best_pose.clone(), best_energy)
}

// Find hydrogen bond interaction, hydrophobic interactions between ligand and protein.
// Find the "perfect" "Het" or "lead" molecule that will act as drug receptor
