#![allow(non_snake_case)]

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
//! - [Zinc](https://zinc.docking.org/)
//!
//!
//! Docking formats: Autodock Vina uses PDBQT files for both target and ligand. The targets, compared
//! to PDB and mmCIF files, have hydrogens added and charges computed. The ligands define which bonds
//! are rotatable. This format is specialized for docking operations.

// todo: Shower thought: Shoot ligands at the dockign site from various angles, in various conformations
// todo etc.

// 4MZI/160355 docking example: https://www.youtube.com/watch?v=vU2aNuP3Y8I

use std::{f32::consts::TAU, time::Instant};

use cudarc::runtime::result::device::set;
use lin_alg::{
    f32::{Vec3 as Vec3F32, f32x8, pack_float, pack_vec3},
    f64::{FORWARD, Quaternion, RIGHT, UP, Vec3},
    linspace,
};
use partial_charge::create_partial_charges;
use rand::Rng;
use rayon::prelude::*;

use crate::{
    bond_inference::create_hydrogen_bonds_one_way,
    docking::{
        dynamics_playback::build_vdw_dynamics,
        prep::{DockingSetup, LIGAND_SAMPLE_RATIO, Torsion},
    },
    element::Element,
    forces,
    forces::{V_lj, V_lj_x8, V_lj_x8_outer},
    molecule::{Atom, Ligand},
};

pub mod dynamics_playback;
pub mod external;
pub mod find_sites;
pub mod partial_charge;
pub mod prep;
pub mod prep_external;
pub mod site_surface;

const GRID_SPACING_SITE_FINDING: f64 = 5.0;

// Don't take into account target atoms that are not within this distance of the docking site center x
// docking site size. Keep in mind, the difference between side len (e.g. of the box drawn), and dist
// from center.
const ATOM_NEAR_SITE_DIST_THRESH: f64 = 1.4;

const HYDROPHOBIC_CUTOFF: f32 = 4.25; // 3.5 - 5 angstrom?

// This must be relatively low (Not much of an approximation): otherwise, the charges
// will cancel too easily when grouped, due to their nature.
// todo: Anything about 0 seems to produe no results.
pub const THETA_BH: f64 = 0.;

// const SOFTENING_FACTOR_SQ_ELECTROSTATIC: f32 = 1e-6;
const SOFTENING_FACTOR_SQ_ELECTROSTATIC: f32 = 1e-6;

const Q: f32 = 1.; // elementary charge.

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

#[derive(Debug, Clone, Default)]
/// Area IVO the docking site.
pub struct DockingSite {
    pub site_center: Vec3,
    pub site_radius: f64,
}

#[derive(Clone, Debug, Default)]
pub enum ConformationType {
    #[default]
    Rigid,
    Flexible {
        torsions: Vec<Torsion>,
    },
}

#[derive(Clone, Debug, Default)]
pub struct Pose {
    // pub anchor_atom: usize, // Index.
    /// The offset of the ligand's anchor atom from the docking center.
    /// todo: Consider normalizing positions to be around the origin, for numerical precision issues.
    pub anchor_posit: Vec3,
    pub orientation: Quaternion,
    pub conformation_type: ConformationType,
}

#[derive(Debug, Clone, Default)]
pub struct BindingEnergy {
    vdw: f32,
    h_bond_count: usize,
    h_bond: f32, // score
    hydrophobic: f32,
    electrostatic: f32,
    /// An ad-hoc metric of making sure the ligand is close to molecules.
    /// a geometric method?
    proximity: f32,
    score: f32,
}

impl BindingEnergy {
    pub fn new(vdw: f32, h_bond_count: usize, hydrophobic: f32, electrostatic: f32) -> Self {
        const E_PER_H_BOND: f32 = -1.2; // todo A/R.

        let h_bond = h_bond_count as f32 * E_PER_H_BOND;

        let weight_vdw = 1.;
        let weight_hydrophobic = 1.;
        let weight_electrostatic = 10.;

        // A low score is considered to be a better pose.
        let score = weight_vdw * vdw
            + h_bond
            + weight_hydrophobic * hydrophobic
            + weight_electrostatic * electrostatic;

        let proximity = 0.; // todo temp

        Self {
            vdw,
            h_bond_count,
            h_bond,
            hydrophobic,
            electrostatic,
            score,
            proximity,
        }
    }
}

/// todo: Improve this.
fn is_hydrophobic(atom: &Atom) -> bool {
    matches!(atom.element, Element::Carbon)
}

/// Calculate binding energy, in kcal/mol. The result will be negative. The maximum (negative) binding
/// energy may be the ideal conformation. This is used as a scoring metric.
pub fn calc_binding_energy(
    setup: &DockingSetup,
    ligand: &Ligand,
    lig_posits: &[Vec3F32],
) -> Option<BindingEnergy> {
    // todo: Integrate CUDA

    let len_rec = setup.rec_atoms_near_site.len();
    let len_lig = lig_posits.len();

    // Cache distances.
    let mut distances = Vec::with_capacity(len_rec * len_lig);
    for i_rec in 0..len_rec {
        for i_lig in 0..len_lig {
            let posit_rec: Vec3F32 = setup.rec_atoms_near_site[i_rec].posit.into();
            let posit_lig = lig_posits[i_lig];

            distances.push((posit_rec - posit_lig).magnitude());
        }
    }

    let (distances_x8, valid_lanes_last_dist) = pack_float(&distances);

    let vdw = if !is_x86_feature_detected!("avx") {
        // todo: Use a neighbor grid or similar? Set it up so there are two separate sides?
        distances
            .par_iter()
            .enumerate()
            .map(|(i, r)| {
                let (sigma, eps) = setup.lj_sigma_eps[i];
                V_lj(*r, sigma, eps)
            })
            .sum()
    } else {
        let vdw_x8: f32x8 = distances_x8
            .par_iter()
            .enumerate()
            .map(|(i, r)| {
                let sigma = setup.lj_sigma_x8[i];
                let eps = setup.lj_eps_x8[i];
                V_lj_x8(*r, sigma, eps)
            })
            .sum();

        vdw_x8.to_array().iter().sum()
    };

    let h_bond_count = {
        // Calculate hydrogen bonds
        let lig_indices: Vec<usize> = (0..len_lig).collect();

        // todo: THis is not efficient; work-in for now.
        let mut lig_atoms_positioned = ligand.molecule.atoms.clone();
        for (i, atom) in lig_atoms_positioned.iter_mut().enumerate() {
            atom.posit = lig_posits[i].into();
        }

        // todo: Use pre-computed dists in H bonds if able.

        // todo: Given you're using a relaxed distance thresh for H bonds, adjust the score
        // todo based on the actual distance of the bond.
        // We keep these separate, so the bond indices are meaningful.
        let h_bonds_rec_donor = create_hydrogen_bonds_one_way(
            &setup.rec_atoms_near_site,
            &setup.rec_indices,
            &setup.rec_bonds_near_site,
            // &ligand.molecule.atoms,
            &lig_atoms_positioned,
            &lig_indices,
            true,
        );

        let h_bonds_lig_donor = create_hydrogen_bonds_one_way(
            &lig_atoms_positioned,
            &lig_indices,
            &ligand.molecule.bonds,
            &setup.rec_atoms_near_site,
            &setup.rec_indices,
            true,
        );

        h_bonds_rec_donor.len() + h_bonds_lig_donor.len()
    };

    // Calculate Hydrophobic (solvation) interactions
    // -- HYDROPHOBIC INTERACTION TERM -- //
    // Example: We’ll collect a simple sum of energies for hydrophobic pairs.
    // For a more advanced approach, you might do a distance-based well function, etc.
    let hydrophobic_score = distances
        .par_iter()
        .enumerate()
        .filter_map(|(i, &r)| {
            if setup.hydrophobic[i] {
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
        .sum();

    // Handle partial charges between target, and ligand. This is a standard electrostatics calculation.
    // We use Barnes Hut to provide an approximate solution at higher speed.
    // todo: Determine if you want to use Rayon, for the expected partial charge count.
    // todo: Rayon.

    // todo: Your barnes_hut lib is only set up up for a single set where everything interacts.
    // todo: how do you set things up so one set interacts with the other?

    // todo: Sort out f32 vs f64 for this.
    let electrostatic = {
        let mut force = Vec3F32::new_zero();
        let partial_charges_lig = create_partial_charges(&ligand.molecule.atoms, Some(lig_posits));

        // In the barnes_hut etc nomenclature, we are iterating over *target* bodies. (Not associated
        // with target=protein=receptor; actually, the opposite!)

        // Note: Ligand positions are already positioned for the pose, by the time they enter this function.
        for q_lig in partial_charges_lig {
            // todo: Experimenting with non-BH to troubleshoot. BH is currently reporting 0 force.
            // todo: Maybe BH isn't suitable here due to local charges summing to 0 ?
            // for q_rec in partial_charges_rec {
            //     let diff: Vec3F32 = (q_rec.posit - q_lig.posit).into();
            //     let dist = distances[i_rec][i_lig];
            //     force += force_elec(
            //         (diff / dist).into(),
            //         q_rec.charge as f64,
            //         q_lig.charge as f64,
            //         dist as f64,
            //         SOFTENING_FACTOR_SQ_ELECTROSTATIC,
            //     )
            //     .into();
            // }
            //
            // continue;

            // Our bh algorithm is currently hard-coded to f64.
            let force_fn = |dir: Vec3, q_src: f64, dist: f64| {
                forces::force_coulomb(
                    dir.into(),
                    dist as f32,
                    q_src as f32,
                    q_lig.charge,
                    SOFTENING_FACTOR_SQ_ELECTROSTATIC,
                )
                .into()
            };

            let f: Vec3F32 = barnes_hut::run_bh(
                q_lig.posit.into(),
                999_999, // N/A, since we're comparing separate sets.
                &setup.charge_tree,
                &setup.bh_config,
                &force_fn,
            )
            .into();

            force += f;

            // force += barnes_hut::run_bh(
            //     q_lig.posit.into(),
            //     999_999, // N/A, since we're comparing separate sets.
            //     charge_tree,
            //     bh_config,
            //     &force_fn,
            // )
            // .into();
        }

        // Force magnitude. Closest to 0 is best, indicating stability?
        force.magnitude()
    };

    Some(BindingEnergy::new(
        vdw,
        h_bond_count,
        hydrophobic_score,
        electrostatic,
    ))
}

/// Brute-force, naive iteration of combinations. (For now)
fn make_posits_orientations(
    init: &DockingSite,
    posit_val: usize,
    num_orientations: usize,
) -> (Vec<Vec3>, Vec<Quaternion>) {
    // We'll break the box into 4 × 5 × 5 = 100 points
    // so that we fill out 'num_posits' exactly:
    let (nx, ny, nz) = (posit_val, posit_val, posit_val);
    let num_posits = nx * ny * nz;

    // The total extent along each axis is 2*site_box_size.
    // We'll divide that into nx, ny, nz intervals respectively.
    let dx = (2.0 * init.site_radius) / nx as f64;
    let dy = (2.0 * init.site_radius) / ny as f64;
    let dz = (2.0 * init.site_radius) / nz as f64;

    // Pre-allocate for efficiency
    let mut ligand_posits = Vec::with_capacity(num_posits);

    // Create points on a regular 3D grid:
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let x = init.site_center.x - init.site_radius + (i as f64 + 0.5) * dx;
                let y = init.site_center.y - init.site_radius + (j as f64 + 0.5) * dy;
                let z = init.site_center.z - init.site_radius + (k as f64 + 0.5) * dz;
                ligand_posits.push(Vec3::new(x, y, z));
            }
        }
    }

    let n_lats = ((num_orientations as f32 / 2.).powf(1. / 3.)) as usize;
    let n_lons = n_lats * 2;
    let n_rolls = n_lons;

    let n_orientations = n_lats * n_lons * n_rolls;
    let mut orientations = Vec::with_capacity(n_orientations);

    for i_lat in 0..n_lats {
        let frac = (i_lat as f32 + 0.5) / n_lats as f32;
        let mu = -1.0 + 2.0 * frac;
        let ϕ = mu.acos();

        for i_lon in 0..n_lons {
            let θ = (i_lon as f32 + 0.5) * TAU / (n_lons as f32);

            // Convert spherical to Cartesian
            let x = ϕ.sin() * θ.cos();
            let y = ϕ.sin() * θ.sin();
            let z = mu;

            let lat_lon_vec = Vec3F32::new(x, y, z).to_normalized();

            let or = Quaternion::from_unit_vecs(Vec3::new(0., 0., 1.), lat_lon_vec.into());

            for roll in 0..n_rolls {
                let angle = roll as f32 * TAU / n_rolls as f32;
                let rotator = Quaternion::from_axis_angle(lat_lon_vec.into(), angle as f64);
                orientations.push(rotator * or);
            }
        }
    }

    (ligand_posits, orientations)
}

/// Pre-generate poses, for a naive system.
pub(crate) fn init_poses(
    site: &DockingSite,
    flexible_bonds: &[usize],
    num_posits: usize,
    num_orientations: usize,
    angles_per_bond: usize,
) -> Vec<Pose> {
    // These positions are of the ligand's anchor atom.
    let (anchor_posits, orientations) =
        make_posits_orientations(site, num_posits, num_orientations);

    let angles = linspace(0., TAU, angles_per_bond);

    let mut result = Vec::new();

    for &anchor_posit in &anchor_posits {
        for &orientation in &orientations {
            // Build all angle-combos across all flexible bonds
            let mut all_combos = vec![Vec::new()]; // start with 1 empty combo

            for &bond_i in flexible_bonds {
                let mut new_combos = Vec::new();
                for existing_combo in &all_combos {
                    for &angle in &angles {
                        let mut extended = existing_combo.clone();
                        extended.push(Torsion {
                            bond: bond_i,
                            dihedral_angle: angle,
                        });
                        new_combos.push(extended);
                    }
                }
                all_combos = new_combos;
            }

            // Produce a Pose for each full combination of angles
            for torsions in all_combos {
                result.push(Pose {
                    anchor_posit,
                    orientation,
                    conformation_type: ConformationType::Flexible { torsions },
                });
            }
        }
    }

    result
}

/// Contains code that is specific to a set of poses. This includes low-cost filters that reduce
/// the downstream number of poses to match.
fn process_poses<'a>(
    poses: &'a [Pose],
    setup: &DockingSetup,
    ligand: &Ligand,
) -> Vec<(usize, BindingEnergy)> {
    // todo: Currently an outer/inner dynamic.
    // Set up the ligand atom positions for each pose; that's all that matters re the pose for now.
    let mut lig_posits = Vec::new();

    // todo: This approach of skipping atoms during iteration isn't great, as it depends on the order
    // todo the atoms are in the Vecs.

    // Optimization; Hydrogens are always close to another atom, and we have many; we can likely rely
    // on that other atom, and save ~n^2 computation here.
    println!("Eliminating poses with atoms too close together...");

    let mut geometry_poses_skip = Vec::new();

    for (i_pose, pose) in poses.iter().enumerate() {
        // todo: Cache distances here?
        let posits_this_pose: Vec<_> = ligand
            .position_atoms(Some(pose))
            .iter()
            .map(|p| (*p).into())
            .collect();

        // A smaller subset, used for some processes to improve performance.
        let lig_posits_sample: Vec<Vec3F32> = posits_this_pose
            .iter()
            .enumerate()
            .filter(|(i, a)| {
                let atom = &ligand.molecule.atoms[*i];
                atom.element == Element::Carbon && i % LIGAND_SAMPLE_RATIO == 0
            })
            .map(|(_i, v)| *v)
            .collect();

        lig_posits.push(posits_this_pose);

        // Remove pose that have any ligand atoms *intersecting* the receptor. We could possibly
        // use a surface mesh of some sort for this. For now, use VDW spheres. Crude, and probably good enough.
        // This pose reduction may significantly speed up the algorithm by discarding unsuitable poses
        // early.

        for rec_atom in &setup.rec_atoms_sample {
            let rec_posit: Vec3F32 = rec_atom.posit.into();

            let vdw_radius = rec_atom.element.vdw_radius() * 1.1;

            let mut end_loop = false;
            for lig_pos in &lig_posits_sample {
                let dist = (rec_posit - *lig_pos).magnitude();

                // todo: We should probably build an ASA surface, then assess if inside or outside of
                // todo that instead of this approach.

                if dist < vdw_radius {
                    geometry_poses_skip.push(i_pose);
                    end_loop = true;
                    break;
                }
            }
            if end_loop {
                break;
            }
        }

        // todo: Not matching scalar results.
        //     for (i_rec, rec_posit) in rec_posits_sample_x8.iter().enumerate() {
        //         let vdw_radius = rec_vdw_sample_x8[i_rec];
        //
        //         let lanes_rec = if i_rec == rec_posits_sample_x8.len() - 1 {
        //             valid_lanes_rec_sample
        //         } else {
        //             8
        //         };
        //
        //         let mut end_loop = false;
        //         for (i_lig, lig_pos) in posits_sample_x8.iter().enumerate() {
        //             let lanes_lig = if i_lig == posits_sample_x8.len() - 1 {
        //                 valid_lanes_lig
        //             } else {
        //                 8
        //             };
        //
        //             let dist = (*rec_posit - *lig_pos).magnitude();
        //
        //             // todo: We should probably build an ASA surface, then assess if inside or outside of
        //             // todo that instead of this approach.
        //
        //             // println!("DIST: {:?} d: {:.2?}, v: {:.2?}, ANY: {:?}", dist.lt(f32x8::splat(5.)), dist.to_array(),
        //                      // vdw_radius.to_array(), dist.lt(f32x8::splat(5.)).any());
        //
        //             // We don't use SIMD any compare, to properly handle the last lane.
        //             let valid_lanes = lanes_rec.min(lanes_lig);
        //             let vdw = vdw_radius.to_array();
        //             for (i_d, dist) in dist.to_array()[..valid_lanes].iter().enumerate() {
        //                 if *dist < vdw[i_d] {
        //                     geometry_poses_skip.push(i_pose);
        //                     end_loop = true;
        //                     break;
        //                 }
        //             }
        //         }
        //         if end_loop {
        //             break;
        //         }
        //     }

        // We use distances in multiple locations; cache here.
        // todo: Note that above, we compute distances, but for sample only.
        // todo: Put this back when ready.
        // let mut distances = Vec::with_capacity(&setup.rec_atoms_sample.len() * lig_posits[i_pose].len());
        // for (i_rec, posit_rec) in &setup.rec_atoms_sample.iter().enumerate() {
        //     for (i_lig, posit_lig) in lig_posits[i_pose].iter().enumerate() {
        //         let dist = (*posit_rec - *posit_lig).magnitude();
        //         distances.push((i_rec, i_lig, dist));
        //     }
        // }
    }

    println!(
        "Complete. iterating through {} poses...",
        poses.len() - geometry_poses_skip.len()
    );

    let result: Vec<_> = poses
        .par_iter()
        .enumerate()
        .filter(|(i_pose, _)| !geometry_poses_skip.contains(i_pose))
        .filter_map(|(i_pose, _pose)| {
            let energy = calc_binding_energy(setup, ligand, &lig_posits[i_pose]);
            if let Some(e) = energy {
                Some((i_pose, e))
            } else {
                None
            }
        })
        .collect();

    result
}

fn vary_pose(pose: &Pose) -> Vec<Pose> {
    let mut result = Vec::new();

    for i in -30..30 {
        if let ConformationType::Flexible { torsions } = &pose.conformation_type {
            let rot_amt = TAU as f64 / 200. * i as f64;

            let rotator_up = Quaternion::from_axis_angle(UP, rot_amt);
            let rotator_right = Quaternion::from_axis_angle(RIGHT, rot_amt);
            let rotator_fwd = Quaternion::from_axis_angle(FORWARD, rot_amt);

            // todo: Try combinations of the above.

            for rot_a in &[rotator_up, rotator_right, rotator_fwd] {
                for rot_b in &[rotator_up, rotator_right, rotator_fwd] {
                    for rot_c in &[rotator_up, rotator_right, rotator_fwd] {
                        result.push(Pose {
                            anchor_posit: pose.anchor_posit,
                            orientation: *rot_a * *rot_b * *rot_c * pose.orientation,
                            conformation_type: ConformationType::Flexible {
                                torsions: torsions.clone(),
                            },
                        });
                    }
                }
            }
        }
    }

    result
}

/// Return best pose, and energy.
///
/// Note: We use the term `receptor` here vice `target`, as `target` is also used in terms of
/// calculating forces between pairs. (These targets may or may not align!)
pub fn find_optimal_pose(setup: &DockingSetup, ligand: &Ligand) -> (Pose, BindingEnergy) {
    // todo: Consider another fn for this part of the setup, so you can re-use it more easily.

    // todo: Evaluate if you can cache EEM charges. Look into how position-dependent they are between ligand flexible
    // todo bond conformations, and lig/receptor interactions.

    println!(
        "Atom counts. Rec: {} Lig: {}",
        setup.rec_atoms_near_site.len(),
        ligand.molecule.atoms.len()
    );

    let start = Instant::now();

    let num_posits = 8; // Gets cubed, although many of these are eliminated.
    let num_orientations = 60;
    let angles_per_bond = 3;

    let poses = init_poses(
        &ligand.docking_site,
        &ligand.flexible_bonds,
        num_posits,
        num_orientations,
        angles_per_bond,
    );
    println!("Initial pose count: {} poses...", poses.len());

    // todo: Increase.
    let top_pose_count = 10;

    // Now process them in parallel and reduce to the single best pose:
    let mut pose_energies = process_poses(&poses, setup, ligand);

    pose_energies.sort_by(|a, b| a.1.score.partial_cmp(&b.1.score).unwrap());
    let best_pose = &poses[pose_energies[0].0];
    let best_energy = pose_energies[0].1.clone();

    // Conduct a molecular dynamics sim on the best poses, refining them further.
    // todo: This appears to not be doing much.
    for (pose_i, energy) in &pose_energies[0..top_pose_count] {
        continue; // todo: Put back when ready.

        let mut lig_this = ligand.clone(); //  todo: DOn't like this clone.
        lig_this.pose = poses[*pose_i].clone();

        let snapshots = build_vdw_dynamics(&lig_this, setup, false);

        let final_snap = &snapshots[snapshots.len() - 1];
        println!(
            "Updated snap: {:?}",
            final_snap.energy.as_ref().unwrap().score
        );
    }

    println!("Complete. \n\nBest pose init: {best_pose:?} \n\nScores: {best_energy:.3?}\n\n");

    // Vary orientations and positiosn of the best poses, pre and/or pose md sim?

    println!("\nBest initial pose: {best_pose:?} \nScores: {best_energy:.3?}\n");

    // Some ad-hoc tweaking.
    //let new_poses = vary_pose(best_pose);

    let elapsed = start.elapsed();
    println!("Time: {}ms", elapsed.as_millis());
    println!("Complete. \n\nBest pose: {best_pose:?} \n\nScores: {best_energy:.3?}\n\n");
    (best_pose.clone(), best_energy.clone())
}

// Find hydrogen bond interaction, hydrophobic interactions between ligand and protein.
// Find the "perfect" "Het" or "lead" molecule that will act as drug receptor
