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
    collections::VecDeque,
    io,
    io::ErrorKind,
    path::Path,
    process::{Command, Stdio},
};

use lin_alg::f64::{Quaternion, Vec3};
use rand::Rng;

use crate::molecule::{Atom, Ligand, Molecule};

pub mod docking_prep;
pub mod docking_prep_external;

const GRID_SPACING_SITE_FINDING: f64 = 1.0;

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
pub fn lj_potential(atom_0: &Atom, atom_1: &Atom) -> f32 {
    // 1. Get *intrinsic* LJ parameters for each element
    let (sigma_a, epsilon_a) = atom_0.element.lj_params();
    let (sigma_b, epsilon_b) = atom_1.element.lj_params();

    // 2. Combine them for the pair (Lorentz-Berthelot)
    let sigma = 0.5 * (sigma_a + sigma_b);
    let epsilon = (epsilon_a * epsilon_b).sqrt();

    let r = (atom_0.posit - atom_1.posit).magnitude() as f32;

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

/// Attempt to find docking sites, using cavity detection.
fn find_docking_sites(mol: &Molecule) -> Vec<DockingInit> {
    // todo: Super chatGPT rough!!

    let mut result = Vec::new();

    // 1. Determine bounding box of the molecule
    let (mut min_x, mut min_y, mut min_z) = (f64::MAX, f64::MAX, f64::MAX);
    let (mut max_x, mut max_y, mut max_z) = (f64::MIN, f64::MIN, f64::MIN);

    if mol.atoms.is_empty() {
        return result; // No atoms, no sites
    }

    for atom in &mol.atoms {
        let p = atom.posit;
        if p.x < min_x {
            min_x = p.x;
        }
        if p.y < min_y {
            min_y = p.y;
        }
        if p.z < min_z {
            min_z = p.z;
        }
        if p.x > max_x {
            max_x = p.x;
        }
        if p.y > max_y {
            max_y = p.y;
        }
        if p.z > max_z {
            max_z = p.z;
        }
    }

    // Pad the bounding box slightly, to ensure we capture surface
    let atom_radius_temp: f64 = 2.0; // todo??
    let pad = 2.0 * atom_radius_temp;
    min_x -= pad;
    min_y -= pad;
    min_z -= pad;
    max_x += pad;
    max_y += pad;
    max_z += pad;

    // Helper function to see if a point is "inside" (within any atom's radius).
    let is_inside_molecule = |x: f64, y: f64, z: f64| -> bool {
        let pt = Vec3 { x, y, z };
        for atom in &mol.atoms {
            let dx = pt.x - atom.posit.x;
            let dy = pt.y - atom.posit.y;
            let dz = pt.z - atom.posit.z;
            let dist2 = dx * dx + dy * dy + dz * dz;
            // Compare squared distances to avoid sqrt call
            if dist2 < atom.element.vdw_radius().powi(2) as f64 {
                // Clashes with an atom => "inside" the molecule volume
                return true;
            }
        }
        false
    };

    // 2. Discretize bounding box into 3D grid
    let nx = ((max_x - min_x) / GRID_SPACING_SITE_FINDING).ceil() as usize;
    let ny = ((max_y - min_y) / GRID_SPACING_SITE_FINDING).ceil() as usize;
    let nz = ((max_z - min_z) / GRID_SPACING_SITE_FINDING).ceil() as usize;

    // We'll store a 3D array of booleans:
    //   true  => inside the molecule
    //   false => empty space
    // For convenience, flatten it into 1D: index = (ix + nx * (iy + ny * iz)).
    let mut grid = vec![false; nx * ny * nz];

    // 3. Fill the grid
    let mut index = 0;
    for iz in 0..nz {
        let zc = min_z + (iz as f64) * GRID_SPACING_SITE_FINDING;
        for iy in 0..ny {
            let yc = min_y + (iy as f64) * GRID_SPACING_SITE_FINDING;
            for ix in 0..nx {
                let xc = min_x + (ix as f64) * GRID_SPACING_SITE_FINDING;
                grid[index] = is_inside_molecule(xc, yc, zc);
                index += 1;
            }
        }
    }

    // 4. We want to find "interior pockets" => empty regions not connected to "outside."
    //    We'll do a flood fill from the boundaries of the grid to mark externally connected empty space.
    let mut visited = vec![false; grid.len()];

    // 4a. Helper to convert (ix, iy, iz) -> linear index
    let to_index = |ix: usize, iy: usize, iz: usize| ix + nx * (iy + ny * iz);

    // BFS function to mark connected empties from a starting cell
    let mut queue = VecDeque::new();
    let neighbors = |ix: usize, iy: usize, iz: usize| -> Vec<(usize, usize, usize)> {
        let mut neighs = Vec::new();
        let ix_i = ix as isize;
        let iy_i = iy as isize;
        let iz_i = iz as isize;
        for (dx, dy, dz) in &[
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ] {
            let nx_i = ix_i + dx;
            let ny_i = iy_i + dy;
            let nz_i = iz_i + dz;
            if nx_i >= 0
                && (nx_i as usize) < nx
                && ny_i >= 0
                && (ny_i as usize) < ny
                && nz_i >= 0
                && (nz_i as usize) < nz
            {
                neighs.push((nx_i as usize, ny_i as usize, nz_i as usize));
            }
        }
        neighs
    };

    // Mark all external empty cells by BFS from the "outer walls"
    // Because the outer boundary is definitely "outside" (just empty space away from the molecule).
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in [0, nz - 1] {
                let idx = to_index(ix, iy, iz);
                if !grid[idx] && !visited[idx] {
                    visited[idx] = true;
                    queue.push_back((ix, iy, iz));
                }
            }
        }
    }
    for iy in 0..ny {
        for iz in 0..nz {
            for ix in [0, nx - 1] {
                let idx = to_index(ix, iy, iz);
                if !grid[idx] && !visited[idx] {
                    visited[idx] = true;
                    queue.push_back((ix, iy, iz));
                }
            }
        }
    }
    for iz in 0..nz {
        for ix in 0..nx {
            for iy in [0, ny - 1] {
                let idx = to_index(ix, iy, iz);
                if !grid[idx] && !visited[idx] {
                    visited[idx] = true;
                    queue.push_back((ix, iy, iz));
                }
            }
        }
    }

    // Flood-fill from these boundary empties to mark them visited
    while let Some((cx, cy, cz)) = queue.pop_front() {
        for (nx_, ny_, nz_) in neighbors(cx, cy, cz) {
            let nidx = to_index(nx_, ny_, nz_);
            if !grid[nidx] && !visited[nidx] {
                visited[nidx] = true;
                queue.push_back((nx_, ny_, nz_));
            }
        }
    }

    // 4b. Now, anything that remains "false" in 'visited' & also false in 'grid' is an unvisited empty cell
    // => a pocket cell. We'll do a BFS over those to find distinct pockets.
    let mut pocket_id = vec![None; grid.len()];
    let mut current_pocket_label = 0;
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let idx = to_index(ix, iy, iz);
                // If it's inside the molecule, or it's visited from outside, skip
                if grid[idx] || visited[idx] {
                    continue;
                }
                // We found an unvisited empty cell => new pocket
                current_pocket_label += 1;
                let label = current_pocket_label;
                // BFS from here to label the entire pocket
                let mut queue2 = VecDeque::new();
                queue2.push_back((ix, iy, iz));
                pocket_id[idx] = Some(label);

                while let Some((cx, cy, cz)) = queue2.pop_front() {
                    for (nx_, ny_, nz_) in neighbors(cx, cy, cz) {
                        let nidx = to_index(nx_, ny_, nz_);
                        if !grid[nidx] && !visited[nidx] && pocket_id[nidx].is_none() {
                            pocket_id[nidx] = Some(label);
                            queue2.push_back((nx_, ny_, nz_));
                        }
                    }
                }
            }
        }
    }

    // 5. Compute a centroid for each pocket, then push a `DockingInit`.
    if current_pocket_label == 0 {
        // No pockets
        return result;
    }

    // Collect all the points for each pocket
    let mut sums = vec![(0.0, 0.0, 0.0, 0usize); current_pocket_label + 1];
    // sums[label] = (sumX, sumY, sumZ, count)
    for iz in 0..nz {
        let zc = min_z + (iz as f64) * GRID_SPACING_SITE_FINDING;
        for iy in 0..ny {
            let yc = min_y + (iy as f64) * GRID_SPACING_SITE_FINDING;
            for ix in 0..nx {
                let xc = min_x + (ix as f64) * GRID_SPACING_SITE_FINDING;
                let idx = to_index(ix, iy, iz);
                if let Some(label) = pocket_id[idx] {
                    let (sx, sy, sz, c) = sums[label];
                    sums[label] = (sx + xc, sy + yc, sz + zc, c + 1);
                }
            }
        }
    }

    // For each pocket, compute centroid => push a DockingInit
    // We also do a very rough bounding size by scanning the extents
    // of the pocket's grid points.
    let mut min_xyzs = vec![(f64::MAX, f64::MAX, f64::MAX); current_pocket_label + 1];
    let mut max_xyzs = vec![(f64::MIN, f64::MIN, f64::MIN); current_pocket_label + 1];

    for iz in 0..nz {
        let zc = min_z + (iz as f64) * GRID_SPACING_SITE_FINDING;
        for iy in 0..ny {
            let yc = min_y + (iy as f64) * GRID_SPACING_SITE_FINDING;
            for ix in 0..nx {
                let xc = min_x + (ix as f64) * GRID_SPACING_SITE_FINDING;
                let idx = to_index(ix, iy, iz);
                if let Some(label) = pocket_id[idx] {
                    let (min_xv, min_yv, min_zv) = min_xyzs[label];
                    let (max_xv, max_yv, max_zv) = max_xyzs[label];
                    min_xyzs[label] = (min_xv.min(xc), min_yv.min(yc), min_zv.min(zc));
                    max_xyzs[label] = (max_xv.max(xc), max_yv.max(yc), max_zv.max(zc));
                }
            }
        }
    }

    for label in 1..=current_pocket_label {
        let (sx, sy, sz, c) = sums[label];
        if c == 0 {
            continue;
        }
        let center = Vec3 {
            x: sx / (c as f64),
            y: sy / (c as f64),
            z: sz / (c as f64),
        };
        let (mnx, mny, mnz) = min_xyzs[label];
        let (mxx, mxy, mxz) = max_xyzs[label];
        // just use the largest dimension as site_box_size
        let dx = mxx - mnx;
        let dy = mxy - mny;
        let dz = mxz - mnz;
        let max_dim = dx.max(dy).max(dz);

        result.push(DockingInit {
            site_center: center,
            site_box_size: max_dim,
        });
    }

    result
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
