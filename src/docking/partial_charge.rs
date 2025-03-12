//! https://acc2.ncbr.muni.cz/
//! https://github.com/sb-ncbr/eem_parameters/

use std::f32::consts::TAU;

use barnes_hut::BodyModel;
use lin_alg::{f32::Vec3, f64::Vec3 as Vec3F64};
use nalgebra::{DMatrix, DVector};
use rand::Rng;

use crate::{
    element::Element,
    molecule::{Atom, Bond, BondCount, BondType, Molecule},
    util::setup_neighbor_pairs,
};

const GRID_SIZE: f64 = 1.6; // Slightly larger than the largest... todo: What?

#[derive(Debug)]
pub struct PartialCharge {
    pub posit: Vec3,
    pub charge: f32,
}

// todo: Ideally, you want f32 here, I believe. Need to modifify barnes_hut lib A/R to support.
impl BodyModel for PartialCharge {
    fn posit(&self) -> Vec3F64 {
        // fn posit(&self) -> Vec3 {
        self.posit.into()
    }

    fn mass(&self) -> f64 {
        // fn mass(&self) -> f32 {
        self.charge.into()
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum EemSet {
    AimB3,
    AimHf,
    MpaB3,
    MpaHf,
    NpaB3,
    NpaHf,
}

#[allow(non_snake_case)]
/// https://github.com/sb-ncbr/eem_parameters/
/// Basis set 6-311G
/// This struct represents one of the 6 sets in this EEM parameter set (Geid et al)
///
/// The inner value sare (reference electronegativity, hardness/self-repulsion parameter.
/// Hardness:  a measure of the resistance the atom has against a change in its partial charge.
/// A large value means the atom strongly resists gaining or losing electrons (it does not
/// easily develop a net charge).
pub struct EemParams {
    Br1: (f32, f32),
    /// e.g. only single bonds. tetra
    C1: (f32, f32),
    /// e.g. a double bond or partial (?) planar. E.g. alkenes or rings.
    C2: (f32, f32),
    /// E.g. a triple bond; only connected to a single atom. E.g. alkanes.
    C3: (f32, f32),
    Cl: (f32, f32),
    F: (f32, f32),
    H: (f32, f32),
    I: (f32, f32),
    N1: (f32, f32),
    N2: (f32, f32),
    N3: (f32, f32),
    O1: (f32, f32),
    O2: (f32, f32),
    P1: (f32, f32),
    P2: (f32, f32),
    S1: (f32, f32),
    S2: (f32, f32),
}

impl EemParams {
    pub fn new(set: EemSet) -> Self {
        match set {
            // AIM B3 data (from AIM B3 table)
            EemSet::AimB3 => Self {
                Br1: (2.46770000, 0.64790000),
                C1: (2.43240000, 0.31920000),
                C2: (2.44660000, 0.30330000),
                C3: (2.29030000, 0.53320000),
                Cl: (2.60260000, 0.82900000),
                F: (3.51580000, 1.93870000),
                H: (2.42380000, 0.55860000),
                I: (2.36140000, 0.85450000),
                N1: (2.59970000, 0.36080000),
                N2: (2.56910000, 0.28930000),
                N3: (2.82230000, 0.57040000),
                O1: (2.66330000, 0.38900000),
                O2: (2.67290000, 0.37700000),
                P1: (1.89340000, 0.43380000),
                P2: (1.95100000, 0.35970000),
                S1: (2.40470000, 0.29550000),
                S2: (2.43540000, 0.20830000),
            },
            // AIM Hf data
            EemSet::AimHf => Self {
                Br1: (2.47030000, 0.53870000),
                C1: (2.42460000, 0.27570000),
                C2: (2.44350000, 0.25400000),
                C3: (2.31210000, 0.42930000),
                Cl: (2.60370000, 0.68220000),
                F: (3.54390000, 1.71380000),
                H: (2.42640000, 0.50540000),
                I: (2.35920000, 0.74760000),
                N1: (2.59980000, 0.29790000),
                N2: (2.56820000, 0.24540000),
                N3: (3.16550000, 0.81880000),
                O1: (2.65750000, 0.32740000),
                O2: (2.66330000, 0.31630000),
                P1: (1.92710000, 0.37950000),
                P2: (1.84280000, 0.34420000),
                S1: (2.39530000, 0.27310000),
                S2: (2.43130000, 0.17920000),
            },
            // MPA B3 data
            EemSet::MpaB3 => Self {
                Br1: (2.34390000, 0.76870000),
                C1: (2.52040000, 0.31150000),
                C2: (2.49600000, 0.28200000),
                C3: (2.50810000, 0.21010000),
                Cl: (2.49910000, 1.00910000),
                F: (3.11060000, 1.67910000),
                H: (2.37700000, 0.71380000),
                I: (2.27430000, 0.74980000),
                N1: (2.61080000, 0.34590000),
                N2: (2.58320000, 0.38360000),
                N3: (2.55440000, 0.93730000),
                O1: (2.63710000, 0.40760000),
                O2: (2.63610000, 0.46120000),
                P1: (2.42480000, 0.16460000),
                P2: (2.14460000, 0.40450000),
                S1: (2.41210000, 0.31250000),
                S2: (2.46260000, 0.20700000),
            },
            // MPA Hf data
            EemSet::MpaHf => Self {
                Br1: (2.34280000, 0.78160000),
                C1: (2.50140000, 0.27230000),
                C2: (2.48850000, 0.24660000),
                C3: (2.51540000, 0.23820000),
                Cl: (2.52590000, 0.87310000),
                F: (3.01860000, 1.11320000),
                H: (2.37510000, 0.66210000),
                I: (2.28840000, 0.64130000),
                N1: (2.61000000, 0.30600000),
                N2: (2.56570000, 0.29110000),
                N3: (2.60770000, 0.75790000),
                O1: (2.63700000, 0.34440000),
                O2: (2.64810000, 0.39920000),
                P1: (2.40710000, 0.15080000),
                P2: (2.16540000, 0.32590000),
                S1: (2.42020000, 0.25020000),
                S2: (2.46420000, 0.18340000),
            },
            // NPA B3 data
            EemSet::NpaB3 => Self {
                Br1: (2.42440000, 0.75110000),
                C1: (2.49920000, 0.32200000),
                C2: (2.50650000, 0.31730000),
                C3: (2.46170000, 0.34890000),
                Cl: (2.51040000, 0.83640000),
                F: (3.00280000, 1.24330000),
                H: (2.38640000, 0.65810000),
                I: (2.32720000, 0.93030000),
                N1: (2.58910000, 0.40720000),
                N2: (2.55680000, 0.29490000),
                N3: (2.53480000, 0.40250000),
                O1: (2.63420000, 0.40410000),
                O2: (2.65880000, 0.42320000),
                P1: (2.38980000, 0.19020000),
                P2: (2.20980000, 0.32810000),
                S1: (2.45060000, 0.24040000),
                S2: (2.48840000, 0.20430000),
            },
            // NPA Hf data
            EemSet::NpaHf => Self {
                Br1: (2.44000000, 0.65270000),
                C1: (2.48870000, 0.30540000),
                C2: (2.49750000, 0.29610000),
                C3: (2.46130000, 0.31760000),
                Cl: (2.52250000, 0.79330000),
                F: (3.02110000, 1.19770000),
                H: (2.39150000, 0.62770000),
                I: (2.32740000, 0.92110000),
                N1: (2.59430000, 0.39200000),
                N2: (2.55000000, 0.27810000),
                N3: (2.53340000, 0.37180000),
                O1: (2.63990000, 0.39450000),
                O2: (2.65050000, 0.38190000),
                P1: (2.39290000, 0.17320000),
                P2: (2.17410000, 0.31920000),
                S1: (2.44710000, 0.23960000),
                S2: (2.49060000, 0.19500000),
            },
        }
    }
}

/// Get EEM parameters for a given atom in a molecule.
fn get_eem_params(
    atom: &Atom,
    atom_i: usize,
    bonds: &[Bond],
    adjacency_list: &[Vec<usize>],
    params: &EemParams,
) -> (f32, f32) {
    use Element::*;

    // 1) Identify the element
    let element = atom.element;

    // 2) Determine the "maximal bond order" for that atom by looking
    //    at the adjacency list or the `Bond` objects.
    //    For example:
    //      - If any bond to this atom is triple => "3"
    //      - Else if any bond is double => "2"
    //      - Else => "1"
    //    For nitrogen or oxygen, same logic: pick N1/N2/N3 or O1/O2 etc.

    let connected_bond_types = adjacency_list[atom_i].iter().map(|&nbr_idx| {
        let bond = bonds
            .iter()
            .find(|b| {
                (b.atom_0 == atom_i && b.atom_1 == nbr_idx)
                    || (b.atom_1 == atom_i && b.atom_0 == nbr_idx)
            })
            .unwrap();
        bond.bond_type
    });

    let max_order = connected_bond_types
        .map(|bond_type| match bond_type {
            BondType::Covalent { count } => match count {
                BondCount::Single => 1,
                BondCount::SingleDoubleHybrid => 2,
                BondCount::Double => 2,
                BondCount::Triple => 3,
                // Possibly handle aromatic or partial bond orders, etc.
                // or treat them like double for classification, etc.
                _ => 1,
            },
            _ => 1, // N/A.
        })
        .max()
        .unwrap_or(1);

    // 3) Return the correct EemParams for that element + bond-order classification:
    match element {
        Carbon => {
            match max_order {
                3 => params.C3, // (chi0, J0)
                2 => params.C2,
                _ => params.C1,
            }
        }
        Nitrogen => match max_order {
            3 => params.N3,
            2 => params.N2,
            _ => params.N1,
        },
        Oxygen => match max_order {
            2 => params.O2,
            _ => params.O1,
        },
        Hydrogen => params.H,
        Fluorine => params.F,
        Chlorine => params.Cl,
        Bromine => params.Br1,
        Iodine => params.I,
        Phosphorus => match max_order {
            2 => params.P2,
            _ => params.P1,
        },
        Sulfur => match max_order {
            2 => params.S2,
            _ => params.S1,
        },
        // etc. for other elements
        _ => {
            // fallback or panic
            (0.0, 0.0)
        }
    }
}

// /// Molecular Electrostatic potential
// fn kollman_mep() {
//     let layers = [1.4_f32, 1.6, 1.8, 2.0];
//
//     let n_gridpoints_per_unit_area = 1;
//     for layer_r in layers.iter() {
//         let area = 2. * TAU * layer_r.powi(2);
//         let n_grid = (n_gridpoints_per_unit_area as f32 * area) as usize;
//         // let mut grid_points = Vec::with_capacity(n_grid);
//     }
//
//     // After evaluating the MEP at all valid grid points located on all four layers, atomic charges
//     // are derived that reproduce the MEP as closely as possible."
// }

pub fn assign_eem_charges(
    atoms: &mut [Atom],
    atom_indices: &[usize],
    bonds: &[Bond],
    adjacency_list: &[Vec<usize>],
    eem_params: &EemParams,
    total_charge: f32,
) {
    let n = atoms.len();
    let dim = n + 1; // we have q_1..q_n plus lambda

    // Build matrix A and vector b
    let mut A = DMatrix::<f32>::zeros(dim, dim);
    let mut b = DVector::<f32>::zeros(dim);

    // 1) Precompute (chi0, J0) and store or build arrays
    let mut chi0 = vec![0.0; n];
    let mut j0 = vec![0.0; n];

    // for (i, atom) in molecule.atoms.iter().enumerate() {
    for (i, atom) in atoms.iter().enumerate() {
        let (c0, j) = get_eem_params(atom, atom_indices[i], bonds, adjacency_list, eem_params);
        chi0[i] = c0;
        j0[i] = j;
    }

    // 2) Fill the top-left N x N submatrix and top N of b
    for i in 0..n {
        // row i =>  A[i][i] += j0[i]  (diagonal)
        //            b[i]    = -chi0[i]
        A[(i, i)] = j0[i];
        b[i] = -chi0[i];

        // Distance-based (or adjacency-based) P(r_ij)
        for j in 0..n {
            if j == i {
                continue;
            }
            // compute distance or adjacency-based factor
            // let r_ij = (molecule.atoms[i].posit - molecule.atoms[j].posit).magnitude() as f32;
            let r_ij = (atoms[i].posit - atoms[j].posit).magnitude() as f32;
            // Some function of r_ij:
            let p_ij = k_over_r(r_ij); // define as needed

            A[(i, j)] = p_ij;
        }

        // The \(\lambda\) column is -1
        A[(i, n)] = -1.0;
    }

    // 3) Last row => sum(q_i) = total_charge
    //     A[n][0..n] = 1, b[n] = total_charge
    for i_col in 0..n {
        A[(n, i_col)] = 1.0;
    }
    // lambda's coefficient is 0
    A[(n, n)] = 0.0;

    b[n] = total_charge;

    // 4) Solve A * x = b
    let maybe_x = A.full_piv_lu().solve(&b);
    if let Some(x) = maybe_x {
        // x[0..n] = partial charges
        // x[n]    = lambda
        for i in 0..n {
            atoms[i].partial_charge = Some(x[i]);
        }
        // optionally store or ignore lambda
        // println!("Lambda = {}", x[n]);
    } else {
        eprintln!("Failed to solve EEM system!");
    }
}

/// For assigning EEM charges.
fn k_over_r(r: f32) -> f32 {
    // Example constant * 1/r function
    let k: f32 = 1.0;
    if r.abs() < 1.0e-6 {
        // avoid singularities; or use some small cutoff
        0.0
    } else {
        k / r
    }
}

/// For now, for use with EEM. `posits` is separate for different ligand pooses.
pub fn create_partial_charges(atoms: &[Atom], posits: Option<&[Vec3F64]>) -> Vec<PartialCharge> {
    let mut result = Vec::with_capacity(atoms.len());

    for (i, atom) in atoms.iter().enumerate() {
        let posit = if let Some(p) = &posits {
            p[i]
        } else {
            atom.posit
        }
        .into();

        result.push(PartialCharge {
            posit,
            charge: atom.partial_charge.unwrap_or_default(),
        })
    }

    result
}
//
// /// Create a set of partial charges around atoms. Rough simulation of electronic density imbalances
// /// in charges molecules, and/or at short distances. It places a single positive charge
// /// at the center of each atom, representing its nucleus. It spreads out a number of smaller-charges
// /// to represent electron density, around that nucleus.
// ///
// /// `charge_density` is how many partial-charge points to generate per electron;
// /// if 4.0, for instance, you’ll get ~4 negative “points” per electron.
// pub fn _create_partial_charges(atoms: &[Atom], charge_density: f32) -> Vec<PartialCharge> {
//     let mut result = Vec::new();
//     let mut rng = rand::rng();
//
//     for atom in atoms {
//         let z = atom.element.atomic_number();
//
//         // If partial_charge is known, interpret it as net formal charge:
//         // e.g., partial_charge = +1 => 1 fewer electron
//         let net_charge = atom.partial_charge.unwrap_or(0.0).round() as i32; // e.g. +1, -1, 0, ...
//         let electron_count = z as i32 - net_charge; // simplistic integer
//
//         // 1) Add nucleus partial charge, +Z or adjusted by net charge if you like.
//         //    For example, if net_charge=+1, you can either keep the nucleus as +Z
//         //    or set it to + (z - net_charge). The usual naive approach is +Z at
//         //    the nucleus, letting the negative electron distribution handle the difference.
//         result.push(PartialCharge {
//             posit: atom.posit.into(),
//             charge: z as f32,
//         });
//
//         // 2) Distribute negative charges around the nucleus
//         if electron_count > 0 {
//             let count_neg_points = (electron_count as f32 * charge_density).round() as i32;
//             if count_neg_points > 0 {
//                 let neg_charge_per_point = -(electron_count as f32) / (count_neg_points as f32);
//
//                 // Decide on a radius scale
//                 // For small molecules, you might pick something like 0.2–0.5 Å for “cloud radius”
//                 // or randomize in a band: [0, r_max].
//                 let r_max = 0.3;
//
//                 for _ in 0..count_neg_points {
//                     // Sample a random direction
//                     // We can do this by sampling spherical coordinates or
//                     // sampling a random vector from a Gaussian, then normalizing
//                     let theta = rng.random_range(0.0..TAU);
//                     let u: f32 = rng.random_range(-1.0..1.0); // for cos(phi)
//
//                     let sqrt_1_minus_u2 = (1.0 - u * u).sqrt();
//                     let dx = sqrt_1_minus_u2 * theta.cos();
//                     let dy = sqrt_1_minus_u2 * theta.sin();
//                     let dz = u;
//
//                     let dir = Vec3::new(dx, dy, dz);
//
//                     // random radius in [0..r_max]
//                     let radius = rng.random_range(0.0..r_max);
//                     let offset = dir * radius;
//
//                     let atom_posit: Vec3 = atom.posit.into();
//                     result.push(PartialCharge {
//                         posit: atom_posit + offset,
//                         charge: neg_charge_per_point,
//                     });
//                 }
//             }
//         }
//     }
//
//     result
// }
//
// /// Create a set of partial charges using a simplified Kollman approach.
// /// This routine does two things:
// ///
// /// 1. For each atom, it assigns a base partial charge from a lookup (mimicking Kollman charges).
// ///
// /// 2. For bonds that are “polar” (i.e. where the electronegativity difference exceeds a threshold),
// ///    it adds a pair of extra point charges to mimic a dipole moment along the bond.
// ///    These extra charges sum to zero, so they do not change the net charge but add a dipole.
// ///    The `dipole_offset` controls how far (in Å) from the nucleus the extra points are placed.
// pub fn _create_kollman_partial_charges(
//     atoms: &[Atom],
//     bonds: &[Bond],
//     dipole_offset: f32,
//     polar_threshold: f32, // e.g. 0.4 on the Pauling scale
// ) -> Vec<PartialCharge> {
//     let mut result = Vec::new();
//
//     // 1. Assign base Kollman charges to each atom.
//     // (These are assumed to be centered on the atomic nucleus.)
//     for atom in atoms {
//         let base_charge = atom.element.kollman_charge();
//         result.push(PartialCharge {
//             posit: atom.posit.into(),
//             charge: base_charge as f32,
//         });
//     }
//
//     // 2. For each bond, check if it is polar.
//     // If the electronegativity difference is larger than a threshold,
//     // add an extra pair of charges to mimic a dipole.
//     for bond in bonds {
//         let atom_i = &atoms[bond.atom_0];
//         let atom_j = &atoms[bond.atom_1];
//
//         let en_i = atom_i.element.electronegativity();
//         let en_j = atom_j.element.electronegativity();
//         let diff = (en_i - en_j).abs();
//
//         if diff > polar_threshold {
//             // Determine the bond vector (from i to j) and its unit vector.
//             let bond_vec = atom_j.posit - atom_i.posit;
//             let unit: Vec3 = bond_vec.to_normalized().into();
//
//             // We'll create a small dipole by placing two extra charges,
//             // one shifted by +dipole_offset and one by -dipole_offset from the midpoint of the bond.
//             // The magnitude of these extra charges is an arbitrary parameter; here we use 0.1e.
//             let extra_charge = 0.1;
//
//             // Compute the midpoint of the bond.
//             let mid: Vec3 = Vec3F64::new(
//                 (atom_i.posit.x + atom_j.posit.x) * 0.5,
//                 (atom_i.posit.y + atom_j.posit.y) * 0.5,
//                 (atom_i.posit.z + atom_j.posit.z) * 0.5,
//             )
//                 .into();
//
//             // Place the extra charges along the bond direction.
//             let offset = unit * dipole_offset;
//             result.push(PartialCharge {
//                 posit: mid + offset,
//                 charge: extra_charge,
//             });
//             result.push(PartialCharge {
//                 posit: mid - offset,
//                 charge: -extra_charge,
//             });
//         }
//     }
//
//     result
// }
//
// #[derive(Clone, Copy, Debug, PartialEq)]
// pub enum PartialChargeType {
//     Gasteiger,
//     Kollman,
// }
//
// /// Note: Hydrogens must already be added prior to adding charges.
// pub fn _setup_partial_charges(atoms: &mut [Atom], charge_type: PartialChargeType) {
//     if charge_type == PartialChargeType::Kollman {
//         unimplemented!()
//     }
//
//     // We use spacial partitioning, so as not to copmare every pair of atoms.
//     let posits: Vec<_> = atoms.iter().map(|a| &a.posit).collect();
//     let indices: Vec<_> = (0..atoms.len()).collect();
//     let neighbor_pairs = setup_neighbor_pairs(&posits, &indices, GRID_SIZE);
//
//     // Run the iterative charge update over all candidate pairs.
//     const ITERATIONS: usize = 6; // More iterations may be needed in practice.
//     for _ in 0..ITERATIONS {
//         let mut charge_updates = vec![0.0; atoms.len()];
//
//         for &(i, j) in &neighbor_pairs {
//             if atoms[i].dock_type.is_none() || atoms[j].dock_type.is_none() {
//                 continue;
//             }
//             let en_i = atoms[i].dock_type.unwrap().gasteiger_electronegativity();
//             let en_j = atoms[j].dock_type.unwrap().gasteiger_electronegativity();
//             // Compute a simple difference-based transfer.
//             let delta = 0.1 * (en_i - en_j);
//             // Transfer charge from atom i to atom j if en_i > en_j.
//             charge_updates[i] -= delta;
//             charge_updates[j] += delta;
//         }
//
//         // Apply the computed updates simultaneously.
//         for (atom, delta) in atoms.iter_mut().zip(charge_updates.iter()) {
//             match &mut atom.partial_charge {
//                 Some(c) => *c += delta,
//                 None => atom.partial_charge = Some(*delta),
//             }
//         }
//     }
// }
