//! For displaying electron density as measured by crytalographics reflection data. From precomputed
//! data, or Miller indices.
//!
//! Note: We currently

#![allow(unused)]

use std::{f64::consts::TAU, sync::Arc, time::Instant};

use bio_apis::{ReqError, rcsb};
use bio_files::{DensityMap, MapHeader, UnitCell};
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaModule, CudaStream, LaunchConfig, PushKernelArg};
use dynamics::ForcesOnWaterMol;
#[cfg(feature = "cuda")]
use lin_alg::f32::{vec3s_from_dev, vec3s_to_dev};
use lin_alg::{f32::Vec3 as Vec3F32, f64::Vec3};
use mcubes::GridPoint;
use rayon::prelude::*;

use crate::{ComputationDevice, molecule::Atom, util::setup_neighbor_pairs};

pub const DENSITY_CELL_MARGIN: f64 = 2.0;

// Density points must be within this distance in Å of a protein atom to be generated.
// This prevents displaying shapes from the neighbor
pub const DENSITY_MAX_DIST: f64 = 4.;
const DIST_TO_ATOMS_SAMPLE_RATIO: usize = 5;

#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum MapStatus {
    /// Ordinary, or observed; the bulk of values.
    Observed,
    FreeSet,
    SystematicallyAbsent,
    OutsideHighResLimit,
    HigherThanResCutoff,
    LowerThanResCutoff,
    /// Ignored
    #[default]
    UnreliableMeasurement,
}

impl MapStatus {
    pub fn from_str(val: &str) -> Option<MapStatus> {
        match val.to_lowercase().as_ref() {
            "o" => Some(MapStatus::Observed),
            // "o" => Some(MapType::M2FOFC),
            // "d" => Some(MapType::DifferenceMap),
            "f" => Some(MapStatus::FreeSet),
            "-" => Some(MapStatus::SystematicallyAbsent),
            "<" => Some(MapStatus::OutsideHighResLimit),
            "h" => Some(MapStatus::HigherThanResCutoff),
            "l" => Some(MapStatus::LowerThanResCutoff),
            "x" => Some(MapStatus::UnreliableMeasurement),
            _ => {
                eprintln!("Fallthrough on map type: {val}");
                None
            }
        }
    }
}

#[allow(unused)]
/// Reflection data for a single Miller index set. Pieced together from 3 formats of CIF
/// file (Structure factors, map 2fo-fc, and map fo-fc), or an MTZ.
#[derive(Clone, Default, Debug)]
pub struct Reflection {
    /// Miller indices.
    pub h: i32,
    pub k: i32,
    pub l: i32,
    pub status: MapStatus,
    /// Amplitude. i.e. F_meas. From SF.
    pub amp: f64,
    /// Standard uncertainty (σ) of amplitude. i.e. F_meas_sigma_au. From SF.
    pub amp_uncertainty: f64,
    /// ie. FWT. From 2fo-fc.
    pub amp_weighted: Option<f64>,
    /// i.e. PHWT. In degrees. From 2fo-fc.
    pub phase_weighted: Option<f64>,
    /// i.e. FOM. From 2fo-fc.
    pub phase_figure_of_merit: Option<f64>,
    /// From fo-fc.
    pub delta_amp_weighted: Option<f64>,
    /// From fo-fc.
    pub delta_phase_weighted: Option<f64>,
    /// From fo-fc.
    pub delta_figure_of_merit: Option<f64>,
}

/// Miller-index-based reflection data.
#[derive(Clone, Debug, Default)]
pub struct ReflectionsData {
    /// X Y Z for a b c?
    pub space_group: String,
    pub cell_len_a: f32,
    pub cell_len_b: f32,
    pub cell_len_c: f32,
    pub cell_angle_alpha: f32,
    pub cell_angle_beta: f32,
    pub cell_angle_gamma: f32,
    pub points: Vec<Reflection>,
}

impl ReflectionsData {
    // /// Load reflections data from RCSB, then parse. (SF, 2fo_fc, and fo_fc)
    // pub fn load_from_rcsb(ident: &str) -> Result<Self, ReqError> {
    //     println!("Downloading structure factors and Map data for {ident}...");
    //
    //     let sf = match rcsb::load_structure_factors_cif(ident) {
    //         Ok(m) => Some(m),
    //         Err(_) => {
    //             eprintln!("Error loading structure factors CIF");
    //             None
    //         }
    //     };
    //
    //     let map_2fo_fc = match rcsb::load_validation_2fo_fc_cif(ident) {
    //         Ok(m) => Some(m),
    //         Err(_) => {
    //             eprintln!("Error loading 2fo_fc map");
    //             None
    //         }
    //     };
    //
    //     let map_fo_fc = match rcsb::load_validation_fo_fc_cif(ident) {
    //         Ok(m) => Some(m),
    //         Err(_) => {
    //             eprintln!("Error loading fo_fc map");
    //             None
    //         }
    //     };
    //
    //     println!("Download complete. Parsing...");
    //     Ok(Self::from_cifs(
    //         sf.as_deref(),
    //         map_2fo_fc.as_deref(),
    //         map_fo_fc.as_deref(),
    //     ))
    // }
}

#[derive(Clone, Debug)]
pub struct ElectronDensity {
    /// In Å
    pub coords: Vec3,
    /// Normalized, using the unit cell volume, as reported in the reflection data.
    pub density: f64,
}

impl GridPoint for ElectronDensity {
    // fn coords(&self) -> Vec3 {self.coords}
    fn value(&self) -> f64 {
        self.density
    }
}

/// One dense 3-D brick of map values. We use this struct to handle symmetry: ensuring full coverage
/// of all atoms.
#[derive(Clone, Debug)]
pub struct DensityRect {
    /// Cartesian coordinate of the *centre* of voxel (0,0,0)
    pub origin_cart: Vec3,
    /// Size of one voxel along a,b,c in Å
    pub step: [f64; 3],
    /// (nx, ny, nz) – number of voxels stored
    pub dims: [usize; 3],
    /// Row-major file-order data: z → y → x fastest
    pub data: Vec<f32>,
}

impl DensityRect {
    /// Extract the smallest cube that covers all atoms plus `margin` Å.
    /// `margin = 0.0` means “touch each atom’s centre”.
    pub fn new(atom_posits: &[Vec3], map: &DensityMap, margin: f64) -> Self {
        let hdr = &map.hdr;
        let cell = &map.cell;

        // Atom bounds in fractional coords, relative to map origin
        let mut min_r = Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut max_r = Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);

        for p in atom_posits {
            // (a) Cartesian → absolute fractional
            let mut f = cell.cartesian_to_fractional(*p);
            // (b) shift so that origin_frac becomes (0,0,0)
            f -= map.origin_frac;

            // keep *unwrapped* values (they can be <0 or >1)
            min_r = Vec3::new(min_r.x.min(f.x), min_r.y.min(f.y), min_r.z.min(f.z));
            max_r = Vec3::new(max_r.x.max(f.x), max_r.y.max(f.y), max_r.z.max(f.z));
        }

        // Extra margin in fractional units
        let margin_r = Vec3::new(margin / cell.a, margin / cell.b, margin / cell.c);
        min_r -= margin_r;
        max_r += margin_r;

        // Convert fractional to voxel indices
        let to_idx = |fr: f64, n: i32| -> isize { (fr * n as f64 - 0.5).floor() as isize };

        let lo_i = [
            to_idx(min_r.x, hdr.mx),
            to_idx(min_r.y, hdr.my),
            to_idx(min_r.z, hdr.mz),
        ];
        let hi_i = [
            to_idx(max_r.x, hdr.mx),
            to_idx(max_r.y, hdr.my),
            to_idx(max_r.z, hdr.mz),
        ];

        // inclusive → dims       (now guaranteed hi_i ≥ lo_i)
        let dims = [
            (hi_i[0] - lo_i[0] + 1) as usize,
            (hi_i[1] - lo_i[1] + 1) as usize,
            (hi_i[2] - lo_i[2] + 1) as usize,
        ];

        let lo_frac = Vec3::new(
            (lo_i[0] as f64 + 0.5) / hdr.mx as f64,
            (lo_i[1] as f64 + 0.5) / hdr.my as f64,
            (lo_i[2] as f64 + 0.5) / hdr.mz as f64,
        ) + map.origin_frac; // back to absolute fractional

        let origin_cart = cell.fractional_to_cartesian(lo_frac);

        // Voxel step vectors in Å
        let step = [
            cell.a / hdr.mx as f64,
            cell.b / hdr.my as f64,
            cell.c / hdr.mz as f64,
        ];

        let mut data = Vec::with_capacity(dims[0] * dims[1] * dims[2]);

        for kz in 0..dims[2] {
            for ky in 0..dims[1] {
                for kx in 0..dims[0] {
                    let idx_c = [
                        lo_i[0] + kx as isize,
                        lo_i[1] + ky as isize,
                        lo_i[2] + kz as isize,
                    ];

                    // Crystallographic → Cartesian center of this voxel
                    let frac = map.origin_frac
                        + Vec3::new(
                            (idx_c[0] as f64 + 0.5) / hdr.mx as f64,
                            (idx_c[1] as f64 + 0.5) / hdr.my as f64,
                            (idx_c[2] as f64 + 0.5) / hdr.mz as f64,
                        );
                    let cart = cell.fractional_to_cartesian(frac);

                    let density = map.density_at_point_trilinear(cart);
                    let dens_sig = map.density_to_sig(density);
                    data.push(dens_sig);
                }
            }
        }

        Self {
            origin_cart,
            step,
            dims,
            data,
        }
    }

    /// Convert a DensityRect into (coords, density) structs, which represent density over 3D space,
    /// and can be visualized. It is computationally intensive currently, due to our filter for only
    /// items near the atom coordinates. This may speed up rendering downstream, and declutter the view.
    ///
    /// We assume `atom_posits` has been filtered to not include Hydrogens, for performance reasons.
    pub fn make_densities(
        &self,
        dev: &ComputationDevice,
        atom_posits: &[Vec3],
        cell: &UnitCell,
        dist_thresh: f64,
    ) -> Vec<ElectronDensity> {
        // todo: Use GPU for this. It is very slow for large sets.
        println!("Making electron densities...");
        let start = Instant::now();

        // Step vectors along a, b, c.
        let cols = cell.ortho.to_cols();

        // length of one voxel along the a-axis in Å = a / mx
        let step_vec_a = cols.0 * (self.step[0] / cell.a); //  = a_vec / mx
        let step_vec_b = cols.1 * (self.step[1] / cell.b); //  = b_vec / my
        let step_vec_c = cols.2 * (self.step[2] / cell.c); //  = c_vec / mz
        let step_vecs = (step_vec_a, step_vec_b, step_vec_c);

        let (nx, ny, nz) = (self.dims[0], self.dims[1], self.dims[2]);

        let mut triplets = Vec::with_capacity(nx * ny * nz);
        for kx in 0..nx {
            for ky in 0..ny {
                for kz in 0..nz {
                    triplets.push((kx, ky, kz));
                }
            }
        }

        // Convert to f64, and don't example every atom. The latter reduces computation time, roughly
        // by a factor of `SAMPLE_RATIO`.
        let atom_posits_sample: Vec<Vec3> = atom_posits
            .iter()
            .enumerate()
            .filter(|(i, _)| i % DIST_TO_ATOMS_SAMPLE_RATIO == 0)
            .map(|(_, a)| (*a).into())
            .collect();

        // Note: We get a big speedup from both Rayon, and then GPU on top of that. e.g.:
        // CPU without rayon: 20,000ms
        // CPU, with rayon (No SIMD): 780ms
        // GPU: 54ms
        let out = match dev {
            ComputationDevice::Cpu => self.make_densities_inner(
                triplets,
                &atom_posits_sample,
                step_vecs,
                dist_thresh,
                nx,
                ny,
            ),
            #[cfg(feature = "cuda")]
            ComputationDevice::Gpu((stream, module)) => self.make_densities_inner_gpu(
                stream,
                module,
                triplets,
                &atom_posits_sample,
                step_vecs,
                dist_thresh.powi(2),
                nx,
                ny,
            ),
        };

        let elapsed = start.elapsed().as_millis();
        println!("Complete, in {elapsed} ms");

        out
    }

    #[cfg(feature = "cuda")]
    /// Separate helper, to isolate from the CPU version.
    fn make_densities_inner_gpu(
        &self,
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        triplets: Vec<(usize, usize, usize)>,
        atom_posits: &[Vec3],
        step_vecs: (Vec3, Vec3, Vec3),
        dist_thresh: f64,
        nx: usize,
        ny: usize,
    ) -> Vec<ElectronDensity> {
        let n = triplets.len();
        let n_atom_posits = atom_posits.len();

        let mut coords_gpu = {
            let v = vec![Vec3F32::new_zero(); n];
            vec3s_to_dev(stream, &v)
        };

        let mut densities_gpu = stream.memcpy_stod(&vec![0.0f32; n]).unwrap();

        let triplets_gpu = {
            let mut trip_flat = Vec::with_capacity(n * 3);
            for (x, y, z) in triplets {
                trip_flat.push(x as u32);
                trip_flat.push(y as u32);
                trip_flat.push(z as u32);
            }
            stream.memcpy_stod(&trip_flat).unwrap()
        };

        let atom_posits_gpu = {
            let p_f32: Vec<Vec3F32> = atom_posits.iter().map(|a| (*a).into()).collect();
            vec3s_to_dev(stream, &p_f32)
        };

        // Convert vec3s to f32.
        let step_vecs_0: Vec3F32 = step_vecs.0.into();
        let step_vecs_1: Vec3F32 = step_vecs.1.into();
        let step_vecs_2: Vec3F32 = step_vecs.2.into();
        let origin_f32: Vec3F32 = self.origin_cart.into();

        let data_gpu = stream.memcpy_stod(&self.data).unwrap();

        let dist_thresh = dist_thresh as f32;

        let kernel = module.load_function("make_densities_kernel").unwrap();

        let cfg = LaunchConfig::for_num_elems(n as u32);

        let mut launch_args = stream.launch_builder(&kernel);

        launch_args.arg(&mut coords_gpu);
        launch_args.arg(&mut densities_gpu);
        //
        launch_args.arg(&triplets_gpu);
        launch_args.arg(&atom_posits_gpu);
        launch_args.arg(&data_gpu);
        //
        launch_args.arg(&step_vecs_0);
        launch_args.arg(&step_vecs_1);
        launch_args.arg(&step_vecs_2);
        launch_args.arg(&origin_f32);
        //
        launch_args.arg(&dist_thresh);
        launch_args.arg(&nx);
        launch_args.arg(&ny);
        launch_args.arg(&n);
        launch_args.arg(&n_atom_posits);

        unsafe { launch_args.launch(cfg) }.unwrap();

        // todo: Consider dtoh; passing to an existing vec instead of re-allocating?
        let coords = vec3s_from_dev(stream, &coords_gpu);
        let densities = stream.memcpy_dtov(&densities_gpu).unwrap();

        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            result.push(ElectronDensity {
                coords: coords[i].into(),
                density: densities[i] as f64,
            })
        }

        result
    }

    /// Separate helper, to isolate from the GPU version.
    fn make_densities_inner(
        &self,
        triplets: Vec<(usize, usize, usize)>,
        atom_posits: &[Vec3],
        step_vecs: (Vec3, Vec3, Vec3),
        dist_thresh: f64,
        nx: usize,
        ny: usize,
    ) -> Vec<ElectronDensity> {
        let dist_thresh_sq = dist_thresh * dist_thresh;

        // Note: We get a big speedup from using rayon here. For example, 200ms vs 5s, or 2.5s vs 70s
        // for a larger file. (9950x CPU)
        triplets
            .par_iter()
            .map(|&(kx, ky, kz)| {
                // Linear index in self.data  (z → y → x fastest)
                let i_data = (kz * ny + ky) * nx + kx;
                let mut density = self.data[i_data] as f64;

                // Cartesian center of this voxel
                let coords = self.origin_cart
                    + step_vecs.0 * kx as f64
                    + step_vecs.1 * ky as f64
                    + step_vecs.2 * kz as f64;

                // todo: Too slow, but can work for now.
                // We don't want to set up density points not near the atom coordinates.
                let mut nearest_dist_sq = f64::MAX;
                for p in atom_posits {
                    let dist_sq = (*p - coords).magnitude_squared();
                    if dist_sq < nearest_dist_sq {
                        nearest_dist_sq = dist_sq;
                    }
                }

                if nearest_dist_sq > dist_thresh_sq {
                    // We set density to 0, vice removing the coordinates; our marching cubes
                    // algorithm requires a regular grid, with no absent values.
                    density = 0.;
                }

                ElectronDensity { coords, density }
            })
            .collect()
    }
}
