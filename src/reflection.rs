//! For displaying electron density as measured by crytalographics reflection data. From precomputed
//! data, or from Miller indices.

use std::{collections::HashMap, f64::consts::TAU, io, time::Instant};

use bio_apis::{ReqError, rcsb};
use lin_alg::{
    complex_nums::{Cplx, IM},
    f64::Vec3,
};
use rayon::prelude::*;

use crate::molecule::Molecule;

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
    /// Load reflections data from RCSB, then parse. (SF, 2fo_fc, and fo_fc)
    pub fn load_from_rcsb(ident: &str) -> Result<Self, ReqError> {
        let sf = rcsb::load_structure_factors_cif(ident)?;

        // todo: Only attempt these maps if we've established as available?

        println!("Downloading reflections data for {ident}...");
        let map_2fo_fc = match rcsb::load_validation_2fo_fc_cif(ident) {
            Ok(m) => Some(m),
            Err(_) => {
                eprintln!("Error loading 2fo_fc map");
                None
            }
        };

        let map_fo_fc = match rcsb::load_validation_fo_fc_cif(ident) {
            Ok(m) => Some(m),
            Err(_) => {
                eprintln!("Error loading fo_fc map");
                None
            }
        };

        println!("Download complete. Parsing...");
        Ok(Self::from_cifs(
            &sf,
            map_2fo_fc.as_deref(),
            map_fo_fc.as_deref(),
        ))
    }

    //     /// 1. Make a regular fractional grid that spans 0–1 along a, b, c.
    //     /// We use this grid for computing electron densitites; it must be converted to real space,
    //     /// e.g. in angstroms, prior to display.
    //     pub fn regular_fractional_grid(&self, n: usize) -> Vec<Vec3> {
    //         let mut pts = Vec::with_capacity(n.pow(3));
    //         let step = 1. / n as f64;
    //
    //         for i in 0..n {
    //             for j in 0..n {
    //                 for k in 0..n {
    //                     pts.push(Vec3 {
    //                         x: i as f64 * step,
    //                         y: j as f64 * step,
    //                         z: k as f64 * step,
    //                     });
    //                 }
    //             }
    //         }
    //
    //         pts
    //     }
    // }

    /// 1. Make a regular fractional grid that spans 0–1 along a, b, c.
    /// We use this grid for computing electron densitites; it must be converted to real space,
    /// e.g. in angstroms, prior to display.
    pub fn regular_fractional_grid(&self, n: usize) -> Vec<Vec3> {
        let step = 1.0 / n as f64;
        let shift = -0.5 + step / 2.0; // put voxel centres at –½…+½
        let mut pts = Vec::with_capacity(n.pow(3));

        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    pts.push(Vec3 {
                        x: i as f64 * step + shift,
                        y: j as f64 * step + shift,
                        z: k as f64 * step + shift,
                    });
                }
            }
        }

        pts
    }
}

#[derive(Clone, Debug)]
pub struct ElectronDensity {
    /// In Å
    pub coords: Vec3,
    /// Noramlized, using the unit cell volume, as reported in the reflection data.
    pub density: f64,
}

fn compute_density(reflections: &[Reflection], posit: Vec3, unit_cell_vol: f32) -> f64 {
    // todo: Use SIMD or GPU for this.

    const EPS: f64 = 0.0000001;
    let mut rho = 0.0;

    for r in reflections {
        if r.status != MapStatus::Observed {
            continue;
        }

        let amp = r.amp_weighted.unwrap_or(r.amp);
        if amp.abs() < EPS {
            continue;
        }

        let Some(phase) = r.phase_weighted else {
            continue;
        };

        //  2π(hx + ky + lz)  (negative sign because CCP4/Coot convention)
        let arg = TAU * (r.h as f64 * posit.x + r.k as f64 * posit.y + r.l as f64 * posit.z);
        //  real part of  F · e^{iφ} · e^{iarg} = amp·cos(φ+arg)

        // todo: Which sign/order?
        // rho += amp * (arg + phase.to_radians()).cos();
        rho += amp * (arg - phase.to_radians()).cos();
    }

    // Normalize.
    rho / unit_cell_vol as f64
}

/// Compute electron density from reflection data.
pub fn compute_density_grid(data: &ReflectionsData) -> Vec<ElectronDensity> {
    let grid = data.regular_fractional_grid(90);
    let unit_cell_vol = data.cell_len_a * data.cell_len_b * data.cell_len_c;

    println!(
        "Computing electron density from refletions onver {} points...",
        grid.len()
    );

    let start = Instant::now();

    let len_a = data.cell_len_a as f64;
    let len_b = data.cell_len_b as f64;
    let len_c = data.cell_len_c as f64;

    let result = grid
        .par_iter()
        .map(|p| ElectronDensity {
            // coords: *p,
            // Convert coords to real space, in angstroms.
            coords: Vec3 {
                x: p.x * len_a,
                y: p.y * len_b,
                z: p.z * len_c,
            },
            // coords: frac_to_cart(
            //     *p,
            //     len_a,
            //     len_b,
            //     len_c,
            //     (data.cell_angle_alpha as f64).to_radians(),
            //     (data.cell_angle_beta as f64).to_radians(),
            //     (data.cell_angle_gamma as f64).to_radians(),
            // ),
            density: compute_density(&data.points, *p, unit_cell_vol),
        })
        .collect();

    let elapsed = start.elapsed().as_millis();

    println!("Complete. Time: {:?}ms", elapsed);
    result
}

/// Convert from fractical coordinates, as used in reflections, to real space in Angstroms.
fn frac_to_cart(fr: Vec3, a: f64, b: f64, c: f64, α: f64, β: f64, γ: f64) -> Vec3 {
    // Angles in radians
    let (ca, cb, cg) = (α.cos(), β.cos(), γ.cos());
    let sg = γ.sin();

    // Volume factor
    let v = (1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg).sqrt();

    // Orthogonalisation matrix (PDB convention 1)
    let ox = Vec3 {
        x: a,
        y: 0.0,
        z: 0.0,
    };
    let oy = Vec3 {
        x: b * cg,
        y: b * sg,
        z: 0.0,
    };
    let oz = Vec3 {
        x: c * cb,
        y: c * (ca - cb * cg) / sg,
        z: c * v / sg,
    };

    Vec3 {
        x: ox.x * fr.x + oy.x * fr.y + oz.x * fr.z,
        y: ox.y * fr.x + oy.y * fr.y + oz.y * fr.z,
        z: ox.z * fr.x + oy.z * fr.y + oz.z * fr.z,
    }
}

// /// Intermediate struct required by the IsoSurface lib.
// struct Source {
//
// }
//
// fn make_mesh(density: &[ElectronDensity], iso_val: f32)