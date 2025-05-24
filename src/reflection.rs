//! For displaying electron density as measured by crytalographics reflection data. From precomputed
//! data, or from Miller indices.

use std::{collections::HashMap, f64::consts::TAU, io};

use bio_apis::{ReqError, rcsb};
use lin_alg::{
    complex_nums::{Cplx, IM},
    f64::Vec3,
};

use crate::molecule::Molecule;

#[derive(Clone, Copy, PartialEq, Debug, Default)]
enum MapType {
    // M2FOFC,
    /// Ordinary, or observed
    Observed,
    // DifferenceMap,
    FreeSet,
    SystematicallyAbsent,
    OutsideHighResLimit,
    HigherThanResCutoff,
    LowerThanResCutoff,
    /// Ignored
    #[default]
    UnreliableMeasurement,
}

impl MapType {
    pub fn from_str(val: &str) -> Option<MapType> {
        match val.to_lowercase().as_ref() {
            "o" => Some(MapType::Observed),
            // "o" => Some(MapType::M2FOFC),
            // "d" => Some(MapType::DifferenceMap),
            "f" => Some(MapType::FreeSet),
            "-" => Some(MapType::SystematicallyAbsent),
            "<" => Some(MapType::OutsideHighResLimit),
            "h" => Some(MapType::HigherThanResCutoff),
            "l" => Some(MapType::LowerThanResCutoff),
            "x" => Some(MapType::UnreliableMeasurement),
            _ => {
                eprintln!("Fallthrough on map type: {val}");
                None
            }
        }
    }
}

/// Reflectdion data for a single Miller index set. Pieced together from 3 formats of CIF
/// file, or an MTZ.
#[derive(Clone, Default, Debug)]
struct Reflection {
    pub h: i32,
    pub k: i32,
    pub l: i32,
    pub status: MapType,
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
#[derive(Clone, Debug)]
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
    /// Build a `ReflectionsData` from the plain SF file plus (optionally) the
    /// 2 Fo–Fc and Fo–Fc map-coefficient CIF files.
    ///
    /// * `sf`              – contents of `*.sf.cif`
    /// * `map_2fo_fc` – optional contents of `*2fo-fc_map_coef.cif`
    /// * `map_fo_fc` – optional contents of `*fo-fc_map_coef.cif`
    pub fn new(sf: &str, map_2fo_fc: Option<&str>, map_fo_fc: Option<&str>) -> Self {
        let mut space_group = String::new();
        let mut cell_len_a = 0.;
        let mut cell_len_b = 0.;
        let mut cell_len_c = 0.;
        let mut cell_angle_alpha = 0.;
        let mut cell_angle_beta = 0.;
        let mut cell_angle_gamma = 0.;

        for line in sf.lines().map(str::trim) {
            if line.starts_with("_space_group.name_H-M_full")
                || line.starts_with("_symmetry.space_group_name_H-M")
            {
                if let Some(val) = line.split_whitespace().nth(1) {
                    space_group = val.trim_matches(&['\'', '"'][..]).to_string();
                }
            } else if line.starts_with("_cell.length_a") {
                if let Some(val) = line.split_whitespace().nth(1) {
                    cell_len_a = val.parse().unwrap_or(0.0);
                }
            } else if line.starts_with("_cell.length_b") {
                if let Some(val) = line.split_whitespace().nth(1) {
                    cell_len_b = val.parse().unwrap_or(0.0);
                }
            } else if line.starts_with("_cell.length_c") {
                if let Some(val) = line.split_whitespace().nth(1) {
                    cell_len_c = val.parse().unwrap_or(0.0);
                }
            } else if line.starts_with("_cell.angle_alpha") {
                if let Some(val) = line.split_whitespace().nth(1) {
                    cell_angle_alpha = val.parse().unwrap_or(0.0);
                }
            } else if line.starts_with("_cell.angle_beta") {
                if let Some(val) = line.split_whitespace().nth(1) {
                    cell_angle_beta = val.parse().unwrap_or(0.0);
                }
            } else if line.starts_with("_cell.angle_gamma") {
                if let Some(val) = line.split_whitespace().nth(1) {
                    cell_angle_gamma = val.parse().unwrap_or(0.0);
                }
            }
        }

        let mut reflections = Vec::<Reflection>::new();
        let mut column_tags = Vec::<String>::new();
        let mut in_loop = false;

        for line in sf.lines().map(str::trim) {
            if line == "loop_" {
                in_loop = true;
                column_tags.clear();
                continue;
            }

            if !in_loop {
                continue;
            }

            // Header lines inside the loop
            if line.starts_with('_') {
                column_tags.push(line.split_whitespace().next().unwrap().to_string());
                continue;
            }

            // Once the first data row appears, we know whether this is *the*
            // reflection loop (it contains _refln.index_h)
            if !column_tags.iter().any(|t| t == "_refln.index_h") {
                in_loop = false; // some other loop – skip it
                continue;
            }

            let vals: Vec<&str> = line.split_whitespace().collect();
            if vals.len() != column_tags.len() {
                in_loop = false; // loop ended
                continue;
            }

            let idx = |tag: &str| column_tags.iter().position(|t| t == tag).unwrap();

            let h = vals[idx("_refln.index_h")].parse().unwrap_or(0);
            let k = vals[idx("_refln.index_k")].parse().unwrap_or(0);
            let l = vals[idx("_refln.index_l")].parse().unwrap_or(0);
            let stat = MapType::from_str(vals[idx("_refln.status")]).unwrap_or_default();
            let amp = vals[idx("_refln.F_meas_au")].parse().unwrap_or(0.0);
            let sig = vals[idx("_refln.F_meas_sigma_au")].parse().unwrap_or(0.0);

            reflections.push(Reflection {
                h,
                k,
                l,
                status: stat,
                amp,
                amp_uncertainty: sig,
                amp_weighted: None,
                phase_weighted: None,
                phase_figure_of_merit: None,
                delta_amp_weighted: None,
                delta_phase_weighted: None,
                delta_figure_of_merit: None,
            });
        }

        let mut lookup: HashMap<(i32, i32, i32), usize> = HashMap::new();
        for (i, r) in reflections.iter().enumerate() {
            lookup.insert((r.h, r.k, r.l), i);
        }

        fn merge_map(
            map_text: &str,
            reflections: &mut [Reflection],
            lookup: &HashMap<(i32, i32, i32), usize>,
            amp_tag: &str,
            phase_tag: &str,
            fom_tag: &str,
        ) {
            let mut in_loop = false;
            let mut tags = Vec::<String>::new();

            for line in map_text.lines().map(str::trim) {
                if line == "loop_" {
                    in_loop = true;
                    tags.clear();
                    continue;
                }
                if !in_loop {
                    continue;
                }

                if line.starts_with('_') {
                    tags.push(line.split_whitespace().next().unwrap().to_string());
                    continue;
                }

                if !tags.iter().any(|t| t == "_refln.index_h") {
                    in_loop = false;
                    continue;
                }

                let vals: Vec<&str> = line.split_whitespace().collect();
                if vals.len() != tags.len() {
                    in_loop = false;
                    continue;
                }

                let idx = |tag: &str| tags.iter().position(|t| t == tag).unwrap();

                let h = vals[idx("_refln.index_h")].parse().unwrap_or(0);
                let k = vals[idx("_refln.index_k")].parse().unwrap_or(0);
                let l = vals[idx("_refln.index_l")].parse().unwrap_or(0);

                if let Some(&pos) = lookup.get(&(h, k, l)) {
                    let rec = &mut reflections[pos];

                    let parse_opt = |s: &str| {
                        if s == "?" || s == "." {
                            None
                        } else {
                            s.parse::<f64>().ok()
                        }
                    };

                    if let Some(col) = tags.iter().position(|t| t == amp_tag) {
                        rec.amp_weighted = parse_opt(vals[col]);
                    }
                    if let Some(col) = tags.iter().position(|t| t == phase_tag) {
                        rec.phase_weighted = parse_opt(vals[col]);
                    }
                    if let Some(col) = tags.iter().position(|t| t == fom_tag) {
                        rec.phase_figure_of_merit = parse_opt(vals[col]);
                    }

                    // Special case for Fo–Fc maps
                    if amp_tag == "_refln.pdbx_DELFWT" {
                        if let Some(col) = tags.iter().position(|t| t == amp_tag) {
                            rec.delta_amp_weighted = parse_opt(vals[col]);
                        }
                        if let Some(col) = tags.iter().position(|t| t == phase_tag) {
                            rec.delta_phase_weighted = parse_opt(vals[col]);
                        }
                        if let Some(col) = tags.iter().position(|t| t == fom_tag) {
                            rec.delta_figure_of_merit = parse_opt(vals[col]);
                        }
                    }
                }
            }
        }

        if let Some(txt) = map_2fo_fc {
            merge_map(
                txt,
                &mut reflections,
                &lookup,
                "_refln.pdbx_FWT",
                "_refln.pdbx_PHWT",
                "_refln.fom",
            );
        }
        if let Some(txt) = map_fo_fc {
            merge_map(
                txt,
                &mut reflections,
                &lookup,
                "_refln.pdbx_DELFWT",
                "_refln.pdbx_DELPHWT",
                "_refln.fom",
            );
        }

        Self {
            space_group,
            cell_len_a,
            cell_len_b,
            cell_len_c,
            cell_angle_alpha,
            cell_angle_beta,
            cell_angle_gamma,
            points: reflections,
        }
    }

    /// Load reflections data from RCSB, then parse.
    pub fn load(ident: &str) -> Result<Self, ReqError> {
        let sf = rcsb::load_structure_factors_cif(ident)?;

        // todo: Only attempt these maps if we've established as available?

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

        Ok(Self::new(&sf, map_2fo_fc.as_deref(), map_fo_fc.as_deref()))
    }
}

pub struct ElectronDensity {
    pub coords: Vec3,
    pub density: f64,
}

// /// Compute electron density at fractional coordinate (x, y, z), from Miller indices.
// fn compute_density(reflections: &[Reflection], posit: Vec3) -> f64 {
//     let mut real = 0.0;
//     let mut imag = 0.0;
//
//     for r in reflections {
//         let phase_rad = r.phase_weighted.to_radians();
//         let arg = -TAU * (r.h as f64 * posit.x + r.k as f64 * posit.y + r.l as f64 * posit.z);
//         let exp = Cplx::from_mag_phase(1.0, arg);
//         let contrib = Cplx::from_mag_phase(1.0, phase_rad) * r.amp * exp;
//
//         real += contrib.real;
//         imag += contrib.im;
//     }
//
//     // Return the real part (imaginary part should ideally cancel)
//     real
// }

pub fn compute_density_grid(data: &ReflectionsData, points: &[Vec3]) -> Vec<f64> {
    let mut result = Vec::new();

    result
}
