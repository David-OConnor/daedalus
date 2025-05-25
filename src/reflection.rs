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
enum MapStatus {
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
struct Reflection {
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
        use std::{collections::HashMap, str::FromStr}; // brings MapType::from_str into scope

        let mut reflections: HashMap<(i32, i32, i32), Reflection> = HashMap::new();

        let mut space_group = String::new();
        let mut a = 0.0_f32;
        let mut b = 0.0_f32;
        let mut c = 0.0_f32;
        let mut α = 0.0_f32;
        let mut β = 0.0_f32;
        let mut γ = 0.0_f32;

        /* === 1. generic one-file parser ====================================== */
        fn parse_file(
            text: &str,
            store: &mut HashMap<(i32, i32, i32), Reflection>,
            precedence: u8, // 0 = lowest, 2 = highest (SF)
            is_sf: bool,
            header_sink: &mut dyn FnMut(&str, &str),
        ) {
            let mut in_loop = false;
            let mut tags: Vec<String> = Vec::new();

            for raw in text.lines() {
                let line = raw.trim();

                /* ---- harvest header (cell, space group) --------------------- */
                if line.starts_with('_') && !line.starts_with("_refln.") {
                    if let Some((tag, val)) = line.split_once(char::is_whitespace) {
                        header_sink(tag, val.trim());
                    }
                }

                if line == "loop_" {
                    in_loop = true;
                    tags.clear();
                    continue;
                }
                if !in_loop {
                    continue;
                }

                if line.starts_with('_') {
                    tags.push(line.split_whitespace().next().unwrap().to_owned());
                    continue;
                }

                /* ensure this loop is a reflection loop */
                if !tags.iter().any(|t| t == "_refln.index_h") {
                    in_loop = false;
                    continue;
                }

                let cols: Vec<&str> = line.split_whitespace().collect();
                if cols.len() != tags.len() {
                    in_loop = false;
                    continue;
                }

                let col = |tag: &str| tags.iter().position(|t| t == tag);

                let h = cols[col("_refln.index_h").unwrap()]
                    .parse::<i32>()
                    .unwrap_or(0);
                let k = cols[col("_refln.index_k").unwrap()]
                    .parse::<i32>()
                    .unwrap_or(0);
                let l = cols[col("_refln.index_l").unwrap()]
                    .parse::<i32>()
                    .unwrap_or(0);
                let key = (h, k, l);

                let rec = store.entry(key).or_insert_with(|| Reflection {
                    h,
                    k,
                    l,
                    ..Default::default()
                });

                /* precedence helpers */
                let overwrite_f64 = |dst: &mut f64, src: Option<f64>| {
                    if let Some(v) = src {
                        if is_sf || *dst == 0.0 {
                            *dst = v;
                        }
                    }
                };
                let overwrite_opt = |dst: &mut Option<f64>, src: Option<f64>| {
                    if let Some(v) = src {
                        if dst.is_none() || is_sf {
                            *dst = Some(v);
                        }
                    }
                };

                /* status (safe) */
                if let Some(i) = col("_refln.status") {
                    rec.status = MapStatus::from_str(cols[i]).unwrap_or_default();
                }

                /* amplitude & sigma */
                overwrite_f64(
                    &mut rec.amp,
                    col("_refln.F_meas_au").and_then(|i| cols[i].parse().ok()),
                );
                overwrite_f64(
                    &mut rec.amp_uncertainty,
                    col("_refln.F_meas_sigma_au").and_then(|i| cols[i].parse().ok()),
                );

                /* 2Fo-Fc coeffs */
                overwrite_opt(
                    &mut rec.amp_weighted,
                    col("_refln.pdbx_FWT").and_then(|i| cols[i].parse().ok()),
                );
                overwrite_opt(
                    &mut rec.phase_weighted,
                    col("_refln.pdbx_PHWT").and_then(|i| cols[i].parse().ok()),
                );
                overwrite_opt(
                    &mut rec.phase_figure_of_merit,
                    col("_refln.fom").and_then(|i| cols[i].parse().ok()),
                );

                /* Fo-Fc coeffs */
                overwrite_opt(
                    &mut rec.delta_amp_weighted,
                    col("_refln.pdbx_DELFWT").and_then(|i| cols[i].parse().ok()),
                );
                overwrite_opt(
                    &mut rec.delta_phase_weighted,
                    col("_refln.pdbx_DELPHWT").and_then(|i| cols[i].parse().ok()),
                );
                overwrite_opt(
                    &mut rec.delta_figure_of_merit,
                    col("_refln.fom").and_then(|i| cols[i].parse().ok()),
                );
            }
        }

        let mut header = |tag: &str, val: &str| match tag {
            "_space_group.name_H-M_full" | "_symmetry.space_group_name_H-M" => {
                if space_group.is_empty() {
                    space_group = val.trim_matches(&['"', '\''][..]).to_owned();
                }
            }
            "_cell.length_a" => a = val.parse().unwrap_or(0.0),
            "_cell.length_b" => b = val.parse().unwrap_or(0.0),
            "_cell.length_c" => c = val.parse().unwrap_or(0.0),
            "_cell.angle_alpha" => α = val.parse().unwrap_or(0.0),
            "_cell.angle_beta" => β = val.parse().unwrap_or(0.0),
            "_cell.angle_gamma" => γ = val.parse().unwrap_or(0.0),
            _ => {}
        };

        /* === 3. parse files in ascending precedence ========================== */
        if let Some(txt) = map_fo_fc {
            parse_file(txt, &mut reflections, 0, false, &mut header);
        }
        if let Some(txt) = map_2fo_fc {
            parse_file(txt, &mut reflections, 1, false, &mut header);
        }
        parse_file(sf, &mut reflections, 2, true, &mut header); // SF wins ties

        /* === 4. assemble result ============================================== */
        Self {
            space_group,
            cell_len_a: a,
            cell_len_b: b,
            cell_len_c: c,
            cell_angle_alpha: α,
            cell_angle_beta: β,
            cell_angle_gamma: γ,
            points: reflections.into_values().collect(),
        }
    }

    /// Load reflections data from RCSB, then parse.
    pub fn load(ident: &str) -> Result<Self, ReqError> {
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
        Ok(Self::new(&sf, map_2fo_fc.as_deref(), map_fo_fc.as_deref()))
    }

    /// 1. Make a regular fractional grid that spans 0–1 along a, b, c.
    /// We use this grid for computing electron densitites; it must be converted to real space,
    /// e.g. in angstroms, prior to display.
    pub fn regular_fractional_grid(&self, n: usize) -> Vec<Vec3> {
        let mut pts = Vec::with_capacity(n.pow(3));
        let grid_size_f = n as f32;
        let step = 1. / n as f64;

        // let dx = self.cell_len_a / grid_size_f;
        // let dy = self.cell_len_b / grid_size_f;
        // let dz = self.cell_len_c / grid_size_f;

        // for i_a in 0..grid_size {
        //     for i_b in 0..grid_size {
        //         for i_c in 0..grid_size {
        //             // todo: Confirm symmetrical around the origin, and that the origin matches the atom coords origin.
        //
        //             pts.push(Vec3 {
        //                 x: (-self.cell_len_a / 2. + i_a as f32 * dx) as f64,
        //                 y: (-self.cell_len_b / 2. + i_b as f32 * dy) as f64,
        //                 z: (-self.cell_len_c / 2. + i_c as f32 * dz) as f64,
        //             });
        //         }
        //     }
        // }

        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    pts.push(Vec3 {
                        x: i as f64 * step,
                        y: j as f64 * step,
                        z: k as f64 * step,
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

    // reflections
    //     .par_iter()
    //     .map(|r| {
    //         let amp = r.amp_weighted.unwrap_or(r.amp);
    //         if amp == 0.0 {
    //             0.
    //         } else {
    //             let phase = r.phase_weighted.unwrap_or(0.0).to_radians();
    //
    //             //  2π(hx + ky + lz)  (negative sign because CCP4/Coot convention)
    //             let arg = -TAU * (r.h as f64 * posit.x + r.k as f64 * posit.y + r.l as f64 * posit.z);
    //
    //             //  real part of  F · e^{iφ} · e^{iarg} = amp·cos(φ+arg)
    //             amp * (phase + arg).cos()
    //         }
    //     })
    //     .sum()

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
        rho += amp * (arg + phase.to_radians()).cos();
        // rho += amp * (arg - phase.to_radians()).cos();
    }

    // Normalize.
    rho  / unit_cell_vol as f64
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
    // todo: STore the div2 variant.

    let result = grid
        .par_iter()
        .map(|p| ElectronDensity {
            // coords: *p,
            // Convert coords to real space, in angstroms.
            coords: Vec3 {
                // x: p.x * len_a,
                // y: p.y * len_b,
                // z: p.z * len_c,

                x: p.x * len_a - len_a / 2.,
                y: p.y * len_b - len_b / 2.,
                z: p.z * len_c - len_c / 2.,
            },
            density: compute_density(&data.points, *p, unit_cell_vol),
        })
        .collect();

    let elapsed = start.elapsed().as_millis();

    println!("Complete. Time: {:?}ms", elapsed);
    result
}
