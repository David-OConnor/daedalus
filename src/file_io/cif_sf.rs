//! For reflections data from CIF dialects: SF, map 2fo_fc and map fo_fc.

use std::collections::HashMap;

use crate::reflection::{MapStatus, Reflection, ReflectionsData};

const EPS: f64 = 0.0000001;

/// Add reflections data to a HashMap from a CIF file. This may be run successively on multiple files.
/// We observe that sometimes reflection data is split across all three file types, and sometimes it's
/// all present in an SF file.
fn parse_file(
    reflections: &mut HashMap<(i32, i32, i32), Reflection>,
    text: &str,
    is_sf: bool,
    header_sink: &mut dyn FnMut(&str, &str),
) {
    let mut in_loop = false;
    let mut tags: Vec<String> = Vec::new();

    for raw in text.lines() {
        let line = raw.trim();

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

        let rec = reflections.entry(key).or_insert_with(|| Reflection {
            h,
            k,
            l,
            ..Default::default()
        });

        let overwrite_f64 = |dst: &mut f64, src: Option<f64>| {
            if let Some(v) = src {
                if is_sf || dst.abs() < EPS {
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

        if let Some(i) = col("_refln.status") {
            rec.status = MapStatus::from_str(cols[i]).unwrap_or_default();
        }

        // SF coeffs
        overwrite_f64(
            &mut rec.amp,
            col("_refln.F_meas_au").and_then(|i| cols[i].parse().ok()),
        );
        overwrite_f64(
            &mut rec.amp_uncertainty,
            col("_refln.F_meas_sigma_au").and_then(|i| cols[i].parse().ok()),
        );

        // 2Fo-Fc coeffs
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

        // Fo-Fc coeffs
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

impl ReflectionsData {
    /// Build a `ReflectionsData` from the plain SF file plus (optionally) the
    /// 2 Fo–Fc and Fo–Fc map-coefficient CIF files.
    ///
    /// * `sf`              – contents of `*.sf.cif`
    /// * `map_2fo_fc` – optional contents of `*2fo-fc_map_coef.cif`
    /// * `map_fo_fc` – optional contents of `*fo-fc_map_coef.cif`
    pub fn from_cifs(sf: &str, map_2fo_fc: Option<&str>, map_fo_fc: Option<&str>) -> Self {
        use std::{collections::HashMap, str::FromStr}; // brings MapType::from_str into scope

        let mut reflections: HashMap<(i32, i32, i32), Reflection> = HashMap::new();

        let mut space_group = String::new();
        let mut a = 0.0_f32;
        let mut b = 0.0_f32;
        let mut c = 0.0_f32;
        let mut α = 0.0_f32;
        let mut β = 0.0_f32;
        let mut γ = 0.0_f32;

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

        // Parse files in ascending precedence.
        if let Some(txt) = map_fo_fc {
            parse_file(&mut reflections, txt, false, &mut header);
        }
        if let Some(txt) = map_2fo_fc {
            parse_file(&mut reflections, txt, false, &mut header);
        }
        parse_file(&mut reflections, sf, true, &mut header); // SF wins ties

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
}
