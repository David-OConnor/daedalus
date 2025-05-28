//! For interoperability between MTZ files, and our reflections structs

use std::{
    fs::File,
    io,
    io::{Read, Write},
    path::Path,
};

use crate::reflection::{MapStatus, Reflection, ReflectionsData};

const HEADER_BLOCK: usize = 80;

#[macro_export]
macro_rules! parse_le {
    ($bytes:expr, $t:ty, $range:expr) => {{ <$t>::from_le_bytes($bytes[$range].try_into().unwrap()) }};
}

#[macro_export]
macro_rules! copy_le {
    ($dest:expr, $src:expr, $range:expr) => {{ $dest[$range].copy_from_slice(&$src.to_le_bytes()) }};
}

impl ReflectionsData {
    /// Small subset of MTZ – merged reflections, one dataset.
    pub fn from_mtz(buf: &[u8]) -> io::Result<ReflectionsData> {
        let mut pos = 0;
        let mut hlines = Vec::new();
        while pos + HEADER_BLOCK <= buf.len() {
            let blk = &buf[pos..pos + HEADER_BLOCK];

            // todo: Don't unwrap; map error.
            let line = std::str::from_utf8(blk).unwrap().trim_end();
            pos += HEADER_BLOCK;
            if line == "END" {
                break;
            }
            hlines.push(line.to_owned());
        }
        if hlines.is_empty() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "no header"));
        }

        // 2) parse the handful of header keywords we need
        //    (CELL, SYMINF, COL ...)
        let mut cell = [0.; 6];
        for l in &hlines {
            if l.starts_with("CELL") {
                // CELL a b c alpha beta gamma
                let nums: Vec<_> = l.split_whitespace().skip(1).collect();
                for (i, v) in nums.iter().take(6).enumerate() {
                    cell[i] = v
                        .parse::<f32>()
                        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad CELL"))?;
                }
            }
        }

        // 3) get column order so we know where F, SIGF, FREE are
        let mut col_labels = Vec::new();
        for l in &hlines {
            if l.starts_with("COLUMN") {
                // Example:  "COLUMN     1 F            1    1"
                let label = l.split_whitespace().nth(2).unwrap();
                col_labels.push(label.to_string());
            }
        }
        let h_i = col_labels.iter().position(|l| l == "H").unwrap();
        let k_i = col_labels.iter().position(|l| l == "K").unwrap();
        let l_i = col_labels.iter().position(|l| l == "L").unwrap();
        let f_i = col_labels.iter().position(|l| l == "F").unwrap();
        let sigf_i = col_labels.iter().position(|l| l == "SIGF").unwrap();
        let freer_i = col_labels.iter().position(|l| l == "FREE");

        // 4) read binary table
        let n_cols = col_labels.len();
        let n_rows = (buf.len() - pos) / (4 * n_cols);
        let mut rdr = &buf[pos..];

        let mut points = Vec::with_capacity(n_rows);
        for _ in 0..n_rows {
            let mut row = Vec::with_capacity(n_cols);
            for _ in 0..n_cols {
                let mut buf = [0; 4];
                rdr.read_exact(&mut buf)?;
                row.push(f32::from_le_bytes(buf))
            }
            points.push(Reflection {
                h: row[h_i] as i32,
                k: row[k_i] as i32,
                l: row[l_i] as i32,
                status: freer_i
                    .map(|i| {
                        if row[i] >= 0.5 {
                            MapStatus::FreeSet
                        } else {
                            MapStatus::Observed
                        }
                    })
                    .unwrap_or(MapStatus::Observed),
                amp: row[f_i] as f64,
                amp_uncertainty: row[sigf_i] as f64,
                ..Default::default()
            });
        }

        println!("CELL: {:?}", cell);

        Ok(Self {
            space_group: "P 1".to_string(), // not stored in minimal header - fake it
            cell_len_a: cell[0],
            cell_len_b: cell[1],
            cell_len_c: cell[2],
            cell_angle_alpha: cell[3],
            cell_angle_beta: cell[4],
            cell_angle_gamma: cell[5],
            points,
        })
    }

    pub fn to_mtz(&self) -> Vec<u8> {
        let mut result = Vec::new();

        result.extend_from_slice(format!("{:<80}\n", "MTZ:V1.1").as_bytes());
        result.extend(format!("TITLE {:<70}\n", "written by Rust").as_bytes());
        result.extend(
            format!(
                "CELL {:8.3} {:8.3} {:8.3} {:7.3} {:7.3} {:7.3}{:10}\n",
                self.cell_len_a,
                self.cell_len_b,
                self.cell_len_c,
                self.cell_angle_alpha,
                self.cell_angle_beta,
                self.cell_angle_gamma,
                ""
            )
            .as_bytes(),
        );

        // column directory
        let cols = ["H", "K", "L", "F", "SIGF", "FREE"];
        for (i, &c) in cols.iter().enumerate() {
            result.extend(format!("COLUMN{:5}{:<8}{:>5}{:>5}\n", "", c, 1, i + 1).as_bytes());
        }
        result.extend("END\n".as_bytes());

        // pad to 80-byte blocks
        while result.len() % HEADER_BLOCK != 0 {
            result.push(b' ');
        }

        let mut bin = Vec::new();
        for p in &self.points {
            copy_le!(bin, p.h as f32, 0..4);
            copy_le!(bin, p.k as f32, 4..8);
            copy_le!(bin, p.l as f32, 8..12);
            copy_le!(bin, p.amp as f32, 12..16);
            copy_le!(bin, p.amp_uncertainty as f32, 20..24);
            copy_le!(
                bin,
                if matches!(p.status, MapStatus::FreeSet) {
                    1.0
                } else {
                    0.0
                } as f32,
                24..28
            );
        }

        result.extend(bin);
        result
    }
}

pub fn load_mtz(path: &Path) -> io::Result<ReflectionsData> {
    let mut file = File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;

    ReflectionsData::from_mtz(&buf)
}

pub fn save_mtz(data: &ReflectionsData, path: &Path) -> io::Result<()> {
    let buf = data.to_mtz();

    let mut file = File::open(path)?;
    file.write_all(&buf)?;

    Ok(())
}
