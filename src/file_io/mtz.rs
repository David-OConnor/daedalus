//! For interoperability between MTZ files, and our reflections structs
//!
//! [Gemmi source](https://github.com/project-gemmi/gemmi/blob/master/src/mtz.cpp)
//! https://github.com/project-gemmi/gemmi/tree/master/include/gemmi
//! [Unnoficial guide (Vague, but has general info)](https://staraniso.globalphasing.org/html/mtzformat.html)

use std::{
    fs::File,
    io,
    io::{ErrorKind, Read, Seek, SeekFrom, Write},
    path::Path,
};

use crate::reflection::{MapStatus, Reflection, ReflectionsData};

const HEADER_BLOCK: usize = 80;

const INITIAL_BYTES: [u8; 4] = *b"MTZ "; // 4D 54 5A 20

#[macro_export]
macro_rules! parse_le {
    ($bytes:expr, $t:ty, $range:expr) => {{ <$t>::from_le_bytes($bytes[$range].try_into().unwrap()) }};
}

#[macro_export]
macro_rules! copy_le {
    ($dest:expr, $src:expr, $range:expr) => {{ $dest[$range].copy_from_slice(&$src.to_le_bytes()) }};
}

fn io_err(text: &str) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, text)
}

// /// Fast, locale-independent atof: parse a `f32` from the start of `s`, advancing the slice.
// fn fast_atof(s: &mut &str) -> io::Result<f32> {
//     // Skip leading whitespace
//     let trimmed = s.trim_start();
//     let advance = s.len() - trimmed.len();
//     *s = &s[advance..];
//     // Skip leading '+'
//     if s.starts_with('+') {
//         *s = &s[1..];
//     }
//     // Parse with lexical_core, getting the number of bytes consumed
//     let bytes = s.as_bytes();
//     let (value, n) = lexical_core::parse_partial::<f64>(bytes);
//     // Advance past the parsed characters
//     *s = &s[n..];
//     Ok(value as f32)
// }

/// Read the main MTZ headers from a byte stream.
pub fn read_main_headers(buf: &[u8], save_headers: Option<&mut Vec<String>>) -> io::Result<()> {
    // todo: Struct?
    // let mut nsymop  = 0;
    let mut header_offset = 0;
    // let mut symops = Vec::new();
    // let mut version_stamp = String::new();
    // let mut title = String::new();
    // let mut batches = Vec::new();
    // let mut cell = Vec::new();
    // let mut min_1_d2 = 0;
    // let mut max_1_d2 = 0;
    // let mut columns = Vec::new();
    // let mut SortOrder = String::new();
    // let mut data = Vec::new();
    // let mut spacegroup_name = String::new();
    // let mut spacegroup_number = 0;

    const HEADER_SIZE: usize = 80;

    // Each header line is 80 bytes
    let mut buf2 = [0u8; HEADER_SIZE];

    // Compute expected position (4 bytes per record, header_offset is 1-based)
    // let header_pos = 4 * (header_offset as u64 - 1);
    // let cur_pos = buf.seek(SeekFrom::Current(0))?;
    // if cur_pos != header_pos {
    //     return Err(io_err("Invalid header pos"));
    // }

    // Counters and flags
    let mut ncol = 0;
    let mut has_batch = false;

    let mut i = 0;
    loop {
        // Read exactly 80 bytes or break at EOF
        // if let Err(_) = buf.read_exact(&mut buf) {
        //     break;
        // }
        // Convert to String (including spaces)
        let line = String::from_utf8_lossy(&buf[i..i + 80]).to_string();

        // Optionally save raw header text
        // if let Some(ref mut hdrs) = save_headers {
        //     hdrs.push(line.clone());
        // }

        // Stop at END record (first three chars)
        if &line[..3] == "END" {
            break;
        }

        // Skip record name and any spaces
        // let mut args = skip_word_and_space(&line);

        // todo temp
        println!("HEADER Type: {:?}", &line[..4]);

        i += HEADER_SIZE;
        //     // Dispatch on the 4-character record code
        //     match &line[..4] {
        //         "VERS" => {
        //             version_stamp = rtrim_str(args);
        //         }
        //         "TITL" => {
        //             title = rtrim_str(args);
        //         }
        //         "NCOL" => {
        //             ncol = simple_atoi(&mut args)?;
        //             nreflections = simple_atoi(&mut args)?;
        //             let nbatches = simple_atoi(&mut args)?;
        //             if nbatches < 0 || nbatches > 10_000_000 {
        //                 return Err(MtzError::WrongNcol);
        //             }
        //             batches.resize(nbatches as usize, Default::default());
        //         }
        //         "CELL" => {
        //            cell = read_cell_parameters(args)?;
        //         }
        //         "SORT" => {
        //             for slot in &mut sort_order {
        //                 *slot = simple_atoi(&mut args)?;
        //             }
        //         }
        //         "SYMI" => {
        //             nsymop = simple_atoi(&mut args)?;
        //             symops.reserve(self.nsymop as usize);
        //             // skip primitive count
        //             let _ = simple_atoi(&mut args)?;
        //             // skip lattice type
        //             args = skip_word_and_space(skip_blank(args));
        //             spacegroup_number = simple_atoi(&mut args)?;
        //             args = skip_blank(args);
        //             // spacegroup name in quotes or bare
        //             if !args.starts_with('"') {
        //                 spacegroup_name = read_word(args).to_string();
        //             } else if let Some(end) = args[1..].find('"') {
        //                 spacegroup_name = args[1..1+end].to_string();
        //             }
        //         }
        //         "SYMM" => {
        //             symops.push(parse_triplet(args));
        //         }
        //         "RESO" => {
        //             min_1_d2 = fast_atof(&mut args)?;
        //             max_1_d2 = fast_atof(&mut args)?;
        //         }
        //         "VALM" => {
        //             if !args.starts_with('N') {
        //                 let v = fast_atof(&mut args)?;
        //                 valm = v; // or assign to self.valm
        //             }
        //         }
        //         "COLU" => {
        //             let mut col = Column::default();
        //             col.label = read_word(&mut args).to_string();
        //             col.col_type = read_word(&mut args).chars().next().unwrap();
        //             col.min_value = fast_atof(&mut args)?;
        //             col.max_value = fast_atof(&mut args)?;
        //             col.dataset_id = simple_atoi(&mut args)?;
        //             col.parent = self as *mut _;
        //             col.idx = columns.len();
        //             columns.push(col);
        //         }
        //         "COLS" => {
        //             let lbl = read_word(&mut args);
        //             if let Some(last) = columns.last_mut() {
        //                 if last.label == lbl {
        //                     last.source = read_word(args).to_string();
        //                 }
        //             }
        //         }
        //         "COLG" => {}
        //         "NDIF" => {
        //             let nd = simple_atoi(&mut args)?;
        //             self.datasets.reserve(nd as usize);
        //         }
        //         "PROJ" => {
        //             let mut ds = Dataset::default();
        //             ds.id = simple_atoi(&mut args)?;
        //             ds.project_name = read_word(skip_word_and_space(&line[5..])).to_string();
        //             self.datasets.push(ds);
        //         }
        //         "CRYS" => {
        //             let id = simple_atoi(&mut args)?;
        //             if id == self.last_dataset().id {
        //                 self.last_dataset_mut().crystal_name = read_word(args).to_string();
        //             }
        //         }
        //         "DATA" => {
        //             let id = simple_atoi(&mut args)?;
        //             if id == self.last_dataset().id {
        //                 self.last_dataset_mut().dataset_name = read_word(args).to_string();
        //             }
        //         }
        //         "DCEL" => {
        //             let id = simple_atoi(&mut args)?;
        //             if id == self.last_dataset().id {
        //                 self.last_dataset_mut().cell = read_cell_parameters(args)?;
        //             }
        //         }
        //         "DWAV" => {
        //             let id = simple_atoi(&mut args)?;
        //             if id == self.last_dataset().id {
        //                 self.last_dataset_mut().wavelength = fast_atof(&mut args)?;
        //             }
        //         }
        //         "BATCH" => {
        //             has_batch = true;
        //         }
        //         _ => {
        //             println!("Unknown header: {}", rtrim_str(&line));
        //         }
        //     }
        // }
        //
        // // consistency checks
        // if ncol != columns.len() as i32 {
        //     return Err(MtzError::ColumnCountMismatch);
        // }
        // if has_batch != !self.batches.is_empty() {
        //     return Err(MtzError::BatchMismatch);
        // }
        // if !data.is_empty() {
        //     let expected = columns.len() * nreflections as usize;
        //     if data.len() > expected {
        //         data.truncate(expected);
        //     } else if data.len() < expected {
        //         return Err(MtzError::DataSizeMismatch);
        //     }
    }

    Ok(())
}

impl ReflectionsData {
    /// Small subset of MTZ – merged reflections, one dataset.
    pub fn from_mtz(buf: &[u8]) -> io::Result<ReflectionsData> {
        let mut pos = 0;

        if buf[0..4] != INITIAL_BYTES {
            return Err(io_err("Invalid MTZ start bytes; should be b'MTZ '."));
        }

        let header_addr = parse_le!(buf, u32, 4..8) as usize;
        println!("Header addr: {:?}", header_addr);

        // This encodes the number formats of the architecture the file was written on. (In fact,
        // the machine stamp is positioned 2 words from the start, where a word is sizeof(float), i.e.
        // typically 8 bytes in. The first 4 half-bytes represent the real, complex, integer and
        // character formats, and the last two bytes are currently unused.)
        let machine_stamp = parse_le!(buf, u32, 8..12);

        // Big endian: 1. Little endian: 4. Other values mean native byte order.
        // e.g. 4 for numerical, and 1 for char.
        let real_fmt = buf[8] >> 4;
        let cplx_fmt = buf[8] & 0xf;
        let int_fmt = buf[9] >> 4;
        let char_fmt = buf[9] & 0xf;

        // Hmm. It seems then we jump to byte 80?

        let refl_data = &buf[21..header_addr];
        let header_data = &buf[header_addr..];

        // read_main_headers(&buf[80..], None);
        read_main_headers(&header_data, None);

        println!("HEader data: {:x?}", &header_data[..100]);

        let points = Vec::new();
        Ok(Self {
            space_group: "P 1".to_string(), // not stored in minimal header - fake it
            cell_len_a: 0.,
            cell_len_b: 0.,
            cell_len_c: 0.,
            cell_angle_alpha: 0.,
            cell_angle_beta: 0.,
            cell_angle_gamma: 0.,
            points,
        })

        // Ok(Self {
        //     space_group: "P 1".to_string(), // not stored in minimal header - fake it
        //     cell_len_a: cell[0],
        //     cell_len_b: cell[1],
        //     cell_len_c: cell[2],
        //     cell_angle_alpha: cell[3],
        //     cell_angle_beta: cell[4],
        //     cell_angle_gamma: cell[5],
        //     points,
        // })
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
