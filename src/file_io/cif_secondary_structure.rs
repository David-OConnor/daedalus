//! For parsing mmCIF files for secondary structure. Easier to implement here than modifying PDBTBX.

use std::{
    collections::HashMap,
    io::{self, BufRead, BufReader, Read, Seek, SeekFrom},
};

use lin_alg::f64::Vec3;

use crate::cartoon_mesh::{BackboneSS, SecondaryStructure};

// todo: PDB support too?

pub fn load_secondary_structure<R: Read + Seek>(mut data: R) -> io::Result<Vec<BackboneSS>> {
    data.seek(SeekFrom::Start(0))?;
    let mut reader = BufReader::new(data);

    // ---------------------------------------------------------------------
    // Pass 1 — read the file once and collect:
    //   • ‘CA’ atom coordinates keyed by (chain-id, seq-id)
    //   • every row of the   _struct_conf   table (we’ll interpret later)
    // ---------------------------------------------------------------------
    let mut ca_coord: HashMap<(String, i32), Vec3> = HashMap::new();
    let mut struct_rows: Vec<(Vec<String>, Vec<String>)> = Vec::new();

    enum Block {
        Nothing,
        StructConf { head: Vec<String> },
        AtomSite { head: Vec<String> },
    }
    let mut block = Block::Nothing;

    // Column indices we care about inside the _atom_site table
    let mut idx_asym = None;
    let mut idx_seq = None;
    let mut idx_atom = None;
    let mut idx_x = None;
    let mut idx_y = None;
    let mut idx_z = None;

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();

        // ──────────────────────────────────────────────────────────────
        // Delimiters
        if trimmed.starts_with("loop_") {
            block = Block::Nothing;
            continue;
        }
        // ──────────────────────────────────────────────────────────────
        // Which block are we in?
        match &mut block {
            Block::Nothing => {
                if trimmed.starts_with("_struct_conf.") {
                    block = Block::StructConf {
                        head: vec![trimmed.to_owned()],
                    };
                } else if trimmed.starts_with("_atom_site.") {
                    block = Block::AtomSite {
                        head: vec![trimmed.to_owned()],
                    };
                }
            }

            // -------------------------  _struct_conf  -----------------
            Block::StructConf { head } => {
                if trimmed.starts_with('_') {
                    head.push(trimmed.to_owned());
                } else if !trimmed.is_empty() {
                    let cols = trimmed.split_whitespace().map(str::to_owned).collect();
                    struct_rows.push((head.clone(), cols));
                } else {
                    block = Block::Nothing;
                }
            }

            // --------------------------  _atom_site  ------------------
            Block::AtomSite { head } => {
                if trimmed.starts_with('_') {
                    head.push(trimmed.to_owned());
                } else if !trimmed.is_empty() {
                    // First data row → derive column indices
                    if idx_asym.is_none() {
                        for (i, h) in head.iter().enumerate() {
                            match &h[h.rfind('.').unwrap() + 1..] {
                                "label_asym_id" => idx_asym = Some(i),
                                "label_seq_id" => idx_seq = Some(i),
                                "label_atom_id" => idx_atom = Some(i),
                                "Cartn_x" => idx_x = Some(i),
                                "Cartn_y" => idx_y = Some(i),
                                "Cartn_z" => idx_z = Some(i),
                                _ => {}
                            }
                        }
                    }
                    let (i_asym, i_seq, i_atom, i_x, i_y, i_z) =
                        match (idx_asym, idx_seq, idx_atom, idx_x, idx_y, idx_z) {
                            (Some(a), Some(s), Some(at), Some(x), Some(y), Some(z)) => {
                                (a, s, at, x, y, z)
                            }
                            _ => continue, // columns missing – ignore
                        };

                    let cols: Vec<&str> = trimmed.split_whitespace().collect();
                    if cols.len() <= i_z {
                        continue;
                    }
                    if cols[i_atom] != "CA" {
                        continue;
                    } // we only need alpha-carbon

                    if let Ok(seq) = cols[i_seq].parse::<i32>() {
                        let x: f64 = cols[i_x].parse().unwrap_or(0.0);
                        let y: f64 = cols[i_y].parse().unwrap_or(0.0);
                        let z: f64 = cols[i_z].parse().unwrap_or(0.0);
                        ca_coord.insert((cols[i_asym].to_owned(), seq), Vec3::new(x, y, z));
                    }
                } else {
                    block = Block::Nothing;
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // Pass 2 — interpret every _struct_conf row and build BackboneSS records
    // ---------------------------------------------------------------------
    let mut result = Vec::new();

    for (head, cols) in struct_rows {
        if head.len() != cols.len() {
            continue;
        }

        // work out once per row where the interesting fields are
        let mut i_conf_type = None;
        let mut i_beg_asym = None;
        let mut i_beg_seq = None;
        let mut i_end_asym = None;
        let mut i_end_seq = None;

        for (i, h) in head.iter().enumerate() {
            match &h[h.rfind('.').unwrap() + 1..] {
                "conf_type_id" => i_conf_type = Some(i),
                "beg_label_asym_id" => i_beg_asym = Some(i),
                "beg_label_seq_id" => i_beg_seq = Some(i),
                "end_label_asym_id" => i_end_asym = Some(i),
                "end_label_seq_id" => i_end_seq = Some(i),
                _ => {}
            }
        }

        let (ic, ib_a, ib_s, ie_a, ie_s) =
            match (i_conf_type, i_beg_asym, i_beg_seq, i_end_asym, i_end_seq) {
                (Some(a), Some(b), Some(c), Some(d), Some(e)) => (a, b, c, d, e),
                _ => continue,
            };

        let conf_tag = &cols[ic];
        let beg_chain = &cols[ib_a];
        let end_chain = &cols[ie_a];

        // Only consider single-chain segments (multi-chain β-sheets are rare in struct_conf)
        if beg_chain != end_chain {
            continue;
        }

        let beg_seq = cols[ib_s].parse::<i32>().ok();
        let end_seq = cols[ie_s].parse::<i32>().ok();
        let (beg_seq, end_seq) = match (beg_seq, end_seq) {
            (Some(b), Some(e)) => (b, e),
            _ => continue,
        };

        let start = match ca_coord.get(&(beg_chain.clone(), beg_seq)) {
            Some(v) => *v,
            None => continue,
        };
        let end = match ca_coord.get(&(end_chain.clone(), end_seq)) {
            Some(v) => *v,
            None => continue,
        };

        let ss = if conf_tag.starts_with("HELX") {
            SecondaryStructure::Helix
        } else if conf_tag.starts_with("STRN") || conf_tag.starts_with("SHEET") {
            SecondaryStructure::Sheet
        } else {
            SecondaryStructure::Coil
        };

        result.push(BackboneSS {
            start,
            end,
            sec_struct: ss,
        });
    }

    Ok(result)
}
