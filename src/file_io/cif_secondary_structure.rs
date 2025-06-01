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
    let mut rdr = BufReader::new(data);

    // ─────────────  caches  ─────────────
    let mut ca_xyz: HashMap<(String, i32), Vec3> = HashMap::new();
    let mut helix_rows: Vec<(Vec<String>, Vec<String>)> = Vec::new();
    let mut sheet_rows: Vec<(Vec<String>, Vec<String>)> = Vec::new();

    enum LoopKind {
        None,
        StructConf,
        AtomSite,
        SheetRange,
    }
    let mut kind = LoopKind::None;
    let mut head: Vec<String> = Vec::new();

    // atom-site column indices (filled on first row)
    let mut a_idx = (None, None, None, None, None, None); // asym, seq, atom, x,y,z

    for line in rdr.lines() {
        let line = line?;
        let t = line.trim();
        if t.is_empty() {
            continue;
        }

        if t == "loop_" {
            kind = LoopKind::None;
            head.clear();
            a_idx = (None, None, None, None, None, None);
            continue;
        }
        if t == "#" {
            kind = LoopKind::None;
            continue;
        }

        match kind {
            LoopKind::None => {
                if t.starts_with("_struct_conf.") {
                    kind = LoopKind::StructConf;
                    head.push(t.to_owned());
                } else if t.starts_with("_atom_site.") {
                    kind = LoopKind::AtomSite;
                    head.push(t.to_owned());
                } else if t.starts_with("_struct_sheet_range.") {
                    kind = LoopKind::SheetRange;
                    head.push(t.to_owned());
                }
            }

            // ───────────── _struct_conf  (helices/turns/strands) ─────────────
            LoopKind::StructConf => {
                if t.starts_with('_') {
                    head.push(t.to_owned());
                    continue;
                }
                let cols: Vec<String> = t.split_whitespace().map(str::to_owned).collect();
                helix_rows.push((head.clone(), cols));
            }

            // ───────────── _struct_sheet_range (β-strands) ─────────────
            LoopKind::SheetRange => {
                if t.starts_with('_') {
                    head.push(t.to_owned());
                    continue;
                }
                let cols: Vec<String> = t.split_whitespace().map(str::to_owned).collect();
                sheet_rows.push((head.clone(), cols));
            }

            // ───────────── _atom_site (coordinates) ─────────────
            LoopKind::AtomSite => {
                if t.starts_with('_') {
                    head.push(t.to_owned());
                    continue;
                }

                // first data row of atom_site → locate columns
                if a_idx.0.is_none() {
                    for (i, h) in head.iter().enumerate() {
                        match &h[h.rfind('.').unwrap() + 1..] {
                            "label_asym_id" => a_idx.0 = Some(i),
                            "label_seq_id" => a_idx.1 = Some(i),
                            "label_atom_id" => a_idx.2 = Some(i),
                            "Cartn_x" => a_idx.3 = Some(i),
                            "Cartn_y" => a_idx.4 = Some(i),
                            "Cartn_z" => a_idx.5 = Some(i),
                            _ => {}
                        }
                    }
                }
                let (ia, isq, iat, ix, iy, iz) = match a_idx {
                    (Some(a), Some(s), Some(at), Some(x), Some(y), Some(z)) => (a, s, at, x, y, z),
                    _ => continue,
                };

                let c: Vec<&str> = t.split_whitespace().collect();
                if c.len() <= iz || c[iat] != "CA" {
                    continue;
                }

                if let Ok(seq) = c[isq].parse::<i32>() {
                    let xyz = Vec3::new(
                        c[ix].parse().unwrap_or(0.0),
                        c[iy].parse().unwrap_or(0.0),
                        c[iz].parse().unwrap_or(0.0),
                    );
                    ca_xyz.insert((c[ia].to_owned(), seq), xyz);
                }
            }
        }
    }

    // ─────────────  build BackboneSS records  ─────────────
    let mut out = Vec::new();

    // ----- helices from _struct_conf -----
    for (h, c) in helix_rows {
        // resolve indices once per header set
        fn find(h: &[String], tag: &str) -> Option<usize> {
            h.iter().position(|s| s.ends_with(tag))
        }
        let i_type = find(&h, "conf_type_id");
        let i_ba = find(&h, "beg_label_asym_id");
        let i_bs = find(&h, "beg_label_seq_id");
        let i_ea = find(&h, "end_label_asym_id");
        let i_es = find(&h, "end_label_seq_id");
        let (i_type, i_ba, i_bs, i_ea, i_es) = match (i_type, i_ba, i_bs, i_ea, i_es) {
            (Some(a), Some(b), Some(c), Some(d), Some(e)) => (a, b, c, d, e),
            _ => continue,
        };

        if !c[i_type].starts_with("HELX") {
            continue;
        } // we only want helices

        let beg_seq = c[i_bs].parse().ok();
        let end_seq = c[i_es].parse().ok();
        if beg_seq.is_none() || end_seq.is_none() {
            continue;
        }

        let start = match ca_xyz.get(&(c[i_ba].clone(), beg_seq.unwrap())) {
            Some(v) => *v,
            None => continue,
        };
        let end = match ca_xyz.get(&(c[i_ea].clone(), end_seq.unwrap())) {
            Some(v) => *v,
            None => continue,
        };

        out.push(BackboneSS {
            start,
            end,
            sec_struct: SecondaryStructure::Helix,
        });
    }

    // ----- β-strands from _struct_sheet_range -----
    for (h, c) in sheet_rows {
        fn idx(h: &[String], tag: &str) -> Option<usize> {
            h.iter().position(|s| s.ends_with(tag))
        }
        let ib_a = idx(&h, "beg_label_asym_id");
        let ib_s = idx(&h, "beg_label_seq_id");
        let ie_a = idx(&h, "end_label_asym_id");
        let ie_s = idx(&h, "end_label_seq_id");
        let (ib_a, ib_s, ie_a, ie_s) = match (ib_a, ib_s, ie_a, ie_s) {
            (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
            _ => continue,
        };

        let beg_seq = c[ib_s].parse().ok();
        let end_seq = c[ie_s].parse().ok();
        if beg_seq.is_none() || end_seq.is_none() {
            continue;
        }

        let start = match ca_xyz.get(&(c[ib_a].clone(), beg_seq.unwrap())) {
            Some(v) => *v,
            None => continue,
        };
        let end = match ca_xyz.get(&(c[ie_a].clone(), end_seq.unwrap())) {
            Some(v) => *v,
            None => continue,
        };

        out.push(BackboneSS {
            start,
            end,
            sec_struct: SecondaryStructure::Sheet,
        });
    }

    Ok(out)
}
