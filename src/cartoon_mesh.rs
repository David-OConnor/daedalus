//! Gets a cartoon mesh for secondary structure.

// todo: You may be able to get useful data from CIF or PDB files, although this is
// todo not parsed by PDBTBX.

use std::{f32::consts::TAU, f64::consts::PI};

use bio_files::ResidueType;
use graphics::{Mesh, Vertex};
use lin_alg::{f32::Vec3 as Vec3F32, f64::Vec3};
use na_seq::{AminoAcid, Element};

use crate::molecule::{Atom, AtomRole, Residue};

/// Radii / dimensions for each cartoon element (Å).
const HELIX_RADIUS: f32 = 0.6;
const COIL_RADIUS: f32 = 0.3;
const SHEET_HALF_W: f32 = 0.9; // half-width of the β-ribbon
const SHEET_THICK: f32 = 0.2;

const CYLINDER_SEGMENTS: usize = 12;

// todo: Eval if you want a second cyilnder mesh of different parameters.

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SecondaryStructure {
    Helix,
    Sheet,
    Coil,
}

#[derive(Clone, Debug)]
pub struct BackboneSS {
    // pub start: Vec3,
    // pub end: Vec3,
    /// Start and end are atom indices.
    pub start: usize,
    pub end: usize,
    pub sec_struct: SecondaryStructure,
}

/// Build a (closed) cylinder running from `a → b`.
fn cylinder(a: Vec3F32, b: Vec3F32, radius: f32, segments: usize) -> (Vec<Vertex>, Vec<usize>) {
    let mut verts = Vec::with_capacity(segments * 2);
    let mut idx = Vec::with_capacity(segments * 6);

    // Axis & orthonormal basis --------------------------------------------
    let axis = (b - a).to_normalized();
    let helper = if axis.z.abs() < 0.999 {
        Vec3F32::new(0.0, 0.0, 1.0)
    } else {
        Vec3F32::new(0.0, 1.0, 0.0)
    };
    let u = axis.cross(helper).to_normalized();
    let v = axis.cross(u);

    // Circle vertices -------------------------------------------------------
    for i in 0..segments {
        let t = TAU * (i as f32) / (segments as f32);
        let dir = (u * t.cos() + v * t.sin()).to_normalized();
        verts.push(Vertex::new((a + dir * radius).to_arr(), dir));
        verts.push(Vertex::new((b + dir * radius).to_arr(), dir));
    }

    // Side faces ------------------------------------------------------------
    for i in 0..segments {
        let i0 = 2 * i;
        let i1 = 2 * i + 1;
        let i2 = 2 * ((i + 1) % segments);
        let i3 = 2 * ((i + 1) % segments) + 1;

        // two triangles per quad
        idx.extend_from_slice(&[i0, i1, i2, i2, i1, i3]);
    }

    (verts, idx)
}

/// Build a thin β-strand “cartoon” ribbon that follows the supplied backbone
/// centres (`backbone_posits`).  
/// `half_w`  – half the visual strand width (Å)  
/// `thick`   – strand thickness (Å)  (± along the local sheet normal)
///
/// ‣ We build a little local frame {t, b, n} for every backbone point:
///     t – tangent  (centre-line direction)
///     b – binormal (points across the ribbon, gives its width)
///     n – normal   (b × t, gives the sheet “thickness” direction)
///
/// ‣ Every cross-section therefore has four vertices:
///     0: left  bottom   (-b  −  n)
///     1: left  top      (-b  +  n)
///     2: right bottom   (+b  −  n)
///     3: right top      (+b  +  n)
///
/// ‣ We stitch successive cross-sections with quads (two triangles each)
///   to make a smooth, curved strip with thickness.
fn sheet_ribbon(backbone_posits: &[Vec3F32], half_w: f32, thick: f32) -> (Vec<Vertex>, Vec<usize>) {
    if backbone_posits.len() < 2 {
        eprintln!("Error loading backbone positions for cartoon mesh");
        return (Vec::new(), Vec::new());
    }

    let n_pts = backbone_posits.len();
    let mut verts = Vec::<Vertex>::with_capacity(n_pts * 4);
    let mut idx = Vec::<usize>::with_capacity((n_pts - 1) * 8 * 3); // 8 tris per segment

    // ---- Helper lambdas ----------------------------------------------------
    let make_frame = |t: Vec3F32| {
        // Binormal points roughly sideways from the strand; fall back to Z/Y if needed
        let helper = if t.z.abs() < 0.999 {
            Vec3F32::new(0.0, 0.0, 1.0)
        } else {
            Vec3F32::new(0.0, 1.0, 0.0)
        };
        let b = t.cross(helper).to_normalized(); // width direction
        let n = b.cross(t).to_normalized(); // sheet normal
        (b, n)
    };

    // ---- Build vertices ----------------------------------------------------
    for (i, &p) in backbone_posits.iter().enumerate() {
        // Central-difference tangent ( …<-p[i-1]  p[i]  p[i+1]->… )
        let prev = if i == 0 {
            backbone_posits[i]
        } else {
            backbone_posits[i - 1]
        };
        let next = if i == n_pts - 1 {
            backbone_posits[i]
        } else {
            backbone_posits[i + 1]
        };
        let t = (next - prev).to_normalized();

        let (b, n) = make_frame(t);

        // Four verts per cross-section
        let left = p - b * half_w;
        let right = p + b * half_w;

        let n_up = n * (thick * 0.5);
        let n_down = -n_up;

        // 0 LL  1 LU  2 RL  3 RU     (see diagram in doc-comment)
        verts.push(Vertex::new((left + n_down).to_arr(), n_down)); // 0
        verts.push(Vertex::new((left + n_up).to_arr(), n_up)); // 1
        verts.push(Vertex::new((right + n_down).to_arr(), n_down)); // 2
        verts.push(Vertex::new((right + n_up).to_arr(), n_up)); // 3
    }

    // ---- Build index buffer ------------------------------------------------
    for seg in 0..(n_pts - 1) {
        let a = seg * 4; // base index for cross-section i
        let b = a + 4; // base for cross-section i+1

        // Top (inner) face  (LU-RU-RU′, LU-RU′-LU′)
        idx.extend_from_slice(&[a + 1, a + 3, b + 3, a + 1, b + 3, b + 1]);

        // Bottom (outer) face  (LL-RL′-RL, LL-LL′-RL′)
        idx.extend_from_slice(&[a + 0, b + 0, a + 2, a + 2, b + 0, b + 2]);

        // Left edge
        idx.extend_from_slice(&[a + 0, a + 1, b + 1, a + 0, b + 1, b + 0]);

        // Right edge
        idx.extend_from_slice(&[a + 2, b + 2, b + 3, a + 2, b + 3, a + 3]);
    }

    // (Optional) ► Add an arrow-shaped cap at the C-terminus here if you like ◄

    (verts, idx)
}

pub fn build_cartoon_mesh(backbone: &[BackboneSS], atoms: &[Atom]) -> Mesh {
    let mut vertices = Vec::<Vertex>::new();
    let mut indices = Vec::<usize>::new();

    for seg in backbone {
        // let mut atom_posits: Vec<Vec3F32> = Vec::with_capacity(seg.end - seg.start);
        let mut atom_posits: Vec<Vec3F32> = Vec::new();
        for i in seg.start..seg.end + 1 {
            if atoms[i].element == Element::Oxygen {
                continue;
            }
            atom_posits.push(atoms[i].posit.into());
        }

        let (mut vtx, mut idx) = match seg.sec_struct {
            SecondaryStructure::Helix =>
            // todo: Do better than a cylinder...
            {
                (Vec::new(), Vec::new())
            }
            // {
            //     cylinder(
            //         seg.start.into(),
            //         seg.end.into(),
            //         HELIX_RADIUS,
            //         CYLINDER_SEGMENTS,
            //     )
            // }
            SecondaryStructure::Coil => (Vec::new(), Vec::new()),
            // cylinder(
            //         seg.start.into(),
            //         seg.end.into(),
            //         COIL_RADIUS,
            //         CYLINDER_SEGMENTS / 2,
            //     ),
            SecondaryStructure::Sheet => {
                // sheet_ribbon(seg.start.into(), seg.end.into(), SHEET_HALF_W, SHEET_THICK)
                sheet_ribbon(&atom_posits, SHEET_HALF_W, SHEET_THICK)
            }
        };

        // offset indices by the number of verts already emitted
        let base = vertices.len();
        idx.iter_mut().for_each(|i| *i += base);
        vertices.extend(vtx);
        indices.extend(idx);
    }

    Mesh {
        vertices,
        indices,
        material: 0,
    }
}
