//! Gets a cartoon mesh for secondary structure.

// todo: You may be able to get useful data from CIF or PDB files, although this is
// todo not parsed by PDBTBX.

use std::f64::consts::PI;

use bio_files::ResidueType;
use graphics::{Mesh, Vertex};
use lin_alg::{f32::Vec3 as Vec3F32, f64::Vec3};
use na_seq::AminoAcid;

use crate::molecule::{Atom, AtomRole, Residue};

/// Radii / dimensions for each cartoon element (Å).
const HELIX_RADIUS: f32 = 0.6;
const COIL_RADIUS:  f32 = 0.3;
const SHEET_HALF_W: f32 = 0.9;   // half-width of the β-ribbon
const SHEET_THICK: f32 = 0.2;

// todo: Eval if you want a second cyilnder mesh of different parameters.

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SecondaryStructure {
    Helix,
    Sheet,
    Coil,
}

#[derive(Clone, Debug)]
pub struct BackboneSS {
    pub start: Vec3,
    pub end: Vec3,
    pub sec_struct: SecondaryStructure,
}

/// Build a thin ribbon/arrow for β-strands.
/// Simplest version: just a box; you can fancy-it-up later.
fn sheet_ribbon(a: Vec3F32,
                b: Vec3F32,
                half_w: f32,
                thick: f32) -> (Vec<Vertex>, Vec<usize>)
{
    let axis   = (b - a).to_normalized();
    let helper = if axis.z.abs() < 0.999 { Vec3F32::new(0.0, 0.0, 1.0) }
    else                     { Vec3F32::new(0.0, 1.0, 0.0) };
    let u = axis.cross(helper).to_normalized(); // width direction
    let v = axis.cross(u).to_normalized();      // thickness direction

    // 8 corners of a rectangular prism (arrow head: add later.)
    let mut verts = Vec::<Vertex>::with_capacity(8);
    for &side in &[-1.0, 1.0] {      // lower/upper faces
        for &edge in &[-1.0, 1.0] {  // left/right edges
            let offset = u * (edge * half_w) + v * (side * thick * 0.5);
            
            verts.push(Vertex::new((a + offset).to_arr(), v * side));
            verts.push(Vertex::new((b + offset).to_arr(), v * side));
        }
    }

    // Twelve triangles (6 quad faces)
    let idx: [usize; 36] = [
        // bottom
        0,2,1,  1,2,3,
        // top
        4,5,6,  5,7,6,
        // sides
        0,1,4,  4,1,5,
        2,6,3,  3,6,7,
        0,4,2,  2,4,6,
        1,3,5,  5,3,7,
    ];
    (verts, idx.to_vec())
}

pub fn build_cartoon_mesh(backbone: &[BackboneSS]) -> Mesh {
    let mut vertices = Vec::<Vertex>::new();
    let mut indices  = Vec::<usize>::new();

    for seg in backbone {
        let (mut vtx, mut idx) = match seg.sec_struct {
            SecondaryStructure::Helix =>
            // todo: Do better than a cylinder...
                (Vec::new(), Vec::new()),
                // cylinder(seg.start, seg.end, HELIX_RADIUS, CYLINDER_SEGMENTS),
            SecondaryStructure::Coil =>
                (Vec::new(), Vec::new()),
                // cylinder(seg.start, seg.end, COIL_RADIUS,  CYLINDER_SEGMENTS / 2),
            SecondaryStructure::Sheet =>
                sheet_ribbon(seg.start.into(), seg.end.into(), SHEET_HALF_W, SHEET_THICK),
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
