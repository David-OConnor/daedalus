//! Gets a cartoon mesh for secondary structure.

// todo: You may be able to get useful data from CIF or PDB files, although this is
// todo not parsed by PDBTBX.

use std::f64::consts::PI;

use graphics::{Mesh, Vertex};
use lin_alg::f64::Vec3;
use na_seq::AminoAcid;

use crate::{
    molecule::{AaRole, Atom, Residue, ResidueType},
    util::vec3_to_f32,
};

/// How many slices around each tube cross-section
const TUBE_SIDES: usize = 8;

#[derive(Clone, Copy, PartialEq)]
enum SecondaryStructure {
    Helix,
    Sheet,
    Coil,
}

struct BackboneSegment {
    start: Vec3,
    end: Vec3,
    sec_struct: SecondaryStructure,
}

// todo: ChatGpt
fn gather_alpha_positions(atoms: &[Atom], residues: &[Residue]) -> Vec<Vec3> {
    let mut alpha_positions = Vec::new();

    // For each residue, find the alpha carbon and push its position
    for residue in residues {
        // Each residue holds indexes to `atoms`
        let mut found_ca = None;
        for &atom_idx in &residue.atoms {
            let atom = &atoms[atom_idx];
            if atom.role == Some(AaRole::C_Alpha) {
                found_ca = Some(atom.posit);
                break;
            }
        }
        if let Some(pos) = found_ca {
            alpha_positions.push(pos);
        }
    }
    alpha_positions
}

// todo: ChatGpt
fn get_secondary_structure_of_residue(res: &Residue) -> SecondaryStructure {
    // Simple placeholder: a real implementation would parse from
    // your PDB data or run a calculation.
    if let ResidueType::AminoAcid(aa) = res.res_type {
        match aa {
            // Maybe just random examples
            AminoAcid::Ala => SecondaryStructure::Helix,
            AminoAcid::Arg => SecondaryStructure::Sheet,
            // ...
            _ => SecondaryStructure::Coil,
        }
    } else {
        SecondaryStructure::Coil
    }
}

// todo: ChatGpt
fn build_backbone_segments(atoms: &[Atom], residues: &[Residue]) -> Vec<BackboneSegment> {
    let alpha_positions = gather_alpha_positions(atoms, residues);
    let mut segments = Vec::new();
    if alpha_positions.len() < 2 {
        return segments;
    }

    for i in 0..(alpha_positions.len() - 1) {
        // For example, use the structure type of the *leading* residue
        let sec_struct = {
            // retrieve from residue i
            let r = &residues[i];
            get_secondary_structure_of_residue(r)
        };
        segments.push(BackboneSegment {
            start: alpha_positions[i],
            end: alpha_positions[i + 1],
            sec_struct,
        });
    }
    segments
}

pub fn mesh_from_atoms(atoms: &[Atom], residues: &[Residue]) -> Mesh {
    let backbone = build_backbone_segments(atoms, residues);

    // Our final mesh
    let mut mesh = Mesh {
        vertices: Vec::new(),
        indices: Vec::new(),
        material: 0, // or whatever
    };

    // Keep track of vertex offset as we add geometry
    let mut base_index = 0;

    for segment in &backbone {
        let direction = (segment.end - segment.start).to_normalized();

        // Decide on radius based on structure type
        let radius = match segment.sec_struct {
            SecondaryStructure::Helix => 0.35,
            SecondaryStructure::Sheet => 0.25,
            SecondaryStructure::Coil => 0.15,
        };

        // Build local coordinate system for the tube cross-section
        // A typical way is to pick a "pseudo-up" vector that isn't parallel to direction,
        // then cross to find a right vector, etc.
        let arbitrary_up = if direction.y.abs() < 0.99 {
            Vec3::new(0.0, 1.0, 0.0)
        } else {
            Vec3::new(1.0, 0.0, 0.0)
        };

        let right = direction.cross(arbitrary_up).to_normalized();
        let up = right.cross(direction).to_normalized();

        // Generate ring of vertices at start
        let start_ring: Vec<(Vec3, Vec3)> = (0..TUBE_SIDES)
            .map(|i| {
                let theta = 2.0 * PI * (i as f64) / (TUBE_SIDES as f64);
                // Normal in cross-section
                let normal = (right * theta.cos() + up * theta.sin()).to_normalized();
                // Vertex position for the start circle
                let position = segment.start + normal * radius;
                (position, normal)
            })
            .collect();

        // Generate ring of vertices at end
        let end_ring: Vec<(Vec3, Vec3)> = (0..TUBE_SIDES)
            .map(|i| {
                let theta = 2.0 * PI * (i as f64) / (TUBE_SIDES as f64);
                let normal = (right * theta.cos() + up * theta.sin()).to_normalized();
                let position = segment.end + normal * radius;
                (position, normal)
            })
            .collect();

        // Add these vertices to the mesh
        for (pos, normal) in &start_ring {
            mesh.vertices.push(Vertex::new(
                vec3_to_f32(*pos).to_arr(),
                vec3_to_f32(*normal),
            ));
        }
        for (pos, normal) in &end_ring {
            mesh.vertices.push(Vertex::new(
                vec3_to_f32(*pos).to_arr(),
                vec3_to_f32(*normal),
            ));
        }

        // Build indices for the quads connecting start_ring to end_ring
        // We have TUBE_SIDES sides in each ring. We'll create triangles in pairs.
        for i in 0..TUBE_SIDES {
            // current and next index, wrapping around
            let i0 = i;
            let i1 = (i + 1) % TUBE_SIDES;

            // Indices in the mesh's vertex buffer:
            // start ring goes from base_index to base_index + TUBE_SIDES-1
            // end ring goes from base_index + TUBE_SIDES to base_index + 2*TUBE_SIDES-1
            let start0 = base_index + i0;
            let start1 = base_index + i1;
            let end0 = base_index + TUBE_SIDES + i0;
            let end1 = base_index + TUBE_SIDES + i1;

            // Two triangles for each quad:
            //   (start0, start1, end0) + (end0, start1, end1)
            mesh.indices.push(start0);
            mesh.indices.push(start1);
            mesh.indices.push(end0);

            mesh.indices.push(end0);
            mesh.indices.push(start1);
            mesh.indices.push(end1);
        }

        // Update base_index for next segment
        base_index += 2 * TUBE_SIDES;
    }

    mesh
}
