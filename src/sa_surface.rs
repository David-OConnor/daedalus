//! For calculating the accessible surface (AS), a proxy for acccessibility of solvents
//! to a molecule. Used for drawing *surface*, *dots*, and related meshes. Related to the van der Waals
//! radius.

use std::{collections::HashMap, time::Instant};

use graphics::{EngineUpdates, Mesh, Vertex};
use lin_alg::f32::{Vec3, Vec3 as Vec3F32};
use mcubes::{MarchingCubes, MeshSide};

use crate::{
    ResColoring, StateUi, ViewSelLevel,
    drawing::{CHARGE_MAP_MAX, CHARGE_MAP_MIN, SAS_ISO_OPACITY, color_viridis_float},
    molecule::{Atom, MoleculePeptide},
    render::MESH_SOLVENT_SURFACE,
    util::res_color,
};
use crate::drawing::MoleculeView;

const SOLVENT_RAD: f32 = 1.4; // water probe

/// Create a mesh of the solvent-accessible surface. We do this using the ball-rolling method
/// based on Van-der-Waals radius, then use the Marching Cubes algorithm to generate an iso mesh with
/// iso value = 0.
pub fn make_sas_mesh(atoms: &[&Atom], mut precision: f32) -> Mesh {
    if atoms.is_empty() {
        return Mesh::default();
    }

    // todo: Experimenting avoiding problems on large mols. We have problems with both surface
    // todo: And dots; this mitigates surface. The dots one is re Instance Buffer max size;
    // todo: This one addresses Vertex buffer being maximum size.
    if atoms.len() > 10_000 {
        precision = 0.6;
    } else if atoms.len() > 20_000 {
        precision = 0.7;
    } else if atoms.len() > 40_000 {
        precision = 0.75;
    }

    // Bounding box and grid
    let mut bb_min = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
    let mut bb_max = Vec3::new(f32::MIN, f32::MIN, f32::MIN);
    let mut r_max: f32 = 0.0;
    for a in atoms {
        let r = a.element.vdw_radius() + SOLVENT_RAD;
        r_max = r_max.max(r);

        bb_min = Vec3::new(
            bb_min.x.min(a.posit.x as f32),
            bb_min.y.min(a.posit.y as f32),
            bb_min.z.min(a.posit.z as f32),
        );

        bb_max = Vec3::new(
            bb_max.x.max(a.posit.x as f32),
            bb_max.y.max(a.posit.y as f32),
            bb_max.z.max(a.posit.z as f32),
        );
    }
    bb_min -= Vec3::splat(r_max + precision);
    bb_max += Vec3::splat(r_max + precision);

    let dim_v = (bb_max - bb_min) / precision;
    let grid_dim = (
        dim_v.x.ceil() as usize + 1,
        dim_v.y.ceil() as usize + 1,
        dim_v.z.ceil() as usize + 1,
    );

    let n_voxels = grid_dim.0 * grid_dim.1 * grid_dim.2;

    // This can be any that is guaranteed to be well outside the SAS surface.
    // It prevents holes from appearing in the mesh due to not having a value outside to compare to.
    let far_val = (r_max + precision).powi(2) + 1.0;
    let mut field = vec![far_val; n_voxels];

    // Helper to flatten (x, y, z)
    let idx = |x: usize, y: usize, z: usize| -> usize { (z * grid_dim.1 + y) * grid_dim.0 + x };

    // Fill signed-squared-distance field
    for a in atoms {
        let center: Vec3 = a.posit.into();
        let rad = a.element.vdw_radius() + SOLVENT_RAD;
        let rad2 = rad * rad;

        let lo = ((center - Vec3::splat(rad)) - bb_min) / precision;
        let hi = ((center + Vec3::splat(rad)) - bb_min) / precision;

        let (xi0, yi0, zi0) = (
            lo.x.floor().max(0.0) as usize,
            lo.y.floor().max(0.0) as usize,
            lo.z.floor().max(0.0) as usize,
        );
        let (xi1, yi1, zi1) = (
            hi.x.ceil().min((grid_dim.0 - 1) as f32) as usize,
            hi.y.ceil().min((grid_dim.1 - 1) as f32) as usize,
            hi.z.ceil().min((grid_dim.2 - 1) as f32) as usize,
        );

        for z in zi0..=zi1 {
            for y in yi0..=yi1 {
                for x in xi0..=xi1 {
                    let p = bb_min + Vec3::new(x as f32, y as f32, z as f32) * precision;
                    let d2 = (p - center).magnitude_squared();
                    let v = d2 - rad2;
                    let f = &mut field[idx(x, y, z)];
                    if v < *f {
                        *f = v;
                    }
                }
            }
        }
    }

    // Convert to a mesh using Marchine Cubes.
    let sampling_interval = (
        grid_dim.0 as f32 - 1.0,
        grid_dim.1 as f32 - 1.0,
        grid_dim.2 as f32 - 1.0,
    );

    //  scale = precision because size / sampling_interval = precision
    let size = (
        sampling_interval.0 * precision,
        sampling_interval.1 * precision,
        sampling_interval.2 * precision,
    );

    // todo: The holes in our mesh seem related to the iso level chosen.
    let mc = MarchingCubes::new(grid_dim, size, sampling_interval, bb_min, field, 0.)
        .expect("marching cubes init");

    // Note: We're experiencing the opposite behavior than we expect here; we really want to draw outside.
    let mc_mesh = mc.generate(MeshSide::InsideOnly);

    let vertices: Vec<Vertex> = mc_mesh
        .vertices
        .iter()
        // I'm not sure why we need to invert the normal here; same reason we use InsideOnly above.
        .map(|v| Vertex::new([v.posit.x, v.posit.y, v.posit.z], -v.normal))
        .collect();

    Mesh {
        vertices,
        indices: mc_mesh.indices,
        material: 0,
    }
}

/// We use this to apply coloring to SAS meshes based on the atoms and residues near them.
pub fn update_sas_mesh_coloring_(mol: &MoleculePeptide, state_ui: &StateUi, meshes: &mut [Mesh]) {
    let start = Instant::now();
    println!("Loading SAS mesh coloring...");

    let opacity = (SAS_ISO_OPACITY * 255.) as u8;

    // todo: We should use a neighbor algorithm too; could result in big speedups.

    // Optimization: Pre-select atoms near the surface, to reduce the number of distances we compute.
    let mut atoms_near_surface = Vec::new();
    for (i, atom) in mol.common.atoms.iter().enumerate() {
        atoms_near_surface.push(i); // todo temp
    }

    for vertex in &mut meshes[MESH_SOLVENT_SURFACE].vertices {
        // todo: This is slow and crude; find a better way! Grouping, neighbors etc.
        let mut closest_atom_dist = f32::INFINITY;
        let mut closest_atom = None;

        // for (i, atom) in atoms_near_surface.iter().enumerate() {}
        for i in &atoms_near_surface {
            let atom = &mol.common.atoms[*i];
            let p: Vec3F32 = atom.posit.into();
            let dist = (p - Vec3F32::from_slice(&vertex.position).unwrap()).magnitude_squared();
            if dist < closest_atom_dist {
                closest_atom_dist = dist;
                closest_atom = Some(i);
            }
        }

        // todo: Once this is working, apply other coloring schemes.
        if let Some(i) = closest_atom {
            let (r, g, b) = match state_ui.view_sel_level {
                ViewSelLevel::Residue => {
                    let atom = &mol.common.atoms[*i];
                    let aa_count = 40; // todo temp!!
                    let res = &mol.residues[atom.residue.unwrap()];

                    res_color(res, state_ui.res_coloring, atom.residue, aa_count)
                }
                _ => {
                    if state_ui.atom_color_by_charge {
                        if let Some(q) = mol.common.atoms[*i].partial_charge {
                            color_viridis_float(q, CHARGE_MAP_MIN, CHARGE_MAP_MAX)
                        } else {
                            (0., 0., 0.)
                        }
                    } else {
                        (0., 0., 0.)
                    }
                }
            };

            vertex.color = Some((
                (r * 255.) as u8,
                (g * 255.) as u8,
                (b * 255.) as u8,
                opacity,
            ));
        }
    }

    let elapsed = start.elapsed().as_millis();
    println!("Colors loaded in {elapsed} ms");
}

// todo: The optimizations are LLM mess

fn cell_key(p: Vec3F32, cell_size: f32) -> (i32, i32, i32) {
    let inv = 1.0 / cell_size;
    (
        (p.x * inv).floor() as i32,
        (p.y * inv).floor() as i32,
        (p.z * inv).floor() as i32,
    )
}

/// We use this to apply coloring to SAS meshes based on the atoms and residues near them.
pub fn update_sas_mesh_coloring(
    mol: &MoleculePeptide,
    state_ui: &StateUi,
    meshes: &mut [Mesh],
    engine_updates: &mut EngineUpdates,
) {
    let start = Instant::now();
    println!("Loading SAS mesh coloring...");

    let opacity = (SAS_ISO_OPACITY * 255.) as u8;

    // Tune these. Units should match your coordinate system (often Å).
    const NEAR_SURFACE_CUTOFF: f32 = 8.0; // atom is "near surface" if within this distance of any SAS vertex
    const GRID_CELL_SIZE: f32 = 6.0; // usually ~= cutoff; larger -> fewer cells, more candidates per cell

    // --- Build grid of SAS vertices (positions only) ---
    let mut sas_grid: HashMap<(i32, i32, i32), Vec<Vec3F32>> = HashMap::new();
    {
        let sas_mesh = &meshes[MESH_SOLVENT_SURFACE];
        for v in &sas_mesh.vertices {
            let vp = Vec3F32::from_slice(&v.position).unwrap();
            let k = cell_key(vp, GRID_CELL_SIZE);
            sas_grid.entry(k).or_default().push(vp);
        }
    }

    // --- Optimization: Pre-select atoms near the surface ---
    let mut atoms_near_surface = Vec::new();
    let cutoff2 = NEAR_SURFACE_CUTOFF * NEAR_SURFACE_CUTOFF;

    for (i, atom) in mol.common.atoms.iter().enumerate() {
        let ap: Vec3F32 = atom.posit.into();
        let (cx, cy, cz) = cell_key(ap, GRID_CELL_SIZE);

        let mut near = false;

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(points) = sas_grid.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &sp in points {
                            if (sp - ap).magnitude_squared() <= cutoff2 {
                                near = true;
                                break;
                            }
                        }
                    }
                    if near {
                        break;
                    }
                }
                if near {
                    break;
                }
            }
            if near {
                break;
            }
        }

        if near {
            atoms_near_surface.push(i);
        }
    }

    // --- Build grid of near-surface atoms (store indices) ---
    let mut atom_grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for &i in &atoms_near_surface {
        let ap: Vec3F32 = mol.common.atoms[i].posit.into();
        let k = cell_key(ap, GRID_CELL_SIZE);
        atom_grid.entry(k).or_default().push(i);
    }

    // --- Color SAS vertices by nearest near-surface atom ---
    for vertex in &mut meshes[MESH_SOLVENT_SURFACE].vertices {
        if !state_ui.color_surface_mesh {
            vertex.color = None;
            continue;
        }

        let vp = Vec3F32::from_slice(&vertex.position).unwrap();
        let (cx, cy, cz) = cell_key(vp, GRID_CELL_SIZE);

        let mut closest_atom_dist = f32::INFINITY;
        let mut closest_atom = None;

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(cands) = atom_grid.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &i in cands {
                            let ap: Vec3F32 = mol.common.atoms[i].posit.into();
                            let dist = (ap - vp).magnitude_squared();
                            if dist < closest_atom_dist {
                                closest_atom_dist = dist;
                                closest_atom = Some(i);
                            }
                        }
                    }
                }
            }
        }

        // Fallback: if the local neighborhood is empty (rare if GRID_CELL_SIZE is sensible),
        // do a full scan over the reduced set so you still color everything correctly.
        if closest_atom.is_none() {
            for &i in &atoms_near_surface {
                let ap: Vec3F32 = mol.common.atoms[i].posit.into();
                let dist = (ap - vp).magnitude_squared();
                if dist < closest_atom_dist {
                    closest_atom_dist = dist;
                    closest_atom = Some(i);
                }
            }
        }

        if let Some(i) = closest_atom {
            let (r, g, b, a) = match state_ui.view_sel_level {
                ViewSelLevel::Residue => {
                    let atom = &mol.common.atoms[i];
                    let aa_count = 40; // todo temp!!
                    let res = &mol.residues[atom.residue.unwrap()];
                    let (r, g, b) = res_color(res, state_ui.res_coloring, atom.residue, aa_count);

                    (r, g, b, opacity)
                }
                _ => {
                    if state_ui.atom_color_by_charge {
                        if let Some(q) = mol.common.atoms[i].partial_charge {
                            let (r, g, b) = color_viridis_float(q, CHARGE_MAP_MIN, CHARGE_MAP_MAX);
                            (r, g, b, opacity)
                        } else {
                            (0., 0., 0., 0)
                        }
                    } else {
                        (0., 0., 0., 0)
                    }
                }
            };

            // Note: If opacity (a) is 0, the engine will use the entity color, and ignore vertex color.
            vertex.color = Some((
                (r * 255.) as u8,
                (g * 255.) as u8,
                (b * 255.) as u8,
                a,
            ));
        }
    }

    engine_updates.meshes = true;

    println!(
        "SAS mesh coloring done: {} atoms near surface, elapsed {:?}",
        atoms_near_surface.len(),
        start.elapsed()
    );
}
