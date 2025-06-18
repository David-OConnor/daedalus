//! For calculating the accessible surface (AS), a proxy for acccessibility of solvents
//! to a molecule. Used for drawing *surface*, *dots*, and related meshes. Related to the van der Waals
//! radius.
//!
//! Uses the Shrake-Rupley, or similar "rolling ball" methods.
//! [This Rust lib](https://github.com/maxall41/RustSASA) appearse to be unsuitable to our purpose;
//! it provides a single 'total SASA value', vice a set of points defining a surface.

use graphics::{Mesh, Vertex};
use lin_alg::f32::Vec3;
use mcubes::{MarchingCubes, MeshSide};

use crate::molecule::Atom;

const SOLVENT_RAD: f32 = 1.4; // water probe
// const GRID_H: f32 = 0.5; // voxel edge length

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

    let nvox = grid_dim.0 * grid_dim.1 * grid_dim.2;

    // This can be any that is guaranteed to be well outside the SAS surface.
    // It prevents holes from appearing in the mesh due to not having a value outside to compare to.
    let far_val = (r_max + precision).powi(2) + 1.0;
    let mut field = vec![far_val; nvox];

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
    //  scale = precision because size / sampling_interval = precision
    let size = (
        (grid_dim.0 as f32 - 1.0) * precision,
        (grid_dim.1 as f32 - 1.0) * precision,
        (grid_dim.2 as f32 - 1.0) * precision,
    );
    let samp = (
        grid_dim.0 as f32 - 1.0,
        grid_dim.1 as f32 - 1.0,
        grid_dim.2 as f32 - 1.0,
    );

    // todo: The holes in our mesh seem related to the iso level chosen.
    let mc =
        MarchingCubes::new(grid_dim, size, samp, bb_min, field, 0.).expect("marching cubes init");

    // I believe even with one side, it draws both, as it creates an isosurface surrounding the value. (?)
    let mc_mesh = mc.generate(MeshSide::OutsideOnly);

    let vertices: Vec<Vertex> = mc_mesh
        .vertices
        .iter()
        .map(|v| Vertex::new([v.posit.x, v.posit.y, v.posit.z], v.normal))
        .collect();

    Mesh {
        vertices,
        indices: mc_mesh.indices,
        material: 0,
    }
}
