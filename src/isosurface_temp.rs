// todo: ROugh!

use graphics::Vertex;
use graphics::Mesh;
use kiddo::{KdTree, SquaredEuclidean};
use lin_alg::f32::Vec3 as Vec3F32;
use lin_alg::f64::Vec3;
use isosurface::{
   LinearHashedMarchingCubes,
    source::ScalarSource,
    math::vector,
   sampler::Sample,
   distance::Signed,
   extractor::Extractor,
};
use isosurface::extractor::IndexedInterleavedNormals;
use crate::reflection::ElectronDensity;


use nalgebra::{Point3, Vector3};

/// Sampled on a regular - **cubic** - grid.
struct GridField {
    data: Vec<f32>,
    dims: (usize, usize, usize),
    origin: [f32; 3],
    h: f32,                       // voxel edge length
}

impl GridField {
    #[inline]
    fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        let (nx, ny, _) = self.dims;
        x + nx * (y + ny * z)
    }
}

impl ScalarSource for GridField {
    /// Trilinear interpolation at **world** position `p`
    fn sample_scalar(&self, p: vector::Vec3) -> Signed {
        let lx = ((p.x - self.origin[0]) / self.h).clamp(0.0, (self.dims.0 - 1) as f32);
        let ly = ((p.y - self.origin[1]) / self.h).clamp(0.0, (self.dims.1 - 1) as f32);
        let lz = ((p.z - self.origin[2]) / self.h).clamp(0.0, (self.dims.2 - 1) as f32);

        let xi = lx.floor() as usize;
        let yi = ly.floor() as usize;
        let zi = lz.floor() as usize;
        let tx = lx - xi as f32;
        let ty = ly - yi as f32;
        let tz = lz - zi as f32;

        let c = |dx, dy, dz| self.data[self.idx(xi + dx, yi + dy, zi + dz)];

        // standard tri-linear blend
        let v000 = c(0, 0, 0);
        let v100 = c(1, 0, 0);
        let v010 = c(0, 1, 0);
        let v110 = c(1, 1, 0);
        let v001 = c(0, 0, 1);
        let v101 = c(1, 0, 1);
        let v011 = c(0, 1, 1);
        let v111 = c(1, 1, 1);

        let v00 = v000 * (1.0 - tx) + v100 * tx;
        let v10 = v010 * (1.0 - tx) + v110 * tx;
        let v01 = v001 * (1.0 - tx) + v101 * tx;
        let v11 = v011 * (1.0 - tx) + v111 * tx;

        let v0 = v00 * (1.0 - ty) + v10 * ty;
        let v1 = v01 * (1.0 - ty) + v11 * ty;

        Signed(v0 * (1.0 - tz) + v1 * tz)
    }
}

impl Sample<Signed> for GridField {
    #[inline]
    fn sample(&self, p: vector::Vec3) -> Signed {
        self.sample_scalar(p)          // simply forward
    }
}


pub fn create_isosurface(points: &[ElectronDensity], iso: f32) -> Mesh {
    /* ---------- 1. Deduce a regular grid from the point cloud ---------- */
    // collect unique sorted coords → infer (nx, ny, nz) and voxel size
    let mut xs: Vec<f64> = points.iter().map(|p| p.coords.x).collect();
    let mut ys: Vec<f64> = points.iter().map(|p| p.coords.y).collect();
    let mut zs: Vec<f64> = points.iter().map(|p| p.coords.z).collect();

    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    zs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    ys.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    zs.dedup_by(|a, b| (*a - *b).abs() < 1e-6);

    let (nx, ny, nz) = (xs.len(), ys.len(), zs.len());
    assert!(nx > 1 && ny > 1 && nz > 1, "need a regular 3-D grid");

    let hx = (xs[1] - xs[0]) as f32;
    let hy = (ys[1] - ys[0]) as f32;
    let hz = (zs[1] - zs[0]) as f32;
    assert!(
        (hx - hy).abs() < 1e-4 && (hx - hz).abs() < 1e-4,
        "grid must be cubic (equal spacing)"
    );
    let h = hx;                                        // voxel edge

    /* ---------- 2. Resample into a dense array ---------- */
    let mut field = vec![0f32; nx * ny * nz];
    let mut tree: KdTree<f64, 3> = KdTree::new();

    let mut extractor = IndexedInterleavedNormals::new(&mut vertices, &mut indices, &sampler);

    // map world → lattice index for direct inserts
    let idx = |ix, iy, iz| ix + nx * (iy + ny * iz);

    for (i, p) in points.iter().enumerate() {
        tree.add(&[p.coords.x, p.coords.y, p.coords.z], i as u64);
        // exact-grid point – safe to cast
        let ix = xs.binary_search_by(|x| x.partial_cmp(&p.coords.x).unwrap()).unwrap();
        let iy = ys.binary_search_by(|y| y.partial_cmp(&p.coords.y).unwrap()).unwrap();
        let iz = zs.binary_search_by(|z| z.partial_cmp(&p.coords.z).unwrap()).unwrap();
        field[idx(ix, iy, iz)] = p.density as f32;
    }

    // optionally fill any gaps by IDW using the k-d tree (omitted for brevity)

    /* ---------- 3. Marching-Cubes ---------- */
    let mut source = GridField {
        data: field,
        dims: (nx, ny, nz),
        origin: [xs[0] as f32, ys[0] as f32, zs[0] as f32],
        h,
    };

    let max_depth = 3; // todo?
    let mut mc = LinearHashedMarchingCubes::new(max_depth);
    mc.extract(&source, iso);

    let vertices: Vec<Vertex> = mc
        .vertices
        .iter()
        .map(|v| {
            // back to world units
            let pos = [
                v.x * h + source.origin[0],
                v.y * h + source.origin[1],
                v.z * h + source.origin[2],
            ];

            // rough gradient for the normal
            let eps = 1.0;
            let sdf = |x: f32, y: f32, z: f32| source.sample_scalar(vector::Vec3::new(x, y, z)).0;

            let grad = Vector3::new(
                sdf(v.x + eps, v.y, v.z) - sdf(v.x - eps, v.y, v.z),
                sdf(v.x, v.y + eps, v.z) - sdf(v.x, v.y - eps, v.z),
                sdf(v.x, v.y, v.z + eps) - sdf(v.x, v.y, v.z - eps),
            )
                .normalize();

            Vertex::new(pos, Vec3::new(grad.x as f64, grad.y as f64, grad.z as f64))
        })
        .collect();

    let indices: Vec<usize> = mc.indices.iter().map(|&i| i as usize).collect();

    Mesh {
        vertices,
        indices,
        material: 0,
    }
}
