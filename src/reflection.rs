//! For displaying electron density as measured by crytalographics reflection data. From precomputed
//! data, or from Miller indices.

#![allow(unused)]

use std::{f64::consts::TAU, time::Instant};

use bio_apis::{ReqError, rcsb};
use bio_files::{DensityMap, MapHeader, UnitCell};
use lin_alg::f64::Vec3;
use mcubes::GridPoint;
use rayon::prelude::*;

use crate::{molecule::Atom, util::setup_neighbor_pairs};

pub const DENSITY_CELL_MARGIN: f64 = 2.0;
// Density points must be within this distance in Å of a (backbone?) atom to be generated.
pub const DENSITY_MAX_DIST: f64 = 3.0;

#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum MapStatus {
    /// Ordinary, or observed; the bulk of values.
    Observed,
    FreeSet,
    SystematicallyAbsent,
    OutsideHighResLimit,
    HigherThanResCutoff,
    LowerThanResCutoff,
    /// Ignored
    #[default]
    UnreliableMeasurement,
}

impl MapStatus {
    pub fn from_str(val: &str) -> Option<MapStatus> {
        match val.to_lowercase().as_ref() {
            "o" => Some(MapStatus::Observed),
            // "o" => Some(MapType::M2FOFC),
            // "d" => Some(MapType::DifferenceMap),
            "f" => Some(MapStatus::FreeSet),
            "-" => Some(MapStatus::SystematicallyAbsent),
            "<" => Some(MapStatus::OutsideHighResLimit),
            "h" => Some(MapStatus::HigherThanResCutoff),
            "l" => Some(MapStatus::LowerThanResCutoff),
            "x" => Some(MapStatus::UnreliableMeasurement),
            _ => {
                eprintln!("Fallthrough on map type: {val}");
                None
            }
        }
    }
}

#[allow(unused)]
/// Reflection data for a single Miller index set. Pieced together from 3 formats of CIF
/// file (Structure factors, map 2fo-fc, and map fo-fc), or an MTZ.
#[derive(Clone, Default, Debug)]
pub struct Reflection {
    /// Miller indices.
    pub h: i32,
    pub k: i32,
    pub l: i32,
    pub status: MapStatus,
    /// Amplitude. i.e. F_meas. From SF.
    pub amp: f64,
    /// Standard uncertainty (σ) of amplitude. i.e. F_meas_sigma_au. From SF.
    pub amp_uncertainty: f64,
    /// ie. FWT. From 2fo-fc.
    pub amp_weighted: Option<f64>,
    /// i.e. PHWT. In degrees. From 2fo-fc.
    pub phase_weighted: Option<f64>,
    /// i.e. FOM. From 2fo-fc.
    pub phase_figure_of_merit: Option<f64>,
    /// From fo-fc.
    pub delta_amp_weighted: Option<f64>,
    /// From fo-fc.
    pub delta_phase_weighted: Option<f64>,
    /// From fo-fc.
    pub delta_figure_of_merit: Option<f64>,
}

/// Miller-index-based reflection data.
#[derive(Clone, Debug, Default)]
pub struct ReflectionsData {
    /// X Y Z for a b c?
    pub space_group: String,
    pub cell_len_a: f32,
    pub cell_len_b: f32,
    pub cell_len_c: f32,
    pub cell_angle_alpha: f32,
    pub cell_angle_beta: f32,
    pub cell_angle_gamma: f32,
    pub points: Vec<Reflection>,
}

impl ReflectionsData {
    /// Load reflections data from RCSB, then parse. (SF, 2fo_fc, and fo_fc)
    pub fn load_from_rcsb(ident: &str) -> Result<Self, ReqError> {
        println!("Downloading structure factors and Map data for {ident}...");

        let sf = match rcsb::load_structure_factors_cif(ident) {
            Ok(m) => Some(m),
            Err(_) => {
                eprintln!("Error loading structure factors CIF");
                None
            }
        };

        let map_2fo_fc = match rcsb::load_validation_2fo_fc_cif(ident) {
            Ok(m) => Some(m),
            Err(_) => {
                eprintln!("Error loading 2fo_fc map");
                None
            }
        };

        let map_fo_fc = match rcsb::load_validation_fo_fc_cif(ident) {
            Ok(m) => Some(m),
            Err(_) => {
                eprintln!("Error loading fo_fc map");
                None
            }
        };

        println!("Download complete. Parsing...");
        Ok(Self::from_cifs(
            sf.as_deref(),
            map_2fo_fc.as_deref(),
            map_fo_fc.as_deref(),
        ))
    }

    //     /// 1. Make a regular fractional grid that spans 0–1 along a, b, c.
    //     /// We use this grid for computing electron densitites; it must be converted to real space,
    //     /// e.g. in angstroms, prior to display.
    //     pub fn regular_fractional_grid(&self, n: usize) -> Vec<Vec3> {
    //         let mut pts = Vec::with_capacity(n.pow(3));
    //         let step = 1. / n as f64;
    //
    //         for i in 0..n {
    //             for j in 0..n {
    //                 for k in 0..n {
    //                     pts.push(Vec3 {
    //                         x: i as f64 * step,
    //                         y: j as f64 * step,
    //                         z: k as f64 * step,
    //                     });
    //                 }
    //             }
    //         }
    //
    //         pts
    //     }
    // }

    /// 1. Make a regular fractional grid that spans 0–1 along a, b, c.
    /// We use this grid for computing electron densitites; it must be converted to real space,
    /// e.g. in angstroms, prior to display.
    pub fn regular_fractional_grid(&self, n: usize) -> Vec<Vec3> {
        let step = 1.0 / n as f64;
        let shift = -0.5 + step / 2.0; // put voxel centres at –½…+½
        let mut pts = Vec::with_capacity(n.pow(3));

        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    pts.push(Vec3 {
                        x: i as f64 * step + shift,
                        y: j as f64 * step + shift,
                        z: k as f64 * step + shift,
                    });
                }
            }
        }

        pts
    }
}

#[derive(Clone, Debug)]
pub struct ElectronDensity {
    /// In Å
    pub coords: Vec3,
    /// Normalized, using the unit cell volume, as reported in the reflection data.
    pub density: f64,
}

impl GridPoint for ElectronDensity {
    // fn coords(&self) -> Vec3 {self.coords}
    fn value(&self) -> f64 {
        self.density
    }
}

fn compute_density(reflections: &[Reflection], posit: Vec3, unit_cell_vol: f32) -> f64 {
    // todo: Use SIMD or GPU for this.

    const EPS: f64 = 0.0000001;
    let mut rho = 0.0;

    for r in reflections {
        if r.status != MapStatus::Observed {
            continue;
        }

        let amp = r.amp_weighted.unwrap_or(r.amp);
        if amp.abs() < EPS {
            continue;
        }

        let Some(phase) = r.phase_weighted else {
            continue;
        };

        //  2π(hx + ky + lz)  (negative sign because CCP4/Coot convention)
        let arg = TAU * (r.h as f64 * posit.x + r.k as f64 * posit.y + r.l as f64 * posit.z);
        //  real part of  F · e^{iφ} · e^{iarg} = amp·cos(φ+arg)

        // todo: Which sign/order?
        rho += amp * (arg + phase.to_radians()).cos();
        // rho += amp * (arg - phase.to_radians()).cos();
    }

    // Normalize.
    // rho / unit_cell_vol as f64

    // todo temp
    rho * 4. / unit_cell_vol as f64
}

/// Compute electron density from reflection data. Simmilar to gemmi's `sf2map`.
pub fn compute_density_grid(data: &ReflectionsData) -> Vec<ElectronDensity> {
    let grid = data.regular_fractional_grid(90);
    let unit_cell_vol = data.cell_len_a * data.cell_len_b * data.cell_len_c;

    println!(
        "Computing electron density from refletions onver {} points...",
        grid.len()
    );

    let start = Instant::now();

    let len_a = data.cell_len_a as f64;
    let len_b = data.cell_len_b as f64;
    let len_c = data.cell_len_c as f64;

    let result = grid
        .par_iter()
        .map(|p| ElectronDensity {
            // coords: *p,
            // Convert coords to real space, in angstroms.
            // coords: Vec3 {
            //     x: p.x * len_a,
            //     y: p.y * len_b,
            //     z: p.z * len_c,
            // },
            // coords: frac_to_cart(
            coords: frac_to_cart3(
                *p,
                len_a,
                len_b,
                len_c,
                (data.cell_angle_alpha as f64).to_radians(),
                (data.cell_angle_beta as f64).to_radians(),
                (data.cell_angle_gamma as f64).to_radians(),
            ),
            density: compute_density(&data.points, *p, unit_cell_vol),
        })
        .collect();

    let elapsed = start.elapsed().as_millis();

    println!("Complete. Time: {:?}ms", elapsed);
    result
}

/// Convert from fractical coordinates, as used in reflections, to real space in Angstroms.
fn frac_to_cart(fr: Vec3, a: f64, b: f64, c: f64, α: f64, β: f64, γ: f64) -> Vec3 {
    // Angles in radians
    let (ca, cb, cg) = (α.cos(), β.cos(), γ.cos());
    let sg = γ.sin();

    // Volume factor
    let v = (1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg).sqrt();

    // Orthogonalisation matrix (PDB convention 1)
    let ox = Vec3 {
        x: a,
        y: 0.0,
        z: 0.0,
    };
    let oy = Vec3 {
        x: b * cg,
        y: b * sg,
        z: 0.0,
    };
    let oz = Vec3 {
        x: c * cb,
        y: c * (ca - cb * cg) / sg,
        z: c * v / sg,
    };

    Vec3 {
        x: ox.x * fr.x + oy.x * fr.y + oz.x * fr.z,
        y: ox.y * fr.x + oy.y * fr.y + oz.y * fr.z,
        z: ox.z * fr.x + oy.z * fr.y + oz.z * fr.z,
    }
}

fn frac_to_cart3(
    frac: Vec3,
    a: f64,
    b: f64,
    c: f64,
    alpha_deg: f64,
    beta_deg: f64,
    gamma_deg: f64,
) -> Vec3 {
    let (alpha, beta, gamma) = (
        alpha_deg.to_radians(),
        beta_deg.to_radians(),
        gamma_deg.to_radians(),
    );

    // cos and sin of the angles
    let (ca, cb, cg) = (alpha.cos(), beta.cos(), gamma.cos());
    let sg = gamma.sin();

    // volume factor (G² in International Tables)
    let v = (1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg).sqrt();

    // International Tables orthogonalisation, PDB convention
    let x = a * frac.x + b * cg * frac.y + c * cb * frac.z;

    let y = b * sg * frac.y + c * (ca - cb * cg) / sg * frac.z;

    let z = c * v / sg * frac.z;

    Vec3 { x, y, z }
}

/// Electron density maps are ususally provided in terms of a cell which may not directly
/// encompass the entire protein. We copy electron density from the opposite side until
/// the protein is enclosed. We also remove parts of the density not near the protein.
pub fn handle_map_symmetry(map: &mut [ElectronDensity], hdr: &MapHeader, atoms: &[Atom]) {}

// /// Intermediate struct required by the IsoSurface lib.
// struct Source {
//
// }
//
// fn make_mesh(density: &[ElectronDensity], iso_val: f32)

//////////////// gpt below...

/// Convert Å → fractional using inverse cell matrix
fn cart_to_frac(p: Vec3, inv: &[[f64; 3]; 3]) -> Vec3 {
    Vec3::new(
        inv[0][0] * p.x + inv[0][1] * p.y + inv[0][2] * p.z,
        inv[1][0] * p.x + inv[1][1] * p.y + inv[1][2] * p.z,
        inv[2][0] * p.x + inv[2][1] * p.y + inv[2][2] * p.z,
    )
}

/// Convert fractional → Å using the cell vectors
fn frac_to_cart2(f: Vec3, ax: Vec3, bx: Vec3, cx: Vec3) -> Vec3 {
    ax * f.x + bx * f.y + cx * f.z
}

/// Build the cell vectors (ax,bx,cx) and inverse matrix (Å⁻¹)
fn cell_matrices(cell: &[f32; 6]) -> (Vec3, Vec3, Vec3, [[f64; 3]; 3]) {
    let (a, b, c) = (cell[0] as f64, cell[1] as f64, cell[2] as f64);
    let (al, be, ga) = (
        cell[3] as f64 * TAU / 360.,
        cell[4] as f64 * TAU / 360.,
        cell[5] as f64 * TAU / 360.,
    );

    let ax = Vec3::new(a, 0.0, 0.0);
    let bx = Vec3::new(b * ga.cos(), b * ga.sin(), 0.0);
    let cx = {
        let cx = c * be.cos();
        let cy = c * (al.cos() - be.cos() * ga.cos()) / ga.sin();
        let cz = c
            * (1.0 - al.cos().powi(2) - be.cos().powi(2) - ga.cos().powi(2)
                + 2.0 * al.cos() * be.cos() * ga.cos())
            .sqrt()
            / ga.sin();
        Vec3::new(cx, cy, cz)
    };

    // inverse(A) where A has columns ax,bx,cx
    let a_inv = {
        let det = ax.x * (bx.y * cx.z - bx.z * cx.y) - bx.x * (ax.y * cx.z - ax.z * cx.y)
            + cx.x * (ax.y * bx.z - ax.z * bx.y);
        let inv = 1.0 / det;
        let m = |v: Vec3| v;
        [
            [
                (m(bx).y * m(cx).z - m(bx).z * m(cx).y) * inv,
                (m(ax).z * m(cx).y - m(ax).y * m(cx).z) * inv,
                (m(ax).y * m(bx).z - m(ax).z * m(bx).y) * inv,
            ],
            [
                (m(bx).z * m(cx).x - m(bx).x * m(cx).z) * inv,
                (m(ax).x * m(cx).z - m(ax).z * m(cx).x) * inv,
                (m(ax).z * m(bx).x - m(ax).x * m(bx).z) * inv,
            ],
            [
                (m(bx).x * m(cx).y - m(bx).y * m(cx).x) * inv,
                (m(ax).y * m(cx).x - m(ax).x * m(cx).y) * inv,
                (m(ax).x * m(bx).y - m(ax).y * m(bx).x) * inv,
            ],
        ]
    };

    (ax, bx, cx, a_inv)
}

/// Wrap every atom coordinate back into the unit cell so the
/// existing *single* asymmetric-unit map encloses them.
///
/// After this call your renderer can sample `map` without ever
/// running out of density.
pub fn wrap_atoms_into_cell(hdr: &MapHeader, atoms: &mut [Atom]) {
    if atoms.is_empty() {
        return;
    }

    let (ax, bx, cx, a_inv) = cell_matrices(&hdr.cell);

    for at in atoms {
        // ● Å → fractional
        let mut f = cart_to_frac(at.posit, &a_inv);

        // ● wrap each fractional coord into [0,1)
        f.x -= f.x.floor();
        f.y -= f.y.floor();
        f.z -= f.z.floor();

        // ● back to Cartesian Å
        at.posit = frac_to_cart2(f, ax, bx, cx);
    }
}

/// One dense 3-D brick of map values. We use this struct to handle symmetry: ensuring full coverage
/// of all atoms.
#[derive(Clone, Debug)]
pub struct DensityRect {
    /// Cartesian coordinate of the *centre* of voxel (0,0,0)
    pub origin_cart: Vec3,
    /// Size of one voxel along a,b,c in Å
    pub step: [f64; 3],
    /// (nx, ny, nz) – number of voxels stored
    pub dims: [usize; 3],
    /// Row-major file-order data:  z → y → x fastest
    pub data: Vec<f32>,
}

impl DensityRect {
    /// Extract the smallest cube that covers all atoms plus `margin` Å.
    /// `margin = 0.0` means “touch each atom’s centre”.
    pub fn new(atom_posits: &[Vec3], map: &DensityMap, margin: f64) -> Self {
        let hdr = &map.hdr;
        let cell = &map.cell;

        // ────────────────────────────────────────────────────────────────
        // 1. Atom bounds in fractional coords *relative to map origin*
        // ────────────────────────────────────────────────────────────────
        let mut min_r = Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut max_r = Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);

        for p in atom_posits {
            // (a) Cartesian → absolute fractional
            let mut f = cell.cartesian_to_fractional(*p);
            // (b) shift so that origin_frac becomes (0,0,0)
            f -= map.origin_frac;

            // keep *unwrapped* values (they can be <0 or >1)
            min_r = Vec3::new(min_r.x.min(f.x), min_r.y.min(f.y), min_r.z.min(f.z));
            max_r = Vec3::new(max_r.x.max(f.x), max_r.y.max(f.y), max_r.z.max(f.z));
        }

        // extra margin in fractional units
        let margin_r = Vec3::new(margin / cell.a, margin / cell.b, margin / cell.c);
        min_r -= margin_r;
        max_r += margin_r;

        // ────────────────────────────────────────────────────────────────
        // 2.  Fractional → voxel indices (no wrapping)
        // ────────────────────────────────────────────────────────────────
        let to_idx = |fr: f64, n: i32| -> isize { (fr * n as f64 - 0.5).floor() as isize };

        let lo_i = [
            to_idx(min_r.x, hdr.mx),
            to_idx(min_r.y, hdr.my),
            to_idx(min_r.z, hdr.mz),
        ];
        let hi_i = [
            to_idx(max_r.x, hdr.mx),
            to_idx(max_r.y, hdr.my),
            to_idx(max_r.z, hdr.mz),
        ];

        // inclusive → dims       (now guaranteed hi_i ≥ lo_i)
        let dims = [
            (hi_i[0] - lo_i[0] + 1) as usize,
            (hi_i[1] - lo_i[1] + 1) as usize,
            (hi_i[2] - lo_i[2] + 1) as usize,
        ];

        // ────────────────────────────────────────────────────────────────
        // 3. Cartesian centre of voxel (0,0,0); add origin_frac only once
        // ────────────────────────────────────────────────────────────────
        let lo_frac = Vec3::new(
            (lo_i[0] as f64 + 0.5) / hdr.mx as f64,
            (lo_i[1] as f64 + 0.5) / hdr.my as f64,
            (lo_i[2] as f64 + 0.5) / hdr.mz as f64,
        ) + map.origin_frac; // back to absolute fractional

        let origin_cart = cell.fractional_to_cartesian(lo_frac);

        // Voxel step vectors in Å
        let step = [
            cell.a / hdr.mx as f64,
            cell.b / hdr.my as f64,
            cell.c / hdr.mz as f64,
        ];

        // ——————————————————————————————————————————
        // 4.  Sample the map
        let mut data = Vec::with_capacity(dims[0] * dims[1] * dims[2]);

        for kz in 0..dims[2] {
            for ky in 0..dims[1] {
                for kx in 0..dims[0] {
                    let idx_c = [
                        lo_i[0] + kx as isize,
                        lo_i[1] + ky as isize,
                        lo_i[2] + kz as isize,
                    ];

                    // crystallographic → Cartesian centre of this voxel
                    let frac = map.origin_frac
                        + Vec3::new(
                            (idx_c[0] as f64 + 0.5) / hdr.mx as f64,
                            (idx_c[1] as f64 + 0.5) / hdr.my as f64,
                            (idx_c[2] as f64 + 0.5) / hdr.mz as f64,
                        );
                    let cart = cell.fractional_to_cartesian(frac);

                    let density = map.density_at_point_trilinear(cart);
                    let dens_sig = map.density_to_sig(density);
                    data.push(dens_sig);
                }
            }
        }

        Self {
            origin_cart,
            step,
            dims,
            data,
        }
    }

    /// Convert a DensityCube into (coords, density) structs.
    ///
    /// Works for any unit-cell.  We rely on the same `UnitCell` you already have,
    /// so non-orthogonal (triclinic, monoclinic, …) cells come out correct.
    pub fn make_densities(
        &self,
        atom_posits: &[Vec3],
        cell: &UnitCell,
        dist_thresh: f64,
    ) -> Vec<ElectronDensity> {
        // Step vectors along a, b, c.
        let cols = cell.ortho.to_cols();

        // length of one voxel along the a-axis in Å  =  a / mx
        let step_vec_a = cols.0 * (self.step[0] / cell.a); //  = a_vec / mx
        let step_vec_b = cols.1 * (self.step[1] / cell.b); //  = b_vec / my
        let step_vec_c = cols.2 * (self.step[2] / cell.c); //  = c_vec / mz

        let (nx, ny, nz) = (self.dims[0], self.dims[1], self.dims[2]);
        let mut out = Vec::with_capacity(nx * ny * nz);

        // todo: Experimenting with neighbor grid here
        let indices: Vec<_> = (0..atom_posits.len()).collect();

        // todo: Temp dealing with the ref vs not.
        // let atom_posits2: Vec<&_> = atom_posits.iter().map(|p| p).collect();
        // const GRID: f64 = 1.0; // todo: Experiment.
        // let neighbor_pairs = setup_neighbor_pairs(&atom_posits2, &indices, GRID);

        for kz in 0..nz {
            for ky in 0..ny {
                for kx in 0..nx {
                    // linear index in self.data  (z → y → x fastest)
                    let idx = (kz * ny + ky) * nx + kx;
                    let mut density = self.data[idx] as f64;

                    // Cartesian centre of this voxel
                    let coords = self.origin_cart
                        + step_vec_a * kx as f64
                        + step_vec_b * ky as f64
                        + step_vec_c * kz as f64;

                    // todo: Insert code here to, using a neighbors algorithm and/or only select
                    // todo backboneo atoms (e.g. only Calpha etc), don't include points pas a certain
                    // todo min distance from this backbone. (Performance saver over checking all atoms)

                    // neighbor_pairs
                    //     .par_iter()
                    //     .filter_map(|(i, j)| {
                    //         let atom_0 = &atom_posits[*i];
                    //         let atom_1 = &atom_posits[*j];
                    //         let dist = (atom_0 - atom_1).magnitude();

                    // todo: Too slow, but can work for now.
                    let mut nearest_dist = 99999.;
                    for p in atom_posits {
                        let dist = (*p - coords).magnitude();
                        if dist < nearest_dist {
                            nearest_dist = dist;
                        }
                    }

                    if nearest_dist > dist_thresh {
                        // We set density to 0, vice removing the coordinates; our marching cubes
                        // algorithm requires a regular grid, with no absent values.
                        density = 0.;
                    }

                    out.push(ElectronDensity { coords, density });
                }
            }
        }
        out
    }
}
