//! For parsing CCP4/MRC .map files. These contain electron density; they are computed
//! from reflection data, using a fourier transform.

use std::{
    fs,
    fs::File,
    io::{self, ErrorKind, Read, Seek, SeekFrom, Write},
    path::Path,
    process::Command,
};

use bio_apis::{ReqError, rcsb};
use byteorder::{LittleEndian, ReadBytesExt};
use lin_alg::f64::{Mat3, Vec3};

use crate::reflection::ElectronDensity;

const HEADER_SIZE: u64 = 1_024;

/// Minimal subset of the 1024-byte CCP4/MRC header
#[derive(Debug)]
pub struct MapHeader {
    pub nx: i32,      // grid points along X
    pub ny: i32,      // grid points along Y
    pub nz: i32,      // grid points along Z
    pub mode: i32,    // data type (0=int8, 1=int16, 2=float32, …)
    pub nxstart: i32, // origin (usually 0,0,0)
    pub nystart: i32,
    pub nzstart: i32,
    pub mx: i32, // sampling (usually = nx,ny,nz)
    pub my: i32,
    pub mz: i32,
    pub cell: [f32; 6], // cell dimensions: a, b, c, alpha, beta, gamma
    pub mapc: i32,      // which axis is fast (1=X, 2=Y, 3=Z)
    pub mapr: i32,      // which axis is medium
    pub maps: i32,      // which axis is slow
    pub dmin: f32,      // minimum density value
    pub dmax: f32,      // maximum density value
    pub dmean: f32,     // mean density
    pub ispg: i16,      // space group number
    /// Number of bytes used by the symmetry block. (Usually 0 for cry-EM, and 80xn for crystallography.)
    pub nsymbt: i16,
    /// origin in Å (MRC-2014) **or** derived from *\*START if absent**
    pub xorigin: Option<f32>,
    pub yorigin: Option<f32>,
    pub zorigin: Option<f32>, // todo: More header items A/R.
}

fn read_map_header<R: Read + Seek>(mut r: R) -> io::Result<MapHeader> {
    // ensure we are at the start
    r.seek(SeekFrom::Start(0))?;

    // Classic part of the header.
    let nx = r.read_i32::<LittleEndian>()?;
    let ny = r.read_i32::<LittleEndian>()?;
    let nz = r.read_i32::<LittleEndian>()?;
    let mode = r.read_i32::<LittleEndian>()?;

    let nxstart = r.read_i32::<LittleEndian>()?;
    let nystart = r.read_i32::<LittleEndian>()?;
    let nzstart = r.read_i32::<LittleEndian>()?;

    let mx = r.read_i32::<LittleEndian>()?;
    let my = r.read_i32::<LittleEndian>()?;
    let mz = r.read_i32::<LittleEndian>()?;

    let mut cell = [0f32; 6];
    for c in &mut cell {
        *c = r.read_f32::<LittleEndian>()?;
    }

    let mapc = r.read_i32::<LittleEndian>()?;
    let mapr = r.read_i32::<LittleEndian>()?;
    let maps = r.read_i32::<LittleEndian>()?;

    let dmin = r.read_f32::<LittleEndian>()?;
    let dmax = r.read_f32::<LittleEndian>()?;
    let dmean = r.read_f32::<LittleEndian>()?;

    let ispg = r.read_i16::<LittleEndian>()?;
    let nsymbt = r.read_i16::<LittleEndian>()?;

    // Words 25-49 are “extra”; skip straight to word 50 (49 × 4 bytes)
    r.seek(SeekFrom::Start(49 * 4))?;

    // words 50-52 = XORIGIN, YORIGIN, ZORIGIN   (MRC-2014)
    let mut xorigin_ = r.read_f32::<LittleEndian>()?;
    let mut yorigin_ = r.read_f32::<LittleEndian>()?;
    let mut zorigin_ = r.read_f32::<LittleEndian>()?;

    let mut xorigin = None;
    let mut yorigin = None;
    let mut zorigin = None;

    let mut tag = [0u8; 4];
    r.seek(SeekFrom::Start(52 * 4))?;
    r.read_exact(&mut tag)?;

    if &tag != b"MAP " {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid MAP tag in header.",
        ));
    }

    const EPS: f32 = 0.0001;

    if xorigin_.abs() > EPS {
        xorigin = Some(xorigin_);
    }

    if yorigin_.abs() > EPS {
        yorigin = Some(yorigin_);
    }

    if zorigin_.abs() > EPS {
        zorigin = Some(zorigin_);
    }

    // skip ahead to end of 1024-byte header
    r.seek(SeekFrom::Start(HEADER_SIZE))?;

    Ok(MapHeader {
        nx,
        ny,
        nz,
        mode,
        nxstart,
        nystart,
        nzstart,
        mx,
        my,
        mz,
        cell,
        mapc,
        mapr,
        maps,
        dmin,
        dmax,
        dmean,
        ispg,
        nsymbt,
        xorigin,
        yorigin,
        zorigin,
    })
}

pub struct UnitCell {
    a: f64,
    b: f64,
    c: f64,
    alpha: f64,
    beta: f64,
    gamma: f64,
    ortho: Mat3,     // frac  -> cart
    ortho_inv: Mat3, // cart -> frac
}

impl UnitCell {
    pub fn new(a: f64, b: f64, c: f64, alpha_deg: f64, beta_deg: f64, gamma_deg: f64) -> Self {
        let (α, β, γ) = (
            alpha_deg.to_radians(),
            beta_deg.to_radians(),
            gamma_deg.to_radians(),
        );

        // components of the three cell vectors in Cartesian space
        let v_a = Vec3::new(a, 0.0, 0.0);
        let v_b = Vec3::new(b * γ.cos(), b * γ.sin(), 0.0);

        let cx = c * β.cos();
        let cy = c * (α.cos() - β.cos() * γ.cos()) / γ.sin();
        let cz = c * (1.0 - β.cos().powi(2) - cy.powi(2) / c.powi(2)).sqrt();
        let v_c = Vec3::new(cx, cy, cz);

        // 3×3 matrix whose columns are a,b,c
        let ortho = Mat3::from_cols(v_a, v_b, v_c);
        let ortho_inv = ortho.inverse().expect("unit-cell matrix is singular");

        Self {
            a,
            b,
            c,
            alpha: α,
            beta: β,
            gamma: γ,
            ortho,
            ortho_inv,
        }
    }

    #[inline]
    pub fn fractional_to_cartesian(&self, f: Vec3) -> Vec3 {
        // todo: Don't clone!
        self.ortho.clone() * f
    }

    #[inline]
    pub fn cartesian_to_fractional(&self, c: Vec3) -> Vec3 {
        // todo: Don't clone!
        self.ortho_inv.clone() * c
    }
}

/// Reads the entire density grid.
/// Assumes `mode == 2` (i.e. 32-bit floats);
pub fn read_map_data(path: &Path) -> io::Result<(MapHeader, Vec<ElectronDensity>)> {
    let mut file = File::open(path)?;
    let hdr = read_map_header(&mut file)?;

    if hdr.mode != 2 {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            format!("Unsupported mode: {}", hdr.mode),
        ));
    }

    file.seek(SeekFrom::Start(HEADER_SIZE + hdr.nsymbt as u64))?;

    // todo: QC all these; simplify A/R.

    // Compute crystal geometry
    let (nx, ny, nz) = (hdr.nx as usize, hdr.ny as usize, hdr.nz as usize);
    let npoints = nx * ny * nz;

    let a = hdr.cell[0] as f64;
    let b = hdr.cell[1] as f64;
    let c = hdr.cell[2] as f64;

    let α = (hdr.cell[3] as f64).to_radians();
    let β = (hdr.cell[4] as f64).to_radians();
    let γ = (hdr.cell[5] as f64).to_radians();

    // Crystallographic basis vectors
    let cosα = α.cos();
    let cosβ = β.cos();
    let cosγ = γ.cos();
    let sinγ = γ.sin();

    let v_a = Vec3::new(a, 0.0, 0.0);
    let v_b = Vec3::new(b * cosγ, b * sinγ, 0.0);
    let cx = c * cosβ;
    let cy = c * (cosα - cosβ * cosγ) / sinγ;
    let cz = c * (1.0 - cosβ * cosβ - ((cosα - cosβ * cosγ) / sinγ).powi(2)).sqrt();
    let v_c = Vec3::new(cx, cy, cz);

    // Origin offset: Voxel units
    let step = [a / hdr.mx as f64, b / hdr.my as f64, c / hdr.mz as f64];

    let start_vox = if let (Some(xo), Some(yo), Some(zo)) = (hdr.xorigin, hdr.yorigin, hdr.zorigin)
    {
        [
            xo as f64 / step[0],
            yo as f64 / step[1],
            zo as f64 / step[2],
        ]
    } else {
        [hdr.nxstart as f64, hdr.nystart as f64, hdr.nzstart as f64]
    };

    // Axis permutation: file -> crystal
    let perm = [
        hdr.mapc as usize - 1, // header values are 1-based
        hdr.mapr as usize - 1,
        hdr.maps as usize - 1,
    ];
    // inverse permutation: crystal_axis → file_axis
    let mut f_of_c = [0usize; 3];
    for (file_axis, cryst_axis) in perm.iter().enumerate() {
        f_of_c[*cryst_axis] = file_axis;
    }

    // unit‐cell volume = |v_a ⋅ (v_b × v_c)|
    // let volume = v_a.dot(v_b.cross(v_c)).abs();

    // Read voxels; build coordinates.

    let mut densities = Vec::with_capacity(npoints);

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let raw = file.read_f32::<LittleEndian>()? as f64;

                // // indices in *file* order, shifted to voxel centre (+0.5)
                // let idx_file = [i as f64 + 0.5, j as f64 + 0.5, k as f64 + 0.5];
                //
                // // fractional crystal coords
                // let frac = [
                //     (idx_file[f_of_c[0]] + start_vox[0]) / hdr.mx as f64,
                //     (idx_file[f_of_c[1]] + start_vox[1]) / hdr.my as f64,
                //     (idx_file[f_of_c[2]] + start_vox[2]) / hdr.mz as f64,
                // ];
                //
                // // Cartesian Å coordinate
                // let coord = v_a * frac[0] + v_b * frac[1] + v_c * frac[2];

                // densities.push(ElectronDensity {
                //     coords: coord,
                //     density: raw, // keep raw value; contour in UI
                // });

                // 1.  Build a proper UnitCell object once
                let cell = UnitCell::new(
                    hdr.cell[0] as f64,
                    hdr.cell[1] as f64,
                    hdr.cell[2] as f64,
                    hdr.cell[3] as f64,
                    hdr.cell[4] as f64,
                    hdr.cell[5] as f64,
                );

                // 2.  Origin in *fractional* coordinates
                let origin_frac = if let (Some(ox), Some(oy), Some(oz)) =
                    (hdr.xorigin, hdr.yorigin, hdr.zorigin)
                {
                    cell.cartesian_to_fractional(Vec3::new(ox as f64, oy as f64, oz as f64))
                } else {
                    Vec3::new(
                        hdr.nxstart as f64 / hdr.mx as f64,
                        hdr.nystart as f64 / hdr.my as f64,
                        hdr.nzstart as f64 / hdr.mz as f64,
                    )
                };

                // 3.  Inside the triple-nested loop
                let idx_file = [i as f64, j as f64, k as f64]; // voxel corners
                let frac = origin_frac
                    + Vec3::new(
                        (idx_file[f_of_c[0]] + 0.5) / hdr.mx as f64,
                        (idx_file[f_of_c[1]] + 0.5) / hdr.my as f64,
                        (idx_file[f_of_c[2]] + 0.5) / hdr.mz as f64,
                    );

                // 4.  Convert with the full orthogonalisation matrix
                let coord = cell.fractional_to_cartesian(frac); // Å

                densities.push(ElectronDensity {
                    coords: coord,
                    density: raw as f64,
                });
            }
        }
    }

    // for k in 0..nz {
    //     for j in 0..ny {
    //         for i in 0..nx {
    //             let raw = file.read_f32::<LittleEndian>()?;
    //             // let density = raw as f64 / volume;
    //
    //             // todo: Temp?
    //             let density = raw as f64 / (hdr.dmax as f64);
    //
    //             // We are assuming the present of one origin means the others are too.
    //             let (x_start, y_start, z_start) = match hdr.xorigin {
    //                 Some(x) => (x as f64, hdr.yorigin.unwrap_or_default() as f64, hdr.zorigin.unwrap_or_default() as f64),
    //                 None => (hdr.nxstart as f64, hdr.nystart as f64, hdr.nzstart as f64)
    //             };
    //
    //             // fractional coords including header offset
    //             let fx = (i as f64 + x_start) / hdr.mx as f64;
    //             let fy = (j as f64 + y_start) / hdr.my as f64;
    //             let fz = (k as f64 + z_start) / hdr.mz as f64;
    //
    //             // todo: Use origins if available.
    //
    //             // Cartesian Å coordinate
    //             let coord = v_a * fx + v_b * fy + v_c * fz;
    //
    //             densities.push(ElectronDensity {
    //                 coords: coord,
    //                 density,
    //             });
    //         }
    //     }
    // }

    Ok((hdr, densities))
}

// /// return 4×4 matrix that converts CIF coords → crystallographic orthogonal Å
// fn cif_scale_matrix(cif: &CifDocument) -> Option<[[f64;4];4]> {
//     let s = |row, col| cif.get("_atom_sites.Cartn_scale_matrix", row, col)?.parse().ok();
//     let v = |row| [s(row,0)?, s(row,1)?, s(row,2)?, cif.get("_atom_sites.Cartn_trans_vector", row, 0)?.parse().ok()?];
//     Some([v(0)?, v(1)?, v(2)?, [0.0, 0.0, 0.0, 1.0]])
// }

/// Stopgap approach?
pub fn density_from_rcsb_gemmi(ident: &str) -> io::Result<Vec<ElectronDensity>> {
    println!("Downloading Map data for {ident}...");

    let map_2fo_fc = rcsb::load_validation_2fo_fc_cif(ident)
        .map_err(|_| io::Error::new(ErrorKind::InvalidData, "Problem loading 2fo-fc from RCSB"))?;

    fs::write("temp_map.cif", map_2fo_fc)?;

    let _status = Command::new("gemmi")
        .args(["sf2map", "temp_map.cif", "temp_map.map"])
        .status()?;

    let (_hdr, map) = read_map_data(Path::new("temp_map.map"))?;

    fs::remove_file(Path::new("temp_map.cif"))?;
    fs::remove_file(Path::new("temp_map.map"))?;

    Ok(map)
}
