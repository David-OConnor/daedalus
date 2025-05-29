//! For parsing CCP4/MRC .map files. These contain electron density; they are computed
//! from reflection data, using a fourier transform.

use std::{
    fs::File,
    io::{self, Read, Seek, SeekFrom},
    path::Path,
};

use byteorder::{LittleEndian, ReadBytesExt};
use lin_alg::f64::Vec3;

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
    pub nsymbt: i16,    // bytes used for symmetry operators (usually 0)
                        // todo: More header items A/R.
}

fn read_map_header<R: Read + Seek>(mut r: R) -> io::Result<MapHeader> {
    // ensure we are at the start
    r.seek(SeekFrom::Start(0))?;

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
    })
}

/// Reads the entire density grid.
/// Assumes `mode == 2` (i.e. 32-bit floats);
pub fn read_map_data(path: &Path) -> io::Result<(MapHeader, Vec<ElectronDensity>)> {
    let mut file = File::open(path)?;
    let hdr = read_map_header(&mut file)?;

    if hdr.mode != 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported mode: {}", hdr.mode),
        ));
    }

    // todo: QC all these; simplify A/R.
    let (nx, ny, nz) = (hdr.nx as usize, hdr.ny as usize, hdr.nz as usize);
    let npoints = nx * ny * nz;

    let a = hdr.cell[0] as f64;
    let b = hdr.cell[1] as f64;
    let c = hdr.cell[2] as f64;
    let alpha = (hdr.cell[3] as f64).to_radians();
    let beta = (hdr.cell[4] as f64).to_radians();
    let gamma = (hdr.cell[5] as f64).to_radians();

    // Crystallographic basis vectors
    let cosα = alpha.cos();
    let cosβ = beta.cos();
    let cosγ = gamma.cos();
    let sinγ = gamma.sin();

    let v_a = Vec3::new(a, 0.0, 0.0);
    let v_b = Vec3::new(b * cosγ, b * sinγ, 0.0);
    let cx = c * cosβ;
    let cy = c * (cosα - cosβ * cosγ) / sinγ;
    let cz = c * (1.0 - cosβ * cosβ - ((cosα - cosβ * cosγ) / sinγ).powi(2)).sqrt();
    let v_c = Vec3::new(cx, cy, cz);

    // unit‐cell volume = |v_a ⋅ (v_b × v_c)|
    let volume = v_a.dot(v_b.cross(v_c)).abs();

    let mut densities = Vec::with_capacity(npoints);

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let raw = file.read_f32::<LittleEndian>()?;
                // let density = raw as f64 / volume;

                // todo: Temp?
                let density = raw as f64 / (hdr.dmax as f64);

                // fractional coords including header offset
                let fx = (i as f64 + hdr.nxstart as f64) / hdr.mx as f64;
                let fy = (j as f64 + hdr.nystart as f64) / hdr.my as f64;
                let fz = (k as f64 + hdr.nzstart as f64) / hdr.mz as f64;

                // Cartesian Å coordinate
                let coord = v_a * fx + v_b * fy + v_c * fz;

                densities.push(ElectronDensity {
                    coords: coord,
                    density,
                });
            }
        }
    }

    Ok((hdr, densities))
}
