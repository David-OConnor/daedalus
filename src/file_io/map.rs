//! For parsing CCP4/MRC .map files. These contain electron density; they are computed
//! from reflection data, using a fourier transform.

use std::{
    fs,
    fs::File,
    io::{self, ErrorKind, Read, Seek, SeekFrom, Write},
    path::Path,
    process::Command,
};

use bio_apis::rcsb;
use byteorder::{LittleEndian, ReadBytesExt};
use lin_alg::f64::{Mat3, Vec3};

use crate::reflection::ElectronDensity;

const HEADER_SIZE: u64 = 1_024;

/// Minimal subset of the 1024-byte CCP4/MRC header
#[allow(unused)]
#[derive(Clone, Debug)]
pub struct MapHeader {
    /// Numbemr of grid points along each axis. nx × ny × nz is the number of voxels.
    pub nx: i32,
    pub ny: i32,
    pub nz: i32,
    pub mode: i32, // data type (0=int8, 1=int16, 2=float32, …)
    /// the `n[xyz]start` values specify the grid starting point (offset) in each dimension.
    /// They are 0 in most cryo-EM maps.
    pub nxstart: i32,
    pub nystart: i32,
    pub nzstart: i32,
    /// the m values give number of grid-sampling intervals along each axis. This is usually equal
    /// to the n values. In this case, each voxel corresponds to one spacing unit. They will differ
    /// in the case of over and under sampling.
    pub mx: i32,
    pub my: i32,
    pub mz: i32,
    /// Unit cell dimensions. [XYZ] length. Then α: Angle between Y and Z, β: Angle
    /// between X and Z, and γ: ANgle between X and Y. Distances are in Å, and angles are in degrees.
    pub cell: [f32; 6], // cell dimensions: a, b, c, alpha, beta, gamma
    pub mapc: i32,  // which axis is fast (1=X, 2=Y, 3=Z)
    pub mapr: i32,  // which axis is medium
    pub maps: i32,  // which axis is slow
    pub dmin: f32,  // minimum density value
    pub dmax: f32,  // maximum density value
    pub dmean: f32, // mean density
    pub ispg: i16,  // space group number
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
            ErrorKind::InvalidData,
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

#[derive(Clone, Debug)]
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

fn read_map_data_raw(path: &Path) -> io::Result<(MapHeader, Vec<f32>)> {
    let mut file = File::open(path)?;
    let hdr = read_map_header(&mut file)?;

    if hdr.mode != 2 {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            format!("Unsupported mode: {}", hdr.mode),
        ));
    }

    file.seek(SeekFrom::Start(HEADER_SIZE + hdr.nsymbt as u64))?;

    let n = (hdr.nx * hdr.ny * hdr.nz) as usize;
    let mut data = Vec::with_capacity(n);

    for _ in 0..n {
        data.push(file.read_f32::<LittleEndian>()?);
    }

    Ok((hdr, data))
}

/// Reads the entire density grid.
/// Assumes `mode == 2` (i.e. 32-bit floats);
pub fn read_map_data(path: &Path) -> io::Result<(MapHeader, Vec<ElectronDensity>)> {
    let (hdr, data) = read_map_data_raw(path)?;

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

    // let v_a = Vec3::new(a, 0.0, 0.0);
    // let v_b = Vec3::new(b * cosγ, b * sinγ, 0.0);
    let cx = c * cosβ;
    let cy = c * (cosα - cosβ * cosγ) / sinγ;
    let cz = c * (1.0 - cosβ * cosβ - ((cosα - cosβ * cosγ) / sinγ).powi(2)).sqrt();
    // let v_c = Vec3::new(cx, cy, cz);

    // // Origin offset: Voxel units
    // let step = [a / hdr.mx as f64, b / hdr.my as f64, c / hdr.mz as f64];

    // let start_vox = if let (Some(xo), Some(yo), Some(zo)) = (hdr.xorigin, hdr.yorigin, hdr.zorigin)
    // {
    //     [
    //         xo as f64 / step[0],
    //         yo as f64 / step[1],
    //         zo as f64 / step[2],
    //     ]
    // } else {
    //     [hdr.nxstart as f64, hdr.nystart as f64, hdr.nzstart as f64]
    // };

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

    // Read voxels; build coordinates.

    let mut densities = Vec::with_capacity(npoints);

    let cell = UnitCell::new(
        hdr.cell[0] as f64,
        hdr.cell[1] as f64,
        hdr.cell[2] as f64,
        hdr.cell[3] as f64,
        hdr.cell[4] as f64,
        hdr.cell[5] as f64,
    );

    // 2.  Origin in *fractional* coordinates
    let origin_frac =
        if let (Some(ox), Some(oy), Some(oz)) = (hdr.xorigin, hdr.yorigin, hdr.zorigin) {
            cell.cartesian_to_fractional(Vec3::new(ox as f64, oy as f64, oz as f64))
        } else {
            Vec3::new(
                hdr.nxstart as f64 / hdr.mx as f64,
                hdr.nystart as f64 / hdr.my as f64,
                hdr.nzstart as f64 / hdr.mz as f64,
            )
        };

    let mut idx = 0;
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let density = data[idx] as f64;
                idx += 1;

                let idx_file = [i as f64, j as f64, k as f64]; // voxel corners
                let frac = origin_frac
                    + Vec3::new(
                        (idx_file[f_of_c[0]] + 0.5) / hdr.mx as f64,
                        (idx_file[f_of_c[1]] + 0.5) / hdr.my as f64,
                        (idx_file[f_of_c[2]] + 0.5) / hdr.mz as f64,
                    );

                let coords = cell.fractional_to_cartesian(frac); // Å

                densities.push(ElectronDensity { coords, density });
            }
        }
    }

    // // todo: Symmetry augmentation experimentation.
    // let mut slid_x_ = Vec::with_capacity(npoints);
    // for density in &densities {
    //     let slid_x = ElectronDensity {
    //         // coords: Vec3::new(density.x - hdr.dmax, density.y, density.z),
    //         coords: Vec3::new(density.coords.x + hdr.cell[0] as f64 * nx as f64, density.coords.y, density.coords.z),
    //         density: density.density,
    //     };
    //     slid_x_.push(slid_x);
    // }
    //
    // densities.append(&mut slid_x_);

    Ok((hdr, densities))
}

#[derive(Clone, Debug)]
/// The 1-D array is stored in **file order**
///   fastest-varying index = MAPC
pub struct DensityMap {
    pub hdr: MapHeader,
    pub cell: UnitCell,
    pub origin_frac: Vec3,    // header origin, already converted to fractional
    pub perm_f2c: [usize; 3], // file-axis → cryst-axis (from MAPC/MAPR/MAPS)
    pub perm_c2f: [usize; 3], // cryst-axis → file-axis (inverse permutation)
    pub data: Vec<f32>,
}

impl DensityMap {
    /// Build the helper once
    pub fn new(path: &Path) -> io::Result<Self> {
        let (hdr, data) = read_map_data_raw(path)?; // see § 7
        let cell = UnitCell::new(
            hdr.cell[0] as f64,
            hdr.cell[1] as f64,
            hdr.cell[2] as f64,
            hdr.cell[3] as f64,
            hdr.cell[4] as f64,
            hdr.cell[5] as f64,
        );

        let perm_f2c = [
            hdr.mapc as usize - 1,
            hdr.mapr as usize - 1,
            hdr.maps as usize - 1,
        ];
        let mut perm_c2f = [0usize; 3];
        for (f, c) in perm_f2c.iter().enumerate() {
            perm_c2f[*c] = f;
        }

        // header origin → fractional
        let origin_frac =
            if let (Some(ox), Some(oy), Some(oz)) = (hdr.xorigin, hdr.yorigin, hdr.zorigin) {
                cell.cartesian_to_fractional(Vec3::new(ox as f64, oy as f64, oz as f64))
            } else {
                Vec3::new(
                    hdr.nxstart as f64 / hdr.mx as f64,
                    hdr.nystart as f64 / hdr.my as f64,
                    hdr.nzstart as f64 / hdr.mz as f64,
                )
            };

        Ok(Self {
            hdr,
            cell,
            origin_frac,
            perm_f2c,
            perm_c2f,
            data,
        })
    }

    /// Positive modulus that always lands in 0..n-1
    #[inline(always)]
    fn pmod(i: isize, n: usize) -> usize {
        ((i % n as isize) + n as isize) as usize % n
    }

    /// Nearest-neighbour lookup – add trilinear if you like
    pub fn rho(&self, cart: Vec3) -> f32 {
        // 1. Cart → frac (wrap immediately to [0,1) )
        let mut frac = self.cell.cartesian_to_fractional(cart);
        frac.x -= frac.x.floor();
        frac.y -= frac.y.floor();
        frac.z -= frac.z.floor();

        // 2. frac → crystallographic voxel index (float)
        let frac_rel = frac - self.origin_frac;
        let ic = [
            (frac_rel.x * self.hdr.mx as f64 - 0.5).round() as isize,
            (frac_rel.y * self.hdr.my as f64 - 0.5).round() as isize,
            (frac_rel.z * self.hdr.mz as f64 - 0.5).round() as isize,
        ];

        // 3. crystallographic → file order, then wrap with pmod
        let ifile = [
            Self::pmod(ic[self.perm_f2c[0]], self.hdr.nx as usize),
            Self::pmod(ic[self.perm_f2c[1]], self.hdr.ny as usize),
            Self::pmod(ic[self.perm_f2c[2]], self.hdr.nz as usize),
        ];

        // 4. linear offset in file order (x fastest)
        let offset = (ifile[2] * self.hdr.ny as usize + ifile[1]) * self.hdr.nx as usize + ifile[0];
        self.data[offset]
    }
}

/// Stopgap approach?
pub fn density_from_rcsb_gemmi(ident: &str) -> io::Result<(MapHeader, Vec<ElectronDensity>)> {
    println!("Downloading Map data for {ident}...");

    let map_2fo_fc = rcsb::load_validation_2fo_fc_cif(ident)
        .map_err(|_| io::Error::new(ErrorKind::InvalidData, "Problem loading 2fo-fc from RCSB"))?;

    fs::write("temp_map.cif", map_2fo_fc)?;

    let _status = Command::new("gemmi")
        .args(["sf2map", "temp_map.cif", "temp_map.map"])
        .status()?;

    let (hdr, map) = read_map_data(Path::new("temp_map.map"))?;

    fs::remove_file(Path::new("temp_map.cif"))?;
    fs::remove_file(Path::new("temp_map.map"))?;

    Ok((hdr, map))
}

pub fn density_from_rcsb_gemmi2(ident: &str) -> io::Result<DensityMap> {
    println!("Downloading Map data for {ident}...");

    let map_2fo_fc = rcsb::load_validation_2fo_fc_cif(ident)
        .map_err(|_| io::Error::new(ErrorKind::InvalidData, "Problem loading 2fo-fc from RCSB"))?;

    fs::write("temp_map.cif", map_2fo_fc)?;

    let _status = Command::new("gemmi")
        .args(["sf2map", "temp_map.cif", "temp_map.map"])
        .status()?;

    let map = DensityMap::new(Path::new("temp_map.map"))?;

    fs::remove_file(Path::new("temp_map.cif"))?;
    fs::remove_file(Path::new("temp_map.map"))?;

    Ok(map)
}
