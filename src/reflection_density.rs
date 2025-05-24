//! For displaying electron density as measured by crytalographics reflection data. From precomputed
//! data, or from Miller indices.

use std::f64::consts::TAU;

use lin_alg::complex_nums::{Cplx, IM};

/// One reflection: Miller index + amplitude + phase (in degrees)
struct Reflection {
    h: i32,
    k: i32,
    l: i32,
    f: f64,       // amplitude
    phi: f64,     // phase in degrees
}

/// Compute electron density at fractional coordinate (x, y, z), from Miller indices.
fn compute_density(reflections: &[Reflection], x: f64, y: f64, z: f64) -> f64 {
    let mut real = 0.0;
    let mut imag = 0.0;

    for r in reflections {
        let phase_rad = r.phi.to_radians();
        let arg = -TAU * (r.h as f64 * x + r.k as f64 * y + r.l as f64 * z);
        let exp = Cplx::from_mag_phase(1.0, arg);
        let contrib = Cplx::from_mag_phase(1.0, phase_rad) * r.f * exp;

        real += contrib.real;
        imag += contrib.im;
    }

    // Return the real part (imaginary part should ideally cancel)
    real
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum MapType {
    M2FOFC,
    DifferenceMap
}

impl MapType {
    pub fn from_str(val: &str) -> Option<MapType> {
        match val.to_lowercase().as_ref() {
            "o" => Some(MapType::M2FOFC),
            "d" => Some(MapType::DifferenceMap),
            _ => {
                eprintln!("Fallthrough on map type: {val}");
                None
            }
        }
    }
}

#[derive(Debug)]
/// E.g. from structural feature mmCIF.
pub struct DensityPt {
    // pub grid_id: GridId, // todo? Generally 1 1 1
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub map_type: MapType,
    pub density: f32,
    pub uncertainty: f32,
}