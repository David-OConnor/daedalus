//! For water molecules, the sim box, thermostat etc.

use lin_alg::f64::Vec3;
use na_seq::Element;

use crate::dynamics::{AtomDynamics, MdState};

// todo: A/R
pub struct Water {
    // todo: Track a posit and velocit for the whole molecule, and model with an
    // todo orientation quaternion?
    // pub posit: Vec3,
    // pub orientation: Quaternion,
    pub o: AtomDynamics,
    pub h0: AtomDynamics,
    pub h1: AtomDynamics,
}

/// Add water molecules
fn hydrate(pressure: f64, temp: f64, bounds: (Vec3, Vec3), n_mols: usize) -> Vec<Water> {
    let mut result = Vec::with_capacity(n_mols);
    // todo: TIP4P to start?
    for _ in 0..n_mols {
        result.push(Water {
            o: AtomDynamics {
                element: Element::Oxygen,
                name: "wo".to_string(), // todo: Qc
                // todo
                posit: Vec3::new_zero(),
                // todo: Init vel based on temp and pressure?
                vel: Vec3::new_zero(),
                accel: Vec3::new_zero(),
                mass: 8.,
                partial_charge: 0.,
                force_field_type: None,
            },
            h0: AtomDynamics {
                // todo
                element: Element::Hydrogen,
                name: "wo".to_string(), // todo: Qc
                posit: Vec3::new_zero(),
                // todo: Init vel based on temp and pressure?
                vel: Vec3::new_zero(),
                accel: Vec3::new_zero(),
                mass: 1.,
                partial_charge: 0.,
                force_field_type: None,
            },
            h1: AtomDynamics {
                element: Element::Hydrogen,
                name: "wo".to_string(), // todo: Qc
                // todo
                posit: Vec3::new_zero(),
                // todo: Init vel based on temp and pressure?
                vel: Vec3::new_zero(),
                accel: Vec3::new_zero(),
                mass: 1.,
                partial_charge: 0.,
                force_field_type: None,
            },
        })
    }

    result
}

/// Simulation cell (orthorhombic for now)
#[derive(Clone, Copy, Default)]
pub struct SimBox {
    pub lo: Vec3,
    pub hi: Vec3,
}

impl SimBox {
    #[inline]
    pub fn extent(&self) -> Vec3 {
        self.hi - self.lo
    }

    /// wrap an absolute coordinate back into the box (orthorhombic)
    #[inline]
    pub fn wrap(&self, p: Vec3) -> Vec3 {
        let ext = self.extent();

        assert!(
            ext.x > 0.0 && ext.y > 0.0 && ext.z > 0.0,
            "SimBox edges must be > 0 (lo={:?}, hi={:?})",
            self.lo,
            self.hi
        );

        // rem_euclid keeps the value in [0, ext)
        let wrapped = Vec3::new(
            (p.x - self.lo.x).rem_euclid(ext.x) + self.lo.x,
            (p.y - self.lo.y).rem_euclid(ext.y) + self.lo.y,
            (p.z - self.lo.z).rem_euclid(ext.z) + self.lo.z,
        );

        wrapped
    }

    /// minimum-image displacement vector (no √)
    #[inline]
    pub fn min_image(&self, dv: Vec3) -> Vec3 {
        let ext = self.extent();
        debug_assert!(ext.x > 0.0 && ext.y > 0.0 && ext.z > 0.0);

        Vec3::new(
            dv.x - (dv.x / ext.x).round() * ext.x,
            dv.y - (dv.y / ext.y).round() * ext.y,
            dv.z - (dv.z / ext.z).round() * ext.z,
        )
    }
}

/// Add `n` TIP3P molecules uniformly in the box.
pub fn add_tip3p(state: &mut MdState, n: usize, rng: &mut impl rand::Rng) {
    use lin_alg::f64::Vec3;
    use na_seq::Element;
    use rand::Rng;
    use rand_distr::{Distribution, UnitSphere};

    use crate::dynamics::{ANG_HOH, AtomDynamics, M_H, M_O, R_OH};
    for _ in 0..n {
        // random COM inside box
        let rand3 = Vec3::new(rng.random::<f64>(), rng.random(), rng.random());

        // todo: What should this be doing?
        let com = state.cell.lo + rand3.hadamard_product(state.cell.extent());

        // random orientation – unit vector + perpendicular
        let z = Vec3::from_slice(&UnitSphere.sample(rng)).unwrap();
        // todo: This is a good idea...
        let x = z.any_perpendicular().to_normalized();
        let y = z.cross(x);

        // two H positions in the HOH plane
        let d = R_OH * (ANG_HOH / 2.0).sin();
        let h0 = com + z * R_OH;
        let h1 = com + z * (-R_OH * ANG_HOH.cos()) + x * d * 2.;

        let mut make = |pos, mass, q, element| AtomDynamics {
            element,
            name: "".to_string(), // todo
            posit: pos,
            vel: Vec3::new_zero(),
            accel: Vec3::new_zero(),
            mass,
            partial_charge: q,
            force_field_type: None,
        };

        state.atoms.push(make(com, M_O, -0.834, Element::Oxygen));
        state.atoms.push(make(h0, M_H, 0.417, Element::Hydrogen));
        state.atoms.push(make(h1, M_H, 0.417, Element::Hydrogen));
    }
    state.build_neighbours(); // list is stale now
}
