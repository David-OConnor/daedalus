#![allow(clippy::excessive_precision)]

//! https://acc2.ncbr.muni.cz/
//! https://github.com/sb-ncbr/eem_parameters/

// todo: Deprecate this module

use barnes_hut::BodyModel;
use lin_alg::{f32::Vec3, f64::Vec3 as Vec3F64};

const GRID_SIZE: f64 = 1.6; // Slightly larger than the largest... todo: What?

#[derive(Debug)]
pub struct PartialCharge {
    pub posit: Vec3,
    pub charge: f32,
}

// todo: Ideally, you want f32 here, I believe. Need to modifify barnes_hut lib A/R to support.
impl BodyModel for PartialCharge {
    fn posit(&self) -> Vec3F64 {
        // fn posit(&self) -> Vec3 {
        self.posit.into()
    }

    fn mass(&self) -> f64 {
        // fn mass(&self) -> f32 {
        self.charge.into()
    }
}
