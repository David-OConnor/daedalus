//! For creating ORCA QM inputs to run, and visualizing output.
//!
//! [ORCA recommendations for methods, basis fns etc](https://www.faccts.de/docs/orca/6.1/manual/contents/quickstartguide/recommendations.html)

use bio_files::orca::{OrcaInput, basis_sets::BasisSetCategory, solvation::Solvator};

#[derive(Default)]
// todo: Some of this is UI state; movem to a place that makes sense A/R.
pub struct StateOrca {
    pub input: OrcaInput,
    pub basis_set_cat: BasisSetCategory,
}
