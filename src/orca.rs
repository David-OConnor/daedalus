//! For creating ORCA QM inputs to run, and visualizing output.
//!
//! [ORCA recommendations for methods, basis fns etc](https://www.faccts.de/docs/orca/6.1/manual/contents/quickstartguide/recommendations.html)

use std::fmt::Display;

use bio_files::orca::{
    GeomOptThresh, OrcaInput, Task,
    basis_sets::BasisSetCategory,
    dynamics::{Dynamics, DynamicsOutput},
    geom::Geom,
};
use dynamics::snapshot::Snapshot;

use crate::State;

#[derive(Default)]
// todo: Some of this is UI state; move to a place that makes sense A/R.
pub struct StateOrca {
    pub input: OrcaInput,
    pub basis_set_cat: BasisSetCategory,
    pub task_type: TaskType,
    /// For the drop-down; only relevant for the geometry optimization task.
    pub geom_opt_thresh: GeomOptThresh,
    // pub dynamics: Dynamics,
}

/// A copy-type, e.g. without the inner values of bio_files::orca::Task.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum TaskType {
    SinglePoint,
    /// Note: Not the Orca default, nor `bio_files::orca::Task` default.
    #[default]
    GeometryOptimization,
    MbisCharges,
    MolDynamics,
}

impl TaskType {
    pub fn help_text(self) -> String {
        match self {
            Self::SinglePoint => "Compute the single-point energy of this molecule.",
            Self::GeometryOptimization => "Compute optimal geometry, and apply this to this molecule's atoms.",
            Self::MbisCharges =>                             "Compute and assign MBIS partial charges for this molecule. This is an accurate QM method, but is very \
                            slow; it may take 10 minutes or longer for a small organic molecule. This replaces any existing partial \
                            charges on this molecule.",
            Self::MolDynamics =>                             "Run MD using ORCA. This is much slower than our normal MD system, but \
                             more accurate Uses settings from the MD section of the UI as well, including number of steps,\
                             dt, and temperature.",
        }.to_string()
    }
}

impl Display for TaskType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let v = match self {
            Self::SinglePoint => "Single pt energy",
            Self::GeometryOptimization => "Optimize geom",
            Self::MbisCharges => "MBIS charges",
            Self::MolDynamics => "Mol dynamics",
        };

        write!(f, "{v}")
    }
}

pub fn update_snapshots(state: &mut State, out: DynamicsOutput) {
    match &mut state.mol_dynamics {
        Some(md) => {
            md.snapshots = Vec::new();
            for (i, step) in out.trajectory.iter().enumerate() {
                let time = i as f32 * state.to_save.md_dt * 1_000.;

                // todo: You still need to reassign the atoms to this
                // todo for the playback.
                let atom_posits: Vec<_> = step.atoms.iter().map(|a| a.posit.into()).collect();

                md.snapshots.push(Snapshot {
                    // todo: You can also get time from the comment.
                    time: time as f64,
                    atom_posits,
                    atom_velocities: Vec::new(),
                    water_o_posits: Vec::new(),
                    water_h0_posits: Vec::new(),
                    water_h1_posits: Vec::new(),
                    water_velocities: Vec::new(),
                    energy_kinetic: 0.,
                    energy_potential: 0.,
                    energy_potential_between_mols: Vec::new(),
                    hydrogen_bonds: Vec::new(),
                    temperature: 0.,
                    pressure: 0.,
                })
            }
        }
        None => {
            // state.mol_dynamics = Some(MdState::new(
            //
            // ))
        }
    }
}
