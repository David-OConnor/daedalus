//! For creating ORCA QM inputs to run, and visualizing output.
//!
//! [ORCA recommendations for methods, basis fns etc](https://www.faccts.de/docs/orca/6.1/manual/contents/quickstartguide/recommendations.html)

use crate::State;
use bio_files::orca::dynamics::DynamicsOutput;
use bio_files::orca::{OrcaInput, basis_sets::BasisSetCategory};
use dynamics::snapshot::Snapshot;

#[derive(Default)]
// todo: Some of this is UI state; move to a place that makes sense A/R.
pub struct StateOrca {
    pub input: OrcaInput,
    pub basis_set_cat: BasisSetCategory,
}

pub fn update_snapshots(state: &mut State, out: DynamicsOutput) {
    match &mut state.mol_dynamics {
        Some(md) => {
            println!("A");

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
            println!("B");
            // state.mol_dynamics = Some(MdState::new(
            //
            // ))
        }
    }
}
