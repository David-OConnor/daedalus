//! A new approach, leveraging our molecular dynamics state and processes.

use bincode::{Decode, Encode};
use lin_alg::f64::{Quaternion, Vec3};
#[derive(Clone, Debug, Default)]
/// Bonds that are marked as flexible, using a semi-rigid conformation.
pub struct Torsion {
    pub bond: usize, // Index.
    pub dihedral_angle: f32,
}

#[derive(Debug, Clone, Encode, Decode)]
/// Area IVO the docking site.
pub struct DockingSite {
    pub site_center: Vec3,
    pub site_radius: f64,
}

impl Default for DockingSite {
    fn default() -> Self {
        Self {
            site_center: Vec3::new_zero(),
            site_radius: 8.,
        }
    }
}

// // todo: Rem if not used.
// #[derive(Clone, Debug, Default)]
// pub enum ConformationType {
//     #[default]
//     /// Don't reposition atoms based on the pose. This is what we use when assigning each atom
//     /// a position using molecular dynamics.
//     AbsolutePosits,
//     // Rigid,
//     /// Certain bonds are marked as flexible, with rotation allowed around them.
//     AssignedTorsions { torsions: Vec<Torsion> },
// }

// todo: Rem if not used.
#[derive(Clone, Debug, Default)]
pub struct Pose {
    // pub conformation_type: ConformationType,
    // /// The offset of the ligand's anchor atom from the docking center.
    // /// Only for rigid and torsion-set-based conformations.
    // /// todo: Consider normalizing positions to be around the origin, for numerical precision issues.
    // pub anchor_posit: Vec3,
    // /// Only for rigid and torsion-set-based conformations.
    // pub orientation: Quaternion,

    pub posits: Vec<Vec3>
}

pub struct DockingPose {
    lig_atom_posits: Vec<Vec3>,
    potential_energy: f64,
}

#[derive(Debug, Default)]
pub struct DockingState {}
