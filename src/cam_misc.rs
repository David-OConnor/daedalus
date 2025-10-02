//! Various controls for moving the cam, moving things to the cam etc.

use graphics::{Camera, EngineUpdates, FWD_VEC, Scene};
use lin_alg::{
    f32::{Quaternion, Vec3 as Vec3F32},
    f64::Vec3,
};

use crate::{
    Selection, State, StateUi,
    lipid::MoleculeLipid,
    mol_lig::MoleculeSmall,
    molecule::{MoleculeCommon, MoleculePeptide},
    nucleic_acid::MoleculeNucleicAcid,
    render::{CAM_INIT_OFFSET, set_flashlight, set_static_light},
    ui::cam::{FOG_DIST_DEFAULT, VIEW_DEPTH_NEAR_MIN},
};

const MOVE_TO_TARGET_DIST: f32 = 15.;
pub const MOVE_CAM_TO_LIG_DIST: f32 = 30.;
const MOVE_TO_CAM_DIST: f32 = 25.;

/// Move the camera to look at a point of interest. Takes the starting location into account.
/// todo: Smooth interpolated zoom.
pub fn cam_look_at(cam: &mut Camera, target: Vec3) {
    let tgt: Vec3F32 = target.into();
    let diff = tgt - cam.position;
    let dir = diff.to_normalized();
    let dist = diff.magnitude();

    // Rotate the camera to look at the target.
    let cam_looking_at = cam.orientation.rotate_vec(FWD_VEC);
    let rotator = Quaternion::from_unit_vecs(cam_looking_at, dir);

    cam.orientation = rotator * cam.orientation;

    // Slide along the path between cam and target until close to it.
    let move_dist = dist - MOVE_TO_TARGET_DIST;
    cam.position += dir * move_dist;
}

pub fn cam_look_at_outside(cam: &mut Camera, target: Vec3F32, alignment: Vec3F32, dist: f32) {
    // Note: This is similar to `cam_look_at`, but we don't call that, as we're positioning
    // with an absolute orientation in mind, vice `cam_look_at`'s use of current cam LOS.

    // Look from the outside in, so our view is unobstructed by the protein. Do this after
    // the camera is positioned.
    let look_vec = (target - alignment).to_normalized();

    cam.position = target + look_vec * dist;
    cam.orientation = Quaternion::from_unit_vecs(FWD_VEC, -look_vec);
}

/// Resets the camera to the *front* view, and related settings. Its beahvior deepends
/// on the size and positions of open molecules.
pub fn reset_camera(
    state: &mut State,
    scene: &mut Scene,
    // view_depth: &mut (u16, u16),
    engine_updates: &mut EngineUpdates,
    // mol: &MoleculeCommon,
    look_vec: Vec3F32, // unit vector the cam is pointing to.
) {
    let mut center = Vec3F32::new_zero();
    let mut size = 8.; // E.g. for small organic molecules.

    if let Some(mol) = &state.peptide {
        // We cache center and size, due to the potential large number of molecules.
        center = mol.center.into();
        size = mol.size;
    } else {
        if let Some(mol) = state.active_mol() {
            center = mol.common().centroid().into();
            // Leaving size at its default for now.
        } else {
            let mut n = 0;
            let mut centroid = Vec3F32::new_zero();
            for mol in &state.ligands[0..10.min(state.ligands.len())] {
                let c: Vec3F32 = mol.common.centroid().into();
                centroid += c;
                n += 1;
            }
            for mol in &state.lipids[0..10.min(state.lipids.len())] {
                let c: Vec3F32 = mol.common.centroid().into();
                centroid += c;
                n += 1;
            }
            centroid /= n as f32;
            center = centroid;
            size = 40.; // A broad view.
        }
    }

    let dist_fm_center = size + CAM_INIT_OFFSET;

    // cam_look_at_outside(&mut scene.camera, center, Vec3F32::new_zero(), dist_fm_center);
    // let look_vec = (target - alignment).to_normalized();

    scene.camera.position = center - look_vec * dist_fm_center;
    scene.camera.orientation = Quaternion::from_unit_vecs(FWD_VEC, look_vec);

    set_static_light(scene, center, size);
    set_flashlight(scene);

    engine_updates.camera = true;
    engine_updates.lighting = true;

    // todo: A/R.
    state.ui.view_depth = (VIEW_DEPTH_NEAR_MIN, FOG_DIST_DEFAULT);
}

/// Move the camera to the selected atom or residue. If there is none, but there
/// is an active molecule, move the camera to that.
pub fn move_cam_to_sel(
    state_ui: &mut StateUi,
    mol_: &Option<MoleculePeptide>,
    ligs: &[MoleculeSmall],
    nucleic_acids: &[MoleculeNucleicAcid],
    lipids: &[MoleculeLipid],
    cam: &mut Camera,
    engine_updates: &mut EngineUpdates,
) {
    let mut selection_found = true;

    match &state_ui.selection {
        Selection::AtomPeptide(_i_atom) => {
            let Some(mol) = mol_ else {
                return;
            };
            let atom_sel = mol.get_sel_atom(&state_ui.selection);

            if let Some(atom) = atom_sel {
                cam_look_at(cam, atom.posit);
            }
        }
        Selection::AtomLig((i_mol, i_atom)) => {
            cam_look_at(cam, ligs[*i_mol].common.atom_posits[*i_atom]);
        }
        Selection::AtomNucleicAcid((i_mol, i_atom)) => {
            // if *i_mol >= nucleic_acids.len() {
            //     return;
            // }
            cam_look_at(cam, nucleic_acids[*i_mol].common.atom_posits[*i_atom]);
        }
        Selection::AtomLipid((i_mol, i_atom)) => {
            cam_look_at(cam, lipids[*i_mol].common.atom_posits[*i_atom]);
        }
        _ => {
            selection_found = false;
        }
    }

    if !selection_found {
        // todo: Get working; need to get active mol.
        // if let Some(mol) = state.active_mol() {
        //     cam_look_at(cam, mol.common().centroid());
        // }
    }

    engine_updates.camera = true;
    state_ui.cam_snapshot = None;
}

pub fn move_mol_to_cam(mol: &mut MoleculeCommon, cam: &Camera) {
    let new_posit = cam.position + cam.orientation.rotate_vec(FWD_VEC) * MOVE_TO_CAM_DIST;
    mol.move_to(new_posit.into());
}
