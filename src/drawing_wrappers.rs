//! Small mol update wrappers. This this focused on index management
//! etc to update entities in place when able.

// todo: Drawing module A/R.

use graphics::{Entity, Scene};

use crate::{
    OperatingMode, State, drawing,
    drawing::EntityClass,
    molecule::{MolGenericRef, MolGenericTrait, MolType},
    util::clear_mol_entity_indices,
};

fn helper_a(scene: &Scene, class: u32) -> (usize, usize) {
    // Assumes all entities are this class are contiguous in the entity list.
    let initial_ent_count = scene.entities.iter().filter(|e| e.class == class).count();

    let mut ent_i_start = 0;
    for (i, ent) in scene.entities.iter().enumerate() {
        if ent.class == class {
            ent_i_start = i;
            break;
        }
    }

    (initial_ent_count, ent_i_start)
}

/// Note: We can get rid of these helpers and simplify things further if we can abstract over
/// mol types with traits of the Ref structs.
fn helper_b(
    state: &mut State,
    scene: &mut Scene,
    entities: Vec<Entity>,
    class: u32,
    initial_ent_count: usize,
    ent_i_start: usize,
    mol_type: MolType,
) {
    if entities.len() == initial_ent_count {
        for (i, ent) in scene.entities[ent_i_start..ent_i_start + initial_ent_count]
            .iter_mut()
            .enumerate()
        {
            // This must include all fields which have changed.
            ent.position = entities[i].position;
            ent.orientation = entities[i].orientation;
            ent.scale = entities[i].scale;
            ent.scale_partial = entities[i].scale_partial;
            ent.color = entities[i].color;
            ent.overlay_text = entities[i].overlay_text.clone();
        }
    } else {
        // Full rebuild
        scene.entities.retain(|ent| ent.class != class);
        scene.entities.extend(entities);

        clear_mol_entity_indices(state, Some(mol_type));
    }
}

pub fn draw_all_ligs(state: &mut State, scene: &mut Scene) {
    // draw_all_mol_of_type(state, scene, &mut state.ligands, MolType::ligand, state.ui.visibility.hide_ligands);
    let class = EntityClass::Ligand as u32;
    let (initial_ent_count, ent_i_start) = helper_a(scene, class);

    if state.ui.visibility.hide_ligand {
        return;
    }

    // Edit small molecules only; not proteins.
    if state.volatile.operating_mode == OperatingMode::MolEditor {
        return;
    }

    let mut entities = Vec::new();
    for (i_mol, mol) in state.ligands.iter_mut().enumerate() {
        let start_i_mol = ent_i_start + entities.len();

        let ents_this_mol = drawing::draw_mol(
            MolGenericRef::Ligand(mol),
            i_mol,
            &state.ui,
            &state.volatile.active_mol,
            state.volatile.mol_manip.mol,
            state.volatile.operating_mode,
        );

        // Note: This may already be set.
        let end_i_mol = start_i_mol + ents_this_mol.len();
        entities.extend(ents_this_mol);

        mol.common.entity_i_range = Some((start_i_mol, end_i_mol));
    }

    helper_b(
        state,
        scene,
        entities,
        class,
        initial_ent_count,
        ent_i_start,
        MolType::Ligand,
    );
}

// todo: You need to generalize your drawing code so you have less repetition, and it's more consistent.
pub fn draw_all_nucleic_acids(state: &mut State, scene: &mut Scene) {
    // draw_all_mol_of_type(state, scene, &mut state.nucleic_acids, MolType::nucleic_acid, state.ui.visibility.hide_nucleic_acids);
    // return;

    let class = EntityClass::NucleicAcid as u32;
    let (initial_ent_count, ent_i_start) = helper_a(scene, class);

    if state.ui.visibility.hide_nucleic_acids {
        return;
    }

    // Edit small molecules only; not proteins.
    if state.volatile.operating_mode == OperatingMode::MolEditor {
        return;
    }

    let mut entities = Vec::new();
    for (i_mol, mol) in state.nucleic_acids.iter_mut().enumerate() {
        let start_i_mol = ent_i_start + entities.len();

        let ents_this_mol = drawing::draw_mol(
            MolGenericRef::NucleicAcid(mol),
            i_mol,
            &state.ui,
            &state.volatile.active_mol,
            state.volatile.mol_manip.mol,
            state.volatile.operating_mode,
        );

        // Note: This may already be set.
        let end_i_mol = start_i_mol + ents_this_mol.len();
        entities.extend(ents_this_mol);

        mol.common.entity_i_range = Some((start_i_mol, end_i_mol));
    }

    helper_b(
        state,
        scene,
        entities,
        class,
        initial_ent_count,
        ent_i_start,
        MolType::NucleicAcid,
    );
}

// todo: You need to generalize your drawing code so you have less repetition, and it's more consistent.
pub fn draw_all_lipids(state: &mut State, scene: &mut Scene) {
    // draw_all_mol_of_type(state, scene, &mut state.lipids, MolType::Lipid, state.ui.visibility.hide_lipids);
    // return;

    let class = EntityClass::Lipid as u32;
    let (initial_ent_count, ent_i_start) = helper_a(scene, class);

    if state.ui.visibility.hide_lipids {
        return;
    }

    // Edit small molecules only; not proteins.
    if state.volatile.operating_mode == OperatingMode::MolEditor {
        return;
    }

    let mut entities = Vec::new();
    for (i_mol, mol) in state.lipids.iter_mut().enumerate() {
        let start_i_mol = ent_i_start + entities.len();

        let ents_this_mol = drawing::draw_mol(
            MolGenericRef::Lipid(mol),
            i_mol,
            &state.ui,
            &state.volatile.active_mol,
            state.volatile.mol_manip.mol,
            state.volatile.operating_mode,
        );

        // Note: This may already be set.
        let end_i_mol = start_i_mol + ents_this_mol.len();
        entities.extend(ents_this_mol);

        mol.common.entity_i_range = Some((start_i_mol, end_i_mol));
    }

    helper_b(
        state,
        scene,
        entities,
        class,
        initial_ent_count,
        ent_i_start,
        MolType::Lipid,
    );
}

pub fn draw_all_mol_of_type<T: MolGenericTrait>(
    state: &mut State,
    scene: &mut Scene,
    mols: &mut [T],
    mol_type: MolType,
    hide_flag: bool,
) {
    let initial_ent_count = scene.entities.len();
    scene
        .entities
        .retain(|ent| ent.class != mol_type.entity_type() as u32);

    if hide_flag {
        return;
    }

    let mut entities = Vec::new();
    for (i, mol) in mols.iter_mut().enumerate() {
        let start_i = entities.len();

        entities.extend(drawing::draw_mol(
            // todo: Sort this out
            mol.to_ref(),
            i,
            &state.ui,
            &state.volatile.active_mol,
            state.volatile.mol_manip.mol,
            state.volatile.operating_mode,
        ));

        let end_i = entities.len();
        mol.common_mut().entity_i_range = Some((start_i, end_i));
    }

    if entities.len() == initial_ent_count {
        // Repleace in-place
    } else {
        scene.entities.extend(entities);
        clear_mol_entity_indices(state, None);
    }
}

/// Updates a single molecule's entities.
fn update_inplace_inner(
    mol: MolGenericRef,
    i: usize,
    ent_i_start: usize,
    ent_i_end: usize,
    state: &State,
    scene: &mut Scene,
) {
    let ents_updated = drawing::draw_mol(
        mol,
        i,
        &state.ui,
        &state.volatile.active_mol,
        state.volatile.mol_manip.mol,
        state.volatile.operating_mode,
    );

    if ents_updated.len() != ent_i_end - ent_i_start {
        eprintln!(
            "Error: Mismatch between new and old mol et counts. Old: {}, new: {}",
            ents_updated.len(),
            ent_i_end - ent_i_start
        );
        return;
    }

    for (i, ent) in scene.entities[ent_i_start..ent_i_end]
        .iter_mut()
        .enumerate()
    {
        ent.position = ents_updated[i].position;
        ent.orientation = ents_updated[i].orientation;
        ent.scale = ents_updated[i].scale;
        ent.scale_partial = ents_updated[i].scale_partial;
        ent.color = ents_updated[i].color;
    }
}

/// Cheaper, but takes some care in synchronization. Only draws
/// This should be run in conjunction with `engine_updates.entities.push_class(EntityClass::Ligand as u32)` or similar.
/// We update entities, then command an instance buffer update only of these updates
/// todo: You can go even further and do it one lig and a time, instead of all ligs.
pub fn update_all_ligs_inplace(state: &State, scene: &mut Scene) {
    for (i, lig) in state.ligands.iter().enumerate() {
        let Some((ent_i_start, ent_i_end)) = lig.common.entity_i_range else {
            eprintln!("Unable to update mol entities in place; missing entity indices");
            continue;
        };

        let mol = MolGenericRef::Ligand(lig);
        update_inplace_inner(mol, i, ent_i_start, ent_i_end, state, scene);
    }
}

pub fn update_all_na_inplace(state: &State, scene: &mut Scene) {
    for (i, na) in state.nucleic_acids.iter().enumerate() {
        let Some((ent_i_start, ent_i_end)) = na.common.entity_i_range else {
            eprintln!("Unable to update mol entities in place; missing entity indices");
            continue;
        };

        let mol = MolGenericRef::NucleicAcid(na);
        update_inplace_inner(mol, i, ent_i_start, ent_i_end, state, scene);
    }
}

pub fn update_all_lipids_inplace(state: &State, scene: &mut Scene) {
    for (i, lipid) in state.lipids.iter().enumerate() {
        let Some((ent_i_start, ent_i_end)) = lipid.common.entity_i_range else {
            eprintln!("Unable to update mol entities in place; missing entity indices");
            continue;
        };

        let mol = MolGenericRef::Lipid(lipid);
        update_inplace_inner(mol, i, ent_i_start, ent_i_end, state, scene);
    }
}

pub fn update_single_ligand_inplace(i: usize, state: &State, scene: &mut Scene) {
    let ligand = &state.ligands[i];
    let Some((ent_i_start, ent_i_end)) = ligand.common.entity_i_range else {
        eprintln!("Unable to update mol entities in place; missing entity indices");
        return;
    };

    let mol = MolGenericRef::Ligand(ligand);
    update_inplace_inner(mol, i, ent_i_start, ent_i_end, state, scene);
}

pub fn update_single_nucleic_acid_inplace(i: usize, state: &State, scene: &mut Scene) {
    let na = &state.nucleic_acids[i];
    let Some((ent_i_start, ent_i_end)) = na.common.entity_i_range else {
        eprintln!("Unable to update mol entities in place; missing entity indices");
        return;
    };

    let mol = MolGenericRef::NucleicAcid(na);
    update_inplace_inner(mol, i, ent_i_start, ent_i_end, state, scene);
}

pub fn update_single_lipid_inplace(i: usize, state: &State, scene: &mut Scene) {
    let lipid = &state.lipids[i];
    let Some((ent_i_start, ent_i_end)) = lipid.common.entity_i_range else {
        eprintln!("Unable to update mol entities in place; missing entity indices");
        return;
    };

    let mol = MolGenericRef::Lipid(lipid);
    update_inplace_inner(mol, i, ent_i_start, ent_i_end, state, scene);
}
