use std::sync::atomic::Ordering;

use bio_files::BondType;
use dynamics::{find_planar_posit, find_tetra_posit_final, find_tetra_posits};
use graphics::{EngineUpdates, Entity, EntityUpdate};
use lin_alg::f64::Vec3;
use na_seq::{Element, Element::Hydrogen};

use crate::{
    StateUi, mol_editor,
    mol_editor::{MolEditorState, NEXT_ATOM_SN},
    molecule::{Atom, Bond},
};

pub fn add_atom(
    editor: &mut MolEditorState,
    entities: &mut Vec<Entity>,
    i_par: usize, // Of the parent atom
    element: Element,
    bond_type: BondType,
    ff_type: Option<String>,
    bond_len: Option<f64>,
    q: f32,
    ui: &mut StateUi,
    updates: &mut EngineUpdates,
) {
    // todo: For readability, we really need somethign like this, but getter borrow errors:
    let common = &mut editor.mol.common;

    let posit_parent = common.atom_posits[i_par];
    let el_parent = common.atoms[i_par].element;

    if element != Hydrogen {
        let mut h_to_del = Vec::new();
        // Remove Hydrogens; we'll add any back as applicable.
        for j in &common.adjacency_list[i_par] {
            if common.atoms[*j].element == Hydrogen {
                h_to_del.push(*j);
            }
        }

        for j in h_to_del {
            if editor.delete_atom(j).is_err() {
                eprintln!("Problem deleting atom {j}");
            }
        }
    }

    // todo: Can't use `common` below here due to the delete_atom code and ownership.

    let neighbor_count = editor.mol.common.adjacency_list[i_par].len();
    let adj_list = &editor.mol.common.adjacency_list;

    let posit = match find_appended_posit(
        i_par,
        posit_parent,
        neighbor_count,
        &editor.mol.common.atoms,
        adj_list,
        bond_len,
    ) {
        Some(p) => p,
        // Can't add an atom; already too many atoms bonded.
        None => return,
    };

    let new_sn = NEXT_ATOM_SN.fetch_add(1, Ordering::AcqRel);
    let new_i = editor.mol.common.atoms.len();

    if i_par >= editor.mol.common.atoms.len() {
        eprintln!("Index out of range when adding atoms: {i_par}");
        return;
        // todo: This return and print are a workaround; find the root cause.
    }

    editor.mol.common.atoms.push(Atom {
        serial_number: new_sn,
        posit,
        element,
        type_in_res: None,
        force_field_type: ff_type,
        partial_charge: Some(q),
        ..Default::default()
    });

    editor.mol.common.bonds.push(Bond {
        bond_type,
        atom_0_sn: editor.mol.common.atoms[i_par].serial_number,
        atom_1_sn: new_sn,
        atom_0: i_par,
        atom_1: new_i,
        is_backbone: false,
    });

    editor.mol.common.atom_posits.push(posit);

    editor.mol.common.adjacency_list[i_par].push(new_i);
    editor.mol.common.adjacency_list.push(vec![i_par]);

    mol_editor::draw_atom(
        entities,
        &editor.mol.common.atoms[editor.mol.common.atoms.len() - 1],
        ui,
    );
    mol_editor::draw_bond(
        entities,
        &editor.mol.common.bonds[editor.mol.common.bonds.len() - 1],
        &editor.mol.common.atoms,
        &editor.mol.common.adjacency_list,
        ui,
    );

    // Up to one recursion to add hydrogens to this parent and to the new atom.
    if element != Hydrogen {
        // Back-fill hydrogens on the parent (it just lost one).
        for (ff_h, bl_h) in
            mol_editor::hydrogens_avail(&editor.mol.common.atoms[i_par].force_field_type)
        {
            // todo: Rough
            let q = match &editor.mol.common.atoms[i_par].element {
                Element::Oxygen => 0.47,
                _ => 0.03,
            };

            add_atom(
                editor,
                entities,
                i_par,
                Hydrogen,
                BondType::Single,
                Some(ff_h.clone()),
                Some(bl_h),
                q,
                ui,
                updates,
            );
        }

        // Populate hydrogens on the newly added heavy atom (eg, make CH3 if appropriate).
        for (ff_h, bl_h) in
            mol_editor::hydrogens_avail(&editor.mol.common.atoms[new_i].force_field_type)
        {
            // todo: Rough
            let q = match &editor.mol.common.atoms[new_i].element {
                Element::Oxygen => 0.47,
                _ => 0.03, // higher in some cases like ring; 0.12?
            };

            add_atom(
                editor,
                entities,
                new_i,
                Hydrogen,
                BondType::Single,
                Some(ff_h.clone()),
                Some(bl_h),
                q,
                ui,
                updates,
            );
        }
    }

    // todo: Ideally just add the single entity, and add it to the
    // index buffer.
    updates.entities = EntityUpdate::All;
}

/// `i` is the parent's index.
fn find_appended_posit(
    i: usize,
    posit_parent: Vec3,
    neighbor_count: usize,
    atoms: &[Atom],
    adj_list: &[Vec<usize>],
    bond_len: Option<f64>,
) -> Option<Vec3> {
    let result = match neighbor_count {
        // todo
        0 => Some(Vec3::new(1.3, 0., 0.)),
        1 => {
            let adj = adj_list[i][0];
            let neighbor = atoms[adj].posit;

            // todo: This probably isn't what you want.
            Some(find_tetra_posits(posit_parent, neighbor, Vec3::new_zero()).0)
        }
        2 => {
            let adj_0 = adj_list[i][0];
            let neighbor_0 = atoms[adj_0].posit;
            let adj_1 = adj_list[i][1];
            let neighbor_1 = atoms[adj_1].posit;

            let (p0, p1) = find_tetra_posits(posit_parent, neighbor_0, neighbor_1);

            // Score a candidate by its minimum distance to any existing neighbor; pick the larger score.
            let neighbors: &[usize] = &adj_list[i];
            let score = |p: Vec3| {
                let mut best = f64::INFINITY;
                for &ni in neighbors {
                    let q = atoms[ni].posit;
                    let d2 = (p - q).magnitude_squared();
                    if d2 < best {
                        best = d2;
                    }
                }
                best
            };
            Some(if score(p0) >= score(p1) { p0 } else { p1 })
        }
        3 => {
            let adj_0 = adj_list[i][0];
            let neighbor_0 = atoms[adj_0].posit;
            let adj_1 = adj_list[i][1];
            let neighbor_1 = atoms[adj_1].posit;
            let adj_2 = adj_list[i][2];
            let neighbor_2 = atoms[adj_2].posit;

            // todo. Check both angles?
            // If the incoming angles are ~τ/3, add in a planar config.
            let bond_0 = neighbor_0 - posit_parent;
            let bond_1 = neighbor_1 - posit_parent;
            let angle = bond_1.to_normalized().dot(bond_0.to_normalized()).acos();

            // Planar; full.
            if angle > 1.95 {
                return None;
            } else {
                Some(find_tetra_posit_final(
                    posit_parent,
                    neighbor_0,
                    neighbor_1,
                    neighbor_2,
                ))
            }
        }
        _ => None,
    };

    // Set len, if applicable.
    // todo: Could be slightly more efficient to bake this length correction into the find_tetra
    // todo etc fns.
    match result {
        Some(p) => match bond_len {
            Some(l) => {
                let rel_pos = (p - posit_parent).to_normalized() * l;
                Some(posit_parent + rel_pos)
            }
            None => Some(p),
        },
        None => None,
    }
}
