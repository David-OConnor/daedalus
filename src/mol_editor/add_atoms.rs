use std::sync::atomic::Ordering;

use bio_files::BondType;
use dynamics::{find_tetra_posit_final, find_tetra_posits};
use egui::Ui;
use graphics::{EngineUpdates, Entity, EntityUpdate};
use lin_alg::f64::{Quaternion, Vec3};
use na_seq::{
    Element,
    Element::{Carbon, Hydrogen, Nitrogen, Oxygen},
};

use crate::{
    StateUi, mol_editor,
    mol_editor::{MolEditorState, NEXT_ATOM_SN, hydrogens_avail, redraw, templates::Template},
    mol_lig::MoleculeSmall,
    molecule::{Atom, Bond, MoleculeCommon},
};

/// `i` is the parent's index.
fn find_appended_posit(
    i: usize,
    posit_parent: Vec3,
    neighbor_count: usize,
    atoms: &[Atom],
    adj_list: &[Vec<usize>],
    bond_len: Option<f64>,
    element: Element,
) -> Option<Vec3> {
    let result = match neighbor_count {
        // This 0 branch should rarely be called; for disconnected parents.
        0 => Some(posit_parent + Vec3::new(1.3, 0., 0.)),
        1 => {
            let adj = adj_list[i][0];
            let neighbor = atoms[adj].posit;

            // For now, pick an arbitrary orientation of the 3 methyl atoms, and let MD sort it out later.
            // todo: choose something that avoids steric clashes.

            // todo: This section is not working properly.
            const TETRA_ANGLE: f64 = 1.91063;
            let bond = (neighbor - posit_parent).to_normalized();
            let axis = bond.any_perpendicular();
            let rotator = Quaternion::from_axis_angle(axis, TETRA_ANGLE);

            // If H, shorten the bond.
            let mut relative_dir = rotator.rotate_vec(bond);
            if element == Hydrogen {
                relative_dir = (relative_dir.to_normalized()) * 1.1;
            }
            Some(posit_parent + relative_dir)
        }
        2 => {
            let adj_0 = adj_list[i][0];
            let neighbor_0 = atoms[adj_0].posit;
            let adj_1 = adj_list[i][1];
            let neighbor_1 = atoms[adj_1].posit;

            // This function uses the distance between the first two params, so it's likely
            // in the case of adding H, this is what we want. (?)
            let (p0, p1) = find_tetra_posits(posit_parent, neighbor_1, neighbor_0);

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

            // None
            Some(if score(p0) >= score(p1) { p0 } else { p1 })
        }
        3 => {
            // None
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

/// A button that adds atoms to the editor molecule from a template. Things can be single atoms, but we are
/// currently using it for rings, functional groups etc.
pub fn add_from_template_btn(
    mol: &mut MoleculeCommon,
    template: Template,
    anchor_i: usize,
    anchor: Vec3,
    r_aligner_i: usize,
    r_aligner: Vec3,
    start_sn: u32,
    start_i: usize,
    ui: &mut Ui,
    redraw: &mut bool,
    rebuild_md: &mut bool,
    abbrev: &str,
    name: &str,
    state_ui: &mut StateUi,
) {
    if ui
        .button(abbrev)
        .on_hover_text(format!("Add a {name} at the current selection"))
        .clicked()
    {
        let (atoms, bonds) = template.atoms_bonds(anchor, r_aligner, start_sn, start_i);
        NEXT_ATOM_SN.fetch_add(atoms.len() as u32, Ordering::AcqRel);

        let mut i_added = Vec::new(); // Used for populating H.

        for atom in &atoms {
            mol.atoms.push(atom.clone());
            i_added.push(mol.atoms.len() - 1);
        }
        for bond in bonds {
            mol.bonds.push(bond);
        }

        // Add back the bond between this atom and the aligner atom.
        mol.bonds.push(Bond {
            bond_type: BondType::Single,
            atom_0_sn: mol.atoms[r_aligner_i].serial_number,
            atom_1_sn: mol.atoms[start_i].serial_number,
            atom_0: r_aligner_i,
            atom_1: start_i,
            is_backbone: false,
        });

        mol.reset_posits();
        mol.build_adjacency_list();

        for (i, atom) in atoms.into_iter().enumerate() {
            populate_hydrogens_on_atom(
                mol,
                i_added[i] - 1,
                atom.element,
                &atom.force_field_type,
                &mut Vec::new(),
                state_ui,
                &mut Default::default(),
            );
        }

        // We are currently replacing the selected atom with the added group's anchor.
        // So, remove it and its H atoms.
        // todo: Fix this; both are causing crashes.
        remove_hydrogens(mol, anchor_i); // Do this prior to removing the atom.
        mol.remove_atom(anchor_i);

        *redraw = true;
        *rebuild_md = true;
    }
}

impl MolEditorState {
    /// Wrapper to ensure we remove hydrogens.
    pub fn remove_atom(&mut self, i: usize) {
        remove_hydrogens(&mut self.mol.common, i);
        self.mol.common.remove_atom(i);
    }
}

/// Remove all hydrogens bonded to an atom.
pub fn remove_hydrogens(mol: &mut MoleculeCommon, i: usize) {
    let mut h_to_del = Vec::new();

    // Remove Hydrogens; we'll add any back as applicable.
    for j in &mol.adjacency_list[i] {
        if mol.atoms[*j].element == Hydrogen {
            h_to_del.push(*j);
        }
    }

    h_to_del.sort_unstable_by(|a, b| b.cmp(a));
    println!("H to del: {:?}", h_to_del);
    for j in h_to_del {
        mol.remove_atom(j);
    }
}

/// Returns the index of the atom added.
pub fn add_atom(
    mol: &mut MoleculeCommon,
    entities: &mut Vec<Entity>,
    i_par: usize, // Of the parent atom
    element: Element,
    bond_type: BondType,
    ff_type: Option<String>,
    bond_len: Option<f64>,
    q: f32,
    ui: &mut StateUi,
    updates: &mut EngineUpdates,
) -> Option<usize> {
    // todo: For readability, we really need somethign like this, but getter borrow errors:
    let posit_parent = mol.atom_posits[i_par];
    let el_parent = mol.atoms[i_par].element;

    if el_parent == Hydrogen {
        return None; // Not supported in our current iteration.
    }

    // Delete hydrogens; we'll add back if required.
    if element != Hydrogen {
        remove_hydrogens(mol, i_par);

        let mol_wrapper = MoleculeSmall {
            common: mol.clone(),
            ..Default::default()
        };

        redraw(entities, &mol_wrapper, ui);
        updates.entities = EntityUpdate::All;
    }

    // todo: Can't use `common` below here due to the delete_atom code and ownership.

    let neighbor_count = mol.adjacency_list[i_par].len();
    let adj_list = &mol.adjacency_list;

    let posit = match find_appended_posit(
        i_par,
        posit_parent,
        neighbor_count,
        &mol.atoms,
        adj_list,
        bond_len,
        element,
    ) {
        Some(p) => p,
        // Can't add an atom; already too many atoms bonded.
        None => return None,
    };

    let new_sn = NEXT_ATOM_SN.fetch_add(1, Ordering::AcqRel);
    let new_i = mol.atoms.len();

    if i_par >= mol.atoms.len() {
        eprintln!("Index out of range when adding atoms: {i_par}");
        return None;
        // todo: This return and print are a workaround; find the root cause.
    }

    let atom_new = Atom {
        serial_number: new_sn,
        posit,
        element: element.clone(),
        type_in_res: None,
        force_field_type: ff_type.clone(),
        partial_charge: Some(q),
        ..Default::default()
    };

    mol.atoms.push(atom_new);

    mol.bonds.push(Bond {
        bond_type,
        atom_0_sn: mol.atoms[i_par].serial_number,
        atom_1_sn: new_sn,
        atom_0: i_par,
        atom_1: new_i,
        is_backbone: false,
    });

    mol.atom_posits.push(posit);

    mol.adjacency_list[i_par].push(new_i);
    mol.adjacency_list.push(vec![i_par]);

    let i_new = mol.atoms.len() - 1;
    let i_new_bond = mol.bonds.len() - 1;

    mol_editor::draw_atom(entities, &mol.atoms[i_new], ui);
    mol_editor::draw_bond(
        entities,
        &mol.bonds[i_new_bond],
        &mol.atoms,
        &mol.adjacency_list,
        ui,
    );

    // Up to one recursion to add hydrogens to this parent and to the new atom.
    if element != Hydrogen {
        populate_hydrogens_on_atom(mol, i_new, element, &ff_type, entities, ui, updates);
    }

    // todo: Ideally just add the single entity, and add it to the
    // index buffer.
    updates.entities = EntityUpdate::All;

    Some(new_i)
}

/// Populate hydrogens on a single atom. Uses tetrahedral, or planar geometry as required
/// based on atoms in the vicinity.
pub fn populate_hydrogens_on_atom(
    mol: &mut MoleculeCommon,
    i: usize,
    el: Element,
    ff_type: &Option<String>,
    entities: &mut Vec<Entity>,
    state_ui: &mut StateUi,
    engine_updates: &mut EngineUpdates,
) {
    // todo. Don't clone!!! Find a better way to fix the borrow error.

    let mut skip = false;

    for bonded_i in &mol.adjacency_list[i] {
        // Don't add H to oxygens double-bonded.
        if mol.atoms[i].element == Oxygen {
            for bond in &mol.bonds {
                if (bond.atom_0 == i && bond.atom_1 == *bonded_i
                    || bond.atom_1 == i && bond.atom_0 == *bonded_i)
                    && matches!(bond.bond_type, BondType::Double)
                {
                    skip = true;
                    break;
                }
            }
        }
    }

    if !skip {
        let adj = &mol.adjacency_list[i];
        let bonds_avail: usize = match el {
            Carbon => 4,
            Oxygen => 2,
            Nitrogen => 3, // todo?
            _ => 4,
        };

        let bonds_remaining = bonds_avail.saturating_sub(adj.len());

        let mut j = 0;
        for (ff_type, bond_len) in hydrogens_avail(ff_type) {
            if j >= bonds_remaining {
                break;
            }
            if mol.atoms[i].serial_number == 3 {
                println!("Attempting to add 1 of {bonds_remaining} atoms...");
            }
            // todo: Rough
            let q = match &mol.atoms[i].element {
                Oxygen => 0.47,
                _ => 0.03,
            };

            add_atom(
                mol,
                entities,
                i,
                Hydrogen,
                BondType::Single,
                Some(ff_type),
                Some(bond_len),
                q,
                state_ui,
                engine_updates,
            );
            j += 1;
        }
    }
}
