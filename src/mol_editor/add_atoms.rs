use std::sync::atomic::Ordering;

use bio_files::BondType;
use dynamics::{find_tetra_posit_final, find_tetra_posits};
use graphics::{EngineUpdates, Entity, EntityUpdate};
use lin_alg::f64::{Quaternion, Vec3};
use na_seq::{
    Element,
    Element::{Carbon, Hydrogen, Nitrogen, Oxygen},
};

use crate::mol_editor::redraw;
use crate::{
    StateUi, mol_editor,
    mol_editor::{MolEditorState, NEXT_ATOM_SN, hydrogens_avail},
    molecule::{Atom, Bond},
};

impl MolEditorState {
    /// Returns the index of the atom added.
    pub fn add_atom(
        &mut self,
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
        let common = &mut self.mol.common;

        let posit_parent = common.atom_posits[i_par];
        let el_parent = common.atoms[i_par].element;

        if el_parent == Hydrogen {
            return None; // Not supported in our current iteration.
        }

        // Delete hydrogens; we'll add back if required.
        if element != Hydrogen {
            let mut h_to_del = Vec::new();
            // Remove Hydrogens; we'll add any back as applicable.
            for j in &common.adjacency_list[i_par] {
                if common.atoms[*j].element == Hydrogen {
                    h_to_del.push(*j);
                }
            }

            for j in h_to_del {
                if self.delete_atom(j).is_err() {
                    eprintln!("Problem deleting atom {j}");
                }
            }
            redraw(entities, &mut self.mol, ui);
            updates.entities = EntityUpdate::All;
        }

        // todo: Can't use `common` below here due to the delete_atom code and ownership.

        let neighbor_count = self.mol.common.adjacency_list[i_par].len();
        let adj_list = &self.mol.common.adjacency_list;

        let posit = match find_appended_posit(
            i_par,
            posit_parent,
            neighbor_count,
            &self.mol.common.atoms,
            adj_list,
            bond_len,
            element,
        ) {
            Some(p) => p,
            // Can't add an atom; already too many atoms bonded.
            None => return None,
        };

        let new_sn = NEXT_ATOM_SN.fetch_add(1, Ordering::AcqRel);
        let new_i = self.mol.common.atoms.len();

        if i_par >= self.mol.common.atoms.len() {
            eprintln!("Index out of range when adding atoms: {i_par}");
            return None;
            // todo: This return and print are a workaround; find the root cause.
        }

        let atom_new = Atom {
            serial_number: new_sn,
            posit,
            element,
            type_in_res: None,
            force_field_type: ff_type,
            partial_charge: Some(q),
            ..Default::default()
        };
        let atom_for_h = atom_new.clone(); // Not a fan of having to clone here.

        self.mol.common.atoms.push(atom_new);

        self.mol.common.bonds.push(Bond {
            bond_type,
            atom_0_sn: self.mol.common.atoms[i_par].serial_number,
            atom_1_sn: new_sn,
            atom_0: i_par,
            atom_1: new_i,
            is_backbone: false,
        });

        self.mol.common.atom_posits.push(posit);

        self.mol.common.adjacency_list[i_par].push(new_i);
        self.mol.common.adjacency_list.push(vec![i_par]);

        let i_new = self.mol.common.atoms.len() - 1;
        let i_new_bond = self.mol.common.bonds.len() - 1;

        mol_editor::draw_atom(entities, &self.mol.common.atoms[i_new], ui);
        mol_editor::draw_bond(
            entities,
            &self.mol.common.bonds[i_new_bond],
            &self.mol.common.atoms,
            &self.mol.common.adjacency_list,
            ui,
        );

        // Up to one recursion to add hydrogens to this parent and to the new atom.
        if element != Hydrogen {
            self.populate_hydrogens_on_atom(i_new, &atom_for_h, entities, ui, updates);
        }

        // todo: Ideally just add the single entity, and add it to the
        // index buffer.

        // todo: this redraw etc is not working.
        updates.entities = EntityUpdate::All;

        Some(new_i)
    }

    /// Populate hydrogens on a single atom. Uses tetrahedral, or planar geometry as required
    /// based on atoms in the vicinity.
    pub(super) fn populate_hydrogens_on_atom(
        &mut self,
        i: usize,
        atom: &Atom,
        entities: &mut Vec<Entity>,
        state_ui: &mut StateUi,
        engine_updates: &mut EngineUpdates,
    ) {
        // todo. Don't clone!!! Find a better way to fix the borrow error.

        let mut skip = false;
        for bonded_i in &self.mol.common.adjacency_list[i] {
            // Don't add H to oxygens double-bonded.
            if self.mol.common.atoms[i].element == Oxygen {
                for bond in &self.mol.common.bonds {
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
            let adj = &self.mol.common.adjacency_list[i];
            let bonds_avail: usize = match atom.element {
                Carbon => 4,
                Oxygen => 2,
                Nitrogen => 3, // todo?
                _ => 4,
            };

            let bonds_remaining = bonds_avail.saturating_sub(adj.len());

            let mut j = 0;
            for (ff_type, bond_len) in hydrogens_avail(&atom.force_field_type) {
                if j >= bonds_remaining {
                    break;
                }
                if self.mol.common.atoms[i].serial_number == 3 {
                    println!("Attempting to add 1 of {bonds_remaining} atoms...");
                }
                // todo: Rough
                let q = match &self.mol.common.atoms[i].element {
                    Oxygen => 0.47,
                    _ => 0.03,
                };

                self.add_atom(
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
}

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
