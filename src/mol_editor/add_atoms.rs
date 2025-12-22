use std::sync::atomic::Ordering;

use bio_files::BondType;
use dynamics::{find_tetra_posit_final, find_tetra_posits};
use egui::Ui;
use graphics::{ControlScheme, EngineUpdates, Entity, EntityUpdate};
use lin_alg::f64::{Quaternion, Vec3};
use na_seq::{
    Element,
    Element::{Carbon, Hydrogen, Nitrogen, Oxygen},
};

use crate::{
    StateUi, mol_editor,
    mol_editor::{MolEditorState, NEXT_ATOM_SN, hydrogens_avail, redraw, templates::Template},
    mol_lig::MoleculeSmall,
    mol_manip::ManipMode,
    molecule::{Atom, Bond, MoleculeCommon},
};

fn find_appended_posit(
    posit_parent: Vec3,
    atoms: &[Atom],
    adj_to_par: &[usize],
    bond_len: Option<f64>,
    element: Element,
) -> Option<Vec3> {
    let neighbor_count = adj_to_par.len();

    // Note on these computations: The parent atom is the "hub" of a tetrahedral or planar
    // hub-and-spoke config. Other spokes are existing atoms bound to this parent, and the atom
    // we're computing the position here to add.
    let result = match neighbor_count {
        // This 0 branch should only be called for disconnected parents.
        0 => Some(posit_parent + Vec3::new(1.3, 0., 0.)),

        1 => {
            // This neighbor is the *grandparent* to the one we're adding; what ties
            // the parent to it. We set up the atom we're adding so it's tau/3 to this
            // grandparent, rotated around the parent.
            let grandparent = atoms[adj_to_par[0]].posit;

            // For now, pick an arbitrary orientation of the 3 methyl atoms (relative to the rest of the system)
            // without regard for steric clashes, and let MD sort it out after.
            // todo: choose something explicitly that avoids steric clashes?

            const TETRA_ANGLE: f64 = 1.91063;

            let bond_par_gp = (grandparent - posit_parent).to_normalized();
            let ax_rot = bond_par_gp.any_perpendicular();
            let rotator = Quaternion::from_axis_angle(ax_rot, TETRA_ANGLE);

            // If H, shorten the bond.
            let mut relative_dir = rotator.rotate_vec(bond_par_gp);
            if element == Hydrogen {
                relative_dir = (relative_dir.to_normalized()) * 1.1;
            }
            Some(posit_parent + relative_dir)
        }
        2 => {
            let neighbor_0 = atoms[adj_to_par[0]].posit;
            let neighbor_1 = atoms[adj_to_par[1]].posit;

            // This function uses the distance between the first two params, so it's likely
            // in the case of adding H, this is what we want. (?)
            let (p0, p1) = find_tetra_posits(posit_parent, neighbor_1, neighbor_0);

            // Score a candidate by its minimum distance to any existing neighbor; pick the larger score.
            let neighbors: &[usize] = &adj_to_par;
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
            let adj_0 = adj_to_par[0];
            let neighbor_0 = atoms[adj_0].posit;
            let adj_1 = adj_to_par[1];
            let neighbor_1 = atoms[adj_1].posit;
            let adj_2 = adj_to_par[2];
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
/// currently using it for rings, functional groups, etc.
pub fn add_from_template(
    mol: &mut MoleculeCommon,
    template: Template,
    anchor_is: &[usize],
    // anchors: &[Vec3], // 0 or 1.
    r_aligner_i: usize,
    r_aligner: Vec3,
    start_sn: u32,
    start_i: usize,
    redraw: &mut bool,
    rebuild_md: &mut bool,
    state_ui: &mut StateUi,
    controls: &mut ControlScheme,
    manip_mode: ManipMode,
) {
    let anchors = anchor_is
        .iter()
        .map(|i| mol.atoms[*i].posit)
        .collect::<Vec<_>>();
    let (atoms, bonds) = template.atoms_bonds(&anchors, r_aligner, start_sn, start_i);
    NEXT_ATOM_SN.fetch_add(atoms.len() as u32, Ordering::AcqRel);

    let mut i_added = Vec::new(); // Used for populating H.

    for atom in &atoms {
        mol.atoms.push(atom.clone());
        i_added.push(mol.atoms.len() - 1);
    }
    for bond in bonds {
        mol.bonds.push(bond);
    }

    if !matches!(template, Template::AromaticRing | Template::PentaRing) {
        // Add back the bond between this atom and the aligner atom.
        mol.bonds.push(Bond {
            bond_type: BondType::Single,
            atom_0_sn: mol.atoms[r_aligner_i].serial_number,
            atom_1_sn: mol.atoms[start_i].serial_number,
            atom_0: r_aligner_i,
            atom_1: start_i,
            is_backbone: false,
        });
    } else {
        if anchor_is.len() == 2 {
            mol.bonds.push(Bond {
                bond_type: BondType::Single,
                atom_0_sn: mol.atoms[anchor_is[0]].serial_number,
                atom_1_sn: mol.atoms[start_i + 1].serial_number,
                atom_0: anchor_is[0],
                atom_1: start_i + 1,
                is_backbone: false,
            });

            mol.bonds.push(Bond {
                bond_type: BondType::Single,
                atom_0_sn: mol.atoms[anchor_is[1]].serial_number,
                atom_1_sn: mol.atoms[start_i + 1].serial_number,
                atom_0: anchor_is[1],
                atom_1: start_i + 1,
                is_backbone: false,
            });
        } else {
            mol.bonds.push(Bond {
                bond_type: BondType::Single,
                atom_0_sn: mol.atoms[anchor_is[0]].serial_number,
                atom_1_sn: mol.atoms[start_i].serial_number,
                atom_0: anchor_is[0],
                atom_1: start_i,
                is_backbone: false,
            });
        }

        // // Remove the anchor atoms, and update the bonds to connect them to the rings.
        // for &anchor_i in anchor_is {
        //     mol.remove_atom(anchor_i);
        // }
    }

    mol.reset_posits();
    mol.build_adjacency_list();

    for (i, atom) in atoms.into_iter().enumerate() {
        populate_hydrogens_on_atom(
            mol,
            i_added[i] - 1,
            // atom.element,
            &atom.force_field_type,
            &mut Vec::new(),
            state_ui,
            &mut Default::default(),
            manip_mode,
        );
    }

    // if matches!(template, Template::AromaticRing | Template::PentaRing) {
    //     // Remove the anchor atoms, and update the bonds to connect them to the rings.
    //     for &anchor_i in anchor_is {
    //         mol.remove_atom(anchor_i);
    //     }
    // }

    *controls = ControlScheme::Arc {
        center: mol.centroid().into(),
    };

    // We are currently replacing the selected atom with the added group's anchor.
    // So, remove it and its H atoms.

    for &anchor_i in anchor_is {
        // todo: Fix this; both are causing crashes.
        remove_hydrogens(mol, anchor_i); // Do this prior to removing the atom.
        mol.remove_atom(anchor_i);
    }

    *redraw = true;
    *rebuild_md = true;
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
    control: &mut ControlScheme,
    manip_mode: ManipMode,
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

        redraw(entities, &mol_wrapper, ui, manip_mode);
        updates.entities = EntityUpdate::All;
    }

    // todo: Can't use `common` below here due to the delete_atom code and ownership.
    let posit = match find_appended_posit(
        posit_parent,
        &mol.atoms,
        &mol.adjacency_list[i_par],
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

    // Add hydrogens back to the parent.
    populate_hydrogens_on_atom(mol, i_par, &ff_type, entities, ui, updates, manip_mode);

    // Add hydrogens to the new atom.
    // Up to one recursion to add hydrogens to this parent and to the new atom.
    if element != Hydrogen {
        populate_hydrogens_on_atom(mol, i_new, &ff_type, entities, ui, updates, manip_mode);
        *control = ControlScheme::Arc {
            center: mol.centroid().into(),
        };
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
    // el: Element,
    ff_type: &Option<String>,
    entities: &mut Vec<Entity>,
    state_ui: &mut StateUi,
    engine_updates: &mut EngineUpdates,
    manip_mode: ManipMode,
) {
    // todo. Don't clone!!! Find a better way to fix the borrow error.
    let el = mol.atoms[i].element;

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
                &mut ControlScheme::None,
                manip_mode,
            );
            j += 1;
        }
    }
}
