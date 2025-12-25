use std::sync::atomic::Ordering;

use bio_files::{BondType, orca::OrcaOutput::Geometry};
use dynamics::{find_planar_posit, find_tetra_posit_final, find_tetra_posits};
use egui::Ui;
use graphics::{ControlScheme, EngineUpdates, Entity, EntityUpdate};
use lin_alg::f64::{Quaternion, Vec3};
use na_seq::{
    Element,
    Element::{Carbon, Hydrogen, Nitrogen, Oxygen},
};

use crate::{
    StateUi, mol_editor,
    mol_editor::{MolEditorState, NEXT_ATOM_SN, redraw, templates::Template},
    mol_lig::MoleculeSmall,
    mol_manip::ManipMode,
    molecule::{Atom, Bond, MoleculeCommon},
};

#[derive(Clone, Copy, PartialEq)]
enum BondGeom {
    Linear,
    Planar,
    Tetrahedral,
}

fn find_appended_posit(
    posit_parent: Vec3,
    atoms: &[Atom],
    adj_to_par: &[usize],
    bond_len: Option<f64>,
    element: Element,
    geom: BondGeom,
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

            match geom {
                BondGeom::Tetrahedral => {
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

                    Some(if score(p0) >= score(p1) { p0 } else { p1 })
                }
                BondGeom::Planar => Some(find_planar_posit(posit_parent, neighbor_0, neighbor_1)),
                BondGeom::Linear => {
                    return None;
                }
            }
        }
        3 => {
            if geom != BondGeom::Tetrahedral {
                return None;
            }

            // None
            let adj_0 = adj_to_par[0];
            let neighbor_0 = atoms[adj_0].posit;
            let adj_1 = adj_to_par[1];
            let neighbor_1 = atoms[adj_1].posit;
            let adj_2 = adj_to_par[2];
            let neighbor_2 = atoms[adj_2].posit;

            // todo. Check both angles?
            // If the incoming angles are ~τ/3, add in a planar config.
            // let bond_0 = neighbor_0 - posit_parent;
            // let bond_1 = neighbor_1 - posit_parent;
            // let angle = bond_1.to_normalized().dot(bond_0.to_normalized()).acos();

            // Planar; full.
            // if angle > 1.95 {
            // todo: Experiment. You may wish to use the character of neighboring bond count
            // todo and type instead of this angle.
            // if angle > 2.11 {
            //     println!("Planar abort!: {angle}"); // todo temp!!
            //     return None;
            // } else {
            Some(find_tetra_posit_final(
                posit_parent,
                neighbor_0,
                neighbor_1,
                neighbor_2,
            ))
            // }
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
    anchor_sns: &[u32],
    anchor_is: &[usize],
    r_aligner_is: &[usize],
    r_aligners: &[Vec3],
    start_sn: u32,
    start_i: usize,
    redraw: &mut bool,
    rebuild_md: &mut bool,
    state_ui: &mut StateUi,
    controls: &mut ControlScheme,
    manip_mode: ManipMode,
) {
    let anchor_posits = anchor_is
        .iter()
        .map(|i| mol.atoms[*i].posit)
        .collect::<Vec<_>>();

    let (atoms, bonds) = template.atoms_bonds(
        anchor_is,
        anchor_sns,
        &anchor_posits,
        r_aligners,
        start_sn,
        start_i,
    );

    NEXT_ATOM_SN.fetch_add(atoms.len() as u32, Ordering::AcqRel);

    let mut i_added = Vec::new(); // Used for populating H.

    for atom in &atoms {
        mol.atoms.push(atom.clone());
        i_added.push(mol.atoms.len() - 1);
    }
    for bond in bonds {
        mol.bonds.push(bond);
    }

    if !template.is_ring() {
        // Add back the bond between this atom and the aligner atom.
        mol.bonds.push(Bond {
            bond_type: BondType::Single,
            atom_0_sn: mol.atoms[r_aligner_is[0]].serial_number,
            atom_1_sn: mol.atoms[start_i].serial_number,
            atom_0: r_aligner_is[0],
            atom_1: start_i,
            is_backbone: false,
        });
    }

    mol.reset_posits();
    mol.build_adjacency_list();

    // Set the anchor bond to Aromatic type if appropriate.
    if template == Template::AromaticRing {
        if anchor_is.len() == 2 {
            let mut atoms_to_update_h = Vec::new(); // avoids db-borrow error.
            for bond in &mut mol.bonds {
                if (bond.atom_0 == anchor_is[0] && bond.atom_1 == anchor_is[1])
                    | (bond.atom_0 == anchor_is[1] && bond.atom_1 == anchor_is[0])
                {
                    bond.bond_type = BondType::Aromatic;
                }

                // todo: This section is dry with the GUI buttons to change bond types. Use a common fn for htis.
                for i in 0..mol.atoms.len() {
                    if bond.atom_0 != i && bond.atom_1 != i {
                        continue;
                    }
                    atoms_to_update_h.push(i);
                }
            }

            for i in atoms_to_update_h {
                remove_hydrogens(mol, i);
                populate_hydrogens_on_atom(
                    mol,
                    i,
                    &mut Vec::new(),
                    state_ui,
                    &mut Default::default(),
                    manip_mode,
                );
            }
        }
    }

    // Get the FF type for this atom prior to adding H.
    // For non-rings, we are currently the selected atom with the added group's anchor.
    // So, remove it and its H atoms.
    for &anchor_i in anchor_is {
        if !template.is_ring() {
            mol.remove_atom(anchor_i);
        }
    }

    for i in i_added {
        populate_hydrogens_on_atom(
            mol,
            i,
            &mut Vec::new(),
            state_ui,
            &mut Default::default(),
            manip_mode,
        );
    }

    for &anchor_i in anchor_is {
        remove_hydrogens(mol, anchor_i);
        populate_hydrogens_on_atom(
            mol,
            anchor_i,
            &mut Vec::new(),
            state_ui,
            &mut Default::default(),
            manip_mode,
        );
    }

    *controls = ControlScheme::Arc {
        center: mol.centroid().into(),
    };

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
        return None;
    }

    // Delete hydrogens; we'll add back if required.
    if element != Hydrogen {
        remove_hydrogens(mol, i_par);

        let mol_wrapper = MoleculeSmall {
            common: mol.clone(),
            ..Default::default()
        };

        redraw(entities, &mol_wrapper, ui, manip_mode);
    }

    let atoms_to_add = bonds_avail(i_par, mol, el_parent);
    let currently_bound_count = mol.adjacency_list[i_par].len();

    // let geom = match atoms_to_add {
    let geom = match atoms_to_add + currently_bound_count {
        4 => BondGeom::Tetrahedral,
        3 => BondGeom::Planar,
        2 => BondGeom::Linear,
        _ => {
            eprintln!("Error: Unexpected atoms to add count.");
            BondGeom::Tetrahedral
        }
    };

    // todo: Can't use `common` below here due to the delete_atom code and ownership.
    let posit = match find_appended_posit(
        posit_parent,
        &mol.atoms,
        &mol.adjacency_list[i_par],
        bond_len,
        element,
        geom,
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

    mol.atom_posits.push(posit);
    mol.adjacency_list[i_par].push(new_i);
    mol.adjacency_list.push(vec![i_par]);

    mol.bonds.push(Bond {
        bond_type,
        atom_0_sn: mol.atoms[i_par].serial_number,
        atom_1_sn: new_sn,
        atom_0: i_par,
        atom_1: new_i,
        is_backbone: false,
    });

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

    // Add hydrogens to the new atom.
    // Up to one recursion to add hydrogens to this parent and to the new atom.
    if element != Hydrogen {
        populate_hydrogens_on_atom(mol, i_new, entities, ui, updates, manip_mode);
        *control = ControlScheme::Arc {
            center: mol.centroid().into(),
        };
    }

    // todo: Ideally just add the single entity, and add it to the
    // todo index buffer.
    updates.entities = EntityUpdate::All;

    Some(new_i)
}

fn bonds_avail(i_atom: usize, mol: &MoleculeCommon, el: Element) -> usize {
    let mut bonds_avail: isize = match el {
        Carbon => 4,
        Oxygen => 2,
        Nitrogen => 3, // todo?
        _ => 4,        // todo?
    };

    let mut ar_count = 0;
    for bond in &mol.bonds {
        if bond.atom_0 != i_atom && bond.atom_1 != i_atom {
            continue;
        }

        match bond.bond_type {
            BondType::Single => bonds_avail -= 1,
            BondType::Double => bonds_avail -= 2,
            BondType::Triple => bonds_avail -= 3,
            BondType::Aromatic => {
                ar_count += 1;
                // bonds_avail -= 2
            }
            _ => bonds_avail -= 1,
        }
    }

    // Special override for the non-integer case of Aromatic bonds (4 - 1.5 x 2 = 1)
    if ar_count == 2 {
        bonds_avail -= 3;
    }

    if bonds_avail < 0 {
        0
    } else {
        bonds_avail as usize
    }
}

/// Populate hydrogens on a single atom. Uses tetrahedral, or planar geometry as required
/// based on atoms in the vicinity.
pub fn populate_hydrogens_on_atom(
    mol: &mut MoleculeCommon,
    i: usize,
    entities: &mut Vec<Entity>,
    state_ui: &mut StateUi,
    engine_updates: &mut EngineUpdates,
    manip_mode: ManipMode,
) {
    if i >= mol.atoms.len() {
        eprintln!("Error: Invalid atom index when populating Hydrogens.");
        return;
    }

    let el = mol.atoms[i].element;

    let h_to_add = bonds_avail(i, mol, el);

    println!(
        "\n Atom {} H to add: {h_to_add}",
        mol.atoms[i].serial_number
    );

    // let bonds_remaining = bonds_avail.saturating_sub(adj.len());

    for _ in 0..h_to_add {
        let atom = &mol.atoms[i];
        let (ff_type, bond_len) = {
            let mut v = (None, 1.1);

            // Grabbing the first, arbitrarily.
            for (ff, bl) in hydrogens_avail(&atom.force_field_type) {
                v.0 = Some(ff);
                v.1 = bl;
                break;
            }

            v
        };

        add_atom(
            mol,
            entities,
            i,
            Hydrogen,
            BondType::Single,
            ff_type,
            Some(bond_len),
            0., // Partial charge will be overwritten.
            state_ui,
            engine_updates,
            &mut ControlScheme::None,
            manip_mode,
        );
    }
}

// todo: I think this approach is wrong. You can add multiple of the same one...
/// This is built from Amber's gaff2.dat. Returns each H FF type that can be bound to a given atom
/// (by force field type), and the bond distance in Å.
fn hydrogens_avail(ff_type: &Option<String>) -> Vec<(String, f64)> {
    let Some(f) = ff_type else { return Vec::new() };
    match f.as_ref() {
        // Water
        "ow" => vec![("hw".to_owned(), 0.9572)],
        "hw" => vec![("hw".to_owned(), 1.5136)],

        // Generic sp carbon (c )
        "c" => vec![
            ("h4".to_owned(), 1.1123),
            ("h5".to_owned(), 1.1053),
            ("ha".to_owned(), 1.1010),
        ],

        // sp2 carbon families
        "c1" => vec![("ha".to_owned(), 1.0666), ("hc".to_owned(), 1.0600)],
        "c2" => vec![
            ("h4".to_owned(), 1.0865),
            ("h5".to_owned(), 1.0908),
            ("ha".to_owned(), 1.0882),
            ("hc".to_owned(), 1.0870),
            ("hx".to_owned(), 1.0836),
        ],
        "c3" => vec![
            ("h1".to_owned(), 1.0969),
            ("h2".to_owned(), 1.0950),
            ("h3".to_owned(), 1.0938),
            ("hc".to_owned(), 1.0962),
            ("hx".to_owned(), 1.0911),
        ],
        "c5" => vec![
            ("h1".to_owned(), 1.0972),
            ("h2".to_owned(), 1.0955),
            ("h3".to_owned(), 1.0958),
            ("hc".to_owned(), 1.0954),
            ("hx".to_owned(), 1.0917),
        ],
        "c6" => vec![
            ("h1".to_owned(), 1.0984),
            ("h2".to_owned(), 1.0985),
            ("h3".to_owned(), 1.0958),
            ("hc".to_owned(), 1.0979),
            ("hx".to_owned(), 1.0931),
        ],

        // Aromatic/condensed ring carbons
        "ca" => vec![
            ("ha".to_owned(), 1.0860),
            ("h4".to_owned(), 1.0885),
            ("h5".to_owned(), 1.0880),
        ],
        "cc" => vec![
            ("h4".to_owned(), 1.0809),
            ("h5".to_owned(), 1.0820),
            ("ha".to_owned(), 1.0838),
            ("hx".to_owned(), 1.0827),
        ],
        "cd" => vec![
            ("h4".to_owned(), 1.0818),
            ("h5".to_owned(), 1.0821),
            ("ha".to_owned(), 1.0835),
            ("hx".to_owned(), 1.0801),
        ],
        "ce" => vec![
            ("h4".to_owned(), 1.0914),
            ("h5".to_owned(), 1.0895),
            ("ha".to_owned(), 1.0880),
        ],
        "cf" => vec![
            ("h4".to_owned(), 1.0942),
            ("ha".to_owned(), 1.0885),
            // table also lists h5-cf (reverse order) at 1.0890
            ("h5".to_owned(), 1.0890),
        ],
        "cg" => Vec::new(), // no H entries shown for cg in the provided snippet

        // Other carbon families frequently seen
        "cu" => vec![("ha".to_owned(), 1.0786)],
        "cv" => vec![("ha".to_owned(), 1.0878)],
        "cx" => vec![
            ("h1".to_owned(), 1.0888),
            ("h2".to_owned(), 1.0869),
            ("hc".to_owned(), 1.0865),
            ("hx".to_owned(), 1.0849),
        ],
        "cy" => vec![
            ("h1".to_owned(), 1.0946),
            ("h2".to_owned(), 1.0930),
            ("hc".to_owned(), 1.0947),
            ("hx".to_owned(), 1.0913),
        ],

        // Nitrogen families: protonated H type is "hn"
        "n1" => vec![("hn".to_owned(), 0.9860)],
        "n2" => vec![("hn".to_owned(), 1.0221)],
        "n3" => vec![("hn".to_owned(), 1.0190)],
        "n4" => vec![("hn".to_owned(), 1.0300)],
        "n" => vec![("hn".to_owned(), 1.0130)],
        "n5" => vec![("hn".to_owned(), 1.0211)],
        "n6" => vec![("hn".to_owned(), 1.0183)],
        "n7" => vec![("hn".to_owned(), 1.0195)],
        "n8" => vec![("hn".to_owned(), 1.0192)],
        "n9" => vec![("hn".to_owned(), 1.0192)],
        "na" => vec![("hn".to_owned(), 1.0095)],
        "nh" => vec![("hn".to_owned(), 1.0120)],
        "nj" => vec![("hn".to_owned(), 1.0130)],
        "nl" => vec![("hn".to_owned(), 1.0476)],
        "no" => vec![("hn".to_owned(), 1.0440)],
        "np" => vec![("hn".to_owned(), 1.0210)],
        "nq" => vec![("hn".to_owned(), 1.0180)],
        "ns" => vec![("hn".to_owned(), 1.0132)],
        "nt" => vec![("hn".to_owned(), 1.0105)],
        "nu" => vec![("hn".to_owned(), 1.0137)],
        "nv" => vec![("hn".to_owned(), 1.0114)],
        "nx" => vec![("hn".to_owned(), 1.0338)],
        "ny" => vec![("hn".to_owned(), 1.0339)],
        "nz" => vec![("hn".to_owned(), 1.0271)],

        // Oxygen families: hydroxyl H type is "ho"
        "o" => vec![("ho".to_owned(), 0.9810)],
        "oh" => vec![("ho".to_owned(), 0.9725)],

        // Sulfur families: thiol H type is "hs"
        "s" => vec![("hs".to_owned(), 1.3530)],
        "s4" => vec![("hs".to_owned(), 1.3928)],
        "s6" => vec![("hs".to_owned(), 1.3709)],
        "sh" => vec![("hs".to_owned(), 1.3503)],
        "sy" => vec![("hs".to_owned(), 1.3716)],

        // Phosphorus families: acidic phosphate H type is "hp"
        "p2" => vec![("hp".to_owned(), 1.4272)],
        "p3" => vec![("hp".to_owned(), 1.4256)],
        "p4" => vec![("hp".to_owned(), 1.4271)],
        "p5" => vec![("hp".to_owned(), 1.4205)],
        "py" => vec![("hp".to_owned(), 1.4150)],

        _ => Vec::new(),
    }
}
