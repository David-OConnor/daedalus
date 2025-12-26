//! For determining which items the user has selected using the mouse cursor. Involves
//! mapping 2D to 3D coordinates, and choosing the right item from what's open and visible.

use graphics::{ControlScheme, Scene};
use lin_alg::{f32::Vec3 as Vec3F32, map_linear};
use na_seq::{Element, Element::Hydrogen};

use crate::{
    Selection, State, StateUi, ViewSelLevel,
    drawing::MoleculeView,
    molecule::{Atom, AtomRole, Bond, Chain, MolType, MoleculeCommon, Residue},
    util::orbit_center,
};

// For hydrogens
const SELECTION_DIST_THRESH_H: f32 = 0.4; // e.g. ball + stick, or stick.
const SELECTION_DIST_THRESH_SMALL: f32 = 0.7; // e.g. ball + stick, or stick.
const SELECTION_DIST_THRESH_BOND: f32 = 0.5; // e.g. ball + stick, or stick.
// Setting this high rel to `THRESH_SMALL` will cause more accidental selections of nearby atoms that
// the cursor is closer to the center of, but are behind the desired one.
// Setting it too low will cause the selector to "miss", even though the cursor is on an atom visual.
const SELECTION_DIST_THRESH_LARGE: f32 = 1.1; // e.g. VDW views like spheres.

const SEL_NEAR_PAD: f32 = 4.;

#[derive(Debug)]
struct Nearest {
    mol_type: MolType,
    mol_i: usize,
    atom_i: usize,
}

impl Nearest {
    pub fn indices(&self) -> (usize, usize) {
        (self.mol_i, self.atom_i)
    }
}

/// From under the cursor; pick the one near the ray, closest to the camera. This function is
/// run after the ray geometry is calculated, and is responsible for determining which atoms, residues, etc
/// are available for selection. It takes into account graphical filters, so only visible items
/// are selected.
///
/// Can select atoms from the protein, or ligand. Returns both the selection, and the distance;
/// we use this if running this for both atoms and bonds, so we can find the closest of the two.
pub fn find_selected_atom_or_bond(
    items_pep_along_ray: &[(usize, usize)],
    items_lig_along_ray: &[(usize, usize)],
    items_na_along_ray: &[(usize, usize)],
    items_lipid_along_ray: &[(usize, usize)],
    atoms_pep: &[Atom],
    ress: &[Residue],
    atoms_lig: &[Vec<Atom>],
    atoms_na: &[Vec<Atom>],
    atoms_lipid: &[Vec<Atom>],
    ray: &(Vec3F32, Vec3F32),
    ui: &StateUi,
    chains: &[Chain],
    // These aren't required if not in bond mode.
    bonds_pep: &[Bond],
    bonds_lig: &[Vec<Bond>],
    bonds_na: &[Vec<Bond>],
    bonds_lipid: &[Vec<Bond>],
    bond_mode: bool,
    shift_held: bool,
) -> (Selection, f32) {
    if items_pep_along_ray.is_empty()
        && items_lig_along_ray.is_empty()
        && items_na_along_ray.is_empty()
        && items_lipid_along_ray.is_empty()
    {
        return (Selection::None, 0.);
    }

    const INIT_DIST: f32 = f32::INFINITY;

    let ray_dir = (ray.1 - ray.0).to_normalized();
    let mut near_t = INIT_DIST;

    let mut nearest = Nearest {
        mol_type: MolType::Peptide,
        mol_i: 0,
        atom_i: 0,
    };
    let mut near_dist = INIT_DIST;

    for (i_mol, i_atom) in items_pep_along_ray {
        let chain_hidden = {
            let chains_this_atom: Vec<&Chain> =
                chains.iter().filter(|c| c.atoms.contains(i_atom)).collect();

            let mut hidden = false;
            for chain in &chains_this_atom {
                if !chain.visible {
                    hidden = true;
                    break;
                }
            }
            hidden
        };

        if chain_hidden {
            continue;
        }

        let atom = &atoms_pep[*i_atom];

        if ui.visibility.hide_sidechains || matches!(ui.mol_view, MoleculeView::Backbone) {
            if let Some(role) = atom.role
                && (role == AtomRole::Sidechain || role == AtomRole::H_Sidechain)
            {
                continue;
            }
        }

        if let Some(role) = atom.role {
            if ui.visibility.hide_sidechains && role == AtomRole::Sidechain {
                continue;
            }
            if role == AtomRole::Water
                && (ui.visibility.hide_water
                    || matches!(
                        ui.mol_view,
                        MoleculeView::SpaceFill | MoleculeView::Backbone
                    ))
            {
                continue;
            }
        }

        if ui.visibility.hide_hydrogen && atom.element == Hydrogen {
            continue;
        }

        if ui.visibility.hide_hetero && atom.hetero {
            continue;
        }

        if ui.visibility.hide_protein && !atom.hetero {
            continue;
        }

        let posit: Vec3F32 = if bond_mode {
            let bond = &bonds_pep[*i_atom];
            let atom_0 = &atoms_pep[bond.atom_0];
            let atom_1 = &atoms_pep[bond.atom_1];

            if ui.visibility.hide_hydrogen
                && (atom_0.element == Hydrogen || atom_1.element == Hydrogen)
            {
                continue;
            }

            ((atom_0.posit + atom_1.posit) / 2.).into()
        } else {
            let atom = &atoms_pep[*i_atom];

            if ui.visibility.hide_hydrogen && atom.element == Hydrogen {
                continue;
            }

            atom.posit.into()
        };

        let (dist_to_ray, t) = ray_metrics(ray.0, ray_dir, posit);
        if t < 0.0 {
            continue;
        }

        let eps = 1e-4;
        if dist_to_ray + eps < near_dist || ((dist_to_ray - near_dist).abs() <= eps && t < near_t) {
            nearest = Nearest {
                mol_type: MolType::Peptide,
                mol_i: *i_mol,
                atom_i: *i_atom,
            };
            near_dist = dist_to_ray;
            near_t = t;
        }
    }

    // Second pass for ligands; we skip most of the hidden checks here that apply to protein atoms.

    nearest_in_group(
        items_lig_along_ray,
        atoms_lig,
        bonds_lig,
        MolType::Ligand,
        bond_mode,
        ui.visibility.hide_hydrogen,
        ray.0,
        ray_dir,
        &mut nearest,
        &mut near_dist,
        &mut near_t,
    );

    nearest_in_group(
        items_na_along_ray,
        atoms_na,
        bonds_na,
        MolType::NucleicAcid,
        bond_mode,
        ui.visibility.hide_hydrogen,
        ray.0,
        ray_dir,
        &mut nearest,
        &mut near_dist,
        &mut near_t,
    );

    nearest_in_group(
        items_lipid_along_ray,
        atoms_lipid,
        bonds_lipid,
        MolType::Lipid,
        bond_mode,
        ui.visibility.hide_hydrogen,
        ray.0,
        ray_dir,
        &mut nearest,
        &mut near_dist,
        &mut near_t,
    );

    // This is equivalent to our empty check above, but catches the case of the atom count being
    // empty due to hidden chains.
    if near_dist == INIT_DIST {
        return (Selection::None, 0.);
    }

    let indices = nearest.indices();

    let sel = match ui.view_sel_level {
        // ViewSelLevel::Atom => match nearest.mol_type {
        ViewSelLevel::Atom | ViewSelLevel::Bond => {
            if bond_mode {
                match nearest.mol_type {
                    // todo: Rework this (with appropriate steps upstream). Get bonds along ray.
                    MolType::Peptide => Selection::BondPeptide(nearest.atom_i),
                    MolType::Ligand => {
                        if shift_held {
                            match &ui.selection {
                                Selection::BondLig((_mol_i_prev, bond_i_prev)) => {
                                    let updated = vec![*bond_i_prev];
                                    multi_sel_helper(
                                        updated,
                                        indices.0,
                                        indices.1,
                                        MolType::Ligand,
                                        bond_mode,
                                    )
                                }
                                Selection::BondsLig((_mol_i_prev, atoms_i_prev)) => {
                                    let updated = atoms_i_prev.clone();
                                    multi_sel_helper(
                                        updated,
                                        indices.0,
                                        indices.1,
                                        MolType::Ligand,
                                        bond_mode,
                                    )
                                }

                                _ => Selection::BondLig(indices),
                            }
                        } else {
                            Selection::BondLig(indices)
                        }
                    }
                    MolType::NucleicAcid => Selection::BondNucleicAcid(indices),
                    MolType::Lipid => Selection::BondLipid(indices),
                    _ => unreachable!(),
                }
            } else {
                match nearest.mol_type {
                    MolType::Peptide => {
                        if shift_held {
                            match &ui.selection {
                                Selection::AtomPeptide(atom_i) => {
                                    let updated = vec![*atom_i];
                                    multi_sel_helper(
                                        updated,
                                        indices.0,
                                        indices.1,
                                        MolType::Peptide,
                                        bond_mode,
                                    )
                                }
                                Selection::AtomsPeptide(atoms_i) => {
                                    let updated = atoms_i.clone();
                                    multi_sel_helper(
                                        updated,
                                        indices.0,
                                        indices.1,
                                        MolType::Peptide,
                                        bond_mode,
                                    )
                                }
                                _ => Selection::AtomPeptide(nearest.atom_i),
                            }
                        } else {
                            Selection::AtomPeptide(nearest.atom_i)
                        }
                    }
                    MolType::Ligand => {
                        if shift_held {
                            match &ui.selection {
                                Selection::AtomLig((_mol_i_prev, atom_i_prev)) => {
                                    let updated = vec![*atom_i_prev];
                                    multi_sel_helper(
                                        updated,
                                        indices.0,
                                        indices.1,
                                        MolType::Ligand,
                                        bond_mode,
                                    )
                                }
                                Selection::AtomsLig((_mol_i_prev, atoms_i_prev)) => {
                                    let updated = atoms_i_prev.clone();
                                    multi_sel_helper(
                                        updated,
                                        indices.0,
                                        indices.1,
                                        MolType::Ligand,
                                        bond_mode,
                                    )
                                }

                                _ => Selection::AtomLig(indices),
                            }
                        } else {
                            Selection::AtomLig(indices)
                        }
                    }
                    MolType::NucleicAcid => Selection::AtomNucleicAcid(indices),
                    MolType::Lipid => Selection::AtomLipid(indices),
                    _ => unreachable!(),
                }
            }
        }

        ViewSelLevel::Residue => {
            match nearest.mol_type {
                MolType::Peptide => {
                    for (i_res, _res) in ress.iter().enumerate() {
                        let atom_near = &atoms_pep[nearest.atom_i];
                        if let Some(i) = atom_near.residue
                            && i == i_res
                        {
                            return (Selection::Residue(i_res), near_dist);
                        }
                    }
                    Selection::None // Selected atom is not in a residue.
                }
                // These are the same as above.
                MolType::Ligand => Selection::AtomLig(indices),
                MolType::NucleicAcid => Selection::AtomNucleicAcid(indices),
                MolType::Lipid => Selection::AtomLipid(indices),
                _ => unreachable!(),
            }
        }
    };

    (sel, near_dist)
}

/// Helper
pub fn points_along_ray_inner(
    result: &mut Vec<(usize, usize)>,
    ray: &(Vec3F32, Vec3F32),
    ray_dir: Vec3F32,
    dist_thresh: f32,
    i_mol: usize,
    i_atom_or_bond: usize,
    posit: Vec3F32,
    el: Option<Element>,
) {
    // Compute the closest point on the ray to the atom position
    let to_atom: Vec3F32 = posit - ray.0;
    let t = to_atom.dot(ray_dir);
    let closest_point = ray.0 + ray_dir * t;

    // Compute the perpendicular distance to the ray
    let dist_to_ray = (posit - closest_point).magnitude();

    // todo: take atom radius into account. E.g. Hydrogens should required a smaller dist.
    // todo: This approach is a bit sloppy, but probably better than not including it.
    // if atom.element == Element::Hydrogen {
    //     // todo: This seems to prevent selecting at all; not sure why.
    //     // dist_thresh *= 0.9;
    // }

    // We render Hydrogens smaller; use a smaller thresh
    let mut thresh = dist_thresh;
    if el == Some(Hydrogen) && thresh == SELECTION_DIST_THRESH_SMALL {
        thresh = SELECTION_DIST_THRESH_H;
    }

    if dist_to_ray < thresh {
        result.push((i_mol, i_atom_or_bond));
    }
}

/// Used for cursor selection. Returns (atom indices prot, atom indices lig)
pub fn points_along_ray_atom(
    ray: (Vec3F32, Vec3F32),
    atoms_peptide: &[Atom],
    atoms_lig: &[Vec<Atom>],
    atoms_na: &[Vec<Atom>],
    atoms_lipid: &[Vec<Atom>],
    dist_thresh: f32,
    // Each tuple is (mol i, atom i in that mol)
) -> (
    Vec<(usize, usize)>,
    Vec<(usize, usize)>,
    Vec<(usize, usize)>,
    Vec<(usize, usize)>,
) {
    let mut result_prot = Vec::new();
    let mut result_lig = Vec::new();
    let mut result_na = Vec::new();
    let mut result_lipid = Vec::new();

    let ray_dir = (ray.1 - ray.0).to_normalized();

    for (i, atom) in atoms_peptide.iter().enumerate() {
        points_along_ray_inner(
            &mut result_prot,
            &ray,
            ray_dir,
            dist_thresh,
            0,
            i,
            atom.posit.into(),
            Some(atom.element),
        );
    }

    for (result, atoms_list) in [
        (&mut result_lig, &atoms_lig),
        (&mut result_na, &atoms_na),
        (&mut result_lipid, &atoms_lipid),
    ] {
        for (i_mol, atoms) in atoms_list.iter().enumerate() {
            for (i, atom) in atoms.iter().enumerate() {
                points_along_ray_inner(
                    result,
                    &ray,
                    ray_dir,
                    dist_thresh,
                    i_mol,
                    i,
                    atom.posit.into(),
                    Some(atom.element),
                );
            }
        }
    }

    (result_prot, result_lig, result_na, result_lipid)
}

/// A helper
fn nearest_in_group(
    items: &[(usize, usize)],
    atoms: &[Vec<Atom>], // All atoms
    bonds: &[Vec<Bond>], // All bonds
    mol_type: MolType,
    bond_mode: bool,
    hide_h: bool,
    ray_origin: Vec3F32,
    ray_dir: Vec3F32,
    nearest: &mut Nearest,
    near_dist: &mut f32,
    near_t: &mut f32, // tie-break: depth along ray
) {
    let eps = 1e-4;

    for (i_mol, i_atom_bond) in items.iter() {
        let posit: Vec3F32 = if bond_mode {
            let bond = &bonds[*i_mol][*i_atom_bond];
            let atom_0 = &atoms[*i_mol][bond.atom_0];
            let atom_1 = &atoms[*i_mol][bond.atom_1];

            if hide_h && (atom_0.element == Hydrogen || atom_1.element == Hydrogen) {
                continue;
            }

            ((atom_0.posit + atom_1.posit) / 2.).into()
        } else {
            let atom = &atoms[*i_mol][*i_atom_bond];
            if hide_h && atom.element == Hydrogen {
                continue;
            }
            atom.posit.into()
        };

        let (dist_to_ray, t) = ray_metrics(ray_origin, ray_dir, posit);
        if t < 0.0 {
            continue;
        }

        if dist_to_ray + eps < *near_dist
            || ((dist_to_ray - *near_dist).abs() <= eps && t < *near_t)
        {
            *nearest = Nearest {
                mol_type,
                mol_i: *i_mol,
                atom_i: *i_atom_bond,
            };
            *near_dist = dist_to_ray;
            *near_t = t;
        }
    }
}

/// Used for cursor selection. Returns (atom indices prot, atom indices lig)
pub fn points_along_ray_bond(
    ray: (Vec3F32, Vec3F32),
    bonds_peptide: &[Bond],
    bonds_lig: &[Vec<Bond>],
    bonds_na: &[Vec<Bond>],
    bonds_lipid: &[Vec<Bond>],
    // We need atoms to get positions
    atoms_peptide: &[Atom],
    atoms_lig: &[Vec<Atom>],
    atoms_na: &[Vec<Atom>],
    atoms_lipid: &[Vec<Atom>],
    dist_thresh: f32,
    // Each tuple is (mol i, bondi in that mol)
) -> (
    Vec<(usize, usize)>,
    Vec<(usize, usize)>,
    Vec<(usize, usize)>,
    Vec<(usize, usize)>,
) {
    let mut result_prot = Vec::new();
    let mut result_lig = Vec::new();
    let mut result_na = Vec::new();
    let mut result_lipid = Vec::new();

    let ray_dir = (ray.1 - ray.0).to_normalized();

    for (i, bond) in bonds_peptide.iter().enumerate() {
        let posit = (atoms_peptide[bond.atom_0].posit + atoms_peptide[bond.atom_1].posit) / 2.;
        points_along_ray_inner(
            &mut result_prot,
            &ray,
            ray_dir,
            dist_thresh,
            0,
            i,
            posit.into(),
            None,
        );
    }

    for (result, atoms_list, bonds_list) in [
        (&mut result_lig, &atoms_lig, &bonds_lig),
        (&mut result_na, &atoms_na, &bonds_na),
        (&mut result_lipid, &atoms_lipid, &bonds_lipid),
    ] {
        for (i_mol, bonds) in bonds_list.iter().enumerate() {
            for (i, bond) in bonds.iter().enumerate() {
                let a0 = atoms_list[i_mol][bond.atom_0].posit;
                let a1 = atoms_list[i_mol][bond.atom_1].posit;
                let posit = (a0 + a1) / 2.0;

                points_along_ray_inner(
                    result,
                    &ray,
                    ray_dir,
                    dist_thresh,
                    i_mol,
                    i,
                    posit.into(),
                    None,
                );
            }
        }
    }

    (result_prot, result_lig, result_na, result_lipid)
}

pub(crate) fn handle_selection_attempt(
    state: &mut State,
    scene: &mut Scene,
    redraw_protein: &mut bool,
    redraw_lig: &mut bool,
    redraw_na: &mut bool,
    redraw_lipid: &mut bool,
) {
    let Some(mut cursor) = state.ui.cursor_pos else {
        return;
    };

    let mut selected_ray = scene.screen_to_render(cursor);

    // Clip the near end of this to prevent false selections that seem to the user
    // to be behind the camera.
    let diff = selected_ray.1 - selected_ray.0;

    selected_ray.0 += diff.to_normalized() * SEL_NEAR_PAD;

    // todo: Lots of DRY here!

    // todo: I don't like this rebuilding.
    fn get_atoms(mol: &MoleculeCommon) -> Vec<Atom> {
        // todo: I don't like this clone!
        mol.atoms
            .iter()
            .enumerate()
            .map(|(i, a)| Atom {
                posit: mol.atom_posits[i],
                element: a.element,
                ..Default::default()
            })
            .collect()
    }

    let mut lig_atoms = Vec::new();
    for mol in &state.ligands {
        lig_atoms.push(get_atoms(&mol.common));
    }

    let mut na_atoms = Vec::new();
    for mol in &state.nucleic_acids {
        na_atoms.push(get_atoms(&mol.common));
    }
    let mut lipid_atoms = Vec::new();
    for mol in &state.lipids {
        lipid_atoms.push(get_atoms(&mol.common));
    }

    let (pep_atoms, pep_res) = match &state.peptide {
        Some(p) => (&p.common.atoms, &p.residues),
        None => (&Vec::new(), &Vec::new()),
    };

    // If we don't scale the selection distance appropriately, an atom etc
    // behind the desired one, but closer to the ray, may be selected; likely
    // this is undesired.
    let dist_thresh = match state.ui.mol_view {
        MoleculeView::SpaceFill => SELECTION_DIST_THRESH_LARGE,
        _ => match state.ui.view_sel_level {
            ViewSelLevel::Bond => SELECTION_DIST_THRESH_BOND,
            _ => SELECTION_DIST_THRESH_SMALL,
        },
    };

    let (selection, _dist) = match state.ui.view_sel_level {
        ViewSelLevel::Bond => {
            let mut pep_bonds = Vec::new();
            // todo: I don' tlike these clones.
            if let Some(mol) = &state.peptide {
                pep_bonds = mol.common.bonds.clone();
            }

            let mut lig_bonds = Vec::new();
            // todo: I don' tlike these clones.
            for mol in &state.ligands {
                lig_bonds.push(mol.common.bonds.clone());
            }

            let mut na_bonds = Vec::new();
            for mol in &state.nucleic_acids {
                na_bonds.push(mol.common.bonds.clone());
            }
            let mut lipid_bonds = Vec::new();
            for mol in &state.lipids {
                lipid_bonds.push(mol.common.bonds.clone());
            }
            let (
                atoms_along_ray_pep,
                atoms_along_ray_lig,
                atoms_along_ray_na,
                atoms_along_ray_lipid,
            ) = points_along_ray_bond(
                selected_ray,
                &pep_bonds,
                &lig_bonds,
                &na_bonds,
                &lipid_bonds,
                pep_atoms,
                &lig_atoms,
                &na_atoms,
                &lipid_atoms,
                dist_thresh,
            );

            find_selected_atom_or_bond(
                &atoms_along_ray_pep,
                &atoms_along_ray_lig,
                &atoms_along_ray_na,
                &atoms_along_ray_lipid,
                pep_atoms,
                &Vec::new(), // todo: Peptide residues. once ready.
                &lig_atoms,
                &na_atoms,
                &lipid_atoms,
                &selected_ray,
                &state.ui,
                &Vec::new(),
                &pep_bonds,
                &lig_bonds,
                &na_bonds,
                &lipid_bonds,
                true,
                state.volatile.inputs_commanded.run,
            )
        }
        _ => {
            let (
                atoms_along_ray_pep,
                atoms_along_ray_lig,
                atoms_along_ray_na,
                atoms_along_ray_lipid,
            ) = points_along_ray_atom(
                selected_ray,
                pep_atoms,
                &lig_atoms,
                &na_atoms,
                &lipid_atoms,
                dist_thresh,
            );

            find_selected_atom_or_bond(
                &atoms_along_ray_pep,
                &atoms_along_ray_lig,
                &atoms_along_ray_na,
                &atoms_along_ray_lipid,
                pep_atoms,
                pep_res,
                &lig_atoms,
                &na_atoms,
                &lipid_atoms,
                &selected_ray,
                &state.ui,
                &Vec::new(),
                &[],
                &[],
                &[],
                &[],
                false,
                state.volatile.inputs_commanded.run,
            )
        }
    };

    match selection {
        Selection::AtomPeptide(_)
        | Selection::AtomsPeptide(_)
        | Selection::BondPeptide(_)
        | Selection::Residue(_) => {
            let mol_i = 0;
            state.volatile.active_mol = Some((MolType::Peptide, mol_i));
        }
        Selection::AtomLig((mol_i, _))
        | Selection::AtomsLig((mol_i, _))
        | Selection::BondLig((mol_i, _))
        | Selection::BondsLig((mol_i, _)) => {
            state.volatile.active_mol = Some((MolType::Ligand, mol_i));
        }
        Selection::AtomNucleicAcid((mol_i, _)) | Selection::BondNucleicAcid((mol_i, _)) => {
            state.volatile.active_mol = Some((MolType::NucleicAcid, mol_i));
        }
        Selection::AtomLipid((mol_i, _)) | Selection::BondLipid((mol_i, _)) => {
            state.volatile.active_mol = Some((MolType::Lipid, mol_i));
        }
        _ => (),
    }

    if selection == state.ui.selection {
        // Toggle.
        state.ui.selection = Selection::None;
    } else {
        state.ui.selection = selection;
    }

    if let ControlScheme::Arc { center } = &mut scene.input_settings.control_scheme {
        state.volatile.orbit_center = state.volatile.active_mol;
        *center = orbit_center(state);
    }

    *redraw_protein = true;
    *redraw_lig = true;
    *redraw_na = true;
    *redraw_lipid = true;
}

/// A stripped-down version, for the mol editor. See notes there where applicable.
/// Note that this allows selecting *either atoms or bonds*, depending on where the click
/// occurs.
pub fn handle_selection_attempt_mol_editor(
    state: &mut State,
    scene: &mut Scene,
    redraw: &mut bool,
) {
    // todo: Allow a sel mode in the Primary mode that lets you pick either atoms or bonds, like this.

    let Some(cursor) = state.ui.cursor_pos else {
        return;
    };

    let mut selected_ray = scene.screen_to_render(cursor);

    let diff = selected_ray.1 - selected_ray.0;

    selected_ray.0 += diff.to_normalized() * SEL_NEAR_PAD;

    let dist_thresh = match state.ui.mol_view {
        MoleculeView::SpaceFill => SELECTION_DIST_THRESH_LARGE,
        _ => SELECTION_DIST_THRESH_SMALL,
    };

    let mol = &state.mol_editor.mol;

    let atoms: Vec<_> = mol
        .common
        .atoms
        .iter()
        .enumerate()
        .map(|(i, a)| Atom {
            posit: mol.common.atom_posits[i],
            element: a.element,
            ..Default::default()
        })
        .collect();

    let (sel_atoms, dist_atoms) = {
        let (atoms_along_ray_pep, atoms_along_ray_lig, atoms_along_ray_na, atoms_along_ray_lipid) =
            points_along_ray_atom(
                selected_ray,
                &Vec::new(),
                &[atoms.clone()], // todo: This clone...
                &[],
                &[],
                dist_thresh,
            );

        find_selected_atom_or_bond(
            &atoms_along_ray_pep,
            &atoms_along_ray_lig,
            &atoms_along_ray_na,
            &atoms_along_ray_lipid,
            &Vec::new(),
            &Vec::new(),
            &[atoms.clone()], // todo: Don't like this.
            &[],
            &[],
            &selected_ray,
            &state.ui,
            &Vec::new(),
            &Vec::new(),
            &[],
            &[],
            &[],
            false,
            state.volatile.inputs_commanded.run,
        )
    };

    let (sel_bonds, dist_bonds) = {
        let bonds = mol.common.bonds.clone(); // todo: This clone...

        let (bonds_along_ray_pep, bonds_along_ray_lig, bonds_along_ray_na, bonds_along_ray_lipid) =
            points_along_ray_bond(
                selected_ray,
                &Vec::new(),
                &[bonds.clone()], // todo: CLone again...
                &[],
                &[],
                &Vec::new(),
                &[atoms.clone()], // todo: This clone...
                &[],
                &[],
                dist_thresh,
            );

        find_selected_atom_or_bond(
            &bonds_along_ray_pep,
            &bonds_along_ray_lig,
            &bonds_along_ray_na,
            &bonds_along_ray_lipid,
            &Vec::new(),
            &Vec::new(),
            &[atoms],
            &[],
            &[],
            &selected_ray,
            &state.ui,
            &Vec::new(),
            &Vec::new(),
            &[bonds],
            &[],
            &[],
            true,
            state.volatile.inputs_commanded.run,
        )
    };

    let selection = if dist_atoms < dist_bonds {
        sel_atoms
    } else {
        sel_bonds
    };

    if selection == state.ui.selection {
        // Toggle.
        state.ui.selection = Selection::None;
    } else {
        state.ui.selection = selection;
    }

    *redraw = true;
}

/// Handles logic regarding selection changes updating multi-atom lists, or reverting
/// to single-atom variants. Uses toggling behavior.
fn multi_sel_helper(
    mut updated: Vec<usize>,
    mol_i: usize,
    atom_or_bond_i: usize,
    mol_type: MolType,
    bond_mode: bool,
) -> Selection {
    if updated.contains(&atom_or_bond_i) {
        updated.retain(|idx| idx != &atom_or_bond_i);
    } else {
        updated.push(atom_or_bond_i);
    }

    // todo: We should handle the case where the mol i isn't from the previous mol.
    match updated.len() {
        0 => Selection::None,
        // Single-selection variants
        1 => {
            let result = (mol_i, updated[0]);
            match mol_type {
                MolType::Ligand => {
                    if bond_mode {
                        Selection::BondLig(result)
                    } else {
                        Selection::AtomLig(result)
                    }
                }
                MolType::Peptide => Selection::AtomPeptide(updated[0]),
                _ => unimplemented!(),
            }
        }
        // Multi-selection variants
        _ => match mol_type {
            MolType::Ligand => {
                if bond_mode {
                    Selection::BondsLig((mol_i, updated))
                } else {
                    Selection::AtomsLig((mol_i, updated))
                }
            }
            MolType::Peptide => Selection::AtomsPeptide(updated),
            _ => unimplemented!(),
        },
    }
}

// todo: Experimenting
fn ray_metrics(ray_origin: Vec3F32, ray_dir: Vec3F32, posit: Vec3F32) -> (f32, f32) {
    let to_p = posit - ray_origin;
    let t = to_p.dot(ray_dir);
    let closest = ray_origin + ray_dir * t;
    let dist_to_ray = (posit - closest).magnitude();
    (dist_to_ray, t)
}
