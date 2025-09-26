//! For determining which items the user has selected using the mouse cursor. Involves
//! mapping 2D to 3D coordinates, and choosing the right item from what's open and visible.

use lin_alg::f32::Vec3 as Vec3F32;
use na_seq::Element;

use crate::{
    Selection, StateUi, ViewSelLevel,
    drawing::MoleculeView,
    molecule::{Atom, AtomRole, Chain, MolType, Residue},
};

/// From under the cursor; pick the one near the ray, closest to the camera. This function is
/// run after the ray geometry is calculated, and is responsible for determing which atoms, residues, etc
/// are available for selection. It takes into account graphical filters, so only visible items
/// are selected.
///
/// Can select atoms from the protein, or ligand.
pub fn find_selected_atom(
    atoms_along_ray: &[(usize, usize)],
    atoms_lig_along_ray: &[(usize, usize)],
    atoms_na_along_ray: &[(usize, usize)],
    atoms_lipid_along_ray: &[(usize, usize)],
    atoms_prot: &[Atom],
    ress: &[Residue],
    atoms_lig: &[Vec<Atom>],
    atoms_na: &[Vec<Atom>],
    atoms_lipid: &[Vec<Atom>],
    ray: &(Vec3F32, Vec3F32),
    ui: &StateUi,
    chains: &[Chain],
) -> Selection {
    if atoms_along_ray.is_empty()
        && atoms_lig_along_ray.is_empty()
        && atoms_na_along_ray.is_empty()
        && atoms_lipid_along_ray.is_empty()
    {
        return Selection::None;
    }

    const INIT_DIST: f32 = f32::INFINITY;

    // todo: Also consider togglign between ones under the cursor near the front,
    // todo and picking the one closest to the ray.

    struct Nearest {
        mol_type: MolType,
        mol_i: usize,
        atom_i: usize,
    }

    let mut nearest = Nearest {
        mol_type: MolType::Peptide,
        mol_i: 0,
        atom_i: 0,
    };
    let mut near_dist = INIT_DIST;

    for (_mol_i, atom_i) in atoms_along_ray {
        let chain_hidden = {
            let chains_this_atom: Vec<&Chain> =
                chains.iter().filter(|c| c.atoms.contains(atom_i)).collect();

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

        let atom = &atoms_prot[*atom_i];

        if ui.visibility.hide_sidechains || matches!(ui.mol_view, MoleculeView::Backbone) {
            if let Some(role) = atom.role {
                if role == AtomRole::Sidechain || role == AtomRole::H_Sidechain {
                    continue;
                }
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

        if ui.visibility.hide_hydrogen && atom.element == Element::Hydrogen {
            continue;
        }

        if ui.visibility.hide_hetero && atom.hetero {
            continue;
        }

        if ui.visibility.hide_non_hetero && !atom.hetero {
            continue;
        }

        let posit: Vec3F32 = atom.posit.into();
        let dist = (posit - ray.0).magnitude();

        if dist < near_dist {
            nearest = Nearest {
                mol_type: MolType::Peptide,
                mol_i: 0,
                atom_i: *atom_i,
            };
            near_dist = dist;
        }
    }

    // Second pass for ligands; we skip most of the hidden checks here that apply to protein atoms.

    // let mut near_i_lig_num = 0;
    // let mut near_i_lig_atom = 0;
    // let mut near_dist_lig = INIT_DIST;

    for (i_mol, i_atom) in atoms_lig_along_ray.iter() {
        let atom = &atoms_lig[*i_mol][*i_atom];

        if ui.visibility.hide_hydrogen && atom.element == Element::Hydrogen {
            continue;
        }

        let posit: Vec3F32 = atom.posit.into();
        let dist = (posit - ray.0).magnitude();

        if dist < near_dist {
            nearest = Nearest {
                mol_type: MolType::Ligand,
                mol_i: *i_mol,
                atom_i: *i_atom,
            };

            near_dist = dist;
        }
    }
    // todo: DRY!
    for (i_mol, i_atom) in atoms_na_along_ray.iter() {
        let atom = &atoms_na[*i_mol][*i_atom];

        if ui.visibility.hide_hydrogen && atom.element == Element::Hydrogen {
            continue;
        }

        let posit: Vec3F32 = atom.posit.into();
        let dist = (posit - ray.0).magnitude();

        if dist < near_dist {
            nearest = Nearest {
                mol_type: MolType::NucleicAcid,
                mol_i: *i_mol,
                atom_i: *i_atom,
            };

            near_dist = dist;
        }
    }

    // todo: DRY!
    for (i_mol, i_atom) in atoms_lipid_along_ray.iter() {
        let atom = &atoms_lipid[*i_mol][*i_atom];

        if ui.visibility.hide_hydrogen && atom.element == Element::Hydrogen {
            continue;
        }

        let posit: Vec3F32 = atom.posit.into();
        let dist = (posit - ray.0).magnitude();

        if dist < near_dist {
            nearest = Nearest {
                mol_type: MolType::Lipid,
                mol_i: *i_mol,
                atom_i: *i_atom,
            };

            near_dist = dist;
        }
    }

    // This is equivalent to our empty check above, but catches the case of the atom count being
    // empty due to hidden chains.
    if near_dist == INIT_DIST {
        return Selection::None;
    }

    match ui.view_sel_level {
        ViewSelLevel::Atom => match nearest.mol_type {
            MolType::Peptide => Selection::Atom(nearest.atom_i),
            MolType::Ligand => Selection::AtomLig((nearest.mol_i, nearest.atom_i)),
            MolType::NucleicAcid => Selection::AtomNucleicAcid((nearest.mol_i, nearest.atom_i)),
            MolType::Lipid => Selection::AtomLipid((nearest.mol_i, nearest.atom_i)),
            _ => unreachable!(),
        },
        ViewSelLevel::Residue => {
            for (i_res, _res) in ress.iter().enumerate() {
                let atom_near = &atoms_prot[nearest.mol_i];
                if let Some(res_i) = atom_near.residue {
                    if res_i == i_res {
                        return Selection::Residue(i_res);
                    }
                }
            }
            Selection::None // Selected atom is not in a residue.
        }
    }
}

/// Helper
pub fn points_along_ray_inner(
    result: &mut Vec<(usize, usize)>,
    ray: &(Vec3F32, Vec3F32),
    ray_dir: Vec3F32,
    dist_thresh: f32,
    i_mol: usize,
    i_atom: usize,
    atom: &Atom,
) {
    let atom_pos: Vec3F32 = atom.posit.into();

    // Compute the closest point on the ray to the atom position
    let to_atom: Vec3F32 = atom_pos - ray.0;
    let t = to_atom.dot(ray_dir);
    let closest_point = ray.0 + ray_dir * t;

    // Compute the perpendicular distance to the ray
    let dist_to_ray = (atom_pos - closest_point).magnitude();

    // todo: take atom radius into account. E.g. Hydrogens should required a smaller dist.
    // todo: This approach is a bit sloppy, but probably better than not including it.
    if atom.element == Element::Hydrogen {
        // todo: This seems to prevent selecting at all; not sure why.
        // dist_thresh *= 0.9;
    }
    if dist_to_ray < dist_thresh {
        result.push((i_mol, i_atom));
    }
}

/// Used for cursor selection. Returns (atom indices prot, atom indices lig)
pub fn points_along_ray(
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
        points_along_ray_inner(&mut result_prot, &ray, ray_dir, dist_thresh, 0, i, atom);
    }

    for (i_mol, atoms) in atoms_lig.iter().enumerate() {
        for (i, atom) in atoms.iter().enumerate() {
            points_along_ray_inner(&mut result_lig, &ray, ray_dir, dist_thresh, i_mol, i, atom);
        }
    }

    for (i_mol, atoms) in atoms_na.iter().enumerate() {
        for (i, atom) in atoms.iter().enumerate() {
            points_along_ray_inner(&mut result_na, &ray, ray_dir, dist_thresh, i_mol, i, atom);
        }
    }

    for (i_mol, atoms) in atoms_lipid.iter().enumerate() {
        for (i, atom) in atoms.iter().enumerate() {
            points_along_ray_inner(
                &mut result_lipid,
                &ray,
                ray_dir,
                dist_thresh,
                i_mol,
                i,
                atom,
            );
        }
    }

    (result_prot, result_lig, result_na, result_lipid)
}
