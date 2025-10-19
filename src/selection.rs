//! For determining which items the user has selected using the mouse cursor. Involves
//! mapping 2D to 3D coordinates, and choosing the right item from what's open and visible.

use lin_alg::f32::Vec3 as Vec3F32;
use na_seq::Element;

use crate::{
    Selection, StateUi, ViewSelLevel,
    drawing::MoleculeView,
    molecule::{Atom, AtomRole, Bond, Chain, MolType, Residue},
};

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
/// run after the ray geometry is calculated, and is responsible for determing which atoms, residues, etc
/// are available for selection. It takes into account graphical filters, so only visible items
/// are selected.
///
/// Can select atoms from the protein, or ligand.
pub fn find_selected_atom(
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
) -> Selection {
    if items_pep_along_ray.is_empty()
        && items_lig_along_ray.is_empty()
        && items_na_along_ray.is_empty()
        && items_lipid_along_ray.is_empty()
    {
        return Selection::None;
    }

    const INIT_DIST: f32 = f32::INFINITY;

    // todo: Also consider togglign between ones under the cursor near the front,
    // todo and picking the one closest to the ray.

    let mut nearest = Nearest {
        mol_type: MolType::Peptide,
        mol_i: 0,
        atom_i: 0,
    };
    let mut near_dist = INIT_DIST;

    for (_mol_i, atom_i) in items_pep_along_ray {
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

        let atom = &atoms_pep[*atom_i];

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

        if ui.visibility.hide_hydrogen && atom.element == Element::Hydrogen {
            continue;
        }

        if ui.visibility.hide_hetero && atom.hetero {
            continue;
        }

        if ui.visibility.hide_protein && !atom.hetero {
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

    nearest_in_group(
        items_lig_along_ray,
        atoms_lig,
        bonds_lig,
        MolType::Ligand,
        bond_mode,
        ui.visibility.hide_hydrogen,
        ray.0,
        &mut nearest,
        &mut near_dist,
    );

    nearest_in_group(
        items_na_along_ray,
        atoms_na,
        bonds_na,
        MolType::NucleicAcid,
        bond_mode,
        ui.visibility.hide_hydrogen,
        ray.0,
        &mut nearest,
        &mut near_dist,
    );

    nearest_in_group(
        items_lipid_along_ray,
        atoms_lipid,
        bonds_lipid,
        MolType::Lipid,
        bond_mode,
        ui.visibility.hide_hydrogen,
        ray.0,
        &mut nearest,
        &mut near_dist,
    );

    // This is equivalent to our empty check above, but catches the case of the atom count being
    // empty due to hidden chains.
    if near_dist == INIT_DIST {
        return Selection::None;
    }

    let indices = nearest.indices();
    match ui.view_sel_level {
        ViewSelLevel::Atom => match nearest.mol_type {
            MolType::Peptide => Selection::AtomPeptide(nearest.atom_i),
            MolType::Ligand => Selection::AtomLig(indices),
            MolType::NucleicAcid => Selection::AtomNucleicAcid(indices),
            MolType::Lipid => Selection::AtomLipid(indices),
            _ => unreachable!(),
        },
        ViewSelLevel::Residue => {
            match nearest.mol_type {
                MolType::Peptide => {
                    for (i_res, _res) in ress.iter().enumerate() {
                        let atom_near = &atoms_pep[nearest.atom_i];
                        if let Some(i) = atom_near.residue
                            && i == i_res
                        {
                            return Selection::Residue(i_res);
                        }
                    }
                    Selection::None // Selected atom is not in a residue.
                }
                // These are the same as bove.
                MolType::Ligand => Selection::AtomLig(indices),
                MolType::NucleicAcid => Selection::AtomNucleicAcid(indices),
                MolType::Lipid => Selection::AtomLipid(indices),
                _ => unreachable!(),
            }
        }
        ViewSelLevel::Bond => match nearest.mol_type {
            // todo: Rework this (with appropriate steps upstream). Get bonds along ray.
            MolType::Peptide => Selection::BondPeptide(nearest.atom_i),
            MolType::Ligand => Selection::BondLig(indices),
            MolType::NucleicAcid => Selection::BondNucleicAcid(indices),
            MolType::Lipid => Selection::BondLipid(indices),
            _ => unreachable!(),
        },
    }
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
    if dist_to_ray < dist_thresh {
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
                );
            }
        }
    }

    (result_prot, result_lig, result_na, result_lipid)
}

/// A helper
fn nearest_in_group(
    items: &[(usize, usize)],
    atoms: &[Vec<Atom>],
    bonds: &[Vec<Bond>],
    mol_type: MolType,
    bond_mode: bool,
    hide_h: bool,
    ray_origin: Vec3F32,
    nearest: &mut Nearest,
    near_dist: &mut f32,
) {
    for (i_mol, i_atom) in items.iter() {
        let posit: Vec3F32 = if bond_mode {
            let bond = &bonds[*i_mol][*i_atom];
            let atom_0 = &atoms[*i_mol][bond.atom_0];
            let atom_1 = &atoms[*i_mol][bond.atom_1];

            if hide_h
                && (atom_0.element == Element::Hydrogen || atom_1.element == Element::Hydrogen)
            {
                continue;
            }

            ((atom_0.posit + atom_1.posit) / 2.).into()
        } else {
            let atom = &atoms[*i_mol][*i_atom];

            if hide_h && atom.element == Element::Hydrogen {
                continue;
            }

            atom.posit.into()
        };

        let dist = (posit - ray_origin).magnitude();
        if dist < *near_dist {
            *nearest = Nearest {
                mol_type,
                mol_i: *i_mol,
                atom_i: *i_atom,
            };
            *near_dist = dist;
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
        let posit = (atoms_peptide[bond.atom_0].posit + atoms_peptide[bond.atom_0].posit) / 2.;
        points_along_ray_inner(
            &mut result_prot,
            &ray,
            ray_dir,
            dist_thresh,
            0,
            i,
            posit.into(),
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

                points_along_ray_inner(result, &ray, ray_dir, dist_thresh, i_mol, i, posit.into());
            }
        }
    }

    (result_prot, result_lig, result_na, result_lipid)
}
