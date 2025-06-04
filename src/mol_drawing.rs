//! Handles drawing molecules, bonds etc.

use std::{fmt, io, io::ErrorKind, str::FromStr};

use bincode::{Decode, Encode, error::EncodeError};
use graphics::{ControlScheme, Entity, FWD_VEC, RIGHT_VEC, Scene, UP_VEC};
use lin_alg::{
    f32::{Quaternion, Vec3},
    map_linear,
};

use crate::{
    Selection, State, ViewSelLevel,
    element::Element,
    molecule::{Atom, AtomRole, BondCount, BondType, Chain, Residue, ResidueType, aa_color},
    reflection::ElectronDensity,
    render::{
        ATOM_SHININESS, BACKGROUND_COLOR, BALL_RADIUS_WATER, BALL_STICK_RADIUS,
        BALL_STICK_RADIUS_H, BODY_SHINYNESS, CAM_INIT_OFFSET, Color, MESH_BOND, MESH_CUBE,
        MESH_DENSITY_SURFACE, MESH_DOCKING_BOX, MESH_SOLVENT_SURFACE, MESH_SPHERE_HIGHRES,
        MESH_SPHERE_LOWRES, MESH_SPHERE_MEDRES, RENDER_DIST_FAR, set_docking_light,
        set_static_light,
    },
    surface::{get_mesh_points, mesh_from_sas_points},
    util::orbit_center,
};

const LIGAND_COLOR: Color = (0., 0.4, 1.);
const LIGAND_COLOR_ANCHOR: Color = (1., 0., 1.);
// i.e a flexible bond.
const LIGAND_COLOR_FLEX: Color = (1., 1., 0.);
const COLOR_AA_NON_RESIDUE: Color = (0., 0.8, 1.0);

const COLOR_SELECTED: Color = (1., 0., 0.);
const COLOR_H_BOND: Color = (1., 0.5, 0.1);
const RADIUS_H_BOND: f32 = 0.2; // A scaler relative to covalent sticks.

const COLOR_SFC_DOT: Color = (0.7, 0.7, 0.7);
const COLOR_DOCKING_BOX: Color = (0.3, 0.3, 0.9);
pub const COLOR_DOCKING_SITE_MESH: Color = (0.5, 0.5, 0.9);

pub const BOND_RADIUS: f32 = 0.10;
pub const BOND_RADIUS_LIGAND_RATIO: f32 = 1.3; // Of bond radius.
// const BOND_CAP_RADIUS: f32 = 1./BOND_RADIUS;
pub const BOND_RADIUS_DOUBLE: f32 = 0.07;

pub const RADIUS_SFC_DOT: f32 = 0.05;

const DOCKING_SITE_OPACITY: f32 = 0.35;

const DIMMED_PEPTIDE_AMT: f32 = 0.92; // Higher value means more dim.

// This allows us to more easily customize sphere mesh resolution.
const MESH_BALL_STICK_SPHERE: usize = MESH_SPHERE_MEDRES;
// todo: I believe this causes performance problems on many machines. But looks
// todo much nicer.
const MESH_SPACEFILL_SPHERE: usize = MESH_SPHERE_HIGHRES;
const MESH_WATER_SPHERE: usize = MESH_SPHERE_MEDRES;
const MESH_BOND_CAP: usize = MESH_SPHERE_LOWRES;
// This should ideally be high res, but we experience anomolies on viewing items inside it, while
// the cam is outside.
// const MESH_DOCKING_SITE: usize = MESH_SPHERE_HIGHRES;
const MESH_DOCKING_SITE: usize = MESH_DOCKING_BOX;
const MESH_SURFACE_DOT: usize = MESH_SPHERE_LOWRES;

// For now at least, we hijack the Entity's id field to mean the type it is.
#[derive(Clone, Copy, PartialEq)]
#[repr(u32)]
pub enum EntityType {
    Protein = 0,
    Ligand = 1,
    Density = 2,
    DensitySurface = 3,
    Other = 4,
}

// todo: For ligands that are flexible, highlight the fleixble bonds in a bright color.

fn blend_color(color_0: Color, color_1: Color, portion: f32) -> Color {
    (
        map_linear(portion, (0., 1.), (color_0.0, color_1.0)),
        map_linear(portion, (0., 1.), (color_0.1, color_1.1)),
        map_linear(portion, (0., 1.), (color_0.2, color_1.2)),
    )
}

/// Make ligands stand out visually, when colored by atom.
fn mod_color_for_ligand(color: &Color) -> Color {
    let blend = (0., 0.3, 1.);
    blend_color(*color, blend, 0.5)
}

#[derive(Clone, Copy, PartialEq, Debug, Default, Encode, Decode)]
pub enum MoleculeView {
    Sticks,
    #[default]
    Backbone,
    BallAndStick,
    /// i.e. Van der Waals radius, or CPK.
    SpaceFill,
    Cartoon,
    Surface,
    Mesh,
    Dots,
}

impl FromStr for MoleculeView {
    type Err = io::Error;

    /// This includes some PyMol standard names, which map to the closest visualization we have.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "sticks" | "lines" => Ok(MoleculeView::Sticks),
            "backbone" => Ok(MoleculeView::Backbone),
            "ballandstick" | "ball_and_stick" | "ball-and-stick" => Ok(MoleculeView::BallAndStick),
            "spacefill" | "space_fill" | "space-fill" | "spheres" => Ok(MoleculeView::SpaceFill),
            "cartoon" | "ribbon" => Ok(MoleculeView::Cartoon),
            "surface" => Ok(MoleculeView::Surface),
            "mesh" => Ok(MoleculeView::Mesh),
            "dots" => Ok(MoleculeView::Dots),
            other => Err(io::Error::new(
                ErrorKind::InvalidData,
                format!("invalid MoleculeView: '{}'", other),
            )),
        }
    }
}

impl fmt::Display for MoleculeView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let val = match self {
            Self::Backbone => "Backbone",
            Self::Sticks => "Sticks",
            Self::BallAndStick => "Ball and stick",
            Self::Cartoon => "Cartoon",
            Self::SpaceFill => "Spacefill",
            Self::Surface => "Surface (Van der Waals)",
            Self::Mesh => "Mesh (Van der Waals)",
            Self::Dots => "Dots (Van der Waals)",
        };

        write!(f, "{val}")
    }
}

/// A linear color map using the viridis scheme.
fn color_viridis(i: usize, min: usize, max: usize) -> Color {
    // Normalize i to [0.0, 1.0]
    let t = if max > min {
        // Compute as f32 and clamp to [0,1]
        let tt = (i.saturating_sub(min) as f32) / ((max - min) as f32);
        tt.clamp(0.0, 1.0)
    } else {
        0.0
    };

    // A small set of Viridis control points (R, G, B), from t=0.0 to t=1.0
    const VIRIDIS: [(f32, f32, f32); 5] = [
        (0.267004, 0.004874, 0.329415), // t = 0.00
        (0.229739, 0.322361, 0.545706), // t = 0.25
        (0.127568, 0.566949, 0.550556), // t = 0.50
        (0.369214, 0.788888, 0.382914), // t = 0.75
        (0.993248, 0.906157, 0.143936), // t = 1.00
    ];

    // Scale t into the control‐point index range [0 .. VIRIDIS.len()-1]
    let n_pts = VIRIDIS.len();
    let scaled = t * ((n_pts - 1) as f32);
    let idx = scaled.floor() as usize;
    let idx_next = (idx + 1).min(n_pts - 1);
    let frac = scaled - (idx as f32);

    let (r1, g1, b1) = VIRIDIS[idx];
    let (r2, g2, b2) = VIRIDIS[idx_next];

    // Linear interpolation between the two nearest control points
    let r = r1 + (r2 - r1) * frac;
    let g = g1 + (g2 - g1) * frac;
    let b = b1 + (b2 - b1) * frac;

    (r, g, b)
}

fn atom_color(
    atom: &Atom,
    i: usize,
    residues: &[Residue],
    selection: &Selection,
    view_sel_level: ViewSelLevel,
    dimmed: bool,
    res_color_by_index: bool,
) -> Color {
    let mut result = match view_sel_level {
        ViewSelLevel::Atom => atom.element.color(),
        ViewSelLevel::Residue => {
            let mut color = COLOR_AA_NON_RESIDUE;

            if let Some(res_i) = &atom.residue {
                let res = &residues[*res_i];
                color = match &res.res_type {
                    ResidueType::AminoAcid(aa) => {
                        if res_color_by_index {
                            let mut c = aa_color(*aa);
                            for (i_res, res) in residues.iter().enumerate() {
                                if let Some(res_i) = atom.residue {
                                    if res_i == i {
                                        c = color_viridis(i_res, 0, residues.len());
                                    }
                                }
                            }
                            c
                        } else {
                            aa_color(*aa)
                        }
                    }
                    _ => COLOR_AA_NON_RESIDUE,
                }
            }
            color
        }
    };

    // If selected, the selected color overrides the element or residue color.
    match selection {
        Selection::Atom(sel_i) => {
            if *sel_i == i {
                result = COLOR_SELECTED;
            }
        }
        Selection::Residue(sel_i) => {
            if let Some(res_i) = atom.residue {
                if res_i == *sel_i {
                    result = COLOR_SELECTED;
                }
            }
        }
        Selection::Atoms(sel_is) => {
            if sel_is.contains(&i) {
                result = COLOR_SELECTED;
            }
        }
        Selection::None => (),
    }

    if dimmed && result != COLOR_SELECTED {
        // Desaturate first; otherwise the more saturated initial colors will be relatively visible, while unsaturated
        // ones will appear blackish.
        result = blend_color(result, BACKGROUND_COLOR, DIMMED_PEPTIDE_AMT)
    }

    result
}

/// Adds a cylindrical bond. This is divided into two halves, so they can be color-coded by their side's
/// atom. Adds optional rounding. `thickness` is relative to BOND_RADIUS.
fn add_bond(
    entities: &mut Vec<Entity>,
    posits: (Vec3, Vec3),
    colors: (Color, Color),
    center: Vec3,
    orientation: Quaternion,
    dist_half: f32,
    caps: bool,
    thickness: f32,
    ligand: bool,
) {
    // Split the bond into two entities, so you can color-code them separately based
    // on which atom the half is closer to.
    let center_0 = (posits.0 + center) / 2.;
    let center_1 = (posits.1 + center) / 2.;

    let entity_type = if ligand {
        EntityType::Ligand
    } else {
        EntityType::Protein
    } as u32;

    let mut entity_0 = Entity::new(
        MESH_BOND,
        center_0,
        orientation,
        1.,
        colors.0,
        BODY_SHINYNESS,
    );
    entity_0.class = entity_type;

    let mut entity_1 = Entity::new(
        MESH_BOND,
        center_1,
        orientation,
        1.,
        colors.1,
        BODY_SHINYNESS,
    );
    entity_1.class = entity_type;

    if caps {
        // These spheres are to put a rounded cap on each bond.
        // todo: You only need a dome; performance implications.
        let mut cap_0 = Entity::new(
            MESH_BOND_CAP,
            posits.0,
            Quaternion::new_identity(),
            BOND_RADIUS * thickness,
            colors.0,
            BODY_SHINYNESS,
        );
        let mut cap_1 = Entity::new(
            MESH_BOND_CAP,
            posits.1,
            Quaternion::new_identity(),
            BOND_RADIUS * thickness,
            colors.1,
            BODY_SHINYNESS,
        );

        cap_0.class = entity_type;
        cap_1.class = entity_type;
        entities.push(cap_0);
        entities.push(cap_1);
    }

    let scale = Some(Vec3::new(thickness, dist_half, thickness));
    entity_0.scale_partial = scale;
    entity_1.scale_partial = scale;

    entities.push(entity_0);
    entities.push(entity_1);
}

fn bond_entities(
    entities: &mut Vec<Entity>,
    posit_0: Vec3,
    posit_1: Vec3,
    mut color_0: Color,
    mut color_1: Color,
    bond_type: BondType,
    ligand: bool,
) {
    // todo: You probably need to update this to display double bonds correctly.

    // todo: YOur multibond plane logic is off.

    let center: Vec3 = (posit_0 + posit_1) / 2.;

    let diff = posit_0 - posit_1;
    let diff_unit = diff.to_normalized();
    let orientation = Quaternion::from_unit_vecs(UP_VEC, diff_unit);
    let dist_half = diff.magnitude() / 2.;

    let caps = true; // todo: Remove caps if ball+ stick

    let bond_count = match bond_type {
        BondType::Covalent { count } => count,
        BondType::Hydrogen => BondCount::Single,
        _ => unimplemented!(),
    };

    if bond_type == BondType::Hydrogen {
        color_0 = COLOR_H_BOND;
        color_1 = COLOR_H_BOND;
    }

    // todo: Put this multibond code back.
    // todo: Lots of DRY!
    match bond_count {
        // BondCount::Single => {
        BondCount::Single | BondCount::SingleDoubleHybrid => {
            let thickness = if bond_type == BondType::Hydrogen {
                RADIUS_H_BOND
            } else {
                if ligand { BOND_RADIUS_LIGAND_RATIO } else { 1. }
            };

            add_bond(
                entities,
                (posit_0, posit_1),
                (color_0, color_1),
                center,
                orientation,
                dist_half,
                caps,
                thickness,
                ligand,
            );
        }
        // todo: Put back once you have a dihedral-angle-based approach.
        // BondCount::SingleDoubleHybrid => {
        //     // Draw two offset bond cylinders.
        //     // todo: Set rot_ortho based on dihedral angle.
        //     let rot_ortho = Quaternion::from_unit_vecs(FWD_VEC, UP_VEC);
        //     let rotator = rot_ortho * orientation;
        //
        //     let offset_a = rotator.rotate_vec(Vec3::new(0.2, 0., 0.));
        //     let offset_b = rotator.rotate_vec(Vec3::new(-0.2, 0., 0.));
        //
        //     // todo: Make this one better
        //
        //     add_bond(
        //         entities,
        //         posit_0 + offset_a,
        //         posit_1 + offset_a,
        //         center + offset_a,
        //         color_0,
        //         color_1,
        //         orientation,
        //         dist_half,
        //         caps,
        //         0.7,
        //     );
        //     add_bond(
        //         entities,
        //         posit_0 + offset_b,
        //         posit_1 + offset_b,
        //         center + offset_b,
        //         color_0,
        //         color_1,
        //         orientation,
        //         dist_half,
        //         caps,
        //         0.4,
        //     );
        // }
        BondCount::Double => {
            // Draw two offset bond cylinders.
            // todo: Set rot_ortho based on dihedral angle.
            let rot_ortho = Quaternion::from_unit_vecs(FWD_VEC, UP_VEC);
            let rotator = rot_ortho * orientation;

            let offset_a = rotator.rotate_vec(Vec3::new(0.15, 0., 0.));
            let offset_b = rotator.rotate_vec(Vec3::new(-0.15, 0., 0.));

            add_bond(
                entities,
                (posit_0 + offset_a, posit_1 + offset_a),
                (color_0, color_1),
                center + offset_a,
                orientation,
                dist_half,
                caps,
                0.5,
                ligand,
            );
            add_bond(
                entities,
                (posit_0 + offset_b, posit_1 + offset_b),
                (color_0, color_1),
                center + offset_b,
                orientation,
                dist_half,
                caps,
                0.5,
                ligand,
            );
        }
        BondCount::Triple => {
            // Draw two offset bond cylinders.
            // todo: Set rot_ortho based on dihedral angle.
            let rot_ortho = Quaternion::from_unit_vecs(FWD_VEC, UP_VEC);
            let rotator = rot_ortho * orientation;

            let offset_a = rotator.rotate_vec(Vec3::new(0.25, 0., 0.));
            let offset_b = rotator.rotate_vec(Vec3::new(-0.25, 0., 0.));

            add_bond(
                entities,
                (posit_0, posit_1),
                (color_0, color_1),
                center,
                orientation,
                dist_half,
                caps,
                0.4,
                ligand,
            );
            add_bond(
                entities,
                (posit_0 + offset_a, posit_1 + offset_a),
                (color_0, color_1),
                center + offset_a,
                orientation,
                dist_half,
                caps,
                0.4,
                ligand,
            );
            add_bond(
                entities,
                (posit_0 + offset_b, posit_1 + offset_b),
                (color_0, color_1),
                center + offset_b,
                orientation,
                dist_half,
                caps,
                0.4,
                ligand,
            );
        }
    }
}

// todo: DRY with/subset of draw_molecule?
pub fn draw_ligand(state: &mut State, scene: &mut Scene) {
    // Hard-coded for sticks for now.

    scene
        .entities
        .retain(|ent| ent.class != EntityType::Ligand as u32);

    let Some(lig) = state.ligand.as_ref() else {
        set_docking_light(scene, None);
        return;
    };

    if state.ui.visibility.hide_ligand {
        return;
    }

    let mol = &lig.molecule;

    if state.ui.show_docking_tools {
        // Add a visual indicator for the docking site.
        scene.entities.push(Entity {
            id: EntityType::Ligand as u32, // todo: A/R
            // todo: High-res spheres are blocking bonds inside them. Likely engine problem.
            mesh: MESH_DOCKING_SITE,
            position: lig.docking_site.site_center.into(),
            scale: lig.docking_site.site_radius as f32,
            color: COLOR_DOCKING_BOX,
            opacity: DOCKING_SITE_OPACITY,
            shinyness: ATOM_SHININESS,
            ..Default::default()
        });
    }

    let mut atoms_positioned = mol.atoms.clone();

    // for atom in &mol.atoms {
    //     scene.entities.push(Entity::new(
    //         MESH_SPHERE,
    //         (atom.posit + ligand.offset).into(),
    //         Quaternion::new_identity(),
    //         BALL_STICK_RADIUS,
    //         atom.element.color(),
    //         ATOM_SHINYNESS,
    //     ));
    // }

    // todo: C+P from draw_molecule. With some removed, but a lot of repeated.
    for (i, bond) in mol.bonds.iter().enumerate() {
        let atom_0 = &mol.atoms[bond.atom_0];
        let atom_1 = &mol.atoms[bond.atom_1];

        if state.ui.visibility.hide_hydrogen
            && (atom_0.element == Element::Hydrogen || atom_1.element == Element::Hydrogen)
        {
            continue;
        }

        let posit_0: Vec3 = lig.atom_posits[bond.atom_0].into();
        let posit_1: Vec3 = lig.atom_posits[bond.atom_1].into();

        let mut color_0 = atom_color(
            atom_0,
            0,
            &mol.residues,
            &Selection::None,
            state.ui.view_sel_level,
            false,
            false,
        );
        let mut color_1 = atom_color(
            atom_1,
            0,
            &mol.residues,
            &Selection::None,
            state.ui.view_sel_level,
            false,
            false,
        );

        color_0 = mod_color_for_ligand(&color_0);
        color_1 = mod_color_for_ligand(&color_1);

        if lig.flexible_bonds.contains(&i) {
            color_0 = LIGAND_COLOR_FLEX;
            color_1 = LIGAND_COLOR_FLEX;
        }

        // Highlight the anchor.
        if bond.atom_0 == lig.anchor_atom {
            color_0 = LIGAND_COLOR_ANCHOR;
        }

        if bond.atom_1 == lig.anchor_atom {
            color_1 = LIGAND_COLOR_ANCHOR;
        }

        bond_entities(
            &mut scene.entities,
            posit_0,
            posit_1,
            color_0,
            color_1,
            bond.bond_type,
            true,
        );
    }

    if !state.ui.visibility.hide_h_bonds {
        for bond in &mol.bonds_hydrogen {
            let atom_donor = &atoms_positioned[bond.donor];
            let atom_acceptor = &atoms_positioned[bond.acceptor];

            let posit_donor: Vec3 = atom_donor.posit.into();

            let posit_acceptor: Vec3 = atom_acceptor.posit.into();

            bond_entities(
                &mut scene.entities,
                posit_donor,
                posit_acceptor,
                COLOR_H_BOND,
                COLOR_H_BOND,
                BondType::Hydrogen,
                true,
            );
        }
    }

    set_docking_light(scene, Some(&state.ligand.as_ref().unwrap().docking_site));
}

/// A visual representation of volumetric electron density,
/// as loaded from .map files or similar.
pub fn draw_density(entities: &mut Vec<Entity>, density: &[ElectronDensity]) {
    entities.retain(|ent| ent.class != EntityType::Density as u32);

    for point in density {
        let mut ent = Entity::new(
            MESH_SPHERE_LOWRES,
            // MESH_CUBE,
            point.coords.into(),
            Quaternion::new_identity(),
            1. * point.density.powf(1.2) as f32,
            // 0.5,
            // (point.density as f32 * 10., 0.0, 1. - point.density as f32),
            (point.density as f32 * 2., 0.0, 0.2),
            // (1., 0.7, 0.5),
            ATOM_SHININESS,
        );
        ent.class = EntityType::Density as u32;

        // ent.opacity =point.density as f32 * 10.;

        entities.push(ent);
    }
}

/// An isosurface of electron density,
/// as loaded from .map files or similar.
pub fn draw_density_surface(entities: &mut Vec<Entity>) {
    entities.retain(|ent| ent.class != EntityType::DensitySurface as u32);

    let mut ent = Entity::new(
        MESH_DENSITY_SURFACE,
        Vec3::new_zero(),
        Quaternion::new_identity(),
        1.,
        (0., 1., 1.),
        ATOM_SHININESS,
    );
    ent.class = EntityType::DensitySurface as u32;
    entities.push(ent);
}

/// Refreshes entities with the model passed.
/// Sensitive to various view configuration parameters.
pub fn draw_molecule(state: &mut State, scene: &mut Scene) {
    let Some(mol) = state.molecule.as_mut() else {
        return;
    };

    scene
        .entities
        .retain(|ent| ent.class != EntityType::Protein as u32);

    let ui = &state.ui;

    let chains_invis: Vec<&Chain> = mol.chains.iter().filter(|c| !c.visible).collect();

    // todo: Figure out how to handle the VDW models A/R.
    // todo: Mesh and/or Surface A/R.
    if ui.mol_view == MoleculeView::Dots {
        if mol.sa_surface_pts.is_none() {
            println!("Starting getting mesh pts...");
            mol.sa_surface_pts = Some(get_mesh_points(&mol.atoms));
            println!("Mesh pts complete.");
        }

        // let mut i = 0;
        for ring in mol.sa_surface_pts.as_ref().unwrap() {
            for sfc_pt in ring {
                let mut entity = Entity::new(
                    MESH_SURFACE_DOT,
                    *sfc_pt,
                    Quaternion::new_identity(),
                    RADIUS_SFC_DOT,
                    COLOR_SFC_DOT,
                    ATOM_SHININESS,
                );
                entity.class = EntityType::Protein as u32;
                scene.entities.push(entity);
            }
            // i += 1;
            // if i > 100 {
            //     break;
            // }
        }
    }

    if ui.mol_view == MoleculeView::Surface {
        if mol.sa_surface_pts.is_none() {
            // todo: DRY with above.
            println!("Starting getting mesh pts...");
            mol.sa_surface_pts = Some(get_mesh_points(&mol.atoms));
            println!("Mesh pts complete.");
        }

        if !mol.mesh_created {
            println!("Building surface mesh...");
            scene.meshes[MESH_SOLVENT_SURFACE] =
                mesh_from_sas_points(mol.sa_surface_pts.as_ref().unwrap());
            mol.mesh_created = true;
            println!("Mesh complete");
        }
    }

    // If sticks view, draw water molecules as balls.
    if ui.mol_view == MoleculeView::Sticks && !state.ui.visibility.hide_water {
        for (i, atom) in mol.atoms.iter().enumerate() {
            if atom.hetero {
                // todo: Excessive nesting.
                if let Some(role) = atom.role {
                    if role == AtomRole::Water {
                        let color_atom = atom_color(
                            atom,
                            i,
                            &mol.residues,
                            &state.selection,
                            state.ui.view_sel_level,
                            false,
                            false,
                        );

                        let mut entity = Entity::new(
                            MESH_WATER_SPHERE,
                            atom.posit.into(),
                            Quaternion::new_identity(),
                            BALL_RADIUS_WATER,
                            color_atom,
                            ATOM_SHININESS,
                        );
                        entity.class = EntityType::Protein as u32;
                        scene.entities.push(entity);
                    }
                }
            }
        }
    }

    // Draw atoms.
    if [MoleculeView::BallAndStick, MoleculeView::SpaceFill].contains(&ui.mol_view) {
        for (i, atom) in mol.atoms.iter().enumerate() {
            if atom.hetero {
                let mut water = false;
                if let Some(role) = atom.role {
                    water = role == AtomRole::Water;
                }
                if !water {
                    // Don't draw VDW spheres for hetero atoms; draw as sticks.
                    continue;
                }
            }

            let mut chain_not_sel = false;
            for chain in &chains_invis {
                if chain.atoms.contains(&i) {
                    chain_not_sel = true;
                    break;
                }
            }
            if chain_not_sel {
                continue;
            }

            if state.ui.visibility.hide_hydrogen && atom.element == Element::Hydrogen {
                continue;
            }

            if let Some(role) = atom.role {
                if state.ui.visibility.hide_sidechains
                    || state.ui.mol_view == MoleculeView::Backbone
                {
                    if matches!(role, AtomRole::Sidechain | AtomRole::H_Sidechain) {
                        continue;
                    }
                }
                if (state.ui.visibility.hide_water || ui.mol_view == MoleculeView::SpaceFill)
                    && role == AtomRole::Water
                {
                    continue;
                }
            }

            if (state.ui.visibility.hide_hetero && atom.hetero)
                || (state.ui.visibility.hide_non_hetero && !atom.hetero)
            {
                continue;
            }

            // We assume only one of near sel, near lig is selectable at a time.
            if ui.show_near_sel_only {
                let atom_sel = mol.get_sel_atom(&state.selection);
                if let Some(a) = atom_sel {
                    if (atom.posit - a.posit).magnitude() as f32 > ui.nearby_dist_thresh as f32 {
                        continue;
                    }
                }
            }
            if let Some(lig) = &state.ligand {
                if ui.show_near_lig_only {
                    let atom_sel = lig.atom_posits[lig.anchor_atom];
                    if (atom.posit - atom_sel).magnitude() as f32 > ui.nearby_dist_thresh as f32 {
                        continue;
                    }
                }
            }

            let (mut radius, mesh) = match ui.mol_view {
                MoleculeView::SpaceFill => (atom.element.vdw_radius(), MESH_SPACEFILL_SPHERE),
                _ => match atom.element {
                    Element::Hydrogen => (BALL_STICK_RADIUS_H, MESH_BALL_STICK_SPHERE),
                    _ => (BALL_STICK_RADIUS, MESH_BALL_STICK_SPHERE),
                },
            };

            if let Some(role) = atom.role {
                if role == AtomRole::Water {
                    radius = BALL_RADIUS_WATER
                }
            }

            let dim_peptide = if state.ligand.is_some() {
                state.ui.visibility.dim_peptide
            } else {
                false
            };

            let color_atom = atom_color(
                atom,
                i,
                &mol.residues,
                &state.selection,
                state.ui.view_sel_level,
                dim_peptide,
                state.ui.res_color_by_index,
            );

            let mut entity = Entity::new(
                mesh,
                atom.posit.into(),
                Quaternion::new_identity(),
                radius,
                color_atom,
                ATOM_SHININESS,
            );
            entity.class = EntityType::Protein as u32;
            scene.entities.push(entity);
        }
    }

    // Draw bonds.
    // if ![MoleculeView::SpaceFill].contains(&ui.mol_view) || atom.hetero {
    for bond in &mol.bonds {
        if ui.mol_view == MoleculeView::Backbone && !bond.is_backbone {
            continue;
        }

        if bond.bond_type == BondType::Hydrogen && ui.visibility.hide_h_bonds {
            continue;
        }

        let atom_0 = &mol.atoms[bond.atom_0];
        let atom_1 = &mol.atoms[bond.atom_1];

        // Don't draw bonds if on the spacefill view, and the atoms aren't hetero.
        if ui.mol_view == MoleculeView::SpaceFill && !atom_0.hetero && !atom_1.hetero {
            continue;
        }

        if ui.show_near_sel_only {
            let atom_sel = mol.get_sel_atom(&state.selection);
            if let Some(a) = atom_sel {
                if (atom_0.posit - a.posit).magnitude() as f32 > ui.nearby_dist_thresh as f32 {
                    continue;
                }
            }
        }
        if let Some(lig) = &state.ligand {
            if ui.show_near_lig_only {
                let atom_sel = lig.atom_posits[lig.anchor_atom];
                if (atom_0.posit - atom_sel).magnitude() as f32 > ui.nearby_dist_thresh as f32 {
                    continue;
                }
            }
        }

        let mut chain_not_sel = false;
        for chain in &chains_invis {
            if chain.atoms.contains(&bond.atom_0) {
                chain_not_sel = true;
                break;
            }
        }
        if chain_not_sel {
            continue;
        }

        if state.ui.visibility.hide_hydrogen
            && (atom_0.element == Element::Hydrogen || atom_1.element == Element::Hydrogen)
        {
            continue;
        }

        // Assuming water won't be bonded to the main molecule.
        if state.ui.visibility.hide_sidechains || state.ui.mol_view == MoleculeView::Backbone {
            if let Some(role_0) = atom_0.role {
                if let Some(role_1) = atom_1.role {
                    if role_0 == AtomRole::Sidechain || role_1 == AtomRole::Sidechain {
                        continue;
                    }
                }
            }
        }

        if (state.ui.visibility.hide_hetero && atom_0.hetero && atom_1.hetero)
            || (state.ui.visibility.hide_non_hetero && !atom_0.hetero && !atom_1.hetero)
        {
            continue;
        }

        let posit_0: Vec3 = atom_0.posit.into();
        let posit_1: Vec3 = atom_1.posit.into();

        let dim_peptide = if state.ligand.is_some() && !&mol.atoms[bond.atom_0].hetero {
            state.ui.visibility.dim_peptide
        } else {
            false
        };

        let color_0 = atom_color(
            atom_0,
            bond.atom_0,
            &mol.residues,
            &state.selection,
            state.ui.view_sel_level,
            dim_peptide,
            state.ui.res_color_by_index,
        );
        let color_1 = atom_color(
            atom_1,
            bond.atom_1,
            &mol.residues,
            &state.selection,
            state.ui.view_sel_level,
            dim_peptide,
            state.ui.res_color_by_index,
        );

        bond_entities(
            &mut scene.entities,
            posit_0,
            posit_1,
            color_0,
            color_1,
            bond.bond_type,
            false,
        );
    }

    // Draw H bonds.
    // todo: DRY with Ligand
    // todo: This incorrectly hides hetero-only H bonds.
    if !state.ui.visibility.hide_h_bonds
        && !state.ui.visibility.hide_non_hetero
        && state.ui.mol_view != MoleculeView::SpaceFill
    {
        for bond in &mol.bonds_hydrogen {
            let atom_donor = &mol.atoms[bond.donor];
            let atom_acceptor = &mol.atoms[bond.acceptor];

            // todo: DRY with above.
            if state.ui.visibility.hide_sidechains || state.ui.mol_view == MoleculeView::Backbone {
                if let Some(role_0) = atom_donor.role {
                    if let Some(role_1) = atom_acceptor.role {
                        if role_0 == AtomRole::Sidechain || role_1 == AtomRole::Sidechain {
                            continue;
                        }
                    }
                }
            }

            // todo: More DRY with cov bonds
            if ui.show_near_sel_only {
                let atom_sel = mol.get_sel_atom(&state.selection);
                if let Some(a) = atom_sel {
                    if (atom_donor.posit - a.posit).magnitude() as f32
                        > ui.nearby_dist_thresh as f32
                    {
                        continue;
                    }
                }
            }
            if let Some(lig) = &state.ligand {
                if ui.show_near_lig_only {
                    let atom_sel = lig.atom_posits[lig.anchor_atom];
                    if (atom_donor.posit - atom_sel).magnitude() as f32
                        > ui.nearby_dist_thresh as f32
                    {
                        continue;
                    }
                }
            }

            let mut chain_not_sel = false;
            for chain in &chains_invis {
                if chain.atoms.contains(&bond.donor) || chain.atoms.contains(&bond.acceptor) {
                    chain_not_sel = true;
                    break;
                }
            }
            if chain_not_sel {
                continue;
            }

            if state.ui.visibility.hide_water {
                if let Some(role) = atom_donor.role {
                    if role == AtomRole::Water {
                        continue;
                    }
                }
                if let Some(role) = atom_acceptor.role {
                    if role == AtomRole::Water {
                        continue;
                    }
                }
            }

            bond_entities(
                &mut scene.entities,
                atom_donor.posit.into(),
                atom_acceptor.posit.into(),
                COLOR_H_BOND,
                COLOR_H_BOND,
                BondType::Hydrogen,
                false,
            );
        }
    }

    if let ControlScheme::Arc { center } = &mut scene.input_settings.control_scheme {
        *center = orbit_center(state);
    }
}
