//! Handles drawing molecules, atoms, bonds, and other items of interest. This
//! adds entities to the scene based on structs.

use std::{fmt, fmt::Display, io, io::ErrorKind, str::FromStr, sync::OnceLock};

use bincode::{Decode, Encode};
use bio_files::{BondType, ResidueType};
use egui::{Color32, FontFamily};
use graphics::{ControlScheme, Entity, Scene, TextOverlay, UP_VEC};
use lin_alg::{
    f32::{Quaternion, Vec3},
    map_linear,
};
use na_seq::Element;

use crate::{
    OperatingMode, ResColoring, Selection, State, StateUi, ViewSelLevel,
    mol_manip::ManipMode,
    molecule::{Atom, AtomRole, Chain, MolGenericRef, MolGenericTrait, MolType, Residue, aa_color},
    reflection::DensityPt,
    render::{
        ATOM_SHININESS, BACKGROUND_COLOR, BALL_RADIUS_WATER_H, BALL_RADIUS_WATER_O,
        BALL_STICK_RADIUS, BALL_STICK_RADIUS_H, BODY_SHINYNESS, Color, MESH_BOND, MESH_CUBE,
        MESH_DENSITY_SURFACE, MESH_SECONDARY_STRUCTURE, MESH_SOLVENT_SURFACE, MESH_SPHERE_HIGHRES,
        MESH_SPHERE_LOWRES, MESH_SPHERE_MEDRES, WATER_BOND_THICKNESS, WATER_OPACITY,
    },
    util::{clear_mol_entity_indices, find_neighbor_posit, orbit_center},
    viridis_lut::VIRIDIS,
};
// const LIGAND_COLOR_ANCHOR: Color = (1., 0., 1.);

const COLOR_MOL_MOVING: Color = (1., 1., 1.);
const COLOR_MOL_ROTATE: Color = (0.65, 1., 0.65);
const COLOR_MD_NEAR_MOL: Color = (0.0, 0., 1.); // Blended into
const BLEND_AMT_MD_NEAR_MOL: f32 = 0.5; // A higher value means it's closer to the special color.

// i.e a flexible bond.
const LIGAND_COLOR_FLEX: Color = (1., 1., 0.);
pub const COLOR_AA_NON_RESIDUE: Color = (0., 0.8, 1.0);
pub const COLOR_AA_NON_RESIDUE_EGUI: Color32 = Color32::from_rgb(0, 204, 255);

pub const COLOR_SELECTED: Color = (1., 0., 0.);
const COLOR_H_BOND: Color = (1., 0.5, 0.1);
const RADIUS_H_BOND: f32 = 0.2; // A scaler relative to covalent sticks.

const COLOR_WATER_BOND: Color = (0.5, 0.5, 0.8);

const COLOR_SFC_DOT: Color = (0.7, 0.7, 0.7);

const LABEL_SIZE_ATOM: f32 = 16.;
const LABEL_SIZE_MOL: f32 = 40.;
const LABEL_COLOR_ATOM: (u8, u8, u8, u8) = (255, 60, 160, 255);
const LABEL_COLOR_ATOM_SEL: (u8, u8, u8, u8) = (255, 20, 20, 255);
const LABEL_COLOR_MOL: (u8, u8, u8, u8) = (255, 120, 150, 255);
const LABEL_COLOR_MOL_SEL: (u8, u8, u8, u8) = (255, 10, 10, 255);

// Hetero residues in protein, so they stand out from the normal protein molecules.
// Lower blend values mean more of the original color.
const COLOR_HETERO_RES: Color = (0.0, 0.0, 1.0);
const BLEND_AMT_HETERO_RES: f32 = 0.4;
const LIGAND_COLOR: Color = (0., 0.4, 1.);
const LIGAND_BLEND_AMT: f32 = 0.4;
const LIPID_COLOR: Color = (1.0, 1.0, 0.);
const LIPID_BLEND_AMT: f32 = 0.3;

const COLOR_DOCKING_BOX: Color = (0.3, 0.3, 0.9);
pub const COLOR_DOCKING_SITE_MESH: Color = (0.5, 0.5, 0.9);
// const DOCKING_SITE_OPACITY: f32 = 0.1;

const COLOR_SA_SURFACE: Color = (0.3, 0.2, 1.);

pub const BOND_RADIUS_BASE: f32 = 0.10; // Absolute unit in Å.

// These ratios scale the base bond radius above.
pub const BOND_RADIUS_LIG_RATIO: f32 = 1.3; // Of bond radius.
pub const BOND_RADIUS_LIG_RATIO_SEL: f32 = 1.6; // Of bond radius.
pub const BOND_RADIUS_RATIO_H: f32 = 0.5; // Of bond radius. Covalent to H.
pub const BOND_RADIUS_LIPID_RATIO: f32 = 0.6; // Of bond radius. Covalent to H.

// Aromatic inner radius, relative to bond radius.
const BOND_RADIUS_AR_INNER_RATIO: f32 = 0.4; // Of bond radius.
const AR_INNER_OFFSET: f32 = 0.3; // Å
// const AR_INNER_SHORTEN_FACTOR: f32 = 0.7;
const AR_SHORTEN_AMT: f32 = 0.4; // Å. Applied to each half.
const DBL_BOND_OFFSET: f32 = 0.1; // Two of these is the separation.
const TRIPLE_BOND_OFFSET: f32 = 0.1; // Two of these is the separation.

pub const SIZE_SFC_DOT: f32 = 0.03;

const DIMMED_PEPTIDE_AMT: f32 = 0.92; // Higher value means more dim.

pub const DENSITY_ISO_OPACITY: f32 = 0.5;
pub const SAS_ISO_OPACITY: f32 = 0.75;

// We use this for mapping partial charge (e.g. as loaded from Amber) to colors.
// This should tightly span the range of expected charges.
// Note that we observe some charges out of this range, but have it narrower
// to show better constrast.
pub const CHARGE_MAP_MIN: f32 = -0.9;
pub const CHARGE_MAP_MAX: f32 = 0.65;

// This allows us to more easily customize sphere mesh resolution.
pub const MESH_BALL_STICK_SPHERE: usize = MESH_SPHERE_MEDRES;
// todo: I believe this causes performance problems on many machines. But looks
// todo much nicer.
pub const MESH_SPACEFILL_SPHERE: usize = MESH_SPHERE_HIGHRES;
pub const MESH_WATER_SPHERE: usize = MESH_SPHERE_MEDRES;
pub const MESH_BOND_CAP: usize = MESH_SPHERE_LOWRES;

// This should ideally be high res, but we experience anomolies on viewing items inside it, while
// the cam is outside.
// const MESH_DOCKING_SITE: usize = MESH_DOCKING_BOX;

// Spheres look slightly better when close, but even our coarsest one leads to performance problems.
const MESH_SURFACE_DOT: usize = MESH_CUBE;

// Cache blend results.
static LIG_C: OnceLock<Color> = OnceLock::new();
static LIG_CL: OnceLock<Color> = OnceLock::new();
static LIG_O: OnceLock<Color> = OnceLock::new();
static LIG_H: OnceLock<Color> = OnceLock::new();
static LIG_N: OnceLock<Color> = OnceLock::new();

// todo: For this and overlay bonds: Make teh atom color based on if the atom is selected;
// todo not if the bond is selected.
fn text_overlay_atoms(
    entity: &mut Entity,
    mol_ident: &str,
    i_atom: usize,
    atom: &Atom,
    sel: bool,
    ui: &StateUi,
) {
    if ui.visibility.labels_atom_sn {
        entity.overlay_text = Some(TextOverlay {
            text: format!("{}", atom.serial_number),
            size: LABEL_SIZE_ATOM,
            color: LABEL_COLOR_ATOM,
            font_family: FontFamily::Proportional,
        });
    }

    let color = if sel {
        LABEL_COLOR_MOL_SEL
    } else {
        LABEL_COLOR_MOL
    };

    if ui.visibility.labels_mol && i_atom == 0 {
        entity.overlay_text = Some(TextOverlay {
            text: format!("{}", mol_ident),
            size: LABEL_SIZE_MOL,
            color,
            font_family: FontFamily::Proportional,
        });
    }
}

fn text_overlay_bonds(
    entity: &mut Entity,
    mol_ident: &str,
    i_atom: usize,
    atom: &Atom,
    sel: bool,
    ui: &StateUi,
) {
    if !matches!(
        ui.mol_view,
        MoleculeView::BallAndStick | MoleculeView::SpaceFill
    ) {
        if ui.visibility.labels_atom_sn {
            entity.overlay_text = Some(TextOverlay {
                text: format!("{}", atom.serial_number),
                size: LABEL_SIZE_ATOM,
                color: LABEL_COLOR_ATOM,
                font_family: FontFamily::Proportional,
            });
        }

        let color = if sel {
            LABEL_COLOR_MOL_SEL
        } else {
            LABEL_COLOR_MOL
        };

        if ui.visibility.labels_mol && i_atom == 0 {
            entity.overlay_text = Some(TextOverlay {
                text: format!("{}", mol_ident),
                size: LABEL_SIZE_MOL,
                color,
                font_family: FontFamily::Proportional,
            });
        }
    }
}

/// We use the Entity's class field to determine which graphics-engine entities to retain and remove.
/// This affects both local drawing logic, and engine-level entity setup.
#[derive(Clone, Copy, PartialEq)]
#[repr(u32)]
pub enum EntityClass {
    Protein = 0,
    Ligand = 1,
    NucleicAcid = 2,
    Lipid = 3,
    DensityPoint = 4,
    DensitySurface = 5,
    SecondaryStructure = 6,
    SaSurface = 7,
    SaSurfaceDots = 8,
    DockingSite = 9,
    WaterModel = 10,
    Other = 11,
}

// todo: For ligands that are flexible, highlight the fleixble bonds in a bright color.

fn blend_color(color_0: Color, color_1: Color, portion: f32) -> Color {
    (
        map_linear(portion, (0., 1.), (color_0.0, color_1.0)),
        map_linear(portion, (0., 1.), (color_0.1, color_1.1)),
        map_linear(portion, (0., 1.), (color_0.2, color_1.2)),
    )
}

fn cache_lig_color(el: Element) -> Option<&'static OnceLock<Color>> {
    match el {
        Element::Carbon => Some(&LIG_C),
        Element::Oxygen => Some(&LIG_O),
        Element::Hydrogen => Some(&LIG_H),
        Element::Nitrogen => Some(&LIG_N),
        Element::Chlorine => Some(&LIG_CL),
        _ => None,
    }
}

/// Make ligands stand out visually, when colored by atom.
fn mod_color_for_ligand(color: &Color, el: Element, color_by_q: bool) -> Color {
    if color_by_q {
        return blend_color(*color, LIGAND_COLOR, LIGAND_BLEND_AMT);
    }

    if let Some(slot) = cache_lig_color(el) {
        slot.get_or_init(|| blend_color(*color, LIGAND_COLOR, LIGAND_BLEND_AMT))
            .clone()
    } else {
        blend_color(*color, LIGAND_COLOR, LIGAND_BLEND_AMT)
    }
}

#[derive(Clone, Copy, PartialEq, Debug, Default, Encode, Decode)]
pub enum MoleculeView {
    Backbone,
    #[default]
    Sticks,
    BallAndStick,
    /// i.e. Van der Waals radius, or CPK.
    SpaceFill,
    Ribbon,
    Surface,
    Dots,
}

impl MoleculeView {
    pub fn next(self) -> Self {
        match self {
            Self::Backbone => Self::Sticks,
            Self::Sticks => Self::BallAndStick,
            Self::BallAndStick => Self::SpaceFill,
            Self::SpaceFill => Self::Surface, // skip ribbon for now
            Self::Ribbon => Self::Surface,
            Self::Surface => Self::Dots,
            Self::Dots => Self::Backbone,
        }
    }

    // todo: repetitive
    pub fn prev(self) -> Self {
        match self {
            Self::Backbone => Self::Dots,
            Self::Sticks => Self::Backbone,
            Self::BallAndStick => Self::Sticks,
            Self::SpaceFill => Self::BallAndStick,
            Self::Ribbon => Self::SpaceFill,
            Self::Surface => Self::SpaceFill,
            Self::Dots => Self::Surface,
        }
    }

    pub fn next_editor(self) -> Self {
        match self {
            Self::Sticks => Self::BallAndStick,
            Self::BallAndStick => Self::SpaceFill,
            Self::SpaceFill => Self::Sticks,
            _ => Self::Sticks,
        }
    }

    // todo: repetitive
    pub fn prev_editor(self) -> Self {
        match self {
            Self::Sticks => Self::SpaceFill,
            Self::BallAndStick => Self::Sticks,
            Self::SpaceFill => Self::BallAndStick,
            _ => Self::Sticks,
        }
    }
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
            "cartoon" | "ribbon" => Ok(MoleculeView::Ribbon),
            "surface" => Ok(MoleculeView::Surface),
            "dots" => Ok(MoleculeView::Dots),
            other => Err(io::Error::new(
                ErrorKind::InvalidData,
                format!("invalid MoleculeView: '{}'", other),
            )),
        }
    }
}

impl Display for MoleculeView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let val = match self {
            Self::Backbone => "Backbone",
            Self::Sticks => "Sticks",
            Self::BallAndStick => "Ball and stick",
            Self::Ribbon => "Ribbon",
            Self::SpaceFill => "Spacefill",
            Self::Surface => "Surface (Van der Waals)",
            Self::Dots => "Dots (Van der Waals)",
        };

        write!(f, "{val}")
    }
}

/// A linear color map using the viridis scheme. Uses a LUT.
pub fn color_viridis(i: usize, min: usize, max: usize) -> Color {
    // Normalize i to [0.0, 1.0]
    let t = if max > min {
        // Compute as f32 and clamp to [0,1]
        let tt = (i.saturating_sub(min) as f32) / ((max - min) as f32);
        tt.clamp(0.0, 1.0)
    } else {
        0.0
    };

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

pub fn color_viridis_float(i: f32, min: f32, max: f32) -> Color {
    const RESOLUTION: usize = 2_048;

    // Normalize i into [0.0, 1.0]
    let t = if max > min {
        ((i - min) / (max - min)).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let idx = (t * (RESOLUTION as f32)).round() as usize;

    color_viridis(idx, 0, RESOLUTION)
}

/// A general function that sets color based criteria like element, partial charge,
/// if selected or not etc.
pub fn atom_color(
    atom: &Atom,
    mol_i: usize,
    i: usize,
    residues: &[Residue],
    aa_count: usize, // # AA residues; used for color-mapping.
    selection: &Selection,
    view_sel_level: ViewSelLevel,
    dimmed: bool,
    res_coloring: ResColoring,
    atom_color_by_q: bool,
    mol_type: MolType,
) -> Color {
    let mut result = match view_sel_level {
        ViewSelLevel::Atom => {
            if atom_color_by_q {
                if let Some(q) = atom.partial_charge {
                    color_viridis_float(q, CHARGE_MAP_MIN, CHARGE_MAP_MAX)
                } else {
                    // Don't revert to atom color, as that could be misinterpreted.
                    (0.5, 0.5, 0.5)
                }
            } else {
                atom.element.color()
            }
        }
        ViewSelLevel::Residue => {
            let mut color = Element::Hydrogen.color(); // todo temp workaround for a bug we haven't tracked down.

            if let Some(res_i) = &atom.residue {
                let res = &residues[*res_i];
                color = match &res.res_type {
                    ResidueType::AminoAcid(aa) => match res_coloring {
                        ResColoring::AminoAcid => aa_color(*aa),
                        ResColoring::Position => match atom.residue {
                            Some(res_i) => color_viridis(res_i, 0, aa_count),
                            None => aa_color(*aa),
                        },
                        ResColoring::Hydrophobicity => {
                            // -4.5 to 4.5
                            // todo: Use hte `hydropathy_doolittle` windowing fn instead?
                            // todo: That may be overkill, or used as a smoothing technique.
                            // These min/maxes are based on possible values of `aa.hydropathicity()`.
                            color_viridis_float(aa.hydropathicity(), -4.5, 4.5)
                        }
                    },
                    _ => COLOR_AA_NON_RESIDUE,
                };

                // Todo: WOrkaround for a problem we're having with Hydrogen's showing like hetero atoms
                // todo in residue mode. Likely due to them not having their AA set.
                // our workaround of setting to 1, 1, 1 is taking effect instead...
                // This sets white vice the residue color, but this may be OK.
                if atom.element == Element::Hydrogen {
                    color = atom.element.color();
                }
            }
            color
        }
        ViewSelLevel::Bond => atom.element.color(), // Handled elsewhere.
    };

    // If selected, the selected color overrides the element or residue color.
    match selection {
        Selection::AtomPeptide(sel_i) => {
            if mol_type == MolType::Peptide && *sel_i == i {
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
        Selection::AtomsPeptide(sel_is) => {
            if sel_is.contains(&i) {
                result = COLOR_SELECTED;
            }
        }
        Selection::AtomLig((i_mol, i_atom)) => {
            if mol_type == MolType::Ligand && *i_atom == i && *i_mol == mol_i {
                result = COLOR_SELECTED;
            }
        }
        Selection::AtomsLig((i_mol, is_atom)) => {
            if mol_type == MolType::Ligand && is_atom.contains(&i) && *i_mol == mol_i {
                result = COLOR_SELECTED;
            }
        }
        Selection::AtomNucleicAcid((lig_i, sel_i)) => {
            if mol_type == MolType::NucleicAcid && *sel_i == i && *lig_i == mol_i {
                result = COLOR_SELECTED;
            }
        }
        Selection::AtomLipid((lig_i, sel_i)) => {
            if mol_type == MolType::Lipid && *sel_i == i && *lig_i == mol_i {
                result = COLOR_SELECTED;
            }
        }
        Selection::BondLig((lig_i, sel_i)) => {
            if mol_type == MolType::Ligand && *sel_i == i && *lig_i == mol_i {
                result = COLOR_SELECTED;
            }
        }
        _ => (), // Other bond types; impl A/R.

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
    posits: (Vec3, Vec3),
    colors: (Color, Color),
    center: Vec3,
    orientation: Quaternion,
    dist_half: f32,
    caps: bool,
    radius_scaler: f32,
    mol_type: MolType,
) -> Vec<Entity> {
    let mut result = Vec::new();

    // Split the bond into two entities, so you can color-code them separately based
    // on which atom the half is closer to.
    let center_0 = (posits.0 + center) / 2.;
    let center_1 = (posits.1 + center) / 2.;

    let entity_type = mol_type.entity_type() as u32;

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
            BOND_RADIUS_BASE * radius_scaler,
            colors.0,
            BODY_SHINYNESS,
        );
        let mut cap_1 = Entity::new(
            MESH_BOND_CAP,
            posits.1,
            Quaternion::new_identity(),
            BOND_RADIUS_BASE * radius_scaler,
            colors.1,
            BODY_SHINYNESS,
        );

        cap_0.class = entity_type;
        cap_1.class = entity_type;
        result.push(cap_0);
        result.push(cap_1);
    }

    let scale = Some(Vec3::new(radius_scaler, dist_half, radius_scaler));
    entity_0.scale_partial = scale;
    entity_1.scale_partial = scale;

    result.push(entity_0);
    result.push(entity_1);

    result
}

pub fn bond_entities(
    posit_0: Vec3,
    posit_1: Vec3,
    mut color_0: Color,
    mut color_1: Color,
    bond_type: BondType,
    mol_type: MolType,
    // No caps for ball and stick
    caps: bool,
    // A Neighbor, in the case of aromataic, double bonds, and triple bonds. We use this to determine how
    // to orient the bond meshes, e.g. in plane with a ring. Second pararm is if the bond is from posit 1,
    // vice posit 0.
    neighbor: (Vec3, bool),
    mol_active: bool,  // i.e. selected
    to_hydrogen: bool, // A covalent bond to H
) -> Vec<Entity> {
    let mut result = Vec::new();

    let center: Vec3 = (posit_0 + posit_1) / 2.;

    let diff = posit_0 - posit_1;
    let diff_unit = diff.to_normalized();
    let orientation = Quaternion::from_unit_vecs(UP_VEC, diff_unit);
    let dist_half = diff.magnitude() / 2.;

    // Dummy not ideal for H bonds semantically, but it's how we currently use it.
    if bond_type == BondType::Dummy {
        color_0 = COLOR_H_BOND;
        color_1 = COLOR_H_BOND;
    }

    let mut radius_scaler = 1.;

    match mol_type {
        MolType::Ligand | MolType::NucleicAcid => {
            if mol_active {
                radius_scaler *= BOND_RADIUS_LIG_RATIO_SEL
            } else {
                radius_scaler *= BOND_RADIUS_LIG_RATIO;
            }
        }
        MolType::Lipid => {
            if mol_active {
                radius_scaler *= BOND_RADIUS_LIG_RATIO_SEL
            } else {
                radius_scaler *= BOND_RADIUS_LIPID_RATIO
            }
        }
        _ => (),
    }

    if to_hydrogen {
        radius_scaler *= BOND_RADIUS_RATIO_H;
    }

    match bond_type {
        // Draw a normal mesh, the same as a single, and a second thinner and shorter inner one.
        BondType::Aromatic => {
            // Compute the dihedral angle so we always place the smaller, offset bond on the inside.

            let (posit_0_inner, posit_1_inner, center_inner, dist_half_inner) = {
                // A vector perpendicular to the plane of the bonds (e.g. the ring)
                // This direction only works in some cases; need a more reliable way.
                // Note: This has problems at the connections to rings, but is works for most aromatic
                // bonds. WHen it fails, it shows the shorter part on the outside.
                let perp_vec = if neighbor.1 {
                    diff.cross(neighbor.0 - posit_1)
                } else {
                    diff.cross(neighbor.0 - posit_0)
                }
                .to_normalized();

                let dir_in = perp_vec.cross(diff.to_normalized()).to_normalized();
                let offset = dir_in * AR_INNER_OFFSET;

                let mut p0 = posit_0 + offset;
                let mut p1 = posit_1 + offset;

                // Make the length shorter.
                let diff = p1 - p0;
                let dist = diff.magnitude();
                let dir = diff / dist;

                let shorten_vec = dir * AR_SHORTEN_AMT;
                p0 += shorten_vec;
                p1 -= shorten_vec;

                (p0, p1, center + offset, (p1 - p0).magnitude() / 2.)
            };

            let thickness_outer = radius_scaler;
            let thickness_inner = radius_scaler * BOND_RADIUS_AR_INNER_RATIO;

            // Primary bond exactly like a normal single bond.
            result.extend(add_bond(
                (posit_0, posit_1),
                (color_0, color_1),
                center,
                orientation,
                dist_half,
                caps,
                thickness_outer,
                mol_type,
            ));

            // Smaller bond on the inside.
            result.extend(add_bond(
                (posit_0_inner, posit_1_inner),
                (color_0, color_1),
                center_inner,
                orientation,
                dist_half_inner,
                caps,
                thickness_inner,
                mol_type,
            ));
        }
        BondType::Double => {
            // Draw two offset bond cylinders.
            // See notes above in the Aromatic section.

            let (offset_a, offset_b) = {
                // The compare doesn't matter here, as it's symmetric.
                let perp_vec = diff.cross(posit_1 - neighbor.0).to_normalized();

                let dir_in = perp_vec.cross(diff.to_normalized()).to_normalized();
                let offset_a = dir_in * DBL_BOND_OFFSET;
                let offset_b = -dir_in * DBL_BOND_OFFSET;

                (offset_a, offset_b)
            };

            result.extend(add_bond(
                (posit_0 + offset_a, posit_1 + offset_a),
                (color_0, color_1),
                center + offset_a,
                orientation,
                dist_half,
                caps,
                0.5,
                mol_type,
            ));

            result.extend(add_bond(
                (posit_0 + offset_b, posit_1 + offset_b),
                (color_0, color_1),
                center + offset_b,
                orientation,
                dist_half,
                caps,
                0.5,
                mol_type,
            ));
        }
        BondType::Triple => {
            // Draw two offset bond cylinders.
            // todo: DRY
            let (offset_a, offset_b) = {
                // The compare doesn't matter here, as it's symmetric.
                let perp_vec = diff.cross(posit_1 - neighbor.0).to_normalized();

                let dir_in = perp_vec.cross(diff.to_normalized()).to_normalized();
                let offset_a = dir_in * TRIPLE_BOND_OFFSET;
                let offset_b = -dir_in * TRIPLE_BOND_OFFSET;

                (offset_a, offset_b)
            };

            result.extend(add_bond(
                (posit_0, posit_1),
                (color_0, color_1),
                center,
                orientation,
                dist_half,
                caps,
                0.4,
                mol_type,
            ));
            result.extend(add_bond(
                (posit_0 + offset_a, posit_1 + offset_a),
                (color_0, color_1),
                center + offset_a,
                orientation,
                dist_half,
                caps,
                0.4,
                mol_type,
            ));
            result.extend(add_bond(
                (posit_0 + offset_b, posit_1 + offset_b),
                (color_0, color_1),
                center + offset_b,
                orientation,
                dist_half,
                caps,
                0.4,
                mol_type,
            ));
        }
        // Single bonds, and others.
        _ => {
            // Hydrogen bond placeholder.
            let thickness = if bond_type == BondType::Dummy {
                RADIUS_H_BOND
            } else {
                radius_scaler
            };

            result.extend(add_bond(
                (posit_0, posit_1),
                (color_0, color_1),
                center,
                orientation,
                dist_half,
                caps,
                thickness,
                mol_type,
            ));
        }
    }

    result
}

/// Water from a MD sim; not from atoms in experimental data.
pub fn draw_water(
    scene: &mut Scene,
    o_pos: &[Vec3],
    h0_pos: &[Vec3],
    h1_pos: &[Vec3],
    hide_water: bool,
) {
    scene
        .entities
        .retain(|ent| ent.class != EntityClass::WaterModel as u32);

    // todo: Borrow mut prob.
    // clear_mol_entity_indices(state);

    if hide_water {
        return;
    }

    for i in 0..o_pos.len() {
        let mut ent = Entity::new(
            MESH_WATER_SPHERE,
            o_pos[i].into(),
            Quaternion::new_identity(),
            BALL_RADIUS_WATER_O,
            Element::Oxygen.color(),
            ATOM_SHININESS,
        );

        ent.opacity = WATER_OPACITY;
        ent.class = EntityClass::WaterModel as u32;
        scene.entities.push(ent);

        for pos in [h0_pos[i], h1_pos[i]].iter() {
            let mut ent = Entity::new(
                MESH_WATER_SPHERE,
                (*pos).into(),
                Quaternion::new_identity(),
                BALL_RADIUS_WATER_H,
                Element::Hydrogen.color(),
                ATOM_SHININESS,
            );

            ent.opacity = WATER_OPACITY;
            ent.class = EntityClass::WaterModel as u32;
            scene.entities.push(ent);
        }

        // Bonds
        for pair in [(o_pos[i], h0_pos[i]), (o_pos[i], h1_pos[i])] {
            let center: Vec3 = ((pair.0 + pair.1) / 2.).into();
            let diff: Vec3 = (pair.0 - pair.1).into();
            let dist = diff.magnitude();

            // This handles the case of atoms in the water molecule split across the periodic boundary
            // condition; don't draw the bond.
            if dist > 5. {
                continue;
            }

            let orientation = Quaternion::from_unit_vecs(UP_VEC, diff.to_normalized());
            let mut ent_bond = Entity::new(
                MESH_BOND,
                center,
                orientation,
                1.,
                COLOR_WATER_BOND,
                BODY_SHINYNESS,
            );
            let scale = Some(Vec3::new(WATER_BOND_THICKNESS, dist, WATER_BOND_THICKNESS));

            ent_bond.opacity = WATER_OPACITY;
            ent_bond.scale_partial = scale;
            ent_bond.class = EntityClass::WaterModel as u32;
            scene.entities.push(ent_bond);
        }
    }
}

/// For all molecule types (for now, not including peptide)
pub fn draw_mol(
    mol: MolGenericRef,
    mol_i: usize,
    ui: &StateUi,
    active_mol: &Option<(MolType, usize)>,
    manip_mode: ManipMode,
    mode: OperatingMode,
) -> Vec<Entity> {
    let mut result = Vec::new();

    // println!("SEL: {:?}", ui.selection);

    if !mol.common().visible {
        return result;
    }

    let mol_active = if let Some((active_mol_type, active_i)) = active_mol {
        mol.mol_type() == *active_mol_type && mol_i == *active_i
    } else {
        false
    };

    // todo: You have problems with transparent objects like the view cube in conjunction
    // todo with the transparent surface; workaround to not draw the cube here.
    if ui.show_docking_tools && ui.mol_view != MoleculeView::Surface {
        // Add a visual indicator for the docking site.

        // scene.entities.push(Entity {
        //     class: EntityType::DockingSite as u32,
        //     // todo: High-res spheres are blocking bonds inside them. Likely engine problem.
        //     mesh: MESH_DOCKING_SITE,
        //     position: lig.docking_site.site_center.into(),
        //     scale: lig.docking_site.site_radius as f32,
        //     color: COLOR_DOCKING_BOX,
        //     opacity: DOCKING_SITE_OPACITY,
        //     shinyness: ATOM_SHININESS,
        //     ..Default::default()
        // });
    }

    let sel = if ui.selection.is_bond() {
        &Selection::None
    } else {
        &ui.selection
    };

    if matches!(
        ui.mol_view,
        MoleculeView::BallAndStick | MoleculeView::SpaceFill
    ) {
        for (i_atom, atom) in mol.common().atoms.iter().enumerate() {
            // if mode == OperatingMode::Primary && ui.mol_view == MoleculeView::SpaceFill {
            //     break;
            // }

            if ui.visibility.hide_hydrogen && atom.element == Element::Hydrogen {
                continue;
            }

            let mut color = (0., 0., 0.);
            let mut manip_active = false;

            match manip_mode {
                ManipMode::Move((mol_type, i)) => match mode {
                    OperatingMode::Primary => {
                        if mol_type == mol.mol_type() && i == mol_i {
                            color = COLOR_MOL_MOVING;
                            manip_active = true;
                        }
                    }
                    OperatingMode::MolEditor => {
                        if i == i_atom {
                            color = COLOR_MOL_MOVING;
                            manip_active = true;
                        }
                    }
                },
                ManipMode::Rotate((mol_type, i)) => {
                    if mol_type == mol.mol_type() && i == mol_i {
                        color = COLOR_MOL_ROTATE;
                        manip_active = true;
                    }
                }
                ManipMode::None => (),
            }

            if !manip_active {
                color = atom_color(
                    atom,
                    mol_i,
                    i_atom,
                    &[],
                    0,
                    sel,
                    ViewSelLevel::Atom, // Always color lipids by atom.
                    false,
                    ui.res_coloring,
                    ui.atom_color_by_charge,
                    mol.mol_type(),
                );

                if color != COLOR_SELECTED {
                    match mol.mol_type() {
                        MolType::Ligand => {
                            if mode == OperatingMode::Primary {
                                color = mod_color_for_ligand(
                                    &color,
                                    atom.element,
                                    ui.atom_color_by_charge,
                                )
                            }
                        }
                        // todo: Lipid and NA caches A/R
                        // todo: Color for NA
                        MolType::NucleicAcid => {
                            color = blend_color(color, LIPID_COLOR, LIPID_BLEND_AMT)
                        }
                        MolType::Lipid => color = blend_color(color, LIPID_COLOR, LIPID_BLEND_AMT),
                        _ => (),
                    }
                }
            }

            let (radius, mesh) = match ui.mol_view {
                MoleculeView::SpaceFill => (atom.element.vdw_radius(), MESH_SPACEFILL_SPHERE),
                _ => match atom.element {
                    Element::Hydrogen => (BALL_STICK_RADIUS_H, MESH_BALL_STICK_SPHERE),
                    _ => (BALL_STICK_RADIUS, MESH_BALL_STICK_SPHERE),
                },
            };

            let mut entity = Entity::new(
                mesh,
                mol.common().atom_posits[i_atom].into(),
                Quaternion::new_identity(),
                radius,
                color,
                ATOM_SHININESS,
            );

            if mode != OperatingMode::MolEditor {
                // Note: We draw these on the bond entities if not in a view that shows atoms.
                text_overlay_atoms(
                    &mut entity,
                    &mol.common().ident,
                    i_atom,
                    &atom,
                    mol_active,
                    ui,
                );
            }

            entity.class = mol.mol_type().entity_type() as u32;
            result.push(entity);
        }
    }

    // todo: C+P from draw_molecule. With some removed, but much repeated.
    for (i_bond, bond) in mol.common().bonds.iter().enumerate() {
        let atom_0 = &mol.common().atoms[bond.atom_0];
        let atom_1 = &mol.common().atoms[bond.atom_1];

        if ui.visibility.hide_hydrogen
            && (atom_0.element == Element::Hydrogen || atom_1.element == Element::Hydrogen)
        {
            continue;
        }

        let posit_0: Vec3 = mol.common().atom_posits[bond.atom_0].into();
        let posit_1: Vec3 = mol.common().atom_posits[bond.atom_1].into();

        // For determining how to orient multiple-bonds. Only run for relevant bonds to save
        // computation.
        let neighbor_posit = match bond.bond_type {
            BondType::Aromatic | BondType::Double | BondType::Triple => {
                let mut hydrogen_is = Vec::with_capacity(mol.common().atoms.len());
                for atom in &mol.common().atoms {
                    hydrogen_is.push(atom.element == Element::Hydrogen);
                }

                let neighbor_i = find_neighbor_posit(
                    &mol.common().adjacency_list,
                    bond.atom_0,
                    bond.atom_1,
                    &hydrogen_is,
                );
                match neighbor_i {
                    Some((i, p1)) => (mol.common().atom_posits[i].into(), p1),
                    None => (mol.common().atom_posits[0].into(), false),
                }
            }
            _ => (Vec3::new_zero(), false),
        };

        let mut color_0 = (0., 0., 0.);
        let mut color_1 = (0., 0., 0.);

        let mut manip_active = false;

        match manip_mode {
            ManipMode::Move((mol_type, i)) => match mode {
                OperatingMode::Primary => {
                    if mol_type == mol.mol_type() && i == mol_i {
                        color_0 = COLOR_MOL_MOVING;
                        color_1 = COLOR_MOL_MOVING;
                        manip_active = true;
                    }
                }
                OperatingMode::MolEditor => {
                    if i == bond.atom_0 {
                        // todo: You may need to clarify manip_1 active manip_0 active or similar,
                        // todo: otherwise the other bond half will not be colored by atom etc.
                        color_0 = COLOR_MOL_MOVING;
                        manip_active = true;
                    }
                    if i == bond.atom_1 {
                        color_1 = COLOR_MOL_MOVING;
                        manip_active = true;
                    }
                }
            },
            ManipMode::Rotate((mol_type, i)) => {
                if mol_type == mol.mol_type() && i == mol_i {
                    color_0 = COLOR_MOL_ROTATE;
                    color_1 = COLOR_MOL_ROTATE;
                    manip_active = true;
                }
            }
            ManipMode::None => (),
        }

        if !manip_active {
            color_0 = atom_color(
                atom_0,
                mol_i,
                bond.atom_0,
                &[],
                0,
                sel,                // ignores bond coloring by adjacent atom if in bond sel mode.
                ViewSelLevel::Atom, // Always color ligands by atom.
                false,
                ui.res_coloring,
                ui.atom_color_by_charge,
                mol.mol_type(),
            );
            color_1 = atom_color(
                atom_1,
                mol_i,
                bond.atom_1,
                &[],
                0,
                sel,                // ignores bond coloring by adjacent atom if in bond sel mode.
                ViewSelLevel::Atom, // Always color ligands by atom.
                false,
                ui.res_coloring,
                ui.atom_color_by_charge,
                mol.mol_type(),
            );

            // If in atom sel mode, we color bonds normally above (The  half of each bond connected
            // to the selected atom). If in bond sel mode, we color the bond between two atoms below.

            match &ui.selection {
                Selection::BondLig((mol_i, bond_i))
                | Selection::BondNucleicAcid((mol_i, bond_i))
                | Selection::BondLipid((mol_i, bond_i)) => {
                    if *bond_i == i_bond {
                        color_0 = COLOR_SELECTED;
                        color_1 = COLOR_SELECTED;
                    }
                }
                Selection::BondsLig((mol_i, bonds_i)) => {
                    if bonds_i.contains(&i_bond) {
                        color_0 = COLOR_SELECTED;
                        color_1 = COLOR_SELECTED;
                    }
                }
                _ => (),
            };

            if color_0 != COLOR_SELECTED {
                match mol.mol_type() {
                    MolType::Ligand => {
                        if mode == OperatingMode::Primary {
                            color_0 = mod_color_for_ligand(
                                &color_0,
                                atom_0.element,
                                ui.atom_color_by_charge,
                            )
                        }
                    }
                    // todo: Color for NA
                    MolType::NucleicAcid => {
                        color_0 = blend_color(color_0, LIPID_COLOR, LIPID_BLEND_AMT)
                    }
                    MolType::Lipid => color_0 = blend_color(color_0, LIPID_COLOR, LIPID_BLEND_AMT),
                    _ => (),
                }
            }
            if color_1 != COLOR_SELECTED {
                match mol.mol_type() {
                    MolType::Ligand => {
                        if mode == OperatingMode::Primary {
                            color_1 = mod_color_for_ligand(
                                &color_1,
                                atom_1.element,
                                ui.atom_color_by_charge,
                            )
                        }
                    }
                    // todo: Color for NA
                    MolType::NucleicAcid => {
                        color_1 = blend_color(color_1, LIPID_COLOR, LIPID_BLEND_AMT)
                    }
                    MolType::Lipid => color_1 = blend_color(color_1, LIPID_COLOR, LIPID_BLEND_AMT),
                    _ => (),
                }
            }
        }

        let to_hydrogen =
            atom_0.element == Element::Hydrogen || atom_1.element == Element::Hydrogen;

        let mut entities = bond_entities(
            posit_0,
            posit_1,
            color_0,
            color_1,
            bond.bond_type,
            mol.mol_type(),
            true,
            neighbor_posit,
            mol_active,
            to_hydrogen,
        );

        // todo: This seems to be related to a bug where atom labels are doubled for some in sticks mode.
        // Draw atom-based labels on bonds if not in a view mode that shows atoms.
        if !entities.is_empty() && mode != OperatingMode::MolEditor {
            text_overlay_bonds(
                &mut entities[0],
                &mol.common().ident,
                bond.atom_0,
                &atom_0,
                mol_active, // todo
                ui,
            );
        }

        result.extend(entities);
    }

    // todo: Add back if you include lig H bonds.
    // if !state.ui.visibility.hide_h_bonds {
    //     for bond in &mol.bonds_hydrogen {
    //         let atom_donor = &atoms_positioned[bond.donor];
    //         let atom_acceptor = &atoms_positioned[bond.acceptor];
    //
    //         let posit_donor: Vec3 = atom_donor.posit.into();
    //
    //         let posit_acceptor: Vec3 = atom_acceptor.posit.into();
    //
    //         bond_entities(
    //             &mut scene.entities,
    //             posit_donor,
    //             posit_acceptor,
    //             COLOR_H_BOND,
    //             COLOR_H_BOND,
    //             BondType::Dummy,
    //             MolType::Ligand,
    //             true,
    //             (Vec3::new_zero(), false),
    //         );
    //     }
    // }

    // set_docking_light(scene, Some(&state.ligand.as_ref().unwrap().docking_site));

    result
}

/// A visual representation of volumetric electron density,
/// as loaded from .map files or similar. This is our point-based approach; not the isosurface.
/// We change size based on density, and not linearly, for visual effect.
pub fn draw_density_point_cloud(entities: &mut Vec<Entity>, density: &[DensityPt]) {
    entities.retain(|ent| ent.class != EntityClass::DensityPoint as u32);
    // clear_mol_entity_indices(state); // todo: Borrow mut problem.

    const EPS: f64 = 0.0000001;

    for point in density {
        // For example, points we filter out for not being near the atoms; we set them to 0 density,
        // vice omitting them. Skipping them here makes rendering more efficient.
        if point.density.abs() < EPS {
            continue;
        }

        // Todo: Sort out how you'll handle this. Currently, You discard these, or they'd go NaN
        // todo on the power computation.
        if point.density < 0.0 {
            continue;
        }

        let mut ent = Entity::new(
            MESH_SPHERE_LOWRES,
            point.coords.into(),
            Quaternion::new_identity(),
            0.03 * point.density.powf(1.3) as f32,
            (point.density as f32 * 2., 0.0, 0.2),
            ATOM_SHININESS,
        );
        ent.class = EntityClass::DensityPoint as u32;

        entities.push(ent);
    }
}

/// An isosurface of electron density,
/// as loaded from .map files or similar.
pub fn draw_density_surface(entities: &mut Vec<Entity>, state: &mut State) {
    entities.retain(|ent| ent.class != EntityClass::DensitySurface as u32);
    clear_mol_entity_indices(state, None);

    let mut ent = Entity::new(
        MESH_DENSITY_SURFACE,
        Vec3::new_zero(),
        Quaternion::new_identity(),
        1.,
        (0., 1., 1.),
        ATOM_SHININESS,
    );
    ent.class = EntityClass::DensitySurface as u32;
    ent.opacity = DENSITY_ISO_OPACITY;
    entities.push(ent);
}

/// The dots view of solvent-accessible-surface
fn draw_dots(update_mesh: &mut bool, mesh_created: bool, scene: &mut Scene) {
    // If the mesh is the default cube, build it. (On demand.)
    if !mesh_created {
        *update_mesh = true;
        return;
    }

    if scene.meshes[MESH_SOLVENT_SURFACE].vertices.len() > 1_000_000 {
        eprintln!("Not drawing dots due to a large-mol rendering problem.");
        return;
    }

    for vertex in &scene.meshes[MESH_SOLVENT_SURFACE].vertices {
        let mut entity = Entity::new(
            MESH_SURFACE_DOT,
            Vec3::from_slice(&vertex.position).unwrap(),
            Quaternion::new_identity(),
            SIZE_SFC_DOT,
            COLOR_SFC_DOT,
            ATOM_SHININESS,
        );
        entity.class = EntityClass::SaSurfaceDots as u32;
        scene.entities.push(entity);
    }
}

/// The mesh view of solvent-accessible-surface
fn draw_sa_surface(update_mesh: &mut bool, mesh_created: bool, scene: &mut Scene) {
    // If the mesh is the default cube, build it. (On demand.)
    if !mesh_created {
        *update_mesh = true;
        return;
    }

    let mut ent = Entity::new(
        MESH_SOLVENT_SURFACE,
        Vec3::new_zero(),
        Quaternion::new_identity(),
        1.,
        COLOR_SA_SURFACE,
        ATOM_SHININESS,
    );
    ent.class = EntityClass::SaSurface as u32;
    ent.opacity = SAS_ISO_OPACITY;
    scene.entities.push(ent);
}

/// Secondary structure, e.g. cartoon.
// pub fn draw_secondary_structure(update_mesh: &mut bool, mesh_created: bool, scene: &mut Scene, state: &mut State) {
pub fn draw_secondary_structure(update_mesh: &mut bool, mesh_created: bool, scene: &mut Scene) {
    // If the mesh is the default cube, build it. (On demand.)
    if !mesh_created {
        *update_mesh = true;
        return;
    }

    scene
        .entities
        .retain(|ent| ent.class != EntityClass::SecondaryStructure as u32);
    // clear_mol_entity_indices(state);

    let mut ent = Entity::new(
        MESH_SECONDARY_STRUCTURE,
        Vec3::new_zero(),
        Quaternion::new_identity(),
        1.,
        (0.7, 0.2, 1.), // todo: Make this customizable etc.
        ATOM_SHININESS,
    );
    ent.class = EntityClass::SecondaryStructure as u32;
    scene.entities.push(ent);
}

// /// Helper
// fn get_atom_posit<'a>(
//     mode: PeptideAtomPosits,
//     posits: Option<&'a Vec<Vec3F64>>,
//     i: usize,
//     atom: &'a Atom,
// ) -> &'a Vec3F64 {
//     match mode {
//         PeptideAtomPosits::Original => &atom.posit,
//         PeptideAtomPosits::Dynamics => match posits {
//             Some(p) => &p[i],
//             None => &atom.posit,
//         },
//     }
// }

/// Refreshes entities with the model passed.
/// Sensitive to various view configuration parameters.
pub fn draw_peptide(state: &mut State, scene: &mut Scene) {
    // todo: You may wish to integrate Cartoon into this workflow.
    let initial_ent_count = scene.entities.len();

    let mol_i = 0; // todo for now.

    let mol_active = if let Some((active_mol_type, active_i)) = state.volatile.active_mol {
        MolType::Peptide == active_mol_type && mol_i == active_i
    } else {
        false
    };

    scene.entities.retain(|ent| {
        ent.class != EntityClass::Protein as u32
            && ent.class != EntityClass::SaSurface as u32
            && ent.class != EntityClass::SaSurfaceDots as u32
    });

    // Edit small molecules only; not proteins.
    if state.volatile.operating_mode == OperatingMode::MolEditor {
        return;
    }

    let Some(mol) = state.peptide.as_ref() else {
        return;
    };

    let start_i = scene.entities.len();
    let mut entities = Vec::new();

    // todo:  Unless colored by res #, set to 0 to save teh computation.
    let aa_count = mol
        .residues
        .iter()
        .filter(|r| {
            if let ResidueType::AminoAcid(_) = r.res_type {
                true
            } else {
                false
            }
        })
        .count();

    let ui = &state.ui;

    if ui.mol_view == MoleculeView::Ribbon {
        draw_secondary_structure(
            &mut state.volatile.flags.update_ss_mesh,
            state.volatile.flags.ss_mesh_created,
            scene,
        );
    }

    // Note that this renders over a sticks model.
    if !state.ui.visibility.hide_protein {
        if ui.mol_view == MoleculeView::Dots {
            draw_dots(
                &mut state.volatile.flags.update_sas_mesh,
                state.volatile.flags.sas_mesh_created,
                scene,
            );
        }
    }

    // todo: Consider if you handle this here, or in a sep fn.
    if !state.ui.visibility.hide_protein {
        if ui.mol_view == MoleculeView::Surface {
            draw_sa_surface(
                &mut state.volatile.flags.update_sas_mesh,
                state.volatile.flags.sas_mesh_created,
                scene,
            );
        }
    }

    let chains_invis: Vec<&Chain> = mol.chains.iter().filter(|c| !c.visible).collect();

    let sel = if ui.selection.is_bond() {
        &Selection::None
    } else {
        &ui.selection
    };

    // If sticks view, draw water molecules as balls.
    if matches!(
        ui.mol_view,
        MoleculeView::Sticks | MoleculeView::BallAndStick
    ) && !state.ui.visibility.hide_water
    {
        for (i_atom, atom) in mol.common.atoms.iter().enumerate() {
            if atom.hetero {
                // todo: Excessive nesting.
                if let Some(role) = atom.role {
                    if role == AtomRole::Water {
                        let color_atom = atom_color(
                            atom,
                            0,
                            i_atom,
                            &mol.residues,
                            aa_count,
                            sel,
                            state.ui.view_sel_level,
                            false,
                            ResColoring::default(),
                            false,
                            MolType::Peptide,
                        );

                        let mut entity = Entity::new(
                            MESH_WATER_SPHERE,
                            mol.common.atom_posits[i_atom].into(),
                            Quaternion::new_identity(),
                            BALL_RADIUS_WATER_O,
                            color_atom,
                            ATOM_SHININESS,
                        );

                        entity.class = EntityClass::Protein as u32;
                        entities.push(entity);
                    }
                }
            }
        }
    }

    // Draw atoms.
    if matches!(
        ui.mol_view,
        MoleculeView::BallAndStick | MoleculeView::SpaceFill
    ) {
        for (i_atom, atom) in mol.common.atoms.iter().enumerate() {
            // let atom_posit = get_atom_posit(
            //     state.ui.peptide_atom_posits,
            //     Some(&mol.common.atom_posits),
            //     i,
            //     atom,
            // );

            if atom.hetero {
                let mut water = false;
                if let Some(role) = atom.role {
                    water = role == AtomRole::Water;
                }
                if !water && ui.mol_view == MoleculeView::SpaceFill {
                    // Don't draw VDW spheres for hetero atoms; draw as sticks.
                    continue;
                }
            }

            let mut chain_not_sel = false;
            for chain in &chains_invis {
                if chain.atoms.contains(&i_atom) {
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
                || (state.ui.visibility.hide_protein && !atom.hetero)
            {
                continue;
            }

            let atom_posit = mol.common.atom_posits[i_atom];

            // We assume only one of near sel, near lig is selectable at a time.
            if ui.show_near_sel_only {
                let atom_sel = mol.get_sel_atom(&state.ui.selection);
                if let Some(a) = atom_sel {
                    // todo: This will fail after moves and dynamics. You mmust pick the selected atom
                    // todo posit correctly!

                    if (atom_posit - a.posit).magnitude() as f32 > ui.nearby_dist_thresh as f32 {
                        continue;
                    }
                }
            }
            if let Some(mol_) = state.active_mol() {
                if ui.show_near_lig_only {
                    let atom_sel = mol_.common().atom_posits[0];
                    if (atom_posit - atom_sel).magnitude() as f32 > ui.nearby_dist_thresh as f32 {
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
                    radius = BALL_RADIUS_WATER_O
                }
            }

            let dim_peptide = state.ui.visibility.dim_peptide && !atom.hetero;

            let mut color_atom = atom_color(
                atom,
                0,
                i_atom,
                &mol.residues,
                aa_count,
                sel,
                state.ui.view_sel_level,
                dim_peptide,
                state.ui.res_coloring,
                state.ui.atom_color_by_charge,
                MolType::Peptide,
            );

            if atom.hetero && color_atom != COLOR_SELECTED {
                color_atom = blend_color(color_atom, COLOR_HETERO_RES, BLEND_AMT_HETERO_RES);
            }

            if state.mol_dynamics.is_some()
                && state.ui.md.peptide_only_near_ligs
                && mol.common.selected_for_md
                && state
                    .ligands
                    .iter()
                    .filter(|l| l.common.selected_for_md)
                    .count()
                    != 0
            {
                if state.volatile.md_peptide_selected.contains(&(0, i_atom)) {
                    color_atom = blend_color(color_atom, COLOR_MD_NEAR_MOL, BLEND_AMT_MD_NEAR_MOL);
                }
            }

            let mut entity = Entity::new(
                mesh,
                atom_posit.into(),
                Quaternion::new_identity(),
                radius,
                color_atom,
                ATOM_SHININESS,
            );

            // Note: We draw these on the bond entities if not in a view that shows atoms.
            text_overlay_atoms(
                &mut entity,
                &mol.common.ident,
                i_atom,
                &atom,
                mol_active,
                ui,
            );

            entity.class = EntityClass::Protein as u32;
            entities.push(entity);
        }
    }

    // For determining inside of rings.
    let mut hydrogen_is = Vec::with_capacity(mol.common.atoms.len());
    for atom in &mol.common.atoms {
        hydrogen_is.push(atom.element == Element::Hydrogen);
    }

    // Draw bonds.
    for (i_bond, bond) in mol.common.bonds.iter().enumerate() {
        if ui.mol_view == MoleculeView::Backbone && !bond.is_backbone {
            continue;
        }

        // Hmm. This could get things confused with other dummy bonds.
        if bond.bond_type == BondType::Dummy && ui.visibility.hide_h_bonds {
            continue;
        }

        let atom_0 = &mol.common.atoms[bond.atom_0];
        let atom_1 = &mol.common.atoms[bond.atom_1];

        let atom_0_posit = mol.common.atom_posits[bond.atom_0];
        let atom_1_posit = mol.common.atom_posits[bond.atom_1];

        // Don't draw bonds if on the spacefill view, and the atoms aren't hetero.
        if ui.mol_view == MoleculeView::SpaceFill && !atom_0.hetero && !atom_1.hetero {
            continue;
        }

        if ui.show_near_sel_only {
            let atom_sel = mol.get_sel_atom(&state.ui.selection);
            if let Some(a) = atom_sel {
                // todo: See note above: You must get teh selected atom posit correctly.
                if (atom_0_posit - a.posit).magnitude() as f32 > ui.nearby_dist_thresh as f32 {
                    continue;
                }
            }
        }
        if let Some(mol_) = state.active_mol() {
            if ui.show_near_lig_only {
                let atom_sel = mol_.common().atom_posits[0];
                if (atom_0_posit - atom_sel).magnitude() as f32 > ui.nearby_dist_thresh as f32 {
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
            || (state.ui.visibility.hide_protein && !atom_0.hetero && !atom_1.hetero)
        {
            continue;
        }

        let posit_0: Vec3 = atom_0_posit.into();
        let posit_1: Vec3 = atom_1_posit.into();

        // For determining how to orient multiple-bonds.
        let neighbor_i = find_neighbor_posit(
            &mol.common.adjacency_list,
            bond.atom_0,
            bond.atom_1,
            &hydrogen_is,
        );
        let neighbor_posit = match neighbor_i {
            Some((i, p1)) => (mol.common.atoms[i].posit.into(), p1),
            None => (mol.common.atoms[0].posit.into(), false),
        };

        let dim_peptide_0 =
            state.ui.visibility.dim_peptide && !mol.common.atoms[bond.atom_0].hetero;
        let dim_peptide_1 =
            state.ui.visibility.dim_peptide && !mol.common.atoms[bond.atom_1].hetero;

        let mut color_0 = atom_color(
            atom_0,
            0,
            bond.atom_0,
            &mol.residues,
            aa_count,
            sel,
            state.ui.view_sel_level,
            dim_peptide_0,
            state.ui.res_coloring,
            state.ui.atom_color_by_charge,
            MolType::Peptide,
        );
        let mut color_1 = atom_color(
            atom_1,
            0,
            bond.atom_1,
            &mol.residues,
            aa_count,
            sel,
            state.ui.view_sel_level,
            dim_peptide_1,
            state.ui.res_coloring,
            state.ui.atom_color_by_charge,
            MolType::Peptide,
        );

        if let Selection::BondPeptide(bond_i) = ui.selection {
            if bond_i == i_bond {
                color_0 = COLOR_SELECTED;
                color_1 = COLOR_SELECTED;
            }
        }

        if atom_0.hetero && color_0 != COLOR_SELECTED {
            color_0 = blend_color(color_0, COLOR_HETERO_RES, BLEND_AMT_HETERO_RES);
        }

        if atom_1.hetero && color_1 != COLOR_SELECTED {
            color_1 = blend_color(color_1, COLOR_HETERO_RES, BLEND_AMT_HETERO_RES);
        }

        if state.mol_dynamics.is_some()
            && state.ui.md.peptide_only_near_ligs
            && mol.common.selected_for_md
            && state
                .ligands
                .iter()
                .filter(|l| l.common.selected_for_md)
                .count()
                != 0
        {
            if state
                .volatile
                .md_peptide_selected
                .contains(&(0, bond.atom_0))
            {
                color_0 = blend_color(color_0, COLOR_MD_NEAR_MOL, BLEND_AMT_MD_NEAR_MOL);
            }
            if state
                .volatile
                .md_peptide_selected
                .contains(&(0, bond.atom_1))
            {
                color_1 = blend_color(color_1, COLOR_MD_NEAR_MOL, BLEND_AMT_MD_NEAR_MOL);
            }
        }

        let to_hydrogen =
            atom_0.element == Element::Hydrogen || atom_1.element == Element::Hydrogen;

        let mut ents_new = bond_entities(
            posit_0,
            posit_1,
            color_0,
            color_1,
            bond.bond_type,
            MolType::Peptide,
            state.ui.mol_view != MoleculeView::BallAndStick,
            neighbor_posit,
            false,
            to_hydrogen,
        );

        if !ents_new.is_empty() {
            text_overlay_bonds(
                &mut ents_new[0],
                &mol.common.ident,
                bond.atom_0,
                &atom_0,
                mol_active,
                ui,
            );
        }

        entities.extend(ents_new);
    }

    // Draw H bonds.
    // todo: DRY with Ligand
    // todo: This incorrectly hides hetero-only H bonds.
    if !state.ui.visibility.hide_h_bonds
        && !state.ui.visibility.hide_protein
        && state.ui.mol_view != MoleculeView::SpaceFill
    {
        for bond in &mol.bonds_hydrogen {
            let atom_donor = &mol.common.atoms[bond.donor];
            let atom_acceptor = &mol.common.atoms[bond.acceptor];

            // todo: DRY with above.
            if state.ui.visibility.hide_sidechains || state.ui.mol_view == MoleculeView::Backbone {
                if let Some(role_0) = atom_donor.role
                    && let Some(role_1) = atom_acceptor.role
                    && (role_0 == AtomRole::Sidechain || role_1 == AtomRole::Sidechain)
                {
                    continue;
                }
            }

            // todo: More DRY with cov bonds
            if ui.show_near_sel_only {
                let atom_sel = mol.get_sel_atom(&state.ui.selection);
                if let Some(a) = atom_sel
                    && (atom_donor.posit - a.posit).magnitude() as f32
                        > ui.nearby_dist_thresh as f32
                {
                    continue;
                }
            }
            if let Some(mol_) = state.active_mol() {
                if ui.show_near_lig_only {
                    let atom_sel = mol_.common().atom_posits[0];
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
                if let Some(role) = atom_donor.role
                    && role == AtomRole::Water
                {
                    continue;
                }
                if let Some(role) = atom_acceptor.role
                    && role == AtomRole::Water
                {
                    continue;
                }
            }

            entities.extend(bond_entities(
                atom_donor.posit.into(),
                atom_acceptor.posit.into(),
                COLOR_H_BOND,
                COLOR_H_BOND,
                BondType::Dummy,
                MolType::Peptide,
                state.ui.mol_view != MoleculeView::BallAndStick,
                (Vec3::new_zero(), false), // N/A
                false,
                false,
            ));
        }
    }

    if let ControlScheme::Arc { center } = &mut scene.input_settings.control_scheme {
        *center = orbit_center(state);
    }

    scene.entities.extend(entities);

    let end_i = scene.entities.len();
    if let Some(mol) = state.peptide.as_mut() {
        mol.common.entity_i_range = Some((start_i, end_i));
    } else {
        eprintln!("Uhoh!")
    }

    if scene.entities.len() != initial_ent_count {
        clear_mol_entity_indices(state, None);
    }
}
