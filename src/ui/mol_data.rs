//! Information and settings for the opened, or to-be opened molecules.

use bio_apis::{drugbank, lmsd, pdbe, pubchem, rcsb};
use bio_files::{ResidueType, md_params::ForceFieldParams};
use dynamics::params::FfParamSet;
use egui::{Color32, RichText, Ui};
use graphics::{ControlScheme, EngineUpdates, EntityUpdate, Scene};
use lin_alg::f64::Vec3;

use crate::{
    ManipMode, Selection, State,
    cam_misc::move_mol_to_cam,
    download_mols, drawing,
    drawing::{CHARGE_MAP_MAX, CHARGE_MAP_MIN, COLOR_AA_NON_RESIDUE_EGUI},
    lipid::MoleculeLipid,
    mol_lig::MoleculeSmall,
    mol_manip::set_manip,
    molecule::{Atom, Bond, MoGenericRefMut, MolGenericRef, MolIdent, MolType, Residue, aa_color},
    nucleic_acid::MoleculeNucleicAcid,
    ui::{
        COL_SPACING, COLOR_ACTIVE, COLOR_ACTIVE_RADIO, COLOR_HIGHLIGHT, COLOR_INACTIVE,
        cam::move_cam_to_active_mol, mol_descrip,
    },
    util::{
        handle_err, handle_success, make_egui_color, make_lig_from_res, move_mol_to_res,
        orbit_center, reset_orbit_center,
    },
};

/// `posit_override` is for example, relative atom positions, such as a positioned ligand.
fn disp_atom_data(atom: &Atom, residues: &[Residue], posit_override: Option<Vec3>, ui: &mut Ui) {
    let role = match atom.role {
        Some(r) => format!("Role: {r}"),
        None => String::new(),
    };

    // Similar to `Vec3`'s format impl, but with fewer digits.
    let posit = match posit_override {
        Some(p) => p,
        None => atom.posit,
    };

    let posit_txt = format!("|{:.3}, {:.3}, {:.3}|", posit.x, posit.y, posit.z);

    let text_0 = format!("#{}", atom.serial_number);
    let text_b = atom.element.to_letter();

    ui.label(RichText::new(text_0).color(Color32::WHITE));

    ui.label(RichText::new(posit_txt).color(Color32::GOLD));

    let atom_color = make_egui_color(atom.element.color());
    ui.label(RichText::new(text_b).color(atom_color));

    if let Some(res_i) = atom.residue {
        // Placeholder for water etc.
        let mut res_color = COLOR_AA_NON_RESIDUE_EGUI;

        let res = &residues[res_i];
        let res_txt = &format!("  {res}");

        if let ResidueType::AminoAcid(aa) = res.res_type {
            res_color = make_egui_color(aa_color(aa));
        }

        ui.label(RichText::new(res_txt).color(res_color));
    }

    ui.label(RichText::new(role).color(Color32::LIGHT_GRAY));

    if let Some(tir) = &atom.type_in_res {
        ui.label(RichText::new(format!("{tir}")).color(Color32::LIGHT_YELLOW));
    }

    if let Some(tir) = &atom.type_in_res_lipid {
        ui.label(RichText::new(tir).color(Color32::LIGHT_YELLOW));
    }

    if let Some(ff) = &atom.force_field_type {
        ui.label(RichText::new(format!("FF: {ff}")).color(Color32::LIGHT_YELLOW));
    }

    if let Some(q) = &atom.partial_charge {
        let plus = if *q > 0. { "+" } else { "" };
        let color = make_egui_color(drawing::color_viridis_float(
            *q,
            CHARGE_MAP_MIN,
            CHARGE_MAP_MAX,
        ));

        // todo: In some cases, this is getting rendered over the initial text? EGUI error?
        ui.label(RichText::new(format!("{plus}q: {q:.2}")).color(color));
    }
}

// todo: This would ideally be a method on FfParamSet, but that lib doesn't have access to our MolType enum.
/// Get params for a single molecule type.
pub fn get_params(set: &FfParamSet, mol_type: MolType) -> &Option<ForceFieldParams> {
    match mol_type {
        MolType::Peptide => &set.peptide,
        MolType::Ligand => &set.small_mol,
        MolType::NucleicAcid => &set.dna, // todo: Could be RNA
        MolType::Lipid => &set.lipids,
        _ => &None,
    }
}

/// `posit_override` is for example, relative atom positions, such as a positioned ligand.
fn disp_bond_data(
    bond: &Bond,
    atoms: &[Atom],
    mol_type: MolType,
    params: &FfParamSet,
    ui: &mut Ui,
) {
    let atom_0 = &atoms[bond.atom_0];
    let atom_1 = &atoms[bond.atom_1];

    ui.label("Bond");
    ui.label(
        RichText::new(format!("#{} - #{}", bond.atom_0_sn, bond.atom_1_sn)).color(Color32::WHITE),
    );

    ui.label(
        RichText::new(format!(
            "{} {} {}",
            atom_0.element.to_letter(),
            bond.bond_type.to_visual_str(),
            atom_1.element.to_letter()
        ))
        .color(Color32::LIGHT_BLUE),
    );

    // todo: Cache??
    let posit_0 = atom_0.posit;
    let posit_1 = atom_1.posit;
    let dist = (posit_1 - posit_0).magnitude();

    ui.label(RichText::new(format!("{dist:.3} Å")).color(Color32::LIGHT_YELLOW));

    if let Some(p) = get_params(params, mol_type) {
        if let (Some(ff_0), Some(ff_1)) = (
            atom_0.force_field_type.as_deref(),
            atom_1.force_field_type.as_deref(),
        ) {
            if let Some(b) = p.get_bond(&(ff_0.to_string(), ff_1.to_string())) {
                ui.label(RichText::new("Param len:"));
                ui.label(RichText::new(format!("{:.3} Å", b.r_0)).color(Color32::LIGHT_BLUE));
            }
        }
    }

    // todo: Frequency (measured/actual)
}

/// Display text of the selected atom or residue.
pub fn selected_data(
    state: &State,
    ligands: &[MoleculeSmall],
    nucleic_acids: &[MoleculeNucleicAcid],
    lipids: &[MoleculeLipid],
    selection: &Selection,
    ui: &mut Ui,
) {
    ui.horizontal(|ui| {
        // ui.horizontal_wrapped(|ui| {
        match selection {
            Selection::AtomPeptide(sel_i) => {
                let Some(mol) = &state.peptide else { return };
                if *sel_i >= mol.common.atoms.len() {
                    return;
                }

                let atom = &mol.common.atoms[*sel_i];
                disp_atom_data(atom, &mol.residues, None, ui);
            }
            Selection::AtomLig((mol_i, atom_i)) => {
                if *mol_i >= ligands.len() {
                    return;
                }
                let mol = &ligands[*mol_i];

                if *atom_i >= mol.common.atoms.len() {
                    return;
                }

                let atom = &mol.common.atoms[*atom_i];
                let posit = mol.common.atom_posits[*atom_i];

                disp_atom_data(atom, &[], Some(posit), ui);
            }
            // todo DRY
            Selection::AtomNucleicAcid((mol_i, atom_i)) => {
                if *mol_i >= nucleic_acids.len() {
                    return;
                }
                let mol = &nucleic_acids[*mol_i];

                if *atom_i >= mol.common.atoms.len() {
                    return;
                }

                let atom = &mol.common.atoms[*atom_i];
                let posit = mol.common.atom_posits[*atom_i];

                disp_atom_data(atom, &[], Some(posit), ui);
            }
            // todo DRY
            Selection::AtomLipid((mol_i, atom_i)) => {
                if *mol_i >= lipids.len() {
                    return;
                }
                let mol = &lipids[*mol_i];

                if *atom_i >= mol.common.atoms.len() {
                    return;
                }

                let atom = &mol.common.atoms[*atom_i];
                let posit = mol.common.atom_posits[*atom_i];

                disp_atom_data(atom, &mol.residues, Some(posit), ui);
            }
            Selection::Residue(sel_i) => {
                if let Some(mol) = &state.peptide {
                    if *sel_i >= mol.residues.len() {
                        return;
                    }

                    let res = &mol.residues[*sel_i];
                    // todo: Color-coding by part like atom, to make easier to view.

                    let mut res_color = COLOR_AA_NON_RESIDUE_EGUI;

                    if let ResidueType::AminoAcid(aa) = res.res_type {
                        res_color = make_egui_color(aa_color(aa));
                    }
                    ui.label(RichText::new(res.to_string()).color(res_color));
                }
            }
            Selection::AtomsPeptide(is) => {
                ui.label(RichText::new(format!("{} atoms", is.len())).color(Color32::GOLD));
            }
            Selection::BondPeptide(bond_i) => {
                let Some(mol) = &state.peptide else {
                    return;
                };
                if *bond_i >= mol.common.bonds.len() {
                    return;
                }

                let bond = &mol.common.bonds[*bond_i];
                disp_bond_data(
                    bond,
                    &mol.common.atoms,
                    MolType::Peptide,
                    &state.ff_param_set,
                    ui,
                );
            }
            Selection::BondLig((mol_i, bond_i)) => {
                if *mol_i >= ligands.len() {
                    return;
                }
                let mol = &ligands[*mol_i];
                if *bond_i >= mol.common.bonds.len() {
                    return;
                }

                let bond = &mol.common.bonds[*bond_i];
                disp_bond_data(
                    bond,
                    &mol.common.atoms,
                    MolType::Ligand,
                    &state.ff_param_set,
                    ui,
                );
            }
            Selection::BondNucleicAcid((mol_i, bond_i)) => {
                if *mol_i >= nucleic_acids.len() {
                    return;
                }
                let mol = &nucleic_acids[*mol_i];
                if *bond_i >= mol.common.bonds.len() {
                    return;
                }

                let bond = &mol.common.bonds[*bond_i];
                disp_bond_data(
                    bond,
                    &mol.common.atoms,
                    MolType::NucleicAcid,
                    &state.ff_param_set,
                    ui,
                );
            }
            Selection::BondLipid((mol_i, bond_i)) => {
                if *mol_i >= lipids.len() {
                    return;
                }
                let mol = &lipids[*mol_i];
                if *bond_i >= mol.common.bonds.len() {
                    return;
                }

                let bond = &mol.common.bonds[*bond_i];
                disp_bond_data(
                    bond,
                    &mol.common.atoms,
                    MolType::Lipid,
                    &state.ff_param_set,
                    ui,
                );
            }
            Selection::None => {}
        }
    });
}

fn mol_picker(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    redraw_pep: &mut bool,
    redraw_lig: &mut bool,
    engine_updates: &mut EngineUpdates,
) {
    // todo: Make this support other types.
    let mut recenter_orbit = false;
    if let Some(mol) = &mut state.peptide {
        let i_mol = 0; // todo: A/R if you add more peptides.

        let active = match state.volatile.active_mol {
            Some((MolType::Peptide, i_)) => i_ == i_mol,
            _ => false,
        };

        let color = if active {
            COLOR_ACTIVE_RADIO
        } else {
            COLOR_INACTIVE
        };

        if ui
            .button(RichText::new(&mol.common.ident).color(color))
            .clicked()
        {
            if active && state.volatile.active_mol.is_some() {
                state.volatile.active_mol = None;
            } else {
                state.volatile.active_mol = Some((MolType::Peptide, i_mol));
                state.volatile.orbit_center = Some((MolType::Peptide, i_mol));

                recenter_orbit = true;
            }

            *redraw_pep = true; // To reflect the change in thickness, color etc.

            let color_vis = if mol.common.visible {
                COLOR_ACTIVE
            } else {
                COLOR_INACTIVE
            };

            if ui.button(RichText::new("👁").color(color_vis)).clicked() {
                mol.common.visible = !mol.common.visible;

                *redraw_lig = true; // todo Overkill; only need to redraw (or even just clear) one.
                // todo: Generalize.
                engine_updates.entities = EntityUpdate::All;
                // engine_updates.entities.push_class(EntityClass::Peptide as u32);
            }
        }
    }

    for (i_mol, mol) in state.ligands.iter_mut().enumerate() {
        let active = match state.volatile.active_mol {
            Some((MolType::Ligand, i_)) => i_ == i_mol,
            _ => false,
        };

        let color = if active {
            COLOR_ACTIVE_RADIO
        } else {
            COLOR_INACTIVE
        };

        if ui
            .button(RichText::new(&mol.common.ident).color(color))
            .clicked()
        {
            if active && state.volatile.active_mol.is_some() {
                state.volatile.active_mol = None;
            } else {
                state.volatile.active_mol = Some((MolType::Ligand, i_mol));
                state.volatile.orbit_center = Some((MolType::Ligand, i_mol));

                recenter_orbit = true;
            }

            *redraw_lig = true; // To reflect the change in thickness, color etc.
        }

        let color_vis = if mol.common.visible {
            COLOR_ACTIVE
        } else {
            COLOR_INACTIVE
        };

        if ui.button(RichText::new("👁").color(color_vis)).clicked() {
            mol.common.visible = !mol.common.visible;

            *redraw_lig = true; // todo Overkill; only need to redraw (or even just clear) one.
            // todo: Generalize.
            engine_updates.entities = EntityUpdate::All;
            // engine_updates.entities.push_class(EntityClass::Ligand as u32);
        }
    }

    if recenter_orbit {
        // reset_orbit_center(state, scene);
        if let ControlScheme::Arc { center } = &mut scene.input_settings.control_scheme {
            *center = orbit_center(state);
        }
    }
}

// todo: Unify this with non-peptide.
pub fn display_mol_data_peptide(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    redraw_peptide: &mut bool,
    redraw_lig: &mut bool,
    close: &mut bool,
    engine_updates: &mut EngineUpdates,
) {
    // These variables prevent double borrows.
    let mut res_to_make = None;
    let mut move_lig_to_res = None;
    let mut move_lig_to_sel = None;
    let mut move_cam = false;

    ui.horizontal(|ui| {
        if let Some(pep) = &state.peptide {
            mol_descrip(&MolGenericRef::Peptide(pep), ui);

            if ui.button(RichText::new("Close").color(Color32::LIGHT_RED)).clicked() {
                *close = true;
            }

            if pep.common.ident.len() <= 5 {
                // todo: You likely need a better approach.
                if ui
                    .button("View on RCSB")
                    .on_hover_text("Open a web browser to the RCSB PDB page for this molecule.")
                    .clicked()
                {
                    rcsb::open_overview(&pep.common.ident);
                }
            }

            if ui.button("Plot dihe")
                .on_hover_text("Draw a Ramachandran plot of the dihedral angles of the peptide.")
                .clicked() {
                state.ui.popup.rama_plot = !state.ui.popup.rama_plot;
            }

            let res_selected = match state.ui.selection {
                Selection::AtomPeptide(sel_i) => {
                    let atom = &pep.common.atoms[sel_i];
                    if let Some(res_i) = &atom.residue {
                        Some(&pep.residues[*res_i])
                    } else {
                        None
                    }
                }
                Selection::Residue(sel_i) => {
                    if sel_i >= pep.residues.len() {
                        handle_err(&mut state.ui, "Residue selection is out of bounds.".to_owned());
                        None
                    } else {
                        Some(&pep.residues[sel_i])
                    }
                },
                _ => None,
            };

            if let Some(res) = res_selected {
                if ui
                    .button(
                        RichText::new(format!("Lig from {}", res.res_type))
                            .color(Color32::GOLD),
                    )
                    .on_hover_text(
                        "Create a ligand from this residue on the peptide. This can be \
                    saved to a Mol2 or SDF file, and used as a ligand. Molecular dynamics can be performed on it.",
                    )
                    .clicked()
                {
                    // todo: I don't like this clone, but it avoids a dbl-borrow.
                    res_to_make = Some(res.clone());
                }
            }

            if let Some(mol) = state.active_mol() {
                for res in &pep.het_residues {
                    // Note: This approach will fail if there are multiple hetero residues of similar len to
                    // this ligand.
                    if (res.atoms.len() - mol.common().atoms.len()) < 3 {
                        let name = match &res.res_type {
                            ResidueType::Other(name) => name,
                            _ => "hetero residue",
                        };
                        ui.add_space(COL_SPACING / 2.);

                        if ui
                            .button(RichText::new(format!("Move lig to {name}")).color(COLOR_HIGHLIGHT))
                            .on_hover_text("Move the ligand to be colocated with this residue. this is intended to \
                    be used to synchronize the ligand with a pre-positioned hetero residue in the protein file, e.g. \
                    prior to docking. In addition to moving \
                    its center, this attempts to align each atom with its equivalent on the residue.")
                            .clicked()
                        {
                            // todo: I don't like this clone, but it avoids a dbl-borrow.
                            move_lig_to_res = Some(res.clone());
                        }
                        break;
                    }
                }
            }

            if let Some((mol_type, _)) = state.volatile.active_mol {
                if mol_type == MolType::Ligand {
                    if !matches!(
                state.ui.selection,
                Selection::None | Selection::AtomLig(_)
            ) {
                        if ui
                            .button(RichText::new("Move lig to sel").color(COLOR_HIGHLIGHT))
                            .on_hover_text("Re-position the ligand to be colacated with the selected atom or residue.")
                            .clicked()
                        {
                            let peptide = state.peptide.as_ref().unwrap();
                            let atom_sel = peptide.get_sel_atom(&state.ui.selection);

                            if let Some(a) = atom_sel {
                                // See note on why we clone above.
                                move_lig_to_sel = Some(a.clone());
                            }
                        }
                    }
                }
            }
        }
    });

    if let Some(res) = res_to_make {
        make_lig_from_res(state, &res, scene, engine_updates);
        // if let Some(pep) = &state.peptide {
        //     move_cam_to_active_mol(state, scene, pep.center, engine_updates);
        // }
    }

    if let Some(res) = move_lig_to_res {
        if let Some((_, i)) = state.volatile.active_mol {
            let mol = &mut state.ligands[i];
            if let Some(pep) = &state.peptide {
                move_mol_to_res(&mut MoGenericRefMut::Ligand(mol), pep, &res);
                move_cam_to_active_mol(state, scene, pep.center, engine_updates);
            }
        }

        *redraw_lig = true;
    }

    if let Some(sel_atom) = move_lig_to_sel {
        let mut mol = state.active_mol_mut().unwrap();
        mol.common_mut().move_to(sel_atom.posit);

        let center = match &state.peptide {
            Some(p) => p.center,
            None => Vec3::new_zero(),
        };
        move_cam_to_active_mol(state, scene, center, engine_updates);

        move_cam = true;

        *redraw_lig = true;
    }

    if move_cam {
        let center = match &state.peptide {
            Some(m) => m.center,
            None => Vec3::new_zero(),
        };

        move_cam_to_active_mol(state, scene, center, engine_updates);
    }

    // Provide convenience functionality for loading ligands based on hetero residues
    // in the protein.
    let mut load_data = None; // Avoids dbl-borrow.

    let mut res_to_load = None;
    if let Some(mol) = &mut state.peptide {
        let mut count_geostd_candidate = 0;
        for res in &mol.het_residues {
            if let ResidueType::Other(name) = &res.res_type {
                if name.len() == 3 {
                    count_geostd_candidate += 1;
                }
            }
        }

        if count_geostd_candidate > 0 {
            ui.horizontal(|ui| {
                ui.label("Make ligs:").on_hover_text(
                    "Attempt to load a ligand molecule and force field \
                            params from a hetero residue included in the protein file.",
                );

                // This mechanism prevents buttons from duplicate hetero residues, e.g.
                // if more than one copy of a ligand is present in the data.
                let mut residue_names = Vec::new();
                for res in &mol.het_residues {
                    let name = match &res.res_type {
                        ResidueType::Other(name) => name,
                        _ => "hetero residue",
                    };
                    if name.len() == 3 {
                        if residue_names.contains(&name) {
                            continue;
                        }
                        residue_names.push(name);

                        if ui
                            .button(RichText::new(name).color(Color32::GOLD))
                            .clicked()
                        {
                            download_mols::load_geostd(name, &mut load_data, &mut state.ui);
                            res_to_load = Some(res.clone()); // Clone avoids borrow error.
                        }
                    }
                }
            });
        }
    }

    // Avoids dbl-borrow
    if let Some(data) = load_data {
        handle_success(
            &mut state.ui,
            format!("Loaded {} from Amber Geostd", data.ident_pdbe),
        );

        // Crude check for success.
        // let lig_count_prev = state.ligands.len();
        state.load_geostd_mol_data(
            &data.ident_pdbe,
            true,
            data.frcmod_avail,
            engine_updates,
            scene,
        );

        // Move camera to ligand; not ligand to camera, since we are generating a ligand
        // that may already be docked to the protein.
        // move_mol_to_cam(&mut state.ligands[i].common, &scene.camera);
        if let Some(mol) = &state.peptide {
            move_cam_to_active_mol(state, scene, mol.center, engine_updates);
        }
    } else {
        if let Some(res) = res_to_load {
            // Use our normal "Lig from" logic.
            make_lig_from_res(state, &res, scene, engine_updates);

            move_cam_to_active_mol(
                state,
                scene,
                state.ligands[0].common.centroid(),
                engine_updates,
            );

            handle_success(
                &mut state.ui,
                "Unable to find FF params for this ligand; added without them".to_string(),
            );
        }
    }
}

pub fn display_mol_data(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    redraw_pep: &mut bool,
    redraw_lig: &mut bool,
    redraw_na: &mut bool,
    redraw_lipid: &mut bool,
    close: &mut bool,
    engine_updates: &mut EngineUpdates,
) {
    ui.horizontal(|ui| {
        mol_picker(state, scene, ui, redraw_pep, redraw_lig, engine_updates);

        let Some((active_mol_type, active_mol_i)) = state.volatile.active_mol else {
            return
        };

        ui.add_space(COL_SPACING);

        if let Some(mol) = state.active_mol() {
            mol_descrip(&mol, ui);
        }

        {
            let mut color_move = COLOR_INACTIVE;
            let mut color_rotate = COLOR_INACTIVE;

            match state.volatile.mol_manip.mol {
                ManipMode::Move((mol_type, mol_i)) => {
                    if mol_type == active_mol_type && mol_i == active_mol_i {
                        color_move = COLOR_ACTIVE;
                    }
                }
                ManipMode::Rotate((mol_type, mol_i)) => {
                    if mol_type == active_mol_type && mol_i == active_mol_i {
                        color_rotate = COLOR_ACTIVE;
                    }
                }
                ManipMode::None => (),
            }

            // ✥ doesn't work in EGUI.
            if ui.button(RichText::new("↔").color(color_move))
                .on_hover_text("(Hotkey: M. M or Esc to stop)) Move the active molecule by clicking and dragging with \
                the mouse. Scroll to move it forward and back.")
                .clicked() {

                set_manip(&mut state.volatile, scene, redraw_lig, redraw_na, redraw_lipid, ManipMode::Move((active_mol_type, active_mol_i)));
            }

            if ui.button(RichText::new("⟳").color(color_rotate))
                .on_hover_text("(Hotkey: R. R or Esc to stop) Rotate the active molecule by clicking and dragging with the mouse. Scroll to roll.")
                .clicked() {

                set_manip(&mut state.volatile, scene, redraw_lig,redraw_na, redraw_lipid, ManipMode::Rotate((active_mol_type, active_mol_i)));
            }
        }

        if let Some(mol) = &mut state.active_mol_mut() {
            if ui
                .button(RichText::new("Move to cam").color(COLOR_HIGHLIGHT))
                .on_hover_text(
                    "Move the molecule to be a short distance in front of the camera.",
                )
                .clicked()
            {
                move_mol_to_cam(mol.common_mut(), &scene.camera);

                match active_mol_type {
                    MolType::Ligand => *redraw_lig = true,
                    MolType::NucleicAcid => *redraw_na = true,
                    MolType::Lipid => *redraw_lipid = true,
                    _ => unimplemented!()
                }
            }

            if ui
                .button(RichText::new("Reset pos").color(COLOR_HIGHLIGHT))
                .on_hover_text(
                    "Move the molecule to its absolute coordinates, e.g. as defined in \
                        its source mmCIF, Mol2 or SDF file.",
                )
                .clicked()
            {
                mol.common_mut().reset_posits();

                // todo: Use the inplace move.
                match active_mol_type {
                    MolType::Ligand => *redraw_lig = true,
                    MolType::NucleicAcid => *redraw_na = true,
                    MolType::Lipid => *redraw_lipid = true,
                    _ => unimplemented!()
                }
            }

            if active_mol_type == MolType::Ligand {
                if ui
                    .button(RichText::new("Similar mols").color(COLOR_HIGHLIGHT))
                    .on_hover_text(
                        "Using PubChem, find, download, and open similar molecules to this one.",
                    )
                    .clicked()
                {
                    let mol = &state.ligands[active_mol_i];

                    // todo: This needs to be in its own thread; long-running blockign call.

                    // todo: Support more than CID. Requires a mode to rcsb api.
                    for ident in &mol.idents {
                        if let MolIdent::PubChem(cid) = ident {
                            // todo: This doesn't show because it doesn't get a chance to render prior to the block.
                            handle_success(&mut state.ui, "Searching for similar molecules...".to_string());
                            match pubchem::find_similar_mols(*cid) {
                                Ok(cids) => {
                                    // todo: This is temp
                                    println!("Similar mols to {cid}: {:?}", cids);
                                    let max_results = 20;
                                    let cids_str = cids
                                        .iter()
                                        .take(max_results)
                                        .map(|x| x.to_string())
                                        .collect::<Vec<_>>()
                                        .join(", ");

                                    handle_success(&mut state.ui, format!("Similar PubChem CIDs: {cids_str}..."));
                                }

                                Err(e) => {
                                    handle_err(&mut state.ui, "Problem finding similar molecules on PubChem".to_owned());
                                }
                            }
                            break;
                        }
                    }
                }
            }
        }

        if ui.button(RichText::new("Close").color(Color32::LIGHT_RED))
            .on_hover_text("(Hotkey: Delete) Close this molecule.")
            .clicked() {
            *close = true;
        }

        if let Some(mol) = state.active_mol() {
            match mol {
                MolGenericRef::Peptide(_) => {}
                MolGenericRef::Ligand(m) => {
                    ui.add_space(COL_SPACING);

                    // todo status color helper?
                    ui.label("Loaded:");
                    let color = if m.ff_params_loaded {
                        Color32::LIGHT_GREEN
                    } else {
                        Color32::LIGHT_RED
                    };
                    ui.label(RichText::new("FF/q").color(color)).on_hover_text(
                        "Green if force field names, and partial charges are assigned \
                for all ligand atoms. Required for ligand moleculer dynamics and docking.",
                    );

                    ui.add_space(COL_SPACING / 4.);

                    let color = if m.frcmod_loaded {
                        Color32::LIGHT_GREEN
                    } else {
                        Color32::LIGHT_RED
                    };
                    ui.label(RichText::new("Frcmod").color(color))
                        .on_hover_text(
                            "Green if molecule-specific Amber force field parameters are \
                loaded for this ligand. Required for ligand molecular dynamics and docking.",
                        );

                    let mut pubchem_cid = None;
                    for ident in &m.idents {
                        match ident {
                            MolIdent::DrugBank(id) => {
                                if ui.button(format!("DrugBank: {id}")).clicked() {
                                    drugbank::open_overview(id);
                                }
                            }
                            MolIdent::PubChem(cid) => {
                                if ui.button(format!("PubChem: {cid}")).clicked() {
                                    pubchem::open_overview(*cid);
                                }
                                pubchem_cid = Some(*cid);
                            }
                            MolIdent::PdbeAmber(id) => {
                                if ui.button(format!("PDBe: {id}")).clicked() {
                                    pdbe::open_overview(id);
                                }
                            }
                        }

                    }

                    if let Some(cid) = pubchem_cid {
                        if ui.button("Find associated structs").clicked() {
                            // todo: Don't block.
                            if m.associated_structures.is_empty() {
                                match pubchem::load_associated_structures(cid) {
                                    Ok(data) => {
                                        // todo: Put back! Borrow issue.
                                        // l.associated_structures = data;
                                        state.ui.popup.show_associated_structures = true;
                                    }
                                    Err(_) => handle_err(
                                        &mut state.ui,
                                        "Unable to find structures for this ligand".to_owned(),
                                    ),
                                }
                            } else {
                                state.ui.popup.show_associated_structures = true;
                            }
                        }
                    }
                }
                MolGenericRef::Lipid(l) => {
                    if ui.button("View on LMSD").clicked() {
                        lmsd::open_overview(&l.lmsd_id);
                    }
                }
                _ => ()
            }
        }

        ui.add_space(COL_SPACING);
    });
}
