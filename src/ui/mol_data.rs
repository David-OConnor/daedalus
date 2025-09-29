//! Information and settings for the opened, or to-be opened molecules.

use bio_apis::{amber_geostd, drugbank, lmsd, pdbe, pubchem, rcsb};
use bio_files::ResidueType;
use egui::{Color32, RichText, Ui};
use graphics::{EngineUpdates, Scene};
use lin_alg::f64::Vec3;

use crate::{
    ManipMode, Selection, State, drawing,
    drawing::{CHARGE_MAP_MAX, CHARGE_MAP_MIN, COLOR_AA_NON_RESIDUE_EGUI},
    lipid::MoleculeLipid,
    mol_lig::MoleculeSmall,
    mol_manip::set_manip,
    molecule::{Atom, MolType, MoleculeGenericRef, MoleculeGenericRefMut, Residue, aa_color},
    nucleic_acid::MoleculeNucleicAcid,
    ui::{
        COL_SPACING, COLOR_ACTIVE, COLOR_ACTIVE_RADIO, COLOR_HIGHLIGHT, COLOR_INACTIVE,
        ROW_SPACING, cam::move_cam_to_lig, mol_descrip,
    },
    util::{handle_err, handle_success, make_egui_color, move_mol_to_res},
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

/// Display text of the selected atom or residue.
pub fn selected_data(
    state: &State,
    ligands: &[MoleculeSmall],
    nucleic_acids: &[MoleculeNucleicAcid],
    lipids: &[MoleculeLipid],
    selection: &Selection,
    ui: &mut Ui,
) {
    // ui.horizontal_wrapped(|ui| {
    match selection {
        Selection::AtomPeptide(sel_i) => {
            if let Some(mol) = &state.peptide {
                if *sel_i >= mol.common.atoms.len() {
                    return;
                }

                let atom = &mol.common.atoms[*sel_i];
                disp_atom_data(atom, &mol.residues, None, ui);
            }
        }
        Selection::AtomLig((lig_i, atom_i)) => {
            if *lig_i >= ligands.len() {
                return;
            }
            let mol = &ligands[*lig_i];

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
        Selection::Atoms(is) => {
            ui.label(RichText::new(format!("{} atoms", is.len())).color(Color32::GOLD));
        }
        Selection::None => (),
    }
    // });
}

fn mol_picker(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    redraw_lig: &mut bool,
    close_lig: &mut bool,
    engine_updates: &mut EngineUpdates,
) {
    // todo: Make this support other types.
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
            engine_updates.entities = true;
        }
    }
}

// todo: Unify this with non-peptide.
pub fn display_mol_data_peptide(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    redraw_lig: &mut bool,
    close: &mut bool,
    engine_updates: &mut EngineUpdates,
) {
    ui.horizontal(|ui| {
        let Some(mol) = &state.peptide else {
            return;
        };

        mol_descrip(&MoleculeGenericRef::Peptide(mol), ui);

        if ui.button(RichText::new("Close").color(Color32::LIGHT_RED)).clicked() {
            *close = true;
        }

        if mol.common.ident.len() <= 5 {
            // todo: You likely need a better approach.
            if ui
                .button("View on RCSB")
                .on_hover_text("Open a web browser to the RCSB PDB page for this molecule.")
                .clicked()
            {
                rcsb::open_overview(&mol.common.ident);
            }
        }

        if ui.button("Plot dihedrals").clicked() {
            state.ui.popup.rama_plot = !state.ui.popup.rama_plot;
        }

        ui.add_space(COL_SPACING);

            let res_selected = match state.ui.selection {
                Selection::AtomPeptide(sel_i) => {
                    let atom = &mol.common.atoms[sel_i];
                    if let Some(res_i) = &atom.residue {
                        Some(&mol.residues[*res_i])
                    } else {
                        None
                    }
                }
                Selection::Residue(sel_i) => {
                    if sel_i >= mol.residues.len() {
                        handle_err(&mut state.ui, "Residue selection is out of bounds.".to_owned());
                        None
                    } else {
                        Some(&mol.residues[sel_i])
                    }
                },
                _ => None,
            };

            let Some(pep) = &state.peptide else {
                return;
            };

            // let mut lig_atom_count = 0;
            // if let Some(lig) = &state.active_lig() {
            //     lig_atom_count = lig.common.atoms.len();
            // }

            for res in &pep.het_residues {
                // Note: This is crude.
                if (res.atoms.len() - pep.common.atoms.len()) < 5 {
                    // todo: Don't list multiple; pick teh closest, at least in len.
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

                        // todo: Put back. Borrow problem.
                        // let mut mol = state.active_mol_mut().unwrap();
                        // let _docking_center = move_mol_to_res(&mut mol, pep, res);
                        //
                        // if let Some(m) = &state.active_mol() {
                        //     move_cam_to_lig(
                        //         // &m,
                        //         // &mut state.ui.cam_snapshot,
                        //         &mut state,
                        //         scene,
                        //         pep.center,
                        //         engine_updates,
                        //     );
                        // }

                        *redraw_lig = true;
                    }
                }
            }
            if let Some(res) = res_selected {
                if ui
                    .button(
                        RichText::new(format!("Make lig from {}", res.res_type))
                            .color(Color32::GOLD),
                    )
                    .on_hover_text(
                        "Create a ligand from this residue on the peptide. This can be \
                    saved to a Mol2 or SDF file, and used as a ligand.",
                    )
                    .clicked()
                {
                    let res_type = res.res_type.clone(); // Avoids dbl-borrow.

                    let mut mol_fm_res = MoleculeSmall::from_res(
                        res,
                        &pep.common.atoms,
                        &pep.common.bonds,
                        false,
                        // &state.ff_params.lig_specific,
                    );
                    // todo: Borrow prob.
                    // mol_fm_res.update_aux(&state);

                    let mut lig_new = MoleculeGenericRefMut::Ligand(&mut mol_fm_res);

                    state.mol_dynamics = None;

                    let docking_center = move_mol_to_res(&mut lig_new, pep, res);

                    // todo: Put this save back / fix dble-borrow?
                    // state.update_docking_site(docking_center);
                    // state.update_sa
                    // ve_prefs(false);
                    // set_docking_light(scene, Some(&lig.docking_site));
                    // engine_updates.lighting = true;

                    *redraw_lig = true;

                    // If creating from an AA, move to the origin (Where we assigned its atom positions).
                    // If from a hetero atom, leave it in place.
                    match &res_type {
                        ResidueType::AminoAcid(_) => {
                            let mut mol = state.active_mol_mut().unwrap();
                            mol.common_mut().reset_posits();
                        }
                        _ => {
                            state.ui.visibility.hide_hetero = true;
                        }
                    }
                    let mut mol = state.active_mol_mut().unwrap();
                    mol = lig_new;

                    // Make it clear that we've added the ligand by showing it, and hiding hetero (if creating from Hetero)
                    state.ui.visibility.hide_ligand = false;
                }
            }

        if !matches!(
            state.ui.selection,
            Selection::None | Selection::AtomLig(_)
        ) {
            let mut move_cam = false;
            if ui
                .button(RichText::new("Move lig to sel").color(COLOR_HIGHLIGHT))
                .on_hover_text("Re-position the ligand to be colacated with the selected atom or residue.")
                .clicked()
            {
                let peptide = &state.peptide.as_ref().unwrap();
                let atom_sel = peptide.get_sel_atom(&state.ui.selection);
                state.mol_dynamics = None;

                if let Some(sel_atom) = atom_sel {
                    // todo: Put back. Borrow problem.

                    // let mut mol = state.active_mol_mut().unwrap();
                    // let diff = sel_atom.posit - mol.common().centroid();
                    // for p in &mut mol.common_mut().atom_posits{
                    //     *p += diff;
                    // }

                    move_cam = true;

                    *redraw_lig = true;
                }
            }

            if move_cam {
                let center = match &state.peptide {
                    Some(m) => m.center,
                    None => Vec3::new_zero()
                };

                // todo: put back. Borrow problem.
                // move_cam_to_lig(
                //     &mut state,
                //     scene,
                //     center,
                //     engine_updates,
                // );
            }
        }
    });

    // If no ligand, provide convenience functionality for loading one based on hetero residues
    // in the protein.
    if state.active_mol().is_some() {
        return;
    }

    let mut load_data = None; // Avoids dbl-borrow.

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
                ui.label("Load Amber Geostd lig from: ").on_hover_text(
                    "Attempt to load a ligand molecule and force field \
                            params from a hetero residue included in the protein file.",
                );

                for res in &mol.het_residues {
                    let name = match &res.res_type {
                        ResidueType::Other(name) => name,
                        _ => "hetero residue",
                    };
                    if name.len() == 3 {
                        if ui
                            .button(RichText::new(name).color(Color32::GOLD))
                            .clicked()
                        {
                            match amber_geostd::find_mols(&name) {
                                Ok(data) => match data.len() {
                                    0 => handle_err(
                                        &mut state.ui,
                                        "Unable to find an Amber molecule for this residue"
                                            .to_string(),
                                    ),
                                    1 => {
                                        load_data = Some(data[0].clone());
                                    }
                                    _ => {
                                        load_data = Some(data[0].clone());
                                        eprintln!("More than 1 geostd items available");
                                    }
                                },
                                Err(e) => handle_err(
                                    &mut state.ui,
                                    format!("Problem loading mol data online: {e:?}"),
                                ),
                            }
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
            format!("Loaded {} from Amber Geostd", data.ident),
        );
        state.load_geostd_mol_data(&data.ident, true, data.frcmod_avail, redraw_lig);

        if state.active_mol().is_some() {
            if let Some(mol) = &state.peptide {
                move_cam_to_lig(state, scene, mol.center, engine_updates);
            }
        }
    }
}

pub fn display_mol_data(
    state: &mut State,
    scene: &mut Scene,
    ui: &mut Ui,
    // mol_type: MolType,
    redraw_lig: &mut bool,
    redraw_na: &mut bool,
    redraw_lipid: &mut bool,
    close: &mut bool,
    engine_updates: &mut EngineUpdates,
) {
    ui.horizontal(|ui| {
        mol_picker(state, scene, ui, redraw_lig, close, engine_updates);

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
                .on_hover_text("Move the active molecule by clicking and dragging with the mouse. Scroll to move it forward and back. (Hotkey: M)")
                .clicked() {

                set_manip(&mut state.volatile, scene, redraw_lig, redraw_na, redraw_lipid, ManipMode::Move((active_mol_type, active_mol_i)));
            }

            if ui.button(RichText::new("⟳").color(color_rotate))
                .on_hover_text("Rotate the active molecule by clicking and dragging with the mouse. Scroll to roll. (Hotkey: R)")
                .clicked() {

                set_manip(&mut state.volatile, scene, redraw_lig,redraw_na, redraw_lipid, ManipMode::Rotate((active_mol_type, active_mol_i)));
            }
        }

        if ui
            .button(RichText::new("Reset posit").color(COLOR_HIGHLIGHT))
            .on_hover_text(
                "Move the moleculeto its absolute coordinates, e.g. as defined in \
                    its source mmCIF, Mol2 or SDF file.",
            )
            .clicked()
        {
            if let Some(mol) = &mut state.active_mol_mut() {
                mol.common_mut().reset_posits();
                // todo: Not working
                println!("Reset positions"); // todo: Temp debug

                match active_mol_type {
                    MolType::Ligand => *redraw_lig = true,
                    MolType::NucleicAcid => *redraw_na = true,
                    MolType::Lipid => *redraw_lipid = true,
                    _ => unimplemented!()
                }

            }
        }

        if ui.button(RichText::new("Close").color(Color32::LIGHT_RED)).clicked() {
            *close = true;
        }

        if let Some(mol) = state.active_mol() {
            match mol {
                MoleculeGenericRef::Peptide(m) => {}
                MoleculeGenericRef::Ligand(l) => {
                    ui.add_space(COL_SPACING);

                    // todo status color helper?
                    ui.label("Loaded:");
                    let color = if l.ff_params_loaded {
                        Color32::LIGHT_GREEN
                    } else {
                        Color32::LIGHT_RED
                    };
                    ui.label(RichText::new("FF/q").color(color)).on_hover_text(
                        "Green if force field names, and partial charges are assigned \
                for all ligand atoms. Required for ligand moleculer dynamics and docking.",
                    );

                    ui.add_space(COL_SPACING / 4.);

                    let color = if l.frcmod_loaded {
                        Color32::LIGHT_GREEN
                    } else {
                        Color32::LIGHT_RED
                    };
                    ui.label(RichText::new("Frcmod").color(color))
                        .on_hover_text(
                            "Green if molecule-specific Amber force field parameters are \
                loaded for this ligand. Required for ligand molecular dynamics and docking.",
                        );

                    if let Some(id) = &l.drugbank_id {
                        if ui.button("View on Drugbank").clicked() {
                            drugbank::open_overview(id);
                        }
                    }

                    if let Some(id) = l.pubchem_cid {
                        if ui.button("View on PubChem").clicked() {
                            pubchem::open_overview(id);
                        }
                    }

                    if let Some(id) = &l.pdbe_id {
                        if ui.button("View on PDBe").clicked() {
                            pdbe::open_overview(id);
                        }
                    }

                    if let Some(cid) = l.pubchem_cid {
                        if ui.button("Find associated structs").clicked() {
                            // todo: Don't block.
                            if l.associated_structures.is_empty() {
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
                MoleculeGenericRef::Lipid(l) => {
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
