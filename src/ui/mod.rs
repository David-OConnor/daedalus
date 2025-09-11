use std::{
    io,
    io::Cursor,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

use bio_apis::{amber_geostd, rcsb};
use egui::{
    Color32, ComboBox, Context, Key, Popup, PopupAnchor, Pos2, RectAlign, RichText, Slider,
    TextEdit, TopBottomPanel, Ui,
};
use graphics::{ControlScheme, EngineUpdates, Scene};
use na_seq::AaIdent;

static INIT_COMPLETE: AtomicBool = AtomicBool::new(false);

use bio_files::{DensityMap, ResidueType, density_from_2fo_fc_rcsb_gemmi};
use md::md_setup;
use mol_data::disp_lig_data;

use crate::{
    CamSnapshot,
    // docking::{
    //     ConformationType, calc_binding_energy, find_optimal_pose, find_sites::find_docking_sites,
    // },
    MsaaSetting,
    Selection,
    State,
    ViewSelLevel,
    cli,
    cli::autocomplete_cli,
    download_mols::{load_sdf_drugbank, load_sdf_pubchem},
    drawing::{
        EntityType, MoleculeView, draw_density_point_cloud, draw_density_surface,
        draw_nucleic_acid, draw_peptide, draw_water,
    },
    file_io::gemmi_path,
    inputs::{MOVEMENT_SENS, ROTATE_SENS},
    molecule::MoleculeGenericRef,
    nucleic_acid::{MoleculeNucleicAcid, NucleicAcidType, Strands},
    render::{set_flashlight, set_static_light},
    ui::{
        cam::{cam_controls, cam_snapshots, move_cam_to_lig},
        misc::section_box,
    },
    util::{
        check_prefs_save, close_lig, close_peptide, cycle_selected, handle_err, handle_scene_flags,
        handle_success, load_atom_coords_rcsb, orbit_center, reset_camera, select_from_search,
    },
};
use crate::{drawing::draw_all_ligs, mol_lig::MoleculeSmall, ui::misc::handle_docking};

pub mod cam;
mod md;
pub mod misc;
mod mol_data;

pub const ROW_SPACING: f32 = 10.;
pub const COL_SPACING: f32 = 30.;

const DENS_ISO_MIN: f32 = 1.0;
const DENS_ISO_MAX: f32 = 3.0;

const NEARBY_THRESH_MIN: u16 = 5;
const NEARBY_THRESH_MAX: u16 = 60;

pub const COLOR_INACTIVE: Color32 = Color32::GRAY;
pub const COLOR_ACTIVE: Color32 = Color32::LIGHT_GREEN;
pub const COLOR_HIGHLIGHT: Color32 = Color32::LIGHT_BLUE;
pub const COLOR_ACTIVE_RADIO: Color32 = Color32::LIGHT_BLUE;
pub const COLOR_ATTENTION: Color32 = Color32::ORANGE;

const COLOR_OUT_ERROR: Color32 = Color32::LIGHT_RED;
const COLOR_OUT_NORMAL: Color32 = Color32::WHITE;
const COLOR_OUT_SUCCESS: Color32 = Color32::LIGHT_GREEN; // Unused for now

pub const COLOR_POPUP: Color32 = Color32::from_rgb(20, 20, 30);

// Number of characters to display. E.g. the molecular description. Often long.
const MAX_TITLE_LEN: usize = 80;

/// Update the tilebar to reflect the current molecule
fn set_window_title(title: &str, scene: &mut Scene) {
    scene.window_title = title.to_owned();
    // ui.ctx().send_viewport_cmd(ViewportCommand::Title(title.to_string()));
}

fn open_lig(state: &mut State, mut mol: MoleculeSmall) {
    // state.ligand =
    //     Some(Ligand::new(mol, &state.ff_params.lig_specific));

    mol.update_aux(state);
    state.ligands.push(mol);

    state.volatile.active_lig = Some(state.ligands.len() - 1);

    state.mol_dynamics = None;
    state.update_from_prefs();

    state.ui.db_input = String::new();
}

pub fn load_file(
    path: &Path,
    state: &mut State,
    redraw: &mut bool,
    reset_cam: &mut bool,
    engine_updates: &mut EngineUpdates,
) -> io::Result<()> {
    state.open(path)?;

    // Clear last map opened here, vice in `open_molecule`, to prevent it clearing the map
    // on init.

    *redraw = true;
    *reset_cam = true;
    engine_updates.entities = true;

    Ok(())
}

// pub fn int_field_w_redraw(val: &mut u32, label: &str, redraw: &mut bool, ui: &mut Ui) {
//     ui.label(label);
//     let mut val_str = val.to_string();
//
//     if ui
//         .add_sized(
//             [70., Ui::available_height(ui)],
//             TextEdit::singleline(&mut val_str),
//         )
//         .changed()
//     {
//         if let Ok(v) = val_str.parse::<u32>() {
//             *val = v;
//             *redraw = true;
//         }
//     }
// }

pub fn num_field<T>(val: &mut T, label: &str, width: u16, ui: &mut Ui)
where
    T: std::fmt::Display + std::str::FromStr,
{
    ui.label(label);
    let mut val_str = val.to_string();

    if ui
        .add_sized(
            [width as f32, Ui::available_height(ui)],
            TextEdit::singleline(&mut val_str),
        )
        .changed()
    {
        if let Ok(v) = val_str.parse::<T>() {
            *val = v;
        }
    }
}

/// Handles keyboard and mouse input not associated with a widget.
pub fn handle_input(
    state: &mut State,
    ui: &mut Ui,
    redraw: &mut bool,
    reset_cam: &mut bool,
    engine_updates: &mut EngineUpdates,
) {
    ui.ctx().input(|ip| {
        // Check for file drop
        if let Some(dropped_files) = ip.raw.dropped_files.first() {
            if let Some(path) = &dropped_files.path {
                if let Err(e) = load_file(path, state, redraw, reset_cam, engine_updates) {
                    handle_err(&mut state.ui, e.to_string());
                }
            }
        }
    });
}

fn get_snap_name(snap: Option<usize>, snaps: &[CamSnapshot]) -> String {
    match snap {
        Some(i) => snaps[i].name.clone(),
        None => "None".to_owned(),
    }
}

fn residue_selector(state: &mut State, scene: &mut Scene, redraw: &mut bool, ui: &mut Ui) {
    // This is a bit fuzzy, as the size varies by residue name (Not always 1 for non-AAs), and index digits.

    let mut update_arc_center = false;

    if let Some(mol) = &state.molecule {
        if let Some(chain_i) = state.ui.chain_to_pick_res {
            if chain_i >= mol.chains.len() {
                return;
            }
            let chain = &mol.chains[chain_i];

            ui.add_space(ROW_SPACING);
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 8.0;

                for (i, res) in mol.residues.iter().enumerate() {
                    // For now, peptide residues only.
                    if let ResidueType::Water = res.res_type {
                        continue;
                    }

                    // Only let the user select residue from the selected chain. This should keep
                    // it more organized, and keep UI space used down.
                    if !chain.residues.contains(&i) {
                        continue;
                    }

                    let name = match &res.res_type {
                        ResidueType::AminoAcid(aa) => aa.to_str(AaIdent::OneLetter),
                        ResidueType::Water => "Water".to_owned(),
                        ResidueType::Other(name) => name.clone(),
                    };

                    let mut color = Color32::GRAY;
                    if let Selection::Residue(sel_i) = state.ui.selection {
                        if sel_i == i {
                            color = COLOR_ACTIVE;
                        }
                    }
                    if ui
                        .button(
                            RichText::new(format!("{} {name}", res.serial_number))
                                .size(10.)
                                .color(color),
                        )
                        .clicked()
                    {
                        state.ui.view_sel_level = ViewSelLevel::Residue;
                        state.ui.selection = Selection::Residue(i);

                        update_arc_center = true; // Avoids borrow error.

                        *redraw = true;
                    }
                }
            });
        }
    }

    if update_arc_center {
        if let ControlScheme::Arc { center } = &mut scene.input_settings.control_scheme {
            *center = orbit_center(state);
        }
    }
}

/// Toggles chain visibility
fn chain_selector(state: &mut State, redraw: &mut bool, ui: &mut Ui) {
    // todo: For now, DRY with res selec
    let Some(mol) = &mut state.molecule else {
        return;
    };

    ui.horizontal(|ui| {
        ui.label("Chain vis:");
        for chain in &mut mol.chains {
            let color = misc::active_color(chain.visible);

            if ui
                .button(RichText::new(chain.id.clone()).color(color))
                .clicked()
            {
                chain.visible = !chain.visible;
                *redraw = true;
            }
        }

        ui.add_space(COL_SPACING);

        ui.label("Select residues from:");

        for (i, chain) in mol.chains.iter().enumerate() {
            let mut color = Color32::GRAY;
            if let Some(i_sel) = state.ui.chain_to_pick_res {
                if i == i_sel {
                    color = COLOR_ACTIVE
                }
            }
            if ui
                .button(RichText::new(chain.id.clone()).color(color))
                .clicked()
            {
                // Toggle behavior.
                if let Some(sel_i) = state.ui.chain_to_pick_res {
                    if i == sel_i {
                        state.ui.chain_to_pick_res = None;
                    } else {
                        state.ui.chain_to_pick_res = Some(i);
                    }
                } else {
                    state.ui.chain_to_pick_res = Some(i);
                }
                state.volatile.ui_height = ui.ctx().used_size().y;
            }
        }

        if state.ui.chain_to_pick_res.is_some() {
            if ui.button("(None)").clicked() {
                state.ui.chain_to_pick_res = None;
                state.volatile.ui_height = ui.ctx().used_size().y;
            }
        }
    });
}

// todo: Update params A/R
fn draw_cli(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    redraw_mol: &mut bool,
    reset_cam: &mut bool,
    ui: &mut Ui,
) {
    ui.horizontal_wrapped(|ui| {
        ui.label("Out: ");
        let color = if state.ui.cmd_line_out_is_err {
            COLOR_OUT_ERROR
        } else {
            COLOR_OUT_NORMAL
        };
        ui.label(RichText::new(&state.ui.cmd_line_output).color(color));
    });

    ui.horizontal(|ui| {
        ui.label("In: ");
        let edit_resp = ui.add(
            TextEdit::singleline(&mut state.ui.cmd_line_input)
                .desired_width(200.)
                // Prevent losing focus on Tab;
                // .cursor_at_end(true)
                .lock_focus(true),
        );

        if edit_resp.changed() {
            // todo: Validate input and color-code?
        }

        ui.add_space(COL_SPACING / 2.);

        let button_clicked = ui.button(RichText::new("Submit")).clicked();
        // This  behavior of lost and has are due to the default EGUI beavhior of those keys.
        // We can use the `lock_focus(true)` method to prevent these, but we generally like them.
        let enter_pressed = edit_resp.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter));
        let tab_pressed = edit_resp.has_focus() && ui.input(|i| i.key_pressed(Key::Tab));
        let up_pressed = edit_resp.has_focus() && ui.input(|i| i.key_pressed(Key::ArrowUp));
        let dn_pressed = edit_resp.has_focus() && ui.input(|i| i.key_pressed(Key::ArrowDown));

        if tab_pressed && !state.ui.cmd_line_input.is_empty() {
            autocomplete_cli(&mut state.ui.cmd_line_input);

            // edit_resp.surrender_focus();
            // state.ui.cmd_line_input.push(' ');
            // state.ui.cmd_line_input.pop();
            // edit_resp.request_focus();
        }

        if up_pressed {
            if state.volatile.cli_input_selected != 0 {
                state.volatile.cli_input_selected -= 1;
            }
            if state.volatile.cli_input_history.len() > state.volatile.cli_input_selected {
                state.ui.cmd_line_input =
                    state.volatile.cli_input_history[state.volatile.cli_input_selected].clone();
            }
        }

        if dn_pressed {
            if state.volatile.cli_input_selected < state.volatile.cli_input_history.len() - 1 {
                state.volatile.cli_input_selected += 1;
            }
            if state.volatile.cli_input_history.len() > state.volatile.cli_input_selected {
                state.ui.cmd_line_input =
                    state.volatile.cli_input_history[state.volatile.cli_input_selected].clone();
            }
        }

        if (button_clicked || enter_pressed) && state.ui.cmd_line_input.len() >= 2 {
            // todo: Error color
            state.ui.cmd_line_output =
                match cli::handle_cmd(state, scene, engine_updates, redraw_mol, reset_cam) {
                    Ok(out) => {
                        state.ui.cmd_line_out_is_err = false;
                        out
                    }
                    Err(e) => {
                        eprintln!("Error processing command");
                        state.ui.cmd_line_out_is_err = true;
                        e.to_string()
                    }
                };

            state.ui.cmd_line_input = String::new();
            // Compensates for the default lose focus behavior; we still want the cursor to remain here.
            edit_resp.request_focus();
        }

        ui.add_space(COL_SPACING);
        residue_search(state, scene, redraw_mol, ui);
    });
}

fn docking(
    state: &mut State,
    scene: &mut Scene,
    redraw_lig: &mut bool,
    engine_updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    // let (Some(mol), Some(lig)) = (&state.molecule, &mut state.ligand) else {
    // // let Some(mol) = &state.molecule else {
    //     return;
    // };

    if state.molecule.is_none() || state.active_lig().is_none() {
        return;
    }

    // let mut docking_posit_update = None;

    ui.horizontal(|ui| {
        // let lig = state.active_lig_mut().unwrap();
        //
        // let Some(lig_data) = &mut lig.lig_data else {
        //     return;
        // };

        if ui.button("Find sites").clicked() {
            // let sites = find_docking_sites(mol);
            // for site in sites {
            //     println!("Docking site: {:?}", site);
            // }
        }

        if ui.button("Dock").clicked() {
            handle_docking(state, scene, ui, engine_updates);
        }

        if ui
            .button(RichText::new("Reset lig posit").color(COLOR_HIGHLIGHT))
            .on_hover_text(
                "Move the ligand to its absolute coordinates, e.g. as defined in \
                    its source Mol2 or SDF file.",
            )
            .clicked()
        {
            let lig = state.active_lig_mut().unwrap();
            lig.reset_posits();
            // state.mol_dynamics = None;

            if !lig.common.atom_posits.is_empty() {
                // docking_posit_update = Some(lig.common.atom_posits[0].into());
                // docking_init_changed = true;
            }

            let center = state.molecule.as_ref().unwrap().center;
            move_cam_to_lig(state, scene, center, engine_updates)
        }

        // if docking_init_changed {
        // *redraw_lig = true;
        // set_docking_light(scene, Some(&lig.docking_site));
        // engine_updates.lighting = true;
        // }
    });

    // if let Some(posit) = docking_posit_update {
    //     // state.update_docking_site(posit);
    //     state.update_save_prefs(false);
    // }
}

fn residue_search(state: &mut State, scene: &mut Scene, redraw: &mut bool, ui: &mut Ui) {
    let (btn_text_p, btn_text_n, search_text) = match state.ui.view_sel_level {
        ViewSelLevel::Atom => ("Prev atom", "Next atom", "Find atom:"),
        ViewSelLevel::Residue => ("Prev AA", "Next AA", "Find res:"),
    };

    ui.label(search_text);
    if ui
        .add(TextEdit::singleline(&mut state.ui.atom_res_search).desired_width(60.))
        .changed()
    {
        select_from_search(state);
        *redraw = true;
    }

    if state.molecule.is_some() {
        if ui
            .button(btn_text_p)
            .on_hover_text("Hotkey: Left arrow")
            .clicked()
        {
            cycle_selected(state, scene, true);
            *redraw = true;
        }
        // todo: DRY

        if ui
            .button(btn_text_n)
            .on_hover_text("Hotkey: Right arrow")
            .clicked()
        {
            cycle_selected(state, scene, false);
            *redraw = true;
        }

        ui.add_space(COL_SPACING * 2.);

        let dock_tools_text = if state.ui.show_docking_tools {
            "Hide docking tools"
        } else {
            "Show docking tools (Broken/WIP)"
        };

        if ui.button(RichText::new(dock_tools_text)).clicked() {
            state.ui.show_docking_tools = !state.ui.show_docking_tools;
        }

        ui.add_space(COL_SPACING / 2.);

        let dock_seq_text = if state.ui.show_aa_seq {
            "Hide seq"
        } else {
            "Show seq"
        };

        if ui.button(RichText::new(dock_seq_text)).clicked() {
            state.ui.show_aa_seq = !state.ui.show_aa_seq;
        }
    }
}

fn add_aa_seq(seq_text: &str, ui: &mut Ui) {
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(seq_text).color(Color32::LIGHT_BLUE));
    });
}

fn selection_section(state: &mut State, redraw: &mut bool, ui: &mut Ui) {
    // todo: DRY with view.
    ui.horizontal_wrapped(|ui| {
        section_box().show(ui, |ui| {
            let help_text = "Hotkeys: square brackets [ ] to cycle";
            ui.label("View/Select:").on_hover_text(help_text);
            let prev_view = state.ui.view_sel_level;

            // Ideally hover text here too, but I'm not sure how.
            ComboBox::from_id_salt(1)
                .width(80.)
                .selected_text(state.ui.view_sel_level.to_string())
                .show_ui(ui, |ui| {
                    for view in &[ViewSelLevel::Atom, ViewSelLevel::Residue] {
                        ui.selectable_value(&mut state.ui.view_sel_level, *view, view.to_string());
                    }
                }).response.on_hover_text(help_text);

            if state.ui.view_sel_level != prev_view {
                *redraw = true;
                // If we change from atom to res, select the prev-selected atom's res. If vice-versa,
                // select that residue's Cα.
                // state.ui.selection = Selection::None;
                if let Some(mol) = &state.molecule {
                    match state.ui.view_sel_level {
                        ViewSelLevel::Residue => {
                            state.ui.selection = match state.ui.selection {
                                Selection::Atom(i) => {
                                    Selection::Residue(mol.common.atoms[i].residue.unwrap_or_default())
                                }
                                _ => Selection::None,
                            };
                        }
                        ViewSelLevel::Atom => {
                            state.ui.selection = match state.ui.selection {
                                // It seems [0] is often N, and [1] is Cα
                                Selection::Residue(i) => {
                                    if mol.residues[i].atoms.len() <= 2 {
                                        Selection::Atom(mol.residues[i].atoms[1])
                                    } else {
                                        Selection::None
                                    }
                                }

                                _ => Selection::None,
                            };
                        }
                    }
                }
            }

            // Buttons to alter the color profile, e.g. for res position, or partial charge.
            ui.add_space(COL_SPACING / 2.);
            match state.ui.view_sel_level {
                ViewSelLevel::Atom => {
                    let color = if state.ui.atom_color_by_charge {
                        COLOR_ACTIVE
                    } else {
                        COLOR_INACTIVE
                    };

                    if ui
                        .button(RichText::new("Color by q").color(color))
                        .on_hover_text("Color the atom by partial charge, instead of element-specific colors")
                        .clicked()
                    {
                        state.ui.atom_color_by_charge = !state.ui.atom_color_by_charge;
                        state.ui.view_sel_level = ViewSelLevel::Atom;
                        *redraw = true;
                    }
                }
                ViewSelLevel::Residue => {
                    let color = if state.ui.res_color_by_index {
                        COLOR_ACTIVE
                    } else {
                        COLOR_INACTIVE
                    };

                    if ui
                        .button(RichText::new("Color by res #").color(color))
                        .on_hover_text("Color the atom by its position in the primary sequence, instead of residue (e.g. AA) -specific colors")
                        .clicked()
                    {
                        state.ui.res_color_by_index = !state.ui.res_color_by_index;
                        state.ui.view_sel_level = ViewSelLevel::Residue;
                        *redraw = true;
                    }
                }
            }

            ui.add_space(COL_SPACING);

            let help = "Hide all atoms not near the selection";
            ui.label("Nearby sel only:").on_hover_text(help);
            if ui.checkbox(&mut state.ui.show_near_sel_only, "")
                .on_hover_text(help)
                .changed() {
                *redraw = true;

                // todo: For now, only allow one of near sel/lig
                if state.ui.show_near_sel_only {
                    state.ui.show_near_lig_only = false
                }
            }

            if state.active_lig().is_some() {
                let help = "Hide all atoms not near the ligand";
                ui.label("Nearby lig only:").on_hover_text(help);
                if ui.checkbox(&mut state.ui.show_near_lig_only, "")
                    .on_hover_text(help)
                    .changed() {
                    *redraw = true;

                    // todo: For now, only allow one of near sel/lig
                    if state.ui.show_near_lig_only {
                        state.ui.show_near_sel_only = false
                    }
                }
            }

            if state.ui.show_near_sel_only || state.ui.show_near_lig_only {
                ui.label("Dist:");
                let dist_prev = state.ui.nearby_dist_thresh;
                ui.spacing_mut().slider_width = 160.;

                ui.add(Slider::new(
                    &mut state.ui.nearby_dist_thresh,
                    NEARBY_THRESH_MIN..=NEARBY_THRESH_MAX,
                ));

                if state.ui.nearby_dist_thresh != dist_prev {
                    *redraw = true;
                }
            }
        });

        if let Some(mol) = &state.molecule {
            mol_data::selected_data(mol, &state.ligands, &state.ui.selection, ui);
        }
    });
}

fn mol_descrip(mol: &MoleculeGenericRef, ui: &mut Ui) {
    ui.heading(RichText::new(mol.common().ident.clone()).color(Color32::GOLD));

    ui.label(format!("{} atoms", mol.common().atoms.len()));

    if let MoleculeGenericRef::Peptide(m) = mol {
        if let Some(method) = m.experimental_method {
            ui.label(method.to_str_short());
        }
    }

    if let Some(title) = mol.common().metadata.get("_struct.title") {
        // Limit size to prevent UI problems.
        let mut title_abbrev: String = title.chars().take(MAX_TITLE_LEN).collect();

        if title_abbrev.len() != title.len() {
            title_abbrev += "...";

            // Allow hovering to see the full title.
            ui.label(RichText::new(title_abbrev).color(Color32::WHITE).size(12.))
                .on_hover_text(title);
        } else {
            ui.label(RichText::new(title_abbrev).color(Color32::WHITE).size(12.));
        }
    }

    if mol.common().ident.len() <= 5 {
        // todo: You likely need a better approach.
        if ui
            .button("View on RCSB")
            .on_hover_text("Open a web browser to the RCSB PDB page for this molecule.")
            .clicked()
        {
            rcsb::open_overview(&mol.common().ident);
        }
    }
}

fn view_settings(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    redraw: &mut bool,
    ui: &mut Ui,
) {
    section_box().show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label("View:");
            let prev_view = state.ui.mol_view;
            ComboBox::from_id_salt(0)
                .width(80.)
                .selected_text(state.ui.mol_view.to_string())
                .show_ui(ui, |ui| {
                    for view in &[
                        MoleculeView::Backbone,
                        MoleculeView::Sticks,
                        MoleculeView::BallAndStick,
                        // MoleculeView::Cartoon,
                        MoleculeView::SpaceFill,
                        MoleculeView::Surface,
                        MoleculeView::Dots,
                    ] {
                        ui.selectable_value(&mut state.ui.mol_view, *view, view.to_string());
                    }
                });

            if state.ui.mol_view != prev_view {
                *redraw = true;
            }

            ui.add_space(COL_SPACING);

            ui.label("Vis:");

            misc::vis_check(
                &mut state.ui.visibility.hide_non_hetero,
                "Peptide",
                ui,
                redraw,
            );
            misc::vis_check(&mut state.ui.visibility.hide_hetero, "Hetero", ui, redraw);

            ui.add_space(COL_SPACING / 2.);

            if !state.ui.visibility.hide_non_hetero {
                // Subset of peptide.
                misc::vis_check(
                    &mut state.ui.visibility.hide_sidechains,
                    "Sidechains",
                    ui,
                    redraw,
                );
            }

            misc::vis_check(&mut state.ui.visibility.hide_hydrogen, "H", ui, redraw);

            // We allow toggling water now regardless of hide hetero, as it's part of our MD sim.
            // if !state.ui.visibility.hide_hetero {
            // Subset of hetero.
            let water_prev = state.ui.visibility.hide_water;
            misc::vis_check(&mut state.ui.visibility.hide_water, "Water", ui, redraw);

            if !state.nucleic_acids.is_empty() {
                misc::vis_check(
                    &mut state.ui.visibility.hide_nucleic_acids,
                    "Nucleic acids",
                    ui,
                    redraw,
                );
            }
            // }

            if let Some(md) = &state.mol_dynamics {
                if state.ui.visibility.hide_water != water_prev {
                    let snap = &md.snapshots[0];

                    draw_water(
                        scene,
                        &snap.water_o_posits,
                        &snap.water_h0_posits,
                        &snap.water_h1_posits,
                        state.ui.visibility.hide_water,
                    );
                }
            }

            if state.active_lig().is_some() {
                let color = misc::active_color(!state.ui.visibility.hide_ligand);
                if ui.button(RichText::new("Lig").color(color)).clicked() {
                    state.ui.visibility.hide_ligand = !state.ui.visibility.hide_ligand;

                    if state.ui.visibility.hide_ligand {
                        scene.entities.retain(|ent| {
                            ent.class != EntityType::Ligand as u32
                                && ent.class != EntityType::DockingSite as u32
                        });
                    } else {
                        draw_all_ligs(state, scene);
                    }

                    engine_updates.entities = true;
                    engine_updates.lighting = true; // docking light.
                }
            }

            misc::vis_check(&mut state.ui.visibility.hide_h_bonds, "H bonds", ui, redraw);
            // vis_check(&mut state.ui.visibility.dim_peptide, "Dim peptide", ui, redraw);

            if state.active_lig().is_some() {
                ui.add_space(COL_SPACING / 2.);
                // Not using `vis_check` for this because its semantics are inverted.
                let color = misc::active_color(state.ui.visibility.dim_peptide);
                if ui
                    .button(RichText::new("Dim peptide").color(color))
                    .clicked()
                {
                    state.ui.visibility.dim_peptide = !state.ui.visibility.dim_peptide;
                    *redraw = true;
                }
            }

            // todo temp
            if ui.button("Load DNA").clicked() {
                if let Some(mol) = &state.molecule {
                    state.nucleic_acids = vec![MoleculeNucleicAcid::from_peptide(
                        &mol,
                        NucleicAcidType::Dna,
                        Strands::Single,
                    )];
                }
                draw_nucleic_acid(state, scene);
                engine_updates.entities = true;
            }

            if ui.button("Load RNA").clicked() {
                if let Some(mol) = &state.molecule {
                    state.nucleic_acids = vec![MoleculeNucleicAcid::from_peptide(
                        &mol,
                        NucleicAcidType::Rna,
                        Strands::Single,
                    )];
                }
                draw_nucleic_acid(state, scene);
                engine_updates.entities = true;
            }

            if let Some(mol) = &state.molecule {
                if let Some(dens) = &mol.elec_density {
                    let mut redraw_dens = false;
                    misc::vis_check(
                        &mut state.ui.visibility.hide_density_point_cloud,
                        "Density",
                        ui,
                        &mut redraw_dens,
                    );

                    if redraw_dens {
                        if state.ui.visibility.hide_density_point_cloud {
                            scene
                                .entities
                                .retain(|ent| ent.class != EntityType::DensityPoint as u32);
                        } else {
                            draw_density_point_cloud(&mut scene.entities, dens);
                        }
                        engine_updates.entities = true;
                    }

                    let mut redraw_dens_surface = false;
                    misc::vis_check(
                        &mut state.ui.visibility.hide_density_surface,
                        "Density sfc",
                        ui,
                        &mut redraw_dens_surface,
                    );

                    if !state.ui.visibility.hide_density_surface {
                        let iso_prev = state.ui.density_iso_level;

                        ui.spacing_mut().slider_width = 300.;
                        ui.add(Slider::new(
                            &mut state.ui.density_iso_level,
                            DENS_ISO_MIN..=DENS_ISO_MAX,
                        ))
                        .on_hover_text("The density value at which to draw the ISO surface");
                        if state.ui.density_iso_level != iso_prev {
                            state.volatile.flags.make_density_iso_mesh = true;
                        }
                    }

                    if redraw_dens_surface {
                        if state.ui.visibility.hide_density_surface {
                            let _ = &mut scene
                                .entities
                                .retain(|ent| ent.class != EntityType::DensitySurface as u32);
                        } else {
                            draw_density_surface(&mut scene.entities);
                        }
                        engine_updates.entities = true;
                    }
                }
            }
        });
    });
}

fn settings(state: &mut State, scene: &mut Scene, ui: &mut Ui) {
    let popup_id = ui.make_persistent_id("settings_popup");
    Popup::new(
        popup_id,
        ui.ctx().clone(),
        PopupAnchor::Position(Pos2::new(60., 60.)),
        ui.layer_id(),
    )
    .align(RectAlign::TOP)
    .open(true)
    .gap(4.0)
    .show(|ui| {
        ui.horizontal(|ui| {
            ui.heading("Settings");
            ui.add_space(COL_SPACING);
            // todo: Make this consistent with your other controls.
            ui.label("MSAA (Restart the program to take effect):");

            let msaa_prev = state.to_save.msaa;
            ComboBox::from_id_salt(10)
                .width(40.)
                .selected_text(state.to_save.msaa.to_str())
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut state.to_save.msaa,
                        MsaaSetting::None,
                        MsaaSetting::None.to_str(),
                    );
                    ui.selectable_value(
                        &mut state.to_save.msaa,
                        MsaaSetting::Four,
                        MsaaSetting::Four.to_str(),
                    );
                });

            if state.to_save.msaa != msaa_prev {
                state.update_save_prefs(false);
            }

            ui.add_space(COL_SPACING);
            ui.label("Movement speed:");
            if ui
                .add(TextEdit::singleline(&mut state.ui.movement_speed_input).desired_width(32.))
                .changed()
            {
                if let Ok(v) = &mut state.ui.movement_speed_input.parse::<u8>() {
                    state.to_save.movement_speed = *v;
                    scene.input_settings.move_sens = *v as f32;

                    state.update_save_prefs(false);
                } else {
                    // reset
                    state.ui.movement_speed_input = state.to_save.movement_speed.to_string();
                }
            }

            ui.add_space(COL_SPACING / 2.);
            ui.label("Rotation sensitivity:");
            if ui
                .add(TextEdit::singleline(&mut state.ui.rotation_sens_input).desired_width(32.))
                .changed()
            {
                if let Ok(v) = &mut state.ui.rotation_sens_input.parse::<u8>() {
                    state.to_save.rotation_sens = *v;
                    scene.input_settings.rotate_sens = *v as f32 / 100.;

                    state.update_save_prefs(false);
                } else {
                    // reset
                    state.ui.rotation_sens_input = state.to_save.rotation_sens.to_string();
                }
            }

            ui.add_space(COL_SPACING / 2.);
            if ui.button("Reset sensitivities").clicked() {
                state.to_save.movement_speed = MOVEMENT_SENS as u8;
                state.ui.movement_speed_input = state.to_save.movement_speed.to_string();
                scene.input_settings.move_sens = MOVEMENT_SENS;

                state.to_save.rotation_sens = (ROTATE_SENS * 100.) as u8;
                state.ui.rotation_sens_input = state.to_save.rotation_sens.to_string();
                scene.input_settings.rotate_sens = ROTATE_SENS;

                state.update_save_prefs(false);
            }
        });

        ui.add_space(ROW_SPACING);

        if ui
            .button(RichText::new("Close").color(Color32::LIGHT_RED))
            .clicked()
        {
            state.ui.popup.show_settings = false;
        }
    });
}

/// This function draws the (immediate-mode) GUI.
/// [UI items](https://docs.rs/egui/latest/egui/struct.Ui.html)
pub fn ui_handler(state: &mut State, ctx: &Context, scene: &mut Scene) -> EngineUpdates {
    let mut engine_updates = EngineUpdates::default();

    // Checks each frame; takes action based on time since last save.
    check_prefs_save(state);

    // todo: Trying to set popup color; Not working
    // let mut style = (*ctx.style()).clone();
    // style.visuals.widgets.noninteractive.bg_fill = COLOR_POPUP;
    // ctx.set_style(style);

    // return  engine_updates;
    let mut redraw_mol = false;
    let mut redraw_lig = false;
    let mut reset_cam = false;

    // For getting DT for certain buttons when held. Does not seem to be the same as the 3D render DT.
    let start = Instant::now();

    TopBottomPanel::top("0").show(ctx, |ui| {
        ui.spacing_mut().slider_width = 120.;

        handle_input(
            state,
            ui,
            &mut redraw_mol,
            &mut reset_cam,
            &mut engine_updates,
        );

        if state.ui.popup.show_settings {
            settings(state, scene, ui);
        }

        ui.horizontal(|ui| {
            let color_settings = if state.ui.popup.show_settings {
                Color32::LIGHT_RED
            } else {
                Color32::GRAY
            };
            if ui
                .button(RichText::new("⚙").color(color_settings))
                .clicked()
            {
                state.ui.popup.show_settings = !state.ui.popup.show_settings;
            }

            let metadata_loaded = false; // avoids borrow error.
            if let Some(mol) = &mut state.molecule {
                mol_descrip(&MoleculeGenericRef::Peptide(&mol), ui);

                if ui.button("Close").clicked() {
                    close_peptide(state, scene, &mut engine_updates);
                }
                ui.add_space(COL_SPACING);
            }

            let color_open_tools = if state.molecule.is_none() {
                Color32::GOLD
            } else {
                COLOR_INACTIVE
            };
            if ui
                .button(RichText::new("Open").color(color_open_tools))
                .clicked()
            {
                state.volatile.dialogs.load.pick_file();
            }

            let mut dm_loaded = None; // avoids a double-borrow error.
            if let Some(mol) = &mut state.molecule {
                // let color = if state.to_save.last_peptide_opened.is_none() {
                //     COLOR_ATTENTION
                // } else {
                //     Color32::GRAY
                // };
                // todo: Put a form of this back.
                let color = Color32::GRAY;

                if ui.button(RichText::new("Save").color(color)).clicked() {
                    let extension = "cif";

                    let filename = {
                        let name = if mol.common.ident.is_empty() {
                            "molecule".to_string()
                        } else {
                            mol.common.ident.clone()
                        };
                        format!("{name}.{extension}")
                    };

                    state.volatile.dialogs.save.config_mut().default_file_name =
                        filename.to_string();
                    state.volatile.dialogs.save.save_file();
                }

                // todo: Move these A/R. LIkely in a sub menu.
                if let Some(files_avail) = &mol.rcsb_files_avail {
                    // if files_avail.structure_factors {
                    //     if ui
                    //         .button(RichText::new("SF").color(COLOR_HIGHLIGHT))
                    //         .clicked()
                    //     {
                    //         match rcsb::load_structure_factors_cif(&mol.common.ident) {
                    //             Ok(data) => {
                    //                 // println!("SF data: {:?}", data);
                    //             }
                    //             Err(_) => {
                    //                 let msg = format!(
                    //                     "Error loading RCSB structure factors for {:?}",
                    //                     &mol.common.ident
                    //                 );
                    //
                    //                 handle_err(&mut state.ui, msg);
                    //             }
                    //         }
                    //     }
                    // }
                    if files_avail.validation_2fo_fc {
                        if ui
                            .button(RichText::new("Fetch elec ρ").color(COLOR_HIGHLIGHT))
                            .on_hover_text("Load 2fo-fc electron density data from RCSB PDB. Convert to CCP4 map format and display.")
                            .clicked()
                        {
                            // todo: For now, we rely on Gemmi being available on the Path.
                            // todo: We will eventually get our own reflections loader working.

                            match density_from_2fo_fc_rcsb_gemmi(&mol.common.ident, gemmi_path()) {
                                Ok(dm) => {
                                    dm_loaded = Some(dm);
                                    handle_success(&mut state.ui, "Loaded density data from RSCB".to_owned());
                                }
                                Err(e) => {
                                    let msg = format!(
                                        "Error loading or processing RCSB 2fo-fc map for {:?}: {e:?}",
                                        &mol.common.ident
                                    );
                                    handle_err(&mut state.ui, msg);
                                }
                            }
                        }
                    }
                    // todo: Add these if you end up with a way to use them. We currently use 2fo-fc only.
                    // if files_avail.validation_fo_fc {
                    //     if ui
                    //         .button(RichText::new("fo-fc").color(COLOR_HIGHLIGHT))
                    //         .clicked()
                    //     {
                    //         match rcsb::load_validation_fo_fc_cif(&mol.common.ident) {
                    //             Ok(data) => {
                    //                 // println!("SF data: {:?}", data);
                    //             }
                    //             Err(_) => {
                    //                 let msg = format!(
                    //                     "Error loading RCSB fo-fc map for {:?}",
                    //                     &mol.common.ident
                    //                 );
                    //                 handle_err(&mut state.ui, msg);
                    //             }
                    //         }
                    //     }
                    // }
                    //
                    // if files_avail.validation {
                    //     if ui
                    //         .button(RichText::new("Val").color(COLOR_HIGHLIGHT))
                    //         .clicked()
                    //     {
                    //         match rcsb::load_validation_cif(&mol.common.ident) {
                    //             Ok(data) => {
                    //                 // println!("VAL DATA: {:?}", data);
                    //             }
                    //             Err(_) => {
                    //                 let msg = format!(
                    //                     "Error loading RCSB validation for {:?}",
                    //                     &mol.common.ident
                    //                 );
                    //                 handle_err(&mut state.ui, msg);
                    //             }
                    //         }
                    //     }
                    // }

                    if files_avail.map {
                        if ui
                            .button(RichText::new("Map").color(COLOR_HIGHLIGHT))
                            .clicked()
                        {
                            match rcsb::load_map(&mol.common.ident) {
                                Ok(data) => {
                                    let mut cursor = Cursor::new(data);
                                    match DensityMap::new(&mut cursor) {
                                        Ok(dm) => {
                                            dm_loaded = Some(dm);
                                            println!("Succsesfully loaded Map rom RSCB.");
                                        }
                                        Err(_) => {
                                            let msg = format!(
                                                "Error loading RCSB Map for {:?}",
                                                &mol.common.ident
                                            );
                                            handle_err(&mut state.ui, msg);
                                        }
                                    }
                                }
                                Err(_) => {
                                    let msg =
                                        format!("Error loading RCSB Map for {:?}", &mol.common.ident);
                                    handle_err(&mut state.ui, msg);
                                }
                            }
                        }
                    }
                }
            }

            if let Some(dm) = dm_loaded {
                state.load_density(dm);
            }

            if let Some(lig) = &state.active_lig() {
                // Highlight the button if we haven't saved this to file, e.g. if opened from online.
                // let color = if state.to_save.last_ligand_opened.is_none() {
                //     COLOR_ATTENTION
                // } else {
                //     Color32::GRAY
                // };
                // todo: Put a form of this back.
                let color = Color32::GRAY;

                if ui.button(RichText::new("Save lig").color(color)).clicked() {
                    let extension = "mol2"; // The default; more robust than SDF.

                    let filename = {
                        let name = if lig.common.ident.is_empty() {
                            "molecule".to_string()
                        } else {
                            lig.common.ident.clone()
                        };
                        format!("{name}.{extension}")
                    };

                    state.volatile.dialogs.save.config_mut().default_file_name =
                        filename.to_string();

                    state.volatile.dialogs.save.save_file();
                }
            }

            if metadata_loaded {
                state.update_save_prefs(false);
            }

            ui.add_space(COL_SPACING);

            let query_help = "From RCSB PDB, PubChem, DrugBank, or Amber Geostd";
            ui.label(RichText::new("Query databases (ident):").color(color_open_tools))
                .on_hover_text(query_help);

            let edit_resp = ui
                .add(TextEdit::singleline(&mut state.ui.db_input).desired_width(40.))
                .on_hover_text(query_help);

            if state.ui.db_input.len() >= 4 {
                let enter_pressed =
                    edit_resp.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter));
                let button_clicked = ui.button("Download from RCSB").clicked();

                // if response.lost_focus() && (button_clicked || enter_pressed)
                if (button_clicked || enter_pressed) && state.ui.db_input.trim().len() == 4 {
                    let ident = state.ui.db_input.clone().trim().to_owned();

                    load_atom_coords_rcsb(
                        &ident,
                        state,
                        scene,
                        &mut engine_updates,
                        &mut redraw_mol,
                        &mut reset_cam,
                    );

                    state.ui.db_input = String::new();
                }
            }
            if state.ui.db_input.len() == 3 {
                let enter_pressed =
                    edit_resp.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter));
                let button_clicked = ui.button("Download Amber Geostd").clicked();

                if button_clicked || enter_pressed {
                    let db_input = &state.ui.db_input.clone(); // Avoids a double borrow.
                    state.load_geostd_mol_data(&db_input, true, true, &mut redraw_lig);

                    state.ui.db_input = String::new();
                }
            }

            if state.ui.db_input.len() >= 4 {
                if state.ui.db_input.to_uppercase().starts_with("DB") {
                    if ui.button("Download from DrugBank").clicked() {
                        match load_sdf_drugbank(&state.ui.db_input) {
                            Ok(mol) => {
                                open_lig(state, mol);
                                redraw_lig = true;
                                reset_cam = true;
                            }
                            Err(_e) => {
                                let msg = "Error loading SDF file".to_owned();
                                handle_err(&mut state.ui, msg);
                            }
                        }
                    }
                }

                if ui.button("Download from PubChem").clicked() {
                    match load_sdf_pubchem(&state.ui.db_input) {
                        Ok(mol) => {
                            open_lig(state, mol);
                            redraw_lig = true;
                            reset_cam = true;
                        }
                        Err(_e) => {
                            let msg = "Error loading SDF file".to_owned();
                            handle_err(&mut state.ui, msg);
                        }
                    }
                }
            }

            if state.molecule.is_none() && state.active_lig().is_none() {
                ui.add_space(COL_SPACING / 2.);
                if ui
                    .button(RichText::new("I'm feeling lucky 🍀").color(color_open_tools))
                    .on_hover_text("Open a random recently-uploaded protein from RCSB PDB.")
                    .clicked()
                {
                    if let Ok(ident) = rcsb::get_newly_released() {
                        load_atom_coords_rcsb(
                            &ident,
                            state,
                            scene,
                            &mut engine_updates,
                            &mut redraw_mol,
                            &mut reset_cam,
                        );
                    }
                }
            }
        });

        ui.add_space(ROW_SPACING);

        let mut close_ligand = false; // to avoid borrow error.
        disp_lig_data(state, scene, ui, &mut redraw_lig, &mut close_ligand, &mut engine_updates);

        ui.add_space(ROW_SPACING);
        selection_section(state, &mut redraw_mol, ui);

        ui.add_space(ROW_SPACING);

        ui.horizontal_wrapped(|ui| {
            cam_controls(scene, state, &mut engine_updates, ui);

            cam_snapshots(state, scene, &mut engine_updates, ui);
        });

        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                view_settings(state, scene, &mut engine_updates, &mut redraw_mol, ui);
                chain_selector(state, &mut redraw_mol, ui);
                // todo: Show hide based on AaCategory? i.e. residue.amino_acid.category(). Hydrophilic, acidic etc.

                residue_selector(state, scene, &mut redraw_mol, ui);
            });
        });

        // ui.add_space(ROW_SPACING);

        md_setup(state, scene, &mut engine_updates, &mut redraw_lig, ui);

        if state.ui.show_docking_tools {
            ui.add_space(ROW_SPACING);

            docking(state, scene, &mut redraw_lig, &mut engine_updates, ui);
        }

        // todo: Allow switching between chains and secondary-structure features here.

        ui.add_space(ROW_SPACING / 2.);

        if state.ui.show_aa_seq {
            if state.molecule.is_some() {
                add_aa_seq(&state.volatile.aa_seq_text, ui);
            }
        }

        // todo: Move A/r.
        draw_cli(
            state,
            scene,
            &mut engine_updates,
            &mut redraw_mol,
            &mut reset_cam,
            ui,
        );

        if state.ui.popup.show_get_geostd {
            let popup_id = ui.make_persistent_id("no_ff_params_popup");
            Popup::new(
                popup_id,
                ui.ctx().clone(), // todo clone???
                // PopupAnchor::PointerFixed,
                // PopupAnchor::ParentRect(),
                PopupAnchor::Position(Pos2::new(60., 60.)),
                ui.layer_id(), // draw on top of the current layer
            )
                .align(RectAlign::TOP)
                // .align(RectAlign::BOTTOM_START)
                .open(true)
                .gap(4.0)
                .show(|ui| {
                    // These vars avoid dbl borrow.
                    let load_ff = !state.active_lig().as_ref().unwrap().ff_params_loaded;
                    let load_frcmod = !state.active_lig().as_ref().unwrap().frcmod_loaded;

                    let Some(lig) = state.active_lig_mut() else {
                        return;
                    };
                    let mut msg = String::from("Not ready for dynamics: ");

                    if !lig.ff_params_loaded {
                        msg += "No FF params or partial charges are present on this ligand."
                    }
                    if !lig.frcmod_loaded {
                        msg += "No FRCMOD parameters loaded for this ligand."
                    }

                    ui.label(RichText::new(msg).color(Color32::LIGHT_RED));

                    ui.add_space(ROW_SPACING);

                    // todo: What about cases where a SDF from pubchem or drugbank doesn't include teh name used by Amber?
                    if ui.button("Check online").clicked() {
                        // let Some(lig) = state.ligand.as_mut() else {
                        //     return;
                        // };

                        match amber_geostd::find_mols(&lig.common.ident) {
                            Ok(data) => {
                                state.ui.popup.get_geostd_items = data;
                            }
                            Err(e) => handle_err(
                                &mut state.ui,
                                format!("Problem loading mol data online: {e:?}"),
                            ),
                        }
                    }

                    // This clone is annoying; db borrow.
                    let items = state.ui.popup.get_geostd_items.clone();
                    for mol_data in items {
                        if ui
                            .button(
                                RichText::new(format!("Load params for {}", mol_data.ident))
                                    .color(COLOR_HIGHLIGHT),
                            )
                            .clicked()
                        {
                            state.load_geostd_mol_data(
                                &mol_data.ident,
                                load_ff,
                                load_frcmod,
                                &mut redraw_lig,
                            );

                            state.ui.popup.show_get_geostd = false;
                        }
                    }

                    ui.add_space(ROW_SPACING);

                    if ui
                        .button(RichText::new("Close").color(Color32::LIGHT_RED))
                        .clicked()
                    {
                        state.ui.popup.show_get_geostd = false;
                    }
                });
        }

        if state.ui.popup.show_associated_structures {
            let mut associated_structs = Vec::new();
            if let Some(lig) = state.active_lig() {
                // todo: I don't like this clone, but not sure how else to do it.
                associated_structs = lig.associated_structures.clone();
            }

            if state.active_lig().is_some() {
                let popup_id = ui.make_persistent_id("associated_structs_popup");
                Popup::new(
                    popup_id,
                    ui.ctx().clone(),
                    PopupAnchor::Position(Pos2::new(300., 60.)),
                    ui.layer_id(), // draw on top of the current layer
                )
                    .align(RectAlign::TOP)
                    .open(true)
                    .gap(4.0)
                    .show(|ui| {
                        for s in &associated_structs {
                            ui.horizontal(|ui| {
                                if ui
                                    .button(
                                        RichText::new(format!("{}", s.pdb_id)).color(COLOR_HIGHLIGHT),
                                    )
                                    .clicked()
                                {
                                    rcsb::open_overview(&s.pdb_id);
                                }
                                ui.add_space(COL_SPACING);

                                if ui
                                    .button(
                                        RichText::new(format!("Open this protein"))
                                            .color(COLOR_HIGHLIGHT),
                                    )
                                    .clicked()
                                {
                                    load_atom_coords_rcsb(
                                        &s.pdb_id,
                                        state,
                                        scene,
                                        &mut engine_updates,
                                        &mut redraw_mol,
                                        &mut reset_cam,
                                    );
                                }
                            });

                            ui.label(RichText::new(format!("{}", s.description)));

                            ui.add_space(ROW_SPACING);
                        }

                        ui.add_space(ROW_SPACING);
                    });

                ui.add_space(ROW_SPACING);
                if ui
                    .button(RichText::new("Close").color(Color32::LIGHT_RED))
                    .clicked()
                {
                    state.ui.popup.show_associated_structures = false;
                }
            }
        }

        // -------UI above; clean-up items (based on flags) below

        if close_ligand {
            if let Some(i) = state.volatile.active_lig {
                close_lig(i, state, scene, &mut engine_updates);
            }
        }

        if let Some(path) = &state.volatile.dialogs.load.take_picked() {
            if let Err(e) = load_file(
                path,
                state,
                &mut redraw_mol,
                &mut reset_cam,
                &mut engine_updates,
            ) {
                handle_err(&mut state.ui, e.to_string());
            }

            set_flashlight(scene);
            engine_updates.lighting = true;
        }

        if let Some(path) = &state.volatile.dialogs.save.take_picked() {
            state.save(path).ok();
        }

        // if let Some(path) = &state.volatile.dialogs.autodock_path.take_picked() {
        //     state.ui.autodock_path_valid = check_adv_avail(path);
        //     if state.ui.autodock_path_valid {
        //         state.to_save.autodock_vina_path = Some(path.to_owned());
        //         state.update_save_prefs();
        //     }
        // }

        // if let Some(path_dir) = &state.volatile.dialogs.save_pdbqt.take_picked() {
        //     if let Some(mol) = &mut state.molecule {
        //         let filename = format!("{}_target.pdbqt", mol.common.ident);
        //         let path = Path::new(path_dir).join(filename);
        //
        //         // todo: You will likely need to add charges earlier, so you can view their data in the UI.
        //         // setup_partial_charges(&mut mol.common.atoms, PartialChargeType::Gasteiger);
        //         // create_partial_charges(&mut mol.common.atoms);
        //
        //         if mol.save_pdbqt(&path, None).is_err() {
        //             eprintln!("Error saving PDBQT target");
        //         }
        //     }
        //
        //     if let Some(lig) = &mut state.ligand {
        //         let filename = format!("{}_ligand.pdbqt", lig.molecule.ident);
        //         let path = Path::new(path_dir).join(filename);
        //
        //         // create_partial_charges(&mut lig.molecule.atoms);
        //         // setup_partial_charges(&mut lig.molecule.atoms, PartialChargeType::Gasteiger);
        //
        //         if lig.molecule.save_pdbqt(&path, None).is_err() {
        //             eprintln!("Error saving PDBQT ligand");
        //         }
        //     }
        // }

        if redraw_mol {
            draw_peptide(state, scene);
            draw_all_ligs(state, scene); // todo: Hmm.

            if let Some(mol) = &state.molecule {
                set_window_title(&mol.common.ident, scene);
            }

            engine_updates.entities = true;

            // For docking light, but may be overkill here.
            if state.active_lig().is_some() {
                engine_updates.lighting = true;
            }
        }

        if redraw_lig {
            draw_all_ligs(state, scene);

            engine_updates.entities = true;

            // For docking light, but may be overkill here.
            if state.active_lig().is_some() {
                engine_updates.lighting = true;
            }
        }

        // Perform cleanup.
        if reset_cam {
            if let Some(mol) = &state.molecule {
                reset_camera(scene, &mut state.ui.view_depth, &mut engine_updates, mol);
            }
        }
    });

    state.volatile.dialogs.load.update(ctx);
    state.volatile.dialogs.save.update(ctx);
    state.volatile.dialogs.autodock_path.update(ctx);

    // todo: Appropriate place for this?
    if state.volatile.inputs_commanded.inputs_present() {
        set_flashlight(scene);
        engine_updates.lighting = true;
    }

    state.ui.dt_render = start.elapsed().as_secs_f32();

    if !INIT_COMPLETE.swap(true, Ordering::AcqRel) {
        if state.volatile.ui_height < f32::EPSILON {
            state.volatile.ui_height = ctx.used_size().y;
        }

        // todo: Move to new_mol_loaded code block?
        if state.molecule.is_some() {
            // if let Some(mol) = &state.molecule {

            // todo: Put back A/R. Dbl borrow error
            // if let Some(lig) = state.active_lig_mut() {
            //     if let Some(data) = &mut lig.lig_data {
            //         if data.anchor_atom >= lig.common.atoms.len() {
            //             let msg = "Error positioning ligand atoms; anchor outside len".to_owned();
            //             handle_err(&mut state.ui, msg);
            //         } else {
            //             lig.position_atoms(None);
            //
            //             let lig_pos: Vec3 = lig.common.atom_posits[data.anchor_atom].into();
            //             let ctr: Vec3 = state.molecule.as_ref().unwrap().center.into();
            //
            //             cam_look_at_outside(&mut scene.camera, lig_pos, ctr);
            //
            //             engine_updates.camera = true;
            //             state.ui.cam_snapshot = None;
            //         }
            //     }
            // }

            set_static_light(
                scene,
                state.molecule.as_ref().unwrap().center.into(),
                state.molecule.as_ref().unwrap().size,
            );
        }
    }

    handle_scene_flags(state, scene, &mut engine_updates);

    engine_updates
}
