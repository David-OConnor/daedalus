use std::{
    io::Cursor,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

use bio_apis::rcsb;
use bio_files::{DensityMap, ResidueType, density_from_2fo_fc_rcsb_gemmi};
use egui::{
    Align, Color32, ComboBox, Context, Key, Layout, Popup, PopupAnchor, Pos2, RectAlign, RichText,
    Slider, TextEdit, TopBottomPanel, Ui,
};
use graphics::{ControlScheme, EngineUpdates, EntityUpdate, FWD_VEC, Scene};
use md::md_setup;
use mol_data::display_mol_data;
use na_seq::AaIdent;

use crate::{
    CamSnapshot, MsaaSetting, OperatingMode, Selection, State, ViewSelLevel, cli,
    cli::autocomplete_cli,
    docking_v2::dock,
    download_mols::{load_atom_coords_rcsb, load_sdf_drugbank, load_sdf_pubchem},
    drawing::{EntityClass, MoleculeView, color_viridis},
    drawing_wrappers::draw_all_lipids,
    file_io::gemmi_path,
    inputs::{MOVEMENT_SENS, ROTATE_SENS, SENS_MOL_MOVE_SCROLL},
    lipid::{LipidShape, make_bacterial_lipids},
    mol_editor::enter_edit_mode,
    molecule::MolGenericRef,
    render::{set_flashlight, set_static_light},
    ui::{
        cam::{cam_controls, cam_snapshots},
        misc::section_box,
        mol_data::display_mol_data_peptide,
        rama_plot::plot_rama,
        util::{
            handle_redraw, init_with_scene, load_file, load_popups, open_lig_from_input,
            update_file_dialogs,
        },
        view::{ui_section_vis, view_settings},
    },
    util::{
        check_prefs_save, clear_mol_entity_indices, close_mol, close_peptide, cycle_selected,
        handle_err, handle_scene_flags, handle_success, orbit_center, reset_orbit_center,
        select_from_search,
    },
};

pub mod cam;
mod md;
pub mod misc;
mod mol_data;
mod mol_editor;
mod rama_plot;
pub mod util;
mod view;

static INIT_COMPLETE: AtomicBool = AtomicBool::new(false);
// This allows us to wait a frame before getting the UI height. This affects the geometry
// between 3d adn 2d  space, e.g. for selecting atoms with the mouse.
pub static UI_HEIGHT_CHANGED: AtomicBool = AtomicBool::new(false);
pub static UI_HEIGHT_CHANGE_DELAY: AtomicBool = AtomicBool::new(false);

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
pub const _COLOR_ATTENTION: Color32 = Color32::ORANGE;

const COLOR_OUT_ERROR: Color32 = Color32::LIGHT_RED;
const COLOR_OUT_NORMAL: Color32 = Color32::WHITE;
const _COLOR_OUT_SUCCESS: Color32 = Color32::LIGHT_GREEN; // Unused for now

pub const _COLOR_POPUP: Color32 = Color32::from_rgb(20, 20, 30);

// Number of characters to display. E.g. the molecular description. Often long.
const MAX_TITLE_LEN: usize = 40;

/// Update the tilebar to reflect the current molecule
fn set_window_title(title: &str, scene: &mut Scene) {
    scene.window_title = title.to_owned();
    // ui.ctx().send_viewport_cmd(ViewportCommand::Title(title.to_string()));
}

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
    scene: &mut Scene,
) {
    ui.ctx().input(|ip| {
        // Check for file drop
        if let Some(dropped_files) = ip.raw.dropped_files.first() {
            if let Some(path) = &dropped_files.path {
                if let Err(e) = load_file(path, state, redraw, reset_cam, engine_updates, scene) {
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

/// Toggles chain visibility
fn chain_selector(state: &mut State, redraw: &mut bool, ui: &mut Ui) {
    // todo: For now, DRY with res selec
    let Some(mol) = &mut state.peptide else {
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

                state.ui.popup.residue_selector = !state.ui.popup.residue_selector;
            }
        }

        if state.ui.chain_to_pick_res.is_some() {
            if ui.button("(None)").clicked() {
                state.ui.chain_to_pick_res = None;
                state.volatile.ui_height = ui.ctx().used_size().y  / ui.ctx().pixels_per_point();
            }
        }
    });
}

// todo: Update params A/R
fn draw_cli(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    redraw_pep: &mut bool,
    redraw_lig: &mut bool,
    redraw_na: &mut bool,
    redraw_lipid: &mut bool,
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
                match cli::handle_cmd(state, scene, engine_updates, redraw_pep, reset_cam) {
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
        residue_search(
            state,
            scene,
            redraw_pep,
            redraw_lig,
            redraw_na,
            redraw_lipid,
            ui,
        );
    });
}

fn residue_search(
    state: &mut State,
    scene: &mut Scene,
    redraw_pep: &mut bool,
    redraw_lig: &mut bool,
    redraw_na: &mut bool,
    redraw_lipid: &mut bool,
    ui: &mut Ui,
) {
    let (btn_text_p, btn_text_n, search_text) = match state.ui.view_sel_level {
        ViewSelLevel::Atom => ("Prev atom", "Next atom", "Find atom:"),
        ViewSelLevel::Residue => ("Prev AA", "Next AA", "Find res:"),
        ViewSelLevel::Bond => ("Prev bond", "Next bond", "Find bond:"),
    };

    ui.label(search_text);
    if ui
        .add(TextEdit::singleline(&mut state.ui.atom_res_search).desired_width(60.))
        .changed()
    {
        select_from_search(state);
        *redraw_pep = true;
    }

    if state.peptide.is_some() || !state.ligands.is_empty() {
        if ui
            .button(btn_text_p)
            .on_hover_text("(Hotkey: Left arrow)")
            .clicked()
        {
            cycle_selected(state, scene, true);

            match state.ui.selection {
                Selection::AtomPeptide(_) | Selection::Residue(_) | Selection::BondPeptide(_) => {
                    *redraw_pep = true
                }
                Selection::AtomLig(_) | Selection::BondLig(_) => *redraw_lig = true,
                Selection::AtomNucleicAcid(_) | Selection::BondNucleicAcid(_) => *redraw_na = true,
                Selection::AtomLipid(_) | Selection::BondLipid(_) => *redraw_lipid = true,
                _ => (),
            }
        }
        // todo: DRY

        if ui
            .button(btn_text_n)
            .on_hover_text("(Hotkey: Right arrow)")
            .clicked()
        {
            cycle_selected(state, scene, false);

            match state.ui.selection {
                Selection::AtomPeptide(_) | Selection::Residue(_) => *redraw_pep = true,
                Selection::AtomLig(_) => *redraw_lig = true,
                Selection::AtomNucleicAcid(_) => *redraw_na = true,
                Selection::AtomLipid(_) => *redraw_lipid = true,
                _ => (),
            }
        }

        // ui.add_space(COL_SPACING * 2.);
        //
        // let dock_tools_text = if state.ui.show_docking_tools {
        //     "Hide docking tools"
        // } else {
        //     "Show docking tools (Broken/WIP)"
        // };
        //
        // if ui.button(RichText::new(dock_tools_text)).clicked() {
        //     state.ui.show_docking_tools = !state.ui.show_docking_tools;
        // }
    }
}

fn add_aa_seq(selection: &mut Selection, seq_text: &str, ui: &mut Ui, redraw: &mut bool) {
    let len = seq_text.len(); // One char per res.
    ui.horizontal_wrapped(|ui| {
        for (i, aa) in seq_text.chars().enumerate() {
            let color = color_viridis(i, 0, len);
            // todo: Find a cheaper way.
            let mut color = Color32::from_rgb(
                (color.0 * 255.) as u8,
                (color.1 * 255.) as u8,
                (color.2 * 255.) as u8,
            );

            if let Selection::Residue(sel) = selection {
                if i == *sel {
                    color = Color32::from_rgb(255, 0, 0); // cheaper, but more maintenance than calling the const.
                }
            }

            if ui.label(RichText::new(aa).color(color)).clicked() {
                *selection = Selection::Residue(i);
                *redraw = true;
            }
        }
    });
}

pub fn view_sel_selector(state: &mut State, redraw: &mut bool, ui: &mut Ui, include_res: bool) {
    let help_text = "(Hotkeys: square brackets [ ])";
    ui.label("View/Select:").on_hover_text(help_text);
    let prev_view = state.ui.view_sel_level;

    let mut views = vec![ViewSelLevel::Atom, ViewSelLevel::Bond];

    if include_res {
        views.push(ViewSelLevel::Residue);
    }

    // Ideally hover text here too, but I'm not sure how.
    ComboBox::from_id_salt(1)
        .width(80.)
        .selected_text(state.ui.view_sel_level.to_string())
        .show_ui(ui, |ui| {
            for view in &views {
                ui.selectable_value(&mut state.ui.view_sel_level, *view, view.to_string());
            }
        })
        .response
        .on_hover_text(help_text);

    if state.ui.view_sel_level != prev_view {
        *redraw = true;
        // If we change from atom to res, select the prev-selected atom's res. If vice-versa,
        // select that residue's Cα.
        // state.ui.selection = Selection::None;
        // todo: This section needs some updates, but isn't critical.
        if let Some(mol) = &state.peptide {
            match state.ui.view_sel_level {
                ViewSelLevel::Residue => {
                    state.ui.selection = match state.ui.selection {
                        Selection::AtomPeptide(i) => {
                            Selection::Residue(mol.common.atoms[i].residue.unwrap_or_default())
                        }
                        _ => Selection::None,
                    };
                }
                ViewSelLevel::Atom => {
                    state.ui.selection = match state.ui.selection {
                        // It seems [0] is often N, and [1] is Cα
                        Selection::Residue(i) => {
                            if i >= mol.residues.len() {
                                handle_err(&mut state.ui, "Residue bounds problem".to_string());
                                Selection::None
                            } else {
                                if mol.residues[i].atoms.len() <= 2 {
                                    Selection::AtomPeptide(mol.residues[i].atoms[1])
                                } else {
                                    Selection::None
                                }
                            }
                        }

                        _ => Selection::None,
                    };
                }
                ViewSelLevel::Bond => {}
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
                .on_hover_text(
                    "Color the atom by partial charge, instead of element-specific colors",
                )
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
        // todo: We could color these based on current vs nominal length and/or frequency.
        ViewSelLevel::Bond => (),
    }

    ui.add_space(COL_SPACING);
}

fn selection_section(state: &mut State, redraw: &mut bool, ui: &mut Ui) {
    // todo: DRY with view.
    ui.horizontal_wrapped(|ui| {
        section_box().show(ui, |ui| {
            view_sel_selector(state, redraw, ui, true);

            let help = "Hide all atoms not near the selection";
            ui.label("Nearby sel only:").on_hover_text(help);
            if ui
                .checkbox(&mut state.ui.show_near_sel_only, "")
                .on_hover_text(help)
                .changed()
            {
                *redraw = true;

                // todo: For now, only allow one of near sel/lig
                if state.ui.show_near_sel_only {
                    state.ui.show_near_lig_only = false
                }
            }

            if state.active_mol().is_some() {
                let help = "Hide all atoms not near the ligand";
                ui.label("Nearby lig only:").on_hover_text(help);
                if ui
                    .checkbox(&mut state.ui.show_near_lig_only, "")
                    .on_hover_text(help)
                    .changed()
                {
                    *redraw = true;

                    // todo: For now, only allow one of near sel/lig
                    if state.ui.show_near_lig_only {
                        state.ui.show_near_sel_only = false
                    }
                }
            }

            ui.label("pH:");
            if ui
                .add_sized(
                    [20., Ui::available_height(ui)],
                    TextEdit::singleline(&mut state.ui.ph_input),
                )
                .changed()
            {
                if let Ok(v) = &mut state.ui.ph_input.parse::<f32>() {
                    state.to_save.ph = *v;

                    // Re-assign hydrogens, and redraw
                    if let Some(mol) = &mut state.peptide {
                        if let Some(ff_map) = &state.ff_param_set.peptide_ff_q_map {
                            match mol.reassign_hydrogens(state.to_save.ph, ff_map) {
                                Ok(_) => *redraw = true,
                                Err(e) => {
                                    let msg = format!("Error reassigning hydrogens: {e:?}");
                                    handle_err(&mut state.ui, msg);
                                }
                            }
                        }
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

        if state.ui.selection != Selection::None {
            section_box().show(ui, |ui| {
                ui.horizontal(|ui| {
                    mol_data::selected_data(
                        &state,
                        &state.ligands,
                        &state.nucleic_acids,
                        &state.lipids,
                        &state.ui.selection,
                        ui,
                    );
                });
            });
        }
    });
}

fn mol_descrip(mol: &MolGenericRef, ui: &mut Ui) {
    ui.heading(RichText::new(mol.common().ident.clone()).color(Color32::GOLD));

    ui.label(format!("{} atoms", mol.common().atoms.len()));

    if let MolGenericRef::Peptide(m) = mol {
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
            ui.label("Cam move speed:");
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
            ui.label("Cam rot sensitivity:");
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

            ui.add_space(COL_SPACING);
            ui.label("Mol scroll move speed:").on_hover_text(
                "When using the scroll wheel to move molecules, this controls how fast they move.",
            );
            if ui
                .add(TextEdit::singleline(&mut state.ui.mol_move_sens_input).desired_width(32.))
                .changed()
            {
                if let Ok(v) = &mut state.ui.mol_move_sens_input.parse::<u8>() {
                    state.to_save.mol_move_sens = *v;
                    state.update_save_prefs(false);
                } else {
                    // reset
                    state.ui.mol_move_sens_input = state.to_save.mol_move_sens.to_string();
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

                state.to_save.mol_move_sens = (SENS_MOL_MOVE_SCROLL * 1_000.) as u8;
                state.ui.mol_move_sens_input = state.to_save.mol_move_sens.to_string();

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

fn residue_selector(state: &mut State, scene: &mut Scene, ui: &mut Ui, redraw: &mut bool) {
    let popup_id = ui.make_persistent_id("res_popup");
    Popup::new(
        popup_id,
        ui.ctx().clone(),
        PopupAnchor::Position(Pos2::new(60., 60.)),
        ui.layer_id(),
    )
    .align(RectAlign::TOP)
    .open(true)
    .width(1_000.)
    .gap(4.0)
    .show(|ui| {
        ui.with_layout(Layout::top_down(Align::RIGHT), |ui| {
            if ui
                .button(RichText::new("Close").color(Color32::LIGHT_RED))
                .clicked()
            {
                state.ui.popup.residue_selector = false;
                state.ui.chain_to_pick_res = None;
            }
        });
        ui.add_space(ROW_SPACING);
        // This is a bit fuzzy, as the size varies by residue name (Not always 1 for non-AAs), and index digits.

        let mut update_arc_center = false;

        if let Some(mol) = &state.peptide {
            if let Some(chain_i) = state.ui.chain_to_pick_res {
                if chain_i >= mol.chains.len() {
                    return;
                }
                let chain = &mol.chains[chain_i];

                ui.add_space(ROW_SPACING);

                // todo: Wrap not working in popup?
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing.x = 8.0;

                    for (i, res) in mol.residues.iter().enumerate() {
                        if i > 800 {
                            break; // todo: Temp workaround to display blocking
                        }
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

    let mut redraw_peptide = false;
    let mut redraw_lig = false;
    let mut redraw_na = false;
    let mut redraw_lipid = false;
    let mut reset_cam = false;

    // For getting DT for certain buttons when held. Does not seem to be the same as the 3D render DT.
    let start = Instant::now();

    TopBottomPanel::top("0").show(ctx, |ui| {
        ui.spacing_mut().slider_width = 120.;

        handle_input(
            state,
            ui,
            &mut redraw_peptide,
            &mut reset_cam,
            &mut engine_updates,
            scene,
        );

        if state.volatile.operating_mode == OperatingMode::MolEditor {
            mol_editor::editor(state, scene, &mut engine_updates,ui);

            if let Err(e) = update_file_dialogs(state, scene, ui, &mut false, &mut false, &mut engine_updates) {
                handle_err(&mut state.ui, format!("Problem saving file: {e:?}"));
            }

            state.mol_editor.md_step(&state.dev, &mut scene.entities, &state.ui, &mut engine_updates);
            return;
        }

        if state.ui.popup.show_settings {
            settings(state, scene, ui);
        }

        if state.ui.popup.residue_selector {
            // todo: Show hide based on AaCategory? i.e. residue.amino_acid.category(). Hydrophilic, acidic etc.
            residue_selector(state, scene, ui, &mut redraw_peptide);
        }

        if state.ui.popup.rama_plot {
            if let Some(mol) = &state.peptide {
                plot_rama(&mol.residues, &mol.common.ident, ui, &mut state.ui.popup.rama_plot);
            }
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

            {
                let mut close = false;
                display_mol_data_peptide(state, scene, ui, &mut redraw_peptide, &mut redraw_lig, &mut close, &mut engine_updates);

                if close {
                    close_peptide(state, scene, &mut engine_updates);
                }
            }

            let color_open_tools = if state.peptide.is_none() {
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
            if let Some(mol) = &mut state.peptide {
                // let color = if state.to_save.last_peptide_opened.is_none() {
                //     COLOR_ATTENTION
                // } else {
                //     Color32::GRAY
                // };
                // todo: Put a form of this back.
                let color = Color32::GRAY;

                if ui.button(RichText::new("Save").color(color)).clicked() {
                    let filename = {
                        let extension = "cif";

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

            let mut mol_to_save = None; // avoids dbl-borrow.
            if let Some(mol) = state.active_mol() {
                // Highlight the button if we haven't saved this to file, e.g. if opened from online.
                // let color = if state.to_save.last_ligand_opened.is_none() {
                //     COLOR_ATTENTION
                // } else {
                //     Color32::GRAY
                // };
                // todo: Put a form of this back.
                let color = Color32::GRAY;

                if ui.button(RichText::new("Save mol").color(color))
                    .on_hover_text("Save the active small molecule, nucleic acid, or lipid to a file.")
                    .clicked() {

                    mol_to_save = Some(mol.common().clone());
                }
            }
            if let Some(mol) = mol_to_save {
                if mol.save(&mut state.volatile.dialogs.save).is_err() {
                    handle_err(&mut state.ui, "Problem saving this file".to_owned());
                }
            }

            if metadata_loaded {
                state.update_save_prefs(false);
            }

            ui.add_space(COL_SPACING);

            let query_help = "Download and view a molecule from RCSB PDB, PubChem, DrugBank, or Amber Geostd";
            ui.label(RichText::new("Query DBs:").color(color_open_tools))
                .on_hover_text(query_help);

            let edit_resp = ui
                .add(TextEdit::singleline(&mut state.ui.db_input).desired_width(60.))
                .on_hover_text(query_help);

            if state.ui.db_input.len() >= 4 {
                let enter_pressed =
                    edit_resp.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter));
                let button_clicked = ui.button("Load from RCSB").clicked();

                if (button_clicked || enter_pressed) && state.ui.db_input.trim().len() == 4 {
                    let ident = state.ui.db_input.clone().trim().to_owned();

                    load_atom_coords_rcsb(
                        &ident,
                        state,
                        scene,
                        &mut engine_updates,
                        &mut redraw_peptide,
                        &mut reset_cam,
                    );

                    state.ui.db_input = String::new();
                }
            }
            if state.ui.db_input.len() == 3 {
                let enter_pressed =
                    edit_resp.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter));
                let button_clicked = ui.button("Load Amber Geostd").clicked();

                if button_clicked || enter_pressed {
                    let db_input = &state.ui.db_input.clone(); // Avoids a double borrow.
                    state.load_geostd_mol_data(&db_input,true, true, &mut redraw_lig, &scene.camera);

                    state.ui.db_input = String::new();
                }
            }

            if state.ui.db_input.len() >= 4 {
                if state.ui.db_input.to_uppercase().starts_with("DB") {
                    if ui.button("Load from DrugBank").clicked() {
                        match load_sdf_drugbank(&state.ui.db_input) {
                            Ok(mol) => {
                                open_lig_from_input(state, &scene.camera, mol);
                                redraw_lig = true;
                                reset_cam = true;
                            }
                            Err(e) => {
                                let msg = format!("Error loading SDF file: {e:?}");
                                handle_err(&mut state.ui, msg);
                            }
                        }
                    }
                }

                if ui.button("Load from PubChem").clicked() {
                    match load_sdf_pubchem(&state.ui.db_input) {
                        Ok(mol) => {
                            open_lig_from_input(state, &scene.camera, mol);
                            redraw_lig = true;
                            reset_cam = true;
                        }
                        Err(e) => {
                            let msg = format!("Error loading SDF file: {e:?}");
                            handle_err(&mut state.ui, msg);
                        }
                    }
                }
            }

            if state.peptide.is_none() && state.active_mol().is_none() {
                ui.add_space(COL_SPACING / 2.);
                if ui
                    .button(RichText::new("I'm feeling lucky 🍀").color(color_open_tools))
                    .on_hover_text("Open a random recently-uploaded protein from RCSB PDB.")
                    .clicked()
                {
                    match rcsb::get_newly_released() {
                        Ok(ident) => {
                            load_atom_coords_rcsb(
                                &ident,
                                state,
                                scene,
                                &mut engine_updates,
                                &mut redraw_peptide,
                                &mut reset_cam,
                            );
                        }
                        Err(e) => handle_err(&mut state.ui, format!("Error loading a protein from RCSB: {e:?}"))
                    }
                }
            }
        });

        let mut close_active_mol = false; // to avoid borrow error.

        ui.horizontal(|ui| {
            // Show the picker, at least.
            if !state.ligands.is_empty() {
                display_mol_data(state, scene, ui, &mut redraw_peptide, &mut redraw_lig, &mut redraw_na, &mut redraw_lipid, &mut close_active_mol, &mut engine_updates);
            }

            if ui.button(RichText::new("Mol editor").color(COLOR_HIGHLIGHT)).clicked() {
                enter_edit_mode(state, scene, &mut engine_updates);
            }
        });

        let redraw_prev = redraw_peptide;
        selection_section(state, &mut redraw_peptide, ui);
        // todo: Kludge
        if redraw_peptide && !redraw_prev {
            redraw_lig = true;
            redraw_na = true;
            redraw_lipid = true;
        }

        ui.horizontal_wrapped(|ui| {
            cam_controls(scene, state, &mut engine_updates, ui);
            cam_snapshots(state, scene, &mut engine_updates, ui);
        });

        ui.horizontal(|ui| {
            view_settings(state, scene, &mut engine_updates, &mut redraw_peptide, &mut redraw_lig, &mut redraw_na, &mut redraw_lipid, ui);

            ui.add_space(COL_SPACING);

            section_box().show(ui, |ui| {
                ui_section_vis(state, ui);
            });
        });

        chain_selector(state, &mut redraw_peptide, ui);

        ui.horizontal(|ui| {
            if state.ui.ui_vis.lipids {
                lipid_section(state, scene, &mut engine_updates, ui);
            }

            if let Some(mol) = &state.active_mol() {
                if let MolGenericRef::Ligand(_) = mol {
                    ui.add_space(COL_SPACING);

                    ui.label("Docking");

                    if ui.button(RichText::new("Dock").color(Color32::GOLD)).clicked() {
                        // The other views make it tough to see the ligand rel the protein.
                        if !matches!(state.ui.mol_view, MoleculeView::SpaceFill | MoleculeView::Surface) {
                            // todo: Dim peptide?
                            state.ui.mol_view = MoleculeView::Surface;
                        }

                        if let Err(e) = dock(state, state.volatile.active_mol.unwrap().1) {
                            handle_err(&mut state.ui, format!("Problem setting up docking: {e:?}"));
                        }
                    }
                }
            }
        });


        if state.ui.ui_vis.dynamics {
            md_setup(state, scene, &mut engine_updates, ui);
        }

        // if state.ui.show_docking_tools {
        //     ui.add_space(ROW_SPACING);
        //
        //     docking(state, scene, &mut redraw_lig, &mut engine_updates, ui);
        // }

        // todo: Allow switching between chains and secondary-structure features here.

        ui.add_space(ROW_SPACING / 2.);

        if state.ui.ui_vis.aa_seq {
            if state.peptide.is_some() {
                add_aa_seq(&mut state.ui.selection, &state.volatile.aa_seq_text, ui, &mut redraw_peptide);
            }
        }

        draw_cli(
            state,
            scene,
            &mut engine_updates,
            &mut redraw_peptide,
            &mut redraw_lig,
            &mut redraw_na,
            &mut redraw_lipid,
            &mut reset_cam,
            ui,
        );

        load_popups(state, scene, ui, &mut redraw_peptide, &mut redraw_lig, &mut reset_cam, &mut engine_updates);

        // -------UI above; clean-up items (based on flags) below

        if close_active_mol {
            if let Some((mol_type, i)) = state.volatile.active_mol {
                close_mol(mol_type, i, state, scene, &mut engine_updates);
            }
        }

        if let Err(e) = update_file_dialogs(state, scene, ui, &mut redraw_peptide, &mut reset_cam, &mut engine_updates) {
            handle_err(&mut state.ui, format!("Problem saving file: {e:?}"));
        }

        handle_redraw(
            state,
            scene, redraw_peptide, redraw_lig, redraw_na, redraw_lipid, reset_cam, &mut engine_updates
        )
    });

    // todo: Appropriate place for this?
    if state.volatile.inputs_commanded.inputs_present() {
        set_flashlight(scene);
        engine_updates.lighting = true;
    }

    // This double-change variable logic is due to some cases needing to wait
    // an additional frame before the height takes effect.
    if UI_HEIGHT_CHANGE_DELAY.swap(false, Ordering::AcqRel) {
        state.volatile.ui_height = ctx.used_size().y / ctx.pixels_per_point();
    }

    if UI_HEIGHT_CHANGED.swap(false, Ordering::AcqRel) {
        UI_HEIGHT_CHANGE_DELAY.store(true, Ordering::Release);
    }

    // We perform init items here that rely on the scene, or UI context.
    if !INIT_COMPLETE.swap(true, Ordering::AcqRel) {
        init_with_scene(state, scene, ctx);
    }

    handle_scene_flags(state, scene, &mut engine_updates);

    // Run one or more MD steps, if a MD computation is in progress.
    state.md_step(scene, &mut engine_updates);

    state.ui.dt_render = start.elapsed().as_secs_f32();

    engine_updates
}

pub fn flag_btn(val: &mut bool, label: &str, hover_text: &str, ui: &mut Ui) {
    let color = if *val { COLOR_ACTIVE } else { COLOR_INACTIVE };
    if ui
        .button(RichText::new(label).color(color))
        .on_hover_text(hover_text)
        .clicked()
    {
        *val = !(*val);
    }
}

/// Add and manage lipids
pub fn lipid_section(
    state: &mut State,
    scene: &mut Scene,
    engine_updates: &mut EngineUpdates,
    ui: &mut Ui,
) {
    if state.ui.lipid_to_add >= state.lipid_templates.len() {
        eprintln!("Error: Not enough lipid templates");
        return;
    }

    section_box().show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label("Add lipids:");

            let add_standard_text = state.lipid_templates[state.ui.lipid_to_add]
                .common
                .ident
                .clone();

            ComboBox::from_id_salt(102)
                .width(90.)
                .selected_text(state.ui.lipid_shape.to_string())
                .show_ui(ui, |ui| {
                    for shape in [LipidShape::Free, LipidShape::Membrane, LipidShape::Lnp] {
                        ui.selectable_value(&mut state.ui.lipid_shape, shape, shape.to_string());
                    }
                })
                .response
                .on_hover_text("Add lipids in this pattern");

            if state.ui.lipid_shape == LipidShape::Free {
                ComboBox::from_id_salt(101)
                    .width(30.)
                    .selected_text(add_standard_text)
                    .show_ui(ui, |ui| {
                        for (i, mol) in state.lipid_templates.iter().enumerate() {
                            ui.selectable_value(&mut state.ui.lipid_to_add, i, &mol.common.ident);
                        }
                    })
                    .response
                    .on_hover_text("Add this lipid to the scene.");
            }

            num_field(&mut state.ui.lipid_mol_count, "# mols", 36, ui);

            // todo: Multiple and sets once this is validated
            if ui.button("+").clicked() {
                // Place in front of the camera.
                let center = scene.camera.position
                    + scene.camera.orientation.rotate_vec(FWD_VEC)
                        * crate::cam_misc::MOVE_TO_CAM_DIST;

                state.lipids.extend(make_bacterial_lipids(
                    state.ui.lipid_mol_count as usize,
                    center.into(),
                    state.ui.lipid_shape,
                    &state.lipid_templates,
                ));
                //
                // let mut mol = state.lipid_templates[state.ui.lipid_to_add].clone();
                // for p in &mut mol.common.atom_posits {
                //     *p = *p + Vec3::new_zero();
                // }
                //
                // state.lipids.push(mol);

                draw_all_lipids(state, scene);
                engine_updates.entities = EntityUpdate::All;
            }

            if !state.lipids.is_empty() {
                if ui
                    .button(RichText::new("Close all lipids").color(Color32::LIGHT_RED))
                    .clicked()
                {
                    state.lipids = Vec::new();
                    scene
                        .entities
                        .retain(|e| e.class != EntityClass::Lipid as u32);
                    clear_mol_entity_indices(state, None);

                    engine_updates.entities = EntityUpdate::All;
                }
            }
        });
    });
}
