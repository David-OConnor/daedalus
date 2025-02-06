use std::path::Path;

use egui::{Color32, ComboBox, Context, RichText, TextEdit, TopBottomPanel, Ui, ViewportCommand};
use graphics::{EngineUpdates, Scene};

use crate::{
    download_pdb::load_rcsb,
    molecule::Molecule,
    pdb::load_pdb,
    render::{draw_molecule, MoleculeView},
    AtomColorCode, Selection, State,
};

pub const ROW_SPACING: f32 = 10.;
pub const COL_SPACING: f32 = 30.;

/// Update the tilebar to reflect the current molecule
fn set_window_title(title: &str, ui: &mut Ui) {
    // todo: Not working. Maybe need a new way when using WGPU.
    // ui.ctx().send_viewport_cmd(ViewportCommand::Title(title.to_string()));
}

fn load_file(
    path: &Path,
    state: &mut State,
    redraw: &mut bool,
    engine_updates: &mut EngineUpdates,
) {
    let pdb = load_pdb(&path);

    if let Ok(p) = pdb {
        state.pdb = Some(p);
        state.molecule = Some(Molecule::from_pdb(state.pdb.as_ref().unwrap()));

        *redraw = true;
        engine_updates.entities = true;
    } else {
        eprintln!("Error loading PDB file");
    }
}

fn int_field(val: &mut usize, label: &str, redraw: &mut bool, ui: &mut Ui) {
    ui.label(label);
    let mut val_str = val.to_string();
    if ui
        .add_sized(
            [60., Ui::available_height(ui)],
            TextEdit::singleline(&mut val_str),
        )
        .changed()
    {
        if let Ok(v) = val_str.parse::<usize>() {
            *val = v;
            *redraw = true;
        }
    }
}

/// Handles keyboard and mouse input not associated with a widget.
pub fn handle_input(
    state: &mut State,
    ui: &mut Ui,
    redraw: &mut bool,
    engine_updates: &mut EngineUpdates,
) {
    ui.ctx().input(|ip| {
        // Check for file drop
        if let Some(dropped_files) = ip.raw.dropped_files.first() {
            if let Some(path) = &dropped_files.path {
                load_file(&path, state, redraw, engine_updates)
            }
        }
    });
}

/// Display text of the selected atom
fn selected_data(mol: &Molecule, selection: Selection, ui: &mut Ui) {
    match selection {
        Selection::Atom(sel) => {
            let atom = &mol.atoms[sel];
            ui.label(format!(
                "El: {:?}, AA: {:?}, Role: {:?}",
                atom.element, atom.amino_acid, atom.role
            ));
        }
        Selection::Residue(sel) => {
            let res = &mol.residues[sel];
            let name = if let Some(aa) = res.aa {
                aa.to_string()
            } else {
                "-".to_owned() // todo temp
            };

            // todo: Sequesnce number etc.
            ui.label(format!("Res: {:?}", name,));
        }
        Selection::None => (),
    }
}

/// This function draws the (immediate-mode) GUI.
/// [UI items](https://docs.rs/egui/latest/egui/struct.Ui.html)
pub fn ui_handler(state: &mut State, ctx: &Context, scene: &mut Scene) -> EngineUpdates {
    let mut engine_updates = EngineUpdates::default();
    let mut redraw = false;
    // let mut reset_window_title = false; // This setup avoids borrow errors.

    TopBottomPanel::top("0").show(ctx, |ui| {
        handle_input(state, ui, &mut redraw, &mut engine_updates);

        ui.horizontal(|ui| {
            if let Some(mol) = &state.molecule {
                ui.heading(RichText::new(mol.ident.clone()).color(Color32::GOLD));
            }

            ui.add_space(COL_SPACING);

            if ui.button("Open").clicked() {
                state.ui.load_dialog.pick_file();
            }

            ui.add_space(COL_SPACING);
            ui.label("RCSB ident:");
            ui.add(TextEdit::singleline(&mut state.ui.rcsb_input).desired_width(60.));

            if ui.button("Load RCSB").clicked() {
                match load_rcsb(&state.ui.rcsb_input) {
                    Ok(pdb) => {
                        state.pdb = Some(pdb);
                        state.molecule = Some(Molecule::from_pdb(state.pdb.as_ref().unwrap()));

                        redraw = true;
                    }
                    Err(_e) => {
                        eprintln!("Error loading PDB file");
                    }
                }
            }

            ui.add_space(COL_SPACING);

            state.ui.load_dialog.update(ctx);

            ui.label("View:");
            let prev_view = state.ui.mol_view;
            ComboBox::from_id_salt(0)
                .width(80.)
                .selected_text(state.ui.mol_view.to_string())
                .show_ui(ui, |ui| {
                    for view in &[
                        MoleculeView::Sticks,
                        MoleculeView::Ribbon,
                        MoleculeView::Spheres,
                        MoleculeView::SpaceFilling,
                        MoleculeView::Cartoon,
                        MoleculeView::Surface,
                        MoleculeView::Mesh,
                        MoleculeView::Dots,
                    ] {
                        ui.selectable_value(&mut state.ui.mol_view, *view, view.to_string());
                    }
                });

            if state.ui.mol_view != prev_view {
                redraw = true;
            }

            ui.add_space(COL_SPACING);

            // todo: DRY with view.
            ui.label("Color code:");
            let prev_view = state.ui.atom_color_code;
            ComboBox::from_id_salt(1)
                .width(80.)
                .selected_text(state.ui.atom_color_code.to_string())
                .show_ui(ui, |ui| {
                    for view in &[AtomColorCode::Atom, AtomColorCode::Residue] {
                        ui.selectable_value(&mut state.ui.atom_color_code, *view, view.to_string());
                    }
                });

            if state.ui.atom_color_code != prev_view {
                redraw = true;
            }

            ui.add_space(COL_SPACING);

            ui.label("Filter nearby:");
            if ui.checkbox(&mut state.ui.show_nearby_only, "").changed() {
                redraw = true;
            }

            ui.add_space(COL_SPACING);
            let mut dist_int = state.ui.nearby_dist_thresh as usize;
            let dist_int_prev = dist_int;
            int_field(&mut dist_int, "Filter thresh:", &mut redraw, ui);

            if dist_int != dist_int_prev {
                state.ui.nearby_dist_thresh = dist_int as f32;
            }

            ui.add_space(COL_SPACING);

            if let Some(mol) = &state.molecule {
                selected_data(mol, state.selection, ui);
            }
        });
        ui.add_space(ROW_SPACING / 2.);

        if let Some(path) = &state.ui.load_dialog.take_picked() {
            load_file(path, state, &mut redraw, &mut engine_updates);
        }

        if redraw {
            if let Some(molecule) = &state.molecule {
                draw_molecule(&mut scene.entities, molecule, &state.ui, state.selection);

                set_window_title(&molecule.ident, ui);
                engine_updates.entities = true;
            }
        }
    });

    state.ui.load_dialog.update(ctx);
    engine_updates
}
