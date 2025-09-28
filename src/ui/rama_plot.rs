//! Draws a Ramachandran plot of protein backbone dihedral angles.

use std::f64::consts::PI;

use egui::{Align, Color32, Layout, Popup, PopupAnchor, Pos2, RectAlign, RichText, Ui, Vec2};
use egui_plot::{GridMark, Plot, PlotPoints, Points, uniform_grid_spacer};

use crate::molecule::Residue;

const POINT_RADIUS: f32 = 2.5;

pub fn plot_rama(residues: &[Residue], protein_ident: &str, ui: &mut Ui, popup_open: &mut bool) {
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
        ui.with_layout(Layout::top_down(Align::RIGHT), |ui| {
            if ui
                .button(RichText::new("Close").color(Color32::LIGHT_RED))
                .clicked()
            {
                *popup_open = false;
            }
        });

        ui.vertical_centered(|ui| {
            ui.heading(
                RichText::new(format!("Ramachandran plot for {protein_ident}"))
                    .color(Color32::WHITE),
            );
        });

        Plot::new("1000")
            .x_axis_label("φ")
            .y_axis_label("ψ")
            // common Rama bounds:
            .include_x(-180.0)
            .include_x(180.0)
            .include_y(-180.0)
            .include_y(180.0)
            .min_size(Vec2::new(800.0, 600.0))
            .allow_drag(false)
            .allow_scroll(false)
            .allow_axis_zoom_drag(false)
            //
            .x_grid_spacer(uniform_grid_spacer(|_in| [45.0, 15.0, 5.0])) // major, minor, tiny
            .y_grid_spacer(uniform_grid_spacer(|_in| [45.0, 15.0, 5.0]))
            // show labels only on the 45° multiples (hide 0 to match your examples)
            .x_axis_formatter(|mark: GridMark, _range| {
                let v = mark.value;
                if (v % 45.0).abs() < 1e-9 && v != 0.0 {
                    format!("{:.0}", v)
                } else {
                    String::new()
                }
            })
            .y_axis_formatter(|mark: GridMark, _range| {
                let v = mark.value;
                if (v % 45.0).abs() < 1e-9 && v != 0.0 {
                    format!("{:.0}", v)
                } else {
                    String::new()
                }
            })
            //
            .show(ui, |plot_ui| {
                let mut pts: Vec<[f64; 2]> = Vec::with_capacity(residues.len());
                for res in residues {
                    if let Some(d) = &res.dihedral {
                        // todo: Cache these?
                        let phi = (d.φ.unwrap_or_default() - PI).to_degrees();
                        let psi = (d.ψ.unwrap_or_default() - PI).to_degrees();
                        pts.push([phi, psi]);
                    }
                }

                if !pts.is_empty() {
                    let plot_pts: PlotPoints = PlotPoints::from(pts);
                    plot_ui.points(Points::new("Residues", plot_pts).radius(POINT_RADIUS));
                }
            });
    });
}
