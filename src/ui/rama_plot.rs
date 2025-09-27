//! Draws a Ramachandran plot of protein backbone dihedral angles.

use egui::{Popup, PopupAnchor, Pos2, RectAlign, Ui};
use egui_plot::{Legend, Plot, PlotPoints, Points};

use crate::molecule::Residue;

const POINT_RADIUS: f32 = 2.5;

pub fn plot_rama(residues: &[Residue], protein_ident: &str, ui: &mut Ui) {
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
        Plot::new(format!("Ramachandran plot for {protein_ident}"))
            .legend(Legend::default())
            .x_axis_label("ϕ")
            .y_axis_label("ψ")
            // common Rama bounds:
            .include_x(-180.0)
            .include_x(180.0)
            .include_y(-180.0)
            .include_y(180.0)
            .show(ui, |plot_ui| {
                let mut pts: Vec<[f64; 2]> = Vec::with_capacity(residues.len());
                for res in residues {
                    if let Some(d) = &res.dihedral {
                        let phi = d.φ.unwrap_or_default() as f64;
                        let psi = d.ψ.unwrap_or_default() as f64;
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
