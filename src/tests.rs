use std::{path::PathBuf, str::FromStr, time::Instant};

use lin_alg::f32::{Vec3 as Vec3F32, pack_float, unpack_slice};
use rayon::{iter::IntoParallelRefIterator, prelude::*};

use super::*;
use crate::{
    docking::{ConformationType, DockingSite},
    forces::{V_lj, V_lj_x8},
};

#[test]
fn test_docking_setup() {
    // todo: Way to cache this load code, then split up the tests?
    // todo: E.g. tests for SIMD matching normal, as well as other more general/functional ones.

    let lj_lut = init_lj_lut();

    // todo: Don't load from file; set up test molecule[s]. For now, this is fine.
    let pdb = load_pdb(&PathBuf::from_str("molecules/1c8k.cif").unwrap()).unwrap();
    let receptor = Molecule::from_pdb(&pdb);

    let mol_ligand = load_sdf(&PathBuf::from_str("molecules/DB03496.sdf").unwrap()).unwrap();
    let mut ligand = Ligand::new(mol_ligand);

    {
        ligand.docking_site = DockingSite {
            site_center: lin_alg::f64::Vec3::new(40.6807, 36.2017, 28.5526),
            site_box_size: 10.,
        };
        ligand.pose.anchor_posit = ligand.docking_site.site_center;
        ligand.pose.orientation = lin_alg::f64::Quaternion::new(0.1156, -0.7155, 0.4165, 0.5488);

        if let ConformationType::Flexible { torsions } = &mut ligand.pose.conformation_type {
            torsions[1].dihedral_angle = 0.884;
            torsions[0].dihedral_angle = 2.553;
        }
    }

    let setup = DockingSetup::new(&receptor, &mut ligand, &lj_lut, &BhConfig::default());

    let poses = docking::init_poses(&ligand.docking_site, &ligand.flexible_bonds, 1, 2, 1);

    // let lig_posits = ligand.position_atoms(Some(&poses[0]));
    let lig_posits = ligand.position_atoms(None);

    let len_rec = setup.rec_atoms_near_site.len();
    let len_lig = lig_posits.len();

    let mut distances = Vec::with_capacity(len_rec * len_lig);
    for i_rec in 0..len_rec {
        for i_lig in 0..len_lig {
            let posit_rec: Vec3F32 = setup.rec_atoms_near_site[i_rec].posit.into();
            let posit_lig: Vec3F32 = lig_posits[i_lig].into();

            distances.push((posit_rec - posit_lig).magnitude());
        }
    }

    let (distances_x8, valid_lanes_last_dist) = pack_float(&distances);

    let vdw_start = Instant::now();
    // todo: Use a neighbor grid or similar? Set it up so there are two separate sides?
    let vdw: f32 = distances
        .par_iter()
        .enumerate()
        .map(|(i, r)| {
            let (sigma, eps) = setup.lj_sigma_eps[i];
            V_lj(*r, sigma, eps)
        })
        .sum();

    let vdw_x8: f32x8 = distances_x8
        .par_iter()
        .enumerate()
        .map(|(i, r)| {
            // if i + 1 == distances_x8.len() {
            //
            // }
            let sigma = setup.lj_sigma_x8[i];
            let eps = setup.lj_eps_x8[i];
            V_lj_x8(*r, sigma, eps)
        })
        .sum();

    let vdw_x8: f32 = vdw_x8.to_array().iter().sum();

    println!("VDW: {vdw} x8: {vdw_x8}");

    // todo: Potentially small(?) differnece. invalid lanes??
    // todo:  Youros answers are coming out similar in mangnute, but sometimes very large?
    assert!((vdw - vdw_x8).abs() < 0.00001);
}
