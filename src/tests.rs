

use super::*;

#[test]
fn test_docking_setup() {
    let receptor = Molecule::new(

    );

    let mol_ligand = Molecule::new();

    let mut ligand = Ligand::new(&mol_ligand);

    let setup = DockingSetup::new(
        &receptor,
        &mut ligand,
        &lj_lut,
        &BhConfig::default(),
    );

    let mut rec_indices_x8_unpacked = Vec::new();
    for rec_i in &setup.rec_indices_x8 {
        for lane in 0..8 {
            rec_indices_x8_unpacked.push(*rec_i[lane]);
        }
    }

    assert_eq!(setup.rec_indices, rec_indices_x8_unpacked);
}
