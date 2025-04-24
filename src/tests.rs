use std::path::PathBuf;
use std::str::FromStr;
use lin_alg::f32::unpack_slice;
use super::*;

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

    let setup = DockingSetup::new(
        &receptor,
        &mut ligand,
        &lj_lut,
        &BhConfig::default(),
    );

    let rec_indices_x8_unpacked = unpack_slice(&setup.rec_indices_x8, setup.rec_indices.len());

    assert_eq!(setup.rec_indices, rec_indices_x8_unpacked);
}
