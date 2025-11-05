//! This is the entry point for a standalone application to train the data.
//! Run `cargo b --release --bin train

use std::{
    collections::{BTreeSet, HashMap},
    fs, io,
    path::{Path, PathBuf},
};

use bio_files::Mol2;

mod param_inference;

use candle_core::{CudaDevice, DType, Device};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder};
use param_inference::{GEOSTD_PATH, GeoStdMol2Dataset, MolGNN};

use crate::param_inference::MODEL_PATH;

/// Find Mol2 paths. Assumes there are per-letter subfolders one-layer deep.
/// todo: FRCmod as well
pub fn find_paths(geostd_dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut result = Vec::new();

    for entry in fs::read_dir(geostd_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            for subentry in fs::read_dir(&path)? {
                let subentry = subentry?;
                let subpath = subentry.path();

                if subpath
                    .extension()
                    .map(|e| e.to_string_lossy().to_lowercase())
                    != Some("mol2".to_string())
                {
                    continue;
                }

                result.push(subpath);
            }
        }
    }

    Ok(result)
}

/// (Element, atom type) maps.
pub fn build_vocabs(
    mol2_paths: &[PathBuf],
) -> candle_core::Result<(HashMap<String, usize>, HashMap<String, usize>)> {
    let mut elems: BTreeSet<String> = BTreeSet::new();
    let mut ff_types: BTreeSet<String> = BTreeSet::new();

    for path in mol2_paths {
        let mol = Mol2::load(path)?;

        let mut skip_mol = false;
        for atom in mol.atoms.iter() {
            let Some(ff) = &atom.force_field_type else {
                skip_mol = true;
                break;
            };

            elems.insert(atom.element.to_letter());
            ff_types.insert(ff.clone());
        }
        if skip_mol {
            continue;
        }
    }

    let mut el_map = HashMap::new();
    for (i, el) in elems.into_iter().enumerate() {
        el_map.insert(el, i);
    }

    let mut atom_type_map = HashMap::new();
    for (i, t) in ff_types.into_iter().enumerate() {
        atom_type_map.insert(t, i);
    }

    Ok((el_map, atom_type_map))
}

fn main() -> candle_core::Result<()> {
    #[cfg(not(feature = "cuda"))]
    let device = Device::Cpu;
    #[cfg(feature = "cuda")]
    let device = Device::Cuda(CudaDevice::new_with_stream(0)?);

    println!("Training on GeoStd data with device: {device:?}");

    let paths = find_paths(Path::new(GEOSTD_PATH))?;
    let (atom_type_vocab, el_vocab) = build_vocabs(&paths)?;

    let dataset = GeoStdMol2Dataset::new(&paths, atom_type_vocab.clone(), el_vocab.clone())?;


    let n_elems = el_vocab.len();
    let n_atom_types = atom_type_vocab.len();
    let hidden_dim = 128;

    let mut varmap = candle_nn::VarMap::new();
    let vb = VarBuilder::from_varmap(&mut varmap, DType::F32, &device);

    let model = MolGNN::new(vb, n_elems, n_atom_types, hidden_dim)?;

    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: 1e-3,
            ..Default::default()
        },
    )?;

    let n_epochs = 3;
    for epoch in 0..n_epochs {
        println!("Running epoch {epoch} / {n_elems}...");
        let mut running_loss = 0f32;

        for i in 0..dataset.len() {
            let batch = dataset.get(i, &device)?;

            let (type_logits, charges_pred) =
                model.forward(&batch.elem_ids, &batch.coords, &batch.edge_index)?;

            // We'll train only the charge head for now.
            // loss = MSE(predicted_charge, true_charge)
            let diff = (charges_pred - &batch.charges)?;
            let loss = diff.sqr()?.mean_all()?;

            opt.backward_step(&loss)?;

            // for logging
            running_loss += f32::from(loss.to_scalar::<f32>()?);

            let _ = type_logits;
        }

        println!(
            "Epoch {epoch} done. Avg loss: {}",
            running_loss / dataset.len() as f32
        );
    }

    // save all learned parameters
    varmap.save(MODEL_PATH)?;
    println!("Saved model to {MODEL_PATH}");

    Ok(())
}
