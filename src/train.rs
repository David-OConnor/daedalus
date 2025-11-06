//! This is the entry point for a standalone application to train the data.
//! Run `cargo b --release --bin train

use std::path::{Path, PathBuf};

use bio_files::Mol2;

mod param_inference;

use candle_core::{CudaDevice, DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, loss};
use graphics::app_utils::save;
use param_inference::{
    MolGNN, Vocabs,
    files::{GEOSTD_PATH, MODEL_PATH, VOCAB_PATH},
};
use rand::seq::SliceRandom;

use crate::param_inference::files::find_paths;

// Higher = perhaps better training, but slower to train.
// todo: Try setting max of 50-100 epochs, and stop early A/R if val loss
// todo hasn't improved.
const N_EPOCHS: u8 = 10;
// Stop training if we have this many epochs without improvement.
const EARLY_STOPPING_PATIENCE: u8 = 7;

// Bigger hidden dim: more capacity to learn patterns, but slower and easier to overfit.
// Smaller hidden dim: faster, less capacity.
const HIDDEN_DIM: usize = 128; // todo: Try 256 as well, and compare.

// Higher learning rate: Faster, but can overshoot. Lower: Safer but slower.
// 1e-3 is a good default.
const LEARNING_RATE: f64 = 1e-3; // todo: What should this be? What is it?

struct GeoStdMol2Dataset {
    mol2_paths: Vec<PathBuf>,
    frcmod_paths: Vec<PathBuf>,
    vocabs: Vocabs,
}

pub struct Batch {
    pub elem_ids: Tensor,
    pub coords: Tensor,
    pub edge_index: Tensor,
    pub type_ids: Tensor,
    pub has_type: Tensor,
    pub charges: Tensor,
    pub num_atoms: usize,
}

impl GeoStdMol2Dataset {
    pub fn new(mol2_paths: &[PathBuf], frcmod_paths: &[PathBuf], vocabs: Vocabs) -> Self {
        Self {
            mol2_paths: mol2_paths.to_vec(),
            frcmod_paths: frcmod_paths.to_vec(),
            vocabs,
        }
    }

    pub fn len(&self) -> usize {
        self.mol2_paths.len()
    }

    pub fn get(&self, idx: usize, device: &Device) -> candle_core::Result<Batch> {
        let mol = Mol2::load(&self.mol2_paths[idx])?;
        let atoms = &mol.atoms;
        let bonds = &mol.bonds;
        let n = atoms.len();

        let mut elem_ids = Vec::with_capacity(n);
        let mut has_type = Vec::with_capacity(n);
        let mut type_ids = Vec::with_capacity(n);
        let mut charges = Vec::with_capacity(n);
        let mut coords = Vec::with_capacity(n * 3);

        let oov_elem_id = self.vocabs.el.len();

        for atom in atoms.iter() {
            let el_id = self
                .vocabs
                .el
                .get(&atom.element.to_letter())
                .cloned()
                .unwrap_or(oov_elem_id);
            elem_ids.push(el_id as i64);

            coords.push(atom.posit.x as f32);
            coords.push(atom.posit.y as f32);
            coords.push(atom.posit.z as f32);

            if let Some(ff) = &atom.force_field_type {
                if let Some(tid) = self.vocabs.atom_type.get(ff) {
                    has_type.push(1.0f32);
                    type_ids.push(*tid as i64);
                } else {
                    has_type.push(0.0f32);
                    type_ids.push(-1);
                }
            } else {
                has_type.push(0.0f32);
                type_ids.push(-1);
            }

            if let Some(pc) = atom.partial_charge {
                charges.push(pc);
            } else {
                charges.push(0.0);
            }
        }

        let mut edge_index_vec: Vec<i64> = Vec::new();
        for bond in bonds.iter() {
            let i = (bond.atom_0_sn - 1) as i64;
            let j = (bond.atom_1_sn - 1) as i64;
            edge_index_vec.push(i);
            edge_index_vec.push(j);
            edge_index_vec.push(j);
            edge_index_vec.push(i);
        }

        let elem_ids = Tensor::from_slice(&elem_ids, (n,), device)?;
        let coords = Tensor::from_slice(&coords, (n, 3), device)?;
        let type_ids = Tensor::from_slice(&type_ids, (n,), device)?;
        let has_type = Tensor::from_slice(&has_type, (n,), device)?;
        let charges = Tensor::from_slice(&charges, (n,), device)?;

        let edge_index = if edge_index_vec.is_empty() {
            Tensor::zeros((0, 2), DType::I64, device)?
        } else {
            let m = edge_index_vec.len() / 2;
            Tensor::from_slice(&edge_index_vec, (m, 2), device)?
        };

        Ok(Batch {
            elem_ids,
            coords,
            edge_index,
            type_ids,
            has_type,
            charges,
            num_atoms: n,
        })
    }
}

fn main() -> candle_core::Result<()> {
    // todo
    // #[cfg(feature = "cuda")]
    // let device = Device::Cuda(CudaDevice::new_with_stream(0)?);
    // #[cfg(not(feature = "cuda"))]
    let device = Device::Cpu;

    println!("Training on GeoStd data with device: {device:?}");

    let (paths_mol2, paths_frcmod) = find_paths(Path::new(GEOSTD_PATH))?;

    let vocabs = Vocabs::new(&paths_mol2)?;
    let n_elems = vocabs.el.len();
    let n_atom_types = vocabs.atom_type.len();

    save(Path::new(VOCAB_PATH), &vocabs)?;
    println!("Vocabs built and saved to {VOCAB_PATH}");

    let dataset = GeoStdMol2Dataset::new(&paths_mol2, &paths_frcmod, vocabs);

    let mut varmap = candle_nn::VarMap::new();
    let vb = VarBuilder::from_varmap(&mut varmap, DType::F32, &device);

    let model = MolGNN::new(vb, n_elems, n_atom_types, HIDDEN_DIM)?;

    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: LEARNING_RATE,
            ..Default::default()
        },
    )?;

    let mut rng = rand::rng();

    // ---- train/val split ----
    let mut all_idxs: Vec<usize> = (0..dataset.len()).collect();
    all_idxs.shuffle(&mut rng);
    let split = (dataset.len() as f32 * 0.8) as usize;
    let train_idxs = all_idxs[..split].to_vec();
    let val_idxs = all_idxs[split..].to_vec();

    let mut best_val_loss = f32::INFINITY;
    let mut epochs_without_improve: u8 = 0;

    for epoch in 0..N_EPOCHS {
        println!("Running epoch {epoch} / {N_EPOCHS}...");

        let mut train_order = train_idxs.clone();
        train_order.shuffle(&mut rng());

        let mut running_loss = 0.;

        for i in train_order.iter() {
            let batch = dataset.get(*i, &device)?;

            let (type_logits, charges_pred) =
                model.forward(&batch.elem_ids, &batch.coords, &batch.edge_index)?;

            let diff = (charges_pred - &batch.charges)?;
            let charge_loss = diff.sqr()?.mean_all()?;

            let type_ids_host = batch.type_ids.to_vec1::<i64>()?;
            let mut valid_idx: Vec<i64> = Vec::new();
            let mut valid_targets: Vec<i64> = Vec::new();
            for (idx, tid) in type_ids_host.iter().enumerate() {
                if *tid >= 0 {
                    valid_idx.push(idx as i64);
                    valid_targets.push(*tid);
                }
            }

            let type_loss = if !valid_idx.is_empty() {
                let idx_tensor = Tensor::from_slice(&valid_idx, (valid_idx.len(),), &device)?;
                let logits_sel = type_logits.index_select(&idx_tensor, 0)?;
                let targets_sel =
                    Tensor::from_slice(&valid_targets, (valid_targets.len(),), &device)?;
                loss::cross_entropy(&logits_sel, &targets_sel)?
            } else {
                Tensor::zeros((), DType::F32, &device)?
            };

            let loss = (&charge_loss + &type_loss)?;

            opt.backward_step(&loss)?;

            running_loss += f32::from(loss.to_scalar::<f32>()?);
        }

        let train_avg = running_loss / train_idxs.len() as f32;

        // ---- validation loss ----
        let mut val_loss_sum = 0f32;
        for i in val_idxs.iter() {
            let batch = dataset.get(*i, &device)?;

            let (type_logits, charges_pred) =
                model.forward(&batch.elem_ids, &batch.coords, &batch.edge_index)?;

            let diff = (charges_pred - &batch.charges)?;
            let charge_loss = diff.sqr()?.mean_all()?;

            let type_ids_host = batch.type_ids.to_vec1::<i64>()?;
            let mut valid_idx: Vec<i64> = Vec::new();
            let mut valid_targets: Vec<i64> = Vec::new();
            for (idx, tid) in type_ids_host.iter().enumerate() {
                if *tid >= 0 {
                    valid_idx.push(idx as i64);
                    valid_targets.push(*tid);
                }
            }

            let type_loss = if !valid_idx.is_empty() {
                let idx_tensor = Tensor::from_slice(&valid_idx, (valid_idx.len(),), &device)?;
                let logits_sel = type_logits.index_select(&idx_tensor, 0)?;
                let targets_sel =
                    Tensor::from_slice(&valid_targets, (valid_targets.len(),), &device)?;
                loss::cross_entropy(&logits_sel, &targets_sel)?
            } else {
                Tensor::zeros((), DType::F32, &device)?
            };

            let loss = (&charge_loss + &type_loss)?;
            val_loss_sum += f32::from(loss.to_scalar::<f32>()?);
        }

        let val_avg = val_loss_sum / val_idxs.len() as f32;

        println!("Epoch {epoch} done. Train avg loss: {train_avg}, Val avg loss: {val_avg}");

        if val_avg < best_val_loss {
            best_val_loss = val_avg;
            epochs_without_improve = 0;
            varmap.save(MODEL_PATH)?; // keep best
            println!("New best val loss. Saved model to {MODEL_PATH}");
        } else {
            epochs_without_improve += 1;
            if epochs_without_improve >= EARLY_STOPPING_PATIENCE {
                println!(
                    "Early stopping at epoch {epoch} (no improvement for {EARLY_STOPPING_PATIENCE} epochs)"
                );
                break;
            }
        }
    }

    // save all learned parameters
    varmap.save(MODEL_PATH)?;

    println!("Saved model to {MODEL_PATH}");

    Ok(())
}
