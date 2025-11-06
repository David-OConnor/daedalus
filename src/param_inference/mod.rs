//! For inferring force field type and partial charge of small organic molecules using Amber's
//! GeoStd library as training data. Uses a neural net.
//!
//! todo: Use this to handle frcmod data as well.

pub mod files; // Pub so the training program can access it.

use std::{
    collections::{BTreeSet, HashMap},
    path::{Path, PathBuf},
};

use bincode::{Decode, Encode};
use bio_files::{AtomGeneric, BondGeneric, mol2::Mol2};
use candle_core::{CudaDevice, DType, Device, IndexOp, Module, Tensor};
use candle_nn as nn;
use candle_nn::{Embedding, Linear, VarBuilder, ops::sigmoid};
use graphics::app_utils::load;

use crate::param_inference::files::{MODEL_PATH, VOCAB_PATH};

/// We save this to file during training, and load it during inference.
#[derive(Debug, Encode, Decode)]
pub struct Vocabs {
    pub el: HashMap<String, usize>,
    pub atom_type: HashMap<String, usize>,
}

impl Vocabs {
    pub fn new(mol2_paths: &[PathBuf]) -> candle_core::Result<Self> {
        let mut elems: BTreeSet<String> = BTreeSet::new();
        let mut ff_types: BTreeSet<String> = BTreeSet::new();

        for path in mol2_paths {
            let mol = Mol2::load(path)?;

            for atom in mol.atoms.iter() {
                elems.insert(atom.element.to_letter());

                // Ideally we won't encounter this with the Geostd data set.
                let Some(ff) = &atom.force_field_type else {
                    eprintln!("Error: Missing FF type on Geostd atom: {atom}");
                    continue;
                };

                ff_types.insert(ff.clone());
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

        Ok(Self {
            el: el_map,
            atom_type: atom_type_map,
        })
    }
}

// -------- GRU cell (hidden_dim -> hidden_dim) --------

struct GruCell {
    w_ih: Linear, // in -> 3*h
    w_hh: Linear, // h -> 3*h
    hidden_dim: usize,
}

impl GruCell {
    fn new(vb: VarBuilder, hidden_dim: usize) -> candle_core::Result<Self> {
        let w_ih = nn::linear(hidden_dim, 3 * hidden_dim, vb.pp("w_ih"))?;
        let w_hh = nn::linear(hidden_dim, 3 * hidden_dim, vb.pp("w_hh"))?;
        Ok(Self {
            w_ih,
            w_hh,
            hidden_dim,
        })
    }

    fn forward(&self, x: &Tensor, h: &Tensor) -> candle_core::Result<Tensor> {
        // x, h: [N, H]
        let ih = self.w_ih.forward(x)?; // [N, 3H]
        let hh = self.w_hh.forward(h)?; // [N, 3H]

        let hsize = self.hidden_dim;
        let i_r = ih.narrow(1, 0, hsize)?;
        let i_z = ih.narrow(1, hsize, hsize)?;
        let i_n = ih.narrow(1, 2 * hsize, hsize)?;

        let h_r = hh.narrow(1, 0, hsize)?;
        let h_z = hh.narrow(1, hsize, hsize)?;
        let h_n = hh.narrow(1, 2 * hsize, hsize)?;

        let r = sigmoid(&(i_r + h_r)?)?;
        let z = sigmoid(&(i_z + h_z)?)?;
        let n = (i_n + (r * h_n)?)?.tanh()?;

        let one = Tensor::ones_like(&z)?;
        let one_minus_z = one.sub(&z)?;

        (&one_minus_z * n)? + (&z * h)?
    }
}

// -------- Message passing layer --------

struct MessagePassingLayer {
    msg: Linear,
    gru: GruCell,
    hidden_dim: usize,
}

impl MessagePassingLayer {
    fn new(vb: VarBuilder, hidden_dim: usize) -> candle_core::Result<Self> {
        let msg = nn::linear(hidden_dim * 2, hidden_dim, vb.pp("msg"))?;
        let gru = GruCell::new(vb.pp("gru"), hidden_dim)?;
        Ok(Self {
            msg,
            gru,
            hidden_dim,
        })
    }

    fn forward(&self, h: &Tensor, edge_index: &Tensor) -> candle_core::Result<Tensor> {
        if edge_index.dims()[0] == 0 {
            return Ok(h.clone());
        }

        let src = edge_index.i((.., 0))?.contiguous()?;
        let dst = edge_index.i((.., 1))?.contiguous()?;

        let h_src = h.index_select(&src, 0)?;
        let h_dst = h.index_select(&dst, 0)?;

        let m_in = Tensor::cat(&[h_src, h_dst], 1)?;
        let m = self.msg.forward(&m_in)?.relu()?;

        let mut agg = Tensor::zeros_like(h)?.contiguous()?;
        let m = m.contiguous()?;

        agg = agg.index_add(&dst, &m, 0)?;

        let h_new = self.gru.forward(&agg, h)?;
        Ok(h_new)
    }
}

// -------- Model --------

pub struct MolGNN {
    elem_emb: Embedding,
    coord_lin: Linear,
    mp1: MessagePassingLayer,
    mp2: MessagePassingLayer,
    mp3: MessagePassingLayer,
    type_head: Linear,
    charge_head: Linear,
}

impl MolGNN {
    pub fn new(
        vb: VarBuilder,
        n_elems: usize,
        n_atom_types: usize,
        hidden_dim: usize,
    ) -> candle_core::Result<Self> {
        let elem_emb = nn::embedding(n_elems + 1, hidden_dim, vb.pp("elem_emb"))?;

        let coord_lin = nn::linear(3, hidden_dim, vb.pp("coord_lin"))?;
        let mp1 = MessagePassingLayer::new(vb.pp("mp1"), hidden_dim)?;
        let mp2 = MessagePassingLayer::new(vb.pp("mp2"), hidden_dim)?;
        let mp3 = MessagePassingLayer::new(vb.pp("mp3"), hidden_dim)?;
        let type_head = nn::linear(hidden_dim, n_atom_types, vb.pp("type_head"))?;
        let charge_head = nn::linear(hidden_dim, 1, vb.pp("charge_head"))?;
        Ok(Self {
            elem_emb,
            coord_lin,
            mp1,
            mp2,
            mp3,
            type_head,
            charge_head,
        })
    }

    pub fn forward(
        &self,
        elem_ids: &Tensor,
        coords: &Tensor,
        edge_index: &Tensor,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let h_emb = self.elem_emb.forward(elem_ids)?;
        let h_coord = self.coord_lin.forward(coords)?;
        let mut h = (h_emb + h_coord)?;

        h = self.mp1.forward(&h, edge_index)?;
        h = self.mp2.forward(&h, edge_index)?;
        h = self.mp3.forward(&h, edge_index)?;

        let type_logits = self.type_head.forward(&h)?;
        let charges = self.charge_head.forward(&h)?.squeeze(1)?;
        Ok((type_logits, charges))
    }
}

// -------- Inference --------

pub fn run_inference(
    model: &MolGNN,
    vocabs: &Vocabs,
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    device: &Device,
) -> candle_core::Result<Vec<(String, f32)>> {
    let mut elem_ids = Vec::with_capacity(atoms.len());
    let mut coords = Vec::with_capacity(atoms.len() * 3);

    let oov_elem_id = vocabs.el.len();

    for atom in atoms.iter() {
        let el = &atom.element;
        elem_ids.push(
            vocabs
                .el
                .get(&el.to_letter())
                .cloned()
                .unwrap_or(oov_elem_id) as i64,
        );

        coords.push(atom.posit.x as f32);
        coords.push(atom.posit.y as f32);
        coords.push(atom.posit.z as f32);
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

    let elem_ids = Tensor::from_slice(&elem_ids, (atoms.len(),), device)?;
    let coords = Tensor::from_slice(&coords, (atoms.len(), 3), device)?;

    let edge_index = if edge_index_vec.is_empty() {
        Tensor::zeros((0, 2), DType::I64, device)?
    } else {
        Tensor::from_slice(&edge_index_vec, (edge_index_vec.len() / 2, 2), device)?
    };

    let (type_logits, charges) = model.forward(&elem_ids, &coords, &edge_index)?;

    let type_ids = type_logits.argmax(1)?.to_dtype(DType::I64)?;
    let type_ids: Vec<i64> = type_ids.to_vec1()?;
    let charges: Vec<f32> = charges.to_vec1()?;

    let inv_type_vocab: HashMap<usize, String> = vocabs
        .atom_type
        .iter()
        .map(|(k, v)| (*v, k.clone()))
        .collect();

    let mut result = Vec::with_capacity(atoms.len());
    for i in 0..atoms.len() {
        let Some(ff) = inv_type_vocab.get(&(type_ids[i] as usize)) else {
            return Err(candle_core::error::Error::Msg("Uhoh".to_string()));
        };

        result.push((ff.to_string(), charges[i]))
    }

    Ok(result)
}

pub fn test_inference() {
    let mol = Mol2::load(Path::new("./molecules/CPB.mol2")).unwrap();

    // todo
    // #[cfg(feature = "cuda")]
    // let dev_candle = Device::Cuda(CudaDevice::new_with_stream(0).unwrap());
    // #[cfg(not(feature = "cuda"))]
    let dev_candle = Device::Cpu;

    println!("Running inference on GeoStd data with device: {dev_candle:?}");

    // let paths = find_paths(&mol2_dir).unwrap();
    // let (el_vocab, atom_type_vocab) = build_vocabs(&paths).unwrap();
    let vocabs: Vocabs = load(&Path::new(VOCAB_PATH)).unwrap();

    let n_elems = vocabs.el.len();
    let n_atom_types = vocabs.atom_type.len();
    let hidden_dim = 128;

    // Make a varmap and LOAD the trained weights
    let mut varmap = candle_nn::VarMap::new();

    // Build the model from the loaded varmap
    let vb = VarBuilder::from_varmap(&mut varmap, DType::F32, &dev_candle);
    let model = MolGNN::new(vb, n_elems, n_atom_types, hidden_dim).unwrap();
    varmap.load(MODEL_PATH).unwrap();

    // Run inference
    let preds = run_inference(&model, &vocabs, &mol.atoms, &mol.bonds, &dev_candle).unwrap();

    for (i, (ff, q)) in preds.iter().enumerate() {
        println!("SN: {}: {ff}  q={q}", i + 1);
    }
}
