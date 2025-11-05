//! For inferring force field type and partial charge of small organic molecules using Amber's
//! GeoStd library as training data. Uses a neural net.
//!
//! todo: Use this to handle frcmod data as well.

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use bio_files::{AtomGeneric, BondGeneric, mol2::Mol2};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn as nn;
use candle_nn::{Embedding, Linear, VarBuilder, ops::sigmoid};

pub const MODEL_PATH: &str = "geostd_model.safetensors";
pub const GEOSTD_PATH: &str = "C:/users/the_a/Desktop/bio_misc/amber_geostd_test";

pub struct GeoStdMol2Dataset {
    mol2_paths: Vec<PathBuf>,
    atom_type_vocab: HashMap<String, usize>,
    elem_vocab: HashMap<String, usize>,
}

impl GeoStdMol2Dataset {
    pub fn new(
        mol2_paths: &[PathBuf],
        // todo: frcmod here?
        atom_type_vocab: HashMap<String, usize>,
        elem_vocab: HashMap<String, usize>,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            mol2_paths: mol2_paths.to_vec(),
            atom_type_vocab,
            elem_vocab,
        })
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

        let oov_elem_id = self.elem_vocab.len();

        for atom in atoms.iter() {
            let el_id = self
                .elem_vocab
                .get(&atom.element.to_letter())
                .cloned()
                .unwrap_or(oov_elem_id);
            elem_ids.push(el_id as i64);

            coords.push(atom.posit.x as f32);
            coords.push(atom.posit.y as f32);
            coords.push(atom.posit.z as f32);

            if let Some(ff) = &atom.force_field_type {
                if let Some(tid) = self.atom_type_vocab.get(ff) {
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

pub struct Batch {
    pub elem_ids: Tensor,
    pub coords: Tensor,
    pub edge_index: Tensor,
    pub type_ids: Tensor,
    pub has_type: Tensor,
    pub charges: Tensor,
    pub num_atoms: usize,
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
    atom_type_vocab: &HashMap<String, usize>,
    elem_vocab: &HashMap<String, usize>,
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
    device: &Device,
) -> candle_core::Result<Vec<(String, f32)>> {
    // ) -> candle_core::Result<Vec<Atom>> {
    let mut elem_ids = Vec::with_capacity(atoms.len());
    let mut coords = Vec::with_capacity(atoms.len() * 3);

    let oov_elem_id = elem_vocab.len();

    for atom in atoms.iter() {
        let el = &atom.element;
        elem_ids.push(
            elem_vocab
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

    let inv_type_vocab: HashMap<usize, String> = atom_type_vocab
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
