//! Fundamental data structures for small organic molecules / ligands

use std::{collections::HashMap, io};

use bio_apis::pubchem::ProteinStructure;
use bio_files::{ChargeType, Mol2, MolType, Pdbqt, ResidueEnd, Sdf};
use lin_alg::f64::{Quaternion, Vec3};
use rayon::prelude::*;

use crate::{
    State,
    docking_v2::{ConformationType, DockingSite, Pose},
    molecule::Chain,
};
use crate::{
    // docking::{ConformationType, DockingSite, Pose, prep::setup_flexibility},
    molecule::{Atom, Bond, MoleculeCommon, Residue},
};

const LIGAND_ABS_POSIT_OFFSET: f64 = 15.; // Å

/// A molecue representing a small organic molecule. Omits mol-generic fields.
#[derive(Debug, Default, Clone)]
pub struct MoleculeSmall {
    pub common: MoleculeCommon,
    pub lig_data: Option<Ligand>,
    pub pubchem_cid: Option<u32>,
    pub drugbank_id: Option<String>,
    /// FF type and partial charge on all atoms. Quick lookup flag.
    pub ff_params_loaded: bool,
    /// E.g., overrides for dihedral angles (part of the *bonded* dynamics calculation) for this
    /// specific molecule, as provided by Amber. Quick lookup flag.
    pub frcmod_loaded: bool,
    /// E.g. loaded proteins from Pubchem.
    pub associated_structures: Vec<ProteinStructure>,
}

impl MoleculeSmall {
    /// This constructor handles assumes details are ingested into a common format upstream. It adds
    /// them to the resulting structure, and augments it with bonds, hydrogen positions, and other things A/R.
    pub fn new(
        ident: String,
        atoms: Vec<Atom>,
        bonds: Vec<Bond>,
        metadata: HashMap<String, String>,
        // ff_params: &HashMap<String, ForceFieldParamsKeyed>,
        // lig_i: Option<usize>,
    ) -> Self {
        let mut frcmod_loaded = false;
        // If we've already loaded FRCMOD data for this ligand, update the status. Alternatively,
        // this will be updated when we load the FRCMOD file after.
        // if ff_params.keys().any(|k| k.eq_ignore_ascii_case(&ident)) {
        //     frcmod_loaded = true;
        // }

        // Handfled elsewhere for now.
        // Offset its position immediately if it's not the first loaded, to prevent ligands
        // from overlapping
        // let mut common = MoleculeCommon::new(ident, atoms, Some(bonds));
        // if let Some(i) = lig_i {
        //     let offset = LIGAND_ABS_POSIT_OFFSET * (i as f64);
        //     for posit in &mut common.atom_posits {
        //         posit.x += offset; // Arbitrary axis and direction.
        //     }
        // }

        let mut pubchem_cid = None;
        let mut drugbank_id = None;

        if let Some(id) = metadata.get("PUBCHEM_COMPOUND_CID") {
            pubchem_cid = Some(id.parse::<u32>().unwrap_or_default());
        }

        if let Some(db_name) = metadata.get("DATABASE_NAME") {
            if db_name.to_lowercase() == "drugbank" {
                if let Some(id) = metadata.get("DATABASE_ID") {
                    drugbank_id = Some(id.clone());
                }
            }
        }

        Self {
            common: MoleculeCommon::new(ident, atoms, Some(bonds), metadata),
            pubchem_cid,
            drugbank_id,
            frcmod_loaded,
            ..Default::default()
        }
    }
}

/// This data is related specifically to docking.
#[derive(Debug, Clone, Default)]
pub struct Ligand {
    pub anchor_atom: usize, // Index.
    /// Note: We may deprecate this in favor of our Amber MD-based approach to flexibility.
    pub flexible_bonds: Vec<usize>, // Index
    pub pose: Pose,
    pub docking_site: DockingSite,
}

impl Ligand {
    pub fn new(mol: &MoleculeSmall) -> Self {
        let mut ff_params_loaded = true;
        for atom in &mol.common.atoms {
            if atom.force_field_type.is_none() || atom.partial_charge.is_none() {
                ff_params_loaded = false;
                break;
            }
        }

        let mut result = Self {
            ..Default::default()
        };

        // result.set_anchor();
        // result.flexible_bonds = setup_flexibility(&mol.common);

        // result.pose.conformation_type = ConformationType::AssignedTorsions {
        //     torsions: result
        //         .flexible_bonds
        //         .iter()
        //         .map(|b| Torsion {
        //             bond: *b,
        //             dihedral_angle: 0.,
        //         })
        //         .collect(),
        // };

        // result.position_atoms(None);

        // result.reset_posits();

        result
    }
}

impl TryFrom<Mol2> for MoleculeSmall {
    type Error = io::Error;
    fn try_from(m: Mol2) -> Result<Self, Self::Error> {
        let atoms: Vec<_> = m.atoms.iter().map(|a| a.into()).collect();

        let bonds: Vec<Bond> = m
            .bonds
            .iter()
            .map(|b| Bond::from_generic(b, &atoms))
            .collect::<Result<_, _>>()?;

        // Note: We don't compute bonds here; we assume they're included in the molecule format.
        Ok(Self::new(m.ident, atoms, bonds, m.metadata.clone()))
    }
}

impl TryFrom<Sdf> for MoleculeSmall {
    type Error = io::Error;
    fn try_from(m: Sdf) -> Result<Self, Self::Error> {
        let atoms: Vec<_> = m.atoms.iter().map(|a| a.into()).collect();
        // let mut residues = Vec::with_capacity(m.residues.len());
        // for res in &m.residues {
        //     residues.push(Residue::from_generic(res, &atoms, ResidueEnd::Hetero)?);
        // }

        // let mut chains = Vec::with_capacity(m.chains.len());
        // for c in &m.chains {
        //     chains.push(Chain::from_generic(c, &atoms, &residues)?);
        // }

        let bonds: Vec<Bond> = m
            .bonds
            .iter()
            .map(|b| Bond::from_generic(b, &atoms))
            .collect::<Result<_, _>>()?;

        Ok(Self::new(m.ident, atoms, bonds, m.metadata.clone()))
    }
}

impl TryFrom<Pdbqt> for MoleculeSmall {
    type Error = io::Error;
    fn try_from(m: Pdbqt) -> Result<Self, Self::Error> {
        let atoms: Vec<_> = m.atoms.iter().map(|a| a.into()).collect();
        let mut residues = Vec::with_capacity(m.residues.len());
        for res in &m.residues {
            residues.push(Residue::from_generic(res, &atoms)?);
        }

        let mut chains = Vec::with_capacity(m.chains.len());
        for c in &m.chains {
            chains.push(Chain::from_generic(c, &atoms, &residues)?);
        }

        let bonds: Vec<Bond> = m
            .bonds
            .iter()
            .map(|b| Bond::from_generic(b, &atoms))
            .collect::<Result<_, _>>()?;

        Ok(Self::new(
            // todo: PDBQT metadata?
            m.ident,
            atoms,
            bonds,
            HashMap::new(),
        ))
    }
}

impl MoleculeSmall {
    /// Creates global positions for all atoms. This takes into account position, orientation, and if applicable,
    /// torsion angles from flexible bonds. Each pivot rotation rotates the side of the flexible bond that
    /// has fewer atoms; the intent is to minimize the overall position changes for these flexible bond angle
    /// changes.
    ///
    /// If we return None, use the existing atom_posits data; it has presumably been already set.
    pub fn position_atoms(&mut self, pose: Option<&Pose>) {
        let Some(data) = &self.lig_data else {
            return;
        };

        let pose_ = match pose {
            Some(p) => p,
            None => &data.pose,
        };

        match &pose_.conformation_type {
            ConformationType::AbsolutePosits => {
                // take no action; we are assigning and accessing the `atom_posits` field directly.
            }
            ConformationType::AssignedTorsions { torsions } => {
                if data.anchor_atom >= self.common.atoms.len() {
                    eprintln!(
                        "Error positioning ligand atoms: Anchor outside atom count. Atom cound: {:?}",
                        self.common.atoms.len()
                    );
                    return;
                }
                let anchor = self.common.atoms[data.anchor_atom].posit;

                let mut result: Vec<_> = self
                    .common
                    .atoms
                    .par_iter()
                    .map(|atom| {
                        let posit_rel = atom.posit - anchor;
                        pose_.anchor_posit + pose_.orientation.rotate_vec(posit_rel)
                    })
                    .collect();
                // Second pass: Rotations. For each flexible bond, divide all atoms into two groups:
                // those upstream of this bond, and those downstream. For all downstream atoms, rotate
                // by `torsions[i]`: The dihedral angle along this bond. If there are ambiguities in this
                // process, it may mean the bond should not have been marked as flexible.
                for torsion in torsions {
                    let bond = &self.common.bonds[torsion.bond];

                    // -- Step 1: measure how many atoms would be "downstream" from each side
                    let side0_downstream = self.find_downstream_atoms(bond.atom_1, bond.atom_0);
                    let side1_downstream = self.find_downstream_atoms(bond.atom_0, bond.atom_1);

                    // -- Step 2: pick the pivot as the side with a larger subtree
                    let (pivot_idx, side_idx, downstream_atom_indices) =
                        if side0_downstream.len() > side1_downstream.len() {
                            // side0_downstream means "downstream from atom_1 ignoring bond to atom_0"
                            // => so pivot is atom_0, side is atom_1
                            (bond.atom_0, bond.atom_1, side1_downstream)
                        } else {
                            // side1_downstream has equal or more
                            (bond.atom_1, bond.atom_0, side0_downstream)
                        };

                    // pivot and side positions
                    let pivot_pos = result[pivot_idx];
                    let side_pos = result[side_idx];
                    let axis_vec = (side_pos - pivot_pos).to_normalized();

                    // Build the Quaternion for this rotation
                    let rotator =
                        Quaternion::from_axis_angle(axis_vec, torsion.dihedral_angle as f64);

                    // Now apply the rotation to each downstream atom:
                    for &atom_idx in &downstream_atom_indices {
                        let old_pos = result[atom_idx];
                        let relative = old_pos - pivot_pos;
                        let new_pos = pivot_pos + rotator.rotate_vec(relative);
                        result[atom_idx] = new_pos;
                    }
                }

                self.common.atom_posits = result;
            }
        }
    }

    /// Reset atom positions to be at their internal values, e.g. as present in the Mol2 or SDF files.
    pub fn reset_posits(&mut self) {
        self.common.atom_posits = self.common.atoms.iter().map(|a| a.posit).collect();
    }

    /// Separate from constructor; run when the pose changes, for now.
    pub fn set_anchor(&mut self) {
        let Some(data) = &mut self.lig_data else {
            return;
        };

        let mut center = Vec3::new_zero();
        for atom in &self.common.atoms {
            center += atom.posit;
        }
        center /= self.common.atoms.len() as f64;

        let mut anchor_atom = 0;
        let mut best_dist = 999999.;

        for (i, atom) in self.common.atoms.iter().enumerate() {
            let dist = (atom.posit - center).magnitude();
            if dist < best_dist {
                best_dist = dist;
                anchor_atom = i;
            }
        }

        data.anchor_atom = anchor_atom;
    }

    /// We use this to rotate flexible molecules around torsion (e.g. dihedral) angles.
    /// `pivot` and `side` are atom indices in the molecule.
    pub fn find_downstream_atoms(&self, pivot: usize, side: usize) -> Vec<usize> {
        // adjacency_list[atom] -> list of neighbors
        // We want all atoms reachable from `side` when we remove the edge (side->pivot).
        let mut visited = vec![false; self.common.atoms.len()];
        let mut stack = vec![side];
        let mut result = vec![];

        visited[side] = true;

        while let Some(current) = stack.pop() {
            result.push(current);

            for &nbr in &self.common.adjacency_list[current] {
                // skip the pivot to avoid going back across the chosen bond
                if nbr == pivot {
                    continue;
                }
                if !visited[nbr] {
                    visited[nbr] = true;
                    stack.push(nbr);
                }
            }
        }

        result
    }

    pub fn to_mol2(&self) -> Mol2 {
        let atoms = self.common.atoms.iter().map(|a| a.to_generic()).collect();
        let bonds = self.common.bonds.iter().map(|b| b.to_generic()).collect();

        Mol2 {
            ident: self.common.ident.clone(),
            metadata: self.common.metadata.clone(),
            mol_type: MolType::Small,
            charge_type: ChargeType::None,
            comment: None,
            atoms,
            bonds,
        }
    }

    pub fn to_sdf(&self) -> Sdf {
        let atoms = self.common.atoms.iter().map(|a| a.to_generic()).collect();
        let bonds = self.common.bonds.iter().map(|b| b.to_generic()).collect();

        let mut metadata = self.common.metadata.clone();

        // Note: These may be redundant with metadata already loaded.
        if let Some(id) = &self.pubchem_cid {
            metadata.insert("PUBCHEM_COMPOUND_CID".to_string(), id.to_string());
        }

        if let Some(id) = &self.drugbank_id {
            metadata.insert("DATABASE_ID".to_string(), id.clone());
            metadata.insert("DATABASE_NAME".to_string(), "drugbank".to_string());
        }

        Sdf {
            ident: self.common.ident.clone(),
            metadata,
            atoms,
            bonds,
            chains: Vec::new(),
            residues: Vec::new(),
        }
    }

    pub fn to_pdbqt(&self) -> Pdbqt {
        let atoms = self.common.atoms.iter().map(|a| a.to_generic()).collect();
        let bonds = self.common.bonds.iter().map(|b| b.to_generic()).collect();

        Pdbqt {
            ident: self.common.ident.clone(),
            mol_type: MolType::Small,
            charge_type: ChargeType::None,
            comment: None,
            atoms,
            bonds,
            chains: Vec::new(),
            residues: Vec::new(),
        }
    }
}

impl MoleculeSmall {
    /// For example, this can be used to create a ligand from a residue that was loaded with a mmCIF
    /// file from RCSB. It can then be used for docking, or saving to a Mol2 or SDF file.
    ///
    /// `atoms` here should be the full set, as indexed by `res`, unless `use_sns` is true.
    /// `use_sns` = false is faster.
    ///
    /// We assume the residue is already populated with hydrogens.
    ///
    /// We reposition its atoms to be around the origin.
    /// todo: How do we get partial charge and ff type? We normally *get* those from Amber-provided
    /// todo Mol2 files. If we do this from an AA, it works, but it doesn't from hereo residues.
    ///
    pub fn from_res(res: &Residue, atoms: &[Atom], bonds: &[Bond], use_sns: bool) -> Self {
        // todo: Handle `use_sns`.
        let mut atoms_this = Vec::with_capacity(res.atoms.len());
        let mut atom_indices = Vec::with_capacity(res.atoms.len());

        for (i, atom_i) in res.atoms.iter().enumerate() {
            atoms_this.push(Atom {
                serial_number: i as u32 + 1,
                // residue: Some(0), // The one and only residue: The one we create this from.
                residue: None,
                chain: None,
                ..atoms[*atom_i].clone()
            });
            atom_indices.push(*atom_i);
        }

        // Reposition atoms so they're near the origin.
        if !atoms_this.is_empty() {
            let move_vec = atoms_this[0].posit;
            for atom in &mut atoms_this {
                atom.posit -= move_vec;
            }
        }

        let bonds_this = bonds
            .iter()
            .filter(|b| atom_indices.contains(&b.atom_0) || atom_indices.contains(&b.atom_1))
            .cloned()
            .collect();

        // This allows saving as Mol2, for example, with residue types, without breaking
        // bindings
        let _res_new = Residue {
            atoms: Vec::new(),
            atom_sns: atoms_this.iter().map(|a| a.serial_number).collect(),
            ..res.clone()
        };

        Self::new(
            res.res_type.to_string(),
            atoms_this,
            bonds_this,
            HashMap::new(),
        )
    }

    /// Updates we wish to do shortly after load, but need access to State for.
    pub fn update_aux(&mut self, state: &State) {
        if let Some(i) = &state.volatile.active_lig {
            let offset = LIGAND_ABS_POSIT_OFFSET * (*i as f64);
            for posit in &mut self.common.atom_posits {
                posit.x += offset; // Arbitrary axis and direction.
            }
        }

        self.ff_params_loaded = true;
        for atom in &self.common.atoms {
            if atom.force_field_type.is_none() || atom.partial_charge.is_none() {
                self.ff_params_loaded = false;
                break;
            }
        }

        if state
            .lig_specific_params
            .keys()
            .any(|k| k.eq_ignore_ascii_case(&self.common.ident))
        {
            self.frcmod_loaded = true;
        }
    }
}
