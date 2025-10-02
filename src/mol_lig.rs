//! Fundamental data structures for small organic molecules / ligands

use std::{collections::HashMap, io, path::PathBuf, time::Instant};

use bio_apis::{amber_geostd, pubchem::ProteinStructure};
use bio_files::{
    ChargeType, Mol2, MolType, Pdbqt, Sdf,
    md_params::{ForceFieldParams, ForceFieldParamsVec},
};
use lin_alg::f64::{Quaternion, Vec3};
use na_seq::Element;
use rayon::prelude::*;

use crate::{
    State,
    docking_v2::{ConformationType, DockingSite, Pose},
    molecule::{
        Atom, Bond, Chain, MolGenericTrait, MolType as Mt, MoleculeCommon, MoleculeGenericRef,
        Residue,
    },
    util::handle_err,
};

const LIGAND_ABS_POSIT_OFFSET: f64 = 15.; // Å

/// A molecue representing a small organic molecule. Omits mol-generic fields.
#[derive(Debug, Default, Clone)]
pub struct MoleculeSmall {
    pub common: MoleculeCommon,
    pub lig_data: Option<Ligand>,
    /// Also used for Amber Geostd.
    pub pdbe_id: Option<String>,
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
        path: Option<PathBuf>,
    ) -> Self {
        let mut pdbe_id = None;
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
                // This seems to be valid for Drugbank-sourced molecules.
                if let Ok(id) = ident.parse::<u32>() {
                    pubchem_cid = Some(id);
                }
            }
        }

        if ident.len() <= 4 {
            // This is a guess
            pdbe_id = Some(ident.clone());
        }

        Self {
            common: MoleculeCommon::new(ident, atoms, bonds, metadata, path),
            pdbe_id,
            pubchem_cid,
            drugbank_id,
            ..Default::default()
        }
    }
}

impl MolGenericTrait for MoleculeSmall {
    fn common(&self) -> &MoleculeCommon {
        &self.common
    }

    fn common_mut(&mut self) -> &mut MoleculeCommon {
        &mut self.common
    }

    fn to_ref(&self) -> MoleculeGenericRef<'_> {
        MoleculeGenericRef::Ligand(self)
    }

    fn mol_type(&self) -> crate::molecule::MolType {
        crate::molecule::MolType::Ligand
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
        // Handle path after; not supported by TryFrom.
        Ok(Self::new(m.ident, atoms, bonds, m.metadata.clone(), None))
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

        // Handle path and state-specific items after; not supported by TryFrom.
        Ok(Self::new(m.ident, atoms, bonds, m.metadata.clone(), None))
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

        // Handle path after; not supported by TryFrom.
        Ok(Self::new(
            m.ident,
            atoms,
            bonds,
            HashMap::new(), // todo: Metadata?
            None,
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
    pub fn from_res(res: &Residue, atoms: &[Atom], bonds: &[Bond]) -> Self {
        // todo: Handle `use_sns`.
        let mut atoms_this = Vec::with_capacity(res.atoms.len());

        // We use this map when rebuilding bonds.
        // Old index: (new index, new sn)
        let mut bond_map = HashMap::new();

        for (i, &atom_i_orig) in res.atoms.iter().enumerate() {
            let atom = &atoms[atom_i_orig];

            let serial_number = i as u32 + 1;
            bond_map.insert(atom_i_orig, (i, serial_number));

            atoms_this.push(Atom {
                serial_number,
                residue: None,
                chain: None,
                ..atom.clone()
            });
        }

        // todo: Consider something better like near the original spot, but spaced out from origin.
        // Reposition atoms so they're near the origin.
        // if !atoms_this.is_empty() {
        //     let move_vec = atoms_this[0].posit;
        //     for atom in &mut atoms_this {
        //         atom.posit -= move_vec;
        //     }
        // }

        let atom_orig_i: Vec<_> = bond_map.keys().collect();
        let mut bonds_this: Vec<_> = bonds
            .iter()
            .filter(|b| atom_orig_i.contains(&&b.atom_0) && atom_orig_i.contains(&&b.atom_1))
            .cloned()
            .collect();

        let mut bonds_new = Vec::with_capacity(bonds_this.len());
        for bond in &bonds_this {
            let (atom_0, atom_0_sn) = bond_map.get(&bond.atom_0).unwrap();
            let (atom_1, atom_1_sn) = bond_map.get(&bond.atom_1).unwrap();

            bonds_new.push(Bond {
                bond_type: bond.bond_type,
                atom_0_sn: *atom_0_sn,
                atom_1_sn: *atom_1_sn,
                atom_0: *atom_0,
                atom_1: *atom_1,
                is_backbone: false,
            })
        }

        let name = res.res_type.to_string();
        let mut result = Self::new(name.clone(), atoms_this, bonds_new, HashMap::new(), None);

        result.pdbe_id = Some(name);
        result
    }

    /// Unfortunately, we can't directly map atoms from our original molecule to
    /// the Geostd one. We could do this with coordinates, but that might be complicated.
    /// For now, we perform a sanity check about atom count by element. If it passes,
    /// we replace molecule atom and bond data with that loaded from the mol2.
    fn replace_with_geostd(
        &mut self,
        ident: &str,
        lig_specific: &mut HashMap<String, ForceFieldParams>,
    ) {
        println!("Attempting to load Amber Geostd dynamics data for this molecule...");
        let start = Instant::now();

        let Ok(data) = amber_geostd::load_mol_files(&ident) else {
            let elapsed = start.elapsed().as_millis();
            println!("Unable to find data, took {elapsed:.1}ms");
            return;
        };

        if !self.ff_params_loaded {
            let Ok(mol2) = Mol2::new(&data.mol2) else {
                return;
            };

            let mut count_c_orig: u32 = 0;
            let mut count_n_orig: u32 = 0;
            let mut count_o_orig: u32 = 0;
            let mut count_h_orig: u32 = 0;
            //
            let mut count_c_amber: u32 = 0;
            let mut count_n_amber: u32 = 0;
            let mut count_o_amber: u32 = 0;
            let mut count_h_amber: u32 = 0;

            for atom in &self.common.atoms {
                match atom.element {
                    Element::Carbon => count_c_orig += 1,
                    Element::Nitrogen => count_n_orig += 1,
                    Element::Oxygen => count_o_orig += 1,
                    Element::Hydrogen => count_h_orig += 1,
                    _ => {}
                }
            }
            for atom in &mol2.atoms {
                match atom.element {
                    Element::Carbon => count_c_amber += 1,
                    Element::Nitrogen => count_n_amber += 1,
                    Element::Oxygen => count_o_amber += 1,
                    Element::Hydrogen => count_h_amber += 1,
                    _ => {}
                }
            }

            if count_c_orig != count_c_amber
                || count_n_orig != count_n_amber
                || count_o_orig != count_o_amber
                || count_h_orig != count_h_amber
            {
                eprintln!(
                    "Unable to load Amber Geostd data for this molecule; atom count mismatch."
                );

                return;
            }

            let mol: Self = match mol2.try_into() {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("Problem loading Mol2 from geostd: {e}");
                    return; // OK only if this fn returns ()
                }
            };

            self.common.atoms = mol.common.atoms;
            self.common.bonds = mol.common.bonds;
            self.common.atom_posits = mol.common.atom_posits;
            self.common.adjacency_list = mol.common.adjacency_list;

            self.ff_params_loaded = true;

            let elapsed = start.elapsed().as_millis();
            println!("Loaded Amber Geostd in {elapsed:.1}ms");
        }

        if !self.frcmod_loaded {
            if let Some(f) = data.frcmod {
                if let Ok(frcmod) = ForceFieldParamsVec::from_frcmod(&f) {
                    lig_specific.insert(self.common.ident.clone(), ForceFieldParams::new(&frcmod));
                    self.frcmod_loaded = true;
                }
            }
        }
    }

    /// Updates we wish to do shortly after load, but need access to State for.
    pub fn update_aux(
        &mut self,
        active_mol: &Option<(Mt, usize)>,
        lig_specific: &mut HashMap<String, ForceFieldParams>,
    ) {
        if let Some((_, i)) = active_mol {
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

        if lig_specific
            .keys()
            .any(|k| k.eq_ignore_ascii_case(&self.common.ident))
        {
            self.frcmod_loaded = true;
        }

        // Attempt to load Amber GeoStd force field names and partial charges.
        // Note: For now at least, we override any existing partial charges.
        // This is probably OK, as we assume Amber parameters elsewhere for now.
        if !self.ff_params_loaded || !self.frcmod_loaded {
            // todo: This lacks nuance. We wish to handle the case of Geostd Mol2 to load frcmod,
            // todo or the case of Drugbank/Pubchem SDF that need both, and have a pubchem id.
            // The reason for our current approach is that a pubchem ID is always valid, but the ident
            // may not be. (e.g. in the case of DrugBank). For Geostd, the Ident is valid.
            let mut ident = self.common.ident.clone();
            if let Some(cid) = &self.pubchem_cid {
                ident = cid.to_string();
            }

            self.replace_with_geostd(&ident, lig_specific);
        }
    }
}
