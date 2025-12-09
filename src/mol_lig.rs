//! Fundamental data structures for small organic molecules / ligands

use std::{
    collections::HashMap,
    io,
    path::{Path, PathBuf},
    sync::{mpsc, mpsc::Receiver},
    thread,
    time::Instant,
};

use bio_apis::{
    ReqError, amber_geostd, amber_geostd::GeostdData, pubchem::ProteinStructure,
};
use bio_files::{
    ChargeType, Mol2, MolType, Pdbqt, Sdf, Xyz, create_bonds,
    md_params::{ForceFieldParams, ForceFieldParamsVec},
};
use dynamics::{
    param_inference::{AmberDefSet, assign_missing_params, find_ff_types},
    partial_charge_inference::infer_charge,
};
use na_seq::Element;

use crate::{
    docking::{DockingSite, Pose},
    molecule::{
        Atom, Bond, Chain, MolGenericRef, MolGenericTrait, MolIdent, MolType as Mt, MoleculeCommon,
        Residue,
    },
};

const LIGAND_ABS_POSIT_OFFSET: f64 = 15.; // Å

/// A molecue representing a small organic molecule. Omits mol-generic fields.
#[derive(Debug, Default, Clone)]
pub struct MoleculeSmall {
    pub common: MoleculeCommon,
    pub lig_data: Option<Ligand>,
    pub idents: Vec<MolIdent>,
    /// FF type and partial charge on all atoms. Quick lookup flag.
    pub ff_params_loaded: bool,
    /// E.g., overrides for dihedral angles (part of the *bonded* dynamics calculation) for this
    /// specific molecule, as provided by Amber. Quick lookup flag.
    pub frcmod_loaded: bool,
    /// E.g. loaded proteins from Pubchem.
    pub associated_structures: Vec<ProteinStructure>,
    /// Simplified Molecular Input Line Entry System
    /// A cache for display as required. This is a text representation of a molecular formula.
    pub smiles: Option<String>,
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
        let mut idents = Vec::new();

        if let Some(id) = metadata.get("PUBCHEM_COMPOUND_CID") {
            if let Ok(cid) = id.parse::<u32>() {
                idents.push(MolIdent::PubChem(cid));
            };
        }

        if let Some(db_name) = metadata.get("DATABASE_NAME") {
            if db_name.to_lowercase() == "drugbank" {
                if let Some(id) = metadata.get("DATABASE_ID") {
                    idents.push(MolIdent::DrugBank(id.clone()));
                }
                // This seems to be valid for Drugbank-sourced molecules.
                if let Ok(id) = ident.parse::<u32>() {
                    idents.push(MolIdent::PubChem(id));
                }
            }
        }

        if ident.len() <= 4 {
            // This is a guess
            idents.push(MolIdent::PdbeAmber(ident.clone()));
        }

        Self {
            common: MoleculeCommon::new(ident, atoms, bonds, metadata, path),
            idents,
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

    fn to_ref(&self) -> MolGenericRef<'_> {
        MolGenericRef::Ligand(self)
    }

    fn mol_type(&self) -> crate::molecule::MolType {
        crate::molecule::MolType::Ligand
    }
}

/// This data is related specifically to docking.
#[derive(Debug, Clone, Default)]
// todo: It appears we use nothing in this struct!
pub struct Ligand {
    pub _anchor_atom: usize, // Index.
    /// Note: We may deprecate this in favor of our Amber MD-based approach to flexibility.
    pub _flexible_bonds: Vec<usize>, // Index
    pub _pose: Pose,
    pub _docking_site: DockingSite,
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

impl MoleculeSmall {
    pub fn from_xyz(m: Xyz, path: &Path) -> io::Result<Self> {
        let atoms: Vec<_> = m.atoms.iter().map(|a| a.into()).collect();

        let bonds_gen = create_bonds(&m.atoms);
        let bonds: Vec<Bond> = bonds_gen
            .iter()
            .map(|b| Bond::from_generic(b, &atoms))
            .collect::<Result<_, _>>()?;

        let filename = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();

        let mut metadata = HashMap::new();
        metadata.insert(String::from("Comment"), m.comment.clone());

        // Handle path and state-specific items after; not supported by TryFrom.
        Ok(Self::new(
            filename,
            atoms,
            bonds,
            metadata,
            Some(path.to_owned()),
        ))
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
    /// We use this to rotate flexible molecules around torsion (e.g. dihedral) angles.
    /// `pivot` and `side` are atom indices in the molecule.
    pub fn _find_downstream_atoms(&self, pivot: usize, side: usize) -> Vec<usize> {
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

        for ident in &self.idents {
            match ident {
                MolIdent::PubChem(cid) => {
                    metadata.insert("PUBCHEM_COMPOUND_CID".to_string(), cid.to_string());
                }
                MolIdent::DrugBank(id) => {
                    metadata.insert("DATABASE_ID".to_string(), id.clone());
                    metadata.insert("DATABASE_NAME".to_string(), "drugbank".to_string());
                }
                _ => (),
            }
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

    pub fn to_xyz(&self) -> Xyz {
        let atoms = self.common.atoms.iter().map(|a| a.to_generic()).collect();

        let comment = match self.common.metadata.get("Comment") {
            Some(v) => v.to_owned(),
            None => String::new(),
        };

        Xyz { atoms, comment }
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

        result.idents.push(MolIdent::PdbeAmber(name));

        result
    }

    pub fn apply_geostd_data(
        &mut self,
        data: GeostdData,
        lig_specific: &mut HashMap<String, ForceFieldParams>,
    ) {
        if !self.ff_params_loaded {
            let Ok(mol2) = Mol2::new(&data.mol2) else {
                eprintln!("Error: No Mol2 available from Geostd");
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
            println!("Loaded Amber Geostd FF data for {}", self.common.ident);
        }

        if !self.frcmod_loaded {
            if let Some(f) = data.frcmod
                && let Ok(frcmod) = ForceFieldParamsVec::from_frcmod(&f)
            {
                lig_specific.insert(self.common.ident.clone(), ForceFieldParams::new(&frcmod));
                self.frcmod_loaded = true;

                println!("Loaded Amber FRCMOD data for {}", self.common.ident);
            }
        }
    }

    /// Attempt to find FF type, partial charge, and FRCMOD overrides for a given molecule.
    /// Launch this in a thread.
    ///
    /// Unfortunately, we can't directly map atoms from our original molecule to
    /// the Geostd one. We could do this with coordinates, but that might be complicated.
    /// For now, we perform a sanity check about atom count by element. If it passes,
    /// we replace molecule atom and bond data with that loaded from the mol2.
    fn search_geostd(
        &mut self,
        ident: &str,
        geostd_thread: &mut Option<Receiver<(usize, Result<GeostdData, ReqError>)>>,
        mol_i: usize,
    ) {
        println!("Attempting to load Amber Geostd dynamics data for this molecule...");

        let (tx, rx) = mpsc::channel(); // one-shot channel
        let ident_for_thread = ident.to_string();

        thread::spawn(move || {
            let data = amber_geostd::load_mol_files(&ident_for_thread);
            let _ = tx.send((mol_i, data));
        });

        *geostd_thread = Some(rx);
    }

    pub fn update_aux(&mut self, active_mol: &Option<(Mt, usize)>) {
        if let Some((_, i)) = active_mol {
            let offset = LIGAND_ABS_POSIT_OFFSET * (*i as f64);
            for posit in &mut self.common.atom_posits {
                posit.x += offset; // Arbitrary axis and direction.
            }
        }
    }

    /// Update partial charges, FF types, and mol-specific params.
    /// Note: Perhaps we restructure? Not all of these need access to state.
    pub fn update_ff_related(
        &mut self,
        mol_specific_param_set: &mut HashMap<String, ForceFieldParams>,
        gaff2: &ForceFieldParams,
    ) {
        self.ff_params_loaded = true;
        for atom in &self.common.atoms {
            if atom.force_field_type.is_none() || atom.partial_charge.is_none() {
                self.ff_params_loaded = false;
                break;
            }
        }

        if mol_specific_param_set
            .keys()
            .any(|k| k.eq_ignore_ascii_case(&self.common.ident))
        {
            self.frcmod_loaded = true;
        }

        println!("Inferring parameter data...");
        // Note: There is an all-in-one `update_small_mol_params` fn we can use as well; it's
        // easier to use nominally, but this approach works better for our this-project Atom and bond types,
        // and loaded flags.

        let mut atoms_gen: Vec<_> = self.common.atoms.iter().map(|a| a.to_generic()).collect();
        let bonds_gen: Vec<_> = self.common.bonds.iter().map(|a| a.to_generic()).collect();

        // todo temp!
        // if true {
        if !self.ff_params_loaded {
            let defs = AmberDefSet::new().unwrap();
            let ff_types = find_ff_types(&atoms_gen, &bonds_gen, &defs);

            // println!("FF types created: {:?}", ff_types); // todo temp

            for (i, atom) in self.common.atoms.iter_mut().enumerate() {
                atom.force_field_type = Some(ff_types[i].clone());

                // We re-use `atoms_gen` for mol specific params below; update atoms gen here.
                atoms_gen[i].force_field_type = Some(ff_types[i].clone());
            }

            let charge = match infer_charge(&atoms_gen, &bonds_gen) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("Error inferring params: {e:?}");
                    return;
                }
            };

            for (i, atom) in self.common.atoms.iter_mut().enumerate() {
                atom.partial_charge = Some(charge[i]);
            }

            // todo: This print and loop are temp.
            println!("\n FF types computed:/n");
            for atom in &self.common.atoms {
                println!(
                    "--{}: {} {:.4}",
                    atom.serial_number,
                    atom.force_field_type.as_ref().unwrap(),
                    atom.partial_charge.unwrap()
                );
            }

            self.ff_params_loaded = true;
        }

        if !self.frcmod_loaded {
            let mol_specific_params =
                match assign_missing_params(&atoms_gen, &self.common.adjacency_list, gaff2) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!(
                            "Error inferring params for mol {}: {e:?}",
                            self.common.ident
                        );
                        return;
                    }
                };

            println!("\n\nDihe FRCMOD created:");
            for p in &mol_specific_params.dihedral {
                println!("\nDihe: {:?}", p);
            }

            println!("\n\nImproper FRCMOD created:");
            for p in &mol_specific_params.improper {
                println!("Improp: {:?}", p);
            }

            println!("Inserting: {:?}", self.common.ident.to_owned());
            mol_specific_param_set.insert(self.common.ident.to_owned(), mol_specific_params);
            self.frcmod_loaded = true;
        }
        println!("Inference complete.");
    }
}
