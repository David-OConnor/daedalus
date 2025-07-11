use std::{
    collections::HashMap,
    io,
    io::{BufReader, ErrorKind, Read, Seek},
    path::Path,
    str::FromStr,
};

use bio_files::{Chain, ResidueType};
use lin_alg::f64::Vec3;
use na_seq::{
    AtomTypeInRes,
    Element::{self, *},
};
use pdbtbx::{Format, PDB, ReadOptions, StrictnessLevel};
use rayon::prelude::*;

use crate::{
    docking::prep::DockType,
    file_io::cif_aux::load_data,
    molecule::{Atom, AtomRole, Molecule, Residue},
};

impl Atom {
    pub fn from_cif_pdb(
        atom_pdb: &pdbtbx::Atom,
        atom_i: usize,
        // aa_map: &HashMap<usize, ResidueType>,
        aa_map: &HashMap<usize, usize>, // atom_i: res_i
        residues: &[Residue],
    ) -> Self {
        // let mut residue_type = ResidueType::Other("".to_owned());
        let mut residue = None;
        let mut role = None;

        if let Some(res_i) = aa_map.get(&atom_i) {
            let res = &residues[*res_i];
            residue = Some(*res_i);

            // if let Some(res_type) = aa_map.get(&atom_i) {
            role = match res.res_type {
                ResidueType::AminoAcid(_aa) => Some(AtomRole::from_name(atom_pdb.name())),
                ResidueType::Water => Some(AtomRole::Water),
                _ => None,
            };

            // residue_type = res_type.clone();
        }

        let name = atom_pdb.name().to_owned();

        Self {
            serial_number: atom_pdb.serial_number() + 1,
            posit: Vec3::new(atom_pdb.x(), atom_pdb.y(), atom_pdb.z()),
            element: el_from_pdb(atom_pdb.element()),
            type_in_res: AtomTypeInRes::from_str(&name).ok(),
            force_field_type: None,
            role,
            residue,
            // residue_type,
            hetero: atom_pdb.hetero(),
            occupancy: None,
            temperature_factor: None,
            partial_charge: None,
            dock_type: Some(DockType::from_str(atom_pdb.name())), // Updated later with Donor/Acceptor
        }
    }
}

impl Molecule {
    /// From `pdbtbx`'s format. Uses raw data too to add secondary structure, which pdbtbx doesn't handle.
    pub fn from_cif_pdb<R: Read + Seek>(pdb: &PDB, raw: R) -> io::Result<Self> {
        // todo: Maybe return the PDB type here, and store that. Also have a way to
        // todo get molecules from it

        println!("Loading mol"); // todo tmep

        // todo: Pdbtbx doesn't implm this yet for CIF.
        for remark in pdb.remarks() {}

        let atoms_pdb: Vec<&pdbtbx::Atom> = pdb.par_atoms().collect();

        // println!("Gather residues...");
        // let res_pdb: Vec<&pdbtbx::Residue> = pdb.par_residues().collect();

        let mut residues: Vec<Residue> = pdb
            .par_residues()
            .map(|res| Residue::from_pdb(res, &atoms_pdb))
            .collect();

        residues.sort_by_key(|r| r.serial_number);

        let mut chains = Vec::with_capacity(pdb.chain_count());
        for chain_pdb in pdb.chains() {
            // println!("Chain: {chain_pdb:?}");

            let mut chain = Chain {
                id: chain_pdb.id().to_owned(),
                atoms: Vec::with_capacity(chain_pdb.atom_count()),
                residues: Vec::with_capacity(chain_pdb.residue_count()),
                visible: true,
            };

            for atom_c in chain_pdb.atoms() {
                let atom_pdb = atoms_pdb
                    .iter()
                    .enumerate()
                    .find(|(_i, a)| a.serial_number() == atom_c.serial_number());
                if let Some((i, _atom)) = atom_pdb {
                    chain.atoms.push(i);
                }
            }

            // We don't have a way to, using serial numbers alone, using PDBTBX, find which residues are associated with
            // which chain. This method is a bit more indirect, using both serial number, and atom indexes.
            for res_c in chain_pdb.residues() {
                for (i, res) in residues.iter().enumerate() {
                    if res.serial_number == res_c.serial_number() {
                        let atom_sns_chain: Vec<usize> =
                            res_c.atoms().map(|a| a.serial_number()).collect();
                        // let atom_sns_res: Vec<usize> = res.atoms.iter().map(|a| a.serial_number).collect();
                        let mut atom_sns_res = Vec::with_capacity(res.atoms.len());
                        for atom_i in &res.atoms {
                            atom_sns_res.push(atoms_pdb[*atom_i].serial_number());
                        }

                        if atom_sns_chain == atom_sns_res {
                            chain.residues.push(i);
                        }
                    }
                }

                // let res = residues
                //     .iter()
                //     .enumerate()
                //     .find(|(i, r)| r.serial_number == res_c.serial_number() && r.atoms );
                // if let Some((i, _res)) = res {
                //     chain.residues.push(i);
                // }
            }

            chains.push(chain);
        }

        // This pre-computation of the AA map is more efficient. { atom_i: res_i}
        let mut aa_map = HashMap::new();
        for (res_i, res) in residues.iter().enumerate() {
            for atom_i in &res.atoms {
                // aa_map.insert(*atom_i, res.res_type.clone());
                aa_map.insert(*atom_i, res_i);
            }
        }

        // todo: This is taking a while.
        let atoms: Vec<Atom> = atoms_pdb
            .into_iter()
            .enumerate()
            .map(|(i, atom)| Atom::from_cif_pdb(atom, i, &aa_map, &residues))
            .collect();

        // todo: We use our own bond inference, since most PDBs seem to lack bond information.

        // todo: Check modern ones?
        // let mut bonds = Vec::new();
        // for (a0, a1, bond) in pdb.bonds() {
        //     bonds.push((Atom::from_pdb(a0), Atom::from_pdb(a1), bond));
        // }

        let mut result = Molecule::new(
            pdb.identifier.clone().unwrap_or_default(),
            atoms,
            chains,
            residues,
            None,
            None,
        );

        (result.secondary_structure, result.method) = load_data(raw)?;

        Ok(result)
    }
}

impl Residue {
    pub fn from_pdb(res_pdb: &pdbtbx::Residue, atoms_pdb: &[&pdbtbx::Atom]) -> Self {
        let res_name = res_pdb.name().unwrap_or_default();

        let res_type = ResidueType::from_str(res_name);

        let mut res = Residue {
            serial_number: res_pdb.serial_number(),
            res_type,
            atoms: Vec::new(),
            dihedral: None,
        };

        for atom_c in res_pdb.atoms() {
            let atom_pdb = atoms_pdb
                .iter()
                .enumerate()
                .find(|(_i, a)| a.serial_number() == atom_c.serial_number());
            if let Some((i, _atom)) = atom_pdb {
                res.atoms.push(i);
            }
        }

        res
    }
}

/// From a string of a CIF or PDB text file.
pub fn read_pdb(pdb_text: &str) -> io::Result<PDB> {
    let reader = BufReader::new(pdb_text.as_bytes());

    let (pdb, _errors) = ReadOptions::default()
        .set_level(StrictnessLevel::Loose)
        .set_format(Format::Mmcif) // Must be set explicitly if  using read_raw.
        .read_raw(reader)
        .map_err(|e| {
            io::Error::new(
                ErrorKind::InvalidData,
                format!("Problem parsing PDB or CIF text: {e:?}"),
            )
        })?;

    Ok(pdb)
}

/// From file
pub fn load_cif_pdb(path: &Path) -> io::Result<PDB> {
    let (pdb, _errors) = ReadOptions::default()
        // At the default strictness level of Medium, we fail to parse a number of files. Medium and Strict
        // ensure closer conformance to the PDB and CIF specs, but many files in the wild do not. Setting
        // loose is required for practical use cases.
        .set_level(StrictnessLevel::Loose)
        .read(path.to_str().unwrap())
        .map_err(|e| {
            io::Error::new(
                ErrorKind::InvalidData,
                format!("Problem opening a PDB or CIF file: {e:?}"),
            )
        })?;

    Ok(pdb)
}

/// Save as PDB or CIF format.
pub fn save_pdb(pdb: &mut PDB, path: &Path) -> io::Result<()> {
    // todo: Update the PDB in state with data from the molecule prior to saving.

    pdbtbx::save(
        pdb,
        path.to_str().unwrap_or_default(),
        StrictnessLevel::Loose,
    )
    .map_err(|e| {
        io::Error::new(
            ErrorKind::InvalidData,
            format!("Problem saving a PDB or CIF file: {e:?}"),
        )
    })

    // todo: Save SS.
}

pub fn el_from_pdb(el: Option<&pdbtbx::Element>) -> Element {
    if let Some(e) = el {
        match e {
            pdbtbx::Element::H => Hydrogen,
            pdbtbx::Element::C => Carbon,
            pdbtbx::Element::O => Oxygen,
            pdbtbx::Element::N => Nitrogen,
            pdbtbx::Element::F => Fluorine,
            pdbtbx::Element::S => Sulfur,
            pdbtbx::Element::P => Phosphorus,
            pdbtbx::Element::Fe => Iron,
            pdbtbx::Element::Cu => Copper,
            pdbtbx::Element::Ca => Calcium,
            pdbtbx::Element::K => Potassium,
            pdbtbx::Element::Al => Aluminum,
            pdbtbx::Element::Pb => Lead,
            pdbtbx::Element::Au => Gold,
            pdbtbx::Element::Ag => Silver,
            pdbtbx::Element::Hg => Mercury,
            pdbtbx::Element::Sn => Tin,
            pdbtbx::Element::Zn => Zinc,
            pdbtbx::Element::Mg => Magnesium,
            pdbtbx::Element::Mn => Manganese,
            pdbtbx::Element::I => Iodine,
            pdbtbx::Element::Cl => Chlorine,
            pdbtbx::Element::W => Tungsten,
            pdbtbx::Element::Te => Tellurium,
            pdbtbx::Element::Se => Selenium,
            pdbtbx::Element::Br => Bromine,
            pdbtbx::Element::Ru => Rubidium,

            _ => {
                eprintln!("Unknown element: {e:?}");
                Element::Other
            }
        }
    } else {
        // todo?
        Element::Other
    }
}
