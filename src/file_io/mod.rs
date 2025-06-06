use std::{
    fs,
    fs::File,
    io,
    io::{ErrorKind, Read},
    path::Path,
};

use lin_alg::f64::Vec3;
use na_seq::AaIdent;

use crate::{
    State,
    file_io::{
        cif_pdb::load_cif_pdb, map::DensityMap, mol2::load_mol2, pdbqt::load_pdbqt, sdf::load_sdf,
    },
    molecule::{Ligand, Molecule},
};

pub mod cif_pdb;
pub mod cif_secondary_structure;
pub mod cif_sf;
pub mod map;
pub mod mol2;
pub mod mtz;
pub mod pdbqt;
pub mod sdf;

impl State {
    /// A single endpoint to open a number of file types
    pub fn open(&mut self, path: &Path) -> io::Result<()> {
        match path
            .extension()
            .unwrap_or_default()
            .to_ascii_lowercase()
            .to_str()
            .unwrap_or_default()
        {
            "sdf" | "mol2" | "pdbqt" | "pdb" | "cif" => self.open_molecule(path)?,
            "map" => {
                let (hdr, mut dens) = map::read_map_data(path)?;

                // println!("Map header: {:#?}", hdr);

                // for pt in &dens[0..100] {
                //     println!("{:.2?}", pt);
                // }
                if let Some(mol) = &mut self.molecule {
                    // handle_map_symmetry(&mut dens, &mol.atoms);

                    mol.elec_density_header = Some(hdr);
                    mol.elec_density = Some(dens);

                    let dm = DensityMap::new(path)?;
                    mol.density_map = Some(dm);

                    self.volatile.make_density_mesh = true;
                }
            }
            _ => {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    "Unsupported file extension",
                ));
            }
        }

        Ok(())
    }

    pub fn open_molecule(&mut self, path: &Path) -> io::Result<()> {
        let binding = path.extension().unwrap_or_default().to_ascii_lowercase();
        let extension = binding;

        let is_ligand = matches!(extension.to_str().unwrap(), "sdf" | "mol2");

        let mut ligand = None;
        let molecule = match extension.to_str().unwrap() {
            "sdf" => load_sdf(path),
            "mol2" => load_mol2(path),
            "pdbqt" => {
                load_pdbqt(path).map(|(molecule, mut lig_loaded)| {
                    if lig_loaded.is_some() {
                        lig_loaded.as_mut().unwrap().molecule = molecule.clone(); // sloppy
                    }
                    ligand = lig_loaded;
                    molecule
                })
            }
            "pdb" | "cif" => {
                let pdb = load_cif_pdb(path)?;
                let mut file = File::open(path)?;

                let mol = Molecule::from_cif_pdb(&pdb, &file)?;
                self.pdb = Some(pdb);

                let mut data_str = String::new();
                file.read_to_string(&mut data_str)?;
                self.cif_pdb_raw = Some(data_str);

                Ok(mol)
            }
            _ => Err(io::Error::new(
                ErrorKind::InvalidData,
                "Invalid file extension",
            )),
        };

        match molecule {
            Ok(mol) => {
                if is_ligand {
                    let het_residues = mol.het_residues.clone();
                    let mol_atoms = mol.atoms.clone();

                    let mut init_posit = Vec3::new_zero();

                    let lig = Ligand::new(mol);

                    // Align to a hetero residue in the open molecule, if there is a match.
                    // todo: Keep this in sync with the UI button-based code; this will have updated.
                    for res in het_residues {
                        if (res.atoms.len() as i16 - lig.molecule.atoms.len() as i16).abs() < 22 {
                            init_posit = mol_atoms[res.atoms[0]].posit;
                        }
                    }

                    self.ligand = Some(lig);
                    self.to_save.last_ligand_opened = Some(path.to_owned());

                    println!("Loaded lig: {:?}", &self.ligand);

                    self.update_docking_site(init_posit);
                } else {
                    self.to_save.last_opened = Some(path.to_owned());

                    self.volatile.aa_seq_text = String::with_capacity(mol.atoms.len());
                    for aa in &mol.aa_seq {
                        self.volatile
                            .aa_seq_text
                            .push_str(&aa.to_str(AaIdent::OneLetter));
                    }
                    self.molecule = Some(mol);
                }

                // Update from prefs based on the molecule-specific items.
                self.update_from_prefs();

                if let Some(mol) = &mut self.molecule {
                    // Only after updating from prefs (to prevent unecesasary loading) do we update data avail.
                    mol.updates_rcsb_data(&mut self.volatile.mol_pending_data_avail);
                }

                // Now, save prefs: This is to save last opened.
                self.update_save_prefs();

                if self.get_make_docking_setup().is_none() {
                    eprintln!("Problem making or getting docking setup.");
                }

                self.ui.new_mol_loaded = true;
            }
            Err(e) => {
                return Err(e);
            }
        }

        Ok(())
    }

    /// A single endpoint to save a number of file types
    pub fn save(&mut self, path: &Path) -> io::Result<()> {
        let binding = path.extension().unwrap_or_default().to_ascii_lowercase();
        let extension = binding;

        match extension.to_str().unwrap_or_default() {
            "pdb" | "cif" => {
                // todo: Eval how you want to handle this. For now, the raw CIF or PDB.
                // if let Some(pdb) = &mut self.pdb {
                //     save_pdb(pdb, path)?;
                //     self.to_save.last_opened = Some(path.to_owned());
                //     self.update_save_prefs()
                // }
                if let Some(data) = &mut self.cif_pdb_raw {
                    fs::write(path, data)?;
                    self.to_save.last_opened = Some(path.to_owned());
                    self.update_save_prefs()
                }
            }
            "sdf" => match &self.ligand {
                Some(lig) => {
                    lig.molecule.save_sdf(path)?;
                    self.to_save.last_ligand_opened = Some(path.to_owned());
                    self.update_save_prefs()
                }
                None => return Err(io::Error::new(ErrorKind::InvalidData, "No ligand to save")),
            },
            "mol2" => match &self.ligand {
                Some(lig) => {
                    lig.molecule.save_mol2(path)?;
                    self.to_save.last_ligand_opened = Some(path.to_owned());
                    self.update_save_prefs()
                }
                None => return Err(io::Error::new(ErrorKind::InvalidData, "No ligand to save")),
            },
            "pdbqt" => match &self.ligand {
                Some(lig) => {
                    lig.molecule.save_pdbqt(path, None)?;
                    self.to_save.last_ligand_opened = Some(path.to_owned());
                    self.update_save_prefs()
                }
                None => return Err(io::Error::new(ErrorKind::InvalidData, "No ligand to save")),
            },
            "map" => {}
            _ => {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    "Unsupported file extension",
                ));
            }
        }

        Ok(())
    }
}
