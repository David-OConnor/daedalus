use std::{
    fs,
    fs::File,
    io,
    io::{ErrorKind, Read},
    path::Path,
    time::Instant,
};

use bio_files::{DensityMap, gemmi_cif_to_map};
use lin_alg::f64::Vec3;
use na_seq::{AaIdent, AminoAcid, Element};

use crate::{
    AMINO_19, FRCMOD_FF19SB, GAFF2, PARM_19, State,
    file_io::{cif_pdb::load_cif_pdb, pdbqt::load_pdbqt},
    molecule::{Ligand, Molecule},
};

pub mod cif_aux;
pub mod cif_pdb;
pub mod cif_sf;
pub mod mtz;
pub mod pdbqt;

use bio_files::{
    Mol2,
    amber_params::{ForceFieldParams, ForceFieldParamsKeyed, parse_amino_charges},
    sdf::Sdf,
};

use crate::{
    docking::prep::DockingSetup,
    dynamics::prep::{merge_params, populate_ff_and_q},
    reflection::{DENSITY_CELL_MARGIN, DENSITY_MAX_DIST, DensityRect, ElectronDensity},
    util::handle_err,
};

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
            "map" => self.open_map(path)?,
            // todo: lib, .dat etc as required. Using Amber force fields and its format
            // todo to start. We assume it'll be generalizable later.
            "frcmod" | "dat" => self.open_force_field(path)?,
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
            "sdf" => Ok(Sdf::load(path)?.into()),
            "mol2" => Ok(Mol2::load(path)?.into()),
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
                // If a 2fo-fc CIF, use gemmi to convert it to Map data.
                // Using the filename to determine if this is a 2fo-fc file, vice atom coordinates,
                // is rough here, but good enough for now.
                // todo: This isn't really opening a molecule, so is out of place. Good enough for now.
                if let Some(name) = path.file_name().and_then(|os| os.to_str()) {
                    if name.contains("2fo") && name.contains("fc") {
                        let dm = gemmi_cif_to_map(path.to_str().unwrap())?;
                        self.load_density(dm);
                    }
                }

                let pdb = load_cif_pdb(path)?;
                let mut file = File::open(path)?;

                let mut mol = Molecule::from_cif_pdb(&pdb, &file)?;
                self.pdb = Some(pdb);

                let mut data_str = String::new();
                file.read_to_string(&mut data_str)?;
                self.cif_pdb_raw = Some(data_str);

                // If we've loaded general FF params, apply them to get FF type and charge.
                if let Some(charge_ff_data) = &self.ff_params.prot_charge_general {
                    if let Err(e) =
                        populate_ff_and_q(&mut mol.atoms, &mol.residues, &charge_ff_data)
                    {
                        eprintln!(
                            "Unable to populate FF charge and FF type for protein atoms: {:?}",
                            e
                        );
                    } else {
                        // Run this to update the ff name and charge data on the set of receptor
                        // atoms near the docking site.
                        if let Some(lig) = &mut self.ligand {
                            self.volatile.docking_setup = Some(DockingSetup::new(
                                &mol,
                                lig,
                                &self.volatile.lj_lookup_table,
                                &self.bh_config,
                            ));
                        }
                    }
                }

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

                    self.update_docking_site(init_posit);
                } else {
                    self.to_save.last_opened = Some(path.to_owned());

                    self.volatile.aa_seq_text = String::with_capacity(mol.atoms.len());
                    for aa in &mol.aa_seq {
                        self.volatile
                            .aa_seq_text
                            .push_str(&aa.to_str(AaIdent::OneLetter));
                    }

                    self.volatile.flags.ss_mesh_created = false;
                    self.volatile.flags.sas_mesh_created = false;

                    self.volatile.flags.clear_density_drawing = true;
                    self.molecule = Some(mol);

                    // Only updating if not loading a ligand.
                    // Update from prefs based on the molecule-specific items.
                    self.update_from_prefs();
                }

                if let Some(mol) = &mut self.molecule {
                    // Only after updating from prefs (to prevent unecesasary loading) do we update data avail.
                    mol.updates_rcsb_data(&mut self.volatile.mol_pending_data_avail);
                }

                // Now, save prefs: This is to save last opened. Note that anomolies happen
                // if we update the molecule here, e.g. with docking site posit.
                self.update_save_prefs_no_mol();

                if self.get_make_docking_setup().is_none() {
                    eprintln!("Problem making or getting docking setup.");
                }

                self.volatile.flags.new_mol_loaded = true;
            }
            Err(e) => {
                return Err(e);
            }
        }

        Ok(())
    }

    pub fn load_density(&mut self, dm: DensityMap) {
        if let Some(mol) = &mut self.molecule {
            // We are filtering for backbone atoms of one type for now, for performance reasons. This is
            // a sample. Good enough?
            let atom_posits: Vec<_> = mol
                .atoms
                .iter()
                // .filter(|a| a.is_backbone() && a.element == Element::Nitrogen)
                .filter(|a| a.element != Element::Hydrogen)
                // .filter(|a| a.is_backbone())
                .map(|a| a.posit)
                .collect();

            let dens_rect = DensityRect::new(&atom_posits, &dm, DENSITY_CELL_MARGIN);
            let dens = dens_rect.make_densities(&atom_posits, &dm.cell, DENSITY_MAX_DIST);

            let elec_dens: Vec<_> = dens
                .iter()
                .map(|d| ElectronDensity {
                    coords: d.coords,
                    density: d.density,
                })
                .collect();

            mol.density_map = Some(dm);
            mol.density_rect = Some(dens_rect);
            mol.elec_density = Some(elec_dens);

            self.volatile.flags.new_density_loaded = true;
            self.volatile.flags.make_density_mesh = true;
        }
    }

    /// An electron density map file, e.g. a .map file.
    /// todo: Support opening MTZ files.
    pub fn open_map(&mut self, path: &Path) -> io::Result<()> {
        let dm = DensityMap::load(path)?;
        self.load_density(dm);

        self.to_save.last_map_opened = Some(path.to_owned());
        self.update_save_prefs();

        Ok(())
    }

    /// Open Amber force field parameters, e.g. dat and frcmod.
    pub fn open_force_field(&mut self, path: &Path) -> io::Result<()> {
        let binding = path.extension().unwrap_or_default().to_ascii_lowercase();
        let extension = binding;

        match extension.to_str().unwrap() {
            "dat" => {
                self.ff_params.lig_general = Some(ForceFieldParamsKeyed::new(
                    &ForceFieldParams::load_dat(path)?,
                ));

                println!("\nLoaded forcefields:");
                let v = &self.ff_params.lig_general.as_ref().unwrap();
                println!("Lin");
                for di in v.bond.values().take(20) {
                    println!("Lin: {:?}, {}, {}", di.atom_types, di.k_b, di.r_0);
                }

                println!("Angle");
                for di in v.angle.values().take(20) {
                    println!("Angle: {:?}, {}, {}", di.atom_types, di.k, di.theta_0);
                }

                println!("Dihe:");
                for di in v.dihedral.values().take(20) {
                    println!(
                        "DH: {:?}, {}, {}",
                        di.atom_types, di.barrier_height, di.phase
                    );
                }

                println!("Dihedral, improper:");
                for di in v.dihedral_improper.values().take(20) {
                    println!(
                        "Imp: {:?}, {}, {}",
                        di.atom_types, di.barrier_height, di.phase
                    );
                }

                // todo: Get VDW loading working.
                println!("Vdw");
                for di in v.van_der_waals.values().take(20) {
                    println!("Vdw: {:?}, {}, {}", di.atom_type, di.sigma, di.eps);
                }

                println!("Loaded general Ligand force fields.");
            }
            "frcmod" => {
                let mol_name = "CPB".to_owned(); // todo temp.

                self.ff_params.lig_specific.insert(
                    mol_name,
                    ForceFieldParamsKeyed::new(&ForceFieldParams::load_frcmod(path)?),
                );
                println!("Loaded molecule-specific force fields.");
            }
            _ => {
                return Err(io::Error::new(
                    ErrorKind::InvalidFilename,
                    "Attempting to parse non-dat or frcmod mod file as a force field.",
                ));
            }
        };

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
                    lig.molecule.to_sdf().save(path)?;

                    self.to_save.last_ligand_opened = Some(path.to_owned());
                    self.update_save_prefs()
                }
                None => return Err(io::Error::new(ErrorKind::InvalidData, "No ligand to save")),
            },
            "mol2" => match &self.ligand {
                Some(lig) => {
                    lig.molecule.to_mol2().save(path)?;

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
            "map" => {
                // todo
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

    /// Load amimo acid partial charges and forcefields from our built-in string. This is fast and
    /// light; do it at init. If we have a molecule loaded, populate its force field and Q data
    /// using it.
    pub fn load_aa_charges_ff(&mut self) {
        match parse_amino_charges(AMINO_19) {
            Ok(charge_ff_data) => {
                if let Some(mol) = &mut self.molecule {
                    if let Err(e) =
                        populate_ff_and_q(&mut mol.atoms, &mol.residues, &charge_ff_data)
                    {
                        eprintln!(
                            "Unable to populate FF charge and FF type for protein atoms: {:?}",
                            e
                        );
                    } else {
                        // Update ff and charges in the receptor atoms.
                        if let Some(lig) = &mut self.ligand {
                            self.volatile.docking_setup = Some(DockingSetup::new(
                                &mol,
                                lig,
                                &self.volatile.lj_lookup_table,
                                &self.bh_config,
                            ));
                        }
                    }
                }

                self.ff_params.prot_charge_general = Some(charge_ff_data);
            }
            Err(e) => handle_err(
                &mut self.ui,
                format!("Unable to load protein charges (static): {e}"),
            ),
        }
    }

    /// Load parameter files for general organic molecules (GAFF2), and proteins/amino acids (PARM19).
    /// This also populates ff type and charge on our protein atoms.
    pub fn load_ffs_general(&mut self) {
        if self.ff_params.prot_general.is_none() {
            // Load general parameters for proteins and AAs.
            match ForceFieldParams::from_dat(PARM_19) {
                Ok(ff) => {
                    self.ff_params.prot_general = Some(ForceFieldParamsKeyed::new(&ff));
                }
                Err(e) => handle_err(
                    &mut self.ui,
                    format!("Unable to load protein FF params (static): {e}"),
                ),
            }

            // Load (updated/patched) general parameters for proteins and AAs.
            match ForceFieldParams::from_frcmod(FRCMOD_FF19SB) {
                Ok(ff) => {
                    let ff_keyed = ForceFieldParamsKeyed::new(&ff);

                    // We just loaded this above.
                    if let Some(ffs) = &mut self.ff_params.prot_general {
                        let params_updated = merge_params(ffs, Some(&ff_keyed));
                        self.ff_params.prot_general = Some(params_updated);
                    }
                }
                Err(e) => handle_err(
                    &mut self.ui,
                    format!("Unable to load protein FF params (static): {e}"),
                ),
            }
        }

        // Note: We may load this at program init
        if self.ff_params.prot_charge_general.is_none() {
            self.load_aa_charges_ff();
            // todo: Handle C and N-terminal files (aminoct12.lib and aminont12.lib)
        }

        // Load general organic molecule, e.g. ligand, parameters.
        if self.ff_params.lig_general.is_none() {
            match ForceFieldParams::from_dat(GAFF2) {
                Ok(ff) => {
                    self.ff_params.lig_general = Some(ForceFieldParamsKeyed::new(&ff));
                }
                Err(e) => handle_err(
                    &mut self.ui,
                    format!("Unable to load ligand FF params (static): {e}"),
                ),
            }
        }
    }
}
