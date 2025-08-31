use std::{
    fs,
    fs::File,
    io,
    io::{ErrorKind, Read},
    path::Path,
    str::FromStr,
    time::Instant,
};

use bio_apis::amber_geostd;
use bio_files::{DensityMap, MmCif, gemmi_sf_to_map};
use na_seq::{AaIdent, Element};

use crate::{
    AMINO_19, AMINO_CT12, AMINO_NT12, FRCMOD_FF19SB, GAFF2, PARM_19, ProtFFTypeChargeMap, State,
    molecule::{Ligand, MoleculePeptide},
};

pub mod pdbqt;

use bio_files::{
    Mol2,
    amber_params::{ForceFieldParams, ForceFieldParamsKeyed, parse_amino_charges},
    sdf::Sdf,
};

use crate::{
    docking::prep::DockingSetup,
    dynamics::prep::{merge_params, populate_ff_and_q},
    file_io::pdbqt::save_pdbqt,
    molecule::{MoleculeGeneric, MoleculeSmall},
    reflection::{DENSITY_CELL_MARGIN, DENSITY_MAX_DIST, DensityRect, ElectronDensity},
    util::{handle_err, handle_success},
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
            // The cif branch here handles 2fo-fc mmCIF files.
            "sdf" | "mol2" | "pdbqt" | "pdb" | "cif" => self.open_molecule(path)?,
            "map" => self.open_map(path)?,
            "mtz" => self.open_mtz(path)?,
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

        handle_success(
            &mut self.ui,
            format!(
                "Loaded file {}",
                Path::new(&path)
                    .file_name()
                    .and_then(|f| f.to_str())
                    .unwrap_or("<unknown>")
            ),
        );

        Ok(())
    }

    // todo: See notes eleswhere on `on_init`.
    pub fn open_molecule(&mut self, path: &Path) -> io::Result<()> {
        let binding = path.extension().unwrap_or_default().to_ascii_lowercase();
        let extension = binding;

        let molecule = match extension.to_str().unwrap() {
            "sdf" => Ok(MoleculeGeneric::Ligand(Sdf::load(path)?.try_into()?)),
            "mol2" | "pdbqt" => Ok(MoleculeGeneric::Ligand(Mol2::load(path)?.try_into()?)),
            // "pdbqt" => Ok(MoleculeGeneric::Ligand(load_pdbqt(path)?.try_into()?)),
            "pdb" | "cif" => {
                // If a 2fo-fc CIF, use gemmi to convert it to Map data.
                // Using the filename to determine if this is a 2fo-fc file, vice atom coordinates,
                // is rough here, but good enough for now.
                // todo: This isn't really opening a molecule, so is out of place. Good enough for now.
                if let Some(name) = path.file_name().and_then(|os| os.to_str()) {
                    // Note: This isn' tthe ideal place to handle 2fo-fc files, but they're in mmCIF format,
                    // so we handle here. We handle map and MTZ files elsewhere, even though they use a
                    // similar pipeline.
                    if name.contains("2fo") && name.contains("fc") {
                        gemmi_sf_to_map(path, gemmi_path())?;
                        let dm = gemmi_sf_to_map(path, gemmi_path())?;
                        self.load_density(dm);
                    }
                }

                let mut file = File::open(path)?;

                let mut data_str = String::new();
                file.read_to_string(&mut data_str)?;

                let cif_data = MmCif::new(&data_str)?;
                // let mut mol: Molecule = cif_data.try_into()?;

                let Some(ff_map) = &self.ff_params.prot_ff_q_map else {
                    return Err(io::Error::new(
                        ErrorKind::Other,
                        "Missing FF map when opening a protein; can't validate H",
                    ));
                };
                let mut mol = MoleculePeptide::from_mmcif(cif_data, &ff_map.internal)?;

                self.cif_pdb_raw = Some(data_str);

                // If we've loaded general FF params, apply them to get FF type and charge.
                if let Some(charge_ff_data) = &self.ff_params.prot_ff_q_map {
                    if let Err(e) =
                        populate_ff_and_q(&mut mol.common.atoms, &mol.residues, &charge_ff_data)
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

                Ok(MoleculeGeneric::Peptide(mol))
            }
            _ => Err(io::Error::new(
                ErrorKind::InvalidData,
                "Invalid file extension",
            )),
        };

        match molecule {
            Ok(mol_gen) => {
                match mol_gen {
                    MoleculeGeneric::Ligand(mol) => {
                        let lig = Ligand::new(mol, &self.ff_params.lig_specific);
                        self.mol_dynamics = None;

                        self.ligand = Some(lig);
                        self.to_save.last_ligand_opened = Some(path.to_owned());

                        // self.update_docking_site(init_posit);
                    }
                    MoleculeGeneric::Peptide(m) => {
                        self.to_save.opened_items = Some(path.to_owned());

                        self.volatile.aa_seq_text = String::with_capacity(m.common.atoms.len());
                        for aa in &m.aa_seq {
                            self.volatile
                                .aa_seq_text
                                .push_str(&aa.to_str(AaIdent::OneLetter));
                        }

                        self.volatile.flags.ss_mesh_created = false;
                        self.volatile.flags.sas_mesh_created = false;

                        self.volatile.flags.clear_density_drawing = true;
                        self.molecule = Some(m);

                        // Only updating if not loading a ligand.
                        // Update from prefs based on the molecule-specific items.
                        self.update_from_prefs();
                    }
                    MoleculeGeneric::NucleicAcid(m) => (), // todo
                }

                if let Some(mol) = &mut self.molecule {
                    // Only after updating from prefs (to prevent unnecessary loading) do we update data avail.
                    mol.updates_rcsb_data(&mut self.volatile.mol_pending_data_avail);
                }

                // Now, save prefs: This is to save last opened. Note that anomolies happen
                // if we update the molecule here, e.g. with docking site posit.
                self.update_save_prefs_no_mol();

                if self.ligand.is_some() {
                    if self.get_make_docking_setup().is_none() {
                        eprintln!("Problem making or getting docking setup.");
                    }
                }

                self.volatile.flags.new_mol_loaded = true;
            }
            Err(e) => {
                return Err(e);
            }
        }

        Ok(())
    }

    pub fn load_density(&mut self, dens_map: DensityMap) {
        if let Some(mol) = &mut self.molecule {
            // We are filtering for backbone atoms of one type for now, for performance reasons. This is
            // a sample. Good enough?
            let atom_posits: Vec<_> = mol
                .common
                .atoms
                .iter()
                .filter(|a| a.element != Element::Hydrogen)
                .map(|a| a.posit)
                .collect();

            let dens_rect = DensityRect::new(&atom_posits, &dens_map, DENSITY_CELL_MARGIN);
            let dens =
                dens_rect.make_densities(&self.dev, &atom_posits, &dens_map.cell, DENSITY_MAX_DIST);

            let elec_dens: Vec<_> = dens
                .iter()
                .map(|d| ElectronDensity {
                    coords: d.coords,
                    density: d.density,
                })
                .collect();

            mol.density_map = Some(dens_map);
            mol.density_rect = Some(dens_rect);
            mol.elec_density = Some(elec_dens);

            self.volatile.flags.new_density_loaded = true;
            self.volatile.flags.make_density_iso_mesh = true;
        }
    }

    /// An electron density map file, e.g. a .map file.
    pub fn open_map(&mut self, path: &Path) -> io::Result<()> {
        let dm = DensityMap::load(path)?;
        self.load_density(dm);

        self.to_save.last_map_opened = Some(path.to_owned());
        self.update_save_prefs(false);

        Ok(())
    }

    /// An electron density MTZ file. We use Gemmi's sf2map functionality, as we do for 2fo-fc files.
    pub fn open_mtz(&mut self, path: &Path) -> io::Result<()> {
        let dm = gemmi_sf_to_map(path, gemmi_path())?;
        self.load_density(dm);

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
                // Good enough for now; works for amber params.
                // todo: Not general though. Could alternatively assume whatever you load is
                // todo for the current molecule.
                // Filename without path or extension.
                let mol_name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .ok_or_else(|| {
                        io::Error::new(
                            ErrorKind::InvalidInput,
                            format!("Invalid frcmod filename: {:?}", path),
                        )
                    })?
                    .to_string();

                self.ff_params.lig_specific.insert(
                    mol_name.to_uppercase(),
                    ForceFieldParamsKeyed::new(&ForceFieldParams::load_frcmod(path)?),
                );

                // Update the lig's FRCMOD status A/R, if the ligand is opened already.
                if let Some(lig) = &mut self.ligand {
                    if &lig.common.ident.to_uppercase() == &mol_name.to_uppercase() {
                        lig.frcmod_loaded = true;
                    }
                }

                self.to_save.last_frcmod_opened = Some(path.to_owned());
                self.update_save_prefs(false);

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
                    self.to_save.opened_items = Some(path.to_owned());
                    self.update_save_prefs(false)
                }
            }
            "sdf" => match &self.ligand {
                Some(lig) => {
                    lig.mol.to_sdf().save(path)?;

                    self.to_save.last_ligand_opened = Some(path.to_owned());
                    self.update_save_prefs(false)
                }
                None => return Err(io::Error::new(ErrorKind::InvalidData, "No ligand to save")),
            },
            "mol2" => match &self.ligand {
                Some(lig) => {
                    lig.mol.to_mol2().save(path)?;

                    self.to_save.last_ligand_opened = Some(path.to_owned());
                    self.update_save_prefs(false)
                }
                None => return Err(io::Error::new(ErrorKind::InvalidData, "No ligand to save")),
            },
            "pdbqt" => match &self.ligand {
                Some(lig) => {
                    save_pdbqt(&lig.mol.to_mol2(), path, None)?;
                    self.to_save.last_ligand_opened = Some(path.to_owned());
                    self.update_save_prefs(false)
                }
                None => return Err(io::Error::new(ErrorKind::InvalidData, "No ligand to save")),
            },
            // todo: Consider if you want to store the original map bytes, as you do with
            // todo mmCIF files, instead of saving what you parsed.
            "map" => match &self.molecule {
                Some(mol) => match &mol.density_map {
                    Some(dm) => {
                        dm.save(path)?;
                        self.to_save.last_map_opened = Some(path.to_owned());
                        self.update_save_prefs(false)
                    }
                    None => {
                        return Err(io::Error::new(
                            ErrorKind::InvalidData,
                            "No density map loaded for this molecule; can't save it.",
                        ));
                    }
                },
                None => {
                    return Err(io::Error::new(
                        ErrorKind::InvalidData,
                        "No molecule open; can't save a density Map.",
                    ));
                }
            },
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
    /// using it. We load normal values, C-terminal values, and N-terminal values to different
    /// fields
    pub fn load_aa_charges_ff(&mut self) -> io::Result<()> {
        let internal = parse_amino_charges(AMINO_19)?;
        let n_terminus = parse_amino_charges(AMINO_NT12)?;
        let c_terminus = parse_amino_charges(AMINO_CT12)?;

        let ff_charge_data = ProtFFTypeChargeMap {
            internal,
            n_terminus,
            c_terminus,
        };

        if let Some(mol) = &mut self.molecule {
            if let Err(e) = populate_ff_and_q(&mut mol.common.atoms, &mol.residues, &ff_charge_data)
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

                // todo: You might need to re-init MD here as well.
            }
        }

        self.ff_params.prot_ff_q_map = Some(ff_charge_data);

        Ok(())
    }

    /// Load parameter files for general organic molecules (GAFF2), and proteins/amino acids (PARM19).
    /// This also populates ff type and charge on our protein atoms. These are built into the application
    /// as static strings.
    ///
    /// This only loads params that haven't already been loaded.
    pub fn load_ffs_general(&mut self) {
        let start = Instant::now();

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
        if self.ff_params.prot_ff_q_map.is_none() {
            if let Err(e) = self.load_aa_charges_ff() {
                handle_err(
                    &mut self.ui,
                    format!("Unable to load protein charges (static): {e}"),
                );
            }
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

        let elapsed = start.elapsed().as_millis();
        println!("Loaded static FF data in {elapsed}ms");
    }

    /// Load Mol2 and optionally, FRCMOD data from our Amber Geostd database into state.
    pub fn load_geostd_mol_data(
        &mut self,
        ident: &str,
        load_mol2: bool,
        load_frcmod: bool,
        redraw_lig: &mut bool,
    ) {
        let ident = ident.trim().to_owned();

        match amber_geostd::load_mol_files(&ident) {
            Ok(data) => {
                // Load FRCmod first, then the Ligand constructor will populate that it loaded.
                if load_frcmod {
                    if let Some(frcmod) = data.frcmod {
                        self.ff_params.lig_specific.insert(
                            ident.to_uppercase(),
                            // todo: Don't unwrap.
                            ForceFieldParamsKeyed::new(
                                &ForceFieldParams::from_frcmod(&frcmod).unwrap(),
                            ),
                        );

                        if let Some(lig) = &mut self.ligand {
                            lig.frcmod_loaded = true;
                        }
                    }
                }

                if let Some(_lib) = data.lib {
                    println!("todo: Lib data available from geostd; download?");
                }

                if load_mol2 {
                    match Mol2::new(&data.mol2) {
                        Ok(mol2) => {
                            let mol: MoleculeSmall = mol2.try_into().unwrap();
                            self.ligand = Some(Ligand::new(mol, &self.ff_params.lig_specific));
                            self.mol_dynamics = None;

                            // self.update_from_prefs();

                            *redraw_lig = true;
                        }
                        Err(e) => handle_err(
                            &mut self.ui,
                            format!("Unable to make a Mol2 from Geostd data: {:?}", e),
                        ),
                    }
                }
            }
            Err(_) => handle_err(
                &mut self.ui,
                format!("Unable to load Amber Geostd data (Server or internet problem?)"),
            ),
        }
    }

    /// We run this at init. Loads all relevant files marked as "last opened".
    pub fn load_last_opened(&mut self) {
        let last_opened = self.to_save.opened_items.clone();
        if let Some(path) = &last_opened {
            if let Err(e) = self.open_molecule(path) {
                handle_err(&mut self.ui, e.to_string());
            }
        }

        // Load map after molecule, so it knows the coordinates.
        let last_map_opened = self.to_save.last_map_opened.clone();
        if let Some(path) = &last_map_opened {
            if let Err(e) = self.open(path) {
                handle_err(&mut self.ui, e.to_string());
            }
        }

        let last_ligand_opened = self.to_save.last_ligand_opened.clone();
        if let Some(path) = &last_ligand_opened {
            if let Err(e) = self.open_molecule(path) {
                handle_err(&mut self.ui, e.to_string());
            }
        }

        let last_frcmod_opened = self.to_save.last_frcmod_opened.clone();
        if let Some(path) = &last_frcmod_opened {
            if let Err(e) = self.open_force_field(path) {
                handle_err(&mut self.ui, e.to_string());
            }
        }
    }
}

/// Utility for finding the Gemmi application, used for opening mmCIF structure factors,
/// and MTZ. This allows gemmi to be distributed in a folder colacated with this program's executable.
pub fn gemmi_path() -> Option<&'static Path> {
    let local_gemmi = Path::new("./gemmi");
    if local_gemmi.exists() {
        Some(&local_gemmi)
    } else {
        // If Gemmi is not in a folder colacated with the application, fall back
        // to the system Path.
        None
    }
}
