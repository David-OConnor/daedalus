use std::{
    fs,
    fs::File,
    io,
    io::{ErrorKind, Read},
    path::Path,
    time::Instant,
};

use bio_files::{
    DensityMap, MmCif, Mol2, Pdbqt, gemmi_sf_to_map, md_params::ForceFieldParams, sdf::Sdf,
};
use chrono::Utc;
use egui_file_dialog::FileDialog;
use graphics::{Camera, ControlScheme, EngineUpdates, EntityUpdate, Scene};
use na_seq::{AaIdent, Element};

use crate::{
    State,
    cam_misc::move_mol_to_cam,
    download_mols,
    drawing::EntityClass,
    drawing_wrappers,
    mol_lig::MoleculeSmall,
    molecule::{
        MoGenericRefMut, MolGenericRef, MolType, MoleculeCommon, MoleculeGeneric, MoleculePeptide,
    },
    prefs::{OpenHistory, OpenType},
    reflection::{DENSITY_CELL_MARGIN, DENSITY_MAX_DIST, DensityRect, ElectronDensity},
    util::{handle_err, handle_success},
};

impl State {
    /// A single endpoint to open a number of file types
    pub fn open(
        &mut self,
        path: &Path,
        scene: Option<&mut Scene>,
        engine_updates: &mut EngineUpdates,
    ) -> io::Result<()> {
        match path
            .extension()
            .unwrap_or_default()
            .to_ascii_lowercase()
            .to_str()
            .unwrap_or_default()
        {
            // The cif branch here also handles 2fo-fc mmCIF files.
            "sdf" | "mol2" | "pdbqt" | "pdb" | "cif" => {
                self.open_molecule(path, scene, engine_updates)?
            }
            "prmtop" => {
                // todo
            }
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

    /// For opening molecule files: Proteins, small organic molecules, nucleic acids etc.
    pub fn open_molecule(
        &mut self,
        path: &Path,
        mut scene: Option<&mut Scene>,
        engine_updates: &mut EngineUpdates,
    ) -> io::Result<()> {
        let binding = path.extension().unwrap_or_default().to_ascii_lowercase();
        let extension = binding;

        let molecule = match extension.to_str().unwrap() {
            "sdf" => {
                let mut m: MoleculeSmall = Sdf::load(path)?.try_into()?;
                m.common.path = Some(path.to_owned());
                Ok(MoleculeGeneric::Ligand(m))
            }
            "mol2" => {
                let mut m: MoleculeSmall = Mol2::load(path)?.try_into()?;
                m.common.path = Some(path.to_owned());
                Ok(MoleculeGeneric::Ligand(m))
            }
            "pdbqt" => {
                let mut m: MoleculeSmall = Pdbqt::load(path)?.try_into()?;
                m.common.path = Some(path.to_owned());
                Ok(MoleculeGeneric::Ligand(m))
            }
            "cif" => {
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

                let Some(ff_map) = &self.ff_param_set.peptide_ff_q_map else {
                    return Err(io::Error::new(
                        ErrorKind::Other,
                        "Missing FF map when opening a protein; can't validate H",
                    ));
                };

                let mol = MoleculePeptide::from_mmcif(
                    cif_data,
                    &ff_map,
                    Some(path.to_owned()),
                    self.to_save.ph,
                )?;
                self.cif_pdb_raw = Some(data_str);

                // Mark all other peptides as not last session.
                for history in &mut self.to_save.open_history {
                    if let OpenType::Peptide = history.type_ {
                        history.last_session = false;
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
                    MoleculeGeneric::Peptide(m) => {
                        self.volatile.aa_seq_text = String::with_capacity(m.common.atoms.len());
                        for aa in &m.aa_seq {
                            self.volatile
                                .aa_seq_text
                                .push_str(&aa.to_str(AaIdent::OneLetter));
                        }

                        self.volatile.flags.ss_mesh_created = false;
                        self.volatile.flags.sas_mesh_created = false;

                        self.volatile.flags.clear_density_drawing = true;

                        self.peptide = Some(m);

                        self.update_history(path, OpenType::Peptide);
                    }
                    MoleculeGeneric::Ligand(mut mol) => {
                        self.mol_dynamics = None;

                        self.volatile.active_mol = Some((MolType::Ligand, self.ligands.len())); // Prior to push; no - 1
                        mol.update_aux(&self.volatile.active_mol, &mut self.lig_specific_params);

                        if let Some(ref mut s) = scene {
                            move_mol_to_cam(&mut mol.common, &s.camera);
                            if let ControlScheme::Arc { center: _ } =
                                s.input_settings.control_scheme
                            {
                                s.input_settings.control_scheme = ControlScheme::Arc {
                                    center: mol.common.centroid().into(),
                                };
                            }
                        }
                        self.ligands.push(mol);

                        // Make sure to draw *after* loaded into state.
                        if let Some(s) = scene {
                            drawing_wrappers::draw_all_ligs(self, s);
                        }

                        engine_updates.entities =
                            EntityUpdate::Classes(vec![EntityClass::Ligand as u32]);
                        self.update_history(path, OpenType::Ligand);
                    }
                    MoleculeGeneric::NucleicAcid(mut mol) => {
                        self.volatile.active_mol =
                            Some((MolType::NucleicAcid, self.nucleic_acids.len())); // Prior to push; no - 1

                        if let Some(ref mut s) = scene {
                            move_mol_to_cam(&mut mol.common, &s.camera);
                            if let ControlScheme::Arc { center: _ } =
                                s.input_settings.control_scheme
                            {
                                s.input_settings.control_scheme = ControlScheme::Arc {
                                    center: mol.common.centroid().into(),
                                };
                            }
                        }
                        self.nucleic_acids.push(mol);

                        if let Some(s) = scene {
                            drawing_wrappers::draw_all_nucleic_acids(self, s);
                        }

                        engine_updates.entities =
                            EntityUpdate::Classes(vec![EntityClass::NucleicAcid as u32]);
                        self.update_history(path, OpenType::NucleicAcid);
                    }
                    MoleculeGeneric::Lipid(mut mol) => {
                        self.volatile.active_mol = Some((MolType::Lipid, self.nucleic_acids.len())); // Prior to push; no - 1

                        if let Some(ref mut s) = scene {
                            move_mol_to_cam(&mut mol.common, &s.camera);
                            if let ControlScheme::Arc { center: _ } =
                                s.input_settings.control_scheme
                            {
                                s.input_settings.control_scheme = ControlScheme::Arc {
                                    center: mol.common.centroid().into(),
                                };
                            }
                        }
                        self.lipids.push(mol);

                        if let Some(s) = scene {
                            drawing_wrappers::draw_all_lipids(self, s);
                        }

                        engine_updates.entities =
                            EntityUpdate::Classes(vec![EntityClass::Lipid as u32]);
                        self.update_history(path, OpenType::Lipid);
                    }
                }
                // Save the open history.
                self.update_save_prefs(false);

                if let Some(mol) = &mut self.peptide {
                    // Only after updating from prefs (to prevent unnecessary loading) do we update data avail.
                    mol.updates_rcsb_data(&mut self.volatile.mol_pending_data_avail);
                }

                // Now, save prefs: This is to save last opened. Note that anomolies happen
                // if we update the molecule here, e.g. with docking site posit.
                self.update_save_prefs_no_mol();

                // if self.ligand.is_some() {
                //     if self.get_make_docking_setup().is_none() {
                //         eprintln!("Problem making or getting docking setup.");
                //     }
                // }

                self.volatile.flags.new_mol_loaded = true;
            }
            Err(e) => {
                return Err(e);
            }
        }

        Ok(())
    }

    pub fn load_density(&mut self, dens_map: DensityMap) {
        if let Some(mol) = &mut self.peptide {
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

            #[cfg(feature = "cuda")]
            let dens = dens_rect.make_densities(
                &self.dev,
                &self.kernel_reflections,
                &atom_posits,
                &dens_map.cell,
                DENSITY_MAX_DIST,
            );

            #[cfg(not(feature = "cuda"))]
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
        let ident = String::new(); // todo: Set this up.
        self.load_density(dm);

        // self.to_save.last_map_opened = Some(path.to_owned());
        // self.update_history(path, OpenType::Map, &ident);
        self.update_history(path, OpenType::Map);

        // Save the open history.
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
                self.ff_param_set.small_mol = Some(ForceFieldParams::load_dat(path)?);

                println!("\nLoaded forcefields:");
                let v = &self.ff_param_set.small_mol.as_ref().unwrap();
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
                    for d in di {
                        println!("DH: {:?}, {}, {}", d.atom_types, d.barrier_height, d.phase);
                    }
                }

                println!("Dihedral, improper:");
                for di in v.improper.values().take(20) {
                    for d in di {
                        println!("Imp: {:?}, {}, {}", d.atom_types, d.barrier_height, d.phase);
                    }
                }

                // todo: Get VDW loading working.
                println!("Vdw");
                for di in v.lennard_jones.values().take(20) {
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

                self.lig_specific_params.insert(
                    mol_name.to_uppercase(),
                    ForceFieldParams::load_frcmod(path)?,
                );

                // Update the lig's FRCMOD status A/R, if the ligand is opened already.
                for lig in &mut self.ligands {
                    if &lig.common.ident.to_uppercase() == &mol_name.to_uppercase() {
                        lig.frcmod_loaded = true;
                    }
                }

                // self.to_save.last_frcmod_opened = Some(path.to_owned());
                // self.update_history(path, OpenType::Frcmod, mol_name.as_str());
                self.update_history(path, OpenType::Frcmod);
                // Save the open history.
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
                // We don't allow editing the protein files yet, so save the raw CIF.
                if let Some(data) = &mut self.cif_pdb_raw {
                    fs::write(path, data)?;

                    let ident = match &self.peptide {
                        Some(mol) => mol.common.ident.clone(),
                        None => String::new(),
                    };

                    // self.to_save.last_peptide_opened = Some(path.to_owned());
                    // self.update_history(path, OpenType::Peptide, &ident);
                    self.update_history(path, OpenType::Peptide);

                    // Save the open history.
                    self.update_save_prefs(false);
                }
            }
            "sdf" => match self.active_mol() {
                Some(lig) => {
                    lig.to_sdf().save(path)?;

                    self.update_history(path, OpenType::Ligand);

                    // Save the open history.
                    self.update_save_prefs(false);
                }
                None => return Err(io::Error::new(ErrorKind::InvalidData, "No ligand to save")),
            },
            "mol2" => match self.active_mol() {
                Some(lig) => {
                    lig.to_mol2().save(path)?;
                    self.update_history(path, OpenType::Ligand);

                    // Save the open history.
                    self.update_save_prefs(false);
                }
                None => return Err(io::Error::new(ErrorKind::InvalidData, "No ligand to save")),
            },
            "prmtop" => (), // todo
            "pdbqt" => match self.active_mol() {
                Some(lig) => {
                    lig.to_pdbqt().save(path)?;
                    self.update_history(path, OpenType::Ligand);

                    // Save the open history.
                    self.update_save_prefs(false);
                }
                None => return Err(io::Error::new(ErrorKind::InvalidData, "No ligand to save")),
            },
            // todo: Consider if you want to store the original map bytes, as you do with
            // todo mmCIF files, instead of saving what you parsed.
            "map" => match &self.peptide {
                Some(mol) => match &mol.density_map {
                    Some(dm) => {
                        dm.save(path)?;
                        // self.to_save.last_map_opened = Some(path.to_owned());

                        // self.update_history(path, OpenType::Map, &mol.common.ident.clone());
                        self.update_history(path, OpenType::Map);

                        // Save the open history.
                        self.update_save_prefs(false);
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

    /// Load Mol2 and optionally, FRCMOD data from our Amber Geostd database into state.
    pub fn load_geostd_mol_data(
        &mut self,
        ident: &str,
        load_mol2: bool,
        load_frcmod: bool,
        redraw_lig: &mut bool,
        cam: &Camera,
    ) {
        let start = Instant::now();
        println!("Loading mol files from Amber Geostd...");

        let ident = ident.trim().to_owned();
        download_mols::load_geostd2(self, cam, &ident, load_mol2, load_frcmod, redraw_lig);

        let elapsed = start.elapsed().as_millis();
        println!("Loaded Amber Geostd in {elapsed:.1}ms");
    }

    /// We run this at init. Loads all relevant files marked as "last opened".
    pub fn load_last_opened(&mut self) {
        let histories = self.to_save.open_history.clone(); // todo: I hate this.

        // This prevents loading duplicates
        // todo: When you place paths in mol.common etc, re-implement this.
        // todo: We must track which files are open.

        for history in &histories {
            if !history.last_session {
                continue;
            }

            match history.type_ {
                OpenType::Peptide => {
                    if let Err(e) = self.open_molecule(&history.path, None, &mut Default::default())
                    {
                        handle_err(&mut self.ui, e.to_string());
                    }
                }
                OpenType::Ligand | OpenType::NucleicAcid | OpenType::Lipid => {
                    if let Err(e) = self.open_molecule(&history.path, None, &mut Default::default())
                    {
                        handle_err(&mut self.ui, e.to_string());
                    }
                }
                OpenType::Map => {
                    if let Err(e) = self.open(&history.path, None, &mut Default::default()) {
                        handle_err(&mut self.ui, e.to_string());
                    }
                }
                OpenType::Frcmod => {
                    if let Err(e) = self.open_force_field(&history.path) {
                        handle_err(&mut self.ui, e.to_string());
                    }
                }
            }
        }
    }

    /// Keeps the history tidy.
    pub fn update_history(&mut self, path: &Path, type_: OpenType) {
        for item in &mut self.to_save.open_history {
            if item.path == *path {
                item.last_session = true;
                item.timestamp = Utc::now();
                return;
            }
        }

        self.to_save
            .open_history
            .push(OpenHistory::new(path, type_));
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

// // todo:
// /// Load general parameter files for proteins, and small organic molecules.
// /// This also populates ff type and charge for protein atoms. These are built into the application
// /// as static strings.
// ///
// /// This is similar to `FfParamSet::new()`, but using static strings.
// pub fn load_ffs_general() -> io::Result<FfParamSet> {
//     let mut result = FfParamSet::default();
//
//     let peptide = ForceFieldParamsKeyed::new(&ForceFieldParams::from_dat(PARM_19)?);
//     let peptide_frcmod = ForceFieldParamsKeyed::new(&ForceFieldParams::from_frcmod(FRCMOD_FF19SB)?);
//     result.peptide = Some(merge_params(&peptide, Some(&peptide_frcmod)));
//
//     let internal = parse_amino_charges(AMINO_19)?;
//     let n_terminus = parse_amino_charges(AMINO_NT12)?;
//     let c_terminus = parse_amino_charges(AMINO_CT12)?;
//
//     result.peptide_ff_q_map = Some(ProtFFTypeChargeMap {
//         internal,
//         n_terminus,
//         c_terminus,
//     });
//
//     result.small_mol = Some(ForceFieldParamsKeyed::new(&ForceFieldParams::from_dat(GAFF2)?));
//
//     let dna = ForceFieldParamsKeyed::new(&ForceFieldParams::from_dat(OL24_LIB)?);
//     let dna_frcmod = ForceFieldParamsKeyed::new(&ForceFieldParams::from_frcmod(OL24_FRCMOD)?);
//     result.dna = Some(merge_params(&dna, Some(&dna_frcmod)));
//
//     result.rna = Some(ForceFieldParamsKeyed::new(&ForceFieldParams::from_dat(RNA_LIB)?));
//
//     Ok(result)
// }

impl MoleculeCommon {
    /// Save to disk.
    pub fn save(&self, dialog: &mut FileDialog) -> io::Result<()> {
        let fname_default = {
            let ext_default = "mol2"; // The default; more robust than SDF.

            let name = if self.ident.is_empty() {
                "molecule".to_string()
            } else {
                self.ident.clone()
            };
            format!("{name}.{ext_default}")
        };

        dialog.config_mut().default_file_name = fname_default.to_string();
        dialog.config_mut().default_file_filter = Some("Molecule (small)".to_owned());

        dialog.save_file();

        Ok(())
    }
}
