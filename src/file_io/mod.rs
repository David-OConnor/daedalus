use std::{fs, io, io::ErrorKind, path::Path, sync::mpsc, thread, time::Instant};

use bio_apis::pubchem;
use bio_files::{
    DensityMap, MmCif, Mol2, Pdbqt, cif_sf::CifStructureFactors, gemmi_sf_to_map,
    md_params::ForceFieldParams, sdf::Sdf,
};
use chrono::Utc;
use egui_file_dialog::FileDialog;
use graphics::{ControlScheme, EngineUpdates, EntityUpdate, Scene};
use lin_alg::f64::Vec3;
use na_seq::{AaIdent, Element};
use rand::Rng;
use rustfft::FftPlanner;

use crate::{
    State,
    cam_misc::move_mol_to_cam,
    download_mols,
    drawing::draw_peptide,
    drawing_wrappers,
    mol_lig::MoleculeSmall,
    molecule::{
        MolGenericTrait, MolIdent, MolType, MoleculeCommon, MoleculeGeneric, MoleculePeptide,
    },
    prefs::{OpenHistory, OpenType},
    reflection::{
        DENSITY_CELL_MARGIN, DENSITY_MAX_DIST, DensityPt, DensityRect, density_map_from_sf,
    },
    util::{handle_err, handle_success},
};

// When opening molecules deconflict; don't allow a mol to be closer than this to another.
const MOL_MIN_DIST_OPEN: f64 = 12.;

impl State {
    /// A single endpoint to open a number of file types. Delegates to functions that handle
    /// specific classes of file to open.
    pub fn open_file(
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
                self.open_mol_from_file(path, scene, engine_updates)?
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
    pub fn open_mol_from_file(
        &mut self,
        path: &Path,
        scene: Option<&mut Scene>,
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
                        // todo: Experimenting with a local impl.
                        gemmi_sf_to_map(path, gemmi_path())?;
                        let dm = gemmi_sf_to_map(path, gemmi_path())?;

                        // let mut fft_planner = FftPlanner::new();
                        // let data = CifStructureFactors::new_from_path(path)?;

                        // let dm = density_map_from_mmcif(&data, &mut fft_planner)?;

                        self.load_density(dm);

                        self.update_history(path, OpenType::Map);
                        // Save the open history.
                        self.update_save_prefs(false);

                        return Ok(());
                    }
                }

                let data_str = fs::read_to_string(path)?;
                let cif_data = MmCif::new(&data_str)?;

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

                Ok(MoleculeGeneric::Peptide(mol))
            }
            _ => Err(io::Error::new(
                ErrorKind::InvalidData,
                "Invalid file extension",
            )),
        };

        match molecule {
            Ok(mol_gen) => self.load_mol_to_state(mol_gen, scene, engine_updates, Some(path)),
            Err(e) => return Err(e),
        }

        Ok(())
    }

    pub fn load_density(&mut self, dens_map: DensityMap) {
        if let Some(mol) = &mut self.peptide {
            // Sample atoms, so we know where to draw the (periodic) density data.
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
                &dens_map.hdr.inner.cell,
                DENSITY_MAX_DIST,
            );

            #[cfg(not(feature = "cuda"))]
            let dens = dens_rect.make_densities(
                &self.dev,
                &atom_posits,
                &dens_map.hdr.inner.cell,
                DENSITY_MAX_DIST,
            );

            let elec_dens: Vec<_> = dens
                .iter()
                .map(|d| DensityPt {
                    coords: d.coords,
                    density: d.density,
                })
                .collect();

            // println!("Rect: {:?}", dens_rect);

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

        self.update_history(path, OpenType::Map);
        // Save the open history.
        self.update_save_prefs(false);

        Ok(())
    }

    /// An electron density MTZ file. We use Gemmi's sf2map functionality, as we do for 2fo-fc files.
    pub fn open_mtz(&mut self, path: &Path) -> io::Result<()> {
        let dm = gemmi_sf_to_map(path, gemmi_path())?;
        self.load_density(dm);

        self.update_history(path, OpenType::Map);
        // Save the open history.
        self.update_save_prefs(false);

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
                    if lig.common.ident.to_uppercase() == mol_name.to_uppercase() {
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
        engine_updates: &mut EngineUpdates,
        scene: &mut Scene,
    ) {
        let start = Instant::now();
        println!("Loading mol files from Amber Geostd...");

        let ident = ident.trim().to_owned();
        download_mols::load_geostd2(self, scene, &ident, load_mol2, load_frcmod, engine_updates);

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
                    if let Err(e) =
                        self.open_mol_from_file(&history.path, None, &mut Default::default())
                    {
                        handle_err(&mut self.ui, e.to_string());
                    } else {
                        if let Some(p) = &history.position {
                            self.peptide.as_mut().unwrap().common.move_to(p.clone());
                        }
                    }
                }
                OpenType::Ligand | OpenType::NucleicAcid | OpenType::Lipid => {
                    if let Err(e) =
                        self.open_mol_from_file(&history.path, None, &mut Default::default())
                    {
                        handle_err(&mut self.ui, e.to_string());
                    } else {
                        if let Some(p) = &history.position {
                            println!("\n\n Hist pos: {:?}", p); // todo temp
                            let i = self.ligands.len() - 1;
                            self.ligands[i].common.move_to(p.clone());
                        }
                    }
                }
                OpenType::Map => {
                    if let Err(e) = self.open_file(&history.path, None, &mut Default::default()) {
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

    /// Add a history event, or update its timestamp and restor the list.
    pub fn update_history(&mut self, path: &Path, type_: OpenType) {
        let mut first_i = None;
        let mut to_delete = Vec::new();

        for (i, item) in self.to_save.open_history.iter_mut().enumerate() {
            if item.path == *path {
                if first_i.is_none() {
                    item.last_session = true;
                    item.timestamp = Utc::now();
                    first_i = Some(i);
                }
                to_delete.push(i);
            }
        }

        to_delete.sort_unstable_by(|a, b| b.cmp(a));

        let mut moved_item = None;
        for i in to_delete {
            let item = self.to_save.open_history.remove(i);
            if Some(i) == first_i {
                moved_item = Some(item);
            }
        }

        if let Some(item) = moved_item {
            self.to_save.open_history.push(item);
        } else {
            self.to_save
                .open_history
                .push(OpenHistory::new(path, type_));
        }
    }

    /// This is a central point for loading a molecule into state. It handles the cases
    /// of loading from file, and online sources. All cases of opening a molecule pass through this.
    ///
    /// It centralizes steps that should be completed upon molecule open, and attempts to consolidate
    /// between different molecule types.
    pub fn load_mol_to_state(
        &mut self,
        mut mol: MoleculeGeneric,
        mut scene: Option<&mut Scene>,
        engine_updates: &mut EngineUpdates,
        path: Option<&Path>,
    ) {
        let mol_type = mol.mol_type();
        let entity_class = mol_type.entity_type() as u32;
        let open_type = mol_type.to_open_type();

        let mut centroid = Vec3::new_zero();
        let mut ident = String::new();

        // The pre-push index.
        let mol_i = match mol_type {
            MolType::Peptide => 0,
            MolType::Ligand => self.ligands.len(),
            MolType::NucleicAcid => self.nucleic_acids.len(),
            MolType::Lipid => self.lipids.len(),
            MolType::Water => unreachable!(),
        };

        match mol {
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

                centroid = m.center;
                ident = m.common.ident.clone();
                self.peptide = Some(m);

                if let Some(ref mut s) = scene {
                    draw_peptide(self, s);
                }
            }
            MoleculeGeneric::Ligand(mut mol) => {
                if let Some(ref mut s) = scene {
                    move_mol_to_cam(&mut mol.common_mut(), &s.camera);
                }

                self.mol_dynamics = None;

                mol.update_aux(&self.volatile.active_mol, &mut self.lig_specific_params);

                if let Some(ref mut s) = scene {
                    let centroid = mol.common.centroid();
                    // If there is already a molecule here, offset.
                    // todo: Apply this logic to other mol types A/R
                    for mol_other in &self.ligands {
                        if (mol_other.common.centroid() - centroid).magnitude() < MOL_MIN_DIST_OPEN
                        {
                            let mut rng = rand::rng();
                            let dir =
                                Vec3::new(rng.random(), rng.random(), rng.random()).to_normalized();

                            let pos_new = centroid + dir * MOL_MIN_DIST_OPEN;

                            mol.common.move_to(pos_new);
                            // Note: No further safeguard in this case.
                            break;
                        }
                    }
                }

                for ident in &mol.idents {
                    // todo: Should we use the pubchem ID? Be flexible? Check both?
                    if !matches!(ident, MolIdent::PdbeAmber(_)) {
                        continue;
                    }

                    match self.to_save.smiles_map.get(&ident) {
                        Some(v) => {
                            println!("Loaded smiles for {ident:?} from our local DB: {v}");
                            mol.smiles = Some(v.clone());
                            break;
                        }
                        None => {
                            let (tx, rx) = mpsc::channel(); // one-shot channel
                            let ident_for_thread = ident.clone();

                            thread::spawn(move || {
                                let data = pubchem::get_smiles(&ident_for_thread.to_str());
                                let _ = tx.send((ident_for_thread, data));
                                println!("Sent thread"); // todo temp.
                            });

                            self.volatile.smiles_pending_data_avail = Some(rx);
                        }
                    }
                }

                centroid = mol.common.centroid();
                ident = mol.common.ident.clone();

                self.ligands.push(mol);

                // Make sure to draw *after* loaded into state.
                if let Some(ref mut s) = scene {
                    drawing_wrappers::draw_all_ligs(self, s);
                }
            }
            MoleculeGeneric::NucleicAcid(mut mol) => {
                if let Some(ref mut s) = scene {
                    move_mol_to_cam(&mut mol.common_mut(), &s.camera);
                }

                centroid = mol.common.centroid();
                ident = mol.common.ident.clone();

                self.nucleic_acids.push(mol);

                if let Some(ref mut s) = scene {
                    drawing_wrappers::draw_all_nucleic_acids(self, s);
                }

                engine_updates.entities = EntityUpdate::Classes(vec![entity_class]);
            }
            MoleculeGeneric::Lipid(mut mol) => {
                if let Some(ref mut s) = scene {
                    move_mol_to_cam(&mut mol.common_mut(), &s.camera);
                }

                centroid = mol.common.centroid();
                ident = mol.common.ident.clone();

                self.lipids.push(mol);

                if let Some(ref mut s) = scene {
                    drawing_wrappers::draw_all_lipids(self, s);
                }
            }
        }

        engine_updates.entities = EntityUpdate::Classes(vec![entity_class]);

        self.volatile.active_mol = Some((mol_type, mol_i));
        self.volatile.orbit_center = Some((mol_type, mol_i));

        if let Some(ref mut s) = scene {
            if let ControlScheme::Arc { center } = &mut s.input_settings.control_scheme {
                *center = centroid.into();
            }
        }

        if mol_type == MolType::Peptide {
            // Mark all other peptides as not last session.
            // We do this as we currently only support one peptide at a time.
            for history in &mut self.to_save.open_history {
                if matches!(history.type_, OpenType::Peptide | OpenType::Map) {
                    history.last_session = false;
                }
            }

            if let Some(mol) = &mut self.peptide {
                // Only after updating from prefs (to prevent unnecessary loading) do we update data avail.
                mol.updates_rcsb_data(&mut self.volatile.mol_pending_data_avail);
            }
        }

        if let Some(p) = path {
            self.update_history(p, open_type);
        }

        // Save the open history.
        self.update_save_prefs(false);

        // Now, save prefs: This is to save last opened. Note that anomalies happen
        // if we update the molecule here, e.g. with docking site posit.
        self.update_save_prefs_no_mol();

        // if self.ligand.is_some() {
        //     if self.get_make_docking_setup().is_none() {
        //         eprintln!("Problem making or getting docking setup.");
        //     }
        // }

        self.volatile.flags.new_mol_loaded = true;

        // Note: This may be overwritten by `load_file` with the full file path.
        handle_success(&mut self.ui, format!("Loaded molecule {ident}"));
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
