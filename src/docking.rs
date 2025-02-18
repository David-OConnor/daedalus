use crate::molecule::{Ligand, Molecule};

/// Calculate binding energy, in kcal/mol. The result will be negative. The maximum (negative) binding
/// energy may be the ideal conformation.
pub fn binding_energy(mol: &Molecule, ligand: &Ligand) -> f32 {
    0.
}

// Find hydrogen bond interaction, hydrophobic interactions between ligand and protein.
// Find the "perfect" "Het" or "lead" molecule that will act as drug receptor
