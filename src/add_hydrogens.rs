use crate::{
    aa_coords::aa_data_from_coords,
    molecule::{Atom, AtomRole, Molecule, ResidueType},
};

impl Molecule {
    /// Adds hydrogens, and populdates residue dihedral angles.
    pub fn populate_hydrogens_angles(&mut self) {
        // todo: Move this fn to this module? Split this and its diehdral component, or not?

        let mut prev_cp_ca = None;

        let res_len = self.residues.len();

        // todo: The Clone avoids a double-borrow error below. Come back to /avoid if possible.
        let res_clone = self.residues.clone();

        for (i, res) in self.residues.iter_mut().enumerate() {
            let atoms: Vec<&Atom> = res.atoms.iter().map(|i| &self.atoms[*i]).collect();

            let mut n_next_pos = None;
            // todo: Messy DRY from the aa_data_from_coords fn.
            if i < res_len - 1 {
                let res_next = &res_clone[i + 1];
                let n_next = res_next.atoms.iter().find(|i| {
                    if let Some(role) = &self.atoms[**i].role {
                        *role == AtomRole::N_Backbone
                    } else {
                        false
                    }
                });
                if let Some(n_next) = n_next {
                    n_next_pos = Some(self.atoms[*n_next].posit);
                }
            }

            if let ResidueType::AminoAcid(aa) = &res.res_type {
                let (dihedral, hydrogens, cp_pos, ca_pos) =
                    aa_data_from_coords(&atoms, *aa, prev_cp_ca, n_next_pos);

                for h in hydrogens {
                    self.atoms.push(h);
                    res.atoms.push(self.atoms.len() - 1);
                }
                prev_cp_ca = Some((cp_pos, ca_pos));
                res.dihedral = Some(dihedral);
            }
        }
    }
}
