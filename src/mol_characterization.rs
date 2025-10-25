//! An experiment to categorize small molecules, and find similar ones

pub struct Classification {
    pub num_atoms: usize,
    pub num_rings: usize,
    pub num_amines: usize,
    pub num_nitrogens: usize,
    pub num_oxygens: usize,
}
