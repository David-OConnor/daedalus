//! Add positions and orientations for all sidechain atoms. Quite repetative!

// todo: This module is a C+P from peptide, modified.

use std::f64::consts::TAU;

use lin_alg::f64::{Quaternion, Vec3};

use crate::aa_coords::{bond_vecs::*, calc_dihedral_angle, sidechain::*};

/// Calculate the orientation, as a quaternion, and position, as a vector, of an atom, given the orientation of a
/// previous atom, and the bond angle. `bond_to_prev_local` is the vector representing the bond to
/// this atom from the previous atom's orientation. `bond_to_prev_local`, `bond_to_next_local`, and
/// `bond_to_this_local` are in the atom's local coordinate space; not worldspace. They are all unit vectors.
///
/// This is solving an iteration of the *forward kinematics problem*.
pub fn find_atom_placement(
    or_prev: Quaternion,
    bond_to_prev_local: Vec3,
    bond_to_next_local: Vec3,
    dihedral_angle: f64,
    posit_prev: Vec3,
    posit_2_back: Vec3,
    bond_to_this_local: Vec3,
    bond_to_this_len: f64,
) -> (Vec3, Quaternion) {
    let prev_bond_world = posit_prev - posit_2_back;

    // Find the position; this is passed directly to the output, and isn't used for further
    // calcualtions within this function.
    let position = posit_prev + or_prev.rotate_vec(bond_to_this_local) * bond_to_this_len;

    // #1: Align the prev atom's bond vector to world space based on the prev atom's orientation.
    let bond_to_this_worldspace = or_prev.rotate_vec(bond_to_this_local);

    // #2: Find the rotation quaternion that aligns the (inverse of) the local-space bond to
    // the prev atom with the world-space "to" bond of the previous atom. This is also the
    // orientation of our atom, without applying the dihedral angle.
    let bond_alignment_rotation =
        Quaternion::from_unit_vecs(bond_to_prev_local * -1., bond_to_this_worldspace);

    // #3: Rotate the orientation around the dihedral angle. We must do this so that our
    // dihedral angle is in relation to the previous and next bonds.
    // Adjust the dihedral angle to be in reference to the previous 2 atoms, per the convention.
    let next_bond_worldspace = bond_alignment_rotation.rotate_vec(bond_to_next_local);

    let dihedral_angle_current = calc_dihedral_angle(
        bond_to_this_worldspace,
        prev_bond_world,
        next_bond_worldspace,
    );

    let angle_dif = dihedral_angle - dihedral_angle_current;

    let rotate_axis = bond_to_this_worldspace;

    let mut dihedral_rotation = Quaternion::from_axis_angle(rotate_axis, angle_dif);

    let dihedral_angle_current2 = calc_dihedral_angle(
        bond_to_this_worldspace,
        prev_bond_world,
        (dihedral_rotation * bond_alignment_rotation).rotate_vec(bond_to_next_local),
    );

    // todo: Quick and dirty approach here. You can perhaps come up with something
    // todo more efficient, ie in one shot.
    if (dihedral_angle - dihedral_angle_current2).abs() > 0.0001 {
        dihedral_rotation = Quaternion::from_axis_angle(rotate_axis, -angle_dif + TAU);
    }

    (position, dihedral_rotation * bond_alignment_rotation)
}

impl Arg {
    // todo: Equiv from `backbone_cart_coords`.
    pub fn sidechain_cart_coords(
        &self,
        c_alpha: Vec3,
        c_alpha_orientation: Quaternion,
        // todo: Do we want our prev bond anchor to be n-calpha?
        n_pos: Vec3,
    ) -> CoordsArg {
        // These are the angles between each of 2 4 equally-spaced atoms on a tetrahedron,
        // with center of (0., 0., 0.). They are the angle formed between 3 atoms.
        // We have chosen the two angles to describe the backbone. We have chosen these arbitrarily.

        let (c_beta, c_beta_orientation) = find_atom_placement(
            c_alpha_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            // Use our info about the previous 2 atoms so we can define the dihedral angle properly.
            // (world space)
            self.χ_1,
            c_alpha,
            n_pos,
            unsafe { CALPHA_R_BOND },
            LEN_SC,
        );

        let (c_gamma, c_gamma_orientation) = find_atom_placement(
            c_beta_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            // Use our info about the previous 2 atoms so we can define the dihedral angle properly.
            // (world space)
            self.χ_2,
            c_beta,
            c_alpha,
            unsafe { TETRA_B },
            LEN_SC,
        );

        let (c_delta, c_delta_orientation) = find_atom_placement(
            c_gamma_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            self.χ_3,
            c_gamma,
            c_beta,
            unsafe { TETRA_B },
            LEN_SC,
        );

        let (n_eps, n_eps_orientation) = find_atom_placement(
            c_delta_orientation,
            PLANAR3_A,
            unsafe { PLANAR3_B },
            self.χ_4,
            c_delta,
            c_gamma,
            unsafe { TETRA_B },
            LEN_SC,
        );

        let (c_zeta, c_zeta_orientation) = find_atom_placement(
            n_eps_orientation,
            PLANAR3_A,
            unsafe { PLANAR3_B },
            self.χ_5,
            n_eps,
            c_delta,
            unsafe { PLANAR3_B },
            LEN_SC,
        );

        let (n_eta1, n_eta1_orientation) = find_atom_placement(
            c_zeta_orientation,
            PLANAR3_A,
            unsafe { PLANAR3_B },
            TAU_DIV2,
            c_zeta,
            n_eps,
            unsafe { PLANAR3_B },
            LEN_SC,
        );

        let (n_eta2, n_eta2_orientation) = find_atom_placement(
            c_zeta_orientation,
            PLANAR3_A,
            unsafe { PLANAR3_B },
            TAU_DIV2,
            c_zeta,
            n_eps,
            unsafe { PLANAR3_C },
            LEN_SC,
        );

        let (h_n_eps, _) = find_atom_placement(
            n_eps_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            n_eps,
            c_delta,
            unsafe { PLANAR3_C },
            LEN_N_H,
        );

        let (h_n_eta1_a, _) = find_atom_placement(
            n_eta1_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            n_eta1,
            c_zeta,
            unsafe { PLANAR3_B },
            LEN_N_H,
        );

        let (h_n_eta1_b, _) = find_atom_placement(
            n_eta1_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            n_eta1,
            c_zeta,
            unsafe { PLANAR3_C },
            LEN_N_H,
        );

        let (h_n_eta2_a, _) = find_atom_placement(
            n_eta2_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            n_eta2,
            c_zeta,
            unsafe { PLANAR3_B },
            LEN_N_H,
        );

        let (h_n_eta2_b, _) = find_atom_placement(
            n_eta2_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            n_eta2,
            c_zeta,
            unsafe { PLANAR3_C },
            LEN_N_H,
        );
        let (h_c_beta_a, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_C },
            LEN_C_H,
        );

        let (h_c_beta_b, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        let (h_c_gamma_a, _) = find_atom_placement(
            c_gamma_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_gamma,
            c_beta,
            unsafe { TETRA_C },
            LEN_C_H,
        );
        let (h_c_gamma_b, _) = find_atom_placement(
            c_gamma_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_gamma,
            c_beta,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        let (h_c_delta_a, _) = find_atom_placement(
            c_delta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_delta,
            c_gamma,
            unsafe { TETRA_C },
            LEN_C_H,
        );

        let (h_c_delta_b, _) = find_atom_placement(
            c_delta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_delta,
            c_gamma,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        CoordsArg {
            c_beta,
            c_gamma,
            c_delta,
            n_eps,
            c_zeta,
            n_eta1,
            n_eta2,
            h_n_eps,
            h_n_eta1_a,
            h_n_eta1_b,
            h_n_eta2_a,
            h_n_eta2_b,
            h_c_beta_a,
            h_c_beta_b,
            h_c_gamma_a,
            h_c_gamma_b,
            h_c_delta_a,
            h_c_delta_b,

            c_beta_orientation,
            c_gamma_orientation,
            c_delta_orientation,
            n_eps_orientation,
            c_zeta_orientation,
            n_eta1_orientation,
            n_eta2_orientation,
        }
    }
}

impl His {
    pub fn sidechain_cart_coords(
        &self,
        c_alpha: Vec3,
        c_alpha_orientation: Quaternion,
        n_pos: Vec3,
    ) -> CoordsHis {
        // todo: ring. Find planar bond angles.
        let (c_beta, c_beta_orientation) = find_atom_placement(
            c_alpha_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            self.χ_1,
            c_alpha,
            n_pos,
            unsafe { CALPHA_R_BOND },
            LEN_SC,
        );

        let (c_gamma, c_gamma_orientation) = find_atom_placement(
            c_beta_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            self.χ_2,
            c_beta,
            c_alpha,
            unsafe { TETRA_B },
            LEN_SC,
        );

        // todo: You can probably get a neater/cleaner ring setup by iterating from one to the next
        // todo instead of splitting into 2 sides.
        let (c_delta1, c_delta1_orientation) = find_atom_placement(
            c_gamma_orientation,
            RING_BOND_IN,
            unsafe { RING5_BOND_OUT },
            1.8849555, // tau / 2 - tau / 5 // todo: Not quite planar, but close.
            c_gamma,
            c_beta,
            unsafe { TETRA_B },
            LEN_SC,
        );

        let (n_delta2, n_delta2_orientation) = find_atom_placement(
            c_gamma_orientation,
            RING_BOND_IN,
            unsafe { RING5_BOND_OUT },
            -1.8849555, // tau / 2 - tau / 5 // todo: Not quite planar, but close.
            c_gamma,
            c_beta,
            unsafe { TETRA_C },
            LEN_SC,
        );

        // todo: Is this right for N in a ring?
        let (n_eps1, n_eps1_orientation) = find_atom_placement(
            c_delta1_orientation,
            RING_BOND_IN,
            unsafe { RING5_BOND_OUT },
            0.,
            c_delta1,
            c_gamma,
            unsafe { RING5_BOND_OUT },
            LEN_SC,
        );

        let (c_eps2, c_eps2_orientation) = find_atom_placement(
            n_delta2_orientation,
            RING_BOND_IN,
            unsafe { RING5_BOND_OUT },
            0.,
            n_delta2,
            c_gamma,
            unsafe { RING5_BOND_OUT },
            LEN_SC,
        );

        let (h_n_delta, _) = find_atom_placement(
            n_delta2_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            n_delta2,
            c_gamma,
            unsafe { PLANAR3_C },
            LEN_N_H,
        );

        // let (h_n_eps, _) = find_atom_placement(
        //     n_eps1_orientation,
        //     H_BOND_IN,
        //     H_BOND_OUT,
        //     TAU_DIV2,
        //     n_eps1,
        //     c_delta1,
        //     unsafe { PLANAR3_C },
        //     LEN_N_H,
        // );

        let (h_c_beta_a, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_C },
            LEN_N_H,
        );

        let (h_c_beta_b, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_D },
            LEN_N_H,
        );

        let (h_c_delta1, _) = find_atom_placement(
            c_delta1_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_delta1,
            c_gamma,
            unsafe { PLANAR3_C },
            LEN_N_H,
        );

        let (h_c_eps2, _) = find_atom_placement(
            c_eps2_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_eps2,
            n_delta2,
            unsafe { PLANAR3_C },
            LEN_N_H,
        );
        // todo: These bond vecs are wrong! Needs to be tighter angles
        // todo due to there only being 5 atoms in the ring.

        CoordsHis {
            c_beta,
            c_gamma,
            c_delta1,
            n_delta2,
            n_eps1,
            c_eps2,
            h_n_delta,
            // h_n_eps,
            h_c_beta_a,
            h_c_beta_b,
            h_c_delta1,
            h_c_eps2,

            c_beta_orientation,
            c_gamma_orientation,
            c_delta1_orientation,
            n_delta2_orientation,
            n_eps1_orientation,
            c_eps2_orientation,
        }
    }
}

// Calpha R bond is tetra C
// SC prev is planar C

impl Lys {
    // todo: Equiv from `backbone_cart_coords`.
    pub fn sidechain_cart_coords(
        &self,
        c_alpha: Vec3,
        c_alpha_orientation: Quaternion,
        // todo: Do we want our prev bond anchor to be n-calpha?
        n_pos: Vec3,
    ) -> CoordsLys {
        // These are the angles between each of 2 4 equally-spaced atoms on a tetrahedron,
        // with center of (0., 0., 0.). They are the angle formed between 3 atoms.
        // We have chosen the two angles to describe the backbone. We have chosen these arbitrarily.

        let (c_beta, c_beta_orientation) = find_atom_placement(
            c_alpha_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            // Use our info about the previous 2 atoms so we can define the dihedral angle properly.
            // (world space)
            self.χ_1,
            c_alpha,
            n_pos,
            unsafe { CALPHA_R_BOND },
            LEN_SC,
        );

        let (c_gamma, c_gamma_orientation) = find_atom_placement(
            c_beta_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            // Use our info about the previous 2 atoms so we can define the dihedral angle properly.
            // (world space)
            self.χ_2,
            c_beta,
            c_alpha,
            unsafe { TETRA_B },
            LEN_SC,
        );

        let (c_delta, c_delta_orientation) = find_atom_placement(
            c_gamma_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            self.χ_3,
            c_gamma,
            c_beta,
            unsafe { TETRA_B },
            LEN_SC,
        );

        let (c_eps, c_eps_orientation) = find_atom_placement(
            c_delta_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            self.χ_4,
            c_delta,
            c_gamma,
            unsafe { TETRA_B },
            LEN_SC,
        );

        let (n_zeta, n_zeta_orientation) = find_atom_placement(
            c_eps_orientation,
            PLANAR3_A,
            unsafe { PLANAR3_B },
            TAU_DIV2,
            c_eps,
            c_delta,
            unsafe { TETRA_B },
            LEN_SC,
        );

        let (h_n_zeta_a, _) = find_atom_placement(
            n_zeta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            n_zeta,
            c_eps,
            unsafe { PLANAR3_B },
            LEN_N_H,
        );

        let (h_n_zeta_b, _) = find_atom_placement(
            n_zeta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            n_zeta,
            c_eps,
            unsafe { PLANAR3_C },
            LEN_N_H,
        );

        let (h_c_beta_a, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_C },
            LEN_C_H,
        );

        let (h_c_beta_b, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        let (h_c_gamma_a, _) = find_atom_placement(
            c_gamma_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_gamma,
            c_beta,
            unsafe { TETRA_C },
            LEN_C_H,
        );

        let (h_c_gamma_b, _) = find_atom_placement(
            c_gamma_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_gamma,
            c_beta,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        let (h_c_delta_a, _) = find_atom_placement(
            c_delta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_delta,
            c_gamma,
            unsafe { TETRA_C },
            LEN_C_H,
        );

        let (h_c_delta_b, _) = find_atom_placement(
            c_delta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_delta,
            c_gamma,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        let (h_c_eps_a, _) = find_atom_placement(
            c_eps_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_eps,
            c_delta,
            unsafe { TETRA_C },
            LEN_C_H,
        );

        let (h_c_eps_b, _) = find_atom_placement(
            c_eps_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_eps,
            c_delta,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        // todo: Third H?

        CoordsLys {
            c_beta,
            c_gamma,
            c_delta,
            c_eps,
            n_zeta,
            h_n_zeta_a,
            h_n_zeta_b,
            h_c_beta_a,
            h_c_beta_b,
            h_c_gamma_a,
            h_c_gamma_b,
            h_c_delta_a,
            h_c_delta_b,
            h_c_eps_a,
            h_c_eps_b,

            c_beta_orientation,
            c_gamma_orientation,
            c_delta_orientation,
            c_eps_orientation,
            n_zeta_orientation,
        }
    }
}

impl Asp {
    pub fn sidechain_cart_coords(
        &self,
        c_alpha: Vec3,
        c_alpha_orientation: Quaternion,
        n_pos: Vec3,
    ) -> CoordsAsp {
        let (c_beta, c_beta_orientation) = find_atom_placement(
            c_alpha_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            // Use our info about the previous 2 atoms so we can define the dihedral angle properly.
            // (world space)
            self.χ_1,
            c_alpha,
            n_pos,
            unsafe { CALPHA_R_BOND },
            LEN_SC,
        );

        let (c_gamma, c_gamma_orientation) = find_atom_placement(
            c_beta_orientation,
            PLANAR3_A,
            unsafe { PLANAR3_B },
            // Use our info about the previous 2 atoms so we can define the dihedral angle properly.
            // (world space)
            self.χ_2,
            c_beta,
            c_alpha,
            unsafe { TETRA_B },
            LEN_SC,
        );

        let (o_delta1, o_delta1_orientation) = find_atom_placement(
            c_gamma_orientation,
            O_BOND_IN,
            unsafe { O_BOND_OUT },
            TAU_DIV2,
            c_gamma,
            c_beta,
            unsafe { PLANAR3_B },
            LEN_SC,
        );

        let (o_delta2, o_delta2_orientation) = find_atom_placement(
            c_gamma_orientation,
            O_BOND_IN,
            unsafe { O_BOND_OUT },
            TAU_DIV2,
            c_gamma,
            c_beta,
            unsafe { PLANAR3_C },
            LEN_SC,
        );

        let (h_c_beta_a, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_C },
            LEN_C_H,
        );

        let (h_c_beta_b, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        CoordsAsp {
            c_beta,
            c_gamma,
            o_delta1,
            o_delta2,
            h_c_beta_a,
            h_c_beta_b,

            c_beta_orientation,
            c_gamma_orientation,
            o_delta1_orientation,
            o_delta2_orientation,
        }
    }
}

impl Glu {
    pub fn sidechain_cart_coords(
        &self,
        c_alpha: Vec3,
        c_alpha_orientation: Quaternion,
        n_pos: Vec3,
    ) -> CoordsGlu {
        let (c_beta, c_beta_orientation) = find_atom_placement(
            c_alpha_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            // Use our info about the previous 2 atoms so we can define the dihedral angle properly.
            // (world space)
            self.χ_1,
            c_alpha,
            n_pos,
            unsafe { CALPHA_R_BOND },
            LEN_SC,
        );

        let (c_gamma, c_gamma_orientation) = find_atom_placement(
            c_beta_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            // Use our info about the previous 2 atoms so we can define the dihedral angle properly.
            // (world space)
            self.χ_2,
            c_beta,
            c_alpha,
            unsafe { TETRA_B },
            LEN_SC,
        );

        let (c_delta, c_delta_orientation) = find_atom_placement(
            c_gamma_orientation,
            PLANAR3_A,
            unsafe { PLANAR3_B },
            // Use our info about the previous 2 atoms so we can define the dihedral angle properly.
            // (world space)
            self.χ_3,
            c_gamma,
            c_beta,
            unsafe { TETRA_B },
            LEN_SC,
        );

        let (o_eps1, o_eps1_orientation) = find_atom_placement(
            c_delta_orientation,
            O_BOND_IN,
            unsafe { O_BOND_OUT },
            TAU_DIV2,
            c_delta,
            c_gamma,
            unsafe { PLANAR3_B },
            LEN_SC,
        );

        let (o_eps2, o_eps2_orientation) = find_atom_placement(
            c_delta_orientation,
            O_BOND_IN,
            unsafe { O_BOND_OUT },
            TAU_DIV2,
            c_delta,
            c_gamma,
            unsafe { PLANAR3_C },
            LEN_SC,
        );

        let (h_c_beta_a, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_C },
            LEN_C_H,
        );

        let (h_c_beta_b, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        let (h_c_gamma_a, _) = find_atom_placement(
            c_gamma_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_gamma,
            c_beta,
            unsafe { TETRA_C },
            LEN_C_H,
        );

        let (h_c_gamma_b, _) = find_atom_placement(
            c_gamma_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_gamma,
            c_beta,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        CoordsGlu {
            c_beta,
            c_gamma,
            c_delta,
            o_eps1,
            o_eps2,
            h_c_beta_a,
            h_c_beta_b,
            h_c_gamma_a,
            h_c_gamma_b,

            c_beta_orientation,
            c_gamma_orientation,
            c_delta_orientation,
            o_eps1_orientation,
            o_eps2_orientation,
        }
    }
}

impl Ser {
    pub fn sidechain_cart_coords(
        &self,
        c_alpha: Vec3,
        c_alpha_orientation: Quaternion,
        n_pos: Vec3,
    ) -> CoordsSer {
        let (c_beta, c_beta_orientation) = find_atom_placement(
            c_alpha_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            self.χ_1,
            c_alpha,
            n_pos,
            unsafe { CALPHA_R_BOND },
            LEN_SC,
        );

        let (o_gamma, o_gamma_orientation) = find_atom_placement(
            c_beta_orientation,
            O_BOND_IN,
            unsafe { O_BOND_OUT },
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_B },
            LEN_SC,
        );

        let (h_c_beta_a, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_IN,
            H_BOND_IN,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_C },
            LEN_C_H,
        );

        let (h_c_beta_b, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_IN,
            H_BOND_IN,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        let (h_o_gamma, _) = find_atom_placement(
            o_gamma_orientation,
            O_BOND_IN,
            O_BOND_IN,
            TAU_DIV2,
            o_gamma,
            c_beta,
            unsafe { TETRA_B },
            LEN_O_H,
        );

        CoordsSer {
            c_beta,
            o_gamma,
            h_c_beta_a,
            h_c_beta_b,
            h_o_gamma,

            c_beta_orientation,
            o_gamma_orientation,
        }
    }
}

impl Thr {
    pub fn sidechain_cart_coords(
        &self,
        c_alpha: Vec3,
        c_alpha_orientation: Quaternion,
        n_pos: Vec3,
    ) -> CoordsThr {
        let (c_beta, c_beta_orientation) = find_atom_placement(
            c_alpha_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            self.χ_1,
            c_alpha,
            n_pos,
            unsafe { CALPHA_R_BOND },
            LEN_SC,
        );

        let (c_gamma2, c_gamma2_orientation) = find_atom_placement(
            c_beta_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_B },
            LEN_SC,
        );

        let (o_gamma1, o_gamma1_orientation) = find_atom_placement(
            c_beta_orientation,
            O_BOND_IN,
            unsafe { O_BOND_OUT },
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_C },
            LEN_SC,
        );

        let (h_c_beta, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        let (h_o, _) = find_atom_placement(
            o_gamma1_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            o_gamma1,
            c_beta,
            unsafe { O_BOND_OUT },
            LEN_O_H,
        );

        let (h_c_gamma1, _) = find_atom_placement(
            c_gamma2_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_gamma2,
            c_beta,
            unsafe { TETRA_B },
            LEN_C_H,
        );

        let (h_c_gamma2, _) = find_atom_placement(
            c_gamma2_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_gamma2,
            c_beta,
            unsafe { TETRA_C },
            LEN_C_H,
        );

        let (h_c_gamma3, _) = find_atom_placement(
            c_gamma2_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_gamma2,
            c_beta,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        CoordsThr {
            c_beta,
            c_gamma2,
            o_gamma1,
            h_c_beta,
            h_o,
            h_c_gamma1,
            h_c_gamma2,
            h_c_gamma3,

            c_beta_orientation,
            c_gamma2_orientation,
            o_gamma1_orientation,
        }
    }
}

impl Asn {
    pub fn sidechain_cart_coords(
        &self,
        c_alpha: Vec3,
        c_alpha_orientation: Quaternion,
        n_pos: Vec3,
    ) -> CoordsAsn {
        let (c_beta, c_beta_orientation) = find_atom_placement(
            c_alpha_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            self.χ_1,
            c_alpha,
            n_pos,
            unsafe { CALPHA_R_BOND },
            LEN_SC,
        );

        let (c_gamma, c_gamma_orientation) = find_atom_placement(
            c_beta_orientation,
            PLANAR3_A,
            unsafe { PLANAR3_B },
            self.χ_2,
            c_beta,
            c_alpha,
            unsafe { TETRA_B },
            LEN_SC,
        );

        let (o_delta1, _) = find_atom_placement(
            c_gamma_orientation,
            O_BOND_IN,
            unsafe { O_BOND_OUT },
            TAU_DIV2,
            c_gamma,
            c_beta,
            unsafe { PLANAR3_B },
            LEN_SC,
        );

        let (n_delta2, n_delta2_orientation) = find_atom_placement(
            c_gamma_orientation,
            PLANAR3_A,
            unsafe { PLANAR3_B },
            TAU_DIV2,
            c_gamma,
            c_beta,
            unsafe { PLANAR3_C },
            LEN_SC,
        );

        let (h_n_delta_a, _) = find_atom_placement(
            n_delta2_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            n_delta2,
            c_gamma,
            unsafe { PLANAR3_B },
            LEN_N_H,
        );

        let (h_n_delta_b, _) = find_atom_placement(
            n_delta2_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            n_delta2,
            c_gamma,
            unsafe { PLANAR3_C },
            LEN_N_H,
        );

        let (h_c_beta_a, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_C },
            LEN_N_H,
        );

        let (h_c_beta_b, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_D },
            LEN_N_H,
        );

        CoordsAsn {
            c_beta,
            c_gamma,
            o_delta1,
            n_delta2,
            h_n_delta_a,
            h_n_delta_b,
            h_c_beta_a,
            h_c_beta_b,

            c_beta_orientation,
            c_gamma_orientation,
            n_delta2_orientation,
        }
    }
}

impl Gln {
    pub fn sidechain_cart_coords(
        &self,
        c_alpha: Vec3,
        c_alpha_orientation: Quaternion,
        n_pos: Vec3,
    ) -> CoordsGln {
        let (c_beta, c_beta_orientation) = find_atom_placement(
            c_alpha_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            self.χ_1,
            c_alpha,
            n_pos,
            unsafe { CALPHA_R_BOND },
            LEN_SC,
        );

        let (c_gamma, c_gamma_orientation) = find_atom_placement(
            c_beta_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            self.χ_2,
            c_beta,
            c_alpha,
            unsafe { TETRA_B },
            LEN_SC,
        );

        let (c_delta, c_delta_orientation) = find_atom_placement(
            c_gamma_orientation,
            PLANAR3_A,
            unsafe { PLANAR3_B },
            self.χ_3,
            c_gamma,
            c_beta,
            unsafe { PLANAR3_B },
            LEN_SC,
        );

        let (o_eps1, _) = find_atom_placement(
            c_delta_orientation,
            O_BOND_IN,
            unsafe { O_BOND_OUT },
            TAU_DIV2,
            c_delta,
            c_gamma,
            unsafe { PLANAR3_C },
            LEN_SC,
        );

        let (n_eps2, n_eps2_orientation) = find_atom_placement(
            c_delta_orientation,
            PLANAR3_A,
            unsafe { PLANAR3_B },
            TAU_DIV2,
            c_delta,
            c_gamma,
            unsafe { PLANAR3_B },
            LEN_SC,
        );

        let (h_n_eps_a, _) = find_atom_placement(
            n_eps2_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            n_eps2,
            c_delta,
            unsafe { PLANAR3_B },
            LEN_N_H,
        );

        let (h_n_eps_b, _) = find_atom_placement(
            n_eps2_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            n_eps2,
            c_delta,
            unsafe { PLANAR3_C },
            LEN_N_H,
        );

        let (h_c_beta_a, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_C },
            LEN_N_H,
        );

        let (h_c_beta_b, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_D },
            LEN_N_H,
        );

        let (h_c_gamma_a, _) = find_atom_placement(
            c_gamma_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_gamma,
            c_beta,
            unsafe { TETRA_C },
            LEN_N_H,
        );

        let (h_c_gamma_b, _) = find_atom_placement(
            c_gamma_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_gamma,
            c_beta,
            unsafe { TETRA_D },
            LEN_N_H,
        );

        CoordsGln {
            c_beta,
            c_gamma,
            c_delta,
            o_eps1,
            n_eps2,
            h_n_eps_a,
            h_n_eps_b,
            h_c_beta_a,
            h_c_beta_b,
            h_c_gamma_a,
            h_c_gamma_b,

            c_beta_orientation,
            c_gamma_orientation,
            c_delta_orientation,
            n_eps2_orientation,
        }
    }
}

impl Cys {
    pub fn sidechain_cart_coords(
        &self,
        c_alpha: Vec3,
        c_alpha_orientation: Quaternion,
        n_pos: Vec3,
    ) -> CoordsCys {
        let (c_beta, c_beta_orientation) = find_atom_placement(
            c_alpha_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            // Use our info about the previous 2 atoms so we can define the dihedral angle properly.
            // (world space)
            self.χ_1,
            c_alpha,
            n_pos,
            unsafe { CALPHA_R_BOND },
            LEN_SC,
        );

        let (s_gamma, s_gamma_orientation) = find_atom_placement(
            c_beta_orientation,
            // todo: Using O bonds for S here. Is this right?
            O_BOND_IN,
            unsafe { O_BOND_OUT },
            // Use our info about the previous 2 atoms so we can define the dihedral angle properly.
            // (world space)
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_B },
            LEN_SC,
        );

        let (h_c_beta_a, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_C },
            LEN_C_H,
        );

        let (h_c_beta_b, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        let (h_s_gamma, _) = find_atom_placement(
            s_gamma_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            s_gamma,
            c_beta,
            unsafe { O_BOND_OUT }, // todo: For S.
            LEN_C_H,
        );

        CoordsCys {
            c_beta,
            s_gamma,
            h_c_beta_a,
            h_c_beta_b,
            h_s_gamma,

            c_beta_orientation,
            s_gamma_orientation,
        }
    }
}

impl Sec {
    pub fn sidechain_cart_coords(
        &self,
        c_alpha: Vec3,
        c_alpha_orientation: Quaternion,
        n_pos: Vec3,
    ) -> CoordsSec {
        let (c_beta, c_beta_orientation) = find_atom_placement(
            c_alpha_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            // Use our info about the previous 2 atoms so we can define the dihedral angle properly.
            // (world space)
            self.χ_1,
            c_alpha,
            n_pos,
            unsafe { CALPHA_R_BOND },
            LEN_SC,
        );

        let (se_gamma, se_gamma_orientation) = find_atom_placement(
            c_beta_orientation,
            // todo: Using O bonds for S here. Is this right?
            O_BOND_IN,
            unsafe { O_BOND_OUT },
            // Use our info about the previous 2 atoms so we can define the dihedral angle properly.
            // (world space)
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_B },
            LEN_SC,
        );

        let (h_c_beta_a, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_C },
            LEN_C_H,
        );

        let (h_c_beta_b, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        CoordsSec {
            c_beta,
            se_gamma,
            h_c_beta_a,
            h_c_beta_b,

            c_beta_orientation,
            se_gamma_orientation,
        }
    }
}

impl Gly {
    pub fn sidechain_cart_coords(
        &self,
        c_alpha: Vec3,
        c_alpha_orientation: Quaternion,
        n_pos: Vec3,
    ) -> CoordsGly {
        // H on the C alpha.
        let (h, _) = find_atom_placement(
            c_alpha_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_alpha,
            n_pos,
            unsafe { TETRA_C },
            LEN_C_H,
        );

        CoordsGly { h }
    }
}

impl Pro {
    pub fn sidechain_cart_coords(
        &self,
        c_alpha: Vec3,
        c_alpha_orientation: Quaternion,
        n_pos: Vec3,
    ) -> CoordsPro {
        let (c_beta, c_beta_orientation) = find_atom_placement(
            c_alpha_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            0.,
            c_alpha,
            n_pos,
            unsafe { CALPHA_R_BOND },
            LEN_SC,
        );

        let (c_gamma, c_gamma_orientation) = find_atom_placement(
            c_beta_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            0.,
            c_beta,
            c_alpha,
            unsafe { TETRA_B },
            LEN_SC,
        );

        let (c_delta, c_delta_orientation) = find_atom_placement(
            c_gamma_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            0.,
            c_gamma,
            c_beta,
            unsafe { TETRA_B },
            LEN_SC,
        );

        let (h_c_beta_a, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_C },
            LEN_C_H,
        );
        let (h_c_beta_b, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        let (h_c_gamma_a, _) = find_atom_placement(
            c_gamma_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_gamma,
            c_beta,
            unsafe { TETRA_C },
            LEN_C_H,
        );
        let (h_c_gamma_b, _) = find_atom_placement(
            c_gamma_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_gamma,
            c_beta,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        let (h_c_delta_a, _) = find_atom_placement(
            c_delta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_delta,
            c_gamma,
            unsafe { TETRA_C },
            LEN_C_H,
        );
        let (h_c_delta_b, _) = find_atom_placement(
            c_delta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_delta,
            c_gamma,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        CoordsPro {
            c_beta,
            c_gamma,
            c_delta,
            h_c_beta_a,
            h_c_beta_b,
            h_c_gamma_a,
            h_c_gamma_b,
            h_c_delta_a,
            h_c_delta_b,

            c_beta_orientation,
            c_gamma_orientation,
            c_delta_orientation,
        }
    }
}

impl Ala {
    pub fn sidechain_cart_coords(
        &self,
        c_alpha: Vec3,
        c_alpha_orientation: Quaternion,
        n_pos: Vec3,
    ) -> CoordsAla {
        let (c_beta, c_beta_orientation) = find_atom_placement(
            c_alpha_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            TAU_DIV2,
            c_alpha,
            n_pos,
            unsafe { CALPHA_R_BOND },
            LEN_SC,
        );

        let (h_c_beta_a, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_B },
            LEN_C_H,
        );

        let (h_c_beta_b, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_C },
            LEN_C_H,
        );

        let (h_c_beta_c, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        CoordsAla {
            c_beta,
            c_beta_orientation,
            h_c_beta_a,
            h_c_beta_b,
            h_c_beta_c,
        }
    }
}

impl Val {
    pub fn sidechain_cart_coords(
        &self,
        c_alpha: Vec3,
        c_alpha_orientation: Quaternion,
        n_pos: Vec3,
    ) -> CoordsVal {
        let (c_beta, c_beta_orientation) = find_atom_placement(
            c_alpha_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            self.χ_1,
            c_alpha,
            n_pos,
            unsafe { CALPHA_R_BOND },
            LEN_SC,
        );

        let (c_gamma1, c_gamma1_orientation) = find_atom_placement(
            c_beta_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_B },
            LEN_SC,
        );

        let (c_gamma2, c_gamma2_orientation) = find_atom_placement(
            c_beta_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_C },
            LEN_SC,
        );

        let (h_c_beta, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        let (h_c_gamma1_a, _) = find_atom_placement(
            c_gamma1_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_gamma1,
            c_beta,
            unsafe { TETRA_B },
            LEN_C_H,
        );

        let (h_c_gamma1_b, _) = find_atom_placement(
            c_gamma1_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_gamma1,
            c_beta,
            unsafe { TETRA_C },
            LEN_C_H,
        );

        let (h_c_gamma1_c, _) = find_atom_placement(
            c_gamma1_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_gamma1,
            c_beta,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        let (h_c_gamma2_a, _) = find_atom_placement(
            c_gamma2_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_gamma2,
            c_beta,
            unsafe { TETRA_B },
            LEN_C_H,
        );

        let (h_c_gamma2_b, _) = find_atom_placement(
            c_gamma2_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_gamma2,
            c_beta,
            unsafe { TETRA_C },
            LEN_C_H,
        );

        let (h_c_gamma2_c, _) = find_atom_placement(
            c_gamma2_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_gamma2,
            c_beta,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        CoordsVal {
            c_beta,
            c_gamma1,
            c_gamma2,
            h_c_beta,
            h_c_gamma1_a,
            h_c_gamma1_b,
            h_c_gamma1_c,
            h_c_gamma2_a,
            h_c_gamma2_b,
            h_c_gamma2_c,

            c_beta_orientation,
            c_gamma1_orientation,
            c_gamma2_orientation,
        }
    }
}

impl Ile {
    pub fn sidechain_cart_coords(
        &self,
        c_alpha: Vec3,
        c_alpha_orientation: Quaternion,
        n_pos: Vec3,
    ) -> CoordsIle {
        let (c_beta, c_beta_orientation) = find_atom_placement(
            c_alpha_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            self.χ_1,
            c_alpha,
            n_pos,
            unsafe { CALPHA_R_BOND },
            LEN_SC,
        );

        let (c_gamma1, c_gamma1_orientation) = find_atom_placement(
            // Non-continuing chain
            c_beta_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_C },
            LEN_SC,
        );

        let (c_gamma2, c_gamma2_orientation) = find_atom_placement(
            c_beta_orientation,
            TETRA_A,
            unsafe { TETRA_B }, // todo?
            self.χ_2,
            c_beta,
            c_alpha,
            unsafe { TETRA_B },
            LEN_SC,
        );

        let (c_delta, c_delta_orientation) = find_atom_placement(
            c_gamma2_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            TAU_DIV2,
            c_gamma2,
            c_beta,
            unsafe { TETRA_B },
            LEN_SC,
        );

        let (h_c_beta, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        let (h_c_gamma1_a, _) = find_atom_placement(
            c_gamma1_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_gamma1,
            c_beta,
            unsafe { TETRA_B },
            LEN_C_H,
        );

        let (h_c_gamma1_b, _) = find_atom_placement(
            c_gamma1_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_gamma1,
            c_beta,
            unsafe { TETRA_C },
            LEN_C_H,
        );

        let (h_c_gamma1_c, _) = find_atom_placement(
            c_gamma1_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_gamma1,
            c_beta,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        let (h_c_gamma2_a, _) = find_atom_placement(
            c_gamma2_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_gamma2,
            c_beta,
            unsafe { TETRA_C },
            LEN_C_H,
        );

        let (h_c_gamma2_b, _) = find_atom_placement(
            c_gamma2_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_gamma2,
            c_beta,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        let (h_c_delta_a, _) = find_atom_placement(
            c_delta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_delta,
            c_gamma2,
            unsafe { TETRA_B },
            LEN_C_H,
        );

        let (h_c_delta_b, _) = find_atom_placement(
            c_delta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_delta,
            c_gamma2,
            unsafe { TETRA_C },
            LEN_C_H,
        );

        let (h_c_delta_c, _) = find_atom_placement(
            c_delta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_delta,
            c_gamma2,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        CoordsIle {
            c_beta,
            c_gamma1,
            c_gamma2,
            c_delta, // off gamma2
            h_c_beta,
            h_c_gamma1_a,
            h_c_gamma1_b,
            h_c_gamma1_c,
            h_c_gamma2_a,
            h_c_gamma2_b,
            h_c_delta_a,
            h_c_delta_b,
            h_c_delta_c,

            c_beta_orientation,
            c_gamma1_orientation,
            c_gamma2_orientation,
            c_delta_orientation,
        }
    }
}

impl Leu {
    pub fn sidechain_cart_coords(
        &self,
        c_alpha: Vec3,
        c_alpha_orientation: Quaternion,
        n_pos: Vec3,
    ) -> CoordsLeu {
        let (c_beta, c_beta_orientation) = find_atom_placement(
            c_alpha_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            self.χ_1,
            c_alpha,
            n_pos,
            unsafe { CALPHA_R_BOND },
            LEN_SC,
        );

        let (c_gamma, c_gamma_orientation) = find_atom_placement(
            c_beta_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            self.χ_2,
            c_beta,
            c_alpha,
            unsafe { TETRA_B },
            LEN_SC,
        );

        let (c_delta1, c_delta1_orientation) = find_atom_placement(
            c_gamma_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            TAU_DIV2,
            c_gamma,
            c_beta,
            unsafe { TETRA_B },
            LEN_SC,
        );

        let (c_delta2, c_delta2_orientation) = find_atom_placement(
            c_gamma_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            TAU_DIV2,
            c_gamma,
            c_beta,
            unsafe { TETRA_C },
            LEN_SC,
        );

        let (h_c_beta_a, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_C },
            LEN_C_H,
        );

        let (h_c_beta_b, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        let (h_c_gamma, _) = find_atom_placement(
            c_gamma_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_gamma,
            c_beta,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        let (h_c_delta1_a, _) = find_atom_placement(
            c_delta1_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_delta1,
            c_gamma,
            unsafe { TETRA_B },
            LEN_C_H,
        );

        let (h_c_delta1_b, _) = find_atom_placement(
            c_delta1_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_delta1,
            c_gamma,
            unsafe { TETRA_C },
            LEN_C_H,
        );

        let (h_c_delta1_c, _) = find_atom_placement(
            c_delta1_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_delta1,
            c_gamma,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        let (h_c_delta2_a, _) = find_atom_placement(
            c_delta1_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_delta2,
            c_gamma,
            unsafe { TETRA_B },
            LEN_C_H,
        );

        let (h_c_delta2_b, _) = find_atom_placement(
            c_delta2_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_delta2,
            c_gamma,
            unsafe { TETRA_C },
            LEN_C_H,
        );

        let (h_c_delta2_c, _) = find_atom_placement(
            c_delta2_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_delta2,
            c_gamma,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        CoordsLeu {
            c_beta,
            c_gamma,
            c_delta1,
            c_delta2,
            h_c_beta_a,
            h_c_beta_b,
            h_c_gamma,
            h_c_delta1_a,
            h_c_delta1_b,
            h_c_delta1_c,
            h_c_delta2_a,
            h_c_delta2_b,
            h_c_delta2_c,

            c_beta_orientation,
            c_gamma_orientation,
            c_delta1_orientation,
            c_delta2_orientation,
        }
    }
}

impl Met {
    pub fn sidechain_cart_coords(
        &self,
        c_alpha: Vec3,
        c_alpha_orientation: Quaternion,
        n_pos: Vec3,
    ) -> CoordsMet {
        let (c_beta, c_beta_orientation) = find_atom_placement(
            c_alpha_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            // Use our info about the previous 2 atoms so we can define the dihedral angle properly.
            // (world space)
            self.χ_1,
            c_alpha,
            n_pos,
            unsafe { CALPHA_R_BOND },
            LEN_SC,
        );

        let (c_gamma, c_gamma_orientation) = find_atom_placement(
            c_beta_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            self.χ_2,
            c_beta,
            c_alpha,
            unsafe { TETRA_B },
            LEN_SC,
        );

        let (s_delta, s_delta_orientation) = find_atom_placement(
            c_gamma_orientation,
            // todo: Is this right for s? Tetra, line, or what?
            PLANAR3_A,            // QC this for S
            unsafe { PLANAR3_B }, // QC this for S
            self.χ_3,
            c_gamma,
            c_beta,
            unsafe { TETRA_B },
            LEN_SC,
        );

        let (c_eps, c_eps_orientation) = find_atom_placement(
            s_delta_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            TAU_DIV2,
            s_delta,
            c_gamma,
            unsafe { PLANAR3_B }, // QC this for S
            LEN_SC,
        );

        let (h_c_beta_a, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_C },
            LEN_C_H,
        );

        let (h_c_beta_b, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        let (h_c_gamma_a, _) = find_atom_placement(
            c_gamma_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_gamma,
            c_beta,
            unsafe { TETRA_C },
            LEN_C_H,
        );

        let (h_c_gamma_b, _) = find_atom_placement(
            c_gamma_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_gamma,
            c_beta,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        let (h_c_eps_a, _) = find_atom_placement(
            c_eps_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_eps,
            s_delta,
            unsafe { TETRA_B },
            LEN_C_H,
        );

        let (h_c_eps_b, _) = find_atom_placement(
            c_eps_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_eps,
            s_delta,
            unsafe { TETRA_C },
            LEN_C_H,
        );

        let (h_c_eps_c, _) = find_atom_placement(
            c_eps_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_eps,
            s_delta,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        CoordsMet {
            c_beta,
            c_gamma,
            s_delta,
            c_eps,
            h_c_beta_a,
            h_c_beta_b,
            h_c_gamma_a,
            h_c_gamma_b,
            h_c_eps_a,
            h_c_eps_b,
            h_c_eps_c,

            c_beta_orientation,
            c_gamma_orientation,
            s_delta_orientation,
            c_eps_orientation,
        }
    }
}

impl Phe {
    pub fn sidechain_cart_coords(
        &self,
        c_alpha: Vec3,
        c_alpha_orientation: Quaternion,
        n_pos: Vec3,
    ) -> CoordsPhe {
        // todo: I think the RING6s you use here are equiv to the `PLANAR3` bonds
        let (c_beta, c_beta_orientation) = find_atom_placement(
            c_alpha_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            self.χ_1,
            c_alpha,
            n_pos,
            unsafe { CALPHA_R_BOND },
            LEN_SC,
        );

        let (c_gamma, c_gamma_orientation) = find_atom_placement(
            c_beta_orientation,
            PLANAR3_A,
            unsafe { PLANAR3_B },
            self.χ_2,
            c_beta,
            c_alpha,
            unsafe { TETRA_C },
            LEN_SC,
        );

        let (c_delta1, c_delta1_orientation) = find_atom_placement(
            c_gamma_orientation,
            PLANAR3_A,
            unsafe { PLANAR3_B },
            TAU_DIV2,
            c_gamma,
            c_beta,
            unsafe { PLANAR3_B },
            LEN_SC,
        );

        let (c_delta2, c_delta2_orientation) = find_atom_placement(
            c_gamma_orientation,
            PLANAR3_A,
            unsafe { PLANAR3_B },
            TAU_DIV2,
            c_gamma,
            c_beta,
            unsafe { PLANAR3_C },
            LEN_SC,
        );

        let (c_eps1, c_eps1_orientation) = find_atom_placement(
            c_delta1_orientation,
            PLANAR3_A,
            unsafe { PLANAR3_B },
            0.,
            c_delta1,
            c_gamma,
            unsafe { PLANAR3_B },
            LEN_SC,
        );

        let (c_eps2, c_eps2_orientation) = find_atom_placement(
            c_delta2_orientation,
            PLANAR3_A,
            unsafe { PLANAR3_B },
            0.,
            c_delta2,
            c_gamma,
            unsafe { PLANAR3_B },
            LEN_SC,
        );

        // We anchor c_zeta off eps1.
        let (c_zeta, c_zeta_orientation) = find_atom_placement(
            c_eps1_orientation,
            PLANAR3_A,
            unsafe { PLANAR3_B },
            TAU_DIV2,
            c_eps1,
            c_delta1,
            unsafe { PLANAR3_B },
            LEN_SC,
        );

        let (h_c_beta_a, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_B }, // Non-standard B vice C here.
            LEN_C_H,
        );

        let (h_c_beta_b, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        let (h_c_delta1, _) = find_atom_placement(
            c_delta1_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_delta1,
            c_gamma,
            unsafe { PLANAR3_C },
            LEN_C_H,
        );

        let (h_c_delta2, _) = find_atom_placement(
            c_delta2_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_delta2,
            c_gamma,
            unsafe { PLANAR3_C },
            LEN_C_H,
        );

        let (h_c_eps1, _) = find_atom_placement(
            c_eps1_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_eps1,
            c_delta1,
            unsafe { PLANAR3_C },
            LEN_C_H,
        );

        let (h_c_eps2, _) = find_atom_placement(
            c_eps2_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_eps2,
            c_delta2,
            unsafe { PLANAR3_C },
            LEN_C_H,
        );

        let (h_c_zeta, _) = find_atom_placement(
            c_zeta_orientation,
            H_BOND_IN,
            H_BOND_OUT,
            TAU_DIV2,
            c_zeta,
            c_eps2,
            unsafe { PLANAR3_B },
            LEN_C_H,
        );

        CoordsPhe {
            c_beta,
            c_gamma,
            c_delta1,
            c_delta2,
            c_eps1,
            c_eps2,
            c_zeta,
            h_c_beta_a,
            h_c_beta_b,
            h_c_delta1,
            h_c_delta2,
            h_c_eps1,
            h_c_eps2,
            h_c_zeta,

            c_beta_orientation,
            c_gamma_orientation,
            c_delta1_orientation,
            c_delta2_orientation,
            c_eps1_orientation,
            c_eps2_orientation,
            c_zeta_orientation,
        }
    }
}

impl Tyr {
    pub fn sidechain_cart_coords(
        &self,
        c_alpha: Vec3,
        c_alpha_orientation: Quaternion,
        n_pos: Vec3,
    ) -> CoordsTyr {
        let (c_beta, c_beta_orientation) = find_atom_placement(
            c_alpha_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            self.χ_1,
            c_alpha,
            n_pos,
            unsafe { CALPHA_R_BOND },
            LEN_SC,
        );

        let (c_gamma, c_gamma_orientation) = find_atom_placement(
            c_beta_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            self.χ_2,
            c_beta,
            c_alpha,
            unsafe { TETRA_B },
            LEN_SC,
        );

        let (c_delta1, c_delta1_orientation) = find_atom_placement(
            c_gamma_orientation,
            PLANAR3_A,
            unsafe { PLANAR3_B },
            TAU_DIV2,
            c_gamma,
            c_beta,
            unsafe { PLANAR3_B },
            LEN_SC,
        );

        let (c_delta2, c_delta2_orientation) = find_atom_placement(
            c_gamma_orientation,
            PLANAR3_A,
            unsafe { PLANAR3_B },
            TAU_DIV2,
            c_gamma,
            c_beta,
            unsafe { PLANAR3_C },
            LEN_SC,
        );

        let (c_eps1, c_eps1_orientation) = find_atom_placement(
            c_delta1_orientation,
            PLANAR3_A,
            unsafe { PLANAR3_B },
            0.,
            c_delta1,
            c_gamma,
            unsafe { PLANAR3_B },
            LEN_SC,
        );

        let (c_eps2, c_eps2_orientation) = find_atom_placement(
            c_delta2_orientation,
            PLANAR3_A,
            unsafe { PLANAR3_B },
            0.,
            c_delta2,
            c_gamma,
            unsafe { PLANAR3_B },
            LEN_SC,
        );

        // We anchor c_zeta off eps1.
        let (c_zeta, c_zeta_orientation) = find_atom_placement(
            c_eps1_orientation,
            PLANAR3_A,
            unsafe { PLANAR3_B },
            TAU_DIV2,
            c_eps1,
            c_delta1,
            unsafe { PLANAR3_B },
            LEN_SC,
        );

        let (o_eta, o_eta_orientation) = find_atom_placement(
            c_zeta_orientation,
            O_BOND_IN,
            unsafe { O_BOND_OUT },
            TAU_DIV2,
            c_zeta,
            c_eps2,
            unsafe { PLANAR3_B },
            LEN_SC,
        );

        let (h_c_beta_a, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_OUT,
            H_BOND_IN,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_C },
            LEN_C_H,
        );

        let (h_c_beta_b, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_OUT,
            H_BOND_IN,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        let (h_c_delta1, _) = find_atom_placement(
            c_delta1_orientation,
            H_BOND_OUT,
            H_BOND_IN,
            TAU_DIV2,
            c_delta1,
            c_gamma,
            unsafe { PLANAR3_C },
            LEN_C_H,
        );

        let (h_c_delta2, _) = find_atom_placement(
            c_delta2_orientation,
            H_BOND_OUT,
            H_BOND_IN,
            TAU_DIV2,
            c_delta2,
            c_gamma,
            unsafe { PLANAR3_C },
            LEN_C_H,
        );

        let (h_c_eps1, _) = find_atom_placement(
            c_eps1_orientation,
            H_BOND_OUT,
            H_BOND_IN,
            TAU_DIV2,
            c_eps1,
            c_delta1,
            unsafe { PLANAR3_C },
            LEN_C_H,
        );

        let (h_c_eps2, _) = find_atom_placement(
            c_eps2_orientation,
            H_BOND_OUT,
            H_BOND_IN,
            TAU_DIV2,
            c_eps2,
            c_delta2,
            unsafe { PLANAR3_C },
            LEN_C_H,
        );

        let (h_o_eta, _) = find_atom_placement(
            o_eta_orientation,
            O_BOND_IN,
            O_BOND_IN,
            TAU_DIV2,
            o_eta,
            c_zeta,
            unsafe { PLANAR3_C },
            LEN_O_H,
        );

        CoordsTyr {
            c_beta,
            c_gamma,
            c_delta1,
            c_delta2,
            c_eps1,
            c_eps2,
            c_zeta,
            o_eta,
            h_c_beta_a,
            h_c_beta_b,
            h_c_delta1,
            h_c_delta2,
            h_c_eps1,
            h_c_eps2,
            h_o_eta,

            c_beta_orientation,
            c_gamma_orientation,
            c_delta1_orientation,
            c_delta2_orientation,
            c_eps1_orientation,
            c_eps2_orientation,
            c_zeta_orientation,
            o_eta_orientation,
        }
    }
}

impl Trp {
    pub fn sidechain_cart_coords(
        &self,
        c_alpha: Vec3,
        c_alpha_orientation: Quaternion,
        n_pos: Vec3,
    ) -> CoordsTrp {
        let (c_beta, c_beta_orientation) = find_atom_placement(
            c_alpha_orientation,
            TETRA_A,
            unsafe { TETRA_B },
            self.χ_1,
            c_alpha,
            n_pos,
            unsafe { CALPHA_R_BOND },
            LEN_SC,
        );

        let (c_gamma, c_gamma_orientation) = find_atom_placement(
            c_beta_orientation,
            TETRA_A,
            unsafe { RING5_BOND_OUT },
            self.χ_2,
            c_beta,
            c_alpha,
            unsafe { TETRA_B },
            LEN_SC,
        );

        let (c_delta, c_delta_orientation) = find_atom_placement(
            c_gamma_orientation,
            RING_BOND_IN,
            unsafe { RING5_BOND_OUT },
            TAU_DIV2,
            c_gamma,
            c_beta,
            unsafe { RING5_BOND_OUT },
            LEN_SC,
        );

        let (n_eps, n_eps_orientation) = find_atom_placement(
            c_delta_orientation,
            RING_BOND_IN,
            unsafe { RING5_BOND_OUT },
            0.,
            c_delta,
            c_gamma,
            unsafe { RING5_BOND_OUT },
            LEN_SC,
        );

        // Between rings
        let (c_zeta, c_zeta_orientation) = find_atom_placement(
            n_eps_orientation,
            RING_BOND_IN,
            unsafe { RING5_BOND_OUT },
            0.,
            n_eps,
            c_delta,
            unsafe { RING5_BOND_OUT },
            LEN_SC,
        );

        // Between rings
        let (c_eta, c_eta_orientation) = find_atom_placement(
            c_zeta_orientation,
            RING_BOND_IN,
            unsafe { PLANAR3_B },
            TAU_DIV2,
            c_zeta,
            n_eps,
            unsafe { RING5_BOND_OUT },
            LEN_SC,
        );

        let (c_theta, c_theta_orientation) = find_atom_placement(
            c_eta_orientation,
            PLANAR3_A,
            unsafe { PLANAR3_B },
            0.,
            c_eta,
            c_zeta,
            unsafe { PLANAR3_B },
            LEN_SC,
        );

        let (c_iota, c_iota_orientation) = find_atom_placement(
            c_theta_orientation,
            PLANAR3_A,
            unsafe { PLANAR3_B },
            0.,
            c_theta,
            c_eta,
            unsafe { PLANAR3_B },
            LEN_SC,
        );

        let (c_kappa, c_kappa_orientation) = find_atom_placement(
            c_iota_orientation,
            PLANAR3_A,
            unsafe { PLANAR3_B },
            0.,
            c_iota,
            c_theta,
            unsafe { PLANAR3_B },
            LEN_SC,
        );

        let (c_lambda, c_lambda_orientation) = find_atom_placement(
            c_kappa_orientation,
            PLANAR3_A,
            unsafe { PLANAR3_B },
            TAU_DIV2,
            c_kappa,
            c_iota,
            unsafe { PLANAR3_B },
            LEN_SC,
        );

        let (h_c_beta_a, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_OUT,
            H_BOND_IN,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_C },
            LEN_C_H,
        );

        let (h_c_beta_b, _) = find_atom_placement(
            c_beta_orientation,
            H_BOND_OUT,
            H_BOND_IN,
            TAU_DIV2,
            c_beta,
            c_alpha,
            unsafe { TETRA_D },
            LEN_C_H,
        );

        let (h_c_delta, _) = find_atom_placement(
            c_delta_orientation,
            H_BOND_OUT,
            H_BOND_IN,
            TAU_DIV2,
            c_delta,
            c_gamma,
            unsafe { PLANAR3_C },
            LEN_C_H,
        );

        let (h_n_eps, _) = find_atom_placement(
            n_eps_orientation,
            H_BOND_OUT,
            H_BOND_IN,
            TAU_DIV2,
            n_eps,
            c_delta,
            unsafe { PLANAR3_C },
            LEN_C_H,
        );
        let (h_c_theta, _) = find_atom_placement(
            c_theta_orientation,
            H_BOND_OUT,
            H_BOND_IN,
            TAU_DIV2,
            c_theta,
            c_eta,
            unsafe { PLANAR3_C },
            LEN_C_H,
        );
        let (h_c_iota, _) = find_atom_placement(
            c_iota_orientation,
            H_BOND_OUT,
            H_BOND_IN,
            TAU_DIV2,
            c_iota,
            c_theta,
            unsafe { PLANAR3_C },
            LEN_C_H,
        );
        let (h_c_kappa, _) = find_atom_placement(
            c_kappa_orientation,
            H_BOND_OUT,
            H_BOND_IN,
            TAU_DIV2,
            c_kappa,
            c_iota,
            unsafe { PLANAR3_C },
            LEN_C_H,
        );
        let (h_c_lambda, _) = find_atom_placement(
            c_lambda_orientation,
            H_BOND_OUT,
            H_BOND_IN,
            TAU_DIV2,
            c_lambda,
            c_kappa,
            unsafe { PLANAR3_B },
            LEN_C_H,
        );

        CoordsTrp {
            c_beta,
            c_gamma,
            c_delta,
            n_eps,
            c_zeta,
            c_eta,
            c_theta,
            c_iota,
            c_kappa,
            c_lambda,
            h_c_beta_a,
            h_c_beta_b,
            h_c_delta,
            h_n_eps,
            h_c_theta,
            h_c_iota,
            h_c_kappa,
            h_c_lambda,

            c_beta_orientation,
            c_gamma_orientation,
            c_delta_orientation,
            n_eps_orientation,
            c_zeta_orientation,
            c_eta_orientation,
            c_theta_orientation,
            c_iota_orientation,
            c_kappa_orientation,
            c_lambda_orientation,
        }
    }
}
