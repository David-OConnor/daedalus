use std::{collections::HashMap, io, io::ErrorKind};

use Element::*;
use pdbtbx::Element::Ni;

#[derive(Clone, Copy, PartialEq, Debug, Hash, Eq)]
pub enum Element {
    Hydrogen,
    Carbon,
    Oxygen,
    Nitrogen,
    Fluorine,
    Sulfur,
    Phosphorus,
    Iron,
    Copper,
    Calcium,
    Potassium,
    Aluminum,
    Lead,
    Gold,
    Silver,
    Mercury,
    Tin,
    Zinc,
    Magnesium,
    Manganese,
    Iodine,
    Chlorine,
    Tungsten,
    Tellurium,
    Selenium,
    Bromine,
    Other,
}

impl Element {
    pub fn valence_typical(&self) -> usize {
        match self {
            Hydrogen => 1,
            Carbon => 4,
            Oxygen => 2,
            Nitrogen => 3,
            Fluorine => 1,
            Sulfur => 2,     // can be 2, 4, or 6, but 2 is a common choice
            Phosphorus => 5, // can be 3 or 5, here we pick 5
            Iron => 2,       // Fe(II) is common (Fe(III) also common)
            Copper => 2,     // Cu(I) and Cu(II) both occur, pick 2 as a naive default
            Calcium => 2,
            Potassium => 1,
            Aluminum => 3,
            Lead => 2,    // Pb(II) or Pb(IV), but Pb(II) is more common/stable
            Gold => 3,    // Au(I) and Au(III) are common, pick 3
            Silver => 1,  // Ag(I) is most common
            Mercury => 2, // Hg(I) and Hg(II), pick 2
            Tin => 4,     // Sn(II) or Sn(IV), pick 4
            Zinc => 2,
            Magnesium => 2,
            Manganese => 7, // todo: Not sure
            Iodine => 1,    // can have higher, but 1 is typical in many simple compounds
            Chlorine => 1,  // can also be 3,5,7, but 1 is the simplest (e.g., HCl)
            Tungsten => 6,  // W can have multiple but 6 is a common oxidation state
            Tellurium => 2, // can also be 4 or 6, pick 2
            Selenium => 2,  // can also be 4 or 6, pick 2
            Bromine => 7,
            Other => 0, // default to 0 for unknown or unhandled elements
        }
    }

    // fn gasteiger_electronegativity(&self) -> f32 {
    //     match self {
    //         Element::Hydrogen => 2.20,
    //         Element::Carbon => 2.55,
    //         Element::Oxygen => 3.44,
    //         Element::Nitrogen => 3.04,
    //         Element::Fluorine => 3.98,
    //         Element::Sulfur => 2.58,
    //         Element::Phosphorus => 2.19,
    //         Element::Iron => 1.83,
    //         Element::Copper => 1.90,
    //         Element::Calcium => 1.00,
    //         Element::Potassium => 0.82,
    //         Element::Aluminum => 1.61,
    //         Element::Lead => 2.33,
    //         Element::Gold => 2.54,
    //         Element::Silver => 1.93,
    //         Element::Mercury => 2.00,
    //         Element::Tin => 1.96,
    //         Element::Zinc => 1.65,
    //         Element::Magnesium => 1.31,
    //         Element::Iodine => 2.66,
    //         Element::Chlorine => 3.16,
    //         Element::Tungsten => 2.36,
    //         Element::Tellurium => 2.10,
    //         Element::Selenium => 2.55,
    //         Element::Other => {
    //             eprintln!(
    //                 "Error: Attempting to get a Gasteiger electronegativity for an unknown element."
    //             );
    //             0.0
    //         }
    //     }
    // }

    pub fn from_pdb(el: Option<&pdbtbx::Element>) -> Self {
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

                _ => {
                    eprintln!("Unknown element: {e:?}");
                    Self::Other
                }
            }
        } else {
            // todo?
            Self::Other
        }
    }

    pub fn from_letter(letter: &str) -> io::Result<Self> {
        match letter.to_uppercase().as_ref() {
            "H" => Ok(Hydrogen),
            "C" => Ok(Carbon),
            "O" => Ok(Oxygen),
            "N" => Ok(Nitrogen),
            "F" => Ok(Fluorine),
            "S" => Ok(Sulfur),
            "P" => Ok(Phosphorus),
            "FE" => Ok(Iron),
            "CU" => Ok(Copper),
            "CA" => Ok(Calcium),
            "K" => Ok(Potassium),
            "AL" => Ok(Aluminum),
            "PB" => Ok(Lead),
            "AU" => Ok(Gold),
            "AG" => Ok(Silver),
            "HG" => Ok(Mercury),
            "SN" => Ok(Tin),
            "ZN" => Ok(Zinc),
            "MG" => Ok(Magnesium),
            "MN" => Ok(Manganese),
            "I" => Ok(Iodine),
            "CL" => Ok(Chlorine),
            "W" => Ok(Tungsten),
            "TE" => Ok(Tellurium),
            "SE" => Ok(Selenium),
            "BR" => Ok(Bromine),
            // todo: Fill in if you need, or remove this fn.
            _ => Err(io::Error::new(
                ErrorKind::InvalidData,
                "Invalid atom letter",
            )),
        }
    }

    pub fn to_letter(&self) -> String {
        match self {
            Hydrogen => "H".into(),
            Carbon => "C".into(),
            Oxygen => "O".into(),
            Nitrogen => "N".into(),
            Fluorine => "F".into(),
            Sulfur => "S".into(),
            Phosphorus => "P".into(),
            Iron => "FE".into(),
            Copper => "CU".into(),
            Calcium => "CA".into(),
            Potassium => "K".into(),
            Aluminum => "AL".into(),
            Lead => "PB".into(),
            Gold => "AU".into(),
            Silver => "AG".into(),
            Mercury => "HG".into(),
            Tin => "SN".into(),
            Zinc => "ZN".into(),
            Magnesium => "MG".into(),
            Manganese => "MN".into(),
            Iodine => "I".into(),
            Chlorine => "CL".into(),
            Tungsten => "W".into(),
            Tellurium => "TE".into(),
            Selenium => "SE".into(),
            Bromine => "Br".into(),
            Other => "X".into(),
        }
    }

    /// From [PyMol](https://pymolwiki.org/index.php/Color_Values)
    pub fn color(&self) -> (f32, f32, f32) {
        match self {
            Hydrogen => (0.9, 0.9, 0.9),
            Carbon => (0.2, 1., 0.2),
            Oxygen => (1., 0.3, 0.3),
            Nitrogen => (0.2, 0.2, 1.0),
            Fluorine => (0.701, 1.0, 1.0),
            Sulfur => (0.9, 0.775, 0.25),
            Phosphorus => (1.0, 0.502, 0.),
            Iron => (0.878, 0.4, 0.2),
            Copper => (0.784, 0.502, 0.2),
            Calcium => (0.239, 1.0, 0.),
            Potassium => (0.561, 0.251, 0.831),
            Aluminum => (0.749, 0.651, 0.651),
            Lead => (0.341, 0.349, 0.380),
            Gold => (1., 0.820, 0.137),
            Silver => (0.753, 0.753, 0.753),
            Mercury => (0.722, 0.722, 0.816),
            Tin => (0.4, 0.502, 0.502),
            Zinc => (0.490, 0.502, 0.690),
            Magnesium => (0.541, 1., 0.),
            Manganese => (0.541, 1., 0.541),
            Iodine => (0.580, 0., 0.580),
            Chlorine => (0.121, 0.941, 0.121),
            Tungsten => (0.129, 0.580, 0.840),
            Tellurium => (0.831, 0.478, 0.),
            Selenium => (1.0, 0.631, 0.),
            Bromine => (1.0, 0.99, 0.),
            Other => (5., 5., 5.),
        }
    }

    #[rustfmt::skip]
    /// Covalent radius, in angstrom.
    /// https://github.com/openbabel/openbabel/blob/master/src/elementtable.h
    /// https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)
    pub fn covalent_radius(self) -> f64 {
        match self {
            Hydrogen   => 0.31,
            Carbon     => 0.76,
            Oxygen     => 0.66,
            Nitrogen   => 0.71,
            Fluorine   => 0.57,
            Sulfur     => 1.05,
            Phosphorus => 1.07,
            Iron       => 1.32,
            Copper     => 1.32,
            Calcium    => 1.76,
            Potassium  => 2.03,
            Aluminum   => 1.21,
            Lead       => 1.46,
            Gold       => 1.36,
            Silver     => 1.45,
            Mercury    => 1.32,
            Tin        => 1.39,
            Zinc       => 1.22,
            Magnesium  => 1.41, // 1.19?
            Manganese  => 1.39,
            Iodine     => 1.39,
            Chlorine   => 1.02,
            Tungsten   => 1.62,
            Tellurium  => 1.38,
            Selenium   => 1.20,
            Bromine  => 1.14, // 1.14 - 1.20
            Other      => 0.00,
        }
    }

    #[rustfmt::skip]
    /// Van-der-wals radius, in angstrom.
    /// https://github.com/openbabel/openbabel/blob/master/src/elementtable.h
    /// https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)
    pub const fn vdw_radius(&self) -> f32 {
        match self {
            Hydrogen   => 1.10, // or 120
            Carbon     => 1.70,
            Oxygen     => 1.52,
            Nitrogen   => 1.55,
            Fluorine   => 1.47,
            Sulfur     => 1.80,
            Phosphorus => 1.80,
            Iron       => 2.05,
            Copper     => 2.00,
            Calcium    => 2.31,
            Potassium  => 2.75,
            Aluminum   => 1.84,
            Lead       => 2.02,
            Gold       => 2.10,
            Silver     => 2.10,
            Mercury    => 2.05,
            Tin        => 1.93,
            Zinc       => 2.10,
            Magnesium  => 1.73,
            Manganese  => 0., // N/A?
            Iodine     => 1.98,
            Chlorine   => 1.75,
            Tungsten   => 2.10,
            Tellurium  => 2.06,
            Selenium   => 1.90,
            Bromine   => 1.85,
            Other      => 0.0,
        }
    }

    /// Returns approximate Lennard-Jones parameters (\sigma, \epsilon) in Å and kJ/mol.
    /// These are *not* real force-field values, just a demonstration.
    pub fn lj_params(self) -> (f32, f32) {
        // For demonstration, we compute sigma from the van der Waals radius
        //   sigma = (2 * vdw_radius) / 2^(1/6).
        // Then guess epsilon from a trivial rule or store a small table.
        // Real simulations typically get these from standard force fields!

        let r_vdw = self.vdw_radius(); // in Å
        // Avoid zero or negative vdw radius
        let r_vdw = if r_vdw <= 0.0 { 1.5 } else { r_vdw };

        // Sigma from naive formula:
        let sigma = (2.0 * r_vdw) / (2_f32.powf(1.0 / 6.0));

        // A naive guess for epsilon
        // (In reality, you’d store carefully fit data or use a better heuristic.)
        // For example, heavier elements get a bigger well depth:
        let approximate_atomic_number = match self {
            Hydrogen => 1,
            Carbon => 6,
            Nitrogen => 7,
            Oxygen => 8,
            Fluorine => 9,
            Sulfur => 16,
            Phosphorus => 15,
            Iron => 26,
            Copper => 29,
            Calcium => 20,
            Potassium => 19,
            Aluminum => 13,
            Lead => 82,
            Gold => 79,
            Silver => 47,
            Mercury => 80,
            Tin => 50,
            Zinc => 30,
            Magnesium => 12,
            Manganese => 25,
            Iodine => 53,
            Chlorine => 17,
            Tungsten => 74,
            Tellurium => 52,
            Selenium => 34,
            Bromine => 35,
            Other => 20, // fallback
        };

        // Pretend epsilon in kJ/mol is something like 0.01 * Z^(0.7)
        let epsilon = 0.01 * (approximate_atomic_number as f32).powf(0.7);

        (sigma, epsilon)
    }
}

fn get_lj_params_inner(el_0: Element, el_1: Element) -> (f32, f32) {
    let (sigma_a, epsilon_a) = el_0.lj_params();
    let (sigma_b, epsilon_b) = el_1.lj_params();
    let sigma = 0.5 * (sigma_a + sigma_b);
    let epsilon = (epsilon_a * epsilon_b).sqrt();

    (sigma, epsilon)
}

/// Get Lennard-Jones potential parameters (Sigma, Epsilon), given two elements.
/// This is essentially a partial LUT. Lorentz-Berthelot combining rule.
///
/// Note: This is a loose approximation.
pub fn get_lj_params(
    el_0: Element,
    el_1: Element,
    lut: &HashMap<(Element, Element), (f32, f32)>,
) -> (f32, f32) {
    // Order doesn't matter.
    if let Some((sigma, epsilon)) = lut.get(&(el_0, el_1)) {
        return (*sigma, *epsilon);
    }
    if let Some((sigma, epsilon)) = lut.get(&(el_1, el_0)) {
        return (*sigma, *epsilon);
    }

    println!("Fallthrough on LJ parans. Els {el_0:?} and {el_1:?}");
    get_lj_params_inner(el_0, el_1)
}

/// Note: Order invariant; insert one for each element pair.
pub fn init_lj_lut() -> HashMap<(Element, Element), (f32, f32)> {
    let mut result = HashMap::new();

    let els = vec![
        Carbon, Hydrogen, Nitrogen, Oxygen, Sulfur, Fluorine, Chlorine,
    ];

    // todo: Prevent duplicates.
    for (i, el_0) in els.iter().enumerate() {
        for (j, el_1) in els.iter().enumerate() {
            // Note: This will allow duplicate orders for same-el combos; acceptible.
            if i <= j {
                result.insert((*el_0, *el_1), get_lj_params_inner(*el_0, *el_1));
            }
        }
    }

    result
}
