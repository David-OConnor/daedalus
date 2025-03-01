use std::{io, io::ErrorKind};

#[derive(Clone, Copy, PartialEq, Debug)]
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
            Self::Hydrogen => 1,
            Self::Carbon => 4,
            Self::Oxygen => 2,
            Self::Nitrogen => 3,
            Self::Fluorine => 1,
            Self::Sulfur => 2,     // can be 2, 4, or 6, but 2 is a common choice
            Self::Phosphorus => 5, // can be 3 or 5, here we pick 5
            Self::Iron => 2,       // Fe(II) is common (Fe(III) also common)
            Self::Copper => 2,     // Cu(I) and Cu(II) both occur, pick 2 as a naive default
            Self::Calcium => 2,
            Self::Potassium => 1,
            Self::Aluminum => 3,
            Self::Lead => 2,    // Pb(II) or Pb(IV), but Pb(II) is more common/stable
            Self::Gold => 3,    // Au(I) and Au(III) are common, pick 3
            Self::Silver => 1,  // Ag(I) is most common
            Self::Mercury => 2, // Hg(I) and Hg(II), pick 2
            Self::Tin => 4,     // Sn(II) or Sn(IV), pick 4
            Self::Zinc => 2,
            Self::Magnesium => 2,
            Self::Manganese => 7, // todo: Not sure
            Self::Iodine => 1,    // can have higher, but 1 is typical in many simple compounds
            Self::Chlorine => 1,  // can also be 3,5,7, but 1 is the simplest (e.g., HCl)
            Self::Tungsten => 6,  // W can have multiple but 6 is a common oxidation state
            Self::Tellurium => 2, // can also be 4 or 6, pick 2
            Self::Selenium => 2,  // can also be 4 or 6, pick 2
            Self::Bromine => 7,
            Self::Other => 0, // default to 0 for unknown or unhandled elements
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
                pdbtbx::Element::H => Self::Hydrogen,
                pdbtbx::Element::C => Self::Carbon,
                pdbtbx::Element::O => Self::Oxygen,
                pdbtbx::Element::N => Self::Nitrogen,
                pdbtbx::Element::F => Self::Fluorine,
                pdbtbx::Element::S => Self::Sulfur,
                pdbtbx::Element::P => Self::Phosphorus,
                pdbtbx::Element::Fe => Self::Iron,
                pdbtbx::Element::Cu => Self::Copper,
                pdbtbx::Element::Ca => Self::Calcium,
                pdbtbx::Element::K => Self::Potassium,
                pdbtbx::Element::Al => Self::Aluminum,
                pdbtbx::Element::Pb => Self::Lead,
                pdbtbx::Element::Au => Self::Gold,
                pdbtbx::Element::Ag => Self::Silver,
                pdbtbx::Element::Hg => Self::Mercury,
                pdbtbx::Element::Sn => Self::Tin,
                pdbtbx::Element::Zn => Self::Zinc,
                pdbtbx::Element::Mg => Self::Magnesium,
                pdbtbx::Element::Mn => Self::Manganese,
                pdbtbx::Element::I => Self::Iodine,
                pdbtbx::Element::Cl => Self::Chlorine,
                pdbtbx::Element::W => Self::Tungsten,
                pdbtbx::Element::Te => Self::Tellurium,
                pdbtbx::Element::Se => Self::Selenium,
                pdbtbx::Element::Br => Self::Bromine,

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
            "H" => Ok(Self::Hydrogen),
            "C" => Ok(Self::Carbon),
            "O" => Ok(Self::Oxygen),
            "N" => Ok(Self::Nitrogen),
            "F" => Ok(Self::Fluorine),
            "S" => Ok(Self::Sulfur),
            "P" => Ok(Self::Phosphorus),
            "FE" => Ok(Self::Iron),
            "CU" => Ok(Self::Copper),
            "CA" => Ok(Self::Calcium),
            "K" => Ok(Self::Potassium),
            "AL" => Ok(Self::Aluminum),
            "PB" => Ok(Self::Lead),
            "AU" => Ok(Self::Gold),
            "AG" => Ok(Self::Silver),
            "HG" => Ok(Self::Mercury),
            "SN" => Ok(Self::Tin),
            "ZN" => Ok(Self::Zinc),
            "MG" => Ok(Self::Magnesium),
            "MN" => Ok(Self::Manganese),
            "I" => Ok(Self::Iodine),
            "CL" => Ok(Self::Chlorine),
            "W" => Ok(Self::Tungsten),
            "TE" => Ok(Self::Tellurium),
            "SE" => Ok(Self::Selenium),
            "BR" => Ok(Self::Bromine),
            // todo: Fill in if you need, or remove this fn.
            _ => Err(io::Error::new(
                ErrorKind::InvalidData,
                "Invalid atom letter",
            )),
        }
    }

    pub fn to_letter(&self) -> String {
        match self {
            Self::Hydrogen => "H".into(),
            Self::Carbon => "C".into(),
            Self::Oxygen => "O".into(),
            Self::Nitrogen => "N".into(),
            Self::Fluorine => "F".into(),
            Self::Sulfur => "S".into(),
            Self::Phosphorus => "P".into(),
            Self::Iron => "FE".into(),
            Self::Copper => "CU".into(),
            Self::Calcium => "CA".into(),
            Self::Potassium => "K".into(),
            Self::Aluminum => "AL".into(),
            Self::Lead => "PB".into(),
            Self::Gold => "AU".into(),
            Self::Silver => "AG".into(),
            Self::Mercury => "HG".into(),
            Self::Tin => "SN".into(),
            Self::Zinc => "ZN".into(),
            Self::Magnesium => "MG".into(),
            Self::Manganese => "MN".into(),
            Self::Iodine => "I".into(),
            Self::Chlorine => "CL".into(),
            Self::Tungsten => "W".into(),
            Self::Tellurium => "TE".into(),
            Self::Selenium => "SE".into(),
            Self::Bromine => "Br".into(),
            Self::Other => "X".into(),
        }
    }

    /// From [PyMol](https://pymolwiki.org/index.php/Color_Values)
    pub fn color(&self) -> (f32, f32, f32) {
        match self {
            Self::Hydrogen => (0.9, 0.9, 0.9),
            Self::Carbon => (0.2, 1., 0.2),
            Self::Oxygen => (1., 0.3, 0.3),
            Self::Nitrogen => (0.2, 0.2, 1.0),
            Self::Fluorine => (0.701, 1.0, 1.0),
            Self::Sulfur => (0.9, 0.775, 0.25),
            Self::Phosphorus => (1.0, 0.502, 0.),
            Self::Iron => (0.878, 0.4, 0.2),
            Self::Copper => (0.784, 0.502, 0.2),
            Self::Calcium => (0.239, 1.0, 0.),
            Self::Potassium => (0.561, 0.251, 0.831),
            Self::Aluminum => (0.749, 0.651, 0.651),
            Self::Lead => (0.341, 0.349, 0.380),
            Self::Gold => (1., 0.820, 0.137),
            Self::Silver => (0.753, 0.753, 0.753),
            Self::Mercury => (0.722, 0.722, 0.816),
            Self::Tin => (0.4, 0.502, 0.502),
            Self::Zinc => (0.490, 0.502, 0.690),
            Self::Magnesium => (0.541, 1., 0.),
            Self::Manganese => (0.541, 1., 0.541),
            Self::Iodine => (0.580, 0., 0.580),
            Self::Chlorine => (0.121, 0.941, 0.121),
            Self::Tungsten => (0.129, 0.580, 0.840),
            Self::Tellurium => (0.831, 0.478, 0.),
            Self::Selenium => (1.0, 0.631, 0.),
            Self::Bromine => (1.0, 0.99, 0.),
            Self::Other => (5., 5., 5.),
        }
    }

    #[rustfmt::skip]
    /// Covalent radius, in angstrom.
    /// https://github.com/openbabel/openbabel/blob/master/src/elementtable.h
    /// https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)
    pub fn covalent_radius(self) -> f64 {
        match self {
            Element::Hydrogen   => 0.31,
            Element::Carbon     => 0.76,
            Element::Oxygen     => 0.66,
            Element::Nitrogen   => 0.71,
            Element::Fluorine   => 0.57,
            Element::Sulfur     => 1.05,
            Element::Phosphorus => 1.07,
            Element::Iron       => 1.32,
            Element::Copper     => 1.32,
            Element::Calcium    => 1.76,
            Element::Potassium  => 2.03,
            Element::Aluminum   => 1.21,
            Element::Lead       => 1.46,
            Element::Gold       => 1.36,
            Element::Silver     => 1.45,
            Element::Mercury    => 1.32,
            Element::Tin        => 1.39,
            Element::Zinc       => 1.22,
            Element::Magnesium  => 1.41, // 1.19?
            Element::Manganese  => 1.39,
            Element::Iodine     => 1.39,
            Element::Chlorine   => 1.02,
            Element::Tungsten   => 1.62,
            Element::Tellurium  => 1.38,
            Element::Selenium   => 1.20,
            Element::Bromine  => 1.14, // 1.14 - 1.20
            Element::Other      => 0.00,
        }
    }

    #[rustfmt::skip]
    /// Van-der-wals radius, in angstrom.
    /// https://github.com/openbabel/openbabel/blob/master/src/elementtable.h
    /// https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)
    pub const fn vdw_radius(&self) -> f32 {
        match self {
            Element::Hydrogen   => 1.10, // or 120
            Element::Carbon     => 1.70,
            Element::Oxygen     => 1.52,
            Element::Nitrogen   => 1.55,
            Element::Fluorine   => 1.47,
            Element::Sulfur     => 1.80,
            Element::Phosphorus => 1.80,
            Element::Iron       => 2.05,
            Element::Copper     => 2.00,
            Element::Calcium    => 2.31,
            Element::Potassium  => 2.75,
            Element::Aluminum   => 1.84,
            Element::Lead       => 2.02,
            Element::Gold       => 2.10,
            Element::Silver     => 2.10,
            Element::Mercury    => 2.05,
            Element::Tin        => 1.93,
            Element::Zinc       => 2.10,
            Element::Magnesium  => 1.73,
            Element::Manganese  => 0., // N/A?
            Element::Iodine     => 1.98,
            Element::Chlorine   => 1.75,
            Element::Tungsten   => 2.10,
            Element::Tellurium  => 2.06,
            Element::Selenium   => 1.90,
            Element::Bromine   => 1.85,
            Element::Other      => 0.0,
        }
    }
}
