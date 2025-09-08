//! We use this to automatically compile CUDA C++ code when building.

use std::env;

#[cfg(feature = "cuda")]
use cuda_setup::{GpuArchitecture, build_host, build_ptx};
use winresource::WindowsResource;

fn main() {
    // #[cfg(feature = "cuda")]
    // build_ptx(
    //     // Select the min supported GPU architecture.
    //     GpuArchitecture::Rtx3,
    //     &["src/cuda/cuda.cu", "src/cuda/util.cu"],
    //     "daedalus",
    // );

    #[cfg(feature = "cuda")]
    build_host(
        // Select the min supported GPU architecture.
        GpuArchitecture::Rtx3,
        &["src/cuda/spme.cu"],
        "spme", // This name is currently hard-coded in the Ewald lib.
    );

    if env::var_os("CARGO_CFG_WINDOWS").is_some() {
        WindowsResource::new()
            // This path can be absolute, or relative to your crate root.
            .set_icon("resources/icon.ico")
            .compile()
            .unwrap();
    }
}
