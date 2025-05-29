//! We use this to automatically compile CUDA C++ code when building.

use std::env;

#[cfg(feature = "cuda")]
use cuda_setup::{GpuArchitecture, build};
use winresource::WindowsResource;

fn main() {
    #[cfg(feature = "cuda")]
    build(
        GpuArchitecture::Rtx4,
        &vec!["src/cuda/cuda.cu", "src/cuda/util.cu"],
    );

    if env::var_os("CARGO_CFG_WINDOWS").is_some() {
        WindowsResource::new()
            // This path can be absolute, or relative to your crate root.
            .set_icon("resources/icon.ico")
            .compile()
            .unwrap();
    }
}
