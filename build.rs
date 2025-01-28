//! We use this to automatically compile CUDA C++ code when building.

#[cfg(feature = "cuda")]
use cuda_setup::{build, GpuArchitecture};

fn main() {
    #[cfg(feature = "cuda")]
    build(
        GpuArchitecture::Rtx4,
        &vec!["src/cuda/cuda.cu", "src/cuda/util.cu"],
    );
}
