// For running FFTs on SPME reciprical code. We invoke cuFFT from the host side
// (Still in this module), which launches its own opaque kernel.

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>



extern "C" {
// Opaque handle we keep in Rust as a pointer
void* spme_make_plan_c2c(int nx, int ny, int nz, void* cu_stream /*CUstream or cudaStream_t*/);
void  spme_exec_inverse_3_c2c(void* plan,
                              cufftComplex* exk,
                              cufftComplex* eyk,
                              cufftComplex* ezk);
void  spme_destroy_plan(void* plan);
void  spme_scale_c2c(cufftComplex* data, size_t n, float scale, void* cu_stream);
}

// #define CUDA_OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ printf("CUDA %s @%s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); return; } } while(0)
// #define CUFFT_OK(x) do { cufftResult r=(x); if(r!=CUFFT_SUCCESS){ printf("CUFFT err %d @%s:%d\n", int(r), __FILE__, __LINE__); return; } } while(0)

static __global__ void scale_c(cufftComplex* a, size_t n, float s){
    size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i < n) { a[i].x *= s; a[i].y *= s; }
}

struct PlanWrap {
    cufftHandle plan;
    size_t n_per_grid;
    cudaStream_t stream;
};

extern "C"
void* spme_make_plan_c2c(int nx, int ny, int nz, void* cu_stream) {
    auto* w = new PlanWrap();
    w->n_per_grid = size_t(nx) * ny * nz;
#if defined(__CUDA_API_VERSION) && __CUDA_API_VERSION >= 11000
    w->stream = reinterpret_cast<cudaStream_t>(cu_stream);
#else
    w->stream = (cudaStream_t)cu_stream;
#endif
    int n[3] = {nx, ny, nz};
    CUFFT_SAFE_CALL(cufftCreate(&w->plan));
    // One plan, 3 batches
    CUFFT_SAFE_CALL(cufftPlanMany(&w->plan, 3, n,
                                  nullptr, 1, w->n_per_grid,
                                  nullptr, 1, w->n_per_grid,
                                  CUFFT_C2C, 3));
    cufftSetStream(w->plan, w->stream);
    return w;
}

extern "C"
void spme_exec_inverse_3_c2c(void* plan, cufftComplex* exk, cufftComplex* eyk, cufftComplex* ezk) {
    auto* w = reinterpret_cast<PlanWrap*>(plan);
    // Execute three batches via separate pointers:
    cufftExecC2C(w->plan, exk, exk, CUFFT_INVERSE);
    cufftExecC2C(w->plan, eyk, eyk, CUFFT_INVERSE);
    cufftExecC2C(w->plan, ezk, ezk, CUFFT_INVERSE);
}

extern "C"
void spme_scale_c2c(cufftComplex* data, size_t n, float scale, void* cu_stream) {
    auto stream = reinterpret_cast<cudaStream_t>(cu_stream);
    int threads = 256;
    int blocks  = int((n + threads - 1) / threads);
    scale_c<<<blocks, threads, 0, stream>>>(data, n, scale);
}

extern "C"
void spme_destroy_plan(void* plan) {
    auto* w = reinterpret_cast<PlanWrap*>(plan);
    if (!w) return;
    cufftDestroy(w->plan);
    delete w;
}