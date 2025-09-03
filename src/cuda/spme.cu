// spme_fft.cu
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cstdio>

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
    w->stream = reinterpret_cast<cudaStream_t>(cu_stream);

    cufftResult r = cufftPlan3d(&w->plan, nx, ny, nz, CUFFT_C2C);
    if (r != CUFFT_SUCCESS) { printf("cufftPlan3d err=%d\n", int(r)); delete w; return nullptr; }

    r = cufftSetStream(w->plan, w->stream);
    if (r != CUFFT_SUCCESS) { printf("cufftSetStream err=%d\n", int(r)); cufftDestroy(w->plan); delete w; return nullptr; }

    return w;
}

extern "C"
void spme_exec_inverse_3_c2c(void* plan, cufftComplex* exk, cufftComplex* eyk, cufftComplex* ezk) {
    auto* w = reinterpret_cast<PlanWrap*>(plan);
    if (!w) return;
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
