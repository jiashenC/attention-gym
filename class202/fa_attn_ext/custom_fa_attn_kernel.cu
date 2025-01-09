#include <cuda_runtime.h>
#include <torch/extension.h>

torch::Tensor apply(torch::Tensor q, torch::Tensor k, torch::Tensor v)
{
    // TODO
}

// Python bindings
PYBIND11_MODULE(custom_fa_attn_kernel, m)
{
    m.def("apply", &apply, "Custom flash attention in CUDA");
}