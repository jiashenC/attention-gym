#include <cuda_runtime.h>
#include <torch/extension.h>

torch::Tensor apply(torch::Tensor q, torch::Tensor k, torch::Tensor v)
{
    // TODO
}

// Python bindings
PYBIND11_MODULE(custom_attn_kernel, m)
{
    // The actual interface for attention kernel
    m.def("apply", &apply, "Custom attention in CUDA");

    // You can commen out those interfaces for testing correctness but not really needed
    // m.def("test_apply_matmul", &test_apply_matmul, "Testing interface for custom matmul in CUDA");
    // m.def("test_apply_matmul_T", &test_apply_matmul_T, "Testing interface for custom transposed matmul in CUDA");
    // m.def("test_apply_softmax", &test_apply_softmax, "Testing interface for custom softmax in CUDA");
}