#include <torch/extension.h>

torch::Tensor forward_flash_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_flash_attention", torch::wrap_pybind_function(forward_flash_attention), "forward flash attention");
}