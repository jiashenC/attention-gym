import torch
import custom_attn_kernel


a = torch.randn(10, 20).to(torch.float).to("cuda")
b = torch.randn(30, 20).to(torch.float).to("cuda")
b_t = torch.randn(20, 30).to(torch.float).to("cuda")
c = torch.randn(2, 2).to(torch.float).to("cuda")

torch_out = torch.matmul(a, b_t)
custom_out = custom_attn_kernel.test_apply_matmul(a, b_t)
assert (torch_out - custom_out).abs().max() < 0.001

torch_out = torch.matmul(a, b.transpose(0, 1))
custom_out = custom_attn_kernel.test_apply_matmul_T(a, b)
assert (torch_out - custom_out).abs().max() < 0.001

for mat_i, mat in enumerate([a, b, c]):
    custom_out = custom_attn_kernel.test_apply_softmax(mat, -1)
    torch_out = torch.softmax(mat, -1)
    assert (torch_out - custom_out).abs().max() < 0.001
