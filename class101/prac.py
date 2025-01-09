import math
import torch
import torch.nn.functional as F


def custom_attn(q, k, v, w_q, w_k, w_v, w_out):
    ###############################################################################
    # Ex.1: To learn the internal of attention mechanism, we are going to emulate
    # the standard attention API of PyTorch. We use @ to denote matrix multiply.
    # The basic operation that we need to emulate is (Q @ K) @ V.
    ###############################################################################

    # ------------------------------------------------------------------------------
    # Ex.1 S.1: Instead of using Q, K, V directly, we will featurize Q, K, V first.
    # By featuring, we meant Q @ W_Q.
    # ------------------------------------------------------------------------------
    # TODO

    # ------------------------------------------------------------------------------
    # Ex.1 S.2: Apply (Q @ K) @ V operation. After applying Q @ K, instead of
    # passing its output directly, we will use a Softmax to constraint its output
    # in a range first. Then, Softmax(Q @ K) @ V.
    # ------------------------------------------------------------------------------
    # TODO

    # ------------------------------------------------------------------------------
    # Ex.1 S.3: Featurize output.
    # ------------------------------------------------------------------------------
    # TODO

    raise NotImplementedError


torch_attn = (
    torch.nn.MultiheadAttention(512, 1, bias=False, batch_first=True)
    .to(torch.float)
    .to("cuda")
    .eval()
)

# Number of tokens in a
seq_len = 10

# Q, K, V
# Their dimension are [# batch, # token, # feature]
q = torch.randn((2, seq_len, 512)).to(torch.float).to("cuda")
k = torch.randn((2, seq_len, 512)).to(torch.float).to("cuda")
v = torch.randn((2, seq_len, 512)).to(torch.float).to("cuda")

# Extract weights from PyTorch module.
all_param = dict(torch_attn.named_parameters())
w_q, w_k, w_v = all_param["in_proj_weight"].chunk(3)
w_q, w_k, w_v = (
    w_q.view(512, 512),
    w_k.view(512, 512),
    w_v.view(512, 512),
)
w_out = all_param["out_proj.weight"].view(512, 512)

# Test correctness (prefilling).
torch_attn_out, _ = torch_attn(q, k, v)
custom_attn_out = custom_attn(q, k, v, w_q, w_k, w_v, w_out)
assert (torch_attn_out - custom_attn_out).abs().max() < 0.0001

# Test correctness (decoding).
q = torch.randn((2, 1, 512)).to(torch.float).to("cuda")
torch_attn_out, _ = torch_attn(q, k, v)
custom_attn_out = custom_attn(q, k, v, w_q, w_k, w_v, w_out)
assert (torch_attn_out - custom_attn_out).abs().max() < 0.0001
