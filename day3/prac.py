import math
import torch


def flash_attention(q, k, v):
    ###############################################################################
    # We emualate the online attention mechanism here using PyTorch APIs.
    ###############################################################################
    k_t = k.transpose(1, 2)
    bsz = q.size()[0]

    o = torch.empty_like(q)
    for bi in range(bsz):
        q_per_b, k_t_per_b, v_per_b = (
            q[bi],
            k_t[bi],
            v[bi],
        )
        for ki, q_row in enumerate(q_per_b):
            m, d = -torch.inf, 0
            for i in range(k_t_per_b.size()[-1]):
                k_t_col = k_t_per_b[:, i]
                q_k = torch.matmul(q_row, k_t_col)
                print(q_k, q_k.size())
                prev_m, prev_d = m, d
                m = max(m, q_k)
                d = prev_d * torch.e ** (prev_m - m) + torch.e ** (q_k - m)
                o[bi, ki] = (
                    o[bi, ki] * (prev_d * torch.e ** (prev_m - m)) / d
                    + torch.e ** (q_k - m) / d * v_per_b[i]
                )
    return o


def custom_attn(q, k, v, w_q, w_k, w_v, w_out):
    feat_q = torch.matmul(q, w_q.T)
    feat_k = torch.matmul(k, w_k.T)
    feat_v = torch.matmul(v, w_v.T)

    feat_q = feat_q * math.sqrt(1.0 / float(feat_q.size(-1)))  # Required
    ###############################################################################
    # Ex.1: To understand the internal of online attention, we first emulate this
    # process in python implementation.
    ###############################################################################
    out = flash_attention(feat_q, feat_k, feat_v)

    feat_out = torch.matmul(out, w_out.T)
    return feat_out


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
