#include <cuda_runtime.h>
#include <torch/extension.h>

template <int TILE_SIZE>
__global__ void fa_attn_kernel(float *q_mat, float *k_mat, float *v_mat, float *o_mat, int Q_N_ROW, int Q_N_COL, int K_N_ROW)
{
    // Input
    __shared__ float q_tile[TILE_SIZE];
    __shared__ float k_tile[TILE_SIZE];

    // Intermediate values
    __shared__ float qk_tile[TILE_SIZE];
    __shared__ float o_tile[TILE_SIZE];
    float m = -FLT_MAX, d = 0;

    int row = blockIdx.x, col = threadIdx.x;

    q_tile[col] = q_mat[row * Q_N_COL + col];
    qk_tile[col] = 0.0f;
    o_tile[col] = 0.0f;
    __syncthreads();

    for (int i = 0; i < K_N_ROW; ++i)
    {
        k_tile[col] = k_mat[i * Q_N_COL + col];
        __syncthreads();

        // Q @ K
        qk_tile[col] = q_tile[col] * k_tile[col];
        __syncthreads();
        for (int k = TILE_SIZE / 2; k > 0; k >>= 1)
        {
            if (col < k)
            {
                qk_tile[col] += qk_tile[col + k];
            }
            __syncthreads();
        }
        float qk = qk_tile[0];

        // Update m and d
        float prev_m = m, prev_d = d;
        m = fmaxf(m, qk);
        d = prev_d * expf(prev_m - m) + expf(qk - m);

        // Update o
        o_tile[col] = o_tile[col] * (prev_d * expf(prev_m - m)) / d + expf(qk - m) / d * v_mat[i * Q_N_COL + col];
        __syncthreads();
    }

    // Write to output
    o_mat[row * Q_N_COL + col] = o_tile[col];
}

torch::Tensor apply(torch::Tensor q, torch::Tensor k, torch::Tensor v)
{
    int bsz = q.size(0);
    auto out = torch::empty(q.sizes(), torch::device(torch::kCUDA));

    for (int bi = 0; bi < bsz; ++bi)
    {
        // Per batch data
        auto q_b = q[bi], k_b = k[bi], v_b = v[bi];
        fa_attn_kernel<512><<<q_b.size(0), 512>>>(q_b.data_ptr<float>(), k_b.data_ptr<float>(), v_b.data_ptr<float>(), out[bi].data_ptr<float>(), q_b.size(0), q_b.size(1), k_b.size(0));
    }

    return out;
}

// Python bindings
PYBIND11_MODULE(custom_fa_attn_kernel, m)
{
    m.def("apply", &apply, "Custom flash attention in CUDA");
}