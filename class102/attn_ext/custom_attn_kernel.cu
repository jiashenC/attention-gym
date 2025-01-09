#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void softmax_kernel(float *inp, float *outp, int NUM_ROW, int NUM_COL)
{
    extern __shared__ float buffer[];

    int row = blockIdx.x;  // Each block processes one row
    int tid = threadIdx.x; // Thread ID within the block

    if (row >= NUM_ROW)
        return;

    float *row_inp = inp + row * NUM_COL;
    float *row_outp = outp + row * NUM_COL;

    // Pass 1: Compute max value in the row for numerical stability
    float max_val = -FLT_MAX;
    for (int i = tid; i < NUM_COL; i += blockDim.x)
    {
        max_val = fmaxf(max_val, row_inp[i]);
    }

    // Reduce maxVal across threads in the block
    buffer[tid] = max_val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            buffer[tid] = fmaxf(buffer[tid], buffer[tid + stride]);
        }
        __syncthreads();
    }
    max_val = buffer[0];

    // Pass 2: Compute exp and sum of exp
    float sum_exp = 0.0f;
    for (int i = tid; i < NUM_COL; i += blockDim.x)
    {
        row_outp[i] = expf(row_inp[i] - max_val);
        sum_exp += row_outp[i];
    }

    // Reduce sum_exp across threads in the block
    buffer[tid] = sum_exp;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            buffer[tid] += buffer[tid + stride];
        }
        __syncthreads();
    }
    sum_exp = buffer[0];

    // Pass 3: Normalize to compute softmax
    for (int i = tid; i < NUM_COL; i += blockDim.x)
    {
        row_outp[i] /= sum_exp;
    }
}

template <int TILE_SIZE>
__global__ void matmul_T_kernel(float *a_mat, float *b_mat, float *out_mat, int M, int N, int K)
{
    __shared__ float a_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float b_tile[TILE_SIZE][TILE_SIZE];

    // Row and column indices of the d_C matrix this thread computes
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0.0f;

    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile)
    {
        // Load tiles into shared memory
        if (row < M && tile * TILE_SIZE + threadIdx.x < K)
        {
            a_tile[threadIdx.y][threadIdx.x] = a_mat[row * K + tile * TILE_SIZE + threadIdx.x];
        }
        else
        {
            a_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && tile * TILE_SIZE + threadIdx.y < K)
        {
            b_tile[threadIdx.y][threadIdx.x] = b_mat[col * K + tile * TILE_SIZE + threadIdx.y];
        }
        else
        {
            b_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial product for this tile
        for (int i = 0; i < TILE_SIZE; ++i)
        {
            value += a_tile[threadIdx.y][i] * b_tile[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result to the output matrix
    if (row < M && col < N)
    {
        out_mat[row * N + col] = value;
    }
}

template <int TILE_SIZE>
__global__ void matmul_kernel(float *a_mat, float *b_mat, float *out_mat, int M, int N, int K)
{
    // Shared memory for submatrices
    __shared__ float a_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float b_tile[TILE_SIZE][TILE_SIZE];

    // Row and column of the output matrix element
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Accumulate the result for out_mat[row][col]
    float value = 0.0f;

    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile)
    {
        // Load elements of A and B into shared memory
        if (row < M && tile * TILE_SIZE + threadIdx.x < K)
            a_tile[threadIdx.y][threadIdx.x] = a_mat[row * K + tile * TILE_SIZE + threadIdx.x];
        else
            a_tile[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && tile * TILE_SIZE + threadIdx.y < K)
            b_tile[threadIdx.y][threadIdx.x] = b_mat[(tile * TILE_SIZE + threadIdx.y) * N + col];
        else
            b_tile[threadIdx.y][threadIdx.x] = 0.0f;

        // Synchronize threads to ensure tiles are loaded
        __syncthreads();

        // Perform computation for this tile
        for (int k = 0; k < TILE_SIZE; ++k)
        {
            value += a_tile[threadIdx.y][k] * b_tile[k][threadIdx.x];
        }

        // Synchronize threads to ensure computation is complete before loading the next tile
        __syncthreads();
    }

    // Write the result to the output matrix
    if (row < M && col < N)
    {
        out_mat[row * N + col] = value;
    }
}

torch::Tensor apply(torch::Tensor q, torch::Tensor k, torch::Tensor v)
{
    const int TILE_SIZE = 16;
    int bsz = q.size(0);

    auto out = torch::empty(q.sizes(), torch::device(torch::kCUDA));
    auto qk_out = torch::empty({q.size(1), k.size(1)}, torch::device(torch::kCUDA));
    auto qk_sm_out = torch::empty({q.size(1), k.size(1)}, torch::device(torch::kCUDA));

    for (int bi = 0; bi < bsz; ++bi)
    {
        // Per batch data
        auto q_b = q[bi], k_b = k[bi], v_b = v[bi];

        // Q @ K
        int q_M = q_b.size(0), qk_K = q_b.size(1), k_N = k_b.size(0);
        dim3 qk_threads_per_block(TILE_SIZE, TILE_SIZE);
        dim3 qk_blocks_per_grid(
            (k_N + TILE_SIZE - 1) / TILE_SIZE,
            (q_M + TILE_SIZE - 1) / TILE_SIZE);
        qk_out.zero_();
        matmul_T_kernel<TILE_SIZE><<<qk_blocks_per_grid, qk_threads_per_block>>>(q_b.data_ptr<float>(), k_b.data_ptr<float>(), qk_out.data_ptr<float>(), q_M, k_N, qk_K);

        // Softmax of QK
        qk_sm_out.zero_();
        softmax_kernel<<<q_M, 256, q_M * sizeof(float)>>>(qk_out.data_ptr<float>(), qk_sm_out.data_ptr<float>(), q_M, k_N);

        // QK @ V
        int v_N = v_b.size(1);
        dim3 qkv_threads_per_block(TILE_SIZE, TILE_SIZE);
        dim3 qkv_blocks_per_grid(
            (v_N + TILE_SIZE - 1) / TILE_SIZE,
            (q_M + TILE_SIZE - 1) / TILE_SIZE);
        matmul_kernel<TILE_SIZE><<<qkv_blocks_per_grid, qkv_threads_per_block>>>(qk_sm_out.data_ptr<float>(), v_b.data_ptr<float>(), out[bi].data_ptr<float>(), q_M, v_N, k_N);
    }

    return out;
}

torch::Tensor test_apply_matmul(torch::Tensor a, torch::Tensor b)
{
    auto a_size = a.sizes();
    auto b_size = b.sizes();
    int M = a_size[0], N = b_size[1], K = a_size[1];

    auto out = torch::empty({M, N}, torch::device(torch::kCUDA));
    const int TILE_SIZE = 16;
    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);
    dim3 blocks_per_grid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_kernel<TILE_SIZE><<<blocks_per_grid, threads_per_block>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), M, N, K);
    return out;
}

torch::Tensor test_apply_matmul_T(torch::Tensor a, torch::Tensor b)
{
    auto a_size = a.sizes();
    auto b_size = b.sizes();
    int M = a_size[0], N = b_size[0], K = a_size[1];

    auto out = torch::empty({M, N}, torch::device(torch::kCUDA));
    const int TILE_SIZE = 16;
    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);
    dim3 blocks_per_grid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_T_kernel<TILE_SIZE><<<blocks_per_grid, threads_per_block>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), M, N, K);
    return out;
}

torch::Tensor test_apply_softmax(torch::Tensor inp, int dim)
{
    int N_DIM = inp.dim();
    if (dim < 0)
    {
        dim += N_DIM;
    }
    if (dim != N_DIM - 1)
    {
        inp = inp.transpose(dim, N_DIM - 1);
    }
    auto inp_sizes = inp.sizes();
    int N_ROW = inp_sizes[0], N_COL = inp_sizes[1];
    dim3 threads_per_block(256);
    dim3 blocks_per_grid(N_ROW);
    auto outp = torch::empty(inp_sizes, torch::device(torch::kCUDA));
    softmax_kernel<<<blocks_per_grid, threads_per_block, N_COL * sizeof(float)>>>(inp.data_ptr<float>(), outp.data_ptr<float>(), N_ROW, N_COL);
    if (dim != N_DIM - 1)
    {
        inp = inp.transpose(dim, N_DIM - 1);
        outp = outp.transpose(dim, N_DIM - 1);
    }
    return outp;
}

// Python bindings
PYBIND11_MODULE(custom_attn_kernel, m)
{
    m.def("apply", &apply, "Custom attention in CUDA");
    m.def("test_apply_matmul", &test_apply_matmul, "Testing interface for custom matmul in CUDA");
    m.def("test_apply_matmul_T", &test_apply_matmul_T, "Testing interface for custom transposed matmul in CUDA");
    m.def("test_apply_softmax", &test_apply_softmax, "Testing interface for custom softmax in CUDA");
}