/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "permute.h"
#include <string.h>

#include <torch/torch.h>
#include <cub/cub.cuh>
#include <cuda_bf16.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ATen/cuda/CUDAContext.h"

#include "cutlass/arch/memory.h"
#include "cutlass/arch/cache_operation.h"
#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"


using torch::Tensor;

namespace grouped_gemm {

template <typename T>
inline T *get_ptr(torch::Tensor &t)
{
    return reinterpret_cast<T *>(t.data_ptr());
}

////////////////////////////////
// modified
///////////////////////////////

static inline __host__ __device__ int round_up_int(int x, int m) {
    return ((x + m - 1) / m) * m;
}

// counts -> padded_offsets(E+1), padded_counts(E)
// single-thread GPU kernel, E typically small(<=256/512)
static __global__ void compute_padded_offsets_kernel(
    const int* __restrict__ counts,     // [E]
    int* __restrict__ padded_offsets,   // [E+1]
    int* __restrict__ padded_counts,    // [E]
    int E,
    int align)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    padded_offsets[0] = 0;
    for (int e = 0; e < E; ++e) {
        int pc = round_up_int(counts[e], align);
        padded_counts[e] = pc;
        padded_offsets[e + 1] = padded_offsets[e] + pc;
    }
}

// build padded_sorted_row_id by expert segments (pad entries = -1)
// Note: input sorted_row_id length is N=num_rows*num_topK (sorted by sorted_indices)
static __global__ void build_padded_sorted_row_id_kernel(
    const int* __restrict__ sorted_row_id,      // [N]
    const int* __restrict__ counts,             // [E]
    const int* __restrict__ padded_offsets,     // [E+1]
    int* __restrict__ padded_sorted_row_id,     // [total_padded]
    int E)
{
    int e = blockIdx.x;
    if (e >= E) return;

    // compute in_start by summing counts[0..e-1]
    // (if E becomes large, change to an exclusive-scan of counts)
    int in_start = 0;
    for (int i = 0; i < e; ++i) in_start += counts[i];
    int in_len = counts[e];

    int out_start = padded_offsets[e];
    int out_end   = padded_offsets[e + 1];
    int out_len   = out_end - out_start;

    for (int i = threadIdx.x; i < out_len; i += blockDim.x) {
        int out_idx = out_start + i;
        padded_sorted_row_id[out_idx] = (i < in_len) ? sorted_row_id[in_start + i] : -1;
    }
}

// scatter row_id_map from padded_sorted_row_id, skip padded (-1) entries safely.
// row_id_map layout: [num_topK][num_rows]
static __global__ void scatter_row_id_map_from_padded_sorted_row_id(
    const int* __restrict__ padded_sorted_row_id, // [num_out_tokens_padded], -1 means pad
    int* __restrict__ row_id_map,                 // [num_topK*num_rows], should be initialized to -1
    int num_out_tokens_padded,
    int num_rows,
    int num_topK,
    int num_negative_one_in_indices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_out_tokens_padded) return;

    int source_row = padded_sorted_row_id[idx];
    if (source_row < 0) return; // pad entry

    int source_token_id = source_row / num_topK;
    int source_topK_id  = source_row - source_token_id * num_topK;

    int dest = idx - num_negative_one_in_indices;
    // keep your existing semantics: entries before removing -1 are invalid
    if (dest < 0) return;

    row_id_map[source_topK_id * num_rows + source_token_id] = dest;
}


// NEW: UB-launch scatter, but only process idx < offsets[E] (true padded total) on GPU.
// This avoids CPU sync to know num_out_tokens_padded.
static __global__ void scatter_row_id_map_from_padded_sorted_row_id_ub(
    const int* __restrict__ padded_sorted_row_id, // [UB], only [0:offsets[E]) written
    int* __restrict__ row_id_map,                 // [num_topK*num_rows], should be initialized to -1
    const int* __restrict__ padded_offsets,       // [E+1]
    int E,
    int num_rows,
    int num_topK,
    int num_negative_one_in_indices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // true total padded length for all experts
    int total = padded_offsets[E];
    if (idx >= total) return;

    int source_row = padded_sorted_row_id[idx];
    if (source_row < 0) return; // pad entry inside real padded area

    int source_token_id = source_row / num_topK;
    int source_topK_id  = source_row - source_token_id * num_topK;

    int dest = idx - num_negative_one_in_indices;
    if (dest < 0) return;

    row_id_map[source_topK_id * num_rows + source_token_id] = dest;
}

// Zero only pad rows (where padded_sorted_row_id[row]==-1).
// This avoids torch::zeros / full memset of output.
// template <typename T, int kElementsPerAccess>
// static __global__ void zero_pad_rows_kernel(
//     T* __restrict__ output,                       // [num_out_tokens_padded, num_cols]
//     const int* __restrict__ padded_sorted_row_id,  // [num_out_tokens_padded]
//     int num_out_tokens_padded,
//     int num_cols)
// {
//     using Frag = cutlass::Array<T, kElementsPerAccess>;

//     int row = blockIdx.x;
//     if (row >= num_out_tokens_padded) return;
//     if (padded_sorted_row_id[row] != -1) return;

//     int tid = threadIdx.x;
//     int64_t num_cols_i64 = (int64_t)num_cols;
//     T* row_ptr = output + (int64_t)row * num_cols_i64;

//     Frag z;
//     #pragma unroll
//     for (int i = 0; i < kElementsPerAccess; ++i) z[i] = T(0);

//     // keep same store granularity as main kernel (float4)
//     for (int c = tid * kElementsPerAccess; c < num_cols; c += blockDim.x * kElementsPerAccess) {
//         *(float4*)(row_ptr + c) = *(float4*)(z.data());
//     }
// }

template <typename T, int kElementsPerAccess>
static __global__ void zero_pad_by_segments_kernel(
    T* __restrict__ output,             // [num_out_tokens_padded, num_cols]
    const int* __restrict__ counts,     // [E]
    const int* __restrict__ offsets,    // [E+1] padded offsets
    int E,
    int num_cols)
{
    using Frag = cutlass::Array<T, kElementsPerAccess>;

    int e = blockIdx.x;
    if (e >= E) return;

    int out_start = offsets[e];
    int out_end   = offsets[e + 1];
    int pad_start = out_start + counts[e];

    int pad_rows = out_end - pad_start;
    if (pad_rows <= 0) return;

    // 2D mapping:
    // blockIdx.y selects which padded row within the segment
    int r = blockIdx.y;
    if (r >= pad_rows) return;

    int row = pad_start + r;

    int tid = threadIdx.x;
    int64_t num_cols_i64 = (int64_t)num_cols;
    T* row_ptr = output + (int64_t)row * num_cols_i64;

    Frag z;
    #pragma unroll
    for (int i = 0; i < kElementsPerAccess; ++i) z[i] = T(0);

    for (int c = tid * kElementsPerAccess; c < num_cols; c += blockDim.x * kElementsPerAccess) {
        *(float4*)(row_ptr + c) = *(float4*)(z.data());
    }
}

///////////////////////////
// end modified
//////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Top K
//
/////////////////////////////////////////////////////////////////////////////////////////////////

static __global__ void moe_permute_topK_row_map(
    const int *sorted_row_id,
    int *row_id_map,
    const int num_rows,  // permute 前的 token 数
    const int num_topK,
    const int num_out_tokens)
{
    // Each block corresponds to one source token
    // row_id_map[num_topK][num_rows]
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int idx = bid * blockDim.x + tid;

    if (idx >= num_rows * num_topK)
        return;

    int source_row = sorted_row_id[idx];
    int source_token_id = source_row / num_topK;
    int source_topK_id = source_row % num_topK;

    if (idx >= num_out_tokens)
    {
        row_id_map[source_topK_id * num_rows + source_token_id] = -1;
    }
    else
    {
        row_id_map[source_topK_id * num_rows + source_token_id] = idx;
    }
}

static __global__ void moe_permute_topK_row_map(
    const int *sorted_row_id,
    int *row_id_map,
    const int num_rows,
    const int num_topK,
    const int num_out_tokens,
    const int num_negative_one_in_indices)
{
    // Each block corresponds to one source token
    // row_id_map[num_topK][num_rows]
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int idx = bid * blockDim.x + tid;
    const int idx_without_negative_one = idx - num_negative_one_in_indices;

    if (idx >= num_rows * num_topK)
        return;

    int source_row = sorted_row_id[idx];
    int source_token_id = source_row / num_topK;
    int source_topK_id = source_row % num_topK;

    if (idx_without_negative_one < 0 || idx_without_negative_one >= num_out_tokens)
    {
        row_id_map[source_topK_id * num_rows + source_token_id] = -1;
    }
    else
    {
        row_id_map[source_topK_id * num_rows + source_token_id] = idx_without_negative_one;
    }
}

template <typename T, typename TCompute, int kElementsPerAccess, bool hasProb>
__global__ void moe_recover_topK_kernel(const T *input,
                                        T *unpermuted_output,
                                        const int *row_id_map,
                                        const float *prob,
                                        const int num_rows,
                                        const int num_topK,
                                        const int num_cols)
{
    extern __shared__ int8_t s_mem[];
    int64_t num_cols_int64 = static_cast<int64_t>(num_cols);
    TCompute *s_prob = reinterpret_cast<TCompute *>(s_mem);

    using FragmentLoadStore = cutlass::Array<T, kElementsPerAccess>;
    using FragmentCompute = cutlass::Array<TCompute, kElementsPerAccess>;

    cutlass::NumericArrayConverter<TCompute, T, kElementsPerAccess> src_converter;
    cutlass::NumericArrayConverter<T, TCompute, kElementsPerAccess> dst_converter;

    // each block corresponds to one source token
    const int source_token = blockIdx.x;
    const int tid = threadIdx.x;

    if (hasProb)
    {
        for (int i = tid; i < num_topK; i += blockDim.x * blockDim.y)
        {
            s_prob[i] = TCompute(prob[source_token * num_topK + i]);
        }
        __syncthreads();
    }

    for (int i = tid * kElementsPerAccess; i < num_cols; i += blockDim.x * kElementsPerAccess)
    {
        FragmentLoadStore frag_load_store;
        FragmentCompute frag_elem;
        FragmentCompute frag_sum;

        int source_row = row_id_map[source_token];

        if (source_row != -1)
        {
            const T *source_row_ptr = input + source_row * num_cols_int64;

            cutlass::arch::global_load<FragmentLoadStore, sizeof(FragmentLoadStore), cutlass::arch::CacheOperation::LastUse>(
                frag_load_store, (source_row_ptr + i), true);
            frag_sum = src_converter(frag_load_store);

            if (hasProb)
            {
                frag_sum = frag_sum * s_prob[0];
            }
        }
        else
        {
            frag_sum.clear();
        }

        for (int k = 1; k < num_topK; k++)
        {
            source_row = row_id_map[k * num_rows + source_token];

            if (source_row == -1)
                continue;

            const T *source_row_ptr = input + source_row * num_cols_int64;

            cutlass::arch::global_load<FragmentLoadStore, sizeof(FragmentLoadStore), cutlass::arch::CacheOperation::LastUse>(
                frag_load_store, (source_row_ptr + i), true);
            frag_elem = src_converter(frag_load_store);

            if (hasProb)
            {
                frag_elem = frag_elem * s_prob[k];
            }

            for (int e = 0; e < kElementsPerAccess; e++)
            {
                frag_sum.at(e) = frag_sum.at(e) + frag_elem.at(e);
            }
        }

        T *dest_row_ptr = unpermuted_output + source_token * num_cols_int64;
        frag_load_store = dst_converter(frag_sum);
        *(float4 *)(dest_row_ptr + i) = *(float4 *)(frag_load_store.data());
    }
}

template <typename T,
          typename TCompute,
          int kElementsPerAccess,
          int topKTile,
          bool hasProb>
__global__ void moe_permute_topK_kernel(const T *input_bwd,
                                        const T *input_fwd,
                                        T *act_grad,
                                        const float *prob,
                                        float *prob_grad,
                                        const int *row_id_map,
                                        const int num_rows,
                                        const int num_topK,
                                        const int num_cols)
{
    int64_t num_cols_int64 = static_cast<int64_t>(num_cols);
    extern __shared__ int8_t s_mem[];
    TCompute *s_prob = reinterpret_cast<TCompute *>(s_mem);

    using FragmentLoadStore = cutlass::Array<T, kElementsPerAccess>;
    using FragmentCompute = cutlass::Array<TCompute, kElementsPerAccess>;

    cutlass::NumericArrayConverter<TCompute, T, kElementsPerAccess> src_converter;
    cutlass::NumericArrayConverter<T, TCompute, kElementsPerAccess> dst_converter;

    const int source_token = blockIdx.x;
    const int tid = threadIdx.x;

    if (hasProb)
    {
        for (int i = tid; i < num_topK; i += blockDim.x)
        {
            s_prob[i] = TCompute(prob[source_token * num_topK + i]);
        }
        __syncthreads();
    }

    float accum[topKTile] = {0.0f};
    FragmentLoadStore frag_load_store;

    const T *source_row_ptr = input_bwd + source_token * num_cols_int64;
    for (int i = tid * kElementsPerAccess; i < num_cols; i += blockDim.x * kElementsPerAccess)
    {
        cutlass::arch::global_load<FragmentLoadStore, sizeof(FragmentLoadStore), cutlass::arch::CacheOperation::LastUse>(
            frag_load_store, (source_row_ptr + i), true);
        FragmentCompute frag_src = src_converter(frag_load_store);

        int index = source_token;

        for (int k = 0; k < topKTile; k++)
        {
            if (k == num_topK) break;

            int dest_row = row_id_map[index];
            index += num_rows;

            if (dest_row == -1)
                continue;

            if (hasProb)
            {
                frag_load_store = dst_converter(frag_src * s_prob[k]);
            }
            else
            {
                frag_load_store = dst_converter(frag_src);
            }

            T *dest_row_ptr = act_grad + dest_row * num_cols_int64;
            *(float4 *)(dest_row_ptr + i) = *(float4 *)(frag_load_store.data());

            if (hasProb)
            {
                const T *input_fwd_ptr = input_fwd + dest_row * num_cols_int64;
                cutlass::arch::global_load<FragmentLoadStore, sizeof(FragmentLoadStore), cutlass::arch::CacheOperation::LastUse>(
                    frag_load_store, (input_fwd_ptr + i), true);
                FragmentCompute frag_input_fwd = src_converter(frag_load_store);

                for (int e = 0; e < kElementsPerAccess; e++)
                {
                    accum[k] += float(frag_src.at(e) * frag_input_fwd.at(e));
                }
            }
        }
    }

    if (hasProb)
    {
        for (int k = 0; k < topKTile; k++)
        {
            if (k == num_topK) break;

            for (int mask = 16; mask > 0; mask /= 2)
            {
                accum[k] = accum[k] + __shfl_xor_sync(0xffffffff, accum[k], mask, 32);
            }
        }

        if (tid == 0)
        {
            for (int k = 0; k < topKTile; k++)
            {
                if (k == num_topK) break;
                prob_grad[source_token * num_topK + k] = accum[k];
            }
        }
    }
}


template <typename T, typename TCompute, bool FWD, int kElementsPerAccess>
void moe_permute_topK_kernel_launcher(
    const T *input,
    T *output,
    const int *sorted_row_id,
    int *row_id_map,
    const float *prob,
    const int num_rows,
    const int num_topK,
    const int num_cols,
    const int num_out_tokens,
    cudaStream_t stream,
    const int num_negative_one_in_indices = 0,
    float *prob_grad = nullptr,
    const T *input_fwd = nullptr)
{
    if (FWD)
    {
        if (prob_grad == nullptr)
        {
            // permute_topK fwd
            int threads = 64;
            int blocks = (num_rows * num_topK + threads - 1) / threads;
            // row_id_map (seq, topk) 第 i 行的 topk 个数字表示 permute 后的 Tensor
            // 的 topk 行要加权求和做 unpermute
            // 例如：
            // permute out = [1, 2, 0, 1, 3, 0, 2, 3]
            // sorted_row_id = [2, 4, 0, 3, 6, 1, 5, 7]
            // row_id_map = [[2, 5], [0, 3], [1, 6], [4, 7]]
            // permute out 的 第 2 行第 5 行是要加权求和的（都对应 permute 前的第 0 个 token）
            if (num_negative_one_in_indices == 0) {
                moe_permute_topK_row_map<<<blocks, threads, 0, stream>>>(
                    sorted_row_id,
                    row_id_map,
                    num_rows,
                    num_topK,
                    num_out_tokens);
            } else {
                moe_permute_topK_row_map<<<blocks, threads, 0, stream>>>(
                    sorted_row_id,
                    row_id_map,
                    num_rows,
                    num_topK,
                    num_out_tokens,
                    num_negative_one_in_indices);
                // size_t num_elements = num_rows * num_topK;
                // std::vector<int> host_row_id_map(num_elements);
                // cudaMemcpy(host_row_id_map.data(), sorted_row_id, num_rows * num_topK * sizeof(int), cudaMemcpyDeviceToHost);
                // for (int i = 0; i < num_rows; ++i) {
                //     for (int j = 0; j < num_topK; j++)
                //         std::cout << host_row_id_map[i * num_topK + j] << " ";
                //     std::cout << std::endl;
                // }
            }

            blocks = num_rows;
            threads = std::min(num_cols / kElementsPerAccess, 1024);
            moe_permute_topK_kernel<T, T, kElementsPerAccess, 128, false><<<blocks, threads, 0, stream>>>(
                input,
                nullptr,
                output,
                nullptr,
                nullptr,
                row_id_map,
                num_rows,
                num_topK,
                num_cols);
        }
        else
        {
            // unpermute_topK bwd
            int blocks = num_rows;
            int threads = 32;
            size_t smem_bytes = num_topK * sizeof(TCompute);

            if (num_topK == 1)
            {
                if (prob == nullptr)
                {
                    moe_permute_topK_kernel<T, T, kElementsPerAccess, 1, false><<<blocks, threads, 0, stream>>>(
                        input,
                        input_fwd,
                        output,
                        prob,
                        prob_grad,
                        row_id_map,
                        num_rows,
                        num_topK,
                        num_cols);
                }
                else
                {
                    moe_permute_topK_kernel<T, TCompute, kElementsPerAccess, 1, true><<<blocks, threads, smem_bytes, stream>>>(
                        input,
                        input_fwd,
                        output,
                        prob,
                        prob_grad,
                        row_id_map,
                        num_rows,
                        num_topK,
                        num_cols);
                }
            }
            else if (num_topK <= 8)
            {
                moe_permute_topK_kernel<T, TCompute, kElementsPerAccess, 8, true><<<blocks, threads, smem_bytes, stream>>>(
                    input,
                    input_fwd,
                    output,
                    prob,
                    prob_grad,
                    row_id_map,
                    num_rows,
                    num_topK,
                    num_cols);
            }
            else if (num_topK <= 16)
            {
                moe_permute_topK_kernel<T, TCompute, kElementsPerAccess, 16, true><<<blocks, threads, smem_bytes, stream>>>(
                    input,
                    input_fwd,
                    output,
                    prob,
                    prob_grad,
                    row_id_map,
                    num_rows,
                    num_topK,
                    num_cols);
            }
            else if (num_topK <= 32)
            {
                moe_permute_topK_kernel<T, TCompute, kElementsPerAccess, 32, true><<<blocks, threads, smem_bytes, stream>>>(
                    input,
                    input_fwd,
                    output,
                    prob,
                    prob_grad,
                    row_id_map,
                    num_rows,
                    num_topK,
                    num_cols);
            }
            else if (num_topK <= 64)
            {
                moe_permute_topK_kernel<T, TCompute, kElementsPerAccess, 64, true><<<blocks, threads, smem_bytes, stream>>>(
                    input,
                    input_fwd,
                    output,
                    prob,
                    prob_grad,
                    row_id_map,
                    num_rows,
                    num_topK,
                    num_cols);
            }
            else if (num_topK <= 128)
            {
                moe_permute_topK_kernel<T, TCompute, kElementsPerAccess, 128, true><<<blocks, threads, smem_bytes, stream>>>(
                    input,
                    input_fwd,
                    output,
                    prob,
                    prob_grad,
                    row_id_map,
                    num_rows,
                    num_topK,
                    num_cols);
            }
            else
            {
                throw std::runtime_error("num_topK cannot exceed 128.");
            }
        }
    }
    else
    {
        int blocks = num_rows;
        int threads = std::min(num_cols / kElementsPerAccess, 1024);
        size_t smem_bytes = num_topK * sizeof(TCompute);


        if (num_topK == 1 && prob == nullptr)
        {
            // permute_topK bwd with topK==1
            moe_recover_topK_kernel<T, T, kElementsPerAccess, false><<<blocks, threads, smem_bytes, stream>>>(
                input,
                output,
                row_id_map,
                prob,
                num_rows,
                num_topK,
                num_cols);
        }
        else if (prob == nullptr)
        {
            // permute_topK bwd
            moe_recover_topK_kernel<T, TCompute, kElementsPerAccess, false><<<blocks, threads, smem_bytes, stream>>>(
                input,
                output,
                row_id_map,
                prob,
                num_rows,
                num_topK,
                num_cols);
        }
        else
        {
            // unpermute_topK fwd
            moe_recover_topK_kernel<T, TCompute, kElementsPerAccess, true><<<blocks, threads, smem_bytes, stream>>>(
                input,
                output,
                row_id_map,
                prob,
                num_rows,
                num_topK,
                num_cols);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Permute_topK OP
//
/////////////////////////////////////////////////////////////////////////////////////////////////

std::tuple<Tensor, Tensor, std::vector<Tensor>> moe_permute_topK_op(
    Tensor              input,
    Tensor              indices,
    int64_t             num_out_tokens,
    std::vector<Tensor> workspace,
    int64_t             num_negative_one_in_indices,
    int64_t             max_expanded_token_num)
{
    const int num_tokens = input.size(0);
    const int num_cols = input.size(1);
    const int num_topK = indices.size(1);

    // initialize the workspace on the first run
    if (workspace.empty()) {
        auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);

        Tensor sorted_indices = torch::empty(max_expanded_token_num, options);
        Tensor row_id = torch::range(0, max_expanded_token_num - 1, 1, options);
        Tensor sorted_row_id =
            torch::empty(max_expanded_token_num, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

        size_t temp_storage_bytes = 0;
        int *temp_ptr = nullptr;
        cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes,
                                        temp_ptr, temp_ptr,
                                        temp_ptr, temp_ptr, max_expanded_token_num);
        Tensor temp_storage =
            torch::empty(temp_storage_bytes, torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));

        workspace.push_back(sorted_indices);
        workspace.push_back(row_id);
        workspace.push_back(sorted_row_id);
        workspace.push_back(temp_storage);
    }

    int *indices_ptr = get_ptr<int>(indices);
    int *sorted_indices_ptr = get_ptr<int>(workspace[0]);
    int *row_id_ptr = get_ptr<int>(workspace[1]);
    int *sorted_row_id_ptr = get_ptr<int>(workspace[2]);

    void *d_temp_storage = get_ptr<void>(workspace[3]);
    size_t temp_storage_bytes = std::numeric_limits<size_t>::max();

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    indices_ptr, sorted_indices_ptr,
                                    row_id_ptr, sorted_row_id_ptr, num_tokens * num_topK,
                                    0, sizeof(std::remove_reference_t<decltype(*indices_ptr)>) * 8, stream);

    // indices 就是 flatten 后的 indices
    // sorted_indices 就是单纯对 indices 排序
    // sorted_row_id 是排序后的第 i 个 token 对应的原始位置
    // indices = [[1, 2], [0, 1], [0, 2], [1, 2]]
    // sorted_indices = [0, 0, 1, 1, 1, 2, 2, 2]
    // sorted_row_id = [2, 4, 0, 3, 6, 1, 5, 7]
    if (std::getenv("PERMUTE_DEBUG_PRINT") != nullptr) {
        // 等当前 stream 完成，否则 CPU 可能读到还没写完的数据
        cudaStreamSynchronize(stream);

        auto n = std::min<int64_t>(num_tokens * num_topK, 64);

        auto indices_cpu = indices.flatten().to(torch::kCPU, /*non_blocking=*/false);
        auto sorted_indices_cpu = workspace[0].slice(0, 0, n).to(torch::kCPU, false);
        auto sorted_row_id_cpu = workspace[2].slice(0, 0, n).to(torch::kCPU, false);

        std::cout << "[permute] n=" << n << "\n";
        std::cout << "indices[0:" << n << "]=" << indices_cpu.slice(0, 0, n) << "\n";
        std::cout << "sorted_indices[0:" << n << "]=" << sorted_indices_cpu << "\n";
        std::cout << "sorted_row_id[0:" << n << "]=" << sorted_row_id_cpu << "\n";
    }

    // activations type
    const at::ScalarType _st = input.scalar_type();

    // Output buffer alloc
    num_out_tokens = (num_out_tokens > 0) ? num_out_tokens : num_tokens * num_topK;
    Tensor permuted_output =
        torch::empty({num_out_tokens, num_cols}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    Tensor row_id_map =
        torch::empty({num_tokens * num_topK}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

    int *row_id_map_ptr = get_ptr<int>(row_id_map);

    switch (_st)
    {
    case at::ScalarType::Float:
    {
        using dType = float;
        using dTypeCompute = float;

        dType *input_ptr = get_ptr<dType>(input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 4>(
            input_ptr,
            permuted_output_ptr,
            sorted_row_id_ptr,
            row_id_map_ptr,
            nullptr,
            num_tokens,
            num_topK,
            num_cols,
            num_out_tokens,
            stream,
            num_negative_one_in_indices);

        break;
    }
    case at::ScalarType::Half:
    {
        using dType = cutlass::half_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 8>(
            input_ptr,
            permuted_output_ptr,
            sorted_row_id_ptr,
            row_id_map_ptr,
            nullptr,
            num_tokens,
            num_topK,
            num_cols,
            num_out_tokens,
            stream,
            num_negative_one_in_indices);

        break;
    }
#ifdef ENABLE_BF16
    case at::ScalarType::BFloat16:
    {
        // Environment variable: PERMUTE_COMPUTE_DTYPE
        // Controls the compute data type used for BFloat16 input tensors in the permute kernel.
        // Valid values:
        //   "fp32" - Use float32 (higher precision) for compute, even if input/output is bfloat16.
        //   (unset or any other value) - Use bfloat16 for compute (default).
        // Set this variable to "fp32" if you want to improve numerical accuracy at the cost of performance.
        static const char* permute_compute_dtype_env = std::getenv("PERMUTE_COMPUTE_DTYPE");

        using dType = cutlass::bfloat16_t;
        dType *input_ptr = get_ptr<dType>(input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        if (permute_compute_dtype_env != nullptr && strcmp(permute_compute_dtype_env, "fp32") == 0){
            using dTypeCompute = float;

            moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 8>(
                input_ptr,
                permuted_output_ptr,
                sorted_row_id_ptr,
                row_id_map_ptr,
                nullptr,
                num_tokens,
                num_topK,
                num_cols,
                num_out_tokens,
                stream,
                num_negative_one_in_indices);
        } else {
            using dTypeCompute = cutlass::bfloat16_t;

            moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 8>(
                input_ptr,
                permuted_output_ptr,
                sorted_row_id_ptr,
                row_id_map_ptr,
                nullptr,
                num_tokens,
                num_topK,
                num_cols,
                num_out_tokens,
                stream,
                num_negative_one_in_indices);
        }
        break;
    }
#endif
#ifdef ENABLE_FP8
    case at::ScalarType::Float8_e5m2:
    {
        using dType = cutlass::float_e5m2_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 16>(
            input_ptr,
            permuted_output_ptr,
            sorted_row_id_ptr,
            row_id_map_ptr,
            nullptr,
            num_tokens,
            num_topK,
            num_cols,
            num_out_tokens,
            stream,
            num_negative_one_in_indices);

        break;
    }
    case at::ScalarType::Float8_e4m3fn:
    {
        using dType = cutlass::float_e4m3_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 16>(
            input_ptr,
            permuted_output_ptr,
            sorted_row_id_ptr,
            row_id_map_ptr,
            nullptr,
            num_tokens,
            num_topK,
            num_cols,
            num_out_tokens,
            stream,
            num_negative_one_in_indices);

        break;
    }
#endif
    default:
        throw std::runtime_error("Wrong activation tensor type.");
    }

    return std::make_tuple(permuted_output, row_id_map, workspace);
}

////////////////////////////////
// permute modified
///////////////////////////////

std::tuple<Tensor, Tensor, std::vector<Tensor>> moe_permute_topK_op_pad(
    Tensor              input,
    Tensor              indices,
    int64_t             num_out_tokens,
    std::vector<Tensor> workspace,
    int64_t             num_negative_one_in_indices,
    int64_t             max_expanded_token_num,
    int64_t             num_experts)
{
    const int num_tokens = input.size(0);
    const int num_cols = input.size(1);
    const int num_topK = indices.size(1);

    // initialize the workspace on the first run
    if (workspace.empty()) {
        auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);

        Tensor sorted_indices = torch::empty(max_expanded_token_num, options);
        Tensor row_id = torch::range(0, max_expanded_token_num - 1, 1, options);
        Tensor sorted_row_id =
            torch::empty(max_expanded_token_num, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

        size_t temp_storage_bytes = 0;
        int *temp_ptr = nullptr;
        cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes,
                                        temp_ptr, temp_ptr,
                                        temp_ptr, temp_ptr, max_expanded_token_num);
        Tensor temp_storage =
            torch::empty(temp_storage_bytes, torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));

        workspace.push_back(sorted_indices);
        workspace.push_back(row_id);
        workspace.push_back(sorted_row_id);
        workspace.push_back(temp_storage);
    }

    int *indices_ptr = get_ptr<int>(indices);
    int *sorted_indices_ptr = get_ptr<int>(workspace[0]);
    int *row_id_ptr = get_ptr<int>(workspace[1]);
    int *sorted_row_id_ptr = get_ptr<int>(workspace[2]);

    void *d_temp_storage = get_ptr<void>(workspace[3]);
    size_t temp_storage_bytes = std::numeric_limits<size_t>::max();

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    indices_ptr, sorted_indices_ptr,
                                    row_id_ptr, sorted_row_id_ptr, num_tokens * num_topK,
                                    0, sizeof(std::remove_reference_t<decltype(*indices_ptr)>) * 8, stream);

    // ...existing debug print...

    // ===== NEW: pad each expert segment to multiple of 128 =====
    constexpr int kAlignTokens = 128;
    const int E = (int)num_experts;
    const int N = num_tokens * num_topK;

    auto i32opt = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);

    // 1) histogram counts per expert over sorted_indices (or indices.flatten(), both ok)
    Tensor expert_counts = torch::zeros({E}, i32opt);      // [E]
    Tensor padded_counts = torch::empty({E}, i32opt);      // [E]
    Tensor padded_offsets = torch::empty({E + 1}, i32opt); // [E+1]

    int* expert_counts_ptr = get_ptr<int>(expert_counts);
    int* padded_counts_ptr = get_ptr<int>(padded_counts);
    int* padded_offsets_ptr = get_ptr<int>(padded_offsets);

    size_t hist_temp_bytes = 0;
    cub::DeviceHistogram::HistogramEven(
        nullptr, hist_temp_bytes,
        sorted_indices_ptr, expert_counts_ptr,
        E + 1, 0, E,
        N, stream);
    Tensor hist_temp = torch::empty((int64_t)hist_temp_bytes, torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));
    cub::DeviceHistogram::HistogramEven(
        get_ptr<void>(hist_temp), hist_temp_bytes,
        sorted_indices_ptr, expert_counts_ptr,
        E + 1, 0, E,
        N, stream);

    // 2) compute padded offsets
    compute_padded_offsets_kernel<<<1, 1, 0, stream>>>(
        expert_counts_ptr, padded_offsets_ptr, padded_counts_ptr, E, kAlignTokens);

    // 3) total padded length: REMOVE CPU sync, use UB allocation instead.
    //
    // Let M be the permute expanded token count (before per-expert padding).
    // Here M = N = num_tokens * num_topK (same as you used).
    // UB formula (your statement): (M + 128*E - M%128).
    // Equivalent: round_up(M,128) + 128*E (when M%128==0, still adds 128*E).
    // We'll use that: cheaper, safe.
    const int64_t M = (int64_t)N;
    const int64_t num_out_tokens_ub = ((M + kAlignTokens - 1) / kAlignTokens) * kAlignTokens + (int64_t)kAlignTokens * (int64_t)E;

    // 4) build padded_sorted_row_id into UB buffer (only [0:padded_offsets[E]) is written by kernel)
    Tensor padded_sorted_row_id =
        torch::empty({num_out_tokens_ub}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    int* padded_sorted_row_id_ptr = get_ptr<int>(padded_sorted_row_id);

    build_padded_sorted_row_id_kernel<<<E, 256, 0, stream>>>(
        sorted_row_id_ptr, expert_counts_ptr, padded_offsets_ptr, padded_sorted_row_id_ptr, E);

    // activations type
    const at::ScalarType _st = input.scalar_type();

    // Output buffer alloc using UB (NOTE: keep EMPTY)
    num_out_tokens = num_out_tokens_ub; // output length becomes UB
    Tensor permuted_output =
        torch::empty({num_out_tokens, num_cols}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));

    // row_id_map init to -1 then scatter valid
    Tensor row_id_map =
        torch::empty({num_tokens * num_topK}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    int *row_id_map_ptr = get_ptr<int>(row_id_map);

    // set all to -1
    cudaMemsetAsync(row_id_map_ptr, 0xFF, (size_t)(num_tokens * num_topK) * sizeof(int), stream);
    {
        int threads = 256;
        int blocks = (int)((num_out_tokens_ub + threads - 1) / threads);
        scatter_row_id_map_from_padded_sorted_row_id_ub<<<blocks, threads, 0, stream>>>(
            padded_sorted_row_id_ptr,
            row_id_map_ptr,
            padded_offsets_ptr,
            E,
            num_tokens,
            num_topK,
            (int)num_negative_one_in_indices);
    }

    // Now run original permute kernel via launcher, but it would rebuild row_id_map internally.
    // So we bypass launcher and call moe_permute_topK_kernel directly (same kernel it uses).

    switch (_st)
    {
    case at::ScalarType::Float:
    {
        using dType = float;
        dType *input_ptr = get_ptr<dType>(input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        int blocks = num_tokens;
        int threads = std::min(num_cols / 4, 1024);
        moe_permute_topK_kernel<dType, dType, 4, 128, false><<<blocks, threads, 0, stream>>>(
            input_ptr,
            nullptr,
            permuted_output_ptr,
            nullptr,
            nullptr,
            row_id_map_ptr,
            num_tokens,
            num_topK,
            num_cols);

        // pad_rows per expert <= 127 always when align=128, so fix grid.y=128
        dim3 grid(E, 128, 1);
        int z_threads = std::min(num_cols / 4, 1024);
        zero_pad_by_segments_kernel<dType, 4><<<grid, z_threads, 0, stream>>>(
            permuted_output_ptr,
            expert_counts_ptr,
            padded_offsets_ptr,
            E,
            num_cols);

        break;
    }
    case at::ScalarType::Half:
    {
        using dType = cutlass::half_t;
        dType *input_ptr = get_ptr<dType>(input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        int blocks = num_tokens;
        int threads = std::min(num_cols / 8, 1024);
        moe_permute_topK_kernel<dType, dType, 8, 128, false><<<blocks, threads, 0, stream>>>(
            input_ptr,
            nullptr,
            permuted_output_ptr,
            nullptr,
            nullptr,
            row_id_map_ptr,
            num_tokens,
            num_topK,
            num_cols);

        // pad_rows per expert <= 127 always when align=128, so fix grid.y=128
        dim3 grid(E, 128, 1);
        int z_threads = std::min(num_cols / 8, 1024);
        zero_pad_by_segments_kernel<dType, 8><<<grid, z_threads, 0, stream>>>(
            permuted_output_ptr,
            expert_counts_ptr,
            padded_offsets_ptr,
            E,
            num_cols);
        

        break;
    }
#ifdef ENABLE_BF16
    case at::ScalarType::BFloat16:
    {
        static const char* permute_compute_dtype_env = std::getenv("PERMUTE_COMPUTE_DTYPE");

        using dType = cutlass::bfloat16_t;
        dType *input_ptr = get_ptr<dType>(input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        int blocks = num_tokens;
        int threads = std::min(num_cols / 8, 1024);

        if (permute_compute_dtype_env != nullptr && strcmp(permute_compute_dtype_env, "fp32") == 0){
            // NOTE: hasProb=false so TCompute not used much, keep signature consistent anyway
            moe_permute_topK_kernel<dType, float, 8, 128, false><<<blocks, threads, 0, stream>>>(
                input_ptr, nullptr, permuted_output_ptr, nullptr, nullptr,
                row_id_map_ptr, num_tokens, num_topK, num_cols);
        } else {
            moe_permute_topK_kernel<dType, dType, 8, 128, false><<<blocks, threads, 0, stream>>>(
                input_ptr, nullptr, permuted_output_ptr, nullptr, nullptr,
                row_id_map_ptr, num_tokens, num_topK, num_cols);
        }

        // pad_rows per expert <= 127 always when align=128, so fix grid.y=128
        dim3 grid(E, 128, 1);
        int z_threads = std::min(num_cols / 8, 1024);
        zero_pad_by_segments_kernel<dType, 8><<<grid, z_threads, 0, stream>>>(
            permuted_output_ptr,
            expert_counts_ptr,
            padded_offsets_ptr,
            E,
            num_cols);

        break;
    }
#endif
#ifdef ENABLE_FP8
    case at::ScalarType::Float8_e5m2:
    {
        using dType = cutlass::float_e5m2_t;
        using dTypeCompute = cutlass::half_t;
        dType *input_ptr = get_ptr<dType>(input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        int blocks = num_tokens;
        int threads = std::min(num_cols / 16, 1024);
        moe_permute_topK_kernel<dType, dTypeCompute, 16, 128, false><<<blocks, threads, 0, stream>>>(
            input_ptr, nullptr, permuted_output_ptr, nullptr, nullptr,
            row_id_map_ptr, num_tokens, num_topK, num_cols);

        // pad_rows per expert <= 127 always when align=128, so fix grid.y=128
        dim3 grid(E, 128, 1);
        int z_threads = std::min(num_cols / 16, 1024);
        zero_pad_by_segments_kernel<dType, 16><<<grid, z_threads, 0, stream>>>(
            permuted_output_ptr,
            expert_counts_ptr,
            padded_offsets_ptr,
            E,
            num_cols);

        break;
    }
    case at::ScalarType::Float8_e4m3fn:
    {
        using dType = cutlass::float_e4m3_t;
        using dTypeCompute = cutlass::half_t;
        dType *input_ptr = get_ptr<dType>(input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        int blocks = num_tokens;
        int threads = std::min(num_cols / 16, 1024);
        moe_permute_topK_kernel<dType, dTypeCompute, 16, 128, false><<<blocks, threads, 0, stream>>>(
            input_ptr, nullptr, permuted_output_ptr, nullptr, nullptr,
            row_id_map_ptr, num_tokens, num_topK, num_cols);

        // pad_rows per expert <= 127 always when align=128, so fix grid.y=128
        dim3 grid(E, 128, 1);
        int z_threads = std::min(num_cols / 16, 1024);
        zero_pad_by_segments_kernel<dType, 16><<<grid, z_threads, 0, stream>>>(
            permuted_output_ptr,
            expert_counts_ptr,
            padded_offsets_ptr,
            E,
            num_cols);

        break;
    }
#endif
    default:
        throw std::runtime_error("Wrong activation tensor type.");
    }

    return std::make_tuple(permuted_output, row_id_map, workspace);
}

/////////////////////////
// permute modified end
/////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Unpermute_topK OP
//
/////////////////////////////////////////////////////////////////////////////////////////////////

// NEW: recover kernel that supports non-padded prob.
// prob is [num_tokens, num_topK] (no padding); input is permuted activations [num_out_tokens, num_cols].
// row_id_map is [num_topK, num_tokens] and may contain -1.
// Semantics: for each token, sum_k input[row_id_map[k, token]] * prob[token, k]; if row_id_map==-1 => contributes 0.
// NOTE: This avoids reading prob using permuted/expert-row indexing.
template <typename T, typename TCompute, int kElementsPerAccess, bool hasProb>
__global__ void moe_recover_topK_kernel_nopad_prob(
    const T *input,
    T *unpermuted_output,
    const int *row_id_map,
    const float *prob,        // [num_rows, num_topK] (no pad)
    const int num_rows,       // num_tokens
    const int num_topK,
    const int num_cols)
{
    extern __shared__ int8_t s_mem[];
    int64_t num_cols_int64 = static_cast<int64_t>(num_cols);
    TCompute *s_prob = reinterpret_cast<TCompute *>(s_mem);

    using FragmentLoadStore = cutlass::Array<T, kElementsPerAccess>;
    using FragmentCompute = cutlass::Array<TCompute, kElementsPerAccess>;

    cutlass::NumericArrayConverter<TCompute, T, kElementsPerAccess> src_converter;
    cutlass::NumericArrayConverter<T, TCompute, kElementsPerAccess> dst_converter;

    const int token = blockIdx.x;
    const int tid = threadIdx.x;

    if (hasProb)
    {
        // prob is indexed by [token, k], NOT by permuted row
        for (int k = tid; k < num_topK; k += blockDim.x)
        {
            s_prob[k] = TCompute(prob[token * num_topK + k]);
        }
        __syncthreads();
    }

    for (int i = tid * kElementsPerAccess; i < num_cols; i += blockDim.x * kElementsPerAccess)
    {
        FragmentLoadStore frag_load_store;
        FragmentCompute frag_sum;
        frag_sum.clear();

        // k=0..num_topK-1
        for (int k = 0; k < num_topK; ++k)
        {
            int src_row = row_id_map[k * num_rows + token];
            if (src_row == -1) continue;

            const T *src_ptr = input + (int64_t)src_row * num_cols_int64;

            cutlass::arch::global_load<FragmentLoadStore, sizeof(FragmentLoadStore), cutlass::arch::CacheOperation::LastUse>(
                frag_load_store, (src_ptr + i), true);

            FragmentCompute frag_elem = src_converter(frag_load_store);

            if constexpr (hasProb)
            {
                frag_elem = frag_elem * s_prob[k];
            }

            #pragma unroll
            for (int e = 0; e < kElementsPerAccess; e++)
            {
                frag_sum.at(e) = frag_sum.at(e) + frag_elem.at(e);
            }
        }

        T *dst_ptr = unpermuted_output + (int64_t)token * num_cols_int64;
        frag_load_store = dst_converter(frag_sum);
        *(float4 *)(dst_ptr + i) = *(float4 *)(frag_load_store.data());
    }
}

Tensor moe_recover_topK_op_unpad(
    Tensor  input,
    Tensor  row_id_map,
    Tensor  prob,
    int64_t num_tokens,
    int64_t num_topK)
{
    const int num_cols = input.size(1);

    // activations type
    const at::ScalarType _st = input.scalar_type();

    // Output buffer alloc
    Tensor unpermuted_output =
        torch::empty({num_tokens, num_cols}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));

    int *row_id_map_ptr = get_ptr<int>(row_id_map);
    float *prob_ptr = (prob.defined()) ? get_ptr<float>(prob) : nullptr;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // We intentionally do NOT pad prob. prob is [num_tokens, num_topK] and row_id_map==-1 contributes 0.
    // Launch one block per token, same as before.
    switch (_st)
    {
    case at::ScalarType::Float:
    {
        using dType = float;
        using dTypeCompute = float;

        dType *input_ptr = get_ptr<dType>(input);
        dType *unpermuted_output_ptr = get_ptr<dType>(unpermuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, false, 4>(
            input_ptr,
            unpermuted_output_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream);

        break;
    }
    case at::ScalarType::Half:
    {
        using dType = cutlass::half_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *unpermuted_output_ptr = get_ptr<dType>(unpermuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, false, 8>(
            input_ptr,
            unpermuted_output_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream);

        break;
    }
#ifdef ENABLE_BF16
    case at::ScalarType::BFloat16:
    {
        static const char* permute_compute_dtype_env = std::getenv("PERMUTE_COMPUTE_DTYPE");

        using dType = cutlass::bfloat16_t;
        dType *input_ptr = get_ptr<dType>(input);
        dType *unpermuted_output_ptr = get_ptr<dType>(unpermuted_output);

        int blocks = (int)num_tokens;
        int threads = std::min(num_cols / 8, 1024);

        if (permute_compute_dtype_env != nullptr && strcmp(permute_compute_dtype_env, "fp32") == 0){
            using dTypeCompute = float;
            size_t smem_bytes = (prob_ptr != nullptr) ? (size_t)num_topK * sizeof(dTypeCompute) : 0;

            if (prob_ptr == nullptr) {
                moe_recover_topK_kernel_nopad_prob<dType, dTypeCompute, 8, false><<<blocks, threads, smem_bytes, stream>>>(
                    input_ptr, unpermuted_output_ptr, row_id_map_ptr, nullptr,
                    (int)num_tokens, (int)num_topK, num_cols);
            } else {
                moe_recover_topK_kernel_nopad_prob<dType, dTypeCompute, 8, true><<<blocks, threads, smem_bytes, stream>>>(
                    input_ptr, unpermuted_output_ptr, row_id_map_ptr, prob_ptr,
                    (int)num_tokens, (int)num_topK, num_cols);
            }
        } else {
            using dTypeCompute = cutlass::bfloat16_t;
            size_t smem_bytes = (prob_ptr != nullptr) ? (size_t)num_topK * sizeof(dTypeCompute) : 0;

            if (prob_ptr == nullptr) {
                moe_recover_topK_kernel_nopad_prob<dType, dTypeCompute, 8, false><<<blocks, threads, smem_bytes, stream>>>(
                    input_ptr, unpermuted_output_ptr, row_id_map_ptr, nullptr,
                    (int)num_tokens, (int)num_topK, num_cols);
            } else {
                moe_recover_topK_kernel_nopad_prob<dType, dTypeCompute, 8, true><<<blocks, threads, smem_bytes, stream>>>(
                    input_ptr, unpermuted_output_ptr, row_id_map_ptr, prob_ptr,
                    (int)num_tokens, (int)num_topK, num_cols);
            }
        }

        // if (permute_compute_dtype_env != nullptr && strcmp(permute_compute_dtype_env, "fp32") == 0){
        //     using dTypeCompute = float;
        //     moe_permute_topK_kernel_launcher<dType, dTypeCompute, false, 8>(
        //         input_ptr,
        //         unpermuted_output_ptr,
        //         nullptr,
        //         row_id_map_ptr,
        //         prob_ptr,
        //         num_tokens,
        //         num_topK,
        //         num_cols,
        //         0,
        //         stream);
        // } else {
        //     using dTypeCompute = cutlass::bfloat16_t;
        //     moe_permute_topK_kernel_launcher<dType, dTypeCompute, false, 8>(
        //         input_ptr,
        //         unpermuted_output_ptr,
        //         nullptr,
        //         row_id_map_ptr,
        //         prob_ptr,
        //         num_tokens,
        //         num_topK,
        //         num_cols,
        //         0,
        //         stream);
        // }

        break;
    }
#endif
#ifdef ENABLE_FP8
    case at::ScalarType::Float8_e5m2:
    {
        using dType = cutlass::float_e5m2_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *unpermuted_output_ptr = get_ptr<dType>(unpermuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, false, 16>(
            input_ptr,
            unpermuted_output_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream);

        break;
    }
    case at::ScalarType::Float8_e4m3fn:
    {
        using dType = cutlass::float_e4m3_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *unpermuted_output_ptr = get_ptr<dType>(unpermuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, false, 16>(
            input_ptr,
            unpermuted_output_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream);

        break;
    }
#endif
    default:
        throw std::runtime_error("Wrong activation tensor type.");
    }

    return unpermuted_output;
}

Tensor moe_recover_topK_op(
    Tensor  input,
    Tensor  row_id_map,
    Tensor  prob,
    int64_t num_tokens,
    int64_t num_topK)
{
    const int num_cols = input.size(1);

    // activations type
    const at::ScalarType _st = input.scalar_type();

    // Output buffer alloc
    Tensor unpermuted_output =
        torch::empty({num_tokens, num_cols}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));

    int *row_id_map_ptr = get_ptr<int>(row_id_map);
    float *prob_ptr = (prob.defined()) ? get_ptr<float>(prob) : nullptr;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // We intentionally do NOT pad prob. prob is [num_tokens, num_topK] and row_id_map==-1 contributes 0.
    // Launch one block per token, same as before.
    switch (_st)
    {
    case at::ScalarType::Float:
    {
        using dType = float;
        using dTypeCompute = float;

        dType *input_ptr = get_ptr<dType>(input);
        dType *unpermuted_output_ptr = get_ptr<dType>(unpermuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, false, 4>(
            input_ptr,
            unpermuted_output_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream);

        break;
    }
    case at::ScalarType::Half:
    {
        using dType = cutlass::half_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *unpermuted_output_ptr = get_ptr<dType>(unpermuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, false, 8>(
            input_ptr,
            unpermuted_output_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream);

        break;
    }
#ifdef ENABLE_BF16
    case at::ScalarType::BFloat16:
    {
        static const char* permute_compute_dtype_env = std::getenv("PERMUTE_COMPUTE_DTYPE");

        using dType = cutlass::bfloat16_t;
        dType *input_ptr = get_ptr<dType>(input);
        dType *unpermuted_output_ptr = get_ptr<dType>(unpermuted_output);

        if (permute_compute_dtype_env != nullptr && strcmp(permute_compute_dtype_env, "fp32") == 0){
            using dTypeCompute = float;
            moe_permute_topK_kernel_launcher<dType, dTypeCompute, false, 8>(
                input_ptr,
                unpermuted_output_ptr,
                nullptr,
                row_id_map_ptr,
                prob_ptr,
                num_tokens,
                num_topK,
                num_cols,
                0,
                stream);
        } else {
            using dTypeCompute = cutlass::bfloat16_t;
            moe_permute_topK_kernel_launcher<dType, dTypeCompute, false, 8>(
                input_ptr,
                unpermuted_output_ptr,
                nullptr,
                row_id_map_ptr,
                prob_ptr,
                num_tokens,
                num_topK,
                num_cols,
                0,
                stream);
        }

        break;
    }
#endif
#ifdef ENABLE_FP8
    case at::ScalarType::Float8_e5m2:
    {
        using dType = cutlass::float_e5m2_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *unpermuted_output_ptr = get_ptr<dType>(unpermuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, false, 16>(
            input_ptr,
            unpermuted_output_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream);

        break;
    }
    case at::ScalarType::Float8_e4m3fn:
    {
        using dType = cutlass::float_e4m3_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *unpermuted_output_ptr = get_ptr<dType>(unpermuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, false, 16>(
            input_ptr,
            unpermuted_output_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream);

        break;
    }
#endif
    default:
        throw std::runtime_error("Wrong activation tensor type.");
    }

    return unpermuted_output;
}

Tensor moe_recover_topK_op_inplace(
    Tensor  input,
    Tensor  unpermuted_output,
    Tensor  row_id_map,
    Tensor  prob,
    int64_t num_tokens,
    int64_t num_topK)
{
    const int num_cols = input.size(1);

    // activations type
    const at::ScalarType _st = input.scalar_type();

    // Output buffer alloc
    // Tensor unpermuted_output =
    //     torch::empty({num_tokens, num_cols}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));

    int *row_id_map_ptr = get_ptr<int>(row_id_map);
    float *prob_ptr = (prob.defined()) ? get_ptr<float>(prob) : nullptr;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    switch (_st)
    {
    case at::ScalarType::Float:
    {
        using dType = float;
        using dTypeCompute = float;

        dType *input_ptr = get_ptr<dType>(input);
        dType *unpermuted_output_ptr = get_ptr<dType>(unpermuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, false, 4>(
            input_ptr,
            unpermuted_output_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream);

        break;
    }
    case at::ScalarType::Half:
    {
        using dType = cutlass::half_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *unpermuted_output_ptr = get_ptr<dType>(unpermuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, false, 8>(
            input_ptr,
            unpermuted_output_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream);

        break;
    }
#ifdef ENABLE_BF16
    case at::ScalarType::BFloat16:
    {
        static const char* permute_compute_dtype_env = std::getenv("PERMUTE_COMPUTE_DTYPE");

        using dType = cutlass::bfloat16_t;
        dType *input_ptr = get_ptr<dType>(input);
        dType *unpermuted_output_ptr = get_ptr<dType>(unpermuted_output);

        if (permute_compute_dtype_env != nullptr && strcmp(permute_compute_dtype_env, "fp32") == 0){
            using dTypeCompute = float;
            moe_permute_topK_kernel_launcher<dType, dTypeCompute, false, 8>(
                input_ptr,
                unpermuted_output_ptr,
                nullptr,
                row_id_map_ptr,
                prob_ptr,
                num_tokens,
                num_topK,
                num_cols,
                0,
                stream);
        } else {
            using dTypeCompute = cutlass::bfloat16_t;
            moe_permute_topK_kernel_launcher<dType, dTypeCompute, false, 8>(
                input_ptr,
                unpermuted_output_ptr,
                nullptr,
                row_id_map_ptr,
                prob_ptr,
                num_tokens,
                num_topK,
                num_cols,
                0,
                stream);
        }

        break;
    }
#endif
#ifdef ENABLE_FP8
    case at::ScalarType::Float8_e5m2:
    {
        using dType = cutlass::float_e5m2_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *unpermuted_output_ptr = get_ptr<dType>(unpermuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, false, 16>(
            input_ptr,
            unpermuted_output_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream);

        break;
    }
    case at::ScalarType::Float8_e4m3fn:
    {
        using dType = cutlass::float_e4m3_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_ptr = get_ptr<dType>(input);
        dType *unpermuted_output_ptr = get_ptr<dType>(unpermuted_output);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, false, 16>(
            input_ptr,
            unpermuted_output_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream);

        break;
    }
#endif
    default:
        throw std::runtime_error("Wrong activation tensor type.");
    }

    return unpermuted_output;
}

std::tuple<Tensor, Tensor> moe_recover_topK_bwd_op(
    Tensor  input_bwd,
    Tensor  input_fwd,
    Tensor  row_id_map,
    Tensor  prob)
{
    const int num_tokens = prob.size(0);
    const int num_topK = prob.size(1);
    const int num_cols = input_bwd.size(1);

    int *row_id_map_ptr = get_ptr<int>(row_id_map);
    float *prob_ptr = get_ptr<float>(prob);

    // activations type
    const at::ScalarType _st = input_bwd.scalar_type();

    // Output buffer alloc
    Tensor act_grad =
        torch::empty({input_fwd.size(0), num_cols}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    Tensor prob_grad =
        torch::empty({num_tokens, num_topK}, torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false));
    float *prob_grad_ptr = get_ptr<float>(prob_grad);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    switch (_st)
    {
    case at::ScalarType::Float:
    {
        using dType = float;
        using dTypeCompute = float;

        dType *input_bwd_ptr = get_ptr<dType>(input_bwd);
        dType *input_fwd_ptr = get_ptr<dType>(input_fwd);
        dType *act_grad_ptr = get_ptr<dType>(act_grad);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 4>(
            input_bwd_ptr,
            act_grad_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream,
            0,
            prob_grad_ptr,
            input_fwd_ptr);

        break;
    }
    case at::ScalarType::Half:
    {
        using dType = cutlass::half_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_bwd_ptr = get_ptr<dType>(input_bwd);
        dType *input_fwd_ptr = get_ptr<dType>(input_fwd);
        dType *act_grad_ptr = get_ptr<dType>(act_grad);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 8>(
            input_bwd_ptr,
            act_grad_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream,
            0,
            prob_grad_ptr,
            input_fwd_ptr);

        break;
    }
#ifdef ENABLE_BF16
    case at::ScalarType::BFloat16:
    {
        using dType = cutlass::bfloat16_t;
        dType *input_bwd_ptr = get_ptr<dType>(input_bwd);
        dType *input_fwd_ptr = get_ptr<dType>(input_fwd);
        dType *act_grad_ptr = get_ptr<dType>(act_grad);

        static const char* permute_compute_dtype_env = std::getenv("PERMUTE_COMPUTE_DTYPE");
        if (permute_compute_dtype_env != nullptr && strcmp(permute_compute_dtype_env, "fp32") == 0){
            using dTypeCompute = float;
            moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 8>(
                input_bwd_ptr,
                act_grad_ptr,
                nullptr,
                row_id_map_ptr,
                prob_ptr,
                num_tokens,
                num_topK,
                num_cols,
                0,
                stream,
                0,
                prob_grad_ptr,
                input_fwd_ptr);
        } else {
            using dTypeCompute = cutlass::bfloat16_t;
            moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 8>(
                input_bwd_ptr,
                act_grad_ptr,
                nullptr,
                row_id_map_ptr,
                prob_ptr,
                num_tokens,
                num_topK,
                num_cols,
                0,
                stream,
                0,
                prob_grad_ptr,
                input_fwd_ptr);
        }

        break;
    }
#endif
#ifdef ENABLE_FP8
    case at::ScalarType::Float8_e5m2:
    {
        using dType = cutlass::float_e5m2_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_bwd_ptr = get_ptr<dType>(input_bwd);
        dType *input_fwd_ptr = get_ptr<dType>(input_fwd);
        dType *act_grad_ptr = get_ptr<dType>(act_grad);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 16>(
            input_bwd_ptr,
            act_grad_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream,
            0,
            prob_grad_ptr,
            input_fwd_ptr);

        break;
    }
    case at::ScalarType::Float8_e4m3fn:
    {
        using dType = cutlass::float_e4m3_t;
        using dTypeCompute = cutlass::half_t;

        dType *input_bwd_ptr = get_ptr<dType>(input_bwd);
        dType *input_fwd_ptr = get_ptr<dType>(input_fwd);
        dType *act_grad_ptr = get_ptr<dType>(act_grad);

        moe_permute_topK_kernel_launcher<dType, dTypeCompute, true, 16>(
            input_bwd_ptr,
            act_grad_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream,
            0,
            prob_grad_ptr,
            input_fwd_ptr);

        break;
    }
#endif
    default:
        throw std::runtime_error("Wrong activation tensor type.");
    }

    return std::make_tuple(act_grad, prob_grad);
}

}  // namespace grouped_gemm
