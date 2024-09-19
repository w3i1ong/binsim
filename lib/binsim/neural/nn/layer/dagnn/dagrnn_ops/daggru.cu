#include "daggru.h"
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
using namespace torch;

template<typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid_(scalar_t x){
    scalar_t one = static_cast<scalar_t>(1.0);
    return one / (one + exp(-x));
}

template<typename scalar_t>
__device__ __forceinline__ scalar_t tanh_(scalar_t x){
    return 2.0 / (1 + exp(-2*x)) - 1;
}

template<typename scalar_t>
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t y){
    scalar_t one = static_cast<scalar_t>(1.0);
    return y * (one - y);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_tanh(scalar_t y){
    scalar_t one = static_cast<scalar_t>(1.0);
    return one - y * y;
}

template<typename scalar_t>
__global__ void fused_gru_partial_forward_kernel(
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, long> last_hidden,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, long> hidden,
        torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, long> gru_wx,
        torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, long> gru_wh,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, long> gru_bias,
        long row_base,
        long row_num,
        long rows_per_thread
){
    const long column = blockIdx.y * blockDim.y + threadIdx.y;
    long start_row = (blockIdx.x * blockDim.x + threadIdx.x) * rows_per_thread + row_base;
    long last_row = row_base + row_num;
    if (column < last_hidden.size(1)) {
        for (int i = 0; i < rows_per_thread && start_row < last_row; i++, start_row++) {
            double a = sigmoid_(double(gru_wh[start_row][0][column]) + gru_wx[start_row][0][column] + gru_bias[0][column]);
            double b = sigmoid_(double(gru_wh[start_row][1][column]) + gru_wx[start_row][1][column] + gru_bias[1][column]);
            double c = tanh_(b * double(gru_wh[start_row][2][column]) + gru_wx[start_row][2][column] + gru_bias[2][column]);
            hidden[start_row][column] = double(last_hidden[start_row][column]) + a * (c - last_hidden[start_row][column]);
        }
    }
}

void fused_gru_partial_forward(Tensor last_hidden, Tensor hidden, Tensor gru_wx, Tensor gru_wh, Tensor gru_bias,
                               long node_base, long node_num){
    unsigned int hidden_dim = last_hidden.size(1);
    unsigned int thread_num;
    const long rows_per_thread=8;
    if(hidden_dim < 32) thread_num = 32;
    else if(hidden_dim < 128) thread_num = 64;
    else thread_num = 128;
    auto stream = c10::cuda::getCurrentCUDAStream(hidden.device().index()).stream();
    const dim3 threads(1, thread_num);
    const dim3 blocks((node_num+rows_per_thread-1)/rows_per_thread, (hidden_dim+thread_num-1)/thread_num);
    AT_DISPATCH_FLOATING_TYPES(last_hidden.scalar_type(), "fused_gru_partial_forward", ([&]{
        fused_gru_partial_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                last_hidden.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, long>() ,
                hidden.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, long>(),
                gru_wx.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, long>(),
                gru_wh.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, long>(),
                gru_bias.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, long>(),
                node_base,
                node_num,
                rows_per_thread
        );
    }));
}

template<typename scalar_t>
__global__ void fused_gru_partial_backward_kernel(
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, long> grad_hidden,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, long> grad_last_hidden,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, long> last_hidden,
        torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, long> gru_wx,
        torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, long> gru_wh,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, long> gru_bias,
        long row_base,
        long row_num,
        long rows_per_thread
){
    const long column = blockIdx.y * blockDim.y + threadIdx.y;
    long start_row = (blockIdx.x * blockDim.x + threadIdx.x) * rows_per_thread + row_base;
    long last_row = row_base + row_num;
    if (column < last_hidden.size(1)) {
        for (int i = 0; i < rows_per_thread && start_row < last_row; i++, start_row++) {
            double a = sigmoid_(double(gru_wh[start_row][0][column]) + gru_wx[start_row][0][column] + gru_bias[0][column]);
            double b = sigmoid_(double(gru_wh[start_row][1][column]) + gru_wx[start_row][1][column] + gru_bias[1][column]);
            double c = tanh_(b * double(gru_wh[start_row][2][column]) + gru_wx[start_row][2][column] + gru_bias[2][column]);
//             hidden[start_row][column] = double(last_hidden[start_row][column]) + a * (c - last_hidden[start_row][column]);
            double grad = grad_hidden[start_row][column];
            grad_last_hidden[start_row][column] = (1 - a) * grad;
            // grad_c
            double grad_a = d_sigmoid(a) * (c - last_hidden[start_row][column]) * grad;
            double grad_c = d_tanh(c) * a * grad;
            double grad_b = d_sigmoid(b) * grad_c * gru_wh[start_row][2][column];

            gru_wh[start_row][2][column] = b * grad_c;
            gru_wx[start_row][2][column] = grad_c;
            gru_wh[start_row][1][column] = gru_wx[start_row][1][column] = grad_b;
            gru_wh[start_row][0][column] = gru_wx[start_row][0][column] = grad_a;
        }
    }
}

void fused_gru_partial_backward(Tensor grad_hidden, Tensor grad_last_hidden, Tensor last_hidden,
                                Tensor gru_wx, Tensor gru_wh, Tensor gru_bias,
                                long node_base, long node_num){
    unsigned int hidden_dim = last_hidden.size(1);
    unsigned int thread_num;
    const long rows_per_thread=8;
    if(hidden_dim < 32) thread_num = 32;
    else if(hidden_dim < 128) thread_num = 64;
    else thread_num = 128;
    auto stream = c10::cuda::getCurrentCUDAStream(grad_hidden.device().index()).stream();
    const dim3 threads(1, thread_num);
    const dim3 blocks((node_num+rows_per_thread-1)/rows_per_thread, (hidden_dim+thread_num-1)/thread_num);
    AT_DISPATCH_FLOATING_TYPES(last_hidden.scalar_type(), "fused_gru_partial_backward", ([&]{
        fused_gru_partial_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                grad_hidden.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, long>() ,
                grad_last_hidden.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, long>() ,
                last_hidden.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, long>() ,
                gru_wx.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, long>(),
                gru_wh.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, long>(),
                gru_bias.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, long>(),
                node_base,
                node_num,
                rows_per_thread
        );
    }));
}

template<typename scalar_t>
__global__ void message_passing_forward_kernel(torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, long> hidden,
                                               torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, long> last_hidden,
                                               torch::PackedTensorAccessor<long, 2, torch::RestrictPtrTraits, long> edge_batch,
                                               torch::PackedTensorAccessor<long, 1, torch::RestrictPtrTraits, long> batch_range){
    long batch_idx = blockIdx.x;
    long column = blockIdx.y * blockDim.y + threadIdx.y;

    if(column < hidden.size(1)){
        long start = batch_range[batch_idx], end = batch_range[batch_idx + 1];
        long dst = edge_batch[start][1];
        double sum = last_hidden[dst][column];
        for(long i = start; i < end; i++){
            long src = edge_batch[i][0], cur_dst = edge_batch[i][1];
            sum += hidden[src][column];
            if(dst != cur_dst){
                printf("Error: %d %d\n", dst, cur_dst);
            }
        }
        last_hidden[dst][column] = sum;
    }
}

void message_passing_forward(Tensor last_hidden, Tensor hidden, Tensor edge_batch, Tensor edge_batch_index){
    unsigned int edge_num = edge_batch.size(0);
    unsigned int batch_num = edge_batch_index.size(0) - 1;
    unsigned int hidden_dim = hidden.size(1);
    unsigned int thread_num = 32;
    if(hidden_dim >= 64)
        thread_num = 64;
    const dim3 threads(1, thread_num);
    const dim3 blocks(batch_num, (hidden_dim+thread_num-1)/thread_num);
    auto stream = c10::cuda::getCurrentCUDAStream(hidden.device().index()).stream();

    AT_DISPATCH_FLOATING_TYPES(hidden.scalar_type(), "message_passing_forward", ([&]{
        message_passing_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                hidden.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, long>(),
                last_hidden.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, long>(),
                edge_batch.packed_accessor<long, 2, torch::RestrictPtrTraits, long>(),
                edge_batch_index.packed_accessor<long, 1, torch::RestrictPtrTraits, long>()
        );
    }));
}
