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
__global__ void fused_lstm_partial_forward_kernel(
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, long> last_cell,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, long> last_hidden,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, long> cell,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, long> hidden,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, long> i_gate,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, long> f_gate,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, long> g_gate,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, long> o_gate,
        long rows_per_thread
){
    const long column = blockIdx.y * blockDim.y + threadIdx.y;
    long start_row = (blockIdx.x * blockDim.x + threadIdx.x) * rows_per_thread;
    long end_row = min(start_row + rows_per_thread, last_hidden.size(0));
    if (column < last_hidden.size(1)) {
        for (; start_row < end_row; start_row++) {
            double i = sigmoid_(i_gate[start_row][column]);
            double f = sigmoid_(f_gate[start_row][column]);
            double g = tanh_(g_gate[start_row][column]);
            double o = sigmoid_(o_gate[start_row][column]);
            double c = f * last_cell[start_row][column] + i * g;
            cell[start_row][column] = c;
            hidden[start_row][column] = o * tanh_(c);
        }
    }
}

void fused_lstm_partial_forward(Tensor last_cell, Tensor last_hidden,
                               Tensor cell, Tensor hidden,
                               Tensor i_gate, Tensor f_gate, Tensor g_gate, Tensor o_gate){
    unsigned int hidden_dim = last_hidden.size(1);
    unsigned int thread_num;
    const long rows_per_thread=8;
    if(hidden_dim < 32) thread_num = 32;
    else if(hidden_dim < 128) thread_num = 64;
    else thread_num = 128;
    auto stream = c10::cuda::getCurrentCUDAStream(hidden.device().index()).stream();
    const dim3 threads(1, thread_num);
    const dim3 blocks((hidden.size(0)+rows_per_thread-1)/rows_per_thread, (hidden_dim+thread_num-1)/thread_num);
    AT_DISPATCH_FLOATING_TYPES(last_hidden.scalar_type(), "fused_lstm_partial_forward", ([&]{
        fused_lstm_partial_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                last_cell.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, long>(),
                last_hidden.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, long>(),
                cell.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, long>(),
                hidden.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, long>(),
                i_gate.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, long>(),
                f_gate.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, long>(),
                g_gate.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, long>(),
                o_gate.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, long>(),
                rows_per_thread
        );
    }));
}

template<typename scalar_t>
__global__ void fused_lstm_partial_backward_kernel(
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, long> grad_cell,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, long> grad_hidden,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, long> grad_last_cell,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, long> grad_last_hidden,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, long> last_cell,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, long> last_hidden,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, long> i_gate,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, long> f_gate,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, long> g_gate,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, long> o_gate,
        long rows_per_thread
){
    const long column = blockIdx.y * blockDim.y + threadIdx.y;
    long start_row = (blockIdx.x * blockDim.x + threadIdx.x) * rows_per_thread;
    long end_row = min(start_row + rows_per_thread, last_hidden.size(0));
    if (column < last_hidden.size(1)) {
        for (; start_row < end_row; start_row++) {
            double i = sigmoid_(i_gate[start_row][column]);
            double f = sigmoid_(f_gate[start_row][column]);
            double g = tanh_(g_gate[start_row][column]);
            double o = sigmoid_(o_gate[start_row][column]);
            double c = f * last_cell[start_row][column] + i * g;
            double tanh_c = tanh_(c);

            o_gate[start_row][column] = grad_hidden[start_row][column] * tanh_c * d_sigmoid(o);
            double grad_c = grad_cell[start_row][column] + grad_hidden[start_row][column] * o * d_tanh(tanh_c);
            i_gate[start_row][column] = grad_c * g * d_sigmoid(i);
            f_gate[start_row][column] = grad_c * last_cell[start_row][column] * d_sigmoid(f);
            g_gate[start_row][column] = grad_c * i * d_tanh(g);
            grad_last_cell[start_row][column] = grad_c * f;
        }
    }
}

void fused_lstm_partial_backward(Tensor grad_cell, Tensor grad_hidden, Tensor grad_last_cell, Tensor grad_last_hidden,
                                 Tensor last_cell, Tensor last_hidden,
                                 Tensor i_gate, Tensor f_gate, Tensor g_gate, Tensor o_gate){
    unsigned int hidden_dim = last_hidden.size(1);
    unsigned int thread_num;
    const long rows_per_thread=8;
    if(hidden_dim < 32) thread_num = 32;
    else if(hidden_dim < 128) thread_num = 64;
    else thread_num = 128;
    auto stream = c10::cuda::getCurrentCUDAStream(grad_hidden.device().index()).stream();
    const dim3 threads(1, thread_num);
    const dim3 blocks((grad_cell.size(0)+rows_per_thread-1)/rows_per_thread, (hidden_dim+thread_num-1)/thread_num);
    AT_DISPATCH_FLOATING_TYPES(last_hidden.scalar_type(), "fused_lstm_partial_backward", ([&]{
        fused_lstm_partial_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                grad_cell.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, long>() ,
                grad_hidden.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, long>() ,
                grad_last_cell.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, long>() ,
                grad_last_hidden.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, long>(),
                last_cell.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, long>(),
                last_hidden.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, long>(),
                i_gate.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, long>(),
                f_gate.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, long>(),
                g_gate.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, long>(),
                o_gate.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, long>(),
                rows_per_thread
        );
    }));
}
