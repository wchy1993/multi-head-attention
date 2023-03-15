#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "multihead_attention.h"

// 定义一个宏来检查和处理CUDA错误
#define CHECK_CUDA_ERROR(err)                                 \
    do {                                                      \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n",      \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(1);                                          \
        }                                                     \
    } while (0)


// init_multihead_attention, destroy_multihead_attention,




// 初始化多头注意力结构的函数
void init_multihead_attention(MultiHeadAttention *mha, int d_model, int num_heads) {
    mha->d_model = d_model;
    mha->num_heads = num_heads;
    mha->head_dim = d_model / num_heads;

    // 为权重矩阵分配内存
    cudaMalloc((void **)&mha->wq_weights, d_model * d_model * sizeof(float));
    cudaMalloc((void **)&mha->wk_weights, d_model * d_model * sizeof(float));
    cudaMalloc((void **)&mha->wv_weights, d_model * d_model * sizeof(float));
    cudaMalloc((void **)&mha->wo_weights, d_model * d_model * sizeof(float));

    // 检查是否有CUDA错误
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// 释放多头注意力结构的函数
void free_multihead_attention(MultiHeadAttention *mha) {
    // 释放权重矩阵的内存
    cudaFree(mha->wq_weights);
    cudaFree(mha->wk_weights);
    cudaFree(mha->wv_weights);
    cudaFree(mha->wo_weights);

    // 检查是否有CUDA错误
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// 线性变换 CUDA 内核函数
__global__ void linear_transform_kernel(float *input, float *weights, float *output, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows && j < cols) {
        float value = 0;
        for (int k = 0; k < cols; k++) {
            value += input[i * cols + k] * weights[k * cols + j];
        }
        output[i * cols + j] = value;
    }
}

// 线性变换 主机函数
void linear_transform(float *d_input, float *d_weights, float *d_output, int rows, int cols, cudaStream_t stream) {
    dim3 blockDim(16, 16);
    dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);

    linear_transform_kernel<<<gridDim, blockDim, 0, stream>>>(d_input, d_weights, d_output, rows, cols);
    CHECK_CUDA_ERROR(cudaGetLastError());
}


__global__ void split_heads_kernel(float *input, float *output, int batch_size, int seq_len, int num_heads, int head_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < batch_size && j < seq_len) {
        for (int h = 0; h < num_heads; h++) {
            for (int d = 0; d < head_dim; d++) {
                int src_idx = i * seq_len * num_heads * head_dim + j * num_heads * head_dim + h * head_dim + d;
                int tgt_idx = i * num_heads * seq_len * head_dim + h * seq_len * head_dim + j * head_dim + d;
                output[tgt_idx] = input[src_idx];
            }
        }
    }
}

void split_heads(float *d_input, float *d_output, int batch_size, int seq_len, int num_heads, int head_dim, cudaStream_t stream) {
    dim3 blockDim(16, 16);
    dim3 gridDim((batch_size + blockDim.x - 1) / blockDim.x, (seq_len + blockDim.y - 1) / blockDim.y);

    split_heads_kernel<<<gridDim, blockDim, 0, stream>>>(d_input, d_output, batch_size, seq_len, num_heads, head_dim);
    CHECK_CUDA_ERROR(cudaGetLastError());
}


__global__ void scaled_dot_product_attention_kernel(float *q, float *k, float *v, float *output, int batch_size, int seq_len, int num_heads, int head_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < batch_size * num_heads && j < seq_len) {
        float sum = 0;
        for (int t = 0; t < seq_len; t++) {
            float dot = 0;
            for (int d = 0; d < head_dim; d++) {
                dot += q[i * seq_len * head_dim + t * head_dim + d] * k[i * seq_len * head_dim + j * head_dim + d];
            }
            dot /= sqrtf((float)head_dim);
            float weight = expf(dot);
            sum += weight;
            output[i * seq_len * seq_len + t * seq_len + j] = weight;
        }

        for (int t = 0; t < seq_len; t++) {
            output[i * seq_len * seq_len + t * seq_len + j] /= sum;
        }
    }
}

void scaled_dot_product_attention(float *d_q, float *d_k, float *d_v, float *d_output, int batch_size, int seq_len, int num_heads, int head_dim, cudaStream_t stream) {
    dim3 blockDim(16, 16);
    dim3 gridDim((batch_size * num_heads + blockDim.x - 1) / blockDim.x, (seq_len + blockDim.y - 1) / blockDim.y);
    scaled_dot_product_attention_kernel<<<gridDim, blockDim, 0, stream>>>(d_q, d_k, d_v, d_output, batch_size, seq_len, num_heads, head_dim);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

__global__ void compute_weighted_sum_kernel(float *attention_weights, float *v, float *output, int batch_size, int seq_len, int num_heads, int head_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < batch_size * num_heads && j < seq_len) {
        for (int d = 0; d < head_dim; d++) {
            float sum = 0;
            for (int t = 0; t < seq_len; t++) {
                sum += attention_weights[i * seq_len * seq_len + t * seq_len + j] * v[i * seq_len * head_dim + t * head_dim + d];
            }
            output[i * seq_len * head_dim + j * head_dim + d] = sum;
        }
    }
}

void compute_weighted_sum(float *d_attention_weights, float *d_v, float *d_output, int batch_size, int seq_len, int num_heads, int head_dim, cudaStream_t stream) {
    dim3 blockDim(16, 16);
    dim3 gridDim((batch_size * num_heads + blockDim.x - 1) / blockDim.x, (seq_len + blockDim.y - 1) / blockDim.y);

    compute_weighted_sum_kernel<<<gridDim, blockDim, 0, stream>>>(d_attention_weights, d_v, d_output, batch_size, seq_len, num_heads, head_dim);
    CHECK_CUDA_ERROR(cudaGetLastError());
}


__global__ void merge_heads_kernel(float *input, float *output, int batch_size, int seq_len, int num_heads, int head_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < batch_size && j < seq_len) {
        for (int h = 0; h < num_heads; h++) {
            for (int d = 0; d < head_dim; d++) {
                int src_idx = i * num_heads * seq_len * head_dim + h * seq_len * head_dim + j * head_dim + d;
                int tgt_idx = i * seq_len * num_heads * head_dim + j * num_heads * head_dim + h * head_dim + d;
                output[tgt_idx] = input[src_idx];
            }
        }
    }
}

void merge_heads(float *d_input, float *d_output, int batch_size, int seq_len, int num_heads, int head_dim, cudaStream_t stream) {
    dim3 blockDim(16, 16);
    dim3 gridDim((batch_size + blockDim.x - 1) / blockDim.x, (seq_len + blockDim.y - 1) / blockDim.y);

    merge_heads_kernel<<<gridDim, blockDim, 0, stream>>>(d_input, d_output, batch_size, seq_len, num_heads, head_dim);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

void destroy_multihead_attention(MultiHeadAttention *mha) {
    // 释放权重内存
    cudaFree(mha->wq_weights);
    cudaFree(mha->wk_weights);
    cudaFree(mha->wv_weights);
    cudaFree(mha->wo_weights);
}



// linear_transform, split_heads, scaled_dot_product_attention,
// compute_weighted_sum, merge_heads




// main 函数
int main() {
    int batch_size = 32;
    int seq_len = 64;
    int d_model = 512;
    int num_heads = 8;
    int head_dim = d_model / num_heads;


    // 创建CUDA事件以记录GPU时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 记录开始时间
    cudaEventRecord(start, 0);

    // 为输入和输出张量分配设备内存
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, batch_size * seq_len * d_model * sizeof(float));
    cudaMalloc((void **)&d_output, batch_size * seq_len * d_model * sizeof(float));

    // 初始化多头注意力结构
    MultiHeadAttention mha;
    init_multihead_attention(&mha, d_model, num_heads);

    // 创建CUDA流以异步执行内核函数
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 1. 线性变换 (Q, K, V)
    float *d_q, *d_k, *d_v;
    cudaMalloc((void **)&d_q, batch_size * seq_len * d_model * sizeof(float));
    cudaMalloc((void **)&d_k, batch_size * seq_len * d_model * sizeof(float));
    cudaMalloc((void **)&d_v, batch_size * seq_len * d_model * sizeof(float));

    linear_transform(d_input, mha.wq_weights, d_q, batch_size * seq_len, d_model, stream);
    linear_transform(d_input, mha.wk_weights, d_k, batch_size * seq_len, d_model, stream);
    linear_transform(d_input, mha.wv_weights, d_v, batch_size * seq_len, d_model, stream);

    // 2. 拆分头
    float *d_q_split, *d_k_split, *d_v_split;
    cudaMalloc((void **)&d_q_split, batch_size * seq_len * num_heads * head_dim * sizeof(float));
    cudaMalloc((void **)&d_k_split, batch_size * seq_len * num_heads * head_dim * sizeof(float));
    cudaMalloc((void **)&d_v_split, batch_size * seq_len * num_heads * head_dim * sizeof(float));

    split_heads(d_q, d_q_split, batch_size, seq_len, num_heads, head_dim, stream);
    split_heads(d_k, d_k_split, batch_size, seq_len, num_heads, head_dim, stream);
    split_heads(d_v, d_v_split, batch_size, seq_len, num_heads, head_dim, stream);

    // 3. 计算注意力权重
    float *d_attention_weights;
    // 继续main函数
    cudaMalloc((void **)&d_attention_weights, batch_size * seq_len * num_heads * seq_len * sizeof(float));

    scaled_dot_product_attention(d_q_split, d_k_split, d_v_split, d_attention_weights, batch_size, seq_len, num_heads, head_dim, stream);

    // 4. 计算加权和
    float *d_weighted_sum;
    cudaMalloc((void **)&d_weighted_sum, batch_size * seq_len * num_heads * head_dim * sizeof(float));
    
    compute_weighted_sum(d_attention_weights, d_v_split, d_weighted_sum, batch_size, seq_len, num_heads, head_dim, stream);

    // 5. 合并头
    float *d_merged_heads;
    cudaMalloc((void **)&d_merged_heads, batch_size * seq_len * d_model * sizeof(float));

    merge_heads(d_weighted_sum, d_merged_heads, batch_size, seq_len, num_heads, head_dim, stream);

    // 6. 输出线性变换
    linear_transform(d_merged_heads, mha.wo_weights, d_output, batch_size * seq_len, d_model, stream);

    // 等待CUDA流中的所有操作完成
    cudaStreamSynchronize(stream);
    cudaEventRecord(stop, 0);

    // 等待GPU完成所有操作
    cudaEventSynchronize(stop);

    // 计算并输出执行时间
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Multi-head attention execution time: %f ms\n", milliseconds);

    // 销毁CUDA事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 清理
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_q_split);
    cudaFree(d_k_split);
    cudaFree(d_v_split);
    cudaFree(d_attention_weights);
    cudaFree(d_weighted_sum);
    cudaFree(d_merged_heads);

    destroy_multihead_attention(&mha);
    cudaStreamDestroy(stream);                                                                              
    return 0;
}