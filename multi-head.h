#ifndef MULTIHEAD_ATTENTION_H
#define MULTIHEAD_ATTENTION_H

#include <cuda_runtime.h>
typedef struct {
    int d_model;
    int num_heads;
    int head_dim;
    float *wq_weights;
    float *wk_weights;
    float *wv_weights;
    float *wo_weights;
} MultiHeadAttention;



void init_multihead_attention(MultiHeadAttention *mha, int d_model, int num_heads);
void destroy_multihead_attention(MultiHeadAttention *mha);

void linear_transform(float *d_input, float *d_weights, float *d_output, int n, int m, cudaStream_t stream);

void split_heads(float *d_input, float *d_output, int batch_size, int seq_len, int num_heads, int head_dim, cudaStream_t stream);

void scaled_dot_product_attention(float *d_q, float *d_k, float *d_v, float *d_attention_weights, int batch_size, int seq_len, int num_heads, int head_dim, cudaStream_t stream);

void compute_weighted_sum(float *d_attention_weights, float *d_v, float *d_output, int batch_size, int seq_len, int num_heads, int head_dim, cudaStream_t stream);

void merge_heads(float *d_input, float *d_output, int batch_size, int seq_len, int num_heads, int head_dim, cudaStream_t stream);

#endif // MULTIHEAD_ATTENTION_H
