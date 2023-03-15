#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <assert.h>
#include <math.h>

const int d_model = 512;
const int num_heads = 16;
const int head_dim = d_model / num_heads;
const int batch_size = 8;
const int seq_len = 64;

void check_cudnn_error(cudnnStatus_t status) {
    if (status != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "Error on line %d: %s\n", __LINE__, cudnnGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

void check_cublas_error(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Error on line %d: %d\n", __LINE__, status);
        exit(EXIT_FAILURE);
    }
}

void check_cuda_error(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "Error on line %d: %s\n", __LINE__, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

void multihead_attention(cudnnHandle_t cudnn, cublasHandle_t cublas, float *query, float *key, float *value, float *output, float *WQ, float *WK, float *WV, float *WO) {
    float *Q, *K, *V, *scores;
    size_t tensor_size = batch_size * seq_len * d_model * sizeof(float);
    cudaMalloc((void **)&Q, tensor_size);
    cudaMalloc((void **)&K, tensor_size);
    cudaMalloc((void **)&V, tensor_size);
    cudaMalloc((void **)&scores, batch_size * num_heads * seq_len * seq_len * sizeof(float));
    cudaMalloc((void **)&Q, tensor_size);
    cudaMalloc((void **)&K, tensor_size);
    cudaMalloc((void **)&V, tensor_size);
    cudaMalloc((void **)&scores, batch_size * num_heads * seq_len * seq_len * sizeof(float));

    float alpha = 1.0f, beta = 0.0f;
    for (int i = 0; i < num_heads; i++) {
        int offset = i * head_dim;
        cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, seq_len, batch_size, d_model, &alpha, query, seq_len, WQ + offset, d_model, &beta, Q + offset, seq_len);
        cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, seq_len, batch_size, d_model, &alpha, key, seq_len, WK + offset, d_model, &beta, K + offset, seq_len);
        cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, seq_len, batch_size, d_model, &alpha, value, seq_len, WV + offset, d_model, &beta, V + offset, seq_len);
    }

    cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, seq_len, seq_len, d_model, &alpha, Q, seq_len, K, seq_len, &beta, scores, seq_len);

    cudnnTensorDescriptor_t scores_desc;
    cudnnCreateTensorDescriptor(&scores_desc);
    cudnnSetTensor4dDescriptor(scores_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size * num_heads, 1, seq_len, seq_len);
    cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, scores_desc, scores, &beta, scores_desc, scores);

    float *context;
    cudaMalloc((void **)&context, tensor_size);

    cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, seq_len, batch_size, seq_len, &alpha, scores, seq_len, V, seq_len, &beta, context, seq_len);

    cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, seq_len, batch_size, d_model, &alpha, context, seq_len, WO, d_model, &beta, output, seq_len);

    cudaFree(Q);
    cudaFree(K);
    cudaFree(V);
    cudaFree(scores);
    cudaFree(context);
    cudnnDestroyTensorDescriptor(scores_desc);
}


int main() {
    cudnnHandle_t cudnn;
    check_cudnn_error(cudnnCreate(&cudnn));

    cublasHandle_t cublas;
    check_cublas_error(cublasCreate(&cublas));
    size_t tensor_size = batch_size * seq_len * d_model * sizeof(float);
    float *query, *key, *value, *output;
    cudaMalloc((void **)&query, tensor_size);
    cudaMalloc((void **)&key, tensor_size);
    cudaMalloc((void **)&value, tensor_size);
    cudaMalloc((void **)&output, tensor_size);

    // Initialize the input tensors (query, key, and value) with appropriate values

    size_t weights_size = d_model * d_model * sizeof(float);
    float *WQ, *WK, *WV, *WO;
    cudaMalloc((void **)&WQ, weights_size);
    cudaMalloc((void **)&WK, weights_size);
    cudaMalloc((void **)&WV, weights_size);
    cudaMalloc((void **)&WO, weights_size);

    // Initialize the weight matrices with appropriate values
 

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    check_cuda_error(cudaEventCreate(&start));
    check_cuda_error(cudaEventCreate(&stop));

    // Record the start event
    check_cuda_error(cudaEventRecord(start, 0));

    multihead_attention(cudnn, cublas, query, key, value, output, WQ, WK, WV, WO);

    // Record the stop event
    check_cuda_error(cudaEventRecord(stop, 0));
    check_cuda_error(cudaEventSynchronize(stop));

    // Calculate elapsed time in milliseconds
    float elapsedTime;
    check_cuda_error(cudaEventElapsedTime(&elapsedTime, start, stop));

    printf("Time elapsed for multihead_attention: %f ms\n", elapsedTime);
    // Process the output tensor as needed (e.g., print, save to a file, etc.)
 

    cudaFree(query);
    cudaFree(key);
    cudaFree(value);
    cudaFree(output);
    cudaFree(WQ);
    cudaFree(WK);
    cudaFree(WV);
    cudaFree(WO);

    // Destroy CUDA events
    check_cuda_error(cudaEventDestroy(start));
    check_cuda_error(cudaEventDestroy(stop));

    cudnnDestroy(cudnn);
    cublasDestroy(cublas);

    return 0;
}

        
