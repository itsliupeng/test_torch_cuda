#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "stdio.h"
#include "stdlib.h"
#include <cuda_runtime_api.h>

// input, weight, output are row-major
template<typename T>
void batch_gemm(
        T *C,
        const T *A,
        const T *B,
        const int batch,
        const int m,
        const int n,
        const int k,
        const int lda,
        const int ldb,
        const int ldc,
        const size_t stridea,
        const size_t strideb,
        const size_t stridec,
        cublasOperation_t trans_a,
        cublasOperation_t trans_b,
        cublasHandle_t cublas_handle,
        float scale = 1.0f) {
    printf("batch %d m %d n %d k %d\n", batch, m, n, k);
    cudaDataType_t Atype, Btype, Ctype, computeType;
    float alpha_float = scale;
    float beta_float = 0.0f;
    half alpha_half = half(scale);
    half beta_half = half(0.0f);
    void *alpha, *beta;
    int cublasAlgo;

    if (std::is_same<T, float>::value) {
        computeType = CUDA_R_32F;
        Atype = CUDA_R_32F;
        Btype = CUDA_R_32F;
        Ctype = CUDA_R_32F;
        alpha = &alpha_float;
        beta = &beta_float;
        cublasAlgo = CUBLAS_GEMM_DEFAULT;
    } else {
        computeType = CUDA_R_16F;
        Atype = CUDA_R_16F;
        Btype = CUDA_R_16F;
        Ctype = CUDA_R_16F;
        alpha = &alpha_half;
        beta = &beta_half;
        cublasAlgo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    }
    cublasGemmStridedBatchedEx(
            cublas_handle,
            trans_a,
            trans_b,
            m,
            n,
            k,
            alpha,
            A,
            Atype,
            lda,
            stridea,
            B,
            Btype,
            ldb,
            strideb,
            beta,
            C,
            Ctype,
            ldc,
            stridec,
            batch,
            computeType,
            static_cast<cublasGemmAlgo_t>(cublasAlgo));
}

int main(int argv, char *argc[]) {
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    int batch = atoi(argc[1]);
    int m = atoi(argc[2]);
    int n = atoi(argc[3]);
    int k = atoi(argc[4]);
    half *buffer;
    cudaMalloc((void**)&buffer, (m * k + k * n + m * n) * batch * sizeof(half));
    half *a = buffer;
    half *b = a + batch * m * k;
    half *c = b + batch * k * n;

    batch_gemm(
            c,
            a,
            b,
            batch,
            n, m, k, n, k, n, n * k, 0, n * m, CUBLAS_OP_N, CUBLAS_OP_N, cublas_handle);
    half h_c[100];
    cudaMemcpy(h_c, c, 1, cudaMemcpyDeviceToHost);
}
