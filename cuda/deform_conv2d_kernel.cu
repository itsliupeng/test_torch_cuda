/**
* deformable convolution
*/
#include "cuda_kernels.h"
#define CUDA_1D_KERNEL_LOOP(i, n) \
    for (int i = (blockIdx.x * blockDim.x) + threadIdx.x; i < (n); i += (blockDim.x * gridDim.x))

template<typename T>
__global__ void add_bias(T *x, const T *bias, int n) {
    const int bid = blockIdx.x;
    auto b = bias[bid];
    for (int tid = threadIdx.x; tid < n; tid += blockDim.x)
        x[bid * n + tid] += b;
}

// [channel, batch, H, W] x + [channel] bias
template<typename T>
void add_bias_kernelLauncher(T *x, const T *bias, int channel, int batch, int H, int W, cudaStream_t stream) {
    dim3 grid(channel);
    int n = W * H * batch;
    int blockSize = n;

    if (blockSize > 1024)
        blockSize = 1024;
    add_bias<<<grid, blockSize, 0, stream>>>(x, bias, n);
}


template<typename T>
__global__ void transpose(T *output, const T *input, int n) {
    int c = blockIdx.y;
    int bs = blockIdx.x;
    for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
        output[(bs * gridDim.y + c) * n + tid] = input[(c * gridDim.x + bs) * n + tid];
    }
}

// [c, n, h, w] => [n, c, h, w]
template<typename T>
void transpose_kernelLauncher(T *output, const T *input, int bs, int c, int h, int w, cudaStream_t stream) {
    dim3 grid(bs, c);
    int n = w * h;
    int blockSize = n;
    if (std::is_same<T, half>::value && n % 2 == 0) {
        blockSize /= 2;
        if (blockSize > 1024) {
            blockSize = 1024;
        }
        transpose<<<grid, blockSize, 0, stream>>>((half2 *) output, (const half2 *) input, n / 2);
    } else {
        if (blockSize > 1024) {
            blockSize = 1024;
        }
        transpose<<<grid, blockSize, 0, stream>>>(output, input, n);
    }
}


template<typename T>
__device__ T bilinear_interpolate(const T *in, int height, int width, T h, T w) {
    if (h <= T(-1) || T(height) <= h || w <= T(-1) || T(width) <= w) {
        return T(0);
    }

    int h_low = floor((float) h);

    int w_low = floor((float) w);
    int h_high = h_low + 1;
    int w_high = w_low + 1;

    T lh = h - T(h_low);
    T lw = w - T(w_low);
    T hh = T(1) - lh, hw = T(1) - lw;

    T v1 = 0;
    if (h_low >= 0 && w_low >= 0)
        v1 = in[h_low * width + w_low];
    T v2 = 0;
    if (h_low >= 0 && w_high <= width - 1)
        v2 = in[h_low * width + w_high];
    T v3 = 0;
    if (h_high <= height - 1 && w_low >= 0)
        v3 = in[h_high * width + w_low];
    T v4 = 0;
    if (h_high <= height - 1 && w_high <= width - 1)
        v4 = in[h_high * width + w_high];

    T w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

    T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}

template<typename T>
__global__ void deformable_im2col_kernel(
        int n,
        const T *input_ptr,
        const T *offset_ptr,
        const T *mask_ptr,
        int height,
        int width,
        int weight_h,
        int weight_w,
        int pad_h,
        int pad_w,
        int stride_h,
        int stride_w,
        int dilation_h,
        int dilation_w,
        int batch_sz,
        int n_in_channels,
        int n_offset_grps,
        int out_h,
        int out_w,
        bool use_mask,
        T *columns_ptr) {
    CUDA_1D_KERNEL_LOOP(index, n) {
        const int out_x = index % out_w;
        const int out_y = (index / out_w) % out_h;
        const int out_b = (index / (out_w * out_h)) % batch_sz;
        const int in_c = index / (out_w * out_h * batch_sz);
        const int out_c = in_c * weight_h * weight_w;

        int c_per_offset_grp = n_in_channels / n_offset_grps;
        const int grp_idx = in_c / c_per_offset_grp;

        columns_ptr += (out_c * (batch_sz * out_h * out_w) + out_b * (out_h * out_w) + out_y * out_w + out_x);

        input_ptr += (out_b * (n_in_channels * height * width) + in_c * (height * width));

        offset_ptr += (out_b * n_offset_grps + grp_idx) * 2 * weight_h * weight_w * out_h * out_w;

        if (use_mask) {
            mask_ptr += (out_b * n_offset_grps + grp_idx) * weight_h * weight_w * out_h * out_w;
        }

        for (int i = 0; i < weight_h; ++i) {
            for (int j = 0; j < weight_w; ++j) {
                const int mask_idx = i * weight_w + j;
                const int offset_idx = 2 * mask_idx;

                T mask_value = 1;
                if (use_mask) {
                    mask_value = mask_ptr[mask_idx * (out_h * out_w) + out_y * out_w + out_x];
                }

                const T offset_h = offset_ptr[offset_idx * (out_h * out_w) + out_y * out_w + out_x];
                const T offset_w = offset_ptr[(offset_idx + 1) * (out_h * out_w) + out_y * out_w + out_x];
                const T y = T(out_y * stride_h - pad_h) + T(i * dilation_h) + offset_h;
                const T x = T(out_x * stride_w - pad_w) + T(j * dilation_w) + offset_w;
                *columns_ptr = mask_value * bilinear_interpolate(input_ptr, height, width, y, x);
                columns_ptr += batch_sz * out_h * out_w;
            }
        }
    }
}

// input, weight, output are row-major
template<typename T>
void gemm(
        T *C,
        const T *A,
        const T *B,
        const int m,
        const int n,
        const int k,
        const int lda,
        const int ldb,
        const int ldc,
        cublasOperation_t trans_a,
        cublasOperation_t trans_b,
        cublasHandle_t cublas_handle,
        float scale = 1.0f) {
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
    cublasGemmEx(
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
            B,
            Btype,
            ldb,
            beta,
            C,
            Ctype,
            ldc,
            computeType,
            static_cast<cublasGemmAlgo_t>(cublasAlgo));
}

template<typename T>
void deform_conv2d_kernel_launcher(
        T *output_ptr,
        T *tmp_output_ptr,
        T *columns_ptr,
        const T *input_ptr,
        const T *offset_ptr,
        const T *mask_ptr,
        const T *weight_ptr,
        const T *bias_ptr,
        int bs,
        int in_h,
        int in_w,
        int out_c,
        int in_c,
        int kernel_h,
        int kernel_w,
        int pad_h,
        int pad_w,
        int stride_h,
        int stride_w,
        int dilation_h,
        int dilation_w,
        int offset_groups,
        int out_h,
        int out_w,
        bool use_mask,
        cublasHandle_t mCublas,
        cudaStream_t stream) {
    int num_kernels = in_c * bs * out_h * out_w;
    const unsigned int threads = 512;
    const unsigned int blocks = (num_kernels + threads - 1) / threads;

    deformable_im2col_kernel<<<blocks, threads, 0, stream>>>(
            num_kernels,
            (const T *) input_ptr,
            (const T *) offset_ptr,
            (const T *) mask_ptr,
            in_h,
            in_w,
            kernel_h,
            kernel_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            bs,
            in_c,
            offset_groups,
            out_h,
            out_w,
            use_mask,
            (T *) columns_ptr);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in deformable_im2col: %s\n", cudaGetErrorString(err));
    }

    int m = out_c;
    int n = bs * out_h * out_w;
    int k = in_c * kernel_h * kernel_w;
    gemm((T *) tmp_output_ptr, (T *) columns_ptr, (T *) weight_ptr, n, m, k, n, k, n, CUBLAS_OP_N, CUBLAS_OP_N, mCublas);

    cudaError_t gemm_err = cudaGetLastError();
    if (gemm_err != cudaSuccess) {
        printf("error in gemm: %s\n", cudaGetErrorString(gemm_err));
    }

    //output [out_c, bs, out_h, out_w]
    add_bias_kernelLauncher((T *) tmp_output_ptr, (const T *) bias_ptr, out_c, bs, out_h, out_w, stream);
    cudaError_t bias_err = cudaGetLastError();
    if (bias_err != cudaSuccess) {
        printf("error in add_bias_kernelLauncher: %s\n", cudaGetErrorString(bias_err));
    }

    // transpose [b, c, h, w]
    transpose_kernelLauncher((T *) output_ptr, (const T *) tmp_output_ptr, bs, out_c, out_h, out_w, stream);
    cudaError_t transpose_err = cudaGetLastError();
    if (transpose_err != cudaSuccess) {
        printf("error in transpose_kernelLauncher: %s\n", cudaGetErrorString(transpose_err));
    }
}
template void deform_conv2d_kernel_launcher(
        float *output_ptr,
        float *tmp_output_ptr,
        float *columns_ptr,
        const float *input_ptr,
        const float *offset_ptr,
        const float *mask_ptr,
        const float *weight_ptr,
        const float *bias_ptr,
        int bs,
        int in_h,
        int in_w,
        int out_c,
        int in_c,
        int kernel_h,
        int kernel_w,
        int pad_h,
        int pad_w,
        int stride_h,
        int stride_w,
        int dilation_h,
        int dilation_w,
        int offset_groups,
        int out_h,
        int out_w,
        bool use_mask,
        cublasHandle_t mCublas,
        cudaStream_t stream);
