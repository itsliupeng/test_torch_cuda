/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cuda_kernels.h"

/**
 * def shift(x, n_segment, fold_div=3):
    nt, c, h, w = x.size()
    n_batch = int(nt / n_segment)
    x = x.view(n_batch, n_segment, c, h, w)
    fold = int(c / fold_div)
    left_side = torch.cat((x[:, 1:, :fold], torch.zeros(n_batch, 1, fold, h, w).to(x.device)), dim=1)
    middle_side = torch.cat((torch.zeros(n_batch, 1, fold, h, w).to(x.device), x[:, :n_segment - 1, fold: 2 * fold]), dim=1)
    out = torch.cat((left_side, middle_side, x[:, :, 2 * fold:]), dim=2)
    return out.view(nt, c, h, w)
 */

// grid(c, n_segment, n_batch)
// block(w*h)
template <typename T>
__global__ void temporal_shift(T *output, const T *input, int n_segment,
                               int fold) {
  // input (n_batch, n_segment, c, h, w)
  const size_t bid =
      (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  const size_t tid = bid * blockDim.x + threadIdx.x;

  if (blockIdx.x < 2 * fold) {
    size_t shift_bid;
    if (blockIdx.x < fold && blockIdx.y >= 1) {
      // left
      shift_bid = (blockIdx.z * gridDim.y + (blockIdx.y - 1)) * gridDim.x + blockIdx.x;
      output[shift_bid * blockDim.x + threadIdx.x] = input[tid];
    } else if (blockIdx.x >= fold && blockIdx.y < n_segment - 1) {
      // middle
      shift_bid = (blockIdx.z * gridDim.y + (blockIdx.y + 1)) * gridDim.x + blockIdx.x;
      output[shift_bid * blockDim.x + threadIdx.x] = input[tid];
    } else {
//        output[tid] = input[tid];
//      output[tid] = (T)0.0f;
    }
  } else {
    output[tid] = input[tid];
  }
//  __syncthreads();
}

// grid(c, n_segment, n_batch)
// block(w*h): < 1024
template <typename T>
void temporal_shift_kernelLauncher(T *output, T *input, int nt, int c, int h, int w, int n_segment, int fold_div, cudaStream_t stream) {
    int n_batch = int(nt / n_segment);
    dim3 grid(c, n_segment, n_batch);
    int blocSize = h * w;
    int fold = int(c / fold_div);
    if (std::is_same<T, half>::value) {
        temporal_shift<<<grid, blocSize, 0, stream>>>((half2 *)output, (const half2 *)input, n_segment, fold);
    } else {
        temporal_shift<<<grid, blocSize, 0, stream>>>(output, input, n_segment, fold);
    }
}


template void temporal_shift_kernelLauncher(float *output, float *input, int nt, int c, int h, int w, int n_segment, int fold_div, cudaStream_t stream);
template void temporal_shift_kernelLauncher(half *output, half *input, int nt, int c, int h, int w, int n_segment, int fold_div, cudaStream_t stream);

//void temporal_shift_kernelLauncher(float *output, float *input, int nt, int c, int h, int w, int n_segment, int fold_div, cudaStream_t stream) {
//    int n_batch = int(nt / n_segment);
//    dim3 grid(n_batch, n_segment, c);
//    int blocSize = h * w;
//    int fold = int(c / fold_div);
//    temporal_shift<<<grid, blocSize, 0, stream>>>(output, input, n_segment, fold);
//}


template <typename T>
__global__ void focus(T *output, const T *input, int h, int w) {
    const size_t bid = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t tid = threadIdx.x;
    const size_t si = bid * blockDim.x + tid;

    bool ix = tid % 2 == 0;
    bool iy = blockIdx.x % 2 == 0;

    // tid < w, blockIdx.x  < h

    size_t dst_bid;
    if (ix && iy) {
        dst_bid = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x / 2 + blockIdx.x / 2;
    } else if ((!ix) && iy) {
        dst_bid = ((blockIdx.z+1) * gridDim.y +  blockIdx.y) * gridDim.x / 2 + blockIdx.x / 2;
    } else if (ix && !iy) {
        dst_bid = ((blockIdx.z+2) * gridDim.y + blockIdx.y) * gridDim.x / 2 + blockIdx.x / 2;
    } else {
        dst_bid = ((blockIdx.z+3) * gridDim.y + blockIdx.y) * gridDim.x / 2 + blockIdx.x / 2;
    }

   auto di = dst_bid * (blockDim.x / 2) + tid / 2;
   output[di] = input[si];
}

template <typename T>
void focus_kernelLauncher(T* output, T* input, int n, int c, int h, int w, cudaStream_t stream) {
    dim3 grid(h, c, n);
    dim3 block(w);
    assert(w <= 1024);
    focus<<<grid, block, 0, stream>>>(output, input, h, w);
}

template void focus_kernelLauncher(float* output, float* input, int n, int c, int h, int w, cudaStream_t stream);
template void focus_kernelLauncher(half* output, half* input, int n, int c, int h, int w, cudaStream_t stream);
template void focus_kernelLauncher(half2* output, half2* input, int n, int c, int h, int w, cudaStream_t stream);

__device__ float sigmoid(float data) {
    return 1.0f / (1.0f + expf(-data));
};

template <typename T>
__global__ void anchor_decode(T* output, const T* input, int w, T* anchors, T stride, int N) {
    if (threadIdx.x >= N)
        return;

    auto sid = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * N + threadIdx.x;
    auto y = sigmoid(input[sid]);

    //  pytorch:
    //   y = x[i].sigmoid()
    //   y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
    //   y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
    // v5: https://github.com/ultralytics/yolov5/issues/471

    if (threadIdx.x < 4) {
        int row = blockIdx.x / w;
        int col = blockIdx.x % w;
        int anchor_idx = blockIdx.y;
        if (threadIdx.x == 0) {
            y = (y * 2.0f - 0.5f + col) * stride;
        } else if (threadIdx.x == 1) {
            y = (y * 2.0f - 0.5f + row) * stride;
        } else if (threadIdx.x == 2) {
            y = powf((y * 2.0f), 2.0f) * anchors[2 * anchor_idx];
        } else {
            y = powf((y * 2.0f), 2.0f) * anchors[2 * anchor_idx + 1];
        }
    }
    output[sid] = y;
}

void anchor_decode_kernelLauncher(float *output, const float *input, int n, int na, int no, int h, int w, float *anchors, float stride, cudaStream_t stream) {
    int TPB = (no + 32 - 1) / 32 * 32;
    assert(TPB <= 1024);
    dim3 grid(w * h, na, n);

    anchor_decode<<<grid, TPB, 0, stream>>>(
            (float *) output, (const float *) input, w, anchors, stride, no);
}