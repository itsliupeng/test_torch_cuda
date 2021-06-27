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
  __syncthreads();
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
