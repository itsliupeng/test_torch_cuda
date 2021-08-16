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

#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <assert.h>
#include <cublas_v2.h>
#include <stdio.h>
#include "common.h"

template <typename T>
void temporal_shift_kernelLauncher(T *output, T *input, int nt, int c, int h, int w, int n_segment, int fold_div, cudaStream_t stream);

//void temporal_shift_kernelLauncher(float *output, float *input, int nt, int c, int h, int w, int n_segment, int fold_div, cudaStream_t stream);

template <typename T>
void focus_kernelLauncher(T* output, T* input, int n, int c, int h, int w, cudaStream_t stream);

void anchor_decode_kernelLauncher(float* output, const float* input, int n, int na, int no, int h, int w, float* anchors, float stride, cudaStream_t stream);

template <typename T>
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
        cudaStream_t stream);