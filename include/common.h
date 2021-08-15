/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
#include <iostream>
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "cudnn.h"
#include <vector>
#include <map>

using namespace std;

static const char *_cudaGetErrorEnum(cudaError_t error)
{
  return cudaGetErrorString(error);
}


static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
  switch (error)
  {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";

  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";

  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";

  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";

  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";

  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";

  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";

  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";

  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";

  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "<unknown>";
}

template <typename T>
void check(T result, char const *const func, const char *const file, int const line)
{
  if (result)
  {
    throw std::runtime_error(std::string("[ERROR] CUDA runtime error: ") +
                             (_cudaGetErrorEnum(result)) + " " + file +
                             ":" + std::to_string(line) + " \n");
  }
}

#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)


#define checkCUDNN(expression)                               \
    {                                                          \
        cudnnStatus_t status = (expression);                     \
        if (status != CUDNN_STATUS_SUCCESS) {                    \
            std::cerr << "Error on file " << __FILE__ << " line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
        std::exit(EXIT_FAILURE);                               \
        }                                                        \
    }

#define TI(tag) \
    cudaEvent_t _event_start_ ##tag; \
    cudaEvent_t _event_end_ ##tag; \
    float _event_time_ ##tag; \
    cudaEventCreate(& _event_start_ ##tag); \
    cudaEventCreate(& _event_end_ ##tag); \
    cudaEventRecord(_event_start_ ##tag);

#define TO(tag, str, times) \
    cudaEventRecord(_event_end_ ##tag); \
    cudaEventSynchronize(_event_end_ ##tag); \
    cudaEventElapsedTime(&_event_time_ ##tag, _event_start_ ##tag, _event_end_ ##tag); \
    float _event_time_once_ ##tag = _event_time_ ##tag / times; \
    printf("%10s:\t %10.3fus\t", str, _event_time_once_ ##tag * 1000); \
    cudaDeviceSynchronize(); \
    printf("%10s string: %s\n",str,  cudaGetErrorString(cudaGetLastError()));


enum class AllocatorType
{
  CUDA,
  TF,
  TH
};

template <typename T>
class SwinTransformerTrait;

template <> class SwinTransformerTrait<half>
{
  public:
    using DataType = half;
};

template <> class SwinTransformerTrait<float>
{
  public:
    using DataType = float;
};

template <typename T>
class SwinTransformerBlockParam
{
public:
  const T *block_norm_gamma = nullptr;
  const T *block_norm_beta = nullptr;
  const T *block_norm2_gamma = nullptr;
  const T *block_norm2_beta = nullptr;
  const T *attention_qkv_kernel = nullptr;
  const T *attention_qkv_bias = nullptr;
  const T *attention_relative_pos_bias = nullptr;
  const T *attention_proj_kernel = nullptr;
  const T *attention_proj_bias = nullptr;
  const T *mlp_linear_kernel = nullptr;
  const T *mlp_linear_bias = nullptr;
  const T *mlp_linear2_kernel = nullptr;
  const T *mlp_linear2_bias = nullptr;
}; //SwinTransformerBlockParam

template <typename T>
class SwinTransformerBasicLayerParam
{
public:
  const T *patchMerge_norm_gamma = nullptr;
  const T *patchMerge_norm_beta = nullptr;
  const T *patchMerge_linear_kernel = nullptr;
  const T *attn_mask = nullptr;
  vector<SwinTransformerBlockParam<T>> block_param_list;
}; //SwinTransformerBasicLayerParam

template <typename T>
class SwinTransformerParam
{
public:
  const T *x = nullptr;
  const T *patchEmbed_proj_kernel = nullptr;
  const T *patchEmbed_proj_bias = nullptr;
  const T *patchEmbed_norm_gamma = nullptr;
  const T *patchEmbed_norm_beta = nullptr;
  const T *norm_gamma = nullptr;
  const T *norm_beta = nullptr;
  vector<SwinTransformerBasicLayerParam<T>> basic_layer_param_list; 
  T* output = nullptr;
  cudnnHandle_t cudnn_handle;
  cublasHandle_t cublas_handle;
  cudaStream_t stream;
}; //class SwinTransformerParam

