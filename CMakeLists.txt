cmake_minimum_required(VERSION 3.20)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(test_cuda LANGUAGES CXX CUDA)

function (set_ifndef variable value)
    if(NOT DEFINED ${variable})
        set(${variable} ${value} PARENT_SCOPE)
    endif()
endfunction()

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -G -Xcompiler -Wall -Wno-deprecated-declarations")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDA 10.2 REQUIRED)

find_library(CUDNN_LIB cudnn HINTS
        ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib)
find_library(CUBLAS_LIB cublas HINTS
        ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib lib/stubs)
find_library(CUBLASLT_LIB cublasLt HINTS
        ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib lib/stubs)

message(STATUS "CUDNN_ROOT_DIR ${CUDNN_ROOT_DIR},  CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR} CUBLAS_LIB ${CUBLAS_LIB} CUBLASLT_LIB ${CUBLASLT_LIB}")

# libtorch
set_ifndef(TORCH_PATH /home/liupeng/anaconda3/envs/py36/lib/python3.6/site-packages/torch)
set(CMAKE_PREFIX_PATH ${TORCH_PATH}/share/cmake)
find_package(Torch 1.7 REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

# torchvision
find_package(TorchVision REQUIRED)
set(TorchVision_INCLUDE_DIR ${TorchVision_INCLUDE_DIR}/torchvision)
set(TorchVision_LIBRARY /usr/local/lib/libtorchvision.so)
message(STATUS "TorchVision_INCLUDE_DIR ${TorchVision_INCLUDE_DIR}, TorchVision_LIBRARY: ${TorchVision_LIBRARY}")
include_directories(${TorchVision_INCLUDE_DIR})

include_directories(
        ${CUDA_INCLUDE_DIRS}
        ${CUDA_TOOLKIT_ROOT_DIR}/include
        ${TORCH_INCLUDE_DIRS}
)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(cuda)

add_executable(anchor_decode anchor_decode.cpp)
target_link_libraries(anchor_decode PUBLIC cuda_kernels ${TORCH_LIBRARIES})

add_executable(focus focus.cpp)
target_link_libraries(focus PUBLIC cuda_kernels ${TORCH_LIBRARIES})

add_executable(temporal_shift temporal_shift.cpp)
target_link_libraries(temporal_shift cuda_kernels ${TORCH_LIBRARIES})

add_executable(deform_conv2d deform_conv2d.cpp)
target_link_libraries(deform_conv2d cuda_kernels ${TORCH_LIBRARIES} ${TorchVision_LIBRARY})


#set_target_properties(
#        main
#        PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)