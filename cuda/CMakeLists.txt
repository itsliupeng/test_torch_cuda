add_library(cuda_kernels STATIC cuda_kernels.cu)
set_property(TARGET cuda_kernels PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cuda_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(cuda_kernels PUBLIC -lcublas -lcudart -lcurand)
