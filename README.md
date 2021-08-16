# Debug CUDA Kernel using libtorch

## build

#### dependency
- build C++ torchvision https://github.com/pytorch/vision#c-api

``` shell
mdkir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DTORCH_PATH=<torch path>
make
```