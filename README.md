# Debug CUDA Kernel using libtorch

## build

``` shell
mdkir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DTORCH_PATH=<torch path>
make
```