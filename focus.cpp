#include <iostream>
#include <torch/script.h>
#include "cuda_kernels.h"
#include "c10/cuda/CUDAStream.h"

torch::Tensor shift(torch::Tensor in) {
    auto in_shape = in.sizes();
    int n = in_shape.at(0);
    int c = in_shape.at(1);
    int h = in_shape.at(2);
    int w = in_shape.at(3);
}

int main() {
    int nt = 1;
    int c = 1;
    int h = 4;
    int w = 4;
    auto in = torch::arange(nt * c * h * w, {at::kCUDA}).reshape({nt, c, h, w}).to({at::kHalf});
    auto out = torch::zeros({nt, 4 * c, h / 2, w / 2}, in.options());
    std::cout << "in tensor: " << in << std::endl;
    auto stream = c10::cuda::getCurrentCUDAStream(0);

    focus_kernelLauncher((half*)out.data_ptr<at::Half>(), (half*)in.data_ptr<at::Half>(), nt, c, h, w, stream.stream());
    std::cout << "out tensor: " << out << std::endl;

    std::cout << "Done." << std::endl;
    return 0;
}

