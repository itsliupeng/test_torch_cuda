#include "c10/cuda/CUDAStream.h"
#include "cuda_kernels.h"
#include <iostream>
#include <torch/script.h>

torch::Tensor focus(torch::Tensor in) {
    auto x = in;

    using namespace torch::indexing;
    return torch::cat(
            {
                    x.index({Ellipsis, Slice(None, None, 2), Slice(None, None, 2)}),
                    x.index({Ellipsis, Slice(1, None, 2), Slice(None, None, 2)}),
                    x.index({Ellipsis, Slice(None, None, 2), Slice(1, None, 2)}),
                    x.index({Ellipsis, Slice(1, None, 2), Slice(1, None, 2)}),
            },
            1);
}

int main() {
    int nt = 1;
    int c = 1;
    int h = 8;
    int w = 8;
    auto in = torch::arange(nt * c * h * w, {at::kCUDA}).reshape({nt, c, h, w}).to({at::kHalf});
    std::cout << "in tensor: " << in << std::endl;

    auto out = torch::zeros({nt, 4 * c, h / 2, w / 2}, in.options());
    std::cout << "real out tensor: " << focus(in) << std::endl;

    auto stream = c10::cuda::getCurrentCUDAStream(0);

    focus_kernelLauncher((half *) out.data_ptr<at::Half>(), (half *) in.data_ptr<at::Half>(), nt, c, h, w, stream.stream());
    std::cout << "out tensor: " << out << std::endl;
    auto diff = torch::abs(focus(in) - out);
    std::cout << "diff max " << diff.max() << ", sum " << diff.sum() << std::endl;
    std::cout << "Done." << std::endl;
    return 0;
}
