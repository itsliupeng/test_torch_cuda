#include <iostream>
#include <torch/script.h>
#include "cuda_kernels.h"
#include "c10/cuda/CUDAStream.h"

//def shift(x, n_segment, fold_div=3, inplace=False):
//    nt, c, h, w = x.size()
//    n_batch = int(nt / n_segment)
//    x = x.view(n_batch, n_segment, c, h, w)
//    fold = int(c / fold_div)
//    left_side = torch.cat((x[:, 1:, :fold], torch.zeros(n_batch, 1, fold, h, w).to(x.device)), dim=1)
//    middle_side = torch.cat((torch.zeros(n_batch, 1, fold, h, w).to(x.device), x[:, :n_segment - 1, fold: 2 * fold]), dim=1)
//    out = torch.cat((left_side, middle_side, x[:, :, 2 * fold:]), dim=2)
//    return out.view(nt, c, h, w)

torch::Tensor shift(torch::Tensor in, int n_segment, int fold_div) {
    auto in_shape = in.sizes();
    int nt = in_shape.at(0);
    int c = in_shape.at(1);
    int h = in_shape.at(2);
    int w = in_shape.at(3);
    auto n_batch = nt / n_segment;
    auto x = in.view({n_batch, n_segment, c, h, w});
    auto fold = c / fold_div;



    using namespace torch::indexing;
    auto left_side = torch::cat({x.index({Slice(), Slice(1), Slice(0, fold)}),
                                 torch::zeros({n_batch, 1, fold, h, w}, x.options())}, 1);

    auto middle_side = torch::cat({
                                        torch::zeros({n_batch, 1, fold, h, w}, x.options()),
                                        x.index({Slice(), Slice(0, n_segment-1), Slice(fold, fold*2)})
                                  }, 1);
    auto out = torch::cat({left_side, middle_side, x.index({Slice(), Slice(),  Slice(2*fold)})}, 2);
    return out.view({nt, c, h, w});

}

int main() {
    int nt = 2;
    int c = 3;
    int h = 2;
    int w = 2;
    int n_segment = nt;
    int fold_div = c;
    auto in = torch::arange(nt * c * h * w, {at::kCUDA}).reshape({nt, c, h, w}).to({at::kFloat});
    auto out = torch::zeros(in.sizes(), in.options());
    auto right_out = shift(in, n_segment, fold_div);
    std::cout << "in tensor: " << in << std::endl;
    std::cout << "right_out: " << right_out << std::endl;
    auto stream = c10::cuda::getCurrentCUDAStream(0);

    temporal_shift_kernelLauncher(out.data_ptr<float>(), in.data_ptr<float>(), nt, c, h, w, n_segment, fold_div, stream.stream());
    std::cout << "out tensor: " << out << std::endl;

    std::cout << "Done." << std::endl;
    return 0;
}

