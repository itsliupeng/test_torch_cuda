#include "c10/cuda/CUDAStream.h"
#include "cuda_kernels.h"
#include <iostream>
#include <torch/script.h>


//torch::Tensor anchor_decode(const torch::TensorList in_tensor_list,
//                            torch::Tensor &anchor_grid,
//                            torch::TensorList grid_list,
//                            torch::Tensor &stride,
//                            int64_t na,
//                            int64_t no) {
//    std::vector<torch::Tensor> results;
//    results.reserve((in_tensor_list.size()));
//    for (int i = 0; i < in_tensor_list.size(); i++) {
//        auto xi = in_tensor_list[i];
//        auto y = xi.sigmoid();
//        auto in_shape = y.sizes();
//        int bs = in_shape[0];
//        int ny = in_shape[2];
//        int nx = in_shape[3];
//
//        y = y.view({bs, na, no, ny, nx}).permute({0, 1, 3, 4, 2}).contiguous();
//
//        using namespace torch::indexing;
//
//        auto a0 =
//                (y.index({Slice(), Slice(), Slice(), Slice(), Slice(None, 2)}) * 2.0 - 0.5 + grid_list[i].to(xi.device())) *
//                stride[i];
//        auto a1 = (y.index({Slice(), Slice(), Slice(), Slice(), Slice(2, 4)}) * 2).pow(2) * anchor_grid[i];
//        auto a2 = y.index({Slice(), Slice(), Slice(), Slice(), Slice(4)});
//
//        results.push_back(torch::cat({a0, a1, a2}, -1).view({bs, -1, no}));
//    }
//    return torch::cat(results, 1);
//}

torch::Tensor anchor_decode(const torch::Tensor in_tensor,
                            torch::Tensor anchor,
                            torch::Tensor grid,
                            float stride,
                            int64_t na,
                            int64_t no) {
    auto xi = in_tensor;
    auto y = xi.sigmoid();
    auto in_shape = y.sizes();
    int bs = in_shape[0];
    int ny = in_shape[2];
    int nx = in_shape[3];

    y = y.view({bs, na, no, ny, nx}).permute({0, 1, 3, 4, 2}).contiguous();

    using namespace torch::indexing;

    auto a0 = (y.index({Ellipsis, Slice(None, 2)}) * 2.0 - 0.5 + grid) * stride;
    auto a1 = (y.index({Ellipsis, Slice(2, 4)}) * 2).pow(2) * anchor;
    auto a2 = y.index({Ellipsis, Slice(4)});
    return torch::cat({a0, a1, a2}, -1).view({bs, -1, no});
}

using namespace torch::indexing;

int main() {
    int n = 1;
    int h = 3;
    int w = 3;
    int na = 3;
    int no = 85;
    auto in_tensor = torch::arange(n * 255 * h * w).reshape({{n, 255, h, w}}).to(torch::kFloat32);

    auto yx = torch::meshgrid({torch::arange(h), torch::arange(w)});

    torch::Tensor grid = torch::stack({yx[1], yx[0]}, 2).view({1, 1, h, w, 2}).to(torch::kFloat32);
    std::cout << "grid: " << grid << std::endl;

    auto anchor = torch::from_blob(std::vector<float>{10, 13, 16, 30, 33, 23}.data(), {1, na, 1, 1, 2}).to(torch::kFloat32);
    std::vector<float> strides({8, 16, 32});

    auto torch_result = anchor_decode(in_tensor, anchor, grid, strides[0], na, no);
    std::cout << "shape: " << torch_result.sizes() << std::endl
              << torch_result.index({Slice(), Slice(0, 10), Slice(0, 6)}) << std::endl;

    auto stream = c10::cuda::getCurrentCUDAStream(0);

    float *_mAnchor;
    size_t total_anchor_len = sizeof(float) * na * 2;
    C10_CUDA_CHECK(cudaMalloc(&_mAnchor, total_anchor_len));
    C10_CUDA_CHECK(cudaMemcpy(_mAnchor, anchor.data_ptr(), total_anchor_len, cudaMemcpyHostToDevice));

    auto a_in = in_tensor.view({n, na, no, h, w}).permute({0, 1, 3, 4, 2}).contiguous().cuda();
    auto a_out = torch::empty({n, na * h * w, no}).cuda();

    auto current_anchor = _mAnchor;
    anchor_decode_kernelLauncher((float *) a_out.data_ptr(), (float *) a_in.data_ptr(), n, na, no, h, w, current_anchor, strides[0], stream.stream());
    stream.synchronize();

    std::cout << "out tensor: " << a_out.index({Slice(), Slice(0, 10), Slice(0, 6)}) << std::endl;

    std::cout << "Done." << std::endl;


    return 0;
}
