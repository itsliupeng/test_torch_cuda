#include "c10/cuda/CUDAStream.h"
#include "cuda_kernels.h"
#include <iostream>
#include <ops/deform_conv2d.h>
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
    int bs = 1;
    int in_c = 3;
    int in_h = 2;
    int in_w = 2;

    int kernel = 3;
    int pad = 1;
    int stride = 1;
    int dilation = 1;

    int out_c = in_c;
    int weight_groups = 1;
    int offset_groups = 1;

    // [n, c, h, w]
    auto dilation_kernel = dilation * (kernel - 1) + 1;
    auto out_h = (in_h + 2 * pad - dilation_kernel) / stride + 1;
    auto out_w = (in_w + 2 * pad - dilation_kernel) / stride + 1;

    auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);

    auto in = torch::arange(bs * in_c * in_h * in_w, options).reshape({bs, in_c, in_h, in_w});
    auto offset = torch::ones({bs, offset_groups * kernel * kernel * 2, out_h, out_w}, options);
    auto mask = torch::ones({bs, offset_groups * kernel * kernel, out_h, out_w}, options);

    auto weight = torch::ones({out_c, in_c, kernel, kernel}, options);
    auto bias = torch::zeros({out_c}, options);

    bool use_mask = true;
    auto vision_out = vision::ops::deform_conv2d(in, weight, offset, mask, bias, stride, stride, pad, pad, dilation, dilation, weight_groups, offset_groups, use_mask);


    std::cout << "input: \n"
              << in << std::endl;

    std::cout << "vision out: \n"
              << vision_out << std::endl;


    auto dcn_out = torch::zeros({bs, out_c, out_h, out_w}, options);
    auto column = torch::zeros({in_c, kernel, kernel, bs, out_h, out_w}, options);

    cublasHandle_t cublas_handle;
    check_cuda_error(cublasCreate(&cublas_handle));
    cudaStream_t stream = torch::cuda::getCurrentCUDAStream().stream();
    check_cuda_error(cublasSetStream(cublas_handle, stream));


    deform_conv2d_kernel_launcher(dcn_out.data_ptr<float>(), column.data_ptr<float>(), in.data_ptr<float>(), mask.data_ptr<float>(), offset.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
            in.size(0), in.size(2), in.size(3),weight.size(0), weight.size(1), weight.size(2), weight.size(3),
                                  pad, pad, stride, stride, dilation, dilation, offset_groups, dcn_out.size(2), dcn_out.size(3), use_mask, cublas_handle, stream);


    std::cout << "dcn out: \n"
              << dcn_out << std::endl;

    auto diff = torch::abs(vision_out - dcn_out);
    std::cout << "diff max " << diff.max() << ", sum " << diff.sum() << std::endl;

    return 0;
}
