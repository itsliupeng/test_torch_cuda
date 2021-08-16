#include "c10/cuda/CUDAStream.h"
#include "cuda_kernels.h"
#include <iostream>
#include <ops/deform_conv2d.h>
#include <torch/script.h>


#include <ops/deform_conv2d.h>
static auto registry_dcn = torch::RegisterOperators().op(
        "kinfer::deform_conv2d(Tensor input, Tensor weight, Tensor offset_mask, Tensor bias, int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, int groups, int offset_groups, bool use_mask) -> (Tensor)",
        [](const torch::Tensor& input,
           const torch::Tensor& weight,
           const torch::Tensor& offset_mask,
           const torch::Tensor& bias,
           int64_t stride_h,
           int64_t stride_w,
           int64_t pad_h,
           int64_t pad_w,
           int64_t dilation_h,
           int64_t dilation_w,
           int64_t groups,
           int64_t offset_groups,
           bool use_mask) {
          auto offset_mask_shape = offset_mask.sizes();
          int c = offset_mask_shape[1];
          at::Tensor offset;
          at::Tensor mask;

          if (use_mask) {
              TORCH_CHECK(c % 3 == 0, "offset_mask chanel should be divided by 3.")
              using namespace torch::indexing;
              auto offset_channel = c / 3 * 2;
              offset = offset_mask.index({Slice(), Slice(0, offset_channel), Slice(), Slice()});
              mask = offset_mask.index({Slice(), Slice(offset_channel), Slice(), Slice()}).sigmoid();
          } else {
              offset = offset_mask;
              mask = at::ones({offset_mask_shape[0], c / 3, offset_mask_shape[2], offset_mask_shape[3]}, offset.options());
          }

//          std::cout << "offset: \n" << offset << std::endl;
//          std::cout << "mask: \n" << mask << std::endl;

          return vision::ops::deform_conv2d(
                  input,
                  weight,
                  offset,
                  mask,
                  bias,
                  stride_h,
                  stride_w,
                  pad_h,
                  pad_w,
                  dilation_h,
                  dilation_w,
                  groups,
                  offset_groups,
                  use_mask);
        });


at::Tensor deform_conv2d(
        const at::Tensor& input,
        const at::Tensor& weight,
        const at::Tensor& offset_mask,
        const at::Tensor& bias,
        int64_t stride_h,
        int64_t stride_w,
        int64_t pad_h,
        int64_t pad_w,
        int64_t dilation_h,
        int64_t dilation_w,
        int64_t groups,
        int64_t offset_groups,
        bool use_mask) {
    static auto op = c10::Dispatcher::singleton()
            .findSchemaOrThrow("kinfer::deform_conv2d", "")
            .typed<decltype(deform_conv2d)>();
    return op.call(
            input,
            weight,
            offset_mask,
            bias,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            groups,
            offset_groups,
            use_mask);
}

int main() {
    int bs = 2;
    int in_c = 1;
    int in_h = 2;
    int in_w = in_h;

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

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);

    auto in = torch::arange(bs * in_c * in_h * in_w, options).reshape({bs, in_c, in_h, in_w});
    auto offset = torch::randn({bs, offset_groups * kernel * kernel * 2, out_h, out_w}, options);
    auto mask = torch::rand({bs, offset_groups * kernel * kernel, out_h, out_w}, options);

//    std::cout << "offset: \n" << offset << std::endl;
//    std::cout << "mask: \n" << mask << std::endl;

    auto weight = torch::ones({out_c, in_c, kernel, kernel}, options);
    auto bias = torch::zeros({out_c}, options);
//    auto bias = torch::rand({out_c}, options);

    bool use_mask = true;
//    auto vision_out = vision::ops::deform_conv2d(in, weight, offset, mask, bias, stride, stride, pad, pad, dilation, dilation, weight_groups, offset_groups, use_mask);

    auto offset_mask = at::cat({offset, mask}, 1);
    auto vision_out = deform_conv2d(in, weight, offset_mask, bias, stride, stride, pad, pad, dilation, dilation, weight_groups, offset_groups, use_mask);


    std::cout << "input: \n"
              << in << std::endl;

    std::cout << "vision out: \n"
              << vision_out << std::endl;


    auto dcn_out = torch::zeros({bs, out_c, out_h, out_w}, options);
    auto column = torch::zeros({weight_groups, in_c / weight_groups * kernel * kernel, bs, out_h, out_w}, options);

    cublasHandle_t cublas_handle;
    check_cuda_error(cublasCreate(&cublas_handle));
    cudaStream_t stream = torch::cuda::getCurrentCUDAStream().stream();
    check_cuda_error(cublasSetStream(cublas_handle, stream));

    auto tmp_dcn_out = at::zeros({out_c, bs, out_h, out_w}, dcn_out.options());


    std::cout << "offset_mask shape: \n"
              << offset_mask.sizes() << std::endl;

    deform_conv2d_kernel_launcher(
            dcn_out.data_ptr<float>(),
            tmp_dcn_out.data_ptr<float>(),
            column.data_ptr<float>(),
            in.data_ptr<float>(),
            offset.data_ptr<float>(),
            mask.data_ptr<float>(),
//            (float*)nullptr,
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            in.size(0),
            in.size(2),
            in.size(3),
            weight.size(0),
            weight.size(1),
            weight.size(2),
            weight.size(3),
            pad,
            pad,
            stride,
            stride,
            dilation,
            dilation,
            offset_groups,
            dcn_out.size(2),
            dcn_out.size(3),
            use_mask,
            cublas_handle,
            stream);


    std::cout << "dcn out: \n"
              << dcn_out << std::endl;

    auto diff = torch::abs(vision_out - dcn_out);
    std::cout << "diff max " << diff.max() << ", sum " << diff.sum() << std::endl;

    return 0;
}
