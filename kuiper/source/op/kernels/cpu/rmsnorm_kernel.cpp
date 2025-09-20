#include "rmsnorm_kernel.h"

namespace kernel {
void rmsnorm_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                        const tensor::Tensor& output, void* stream) {
  UNUSED(stream);
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == base::DeviceType::kDeviceCPU &&
        weight.device_type() == base::DeviceType::kDeviceCPU &&
        output.device_type() == base::DeviceType::kDeviceCPU);

  // in_ptr和out_ptr分别是该算子输入输出数据指向的指针
  const float* in_ptr = input.ptr<float>();
  const float* wei_ptr = weight.ptr<float>();
  const float* out_ptr = output.ptr<float>();
  const int32_t dim = static_cast<int32_t>(input.size());

  arma::fvec in_tensor(const_cast<float*>(in_ptr), dim, false, true);
  arma::fvec out_tensor(const_cast<float*>(out_ptr), dim, false, true);
  arma::fvec wei_tensor(const_cast<float*>(wei_ptr), dim, false, true);

#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
  const float eps = 1e-6f;
#else
  const float eps = 1e-5f;
#endif

  const float mean = arma::as_scalar(arma::mean(arma::pow(in_tensor, 2))) + eps;
  const float rsqrt = 1.f / std::sqrt(mean);
  out_tensor = wei_tensor % (rsqrt * in_tensor);
}
}  // namespace kernel

/*不用数学库的实现
void rmsnorm_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
  const tensor::Tensor& output, void* stream) {

float* in_ptr = const_cast<float*>(input.ptr<float>());
float* out_ptr = const_cast<float*>(output.ptr<float>());

int size = static_cast<int32_t>(input.size());
float sum = 0.f;
for (int i = 0; i < size; ++i) {
float input_value = input.index<float>(i);
sum += input_value * input_value;
}
const float eps = 1e-5f;
float mean = sum / float(size) + eps;

const float rsqrt = 1.f / std::sqrt(mean);
for (int i = 0; i < size; ++i) {
*(out_ptr + i) = weight.index<float>(i) * (rsqrt * (*(in_ptr + i)));
}
}
*/