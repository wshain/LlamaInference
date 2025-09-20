#include "matmul_kernel.h"
#include "../kernels_interface.h"
#include "base/base.h"
namespace kernel {
/**
 * @brief 矩阵乘法的 CPU 实现
 *
 * 计算 output = scale * (input @ weight^T)
 *
 * @param input 输入张量，形状为 [in_dim0] 或 [in_dim0, in_dim1]
 * @param weight 权重张量，形状为 [out_dim, in_dim0]（存储为行优先）
 * @param output 输出张量，形状为 [in_dim1, out_dim]（若 input 为 2D）或 [out_dim]（若 input 为 1D）
 * @param scale 缩放因子，结果会乘以该值
 * @param config CUDA 配置参数（在 CPU 版本中未使用）
 */
void matmul_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, float scale, const CudaConfig* config) {
  // 忽略 config 参数，避免编译警告
  UNUSED(config);
  // 参数合法性检查
  CHECK(input.is_empty() == false);
  CHECK(weight.is_empty() == false);
  CHECK(output.is_empty() == false);

  // 确保所有 tensor 都位于 CPU 上
  CHECK(input.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(output.device_type() == base::DeviceType::kDeviceCPU);

  // 获取指针
  const float* input_ptr = input.ptr<float>();
  const float* weight_ptr = weight.ptr<float>();
  const float* output_ptr = output.ptr<float>();

  // 解析输入 tensor 的维度
  int32_t in_dim1 = 1;
  int32_t in_dim0 = 1;
  if (input.dims_size() == 2) {
    // 输入为二维：[batch_size, feature_dim]
    in_dim0 = input.get_dim(0);
    in_dim1 = input.get_dim(1);
  } else if (input.dims_size() == 1) {
    // 输入为一维：[feature_dim]
    in_dim0 = input.get_dim(0);
  } else {
    LOG(FATAL) << "The input tensor has a wrong dim size.";
  }

  // 检查权重 tensor 的维度（必须为 2D）
  CHECK_EQ(weight.dims_size(), 2);
  const int32_t wei_dim0 = weight.get_dim(0);
  const int32_t wei_dim1 = weight.get_dim(1);

  // 确保矩阵乘法维度匹配：input[*, in_dim0] 与 weight[wei_dim0, wei_dim1] 相乘时，in_dim0 ==
  // wei_dim1
  // x = 【1，2，3】 = (in_dim1,in_dim0) = (1*3)
  // A = [[1, 2, 3],
  //     [4, 5, 6]] = (2*3) = (weight_dim0, weight_dim1) 转置之后(weight_dim1,weight_dim0)=(3*2)
  CHECK_EQ(in_dim0, wei_dim1);

  // 验证输出 tensor 的总元素数量是否正确
  CHECK_EQ(output.size(), wei_dim0 * in_dim1);

  // arma在内存中按列取
  arma::fmat input_mat(const_cast<float*>(input_ptr), in_dim1, in_dim0, false,
                       true);  // 维度设置正确，x不会变化（in_dim1,in_dim0）
  arma::fmat weight_mat(const_cast<float*>(weight_ptr), wei_dim1, wei_dim0, false,
                        true);  // weight维度交换，相当于转置 （wei_dim1, wei_dim0）
  arma::fmat output_mat(const_cast<float*>(output_ptr), in_dim1, wei_dim0, false, true);
  output_mat = ((input_mat * weight_mat)) * scale;
}
}  // namespace kernel