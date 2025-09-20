#include "swiglu_kernel.h"
namespace kernel {
// SwiGLU 核心函数：计算 output = (input1 * sigmoid(input1)) ⊙ input2
// 其中 ⊙ 表示逐元素相乘（Hadamard product）
void swiglu_kernel_cpu(const tensor::Tensor& input1,  // 第一个输入张量（门控信号）
                       const tensor::Tensor& input2,  // 第二个输入张量（通常是线性变换输出）
                       const tensor::Tensor& output,  // 输出张量
                       void* stream                   // CUDA 流指针（CPU 实现中不使用）
) {
  // 声明 stream 参数未被使用，避免编译警告
  UNUSED(stream);
  // 检查输入和输出张量都不是空的
  CHECK_EQ(input1.is_empty(), false);
  CHECK_EQ(input2.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);

  // 确保所有张量都在 CPU 设备上（因为这是 CPU 版本的 kernel）
  CHECK(input1.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(input2.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(output.device_type() == base::DeviceType::kDeviceCPU);

  // 使用 Armadillo 库的 fvec（单精度浮点向量）封装原始指针
  // 将 input1、input2 和 output 的数据视为一维向量进行操作
  // 参数说明：
  //   - const_cast<float*>(...)：去掉 const 以允许修改（这里 input1_vec 会先被修改为 sigmoid 结果）
  //   - input1.size()：张量中元素总数
  //   - false：不拥有内存所有权（即不管理内存释放）
  //   - true：允许共享内存（即指向外部数据）
  arma::fvec input1_vec(const_cast<float*>(input1.ptr<float>()), input1.size(), false, true);
  arma::fvec input2_vec(const_cast<float*>(input2.ptr<float>()), input2.size(), false, true);
  arma::fvec output_vec(const_cast<float*>(output.ptr<float>()), output.size(), false, true);

  // 计算 SwiGLU 的核心部分：
  // input1_vec 变为 input1 * sigmoid(input1)
  // 即：x * σ(x)，其中 σ(x) = 1 / (1 + exp(-x))
  input1_vec %= (1.0f / (1.0f + arma::exp(-input1_vec)));
  // 这里 %= 是按元素乘并赋值

  output_vec = input1_vec % input2_vec;  // % 是 Armadillo 中的逐元素乘法
}
}  // namespace kernel