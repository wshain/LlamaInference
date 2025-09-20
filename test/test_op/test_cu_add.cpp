#include <cuda_runtime_api.h>  // CUDA API（如 cudaMemcpy、cudaDeviceSynchronize）
#include <glog/logging.h>      // Google 日志库
#include <gtest/gtest.h>       // Google Test 测试框架
#include "../source/op/kernels/kernels_interface.h"  // 算子接口定义
#include "../utils.cuh"                              // 自定义 CUDA 工具函数（如 set_value_cu）
#include "base/buffer.h"                             // Buffer 类定义

// 测试用例：add1_nostream
// 功能：测试两个 float 张量在无 CUDA Stream 情况下的加法操作
TEST(test_add_cu, add1_nostream) {
  // 获取 CUDA 内存分配器实例
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  // 定义张量大小：32 * 151 = 4832 个 float 元素
  int32_t size = 32 * 151;

  // 创建三个 Tensor：
  // t1: 初始化为全 2.0f
  // t2: 初始化为全 3.0f
  // out: 用于保存加法结果
  tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  tensor::Tensor t2(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  tensor::Tensor out(base::DataType::kDataTypeFp32, size, true, alloc_cu);

  // 使用 CUDA 设置张量数据为固定值
  set_value_cu(static_cast<float*>(t1.get_buffer()->ptr()), size, 2.f);
  set_value_cu(static_cast<float*>(t2.get_buffer()->ptr()), size, 3.f);

  // 调用加法核函数：out = t1 + t2
  kernel::get_add_kernel(base::DeviceType::kDeviceCUDA)(t1, t2, out, nullptr);

  // 同步设备，确保计算完成
  cudaDeviceSynchronize();

  // 在主机上申请内存，用于拷贝结果
  float* output = new float[size];

  // 将 GPU 结果拷贝回 CPU 主机内存
  cudaMemcpy(output, out.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);

  // 验证每个输出元素是否等于 5.0f
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(output[i], 5.f);  // 断言：结果应为 5.0f
  }

  delete[] output;
}

// 测试用例：add1_stream
// 功能：测试使用 CUDA Stream 的加法操作
TEST(test_add_cu, add1_stream) {
  // 获取 CUDA 分配器
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151;

  // 创建三个 Tensor
  tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  tensor::Tensor t2(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  tensor::Tensor out(base::DataType::kDataTypeFp32, size, true, alloc_cu);

  // 初始化张量内容
  set_value_cu(static_cast<float*>(t1.get_buffer()->ptr()), size, 2.f);
  set_value_cu(static_cast<float*>(t2.get_buffer()->ptr()), size, 3.f);

  // 创建一个 CUDA Stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // 在指定的 CUDA Stream 中执行加法核函数
  kernel::get_add_kernel(base::DeviceType::kDeviceCUDA)(t1, t2, out, stream);

  // 显式同步设备，等待该 Stream 中所有任务完成
  cudaDeviceSynchronize();

  // 准备主机内存接收结果
  float* output = new float[size];
  cudaMemcpy(output, out.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);

  // 校验每个结果是否正确
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(output[i], 5.f);
  }

  // 清理资源
  cudaStreamDestroy(stream);
  delete[] output;
}

// 测试用例：add_align1
// 功能：测试非对齐尺寸的浮点数加法运算，允许一定误差范围
TEST(test_add_cu, add_align1) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151 * 13;  // 更大的张量大小，用于测试不同内存对齐情况

  // 创建三个 Tensor
  tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  tensor::Tensor t2(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  tensor::Tensor out(base::DataType::kDataTypeFp32, size, true, alloc_cu);

  // 初始化张量内容为 2.1f 和 3.3f
  set_value_cu(static_cast<float*>(t1.get_buffer()->ptr()), size, 2.1f);
  set_value_cu(static_cast<float*>(t2.get_buffer()->ptr()), size, 3.3f);

  // 执行加法核函数：out = t1 + t2
  kernel::get_add_kernel(base::DeviceType::kDeviceCUDA)(t1, t2, out, nullptr);

  cudaDeviceSynchronize();  // 等待计算完成

  // 主机内存准备接收结果
  float* output = new float[size];
  cudaMemcpy(output, out.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);

  // 使用 ASSERT_NEAR 判断结果是否接近预期值（允许 ±0.1 误差）
  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(output[i], 5.4f, 0.1f);  // 因为是浮点数运算，可能存在精度损失
  }

  delete[] output;
}