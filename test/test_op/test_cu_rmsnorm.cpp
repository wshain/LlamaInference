#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../source/op/kernels/kernels_interface.h"
#include "../utils.cuh"
#include "base/buffer.h"
//测试 RMSNorm 在没有显式 CUDA 流（null stream）下的功能正确性
TEST(test_rmsnorm_cu, rmsnorm_nostream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 15;

  tensor::Tensor in_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor wei_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < size; ++i) {
    in_cpu.index<float>(i) = dist(mt);
    wei_cpu.index<float>(i) = dist(mt);
  }
  tensor::Tensor in_cu = in_cpu.clone();
  tensor::Tensor wei_cu = wei_cpu.clone();
  tensor::Tensor out_cu = out_cpu.clone();
  in_cu.to_cuda(nullptr);
  wei_cu.to_cuda(nullptr);
  out_cu.to_cuda(nullptr);

  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCUDA)(in_cu, wei_cu, out_cu, nullptr);
  out_cu.to_cpu();
  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCPU)(in_cpu, wei_cpu, out_cpu, nullptr);

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(out_cu.index<float>(i), out_cpu.index<float>(i), 1e-5f);
  }
}
//测试 RMSNorm 在指定维度上的操作（可能处理多维张量）以及流的使用
TEST(test_rmsnorm_cu_dim, rmsnorm_stream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int dim_size = 4;
  int size = 1024;
  tensor::Tensor in_cpu(base::DataType::kDataTypeFp32, dim_size, size, true, alloc_cpu);
  tensor::Tensor wei_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, dim_size, size, true, alloc_cpu);

  for (int i = 0; i < dim_size; ++i) {
    for (int j = 0; j < size; ++j) {
      wei_cpu.index<float>(j) = float(j);
      in_cpu.index<float>(i * size + j) = float(j);
    }
  }

  tensor::Tensor in_cu = in_cpu.clone();
  tensor::Tensor wei_cu = wei_cpu.clone();
  tensor::Tensor out_cu = out_cpu.clone();
  in_cu.to_cuda(nullptr);
  wei_cu.to_cuda(nullptr);
  out_cu.to_cuda(nullptr);
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  kernel::get_rmsnorm_dim_kernel(base::DeviceType::kDeviceCUDA)(in_cu, wei_cu, out_cu, 1, nullptr);
  kernel::get_rmsnorm_dim_kernel(base::DeviceType::kDeviceCUDA)(in_cu, wei_cu, in_cu, 1, nullptr);

  out_cu.to_cpu();
  in_cu.to_cpu();

  tensor::Tensor in_cpu_golden(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor wei_cpu_golden(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor out_golden(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess);

  for (int j = 0; j < size; ++j) {
    wei_cpu_golden.index<float>(j) = float(j);
    in_cpu_golden.index<float>(j) = float(j);
  }
  tensor::Tensor in_cu_golden = in_cpu_golden.clone();
  tensor::Tensor wei_cu_golden = wei_cpu_golden.clone();
  tensor::Tensor out_cu_golden = out_cpu.clone();
  in_cu_golden.to_cuda();
  wei_cu_golden.to_cuda();
  out_cu_golden.to_cuda();

  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCUDA)(in_cu_golden, wei_cu_golden,
                                                            out_cu_golden, nullptr);

  out_cu_golden.to_cpu();

  for (int i = 0; i < dim_size; ++i) {
    for (int j = 0; j < size; ++j) {
      ASSERT_EQ(out_cu.index<float>(i * size + j), out_cu_golden.index<float>(j))
          << "i: " << i << " j: " << j;
      ASSERT_EQ(in_cu.index<float>(i * size + j), out_cu_golden.index<float>(j))
          << "i: " << i << " j: " << j;
    }
  }
}
//测试 RMSNorm 在显式 CUDA 流下的功能正确性
TEST(test_rmsnorm_cu, rmsnorm_stream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 32;

  tensor::Tensor in_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor wei_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < size; ++i) {
    in_cpu.index<float>(i) = dist(mt);
    wei_cpu.index<float>(i) = dist(mt);
  }

  tensor::Tensor in_cu = in_cpu.clone();
  tensor::Tensor wei_cu = wei_cpu.clone();
  tensor::Tensor out_cu = out_cpu.clone();
  in_cu.to_cuda(nullptr);
  wei_cu.to_cuda(nullptr);
  out_cu.to_cuda(nullptr);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCUDA)(in_cu, wei_cu, out_cu, stream);
  out_cu.to_cpu();

  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCPU)(in_cpu, wei_cpu, out_cpu, nullptr);

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(out_cu.index<float>(i), out_cpu.index<float>(i), 1e-5f);
  }
  cudaStreamDestroy(stream);
}
//测试 RMSNorm 在大向量和显式流下的功能正确性
TEST(test_rmsnorm_cu, rmsnorm_stream2) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151 * 15;

  tensor::Tensor in_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor wei_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < size; ++i) {
    in_cpu.index<float>(i) = dist(mt);
    wei_cpu.index<float>(i) = dist(mt);
  }

  tensor::Tensor in_cu = in_cpu.clone();
  tensor::Tensor wei_cu = wei_cpu.clone();
  tensor::Tensor out_cu = out_cpu.clone();
  in_cu.to_cuda(nullptr);
  wei_cu.to_cuda(nullptr);
  out_cu.to_cuda(nullptr);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCUDA)(in_cu, wei_cu, out_cu, stream);
  out_cu.to_cpu();

  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCPU)(in_cpu, wei_cpu, out_cpu, nullptr);

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(out_cu.index<float>(i), out_cpu.index<float>(i), 1e-5f);
  }
  cudaStreamDestroy(stream);
}