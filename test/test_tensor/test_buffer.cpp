#include <cuda_runtime_api.h>  // CUDA API 支持
#include <glog/logging.h>      // Google 日志库
#include <gtest/gtest.h>       // Google Test 单元测试框架
#include "../utils.cuh"        // 自定义 CUDA 工具函数（如 set_value_cu）
#include "base/buffer.h"       // Buffer 类定义

// 测试 1：内存分配功能
TEST(test_buffer, allocate) {
  using namespace base;

  // 创建一个 CPU 内存分配器实例（单例）
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();

  // 构造一个大小为 32 字节的 Buffer 对象，并使用该分配器进行内存分配
  Buffer buffer(32, alloc);

  // 断言：buffer.ptr() 不为空，表示内存分配成功
  ASSERT_NE(buffer.ptr(), nullptr);
}

// 测试 2：外部内存管理功能
TEST(test_buffer, use_external) {
  using namespace base;

  // 获取 CPU 分配器实例
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();

  // 手动申请一块 CPU 内存（32 个 float）
  float* ptr = new float[32];

  // 构造 Buffer 时传入外部指针 ptr，并标记为外部内存（use_external = true）
  Buffer buffer(32, nullptr, ptr, true);

  // 断言：buffer.is_external() 返回 true，确认是外部内存
  ASSERT_EQ(buffer.is_external(), true);

  // 手动释放外部内存（因为 Buffer 不会自动释放外部内存）
  delete[] ptr;
}

// 测试 3：从 CPU 拷贝到 GPU 并验证数据一致性
TEST(test_buffer, cuda_memcpy1) {
  using namespace base;

  // 获取 CPU 和 GPU 分配器实例
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  // 定义缓冲区大小（32 个 float）
  int32_t size = 32;

  // 在 CPU 上手动申请内存并初始化数据（0 到 31）
  float* ptr = new float[size];
  for (int i = 0; i < size; ++i) {
    ptr[i] = float(i);
  }

  // 创建一个 Buffer 对象，使用外部 CPU 内存 ptr
  Buffer buffer(size * sizeof(float), nullptr, ptr, true);

  // 设置设备类型为 CPU
  buffer.set_device_type(DeviceType::kDeviceCPU);

  // 确认当前 Buffer 使用的是外部内存
  ASSERT_EQ(buffer.is_external(), true);

  // 创建一个 GPU Buffer 对象，使用 CUDADeviceAllocator 分配显存
  Buffer cu_buffer(size * sizeof(float), alloc_cu);

  // 调用 copy_from 方法将 CPU 数据拷贝到 GPU
  cu_buffer.copy_from(buffer);

  // 在 CPU 上申请内存用于结果验证
  float* ptr2 = new float[size];

  // 将 GPU 数据拷贝回 CPU
  cudaMemcpy(ptr2, cu_buffer.ptr(), sizeof(float) * size, cudaMemcpyDeviceToHost);

  // 验证 GPU 中的数据是否与原始 CPU 数据一致
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(ptr2[i], float(i));
  }

  // 释放手动申请的内存
  delete[] ptr;
  delete[] ptr2;
}

// 测试 4：再次测试 CPU 到 GPU 的拷贝（与上一测试类似）
TEST(test_buffer, cuda_memcpy2) {
  using namespace base;

  // 同样获取 CPU/GPU 分配器
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  // 定义大小并初始化 CPU 数据
  int32_t size = 32;
  float* ptr = new float[size];
  for (int i = 0; i < size; ++i) {
    ptr[i] = float(i);
  }

  // 创建外部 CPU Buffer
  Buffer buffer(size * sizeof(float), nullptr, ptr, true);
  buffer.set_device_type(DeviceType::kDeviceCPU);
  ASSERT_EQ(buffer.is_external(), true);

  // 创建 GPU Buffer
  Buffer cu_buffer(size * sizeof(float), alloc_cu);

  // CPU -> GPU 拷贝
  cu_buffer.copy_from(buffer);

  // 拷贝回 CPU 并验证
  float* ptr2 = new float[size];
  cudaMemcpy(ptr2, cu_buffer.ptr(), sizeof(float) * size, cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(ptr2[i], float(i));
  }

  // 释放资源
  delete[] ptr;
  delete[] ptr2;
}

// 测试 5：GPU 到 GPU 的拷贝
TEST(test_buffer, cuda_memcpy3) {
  using namespace base;

  // 获取分配器
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  // 定义大小
  int32_t size = 32;

  // 创建两个 GPU Buffer
  Buffer cu_buffer1(size * sizeof(float), alloc_cu);
  Buffer cu_buffer2(size * sizeof(float), alloc_cu);

  // 使用 CUDA kernel 函数设置 GPU 缓冲区数据为 1.0f
  set_value_cu((float*)cu_buffer2.ptr(), size);

  // 验证两个 Buffer 的设备类型都是 GPU
  ASSERT_EQ(cu_buffer1.device_type(), DeviceType::kDeviceCUDA);
  ASSERT_EQ(cu_buffer2.device_type(), DeviceType::kDeviceCUDA);

  // GPU -> GPU 拷贝
  cu_buffer1.copy_from(cu_buffer2);

  // 拷贝回 CPU 并验证值是否为 1.0f
  float* ptr2 = new float[size];
  cudaMemcpy(ptr2, cu_buffer1.ptr(), sizeof(float) * size, cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(ptr2[i], 1.f);
  }

  // 释放 CPU 内存
  delete[] ptr2;
}

// 测试 6：GPU 到 CPU 的拷贝
TEST(test_buffer, cuda_memcpy4) {
  using namespace base;

  // 获取分配器
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  // 定义大小
  int32_t size = 32;

  // 创建两个 Buffer：
  // cu_buffer1 是 GPU 内存
  // cu_buffer2 是 CPU 内存
  Buffer cu_buffer1(size * sizeof(float), alloc_cu);
  Buffer cu_buffer2(size * sizeof(float), alloc);

  // 验证设备类型
  ASSERT_EQ(cu_buffer1.device_type(), DeviceType::kDeviceCUDA);
  ASSERT_EQ(cu_buffer2.device_type(), DeviceType::kDeviceCPU);

  // 使用 CUDA kernel 设置 GPU 数据为 1.0f
  set_value_cu((float*)cu_buffer1.ptr(), size);

  // GPU -> CPU 拷贝
  cu_buffer2.copy_from(cu_buffer1);

  // 直接访问 CPU Buffer 的指针并验证数据
  float* ptr2 = (float*)cu_buffer2.ptr();
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(ptr2[i], 1.f);
  }
}