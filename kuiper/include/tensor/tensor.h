#ifndef KUIPER_INCLUDE_TENSOR_TENSOR_H_
#define KUIPER_INCLUDE_TENSOR_TENSOR_H_

#include <driver_types.h>         // CUDA driver 类型定义（如 cudaStream_t）
#include <glog/logging.h>         // Google logging 库
#include <armadillo>              // 可能用于调试或 CPU 数值计算（这里未使用）
#include <memory>                 // shared_ptr、unique_ptr 等智能指针支持
#include <vector>                 // std::vector 支持
#include "base/base.h"            // 基础类型定义（如 DataType）
#include "base/buffer.h"          // Buffer 类，用于统一管理内存

namespace tensor {

// Tensor 类：封装张量数据，包括维度信息和底层内存缓冲区
class Tensor {
 public:
  // 默认构造函数
  explicit Tensor() = default;

  // 构造函数：1维张量
  explicit Tensor(base::DataType data_type, int32_t dim0,
                  bool need_alloc = false,
                  std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                  void* ptr = nullptr);

  // 构造函数：2维张量
  explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1,
                  bool need_alloc = false,
                  std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                  void* ptr = nullptr);

  // 构造函数：3维张量
  explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2,
                  bool need_alloc = false,
                  std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                  void* ptr = nullptr);

  // 构造函数：4维张量
  explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
                  bool need_alloc = false,
                  std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                  void* ptr = nullptr);

  // 构造函数：任意维度张量
  explicit Tensor(base::DataType data_type, std::vector<int32_t> dims,
                  bool need_alloc = false,
                  std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                  void* ptr = nullptr);

  // 将张量数据转移到 CPU
  void to_cpu();

  // 将张量数据转移到 GPU（CUDA）
  void to_cuda(cudaStream_t stream = nullptr);

  // 判断当前张量是否为空
  bool is_empty() const;

  // 初始化底层 Buffer（内存缓冲区）
  void init_buffer(std::shared_ptr<base::DeviceAllocator> alloc,
                   base::DataType data_type,
                   bool need_alloc,
                   void* ptr);

  // 获取指向数据的指针（模板版本，类型安全）
  template <typename T>
  T* ptr();

  // const 版本，只读访问
  template <typename T>
  const T* ptr() const;

  // reshape 操作：修改张量形状，不改变数据内容
  void reshape(const std::vector<int32_t>& dims);

  // 获取底层 Buffer 智能指针
  std::shared_ptr<base::Buffer> get_buffer() const;

  // 返回张量中元素总数量
  size_t size() const;

  // 返回张量所占字节数
  size_t byte_size() const;

  // 返回张量维度个数
  int32_t dims_size() const;

  // 获取张量的数据类型（float/fp16/int8 等）
  base::DataType data_type() const;

  // 获取指定索引处的维度大小
  int32_t get_dim(int32_t idx) const;

  // 获取维度数组
  const std::vector<int32_t>& dims() const;

  // 计算并返回 strides（步长），用于索引转换
  std::vector<size_t> strides() const;

  // 分配一个新的 Buffer 给当前 Tensor 使用
  bool assign(std::shared_ptr<base::Buffer> buffer);

  // 重置张量的数据类型和维度
  void reset(base::DataType data_type, const std::vector<int32_t>& dims);

  // 设置设备类型（const 方法表示不会修改对象状态）
  void set_device_type(base::DeviceType device_type) const;

  // 获取当前设备类型（CPU/GPU）
  base::DeviceType device_type() const;

  // 显式分配内存（可选重新分配）
  bool allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_realloc = false);

  // 获取第 index 个元素的指针（模板 + 带索引）
  template <typename T>
  T* ptr(int64_t index);

  template <typename T>
  const T* ptr(int64_t index) const;

  // 获取第 offset 个元素的引用（模板 + 偏移）
  template <typename T>
  T& index(int64_t offset);

  template <typename T>
  const T& index(int64_t offset) const;

  // 克隆当前张量（深拷贝）
  tensor::Tensor clone() const;

 private:
  size_t size_ = 0;                             // 张量元素总数
  std::vector<int32_t> dims_;                   // 张量各维度大小
  std::shared_ptr<base::Buffer> buffer_;        // 底层缓冲区（共享指针）
  base::DataType data_type_ = base::DataType::kDataTypeUnknown;  // 数据类型
};

// 获取第 offset 个元素的引用（非 const）
template <typename T>
T& Tensor::index(int64_t offset) {
  CHECK_GE(offset, 0);   // 检查偏移是否合法
  CHECK_LT(offset, this->size());
  T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
  return val;
}

// const 版本，获取第 offset 个元素的引用
template <typename T>
const T& Tensor::index(int64_t offset) const {
  CHECK_GE(offset, 0);
  CHECK_LT(offset, this->size());
  const T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
  return val;
}

// 获取 const 指针（模板）
template <typename T>
const T* Tensor::ptr() const {
  if (!buffer_) {
    return nullptr;
  }
  return const_cast<const T*>(reinterpret_cast<T*>(buffer_->ptr()));
}

// 获取非 const 指针（模板）
template <typename T>
T* Tensor::ptr() {
  if (!buffer_) {
    return nullptr;
  }
  return reinterpret_cast<T*>(buffer_->ptr());
}

// 获取第 index 个元素的指针（非 const）
template <typename T>
T* Tensor::ptr(int64_t index) {
  CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
      << "The data area buffer of this tensor is empty or it points to a null pointer.";
  return const_cast<T*>(reinterpret_cast<const T*>(buffer_->ptr())) + index;
}

// 获取第 index 个元素的 const 指针
template <typename T>
const T* Tensor::ptr(int64_t index) const {
  CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
      << "The data area buffer of this tensor is empty or it points to a null pointer.";
  return reinterpret_cast<const T*>(buffer_->ptr()) + index;
}

}  // namespace tensor

#endif  // KUIPER_INCLUDE_TENSOR_TENSOR_H_