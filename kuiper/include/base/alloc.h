// 头文件保护宏，防止同一个头文件被多次包含导致重复定义 #ifndef之间#endif
#ifndef KUIPER_INCLUDE_BASE_ALLOC_H_

#define KUIPER_INCLUDE_BASE_ALLOC_H_
#include <map>     // 用于管理内存池
#include <memory>  // 使用智能指针shared_ptr
#include "base.h"
namespace base {
// 拷贝方向
enum class MemcpyKind {
  kMemcpyCPU2CPU = 0,
  kMemcpyCPU2CUDA = 1,
  kMemcpyCUDA2CPU = 2,
  kMemcpyCUDA2CUDA = 3,
};

class DeviceAllocator {
 public:
  // 构造函数，指定设备类型
  // explicit：防止自动隐式转换
  //: device_type_(device_type) 成员初始化列表， 初始化device_type 为device_type_
  explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type) {}
  // 虚函数，表示子类可以重写这个函数
  // 返回设备类型
  virtual DeviceType device_type() const { return device_type_; }
  // 纯虚函数（const = 0）
  // 含有纯虚函数的类为抽象类，不能直接创建实例，子类必须实现这些函数才可以使用
  // 释放内存
  virtual void release(void* ptr) const = 0;
  // 分配指定大小的内存
  virtual void* allocate(size_t byte_size) const = 0;

  // 把一块内存的内容复制到另一块内存
  // src_ptr 指向源数据的指针，不能通过这个指针修改数据
  // dest_ptr 指向目标内存的指针，可以修改
  // byte_size：要复制的数据大小（单位是字节）
  // memcpy_kind：拷贝方向（比如 CPU2CPU）
  // stream,cuda 流用于异步操作
  // need_sync：是否需要同步
  virtual void memcpy(const void* src_ptr, void* dest_ptr, size_t byte_size,
                      MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU, void* stream = nullptr,
                      bool need_sync = false) const;
  // 指定内存区域清零
  virtual void memset_zero(void* ptr, size_t byte_size, void* stream, bool need_sync = false);

  // 私有成员变量，设置默认初始化设备类型
 private:
  DeviceType device_type_ = DeviceType::kDeviceUnknown;
};

// 定义新的类继承（public）自基类 device -cpu
class CPUDeviceAllocator : public DeviceAllocator {
 public:
  explicit CPUDeviceAllocator();
  // override 表示覆盖基类的虚函数，const表示常量成员函数，不会修改类的成员变量
  void* allocate(size_t byte_size) const override;

  void release(void* ptr) const override;
};
// 记录一块cuda内存的信息
// data 为指针地址
// byte_size 大小
// busy 是否被占用
struct CudaMemoryBuffer {
  void* data;
  size_t byte_size;
  bool busy;

  CudaMemoryBuffer() = default;

  CudaMemoryBuffer(void* data, size_t byte_size, bool busy)
      : data(data), byte_size(byte_size), busy(busy) {}
};
// 定义新的类继承（public） 自基类 device- gpu cuda
class CUDADeviceAllocator : public DeviceAllocator {
 public:
  // 构造函数
  explicit CUDADeviceAllocator();
  // 分配内存
  void* allocate(size_t byte_size) const override;
  // 释放内存
  void release(void* ptr) const override;

 private:
  // const 修饰的成员函数不能修改类的成员变量，mutable修饰之后，可以在const函数中被修改
  mutable std::map<int, size_t> no_busy_cnt_;  // 记录未被占用的内存块数量
  // 记录不同大小的内存块池
  mutable std::map<int, std::vector<CudaMemoryBuffer>> big_buffers_map_;   // > 1MB  使用
  mutable std::map<int, std::vector<CudaMemoryBuffer>> cuda_buffers_map_;  // <= 1MB 使用
};

// 单例模式
// 工厂类。确保一个类在整个程序中只有一个实例
class CPUDeviceAllocatorFactory {
 public:
  // shared_ptr 智能指针
  static std::shared_ptr<CPUDeviceAllocator> get_instance() {
    if (instance == nullptr) {
      instance = std::make_shared<CPUDeviceAllocator>();
    }
    return instance;
  }

 private:
  static std::shared_ptr<CPUDeviceAllocator> instance;
};

// 工厂类
class CUDADeviceAllocatorFactory {
 public:
  static std::shared_ptr<CUDADeviceAllocator> get_instance() {
    if (instance == nullptr) {
      instance = std::make_shared<CUDADeviceAllocator>();
    }
    return instance;
  }

 private:
  static std::shared_ptr<CUDADeviceAllocator> instance;
};
}  // namespace base
#endif  // KUIPER_INCLUDE_BASE_ALLOC_H_