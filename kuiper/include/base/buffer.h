#ifndef KUIPER_INCLUDE_BASE_BUFFER_H_
#define KUIPER_INCLUDE_BASE_BUFFER_H_

#include <memory>        // 提供 shared_ptr 等智能指针支持
#include "base/alloc.h"  // 自定义内存分配器头文件

namespace base {

// Buffer 类用于统一管理 CPU/GPU 上的数据缓冲区
// 继承关系说明：
// - NoCopyable：禁止拷贝构造和赋值操作（防误用）
// - std::enable_shared_from_this<Buffer>：允许从 this 获取 shared_ptr<Buffer>
class Buffer : public NoCopyable, std::enable_shared_from_this<Buffer> {
 private:
  size_t byte_size_ = 0;                                 // 缓冲区大小（字节）
  void* ptr_ = nullptr;                                  // 数据指针（指向实际内存）
  bool use_external_ = false;                            // 是否使用外部提供的内存？
  DeviceType device_type_ = DeviceType::kDeviceUnknown;  // 当前数据所在的设备类型（CPU/GPU）
  std::shared_ptr<DeviceAllocator> allocator_;           // 使用的内存分配器（CPU 或 GPU）

 public:
  // 默认构造函数
  explicit Buffer() = default;

  // 带参构造函数
  // 参数说明：
  // - byte_size: 缓冲区大小（以字节为单位）
  // - allocator: 内存分配器（可为空，表示不立即分配内存）
  // - ptr: 如果不为空，则使用传入的指针作为内存（外部内存）
  // - use_external: 是否使用外部内存（true 表示 Buffer 不负责释放内存）
  explicit Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator = nullptr,
                  void* ptr = nullptr, bool use_external = false);

  // 虚析构函数，确保派生类对象在析构时能正确调用析构函数
  virtual ~Buffer();

  // 分配内存（如果尚未分配），返回是否成功
  bool allocate();

  // 拷贝函数：从另一个 Buffer 对象中拷贝数据（常量引用）
  void copy_from(const Buffer& buffer) const;

  // 拷贝函数：从另一个 Buffer 指针中拷贝数据（常量指针）
  void copy_from(const Buffer* buffer) const;

  // 返回数据指针（非 const 版本，允许修改数据）
  void* ptr();

  // 返回数据指针（const 版本，只读访问）
  const void* ptr() const;

  // 返回缓冲区大小（字节数）
  size_t byte_size() const;

  // 返回使用的内存分配器（shared_ptr 形式）
  std::shared_ptr<DeviceAllocator> allocator() const;

  // 返回当前缓冲区所在的设备类型（CPU/GPU）
  DeviceType device_type() const;

  // 设置缓冲区所在的设备类型
  void set_device_type(DeviceType device_type);

  // 安全地从 this 获取一个 shared_ptr<Buffer>
  // 避免手动 new shared_ptr<T>(this)，防止未定义行为
  std::shared_ptr<Buffer> get_shared_from_this();

  // 判断是否使用了外部内存
  bool is_external() const;
};

}  // namespace base

#endif  // KUIPER_INCLUDE_BASE_BUFFER_H_

/**
class Base {
 public:
  Base() { std::cout << "Base constructor\n"; }
  virtual ~Base() { std::cout << "Base destructor\n"; }  // 虚析构函数
};

class Derived : public Base {
 public:
  Derived() { std::cout << "Derived constructor\n"; }
  ~Derived() override { std::cout << "Derived destructor\n"; }
};

int main() {
  Base* obj = new Derived();//这是一种典型的“多态”写法：父类指针指向子类对象
  delete obj;
  return 0;
}

输出：
Base constructor
Derived constructor
Derived destructor
Base destructor

当使用基类指针指向子类对象，并通过 delete 删除该指针时，如何确保子类的析构函数也能被调用 ——
这就是虚析构函数的意义。
**/