#ifndef KUIPER_INCLUDE_OP_LAYER_H_
#define KUIPER_INCLUDE_OP_LAYER_H_

#include <base/cuda_config.h>  // CUDA 配置类定义
#include <string>              // std::string
#include <vector>              // std::vector
#include "base/base.h"         // 基础类型定义（如 DeviceType、DataType）
#include "tensor/tensor.h"     // Tensor 类定义

namespace op {

// LayerType 枚举：表示不同类型的神经网络层
enum class LayerType : uint8_t {
  kLayerUnknown = 0,    // 未知类型
  kLayerLinear = 1,     // 线性层（全连接层）
  kLayerEncode = 2,     // 编码层
  kLayerEmbedding = 3,  // Embedding 层
  kLayerRMSNorm = 4,    // RMS 归一化层
  kLayerMatmul = 5,     // 矩阵乘法操作
  kLayerRoPe = 6,       // RoPE 位置编码（Rotary Positional Encoding）
  kLayerMHA = 7,        // 多头注意力（Multi-Head Attention）
  kLayerSoftmax = 8,    // Softmax 激活函数
  kLayerAdd = 9,        // 加法操作
  kLayerSwiGLU = 10,    // SwiGLU 激活函数
};

// BaseLayer 抽象基类：定义所有 Layer 的统一接口
class BaseLayer {
 public:
  // 构造函数
  explicit BaseLayer(base::DeviceType device_type,  // 设备类型（CPU/GPU）
                     LayerType layer_type,          // 层类型
                     base::DataType data_type,      // 数据类型（float/fp16/int8 等）
                     std::string layer_name = "");  // 层名称（可选）

  // 获取当前数据的数据类型
  base::DataType data_type() const;

  // 获取当前层的类型
  LayerType layer_type() const;

  // 初始化层资源（抽象函数，子类必须实现）
  virtual base::Status init() = 0;

  // 前向传播方法（多种重载形式，支持不同数量输入张量）
  virtual base::Status forward() = 0;
  virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& output1) = 0;
  virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                               const tensor::Tensor& output1) = 0;
  virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                               const tensor::Tensor& input3, const tensor::Tensor& output1) = 0;
  virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                               const tensor::Tensor& input3, const tensor::Tensor& input4,
                               const tensor::Tensor& output1) = 0;
  virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                               const tensor::Tensor& input3, const tensor::Tensor& input4,
                               const tensor::Tensor& input5, const tensor::Tensor& output1) = 0;

  // 设置第 idx 个输入张量
  virtual void set_input(int32_t idx, const tensor::Tensor& input) = 0;

  // 设置第 idx 个输出张量
  virtual void set_output(int32_t idx, const tensor::Tensor& output) = 0;

  // 返回输入张量的数量
  virtual size_t input_size() const = 0;

  // 返回输出张量的数量
  virtual size_t output_size() const = 0;

  // 校验输入输出是否符合要求
  virtual base::Status check() const = 0;

  // 获取第 idx 个输入/输出张量（非 const 和 const 版本）
  virtual tensor::Tensor& get_input(int32_t idx) = 0;
  virtual tensor::Tensor& get_output(int32_t idx) = 0;
  virtual const tensor::Tensor& get_input(int32_t idx) const = 0;
  virtual const tensor::Tensor& get_output(int32_t idx) const = 0;

  // 设置权重张量（按索引）
  virtual base::Status set_weight(int32_t idx, const tensor::Tensor& weight);

  // 使用维度和原始指针设置权重
  virtual base::Status set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                  const void* weight_ptr,
                                  base::DeviceType device_type = base::DeviceType::kDeviceUnknown);

  // 获取层名称
  const std::string& get_layer_name() const;

  // 设置层名称
  void set_layer_name(const std::string& layer_name);

  // 获取设备类型
  base::DeviceType device_type() const;

  // 设置设备类型
  void set_device_type(base::DeviceType device_type);

 protected:
  std::string layer_name_;        // 层名称
  LayerType layer_type_;          // 层类型
  base::DataType data_type_;      // 数据类型
  base::DeviceType device_type_;  // 设备类型
};

// Layer 类：BaseLayer 的具体实现，管理输入输出张量
class Layer : public BaseLayer {
 public:
  // 构造函数
  explicit Layer(base::DeviceType device_type, LayerType layer_type, std::string layer_name = "");

  // 实现 init 接口
  base::Status init() override;

  // 检查张量是否满足指定的设备和数据类型要求
  base::Status check_tensor(const tensor::Tensor& tensor, base::DeviceType device_type,
                            base::DataType data_type) const;

  // 检查张量是否满足指定的设备、数据类型和维度要求
  base::Status check_tensor_with_dim(const tensor::Tensor& tensor, base::DeviceType device_type,
                                     base::DataType data_type, ...) const;

  // 实现 check 接口：验证输入输出是否合法
  base::Status check() const override;

  // 实现多个 forward 方法（转发调用到具体的实现）
  base::Status forward() override;
  base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& output1) override;
  base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& output1) override;
  base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& input3, const tensor::Tensor& output1) override;
  base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& input3, const tensor::Tensor& input4,
                       const tensor::Tensor& output1) override;
  base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& input3, const tensor::Tensor& input4,
                       const tensor::Tensor& input5, const tensor::Tensor& output1) override;

  // 设置输入/输出张量
  void set_input(int32_t idx, const tensor::Tensor& input) override;
  void set_output(int32_t idx, const tensor::Tensor& output) override;

  // 获取输入/输出张量（const & non-const）
  const tensor::Tensor& get_input(int32_t idx) const override;
  const tensor::Tensor& get_output(int32_t idx) const override;
  tensor::Tensor& get_input(int32_t idx) override;
  tensor::Tensor& get_output(int32_t idx) override;

  // 输入输出张量数量
  size_t input_size() const override;
  size_t output_size() const override;

  // 重设输入/输出张量数量
  void reset_input_size(size_t size);
  void reset_output_size(size_t size);

  // 将该层转移到 GPU 上运行
  virtual void to_cuda();

  // 设置 CUDA 配置参数
  void set_cuda_config(std::shared_ptr<kernel::CudaConfig> config);

  // 获取 CUDA 配置参数
  std::shared_ptr<kernel::CudaConfig> cuda_config() const;

 protected:
  std::vector<tensor::Tensor> inputs_;               // 输入张量列表
  std::vector<tensor::Tensor> outputs_;              // 输出张量列表
  std::shared_ptr<kernel::CudaConfig> cuda_config_;  // CUDA 相关配置
};

// LayerParam 类：继承自 Layer，支持带权重参数的层
class LayerParam : public Layer {
 public:
  // 构造函数
  explicit LayerParam(base::DeviceType device_type, LayerType layer_type,
                      bool is_quant_layer = false,  // 是否是量化层
                      std::string layer_name = "");

  // 获取权重数量
  size_t weight_size() const;

  // 重设权重数量
  void reset_weight_size(size_t size);

  // 获取第 idx 个权重张量
  tensor::Tensor& get_weight(int32_t idx);
  const tensor::Tensor& get_weight(int32_t idx) const;

  // 将该层转移到 GPU 上运行（覆盖父类方法）
  void to_cuda() override;

  // 设置 CUDA 配置
  void set_cuda_config(std::shared_ptr<kernel::CudaConfig> config);

  // 设置权重（从 Tensor 或原始内存拷贝）
  base::Status set_weight(int32_t idx, const tensor::Tensor& weight) override;
  base::Status set_weight(int32_t idx, const std::vector<int32_t>& dims, const void* weight_ptr,
                          base::DeviceType device_type = base::DeviceType::kDeviceUnknown) override;

  // 设置量化缩放因子张量
  void set_scales(const tensor::Tensor& scales);

  // 设置量化组大小
  void set_group_size(int32_t group_size);

  // 获取缩放因子数量
  int32_t get_scale_num() const;

 protected:
  int32_t group_size_ = 0;               // 量化组大小
  bool is_quant_layer_ = false;          // 是否是量化层
  tensor::Tensor scales_;                // 量化缩放因子
  std::vector<tensor::Tensor> weights_;  // 权重张量列表
};

}  // namespace op

#endif  // KUIPER_INCLUDE_OP_LAYER_H_