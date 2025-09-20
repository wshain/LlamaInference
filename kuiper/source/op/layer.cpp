#include "op/layer.h"
#include <base/cuda_config.h>
#include <glog/logging.h>
#include <cstdarg>
#include <numeric>
#include <utility>

namespace op {

// BaseLayer 是所有层的基类，提供通用属性和接口
BaseLayer::BaseLayer(base::DeviceType device_type, LayerType layer_type, base::DataType data_type,
                     std::string layer_name)
    : device_type_(device_type),
      layer_type_(layer_type),
      data_type_(data_type),
      layer_name_(std::move(layer_name)) {}

// 获取当前层的数据类型
base::DataType BaseLayer::data_type() const { return data_type_; }

// 获取当前层的类型
LayerType BaseLayer::layer_type() const { return layer_type_; }

// 设置权重（Tensor形式）——默认未实现
base::Status BaseLayer::set_weight(int32_t idx, const tensor::Tensor& weight) {
  return base::error::FunctionNotImplement();
}

// 设置权重（指针形式）——默认未实现
base::Status BaseLayer::set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                   const void* weight_ptr, base::DeviceType device_type) {
  return base::error::FunctionNotImplement();
}

// 获取层名称
const std::string& BaseLayer::get_layer_name() const { return layer_name_; }

// 设置层名称
void BaseLayer::set_layer_name(const std::string& layer_name) { layer_name_ = layer_name; }

// 获取当前设备类型（CPU/GPU）
base::DeviceType BaseLayer::device_type() const { return device_type_; }

// 设置设备类型
void BaseLayer::set_device_type(base::DeviceType device_type) { device_type_ = device_type; }

// Layer 是 BaseLayer 的子类，代表一个通用层，数据类型为 float32
Layer::Layer(base::DeviceType device_type, LayerType layer_type, std::string layer_name)
    : BaseLayer(device_type, layer_type, base::DataType::kDataTypeFp32, std::move(layer_name)) {}

// 初始化函数，默认成功
base::Status Layer::init() { return base::error::Success(); }

// 前向传播函数，默认未实现
base::Status Layer::forward() { return base::error::FunctionNotImplement(""); }

// 检查 Tensor 是否满足设备和数据类型要求
base::Status Layer::check_tensor(const tensor::Tensor& tensor, base::DeviceType device_type,
                                 base::DataType data_type) const {
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("The tensor parameter is empty.");
  }
  if (tensor.device_type() != device_type) {
    return base::error::InvalidArgument("The tensor has a wrong device type.");
  }
  if (tensor.data_type() != data_type) {
    return base::error::InvalidArgument("The tensor has a wrong data type.");
  }
  return base::error::Success();
}

// 检查 Tensor 是否满足设备、数据类型及维度要求
base::Status Layer::check_tensor_with_dim(const tensor::Tensor& tensor,
                                          base::DeviceType device_type, base::DataType data_type,
                                          ...) const {
  std::va_list args;
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("The tensor parameter is empty.");
  }
  if (tensor.device_type() != device_type) {
    return base::error::InvalidArgument("The tensor has a wrong device type.");
  }
  if (tensor.data_type() != data_type) {
    return base::error::InvalidArgument("The tensor has a wrong data type.");
  }

  va_start(args, data_type);
  int32_t dims = tensor.dims_size();
  for (int32_t i = 0; i < dims; ++i) {
    int32_t dim = va_arg(args, int32_t);
    if (dim != tensor.get_dim(i)) {
      return base::error::InvalidArgument("The tensor has a wrong dim in dim" + std::to_string(i));
    }
  }
  va_end(args);
  return base::error::Success();
}

// 设置输入张量
void Layer::set_input(int32_t idx, const tensor::Tensor& input) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  this->inputs_.at(idx) = input;
}

// 设置输出张量
void Layer::set_output(int32_t idx, const tensor::Tensor& output) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  this->outputs_.at(idx) = output;
}

// 获取输入张量（const版本）
const tensor::Tensor& Layer::get_input(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  return inputs_.at(idx);
}

// 获取输入张量（非const版本）
tensor::Tensor& Layer::get_input(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  return inputs_.at(idx);
}

// 获取输出张量（非const版本）
tensor::Tensor& Layer::get_output(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  return outputs_.at(idx);
}

// 检查层是否配置正确（默认未实现）
base::Status Layer::check() const {
  return base::error::FunctionNotImplement("The check function is not implement yet");
}

// 获取输出张量（const版本）
const tensor::Tensor& Layer::get_output(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  return outputs_.at(idx);
}

// 调整输入张量的数量
void Layer::reset_input_size(size_t size) { inputs_.resize(size); }

// 调整输出张量的数量
void Layer::reset_output_size(size_t size) { outputs_.resize(size); }

// 将输入输出张量移动到 GPU
void Layer::to_cuda() {
  for (auto& input : inputs_) {
    if (!input.is_empty()) {
      input.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
  }
  for (auto& output : outputs_) {
    if (!output.is_empty()) {
      output.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
  }
}

// 设置 CUDA 配置
void Layer::set_cuda_config(std::shared_ptr<kernel::CudaConfig> config) {
  if (!config) {
    return;
  }
  this->cuda_config_ = config;
}

// 获取 CUDA 配置
std::shared_ptr<kernel::CudaConfig> Layer::cuda_config() const { return cuda_config_; }

// 获取输入张量数量
size_t Layer::input_size() const { return inputs_.size(); }

// 获取输出张量数量
size_t Layer::output_size() const { return outputs_.size(); }

// LayerParam 是 Layer 的子类，支持权重和量化参数
LayerParam::LayerParam(base::DeviceType device_type, LayerType layer_type, bool is_quant_layer,
                       std::string layer_name)
    : Layer(device_type, layer_type, std::move(layer_name)), is_quant_layer_(is_quant_layer) {}

// 设置权重（Tensor形式）
base::Status LayerParam::set_weight(int32_t idx, const tensor::Tensor& weight) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  CHECK(weight.data_type() == base::DataType::kDataTypeFp32);
  if (!weight.is_empty()) {
    CHECK(weight.device_type() == device_type_);
  }
  weights_.at(idx) = weight;
  return base::error::Success();
}

// 获取权重（const版本）
const tensor::Tensor& LayerParam::get_weight(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  return weights_.at(idx);
}

// 将权重和 scales 移动到 GPU
void LayerParam::to_cuda() {
  Layer::to_cuda();
  for (auto& weight : weights_) {
    weight.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
  }
  if (!scales_.is_empty()) {
    scales_.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
  }
}

// 设置权重（指针形式），支持量化处理
base::Status LayerParam::set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                    const void* weight_ptr, base::DeviceType device_type) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  CHECK_NE(weight_ptr, nullptr);
  // 3.1 计算权重的数量，计算方法是将维度依次累计相乘
  size_t size = std::accumulate(dims.begin(), dims.end(), sizeof(float), std::multiplies<>());
  /**
  3.2 将权重指针（这里来自于weight_data）赋值给一个buffer，
  buffer就是我们以往课程中说过的，用于管理内存资源的一个类。
  3.2 buffer是可以共享计数的，当没有使用者且拥有这块
  指针的所有权时buffer会将指针释放，以获得自动管理内存的功能。
  */
  std::shared_ptr<base::Buffer> buffer =
      std::make_shared<base::Buffer>(size, nullptr, const_cast<void*>(weight_ptr), true);
  if (device_type != base::DeviceType::kDeviceUnknown) {
    buffer->set_device_type(device_type);
  }
  /**
   * 3.3 创建每个算子关联的权重，它是一个张量。默认赋值的张量在cpu上的，
     3.3 等下我们需要把它upload到gpu中。
  */
  if (!is_quant_layer_) {
    // 非量化层，使用 float32 权重
    tensor::Tensor weight(base::DataType::kDataTypeFp32, dims);
    weight.set_device_type(device_type);
    CHECK(weight.assign(buffer));
    weights_.at(idx) = weight;
  } else {
    // 量化层，使用 int8 权重，并计算 scales
    tensor::Tensor weight(base::DataType::kDataTypeInt8, dims);
    weight.set_device_type(device_type);
    CHECK(weight.assign(buffer));
    weights_.at(idx) = weight;

    const int32_t weight_size = static_cast<int32_t>(weight.size());
    CHECK(weight_size % group_size_ == 0);

    int32_t scale_nums = weight_size / group_size_;
    scales_ = tensor::Tensor{base::DataType::kDataTypeFp32, scale_nums, false, nullptr,
                             reinterpret_cast<float*>((int8_t*)weight_ptr + weight_size)};
    scales_.set_device_type(device_type);
  }

  return base::error::Success();
}

// 设置 scales 张量
void LayerParam::set_scales(const tensor::Tensor& scales) {
  CHECK(!scales.is_empty());
  this->scales_ = scales;
}

// 设置量化分组大小
void LayerParam::set_group_size(int32_t group_size) { this->group_size_ = group_size; }

// 获取 scale 的数量
int32_t LayerParam::get_scale_num() const {
  CHECK(!scales_.is_empty());
  return static_cast<int32_t>(scales_.size());
}

// 调整权重张量的数量
void LayerParam::reset_weight_size(size_t size) { weights_.resize(size); }

// 获取权重张量数量
size_t LayerParam::weight_size() const { return weights_.size(); }

// 多个 forward 重载函数，用于设置输入输出并调用前向传播
base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_output(0, output1);
  return this->forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);

  this->set_output(0, output1);
  return this->forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3, const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);

  this->set_output(0, output1);
  return this->forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3, const tensor::Tensor& input4,
                            const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);
  this->set_input(3, input4);

  this->set_output(0, output1);
  return this->forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3, const tensor::Tensor& input4,
                            const tensor::Tensor& input5, const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);
  this->set_input(3, input4);
  this->set_input(4, input5);

  this->set_output(0, output1);
  return this->forward();
}

// 获取权重张量（非const版本）
tensor::Tensor& LayerParam::get_weight(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  return weights_.at(idx);
}

}  // namespace op