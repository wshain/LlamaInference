#include "model/llama3.h"
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <op/matmul.h>
#include <op/mha.h>
#include <op/rmsnorm.h>
#include <sentencepiece_processor.h>
#include <utility>
#include "../op/kernels/cpu/rope_kernel.h"
#include "../op/kernels/cuda/rope_kernel.cuh"
#include "base/tick.h"
namespace model {
/**
 * 将当前 LLaMA 模型的所有层从 CPU 内存迁移到 GPU（CUDA）设备上
 * 同时为各层设置 CUDA 运行所需的配置（如 stream、device 等）
 *
 * @param config: 共享的 CUDA 配置对象，包含 stream、device_id 等信息
 */
void LLama2Layers::to_cuda(std::shared_ptr<kernel::CudaConfig> config) {
  if (add_layer_) {
    add_layer_->set_cuda_config(config);
    add_layer_->to_cuda();
  }

  if (rope_layer_) {
    rope_layer_->set_cuda_config(config);
    rope_layer_->to_cuda();
  }

  if (swiglu_layer_) {
    swiglu_layer_->set_cuda_config(config);
    swiglu_layer_->to_cuda();
  }

  if (cls_layer_) {
    cls_layer_->set_cuda_config(config);
    cls_layer_->to_cuda();
  }

  if (embedding_layer_) {
    embedding_layer_->set_cuda_config(config);
    embedding_layer_->to_cuda();
  }

  if (mha_layer_) {
    mha_layer_->set_cuda_config(config);
    mha_layer_->to_cuda();
  }

  for (auto& weight_layer : wq_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : wk_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : wv_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : wo_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : w1_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : w2_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : w3_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& rms_norm_layer : rmsnorm_layers_) {
    if (rms_norm_layer) {
      rms_norm_layer->to_cuda();
      rms_norm_layer->set_cuda_config(config);
    }
  }
}
/**
 * LLama2Model 构造函数
 *
 * @param tokenizer_type: 词 tokenizer 类型（如 SentencePiece）
 * @param token_path:     tokenizer 模型文件路径（如 tokenizer.model）
 * @param model_path:     模型权重文件路径（如 llama2.bin 或 gguf 格式）
 * @param is_quant_model: 是否为量化模型（例如 int8、4-bit 等）
 */
LLama2Model::LLama2Model(base::TokenizerType tokenizer_type, std::string token_path,
                         std::string model_path, bool is_quant_model)
    : Model(tokenizer_type, base::ModelType::kModelTypeLLama2, std::move(token_path),
            std::move(model_path), is_quant_model) {}

base::Status LLama2Model::init(base::DeviceType device_type) {
  using namespace base;
  if (token_path_.empty()) {
    return error::PathNotValid(token_path_);
  }
  if (device_type == base::DeviceType::kDeviceCPU && is_quant_model_) {
    return error::InternalError("The cpu device do not support int8 quant model.");
  }

  device_type_ = device_type;
  if (device_type == DeviceType::kDeviceCUDA) {
    cudaSetDevice(0);
    cuda_config_ = std::make_shared<kernel::CudaConfig>();
    cudaStreamCreate(&cuda_config_->stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      return error::InternalError("The cuda hanle create failed.");
    }
  }

  Status read_status = gen_model_from_file();
  if (!read_status) {
    return read_status;
  }
  init_mem();
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    kernel::sin_cos_cache_calc_cpu(config_->head_size_, config_->seq_len_,
                                   get_buffer(ModelBufferType::kSinCache).ptr<float>(),
                                   get_buffer(ModelBufferType::kCosCache).ptr<float>());
  } else {
    CHECK_NE(cuda_config_, nullptr);
    kernel::sin_cos_cache_calc_cu(config_->head_size_, config_->seq_len_,
                                  get_buffer(ModelBufferType::kSinCache),
                                  get_buffer(ModelBufferType::kCosCache), cuda_config_->stream);
  }

  sampler_ = std::make_unique<sampler::ArgmaxSampler>(device_type_);
  return error::Success();
}
/**
 * 前向传播函数：执行一次模型推理，计算下一个 token 的预测结果
 *
 * @param input       当前输入 token 的张量（形状通常是 [1] 或 [seq_len]）
 * @param pos_tensor  当前位置信息张量（用于 RoPE 位置编码）
 * @param next        输出参数，用于返回预测的下一个 token ID
 *
 * @return base::Status 表示推理是否成功
 */
base::Status LLama2Model::forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                  int& next) const {
  if (input.is_empty()) {
    return base::error::InvalidArgument("The input tensor is empty.");
  }
  if (device_type_ == base::DeviceType::kDeviceCPU && is_quant_model_) {
    return base::error::InternalError("Unsupported int8 quant in the cpu device");
  }

  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    attention_rms(layer_idx, input);
    // attention (wq wk wv @ input)
    attention_qkv(layer_idx, pos_tensor);
    // multi-head attention
    attention_mha(layer_idx, pos_tensor);
    // feed forward
    feed_forward(layer_idx, input);
  }
  cls_logits(input);
  return base::error::Success();
}
/**
 * 创建 LLaMA-2 模型中的“无参数”操作层（non-parameter layers）
 *
 * 所谓“无参数”，是指这些层：
 *   - 不包含可学习权重（如 Wq, Wk, Wo 等）
 *   - 主要是数学运算或结构调度功能
 *   - 在推理过程中复用，不随 layer_idx 变化
 *
 * 这些层通常在整个模型中**只创建一份实例**，被所有 Transformer 层共享使用。
 */
void LLama2Model::create_nonparam_layers() {
  CHECK(llama_layers_ != nullptr);
  llama_layers_->rope_layer_ = std::make_shared<op::RoPELayer>(
      device_type_, config_->dim_, config_->kv_dim_, config_->head_size_);

  llama_layers_->mha_layer_ = std::make_shared<op::MultiHeadAttention>(
      device_type_, 0, config_->kv_mul_, config_->kv_dim_, config_->seq_len_, config_->head_num_,
      config_->head_size_);

  llama_layers_->add_layer_ = std::make_shared<op::VecAddLayer>(device_type_);

  llama_layers_->swiglu_layer_ =
      std::make_shared<op::SwiGLULayer>(device_type_, config_->hidden_dim_);
}
/**
 * 创建 LLaMA-2 模型中所有带参数的量化层（Quantized Parameter Layers）
 *
 * 该函数仅在 is_quant_model_ == true 时调用
 * 负责从 raw_model_data_ 中按顺序解析并加载：
 *   - 所有线性层（MatmulLayer）的量化权重（int8 + scale）
 *   - Embedding 层
 *   - RMSNorm 层（float32 权重）
 *
 * 权重数据是连续存储的，通过 pos 偏移量逐个读取
 */
void LLama2Model::create_param_quant_layers() {
  CHECK(is_quant_model_);
  CHECK(llama_layers_ != nullptr);

  size_t pos = 0;
  int32_t dim = config_->dim_;
  auto cpu_device_type = base::DeviceType::kDeviceCPU;

  // query
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, true);
    wq->set_group_size(group_size_);
    wq->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wq_layers_.push_back(wq);
    pos = pos + dim * dim + wq->get_scale_num() * sizeof(float);
  }

  // key
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wk = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim, true);
    wk->set_group_size(group_size_);
    wk->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wk_layers_.push_back(wk);
    pos = pos + config_->kv_dim_ * dim + wk->get_scale_num() * sizeof(float);
  }

  // value
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wv = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim, true);
    wv->set_group_size(group_size_);
    wv->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wv_layers_.push_back(wv);
    pos += config_->kv_dim_ * dim + wv->get_scale_num() * sizeof(float);
  }

  // output
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, true);
    wo->set_group_size(group_size_);
    wo->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wo_layers_.push_back(wo);
    pos = pos + dim * dim + wo->get_scale_num() * sizeof(float);
  }

  // w1 layers
  int32_t hidden_dim = config_->hidden_dim_;
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w1 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim, true);
    w1->set_group_size(group_size_);
    w1->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w1_layers_.push_back(w1);
    pos = pos + dim * hidden_dim + w1->get_scale_num() * sizeof(float);
  }

  // w2 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w2 = std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim, true);
    w2->set_group_size(group_size_);
    w2->set_weight(0, {dim, hidden_dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w2_layers_.push_back(w2);
    pos = pos + dim * hidden_dim + w2->get_scale_num() * sizeof(float);
  }

  // w3 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w3 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim, true);
    w3->set_group_size(group_size_);
    w3->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w3_layers_.push_back(w3);
    pos = pos + dim * hidden_dim + w3->get_scale_num() * sizeof(float);
  }

  // wcls layer
  auto cls_layer = std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim, true);
  cls_layer->set_group_size(group_size_);
  if (config_->is_shared_weight_) {
    // using token embedding weight
    cls_layer->set_weight(0, {config_->vocab_size_, dim}, this->raw_model_data_->weight(pos),
                          cpu_device_type);
  } else {
    // no shared
    cls_layer->set_weight(0, {config_->vocab_size_, dim}, this->raw_model_data_->weight(pos),
                          cpu_device_type);
    pos = pos + config_->vocab_size_ * dim + cls_layer->get_scale_num() * sizeof(float);
  }
  llama_layers_->cls_layer_ = cls_layer;

  // embedding layer 创建层本身
  float* weight_ptr = (float*)raw_model_data_->weight(pos);
  llama_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
      device_type_, config_->dim_, config_->seq_len_, std::abs(config_->vocab_size_));
  llama_layers_->embedding_layer_->set_weight(0, {std::abs(config_->vocab_size_), dim}, weight_ptr,
                                              cpu_device_type);
  weight_ptr += config_->vocab_size_ * dim;

  // rmsnorm attention attention,ffn,final
  for (int32_t i = 0; i < 2 * config_->layer_num_ + 1; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, dim);

    rms_norm_layer->set_weight(0, {dim}, weight_ptr, cpu_device_type);
    llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    weight_ptr += dim;
  }
}
/**
 * 创建 LLaMA-2 模型中所有带参数的非量化层（浮点权重）
 *  将从磁盘加载的原始二进制模型数据（raw_model_data_）
 *  解析为结构化的神经网络层（如 MatmulLayer, RMSNormLayer, EmbeddingLayer 等），
 *  并按 LLaMA 架构组织成可用的计算图
 *
 * 该函数仅在 is_quant_model_ == false 时调用
 * 负责从 raw_model_data_ 中按顺序加载：
 *   - Embedding 层
 *   - 所有线性层（WQ, WK, WV, WO, W1, W2, W3, CLS）
 *   - 所有 RMSNorm 层（包含 attention 和 ffn 的 pre-norm，以及 final norm）
 *
 * 权重数据是连续存储的，通过 pos 偏移量逐个读取
 */
void LLama2Model::create_param_layers() {
  // 确保当前不是量化模型，且 模型层容器必须已初始化
  CHECK(!is_quant_model_);
  CHECK(llama_layers_ != nullptr);

  // The embedding layer
  // 定义 CPU 设备类型，用于设置权重所在的设备
  auto cpu_device_type = base::DeviceType::kDeviceCPU;

  // ========================================================================
  // 1. 创建词嵌入层 (Embedding Layer)
  // 输入：token ID -> 输出：dim 维向量
  // 权重形状：[vocab_size, dim]
  // ========================================================================
  llama_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
      device_type_,                   // 计算设备（如 GPU）
      config_->dim_,                  // 词向量维度
      config_->seq_len_,              // 最大序列长度（用于缓存）
      std::abs(config_->vocab_size_)  // 词汇表大小（取绝对值防止配置错误）
  );

  // 从 raw_model_data_ 中获取第 0 个权重块（即 embedding 权重）
  const void* weight_embedding = raw_model_data_->weight(0);
  // 设置 embedding 层的权重
  llama_layers_->embedding_layer_->set_weight(
      0,                                                // 权重索引（第0个）
      {std::abs(config_->vocab_size_), config_->dim_},  // 权重形状
      weight_embedding,                                 // 指向权重数据的指针
      cpu_device_type                                   // 权重当前所在设备
  );

  // create all matmul layer
  // ========================================================================
  // 2. 初始化权重偏移量 `pos`
  // 权重是连续存储的：
  //   [embedding_weights][norm_weights][WQ][WK][WV][WO][W1][W2][W3][CLS]
  // 当前已读取：
  //   - embedding: vocab_size * dim
  //   - 第一组 RMSNorm（attention 输入前的归一化）: layer_num * dim
  // 所以下一个位置是这两个之后
  // ========================================================================
  int32_t dim = config_->dim_;
  size_t pos = dim * std::abs(config_->vocab_size_) + dim * config_->layer_num_;

  // create weight matrix for query
  // ========================================================================
  // 3. 创建注意力模块中的 WQ（Query 投影矩阵）
  // 每层一个 MatmulLayer，形状：[dim, dim]
  // 总共 layer_num_ 层
  // ========================================================================
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim);  // 创建一个新的matmul层
    wq->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wq_layers_.push_back(wq);  // 每个wq layer的维度是dim×dim
    pos += dim * dim;                         // 移动偏移量
  }

  // create weight matrix for key
  // ========================================================================
  // 4. 创建 WK（Key 投影矩阵）
  // 注意：KV 头数可能小于 Q 头数（GQA 支持），所以维度是 kv_dim_
  // 形状：[kv_dim_, dim]，其中 kv_dim_ = kv_head_num_ * head_size_
  // ========================================================================
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wk = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim);
    wk->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wk_layers_.push_back(wk);
    pos += config_->kv_dim_ * dim;
  }

  // create weight matrix for value
  // ========================================================================
  // 5. 创建 WV（Value 投影矩阵）
  // 形状与 WK 相同：[kv_dim_, dim]
  // ========================================================================
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wv = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim);
    wv->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wv_layers_.push_back(wv);
    pos += config_->kv_dim_ * dim;
  }

  // create weight matrix for output
  // ========================================================================
  // 6. 创建 WO（输出投影矩阵）
  // 将多头输出投影回 dim 维
  // 形状：[dim, dim]
  // ========================================================================
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim);
    wo->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wo_layers_.push_back(wo);
    pos += dim * dim;
  }

  // skip ffn rmsnorm
  // ========================================================================
  // 7. 跳过 FFN 前的 RMSNorm 权重
  // 这些权重在后面统一加载（见 RMSNorm 部分）
  // 占用空间：layer_num_ * dim
  // ========================================================================
  pos += config_->layer_num_ * dim;

  // w1 layers
  // ========================================================================
  // 8. 创建 FFN 中的 W1（第一个升维投影）
  // SwiGLU 结构中的 up_proj 或 gate_proj
  // 形状：[hidden_dim, dim]，hidden_dim 通常是 dim 的 4 倍
  // ========================================================================
  int32_t hidden_dim = config_->hidden_dim_;
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w1 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim);
    w1->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w1_layers_.push_back(w1);
    pos += dim * hidden_dim;
  }

  // w2 layers
  // ========================================================================
  // 9. 创建 W2（输出投影）
  // SwiGLU 中的 down_proj
  // 形状：[dim, hidden_dim]
  // ========================================================================
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w2 = std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim);
    w2->set_weight(0, {dim, hidden_dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w2_layers_.push_back(w2);
    pos += dim * hidden_dim;
  }

  // w3 layers
  // ========================================================================
  // 10. 创建 W3（gate 投影）
  // 在 SwiGLU 中用于计算门控信号
  // 形状：[hidden_dim, dim]，与 W1 并行
  // ========================================================================
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w3 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim);
    w3->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w3_layers_.push_back(w3);
    pos += dim * hidden_dim;
  }

  // skip final rms weight
  // ========================================================================
  // 11. 跳过最终输出层的 RMSNorm 权重（稍后加载）
  // 占用：dim 大小
  // ========================================================================
  pos += dim;
  // skip freqs_cos and freqs_sin weight
  // ========================================================================
  // 12. 跳过预计算的旋转位置编码（RoPE）
  // freqs_cos 和 freqs_sin，用于注意力中的位置编码
  // 每个大小为 seq_len * head_size_
  // ========================================================================
  pos += config_->seq_len_ * config_->head_size_;

  // ========================================================================
  // 13. 创建分类头（输出投影到词汇表）
  // 如果 config_->is_shared_weight_ 为 true，则共享 embedding 权重（权重绑定）
  // 否则使用独立的 CLS 权重
  // ========================================================================
  llama_layers_->cls_layer_ =
      std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim);
  if (config_->is_shared_weight_) {
    // using token embedding weight
    // 共享 embedding 权重（参数高效）
    llama_layers_->cls_layer_->set_weight(0, {config_->vocab_size_, dim},
                                          this->raw_model_data_->weight(0),  // 指向 embedding 权重
                                          cpu_device_type);
  } else {
    // 使用独立权重
    llama_layers_->cls_layer_->set_weight(
        0, {config_->vocab_size_, dim},
        this->raw_model_data_->weight(pos),  // pos 当前指向 CLS 权重
        cpu_device_type);
  }

  // create rmsnorm layer
  // ========================================================================
  // 14. 开始加载 RMSNorm 层（第一组：每层注意力前的归一化）
  // 起始位置：embedding 权重之后，即 vocab_size * dim
  // ========================================================================
  size_t rmsnorm_pos = config_->dim_ * std::abs(config_->vocab_size_);

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);

    const void* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
    rms_norm_layer->set_weight(0, {config_->dim_}, weight_rmsnorm, cpu_device_type);
    llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    rmsnorm_pos += config_->dim_;
  }

  // skip attention.wq attention.wk attention.wv attention.wo
  // ========================================================================
  // 15. 跳过所有注意力投影层的权重
  // 用于定位第二组 RMSNorm（FFN 前）的位置
  // 包括：WQ, WK, WV, WO 四类矩阵的总大小
  // ========================================================================
  rmsnorm_pos += config_->layer_num_ * config_->dim_ * config_->dim_;
  rmsnorm_pos +=
      config_->layer_num_ * config_->dim_ * (config_->kv_head_num_ * config_->head_size_);
  rmsnorm_pos +=
      config_->layer_num_ * config_->dim_ * (config_->kv_head_num_ * config_->head_size_);
  rmsnorm_pos += config_->layer_num_ * config_->dim_ * config_->dim_;

  // ========================================================================
  // 16. 加载第二组 RMSNorm（FFN 前的归一化）
  // 每层一个，共 layer_num_ 个
  // ========================================================================
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);
    const void* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
    rms_norm_layer->set_weight(0, {config_->dim_}, weight_rmsnorm, cpu_device_type);
    llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);

    rmsnorm_pos += config_->dim_;
  }

  // skip ffn.w1 ffn.w2 ffn.w3
  // ========================================================================
  // 17. 跳过 FFN 中的三个线性层权重（W1, W2, W3）
  // 为 final RMSNorm 定位
  // ========================================================================
  rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
  rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
  rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;

  // ========================================================================
  // 18. 加载最终的 RMSNorm（输出前的归一化）
  // 只有一个
  // ========================================================================
  std::shared_ptr<op::RmsNormLayer> rms_final_layer =
      std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);

  const void* weight_rmsnorm_final = raw_model_data_->weight(rmsnorm_pos);
  rms_final_layer->set_weight(0, {config_->dim_}, weight_rmsnorm_final, cpu_device_type);
  llama_layers_->rmsnorm_layers_.push_back(rms_final_layer);
}
/**
 * 初始化模型推理所需的临时张量（Tensors）和 KV Cache
 *
 * 该函数负责：
 *   1. 获取设备内存分配器（CPU/GPU）
 *   2. 将模型层结构迁移到目标设备（如 CUDA）
 *   3. 创建所有前向传播中需要的临时缓冲区（如输入、中间结果、KV Cache）
 *   4. 注册这些缓冲区到模型的 buffer 管理系统中
 *
 * 所有缓冲区根据设备类型（CPU/CUDA）分配在对应内存空间
 */
void LLama2Model::init_mem() {
  // 根据模型运行设备类型选择对应的设备内存分配器
  std::shared_ptr<base::DeviceAllocator> alloc;
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    // 如果运行在 CPU 上，使用 CPU 分配器
    alloc = base::CPUDeviceAllocatorFactory::get_instance();
  } else {
    // 否则使用 CUDA 分配器（GPU）
    alloc = base::CUDADeviceAllocatorFactory::get_instance();
  }
  // 若设备为 CUDA，则将模型的权重层移动到 GPU 上
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK_NE(cuda_config_, nullptr);       // 确保 CUDA 配置已初始化
    llama_layers_->to_cuda(cuda_config_);  // 将模型参数和层结构迁移到 GPU
  }

  // 获取 CPU 和 GPU 的分配器实例，用于创建跨设备的张量（如输入/位置在 CPU，计算在 GPU)
  std::shared_ptr<base::DeviceAllocator> alloc_cpu =
      base::CPUDeviceAllocatorFactory::get_instance();  // CPU 分配器
  std::shared_ptr<base::DeviceAllocator> alloc_cu =
      base::CUDADeviceAllocatorFactory::get_instance();  // GPU 分配器

  // ------------------- 输入相关张量 -------------------
  // 输入 token ID（整数序列），在 CPU 上存储（通常来自 host 输入）
  tensor::Tensor input_tokens(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);

  // 词嵌入输出：将 token 映射为向量表示，维度为 [1, dim_]，根据设备决定存放位置
  tensor::Tensor input_embeddings(base::DataType::kDataTypeFp32, 1, config_->dim_, true, alloc);

  // RoPE 位置编码所需的 sin 和 cos 缓存（预计算缓存），用于旋转位置编码
  // 总长度：每个 head 的大小 * 最大序列长度
  tensor::Tensor sin_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,
                           true, alloc);
  tensor::Tensor cos_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,
                           true, alloc);

  // 将上述张量注册到模型的缓冲区管理器中，便于后续按类型访问
  CHECK(insert_buffer(ModelBufferType::kSinCache, sin_cache));
  CHECK(insert_buffer(ModelBufferType::kCosCache, cos_cache));

  CHECK(insert_buffer(ModelBufferType::kInputTokens, input_tokens));
  CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));

  // ------------------- 中间层归一化与输出张量 -------------------
  // RMSNorm 输出张量，维度为 [dim_]，用于残差连接后的归一化
  tensor::Tensor rms_output(base::DataType::kDataTypeFp32, config_->dim_, true, alloc);
  // 注册多个共享相同形状的中间输出缓冲区
  CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output));  // 输出层归一化
  CHECK(insert_buffer(ModelBufferType::kOutputMHA, rms_output));      // 多头注意力输入前的归一化
  CHECK(insert_buffer(ModelBufferType::kW2Output, rms_output));       // FFN 中 W2 的输出
  CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, rms_output));     // FFN 前的归一化

  // ------------------- Feed-Forward Network (FFN) 中间输出 -------------------
  // Llama 使用 SwiGLU 结构，包含 W1 和 W3 两个线性变换
  tensor::Tensor w1_output(base::DataType::kDataTypeFp32, config_->hidden_dim_, true, alloc);
  tensor::Tensor w3_output(base::DataType::kDataTypeFp32, config_->hidden_dim_, true, alloc);

  CHECK(insert_buffer(ModelBufferType::kW1Output, w1_output));  // W1 的输出
  CHECK(insert_buffer(ModelBufferType::kW3Output, w3_output));  // W3 的输出

  // kv cache
  // ------------------- KV Cache（用于加速自回归生成）-------------------
  // 存储每层的 Key 和 Value 缓存，形状为 [layer_num, seq_len, kv_dim]
  // 用于避免重复计算历史 token 的 K/V，提升生成速度
  tensor::Tensor key_cache(base::DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_,
                           config_->kv_dim_, true, alloc);
  tensor::Tensor value_cache(base::DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_,
                             config_->kv_dim_, true, alloc);

  CHECK(insert_buffer(ModelBufferType::kKeyCache, key_cache));
  CHECK(insert_buffer(ModelBufferType::kValueCache, value_cache));

  // ------------------- 多头注意力（MHA）中间张量 -------------------
  // Wq query output
  // Query 向量输出，维度为 [dim_]，作为注意力计算的 Q
  tensor::Tensor query(base::DataType::kDataTypeFp32, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kQuery, query));

  // Pos tensor
  // 输入位置索引（当前生成的位置），通常在 CPU 上维护
  tensor::Tensor pos_tensor(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  CHECK(insert_buffer(ModelBufferType::kInputPos, pos_tensor));

  // Attention output
  // 注意力分数存储（score）和注意力输出
  tensor::Tensor attn(base::DataType::kDataTypeFp32, config_->head_num_, config_->seq_len_, true,
                      alloc);  // 每个头的注意力权重 [head_num, seq_len]
  CHECK(insert_buffer(ModelBufferType::kScoreStorage, attn));  // 注意力分数缓存
  CHECK(insert_buffer(ModelBufferType::kAttnOutput, query));   // 注意力输出（复用 query 形状）

  // ------------------- 最终输出张量 -------------------
  // final forward output
  // 模型最终输出：词汇表上的 logits，维度 [vocab_size_]
  tensor::Tensor forward_output(base::DataType::kDataTypeFp32, config_->vocab_size_, true, alloc);
  // 如果运行在 GPU 上，还需要一个 CPU 上的副本用于获取最终结果（如采样）
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    tensor::Tensor forward_output_cpu(base::DataType::kDataTypeFp32, config_->vocab_size_, true,
                                      alloc_cpu);  // 在 CPU 上分配
    CHECK(insert_buffer(ModelBufferType::kForwardOutputCPU, forward_output_cpu));
  }

  CHECK(insert_buffer(ModelBufferType::kForwardOutput, forward_output));
}
/**
 * 创建 LLaMA-2 模型的完整网络结构
 *
 * 流程：
 *   1. 初始化 llama_layers_ 容器
 *   2. 根据是否量化，创建带参数的层（权重加载）
 *   3. 创建无参数的操作层（如 Add, RoPE, MHA, SwiGLU 等）
 *   4. 进行完整性校验（确保所有层都正确创建）
 *
 * @return base::Status 表示创建成功或失败
 */
base::Status LLama2Model::create_layers() {
  using namespace base;
  if (!llama_layers_) {
    llama_layers_ = std::make_unique<LLama2Layers>();
  }

  if (!is_quant_model_) {
    create_param_layers();
  } else {
    create_param_quant_layers();
  }
  create_nonparam_layers();

  if (!llama_layers_->embedding_layer_) {
    return error::InternalError("Create the embedding layer for the llama model failed!");
  }

  if (llama_layers_->rmsnorm_layers_.size() != 2 * config_->layer_num_ + 1) {
    return error::InternalError("Create the rmsnorm layers for the llama model failed!");
  }

  if (llama_layers_->wq_layers_.size() != config_->layer_num_ ||
      llama_layers_->wk_layers_.size() != config_->layer_num_ ||
      llama_layers_->wv_layers_.size() != config_->layer_num_ ||
      llama_layers_->wo_layers_.size() != config_->layer_num_) {
    return error::InternalError(
        "Create the matmul layer in the attention and ffn attention layers for "
        "the llama model "
        "failed.");
  }

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    if (!llama_layers_->wq_layers_.at(i) || !llama_layers_->wk_layers_.at(i) ||
        !llama_layers_->wv_layers_.at(i) || !llama_layers_->wo_layers_.at(i)) {
      return error::InternalError(
          "Create the matmul layer in the attention and ffn attention layers for "
          "the llama model "
          "failed.");
    }
  }

  if (llama_layers_->w1_layers_.size() != config_->layer_num_ ||
      llama_layers_->w2_layers_.size() != config_->layer_num_ ||
      llama_layers_->w3_layers_.size() != config_->layer_num_) {
    return error::InternalError(
        "Create the matmul layer in the feedforward layers for the llama model "
        "failed.");
  }

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    if (!llama_layers_->w1_layers_.at(i) || !llama_layers_->w2_layers_.at(i) ||
        !llama_layers_->w3_layers_.at(i)) {
      return error::InternalError(
          "Create the matmul layer in the feedforward layers for the llama model "
          "failed.");
    }
  }

  if (!llama_layers_->rope_layer_) {
    return error::InternalError("Create the rope layer for the llama model failed!");
  }

  if (!llama_layers_->add_layer_) {
    return error::InternalError("Create the add layer for the llama model failed!");
  }

  if (!llama_layers_->mha_layer_) {
    return error::InternalError("Create the mha layer for the llama model failed!");
  }

  if (!llama_layers_->swiglu_layer_) {
    return error::InternalError("Create the SwiGLU layer for the llama model failed!");
  }
  return error::Success();
}
// 将输入的 token ID 列表 转换为对应的 词嵌入向量（embedding vectors），是 LLaMA-2
// 模型前向传播的第一步。
op::EmbeddingOutput LLama2Model::embedding(const std::vector<int>& tokens) const {
  auto input_tokens = get_buffer(ModelBufferType::kInputTokens);
  auto input_embeddings = get_buffer(ModelBufferType::kInputEmbeddings);
  if (input_tokens.size() != tokens.size()) {
    input_tokens.reshape({static_cast<int32_t>(tokens.size())});
    input_embeddings.reshape({static_cast<int32_t>(tokens.size()), config_->dim_});
  }
  for (int32_t i = 0; i < tokens.size(); ++i) {
    input_tokens.index<int32_t>(i) = tokens.at(i);
  }

  auto input_token_num =
      tensor::Tensor(base::DataType::kDataTypeInt32, static_cast<int32_t>(tokens.size()));
  LOG_IF(FATAL, !llama_layers_->embedding_layer_)
      << "The embedding layer in the llama2 model is null pointer.";
  STATUS_CHECK(
      llama_layers_->embedding_layer_->forward(input_tokens, input_token_num, input_embeddings));

  op::EmbeddingOutput output(input_tokens, input_embeddings, input_token_num);
  return output;
}
// 在 Multi-Head Attention（MHA）之前，对输入进行 RMSNorm 归一化
void LLama2Model::attention_rms(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(llama_layers_ != nullptr);
  // attn rmsnorm
  tensor::Tensor rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  std::shared_ptr<op::Layer> rmsnorm_layer = llama_layers_->rmsnorm_layers_.at(layer_idx);
  if (!rmsnorm_layer) {
    LOG(FATAL) << "The attention rmsnorm layer is a null pointer in the llama2 model";
  }
  STATUS_CHECK(rmsnorm_layer->forward(input, rmsnorm_output));
}

// 多头注意力（Multi-Head Attention）的 Q、K、V 投影与 RoPE 旋转位置编码
/**
 * @brief 执行第 layer_idx 层注意力机制的 Q、K、V 计算与 RoPE 旋转编码
 *
 * 该函数完成以下操作：
 * 1. 从上一层 RMSNorm 的输出中计算 Query、Key、Value 向量
 * 2. 将 Key 和 Value 写入 KV Cache（用于加速自回归生成）
 * 3. 对 Q 和 K 应用 RoPE（Rotary Position Embedding）位置编码
 *
 * @param layer_idx 当前处理的 Transformer 层索引
 * @param pos_tensor 当前推理的位置索引（例如：第 5 个生成的 token）
 */
void LLama2Model::attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  // 确保模型的层结构已正确初始化
  CHECK(llama_layers_ != nullptr);

  // kv cache
  // 获取 Query 输出缓冲区（形状: [dim_]）
  // 这个 buffer 将接收 wq_layer 的输出结果
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);

  // 从 pos_tensor 中提取当前 token 的位置索引（通常是一个长度为1的 int32 Tensor）
  int32_t pos = pos_tensor.index<int32_t>(0);

  // wq wk wv @ input
  // ==== Step 1: 获取当前层、当前位置的 Key/Value Cache 切片 ===
  // slice_kv_cache 返回一对 Tensor：key 和 val
  // 它们是 KV Cache 中对应 (layer_idx, pos) 位置的“视图”（view），不复制数据
  // 写入这些 Tensor 会直接修改 KV Cache 的底层内存
  const auto& [key, val] = slice_kv_cache(layer_idx, pos);

  // query
  // === Step 2: 获取当前层的 Q、K、V 投影层（即 Wq, Wk, Wv 权重层） ===
  const auto& query_layer = llama_layers_->wq_layers_.at(layer_idx);
  CHECK_NE(query_layer, nullptr) << "The query layer in the attention block is null pointer.";

  // 获取 RMSNorm 的输出（即当前层的输入特征）
  // 在 Llama 中，每个注意力块前会先对输入做 RMSNorm
  auto rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);

  // === Step 3: 计算 Query = Wq @ rmsnorm_output ===
  STATUS_CHECK(query_layer->forward(rmsnorm_output, query));
  // 输出写入 query buffer（形状: [dim_]）

  // key
  // === Step 4: 计算 Key = Wk @ rmsnorm_output ===
  const auto& key_layer = llama_layers_->wk_layers_.at(layer_idx);
  CHECK_NE(key_layer, nullptr) << "The key layer in the attention block is null pointer.";
  STATUS_CHECK(key_layer->forward(rmsnorm_output, key));
  // 输出直接写入 KV Cache 的 key 视图中（避免额外拷贝）

  // value
  // === Step 5: 计算 Value = Wv @ rmsnorm_output ===
  const auto& value_layer = llama_layers_->wv_layers_.at(layer_idx);
  CHECK_NE(value_layer, nullptr) << "The value layer in the attention block is null pointer.";
  STATUS_CHECK(value_layer->forward(rmsnorm_output, val));
  // 输出直接写入 KV Cache 的 val 视图中

  // rope
  // === Step 6: 对 Q 和 K 应用 RoPE（旋转位置编码） ===
  // RoPE 编码依赖于当前 token 的位置（pos），使模型能感知序列顺序
  CHECK_NE(llama_layers_->rope_layer_, nullptr)
      << "The RoPE layer in the attention block is null pointer.";
  STATUS_CHECK(llama_layers_->rope_layer_->forward(
      query,                                   // 输入：Query 向量
      key,                                     // 输入：Key 向量（已写入 KV Cache）
      pos_tensor,                              // 输入：当前 token 的位置索引
      get_buffer(ModelBufferType::kSinCache),  // 预计算的 sin 缓存
      get_buffer(ModelBufferType::kCosCache),  // 预计算的 cos 缓存
      tensor::Tensor{}                         // 可选输出（此处未使用）
      ));  // 经过 RoPE 后，query 和 key 已包含位置信息，可用于后续注意力计算
}
// 核心推理接口，封装了从输入张量到生成下一个 token
base::Status LLama2Model::predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                  bool is_prompt, int& next) const {
  auto status = forward(input, pos_tensor, next);
  if (!status) {
    return status;
  }
  next = post_processing(pos_tensor, is_prompt);
  return base::error::Success();
}
// 多头注意力（Multi-Head Attention, MHA）的计算与输出投影
void LLama2Model::attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(llama_layers_ != nullptr);
  // mha
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  // VAL = [val1,val2,...val t]
  // output @ VAL = 最终的结果
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);

  tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
  tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);

  const auto& mha_layer = llama_layers_->mha_layer_;
  CHECK_NE(mha_layer, nullptr) << "The multi head attention layer is null pointer.";
  int pos = pos_tensor.index<int32_t>(0);
  std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_pos(pos);
  std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_layer_idx(layer_idx);
  STATUS_CHECK(mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output));

  // wo @ attention output
  tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
  const auto& wo_layer = llama_layers_->wo_layers_.at(layer_idx);
  CHECK_NE(wo_layer, nullptr) << "The weight output layer is null pointer.";
  STATUS_CHECK(wo_layer->forward(mha_output, attn_output));
}

/**
 * @brief LLaMA-2 模型中 Transformer 层的前馈网络（Feed-Forward Network, FFN）子层。
 *
 * 该函数实现了 LLaMA-2 架构中的 FFN 子层，采用 SwiGLU 激活函数和残差连接。
 * FFN 是每个 Transformer 层的第二个子层，紧跟在 Multi-Head Attention 之后。
 * 其结构为：x + FFN(RMSNorm(x + Attention(x)))
 * 负责对注意力输出进行非线性变换和特征增强
 *
 * 注意：此 FFN 使用了门控线性单元（GLU）变体 SwiGLU，包含三个线性投影：
 *   - w1: 用于生成激活值（gate）
 *   - w3: 用于生成门控信号（gate control）
 *   - w2: 用于最终输出投影
 * 计算公式为：FFN(x) = SwiGLU(w1(x), w3(x)) @ w2(x)
 *
 * @param layer_idx 当前处理的 Transformer 层索引
 * @param input     Attention 子层的输出张量（同时也是 Attention 残差连接后的结果）
 *                  输入尺寸: [seq_len, hidden_dim]
 */
void LLama2Model::feed_forward(int32_t layer_idx, const tensor::Tensor& input) const {
  // 断言模型层结构已初始化
  CHECK(llama_layers_ != nullptr);

  // residual add
  //=============================================================================
  // Step 1: 第一次残差连接（Attention 残差）
  //   input = input + attention_output
  // 注意：此处的 input 已经是 Attention 残差后的结果。
  // 但在某些实现中，add_layer_ 可能在此处再次用于确保残差已加。
  // 实际上，Attention 后的残差通常已在 attention 层完成，此处可能是冗余或通用设计。
  // =============================================================================
  CHECK_NE(llama_layers_->add_layer_, nullptr)
      << "The add layer in the feedforward block is null pointer";
  STATUS_CHECK(
      llama_layers_->add_layer_->forward(input, get_buffer(ModelBufferType::kAttnOutput), input));
  // 此处将 input（当前状态）与 kAttnOutput（Attention 原始输出）相加，结果写回 input
  // 相当于：input = input + attn_output

  // ffn rmsnorm
  // =============================================================================
  // Step 2: RMSNorm 归一化（FFN 前的归一化）
  //   ffn_norm_output = RMSNorm(input)
  // 使用第 (layer_idx + config_->layer_num_) 个 RMSNorm 层（因为前 config_->layer_num_
  // 个用于 Attention 前的归一化）
  // =============================================================================
  tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
  const auto& ffn_rmsnorm = llama_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
  CHECK_NE(ffn_rmsnorm, nullptr)
      << "The final rmsnorm layer in the feedforward block is null pointer";
  STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_output));
  // 输出：ffn_norm_output = RMSNorm(input)

  // w1
  // =============================================================================
  // Step 3: 线性投影 w1（Gate 投影）
  //   w1_output = ffn_norm_output @ W1^T
  // W1 将 hidden_dim 映射到 intermediate_dim（通常是 4 * hidden_dim 或更大）
  // =============================================================================
  tensor::Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
  const auto& w1_layer = llama_layers_->w1_layers_.at(layer_idx);
  CHECK_NE(w1_layer, nullptr) << "The w1 layer in the feedforward block is null pointer";
  STATUS_CHECK(w1_layer->forward(ffn_norm_output, w1_output));
  // w1_output 维度: [seq_len, intermediate_dim]

  // w3
  // =============================================================================
  // Step 4: 线性投影 w3（Gate 控制投影）
  //   w3_output = ffn_norm_output @ W3^T
  // W3 与 W1 共享输入，但参数不同，也映射到 intermediate_dim
  // =============================================================================
  tensor::Tensor w3_ouput = get_buffer(ModelBufferType::kW3Output);
  const auto& w3_layer = llama_layers_->w3_layers_.at(layer_idx);
  CHECK_NE(w3_layer, nullptr) << "The w3 layer in the feedforward block is null pointer";
  STATUS_CHECK(w3_layer->forward(ffn_norm_output, w3_ouput));
  // w3_output 维度: [seq_len, intermediate_dim]

  // SwiGLU
  // =============================================================================
  // Step 5: SwiGLU 激活函数
  //   SwiGLU(gate, gate_ctrl) = SiLU(gate) * gate_ctrl
  //   其中 gate = w1_output, gate_ctrl = w3_output
  // 在 LLaMA-2 中，w1 提供 gate，w3 提供 gate_ctrl
  // 结果仍存入 w1_output（覆盖）
  // =============================================================================
  CHECK_NE(llama_layers_->swiglu_layer_, nullptr)
      << "The swiglu layer in the feedforward block is null pointer";
  STATUS_CHECK(llama_layers_->swiglu_layer_->forward(w1_output, w3_ouput, w1_output));
  // 此处执行：w1_output = SiLU(w1_output) * w3_output
  // 输出维度不变：[seq_len, intermediate_dim]

  // w2
  // =============================================================================
  // Step 6: 线性投影 w2（输出投影）
  //   w2_output = w1_output @ W2^T
  // W2 将 intermediate_dim 映射回 hidden_dim
  // =============================================================================
  tensor::Tensor w2_output = get_buffer(ModelBufferType::kW2Output);
  const auto& w2_layer = llama_layers_->w2_layers_.at(layer_idx);
  CHECK_NE(w2_layer, nullptr) << "The w2 layer in the feedforward block is null pointer";
  STATUS_CHECK(w2_layer->forward(w1_output, w2_output));
  // w2_output 维度: [seq_len, hidden_dim]

  // residual add
  // =============================================================================
  // Step 7: 第二次残差连接（FFN 残差）
  //   output = input + w2_output
  // 将 FFN 的输出加回到原始输入（即 Attention 残差后的结果）
  // 实现完整的 Transformer 子层结构
  // =============================================================================
  CHECK_NE(llama_layers_->add_layer_, nullptr)
      << "The add layer in the feedforward block is null pointer";
  STATUS_CHECK(llama_layers_->add_layer_->forward(input, w2_output, input));
  // 最终：input = input + w2_output
  // 函数结束后，input 即为当前层的完整输出，可用于下一层
}

// 最终分类头（Classification Head） 的实现，负责将 Transformer 的最后一层隐藏状态转换为词汇表上的
// logits，用于预测下一个 token
void LLama2Model::cls_logits(const tensor::Tensor& input) const {
  CHECK(llama_layers_ != nullptr);
  const auto& norm = llama_layers_->rmsnorm_layers_.at(2 * config_->layer_num_);
  CHECK_NE(norm, nullptr);
  STATUS_CHECK(norm->forward(input, input));

  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  CHECK_NE(llama_layers_->cls_layer_, nullptr);
  STATUS_CHECK(llama_layers_->cls_layer_->forward(input, forward_output));
}
// LLaMA-2 模型推理流程中的最后一步：从 logits 中采样生成下一个 token ID
// 这个函数根据当前所处的推理阶段（prompt 预填充 or 自回归生成），决定如何选择下一个 token，是
// 解码策略的核心控制点
int32_t LLama2Model::post_processing(const tensor::Tensor& pos, bool is_prompt) const {
  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  const float* forward_logits = forward_output.ptr<float>();

  int32_t next = 0;
  if (is_prompt) {
    next = -1;
  } else {
    next = static_cast<int32_t>(sampler_->sample(forward_logits, forward_output.size(),
                                                 cuda_config_ ? cuda_config_->stream : nullptr));
  }
  return next;
}

}  // namespace model