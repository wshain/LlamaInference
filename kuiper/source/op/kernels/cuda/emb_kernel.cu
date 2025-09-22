#include "emb_kernel.cuh"
namespace kernel {
/**
 * CUDA kernel: 实现 Embedding 查表操作 (float32 版本)
 *
 * 功能: 对每个输入 token，从 weight 矩阵中取出对应的 embedding 向量
 * 相当于 PyTorch 中的: output = weight[input]
 *
 * 参数:
 *   vocab_size   - 词表大小
 *   token_num    - 当前 batch 中 token 的总数（序列长度 × batch size）
 *   weight_dim   - 每个 embedding 向量的维度（如 256, 4096）
 *   input_ptr    - 输入 token IDs 的指针，shape: [token_num]
 *   weight_ptr   - embedding 权重矩阵指针，shape: [vocab_size, weight_dim]
 *   output_ptr   - 输出 embedding 结果指针，shape: [token_num, weight_dim]
 */
__global__ void emb_kernel_cu_fp32(int32_t vocab_size, int32_t token_num, int32_t weight_dim,
                                   const int32_t* input_ptr,  // 输入：token ID 数组
                                   const float* weight_ptr,   // 输入：embedding 权重矩阵
                                   float* output_ptr          // 输出：查表后的 embedding 向量
) {
  // 每个 CUDA block 处理一个 token
  int32_t token_idx = blockIdx.x;  // 默认token_num < block_num 所有block可以处理完一次的token
  if (token_idx >= token_num) {
    return;
  }
  // 获取当前 token 的 ID（即在词表中的索引）
  int32_t token = input_ptr[token_idx];
  // 安全检查：确保 token ID 在合法范围内
  if (token >= vocab_size) {
    return;
  }
  // 计算输出位置：第 token_idx 个 token 的输出起始地址
  float* output_ptr_start = output_ptr + token_idx * weight_dim;
  // 计算权重位置：第 token 个 embedding 向量的起始地址
  const float* weight_ptr_start = weight_ptr + token * weight_dim;

  // 使用 CUDA 线程并行拷贝 embedding 向量的每个元素
  // threadIdx.x 是当前线程在 block 中的 ID
  // blockDim.x 是 block 中的总线程数
  // i += blockDim.x 实现 "grid-stride loop"，支持 weight_dim > blockDim.x
  for (int32_t i = threadIdx.x; i < weight_dim; i += blockDim.x) {
    output_ptr_start[i] = weight_ptr_start[i];
  }
}

void emb_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                   const tensor::Tensor& output, int32_t vocab_size, void* stream) {
  tensor::Tensor input_cu;
  if (input.device_type() != base::DeviceType::kDeviceCUDA) {
    input_cu = input.clone();
    input_cu.to_cuda();
  }
  const int32_t input_num = static_cast<int32_t>(input.size());
  const int32_t weight_dim = weight.get_dim(1);
  CHECK(weight.device_type() == output.device_type());
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

  constexpr int32_t max_seq_len = 512;
  constexpr int32_t thread_num = 128;
  int32_t* in_ptr = input_cu.ptr<int32_t>();
  float* wei_ptr = const_cast<float*>(weight.ptr<float>());
  float* out_ptr = const_cast<float*>(output.ptr<float>());
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    emb_kernel_cu_fp32<<<max_seq_len, thread_num, 0, stream_>>>(vocab_size, input_num, weight_dim,
                                                                in_ptr, wei_ptr, out_ptr);
  } else {
    emb_kernel_cu_fp32<<<max_seq_len, thread_num>>>(vocab_size, input_num, weight_dim, in_ptr,
                                                    wei_ptr, out_ptr);
  }
}
}  // namespace kernel