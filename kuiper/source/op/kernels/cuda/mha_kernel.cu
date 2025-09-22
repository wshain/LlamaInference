#include <base/cuda_config.h>
#include <base/tick.h>
#include <tensor/tensor.h>
#include <cfloat>
#include <cub/cub.cuh>
#include "mha_kernel.cuh"
namespace kernel {
constexpr static int thread_num = 256;
/**
 * 在 GPU 上对一个数组执行 in-place Softmax 操作
 *
 * 功能: 对输入数组 x 执行 softmax: x[i] = exp(x[i] - max) / sum(exp(x[j] - max))
 * 使用 CUB 库进行高效的 block 内归约（reduce）
 *
 * 参数:
 *   x     - 输入/输出数组，shape: [size]，softmax 后结果写回原数组
 *   size  - 数组长度（通常是 seq_len，即注意力分数的序列长度）
 */
__device__ void softmax_gpu(float* __restrict__ x, int size) {
  int tid = threadIdx.x;
  int step = blockDim.x;

  // find max value (for numerical stability)
  // this should be FLT_MAX, not 0 !!!!
  // otherwise, the softmax may be occur nan when head_dim < 128 threads
  // ================= 第一步：找最大值（用于数值稳定）=================
  // 初始化 max_val：如果线程 ID 小于 size，则取 x[tid]，否则设为 -FLT_MAX
  // 这是为了避免越界线程影响最大值计算
  float max_val = tid < size ? x[tid] : -FLT_MAX;
  // 使用 grid-stride loop 在 block 内并行查找局部最大值
  // 每个线程从 tid + step 开始，每隔 step 个元素检查一次
  for (int i = tid + step; i < size; i += step) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  // 使用 CUB 的 BlockReduce 进行 block 内归约（求最大值）
  using BlockReduce = cub::BlockReduce<float, thread_num>;  // 假设 thread_num 是编译期常量
  __shared__ BlockReduce::TempStorage temp;                 // 共享内存，用于 CUB 归约
  __shared__ float shared_val;                              // 用于存储归约结果（max 和 sum）
  // 执行归约操作：所有线程协作，找出整个 block 中的最大值
  max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
  // 只有主线程（threadIdx.x == 0）保存结果到共享内存
  if (threadIdx.x == 0) {
    shared_val = max_val;
  }
  __syncthreads();
  // 所有线程读取最大值
  max_val = shared_val;

  // ================= 第二步：计算 exp(x[i] - max_val) 并求和 =================
  float sum = 0.0f;
  for (int i = tid; i < size; i += step) {
    x[i] = expf(x[i] - max_val);  // in-place 计算 exp(x[i] - max)，提升数值稳定性
    sum += x[i];                  // 累加局部和
  }
  // 使用 BlockReduce 对 sum 进行 block 内归约（求和）
  sum = BlockReduce(temp).Sum(sum);
  // 主线程保存总和
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  // 所有线程读取总和
  sum = shared_val;
  // ================= 第三步：归一化：x[i] /= sum =================
  for (int i = tid; i < size; i += step) {
    // 每个元素除以总和，完成 softmax
    x[i] /= sum;
  }
}

__global__ void multi_head_attention_kernel(int32_t pos, int32_t seq_len, float* query,
                                            float* score_ptr, float* output, float* key_cache,
                                            float* value_cache, int32_t kv_dim, int32_t kv_mul,
                                            int32_t head_num, int32_t head_size,
                                            int32_t layer_offset) {
  int head = blockIdx.x;
  if (head >= head_num) {
    return;
  }

  extern __shared__ float s_query_head[];
  float scale = 1.f / sqrtf(float(head_size));
  float* query_head = query + head * head_size;

  // 预加载query到共享内存
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    s_query_head[i] = query_head[i];
  }
  __syncthreads();

  float* score_head = score_ptr + head * seq_len;
  // head当前的注意力头索引，kv_mul用于gqa，head_size表示一个自注意力头的维度
  // kv_dim = head_size * head_num，多头自注意力情况下的key,value 维度
  // kv_dim = head_size * head_num / kv_num，GQA情况下的key,value 维度
  int head_offset = (head / kv_mul) * head_size;
  // 计算自注意力分数
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    float* key_head = key_cache + layer_offset + t * kv_dim + head_offset;

    float score = 0.0f;
    for (int i = 0; i < head_size; i += 4) {
      float4 key_val = *reinterpret_cast<float4*>(key_head + i);
      float4 query_val = *reinterpret_cast<float4*>(s_query_head + i);

      score += key_val.x * query_val.x + key_val.y * query_val.y + key_val.z * query_val.z +
               key_val.w * query_val.w;
    }

    score *= scale;
    score_head[t] = score;
  }
  __syncthreads();

  softmax_gpu(score_head, pos + 1);
  __syncthreads();

  float* output_head = output + head * head_size;
  // 使用自注意力分数对value矩阵加权
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float value = 0.0f;
    for (int t = 0; t <= pos; t++) {
      float* value_head = value_cache + layer_offset + t * kv_dim + head_offset;
      float score = score_head[t];
      value += score * value_head[i];
    }
    output_head[i] = value;
  }
}

void mha_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                   int32_t kv_dim, int32_t kv_mul, int32_t head_size, const tensor::Tensor& mha_out,
                   const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                   const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
                   base::DeviceType device_type, CudaConfig* config) {
  UNUSED(device_type);
  int32_t layer_offset = layer_index * seq_len * kv_dim;
  float* query = const_cast<float*>(query_tensor.ptr<float>());
  float* score = const_cast<float*>(score_tensor.ptr<float>());
  float* output = const_cast<float*>(mha_out.ptr<float>());

  float* key_cache = const_cast<float*>(key_cache_tensor.ptr<float>());
  float* value_cache = const_cast<float*>(value_cache_tensor.ptr<float>());

  cudaStream_t stream = config->stream;
  multi_head_attention_kernel<<<head_num, thread_num, head_size * sizeof(float), stream>>>(
      pos, seq_len, query, score, output, key_cache, value_cache, kv_dim, kv_mul, head_num,
      head_size, layer_offset);
}

}  // namespace kernel