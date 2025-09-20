#include <tensor/tensor.h>
#include <cub/block/block_reduce.cuh>
#include "../kernels_interface.h"
#include "matmul_kernel.cuh"
namespace kernel {

/**
 * @brief CUDA 内核：单精度浮点矩阵乘法 (FP32)
 *
 * 计算 output[p] = dot(input, weight[p, :]) for p in [0, K)
 * 即：output = input @ weight^T，其中 input 是 (M,) 或 (1,M)，weight 是 (K,M)
 *
 * 每个 block 负责计算 ROW_PER_BLOCK 行输出（通常为1），每个 block 有 THREAD_PER_BLOCK 个线程。
 * 所有 blocks 并行处理 weight 的不同行。
 *
 * @tparam THREAD_PER_BLOCK 每个 block 的线程数（建议 128/256）
 * @tparam ROW_PER_BLOCK    每个 block 处理的输出行数（通常为1）
 * @param input     输入向量指针，形状 [M]
 * @param weight    权重矩阵指针，形状 [K, M]，按行主序存储
 * @param output    输出向量指针，形状 [K]
 * @param M         特征维度（input 长度，weight 每行长度）
 * @param K         输出维度（weight 行数）
 */

template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32(const float* input, const float* weight, float* output, int M,
                                      int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  // 当前 block 负责的输出行范围 [start_row, end_row)
  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }

  // 向量化打包大小：使用 float4 一次处理 4 个 float
  constexpr int pack_size = 4;
  // 可以用 float4 处理的组数
  const int pack_num = M / pack_size;
  // 剩余未对齐的元素起始位置
  const int pack_off = pack_size * pack_num;

#pragma unroll
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    // 当前行 weight[p, :] 在 weight 数组中的偏移
    int row_offset = p * M;
    // 将 input 和 weight 行指针转为 float4*，启用向量加载
    float4* input_float4_ptr = (float4*)input;
    float4* weight_float4_ptr = (float4*)(weight + row_offset);

    // 【阶段1】使用 float4 向量指令处理对齐部分 (0 ~ pack_off)
#pragma unroll
    for (int i = tid; i < pack_num; i += blockDim.x) {
      float4 input_float4 = *(input_float4_ptr + i);
      float4 weight_float4 = *(weight_float4_ptr + i);

      // 手动计算点积的四个分量（避免编译器优化问题）
      float part_sum = input_float4.x * weight_float4.x + input_float4.y * weight_float4.y +
                       input_float4.z * weight_float4.z + input_float4.w * weight_float4.w;
      sdata[tid] += part_sum;
    }
    // 【阶段2】处理剩余未对齐的元素 (pack_off ~ M)
    for (int i = pack_off + tid; i < M; i += blockDim.x) {
      sdata[tid] += input[i] * weight[row_offset + i];
    }

    __syncthreads();
    // 使用 CUB 库的 BlockReduce 进行块内归约求和
    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

/**
 * @brief CUDA 内核：混合精度矩阵乘法 (FP32 input × INT8 weight)
 *
 * 支持分组量化（Group-wise Quantization）：
 *   weight 被分为多个 group（每 group_size 个元素一组），每组有一个 scale。
 *   实际值 = int8_weight × scale[group_idx]
 *
 * 计算：output[p] = sum_i (input[i] * scales[group(i)] * weight[p*M + i])
 *
 * @tparam THREAD_PER_BLOCK 每个 block 的线程数
 * @tparam ROW_PER_BLOCK    每个 block 处理的输出行数
 * @param input       FP32 输入，[M]
 * @param weight      INT8 量化权重，[K, M]
 * @param scales      每组的缩放因子，[num_groups]，num_groups = ceil(M / group_size)
 * @param group_size  量化分组大小（如 128）
 * @param output      输出，[K]
 * @param M           特征维度
 * @param K           输出维度
 */
template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32int8(const float* input, const int8_t* weight,
                                          const float* scales, const int32_t group_size,
                                          float* output, int M, int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    for (int i = tid; i < M; i += THREAD_PER_BLOCK) {
      const int weight_idx = p * M + i;
      const int group_idx = weight_idx / group_size;
      // 反量化：int8 -> float，并与 input 相乘累加
      sdata[tid] += input[i] * scales[group_idx] * static_cast<float>(weight[weight_idx]);
    }
    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

/**
 * @brief 外部接口：调用 FP32 矩阵乘法内核
 *
 * 计算 output = input @ weight^T （无缩放，scale 参数未使用）
 *
 * @param input   输入 tensor，形状 [M] 或 [1,M]
 * @param weight  权重 tensor，形状 [K, M]
 * @param output  输出 tensor，形状 [K]
 * @param scale   缩放因子（当前未使用）
 * @param config  CUDA 配置（含 stream）
 */
void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, const float scale, const CudaConfig* config) {
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);  // row
  const int32_t M = weight.get_dim(1);  // col
  int packet_size = 4;
  // CHECK_EQ(M % packet_size, 0);

  CHECK_EQ(M, input.get_dim(0));
  // 实例化模板：128 线程/block，每个 block 处理 1 行输出
  if (config && config->stream) {
    // 使用指定 stream 异步执行
    matmul_kernel_cu_fp32<128, 1><<<K, 128, 0, config->stream>>>(
        input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), M, K);
  } else {
    // 默认流同步执行
    matmul_kernel_cu_fp32<128, 1><<<K, 128>>>(input.ptr<float>(), weight.ptr<float>(),
                                              const_cast<float*>(output.ptr<float>()), M, K);
  }
}
/**
 * @brief 外部接口：调用 INT8 量化矩阵乘法内核
 *
 * 支持分组量化反量化计算：output = input @ dequantize(weight)
 *
 * @param input      FP32 输入 [M]
 * @param weight     INT8 量化权重 [K, M]
 * @param output     输出 [K]
 * @param group_size 量化分组大小
 * @param scale      每组的缩放因子 [ceil(M/group_size)]
 * @param config     CUDA 配置
 */
void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, int32_t group_size,
                            const tensor::Tensor& scale, const CudaConfig* config) {
  CHECK(config != nullptr);
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);  // row
  const int32_t M = weight.get_dim(1);  // col
  int packet_size = 4;
  CHECK_EQ(M % packet_size, 0);
  CHECK_EQ(M, input.get_dim(0));
  if (config->stream) {
    matmul_kernel_cu_fp32int8<128, 1><<<K, 128, 0, config->stream>>>(
        input.ptr<float>(), weight.ptr<int8_t>(), scale.ptr<float>(), group_size,
        const_cast<float*>(output.ptr<float>()), M, K);
  } else {
    matmul_kernel_cu_fp32int8<128, 1><<<K, 128>>>(input.ptr<float>(), weight.ptr<int8_t>(),
                                                  scale.ptr<float>(), group_size,
                                                  const_cast<float*>(output.ptr<float>()), M, K);
  }
}
}  // namespace kernel