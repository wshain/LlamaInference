#include <device_launch_parameters.h>
#include <cub/block/block_reduce.cuh>
#include "rmsnorm_kernel.cuh"
namespace kernel {
/**
 * 计算多维输入 in = (dim1, dim2), 计算在dim2维度上的rmsnorm
 */
static __global__ void row_rmsnorm_f32_dim(float* in, float* wei, float* out, int dim_size,
                                           int size, float eps) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  if (bid >= dim_size) {
    return;
  }

  float* block_in = in + bid * size;
  float* block_out = out + bid * size;
  constexpr int pack_size = 4;
  const int pack_num = size / pack_size;
  const int pack_off = pack_size * pack_num;

  float sum = 0.0f;
  float4* in_pack = reinterpret_cast<float4*>(block_in);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    sum += in_float4.x * in_float4.x;
    sum += in_float4.y * in_float4.y;
    sum += in_float4.z * in_float4.z;
    sum += in_float4.w * in_float4.w;
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    sum += block_in[i] * block_in[i];
  }

  using BlockReduce = cub::BlockReduce<float, 128>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  float4* wei_pack = reinterpret_cast<float4*>(wei);
  float4* out_pack = reinterpret_cast<float4*>(block_out);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    float4 wei_float4 = *(wei_pack + i);
    *(out_pack + i) =
        make_float4(scale * in_float4.x * wei_float4.x, scale * in_float4.y * wei_float4.y,
                    scale * in_float4.z * wei_float4.z, scale * in_float4.w * wei_float4.w);
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    block_out[i] = wei[i] * block_in[i] * scale;
  }
}

template <int32_t BLOCK_DIM>
// 单行归一化
static __global__ void row_rmsnorm_f32(float* in, float* wei, float* out, int size, float eps) {
  const int tid = threadIdx.x;

  constexpr int pack_size = 4;
  const int pack_num = size / pack_size;
  const int pack_off = pack_size * pack_num;

  float sum = 0.0f;
  float4* in_pack = reinterpret_cast<float4*>(in);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    sum += in_float4.x * in_float4.x;
    sum += in_float4.y * in_float4.y;
    sum += in_float4.z * in_float4.z;
    sum += in_float4.w * in_float4.w;
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    sum += in[i] * in[i];
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  float4* wei_pack = reinterpret_cast<float4*>(wei);
  float4* out_pack = reinterpret_cast<float4*>(out);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    float4 wei_float4 = *(wei_pack + i);
    *(out_pack + i) =
        make_float4(scale * in_float4.x * wei_float4.x, scale * in_float4.y * wei_float4.y,
                    scale * in_float4.z * wei_float4.z, scale * in_float4.w * wei_float4.w);
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    out[i] = wei[i] * in[i] * scale;
  }
}
//---------------1/warp规约
// static __global__ void row_rmsnorm_f32(const float* in, const float* wei, float* out,  const int
// size, const float eps) {
//       const int tid = threadIdx.x; // 线程 id 多个线程是同时进来的。
//       const int lane_id = tid % warpSize; // 这个线程id在warp内的编号

//       float sum = 0.0f;
//         // lane_id从0到31都有，是同时执行的
//       for (int i = lane_id; i < size; i += warpSize) {
//         sum += in[i] * in[i];
//       }
//       // 根据局部和求出sum的全局和
//       using WarpReduce = cub::WarpReduce<float, 32>;
//       __shared__ typename WarpReduce::TempStorage temp;
//       __shared__ float shared_val;
//       sum = WarpReduce(temp).Reduce(sum, cub::Sum());

//       const float scale = rsqrtf(sum / static_cast<float>(size) + eps);
//       for (int i = lane_id; i < size; i += warpSize) {
//         out[i] = scale * in[i] * wei[i];
//       }
//   }
//------------------ 2/块规约
// template<int32_t BLOCK_DIM>
// static __global__ void row_rmsnorm_f32(const float* in, const float* wei, float* out,
//                                        const int size, const float eps) {
//   const int tid = threadIdx.x;

//   float sum = 0.0f;
//     // tid = 0 , in[0]+in[128]+in[256]+...+
//   for (int i = tid; i < size; i += blockDim.x) {
//     sum += in[i] * in[i];
//   }

//   using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
//   __shared__ typename BlockReduce::TempStorage temp;
//   __shared__ float shared_val;
//   sum = BlockReduce(temp).Sum(sum);
//   if (threadIdx.x == 0) {
//     shared_val = sum;
//   }
//   __syncthreads();
//   sum = shared_val;
//   const float scale = rsqrtf(sum / static_cast<float>(size) + eps);
//   for (int i = tid; i < size; i += blockDim.x) {
//     out[i] = scale * in[i] * wei[i];
//   }
// }
//+++++++++++++++++++手动实现BlockReduce
// template<const int NUM_THREADS=128>
// // 传进来的参数val，是每个线程累计的局部和
// __device__ __forceinline__ float block_reduce_sum(float val) {
//   // always <= 32 warps per block (limited by 1024 threads per block)
//   constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
//   int warp = threadIdx.x / WARP_SIZE;
//   int lane = threadIdx.x % WARP_SIZE;
//   static __shared__ float shared[NUM_WARPS]; // 共享内存数据

//   val = warp_reduce_sum<WARP_SIZE>(val);
//     // 0-31线程上的做一个规约，32-63线程上的做一个规约，32个线程的各自规约结果放到val上。
//   if (lane == 0)
//       shared[warp] = val; // shared[0]存放0-31个线程上的结果，shared[1]存放32-63上的结果。
//   __syncthreads();

//    // tid = 0的时候，val = shared[0] ,
//    // tid = 1的时候，val = shared[1]
//    // tid = 2的时候，val = shared[2]
//    // tid = 3的时候，val = shared[3]
//   val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
//   val = warp_reduce_sum<NUM_WARPS>(val);
//   return val;

void rmsnorm_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
        weight.device_type() == base::DeviceType::kDeviceCUDA &&
        output.device_type() == base::DeviceType::kDeviceCUDA);

#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
  const float eps = 1e-6f;
#else
  const float eps = 1e-5f;
#endif
  const int32_t size = static_cast<int32_t>(input.size());
  float* in_ptr = const_cast<float*>(input.ptr<float>());
  float* wei_ptr = const_cast<float*>(weight.ptr<float>());
  float* out_ptr = const_cast<float*>(output.ptr<float>());
  constexpr int threads_num = 128;
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    row_rmsnorm_f32<128><<<1, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, size, eps);
  } else {
    row_rmsnorm_f32<128><<<1, threads_num>>>(in_ptr, wei_ptr, out_ptr, size, eps);
  }
}

void rmsnorm_kernel_cu_dim(const tensor::Tensor& input, const tensor::Tensor& weight,
                           const tensor::Tensor& output, int32_t dim, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
        weight.device_type() == base::DeviceType::kDeviceCUDA &&
        output.device_type() == base::DeviceType::kDeviceCUDA);

  const float eps = 1e-6f;
  const int32_t total_size = static_cast<int32_t>(input.size());
  const int32_t size = input.get_dim(input.dims_size() - 1);
  const int32_t dim_size = total_size / size;

  float* in_ptr = const_cast<float*>(input.ptr<float>());
  float* wei_ptr = const_cast<float*>(weight.ptr<float>());
  float* out_ptr = const_cast<float*>(output.ptr<float>());
  constexpr int threads_num = 128;
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    row_rmsnorm_f32_dim<<<dim_size, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, dim_size,
                                                               size, eps);
  } else {
    row_rmsnorm_f32_dim<<<dim_size, threads_num>>>(in_ptr, wei_ptr, out_ptr, dim_size, size, eps);
  }
}
}  // namespace kernel
