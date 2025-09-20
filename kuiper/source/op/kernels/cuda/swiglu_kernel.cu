#include <tensor/tensor.h>
#include "swiglu_kernel.cuh"
namespace kernel {
// XXXXX 实际上，这里S使用共享内存并没有性能优势，因为加入共享内存的元素只使用了一次 XXXXX
__global__ void swiglu_kernel_cu_fp32(int size,          // 输入张量的总元素个数
                                      const float* in1,  // 第一个输入指针（通常是 x）
                                      const float* in2,  // 第二个输入指针（通常是 gate / y）
                                      float* out         // 输出指针（结果：(x * sigmoid(x)) * y）
) {
  int tid = threadIdx.x;
  // 计算全局线程 ID：对应当前处理的数据下标
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size) {
    return;
  }
  // 声明动态共享内存（每个 block 分配 shmem 字节）
  extern __shared__ float shared_mem[];
  // 将共享内存划分为两块：
  // - smem1: 缓存 in1 的 blockDim.x 个元素
  // - smem2: 缓存 in2 的 blockDim.x 个元素
  float* smem1 = shared_mem;
  float* smem2 = shared_mem + blockDim.x;

  // 将全局内存数据加载到共享内存中（共 blockDim.x 个线程并行）
  smem1[tid] = in1[idx];
  smem2[tid] = in2[idx];
  __syncthreads();

  // 计算 sigmoid(smem1[tid]) = 1 / (1 + exp(-x))
  float value = 1.0f / (1.0f + exp(-smem1[tid]));
  // 原地计算 Swish(x) = x * sigmoid(x)
  smem1[tid] = smem1[tid] * value;

  // 计算最终输出：Swish(x) * y，并写回全局内存
  out[idx] = smem1[tid] * smem2[tid];
}

void swiglu_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                      const tensor::Tensor& output, void* stream  // CUDA 流（用于异步执行）
) {
  // 检查in1,in2,out不为空，并且在gpu上
  CHECK_EQ(input1.is_empty(), false);
  CHECK(input1.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK_EQ(input2.is_empty(), false);
  CHECK(input2.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK_EQ(output.is_empty(), false);
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);
  // 获取张量总元素数
  int size = static_cast<int32_t>(input1.size());

  int threads = 128;
  // 设置每个 block 的线程数（通常选 128 或 256）
  int blocks = (size + threads - 1) / threads;

  // 计算每个 block 所需的动态共享内存大小：
  // 两个 float 数组，每数组长度为 blockDim.x（即 threads）
  const size_t shmem = threads * sizeof(float) * 2;

  // 判断是否传入了 CUDA 流
  if (!stream) {
    // 若未传入流，则使用默认流（同步执行）
    swiglu_kernel_cu_fp32<<<blocks, threads, shmem>>>(
        size, input1.ptr<float>(),               // x 数据指针
        input2.ptr<float>(),                     // y 数据指针
        const_cast<float*>(output.ptr<float>())  // 输出指针（去 const）
    );
  } else {
    // 若传入流，则转换为 cudaStream_t 并在指定流上异步执行
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    swiglu_kernel_cu_fp32<<<blocks, threads, shmem, stream_>>>(
        size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
  }
}
}  // namespace kernel