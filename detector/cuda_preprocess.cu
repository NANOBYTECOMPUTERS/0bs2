#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_types.hpp>

static __global__ void bgr8_to_rgb_chw_norm_kernel(
    const unsigned char* __restrict__ srcBgr, size_t srcStepBytes,
    float* __restrict__ dstChw,
    int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int hw = height * width;
    const int idx = y * width + x;

    const unsigned char* p = srcBgr + y * srcStepBytes + x * 3;
    constexpr float inv255 = 1.0f / 255.0f;

    dstChw[0 * hw + idx] = static_cast<float>(p[2]) * inv255; // R
    dstChw[1 * hw + idx] = static_cast<float>(p[1]) * inv255; // G
    dstChw[2 * hw + idx] = static_cast<float>(p[0]) * inv255; // B
}

void launch_hwc_to_chw_norm(
    const cv::cuda::GpuMat& hwcBgr8,
    float* dstChw,
    int width,
    int height,
    cudaStream_t stream)
{
    if (hwcBgr8.empty() || hwcBgr8.type() != CV_8UC3 || !dstChw || width <= 0 || height <= 0)
        return;

    const dim3 block(16, 16);
    const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    const unsigned char* srcPtr = hwcBgr8.ptr<unsigned char>();

    bgr8_to_rgb_chw_norm_kernel << <grid, block, 0, stream >> > (
        srcPtr, hwcBgr8.step, dstChw, width, height
        );
}
#endif
