#pragma once

#ifdef USE_CUDA

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/core/cuda.hpp>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "postProcess.h"

struct WorkerDetection
{
    int classId = 0;
    float confidence = 0.0f;
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
};

struct WorkerDetectionResult
{
    std::string image;
    int width = 0;
    int height = 0;
    std::vector<WorkerDetection> detections;
    std::string error;
};

struct LetterboxTransform
{
    float gain = 1.0f;
    float padX = 0.0f;
    float padY = 0.0f;
    int resizedWidth = 0;
    int resizedHeight = 0;
};

class TensorRtBatchDetector
{
public:
    TensorRtBatchDetector();
    ~TensorRtBatchDetector();

    TensorRtBatchDetector(const TensorRtBatchDetector&) = delete;
    TensorRtBatchDetector& operator=(const TensorRtBatchDetector&) = delete;

    void setClassCount(int classCount);
    bool loadEngine(const std::string& enginePath, std::string& error);
    WorkerDetectionResult detectImage(
        const std::string& imagePath,
        float confidenceThreshold,
        float nmsThreshold);
    std::vector<WorkerDetectionResult> detectBatch(
        const std::vector<std::string>& imagePaths,
        float confidenceThreshold,
        float nmsThreshold);

private:
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;

    cudaStream_t stream = nullptr;
    cv::cuda::Stream cvStream;

    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    std::map<std::string, size_t> inputSizes;
    std::map<std::string, size_t> outputSizes;
    std::map<std::string, nvinfer1::DataType> outputTypes;
    std::map<std::string, std::vector<int64_t>> outputShapes;
    std::map<std::string, void*> inputBindings;
    std::map<std::string, void*> outputBindings;
    std::map<std::string, std::vector<unsigned char>> outputHostBuffers;
    std::map<std::string, std::vector<float>> outputFloatBuffers;
    std::map<std::string, std::vector<float>> fp16OutputScratch;

    std::string inputName;
    int inputChannels = 3;
    int inputWidth = 640;
    int inputHeight = 640;
    int minBatchSize = 1;
    int maxBatchSize = 1;
    int currentBatchSize = 0;
    bool inputHasBatchDim = false;
    bool inputDynamic = false;
    int numClasses = 0;
    int explicitClassCount = 0;

    cv::cuda::GpuMat gpuFrameBuffer;
    cv::cuda::GpuMat gpuBgrBuffer;
    cv::cuda::GpuMat gpuResizedBuffer;
    cv::cuda::GpuMat gpuLetterboxBuffer;
    cv::cuda::GpuMat gpuFloatBuffer;
    LetterboxTransform lastLetterboxTransform;

    bool ensureStream(std::string& error);
    bool allocateBindings(std::string& error);
    bool configureBatch(int batchSize, std::string& error);
    bool resolveBatchShape(int batchSize, nvinfer1::Dims& resolved, std::string& error) const;
    bool refreshTensorShapesAndBindings(std::string& error);
    bool preprocess(const cv::Mat& frame, int slot, LetterboxTransform& transform, std::string& error);
    bool copyOutputsToHost(std::string& error);
    bool copyOutputToFloat(const std::string& outputName, std::vector<float>& out, std::string& error);
    bool outputSliceForSlot(
        const std::string& outputName,
        int executionBatchSize,
        int slot,
        const float*& data,
        std::vector<int64_t>& shape,
        std::string& error) const;
    bool decodeMultiOutputNms(
        float confidenceThreshold,
        float nmsThreshold,
        std::vector<Detection>& detections);
    bool decodeMultiOutputNmsForSlot(
        int executionBatchSize,
        int slot,
        float confidenceThreshold,
        float nmsThreshold,
        std::vector<Detection>& detections,
        std::string& error) const;
    bool decodeDetectionsForSlot(
        int executionBatchSize,
        int slot,
        float confidenceThreshold,
        float nmsThreshold,
        std::vector<Detection>& detections,
        std::string& error) const;
    void appendWorkerDetections(
        WorkerDetectionResult& result,
        const std::vector<Detection>& detections,
        const LetterboxTransform& transform) const;
    std::string choosePrimaryOutputName() const;
    void releaseBindings();
    void reset();

    int inferClassCountFromShape(const std::vector<int64_t>& shape) const;
    LetterboxTransform computeLetterboxTransform(int imageWidth, int imageHeight) const;
    size_t getSizeByDim(const nvinfer1::Dims& dims) const;
    size_t getElementSize(nvinfer1::DataType dtype) const;
};

#endif
