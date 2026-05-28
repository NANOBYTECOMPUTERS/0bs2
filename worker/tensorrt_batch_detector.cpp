#ifdef USE_CUDA

#include "tensorrt_batch_detector.h"

#include <NvInferPlugin.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

#include "cuda_preprocess.h"

namespace
{
    class WorkerTrtLogger : public nvinfer1::ILogger
    {
    public:
        void log(Severity severity, char const* msg) noexcept override
        {
            if (severity <= Severity::kWARNING)
            {
                std::cerr << "[YOLO Worker TensorRT] " << (msg ? msg : "") << std::endl;
            }
        }
    };

    WorkerTrtLogger gWorkerTrtLogger;

    bool setError(std::string& error, const std::string& message)
    {
        error = message;
        return false;
    }

    bool tryGetPositiveDimInt(int64_t value, int& out)
    {
        if (value <= 0 || value > std::numeric_limits<int>::max())
        {
            return false;
        }
        out = static_cast<int>(value);
        return true;
    }

    bool ensureTensorRtPlugins(std::string& error);

    std::unique_ptr<nvinfer1::ICudaEngine> loadSerializedEngine(
        const std::string& enginePath,
        nvinfer1::IRuntime* runtime,
        std::string& error)
    {
        if (!runtime)
        {
            setError(error, "TensorRT runtime is not initialized");
            return nullptr;
        }

        std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
        if (!file)
        {
            setError(error, "Failed to open TensorRT engine: " + enginePath);
            return nullptr;
        }

        const std::streamsize size = file.tellg();
        if (size <= 0)
        {
            setError(error, "TensorRT engine is empty: " + enginePath);
            return nullptr;
        }

        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(static_cast<size_t>(size));
        if (!file.read(buffer.data(), size))
        {
            setError(error, "Failed to read TensorRT engine: " + enginePath);
            return nullptr;
        }

        if (!ensureTensorRtPlugins(error))
        {
            return nullptr;
        }

        std::unique_ptr<nvinfer1::ICudaEngine> loaded(
            runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
        if (!loaded)
        {
            setError(error, "Failed to deserialize TensorRT engine: " + enginePath);
        }
        return loaded;
    }

    std::vector<int64_t> dimsToShape(const nvinfer1::Dims& dims)
    {
        std::vector<int64_t> shape;
        shape.reserve(static_cast<size_t>(std::max(0, dims.nbDims)));
        for (int i = 0; i < dims.nbDims; ++i)
        {
            shape.push_back(dims.d[i]);
        }
        return shape;
    }

    bool dimsHaveDynamicValue(const nvinfer1::Dims& dims)
    {
        for (int i = 0; i < dims.nbDims; ++i)
        {
            if (dims.d[i] <= 0)
            {
                return true;
            }
        }
        return false;
    }

    bool validDims(const nvinfer1::Dims& dims)
    {
        if (dims.nbDims <= 0)
        {
            return false;
        }
        for (int i = 0; i < dims.nbDims; ++i)
        {
            if (dims.d[i] <= 0)
            {
                return false;
            }
        }
        return true;
    }

    bool getPositiveDimOrFallback(int64_t value, int fallback, int& out)
    {
        if (tryGetPositiveDimInt(value, out))
        {
            return true;
        }
        out = fallback;
        return out > 0;
    }

    bool resolveMatrixShape(const std::vector<int64_t>& shape, int64_t& rows, int64_t& cols)
    {
        if (shape.size() == 3)
        {
            rows = shape[1];
            cols = shape[2];
            return rows > 0 && cols > 0;
        }
        if (shape.size() == 2)
        {
            rows = shape[0];
            cols = shape[1];
            return rows > 0 && cols > 0;
        }
        return false;
    }

    int clampInt(int value, int lower, int upper)
    {
        return std::max(lower, std::min(value, upper));
    }

    bool squeezeMatrixShape(const std::vector<int64_t>& shape, int64_t& rows, int64_t& cols)
    {
        if (shape.size() == 3 && shape[0] > 0)
        {
            rows = shape[1];
            cols = shape[2];
            return rows > 0 && cols > 0;
        }
        if (shape.size() == 2)
        {
            rows = shape[0];
            cols = shape[1];
            return rows > 0 && cols > 0;
        }
        return false;
    }

    size_t elementCountFromShape(const std::vector<int64_t>& shape)
    {
        size_t count = 1;
        for (const int64_t dim : shape)
        {
            if (dim <= 0) return 0;
            count *= static_cast<size_t>(dim);
        }
        return count;
    }

    bool looksLikeVectorShape(const std::vector<int64_t>& shape)
    {
        if (shape.empty()) return false;
        if (shape.size() == 1) return shape[0] > 0;
        if (shape.size() == 2) return (shape[0] == 1 && shape[1] > 0) || (shape[1] == 1 && shape[0] > 0);
        return false;
    }

    bool looksLikeBoxesShape(const std::vector<int64_t>& shape, size_t& count)
    {
        if (shape.size() == 3 && shape[0] > 0 && shape[2] == 4 && shape[1] > 0)
        {
            count = static_cast<size_t>(shape[1]);
            return true;
        }
        if (shape.size() == 2 && shape[1] == 4 && shape[0] > 0)
        {
            count = static_cast<size_t>(shape[0]);
            return true;
        }
        return false;
    }

    std::string lowerName(std::string value)
    {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        return value;
    }

    bool ensureTensorRtPlugins(std::string& error)
    {
        static bool attempted = false;
        static bool initialized = false;
        if (!attempted)
        {
            initialized = initLibNvInferPlugins(&gWorkerTrtLogger, "");
            attempted = true;
        }
        if (!initialized)
        {
            return setError(error, "Failed to initialize TensorRT plugins");
        }
        return true;
    }
}

TensorRtBatchDetector::TensorRtBatchDetector() = default;

TensorRtBatchDetector::~TensorRtBatchDetector()
{
    reset();
}

void TensorRtBatchDetector::setClassCount(int classCount)
{
    explicitClassCount = std::max(0, classCount);
}

void TensorRtBatchDetector::reset()
{
    releaseBindings();
    outputHostBuffers.clear();
    outputFloatBuffers.clear();
    outputTypes.clear();
    outputShapes.clear();
    inputSizes.clear();
    outputSizes.clear();
    inputNames.clear();
    outputNames.clear();
    fp16OutputScratch.clear();
    inputName.clear();
    minBatchSize = 1;
    maxBatchSize = 1;
    currentBatchSize = 0;
    inputHasBatchDim = false;
    inputDynamic = false;
    context.reset();
    engine.reset();
    runtime.reset();
    gpuFrameBuffer.release();
    gpuBgrBuffer.release();
    gpuResizedBuffer.release();
    gpuLetterboxBuffer.release();
    gpuFloatBuffer.release();
    if (stream)
    {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }
}

void TensorRtBatchDetector::releaseBindings()
{
    for (auto& binding : inputBindings)
    {
        if (binding.second)
        {
            cudaFree(binding.second);
        }
    }
    for (auto& binding : outputBindings)
    {
        if (binding.second)
        {
            cudaFree(binding.second);
        }
    }
    inputBindings.clear();
    outputBindings.clear();
}

bool TensorRtBatchDetector::ensureStream(std::string& error)
{
    if (stream)
    {
        return true;
    }

    const cudaError_t status = cudaStreamCreate(&stream);
    if (status != cudaSuccess)
    {
        return setError(error, std::string("cudaStreamCreate failed: ") + cudaGetErrorString(status));
    }

    cvStream = cv::cuda::StreamAccessor::wrapStream(stream);
    return true;
}

bool TensorRtBatchDetector::loadEngine(const std::string& enginePath, std::string& error)
{
    reset();

    if (enginePath.empty())
    {
        return setError(error, "loadEngine requires a TensorRT engine path");
    }
    if (!std::filesystem::exists(std::filesystem::path(enginePath)))
    {
        return setError(error, "TensorRT engine does not exist: " + enginePath);
    }
    if (!ensureStream(error))
    {
        return false;
    }

    runtime.reset(nvinfer1::createInferRuntime(gWorkerTrtLogger));
    if (!runtime)
    {
        return setError(error, "Failed to create TensorRT runtime");
    }

    engine = loadSerializedEngine(enginePath, runtime.get(), error);
    if (!engine)
    {
        return false;
    }

    context.reset(engine->createExecutionContext());
    if (!context)
    {
        return setError(error, "Failed to create TensorRT execution context");
    }

    for (int i = 0; i < engine->getNbIOTensors(); ++i)
    {
        const char* tensorName = engine->getIOTensorName(i);
        if (!tensorName)
        {
            continue;
        }
        if (engine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT)
        {
            inputNames.emplace_back(tensorName);
        }
        else if (engine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kOUTPUT)
        {
            outputNames.emplace_back(tensorName);
            outputTypes[tensorName] = engine->getTensorDataType(tensorName);
        }
    }

    if (inputNames.empty())
    {
        return setError(error, "TensorRT engine has no input tensors");
    }
    if (outputNames.empty())
    {
        return setError(error, "TensorRT engine has no output tensors");
    }

    inputName = inputNames.front();
    const nvinfer1::Dims engineInputDims = engine->getTensorShape(inputName.c_str());
    inputDynamic = dimsHaveDynamicValue(engineInputDims);
    inputHasBatchDim = engineInputDims.nbDims == 4;

    if (engineInputDims.nbDims != 3 && engineInputDims.nbDims != 4)
    {
        return setError(error, "TensorRT input rank must be 3 or 4");
    }

    nvinfer1::Dims referenceInputDims = engineInputDims;
    if (inputDynamic && engine->getNbOptimizationProfiles() > 0)
    {
        if (!context->setOptimizationProfileAsync(0, stream))
        {
            return setError(error, "Failed to select TensorRT optimization profile 0");
        }
        referenceInputDims = engine->getProfileShape(
            inputName.c_str(),
            0,
            nvinfer1::OptProfileSelector::kOPT);
        const nvinfer1::Dims minInputDims = engine->getProfileShape(
            inputName.c_str(),
            0,
            nvinfer1::OptProfileSelector::kMIN);
        const nvinfer1::Dims maxInputDims = engine->getProfileShape(
            inputName.c_str(),
            0,
            nvinfer1::OptProfileSelector::kMAX);

        if (maxInputDims.nbDims == referenceInputDims.nbDims)
        {
            const int startDim = inputHasBatchDim ? 1 : 0;
            for (int dim = startDim; dim < referenceInputDims.nbDims; ++dim)
            {
                if (maxInputDims.d[dim] > 0)
                {
                    referenceInputDims.d[dim] = maxInputDims.d[dim];
                }
            }
        }

        if (inputHasBatchDim)
        {
            if (!tryGetPositiveDimInt(minInputDims.d[0], minBatchSize))
            {
                minBatchSize = 1;
            }
            if (!tryGetPositiveDimInt(maxInputDims.d[0], maxBatchSize))
            {
                maxBatchSize = std::max(1, minBatchSize);
            }
        }
    }

    if (inputHasBatchDim)
    {
        if (!inputDynamic)
        {
            if (!tryGetPositiveDimInt(engineInputDims.d[0], maxBatchSize))
            {
                return setError(error, "TensorRT input batch dimension must be positive");
            }
            minBatchSize = maxBatchSize;
        }

        if (!getPositiveDimOrFallback(referenceInputDims.d[1], 3, inputChannels) ||
            !getPositiveDimOrFallback(referenceInputDims.d[2], 640, inputHeight) ||
            !getPositiveDimOrFallback(referenceInputDims.d[3], 640, inputWidth))
        {
            return setError(error, "TensorRT input shape must be NCHW with positive dimensions");
        }
    }
    else
    {
        minBatchSize = 1;
        maxBatchSize = 1;
        if (!getPositiveDimOrFallback(referenceInputDims.d[0], 3, inputChannels) ||
            !getPositiveDimOrFallback(referenceInputDims.d[1], 640, inputHeight) ||
            !getPositiveDimOrFallback(referenceInputDims.d[2], 640, inputWidth))
        {
            return setError(error, "TensorRT input shape must be CHW with positive dimensions");
        }
    }

    if (inputChannels != 3)
    {
        return setError(error, "TensorRT worker expects a 3-channel input tensor");
    }

    minBatchSize = std::max(1, minBatchSize);
    maxBatchSize = std::max(minBatchSize, maxBatchSize);
    const int initialBatchSize = inputDynamic ? minBatchSize : maxBatchSize;
    if (!configureBatch(initialBatchSize, error))
    {
        return false;
    }

    numClasses = explicitClassCount;
    for (const auto& name : outputNames)
    {
        if (explicitClassCount <= 0)
        {
            numClasses = std::max(numClasses, inferClassCountFromShape(outputShapes[name]));
        }
    }

    gpuResizedBuffer.create(inputHeight, inputWidth, CV_8UC3);
    gpuLetterboxBuffer.create(inputHeight, inputWidth, CV_8UC3);
    gpuFloatBuffer.create(inputHeight, inputWidth, CV_32FC3);
    return true;
}

bool TensorRtBatchDetector::resolveBatchShape(int batchSize, nvinfer1::Dims& resolved, std::string& error) const
{
    if (!engine)
    {
        return setError(error, "TensorRT engine is not loaded");
    }

    resolved = engine->getTensorShape(inputName.c_str());
    if (resolved.nbDims != 3 && resolved.nbDims != 4)
    {
        return setError(error, "TensorRT input rank must be 3 or 4");
    }

    if (resolved.nbDims == 4)
    {
        resolved.d[0] = batchSize;
        if (resolved.d[1] <= 0) resolved.d[1] = inputChannels;
        if (resolved.d[2] <= 0) resolved.d[2] = inputHeight;
        if (resolved.d[3] <= 0) resolved.d[3] = inputWidth;
    }
    else
    {
        if (batchSize != 1)
        {
            return setError(error, "TensorRT input tensor does not have a batch dimension");
        }
        if (resolved.d[0] <= 0) resolved.d[0] = inputChannels;
        if (resolved.d[1] <= 0) resolved.d[1] = inputHeight;
        if (resolved.d[2] <= 0) resolved.d[2] = inputWidth;
    }

    if (!validDims(resolved))
    {
        return setError(error, "TensorRT input shape could not be resolved");
    }
    return true;
}

bool TensorRtBatchDetector::refreshTensorShapesAndBindings(std::string& error)
{
    if (!context || !engine)
    {
        return setError(error, "TensorRT engine is not loaded");
    }

    const int32_t shapeStatus = context->inferShapes(0, nullptr);
    if (shapeStatus != 0)
    {
        if (shapeStatus > 0)
        {
            return setError(error, "TensorRT input shapes are not fully specified");
        }
        return setError(error, "TensorRT shape inference failed");
    }

    inputSizes.clear();
    outputSizes.clear();
    outputShapes.clear();

    for (const auto& name : inputNames)
    {
        const nvinfer1::Dims dims = context->getTensorShape(name.c_str());
        inputSizes[name] = getSizeByDim(dims) * getElementSize(engine->getTensorDataType(name.c_str()));
    }
    for (const auto& name : outputNames)
    {
        const nvinfer1::Dims dims = context->getTensorShape(name.c_str());
        outputShapes[name] = dimsToShape(dims);
        outputSizes[name] = getSizeByDim(dims) * getElementSize(engine->getTensorDataType(name.c_str()));
    }

    return allocateBindings(error);
}

bool TensorRtBatchDetector::configureBatch(int batchSize, std::string& error)
{
    if (!context)
    {
        return setError(error, "TensorRT engine is not loaded");
    }
    if (batchSize < 1 || batchSize > maxBatchSize)
    {
        return setError(error, "Requested TensorRT batch size is outside the loaded engine profile");
    }

    if (currentBatchSize == batchSize && !inputBindings.empty() && !outputBindings.empty())
    {
        return true;
    }

    if (inputDynamic)
    {
        nvinfer1::Dims resolved;
        if (!resolveBatchShape(batchSize, resolved, error))
        {
            return false;
        }
        if (!context->setInputShape(inputName.c_str(), resolved))
        {
            return setError(error, "Failed to set TensorRT input shape for requested batch");
        }
    }
    else if (inputHasBatchDim && batchSize != maxBatchSize)
    {
        return setError(error, "Static TensorRT engine requires its fixed batch size");
    }

    currentBatchSize = batchSize;
    return refreshTensorShapesAndBindings(error);
}

bool TensorRtBatchDetector::allocateBindings(std::string& error)
{
    releaseBindings();
    outputHostBuffers.clear();
    outputFloatBuffers.clear();

    for (const auto& name : inputNames)
    {
        const size_t bytes = inputSizes[name];
        if (bytes == 0)
        {
            return setError(error, "Input tensor has unsupported shape or data type: " + name);
        }

        void* ptr = nullptr;
        const cudaError_t status = cudaMalloc(&ptr, bytes);
        if (status != cudaSuccess)
        {
            return setError(error, std::string("cudaMalloc input failed: ") + cudaGetErrorString(status));
        }
        inputBindings[name] = ptr;
        if (!context->setTensorAddress(name.c_str(), ptr))
        {
            return setError(error, "Failed to bind TensorRT input tensor: " + name);
        }
    }

    for (const auto& name : outputNames)
    {
        const size_t bytes = outputSizes[name];
        if (bytes == 0)
        {
            return setError(error, "Output tensor has unsupported shape or data type: " + name);
        }

        void* ptr = nullptr;
        const cudaError_t status = cudaMalloc(&ptr, bytes);
        if (status != cudaSuccess)
        {
            return setError(error, std::string("cudaMalloc output failed: ") + cudaGetErrorString(status));
        }
        outputBindings[name] = ptr;
        outputHostBuffers[name].resize(bytes);
        if (!context->setTensorAddress(name.c_str(), ptr))
        {
            return setError(error, "Failed to bind TensorRT output tensor: " + name);
        }
    }

    return true;
}

LetterboxTransform TensorRtBatchDetector::computeLetterboxTransform(int imageWidth, int imageHeight) const
{
    LetterboxTransform transform;
    if (imageWidth <= 0 || imageHeight <= 0 || inputWidth <= 0 || inputHeight <= 0)
    {
        return transform;
    }

    transform.gain = std::min(
        static_cast<float>(inputWidth) / static_cast<float>(imageWidth),
        static_cast<float>(inputHeight) / static_cast<float>(imageHeight));
    transform.resizedWidth = std::max(1, static_cast<int>(std::round(imageWidth * transform.gain)));
    transform.resizedHeight = std::max(1, static_cast<int>(std::round(imageHeight * transform.gain)));
    transform.resizedWidth = std::min(transform.resizedWidth, inputWidth);
    transform.resizedHeight = std::min(transform.resizedHeight, inputHeight);
    transform.padX = static_cast<float>(inputWidth - transform.resizedWidth) * 0.5f;
    transform.padY = static_cast<float>(inputHeight - transform.resizedHeight) * 0.5f;
    return transform;
}

bool TensorRtBatchDetector::preprocess(
    const cv::Mat& frame,
    int slot,
    LetterboxTransform& transform,
    std::string& error)
{
    if (frame.empty())
    {
        return setError(error, "Cannot preprocess an empty image");
    }

    const auto inputIt = inputBindings.find(inputName);
    if (inputIt == inputBindings.end() || !inputIt->second)
    {
        return setError(error, "TensorRT input buffer is not allocated");
    }
    if (slot < 0 || slot >= currentBatchSize)
    {
        return setError(error, "TensorRT preprocessing slot is outside the active batch");
    }

    gpuFrameBuffer.upload(frame, cvStream);

    cv::cuda::GpuMat bgrFrame;
    if (frame.channels() == 4)
    {
        cv::cuda::cvtColor(gpuFrameBuffer, gpuBgrBuffer, cv::COLOR_BGRA2BGR, 0, cvStream);
        bgrFrame = gpuBgrBuffer;
    }
    else if (frame.channels() == 1)
    {
        cv::cuda::cvtColor(gpuFrameBuffer, gpuBgrBuffer, cv::COLOR_GRAY2BGR, 0, cvStream);
        bgrFrame = gpuBgrBuffer;
    }
    else if (frame.channels() == 3)
    {
        bgrFrame = gpuFrameBuffer;
    }
    else
    {
        return setError(error, "Unsupported image channel count");
    }

    transform = computeLetterboxTransform(frame.cols, frame.rows);
    lastLetterboxTransform = transform;
    const int padLeft = static_cast<int>(std::round(transform.padX));
    const int padTop = static_cast<int>(std::round(transform.padY));

    cv::cuda::resize(
        bgrFrame,
        gpuResizedBuffer,
        cv::Size(transform.resizedWidth, transform.resizedHeight),
        0,
        0,
        cv::INTER_LINEAR,
        cvStream);

    gpuLetterboxBuffer.setTo(cv::Scalar(114, 114, 114), cvStream);
    cv::cuda::GpuMat roi = gpuLetterboxBuffer(
        cv::Rect(padLeft, padTop, transform.resizedWidth, transform.resizedHeight));
    gpuResizedBuffer.copyTo(roi, cvStream);

    gpuLetterboxBuffer.convertTo(gpuFloatBuffer, CV_32FC3, 1.0 / 255.0, 0.0, cvStream);
    const size_t slotElements = static_cast<size_t>(inputChannels) *
        static_cast<size_t>(inputWidth) *
        static_cast<size_t>(inputHeight);
    float* dst = reinterpret_cast<float*>(inputIt->second);
    if (inputHasBatchDim)
    {
        dst += static_cast<size_t>(slot) * slotElements;
    }
    launch_hwc_to_chw_norm(
        gpuFloatBuffer,
        dst,
        inputWidth,
        inputHeight,
        stream);

    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        return setError(error, std::string("CUDA preprocess failed: ") + cudaGetErrorString(status));
    }

    return true;
}

bool TensorRtBatchDetector::copyOutputsToHost(std::string& error)
{
    for (const auto& name : outputNames)
    {
        const cudaError_t copyStatus = cudaMemcpyAsync(
            outputHostBuffers[name].data(),
            outputBindings[name],
            outputSizes[name],
            cudaMemcpyDeviceToHost,
            stream);
        if (copyStatus != cudaSuccess)
        {
            return setError(error, std::string("Failed to copy TensorRT output: ") + cudaGetErrorString(copyStatus));
        }
    }

    const cudaError_t syncStatus = cudaStreamSynchronize(stream);
    if (syncStatus != cudaSuccess)
    {
        return setError(error, std::string("CUDA stream synchronization failed: ") + cudaGetErrorString(syncStatus));
    }

    outputFloatBuffers.clear();
    for (const auto& name : outputNames)
    {
        std::vector<float>& output = outputFloatBuffers[name];
        if (!copyOutputToFloat(name, output, error))
        {
            return false;
        }
    }

    return true;
}

bool TensorRtBatchDetector::copyOutputToFloat(const std::string& outputName, std::vector<float>& out, std::string& error)
{
    const auto hostIt = outputHostBuffers.find(outputName);
    const auto sizeIt = outputSizes.find(outputName);
    const auto typeIt = outputTypes.find(outputName);
    if (hostIt == outputHostBuffers.end() || sizeIt == outputSizes.end() || typeIt == outputTypes.end())
    {
        return setError(error, "TensorRT output buffer is missing: " + outputName);
    }

    const unsigned char* bytes = hostIt->second.data();
    const size_t byteCount = sizeIt->second;
    switch (typeIt->second)
    {
    case nvinfer1::DataType::kFLOAT:
    {
        const size_t elements = byteCount / sizeof(float);
        const float* values = reinterpret_cast<const float*>(bytes);
        out.assign(values, values + elements);
        return true;
    }
    case nvinfer1::DataType::kHALF:
    {
        const size_t elements = byteCount / sizeof(__half);
        const __half* values = reinterpret_cast<const __half*>(bytes);
        out.resize(elements);
        for (size_t i = 0; i < elements; ++i)
        {
            out[i] = __half2float(values[i]);
        }
        return true;
    }
    case nvinfer1::DataType::kINT32:
    {
        const size_t elements = byteCount / sizeof(int32_t);
        const int32_t* values = reinterpret_cast<const int32_t*>(bytes);
        out.resize(elements);
        for (size_t i = 0; i < elements; ++i)
        {
            out[i] = static_cast<float>(values[i]);
        }
        return true;
    }
    case nvinfer1::DataType::kINT8:
    {
        const size_t elements = byteCount / sizeof(int8_t);
        const int8_t* values = reinterpret_cast<const int8_t*>(bytes);
        out.resize(elements);
        for (size_t i = 0; i < elements; ++i)
        {
            out[i] = static_cast<float>(values[i]);
        }
        return true;
    }
    case nvinfer1::DataType::kBOOL:
    {
        const size_t elements = byteCount / sizeof(bool);
        const bool* values = reinterpret_cast<const bool*>(bytes);
        out.resize(elements);
        for (size_t i = 0; i < elements; ++i)
        {
            out[i] = values[i] ? 1.0f : 0.0f;
        }
        return true;
    }
    default:
        return setError(error, "Unsupported TensorRT output type: " + outputName);
    }
}

bool TensorRtBatchDetector::outputSliceForSlot(
    const std::string& outputName,
    int executionBatchSize,
    int slot,
    const float*& data,
    std::vector<int64_t>& shape,
    std::string& error) const
{
    data = nullptr;
    shape.clear();

    const auto bufferIt = outputFloatBuffers.find(outputName);
    const auto shapeIt = outputShapes.find(outputName);
    if (bufferIt == outputFloatBuffers.end() || shapeIt == outputShapes.end())
    {
        return setError(error, "TensorRT output buffer is missing: " + outputName);
    }
    if (slot < 0 || slot >= executionBatchSize)
    {
        return setError(error, "TensorRT output slot is outside the active batch");
    }

    const std::vector<float>& buffer = bufferIt->second;
    const std::vector<int64_t>& fullShape = shapeIt->second;
    if (executionBatchSize <= 1)
    {
        data = buffer.data();
        shape = fullShape;
        return true;
    }

    if (fullShape.empty() || fullShape[0] != executionBatchSize)
    {
        return setError(error, "TensorRT output tensor cannot be safely split by batch: " + outputName);
    }

    shape.assign(fullShape.begin() + 1, fullShape.end());
    const size_t perSlotElements = elementCountFromShape(shape);
    const size_t offset = static_cast<size_t>(slot) * perSlotElements;
    if (perSlotElements == 0 || offset + perSlotElements > buffer.size())
    {
        return setError(error, "TensorRT output tensor has an invalid batched shape: " + outputName);
    }

    data = buffer.data() + offset;
    return true;
}

bool TensorRtBatchDetector::decodeMultiOutputNmsForSlot(
    int executionBatchSize,
    int slot,
    float confidenceThreshold,
    float nmsThreshold,
    std::vector<Detection>& detections,
    std::string& error) const
{
    std::string boxesName;
    std::string scoresName;
    std::string classesName;
    std::string countName;
    size_t boxCount = 0;
    std::vector<std::string> vectorOutputs;
    std::map<std::string, const float*> outputData;
    std::map<std::string, std::vector<int64_t>> slotShapes;

    for (const auto& name : outputNames)
    {
        const float* data = nullptr;
        std::vector<int64_t> slotShape;
        if (!outputSliceForSlot(name, executionBatchSize, slot, data, slotShape, error))
        {
            return false;
        }
        outputData[name] = data;
        slotShapes[name] = slotShape;

        size_t candidateBoxCount = 0;
        const std::string lowered = lowerName(name);
        if (looksLikeBoxesShape(slotShape, candidateBoxCount))
        {
            if (boxesName.empty() || lowered.find("box") != std::string::npos)
            {
                boxesName = name;
                boxCount = candidateBoxCount;
            }
            continue;
        }

        const bool scalarShape = slotShape.empty();
        if (scalarShape || looksLikeVectorShape(slotShape))
        {
            if ((scalarShape || elementCountFromShape(slotShape) == 1) && countName.empty())
            {
                countName = name;
                continue;
            }
            vectorOutputs.push_back(name);
            if (scoresName.empty() && (lowered.find("score") != std::string::npos || lowered.find("conf") != std::string::npos))
            {
                scoresName = name;
            }
            else if (classesName.empty() && (lowered.find("class") != std::string::npos || lowered.find("label") != std::string::npos))
            {
                classesName = name;
            }
        }
    }

    if (boxesName.empty())
    {
        return false;
    }
    if (scoresName.empty() || classesName.empty())
    {
        for (const auto& name : vectorOutputs)
        {
            if (name == scoresName || name == classesName)
            {
                continue;
            }
            if (scoresName.empty())
            {
                scoresName = name;
                continue;
            }
            if (classesName.empty())
            {
                classesName = name;
                break;
            }
        }
    }
    if (scoresName.empty() || classesName.empty())
    {
        return false;
    }

    const float* boxes = outputData[boxesName];
    const float* scores = outputData[scoresName];
    const float* classes = outputData[classesName];
    size_t detectionCount = std::min({
        boxCount,
        elementCountFromShape(slotShapes[scoresName]),
        elementCountFromShape(slotShapes[classesName]) });
    if (!countName.empty() && outputData[countName])
    {
        const int reportedCount = static_cast<int>(std::round(outputData[countName][0]));
        if (reportedCount >= 0)
        {
            detectionCount = std::min(detectionCount, static_cast<size_t>(reportedCount));
        }
    }

    for (size_t i = 0; i < detectionCount; ++i)
    {
        const float confidence = scores[i];
        if (confidence <= confidenceThreshold)
        {
            continue;
        }

        const size_t boxOffset = i * 4;
        Detection detection;
        detection.box.x = static_cast<int>(boxes[boxOffset]);
        detection.box.y = static_cast<int>(boxes[boxOffset + 1]);
        detection.box.width = static_cast<int>(boxes[boxOffset + 2] - boxes[boxOffset]);
        detection.box.height = static_cast<int>(boxes[boxOffset + 3] - boxes[boxOffset + 1]);
        detection.confidence = confidence;
        detection.classId = static_cast<int>(std::round(classes[i]));
        detections.push_back(detection);
    }

    NMS(detections, nmsThreshold, nullptr);
    return true;
}

bool TensorRtBatchDetector::decodeMultiOutputNms(
    float confidenceThreshold,
    float nmsThreshold,
    std::vector<Detection>& detections)
{
    std::string boxesName;
    std::string scoresName;
    std::string classesName;
    std::string countName;
    size_t boxCount = 0;
    std::vector<std::string> vectorOutputs;

    for (const auto& name : outputNames)
    {
        size_t candidateBoxCount = 0;
        const auto shapeIt = outputShapes.find(name);
        if (shapeIt == outputShapes.end())
        {
            continue;
        }

        const std::string lowered = lowerName(name);
        if (looksLikeBoxesShape(shapeIt->second, candidateBoxCount))
        {
            if (boxesName.empty() || lowered.find("box") != std::string::npos)
            {
                boxesName = name;
                boxCount = candidateBoxCount;
            }
            continue;
        }

        if (looksLikeVectorShape(shapeIt->second))
        {
            if (elementCountFromShape(shapeIt->second) == 1 && countName.empty())
            {
                countName = name;
                continue;
            }
            vectorOutputs.push_back(name);
            if (scoresName.empty() && (lowered.find("score") != std::string::npos || lowered.find("conf") != std::string::npos))
            {
                scoresName = name;
            }
            else if (classesName.empty() && (lowered.find("class") != std::string::npos || lowered.find("label") != std::string::npos))
            {
                classesName = name;
            }
        }
    }

    if (boxesName.empty())
    {
        return false;
    }
    if (scoresName.empty() || classesName.empty())
    {
        for (const auto& name : vectorOutputs)
        {
            if (name == scoresName || name == classesName)
            {
                continue;
            }
            if (scoresName.empty())
            {
                scoresName = name;
                continue;
            }
            if (classesName.empty())
            {
                classesName = name;
                break;
            }
        }
    }
    if (scoresName.empty() || classesName.empty())
    {
        return false;
    }

    const auto& boxes = outputFloatBuffers[boxesName];
    const auto& scores = outputFloatBuffers[scoresName];
    const auto& classes = outputFloatBuffers[classesName];
    size_t detectionCount = std::min({ boxCount, scores.size(), classes.size() });
    if (!countName.empty() && !outputFloatBuffers[countName].empty())
    {
        const int reportedCount = static_cast<int>(std::round(outputFloatBuffers[countName][0]));
        if (reportedCount >= 0)
        {
            detectionCount = std::min(detectionCount, static_cast<size_t>(reportedCount));
        }
    }

    for (size_t i = 0; i < detectionCount; ++i)
    {
        const float confidence = scores[i];
        if (confidence <= confidenceThreshold)
        {
            continue;
        }

        const size_t boxOffset = i * 4;
        if (boxOffset + 3 >= boxes.size())
        {
            break;
        }

        Detection detection;
        detection.box.x = static_cast<int>(boxes[boxOffset]);
        detection.box.y = static_cast<int>(boxes[boxOffset + 1]);
        detection.box.width = static_cast<int>(boxes[boxOffset + 2] - boxes[boxOffset]);
        detection.box.height = static_cast<int>(boxes[boxOffset + 3] - boxes[boxOffset + 1]);
        detection.confidence = confidence;
        detection.classId = static_cast<int>(std::round(classes[i]));
        detections.push_back(detection);
    }

    NMS(detections, nmsThreshold, nullptr);
    return true;
}

std::string TensorRtBatchDetector::choosePrimaryOutputName() const
{
    std::string bestName;
    size_t bestElements = 0;
    for (const auto& name : outputNames)
    {
        const auto shapeIt = outputShapes.find(name);
        if (shapeIt == outputShapes.end())
        {
            continue;
        }

        int64_t rows = 0;
        int64_t cols = 0;
        if (!squeezeMatrixShape(shapeIt->second, rows, cols))
        {
            continue;
        }

        const size_t elements = elementCountFromShape(shapeIt->second);
        if (elements > bestElements)
        {
            bestElements = elements;
            bestName = name;
        }
    }
    return bestName;
}

bool TensorRtBatchDetector::decodeDetectionsForSlot(
    int executionBatchSize,
    int slot,
    float confidenceThreshold,
    float nmsThreshold,
    std::vector<Detection>& detections,
    std::string& error) const
{
    detections.clear();
    if (decodeMultiOutputNmsForSlot(
        executionBatchSize,
        slot,
        confidenceThreshold,
        nmsThreshold,
        detections,
        error))
    {
        return true;
    }
    if (!error.empty())
    {
        return false;
    }

    const std::string outputName = choosePrimaryOutputName();
    if (outputName.empty())
    {
        return setError(error, "No supported TensorRT output tensor was found");
    }

    const float* output = nullptr;
    std::vector<int64_t> outputShape;
    if (!outputSliceForSlot(outputName, executionBatchSize, slot, output, outputShape, error))
    {
        return false;
    }

    std::chrono::duration<double, std::milli> nmsTime;
    detections = postProcessYoloScaled(
        output,
        outputShape,
        numClasses,
        confidenceThreshold,
        nmsThreshold,
        1.0f,
        &nmsTime);
    return true;
}

void TensorRtBatchDetector::appendWorkerDetections(
    WorkerDetectionResult& result,
    const std::vector<Detection>& detections,
    const LetterboxTransform& transform) const
{
    const cv::Rect imageBounds(0, 0, result.width, result.height);

    result.detections.reserve(result.detections.size() + detections.size());
    for (const auto& det : detections)
    {
        const float x1 = (static_cast<float>(det.box.x) - transform.padX) / transform.gain;
        const float y1 = (static_cast<float>(det.box.y) - transform.padY) / transform.gain;
        const float x2 = (static_cast<float>(det.box.x + det.box.width) - transform.padX) / transform.gain;
        const float y2 = (static_cast<float>(det.box.y + det.box.height) - transform.padY) / transform.gain;

        cv::Rect imageBox;
        imageBox.x = static_cast<int>(std::round(x1));
        imageBox.y = static_cast<int>(std::round(y1));
        imageBox.width = static_cast<int>(std::round(x2 - x1));
        imageBox.height = static_cast<int>(std::round(y2 - y1));
        imageBox = imageBox & imageBounds;
        if (imageBox.width <= 0 || imageBox.height <= 0)
        {
            continue;
        }

        WorkerDetection workerDetection;
        workerDetection.classId = det.classId;
        workerDetection.confidence = det.confidence;
        workerDetection.x = clampInt(imageBox.x, 0, result.width);
        workerDetection.y = clampInt(imageBox.y, 0, result.height);
        workerDetection.width = imageBox.width;
        workerDetection.height = imageBox.height;
        result.detections.push_back(workerDetection);
    }
}

WorkerDetectionResult TensorRtBatchDetector::detectImage(
    const std::string& imagePath,
    float confidenceThreshold,
    float nmsThreshold)
{
    std::vector<WorkerDetectionResult> results = detectBatch(
        std::vector<std::string>{ imagePath },
        confidenceThreshold,
        nmsThreshold);
    if (results.empty())
    {
        WorkerDetectionResult result;
        result.image = imagePath;
        result.error = "TensorRT batch detection returned no result";
        return result;
    }
    return results.front();
}

std::vector<WorkerDetectionResult> TensorRtBatchDetector::detectBatch(
    const std::vector<std::string>& imagePaths,
    float confidenceThreshold,
    float nmsThreshold)
{
    std::vector<WorkerDetectionResult> results(imagePaths.size());
    std::vector<cv::Mat> frames(imagePaths.size());
    std::vector<size_t> validIndices;
    validIndices.reserve(imagePaths.size());

    for (size_t i = 0; i < imagePaths.size(); ++i)
    {
        results[i].image = imagePaths[i];
        if (!context)
        {
            results[i].error = "TensorRT engine is not loaded";
            continue;
        }

        frames[i] = cv::imread(imagePaths[i], cv::IMREAD_UNCHANGED);
        if (frames[i].empty())
        {
            results[i].error = "Failed to read image";
            continue;
        }

        results[i].width = frames[i].cols;
        results[i].height = frames[i].rows;
        validIndices.push_back(i);
    }

    if (!context || validIndices.empty())
    {
        return results;
    }

    const int chunkCapacity = std::max(1, maxBatchSize);
    for (size_t offset = 0; offset < validIndices.size(); offset += static_cast<size_t>(chunkCapacity))
    {
        const int requestedBatchSize = static_cast<int>(
            std::min(static_cast<size_t>(chunkCapacity), validIndices.size() - offset));
        const int executionBatchSize = inputDynamic
            ? std::max(minBatchSize, requestedBatchSize)
            : (inputHasBatchDim ? maxBatchSize : 1);

        std::string error;
        if (!configureBatch(executionBatchSize, error))
        {
            for (int slot = 0; slot < requestedBatchSize; ++slot)
            {
                results[validIndices[offset + static_cast<size_t>(slot)]].error = error;
            }
            continue;
        }

        std::vector<LetterboxTransform> transforms(static_cast<size_t>(requestedBatchSize));
        bool chunkReady = true;
        for (int slot = 0; slot < requestedBatchSize; ++slot)
        {
            const size_t imageIndex = validIndices[offset + static_cast<size_t>(slot)];
            if (!preprocess(frames[imageIndex], slot, transforms[static_cast<size_t>(slot)], error))
            {
                results[imageIndex].error = error;
                chunkReady = false;
                break;
            }
        }

        if (chunkReady && executionBatchSize > requestedBatchSize)
        {
            const size_t padIndex = validIndices[offset + static_cast<size_t>(requestedBatchSize - 1)];
            LetterboxTransform padTransform;
            for (int slot = requestedBatchSize; slot < executionBatchSize; ++slot)
            {
                if (!preprocess(frames[padIndex], slot, padTransform, error))
                {
                    chunkReady = false;
                    break;
                }
            }
        }

        if (!chunkReady)
        {
            for (int slot = 0; slot < requestedBatchSize; ++slot)
            {
                WorkerDetectionResult& result = results[validIndices[offset + static_cast<size_t>(slot)]];
                if (result.error.empty())
                {
                    result.error = error.empty() ? "TensorRT preprocessing failed" : error;
                }
            }
            continue;
        }

        if (!context->enqueueV3(stream))
        {
            for (int slot = 0; slot < requestedBatchSize; ++slot)
            {
                results[validIndices[offset + static_cast<size_t>(slot)]].error = "TensorRT enqueueV3 failed";
            }
            continue;
        }

        if (!copyOutputsToHost(error))
        {
            for (int slot = 0; slot < requestedBatchSize; ++slot)
            {
                results[validIndices[offset + static_cast<size_t>(slot)]].error = error;
            }
            continue;
        }

        for (int slot = 0; slot < requestedBatchSize; ++slot)
        {
            WorkerDetectionResult& result = results[validIndices[offset + static_cast<size_t>(slot)]];
            std::vector<Detection> detections;
            error.clear();
            if (!decodeDetectionsForSlot(
                executionBatchSize,
                slot,
                confidenceThreshold,
                nmsThreshold,
                detections,
                error))
            {
                result.error = error.empty() ? "TensorRT output decode failed" : error;
                continue;
            }

            appendWorkerDetections(result, detections, transforms[static_cast<size_t>(slot)]);
        }
    }

    return results;
}

int TensorRtBatchDetector::inferClassCountFromShape(const std::vector<int64_t>& shape) const
{
    int64_t rows = 0;
    int64_t cols = 0;
    if (!squeezeMatrixShape(shape, rows, cols))
    {
        return 0;
    }

    if (cols == 6)
    {
        return 0;
    }

    if (rows >= 5 && rows <= cols)
    {
        return static_cast<int>(rows - 4);
    }
    if (cols >= 5)
    {
        return static_cast<int>(cols - 4);
    }

    return 0;
}

size_t TensorRtBatchDetector::getSizeByDim(const nvinfer1::Dims& dims) const
{
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] <= 0)
        {
            return 0;
        }
        size *= static_cast<size_t>(dims.d[i]);
    }
    return size;
}

size_t TensorRtBatchDetector::getElementSize(nvinfer1::DataType dtype) const
{
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT: return sizeof(float);
    case nvinfer1::DataType::kHALF: return sizeof(__half);
    case nvinfer1::DataType::kINT32: return sizeof(int32_t);
    case nvinfer1::DataType::kINT8: return sizeof(int8_t);
    case nvinfer1::DataType::kBOOL: return sizeof(bool);
    default: return 0;
    }
}

#endif
