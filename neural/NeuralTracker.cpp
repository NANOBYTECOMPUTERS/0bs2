#include "NeuralTracker.h"

#include <onnxruntime_cxx_api.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <Windows.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "tensorrt/nvinf.h"
#endif

namespace aim::neural
{
namespace
{
std::filesystem::path resolveNeuralModelPath(const std::string& modelPath)
{
    std::filesystem::path configured(modelPath);
    std::error_code ec;

    if (configured.is_absolute() && std::filesystem::exists(configured, ec))
        return configured;

    std::vector<std::filesystem::path> bases;
    bases.push_back(std::filesystem::current_path(ec));

    char exePath[MAX_PATH] = {};
    if (GetModuleFileNameA(nullptr, exePath, MAX_PATH) > 0)
    {
        std::filesystem::path exeDir = std::filesystem::path(exePath).parent_path();
        bases.push_back(exeDir);
        bases.push_back(exeDir.parent_path());
        bases.push_back(exeDir.parent_path().parent_path());
    }

    for (const auto& base : bases)
    {
        if (base.empty())
            continue;
        std::filesystem::path candidate = base / configured;
        if (std::filesystem::exists(candidate, ec))
            return candidate;
    }

    return configured;
}

float normalizeOutputScore(float value)
{
    if (!std::isfinite(value))
        return 0.5f;
    if (value < 0.0f || value > 1.0f)
        value = 1.0f / (1.0f + std::exp(-value));
    if (value < 0.0f) value = 0.0f;
    if (value > 1.0f) value = 1.0f;
    return value;
}

std::string lowerExtension(const std::filesystem::path& path)
{
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

class OnnxNeuralTracker final : public INeuralTracker
{
public:
    explicit OnnxNeuralTracker(const std::filesystem::path& modelPath)
        : env(ORT_LOGGING_LEVEL_WARNING, "NeuralTracker"),
          memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    {
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
        sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetInterOpNumThreads(1);

        const std::wstring widePath = modelPath.wstring();
        session = Ort::Session(env, widePath.c_str(), sessionOptions);

        auto inputName = session.GetInputNameAllocated(0, allocator);
        auto outputName = session.GetOutputNameAllocated(0, allocator);
        inputNameStorage = inputName.get();
        outputNameStorage = outputName.get();

        auto inputInfo = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        const auto inputShape = inputInfo.GetShape();
        if (!inputShape.empty())
        {
            const int64_t featureDim = inputShape.back();
            if (featureDim > 0 && featureDim != static_cast<int64_t>(NeuralTrackerFeatureCount))
            {
                std::cerr << "[NeuralTracker] ONNX input feature count mismatch. Model expects "
                          << featureDim << ", runtime provides " << NeuralTrackerFeatureCount
                          << "." << std::endl;
                availableFlag = false;
                return;
            }
        }

        availableFlag = !inputNameStorage.empty() && !outputNameStorage.empty();
    }

    bool available() const override
    {
        return availableFlag;
    }

    NeuralTrackerResult score(const NeuralTrackerFeatures& features) override
    {
        NeuralTrackerResult result;
        if (!availableFlag)
            return result;

        try
        {
            std::array<float, NeuralTrackerFeatureCount> featureArray = neuralTrackerFeatureArray(features);
            std::array<int64_t, 2> inputShape{ 1, static_cast<int64_t>(NeuralTrackerFeatureCount) };
            Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                memoryInfo,
                featureArray.data(),
                featureArray.size(),
                inputShape.data(),
                inputShape.size());

            const char* inputNames[] = { inputNameStorage.c_str() };
            const char* outputNames[] = { outputNameStorage.c_str() };
            auto outputs = session.Run(
                Ort::RunOptions{ nullptr },
                inputNames,
                &inputTensor,
                1,
                outputNames,
                1);

            if (outputs.empty() || !outputs[0].IsTensor())
                return result;

            auto info = outputs[0].GetTensorTypeAndShapeInfo();
            if (info.GetElementCount() < 1)
                return result;

            const float* values = outputs[0].GetTensorData<float>();
            result.neuralScore = normalizeOutputScore(values[0]);
            result.valid = true;
        }
        catch (const Ort::Exception& e)
        {
            availableFlag = false;
            std::cerr << "[NeuralTracker] ONNX inference disabled after error: " << e.what() << std::endl;
        }
        catch (...)
        {
            availableFlag = false;
            std::cerr << "[NeuralTracker] ONNX inference disabled after unknown error." << std::endl;
        }

        return result;
    }

private:
    Ort::Env env;
    Ort::SessionOptions sessionOptions;
    Ort::Session session{ nullptr };
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memoryInfo;
    std::string inputNameStorage;
    std::string outputNameStorage;
    bool availableFlag = false;
};

#ifdef USE_CUDA
size_t trtElementCount(const nvinfer1::Dims& dims)
{
    if (dims.nbDims <= 0)
        return 0;

    size_t count = 1;
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] <= 0)
            return 0;
        count *= static_cast<size_t>(dims.d[i]);
    }
    return count;
}

size_t trtElementSize(nvinfer1::DataType type)
{
    switch (type)
    {
    case nvinfer1::DataType::kFLOAT: return sizeof(float);
    case nvinfer1::DataType::kHALF: return sizeof(uint16_t);
    case nvinfer1::DataType::kINT32: return sizeof(int32_t);
    case nvinfer1::DataType::kINT8: return sizeof(int8_t);
    default: return 0;
    }
}

float halfBitsToFloat(uint16_t value)
{
    const int sign = (value & 0x8000u) ? -1 : 1;
    const int exponent = (value >> 10) & 0x1f;
    const int mantissa = value & 0x03ff;

    if (exponent == 0)
    {
        if (mantissa == 0)
            return sign < 0 ? -0.0f : 0.0f;
        return static_cast<float>(sign) * std::ldexp(static_cast<float>(mantissa), -24);
    }
    if (exponent == 31)
    {
        if (mantissa == 0)
            return sign < 0 ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        return std::numeric_limits<float>::quiet_NaN();
    }

    return static_cast<float>(sign) * std::ldexp(static_cast<float>(1024 + mantissa), exponent - 25);
}

class CudaDeviceBuffer
{
public:
    CudaDeviceBuffer() = default;
    CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
    CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;

    ~CudaDeviceBuffer()
    {
        release();
    }

    bool allocate(size_t bytes)
    {
        release();
        if (bytes == 0)
            return false;

        cudaError_t err = cudaMalloc(&ptr_, bytes);
        if (err != cudaSuccess)
        {
            std::cerr << "[NeuralTracker] CUDA allocation failed: "
                      << cudaGetErrorString(err) << std::endl;
            ptr_ = nullptr;
            bytes_ = 0;
            return false;
        }

        bytes_ = bytes;
        return true;
    }

    void release()
    {
        if (ptr_)
        {
            cudaFree(ptr_);
            ptr_ = nullptr;
            bytes_ = 0;
        }
    }

    void* get() const { return ptr_; }
    size_t size() const { return bytes_; }

private:
    void* ptr_ = nullptr;
    size_t bytes_ = 0;
};

class CudaStreamHandle
{
public:
    CudaStreamHandle() = default;
    CudaStreamHandle(const CudaStreamHandle&) = delete;
    CudaStreamHandle& operator=(const CudaStreamHandle&) = delete;

    ~CudaStreamHandle()
    {
        if (stream_)
            cudaStreamDestroy(stream_);
    }

    bool create()
    {
        if (stream_)
            return true;

        cudaError_t err = cudaStreamCreate(&stream_);
        if (err != cudaSuccess)
        {
            std::cerr << "[NeuralTracker] CUDA stream creation failed: "
                      << cudaGetErrorString(err) << std::endl;
            stream_ = nullptr;
            return false;
        }
        return true;
    }

    cudaStream_t get() const { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

std::filesystem::path resolveNeuralEnginePath(const std::filesystem::path& modelPath)
{
    const std::string ext = lowerExtension(modelPath);
    if (ext == ".engine")
        return modelPath;
    if (ext == ".onnx")
    {
        std::filesystem::path enginePath = modelPath;
        enginePath.replace_extension(".engine");
        return enginePath;
    }
    return {};
}

bool buildAndSaveNeuralEngine(const std::filesystem::path& onnxPath, const std::filesystem::path& enginePath)
{
    std::unique_ptr<nvinfer1::ICudaEngine> builtEngine(buildEngineFromOnnx(onnxPath.string(), gLogger));
    if (!builtEngine)
        return false;

    std::unique_ptr<nvinfer1::IHostMemory> serializedEngine(builtEngine->serialize());
    if (!serializedEngine || serializedEngine->size() == 0)
    {
        std::cerr << "[NeuralTracker] TensorRT engine serialization failed." << std::endl;
        return false;
    }

    std::error_code ec;
    if (enginePath.has_parent_path())
        std::filesystem::create_directories(enginePath.parent_path(), ec);

    std::ofstream engineFile(enginePath, std::ios::binary);
    if (!engineFile)
    {
        std::cerr << "[NeuralTracker] Failed to create TensorRT engine file: "
                  << enginePath.string() << std::endl;
        return false;
    }

    engineFile.write(reinterpret_cast<const char*>(serializedEngine->data()), serializedEngine->size());
    if (!engineFile)
    {
        std::cerr << "[NeuralTracker] Failed to write TensorRT engine file: "
                  << enginePath.string() << std::endl;
        return false;
    }

    std::cout << "[NeuralTracker] TensorRT engine saved to: " << enginePath.string() << std::endl;
    return true;
}

class TrtNeuralTracker final : public INeuralTracker
{
public:
    explicit TrtNeuralTracker(const std::filesystem::path& modelPath)
    {
        availableFlag = initialize(modelPath);
    }

    bool available() const override
    {
        return availableFlag;
    }

    NeuralTrackerResult score(const NeuralTrackerFeatures& features) override
    {
        NeuralTrackerResult result;
        if (!availableFlag)
            return result;

        std::lock_guard<std::mutex> lock(inferenceMutex);
        const std::array<float, NeuralTrackerFeatureCount> featureArray = neuralTrackerFeatureArray(features);

        cudaError_t err = cudaMemcpyAsync(
            inputBuffer.get(),
            featureArray.data(),
            inputBytes,
            cudaMemcpyHostToDevice,
            stream.get());
        if (err != cudaSuccess)
            return disableAfterCudaError("input copy", err);

        if (!context->enqueueV3(stream.get()))
        {
            availableFlag = false;
            std::cerr << "[NeuralTracker] TensorRT enqueue failed; CUDA neural tracker disabled." << std::endl;
            return result;
        }

        void* outputHost = outputType == nvinfer1::DataType::kHALF
            ? static_cast<void*>(outputHalfHost.data())
            : static_cast<void*>(outputFloatHost.data());
        err = cudaMemcpyAsync(
            outputHost,
            outputBuffer.get(),
            outputBytes,
            cudaMemcpyDeviceToHost,
            stream.get());
        if (err != cudaSuccess)
            return disableAfterCudaError("output copy", err);

        err = cudaStreamSynchronize(stream.get());
        if (err != cudaSuccess)
            return disableAfterCudaError("stream sync", err);

        const float rawScore = outputType == nvinfer1::DataType::kHALF
            ? halfBitsToFloat(outputHalfHost[0])
            : outputFloatHost[0];

        result.neuralScore = normalizeOutputScore(rawScore);
        result.valid = true;
        return result;
    }

private:
    bool initialize(const std::filesystem::path& modelPath)
    {
        const std::string modelExt = lowerExtension(modelPath);
        const std::filesystem::path enginePath = resolveNeuralEnginePath(modelPath);
        if (enginePath.empty())
        {
            std::cerr << "[NeuralTracker] CUDA runtime expects .onnx or .engine model: "
                      << modelPath.string() << std::endl;
            return false;
        }

        std::error_code ec;
        if (modelExt == ".onnx" && !std::filesystem::exists(enginePath, ec))
        {
            std::cout << "[NeuralTracker] Building TensorRT engine from ONNX model." << std::endl;
            if (!buildAndSaveNeuralEngine(modelPath, enginePath))
                return false;
        }

        if (!stream.create())
            return false;

        runtime.reset(nvinfer1::createInferRuntime(gLogger));
        if (!runtime)
        {
            std::cerr << "[NeuralTracker] Failed to create TensorRT runtime." << std::endl;
            return false;
        }

        engine.reset(loadEngineFromFile(enginePath.string(), runtime.get()));
        if (!engine)
        {
            std::cerr << "[NeuralTracker] Failed to load TensorRT engine: "
                      << enginePath.string() << std::endl;
            return false;
        }

        context.reset(engine->createExecutionContext());
        if (!context)
        {
            std::cerr << "[NeuralTracker] Failed to create TensorRT execution context." << std::endl;
            return false;
        }

        if (!findTensorNames())
            return false;

        nvinfer1::Dims inputDims = context->getTensorShape(inputName.c_str());
        bool dynamicInput = false;
        for (int32_t i = 0; i < inputDims.nbDims; ++i)
            dynamicInput = dynamicInput || inputDims.d[i] <= 0;

        if (dynamicInput)
        {
            nvinfer1::Dims fixedDims{};
            fixedDims.nbDims = 2;
            fixedDims.d[0] = 1;
            fixedDims.d[1] = static_cast<int64_t>(NeuralTrackerFeatureCount);
            if (!context->setInputShape(inputName.c_str(), fixedDims)
                || !context->allInputDimensionsSpecified())
            {
                std::cerr << "[NeuralTracker] Failed to set TensorRT input shape." << std::endl;
                return false;
            }
            inputDims = context->getTensorShape(inputName.c_str());
        }

        const size_t inputCount = trtElementCount(inputDims);
        const nvinfer1::DataType inputType = engine->getTensorDataType(inputName.c_str());
        if (inputType != nvinfer1::DataType::kFLOAT || inputCount != NeuralTrackerFeatureCount)
        {
            std::cerr << "[NeuralTracker] TensorRT neural input must be 1x"
                      << NeuralTrackerFeatureCount << " FP32 features." << std::endl;
            return false;
        }

        nvinfer1::Dims outputDims = context->getTensorShape(outputName.c_str());
        outputElementCount = trtElementCount(outputDims);
        outputType = engine->getTensorDataType(outputName.c_str());
        const size_t outputElementBytes = trtElementSize(outputType);
        if (outputElementCount < 1
            || (outputType != nvinfer1::DataType::kFLOAT && outputType != nvinfer1::DataType::kHALF)
            || outputElementBytes == 0)
        {
            std::cerr << "[NeuralTracker] TensorRT neural output must contain at least one FP32/FP16 score." << std::endl;
            return false;
        }

        inputBytes = NeuralTrackerFeatureCount * sizeof(float);
        outputBytes = outputElementCount * outputElementBytes;
        if (!inputBuffer.allocate(inputBytes) || !outputBuffer.allocate(outputBytes))
            return false;

        if (outputType == nvinfer1::DataType::kHALF)
            outputHalfHost.resize(outputElementCount);
        else
            outputFloatHost.resize(outputElementCount);

        if (!context->setTensorAddress(inputName.c_str(), inputBuffer.get())
            || !context->setTensorAddress(outputName.c_str(), outputBuffer.get()))
        {
            std::cerr << "[NeuralTracker] Failed to bind TensorRT neural tensors." << std::endl;
            return false;
        }

        std::cout << "[NeuralTracker] Loaded CUDA TensorRT association engine: "
                  << enginePath.string() << std::endl;
        return true;
    }

    bool findTensorNames()
    {
        for (int32_t i = 0; i < engine->getNbIOTensors(); ++i)
        {
            const char* name = engine->getIOTensorName(i);
            if (!name)
                continue;

            if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT && inputName.empty())
                inputName = name;
            else if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT && outputName.empty())
                outputName = name;
        }

        if (inputName.empty() || outputName.empty())
        {
            std::cerr << "[NeuralTracker] TensorRT neural engine must expose one input and one output." << std::endl;
            return false;
        }
        return true;
    }

    NeuralTrackerResult disableAfterCudaError(const char* phase, cudaError_t err)
    {
        availableFlag = false;
        std::cerr << "[NeuralTracker] CUDA neural tracker disabled after " << phase
                  << " failed: " << cudaGetErrorString(err) << std::endl;
        return {};
    }

    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    CudaStreamHandle stream;
    CudaDeviceBuffer inputBuffer;
    CudaDeviceBuffer outputBuffer;
    std::vector<float> outputFloatHost;
    std::vector<uint16_t> outputHalfHost;
    std::string inputName;
    std::string outputName;
    nvinfer1::DataType outputType = nvinfer1::DataType::kFLOAT;
    size_t inputBytes = 0;
    size_t outputBytes = 0;
    size_t outputElementCount = 0;
    bool availableFlag = false;
    std::mutex inferenceMutex;
};

std::shared_ptr<INeuralTracker> createCudaNeuralTracker(const std::filesystem::path& resolved)
{
    try
    {
        auto tracker = std::make_shared<TrtNeuralTracker>(resolved);
        if (!tracker->available())
            return nullptr;
        return tracker;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[NeuralTracker] Failed to load CUDA neural tracker: "
                  << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "[NeuralTracker] Failed to load CUDA neural tracker: unknown error." << std::endl;
    }
    return nullptr;
}
#endif

std::mutex g_logMutex;
std::ofstream g_logFile;
std::string g_logFilePath;

void ensureLogOpen(const std::string& logPath)
{
    if (logPath.empty())
        return;

    if (g_logFile.is_open() && g_logFilePath == logPath)
        return;

    if (g_logFile.is_open())
        g_logFile.close();

    std::filesystem::path path(logPath);
    std::error_code ec;
    if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path(), ec);

    const bool writeHeader = !std::filesystem::exists(path, ec) || std::filesystem::file_size(path, ec) == 0;
    g_logFile.open(path, std::ios::out | std::ios::app);
    g_logFilePath = logPath;

    if (g_logFile.is_open() && writeHeader)
    {
        g_logFile
            << "timestamp_ms,distance_norm,iou,size_log_ratio,detection_confidence,track_confidence,"
            << "heading_alignment,track_missed_norm,track_hits_norm,is_locked,class_compatible,dt,"
            << "speed_norm,target_size_norm,pivot_offset_x_norm,pivot_offset_y_norm,relaxed_gate,"
            << "neural_score,classical_score,final_score,accepted,chosen\n";
    }
}
}

std::array<float, NeuralTrackerFeatureCount> neuralTrackerFeatureArray(const NeuralTrackerFeatures& features)
{
    return {
        features.distanceNorm,
        features.iou,
        features.sizeLogRatio,
        features.detectionConfidence,
        features.trackConfidence,
        features.headingAlignment,
        features.trackMissedNorm,
        features.trackHitsNorm,
        features.isLocked,
        features.classCompatible,
        features.dt,
        features.speedNorm,
        features.targetSizeNorm,
        features.pivotOffsetXNorm,
        features.pivotOffsetYNorm,
        features.relaxedGate,
    };
}

std::shared_ptr<INeuralTracker> createCpuNeuralTracker(const std::string& modelPath)
{
    const std::filesystem::path resolved = resolveNeuralModelPath(modelPath);
    std::error_code ec;
    if (!std::filesystem::exists(resolved, ec))
    {
        std::cerr << "[NeuralTracker] ONNX model missing, using classical tracker only: "
                  << resolved.string() << std::endl;
        return nullptr;
    }

    try
    {
        auto tracker = std::make_shared<OnnxNeuralTracker>(resolved);
        if (!tracker->available())
            return nullptr;
        std::cout << "[NeuralTracker] Loaded ONNX association model: " << resolved.string() << std::endl;
        return tracker;
    }
    catch (const Ort::Exception& e)
    {
        std::cerr << "[NeuralTracker] Failed to load ONNX model, using classical tracker only: "
                  << e.what() << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[NeuralTracker] Failed to load ONNX model, using classical tracker only: "
                  << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "[NeuralTracker] Failed to load ONNX model, using classical tracker only: unknown error." << std::endl;
    }

    return nullptr;
}

std::shared_ptr<INeuralTracker> createOnnxNeuralTracker(const std::string& modelPath)
{
    return createCpuNeuralTracker(modelPath);
}

std::shared_ptr<INeuralTracker> createNeuralTracker(const std::string& modelPath, const std::string& runtime)
{
    if (runtime == "CPU")
        return createCpuNeuralTracker(modelPath);

    if (runtime == "CUDA")
    {
#ifdef USE_CUDA
        const std::filesystem::path resolved = resolveNeuralModelPath(modelPath);
        std::error_code ec;
        if (!std::filesystem::exists(resolved, ec))
        {
            std::cerr << "[NeuralTracker] CUDA model missing; neural tracker unavailable: "
                      << resolved.string() << std::endl;
            return nullptr;
        }

        return createCudaNeuralTracker(resolved);
#else
        std::cerr << "[NeuralTracker] CUDA neural tracker requested in a non-CUDA build." << std::endl;
        return nullptr;
#endif
    }

    std::cerr << "[NeuralTracker] Unknown runtime '" << runtime
              << "'; neural tracker unavailable." << std::endl;
    return nullptr;
}

void logNeuralTrackerAssociation(
    const std::string& logPath,
    const NeuralTrackerFeatures& features,
    float neuralScore,
    float classicalScore,
    float finalScore,
    bool accepted,
    bool chosen)
{
    std::lock_guard<std::mutex> lock(g_logMutex);
    ensureLogOpen(logPath);
    if (!g_logFile.is_open())
        return;

    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();

    g_logFile
        << ms << ','
        << features.distanceNorm << ','
        << features.iou << ','
        << features.sizeLogRatio << ','
        << features.detectionConfidence << ','
        << features.trackConfidence << ','
        << features.headingAlignment << ','
        << features.trackMissedNorm << ','
        << features.trackHitsNorm << ','
        << features.isLocked << ','
        << features.classCompatible << ','
        << features.dt << ','
        << features.speedNorm << ','
        << features.targetSizeNorm << ','
        << features.pivotOffsetXNorm << ','
        << features.pivotOffsetYNorm << ','
        << features.relaxedGate << ','
        << neuralScore << ','
        << classicalScore << ','
        << finalScore << ','
        << (accepted ? 1 : 0) << ','
        << (chosen ? 1 : 0) << '\n';
}
}
