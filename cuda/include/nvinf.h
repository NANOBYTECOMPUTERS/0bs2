#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "trt_monitor.h"

class TrtLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, char const* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            std::cerr << "[TensorRT] " << (msg ? msg : "") << std::endl;
        }
    }
};

inline TrtLogger gLogger;

inline nvinfer1::ICudaEngine* loadEngineFromFile(const std::string& engineFile, nvinfer1::IRuntime* runtime)
{
    if (!runtime)
    {
        std::cerr << "[TensorRT] Cannot load engine without a runtime." << std::endl;
        return nullptr;
    }

    std::ifstream file(engineFile, std::ios::binary | std::ios::ate);
    if (!file)
    {
        std::cerr << "[TensorRT] Failed to open engine: " << engineFile << std::endl;
        return nullptr;
    }

    const std::streamsize size = file.tellg();
    if (size <= 0)
    {
        std::cerr << "[TensorRT] Engine file is empty: " << engineFile << std::endl;
        return nullptr;
    }

    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(static_cast<size_t>(size));
    if (!file.read(buffer.data(), size))
    {
        std::cerr << "[TensorRT] Failed to read engine: " << engineFile << std::endl;
        return nullptr;
    }

    return runtime->deserializeCudaEngine(buffer.data(), buffer.size());
}

inline nvinfer1::ICudaEngine* buildEngineFromOnnx(const std::string& onnxFile, nvinfer1::ILogger& logger)
{
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder)
    {
        std::cerr << "[TensorRT] Failed to create builder." << std::endl;
        return nullptr;
    }

    const auto explicitBatch =
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(explicitBatch));
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!network || !config)
    {
        std::cerr << "[TensorRT] Failed to create network/config." << std::endl;
        return nullptr;
    }

    ImGuiProgressMonitor progressMonitor;
    config->setProgressMonitor(&progressMonitor);
    TrtExportResetState();
    gIsTrtExporting = true;

    struct ScopedProgressState
    {
        ~ScopedProgressState()
        {
            std::lock_guard<std::mutex> lock(gProgressMutex);
            gProgressPhases.clear();
            gIsTrtExporting = false;
            gTrtExportCancelRequested = false;
            gTrtExportLastUpdateMs = TrtNowMs();
        }
    } progressState;

    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    if (!parser)
    {
        std::cerr << "[TensorRT] Failed to create ONNX parser." << std::endl;
        return nullptr;
    }

    if (!parser->parseFromFile(onnxFile.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
    {
        std::cerr << "[TensorRT] Failed to parse ONNX model: " << onnxFile << std::endl;
        return nullptr;
    }

    if (builder->platformHasFastFp16())
        config->setFlag(nvinfer1::BuilderFlag::kFP16);

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, size_t{ 1 } << 30);

    for (int32_t i = 0; i < network->getNbInputs(); ++i)
    {
        nvinfer1::ITensor* input = network->getInput(i);
        if (!input)
            continue;

        const nvinfer1::Dims dims = input->getDimensions();
        bool dynamic = false;
        for (int32_t d = 0; d < dims.nbDims; ++d)
            dynamic = dynamic || dims.d[d] < 0;

        if (!dynamic)
            continue;

        nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
        if (!profile)
        {
            std::cerr << "[TensorRT] Failed to create optimization profile." << std::endl;
            return nullptr;
        }

        auto makeDims = [&](int spatial) {
            nvinfer1::Dims profileDims = dims;
            for (int32_t d = 0; d < profileDims.nbDims; ++d)
            {
                if (profileDims.d[d] < 0)
                {
                    profileDims.d[d] = (d == 0) ? 1 : spatial;
                }
            }
            return profileDims;
        };

        const char* inputName = input->getName();
        bool ok = profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, makeDims(160));
        ok = ok && profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, makeDims(320));
        ok = ok && profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, makeDims(640));
        if (!ok || !profile->isValid())
        {
            std::cerr << "[TensorRT] Invalid optimization profile for input: " << inputName << std::endl;
            return nullptr;
        }

        config->addOptimizationProfile(profile);
    }

    std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
    if (!plan)
    {
        std::cerr << "[TensorRT] Failed to build serialized engine." << std::endl;
        return nullptr;
    }

    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    if (!runtime)
    {
        std::cerr << "[TensorRT] Failed to create runtime for built engine." << std::endl;
        return nullptr;
    }

    return runtime->deserializeCudaEngine(plan->data(), plan->size());
}
