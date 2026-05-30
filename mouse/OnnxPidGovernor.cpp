#include "OnnxPidGovernor.h"

#include <onnxruntime_cxx_api.h>
#include <Windows.h>

#include <array>
#include <filesystem>
#include <iostream>
#include <vector>

namespace aim
{
namespace
{
std::filesystem::path resolveModelPath(const std::string& modelPath)
{
    std::filesystem::path configured(modelPath);
    std::error_code ec;

    if (configured.is_absolute() && std::filesystem::exists(configured, ec))
        return configured;

    // Build search bases (exe dir, parents, current)
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

    // Preferred subfolders for neural models
    std::vector<std::filesystem::path> subdirs = {
        "neural_models",
        "models",
        "training/models"
    };

    for (const auto& base : bases)
    {
        if (base.empty()) continue;
        for (const auto& sub : subdirs)
        {
            std::filesystem::path candidate = base / sub / configured.filename();
            if (std::filesystem::exists(candidate, ec))
                return candidate;
        }
        // Also try direct relative to base
        std::filesystem::path direct = base / configured;
        if (std::filesystem::exists(direct, ec))
            return direct;
    }

    // Final fallback: neural_models
    return std::filesystem::absolute(std::filesystem::path("neural_models") / configured);
}

class OnnxPidGovernor final : public IPidGovernor
{
public:
    explicit OnnxPidGovernor(const std::filesystem::path& modelPath)
        : env(ORT_LOGGING_LEVEL_WARNING, "PidGovernor"),
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
            if (featureDim > 0 && featureDim != PidGovernorFeatureCount)
            {
                std::cerr << "[Mouse] PID governor ONNX input feature count mismatch. Model expects "
                          << featureDim << ", runtime provides " << PidGovernorFeatureCount
                          << ". Retrain/export the 8-axis governor model." << std::endl;
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

    PidGovernorScales evaluate(const PidGovernorInput& input) override
    {
        PidGovernorScales scales;
        if (!availableFlag)
            return scales;

        try
        {
            std::array<float, PidGovernorFeatureCount> features = pidGovernorFeatures(input);
            std::array<int64_t, 2> inputShape{ 1, PidGovernorFeatureCount };
            Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                memoryInfo,
                features.data(),
                features.size(),
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
                return scales;

            auto info = outputs[0].GetTensorTypeAndShapeInfo();
            const size_t outputCount = info.GetElementCount();
            if (outputCount < PidGovernorOutputCount)
                return scales;

            const float* values = outputs[0].GetTensorData<float>();
            scales.kp = values[0];
            scales.ki = values[1];
            scales.kd = values[2];
            scales.speed = values[3];
            scales.valid = true;
        }
        catch (const Ort::Exception& e)
        {
            availableFlag = false;
            std::cerr << "[Mouse] PID governor ONNX inference disabled after error: " << e.what() << std::endl;
        }
        catch (...)
        {
            availableFlag = false;
            std::cerr << "[Mouse] PID governor ONNX inference disabled after unknown error." << std::endl;
        }

        return scales;
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
}

std::shared_ptr<IPidGovernor> createOnnxPidGovernor(const std::string& modelPath)
{
    const std::filesystem::path resolved = resolveModelPath(modelPath);
    std::error_code ec;
    if (!std::filesystem::exists(resolved, ec))
    {
        std::cerr << "[Mouse] PID governor ONNX model missing, using pure PID: " << resolved.string() << std::endl;
        return nullptr;
    }

    try
    {
        auto governor = std::make_shared<OnnxPidGovernor>(resolved);
        if (!governor->available())
            return nullptr;
        std::cout << "[Mouse] Loaded PID governor ONNX: " << resolved.string() << std::endl;
        return governor;
    }
    catch (const Ort::Exception& e)
    {
        std::cerr << "[Mouse] Failed to load PID governor ONNX, using pure PID: " << e.what() << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Mouse] Failed to load PID governor ONNX, using pure PID: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "[Mouse] Failed to load PID governor ONNX, using pure PID: unknown error." << std::endl;
    }

    return nullptr;
}
}
