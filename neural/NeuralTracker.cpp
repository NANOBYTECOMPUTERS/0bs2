#include "NeuralTracker.h"

#include <onnxruntime_cxx_api.h>
#include <Windows.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <vector>

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

std::shared_ptr<INeuralTracker> createOnnxNeuralTracker(const std::string& modelPath)
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
