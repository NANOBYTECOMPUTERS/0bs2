#include "NeuralTargetingHead.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>

#include <onnxruntime_cxx_api.h>

namespace aim::neural
{
namespace
{
constexpr size_t NeuralTargetingBaseFeatureCount = 10;

std::filesystem::path resolveTargetingModelPath(const std::string& modelPath)
{
    if (modelPath.empty())
        return std::filesystem::path("models") / "neural_targeting_head.onnx";

    std::filesystem::path path(modelPath);
    if (path.is_absolute())
        return path;
    return std::filesystem::absolute(path);
}

float sanitizeFeature(float value, float scale)
{
    if (!std::isfinite(value))
        return 0.0f;
    return std::clamp(value / std::max(1.0f, scale), -4.0f, 4.0f);
}

std::vector<float> buildFeatureVector(
    const NeuralTargetingHead::Input& input,
    size_t expectedFeatureCount)
{
    std::vector<float> features;
    const size_t predictedCount = std::min(input.predicted_x.size(), input.predicted_y.size());
    features.reserve(NeuralTargetingBaseFeatureCount + predictedCount * 2);

    features.push_back(sanitizeFeature(input.center_x, 640.0f));
    features.push_back(sanitizeFeature(input.center_y, 640.0f));
    features.push_back(sanitizeFeature(input.width, 320.0f));
    features.push_back(sanitizeFeature(input.height, 320.0f));
    features.push_back(sanitizeFeature(input.vx, 1500.0f));
    features.push_back(sanitizeFeature(input.vy, 1500.0f));
    features.push_back(sanitizeFeature(input.box_scale_vel, 800.0f));
    features.push_back(std::clamp(input.confidence, 0.0f, 1.0f));
    features.push_back(sanitizeFeature(input.refinement_x, 80.0f));
    features.push_back(sanitizeFeature(input.refinement_y, 80.0f));

    for (size_t i = 0; i < predictedCount; ++i)
    {
        features.push_back(sanitizeFeature(input.predicted_x[i] - input.center_x, 640.0f));
        features.push_back(sanitizeFeature(input.predicted_y[i] - input.center_y, 640.0f));
    }

    if (expectedFeatureCount > 0)
    {
        if (features.size() < expectedFeatureCount)
            features.resize(expectedFeatureCount, 0.0f);
        else if (features.size() > expectedFeatureCount)
            features.resize(expectedFeatureCount);
    }

    return features;
}

std::pair<float, float> clampOffset(float x, float y, float maxPx)
{
    const float mag = std::hypot(x, y);
    if (!std::isfinite(mag) || mag <= maxPx || mag <= 1e-6f)
        return { std::isfinite(x) ? x : 0.0f, std::isfinite(y) ? y : 0.0f };

    const float scale = maxPx / mag;
    return { x * scale, y * scale };
}
}

class NeuralTargetingHead::Impl
{
public:
    Impl()
        : env(ORT_LOGGING_LEVEL_WARNING, "NeuralTargetingHead"),
          memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    {
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
        sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetInterOpNumThreads(1);
    }

    bool load(const std::string& onnx_path)
    {
        availableFlag = false;
        disabledAfterError = false;
        expectedFeatureCount = 0;
        inputNameStorage.clear();
        outputNameStorage.clear();
        session = Ort::Session(nullptr);

        const std::filesystem::path resolved = resolveTargetingModelPath(onnx_path);
        std::error_code ec;
        if (!std::filesystem::is_regular_file(resolved, ec))
        {
            std::cerr << "[NeuralTargetingHead] ONNX model missing; targeting head disabled: "
                      << resolved.string() << std::endl;
            return false;
        }

        try
        {
            const std::wstring widePath = resolved.wstring();
            session = Ort::Session(env, widePath.c_str(), sessionOptions);

            auto inputName = session.GetInputNameAllocated(0, allocator);
            auto outputName = session.GetOutputNameAllocated(0, allocator);
            inputNameStorage = inputName.get();
            outputNameStorage = outputName.get();

            auto inputInfo = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
            const auto inputShape = inputInfo.GetShape();
            if (!inputShape.empty() && inputShape.back() > 0)
                expectedFeatureCount = static_cast<size_t>(inputShape.back());

            availableFlag = !inputNameStorage.empty() && !outputNameStorage.empty();
        }
        catch (const Ort::Exception& e)
        {
            std::cerr << "[NeuralTargetingHead] Failed to load ONNX model: " << e.what() << std::endl;
            availableFlag = false;
        }
        catch (const std::exception& e)
        {
            std::cerr << "[NeuralTargetingHead] Failed to load ONNX model: " << e.what() << std::endl;
            availableFlag = false;
        }
        catch (...)
        {
            std::cerr << "[NeuralTargetingHead] Failed to load ONNX model: unknown error." << std::endl;
            availableFlag = false;
        }

        if (availableFlag)
            std::cout << "[NeuralTargetingHead] Loaded ONNX targeting head: "
                      << resolved.string() << std::endl;

        return availableFlag;
    }

    NeuralTargetingHead::Output compute(const NeuralTargetingHead::Input& input)
    {
        NeuralTargetingHead::Output result;
        if (!enabled || !available() || input.predicted_x.empty() || input.predicted_y.empty())
            return result;

        NeuralTargetingHead::Input iterInput = input;
        std::array<float, 3> rawOutput{ 0.0f, 0.0f, 0.0f };
        double inferenceMs = 0.0;

        try
        {
            for (int iter = 0; iter < max_iterations; ++iter)
            {
                std::vector<float> features = buildFeatureVector(iterInput, expectedFeatureCount);
                if (features.size() < NeuralTargetingBaseFeatureCount)
                    return {};

                std::array<int64_t, 2> inputShape{ 1, static_cast<int64_t>(features.size()) };
                Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                    memoryInfo,
                    features.data(),
                    features.size(),
                    inputShape.data(),
                    inputShape.size());

                const char* inputNames[] = { inputNameStorage.c_str() };
                const char* outputNames[] = { outputNameStorage.c_str() };

                const auto t0 = std::chrono::steady_clock::now();
                auto outputs = session.Run(
                    Ort::RunOptions{ nullptr },
                    inputNames,
                    &inputTensor,
                    1,
                    outputNames,
                    1);
                const auto t1 = std::chrono::steady_clock::now();
                inferenceMs += std::chrono::duration<double, std::milli>(t1 - t0).count();

                if (outputs.empty() || !outputs[0].IsTensor())
                    return {};

                auto info = outputs[0].GetTensorTypeAndShapeInfo();
                const size_t outputCount = info.GetElementCount();
                if (outputCount < 3)
                    return {};

                const float* values = outputs[0].GetTensorData<float>();
                rawOutput = { values[0], values[1], values[2] };
                auto clamped = clampOffset(rawOutput[0], rawOutput[1], maxRefinementPx);
                iterInput.refinement_x = clamped.first;
                iterInput.refinement_y = clamped.second;
            }

            result.refinement_offset_x = iterInput.refinement_x;
            result.refinement_offset_y = iterInput.refinement_y;
            result.confidence = std::clamp(rawOutput[2], 0.0f, 1.0f);
            result.inference_ms = inferenceMs;
            result.valid =
                std::isfinite(result.refinement_offset_x) &&
                std::isfinite(result.refinement_offset_y) &&
                result.confidence > 0.0f;
        }
        catch (const Ort::Exception& e)
        {
            disabledAfterError = true;
            std::cerr << "[NeuralTargetingHead] ONNX inference disabled after error: " << e.what() << std::endl;
        }
        catch (...)
        {
            disabledAfterError = true;
            std::cerr << "[NeuralTargetingHead] ONNX inference disabled after unknown error." << std::endl;
        }

        return result;
    }

    void setEnabled(bool value)
    {
        enabled = value;
    }

    void setMaxRefinementPx(float maxPx)
    {
        maxRefinementPx = std::clamp(maxPx, 1.0f, 160.0f);
    }

    void setMaxIterations(int iterations)
    {
        max_iterations = std::clamp(iterations, 1, 2);
    }

    bool available() const
    {
        return availableFlag && !disabledAfterError;
    }

private:
    Ort::Env env;
    Ort::SessionOptions sessionOptions;
    Ort::Session session{ nullptr };
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memoryInfo;
    std::string inputNameStorage;
    std::string outputNameStorage;
    size_t expectedFeatureCount = 0;
    bool enabled = true;
    bool availableFlag = false;
    bool disabledAfterError = false;
    float maxRefinementPx = 35.0f;
    int max_iterations = 2;
};

NeuralTargetingHead::NeuralTargetingHead()
    : impl_(std::make_unique<Impl>())
{
}

NeuralTargetingHead::~NeuralTargetingHead() = default;

bool NeuralTargetingHead::load(const std::string& onnx_path)
{
    return impl_->load(onnx_path);
}

NeuralTargetingHead::Output NeuralTargetingHead::compute(const Input& input)
{
    return impl_->compute(input);
}

void NeuralTargetingHead::setEnabled(bool enabled)
{
    impl_->setEnabled(enabled);
}

void NeuralTargetingHead::setMaxRefinementPx(float max_px)
{
    impl_->setMaxRefinementPx(max_px);
}

void NeuralTargetingHead::setMaxIterations(int iterations)
{
    impl_->setMaxIterations(iterations);
}

bool NeuralTargetingHead::available() const
{
    return impl_->available();
}

std::shared_ptr<NeuralTargetingHead> createNeuralTargetingHead(const std::string& modelPath)
{
    auto head = std::make_shared<NeuralTargetingHead>();
    head->load(modelPath.empty() ? "models/neural_targeting_head.onnx" : modelPath);
    return head;
}

class NeuralTargetingWorker::Impl
{
public:
    Impl() = default;

    ~Impl()
    {
        stop();
    }

    void submit(const NeuralTargetingWorker::Request& request)
    {
        if (request.track_id < 0 || request.input.predicted_x.empty() || request.input.predicted_y.empty())
            return;

        {
            std::lock_guard<std::mutex> lock(mtx);
            if (stopping)
                return;

            ensureThreadLocked();
            pendingRequests.erase(
                std::remove_if(
                    pendingRequests.begin(),
                    pendingRequests.end(),
                    [&](const NeuralTargetingWorker::Request& pending) {
                        return pending.track_id == request.track_id;
                    }),
                pendingRequests.end());

            pendingRequests.push_back(request);
            while (pendingRequests.size() > MaxPendingRequests)
                pendingRequests.pop_front();
        }
        cv.notify_one();
    }

    bool tryGet(int trackId, NeuralTargetingWorker::Result& result)
    {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = latestResults.find(trackId);
        if (it == latestResults.end())
            return false;

        result = it->second;
        latestResults.erase(it);
        return true;
    }

    void clear()
    {
        std::lock_guard<std::mutex> lock(mtx);
        pendingRequests.clear();
        latestResults.clear();
    }

    void stop()
    {
        {
            std::lock_guard<std::mutex> lock(mtx);
            stopping = true;
            pendingRequests.clear();
        }
        cv.notify_all();

        if (worker.joinable())
            worker.join();
    }

private:
    static constexpr size_t MaxPendingRequests = 32;

    void ensureThreadLocked()
    {
        if (!worker.joinable())
            worker = std::thread(&NeuralTargetingWorker::Impl::workerLoop, this);
    }

    void workerLoop()
    {
        while (true)
        {
            NeuralTargetingWorker::Request request;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&] { return stopping || !pendingRequests.empty(); });
                if (stopping && pendingRequests.empty())
                    break;

                request = pendingRequests.front();
                pendingRequests.pop_front();
            }

            auto result = runRequest(request);
            {
                std::lock_guard<std::mutex> lock(mtx);
                if (!stopping)
                    latestResults[result.track_id] = result;
            }
        }
    }

    NeuralTargetingWorker::Result runRequest(const NeuralTargetingWorker::Request& request)
    {
        NeuralTargetingWorker::Result result;
        result.track_id = request.track_id;
        result.frame_id = request.frame_id;

        try
        {
            const std::string wantedPath = request.model_path.empty()
                ? std::string("models/neural_targeting_head.onnx")
                : request.model_path;
            const bool pathChanged = wantedPath != loadedPath;
            const auto now = std::chrono::steady_clock::now();

            if (pathChanged || !head || !head->available())
            {
                if (pathChanged || now >= nextLoadAttempt)
                {
                    head = createNeuralTargetingHead(wantedPath);
                    loadedPath = wantedPath;
                    if (!head || !head->available())
                        nextLoadAttempt = now + std::chrono::seconds(2);
                    else
                        nextLoadAttempt = {};
                }
            }

            if (!head || !head->available())
                return result;

            head->setEnabled(true);
            head->setMaxRefinementPx(request.max_refinement_px);
            head->setMaxIterations(request.max_iterations);
            result.output = head->compute(request.input);
            result.valid = result.output.valid;
        }
        catch (const std::exception& e)
        {
            std::cerr << "[NeuralTargetingHead] Async targeting failed: " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "[NeuralTargetingHead] Async targeting failed: unknown error." << std::endl;
        }

        return result;
    }

    std::mutex mtx;
    std::condition_variable cv;
    std::thread worker;
    bool stopping = false;
    std::deque<NeuralTargetingWorker::Request> pendingRequests;
    std::unordered_map<int, NeuralTargetingWorker::Result> latestResults;
    std::shared_ptr<NeuralTargetingHead> head;
    std::string loadedPath;
    std::chrono::steady_clock::time_point nextLoadAttempt{};
};

NeuralTargetingWorker& NeuralTargetingWorker::instance()
{
    static NeuralTargetingWorker worker;
    return worker;
}

NeuralTargetingWorker::NeuralTargetingWorker()
    : impl_(std::make_unique<Impl>())
{
}

NeuralTargetingWorker::~NeuralTargetingWorker() = default;

void NeuralTargetingWorker::submit(const Request& request)
{
    impl_->submit(request);
}

bool NeuralTargetingWorker::tryGet(int trackId, Result& result)
{
    return impl_->tryGet(trackId, result);
}

void NeuralTargetingWorker::clear()
{
    impl_->clear();
}

void NeuralTargetingWorker::stop()
{
    impl_->stop();
}
}
