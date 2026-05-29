#include "TemporalPredictor.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace aim::neural
{
namespace
{
std::filesystem::path resolveModelPath(const std::string& modelPath)
{
    if (modelPath.empty())
        return std::filesystem::path("models") / "temporal_predictor.onnx";

    std::filesystem::path path(modelPath);
    if (path.is_absolute())
        return path;
    return std::filesystem::absolute(path);
}
}

class TemporalPredictor::Impl
{
public:
    Impl()
        : env(ORT_LOGGING_LEVEL_WARNING, "TemporalPredictor"),
          memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    {
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
        sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetInterOpNumThreads(1);
    }

    bool loadModel(const std::string& onnx_path)
    {
        availableFlag = false;
        disabledAfterError = false;
        inputNameStorage.clear();
        outputNameStorage.clear();
        session = Ort::Session(nullptr);

        const std::filesystem::path resolved = resolveModelPath(onnx_path);
        std::error_code ec;
        if (!std::filesystem::is_regular_file(resolved, ec))
        {
            std::cerr << "[TemporalPredictor] ONNX model missing; learned prediction disabled: "
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
            inputShapeTemplate = inputInfo.GetShape();
            if (!inputShapeTemplate.empty())
            {
                const int64_t lastDim = inputShapeTemplate.back();
                if (lastDim > 0 && lastDim != TemporalPredictorFeatureCount)
                {
                    std::cerr << "[TemporalPredictor] ONNX input feature count mismatch. Model expects "
                              << lastDim << ", runtime provides " << TemporalPredictorFeatureCount
                              << "." << std::endl;
                    return false;
                }
            }

            availableFlag = !inputNameStorage.empty() && !outputNameStorage.empty();
        }
        catch (const Ort::Exception& e)
        {
            std::cerr << "[TemporalPredictor] Failed to load ONNX model: " << e.what() << std::endl;
            availableFlag = false;
        }
        catch (const std::exception& e)
        {
            std::cerr << "[TemporalPredictor] Failed to load ONNX model: " << e.what() << std::endl;
            availableFlag = false;
        }
        catch (...)
        {
            std::cerr << "[TemporalPredictor] Failed to load ONNX model: unknown error." << std::endl;
            availableFlag = false;
        }

        if (availableFlag)
            std::cout << "[TemporalPredictor] Loaded ONNX temporal predictor: "
                      << resolved.string() << std::endl;

        return availableFlag;
    }

    TemporalPredictor::Output predict(const TemporalPredictor::Input& input)
    {
        TemporalPredictor::Output result;
        result.prediction_horizon = predictionHorizon;

        const int historyLength = std::clamp(input.history_length, 2, 64);
        const size_t required = static_cast<size_t>(historyLength) * TemporalPredictorFeatureCount;
        if (!availableFlag || disabledAfterError || input.history.size() < required)
        {
            result.valid = false;
            return result;
        }

        try
        {
            std::vector<float> history(required);
            std::copy(input.history.end() - static_cast<std::ptrdiff_t>(required), input.history.end(), history.begin());

            std::vector<int64_t> inputShape;
            if (inputShapeTemplate.size() >= 3)
            {
                inputShape = { 1, historyLength, TemporalPredictorFeatureCount };
            }
            else
            {
                inputShape = { 1, static_cast<int64_t>(required) };
            }

            Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                memoryInfo,
                history.data(),
                history.size(),
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
            result.inference_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            if (outputs.empty() || !outputs[0].IsTensor())
                return result;

            auto info = outputs[0].GetTensorTypeAndShapeInfo();
            const size_t outputCount = info.GetElementCount();
            if (outputCount < 2)
                return result;

            const float* values = outputs[0].GetTensorData<float>();
            const size_t pointCount = std::min(
                static_cast<size_t>(predictionHorizon),
                outputCount / 2);

            result.future_x.reserve(pointCount);
            result.future_y.reserve(pointCount);
            for (size_t i = 0; i < pointCount; ++i)
            {
                const float x = values[i * 2];
                const float y = values[i * 2 + 1];
                if (!std::isfinite(x) || !std::isfinite(y))
                    continue;
                result.future_x.push_back(x);
                result.future_y.push_back(y);
            }

            if (outputCount >= pointCount * 2 + 1)
                result.confidence = std::clamp(values[pointCount * 2], 0.0f, 1.0f);
            else
                result.confidence = 1.0f;

            result.valid = !result.future_x.empty() && result.future_x.size() == result.future_y.size();
        }
        catch (const Ort::Exception& e)
        {
            disabledAfterError = true;
            result.valid = false;
            std::cerr << "[TemporalPredictor] ONNX inference disabled after error: "
                      << e.what() << std::endl;
        }
        catch (...)
        {
            disabledAfterError = true;
            result.valid = false;
            std::cerr << "[TemporalPredictor] ONNX inference disabled after unknown error." << std::endl;
        }

        return result;
    }

    void setHistoryLength(int len)
    {
        historyLength = std::clamp(len, 2, 64);
    }

    void setPredictionHorizon(int horizon)
    {
        predictionHorizon = std::clamp(horizon, 1, 64);
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
    std::vector<int64_t> inputShapeTemplate;
    bool availableFlag = false;
    bool disabledAfterError = false;
    int historyLength = 12;
    int predictionHorizon = 16;
};

TemporalPredictor::TemporalPredictor()
    : impl_(std::make_unique<Impl>())
{
}

TemporalPredictor::~TemporalPredictor() = default;

bool TemporalPredictor::loadModel(const std::string& onnx_path)
{
    return impl_->loadModel(onnx_path);
}

TemporalPredictor::Output TemporalPredictor::predict(const Input& input)
{
    return impl_->predict(input);
}

void TemporalPredictor::setHistoryLength(int len)
{
    impl_->setHistoryLength(len);
}

void TemporalPredictor::setPredictionHorizon(int horizon)
{
    impl_->setPredictionHorizon(horizon);
}

bool TemporalPredictor::available() const
{
    return impl_->available();
}

std::shared_ptr<TemporalPredictor> createTemporalPredictor(const std::string& modelPath)
{
    auto predictor = std::make_shared<TemporalPredictor>();
    predictor->loadModel(modelPath.empty() ? "models/temporal_predictor.onnx" : modelPath);
    return predictor;
}

class TemporalPredictionWorker::Impl
{
public:
    Impl() = default;

    ~Impl()
    {
        stop();
    }

    void submit(const TemporalPredictionWorker::Request& request)
    {
        if (request.track_id < 0 || request.input.history.empty())
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
                    [&](const TemporalPredictionWorker::Request& pending) {
                        return pending.track_id == request.track_id;
                    }),
                pendingRequests.end());

            pendingRequests.push_back(request);
            while (pendingRequests.size() > MaxPendingRequests)
                pendingRequests.pop_front();
        }
        cv.notify_one();
    }

    bool tryGet(int trackId, TemporalPredictionWorker::Result& result)
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
            worker = std::thread(&TemporalPredictionWorker::Impl::workerLoop, this);
    }

    void workerLoop()
    {
        while (true)
        {
            TemporalPredictionWorker::Request request;
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

    TemporalPredictionWorker::Result runRequest(const TemporalPredictionWorker::Request& request)
    {
        TemporalPredictionWorker::Result result;
        result.track_id = request.track_id;
        result.frame_id = request.frame_id;

        try
        {
            const std::string wantedPath = request.model_path.empty()
                ? std::string("models/temporal_predictor.onnx")
                : request.model_path;
            const bool pathChanged = wantedPath != loadedPath;
            const auto now = std::chrono::steady_clock::now();

            if (pathChanged || !predictor || !predictor->available())
            {
                if (pathChanged || now >= nextLoadAttempt)
                {
                    predictor = createTemporalPredictor(wantedPath);
                    loadedPath = wantedPath;
                    if (!predictor || !predictor->available())
                        nextLoadAttempt = now + std::chrono::seconds(2);
                    else
                        nextLoadAttempt = {};
                }
            }

            if (!predictor || !predictor->available())
                return result;

            predictor->setHistoryLength(request.history_length);
            predictor->setPredictionHorizon(request.prediction_horizon);
            result.output = predictor->predict(request.input);
            result.valid = result.output.valid;
        }
        catch (const std::exception& e)
        {
            std::cerr << "[TemporalPredictor] Async prediction failed: " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "[TemporalPredictor] Async prediction failed: unknown error." << std::endl;
        }

        return result;
    }

    std::mutex mtx;
    std::condition_variable cv;
    std::thread worker;
    bool stopping = false;
    std::deque<TemporalPredictionWorker::Request> pendingRequests;
    std::unordered_map<int, TemporalPredictionWorker::Result> latestResults;
    std::shared_ptr<TemporalPredictor> predictor;
    std::string loadedPath;
    std::chrono::steady_clock::time_point nextLoadAttempt{};
};

TemporalPredictionWorker& TemporalPredictionWorker::instance()
{
    static TemporalPredictionWorker worker;
    return worker;
}

TemporalPredictionWorker::TemporalPredictionWorker()
    : impl_(std::make_unique<Impl>())
{
}

TemporalPredictionWorker::~TemporalPredictionWorker() = default;

void TemporalPredictionWorker::submit(const Request& request)
{
    impl_->submit(request);
}

bool TemporalPredictionWorker::tryGet(int trackId, Result& result)
{
    return impl_->tryGet(trackId, result);
}

void TemporalPredictionWorker::clear()
{
    impl_->clear();
}

void TemporalPredictionWorker::stop()
{
    impl_->stop();
}
}
