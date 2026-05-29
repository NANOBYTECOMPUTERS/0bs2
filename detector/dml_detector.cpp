#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>
#include <dml_provider_factory.h>
#include <wrl/client.h>
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <limits>
#include <algorithm>
#include <dxgi.h>
#include <opencv2/dnn.hpp>

#include "dml_detector.h"
#include "0BS_box_2.h"
#include "postProcess.h"
#include "capture.h"
#include "other_tools.h"
#ifdef USE_CUDA
#include "depth/depth_mask.h"
#endif

extern std::atomic<bool> detector_model_changed;
extern std::atomic<bool> detection_resolution_changed;
extern std::atomic<unsigned long long> detection_resolution_generation;
extern std::atomic<bool> detectionPaused;

namespace
{
bool tryInt64ToInt(int64_t value, int* out)
{
    if (!out)
    {
        return false;
    }

    if (value < static_cast<int64_t>(std::numeric_limits<int>::min()) ||
        value > static_cast<int64_t>(std::numeric_limits<int>::max()))
    {
        return false;
    }

    *out = static_cast<int>(value);
    return true;
}

#ifdef USE_CUDA
bool intersectsDepthMask(const cv::Rect& box, const cv::Mat& mask)
{
    if (box.width <= 0 || box.height <= 0 || mask.empty() || mask.type() != CV_8UC1)
        return false;

    const cv::Rect imageBounds(0, 0, mask.cols, mask.rows);
    const cv::Rect clipped = box & imageBounds;
    if (clipped.width <= 0 || clipped.height <= 0)
        return false;

    const int cx = clipped.x + clipped.width / 2;
    const int cy = clipped.y + clipped.height / 2;
    if (mask.at<uint8_t>(cy, cx) != 0)
        return true;

    const cv::Mat roi = mask(clipped);
    return cv::countNonZero(roi) > 0;
}

void filterDetectionsByDepthMask(std::vector<Detection>& detections)
{
    static cv::Mat holdTtl;

    if (detections.empty())
        return;

    if (!config.depth_inference_enabled || !config.depth_mask_enabled)
    {
        holdTtl.release();
        return;
    }

    const int holdFrames = std::clamp(config.depth_mask_hold_frames, 0, 120);
    cv::Mat currentMask = getCurrentDetectionSuppressionMask();
    cv::Mat suppressionMask;

    if (holdFrames <= 0)
    {
        holdTtl.release();
        suppressionMask = currentMask;
    }
    else
    {
        if (!currentMask.empty() && currentMask.type() == CV_8UC1)
        {
            if (holdTtl.empty() || holdTtl.size() != currentMask.size())
                holdTtl = cv::Mat::zeros(currentMask.size(), CV_16UC1);
            cv::subtract(holdTtl, cv::Scalar(1), holdTtl);
            holdTtl.setTo(cv::Scalar(static_cast<uint16_t>(holdFrames)), currentMask);
        }
        else if (!holdTtl.empty())
        {
            cv::subtract(holdTtl, cv::Scalar(1), holdTtl);
        }

        if (!holdTtl.empty() && cv::countNonZero(holdTtl) > 0)
        {
            cv::compare(holdTtl, cv::Scalar(0), suppressionMask, cv::CMP_GT);
        }
        else
        {
            suppressionMask.release();
        }
    }

    if (suppressionMask.empty() || suppressionMask.type() != CV_8UC1)
        return;

    detections.erase(
        std::remove_if(detections.begin(), detections.end(),
            [&suppressionMask](const Detection& det) { return intersectsDepthMask(det.box, suppressionMask); }),
        detections.end());
}
#else
void filterDetectionsByDepthMask(std::vector<Detection>&)
{
}
#endif
}

std::string GetDMLDeviceName(int deviceId)
{
    Microsoft::WRL::ComPtr<IDXGIFactory1> dxgiFactory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory))))
        return "Unknown";

    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
    if (FAILED(dxgiFactory->EnumAdapters1(deviceId, &adapter)))
        return "Invalid device ID";

    DXGI_ADAPTER_DESC1 desc;
    if (FAILED(adapter->GetDesc1(&desc)))
        return "Failed to get description";

    std::wstring wname(desc.Description);
    return WideToUtf8(wname);
}

DirectMLDetector::DirectMLDetector(const std::string& model_path)
    :
    env(ORT_LOGGING_LEVEL_WARNING, "DML_Detector"),
    memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);
    session_options.DisableMemPattern();

    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(session_options, config.dml_device_id));

    if (config.verbose)
        std::cout << "[DirectML] Using adapter: " << GetDMLDeviceName(config.dml_device_id) << std::endl;

    initializeModel(model_path);
}

DirectMLDetector::~DirectMLDetector()
{
    shouldExit = true;
    inferenceCV.notify_all();
}

void DirectMLDetector::initializeModel(const std::string& model_path)
{
    std::wstring model_path_wide(model_path.begin(), model_path.end());
    session = Ort::Session(env, model_path_wide.c_str(), session_options);

    input_name = session.GetInputNameAllocated(0, allocator).get();
    output_name = session.GetOutputNameAllocated(0, allocator).get();

    Ort::TypeInfo input_type_info = session.GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    input_shape = input_tensor_info.GetShape();

    bool isStatic = true;
    for (auto d : input_shape) if (d <= 0) isStatic = false;
    cachedModelHeight = -1;
    cachedModelWidth = -1;
    if (input_shape.size() > 2)
    {
        int converted = 0;
        if (tryInt64ToInt(input_shape[2], &converted))
            cachedModelHeight = converted;
    }
    if (input_shape.size() > 3)
    {
        int converted = 0;
        if (tryInt64ToInt(input_shape[3], &converted))
            cachedModelWidth = converted;
    }
    cachedFixedInput = isStatic && cachedModelHeight > 0 && cachedModelWidth > 0;

    if (isStatic != config.fixed_input_size)
    {
        config.fixed_input_size = isStatic;
        detector_model_changed.store(true);
        std::cout << "[DML] Automatically set fixed_input_size = " << (isStatic ? "true" : "false") << std::endl;
    }

    cachedNumClasses = getNumberOfClasses();
}

std::vector<Detection> DirectMLDetector::detect(const cv::Mat& input_frame)
{
    std::vector<cv::Mat> batch = { input_frame };
    auto batchResult = detectBatch(batch);
    if (!batchResult.empty())
        return batchResult[0];
    else
        return {};
}

std::vector<std::vector<Detection>> DirectMLDetector::detectBatch(const std::vector<cv::Mat>& frames)
{
    std::vector<std::vector<Detection>> empty;
    if (frames.empty()) return empty;
    const size_t batch_size = frames.size();

    const bool useFixed = config.fixed_input_size && cachedModelHeight > 0 && cachedModelWidth > 0;

    const int target_h = useFixed ? cachedModelHeight : config.detection_resolution;
    const int target_w = useFixed ? cachedModelWidth : config.detection_resolution;

    auto t0 = std::chrono::steady_clock::now();

    auto normalizeChannels = [](const cv::Mat& frame, cv::Mat& scratch) -> cv::Mat
    {
        switch (frame.channels())
        {
        case 4:
            cv::cvtColor(frame, scratch, cv::COLOR_BGRA2BGR);
            return scratch;
        case 3:
            return frame;
        case 1:
            cv::cvtColor(frame, scratch, cv::COLOR_GRAY2BGR);
            return scratch;
        default:
            scratch = cv::Mat::zeros(frame.size(), CV_8UC3);
            return scratch;
        }
    };

    if (batch_size == 1)
    {
        bgrFrame = normalizeChannels(frames.front(), bgrFrame);
        cv::Mat source = bgrFrame;
        if (bgrFrame.cols != target_w || bgrFrame.rows != target_h)
        {
            cv::resize(source, resizedFrame, cv::Size(target_w, target_h));
            source = resizedFrame;
        }
        cv::dnn::blobFromImage(
            source,
            inputBlob,
            1.0f / 255.0f,
            cv::Size(target_w, target_h),
            cv::Scalar(),
            true,
            false,
            CV_32F);
    }
    else
    {
        batchPreprocessFrames.clear();
        batchPreprocessFrames.reserve(batch_size);
        std::vector<cv::Mat> channelScratch(batch_size);
        std::vector<cv::Mat> resizeScratch(batch_size);
        for (size_t b = 0; b < batch_size; ++b)
        {
            cv::Mat source = normalizeChannels(frames[b], channelScratch[b]);
            if (source.cols != target_w || source.rows != target_h)
            {
                cv::resize(source, resizeScratch[b], cv::Size(target_w, target_h));
                source = resizeScratch[b];
            }
            batchPreprocessFrames.push_back(source);
        }
        cv::dnn::blobFromImages(
            batchPreprocessFrames,
            inputBlob,
            1.0f / 255.0f,
            cv::Size(target_w, target_h),
            cv::Scalar(),
            true,
            false,
            CV_32F);
    }
    auto t1 = std::chrono::steady_clock::now();

    ortInputShape = {
        static_cast<int64_t>(batch_size),
        3,
        static_cast<int64_t>(target_h),
        static_cast<int64_t>(target_w)
    };
    const size_t inputElementCount = inputBlob.total();
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        inputBlob.ptr<float>(),
        inputElementCount,
        ortInputShape.data(),
        ortInputShape.size());

    const char* input_names[] = { input_name.c_str() };
    const char* output_names[] = { output_name.c_str() };

    auto t2 = std::chrono::steady_clock::now();
    auto output_tensors = session.Run(Ort::RunOptions{ nullptr },
        input_names, &input_tensor, 1,
        output_names, 1);
    auto t3 = std::chrono::steady_clock::now();

    float* outData = output_tensors.front().GetTensorMutableData<float>();
    Ort::TensorTypeAndShapeInfo outInfo = output_tensors.front().GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outShape = outInfo.GetShape(); // [B, rows, cols]
    if (outShape.size() < 3)
    {
        std::cerr << "[DirectMLDetector] Unexpected output tensor rank." << std::endl;
        return empty;
    }

    int rows = 0;
    int cols = 0;
    if (!tryInt64ToInt(outShape[1], &rows) || !tryInt64ToInt(outShape[2], &cols) || rows <= 0 || cols <= 0)
    {
        std::cerr << "[DirectMLDetector] Output tensor dimensions are invalid." << std::endl;
        return empty;
    }
    const int num_classes = (cachedNumClasses > 0) ? cachedNumClasses : (rows - 4);

    std::vector<std::vector<Detection>> batchDetections(batch_size);
    float conf_thr = config.confidence_threshold;
    float nms_thr = config.nms_threshold;

    auto t4 = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> nmsTimeTmp{ 0 };

    outputShape2d = { static_cast<int64_t>(rows), static_cast<int64_t>(cols) };
    std::vector<int64_t> shp = { outputShape2d[0], outputShape2d[1] };

    for (size_t b = 0; b < batch_size; ++b)
    {
        const float* ptr = outData + b * rows * cols;
        std::vector<Detection> detections;

        detections = postProcessYoloDML(ptr, shp, num_classes, conf_thr, nms_thr, &nmsTimeTmp);

        if (useFixed && (target_w != config.detection_resolution || target_h != config.detection_resolution))
        {
            const float scaleX = static_cast<float>(config.detection_resolution) / target_w;
            const float scaleY = static_cast<float>(config.detection_resolution) / target_h;
            for (auto& d : detections)
            {
                d.box.x = static_cast<int>(d.box.x * scaleX);
                d.box.y = static_cast<int>(d.box.y * scaleY);
                d.box.width = static_cast<int>(d.box.width * scaleX);
                d.box.height = static_cast<int>(d.box.height * scaleY);
            }
        }

        batchDetections[b] = std::move(detections);
    }
    auto t5 = std::chrono::steady_clock::now();

    lastPreprocessTimeDML = t1 - t0;
    lastInferenceTimeDML = t3 - t2;
    lastCopyTimeDML = t4 - t3;
    lastPostprocessTimeDML = t5 - t4;
    lastNmsTimeDML = nmsTimeTmp;

    return batchDetections;
}

int DirectMLDetector::getNumberOfClasses()
{
    Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(0);
    auto tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = tensor_info.GetShape();

    if (output_shape.size() == 3)
    {
        int channels = 0;
        if (!tryInt64ToInt(output_shape[1], &channels))
        {
            std::cerr << "[DirectMLDetector] Output tensor channel dimension is invalid." << std::endl;
            return -1;
        }
        int num_classes = channels - 4;
        return num_classes;
    }
    else
    {
        std::cerr << "[DirectMLDetector] Unexpected output tensor shape." << std::endl;
        return -1;
    }
}

void DirectMLDetector::processFrame(const cv::Mat& frame)
{
    static std::atomic<uint64_t> fallbackFrameId{ 0 };
    processFrame(
        frame,
        fallbackFrameId.fetch_add(1, std::memory_order_relaxed) + 1,
        std::chrono::steady_clock::now());
}

void DirectMLDetector::processFrame(
    const cv::Mat& frame,
    uint64_t frameId,
    std::chrono::steady_clock::time_point captureTimestamp)
{
    processFrame(
        frame,
        frameId,
        captureTimestamp,
        CaptureFrameGeometry::FromCrop(
            0,
            0,
            frame.cols,
            frame.rows,
            frame.cols,
            frame.rows,
            frame.cols,
            frame.rows,
            false));
}

void DirectMLDetector::processFrame(
    const cv::Mat& frame,
    uint64_t frameId,
    std::chrono::steady_clock::time_point captureTimestamp,
    const CaptureFrameGeometry& frameGeometry)
{
    if (detectionPaused)
    {
        detectionBuffer.clear(frameId, captureTimestamp, frameGeometry);
        return;
    }
    std::unique_lock<std::mutex> lock(inferenceMutex);
    currentFrame = frame;
    currentFrameId = frameId;
    currentFrameCaptureTimestamp = captureTimestamp;
    currentFrameGeometry = frameGeometry;
    frameReady = true;
    inferenceCV.notify_one();
}

void DirectMLDetector::dmlInferenceThread()
{
    try
    {
        while (!shouldExit)
        {
            if (detector_model_changed.load())
            {
                initializeModel("models/" + config.ai_model);
                detection_resolution_changed.store(true);
                detection_resolution_generation.fetch_add(1, std::memory_order_relaxed);
                detector_model_changed.store(false);
                std::cout << "[DML] Detector reloaded: " << config.ai_model << std::endl;
            }

            if (detectionPaused)
            {
                detectionBuffer.clear();
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            cv::Mat frame;
            uint64_t frameId = 0;
            std::chrono::steady_clock::time_point frameCaptureTimestamp{};
            CaptureFrameGeometry frameGeometry;
            bool hasNewFrame = false;
            {
                std::unique_lock<std::mutex> lock(inferenceMutex);
                if (!frameReady && !shouldExit)
                    inferenceCV.wait(lock, [this] { return frameReady || shouldExit; });

                if (shouldExit) break;

                if (frameReady)
                {
                    frame = std::move(currentFrame);
                    frameId = currentFrameId;
                    frameCaptureTimestamp = currentFrameCaptureTimestamp;
                    frameGeometry = currentFrameGeometry;
                    frameReady = false;
                    hasNewFrame = true;
                }
            }

            if (hasNewFrame && !frame.empty())
            {
                std::vector<cv::Mat> batchFrames = { frame };
                auto detectionsBatch = detectBatch(batchFrames);
                if (detectionsBatch.empty())
                {
                    detectionBuffer.publish(
                        std::vector<cv::Rect>{},
                        std::vector<int>{},
                        std::vector<float>{},
                        frameId,
                        frameCaptureTimestamp,
                        frameGeometry);
                    continue;
                }
                std::vector<Detection> filteredDetections = std::move(detectionsBatch.back());
                filterDetectionsByDepthMask(filteredDetections);

                std::vector<cv::Rect> boxes;
                std::vector<int> classes;
                std::vector<float> confidences;
                boxes.reserve(filteredDetections.size());
                classes.reserve(filteredDetections.size());
                confidences.reserve(filteredDetections.size());
                for (const auto& d : filteredDetections)
                {
                    boxes.push_back(d.box);
                    classes.push_back(d.classId);
                    confidences.push_back(d.confidence);
                }
                detectionBuffer.publish(
                    std::move(boxes),
                    std::move(classes),
                    std::move(confidences),
                    frameId,
                    frameCaptureTimestamp,
                    frameGeometry);
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "[DML] Inference thread crashed: " << e.what() << std::endl;
        shouldExit = true;
        inferenceCV.notify_all();
        detectionBuffer.cv.notify_all();
    }
    catch (...)
    {
        std::cerr << "[DML] Inference thread crashed: unknown exception." << std::endl;
        shouldExit = true;
        inferenceCV.notify_all();
        detectionBuffer.cv.notify_all();
    }
}
