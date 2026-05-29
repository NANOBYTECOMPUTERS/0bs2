#ifndef DIRECTML_DETECTOR_H
#define DIRECTML_DETECTOR_H

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <vector>

#include "capture/capture_geometry.h"
#include "postProcess.h"

class DirectMLDetector
{
public:
    DirectMLDetector(const std::string& model_path);
    ~DirectMLDetector();

    std::vector<Detection> detect(const cv::Mat& input_frame);
    std::vector<std::vector<Detection>> detectBatch(const std::vector<cv::Mat>& frames);

    void dmlInferenceThread();
    void processFrame(const cv::Mat& frame);
    void processFrame(const cv::Mat& frame, uint64_t frameId, std::chrono::steady_clock::time_point captureTimestamp);
    void processFrame(
        const cv::Mat& frame,
        uint64_t frameId,
        std::chrono::steady_clock::time_point captureTimestamp,
        const CaptureFrameGeometry& frameGeometry);

    int getNumberOfClasses();

    std::chrono::duration<double, std::milli> lastInferenceTimeDML;
    std::chrono::duration<double, std::milli> lastPreprocessTimeDML;
    std::chrono::duration<double, std::milli> lastCopyTimeDML;
    std::chrono::duration<double, std::milli> lastPostprocessTimeDML;
    std::chrono::duration<double, std::milli> lastNmsTimeDML;

    std::condition_variable inferenceCV;
    std::atomic<bool> shouldExit = false;

private:
    Ort::Env env;
    Ort::Session session{ nullptr };
    Ort::SessionOptions session_options;
    Ort::AllocatorWithDefaultOptions allocator;

    std::string input_name;
    std::string output_name;
    std::vector<int64_t> input_shape;
    int cachedModelHeight = -1;
    int cachedModelWidth = -1;
    bool cachedFixedInput = false;
    int cachedNumClasses = -1;

    cv::Mat inputBlob;
    cv::Mat bgrFrame;
    cv::Mat resizedFrame;
    std::vector<cv::Mat> batchPreprocessFrames;
    std::array<int64_t, 4> ortInputShape{};
    std::array<int64_t, 2> outputShape2d{};

    std::mutex inferenceMutex;
    cv::Mat currentFrame;
    uint64_t currentFrameId = 0;
    std::chrono::steady_clock::time_point currentFrameCaptureTimestamp{};
    CaptureFrameGeometry currentFrameGeometry;
    bool frameReady = false;

    void initializeModel(const std::string& model_path);
    Ort::MemoryInfo memory_info;
};

#endif // DIRECTML_DETECTOR_H
