#pragma once
#include <vector>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cstdint>
#include <opencv2/opencv.hpp>

#include "capture/capture_geometry.h"

struct DetectionBuffer
{
    std::mutex mutex;
    std::condition_variable cv;
    int version = 0;
    uint64_t frameId = 0;
    std::vector<cv::Rect> boxes;
    std::vector<int> classes;
    std::vector<float> confidences;
    CaptureFrameGeometry geometry;
    std::chrono::steady_clock::time_point captureTimestamp{};
    std::chrono::steady_clock::time_point publishTimestamp{};
    std::chrono::steady_clock::time_point timestamp{};

    void clear(
        uint64_t newFrameId = 0,
        std::chrono::steady_clock::time_point newCaptureTimestamp = {},
        const CaptureFrameGeometry& newGeometry = CaptureFrameGeometry())
    {
        const auto now = std::chrono::steady_clock::now();
        if (newCaptureTimestamp.time_since_epoch().count() == 0)
            newCaptureTimestamp = now;

        {
            std::lock_guard<std::mutex> lock(mutex);
            boxes.clear();
            classes.clear();
            confidences.clear();
            geometry = newGeometry;
            frameId = newFrameId;
            captureTimestamp = newCaptureTimestamp;
            publishTimestamp = now;
            timestamp = publishTimestamp;
            ++version;
        }
        cv.notify_all();
    }

    void publish(
        std::vector<cv::Rect>&& newBoxes,
        std::vector<int>&& newClasses,
        std::vector<float>&& newConfidences,
        uint64_t newFrameId,
        std::chrono::steady_clock::time_point newCaptureTimestamp,
        const CaptureFrameGeometry& newGeometry = CaptureFrameGeometry())
    {
        const auto now = std::chrono::steady_clock::now();
        if (newCaptureTimestamp.time_since_epoch().count() == 0)
            newCaptureTimestamp = now;

        {
            std::lock_guard<std::mutex> lock(mutex);
            boxes = std::move(newBoxes);
            classes = std::move(newClasses);
            confidences = std::move(newConfidences);
            if (confidences.size() != boxes.size())
                confidences.assign(boxes.size(), 1.0f);
            geometry = newGeometry;
            frameId = newFrameId;
            captureTimestamp = newCaptureTimestamp;
            publishTimestamp = now;
            timestamp = publishTimestamp;
            ++version;
        }
        cv.notify_all();
    }

    void set(
        const std::vector<cv::Rect>& newBoxes,
        const std::vector<int>& newClasses,
        const std::vector<float>& newConfidences)
    {
        publish(
            std::vector<cv::Rect>(newBoxes),
            std::vector<int>(newClasses),
            std::vector<float>(newConfidences),
            0,
            std::chrono::steady_clock::now(),
            CaptureFrameGeometry());
    }

    void get(
        std::vector<cv::Rect>& outBoxes,
        std::vector<int>& outClasses,
        std::vector<float>& outConfidences,
        int& outVersion)
    {
        CaptureFrameGeometry ignoredGeometry;
        get(outBoxes, outClasses, outConfidences, outVersion, ignoredGeometry);
    }

    void get(
        std::vector<cv::Rect>& outBoxes,
        std::vector<int>& outClasses,
        std::vector<float>& outConfidences,
        int& outVersion,
        CaptureFrameGeometry& outGeometry)
    {
        std::lock_guard<std::mutex> lock(mutex);
        outBoxes = boxes;
        outClasses = classes;
        outConfidences = confidences;
        outVersion = version;
        outGeometry = geometry;
    }
};
