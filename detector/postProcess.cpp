#include <algorithm>
#include <numeric>
#include <chrono>
#include <cmath>
#include <limits>

#include "postProcess.h"
#ifndef YOLO_ANNOTATION_WORKER
#include "0BS_box_2.h"
#ifdef USE_CUDA
#include "trt_detector.h"
#endif
#endif

namespace
{
    void limitDetectionsForNms(std::vector<Detection>& detections)
    {
#ifdef YOLO_ANNOTATION_WORKER
        (void)detections;
#else
        const int maxDetectionsConfig = config.max_detections;
        if (maxDetectionsConfig <= 0)
        {
            detections.clear();
            return;
        }

        const size_t maxDetections = static_cast<size_t>(maxDetectionsConfig);
        if (detections.size() <= maxDetections) return;

        std::nth_element(
            detections.begin(),
            detections.begin() + maxDetections,
            detections.end(),
            [](const Detection& a, const Detection& b)
            {
                return a.confidence > b.confidence;
            }
        );
        detections.resize(maxDetections);
#endif
    }

#ifdef USE_CUDA
    struct YoloDecoderLayout
    {
        bool attributeMajor = false;
        bool nmsOutput = false;
        bool hasObjectness = false;
        int64_t boxCount = 0;
        int64_t attributeCount = 0;
        int classCount = 0;
    };

    bool squeezeYoloShape(const std::vector<int64_t>& shape, int64_t& rows, int64_t& cols)
    {
        if (shape.size() == 3 && shape[0] == 1)
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

    float readYoloAttribute(const float* output, const YoloDecoderLayout& layout, int64_t boxIndex, int64_t attributeIndex)
    {
        return layout.attributeMajor
            ? output[attributeIndex * layout.boxCount + boxIndex]
            : output[boxIndex * layout.attributeCount + attributeIndex];
    }

    bool looksIntegerLike(float value)
    {
        return std::isfinite(value) && std::fabs(value - std::round(value)) <= 0.001f;
    }

    bool looksLikeClassIdRow(const float* output, int64_t boxCount, int explicitClassCount)
    {
        if (boxCount <= 0) return false;

        const int64_t samples = std::min<int64_t>(boxCount, 64);
        int checked = 0;
        int classIdLike = 0;
        for (int64_t i = 0; i < samples; ++i)
        {
            const int64_t boxIndex = (samples == boxCount) ? i : (i * boxCount / samples);
            const float confidence = output[4 * boxCount + boxIndex];
            const float classValue = output[5 * boxCount + boxIndex];
            if (!std::isfinite(confidence) || !std::isfinite(classValue))
            {
                continue;
            }
            if (confidence < 0.0f || confidence > 1.5f || classValue < 0.0f)
            {
                continue;
            }

            ++checked;
            const bool inClassRange = explicitClassCount <= 0 || classValue < static_cast<float>(explicitClassCount);
            if (inClassRange && looksIntegerLike(classValue))
            {
                ++classIdLike;
            }
        }

        return checked > 0 && classIdLike * 4 >= checked * 3;
    }

    bool makeScoreLayout(
        bool attributeMajor,
        int64_t attributeCount,
        int64_t boxCount,
        int explicitClassCount,
        YoloDecoderLayout& layout)
    {
        if (attributeCount < 5 || boxCount <= 0) return false;

        if (explicitClassCount > 0)
        {
            if (attributeCount == 4 + explicitClassCount)
            {
                layout = YoloDecoderLayout{ attributeMajor, false, false, boxCount, attributeCount, explicitClassCount };
                return true;
            }
            if (attributeCount == 5 + explicitClassCount)
            {
                layout = YoloDecoderLayout{ attributeMajor, false, true, boxCount, attributeCount, explicitClassCount };
                return true;
            }
            return false;
        }

        const int fallbackClassCount = static_cast<int>(attributeCount - 4);
        if (fallbackClassCount <= 0) return false;
        layout = YoloDecoderLayout{ attributeMajor, false, false, boxCount, attributeCount, fallbackClassCount };
        return true;
    }

    void collectExplicitScoreLayout(
        bool attributeMajor,
        int64_t attributeCount,
        int64_t boxCount,
        int explicitClassCount,
        YoloDecoderLayout& objectnessLayout,
        bool& hasObjectnessLayout,
        YoloDecoderLayout& plainLayout,
        bool& hasPlainLayout)
    {
        YoloDecoderLayout candidate;
        if (makeScoreLayout(attributeMajor, attributeCount, boxCount, explicitClassCount, candidate))
        {
            if (candidate.hasObjectness)
            {
                objectnessLayout = candidate;
                hasObjectnessLayout = true;
            }
            else
            {
                plainLayout = candidate;
                hasPlainLayout = true;
            }
        }

        if (explicitClassCount > 1 &&
            makeScoreLayout(attributeMajor, attributeCount, boxCount, explicitClassCount - 1, candidate) &&
            candidate.hasObjectness)
        {
            objectnessLayout = candidate;
            hasObjectnessLayout = true;
        }
    }

    bool resolveYoloDecoderLayout(
        const float* output,
        const std::vector<int64_t>& shape,
        int explicitClassCount,
        YoloDecoderLayout& layout)
    {
        int64_t rows = 0;
        int64_t cols = 0;
        if (!squeezeYoloShape(shape, rows, cols))
        {
            return false;
        }
        if (rows > std::numeric_limits<int>::max() || cols > std::numeric_limits<int>::max())
        {
            return false;
        }

        if (cols == 6)
        {
            layout = YoloDecoderLayout{ false, true, false, rows, cols, explicitClassCount };
            return true;
        }

        if (rows == 6 && looksLikeClassIdRow(output, cols, explicitClassCount))
        {
            layout = YoloDecoderLayout{ true, true, false, cols, rows, explicitClassCount };
            return true;
        }

        if (explicitClassCount > 0)
        {
            YoloDecoderLayout objectnessLayout;
            YoloDecoderLayout plainLayout;
            bool hasObjectnessLayout = false;
            bool hasPlainLayout = false;

            collectExplicitScoreLayout(
                true,
                rows,
                cols,
                explicitClassCount,
                objectnessLayout,
                hasObjectnessLayout,
                plainLayout,
                hasPlainLayout);
            collectExplicitScoreLayout(
                false,
                cols,
                rows,
                explicitClassCount,
                objectnessLayout,
                hasObjectnessLayout,
                plainLayout,
                hasPlainLayout);

            if (hasObjectnessLayout)
            {
                layout = objectnessLayout;
                return true;
            }
            if (hasPlainLayout)
            {
                layout = plainLayout;
                return true;
            }
        }

        if (makeScoreLayout(true, rows, cols, explicitClassCount, layout))
        {
            return true;
        }
        if (makeScoreLayout(false, cols, rows, explicitClassCount, layout))
        {
            return true;
        }

        if (rows >= 5 && rows <= cols && makeScoreLayout(true, rows, cols, 0, layout))
        {
            return true;
        }
        if (cols >= 5 && makeScoreLayout(false, cols, rows, 0, layout))
        {
            return true;
        }

        return false;
    }
#endif
}

void NMS(std::vector<Detection>& detections, float nmsThreshold, std::chrono::duration<double, std::milli>* nmsTime)
{
    if (detections.empty()) return;

    limitDetectionsForNms(detections);
    if (detections.empty())
    {
        if (nmsTime)
        {
            *nmsTime = std::chrono::duration<double, std::milli>(0);
        }
        return;
    }

    if (nmsThreshold <= 0.0f)
    {
        if (nmsTime)
        {
            *nmsTime = std::chrono::duration<double, std::milli>(0);
        }
        return;
    }

    auto t0 = std::chrono::steady_clock::now();

    std::sort(
        detections.begin(),
        detections.end(),
        [](const Detection& a, const Detection& b)
        {
            return a.confidence > b.confidence;
        }
    );

    std::vector<bool> suppress(detections.size(), false);
    std::vector<Detection> result;
    result.reserve(detections.size());

    for (size_t i = 0; i < detections.size(); ++i)
    {
        if (suppress[i]) continue;

        result.push_back(detections[i]);

        const cv::Rect& box_i = detections[i].box;
        const float area_i = static_cast<float>(box_i.area());

        for (size_t j = i + 1; j < detections.size(); ++j)
        {
            if (suppress[j]) continue;
            if (detections[j].classId != detections[i].classId) continue;

            const cv::Rect& box_j = detections[j].box;
            const cv::Rect intersection = box_i & box_j;

            if (intersection.width > 0 && intersection.height > 0)
            {
                const float intersection_area = static_cast<float>(intersection.area());
                const float union_area = area_i + static_cast<float>(box_j.area()) - intersection_area;

                if (intersection_area / union_area > nmsThreshold)
                {
                    suppress[j] = true;
                }
            }
        }
    }

    detections = std::move(result);
    limitDetectionsForNms(detections);

    auto t1 = std::chrono::steady_clock::now();
    if (nmsTime)
    {
        *nmsTime = t1 - t0;
    }
}

#ifdef USE_CUDA
std::vector<Detection> postProcessYoloScaled(
    const float* output,
    const std::vector<int64_t>& shape,
    int numClasses,
    float confThreshold,
    float nmsThreshold,
    float imgScale,
    std::chrono::duration<double, std::milli>* nmsTime
)
{
    std::vector<Detection> detections;
    detections.reserve(256);

    YoloDecoderLayout layout;
    if (!resolveYoloDecoderLayout(output, shape, numClasses, layout))
    {
        return detections;
    }

    if (layout.nmsOutput)
    {
        for (int64_t i = 0; i < layout.boxCount; ++i)
        {
            const float confidence = readYoloAttribute(output, layout, i, 4);
            if (confidence <= confThreshold) continue;

            const float x1 = readYoloAttribute(output, layout, i, 0);
            const float y1 = readYoloAttribute(output, layout, i, 1);
            const float x2 = readYoloAttribute(output, layout, i, 2);
            const float y2 = readYoloAttribute(output, layout, i, 3);
            const int classId = static_cast<int>(std::round(readYoloAttribute(output, layout, i, 5)));

            Detection detection;
            detection.box.x = static_cast<int>(x1 * imgScale);
            detection.box.y = static_cast<int>(y1 * imgScale);
            detection.box.width = static_cast<int>((x2 - x1) * imgScale);
            detection.box.height = static_cast<int>((y2 - y1) * imgScale);
            detection.confidence = confidence;
            detection.classId = classId;
            detections.push_back(detection);
        }
    }
    else
    {
        const int classOffset = layout.hasObjectness ? 5 : 4;
        if (layout.classCount <= 0 || classOffset + layout.classCount > layout.attributeCount)
        {
            return detections;
        }

        for (int64_t i = 0; i < layout.boxCount; ++i)
        {
            const float objectness = layout.hasObjectness ? readYoloAttribute(output, layout, i, 4) : 1.0f;
            float maxScore = std::numeric_limits<float>::lowest();
            int maxClassId = 0;
            for (int c = 0; c < layout.classCount; ++c)
            {
                const float score = objectness * readYoloAttribute(output, layout, i, classOffset + c);
                if (score > maxScore)
                {
                    maxScore = score;
                    maxClassId = c;
                }
            }

            if (maxScore > confThreshold)
            {
                const float cx = readYoloAttribute(output, layout, i, 0);
                const float cy = readYoloAttribute(output, layout, i, 1);
                const float ow = readYoloAttribute(output, layout, i, 2);
                const float oh = readYoloAttribute(output, layout, i, 3);
                const float half_ow = 0.5f * ow;
                const float half_oh = 0.5f * oh;

                Detection det;
                det.box.x = static_cast<int>((cx - half_ow) * imgScale);
                det.box.y = static_cast<int>((cy - half_oh) * imgScale);
                det.box.width = static_cast<int>(ow * imgScale);
                det.box.height = static_cast<int>(oh * imgScale);
                det.confidence = maxScore;
                det.classId = maxClassId;

                detections.push_back(det);
            }
        }
    }

    NMS(detections, nmsThreshold, nmsTime);
    return detections;
}

#ifndef YOLO_ANNOTATION_WORKER
std::vector<Detection> postProcessYolo(
    const float* output,
    const std::vector<int64_t>& shape,
    int numClasses,
    float confThreshold,
    float nmsThreshold,
    std::chrono::duration<double, std::milli>* nmsTime
)
{
    return postProcessYoloScaled(
        output,
        shape,
        numClasses,
        confThreshold,
        nmsThreshold,
        trt_detector.img_scale,
        nmsTime
    );
}
#endif
#endif

std::vector<Detection> postProcessYoloDML(
    const float* output,
    const std::vector<int64_t>& shape,
    int numClasses,
    float confThreshold,
    float nmsThreshold,
    std::chrono::duration<double, std::milli>* nmsTime
)
{
    std::vector<Detection> detections;
    if (shape.size() != 2) return detections;

    int64_t rows = shape[0];
    int64_t cols = shape[1];
    if (rows <= 0 || cols <= 0) return detections;
    if (rows > std::numeric_limits<int>::max() || cols > std::numeric_limits<int>::max()) return detections;
    const int rows_i = static_cast<int>(rows);
    const int cols_i = static_cast<int>(cols);

    if (cols_i == 6)
    {
        const int numDetections = rows_i;
        detections.reserve(static_cast<size_t>(numDetections));
        for (int i = 0; i < numDetections; ++i)
        {
            const float* det = output + i * cols_i;
            float confidence = det[4];
            if (confidence > confThreshold)
            {
                int classId = static_cast<int>(det[5]);
                float cx = det[0];
                float cy = det[1];
                float dx = det[2];
                float dy = det[3];

                cv::Rect box;
                box.x = static_cast<int>(cx);
                box.y = static_cast<int>(cy);
                box.width = static_cast<int>(dx - cx);
                box.height = static_cast<int>(dy - cy);
                detections.push_back(Detection{ box, confidence, classId });
            }
        }
        NMS(detections, nmsThreshold, nmsTime);
        return detections;
    }

    if (numClasses <= 0 || rows_i < 4 || numClasses > rows_i - 4) return detections;

    detections.reserve(static_cast<size_t>(cols_i));
    for (int i = 0; i < cols_i; ++i)
    {
        float score = std::numeric_limits<float>::lowest();
        int classId = 0;
        for (int c = 0; c < numClasses; ++c)
        {
            const float classScore = output[(4 + c) * cols_i + i];
            if (classScore > score)
            {
                score = classScore;
                classId = c;
            }
        }

        if (score > confThreshold)
        {
            float cx = output[0 * cols_i + i];
            float cy = output[1 * cols_i + i];
            float ow = output[2 * cols_i + i];
            float oh = output[3 * cols_i + i];
            const float half_ow = 0.5f * ow;
            const float half_oh = 0.5f * oh;
            cv::Rect box;
            box.x = static_cast<int>(cx - half_ow);
            box.y = static_cast<int>(cy - half_oh);
            box.width = static_cast<int>(ow);
            box.height = static_cast<int>(oh);
            detections.push_back(Detection{ box, score, classId });
        }
    }
    if (!detections.empty())
    {
        NMS(detections, nmsThreshold, nmsTime);
    }
    return detections;
}