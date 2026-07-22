#include "postProcess.h"
#include "test_harness.h"

#include <algorithm>
#include <chrono>
#include <vector>

namespace
{
const Detection* findDetection(const std::vector<Detection>& detections, int classId)
{
    const auto it = std::find_if(
        detections.begin(),
        detections.end(),
        [classId](const Detection& detection)
        {
            return detection.classId == classId;
        });
    return it == detections.end() ? nullptr : &(*it);
}

void testNmsSuppressesOnlyWithinClass()
{
    std::vector<Detection> detections = {
        { cv::Rect(10, 10, 100, 100), 0.95f, 0 },
        { cv::Rect(15, 15, 100, 100), 0.80f, 0 },
        { cv::Rect(16, 16, 100, 100), 0.70f, 1 },
        { cv::Rect(300, 300, 40, 40), 0.60f, 0 },
    };

    std::chrono::duration<double, std::milli> nmsTime{};
    NMS(detections, 0.50f, &nmsTime);

    REQUIRE(detections.size() == 3);
    REQUIRE(detections[0].confidence >= detections[1].confidence);
    REQUIRE(findDetection(detections, 1) != nullptr);
    REQUIRE(nmsTime.count() >= 0.0);

    const int classZeroCount = static_cast<int>(std::count_if(
        detections.begin(),
        detections.end(),
        [](const Detection& detection)
        {
            return detection.classId == 0;
        }));
    REQUIRE(classZeroCount == 2);
}

void testCudaScaledDecoderHandlesAttributeMajorObjectnessScores()
{
    const std::vector<int64_t> shape = { 1, 7, 3 };
    const std::vector<float> output = {
        100.0f, 200.0f, 300.0f,
        80.0f, 120.0f, 180.0f,
        20.0f, 30.0f, 50.0f,
        40.0f, 50.0f, 60.0f,
        0.95f, 0.80f, 0.50f,
        0.20f, 0.70f, 0.10f,
        0.88f, 0.10f, 0.20f,
    };

    const std::vector<Detection> detections = postProcessYoloScaled(
        output.data(),
        shape,
        2,
        0.50f,
        0.40f,
        2.0f);

    REQUIRE(detections.size() == 2);

    const Detection* classOne = findDetection(detections, 1);
    REQUIRE(classOne != nullptr);
    REQUIRE(classOne->box.x == 180);
    REQUIRE(classOne->box.y == 120);
    REQUIRE(classOne->box.width == 40);
    REQUIRE(classOne->box.height == 80);
    REQUIRE_NEAR(classOne->confidence, 0.836, 1e-6);

    const Detection* classZero = findDetection(detections, 0);
    REQUIRE(classZero != nullptr);
    REQUIRE(classZero->box.x == 370);
    REQUIRE(classZero->box.y == 190);
    REQUIRE(classZero->box.width == 60);
    REQUIRE(classZero->box.height == 100);
    REQUIRE_NEAR(classZero->confidence, 0.56, 1e-6);
}

void testCudaScaledDecoderBypassesLocalNmsForFinalRows()
{
    const std::vector<int64_t> shape = { 2, 6 };
    const std::vector<float> output = {
        10.0f, 10.0f, 60.0f, 60.0f, 0.93f, 0.0f,
        12.0f, 12.0f, 62.0f, 62.0f, 0.88f, 0.0f,
    };

    const std::vector<Detection> detections = postProcessYoloScaled(
        output.data(),
        shape,
        1,
        0.50f,
        0.01f,
        1.0f);

    REQUIRE(detections.size() == 2);
    REQUIRE(detections[0].box == cv::Rect(10, 10, 50, 50));
    REQUIRE(detections[1].box == cv::Rect(12, 12, 50, 50));
}

void testDmlDecoderHandlesFinalRows()
{
    const std::vector<int64_t> shape = { 3, 6 };
    const std::vector<float> output = {
        10.0f, 20.0f, 50.0f, 80.0f, 0.91f, 2.0f,
        12.0f, 22.0f, 52.0f, 82.0f, 0.89f, 2.0f,
        100.0f, 110.0f, 130.0f, 140.0f, 0.20f, 1.0f,
    };

    const std::vector<Detection> detections = postProcessYoloDML(
        output.data(),
        shape,
        3,
        0.50f,
        0.01f);

    REQUIRE(detections.size() == 2);
    REQUIRE(detections[0].classId == 2);
    REQUIRE(detections[0].box == cv::Rect(10, 20, 40, 60));
    REQUIRE(detections[1].box == cv::Rect(12, 22, 40, 60));
}

void testDmlAttributeMajorDecoderAppliesNms()
{
    const std::vector<int64_t> shape = { 6, 2 };
    const std::vector<float> output = {
        50.0f, 52.0f,
        60.0f, 62.0f,
        40.0f, 40.0f,
        40.0f, 40.0f,
        0.90f, 0.70f,
        0.10f, 0.20f,
    };

    const std::vector<Detection> detections = postProcessYoloDML(
        output.data(),
        shape,
        2,
        0.50f,
        0.30f);

    REQUIRE(detections.size() == 1);
    REQUIRE(detections[0].classId == 0);
    REQUIRE(detections[0].box == cv::Rect(30, 40, 40, 40));
    REQUIRE_NEAR(detections[0].confidence, 0.90, 1e-6);
}
}

int main()
{
    return obs2test::runTests(
        {
            { "NMS suppresses only within class", testNmsSuppressesOnlyWithinClass },
            { "CUDA scaled decoder handles attribute-major objectness scores", testCudaScaledDecoderHandlesAttributeMajorObjectnessScores },
            { "CUDA scaled decoder bypasses local NMS for final rows", testCudaScaledDecoderBypassesLocalNmsForFinalRows },
            { "DML decoder handles final rows", testDmlDecoderHandlesFinalRows },
            { "DML attribute-major decoder applies NMS", testDmlAttributeMajorDecoderAppliesNms },
        },
        "postprocess");
}
