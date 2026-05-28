#include "ndjson_protocol.h"
#include "tensorrt_batch_detector.h"

#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

static std::string loadedModel;
static std::string loadedBackend;
static std::string detectorError;
static int loadedClassCount = 0;

#ifdef USE_CUDA
static TensorRtBatchDetector detector;
static bool detectorReady = false;
#endif

namespace
{
    std::string serializeDetection(const WorkerDetection& detection)
    {
        std::ostringstream body;
        body << "{\"class_id\":" << detection.classId
            << ",\"confidence\":" << detection.confidence
            << ",\"x\":" << detection.x
            << ",\"y\":" << detection.y
            << ",\"width\":" << detection.width
            << ",\"height\":" << detection.height
            << "}";
        return body.str();
    }

    std::string serializeResult(const WorkerDetectionResult& result)
    {
        std::string body = "{\"image\":\"" + jsonEscape(result.image) +
            "\",\"width\":" + std::to_string(result.width) +
            ",\"height\":" + std::to_string(result.height) +
            ",\"detections\":[";

        for (size_t i = 0; i < result.detections.size(); ++i)
        {
            if (i > 0) body += ",";
            body += serializeDetection(result.detections[i]);
        }

        body += "]";
        if (!result.error.empty())
        {
            body += ",\"error\":\"" + jsonEscape(result.error) + "\"";
        }
        body += "}";
        return body;
    }
}

std::string handleHello()
{
    return okObject("\"worker\":\"yolo_annotation_worker\",\"version\":\"0.1.0\",\"backends\":[\"TRT\"]");
}

std::string handleLoadModel(const JsonRequest& request)
{
    auto modelIt = request.strings.find("model_path");
    if (modelIt == request.strings.end()) return errorObject("load_model requires model_path");

    loadedModel = modelIt->second;
    auto backendIt = request.strings.find("backend");
    loadedBackend = backendIt == request.strings.end() ? "TRT" : backendIt->second;
    loadedClassCount = request.classCount;
    detectorError.clear();

#ifdef USE_CUDA
    if (loadedBackend != "TRT")
    {
        detectorReady = false;
        return errorObject("yolo_annotation_worker supports only the TRT backend");
    }

    if (!std::filesystem::exists(std::filesystem::path(loadedModel)))
    {
        detectorReady = false;
        detectorError = "model file does not exist: " + loadedModel;
        return okObject(
            "\"model_path\":\"" + jsonEscape(loadedModel) +
            "\",\"backend\":\"" + jsonEscape(loadedBackend) +
            "\",\"warning\":\"" + jsonEscape(detectorError) + "\""
        );
    }

    std::string error;
    detector.setClassCount(loadedClassCount);
    detectorReady = detector.loadEngine(loadedModel, error);
    if (!detectorReady)
    {
        detectorError = error;
        return errorObject(error);
    }

    return okObject(
        "\"model_path\":\"" + jsonEscape(loadedModel) +
        "\",\"backend\":\"" + jsonEscape(loadedBackend) +
        "\",\"class_count\":" + std::to_string(loadedClassCount)
    );
#else
    return errorObject("CUDA support is not enabled in yolo_annotation_worker");
#endif
}

std::string handleAnnotateBatch(const JsonRequest& request)
{
    if (loadedModel.empty()) return errorObject("model must be loaded before annotate_batch");

    std::string body = "\"results\":[";
#ifdef USE_CUDA
    std::vector<WorkerDetectionResult> results;
    results.reserve(request.images.size());

    if (request.classCount > 0 && loadedClassCount > 0 && request.classCount != loadedClassCount)
    {
        for (const auto& image : request.images)
        {
            WorkerDetectionResult result;
            result.image = image;
            result.error = "annotate_batch class_count does not match loaded model";
            results.push_back(std::move(result));
        }
    }
    else
    {
        if (request.classCount > 0 && loadedClassCount <= 0)
        {
            loadedClassCount = request.classCount;
            detector.setClassCount(loadedClassCount);
        }

        if (detectorReady)
        {
            results = detector.detectBatch(
                request.images,
                static_cast<float>(request.confidenceThreshold),
                static_cast<float>(request.nmsThreshold));
        }
        else
        {
            for (const auto& image : request.images)
            {
                WorkerDetectionResult result;
                result.image = image;
                result.error = detectorError.empty() ? "inference not initialized" : detectorError;
                results.push_back(std::move(result));
            }
        }
    }

    for (size_t i = 0; i < results.size(); ++i)
    {
        if (i > 0) body += ",";
        body += serializeResult(results[i]);
    }
#else
    for (size_t i = 0; i < request.images.size(); ++i)
    {
        if (i > 0) body += ",";
        WorkerDetectionResult result;
        result.image = request.images[i];
        result.error = "CUDA support is not enabled in yolo_annotation_worker";
        body += serializeResult(result);
    }
#endif
    body += "]";
    return okObject(body);
}

std::string handleShutdown()
{
    return okObject("");
}

int main()
{
    std::string line;
    while (std::getline(std::cin, line))
    {
        JsonRequest request;
        std::string error;
        if (!parseJsonRequest(line, request, error))
        {
            std::cout << errorObject(error) << std::endl;
            continue;
        }

        if (request.command == "hello")
        {
            std::cout << handleHello() << std::endl;
        }
        else if (request.command == "load_model")
        {
            std::cout << handleLoadModel(request) << std::endl;
        }
        else if (request.command == "annotate_batch")
        {
            std::cout << handleAnnotateBatch(request) << std::endl;
        }
        else if (request.command == "shutdown")
        {
            std::cout << handleShutdown() << std::endl;
            break;
        }
        else
        {
            std::cout << errorObject("unknown command: " + request.command) << std::endl;
        }
    }

    return 0;
}
