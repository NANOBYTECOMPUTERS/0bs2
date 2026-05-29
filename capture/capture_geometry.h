#pragma once

#include <algorithm>
#include <cstdint>
#include <opencv2/core.hpp>

struct CaptureFrameGeometry
{
    int modelWidth = 0;
    int modelHeight = 0;
    int cropX = 0;
    int cropY = 0;
    int cropWidth = 0;
    int cropHeight = 0;
    int screenWidth = 0;
    int screenHeight = 0;
    bool validScreenMapping = false;
    uint64_t generation = 0;

    bool hasValidModel() const
    {
        return modelWidth > 0 && modelHeight > 0 && cropWidth > 0 && cropHeight > 0;
    }

    static CaptureFrameGeometry FromCrop(
        int cropX_,
        int cropY_,
        int cropWidth_,
        int cropHeight_,
        int modelWidth_,
        int modelHeight_,
        int screenWidth_,
        int screenHeight_,
        bool validScreenMapping_ = true,
        uint64_t generation_ = 0)
    {
        CaptureFrameGeometry geometry;
        geometry.modelWidth = std::max(1, modelWidth_);
        geometry.modelHeight = std::max(1, modelHeight_);
        geometry.cropX = cropX_;
        geometry.cropY = cropY_;
        geometry.cropWidth = std::max(1, cropWidth_);
        geometry.cropHeight = std::max(1, cropHeight_);
        geometry.screenWidth = std::max(1, screenWidth_);
        geometry.screenHeight = std::max(1, screenHeight_);
        geometry.validScreenMapping = validScreenMapping_;
        geometry.generation = generation_;
        return geometry;
    }

    static CaptureFrameGeometry FromCenterCrop(
        int screenWidth_,
        int screenHeight_,
        int cropWidth_,
        int cropHeight_,
        int modelWidth_,
        int modelHeight_,
        bool validScreenMapping_ = true,
        uint64_t generation_ = 0)
    {
        const int clampedScreenWidth = std::max(1, screenWidth_);
        const int clampedScreenHeight = std::max(1, screenHeight_);
        const int clampedCropWidth = std::max(1, std::min(cropWidth_, clampedScreenWidth));
        const int clampedCropHeight = std::max(1, std::min(cropHeight_, clampedScreenHeight));
        return FromCrop(
            std::max(0, (clampedScreenWidth - clampedCropWidth) / 2),
            std::max(0, (clampedScreenHeight - clampedCropHeight) / 2),
            clampedCropWidth,
            clampedCropHeight,
            modelWidth_,
            modelHeight_,
            clampedScreenWidth,
            clampedScreenHeight,
            validScreenMapping_,
            generation_);
    }
};

class CaptureToScreenConverter
{
public:
    explicit CaptureToScreenConverter(const CaptureFrameGeometry& frameGeometry)
        : geometry(frameGeometry)
    {
    }

    bool isUsable() const
    {
        return geometry.hasValidModel();
    }

    double scaleX() const
    {
        return isUsable()
            ? static_cast<double>(geometry.cropWidth) / static_cast<double>(geometry.modelWidth)
            : 1.0;
    }

    double scaleY() const
    {
        return isUsable()
            ? static_cast<double>(geometry.cropHeight) / static_cast<double>(geometry.modelHeight)
            : 1.0;
    }

    double averageScale() const
    {
        return (scaleX() + scaleY()) * 0.5;
    }

    cv::Point2d modelToScreenPoint(double x, double y) const
    {
        return {
            geometry.cropX + x * scaleX(),
            geometry.cropY + y * scaleY()
        };
    }

    cv::Point2d screenToModelPoint(double x, double y) const
    {
        const double sx = scaleX();
        const double sy = scaleY();
        return {
            sx > 1e-9 ? (x - geometry.cropX) / sx : 0.0,
            sy > 1e-9 ? (y - geometry.cropY) / sy : 0.0
        };
    }

    cv::Rect2d modelToScreenRect(const cv::Rect& rect) const
    {
        const cv::Point2d p1 = modelToScreenPoint(rect.x, rect.y);
        const cv::Point2d p2 = modelToScreenPoint(rect.x + rect.width, rect.y + rect.height);
        return { p1.x, p1.y, p2.x - p1.x, p2.y - p1.y };
    }

    bool isInsideModel(double x, double y) const
    {
        return x >= 0.0 && y >= 0.0 &&
            x < static_cast<double>(geometry.modelWidth) &&
            y < static_cast<double>(geometry.modelHeight);
    }

    bool isInsideScreenCrop(double x, double y) const
    {
        return x >= static_cast<double>(geometry.cropX) &&
            y >= static_cast<double>(geometry.cropY) &&
            x < static_cast<double>(geometry.cropX + geometry.cropWidth) &&
            y < static_cast<double>(geometry.cropY + geometry.cropHeight);
    }

    const CaptureFrameGeometry& frameGeometry() const
    {
        return geometry;
    }

private:
    CaptureFrameGeometry geometry;
};
