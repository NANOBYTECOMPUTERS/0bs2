#include "capture/capture_geometry.h"
#include "test_harness.h"

#include <vector>

namespace
{
void testCenterCropClampsAndCenters()
{
    const CaptureFrameGeometry centered = CaptureFrameGeometry::FromCenterCrop(
        1920,
        1080,
        640,
        640,
        320,
        320,
        true,
        42);

    REQUIRE(centered.hasValidModel());
    REQUIRE(centered.cropX == 640);
    REQUIRE(centered.cropY == 220);
    REQUIRE(centered.cropWidth == 640);
    REQUIRE(centered.cropHeight == 640);
    REQUIRE(centered.screenWidth == 1920);
    REQUIRE(centered.screenHeight == 1080);
    REQUIRE(centered.validScreenMapping);
    REQUIRE(centered.generation == 42);

    const CaptureFrameGeometry clamped = CaptureFrameGeometry::FromCenterCrop(
        100,
        80,
        640,
        640,
        320,
        320);

    REQUIRE(clamped.cropX == 0);
    REQUIRE(clamped.cropY == 0);
    REQUIRE(clamped.cropWidth == 100);
    REQUIRE(clamped.cropHeight == 80);
    REQUIRE(clamped.screenWidth == 100);
    REQUIRE(clamped.screenHeight == 80);
}

void testModelScreenRoundTrip()
{
    const CaptureFrameGeometry geometry = CaptureFrameGeometry::FromCrop(
        100,
        50,
        800,
        600,
        400,
        300,
        1920,
        1080);
    const CaptureToScreenConverter converter(geometry);

    REQUIRE(converter.isUsable());
    REQUIRE_NEAR(converter.scaleX(), 2.0, 1e-9);
    REQUIRE_NEAR(converter.scaleY(), 2.0, 1e-9);
    REQUIRE_NEAR(converter.averageScale(), 2.0, 1e-9);

    const cv::Point2d screenPoint = converter.modelToScreenPoint(125.0, 75.0);
    REQUIRE_NEAR(screenPoint.x, 350.0, 1e-9);
    REQUIRE_NEAR(screenPoint.y, 200.0, 1e-9);

    const cv::Point2d modelPoint = converter.screenToModelPoint(screenPoint.x, screenPoint.y);
    REQUIRE_NEAR(modelPoint.x, 125.0, 1e-9);
    REQUIRE_NEAR(modelPoint.y, 75.0, 1e-9);

    const cv::Rect2d screenRect = converter.modelToScreenRect(cv::Rect(10, 20, 30, 40));
    REQUIRE_NEAR(screenRect.x, 120.0, 1e-9);
    REQUIRE_NEAR(screenRect.y, 90.0, 1e-9);
    REQUIRE_NEAR(screenRect.width, 60.0, 1e-9);
    REQUIRE_NEAR(screenRect.height, 80.0, 1e-9);
}

void testBoundaryChecksUseExclusiveFarEdges()
{
    const CaptureFrameGeometry geometry = CaptureFrameGeometry::FromCrop(
        10,
        20,
        100,
        80,
        50,
        40,
        800,
        600);
    const CaptureToScreenConverter converter(geometry);

    REQUIRE(converter.isInsideModel(0.0, 0.0));
    REQUIRE(converter.isInsideModel(49.999, 39.999));
    REQUIRE(!converter.isInsideModel(-0.001, 0.0));
    REQUIRE(!converter.isInsideModel(50.0, 20.0));
    REQUIRE(!converter.isInsideModel(20.0, 40.0));

    REQUIRE(converter.isInsideScreenCrop(10.0, 20.0));
    REQUIRE(converter.isInsideScreenCrop(109.999, 99.999));
    REQUIRE(!converter.isInsideScreenCrop(9.999, 20.0));
    REQUIRE(!converter.isInsideScreenCrop(110.0, 40.0));
    REQUIRE(!converter.isInsideScreenCrop(40.0, 100.0));
}

void testDefaultGeometryFallsBackToUnitScale()
{
    const CaptureFrameGeometry geometry;
    const CaptureToScreenConverter converter(geometry);

    REQUIRE(!geometry.hasValidModel());
    REQUIRE(!converter.isUsable());
    REQUIRE_NEAR(converter.scaleX(), 1.0, 1e-9);
    REQUIRE_NEAR(converter.scaleY(), 1.0, 1e-9);
    REQUIRE_NEAR(converter.averageScale(), 1.0, 1e-9);
}
}

int main()
{
    return obs2test::runTests(
        {
            { "center crop clamps and centers", testCenterCropClampsAndCenters },
            { "model screen round trip", testModelScreenRoundTrip },
            { "boundary checks use exclusive far edges", testBoundaryChecksUseExclusiveFarEdges },
            { "default geometry falls back to unit scale", testDefaultGeometryFallsBackToUnitScale },
        },
        "capture geometry");
}
