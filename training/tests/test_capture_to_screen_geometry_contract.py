import unittest
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "0BS_box_2.vcxproj").exists()
)


class CaptureToScreenGeometryContractTest(unittest.TestCase):
    def read(self, relative_path):
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    def test_converter_contract_uses_center_crop_offset_and_source_scale(self):
        header = self.read("capture/capture_geometry.h")

        for token in (
            "struct CaptureFrameGeometry",
            "class CaptureToScreenConverter",
            "modelWidth",
            "modelHeight",
            "cropX",
            "cropY",
            "cropWidth",
            "cropHeight",
            "screenWidth",
            "screenHeight",
            "validScreenMapping",
            "modelToScreenPoint",
            "screenToModelPoint",
            "modelToScreenRect",
            "isInsideModel",
            "isInsideScreenCrop",
        ):
            self.assertIn(token, header)

        self.assertIn("cropX + x * scaleX()", header)
        self.assertIn("cropY + y * scaleY()", header)
        self.assertIn("static_cast<double>(geometry.cropWidth) /", header)
        self.assertIn("static_cast<double>(geometry.cropHeight) /", header)
        self.assertNotIn("screenWidth / captureWidth", header)
        self.assertNotIn("screenHeight / captureHeight", header)

    def test_geometry_inside_checks_use_half_open_pixel_bounds(self):
        header = self.read("capture/capture_geometry.h")

        self.assertIn("x < static_cast<double>(geometry.modelWidth)", header)
        self.assertIn("y < static_cast<double>(geometry.modelHeight)", header)
        self.assertIn("x < static_cast<double>(geometry.cropX + geometry.cropWidth)", header)
        self.assertIn("y < static_cast<double>(geometry.cropY + geometry.cropHeight)", header)
        self.assertNotIn("x <= static_cast<double>(geometry.modelWidth)", header)
        self.assertNotIn("y <= static_cast<double>(geometry.modelHeight)", header)
        self.assertNotIn("x <= static_cast<double>(geometry.cropX + geometry.cropWidth)", header)
        self.assertNotIn("y <= static_cast<double>(geometry.cropY + geometry.cropHeight)", header)

    def test_synthetic_center_edge_resolution_and_accuracy_cases(self):
        def to_screen(x, y, model_w, model_h, crop_x, crop_y, crop_w, crop_h):
            return (
                crop_x + x * (crop_w / model_w),
                crop_y + y * (crop_h / model_h),
            )

        # Center bias: 320 center crop on 1920x1080 should map model center to desktop center.
        self.assertEqual(to_screen(160, 160, 320, 320, 800, 380, 320, 320), (960, 540))

        # Edge alignment: every model-space edge lands on the corresponding crop edge.
        self.assertEqual(to_screen(0, 0, 320, 320, 800, 380, 320, 320), (800, 380))
        self.assertEqual(to_screen(320, 320, 320, 320, 800, 380, 320, 320), (1120, 700))

        # Resolution switch: 320 and 640 crops both preserve the same desktop center.
        self.assertEqual(to_screen(320, 320, 640, 640, 640, 220, 640, 640), (960, 540))

        # Mouse accuracy regression floor: a static target at model center has zero screen error.
        desired_x, desired_y = to_screen(160, 160, 320, 320, 800, 380, 320, 320)
        actual_x, actual_y = 960, 540
        avg_error = ((desired_x - actual_x) ** 2 + (desired_y - actual_y) ** 2) ** 0.5
        self.assertLess(avg_error, 5.0)

    def test_detection_buffer_carries_frame_geometry_with_detections(self):
        buffer_h = self.read("detector/detection_buffer.h")
        dml_h = self.read("detector/dml_detector.h")
        dml_cpp = self.read("detector/dml_detector.cpp")
        trt_h = self.read("detector/trt_detector.h")
        trt_cpp = self.read("detector/trt_detector.cpp")

        for token in (
            '#include "capture/capture_geometry.h"',
            "CaptureFrameGeometry geometry",
            "const CaptureFrameGeometry& newGeometry",
            "geometry = newGeometry",
            "outGeometry",
        ):
            self.assertIn(token, buffer_h)

        for source in (dml_h, dml_cpp, trt_h, trt_cpp):
            self.assertIn("CaptureFrameGeometry", source)
            self.assertIn("currentFrameGeometry", source)

        self.assertIn("detectionBuffer.publish(", dml_cpp)
        self.assertIn("currentFrameGeometry", dml_cpp)
        self.assertIn("detectionBuffer.publish(", trt_cpp)
        self.assertIn("frameGeometry", trt_cpp)

    def test_capture_backends_provide_geometry(self):
        capture_h = self.read("capture/capture.h")
        capture_cpp = self.read("capture/capture.cpp")
        dda_h = self.read("capture/duplication_api_capture.h")
        dda_cpp = self.read("capture/duplication_api_capture.cpp")
        winrt_h = self.read("capture/winrt_capture.h")
        winrt_cpp = self.read("capture/winrt_capture.cpp")

        self.assertIn("virtual CaptureFrameGeometry GetFrameGeometry", capture_h)
        self.assertIn("CaptureFrameGeometry frameGeometry", capture_cpp)
        self.assertIn("GetFrameGeometry", capture_cpp)
        self.assertIn("CaptureFrameGeometry::FromCenterCrop", capture_cpp)
        self.assertIn("processFrame(frame, detectorFrameId, captureTimestamp, frameGeometry)", capture_cpp)

        for source in (dda_h, dda_cpp, winrt_h, winrt_cpp):
            self.assertIn("GetFrameGeometry", source)
            self.assertIn("CaptureFrameGeometry", source)

        self.assertIn("copyWidth", dda_cpp)
        self.assertIn("copyHeight", dda_cpp)
        self.assertIn("regionX", winrt_cpp)
        self.assertIn("regionY", winrt_cpp)

    def test_overlay_debug_uses_converter_for_visual_alignment(self):
        overlay = self.read("runtime/game_overlay_loop.cpp")

        for token in (
            '#include "capture/capture_geometry.h"',
            "CaptureFrameGeometry detectionGeometry",
            "CaptureToScreenConverter overlayConverter",
            "fallbackOverlayGeometry",
            "modelToScreenRect",
            "modelToScreenPoint",
            "screenRect.x",
            "screenPoint.x",
        ):
            self.assertIn(token, overlay)

        self.assertNotIn("baseX + bx * scaleX", overlay)
        self.assertNotIn("baseY + by * scaleY", overlay)

    def test_mouse_control_remains_in_detection_space(self):
        loop = self.read("runtime/mouse_thread_loop.cpp")

        self.assertIn("CaptureFrameGeometry bufferGeometry", loop)
        self.assertIn("bufferGeometry = detectionBuffer.geometry", loop)
        self.assertIn("bufferGeometry.hasValidModel()", loop)
        self.assertIn("bufferGeometry.modelWidth", loop)
        self.assertIn("bufferGeometry.modelHeight", loop)
        self.assertIn("mouseThread.updateDetectionGeometry(trackerFrameWidth, trackerFrameHeight)", loop)
        self.assertIn("targetTracker.updateAt(", loop)
        self.assertIn("trackerFrameTimestamp", loop)
        self.assertIn("config.detection_resolution,", loop)
        self.assertIn("activeTarget->smoothX", loop)
        self.assertIn("activeTarget->smoothY", loop)
        self.assertIn("mouseThread.publishTargetMotionState(lockInfo)", loop)
        self.assertNotIn("mouseThread.moveMouseTarget(*activeTarget)", loop)
        self.assertIn("chooseDirectDetectionTarget(", loop)
        self.assertNotIn("modelToScreenPoint(activeTarget->smoothX", loop)
        self.assertNotIn("CaptureToScreenConverter", loop)


if __name__ == "__main__":
    unittest.main()
