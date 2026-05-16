import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]


class RealtimeDetectionContractTest(unittest.TestCase):
    def read(self, relative_path):
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    def test_detection_buffer_publishes_freshness_metadata_with_move_swaps(self):
        buffer_h = self.read("detector/detection_buffer.h")

        self.assertIn("uint64_t frameId", buffer_h)
        self.assertIn("captureTimestamp", buffer_h)
        self.assertIn("publishTimestamp", buffer_h)
        self.assertIn("void publish(", buffer_h)
        self.assertIn("std::vector<cv::Rect>&& newBoxes", buffer_h)
        self.assertIn("boxes = std::move(newBoxes)", buffer_h)
        self.assertIn("cv.notify_all();", buffer_h)

    def test_capture_and_consumers_use_generation_counter_for_resolution_changes(self):
        main_cpp = self.read("0BS_box_2.cpp")
        capture_cpp = self.read("capture/capture.cpp")
        mouse_loop = self.read("runtime/mouse_thread_loop.cpp")

        self.assertIn("std::atomic<unsigned long long> detection_resolution_generation", main_cpp)
        self.assertIn("detection_resolution_generation.fetch_add", main_cpp)
        self.assertIn("seenDetectionResolutionGeneration", capture_cpp)
        self.assertIn("detection_resolution_generation.load", capture_cpp)
        self.assertNotIn("detection_resolution_changed.exchange(false)", capture_cpp)
        self.assertIn("seenDetectionResolutionGeneration", mouse_loop)
        self.assertIn("detection_resolution_generation.load", mouse_loop)
        self.assertNotIn("detection_resolution_changed.store(false)", mouse_loop)

    def test_detector_frame_handoff_carries_frame_id_and_capture_timestamp(self):
        dml_h = self.read("detector/dml_detector.h")
        dml_cpp = self.read("detector/dml_detector.cpp")
        capture_cpp = self.read("capture/capture.cpp")
        mouse_loop = self.read("runtime/mouse_thread_loop.cpp")

        self.assertIn("processFrame(const cv::Mat& frame, uint64_t frameId", dml_h)
        self.assertIn("currentFrameCaptureTimestamp", dml_h)
        self.assertIn("frameCaptureTimestamp", capture_cpp)
        self.assertIn("detectorFrameId", capture_cpp)
        self.assertIn("processFrame(detectionFrame, detectorFrameId", capture_cpp)
        self.assertIn("detectionBuffer.publish(", dml_cpp)
        self.assertIn("detectionBuffer.captureTimestamp", mouse_loop)
        self.assertIn("maxDetectionAgeMs", mouse_loop)

    def test_detectors_publish_newest_completed_frame_without_submission_starvation(self):
        dml_cpp = self.read("detector/dml_detector.cpp")
        dml_h = self.read("detector/dml_detector.h")
        trt_cpp = self.read("detector/trt_detector.cpp")
        trt_h = self.read("detector/trt_detector.h")

        self.assertNotIn("latestSubmittedFrameId", dml_h)
        self.assertNotIn("latestSubmittedFrameId", dml_cpp)
        self.assertNotIn("frameId < latestSubmittedFrameId.load", dml_cpp)
        self.assertNotIn("latestSubmittedFrameId", trt_h)
        self.assertNotIn("latestSubmittedFrameId", trt_cpp)
        self.assertNotIn("frameId < latestSubmittedFrameId.load", trt_cpp)

    def test_dml_preprocess_reuses_blob_and_skips_redundant_resize(self):
        dml_h = self.read("detector/dml_detector.h")
        dml_cpp = self.read("detector/dml_detector.cpp")

        self.assertIn("cv::Mat inputBlob", dml_h)
        self.assertIn("cv::Mat resizedFrame", dml_h)
        self.assertIn("cachedModelHeight", dml_h)
        self.assertIn("session_options.DisableMemPattern()", dml_cpp)
        self.assertIn("cv::dnn::blobFromImage", dml_cpp)
        self.assertIn("bgrFrame.cols != target_w || bgrFrame.rows != target_h", dml_cpp)
        self.assertNotIn("for (int h = 0; h < target_h", dml_cpp)
        self.assertNotIn("input_tensor_values(", dml_cpp)

    def test_postprocess_uses_max_detection_caps_and_class_aware_nms(self):
        post_cpp = self.read("detector/postProcess.cpp")

        self.assertIn("config.max_detections", post_cpp)
        self.assertIn("limitDetectionsForNms", post_cpp)
        self.assertIn("detections[j].classId != detections[i].classId", post_cpp)
        self.assertIn("std::nth_element", post_cpp)
        self.assertIn("output[(4 + c) * cols_i + i]", post_cpp)
        self.assertNotIn("cv::minMaxLoc(classes_scores", post_cpp)


if __name__ == "__main__":
    unittest.main()
