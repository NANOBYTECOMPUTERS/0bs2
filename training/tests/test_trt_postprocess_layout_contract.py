import unittest
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "0BS_box_2.vcxproj").exists()
)


class TrtPostProcessLayoutContractTests(unittest.TestCase):
    def read_postprocess(self):
        return (REPO_ROOT / "detector" / "postProcess.cpp").read_text(encoding="utf-8")

    def test_nms_shaped_outputs_are_classified_before_score_layouts(self):
        post_cpp = self.read_postprocess()
        resolver_start = post_cpp.index("bool resolveYoloDecoderLayout(")
        resolver_end = post_cpp.index("#endif", resolver_start)
        resolver = post_cpp[resolver_start:resolver_end]

        cols_nms = resolver.index("if (cols == 6)")
        rows_nms = resolver.index("if (rows == 6 && looksLikeClassIdRow")
        explicit_score = resolver.index("if (explicitClassCount > 0)")

        self.assertLess(cols_nms, explicit_score)
        self.assertLess(rows_nms, explicit_score)
        self.assertIn("YoloDecoderLayout{ false, true, false, rows, cols", resolver)
        self.assertIn("YoloDecoderLayout{ true, true, false, cols, rows", resolver)

    def test_trt_detector_can_pass_positive_class_count_for_nms_shape(self):
        trt_cpp = (REPO_ROOT / "detector" / "trt_detector.cpp").read_text(encoding="utf-8")

        self.assertIn("const int64_t outChannels = outDims.d[1]", trt_cpp)
        self.assertIn("const int64_t classes64 = (outChannels > 4) ? (outChannels - 4) : 1", trt_cpp)


if __name__ == "__main__":
    unittest.main()
