from __future__ import annotations

import unittest


class VisualMotionAnalyzerShimTest(unittest.TestCase):
    def test_vsp_visual_motion_analyzer_is_deprecated_delegation_shim(self) -> None:
        from vision_snapshot_processor import visual_motion_analyzer as shim
        from self_mirror_visual_analyzer import visual_motion_analyzer as runtime

        self.assertIs(shim.analyze_frames, runtime.analyze_frames)
        self.assertIs(shim.analyze_config, runtime.analyze_config)
        self.assertIs(shim.write_outputs, runtime.write_outputs)
        self.assertIs(shim.main, runtime.main)


if __name__ == "__main__":
    unittest.main()
