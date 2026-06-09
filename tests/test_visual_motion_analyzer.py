from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from vision_snapshot_processor.visual_motion_analyzer import analyze_frames, main


WINDOWS = [
    {"window_id": "active", "start_ms": 0, "end_ms": 300},
    {"window_id": "release", "start_ms": 300, "end_ms": 500},
    {"window_id": "settle", "start_ms": 500, "end_ms": 800},
]

ROIS = [
    {
        "roi_id": "avatar_face_head",
        "kind": "avatar",
        "counts_as_avatar_motion": True,
        "expected_for_pass": True,
        "rect_norm": {"x": 0.0, "y": 0.0, "w": 0.5, "h": 1.0},
    },
    {
        "roi_id": "speech_bubble",
        "kind": "guard_ui",
        "counts_as_avatar_motion": False,
        "expected_for_pass": False,
        "rect_norm": {"x": 0.5, "y": 0.0, "w": 0.5, "h": 1.0},
    },
]


class VisualMotionAnalyzerTest(unittest.TestCase):
    def test_expected_avatar_roi_motion_passes_without_counting_guard_roi(self) -> None:
        frames = [_frame() for _ in range(6)]
        frames[1][10:26, 10:22] = 255
        frames[2][10:26, 12:24] = 255

        summary, rows = analyze_frames(
            frames,
            analysis_run_id="vismot_run_test_avatar_001",
            scenario_id="rr003.visible_motion.smile.no_live.v0",
            motion_event_id="mot_evt_test_avatar_001",
            stimulus_instance_id="mot_inst_test_avatar_001",
            driver_result_id="mot_drv_test_avatar_001",
            sample_rate_fps=10,
            windows=WINDOWS,
            rois=ROIS,
        )

        self.assertEqual(summary["result"], "visual-pass")
        avatar = _roi(summary, "avatar_face_head")
        guard = _roi(summary, "speech_bubble")
        self.assertEqual(avatar["pass_label"], "visual-motion-detected")
        self.assertEqual(guard["pass_label"], "guard-ui-motion-excluded")
        self.assertTrue(all(not row["counts_as_avatar_motion"] for row in rows if row["roi_id"] == "speech_bubble"))

    def test_guard_only_motion_is_not_avatar_motion(self) -> None:
        frames = [_frame() for _ in range(6)]
        frames[1][10:26, 44:56] = 255
        frames[2][10:26, 46:58] = 255

        summary, _rows = analyze_frames(
            frames,
            analysis_run_id="vismot_run_test_guard_001",
            scenario_id="rr003.visible_motion.smile.no_live.v0",
            motion_event_id="mot_evt_test_guard_001",
            stimulus_instance_id="mot_inst_test_guard_001",
            driver_result_id="mot_drv_test_guard_001",
            sample_rate_fps=10,
            windows=WINDOWS,
            rois=ROIS,
        )

        self.assertEqual(summary["result"], "ui-only-motion-not-avatar-motion")
        avatar = _roi(summary, "avatar_face_head")
        self.assertEqual(avatar["pass_label"], "visual-missing-motion")

    def test_one_active_sample_does_not_pass_min_consecutive_threshold(self) -> None:
        windows = [
            {"window_id": "active", "start_ms": 0, "end_ms": 150},
            {"window_id": "release", "start_ms": 150, "end_ms": 300},
            {"window_id": "settle", "start_ms": 300, "end_ms": 500},
        ]
        frames = [_frame() for _ in range(5)]
        frames[1][10:26, 10:22] = 255

        summary, _rows = analyze_frames(
            frames,
            analysis_run_id="vismot_run_test_single_spike_001",
            scenario_id="rr003.visible_motion.smile.no_live.v0",
            motion_event_id="mot_evt_test_single_spike_001",
            stimulus_instance_id="mot_inst_test_single_spike_001",
            driver_result_id="mot_drv_test_single_spike_001",
            sample_rate_fps=10,
            windows=windows,
            rois=ROIS,
        )

        self.assertEqual(summary["result"], "visual-missing-motion")
        avatar = _roi(summary, "avatar_face_head")
        self.assertEqual(avatar["pass_label"], "visual-missing-motion")

    def test_pretrigger_avatar_motion_is_flagged(self) -> None:
        windows = [
            {"window_id": "pretrigger", "start_ms": 0, "end_ms": 200},
            {"window_id": "active", "start_ms": 200, "end_ms": 500},
            {"window_id": "release", "start_ms": 500, "end_ms": 700},
            {"window_id": "settle", "start_ms": 700, "end_ms": 900},
        ]
        frames = [_frame() for _ in range(9)]
        frames[1][10:26, 10:22] = 255

        summary, _rows = analyze_frames(
            frames,
            analysis_run_id="vismot_run_test_pretrigger_001",
            scenario_id="rr003.visible_motion.smile.no_live.v0",
            motion_event_id="mot_evt_test_pretrigger_001",
            stimulus_instance_id="mot_inst_test_pretrigger_001",
            driver_result_id="mot_drv_test_pretrigger_001",
            sample_rate_fps=10,
            windows=windows,
            rois=ROIS,
        )

        self.assertEqual(summary["result"], "visual-pretrigger-motion")
        avatar = _roi(summary, "avatar_face_head")
        self.assertEqual(avatar["pass_label"], "visual-pretrigger-motion")
        self.assertGreater(avatar["pretrigger_peak_motion_score"], 0)

    def test_source_ref_id_is_redacted_when_path_like_value_is_supplied(self) -> None:
        frames = [_frame() for _ in range(4)]

        summary, _rows = analyze_frames(
            frames,
            analysis_run_id="vismot_run_test_redaction_001",
            scenario_id="rr003.visible_motion.smile.no_live.v0",
            motion_event_id="mot_evt_test_redaction_001",
            stimulus_instance_id="mot_inst_test_redaction_001",
            driver_result_id="mot_drv_test_redaction_001",
            sample_rate_fps=10,
            windows=WINDOWS,
            rois=ROIS,
            source_ref_id="private/source/frame001.png",
        )

        source_ref = summary["source_ref"]
        self.assertTrue(source_ref["source_ref_id"].startswith("redacted_source_"))
        self.assertNotIn("/", source_ref["source_ref_id"])

    def test_browser_frame_provider_source_kind_is_preserved(self) -> None:
        frames = [_frame() for _ in range(4)]

        summary, _rows = analyze_frames(
            frames,
            analysis_run_id="vismot_run_test_browser_kind_001",
            scenario_id="rr003.visible_motion.browser.self_mirror.v0",
            motion_event_id="mot_evt_test_browser_kind_001",
            stimulus_instance_id="mot_inst_test_browser_kind_001",
            driver_result_id="mot_drv_test_browser_kind_001",
            sample_rate_fps=10,
            windows=WINDOWS,
            rois=ROIS,
            source_ref_id="redacted_browser_projection_visual_001",
            source_ref_kind="browser_frame_provider",
            proof_layer="visible_motion",
        )

        self.assertEqual(summary["source_ref"]["kind"], "browser_frame_provider")
        self.assertEqual(summary["proof_layer"], "visible_motion")

    def test_synthetic_self_mirror_fixture_does_not_need_frame_paths(self) -> None:
        synthetic_rois = [
            *ROIS,
            {
                "roi_id": "avatar_torso",
                "kind": "avatar",
                "counts_as_avatar_motion": True,
                "expected_for_pass": False,
                "rect_norm": {"x": 0.20, "y": 0.0, "w": 0.30, "h": 1.0},
            },
        ]
        config = {
            "analysis_run_id": "vismot_run_test_synthetic_self_mirror_001",
            "scenario_id": "rr003.visible_motion.self_mirror.synthetic.v0",
            "motion_event_id": "mot_evt_test_synthetic_self_mirror_001",
            "stimulus_instance_id": "mot_inst_test_synthetic_self_mirror_001",
            "driver_result_id": "mot_drv_test_synthetic_self_mirror_001",
            "proof_layer": "no_live_runtime",
            "source_ref": {
                "kind": "synthetic_test_frames",
                "source_ref_id": "redacted_synthetic_self_mirror_001",
            },
            "sampling": {"sample_rate_fps": 8},
            "synthetic_fixture": {
                "width": 320,
                "height": 180,
                "frame_count": 24,
                "avatar_motion_roi_ids": ["avatar_face_head"],
                "guard_motion_roi_ids": ["speech_bubble"],
            },
            "windows": [
                {"window_id": "pretrigger", "start_ms": 0, "end_ms": 500},
                {"window_id": "active", "start_ms": 500, "end_ms": 1800},
                {"window_id": "release", "start_ms": 1800, "end_ms": 2400},
                {"window_id": "settle", "start_ms": 2400, "end_ms": 3000},
            ],
            "rois": synthetic_rois,
            "thresholds": {
                "active_motion_min_score": 0.08,
                "settle_motion_max_score": 0.05,
                "min_consecutive_samples": 2,
            },
        }

        from vision_snapshot_processor.visual_motion_analyzer import analyze_config

        summary, rows = analyze_config(config)

        self.assertEqual(summary["source_ref"]["kind"], "synthetic_test_frames")
        self.assertEqual(summary["source_ref"]["source_ref_id"], "redacted_synthetic_self_mirror_001")
        self.assertEqual(summary["result"], "visual-pass")
        self.assertEqual(_roi(summary, "avatar_face_head")["pass_label"], "visual-motion-detected")
        self.assertEqual(_roi(summary, "avatar_torso")["pass_label"], "avatar-motion-not-required")
        self.assertEqual(_roi(summary, "speech_bubble")["pass_label"], "guard-ui-motion-excluded")
        self.assertEqual(len(rows), (24 - 1) * len(synthetic_rois))

    def test_cli_json_uses_artifact_basenames_not_local_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            frame_paths = []
            for index in range(4):
                frame = _frame()
                if index in (1, 2):
                    frame[10:26, 10:22] = 255
                path = root / f"frame_{index}.png"
                self.assertTrue(cv2.imwrite(str(path), frame))
                frame_paths.append(str(path))

            config_path = root / "config.json"
            output_dir = root / "out"
            config = {
                "analysis_run_id": "vismot_run_test_cli_001",
                "scenario_id": "rr003.visible_motion.smile.no_live.v0",
                "motion_event_id": "mot_evt_test_cli_001",
                "stimulus_instance_id": "mot_inst_test_cli_001",
                "driver_result_id": "mot_drv_test_cli_001",
                "proof_layer": "no_live_runtime",
                "frame_paths": frame_paths,
                "source_ref": {"source_ref_id": str(root / "private_source")},
                "sampling": {"sample_rate_fps": 10},
                "windows": WINDOWS,
                "rois": ROIS,
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(
                    main(["--config", str(config_path), "--output-dir", str(output_dir), "--json"]),
                    0,
                )

            output = stdout.getvalue()
            payload = json.loads(output)
            self.assertNotIn(str(root), output)
            self.assertEqual(payload["summary_file"], "visual_motion_summary.json")
            self.assertEqual(payload["timeseries_file"], "visual_motion_roi_timeseries.csv")
            self.assertNotIn("summary", payload)
            self.assertNotIn("timeseries", payload)

            summary_path = output_dir / "visual_motion_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertTrue(summary["source_ref"]["source_ref_id"].startswith("redacted_source_"))


def _frame() -> np.ndarray:
    return np.zeros((64, 64, 3), dtype=np.uint8)


def _roi(summary: dict[str, object], roi_id: str) -> dict[str, object]:
    for row in summary["roi_results"]:  # type: ignore[index]
        if row["roi_id"] == roi_id:
            return row
    raise AssertionError(f"ROI not found: {roi_id}")


if __name__ == "__main__":
    unittest.main()
