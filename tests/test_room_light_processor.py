from __future__ import annotations

import json
import unittest

import numpy as np

from vision_snapshot_processor.room_light_evaluator import (
    RoomLightEvaluationConfig,
    evaluate_frames,
    summarize_observations,
)
from vision_snapshot_processor.processors.room_light import RoomLightSnapshotProcessor
from vision_snapshot_processor.processors import room_light
from vision_snapshot_processor.topics import (
    MSG_TYPE_ROOM_LIGHT_STATE,
    ROOM_LIGHT_STATE_TOPIC,
    topic_json,
)


def _feature_frame(
    *,
    frame_id: int,
    stamp: float,
    luma_mean: float,
    luma_std: float,
    dynamic_range: float,
    saturation_mean: float,
    warm_ratio: float,
    blue_ratio: float,
    lab_b_mean: float,
    edge_density: float,
    underexposed_fraction: float,
    overexposed_fraction: float,
) -> room_light.FrameFeatures:
    return room_light.FrameFeatures(
        frame_id=frame_id,
        stamp=stamp,
        luma_mean=luma_mean,
        luma_std=luma_std,
        luma_p10=0.0,
        luma_p90=dynamic_range,
        dynamic_range=dynamic_range,
        saturation_mean=saturation_mean,
        warm_ratio=warm_ratio,
        blue_ratio=blue_ratio,
        lab_b_mean=lab_b_mean,
        edge_density=edge_density,
        underexposed_fraction=underexposed_fraction,
        overexposed_fraction=overexposed_fraction,
        temporal_delta=0.0,
        gray_small=np.zeros((2, 2), dtype=np.float32),
    )


class RoomLightSnapshotProcessorTest(unittest.TestCase):
    def test_requires_two_frames_before_state(self) -> None:
        processor = RoomLightSnapshotProcessor(min_frames=2, window_ms=1000)
        frame = np.zeros((48, 64, 3), dtype=np.uint8)

        self.assertIsNone(processor.observe(frame, frame_id=1, stamp=1.0))
        state = processor.observe(frame, frame_id=2, stamp=1.5)

        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(state.frame_count, 2)
        self.assertGreaterEqual(state.temporal_window_ms, 500)

    def test_dark_frames_are_electric_off_with_confidence(self) -> None:
        processor = RoomLightSnapshotProcessor(min_frames=2, window_ms=1000)
        frame = np.full((48, 64, 3), 2, dtype=np.uint8)

        processor.observe(frame, frame_id=1, stamp=1.0)
        state = processor.observe(frame, frame_id=2, stamp=1.5)

        assert state is not None
        payload = state.to_payload()
        self.assertEqual(payload["state"], "off")
        self.assertEqual(payload["lighting_type"], "dark")
        self.assertGreater(payload["confidence"], 0.5)
        self.assertEqual(payload["sequence"]["frame_count"], 2)

    def test_topic_json_wraps_room_light_payload(self) -> None:
        processor = RoomLightSnapshotProcessor(min_frames=2, window_ms=1000)
        frame = np.full((48, 64, 3), (20, 160, 230), dtype=np.uint8)
        processor.observe(frame, frame_id=10, stamp=10.0)
        state = processor.observe(frame, frame_id=11, stamp=11.0)
        assert state is not None

        envelope = json.loads(
            topic_json(
                ROOM_LIGHT_STATE_TOPIC,
                MSG_TYPE_ROOM_LIGHT_STATE,
                state.to_payload(),
                sequence=11,
                stamp=11.0,
                frame_id="cam0",
            )
        )

        self.assertEqual(envelope["topic"], "/vision/room_light/state")
        self.assertEqual(envelope["msg_type"], "vision_snapshot_processor/RoomLightState")
        self.assertEqual(envelope["header"]["frame_id"], "cam0")
        self.assertEqual(envelope["payload"]["type"], "room_light_state")

    def test_daylight_switch_calibration_distinguishes_electric_light(self) -> None:
        on_state = room_light._daylight_electric_switch_state(
            {
                "luma_mean": 0.56,
                "dynamic_range": 0.74,
                "overexposed_fraction": 0.09,
                "underexposed_fraction": 0.002,
                "edge_density": 0.50,
            },
            electric_probability=0.62,
            daylight_probability=0.78,
            dark_probability=0.05,
        )
        off_state = room_light._daylight_electric_switch_state(
            {
                "luma_mean": 0.48,
                "dynamic_range": 0.60,
                "overexposed_fraction": 0.02,
                "underexposed_fraction": 0.04,
                "edge_density": 0.45,
            },
            electric_probability=0.59,
            daylight_probability=0.66,
            dark_probability=0.12,
        )

        self.assertIsNotNone(on_state)
        self.assertIsNotNone(off_state)
        assert on_state is not None
        assert off_state is not None
        self.assertEqual(on_state[0], "on")
        self.assertEqual(off_state[0], "off")

    def test_daylight_switch_calibration_updates_lighting_type(self) -> None:
        on_state = room_light._classify(
            [
                _feature_frame(
                    frame_id=1,
                    stamp=1.0,
                    luma_mean=0.55737,
                    luma_std=0.27418,
                    dynamic_range=0.75229,
                    saturation_mean=0.1178,
                    warm_ratio=1.01254,
                    blue_ratio=1.16307,
                    lab_b_mean=-0.00132,
                    edge_density=0.53244,
                    underexposed_fraction=0.00098,
                    overexposed_fraction=0.09462,
                ),
                _feature_frame(
                    frame_id=2,
                    stamp=1.5,
                    luma_mean=0.55737,
                    luma_std=0.27418,
                    dynamic_range=0.75229,
                    saturation_mean=0.1178,
                    warm_ratio=1.01254,
                    blue_ratio=1.16307,
                    lab_b_mean=-0.00132,
                    edge_density=0.53244,
                    underexposed_fraction=0.00098,
                    overexposed_fraction=0.09462,
                ),
            ]
        )
        off_state = room_light._classify(
            [
                _feature_frame(
                    frame_id=1,
                    stamp=1.0,
                    luma_mean=0.52855,
                    luma_std=0.23753,
                    dynamic_range=0.62294,
                    saturation_mean=0.13802,
                    warm_ratio=1.06429,
                    blue_ratio=1.21429,
                    lab_b_mean=-0.00318,
                    edge_density=0.46249,
                    underexposed_fraction=0.02685,
                    overexposed_fraction=0.02578,
                ),
                _feature_frame(
                    frame_id=2,
                    stamp=1.5,
                    luma_mean=0.52855,
                    luma_std=0.23753,
                    dynamic_range=0.62294,
                    saturation_mean=0.13802,
                    warm_ratio=1.06429,
                    blue_ratio=1.21429,
                    lab_b_mean=-0.00318,
                    edge_density=0.46249,
                    underexposed_fraction=0.02685,
                    overexposed_fraction=0.02578,
                ),
            ]
        )

        self.assertEqual(on_state.state, "on")
        self.assertEqual(on_state.lighting_type, "mixed")
        self.assertEqual(off_state.state, "off")
        self.assertEqual(off_state.lighting_type, "daylight")

    def test_payload_keeps_review_safe_source_and_confidence_fields(self) -> None:
        state = room_light._classify(
            [
                _feature_frame(
                    frame_id=3,
                    stamp=3.0,
                    luma_mean=0.55737,
                    luma_std=0.27418,
                    dynamic_range=0.75229,
                    saturation_mean=0.1178,
                    warm_ratio=1.01254,
                    blue_ratio=1.16307,
                    lab_b_mean=-0.00132,
                    edge_density=0.53244,
                    underexposed_fraction=0.00098,
                    overexposed_fraction=0.09462,
                ),
                _feature_frame(
                    frame_id=4,
                    stamp=3.5,
                    luma_mean=0.55737,
                    luma_std=0.27418,
                    dynamic_range=0.75229,
                    saturation_mean=0.1178,
                    warm_ratio=1.01254,
                    blue_ratio=1.16307,
                    lab_b_mean=-0.00132,
                    edge_density=0.53244,
                    underexposed_fraction=0.00098,
                    overexposed_fraction=0.09462,
                ),
            ]
        )

        payload = state.to_payload()

        self.assertEqual(payload["model"]["name"], room_light.ROOM_LIGHT_MODEL_NAME)
        self.assertIn("confidence", payload)
        self.assertIn("lighting_type", payload)
        self.assertIn("electric_light", payload)
        self.assertIn("daylight", payload)
        self.assertIn("probabilities", payload)
        self.assertIn("observation_id", payload)
        self.assertNotIn("frame", payload)
        self.assertNotIn("path", payload)
        self.assertNotIn("entity_id", payload)

    def test_redacted_evaluator_summarizes_local_media_without_source_path(self) -> None:
        frames = [
            np.full((48, 64, 3), (20, 160, 230), dtype=np.uint8),
            np.full((48, 64, 3), (20, 160, 230), dtype=np.uint8),
            np.full((48, 64, 3), (20, 160, 230), dtype=np.uint8),
        ]

        summary = evaluate_frames(
            frames,
            config=RoomLightEvaluationConfig(
                sample_id="vision.room_light.synthetic_on",
                expected_state="on",
                max_frames=3,
            ),
        )

        self.assertEqual(summary["type"], "room_light_local_media_evaluation")
        self.assertEqual(summary["sample_id"], "vision.room_light.synthetic_on")
        self.assertEqual(summary["result"], "pass")
        self.assertEqual(summary["frames_seen"], 3)
        self.assertEqual(summary["frames_sampled"], 3)
        self.assertFalse(summary["raw_media_included"])
        self.assertFalse(summary["raw_frame_included"])
        self.assertFalse(summary["source_path_included"])
        self.assertIn("final", summary)
        self.assertNotIn("input", summary)
        self.assertNotIn("source_path", summary)
        self.assertIn(
            "room_light_estimate_not_physical_switch_state",
            summary["non_claims"],
        )

    def test_redacted_evaluator_reports_missing_observations_as_not_evaluated(self) -> None:
        summary = summarize_observations(
            [],
            config=RoomLightEvaluationConfig(sample_id="empty"),
            frames_seen=0,
            frames_sampled=0,
        )

        self.assertEqual(summary["result"], "not_evaluated")
        self.assertEqual(summary["observations"], 0)
        self.assertNotIn("final", summary)


if __name__ == "__main__":
    unittest.main()
