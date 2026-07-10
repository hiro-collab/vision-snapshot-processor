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
    MSG_TYPE_ROOM_LIGHT_OBSERVATION,
    ROOM_LIGHT_OBSERVATION_TOPIC,
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
    def test_requires_two_frames_before_observation(self) -> None:
        processor = RoomLightSnapshotProcessor(min_frames=2, window_ms=1000)
        frame = np.zeros((48, 64, 3), dtype=np.uint8)

        self.assertIsNone(processor.observe(frame, frame_id=1, stamp=1.0))
        observation = processor.observe(frame, frame_id=2, stamp=1.5)

        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(observation.frame_count, 2)
        self.assertGreaterEqual(observation.temporal_window_ms, 500)

    def test_dark_frames_produce_dark_observation_with_confidence(self) -> None:
        processor = RoomLightSnapshotProcessor(min_frames=2, window_ms=1000)
        frame = np.full((48, 64, 3), 2, dtype=np.uint8)

        processor.observe(frame, frame_id=1, stamp=1.0)
        observation = processor.observe(frame, frame_id=2, stamp=1.5)

        assert observation is not None
        payload = observation.to_payload()
        self.assertEqual(payload["observation_bucket"], "dark")
        self.assertGreater(payload["cue_likelihoods"]["darkness"], 0.5)
        self.assertGreater(payload["confidence"], 0.5)
        self.assertEqual(payload["sequence"]["frame_count"], 2)

    def test_topic_json_wraps_room_light_payload(self) -> None:
        processor = RoomLightSnapshotProcessor(min_frames=2, window_ms=1000)
        frame = np.full((48, 64, 3), (20, 160, 230), dtype=np.uint8)
        processor.observe(frame, frame_id=10, stamp=10.0)
        observation = processor.observe(frame, frame_id=11, stamp=11.0)
        assert observation is not None

        envelope = json.loads(
            topic_json(
                ROOM_LIGHT_OBSERVATION_TOPIC,
                MSG_TYPE_ROOM_LIGHT_OBSERVATION,
                observation.to_payload(),
                sequence=11,
                stamp=11.0,
                frame_id="cam0",
            )
        )

        self.assertEqual(envelope["topic"], "/vision/room_light/observation")
        self.assertEqual(envelope["msg_type"], "vision_snapshot_processor/RoomLightObservation")
        self.assertEqual(envelope["header"]["frame_id"], "cam0")
        self.assertEqual(envelope["payload"]["type"], "room_light_observation")

    def test_warm_and_daylight_cues_raise_daylight_ambiguity(self) -> None:
        observation = room_light._classify(
            [
                _feature_frame(
                    frame_id=1,
                    stamp=1.0,
                    luma_mean=0.56,
                    luma_std=0.27,
                    dynamic_range=0.74,
                    saturation_mean=0.12,
                    warm_ratio=1.12,
                    blue_ratio=1.08,
                    lab_b_mean=0.04,
                    edge_density=0.50,
                    underexposed_fraction=0.002,
                    overexposed_fraction=0.09,
                ),
                _feature_frame(
                    frame_id=2,
                    stamp=1.5,
                    luma_mean=0.56,
                    luma_std=0.27,
                    dynamic_range=0.74,
                    saturation_mean=0.12,
                    warm_ratio=1.12,
                    blue_ratio=1.08,
                    lab_b_mean=0.04,
                    edge_density=0.50,
                    underexposed_fraction=0.002,
                    overexposed_fraction=0.09,
                ),
            ]
        )

        self.assertEqual(observation.observation_bucket, "bright")
        self.assertEqual(observation.daylight_ambiguity, "high")
        self.assertIn("warm_light", observation.cue_likelihoods)
        self.assertIn("daylight", observation.cue_likelihoods)

    def test_payload_keeps_review_safe_source_and_confidence_fields(self) -> None:
        observation = room_light._classify(
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

        payload = observation.to_payload()

        self.assertEqual(payload["model"]["name"], room_light.ROOM_LIGHT_MODEL_NAME)
        self.assertEqual(payload["type"], "room_light_observation")
        self.assertIn("observation_bucket", payload)
        self.assertIn("confidence", payload)
        self.assertIn("daylight_ambiguity", payload)
        self.assertEqual(set(payload["cue_likelihoods"]), {"warm_light", "daylight", "darkness"})
        self.assertEqual(payload["source"], "vision_snapshot_processor")
        self.assertEqual(payload["source_class"], "camera_environment_estimate")
        self.assertEqual(payload["proof_ceiling"], "camera_environment_estimate_only")
        self.assertEqual(
            payload["does_not_prove"],
            ["physical_room_light_state", "home_assistant_light_state"],
        )
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
                sample_id="vision.room_light.synthetic_warm",
                expected_observation_bucket="balanced",
                max_frames=3,
            ),
        )

        self.assertEqual(summary["type"], "room_light_local_media_evaluation")
        self.assertEqual(summary["sample_id"], "vision.room_light.synthetic_warm")
        self.assertEqual(summary["result"], "pass")
        self.assertEqual(summary["frames_seen"], 3)
        self.assertEqual(summary["frames_sampled"], 3)
        self.assertFalse(summary["raw_media_included"])
        self.assertFalse(summary["raw_frame_included"])
        self.assertFalse(summary["source_path_included"])
        self.assertIn("final", summary)
        self.assertIn("observation_bucket_counts", summary)
        self.assertNotIn("input", summary)
        self.assertNotIn("source_path", summary)
        self.assertIn(
            "room_light_observation_not_physical_switch_proof",
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
