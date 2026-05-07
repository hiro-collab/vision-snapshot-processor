from __future__ import annotations

import json
import unittest

import numpy as np

from vision_snapshot_processor.processors.room_light import RoomLightSnapshotProcessor
from vision_snapshot_processor.topics import (
    MSG_TYPE_ROOM_LIGHT_STATE,
    ROOM_LIGHT_STATE_TOPIC,
    topic_json,
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


if __name__ == "__main__":
    unittest.main()
