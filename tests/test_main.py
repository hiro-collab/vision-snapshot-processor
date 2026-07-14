from __future__ import annotations

import argparse
import asyncio
import os
import unittest
from unittest import mock

import cv2

from vision_snapshot_processor.main import (
    CAPTURE_OPEN_TIMEOUT_MS,
    CAPTURE_READ_TIMEOUT_MS,
    camera_source_class,
    normalize_camera_source_for_capture,
    parse_camera_source,
    RecoveringVideoCapture,
)


class CameraSourceTest(unittest.TestCase):
    def test_parse_camera_source_accepts_file_url(self) -> None:
        source = "file:///C:/workspace/local/media/movie/light_on.mp4"

        self.assertEqual(parse_camera_source(source), source)

    def test_parse_camera_source_rejects_plain_path(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_camera_source("C:/workspace/local/media/movie/light_on.mp4")

    def test_normalize_file_url_to_local_path_for_capture(self) -> None:
        source = "file:///C:/Sword%20Agent/local/media/movie/light_on.mp4"

        normalized = normalize_camera_source_for_capture(source)

        self.assertNotIn("file:", normalized)
        self.assertIn("Sword Agent", normalized)
        self.assertTrue(
            normalized.endswith(os.path.join("local", "media", "movie", "light_on.mp4"))
            or normalized.endswith("local/media/movie/light_on.mp4")
        )

    def test_normalize_stream_url_is_unchanged(self) -> None:
        source = "rtsp://127.0.0.1:8554/cam0"

        self.assertEqual(normalize_camera_source_for_capture(source), source)

    def test_camera_source_class_never_echoes_credentials_tokens_or_paths(self) -> None:
        cases = {
            "rtsp://user:secret@127.0.0.1/live?token=private": "rtsp_stream",
            "https://camera.invalid/live?token=private": "http_stream",
            "file:///C:/private/person/sample.mp4": "local_file",
        }

        for source, expected in cases.items():
            with self.subTest(source=expected):
                result = camera_source_class(source)
                self.assertEqual(result, expected)
                self.assertNotIn("secret", result)
                self.assertNotIn("private", result)
                self.assertNotIn("sample", result)


class RecoveringVideoCaptureTest(unittest.TestCase):
    @staticmethod
    def _args() -> argparse.Namespace:
        return argparse.Namespace(
            camera_width=None,
            camera_height=None,
            camera_fps=None,
        )

    def test_initial_unavailable_source_reopens_without_substitution(self) -> None:
        class FakeCapture:
            def __init__(self, opened: bool, reads=()) -> None:
                self.opened = opened
                self.reads = iter(reads)
                self.released = False
                self.set_calls = []

            def isOpened(self):
                return self.opened

            def set(self, key, value):
                self.set_calls.append((key, value))
                return True

            def read(self):
                return next(self.reads, (False, None))

            def release(self):
                self.released = True

        unavailable = FakeCapture(False)
        recovered = FakeCapture(True, reads=[(True, "fresh-frame")])
        opened_sources = []

        def opener(source):
            opened_sources.append(source)
            return unavailable if len(opened_sources) == 1 else recovered

        capture = RecoveringVideoCapture(
            "rtsp://127.0.0.1:8554/cam0",
            self._args(),
            opener=opener,
        )

        self.assertEqual(capture.open_once(), (False, 0.25))
        self.assertTrue(unavailable.released)
        self.assertEqual(capture.open_once(), (True, 0.0))
        self.assertEqual(capture.read(), (True, "fresh-frame"))
        capture.close()

        self.assertEqual(
            opened_sources,
            ["rtsp://127.0.0.1:8554/cam0"] * 2,
        )
        self.assertTrue(recovered.released)

    def test_three_read_failures_release_and_reopen_same_source(self) -> None:
        class FakeCapture:
            def __init__(self, reads) -> None:
                self.reads = iter(reads)
                self.released = False

            def isOpened(self):
                return True

            def set(self, _key, _value):
                return True

            def read(self):
                return next(self.reads)

            def release(self):
                self.released = True

        disconnected = FakeCapture([(False, None)] * 3)
        recovered = FakeCapture([(True, "fresh-frame")])
        captures = iter((disconnected, recovered))
        capture = RecoveringVideoCapture(
            "rtsp://127.0.0.1:8554/cam0",
            self._args(),
            opener=lambda _source: next(captures),
        )

        self.assertEqual(capture.open_once(), (True, 0.0))
        self.assertEqual(capture.read(), (False, None))
        self.assertEqual(capture.read(), (False, None))
        self.assertEqual(capture.read(), (False, None))
        self.assertFalse(capture.is_opened)
        self.assertTrue(disconnected.released)
        self.assertEqual(capture.open_once(), (True, 0.0))
        self.assertEqual(capture.read(), (True, "fresh-frame"))
        capture.close()

        self.assertTrue(recovered.released)

    def test_open_exceptions_release_candidates_and_keep_bounded_retry(self) -> None:
        class FakeCapture:
            def __init__(self, *, configure_error=False, opened_error=False) -> None:
                self.configure_error = configure_error
                self.opened_error = opened_error
                self.released = False

            def isOpened(self):
                if self.opened_error:
                    raise RuntimeError("open check failed")
                return True

            def set(self, _key, _value):
                if self.configure_error:
                    raise RuntimeError("configuration failed")
                return True

            def release(self):
                self.released = True

        configured = FakeCapture(configure_error=True)
        checked = FakeCapture(opened_error=True)
        calls = iter((RuntimeError("open failed"), configured, checked))

        def opener(_source):
            value = next(calls)
            if isinstance(value, Exception):
                raise value
            return value

        capture = RecoveringVideoCapture(
            "rtsp://127.0.0.1:8554/cam0",
            self._args(),
            opener=opener,
        )

        self.assertEqual(capture.open_once(), (False, 0.25))
        self.assertEqual(capture.open_once(), (False, 0.5))
        self.assertTrue(configured.released)
        self.assertEqual(capture.open_once(), (False, 1.0))
        self.assertTrue(checked.released)
        self.assertFalse(capture.is_opened)

    def test_default_opener_sets_fixed_open_and_read_timeouts(self) -> None:
        candidate = mock.Mock()
        candidate.isOpened.return_value = True
        with mock.patch(
            "vision_snapshot_processor.main.cv2.VideoCapture",
            return_value=candidate,
        ) as opener:
            capture = RecoveringVideoCapture(
                "rtsp://127.0.0.1:8554/cam0",
                self._args(),
            )

            self.assertEqual(capture.open_once(), (True, 0.0))
            capture.close()

        opener.assert_called_once_with(
            "rtsp://127.0.0.1:8554/cam0",
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                CAPTURE_OPEN_TIMEOUT_MS,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                CAPTURE_READ_TIMEOUT_MS,
            ],
        )
        self.assertLessEqual(CAPTURE_OPEN_TIMEOUT_MS, 1000)
        self.assertLessEqual(CAPTURE_READ_TIMEOUT_MS, 1000)

    def test_failed_open_backoff_caps_and_resets_after_success(self) -> None:
        class FakeCapture:
            def __init__(self, opened: bool) -> None:
                self.opened = opened

            def isOpened(self):
                return self.opened

            def set(self, _key, _value):
                return True

            def release(self):
                pass

        candidates = iter(
            [FakeCapture(False) for _ in range(6)]
            + [FakeCapture(True), FakeCapture(False)]
        )
        capture = RecoveringVideoCapture(
            "rtsp://127.0.0.1:8554/cam0",
            self._args(),
            opener=lambda _source: next(candidates),
        )

        self.assertEqual(
            [capture.open_once()[1] for _ in range(6)],
            [0.25, 0.5, 1.0, 2.0, 3.0, 3.0],
        )
        self.assertEqual(capture.open_once(), (True, 0.0))
        self.assertEqual(capture.open_once(), (False, 0.25))


class RunLoopReconnectTest(unittest.IsolatedAsyncioTestCase):
    async def test_each_successful_capture_generation_resets_before_observe(self) -> None:
        events = []

        class StopRun(Exception):
            pass

        class FakeCapture:
            def __init__(self) -> None:
                self.opened = False
                self.generation = 0
                self.closed = False

            @property
            def is_opened(self):
                return self.opened

            def open_once(self):
                self.generation += 1
                self.opened = True
                events.append(f"open-{self.generation}")
                return True, 0.0

            def read(self):
                events.append(f"read-{self.generation}")
                if self.generation == 1:
                    self.opened = False
                return True, f"frame-{self.generation}"

            def close(self):
                self.closed = True

        class FakeProcessor:
            def reset(self):
                events.append("reset")

            def observe(self, frame, *, frame_id, stamp):
                events.append(f"observe-{frame}-{frame_id}")
                if frame_id == 2:
                    raise StopRun()
                return None

        class FakeBroadcaster:
            async def __aenter__(self):
                return self

            async def __aexit__(self, _exc_type, _exc, _traceback):
                return False

            async def publish(self, _payload):
                raise AssertionError("no observation should publish in this seam")

        args = argparse.Namespace(
            opencv_ffmpeg_capture_options="none",
            camera_source="rtsp://127.0.0.1:8554/cam0",
            camera_width=None,
            camera_height=None,
            camera_fps=None,
            processor=["room_light"],
            min_frames=2,
            window_ms=1000,
            resize_width=160,
            host="127.0.0.1",
            port=8776,
            max_clients=1,
            max_message_bytes=8192,
            sample_every=0.001,
            frame_id="cam0",
        )
        capture = FakeCapture()
        processor = FakeProcessor()
        loop = asyncio.get_running_loop()
        with (
            mock.patch(
                "vision_snapshot_processor.main.RecoveringVideoCapture",
                return_value=capture,
            ),
            mock.patch(
                "vision_snapshot_processor.main.RoomLightSnapshotProcessor",
                return_value=processor,
            ),
            mock.patch(
                "vision_snapshot_processor.main.WebSocketTopicBroadcaster",
                return_value=FakeBroadcaster(),
            ),
            mock.patch.object(
                loop,
                "add_signal_handler",
                side_effect=NotImplementedError,
            ),
            mock.patch("vision_snapshot_processor.main.signal.signal"),
        ):
            from vision_snapshot_processor.main import run

            with self.assertRaises(StopRun):
                await run(args)

        self.assertEqual(
            events,
            [
                "open-1",
                "reset",
                "read-1",
                "observe-frame-1-1",
                "open-2",
                "reset",
                "read-2",
                "observe-frame-2-2",
            ],
        )
        self.assertTrue(capture.closed)


if __name__ == "__main__":
    unittest.main()
