from __future__ import annotations

import argparse
import os
import unittest

from vision_snapshot_processor.main import (
    normalize_camera_source_for_capture,
    parse_camera_source,
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


if __name__ == "__main__":
    unittest.main()
