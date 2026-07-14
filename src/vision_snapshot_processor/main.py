from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
import time
from urllib.parse import unquote, urlsplit
from urllib.request import url2pathname

import cv2

from .processors.room_light import RoomLightSnapshotProcessor
from .topics import MSG_TYPE_ROOM_LIGHT_OBSERVATION, ROOM_LIGHT_OBSERVATION_TOPIC, topic_json
from .websocket import WebSocketTopicBroadcaster


DEFAULT_OPENCV_FFMPEG_CAPTURE_OPTIONS = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|reorder_queue_size;0"
)
CAPTURE_OPEN_TIMEOUT_MS = 1000
CAPTURE_READ_TIMEOUT_MS = 1000


def open_bounded_video_capture(value: str):
    return cv2.VideoCapture(
        value,
        cv2.CAP_FFMPEG,
        [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            CAPTURE_OPEN_TIMEOUT_MS,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            CAPTURE_READ_TIMEOUT_MS,
        ],
    )


def parse_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("--port must be between 1 and 65535")
    return port


def parse_positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def parse_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def parse_min_frames(value: str) -> int:
    parsed = parse_positive_int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("--min-frames must be 2 or greater")
    return parsed


def parse_camera_source(value: str) -> str:
    parsed = value.strip()
    if not parsed:
        raise argparse.ArgumentTypeError("--camera-source must not be empty")
    parts = urlsplit(parsed)
    if parts.scheme.lower() not in {"rtsp", "rtsps", "http", "https", "file"}:
        raise argparse.ArgumentTypeError(
            "--camera-source must be a stream/file URL such as rtsp://127.0.0.1:8554/cam0"
        )
    return parsed


def normalize_camera_source_for_capture(value: str) -> str:
    parts = urlsplit(value)
    if parts.scheme.lower() != "file":
        return value

    if parts.netloc and parts.netloc.lower() != "localhost":
        url_path = f"//{parts.netloc}{parts.path}"
    else:
        url_path = parts.path
    path = url2pathname(unquote(url_path))
    if len(path) >= 3 and path[0] in {"/", "\\"} and path[2] == ":":
        path = path[1:]
    return path


def camera_source_class(value: str) -> str:
    scheme = urlsplit(value).scheme.lower()
    if scheme == "file":
        return "local_file"
    if scheme in {"rtsp", "rtsps"}:
        return "rtsp_stream"
    if scheme in {"http", "https"}:
        return "http_stream"
    return "unsupported"


class RecoveringVideoCapture:
    """Own one configured source and reopen only that source after loss."""

    def __init__(self, source: str, args: argparse.Namespace, *, opener=None) -> None:
        self.source = source
        self.args = args
        self._opener = opener or open_bounded_video_capture
        self._capture = None
        self._read_failures = 0
        self._retry_delay = 0.25

    @property
    def is_opened(self) -> bool:
        return self._capture is not None and bool(self._capture.isOpened())

    def open_once(self) -> tuple[bool, float]:
        self._release_current()
        candidate = None
        try:
            candidate = self._opener(self.source)
            self._configure(candidate)
            if not candidate.isOpened():
                return self._reject_candidate(candidate)
        except Exception:
            return self._reject_candidate(candidate)
        self._capture = candidate
        self._read_failures = 0
        self._retry_delay = 0.25
        return True, 0.0

    def read(self):
        if not self.is_opened:
            return False, None
        try:
            ok, frame = self._capture.read()
        except Exception:
            ok, frame = False, None
        if ok and frame is not None:
            self._read_failures = 0
            return True, frame
        self._read_failures += 1
        if self._read_failures >= 3:
            self._release_current()
        return False, None

    def close(self) -> None:
        self._release_current()

    def _release_current(self) -> None:
        capture = self._capture
        self._capture = None
        self._read_failures = 0
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass

    def _reject_candidate(self, candidate) -> tuple[bool, float]:
        if candidate is not None:
            try:
                candidate.release()
            except Exception:
                pass
        delay = self._retry_delay
        self._retry_delay = min(max(delay * 2.0, 0.25), 3.0)
        return False, delay

    def _configure(self, capture) -> None:
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.args.camera_width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.camera_width)
        if self.args.camera_height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.camera_height)
        if self.args.camera_fps is not None:
            capture.set(cv2.CAP_PROP_FPS, self.args.camera_fps)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run snapshot-based vision processors and publish topic envelopes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=parse_port, default=8776)
    parser.add_argument("--camera-source", type=parse_camera_source, default="rtsp://127.0.0.1:8554/cam0")
    parser.add_argument("--frame-id", default="cam0")
    parser.add_argument("--processor", action="append", choices=["room_light"], default=["room_light"])
    parser.add_argument("--sample-every", type=parse_positive_float, default=1.0)
    parser.add_argument("--window-ms", type=parse_positive_int, default=1000)
    parser.add_argument("--min-frames", type=parse_min_frames, default=2)
    parser.add_argument("--resize-width", type=parse_positive_int, default=160)
    parser.add_argument("--camera-width", type=parse_positive_int)
    parser.add_argument("--camera-height", type=parse_positive_int)
    parser.add_argument("--camera-fps", type=parse_positive_float)
    parser.add_argument("--opencv-ffmpeg-capture-options", default=DEFAULT_OPENCV_FFMPEG_CAPTURE_OPTIONS)
    parser.add_argument("--max-clients", type=parse_positive_int, default=8)
    parser.add_argument("--max-message-bytes", type=parse_positive_int, default=8192)
    return parser


async def run(args: argparse.Namespace) -> None:
    if args.opencv_ffmpeg_capture_options.strip().lower() != "none":
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = args.opencv_ffmpeg_capture_options

    capture_source = normalize_camera_source_for_capture(args.camera_source)
    capture = RecoveringVideoCapture(capture_source, args)
    try:
        room_light = None
        if "room_light" in set(args.processor):
            room_light = RoomLightSnapshotProcessor(
                min_frames=args.min_frames,
                window_ms=args.window_ms,
                resize_width=args.resize_width,
            )

        broadcaster = WebSocketTopicBroadcaster(
            args.host,
            args.port,
            max_clients=args.max_clients,
            max_message_bytes=args.max_message_bytes,
        )
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        def request_stop(_signum: int, _frame: object) -> None:
            loop.call_soon_threadsafe(stop_event.set)

        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(signum, stop_event.set)
            except (NotImplementedError, RuntimeError):
                signal.signal(signum, request_stop)

        frame_number = 0
        next_sample_at = time.monotonic()
        async with broadcaster:
            print(f"vision snapshot processor listening on ws://{args.host}:{args.port}", flush=True)
            print(
                f"camera source class: {camera_source_class(args.camera_source)}",
                flush=True,
            )
            print(f"processors: {', '.join(sorted(set(args.processor)))}", flush=True)
            while not stop_event.is_set():
                if not capture.is_opened:
                    opened, retry_delay = await asyncio.to_thread(capture.open_once)
                    if not opened:
                        try:
                            await asyncio.wait_for(
                                stop_event.wait(), timeout=retry_delay
                            )
                        except TimeoutError:
                            pass
                        continue
                    if room_light is not None:
                        room_light.reset()
                delay = next_sample_at - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(min(delay, 0.1))
                    continue
                next_sample_at = max(next_sample_at + args.sample_every, time.monotonic())
                ok, frame = await asyncio.to_thread(capture.read)
                stamp = time.time()
                if not ok or frame is None:
                    await asyncio.sleep(0.1)
                    continue
                frame_number += 1
                if room_light is not None:
                    observation = room_light.observe(frame, frame_id=frame_number, stamp=stamp)
                    if observation is not None:
                        await broadcaster.publish(
                            topic_json(
                                ROOM_LIGHT_OBSERVATION_TOPIC,
                                MSG_TYPE_ROOM_LIGHT_OBSERVATION,
                                observation.to_payload(),
                                sequence=frame_number,
                                stamp=observation.observed_at,
                                frame_id=args.frame_id,
                            )
                        )
    finally:
        capture.close()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
