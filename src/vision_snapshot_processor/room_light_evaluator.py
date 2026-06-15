from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .processors.room_light import ROOM_LIGHT_MODEL_NAME, RoomLightSnapshotProcessor, RoomLightState


@dataclass(frozen=True)
class RoomLightEvaluationConfig:
    sample_id: str
    expected_state: str | None = None
    max_frames: int = 120
    sample_every_frames: int = 1
    min_frames: int = 2
    window_ms: int = 1000
    resize_width: int = 160


def evaluate_frames(
    frames: Iterable[np.ndarray],
    *,
    config: RoomLightEvaluationConfig,
    start_stamp: float = 0.0,
    frame_interval_s: float = 0.25,
) -> dict[str, Any]:
    processor = RoomLightSnapshotProcessor(
        min_frames=config.min_frames,
        window_ms=config.window_ms,
        resize_width=config.resize_width,
    )
    observations: list[RoomLightState] = []
    frames_seen = 0
    frames_sampled = 0

    for frame in frames:
        frames_seen += 1
        if frames_seen > config.max_frames:
            break
        if (frames_seen - 1) % config.sample_every_frames != 0:
            continue
        frames_sampled += 1
        state = processor.observe(
            frame,
            frame_id=frames_seen,
            stamp=start_stamp + (frames_seen - 1) * frame_interval_s,
        )
        if state is not None:
            observations.append(state)

    return summarize_observations(
        observations,
        config=config,
        frames_seen=frames_seen,
        frames_sampled=frames_sampled,
    )


def evaluate_video_file(path: Path, *, config: RoomLightEvaluationConfig) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("room-light evaluation input could not be opened")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_interval_s = 1.0 / fps if fps and fps > 0 else 0.25

    def iter_frames() -> Iterable[np.ndarray]:
        try:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                yield frame
        finally:
            capture.release()

    return evaluate_frames(iter_frames(), config=config, frame_interval_s=frame_interval_s)


def summarize_observations(
    observations: list[RoomLightState],
    *,
    config: RoomLightEvaluationConfig,
    frames_seen: int,
    frames_sampled: int,
) -> dict[str, Any]:
    final_state = observations[-1] if observations else None
    expected_state = _clean_expected_state(config.expected_state)
    result = "not_evaluated"
    if expected_state is not None:
        result = "pass" if final_state is not None and final_state.state == expected_state else "fail"
    elif final_state is not None:
        result = "observed"

    state_counts: dict[str, int] = {}
    for observation in observations:
        state_counts[observation.state] = state_counts.get(observation.state, 0) + 1

    summary: dict[str, Any] = {
        "type": "room_light_local_media_evaluation",
        "schema_version": 1,
        "sample_id": config.sample_id,
        "expected_state": expected_state,
        "result": result,
        "model": ROOM_LIGHT_MODEL_NAME,
        "frames_seen": frames_seen,
        "frames_sampled": frames_sampled,
        "observations": len(observations),
        "state_counts": state_counts,
        "raw_media_included": False,
        "raw_frame_included": False,
        "source_path_included": False,
        "non_claims": [
            "local_media_result_not_live_room_proof",
            "room_light_estimate_not_physical_switch_state",
            "electric_light_probability_not_robust_electric_light_proof",
        ],
    }
    if final_state is not None:
        payload = final_state.to_payload()
        summary["final"] = {
            "state": payload["state"],
            "confidence": payload["confidence"],
            "lighting_type": payload["lighting_type"],
            "electric_light": payload["electric_light"],
            "daylight": payload["daylight"],
            "probabilities": payload["probabilities"],
            "observed_at": payload["observed_at"],
            "observation_id": payload["observation_id"],
            "sequence": payload["sequence"],
            "model": payload["model"],
        }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a redacted local-media room-light evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Local media path. The path is never echoed in JSON output.")
    parser.add_argument("--sample-id", required=True, help="Stable sample id to include in redacted output.")
    parser.add_argument("--expected-state", choices=["on", "off", "unknown"])
    parser.add_argument("--max-frames", type=_positive_int, default=120)
    parser.add_argument("--sample-every-frames", type=_positive_int, default=1)
    parser.add_argument("--min-frames", type=_min_frames, default=2)
    parser.add_argument("--window-ms", type=_positive_int, default=1000)
    parser.add_argument("--resize-width", type=_positive_int, default=160)
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = RoomLightEvaluationConfig(
        sample_id=args.sample_id,
        expected_state=args.expected_state,
        max_frames=args.max_frames,
        sample_every_frames=args.sample_every_frames,
        min_frames=args.min_frames,
        window_ms=args.window_ms,
        resize_width=args.resize_width,
    )
    summary = evaluate_video_file(Path(args.input), config=config)
    print(json.dumps(summary, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True))
    return 0


def _clean_expected_state(value: str | None) -> str | None:
    if value is None:
        return None
    if value not in {"on", "off", "unknown"}:
        raise ValueError("expected_state must be on, off, unknown, or None")
    return value


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def _min_frames(value: str) -> int:
    parsed = _positive_int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("--min-frames must be 2 or greater")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
