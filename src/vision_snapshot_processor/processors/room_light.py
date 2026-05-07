from __future__ import annotations

import hashlib
import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import cv2
import numpy as np


ROOM_LIGHT_MODEL_NAME = "room-light-heuristic-snapshot-v1"


@dataclass(frozen=True)
class FrameFeatures:
    frame_id: int
    stamp: float
    luma_mean: float
    luma_std: float
    luma_p10: float
    luma_p90: float
    dynamic_range: float
    saturation_mean: float
    warm_ratio: float
    blue_ratio: float
    lab_b_mean: float
    edge_density: float
    underexposed_fraction: float
    overexposed_fraction: float
    temporal_delta: float
    gray_small: np.ndarray


@dataclass(frozen=True)
class RoomLightState:
    state: str
    confidence: float
    lighting_type: str
    electric_probability: float
    daylight_probability: float
    dark_probability: float
    observed_at: float
    first_frame_id: int
    last_frame_id: int
    frame_count: int
    temporal_window_ms: int
    feature_summary: dict[str, float]

    def to_payload(self) -> dict[str, Any]:
        observed_at = datetime.fromtimestamp(self.observed_at, tz=UTC).isoformat()
        observation_id = _observation_id(
            self.first_frame_id,
            self.last_frame_id,
            self.observed_at,
            self.electric_probability,
            self.daylight_probability,
        )
        daylight_state = _ternary_state(self.daylight_probability, high=0.65, low=0.35)
        if daylight_state == "on":
            daylight_state = "present"
        elif daylight_state == "off":
            daylight_state = "absent"

        return {
            "type": "room_light_state",
            "schema_version": 1,
            "label": self.lighting_type,
            "state": self.state,
            "confidence": round(self.confidence, 4),
            "lighting_type": self.lighting_type,
            "electric_light": {
                "state": self.state,
                "probability": round(self.electric_probability, 4),
            },
            "daylight": {
                "state": daylight_state,
                "probability": round(self.daylight_probability, 4),
            },
            "probabilities": {
                "electric_on": round(self.electric_probability, 4),
                "daylight_present": round(self.daylight_probability, 4),
                "dark": round(self.dark_probability, 4),
            },
            "observed_at": observed_at,
            "observation_id": observation_id,
            "sequence": {
                "frame_count": self.frame_count,
                "first_frame_id": self.first_frame_id,
                "last_frame_id": self.last_frame_id,
                "temporal_window_ms": self.temporal_window_ms,
            },
            "model": {
                "name": ROOM_LIGHT_MODEL_NAME,
                "kind": "heuristic",
            },
            "evidence": {
                "model": {
                    "name": ROOM_LIGHT_MODEL_NAME,
                    "kind": "heuristic",
                },
                "frames": self.frame_count,
                "temporal_window_ms": self.temporal_window_ms,
                "features": self.feature_summary,
            },
        }


class RoomLightSnapshotProcessor:
    def __init__(
        self,
        *,
        min_frames: int = 2,
        window_ms: int = 1000,
        resize_width: int = 160,
    ) -> None:
        if min_frames < 2:
            raise ValueError("min_frames must be 2 or greater")
        if window_ms <= 0:
            raise ValueError("window_ms must be greater than 0")
        if resize_width <= 0:
            raise ValueError("resize_width must be greater than 0")
        self.min_frames = int(min_frames)
        self.window_seconds = float(window_ms) / 1000.0
        self.resize_width = int(resize_width)
        self._frames: deque[FrameFeatures] = deque()

    def observe(self, frame_bgr: np.ndarray, *, frame_id: int, stamp: float | None = None) -> RoomLightState | None:
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            return None
        observed_at = time.time() if stamp is None else float(stamp)
        previous_gray = self._frames[-1].gray_small if self._frames else None
        features = _extract_features(
            frame_bgr,
            frame_id=int(frame_id),
            stamp=observed_at,
            resize_width=self.resize_width,
            previous_gray=previous_gray,
        )
        self._frames.append(features)
        self._trim(observed_at)
        if len(self._frames) < self.min_frames:
            return None
        return _classify(list(self._frames))

    def _trim(self, now: float) -> None:
        while self._frames and (now - self._frames[0].stamp) > self.window_seconds:
            if len(self._frames) <= self.min_frames:
                break
            self._frames.popleft()


def _extract_features(
    frame_bgr: np.ndarray,
    *,
    frame_id: int,
    stamp: float,
    resize_width: int,
    previous_gray: np.ndarray | None,
) -> FrameFeatures:
    height, width = frame_bgr.shape[:2]
    scale = resize_width / max(1, width)
    resize_height = max(1, int(round(height * scale)))
    small = cv2.resize(frame_bgr, (resize_width, resize_height), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    bgr = small.astype(np.float32) + 1.0
    blue = bgr[:, :, 0]
    red = bgr[:, :, 2]

    luma_mean = _finite_mean(gray)
    luma_std = float(np.std(gray))
    luma_p10 = float(np.percentile(gray, 10))
    luma_p90 = float(np.percentile(gray, 90))
    dynamic_range = max(0.0, luma_p90 - luma_p10)
    saturation_mean = _finite_mean(hsv[:, :, 1] / 255.0)
    warm_ratio = _clamp_float(float(np.mean(red / blue)), 0.0, 4.0)
    blue_ratio = _clamp_float(float(np.mean(blue / red)), 0.0, 4.0)
    lab_b_mean = _clamp_float(float(np.mean((lab[:, :, 2] - 128.0) / 128.0)), -1.0, 1.0)

    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    edge_density = _clamp_float(float(np.mean(np.abs(laplacian) > 0.055)), 0.0, 1.0)
    underexposed_fraction = _clamp_float(float(np.mean(gray < 0.08)), 0.0, 1.0)
    overexposed_fraction = _clamp_float(float(np.mean(gray > 0.94)), 0.0, 1.0)
    temporal_delta = 0.0
    if previous_gray is not None and previous_gray.shape == gray.shape:
        temporal_delta = _clamp_float(float(np.mean(np.abs(gray - previous_gray))), 0.0, 1.0)

    return FrameFeatures(
        frame_id=frame_id,
        stamp=stamp,
        luma_mean=luma_mean,
        luma_std=luma_std,
        luma_p10=luma_p10,
        luma_p90=luma_p90,
        dynamic_range=dynamic_range,
        saturation_mean=saturation_mean,
        warm_ratio=warm_ratio,
        blue_ratio=blue_ratio,
        lab_b_mean=lab_b_mean,
        edge_density=edge_density,
        underexposed_fraction=underexposed_fraction,
        overexposed_fraction=overexposed_fraction,
        temporal_delta=temporal_delta,
        gray_small=gray,
    )


def _classify(frames: list[FrameFeatures]) -> RoomLightState:
    summary = {
        "luma_mean": _mean(frames, "luma_mean"),
        "luma_std": _mean(frames, "luma_std"),
        "dynamic_range": _mean(frames, "dynamic_range"),
        "saturation_mean": _mean(frames, "saturation_mean"),
        "warm_ratio": _mean(frames, "warm_ratio"),
        "blue_ratio": _mean(frames, "blue_ratio"),
        "lab_b_mean": _mean(frames, "lab_b_mean"),
        "edge_density": _mean(frames, "edge_density"),
        "underexposed_fraction": _mean(frames, "underexposed_fraction"),
        "overexposed_fraction": _mean(frames, "overexposed_fraction"),
        "temporal_delta": _mean(frames[1:], "temporal_delta") if len(frames) > 1 else 0.0,
    }

    dark_probability = _sigmoid(
        (0.20 - summary["luma_mean"]) * 7.0
        + (0.16 - summary["luma_std"]) * 3.0
        + summary["underexposed_fraction"] * 2.8
        - summary["overexposed_fraction"] * 2.0
    )
    daylight_probability = _sigmoid(
        (summary["luma_mean"] - 0.44) * 2.4
        + (summary["dynamic_range"] - 0.38) * 2.0
        + (summary["blue_ratio"] - 1.02) * 2.2
        - (summary["lab_b_mean"] * 1.4)
        - dark_probability * 1.2
    )
    electric_probability = _sigmoid(
        (summary["warm_ratio"] - 1.04) * 3.0
        + (summary["lab_b_mean"] - 0.015) * 2.2
        + (summary["saturation_mean"] - 0.12) * 1.1
        + (summary["edge_density"] - 0.04) * 1.0
        + (summary["luma_mean"] - 0.28) * 1.1
        - dark_probability * 1.8
        - max(0.0, daylight_probability - 0.62) * 0.9
    )

    lighting_type = _lighting_type(electric_probability, daylight_probability, dark_probability)
    state = "unknown"
    confidence = 0.0
    if electric_probability >= 0.68 and dark_probability < 0.65:
        state = "on"
        confidence = min(1.0, electric_probability)
    elif electric_probability <= 0.28 and dark_probability >= 0.58 and daylight_probability <= 0.42:
        state = "off"
        confidence = min(1.0, (1.0 - electric_probability) * dark_probability)

    first = frames[0]
    last = frames[-1]
    temporal_window_ms = max(0, int(round((last.stamp - first.stamp) * 1000.0)))
    return RoomLightState(
        state=state,
        confidence=_clamp_float(confidence, 0.0, 1.0),
        lighting_type=lighting_type,
        electric_probability=_clamp_float(electric_probability, 0.0, 1.0),
        daylight_probability=_clamp_float(daylight_probability, 0.0, 1.0),
        dark_probability=_clamp_float(dark_probability, 0.0, 1.0),
        observed_at=last.stamp,
        first_frame_id=first.frame_id,
        last_frame_id=last.frame_id,
        frame_count=len(frames),
        temporal_window_ms=temporal_window_ms,
        feature_summary={key: round(value, 5) for key, value in summary.items()},
    )


def _lighting_type(electric: float, daylight: float, dark: float) -> str:
    if dark >= 0.72 and electric < 0.45 and daylight < 0.45:
        return "dark"
    if electric >= 0.62 and daylight >= 0.50:
        return "mixed"
    if electric >= 0.68:
        return "electric"
    if daylight >= 0.68:
        return "daylight"
    return "unknown"


def _ternary_state(probability: float, *, high: float, low: float) -> str:
    if probability >= high:
        return "on"
    if probability <= low:
        return "off"
    return "unknown"


def _observation_id(*values: object) -> str:
    digest = hashlib.sha1("|".join(str(value) for value in values).encode("utf-8")).hexdigest()
    return f"room-light-{digest[:12]}"


def _mean(frames: list[FrameFeatures], field: str) -> float:
    if not frames:
        return 0.0
    return _clamp_float(float(sum(float(getattr(frame, field)) for frame in frames) / len(frames)), -10.0, 10.0)


def _finite_mean(values: np.ndarray) -> float:
    value = float(np.mean(values))
    return value if math.isfinite(value) else 0.0


def _sigmoid(value: float) -> float:
    if value >= 60:
        return 1.0
    if value <= -60:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def _clamp_float(value: float, low: float, high: float) -> float:
    if not math.isfinite(value):
        return low
    return min(high, max(low, value))
