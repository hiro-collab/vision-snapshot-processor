from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


AVATAR_PASS_LABEL = "visual-motion-detected"
AVATAR_NOT_REQUIRED_LABEL = "avatar-motion-not-required"
GUARD_EXCLUDED_LABEL = "guard-ui-motion-excluded"
UI_ONLY_LABEL = "ui-only-motion-not-avatar-motion"
MISSING_MOTION_LABEL = "visual-missing-motion"
PRETRIGGER_LABEL = "visual-pretrigger-motion"
SETTLE_JITTER_LABEL = "visual-settle-jitter"
VISUAL_PASS = "visual-pass"
SAFE_REDACTED_SOURCE_REF_PATTERN = re.compile(r"^redacted_[a-z0-9_.:-]+$")
SOURCE_REF_KINDS = {
    "local_frame_sequence",
    "local_video_file",
    "browser_frame_provider",
    "synthetic_test_frames",
}


@dataclass(frozen=True)
class Window:
    window_id: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class Roi:
    roi_id: str
    kind: str
    counts_as_avatar_motion: bool
    expected_for_pass: bool
    rect_norm: dict[str, float]


def analyze_frames(
    frames_bgr: list[np.ndarray],
    *,
    analysis_run_id: str,
    scenario_id: str,
    motion_event_id: str,
    stimulus_instance_id: str,
    driver_result_id: str,
    sample_rate_fps: float,
    windows: list[dict[str, Any]],
    rois: list[dict[str, Any]],
    thresholds: dict[str, Any] | None = None,
    source_ref_id: str = "redacted_local_source",
    source_ref_kind: str = "local_frame_sequence",
    proof_layer: str = "no_live_runtime",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not frames_bgr:
        raise ValueError("frames_bgr must contain at least one frame")
    if sample_rate_fps <= 0:
        raise ValueError("sample_rate_fps must be greater than 0")

    parsed_windows = [_parse_window(window) for window in windows]
    parsed_rois = [_parse_roi(roi) for roi in rois]
    limits = {
        "active_motion_min_score": 0.12,
        "settle_motion_max_score": 0.05,
        "min_consecutive_samples": 2,
    }
    if thresholds:
        limits.update(thresholds)

    rows: list[dict[str, Any]] = []
    baseline = frames_bgr[0]
    previous = frames_bgr[0]
    for frame_index, frame in enumerate(frames_bgr):
        time_ms = int(round(frame_index * 1000.0 / sample_rate_fps))
        window_id = _window_id_at(time_ms, parsed_windows)
        if frame_index == 0:
            previous = frame
            continue
        for roi in parsed_rois:
            rows.append(
                _measure_roi(
                    frame,
                    previous,
                    baseline,
                    roi=roi,
                    analysis_run_id=analysis_run_id,
                    time_ms=time_ms,
                    window_id=window_id,
                )
            )
        previous = frame

    roi_results = _summarize_rois(rows, parsed_rois, limits)
    result = _overall_result(roi_results, limits)
    evaluation_window_ms = max((window.end_ms for window in parsed_windows), default=0)
    summary = {
        "schema_version": "visual_motion_analysis.v0",
        "analysis_run_id": analysis_run_id,
        "scenario_id": scenario_id,
        "proof_layer": proof_layer,
        "motion_event_id": motion_event_id,
        "stimulus_instance_id": stimulus_instance_id,
        "driver_result_id": driver_result_id,
        "mixer_tick_ids": [],
        "source_ref": {
            "kind": _safe_source_ref_kind(source_ref_kind),
            "source_ref_id": _safe_source_ref_id(source_ref_id),
            "raw_source_shared": False,
        },
        "sampling": {
            "sample_rate_fps": sample_rate_fps,
            "evaluation_window_ms": evaluation_window_ms,
            "frame_count": len(frames_bgr),
        },
        "windows": [
            {"window_id": window.window_id, "start_ms": window.start_ms, "end_ms": window.end_ms}
            for window in parsed_windows
        ],
        "roi_config": {
            "viewport": {"width": int(frames_bgr[0].shape[1]), "height": int(frames_bgr[0].shape[0])},
            "rois": [
                {
                    "roi_id": roi.roi_id,
                    "kind": roi.kind,
                    "counts_as_avatar_motion": roi.counts_as_avatar_motion,
                    "expected_for_pass": roi.expected_for_pass,
                    "rect_norm": roi.rect_norm,
                }
                for roi in parsed_rois
            ],
        },
        "thresholds": {
            "active_motion_min_score": float(limits["active_motion_min_score"]),
            "settle_motion_max_score": float(limits["settle_motion_max_score"]),
            "min_consecutive_samples": int(limits["min_consecutive_samples"]),
        },
        "result": result,
        "roi_results": roi_results,
        "artifact_policy": {
            "raw_frames_shared": False,
            "raw_paths_shared": False,
            "chart_shared": False,
            "cleanup_note_required": True,
        },
        "redaction": {
            "redaction_status": "summary_only",
            "shareability_class": "review_packet",
            "public_safe": False,
        },
        "safety": {
            "raw_prompt_shared": False,
            "raw_transcript_shared": False,
            "raw_log_shared": False,
            "raw_media_shared": False,
            "raw_path_shared": False,
            "raw_asset_filename_shared": False,
            "provider_payload_shared": False,
            "private_endpoint_shared": False,
            "home_assistant_route_retained": False,
        },
    }
    return summary, rows


def analyze_config(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_ref = config.get("source_ref", {})
    sampling = config.get("sampling", {})
    sample_rate_fps = float(sampling.get("sample_rate_fps", config.get("sample_rate_fps", 8)))
    windows = list(config["windows"])
    rois = list(config["rois"])
    if "synthetic_fixture" in config:
        frames = _generate_synthetic_fixture_frames(
            dict(config["synthetic_fixture"]),
            windows=windows,
            rois=rois,
            sample_rate_fps=sample_rate_fps,
        )
        source_ref_kind = str(source_ref.get("kind", "synthetic_test_frames"))
    else:
        frame_paths = [Path(path) for path in config.get("frame_paths", [])]
        if not frame_paths:
            raise ValueError("config.frame_paths must contain local frame paths or synthetic_fixture")
        frames = [_read_frame(path) for path in frame_paths]
        source_ref_kind = str(source_ref.get("kind", "local_frame_sequence"))

    return analyze_frames(
        frames,
        analysis_run_id=str(config["analysis_run_id"]),
        scenario_id=str(config["scenario_id"]),
        motion_event_id=str(config["motion_event_id"]),
        stimulus_instance_id=str(config["stimulus_instance_id"]),
        driver_result_id=str(config["driver_result_id"]),
        sample_rate_fps=sample_rate_fps,
        windows=windows,
        rois=rois,
        thresholds=dict(config.get("thresholds", {})),
        source_ref_id=str(source_ref.get("source_ref_id", "redacted_local_source")),
        source_ref_kind=source_ref_kind,
        proof_layer=str(config.get("proof_layer", "no_live_runtime")),
    )


def write_outputs(summary: dict[str, Any], rows: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "visual_motion_summary.json"
    csv_path = output_dir / "visual_motion_roi_timeseries.csv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fieldnames = [
        "analysis_run_id",
        "time_ms",
        "window_id",
        "roi_id",
        "roi_kind",
        "counts_as_avatar_motion",
        "changed_pixel_ratio",
        "optical_flow_mean",
        "optical_flow_p95",
        "bbox_delta",
        "centroid_delta",
        "ssim_to_baseline",
        "motion_score",
        "pass_label",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return summary_path, csv_path


def _parse_window(value: dict[str, Any]) -> Window:
    start_ms = int(value["start_ms"])
    end_ms = int(value["end_ms"])
    if end_ms <= start_ms:
        raise ValueError(f"window end_ms must be greater than start_ms: {value}")
    return Window(window_id=str(value["window_id"]), start_ms=start_ms, end_ms=end_ms)


def _parse_roi(value: dict[str, Any]) -> Roi:
    return Roi(
        roi_id=str(value["roi_id"]),
        kind=str(value["kind"]),
        counts_as_avatar_motion=bool(value["counts_as_avatar_motion"]),
        expected_for_pass=bool(value["expected_for_pass"]),
        rect_norm={key: float(value["rect_norm"][key]) for key in ("x", "y", "w", "h")},
    )


def _safe_source_ref_id(value: str) -> str:
    normalized = value.strip().lower()
    if (
        SAFE_REDACTED_SOURCE_REF_PATTERN.fullmatch(normalized)
        and "/" not in normalized
        and "\\" not in normalized
    ):
        return normalized
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"redacted_source_{digest}"


def _safe_source_ref_kind(value: str) -> str:
    return value if value in SOURCE_REF_KINDS else "local_frame_sequence"


def _read_frame(path: Path) -> np.ndarray:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("failed to read frame")
    return frame


def _generate_synthetic_fixture_frames(
    fixture: dict[str, Any],
    *,
    windows: list[dict[str, Any]],
    rois: list[dict[str, Any]],
    sample_rate_fps: float,
) -> list[np.ndarray]:
    frame_count = int(fixture.get("frame_count", 48))
    width = int(fixture.get("width", 640))
    height = int(fixture.get("height", 360))
    if frame_count <= 0:
        raise ValueError("synthetic_fixture.frame_count must be greater than 0")
    if width <= 0 or height <= 0:
        raise ValueError("synthetic_fixture width and height must be greater than 0")

    parsed_windows = [_parse_window(window) for window in windows]
    parsed_rois = [_parse_roi(roi) for roi in rois]
    expected_avatar_roi_ids = {
        str(value)
        for value in fixture.get(
            "avatar_motion_roi_ids",
            [
                roi.roi_id
                for roi in parsed_rois
                if roi.counts_as_avatar_motion and roi.expected_for_pass
            ],
        )
    }
    guard_motion_roi_ids = {str(value) for value in fixture.get("guard_motion_roi_ids", [])}
    motion_amplitude_px = int(fixture.get("motion_amplitude_px", max(8, width // 64)))
    frames: list[np.ndarray] = []

    for frame_index in range(frame_count):
        time_ms = int(round(frame_index * 1000.0 / sample_rate_fps))
        window_id = _window_id_at(time_ms, parsed_windows)
        frame = _synthetic_base_frame(width, height, parsed_rois)
        for roi in parsed_rois:
            if roi.roi_id in expected_avatar_roi_ids:
                _draw_synthetic_motion_marker(
                    frame,
                    roi,
                    frame_index=frame_index,
                    window_id=window_id,
                    amplitude_px=motion_amplitude_px,
                    color=(238, 246, 255),
                )
            elif roi.roi_id in guard_motion_roi_ids:
                _draw_synthetic_motion_marker(
                    frame,
                    roi,
                    frame_index=frame_index,
                    window_id=window_id,
                    amplitude_px=motion_amplitude_px,
                    color=(255, 225, 82),
                )
        frames.append(frame)
    return frames


def _synthetic_base_frame(width: int, height: int, rois: list[Roi]) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (20, 92, 46)
    for roi in rois:
        x0, y0, x1, y1 = _roi_pixels(roi, width, height)
        if roi.kind == "avatar":
            cv2.rectangle(frame, (x0, y0), (x1, y1), (66, 82, 102), thickness=-1)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (118, 141, 166), thickness=2)
        elif roi.kind == "guard_ui":
            cv2.rectangle(frame, (x0, y0), (x1, y1), (30, 34, 42), thickness=-1)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (88, 104, 126), thickness=2)
        else:
            cv2.rectangle(frame, (x0, y0), (x1, y1), (18, 70, 84), thickness=-1)
    return frame


def _draw_synthetic_motion_marker(
    frame: np.ndarray,
    roi: Roi,
    *,
    frame_index: int,
    window_id: str,
    amplitude_px: int,
    color: tuple[int, int, int],
) -> None:
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = _roi_pixels(roi, width, height)
    roi_width = max(1, x1 - x0)
    roi_height = max(1, y1 - y0)
    marker_width = max(4, roi_width // 3)
    marker_height = max(4, roi_height // 3)
    base_x = x0 + roi_width // 2 - marker_width // 2
    base_y = y0 + roi_height // 2 - marker_height // 2
    if window_id == "active":
        offset_x = int(round(math.sin(frame_index * 0.9) * amplitude_px))
        offset_y = int(round(math.cos(frame_index * 0.7) * max(2, amplitude_px // 2)))
    else:
        offset_x = 0
        offset_y = 0
    px0 = max(x0, min(x1 - marker_width, base_x + offset_x))
    py0 = max(y0, min(y1 - marker_height, base_y + offset_y))
    cv2.rectangle(frame, (px0, py0), (px0 + marker_width, py0 + marker_height), color, thickness=-1)


def _roi_pixels(roi: Roi, width: int, height: int) -> tuple[int, int, int, int]:
    rect = roi.rect_norm
    x0 = int(round(_clamp(rect["x"], 0.0, 1.0) * width))
    y0 = int(round(_clamp(rect["y"], 0.0, 1.0) * height))
    x1 = int(round(_clamp(rect["x"] + rect["w"], 0.0, 1.0) * width))
    y1 = int(round(_clamp(rect["y"] + rect["h"], 0.0, 1.0) * height))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))
    return x0, y0, x1, y1


def _window_id_at(time_ms: int, windows: Iterable[Window]) -> str:
    for window in windows:
        if window.start_ms <= time_ms < window.end_ms:
            return window.window_id
    return "outside"


def _measure_roi(
    frame: np.ndarray,
    previous: np.ndarray,
    baseline: np.ndarray,
    *,
    roi: Roi,
    analysis_run_id: str,
    time_ms: int,
    window_id: str,
) -> dict[str, Any]:
    crop = _crop(frame, roi)
    prev_crop = _crop(previous, roi)
    base_crop = _crop(baseline, roi)
    gray = _gray(crop)
    prev_gray = _gray(prev_crop)
    base_gray = _gray(base_crop)

    diff_prev = np.abs(gray - prev_gray)
    diff_base = np.abs(gray - base_gray)
    changed_pixel_ratio = _clamp(float(np.mean(diff_prev > 0.06)), 0.0, 1.0)
    flow_mean, flow_p95 = _optical_flow(prev_gray, gray)
    bbox_delta, centroid_delta = _motion_mask_stats(diff_prev > 0.06)
    ssim_to_baseline = _ssim(base_gray, gray)
    motion_score = _clamp(
        max(
            changed_pixel_ratio,
            min(flow_p95 / 12.0, 1.0),
            bbox_delta,
            centroid_delta,
            1.0 - ssim_to_baseline if np.mean(diff_base) > 0.01 else 0.0,
        ),
        0.0,
        1.0,
    )
    return {
        "analysis_run_id": analysis_run_id,
        "time_ms": time_ms,
        "window_id": window_id,
        "roi_id": roi.roi_id,
        "roi_kind": roi.kind,
        "counts_as_avatar_motion": roi.counts_as_avatar_motion,
        "changed_pixel_ratio": round(changed_pixel_ratio, 6),
        "optical_flow_mean": round(flow_mean, 6),
        "optical_flow_p95": round(flow_p95, 6),
        "bbox_delta": round(bbox_delta, 6),
        "centroid_delta": round(centroid_delta, 6),
        "ssim_to_baseline": round(ssim_to_baseline, 6),
        "motion_score": round(motion_score, 6),
        "pass_label": "",
    }


def _crop(frame: np.ndarray, roi: Roi) -> np.ndarray:
    height, width = frame.shape[:2]
    rect = roi.rect_norm
    x0 = int(round(_clamp(rect["x"], 0.0, 1.0) * width))
    y0 = int(round(_clamp(rect["y"], 0.0, 1.0) * height))
    x1 = int(round(_clamp(rect["x"] + rect["w"], 0.0, 1.0) * width))
    y1 = int(round(_clamp(rect["y"] + rect["h"], 0.0, 1.0) * height))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))
    return frame[y0:y1, x0:x1]


def _gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def _optical_flow(previous: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    if previous.shape != current.shape or previous.size == 0:
        return 0.0, 0.0
    flow = cv2.calcOpticalFlowFarneback(
        previous,
        current,
        None,
        pyr_scale=0.5,
        levels=1,
        winsize=9,
        iterations=2,
        poly_n=5,
        poly_sigma=1.1,
        flags=0,
    )
    magnitude = np.sqrt(flow[..., 0] * flow[..., 0] + flow[..., 1] * flow[..., 1])
    return _clamp(float(np.mean(magnitude)), 0.0, 1000.0), _clamp(float(np.percentile(magnitude, 95)), 0.0, 1000.0)


def _motion_mask_stats(mask: np.ndarray) -> tuple[float, float]:
    points = cv2.findNonZero(mask.astype(np.uint8))
    if points is None:
        return 0.0, 0.0
    x, y, width, height = cv2.boundingRect(points)
    bbox_delta = _clamp(float(width * height) / float(mask.shape[0] * mask.shape[1]), 0.0, 1.0)
    moments = cv2.moments(points)
    if moments["m00"] == 0:
        return bbox_delta, 0.0
    cx = float(moments["m10"] / moments["m00"])
    cy = float(moments["m01"] / moments["m00"])
    center_x = mask.shape[1] / 2.0
    center_y = mask.shape[0] / 2.0
    diagonal = math.sqrt(mask.shape[0] * mask.shape[0] + mask.shape[1] * mask.shape[1])
    centroid_delta = math.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2) / max(1.0, diagonal)
    return bbox_delta, _clamp(centroid_delta, 0.0, 1.0)


def _ssim(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape or first.size == 0:
        return 0.0
    mu_x = float(np.mean(first))
    mu_y = float(np.mean(second))
    var_x = float(np.var(first))
    var_y = float(np.var(second))
    cov_xy = float(np.mean((first - mu_x) * (second - mu_y)))
    c1 = 0.01 * 0.01
    c2 = 0.03 * 0.03
    numerator = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    if denominator == 0:
        return 1.0
    return _clamp(numerator / denominator, 0.0, 1.0)


def _summarize_rois(rows: list[dict[str, Any]], rois: list[Roi], thresholds: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    active_threshold = float(thresholds["active_motion_min_score"])
    settle_threshold = float(thresholds["settle_motion_max_score"])
    min_consecutive_samples = max(1, int(thresholds["min_consecutive_samples"]))
    for roi in rois:
        roi_rows = [row for row in rows if row["roi_id"] == roi.roi_id]
        pretrigger_peak = _peak(roi_rows, "pretrigger")
        active_peak = _peak(roi_rows, "active")
        release_peak = _peak(roi_rows, "release")
        settle_peak = _peak(roi_rows, "settle")
        if roi.counts_as_avatar_motion and roi.expected_for_pass:
            active_run = _max_consecutive_at_or_above(roi_rows, "active", active_threshold)
            if pretrigger_peak >= active_threshold:
                label = PRETRIGGER_LABEL
            elif active_run >= min_consecutive_samples and settle_peak <= settle_threshold:
                label = AVATAR_PASS_LABEL
            elif settle_peak > settle_threshold:
                label = SETTLE_JITTER_LABEL
            else:
                label = MISSING_MOTION_LABEL
        elif roi.counts_as_avatar_motion:
            label = AVATAR_NOT_REQUIRED_LABEL
        else:
            label = GUARD_EXCLUDED_LABEL
        results.append(
            {
                "roi_id": roi.roi_id,
                "kind": roi.kind,
                "counts_as_avatar_motion": roi.counts_as_avatar_motion,
                "pretrigger_peak_motion_score": round(pretrigger_peak, 6),
                "active_peak_motion_score": round(active_peak, 6),
                "release_peak_motion_score": round(release_peak, 6),
                "settle_peak_motion_score": round(settle_peak, 6),
                "pass_label": label,
            }
        )
    return results


def _overall_result(roi_results: list[dict[str, Any]], thresholds: dict[str, Any]) -> str:
    expected = [
        row
        for row in roi_results
        if row["counts_as_avatar_motion"] and row["pass_label"] != AVATAR_NOT_REQUIRED_LABEL
    ]
    guards = [row for row in roi_results if not row["counts_as_avatar_motion"]]
    expected_pretrigger = any(row["pass_label"] == PRETRIGGER_LABEL for row in expected)
    expected_pass = any(row["pass_label"] == AVATAR_PASS_LABEL for row in expected)
    expected_jitter = any(row["pass_label"] == SETTLE_JITTER_LABEL for row in expected)
    guard_motion = any(row["active_peak_motion_score"] >= float(thresholds["active_motion_min_score"]) for row in guards)
    if expected_pretrigger:
        return PRETRIGGER_LABEL
    if expected_pass:
        return VISUAL_PASS
    if guard_motion and expected:
        return UI_ONLY_LABEL
    if expected_jitter:
        return SETTLE_JITTER_LABEL
    return MISSING_MOTION_LABEL


def _peak(rows: list[dict[str, Any]], window_id: str) -> float:
    values = [float(row["motion_score"]) for row in rows if row["window_id"] == window_id]
    return max(values) if values else 0.0


def _max_consecutive_at_or_above(rows: list[dict[str, Any]], window_id: str, threshold: float) -> int:
    longest = 0
    current = 0
    for row in sorted((row for row in rows if row["window_id"] == window_id), key=lambda item: int(item["time_ms"])):
        if float(row["motion_score"]) >= threshold:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _clamp(value: float, low: float, high: float) -> float:
    if not math.isfinite(value):
        return low
    return min(high, max(low, value))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run bounded RR003 visual motion analysis.")
    parser.add_argument("--config", required=True, help="Local-only visual analyzer config JSON.")
    parser.add_argument("--output-dir", required=True, help="Output directory for summary JSON and CSV.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable result summary.")
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    summary, rows = analyze_config(config)
    summary_path, csv_path = write_outputs(summary, rows, Path(args.output_dir))
    result = {
        "status": "ok",
        "proof_layer": summary["proof_layer"],
        "result": summary["result"],
        "summary_file": summary_path.name,
        "timeseries_file": csv_path.name,
        "raw_frames_shared": False,
        "raw_paths_shared": False,
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(f"Visual Motion Analyzer: {result['status']}")
        print(f"proof_layer={result['proof_layer']}")
        print(f"result={result['result']}")
        print("raw_frames_shared=false")
        print("raw_paths_shared=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
