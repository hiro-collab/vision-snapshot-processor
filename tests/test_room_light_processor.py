from __future__ import annotations

import copy
import json
import math
import os
import tempfile
import traceback
from datetime import UTC, datetime
from pathlib import Path
import unittest

import numpy as np

from vision_snapshot_processor.room_light_evaluator import (
    RoomLightEvaluationConfig,
    evaluate_frames,
    summarize_observations,
)
from vision_snapshot_processor.processors.room_light import RoomLightSnapshotProcessor
from vision_snapshot_processor.processors.room_light import RoomLightObservation
from vision_snapshot_processor.processors import room_light
from vision_snapshot_processor.topics import (
    MSG_TYPE_ROOM_LIGHT_OBSERVATION,
    ROOM_LIGHT_OBSERVATION_TOPIC,
    topic_json,
)


_SHARED_VECTOR_ENV = "SWORD_T1_ROOM_LIGHT_SHARED_VECTOR_PATH"
_MAX_SHARED_VECTOR_PATH_CHARS = 2048
_MAX_SHARED_VECTOR_BYTES = 32 * 1024
_FIXTURE_UNAVAILABLE = "room_light_fixture_unavailable"
_FIXTURE_INVALID = "room_light_fixture_invalid"
_ORDERED_CASE_IDS = (
    "canonical_camera_hub",
    "canonical_vision_snapshot_processor",
    "malformed_nested_sequence",
    "wrong_numeric_type",
    "nonfinite_numeric",
    "out_of_range_numeric",
    "wrong_case",
    "stale_freshness",
    "reversed_ordered_nonclaims",
    "non_room_light",
    "unknown_field_non_echo",
    "wrong_proof_ceiling",
    "responsiveness_same_identity_material_movement",
    "responsiveness_changed_identity_no_material_movement",
    "responsiveness_changed_identity_material_movement",
)
_DOES_NOT_PROVE = [
    "physical_room_light_state",
    "home_assistant_light_state",
]
_EXPECTED_CLASSES = {
    "canonical_camera_hub": ("valid", "camera-environment-estimate-high-confidence", "pass", "material_camera_environment_estimate_change_with_new_observation", "not_echoed"),
    "canonical_vision_snapshot_processor": ("valid", "camera-environment-estimate-high-confidence", "pass", "material_camera_environment_estimate_change_with_new_observation", "not_echoed"),
    "malformed_nested_sequence": ("invalid", "unavailable", "fail", "noncanonical_camera_environment_estimate", "not_echoed"),
    "wrong_numeric_type": ("invalid", "unavailable", "fail", "noncanonical_camera_environment_estimate", "not_echoed"),
    "nonfinite_numeric": ("invalid", "unavailable", "fail", "noncanonical_camera_environment_estimate", "not_echoed"),
    "out_of_range_numeric": ("invalid", "unavailable", "fail", "noncanonical_camera_environment_estimate", "not_echoed"),
    "wrong_case": ("invalid", "unavailable", "fail", "noncanonical_camera_environment_estimate", "not_echoed"),
    "stale_freshness": ("valid", "unavailable", "partial", "material_camera_environment_estimate_change_with_new_observation", "not_echoed"),
    "reversed_ordered_nonclaims": ("invalid", "unavailable", "fail", "noncanonical_camera_environment_estimate", "not_echoed"),
    "non_room_light": ("invalid", "unavailable", "fail", "noncanonical_camera_environment_estimate", "not_echoed"),
    "unknown_field_non_echo": ("valid", "camera-environment-estimate-high-confidence", "pass", "material_camera_environment_estimate_change_with_new_observation", "not_echoed"),
    "wrong_proof_ceiling": ("invalid", "unavailable", "fail", "noncanonical_camera_environment_estimate", "not_echoed"),
    "responsiveness_same_identity_material_movement": ("valid", "camera-environment-estimate-high-confidence", "fail", "material_camera_environment_estimate_change_without_new_observation", "not_echoed"),
    "responsiveness_changed_identity_no_material_movement": ("valid", "camera-environment-estimate-high-confidence", "fail", "new_observation_without_material_camera_environment_estimate_change", "not_echoed"),
    "responsiveness_changed_identity_material_movement": ("valid", "camera-environment-estimate-high-confidence", "pass", "material_camera_environment_estimate_change_with_new_observation", "not_echoed"),
}
_PAYLOAD_KEYS = {
    "type", "schema_version", "observation_bucket", "confidence",
    "daylight_ambiguity", "cue_likelihoods", "source", "source_class",
    "observed_at", "observation_id", "source_snapshot_id", "sequence",
    "model", "freshness", "proof_ceiling", "does_not_prove",
}


def _contract(condition: bool) -> None:
    if not condition:
        raise AssertionError(_FIXTURE_INVALID) from None


def _fixture_unavailable() -> None:
    raise AssertionError(_FIXTURE_UNAVAILABLE) from None


def _no_duplicate_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        _contract(key not in result)
        result[key] = value
    return result


def _load_shared_vectors(path_text: str) -> dict[str, object]:
    try:
        _contract(isinstance(path_text, str))
        _contract(0 < len(path_text) <= _MAX_SHARED_VECTOR_PATH_CHARS)
        _contract(not any(ord(char) < 32 or ord(char) == 127 for char in path_text))
        path = Path(path_text)
    except (TypeError, ValueError, OverflowError):
        raise AssertionError(_FIXTURE_INVALID) from None

    try:
        if not path.is_file():
            _fixture_unavailable()
        size = path.stat().st_size
        raw = path.read_bytes()
    except AssertionError:
        raise
    except OSError:
        raise AssertionError(_FIXTURE_UNAVAILABLE) from None

    _contract(0 < size <= _MAX_SHARED_VECTOR_BYTES)
    _contract(len(raw) == size)
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_no_duplicate_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                AssertionError(_FIXTURE_INVALID)
            ),
        )
    except (AssertionError, UnicodeError, ValueError, json.JSONDecodeError, TypeError):
        raise AssertionError(_FIXTURE_INVALID) from None
    _contract(isinstance(document, dict))

    try:
        _assert_shared_vector_contract(document)
    except (AssertionError, AttributeError, IndexError, KeyError, OverflowError, TypeError, ValueError):
        raise AssertionError(_FIXTURE_INVALID) from None
    return document


def _bounded_tree(value: object, *, depth: int = 0) -> None:
    _contract(depth <= 5)
    if isinstance(value, dict):
        _contract(len(value) <= 20)
        for key, child in value.items():
            _contract(isinstance(key, str) and 0 < len(key) <= 64 and key.isascii())
            _bounded_tree(child, depth=depth + 1)
    elif isinstance(value, list):
        _contract(len(value) <= 20)
        for child in value:
            _bounded_tree(child, depth=depth + 1)
    elif isinstance(value, str):
        _contract(0 < len(value) <= 128 and value.isascii())
        _contract(not any(ord(char) < 32 or ord(char) == 127 for char in value))
    elif isinstance(value, bool) or value is None:
        pass
    elif isinstance(value, (int, float)):
        _contract(math.isfinite(value) and abs(value) <= 10_000)
    else:
        _contract(False)


def _expected_dict(values: tuple[str, ...]) -> dict[str, str]:
    return dict(
        zip(
            ("validation_class", "claim_class", "responsiveness_class", "delta_class", "unknown_echo_class"),
            values,
            strict=True,
        )
    )


def _assert_payload_contract(payload: object, *, case_id: str, followup: bool) -> None:
    _contract(isinstance(payload, dict))
    allowed_keys = set(_PAYLOAD_KEYS)
    if case_id == "unknown_field_non_echo" and followup:
        allowed_keys.add("unknown_test_field")
    _contract(set(payload) == allowed_keys)
    _contract(isinstance(payload["cue_likelihoods"], dict))
    _contract(isinstance(payload["sequence"], dict))
    _contract(isinstance(payload["model"], dict))
    _contract(isinstance(payload["freshness"], dict))
    _contract(isinstance(payload["does_not_prove"], list))
    _contract(set(payload["cue_likelihoods"]) == {"warm_light", "daylight", "darkness"})
    _contract(set(payload["sequence"]) == {"first_frame_id", "last_frame_id", "frame_count", "temporal_window_ms"})
    _contract(set(payload["model"]) == {"name", "kind"})
    _contract(set(payload["freshness"]) == {"level"})
    _contract(payload["schema_version"] == 1 and not isinstance(payload["schema_version"], bool))
    _contract(payload["source_class"] == "camera_environment_estimate")
    _contract(payload["model"] == {"name": room_light.ROOM_LIGHT_MODEL_NAME, "kind": "heuristic"})
    expected_source = "vision_snapshot_processor" if case_id == "canonical_vision_snapshot_processor" else "camera_hub"
    _contract(payload["source"] == expected_source)

    numeric_type_exception = followup and case_id == "wrong_numeric_type"
    range_exception = followup and case_id == "out_of_range_numeric"
    _contract(isinstance(payload["confidence"], (int, float)) and not isinstance(payload["confidence"], bool) or numeric_type_exception)
    if numeric_type_exception:
        _contract(payload["confidence"] == "0.95")
    else:
        _contract(math.isfinite(payload["confidence"]))
        _contract(0.0 <= payload["confidence"] <= 1.0 or range_exception)
        _contract(payload["confidence"] == (1.25 if range_exception else 0.95))
    for likelihood in payload["cue_likelihoods"].values():
        _contract(isinstance(likelihood, (int, float)) and not isinstance(likelihood, bool))
        _contract(math.isfinite(likelihood) and 0.0 <= likelihood <= 1.0)

    sequence_type_exception = followup and case_id == "malformed_nested_sequence"
    for name in ("first_frame_id", "last_frame_id", "frame_count", "temporal_window_ms"):
        value = payload["sequence"][name]
        _contract(isinstance(value, int) and not isinstance(value, bool) or sequence_type_exception and name == "frame_count")
        if isinstance(value, int):
            _contract(0 <= value <= 10_000)

    if case_id == "canonical_vision_snapshot_processor":
        expected_sequence = {
            "first_frame_id": 22 if followup else 20,
            "last_frame_id": 23 if followup else 21,
            "frame_count": 2,
            "temporal_window_ms": 100,
        }
    else:
        expected_sequence = {
            "first_frame_id": 12 if followup else 10,
            "last_frame_id": 13 if followup else 11,
            "frame_count": "2" if sequence_type_exception else 2,
            "temporal_window_ms": 100,
        }
    _contract(payload["sequence"] == expected_sequence)

    expected_type = "ambient_environment_observation" if followup and case_id == "non_room_light" else "room_light_observation"
    if followup and case_id == "wrong_case":
        expected_bucket = "DARK"
    elif followup and case_id != "responsiveness_changed_identity_no_material_movement":
        expected_bucket = "dark"
    else:
        expected_bucket = "bright"
    expected_freshness = "stale" if followup and case_id == "stale_freshness" else "fresh"
    expected_proof = "physical_room_light_state" if followup and case_id == "wrong_proof_ceiling" else "camera_environment_estimate_only"
    expected_nonclaims = list(reversed(_DOES_NOT_PROVE)) if followup and case_id == "reversed_ordered_nonclaims" else _DOES_NOT_PROVE
    _contract(payload["type"] == expected_type)
    _contract(payload["observation_bucket"] == expected_bucket)
    _contract(payload["daylight_ambiguity"] == "low")
    _contract(payload["freshness"]["level"] == expected_freshness)
    _contract(payload["proof_ceiling"] == expected_proof)
    _contract(payload["does_not_prove"] == expected_nonclaims)
    no_movement = followup and case_id == "responsiveness_changed_identity_no_material_movement"
    expected_cues = (
        {"warm_light": 0.9, "daylight": 0.05, "darkness": 0.05}
        if not followup or no_movement
        else {"warm_light": 0.05, "daylight": 0.05, "darkness": 0.9}
    )
    _contract(payload["cue_likelihoods"] == expected_cues)
    _contract(payload["observed_at"] == ("2026-01-01T00:00:01Z" if followup else "2026-01-01T00:00:00Z"))
    id_prefix = "synthetic-vsp-" if case_id == "canonical_vision_snapshot_processor" else "synthetic-"
    id_number = "001" if not followup or case_id == "responsiveness_same_identity_material_movement" else "002"
    _contract(payload["observation_id"] == f"{id_prefix}observation-{id_number}")
    snapshot_number = "002" if followup else "001"
    _contract(payload["source_snapshot_id"] == f"{id_prefix}snapshot-{snapshot_number}")


def _assert_shared_vector_contract(document: dict[str, object]) -> None:
    _contract(set(document) == {"fixture_version", "fixture_kind", "unknown_field_sentinel", "cases"})
    _contract(document["fixture_version"] == "room-light-shared-vectors.v1")
    _contract(document["fixture_kind"] == "non_schema_test_vectors")
    _contract(document["unknown_field_sentinel"] == "fixed-unknown-room-light-sentinel-7e57")
    _contract(isinstance(document["cases"], list))
    cases = document["cases"]
    _contract([case.get("case_id") if isinstance(case, dict) else None for case in cases] == list(_ORDERED_CASE_IDS))
    _bounded_tree(document)
    for case in cases:
        case_id = case["case_id"]
        expected_row_keys = {"case_id", "baseline", "followup", "expected"}
        if case_id == "nonfinite_numeric":
            expected_row_keys.add("synthetic_numeric_class")
            _contract(case["synthetic_numeric_class"] == "followup_confidence_nan")
        _contract(set(case) == expected_row_keys)
        _contract(case["expected"] == _expected_dict(_EXPECTED_CLASSES[case_id]))
        _assert_payload_contract(case["baseline"], case_id=case_id, followup=False)
        _assert_payload_contract(case["followup"], case_id=case_id, followup=True)
        if case_id == "unknown_field_non_echo":
            _contract(case["followup"]["unknown_test_field"] == document["unknown_field_sentinel"])


def _vsp_observation(vector: dict[str, object]) -> RoomLightObservation:
    observed_at = datetime.fromisoformat(vector["observed_at"].replace("Z", "+00:00")).astimezone(UTC).timestamp()
    sequence = vector["sequence"]
    return RoomLightObservation(
        observation_bucket=vector["observation_bucket"],
        confidence=vector["confidence"],
        daylight_ambiguity=vector["daylight_ambiguity"],
        cue_likelihoods=dict(vector["cue_likelihoods"]),
        observed_at=observed_at,
        first_frame_id=sequence["first_frame_id"],
        last_frame_id=sequence["last_frame_id"],
        frame_count=sequence["frame_count"],
        temporal_window_ms=sequence["temporal_window_ms"],
        feature_summary={},
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
    def test_reset_prevents_pre_disconnect_frame_from_joining_new_generation(self) -> None:
        processor = RoomLightSnapshotProcessor(min_frames=2, window_ms=1000)
        frame = np.zeros((48, 64, 3), dtype=np.uint8)

        self.assertIsNone(processor.observe(frame, frame_id=1, stamp=1.0))
        processor.reset()
        self.assertIsNone(processor.observe(frame, frame_id=2, stamp=20.0))
        observation = processor.observe(frame, frame_id=3, stamp=20.1)

        self.assertIsNotNone(observation)
        self.assertEqual(observation.first_frame_id, 2)
        self.assertEqual(observation.last_frame_id, 3)

    def test_shared_vector_loader_failures_are_fixed_non_echo(self) -> None:
        original_present = _SHARED_VECTOR_ENV in os.environ
        original_value = os.environ.get(_SHARED_VECTOR_ENV)
        temporary_directory = tempfile.TemporaryDirectory()

        try:
            root = Path(temporary_directory.name)
            missing = root / "missing-fixture-os-detail-2f83.json"
            malformed = root / "malformed-fixture-case-4c91.json"
            oversized = root / "oversized-fixture-value-8a27.json"
            malformed.write_text(
                '{"case_id":"injected-case-5d31","value":"injected-value-9b62",'
                '"sentinel":"injected-sentinel-7f44"',
                encoding="utf-8",
            )
            oversized.write_bytes(b"x" * (_MAX_SHARED_VECTOR_BYTES + 1))
            unsafe = str(root / "unsafe-fixture-os-detail-6e18.json") + "\nunsafe-path-detail-3a75"

            failures = (
                (str(missing), _FIXTURE_UNAVAILABLE),
                (str(malformed), _FIXTURE_INVALID),
                (str(oversized), _FIXTURE_INVALID),
                (unsafe, _FIXTURE_INVALID),
            )
            forbidden = (
                str(missing), missing.name,
                str(malformed), malformed.name,
                str(oversized), oversized.name,
                unsafe, "unsafe-fixture-os-detail-6e18.json",
                str(_MAX_SHARED_VECTOR_BYTES + 1),
                "injected-case-5d31", "injected-value-9b62",
                "injected-sentinel-7f44", "unsafe-path-detail-3a75",
                "The system cannot find the file specified",
                "FileNotFoundError", "PermissionError", "JSONDecodeError",
            )

            for configured_path, expected_class in failures:
                os.environ[_SHARED_VECTOR_ENV] = configured_path
                try:
                    _load_shared_vectors(os.environ[_SHARED_VECTOR_ENV])
                except AssertionError as error:
                    self.assertEqual(error.args, (expected_class,))
                    self.assertTrue(error.__suppress_context__)
                    serialized = "".join(traceback.format_exception(error))
                else:
                    self.fail("room_light_fixture_failure_expected")

                self.assertEqual(serialized.count(expected_class), 1)
                for secret in forbidden:
                    self.assertNotIn(secret, serialized)
        finally:
            if original_present:
                os.environ[_SHARED_VECTOR_ENV] = original_value or ""
            else:
                os.environ.pop(_SHARED_VECTOR_ENV, None)
            temporary_directory.cleanup()

    def test_env_opt_in_shared_room_light_vectors(self) -> None:
        configured_path = os.environ.get(_SHARED_VECTOR_ENV)
        if configured_path is None:
            self.skipTest("shared_vector_env_not_configured")

        document = _load_shared_vectors(configured_path)
        _assert_shared_vector_contract(document)

        cases = {case["case_id"]: case for case in document["cases"]}
        vsp_case = cases["canonical_vision_snapshot_processor"]
        sentinel = document["unknown_field_sentinel"]
        expected_product_keys = _PAYLOAD_KEYS - {"source_snapshot_id", "freshness"}
        observations = []
        for phase in ("baseline", "followup"):
            vector = vsp_case[phase]
            observation = _vsp_observation(vector)
            observations.append(observation)
            produced = observation.to_payload()
            self.assertTrue(set(produced) == expected_product_keys, "vsp_product_contract_regression")
            for field in (
                "type", "schema_version", "observation_bucket", "confidence",
                "daylight_ambiguity", "cue_likelihoods", "source", "source_class",
                "sequence", "model", "proof_ceiling", "does_not_prove",
            ):
                self.assertTrue(produced[field] == vector[field], "vsp_product_contract_regression")
            self.assertTrue(sentinel not in json.dumps(produced, sort_keys=True), "vsp_unknown_echo_regression")
            self.assertTrue("unknown_test_field" not in produced, "vsp_unknown_echo_regression")

        summary = summarize_observations(
            observations,
            config=RoomLightEvaluationConfig(
                sample_id="vision.room_light.shared_vector",
                expected_observation_bucket="dark",
                max_frames=2,
            ),
            frames_seen=2,
            frames_sampled=2,
        )
        self.assertTrue(summary["result"] == "pass", "vsp_evaluator_contract_regression")
        self.assertTrue(summary["final"]["sequence"] == vsp_case["followup"]["sequence"], "vsp_evaluator_contract_regression")
        self.assertTrue(summary["final"]["does_not_prove"] == _DOES_NOT_PROVE, "vsp_evaluator_contract_regression")
        self.assertTrue(not summary["raw_media_included"], "vsp_evaluator_raw_echo_regression")
        self.assertTrue(not summary["raw_frame_included"], "vsp_evaluator_raw_echo_regression")
        self.assertTrue(not summary["source_path_included"], "vsp_evaluator_raw_echo_regression")
        self.assertTrue(sentinel not in json.dumps(summary, sort_keys=True), "vsp_unknown_echo_regression")

        malformed = copy.deepcopy(document)
        malformed["cases"] = "not-a-case-sequence"
        with self.assertRaisesRegex(AssertionError, "^room_light_fixture_invalid$"):
            _assert_shared_vector_contract(malformed)

        reordered = copy.deepcopy(document)
        reordered["cases"][0], reordered["cases"][1] = reordered["cases"][1], reordered["cases"][0]
        with self.assertRaisesRegex(AssertionError, "^room_light_fixture_invalid$"):
            _assert_shared_vector_contract(reordered)

        extra_field = copy.deepcopy(document)
        extra_field["cases"][0]["unexpected"] = "bounded-synthetic-value"
        with self.assertRaisesRegex(AssertionError, "^room_light_fixture_invalid$"):
            _assert_shared_vector_contract(extra_field)

        changed_nonclaim = copy.deepcopy(document)
        changed_nonclaim["cases"][0]["baseline"]["does_not_prove"] = ["physical_room_light_state"]
        with self.assertRaisesRegex(AssertionError, "^room_light_fixture_invalid$"):
            _assert_shared_vector_contract(changed_nonclaim)

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
