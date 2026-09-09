"""Versioned calibration artifacts; policy thresholds remain a separate layer."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Protocol

from .artifacts import ArtifactError


class Calibrator(Protocol):
    method: str
    def calibrate(self, raw_probability: float, *, task: str | None = None) -> float: ...


def _logit(probability: float) -> float:
    p = min(max(float(probability), 1e-7), 1 - 1e-7)
    return math.log(p / (1 - p))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


@dataclass(frozen=True, slots=True)
class PlattCalibrator:
    slope: float
    intercept: float
    method: str = "platt"

    def calibrate(self, raw_probability: float, *, task: str | None = None) -> float:
        return _sigmoid(self.slope * _logit(raw_probability) + self.intercept)


@dataclass(frozen=True, slots=True)
class TemperatureCalibrator:
    temperature: float
    method: str = "temperature"

    def __post_init__(self) -> None:
        if self.temperature <= 0 or not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite and positive")

    def calibrate(self, raw_probability: float, *, task: str | None = None) -> float:
        return _sigmoid(_logit(raw_probability) / self.temperature)


@dataclass(frozen=True, slots=True)
class PerTaskCalibrator:
    calibrators: Mapping[str, Calibrator]
    method: str = "per_task"

    def calibrate(self, raw_probability: float, *, task: str | None = None) -> float:
        if task is None or task not in self.calibrators:
            raise ArtifactError(f"calibrator has no parameters for task {task!r}")
        return self.calibrators[task].calibrate(raw_probability, task=task)


@dataclass(frozen=True, slots=True)
class CalibrationArtifact:
    schema_version: str
    model_id: str
    endpoint: str
    method: str
    parameters: Mapping[str, object]
    calibration_split_sha256: str
    fitted_at: datetime
    metrics_before: Mapping[str, float]
    metrics_after: Mapping[str, float]

    def canonical_json(self) -> str:
        return json.dumps({
            "schema_version": self.schema_version, "model_id": self.model_id,
            "endpoint": self.endpoint, "method": self.method,
            "parameters": self.parameters,
            "calibration_split_sha256": self.calibration_split_sha256,
            "fitted_at": self.fitted_at.isoformat(), "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after,
        }, sort_keys=True, separators=(",", ":"))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


def load_calibration_artifact(path: Path) -> CalibrationArtifact:
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != "toxpred-calibration-v1":
        raise ArtifactError("unsupported calibration artifact schema")
    return CalibrationArtifact(
        schema_version=data["schema_version"], model_id=data["model_id"],
        endpoint=data["endpoint"], method=data["method"], parameters=data["parameters"],
        calibration_split_sha256=data["calibration_split_sha256"],
        fitted_at=datetime.fromisoformat(data["fitted_at"]),
        metrics_before=data["metrics_before"], metrics_after=data["metrics_after"],
    )
