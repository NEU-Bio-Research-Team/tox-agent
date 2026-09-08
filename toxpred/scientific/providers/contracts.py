"""Immutable outputs shared by scientific providers and the application layer.

Provider ownership is kept by :class:`ProviderBatchResult`; rows are never
flattened together.  Mapping compatibility is intentionally read-only so the
frozen numerical benchmark can continue to inspect fields while callers move
to typed attributes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Generic, Iterator, Mapping, Sequence, TypeVar

from ...domain.endpoints import TOX21_TASKS
from ..artifacts import ArtifactError


def validated_probability(value: float, *, model_id: str, field: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ArtifactError(
            f"[{model_id}] {field} must be a finite probability in [0, 1], got {value!r}"
        )
    return value


class _ReadOnlyRow(Mapping[str, object]):
    """A transitional, immutable mapping view for benchmark compatibility."""

    def _mapping(self) -> Mapping[str, object]:
        raise NotImplementedError

    def __getitem__(self, key: str) -> object:
        return self._mapping()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._mapping())

    def __len__(self) -> int:
        return len(self._mapping())


@dataclass(frozen=True, slots=True)
class TokenizationProvenance:
    input_token_count: int
    encoded_token_count: int
    max_length: int
    truncated: bool

    def __post_init__(self) -> None:
        if self.input_token_count < 0 or self.encoded_token_count < 0:
            raise ValueError("token counts cannot be negative")
        if self.max_length <= 0 or self.encoded_token_count > self.max_length:
            raise ValueError("encoded token count must not exceed max_length")
        if self.truncated != (self.input_token_count > self.max_length):
            raise ValueError("truncated must reflect the individual input token count")

    def to_dict(self) -> dict[str, object]:
        return {
            "input_token_count": self.input_token_count,
            "encoded_token_count": self.encoded_token_count,
            "max_length": self.max_length,
            "truncated": self.truncated,
        }


@dataclass(frozen=True, slots=True)
class HergTox21RawOutput(_ReadOnlyRow):
    model_id: str
    herg_probability_blocker: float
    tox21_probability_activity: Mapping[str, float]
    tokenization: TokenizationProvenance

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "herg_probability_blocker",
            validated_probability(
                self.herg_probability_blocker,
                model_id=self.model_id,
                field="herg_probability_blocker",
            ),
        )
        tasks = set(self.tox21_probability_activity)
        if tasks != set(TOX21_TASKS):
            missing = sorted(set(TOX21_TASKS) - tasks)
            extra = sorted(tasks - set(TOX21_TASKS))
            raise ArtifactError(
                f"[{self.model_id}] malformed Tox21 output; missing={missing}, extra={extra}"
            )
        values = {
            task: validated_probability(
                self.tox21_probability_activity[task],
                model_id=self.model_id,
                field=f"tox21_probability_activity.{task}",
            )
            for task in TOX21_TASKS
        }
        object.__setattr__(self, "tox21_probability_activity", values)

    def _mapping(self) -> Mapping[str, object]:
        # n_tokens/truncated remain as deprecated read aliases for frozen v1
        # benchmark readers; v2 consumers use ``tokenization``.
        return {
            "model_id": self.model_id,
            "herg_probability_blocker": self.herg_probability_blocker,
            "tox21_probability_activity": self.tox21_probability_activity,
            "input_token_count": self.tokenization.input_token_count,
            "encoded_token_count": self.tokenization.encoded_token_count,
            "max_length": self.tokenization.max_length,
            "n_tokens": self.tokenization.encoded_token_count,
            "truncated": self.tokenization.truncated,
        }


@dataclass(frozen=True, slots=True)
class ClinToxRawOutput(_ReadOnlyRow):
    model_id: str
    clintox_probability_toxicity: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "clintox_probability_toxicity",
            validated_probability(
                self.clintox_probability_toxicity,
                model_id=self.model_id,
                field="clintox_probability_toxicity",
            ),
        )

    def _mapping(self) -> Mapping[str, object]:
        return {
            "model_id": self.model_id,
            "clintox_probability_toxicity": self.clintox_probability_toxicity,
        }


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ProviderBatchResult(Generic[T], Sequence[T]):
    provider_id: str
    rows: tuple[T, ...]

    def __getitem__(self, index):  # type: ignore[no-untyped-def]
        return self.rows[index]

    def __len__(self) -> int:
        return len(self.rows)

    def validate_count(self, expected: int) -> "ProviderBatchResult[T]":
        if len(self.rows) != expected:
            raise ArtifactError(
                f"[{self.provider_id}] returned {len(self.rows)} rows for {expected} molecules"
            )
        return self
