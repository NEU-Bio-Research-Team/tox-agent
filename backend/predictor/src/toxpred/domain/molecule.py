"""Molecule value object. Chemistry toolkits stay out of the domain layer."""
from __future__ import annotations

from dataclasses import dataclass


class InvalidSmilesError(ValueError):
    """Raised for input that is not a parseable, non-empty molecule.

    Mapped to a typed 400 at the API boundary. It is never absorbed into a
    zero-probability prediction: the previous implementation returned
    ``{"label": "PARSE_ERROR", "p_toxic": 0.0}``, a shape a caller can easily
    read as "predicted non-toxic".
    """

    def __init__(self, smiles: str, reason: str) -> None:
        super().__init__(f"invalid SMILES {smiles!r}: {reason}")
        self.smiles = smiles
        self.reason = reason


@dataclass(frozen=True, slots=True)
class Molecule:
    input_smiles: str
    canonical_smiles: str
    elements: frozenset[str]
    num_atoms: int
