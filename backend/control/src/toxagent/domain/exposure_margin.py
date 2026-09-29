"""The hERG exposure margin: a potency over a free plasma concentration (W9-13).

RETHINK §4.7/§4.10: "a new calculation" is exactly what a skill cannot add. The
``assess-conflicting-evidence`` reference already teaches how to read an
IC50 against exposure (the 30-fold margin of Redfern et al. 2003, Cardiovasc Res
58(1):32–45); before this module the model had to do the division in its head
and could not cite the result. Now the server computes it, deterministically,
and the result is an observation a claim can cite.

What it will not do:

* **Invent an input.** Both concentrations must come from a source the session
  holds — a context item the researcher supplied or an evidence record — and
  the number given must appear in that source's text as written
  (``transcription_check``). A value the model recalls is not accepted.
* **Guess a molecular weight.** Molar units only; a mass concentration needs a
  conversion this module does not pretend to know.
* **Issue a verdict.** A margin is a ratio of two measurements under their own
  assay and exposure conditions; the result carries
  ``screening_not_safety_assessment``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

METHOD_VERSION = "exposure-margin-v1"

#: To nanomolar. ``uM`` is accepted for keyboards without µ.
_TO_NM = {"pM": Decimal("0.001"), "nM": Decimal(1), "µM": Decimal(1000), "uM": Decimal(1000),
          "mM": Decimal(1_000_000), "M": Decimal(1_000_000_000)}
UNITS = tuple(_TO_NM)


class InvalidMarginInput(ValueError):
    """Refused; the message is written for the model that sent it."""


@dataclass(frozen=True, slots=True)
class Concentration:
    value: float
    unit: str
    source_ref: str

    def nanomolar(self) -> Decimal:
        if self.unit not in _TO_NM:
            raise InvalidMarginInput(f"unit must be one of {list(UNITS)}; got {self.unit!r}")
        if not self.value > 0:
            raise InvalidMarginInput("a concentration must be greater than zero")
        return Decimal(repr(float(self.value))) * _TO_NM[self.unit]


def _number_forms(value: float) -> set[str]:
    text = format(Decimal(repr(float(value))).normalize(), "f")
    forms = {text, text.replace(".", ",")}
    if "." not in text:
        forms |= {f"{text}.0", f"{text},0"}
    return forms


def transcription_check(value: float, source_text: str) -> bool:
    """Whether ``value`` is written in ``source_text`` (as a whole number token)."""
    for form in _number_forms(value):
        if re.search(rf"(?<![\d.,]){re.escape(form)}(?![\d])", source_text):
            return True
    return False


def compute(ic50: Concentration, cmax: Concentration, *,
            fraction_unbound: float | None = None, fu_source_ref: str | None = None) -> dict[str, Any]:
    """The margin and everything it was computed from. Pure."""
    ic50_nm = ic50.nanomolar()
    cmax_nm = cmax.nanomolar()
    if fraction_unbound is not None:
        if not 0 < fraction_unbound <= 1:
            raise InvalidMarginInput("fraction_unbound must be in (0, 1]")
        free_nm = cmax_nm * Decimal(repr(float(fraction_unbound)))
        cmax_kind = "total, corrected by fraction unbound"
    else:
        free_nm = cmax_nm
        cmax_kind = "free"
    margin = float(ic50_nm / free_nm)
    return {
        "method_version": METHOD_VERSION,
        "formula": "margin = IC50 / free Cmax (both in nM)",
        "margin": round(margin, 6),
        "ic50_nM": float(ic50_nm),
        "free_cmax_nM": float(free_nm),
        "inputs": {
            "ic50": {"value": ic50.value, "unit": ic50.unit, "source_ref": ic50.source_ref},
            "cmax": {"value": cmax.value, "unit": cmax.unit, "source_ref": cmax.source_ref,
                     "kind": cmax_kind},
            "fraction_unbound": (
                {"value": fraction_unbound, "source_ref": fu_source_ref}
                if fraction_unbound is not None else None
            ),
        },
        "reading": (
            "A ratio of two measurements, each under its own assay and exposure conditions. "
            "It is not a safety verdict; how large a margin is reassuring is a judgement "
            "about those conditions."
        ),
    }
