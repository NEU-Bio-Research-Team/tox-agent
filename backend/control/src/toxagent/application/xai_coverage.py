"""How much of an attribution actually landed on the molecule.

P1-6 of the 2026-09-13 audit: the CCO/hERG explanation put 35.84% of its total
importance mass on tokens that are not part of the structure at all — the
tokenizer's ``[CLS]`` and ``[SEP]`` markers — and the report described it as
having no unmapped mass. A reader looking at the highlighted atoms had no way
to know that a third of what moved the model is not in the picture.

``unmapped_importance`` already carried that number. What it could not say is
*what* the unmapped mass was. Two cases look identical in one float and mean
different things:

**Special tokens.** The model's attention on its own sequence markers. This is
an artefact of how the model reads SMILES, not a chemical statement, and a high
fraction is a reason to distrust the picture rather than to reason about the
molecule.

**Everything else.** Structural characters the aligner could not attach to an
atom or a bond — ring closures, branch parentheses, stereo markers. These are
about the structure but not about one atom.

Nothing here renormalizes. Dropping the special tokens and rescaling the rest
to 100% is the tempting move and it is exactly wrong: it produces a picture
that looks fully explained and quietly deletes the evidence that it is not.
The relative-importance-among-mapped-atoms view exists as a *separate*,
explicitly named number for presentation, and the raw fractions travel with it
everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

#: Tokens a SMILES tokenizer adds that are not part of the molecule. Matched
#: case-insensitively, and the bracket shape is checked too: a lone ``[C@H]``
#: is a real atom token and must never be classified as a marker.
SPECIAL_TOKEN_NAMES = frozenset(
    {"[cls]", "[sep]", "[pad]", "[unk]", "[mask]", "[bos]", "[eos]", "<s>", "</s>", "<pad>"}
)

#: Above this, the picture is not a fair summary of what moved the model.
LOW_COVERAGE_BELOW = 0.50
LIMITED_COVERAGE_BELOW = 0.80

#: Bumped whenever a threshold or the classification rule changes, so an
#: explanation records which policy produced its status rather than being
#: re-judged years later under different numbers.
COVERAGE_POLICY_VERSION = "xai-coverage-1"


def is_special_token(token: Any) -> bool:
    if not isinstance(token, str):
        return False
    return token.strip().lower() in SPECIAL_TOKEN_NAMES


@dataclass(frozen=True, slots=True)
class ExplanationCoverage:
    """The accounting of one attribution's importance mass.

    The three fractions sum to 1.0 within floating-point tolerance whenever the
    payload reported enough to compute them. When it did not, they are ``None``
    — "the explainer did not tell us" and "none of the mass was unmapped" are
    different facts, and only one of them is a zero.
    """

    mapped_importance_fraction: float | None
    special_token_importance_fraction: float | None
    other_unmapped_importance_fraction: float | None
    unmapped_importance: float | None
    coverage_status: str
    policy_version: str = COVERAGE_POLICY_VERSION
    positive_contributor_count: int = 0
    negative_contributor_count: int = 0

    @property
    def is_known(self) -> bool:
        return self.mapped_importance_fraction is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mapped_importance_fraction": self.mapped_importance_fraction,
            "special_token_importance_fraction": self.special_token_importance_fraction,
            "other_unmapped_importance_fraction": self.other_unmapped_importance_fraction,
            "unmapped_importance": self.unmapped_importance,
            "coverage_status": self.coverage_status,
            "coverage_policy_version": self.policy_version,
            "positive_contributor_count": self.positive_contributor_count,
            "negative_contributor_count": self.negative_contributor_count,
        }

    def summary_sentence(self) -> str:
        """The one sentence every section must reuse rather than restate.

        The compiler owns this string. A section that writes its own version of
        it is how an executive summary came to deny its own explanation data
        (P0-2), so there is exactly one place this wording is produced.
        """
        if not self.is_known:
            return (
                "The explainer did not report how much of the attribution mass "
                "reached the structure, so the coverage of this picture is unknown."
            )
        mapped = (self.mapped_importance_fraction or 0.0) * 100
        special = (self.special_token_importance_fraction or 0.0) * 100
        contributors = self.positive_contributor_count + self.negative_contributor_count
        parts = [
            f"{mapped:.1f}% of the attribution mass maps to atoms or bonds",
            f"{contributors} contributor(s) are shown "
            f"({self.positive_contributor_count} positive, "
            f"{self.negative_contributor_count} negative)",
        ]
        if special > 0:
            parts.append(
                f"{special:.1f}% falls on the tokenizer's own sequence markers, "
                "which are not part of the molecule"
            )
        other = (self.other_unmapped_importance_fraction or 0.0) * 100
        if other > 0:
            parts.append(
                f"{other:.1f}% falls on structural characters that could not be "
                "attached to a single atom or bond"
            )
        return "; ".join(parts) + "."


def classify_coverage(mapped_fraction: float | None) -> str:
    """``high`` | ``limited`` | ``low`` | ``unknown``."""
    if mapped_fraction is None:
        return "unknown"
    if mapped_fraction < LOW_COVERAGE_BELOW:
        return "low"
    if mapped_fraction < LIMITED_COVERAGE_BELOW:
        return "limited"
    return "high"


def _importance(item: Mapping[str, Any]) -> float | None:
    for field in ("importance", "magnitude"):
        value = item.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value == value:
            return abs(float(value))
    return None


def compute_coverage(
    payload: Mapping[str, Any],
    *,
    positive_contributor_count: int = 0,
    negative_contributor_count: int = 0,
) -> ExplanationCoverage:
    """Derive the coverage accounting from one explainer payload.

    ``unmapped_importance`` is taken from the payload verbatim — the predictor
    computed it against the same denominator it used for every
    ``relative_importance``, and recomputing it here from the tokens would give
    a second, subtly different number for the same fact.

    The *split* of that unmapped mass is computed here, because only the token
    list knows which unmapped tokens were sequence markers.
    """
    unmapped = payload.get("unmapped_importance")
    unmapped = (
        float(unmapped)
        if isinstance(unmapped, (int, float)) and not isinstance(unmapped, bool)
        else None
    )
    if unmapped is None:
        return ExplanationCoverage(
            mapped_importance_fraction=None,
            special_token_importance_fraction=None,
            other_unmapped_importance_fraction=None,
            unmapped_importance=None,
            coverage_status="unknown",
            positive_contributor_count=positive_contributor_count,
            negative_contributor_count=negative_contributor_count,
        )

    mapped_fraction = max(0.0, 1.0 - unmapped)

    tokens: Sequence[Mapping[str, Any]] = payload.get("tokens") or ()
    total_mass = 0.0
    special_mass = 0.0
    for token in tokens:
        if not isinstance(token, Mapping):
            continue
        importance = _importance(token)
        if importance is None:
            continue
        total_mass += importance
        if is_special_token(token.get("token")):
            special_mass += importance

    if total_mass > 0:
        special_fraction = min(special_mass / total_mass, unmapped)
    else:
        # No token list: the mass is unmapped but we cannot say what it fell
        # on. Attributing all of it to special tokens would be a guess, and
        # attributing none of it would be a different guess — say neither.
        special_fraction = None

    other_fraction = None if special_fraction is None else max(0.0, unmapped - special_fraction)

    return ExplanationCoverage(
        mapped_importance_fraction=mapped_fraction,
        special_token_importance_fraction=special_fraction,
        other_unmapped_importance_fraction=other_fraction,
        unmapped_importance=unmapped,
        coverage_status=classify_coverage(mapped_fraction),
        positive_contributor_count=positive_contributor_count,
        negative_contributor_count=negative_contributor_count,
    )


def mapped_only_relative_importance(
    atoms: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Atom shares renormalized over the mapped mass only — for drawing.

    Explicitly named and explicitly separate. A figure has to allocate colour
    over the atoms it can draw, and this is that allocation; it is never the
    provenance number, and the raw fractions travel alongside it so a caption
    can say what share of the whole this view represents.
    """
    scored = [
        (atom, _importance(atom))
        for atom in atoms
        if isinstance(atom, Mapping) and _importance(atom) is not None
    ]
    total = sum(value for _, value in scored)
    if total <= 0:
        return [
            {**dict(atom), "mapped_only_relative_importance": 0.0} for atom, _ in scored
        ]
    return [
        {**dict(atom), "mapped_only_relative_importance": value / total}
        for atom, value in scored
    ]


def rank_atoms_by_absolute_total(
    atoms: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    """Atoms ordered by how much they moved the model, direction ignored.

    Ranking by signed value puts the strongest *negative* contributor last,
    which reads as "least important" when it is the opposite.
    """
    return sorted(
        (atom for atom in atoms if isinstance(atom, Mapping)),
        key=lambda atom: (-(_importance(atom) or 0.0), atom.get("atom_index") or 0),
    )
