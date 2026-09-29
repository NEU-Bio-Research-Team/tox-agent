"""Resolve a v2 draft into the internal candidate shape (P1-4).

The server, not the model, owns three things here:

* **identity** — the ``clm_`` id is issued here, from the same generator every
  other domain id comes from. A model's ``local_ref`` is a label it uses to
  point at its own claims, and it never leaves this function;
* **the value** — a numeric or classification claim's ``source_value`` is read
  out of the immutable observation the claim names, not copied from the draft;
* **the rendering** — ``rendered_value`` is produced from that value under the
  declared transform and the run's locale, so a Vietnamese decimal comma is a
  presentation rule applied once rather than a string a model has to get right.

What comes out is an ordinary ``GroundedAnswerCandidate``: every validator
downstream keeps working unchanged, and the numeric checks it runs become
tautological for field-backed claims — which is the point. A check that can no
longer fail has been replaced by a construction that cannot go wrong.

Resolution can still fail, and when it does it fails the way everything else in
this package does: typed, correctable violations naming the exact path.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping

from ...domain.errors import Violation
from ...domain.ids import CLAIM, new_id
from .candidate_wire import ClaimCandidate, GroundedAnswerCandidate, RecommendationCandidate
from .draft_wire import FIELD_BACKED, GroundedAnswerDraftV2

#: Locales whose decimal separator is a comma. The rendering rule lives here
#: rather than in a prompt, because "write 0,731 not 0.731" is a formatting
#: convention and a model applying it by hand is a number it can mistype.
_COMMA_DECIMAL_LANGUAGES = frozenset({"vi"})

#: ``{{herg_p}}`` in answer_markdown: "put this claim's value here". The server
#: renders a claim's value, so it is the only party that knows the exact string;
#: the model names the claim and the server writes the number (W9-02).
_PLACEHOLDER = re.compile(r"\{\{\s*([a-z][a-z0-9_]{0,31})\s*\}\}")


@dataclass(frozen=True)
class ResolvedDraft:
    candidate: GroundedAnswerCandidate | None
    violations: tuple[Violation, ...] = ()
    #: local_ref -> the id the server issued for it. Useful to a caller that
    #: wants to report back which claim a violation belongs to in the model's
    #: own vocabulary.
    issued_ids: Mapping[str, str] = None  # type: ignore[assignment]
    #: The numeric values the server resolved for this draft's claims, for the
    #: prose coverage check (``coverage.validate_markdown_numeric_coverage``).
    claimed_values: tuple[float, ...] = ()

    @property
    def ok(self) -> bool:
        return self.candidate is not None and not self.violations


def render_number(value: float, transform: str, *, language: str) -> str:
    """The canonical rendering of one value under one transform.

    Deliberately total over the transform allowlist, and deliberately plain:
    no thousands separators, no units, no "0,0315 (3,15%)" compounds. ADR 0005
    says the server renders the canonical value and prose carries the phrasing,
    and a renderer that starts formatting for display is how the two drift.
    """
    if transform.startswith("percent:"):
        digits = int(transform.split(":")[1])
        text = f"{value * 100.0:.{digits}f}%"
    elif transform.startswith("round:"):
        digits = int(transform.split(":")[1])
        text = f"{value:.{digits}f}"
    else:
        # identity, difference, ratio: the value as it is, without inventing a
        # precision the source did not have.
        text = _plain(value)
    if language in _COMMA_DECIMAL_LANGUAGES:
        text = text.replace(".", ",")
    return text


#: Binary floating point cannot hold 0.7312 - 0.2 exactly; it holds
#: 0.5311999999999999. Printing that is not extra honesty, it is noise from the
#: representation being shown as though it were measured precision. Twelve
#: decimal places is far beyond anything a predictor reports and far short of
#: where the artefact lives, so rounding here removes the artefact and cannot
#: remove a real digit.
_REPRESENTATION_NOISE_DIGITS = 12


def _plain(value: float) -> str:
    """A float without scientific notation or a trailing ``.0`` it never had."""
    value = round(float(value), _REPRESENTATION_NOISE_DIGITS)
    if value.is_integer():
        return str(int(value))
    return format(Decimal(repr(value)).normalize(), "f")


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _resolve_claim(
    claim,
    *,
    index: int,
    observations_by_id: Mapping[str, Any],
    claim_id: str,
    ids_by_ref: Mapping[str, str],
    language: str,
) -> tuple[ClaimCandidate | None, list[Violation]]:
    path = f"claims[{index}]"
    violations: list[Violation] = []
    source_value: Any = None
    rendered_value: str | None = None

    if claim.kind in FIELD_BACKED:
        if not claim.observation_id:
            violations.append(
                Violation(
                    "claim_observation_missing",
                    f"a {claim.kind} claim must name the observation its value comes from",
                    path=f"{path}.observation_id",
                )
            )
        else:
            observation = observations_by_id.get(claim.observation_id)
            if observation is None:
                violations.append(
                    Violation(
                        "claim_observation_not_found",
                        f"observation {claim.observation_id!r} was not read by this run",
                        path=f"{path}.observation_id",
                        actual=claim.observation_id,
                    )
                )
            else:
                try:
                    source_value = observation.value_at(claim.field_path)
                except Exception as exc:  # noqa: BLE001 - surfaced as a violation
                    violations.append(
                        Violation(
                            "claim_field_path_unresolvable",
                            str(exc),
                            path=f"{path}.field_path",
                            actual=claim.field_path,
                        )
                    )
                else:
                    number = _numeric(source_value)
                    if claim.kind == "numeric":
                        if number is None:
                            violations.append(
                                Violation(
                                    "claim_field_not_numeric",
                                    f"{claim.field_path} is not a numeric field",
                                    path=f"{path}.field_path",
                                    actual=source_value,
                                )
                            )
                        else:
                            rendered_value = render_number(
                                number, claim.transform, language=language
                            )
                    else:
                        rendered_value = str(source_value)

    if violations:
        return None, violations

    return (
        ClaimCandidate(
            claim_id=claim_id,
            kind=claim.kind,
            text=claim.text,
            observation_id=claim.observation_id,
            field_path=claim.field_path,
            source_value=source_value,
            rendered_value=rendered_value,
            transform=claim.transform,
            citation_ids=list(claim.citation_ids),
            input_claim_ids=[ids_by_ref[ref] for ref in claim.input_local_refs],
        ),
        [],
    )


def resolve_draft(
    draft: GroundedAnswerDraftV2,
    *,
    observations_by_id: Mapping[str, Any],
    language: str = "en",
) -> ResolvedDraft:
    """Turn a v2 draft into the internal v1 candidate, issuing ids and values.

    Ids are issued for every claim before any of them is resolved, so a
    comparison claim can name its inputs regardless of the order the model
    listed them in.
    """
    ids_by_ref = {claim.local_ref: new_id(CLAIM) for claim in draft.claims}

    claims: list[ClaimCandidate] = []
    violations: list[Violation] = []
    for index, claim in enumerate(draft.claims):
        resolved, claim_violations = _resolve_claim(
            claim,
            index=index,
            observations_by_id=observations_by_id,
            claim_id=ids_by_ref[claim.local_ref],
            ids_by_ref=ids_by_ref,
            language=language,
        )
        violations.extend(claim_violations)
        if resolved is not None:
            claims.append(resolved)

    # A comparison is computed, not submitted: its inputs may appear after it
    # in the draft, so it is resolved once every other claim has a value.
    by_id = {claim.claim_id: claim for claim in claims}
    for index, claim in enumerate(draft.claims):
        if claim.kind != "comparison" or claim.transform not in {"difference", "ratio"}:
            continue
        resolved = by_id.get(ids_by_ref[claim.local_ref])
        if resolved is None:
            continue
        path = f"claims[{index}]"
        if len(claim.input_local_refs) != 2:
            violations.append(
                Violation(
                    "claim_derived_inputs_invalid",
                    f"a {claim.transform} claim needs exactly two input_local_refs, got "
                    f"{len(claim.input_local_refs)}",
                    path=f"{path}.input_local_refs",
                )
            )
            continue
        inputs = [by_id.get(ids_by_ref[ref]) for ref in claim.input_local_refs]
        if any(item is None or item.source_value is None for item in inputs):
            violations.append(
                Violation(
                    "claim_derived_input_missing",
                    f"claim {claim.local_ref!r} compares claims that carry no value",
                    path=f"{path}.input_local_refs",
                )
            )
            continue
        value = derived_value(claim.transform, inputs[0].source_value, inputs[1].source_value)
        if value is None:
            violations.append(
                Violation(
                    "claim_derived_division_by_zero"
                    if claim.transform == "ratio"
                    else "claim_derived_inputs_invalid",
                    f"the {claim.transform} of these two claims is not a number",
                    path=f"{path}.input_local_refs",
                )
            )
            continue
        claims[claims.index(resolved)] = replace_claim(
            resolved,
            source_value=value,
            rendered_value=render_number(value, claim.transform, language=language),
        )

    # Read from ``claims``, not ``by_id``: a comparison was replaced above.
    rendered_by_ref = {
        ref: next((c.rendered_value for c in claims if c.claim_id == claim_id), None)
        for ref, claim_id in ids_by_ref.items()
    }
    answer_markdown, placeholder_violations = _fill_placeholders(
        draft.answer_markdown, rendered_by_ref,
        {claim.local_ref: claim.text for claim in draft.claims},
    )
    violations.extend(placeholder_violations)

    if violations:
        return ResolvedDraft(candidate=None, violations=tuple(violations), issued_ids=ids_by_ref)

    claimed_values = tuple(
        float(claim.source_value) for claim in claims
        if claim.kind in ("numeric", "comparison") and _numeric(claim.source_value) is not None
    )
    candidate = GroundedAnswerCandidate(
        answer_markdown=answer_markdown,
        claims=claims,
        limitations=list(draft.limitations),
        recommended_next_steps=[
            RecommendationCandidate(
                text=step.text,
                basis_claim_ids=[ids_by_ref[ref] for ref in step.basis_local_refs],
            )
            for step in draft.recommended_next_steps
        ],
    )
    return ResolvedDraft(
        candidate=candidate, violations=(), issued_ids=ids_by_ref, claimed_values=claimed_values,
    )


def _fill_placeholders(
    markdown: str, rendered_by_ref: Mapping[str, str | None], text_by_ref: Mapping[str, str],
) -> tuple[str, list[Violation]]:
    """Replace every ``{{local_ref}}`` with that claim's server-rendered value,
    or, for a claim that carries no value, with the claim's own text.

    The second half was learnt from a measurement (W9-A, C + answer_draft_v2):
    models read "write its local_ref in double braces" as "place this claim
    here" and marked scientific and limitation claims too; refusing those cost
    26 first drafts. Their text is what the model asserted and cited, and it is
    checked by every wording rule once it is in the prose, so putting it where
    the model marked it is what the draft meant.
    """
    violations: list[Violation] = []

    def fill(match: re.Match) -> str:
        ref = match.group(1)
        if ref not in rendered_by_ref:
            violations.append(Violation(
                "answer_placeholder_unknown",
                f"answer_markdown names {{{{{ref}}}}} but no claim in this draft has that local_ref",
                path="answer_markdown", actual=match.group(0),
            ))
            return match.group(0)
        value = rendered_by_ref[ref]
        return value if value is not None else text_by_ref[ref]

    return _PLACEHOLDER.sub(fill, markdown), violations


def replace_claim(claim: ClaimCandidate, **changes: Any) -> ClaimCandidate:
    """A copy with fields replaced. ``ClaimCandidate`` is a pydantic model, so
    this is ``model_copy``, named for what it does at the call site."""
    return claim.model_copy(update=changes)


def derived_value(transform: str, first: Any, second: Any) -> float | None:
    """The value a comparison claim carries, computed rather than submitted.

    ``None`` when the inputs are not both numbers, or when a ratio would divide
    by zero — the caller turns that into a violation rather than into an
    infinity a renderer would happily print.
    """
    left, right = _numeric(first), _numeric(second)
    if left is None or right is None:
        return None
    if transform == "difference":
        return left - right
    if transform == "ratio":
        return None if right == 0 else left / right
    return None
