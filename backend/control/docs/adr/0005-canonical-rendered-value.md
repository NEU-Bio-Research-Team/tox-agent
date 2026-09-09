# ADR 0005 — `rendered_value` is one canonical number

**Status:** accepted · **Date:** 2026-09-04 · **Plan:** §5.7, §9.1.5 (DEC-08)

## Context

The first live Phase 3 report-Q&A run (`run_5e5875e4…`) fell back to the
deterministic answer. One of the reasons both candidates were rejected: with
`preferred_language: vi` the model rendered a numeric claim's `rendered_value`
as `"0,0315 (3,15%)"` — the raw probability and its percentage in one string.
`validation/numeric.py` could not parse it and returned
`claim_rendered_value_unparseable`.

The plan already constrains this field. §9.1.5 says `rendered_value` "matches
the transform", and the §5.7 example shows a bare `"0.731"`. A string that
carries both a `round:n` value and a `percent:n` value at once matches neither
transform, so it was never valid — the parser was simply lax about a decimal
comma and silent about everything else.

Two options were considered:

1. **Loosen the parser** to extract the leading number from a compound render.
2. **Constrain `rendered_value` to one canonical number** and reject anything
   else with a correction message.

## Decision

Option 2.

A numeric claim's `rendered_value` is a **single number**: an optional leading
sign, digits, one optional decimal separator that may be a dot **or** a
Vietnamese comma, an optional trailing `%`. Regex: `^-?\d+(?:[.,]\d+)?%?$`.

- No thousands separator, no spaces, no units, no parenthetical, no words.
- Display phrasing such as `"0,0315 (3,15%)"` belongs in the claim's `text`
  and in `answer_markdown`, which are prose and are graded as prose.
- A non-canonical render is returned as `claim_rendered_value_unparseable`
  whose message names the fix ("move the phrasing into the claim text"),
  spending the run's one correction attempt on a stated, mechanical error.
- The rule is also stated up front — in the `submit_grounded_answer` tool
  description and in the `ANSWER_FORMAT` block of every system prompt — so a
  correction attempt is rarely needed.

## Why not option 1

`numeric.py` exists so that "a number in an accepted answer equals the
canonical field it cites, exactly, under a declared and checkable transform …
no string-replacement fix-up of a wrong answer". Silently taking the first
token of `"0,0315 (3,15%)"` is exactly that fix-up: it discards half of what
the model wrote and would never catch a `(3,15%)` half that disagreed with the
`0,0315` half. Keeping `rendered_value` to one number keeps the check
mechanical and total.

## Scope

Numeric claims only. A `classification` claim's `rendered_value` is still the
raw enum/label string it cites (`classification.py`), unaffected by this ADR.

## Verification

`tests/unit/test_validation_numeric.py` covers the canonical accept set, the
compound/annotated reject set (including the exact `"0,0315 (3,15%)"` render
from the live run), and that the rejection is a correctable violation rather
than a crash. The exit gate — a live report-Q&A that commits with
`is_fallback: 0` — is proven by `scripts/smoke_local_phase3.sh`, not by unit
tests.
