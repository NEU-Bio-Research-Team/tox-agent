# ADR 0002 — No aggregate toxicity verdict

**Status:** accepted · **Date:** 2026-09-04 · **Plan:** §2.2 (SCI-01…SCI-10), §9

## Decision

No schema in this project may carry an aggregate toxicity, safety, or severity
score, and no answer may assert one. hERG blockade, the twelve Tox21 assay
activities, and ClinTox clinical-trial toxicity are three different
measurements and stay three different fields, end to end.

## Enforcement, in the order a claim meets it

1. **Schema.** `GroundedAnswer` has no aggregate field to populate. Claims are
   `numeric | classification | scientific | comparison | limitation |
   recommendation`; each numeric claim names one observation and one field path.
2. **Numeric validator.** `source_value` must equal the canonical value at that
   field path, and `rendered_value` must be a declared transform of it. There is
   no path by which a model-authored number reaches an accepted answer.
3. **Classification validator.** Labels are compared as exact enums.
   `non_blocker` cannot become `safe`; `applicability.ok` cannot become
   `in_distribution`.
4. **Prohibited-claim validator.** Cross-endpoint substitution, hit-count-as-
   severity, and clinical/regulatory conclusions are rejected by code, per
   endpoint, with a typed violation.
5. **Required limitations.** Interpreting a probability requires
   `uncalibrated_probability`; naming applicability requires
   `applicability_is_rule_based`; attribution requires
   `attribution_not_causality`.

## Consequences

- A caller that wants "how many Tox21 assays are active" can count the assays
  themselves; the product will not present that count as a severity.
- An unavailable endpoint stays unavailable. There is no fallback endpoint and
  no borrowed probability (SCI-06).
