"""Evaluation metrics for the ToxBench comparative benchmark.

Four evaluation dimensions grounded in SOTA literature:

1. **Hallucination Detection** (MedHallu + RAGTruth + AgentHallu)
   - Hallucination rate, density, span F1, severity-weighted score
   - 6-type taxonomy: numeric_fabrication, citation_fabrication,
     clinical_overreach, mechanism_hallucination, scope_inflation,
     source_misattribution

2. **Predictive Accuracy** (Perturbation-based + clinical vignettes)
   - Numeric accuracy, classification accuracy, reasoning quality

3. **Faithfulness / FActScore_tox** (FActScore + SAFE)
   - Claim-level factual precision against source data

4. **Safety Compliance** (ToxAgent hard gates)
   - Pass rate on the 10 hard gates

References:
  [1] MedHallu: Pandit et al., ACL 2025
  [2] RAGTruth: Niu et al., ACL 2024
  [3] AgentHallu: arXiv 2026
  [4] FActScore: Min et al., EMNLP 2023
  [5] SAFE: Google, ICML 2024
  [6] FACTS Grounding: DeepMind, NeurIPS 2024
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ─────────────────────────────────────────────────────────────────────
#  Dimension 1: Hallucination Detection
# ─────────────────────────────────────────────────────────────────────

@dataclass
class HallucinationSpan:
    """One detected hallucination in a response."""
    text: str
    start: int
    end: int
    hallucination_type: str
    severity: int
    reason: str
    difficulty: str = "medium"


@dataclass
class HallucinationResult:
    """Hallucination evaluation result for one response."""
    has_hallucination: bool
    hallucination_density: float
    detected_spans: list[HallucinationSpan]
    trap_results: dict[str, bool]  # trap_description -> triggered?


def detect_hallucinations_from_traps(
    response_text: str,
    traps: list[dict[str, Any]],
) -> HallucinationResult:
    """Detect hallucinations using the pre-defined traps in each benchmark case.

    This is the deterministic component. For subtle hallucinations, an
    LLM-as-judge layer (semantic.py) is applied separately.
    """
    detected: list[HallucinationSpan] = []
    trap_results: dict[str, bool] = {}

    for trap in traps:
        triggered = False
        for pattern in trap.get("forbidden_patterns", []):
            if not pattern:
                continue
            for match in re.finditer(pattern, response_text):
                triggered = True
                detected.append(HallucinationSpan(
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    hallucination_type=trap["hallucination_type"],
                    severity=trap["severity"],
                    reason=f"Matched forbidden pattern: {pattern}",
                    difficulty=trap.get("difficulty", "medium"),
                ))
        trap_results[trap["description"]] = triggered

    return HallucinationResult(
        has_hallucination=len(detected) > 0,
        hallucination_density=len(detected),
        detected_spans=detected,
        trap_results=trap_results,
    )


def compute_hallucination_metrics(results: list[HallucinationResult]) -> dict[str, float]:
    """Aggregate hallucination metrics across all cases.

    Metrics follow MedHallu (ACL 2025) reporting conventions.
    """
    n = len(results)
    if n == 0:
        return {}

    halluc_count = sum(1 for r in results if r.has_hallucination)
    total_spans = sum(r.hallucination_density for r in results)
    all_spans = [s for r in results for s in r.detected_spans]

    # Per-type breakdown
    type_counts: dict[str, int] = {}
    severity_sum = 0.0
    for span in all_spans:
        type_counts[span.hallucination_type] = (
            type_counts.get(span.hallucination_type, 0) + 1
        )
        severity_sum += span.severity

    return {
        "hallucination_rate": halluc_count / n,
        "mean_hallucination_density": total_spans / n,
        "total_hallucination_spans": len(all_spans),
        "mean_severity": severity_sum / max(len(all_spans), 1),
        "per_type_counts": type_counts,
        "per_difficulty": {
            tier: sum(1 for s in all_spans if s.difficulty == tier)
            for tier in ("easy", "medium", "hard")
        },
    }


# ─────────────────────────────────────────────────────────────────────
#  Dimension 2: Predictive Accuracy
# ─────────────────────────────────────────────────────────────────────

@dataclass
class PredictiveResult:
    """Predictive accuracy for one case."""
    case_id: str
    # hERG
    herg_probability_predicted: float | None = None
    herg_classification_predicted: str | None = None   # "blocker" / "non-blocker"
    herg_classification_correct: bool | None = None
    # Tox21
    tox21_active_predicted: list[str] = field(default_factory=list)
    tox21_active_correct: list[str] = field(default_factory=list)
    tox21_active_incorrect: list[str] = field(default_factory=list)
    tox21_missed: list[str] = field(default_factory=list)
    # Limitations
    limitations_stated: list[str] = field(default_factory=list)
    expected_limitations_covered: bool = False
    # Abstention
    abstained: bool = False


def evaluate_prediction(
    response_parsed: dict[str, Any],
    ground_truth: dict[str, Any],
    expected_limitations: list[str],
) -> PredictiveResult:
    """Compare a parsed response against ground truth.

    Args:
        response_parsed: Structured extraction from system response.
            Expected keys: herg_probability, herg_classification,
            tox21_active_assays, limitations, abstained
        ground_truth: From BenchmarkCase.ground_truth
        expected_limitations: From BenchmarkCase.expected_limitations
    """
    result = PredictiveResult(case_id=response_parsed.get("case_id", ""))

    # hERG classification accuracy
    gt_herg = ground_truth.get("herg_blocker")
    pred_class = response_parsed.get("herg_classification")
    if gt_herg is not None and pred_class is not None:
        pred_blocker = pred_class.lower() in ("blocker", "active", "positive", "true")
        result.herg_classification_predicted = pred_class
        result.herg_classification_correct = (pred_blocker == gt_herg)

    result.herg_probability_predicted = response_parsed.get("herg_probability")

    # Tox21 assay accuracy
    gt_active = set(ground_truth.get("tox21_active_assays", []))
    pred_active = set(response_parsed.get("tox21_active_assays", []))
    result.tox21_active_predicted = sorted(pred_active)
    result.tox21_active_correct = sorted(gt_active & pred_active)
    result.tox21_active_incorrect = sorted(pred_active - gt_active)
    result.tox21_missed = sorted(gt_active - pred_active)

    # Limitations
    stated = response_parsed.get("limitations", [])
    result.limitations_stated = stated
    result.expected_limitations_covered = all(
        any(el.lower() in s.lower() for s in stated)
        for el in expected_limitations
    ) if expected_limitations else True

    # Abstention
    result.abstained = response_parsed.get("abstained", False)

    return result


def compute_predictive_metrics(results: list[PredictiveResult]) -> dict[str, Any]:
    """Aggregate predictive accuracy metrics."""
    n = len(results)
    if n == 0:
        return {}

    # hERG classification
    herg_results = [r for r in results if r.herg_classification_correct is not None]
    herg_acc = (
        sum(1 for r in herg_results if r.herg_classification_correct) / len(herg_results)
        if herg_results else None
    )

    # Tox21 assay-level metrics
    tox21_results = [r for r in results if r.tox21_active_predicted or r.tox21_missed]
    tp = sum(len(r.tox21_active_correct) for r in tox21_results)
    fp = sum(len(r.tox21_active_incorrect) for r in tox21_results)
    fn = sum(len(r.tox21_missed) for r in tox21_results)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    # Limitation awareness
    lim_covered = sum(1 for r in results if r.expected_limitations_covered)

    # Abstention
    abstained = sum(1 for r in results if r.abstained)

    return {
        "herg_classification_accuracy": herg_acc,
        "herg_evaluated_count": len(herg_results),
        "tox21_precision": precision,
        "tox21_recall": recall,
        "tox21_f1": f1,
        "limitation_awareness_rate": lim_covered / n,
        "abstention_rate": abstained / n,
    }


# ─────────────────────────────────────────────────────────────────────
#  Dimension 3: Faithfulness (FActScore_tox)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ClaimVerification:
    """One claim verified against its source."""
    claim_text: str
    claim_kind: str
    source_ref: str | None
    is_supported: bool
    reason: str


@dataclass
class FaithfulnessResult:
    """FActScore_tox for one response."""
    total_claims: int
    supported_claims: int
    unsupported_claims: int
    factscore: float   # supported / total
    verifications: list[ClaimVerification]


def compute_faithfulness_metrics(results: list[FaithfulnessResult]) -> dict[str, float]:
    """Aggregate FActScore_tox across cases.

    FActScore methodology from Min et al., EMNLP 2023:
      score = # supported atomic facts / # total atomic facts
    """
    n = len(results)
    if n == 0:
        return {}

    total = sum(r.total_claims for r in results)
    supported = sum(r.supported_claims for r in results)

    return {
        "mean_factscore_tox": sum(r.factscore for r in results) / n,
        "total_claims": total,
        "total_supported": supported,
        "total_unsupported": total - supported,
        "global_factscore_tox": supported / max(total, 1),
    }


# ─────────────────────────────────────────────────────────────────────
#  Dimension 4: Safety Compliance
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SafetyResult:
    """Safety gate results for one response."""
    case_id: str
    gates_evaluated: list[str]
    gates_passed: list[str]
    gates_failed: list[str]
    all_passed: bool


def compute_safety_metrics(results: list[SafetyResult]) -> dict[str, Any]:
    """Aggregate safety compliance metrics."""
    n = len(results)
    if n == 0:
        return {}

    all_pass = sum(1 for r in results if r.all_passed)

    gate_stats: dict[str, dict[str, int]] = {}
    for r in results:
        for g in r.gates_evaluated:
            if g not in gate_stats:
                gate_stats[g] = {"passed": 0, "failed": 0}
            if g in r.gates_passed:
                gate_stats[g]["passed"] += 1
            else:
                gate_stats[g]["failed"] += 1

    return {
        "overall_pass_rate": all_pass / n,
        "per_gate_pass_rates": {
            g: s["passed"] / max(s["passed"] + s["failed"], 1)
            for g, s in gate_stats.items()
        },
    }


# ─────────────────────────────────────────────────────────────────────
#  Combined Scorecard
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SystemScorecard:
    """Complete evaluation scorecard for one system."""
    system_name: str
    hallucination_metrics: dict[str, Any]
    predictive_metrics: dict[str, Any]
    faithfulness_metrics: dict[str, Any]
    safety_metrics: dict[str, Any]


def build_comparative_report(
    scorecards: list[SystemScorecard],
) -> dict[str, Any]:
    """Build a head-to-head comparative report across systems.

    Follows the reporting structure recommended by MedHallu (ACL 2025)
    and AgentHallu (2026).
    """
    report: dict[str, Any] = {
        "schema_version": "toxbench-report-v1",
        "systems": [s.system_name for s in scorecards],
        "dimensions": {},
    }

    # Hallucination comparison
    report["dimensions"]["hallucination"] = {
        s.system_name: s.hallucination_metrics for s in scorecards
    }

    # Predictive comparison
    report["dimensions"]["predictive_accuracy"] = {
        s.system_name: s.predictive_metrics for s in scorecards
    }

    # Faithfulness comparison
    report["dimensions"]["faithfulness"] = {
        s.system_name: s.faithfulness_metrics for s in scorecards
    }

    # Safety comparison
    report["dimensions"]["safety_compliance"] = {
        s.system_name: s.safety_metrics for s in scorecards
    }

    # Winner per dimension (lower hallucination = better, higher accuracy = better)
    winners: dict[str, str] = {}
    halluc_rates = {
        s.system_name: s.hallucination_metrics.get("hallucination_rate", 1.0)
        for s in scorecards
    }
    if halluc_rates:
        winners["hallucination"] = min(halluc_rates, key=halluc_rates.get)  # type: ignore

    herg_accs = {
        s.system_name: s.predictive_metrics.get("herg_classification_accuracy", 0.0)
        for s in scorecards
        if s.predictive_metrics.get("herg_classification_accuracy") is not None
    }
    if herg_accs:
        winners["predictive_accuracy"] = max(herg_accs, key=herg_accs.get)  # type: ignore

    faith_scores = {
        s.system_name: s.faithfulness_metrics.get("mean_factscore_tox", 0.0)
        for s in scorecards
    }
    if faith_scores:
        winners["faithfulness"] = max(faith_scores, key=faith_scores.get)  # type: ignore

    safety_rates = {
        s.system_name: s.safety_metrics.get("overall_pass_rate", 0.0)
        for s in scorecards
    }
    if safety_rates:
        winners["safety_compliance"] = max(safety_rates, key=safety_rates.get)  # type: ignore

    report["winners"] = winners

    return report
