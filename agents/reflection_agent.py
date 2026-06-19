"""reflection_agent.py — ToxAgent ReflectionAgent

An independent scientific reviewer that evaluates toxicity reports produced by
WriterAgent and returns a structured quality assessment artifact.

Design decisions (approved):
  Q1 — Flag Only: needs_revision=True is a flag, not a retry trigger.
  Q2 — Env-Gated + Complexity-Aware:
         REFLECTION_ENABLED env var (default "1");
         LLM path only for MEDIUM/HIGH complexity.
         LOW complexity gets deterministic fast-path only.
  Q3 — Direct GenAI API: same build_genai_client_candidates + call_with_retry
         pattern as writer_agent.py. No ADK runner.

ReflectionAgent is read-only — it never modifies or regenerates the report.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from .adk_compat import LlmAgent

logger = logging.getLogger(__name__)

REFLECTION_MODEL = os.getenv(
    "AGENT_MODEL_FAST",
    os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
)
REFLECTION_FALLBACK_MODEL = os.getenv(
    "AGENT_MODEL_FAST",
    os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
)


def _reflection_enabled() -> bool:
    """Read REFLECTION_ENABLED env var (default: enabled)."""
    raw = os.getenv("REFLECTION_ENABLED", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------


class ReflectionResult(BaseModel):
    """Structured quality assessment of a ToxAgent report.

    Fields
    ------
    score               : 0-100 scientific quality score.
    supported           : True if all conclusions are evidence-backed.
    missing_evidence    : Evidence gaps that would strengthen the report.
    unsupported_claims  : Claims made without adequate supporting evidence.
    ood_risk            : True if the compound is flagged as out-of-distribution.
    needs_revision      : True when unsupported claims exist or score < 50.
    recommended_actions : Corrective actions for the report author / reviewer.
    score_breakdown     : Per-dimension scores for UI breakdown display.
    reflection_source   : 'deterministic' | 'llm' | 'llm_fallback'.
    """

    score: int = Field(default=0, ge=0, le=100)
    supported: bool = Field(default=True)
    missing_evidence: List[str] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)
    ood_risk: bool = Field(default=False)
    needs_revision: bool = Field(default=False)
    recommended_actions: List[str] = Field(default_factory=list)
    score_breakdown: Dict[str, int] = Field(default_factory=dict)
    reflection_source: str = Field(default="deterministic")

    @model_validator(mode="after")
    def _derive_needs_revision(self) -> "ReflectionResult":
        """needs_revision is True when claims are unsupported or score < 50."""
        if self.unsupported_claims or self.score < 50:
            object.__setattr__(self, "needs_revision", True)
            object.__setattr__(self, "supported", not bool(self.unsupported_claims))
        return self


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Deterministic scoring — 7 dimensions, 100 points total
# ---------------------------------------------------------------------------


def _score_scientific_validity(
    ood_assessment: Dict[str, Any],
    clinical: Dict[str, Any],
) -> Tuple[int, List[str]]:
    """0-20 pts: OOD flag + model confidence."""
    notes: List[str] = []
    score = 20

    ood_flag = bool(ood_assessment.get("flag", False))
    ood_risk = str(ood_assessment.get("ood_risk") or "LOW").upper()
    confidence = str(clinical.get("confidence") or "HIGH").upper()

    if ood_flag:
        if ood_risk == "HIGH":
            score -= 12
            notes.append("OOD flag raised with HIGH risk level")
        elif ood_risk == "MEDIUM":
            score -= 8
            notes.append("OOD flag raised with MEDIUM risk level")
        else:
            score -= 4
            notes.append("OOD flag raised with LOW risk level")

    if confidence == "LOW":
        score -= 4
        notes.append("Model confidence is LOW")
    elif confidence == "MEDIUM":
        score -= 2
        notes.append("Model confidence is MEDIUM")

    return max(0, score), notes


def _score_evidence_sufficiency(
    research: Dict[str, Any],
    evidence_qa: Optional[Dict[str, Any]],
) -> Tuple[int, List[str]]:
    """0-20 pts: article count + evidence_confidence from EvidenceQA."""
    notes: List[str] = []
    score = 0

    # Use QA-curated count if available, fall back to raw literature count
    curated_count = 0
    evidence_confidence = "UNKNOWN"

    if evidence_qa:
        curated_count = _safe_int(evidence_qa.get("total_articles_curated"))
        evidence_confidence = str(
            evidence_qa.get("evidence_confidence") or "UNKNOWN"
        ).upper()
    else:
        literature = _to_dict(research.get("literature"))
        articles = literature.get("articles")
        curated_count = len(articles) if isinstance(articles, list) else 0
        evidence_confidence = "UNKNOWN"

    # Score from confidence level
    conf_score = {"HIGH": 12, "MEDIUM": 8, "LOW": 4, "UNKNOWN": 2}.get(
        evidence_confidence, 2
    )
    score += conf_score

    # Score from article count
    if curated_count >= 3:
        score += 8
    elif curated_count >= 1:
        score += 4
    else:
        notes.append("No curated literature articles available")

    if evidence_confidence in ("LOW", "UNKNOWN"):
        notes.append(f"Evidence confidence is {evidence_confidence}")

    return min(20, score), notes


def _score_consistency(
    clinical: Dict[str, Any],
    mechanism: Dict[str, Any],
    final_report: Dict[str, Any],
) -> Tuple[int, List[str]]:
    """0-20 pts: risk_level vs p_toxic vs assay_hits internal consistency."""
    notes: List[str] = []
    score = 20

    p_toxic = _safe_float(clinical.get("p_toxic"))
    assay_hits = _safe_int(mechanism.get("assay_hits"))
    risk_level = str(
        final_report.get("risk_level") or "UNKNOWN"
    ).upper()
    label = str(clinical.get("label") or "").upper()

    # CRITICAL risk should have both high p_toxic AND high assay_hits
    if risk_level == "CRITICAL":
        if p_toxic < 0.6:
            score -= 8
            notes.append(
                f"CRITICAL risk declared but p_toxic={p_toxic:.2f} is below 0.60"
            )
        if assay_hits < 2:
            score -= 6
            notes.append(
                f"CRITICAL risk declared but assay_hits={assay_hits} < 2"
            )

    # HIGH risk should have meaningful signals
    elif risk_level == "HIGH":
        if p_toxic < 0.4 and assay_hits < 1:
            score -= 10
            notes.append(
                f"HIGH risk declared but p_toxic={p_toxic:.2f} and assay_hits={assay_hits} are both low"
            )
        elif p_toxic < 0.4 or assay_hits < 1:
            score -= 4
            notes.append("HIGH risk partially inconsistent with underlying signals")

    # TOXIC label should align with p_toxic
    if label == "TOXIC" and p_toxic < 0.35:
        score -= 6
        notes.append(
            f"TOXIC label but p_toxic={p_toxic:.2f} is below default threshold 0.35"
        )
    elif label == "NON-TOXIC" and p_toxic > 0.65:
        score -= 6
        notes.append(
            f"NON-TOXIC label but p_toxic={p_toxic:.2f} is unexpectedly high"
        )

    return max(0, score), notes


def _score_missing_information(
    research: Dict[str, Any],
    mechanism: Dict[str, Any],
) -> Tuple[int, List[str]]:
    """0-15 pts: bioassay / literature / mechanism data presence."""
    notes: List[str] = []
    score = 0

    has_bioassay = bool(research.get("bioassay_summary"))
    has_literature = bool(
        _to_dict(research.get("literature")).get("articles")
    )
    has_mechanism = _safe_int(mechanism.get("assay_hits")) > 0 or bool(
        mechanism.get("active_tasks")
    )

    if has_bioassay:
        score += 5
    else:
        notes.append("PubChem bioassay data is unavailable for this compound")

    if has_literature:
        score += 5
    else:
        notes.append("No published literature was retrieved for this compound")

    if has_mechanism:
        score += 5
    else:
        notes.append("Mechanistic Tox21 assay data shows no active hits")

    return score, notes


def _score_reliability(
    ood_assessment: Dict[str, Any],
    screening: Dict[str, Any],
) -> Tuple[int, List[str]]:
    """0-10 pts: reliability_warning + OOD risk level severity."""
    notes: List[str] = []
    score = 10

    reliability_warning = screening.get("reliability_warning")
    ood_risk = str(ood_assessment.get("ood_risk") or "LOW").upper()

    if reliability_warning:
        score -= 4
        notes.append(f"Reliability warning: {str(reliability_warning)[:120]}")

    penalty = {"HIGH": 5, "MEDIUM": 3, "LOW": 0}.get(ood_risk, 0)
    score -= penalty
    if penalty:
        notes.append(f"OOD risk is {ood_risk}")

    return max(0, score), notes


def _score_ood_warnings(ood_assessment: Dict[str, Any]) -> Tuple[int, List[str]]:
    """0-10 pts: OOD flag and risk severity."""
    notes: List[str] = []
    ood_flag = bool(ood_assessment.get("flag", False))
    ood_risk = str(ood_assessment.get("ood_risk") or "LOW").upper()

    if not ood_flag:
        return 10, []

    score = {"HIGH": 0, "MEDIUM": 3, "LOW": 6}.get(ood_risk, 5)
    notes.append(
        f"Compound is outside the model training distribution (OOD risk={ood_risk})"
    )
    reason = str(ood_assessment.get("reason") or "").strip()
    if reason:
        notes.append(f"OOD reason: {reason[:120]}")
    return score, notes


def _score_overconfidence(
    clinical: Dict[str, Any],
    ood_assessment: Dict[str, Any],
) -> Tuple[int, List[str]]:
    """0-5 pts: HIGH confidence despite OOD or low p_toxic margin."""
    notes: List[str] = []
    confidence = str(clinical.get("confidence") or "").upper()
    ood_flag = bool(ood_assessment.get("flag", False))
    p_toxic = _safe_float(clinical.get("p_toxic"))
    threshold = _safe_float(clinical.get("threshold_used"), 0.35)

    # Overconfident = HIGH confidence when OOD is flagged
    if confidence == "HIGH" and ood_flag:
        notes.append(
            "HIGH model confidence reported despite OOD flag — may indicate overconfidence"
        )
        return 1, notes

    # Borderline prediction with high confidence
    margin = abs(p_toxic - threshold)
    if confidence == "HIGH" and margin < 0.08:
        notes.append(
            f"HIGH confidence on borderline prediction (margin={margin:.3f} from threshold)"
        )
        return 2, notes

    return 5, notes


# ---------------------------------------------------------------------------
# Missing evidence and unsupported claim detection
# ---------------------------------------------------------------------------


def _detect_missing_evidence(
    research: Dict[str, Any],
    ood_assessment: Dict[str, Any],
    evidence_qa: Optional[Dict[str, Any]],
    screening: Dict[str, Any],
) -> List[str]:
    """Build a list of missing evidence gaps (for UI display)."""
    gaps: List[str] = []

    # Literature
    if not _to_dict(research.get("literature")).get("articles"):
        gaps.append(
            "No published PubMed literature retrieved for this compound — "
            "manual literature search is recommended."
        )

    # Bioassay
    if not research.get("bioassay_summary"):
        gaps.append(
            "PubChem bioassay data is unavailable — "
            "experimental assay confirmation is needed."
        )

    # OOD
    if ood_assessment.get("flag"):
        ood_risk = str(ood_assessment.get("ood_risk") or "UNKNOWN").upper()
        gaps.append(
            f"Compound is out-of-distribution (OOD risk={ood_risk}) — "
            "model predictions may be unreliable; experimental validation required."
        )

    # Evidence quality
    if evidence_qa:
        conf = str(evidence_qa.get("evidence_confidence") or "").upper()
        if conf in ("LOW", "UNKNOWN"):
            gaps.append(
                f"Curated literature evidence quality is {conf} — "
                "higher-quality or more relevant publications are needed."
            )

    # Mechanism
    mechanism = _to_dict(screening.get("mechanism"))
    if _safe_int(mechanism.get("assay_hits")) == 0:
        gaps.append(
            "No Tox21 assay hits detected — mechanistic toxicity basis is absent."
        )

    return gaps


def _detect_unsupported_claims(
    final_report: Dict[str, Any],
    clinical: Dict[str, Any],
    mechanism: Dict[str, Any],
    ood_assessment: Dict[str, Any],
    evidence_qa: Optional[Dict[str, Any]],
) -> List[str]:
    """Detect claims in the report that are not adequately supported by evidence."""
    claims: List[str] = []

    risk_level = str(final_report.get("risk_level") or "UNKNOWN").upper()
    p_toxic = _safe_float(clinical.get("p_toxic"))
    assay_hits = _safe_int(mechanism.get("assay_hits"))
    label = str(clinical.get("label") or "").upper()
    ood_flag = bool(ood_assessment.get("flag", False))
    ood_risk = str(ood_assessment.get("ood_risk") or "LOW").upper()

    # CRITICAL without strong evidence
    if risk_level == "CRITICAL" and (p_toxic < 0.6 or assay_hits < 2):
        claims.append(
            f"Report concludes CRITICAL risk (p_toxic={p_toxic:.2f}, assay_hits={assay_hits}) "
            "but the underlying signals do not consistently support this severity."
        )

    # TOXIC label with low p_toxic
    if label == "TOXIC" and p_toxic < 0.35:
        claims.append(
            f"TOXIC verdict is asserted with p_toxic={p_toxic:.2f} below the standard threshold — "
            "this classification requires explicit justification."
        )

    # HIGH risk + LOW evidence confidence
    if evidence_qa:
        ev_conf = str(evidence_qa.get("evidence_confidence") or "").upper()
        if risk_level in ("HIGH", "CRITICAL") and ev_conf == "LOW":
            claims.append(
                f"{risk_level} risk level is stated but literature evidence confidence is LOW — "
                "the conclusion is not adequately supported by published evidence."
            )

    # OOD flagged but no caveat in report metadata
    report_meta = _to_dict(final_report.get("report_metadata"))
    if ood_flag and ood_risk == "HIGH" and not report_meta.get("ood_caveat"):
        claims.append(
            "The compound is OOD (HIGH risk) but the report does not include "
            "an explicit out-of-distribution caveat in its conclusions."
        )

    return claims


# ---------------------------------------------------------------------------
# Deterministic reflection — assembles full result without LLM
# ---------------------------------------------------------------------------


def _run_deterministic_reflection(
    final_report: Dict[str, Any],
    screening: Dict[str, Any],
    research: Dict[str, Any],
    evidence_qa: Optional[Dict[str, Any]],
) -> ReflectionResult:
    """Score the report across 7 dimensions using only structured data.

    Always runs regardless of env flags or complexity — it is the safe baseline.
    """
    clinical = _to_dict(screening.get("clinical"))
    mechanism = _to_dict(screening.get("mechanism"))
    ood_assessment = _to_dict(screening.get("ood_assessment"))

    # --- 7-dimension scorer ---
    s1, n1 = _score_scientific_validity(ood_assessment, clinical)
    s2, n2 = _score_evidence_sufficiency(research, evidence_qa)
    s3, n3 = _score_consistency(clinical, mechanism, final_report)
    s4, n4 = _score_missing_information(research, mechanism)
    s5, n5 = _score_reliability(ood_assessment, screening)
    s6, n6 = _score_ood_warnings(ood_assessment)
    s7, n7 = _score_overconfidence(clinical, ood_assessment)

    total = s1 + s2 + s3 + s4 + s5 + s6 + s7

    score_breakdown = {
        "scientific_validity": s1,
        "evidence_sufficiency": s2,
        "consistency": s3,
        "missing_information": s4,
        "reliability": s5,
        "ood_warnings": s6,
        "overconfidence": s7,
    }

    missing_evidence = _detect_missing_evidence(
        research, ood_assessment, evidence_qa, screening
    )
    unsupported_claims = _detect_unsupported_claims(
        final_report, clinical, mechanism, ood_assessment, evidence_qa
    )

    # Build recommended actions from scorer notes
    all_notes = n1 + n2 + n3 + n4 + n5 + n6 + n7
    recommended_actions: List[str] = []
    if unsupported_claims:
        recommended_actions.append(
            "Review and revise conclusions — one or more claims are not fully supported."
        )
    if missing_evidence:
        recommended_actions.append(
            "Acquire additional evidence: "
            + "; ".join(e.split("—")[0].strip() for e in missing_evidence[:2])
            + "."
        )
    if ood_assessment.get("flag"):
        recommended_actions.append(
            "Flag for expert adjudication: compound is out-of-distribution."
        )
    if total < 50:
        recommended_actions.append(
            "Major scientific concerns detected — do not use this report for regulatory decisions."
        )
    elif total < 70:
        recommended_actions.append(
            "Minor weaknesses identified — additional experimental validation is recommended."
        )

    ood_risk_flag = bool(ood_assessment.get("flag", False))

    return ReflectionResult(
        score=total,
        supported=not bool(unsupported_claims),
        missing_evidence=missing_evidence,
        unsupported_claims=unsupported_claims,
        ood_risk=ood_risk_flag,
        needs_revision=bool(unsupported_claims) or total < 50,
        recommended_actions=recommended_actions,
        score_breakdown=score_breakdown,
        reflection_source="deterministic",
    )


# ---------------------------------------------------------------------------
# LLM reflection — direct GenAI API (same pattern as writer_agent.py)
# ---------------------------------------------------------------------------


def _build_reflection_prompt(
    final_report: Dict[str, Any],
    screening: Dict[str, Any],
    research: Dict[str, Any],
    evidence_qa: Optional[Dict[str, Any]],
    deterministic: ReflectionResult,
) -> str:
    """Build the prompt for the LLM reflection reviewer."""
    clinical = _to_dict(screening.get("clinical"))
    mechanism = _to_dict(screening.get("mechanism"))
    ood_assessment = _to_dict(screening.get("ood_assessment"))

    compound_info = _to_dict(research.get("compound_info"))
    literature = _to_dict(research.get("literature"))
    articles = literature.get("articles") or []
    article_titles = [
        str(a.get("title") or "").strip()
        for a in (articles[:4] if isinstance(articles, list) else [])
        if isinstance(a, dict) and a.get("title")
    ]

    payload = {
        "report_summary": {
            "risk_level": final_report.get("risk_level"),
            "executive_summary": str(
                final_report.get("executive_summary") or ""
            )[:400],
        },
        "screening_signals": {
            "clinical_label": clinical.get("label"),
            "p_toxic": clinical.get("p_toxic"),
            "threshold_used": clinical.get("threshold_used"),
            "confidence": clinical.get("confidence"),
            "assay_hits": mechanism.get("assay_hits"),
            "highest_risk_task": mechanism.get("highest_risk_task"),
            "highest_risk_score": mechanism.get("highest_risk_score"),
            "final_verdict": screening.get("final_verdict"),
        },
        "ood_assessment": {
            "flag": ood_assessment.get("flag"),
            "ood_risk": ood_assessment.get("ood_risk"),
            "reason": ood_assessment.get("reason"),
        },
        "evidence_quality": {
            "curated_articles": (evidence_qa or {}).get("total_articles_curated"),
            "evidence_confidence": (evidence_qa or {}).get("evidence_confidence"),
            "article_titles": article_titles,
            "has_bioassay": bool(research.get("bioassay_summary")),
            "compound_name": compound_info.get("common_name")
            or compound_info.get("iupac_name"),
        },
        "deterministic_pre_score": deterministic.score,
        "deterministic_missing_evidence": deterministic.missing_evidence,
        "deterministic_unsupported_claims": deterministic.unsupported_claims,
    }

    output_schema = {
        "score": "integer 0-100",
        "supported": "boolean",
        "missing_evidence": ["string"],
        "unsupported_claims": ["string"],
        "ood_risk": "boolean",
        "needs_revision": "boolean",
        "recommended_actions": ["string"],
    }

    return (
        "You are ReflectionAgent for ToxAgent.\n"
        "Your responsibility is to critically evaluate the toxicity report below "
        "as an independent scientific reviewer.\n\n"
        "Review dimensions:\n"
        "1. Scientific validity\n"
        "2. Evidence sufficiency\n"
        "3. Consistency between conclusions and evidence\n"
        "4. Missing information\n"
        "5. Reliability concerns\n"
        "6. OOD warnings\n"
        "7. Overconfidence detection\n\n"
        "Scoring:\n"
        "90-100: Strong evidence support.\n"
        "70-89:  Acceptable but minor weaknesses.\n"
        "50-69:  Insufficient support.\n"
        "Below 50: Major scientific concerns.\n\n"
        "If unsupported claims exist, set needs_revision=true.\n\n"
        "Output ONLY valid JSON with this schema:\n"
        f"{json.dumps(output_schema, indent=2)}\n\n"
        "A deterministic pre-scorer has already assigned a preliminary score of "
        f"{deterministic.score}/100. Your LLM review may adjust this score up or down "
        "based on subtle scientific reasoning the rule-based system cannot assess.\n\n"
        f"Report data: {json.dumps(payload, ensure_ascii=False)}"
    )


def _call_llm_reflection(
    final_report: Dict[str, Any],
    screening: Dict[str, Any],
    research: Dict[str, Any],
    evidence_qa: Optional[Dict[str, Any]],
    deterministic: ReflectionResult,
) -> Tuple[Optional[ReflectionResult], str]:
    """Call GenAI API to get an LLM-based reflection.

    Returns (ReflectionResult, status_string).
    Returns (None, error_string) on failure — caller falls back to deterministic.
    """
    try:
        from services.genai_runtime import (  # type: ignore[import]
            build_genai_client_candidates,
            call_with_retry,
            dedupe_strings,
            is_model_unavailable_error,
            is_resource_exhausted_error,
        )
    except ImportError as exc:
        logger.warning("reflection: genai_runtime unavailable (%s)", exc)
        return None, "genai_runtime_unavailable"

    client_candidates = build_genai_client_candidates()
    if not client_candidates:
        return None, "genai_client_unavailable"

    prompt = _build_reflection_prompt(
        final_report, screening, research, evidence_qa, deterministic
    )
    model_candidates = dedupe_strings([REFLECTION_MODEL, REFLECTION_FALLBACK_MODEL])
    errors: List[str] = []

    for client, auth_mode in client_candidates:
        for model_name in model_candidates:
            try:
                response = call_with_retry(
                    lambda: client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config={
                            "temperature": 0.2,
                            "response_mime_type": "application/json",
                        },
                    )
                )
                raw = str(getattr(response, "text", "") or "").strip()
                if not raw:
                    errors.append(f"empty_response:{auth_mode}:{model_name}")
                    continue

                # Strip code fences if present
                if raw.startswith("```"):
                    lines = raw.splitlines()
                    raw = "\n".join(
                        lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
                    ).strip()

                data = json.loads(raw)

                result = ReflectionResult(
                    score=int(data.get("score", deterministic.score)),
                    supported=bool(data.get("supported", deterministic.supported)),
                    missing_evidence=list(
                        data.get("missing_evidence", deterministic.missing_evidence)
                    ),
                    unsupported_claims=list(
                        data.get("unsupported_claims", deterministic.unsupported_claims)
                    ),
                    ood_risk=bool(data.get("ood_risk", deterministic.ood_risk)),
                    needs_revision=bool(
                        data.get("needs_revision", deterministic.needs_revision)
                    ),
                    recommended_actions=list(
                        data.get("recommended_actions", deterministic.recommended_actions)
                    ),
                    score_breakdown=deterministic.score_breakdown,  # keep rule-based breakdown
                    reflection_source="llm",
                )
                return result, f"llm_success:{auth_mode}:{model_name}"

            except Exception as exc:  # noqa: BLE001
                errors.append(
                    f"llm_error:{type(exc).__name__}:{auth_mode}:{model_name}:{str(exc)[:150]}"
                )
                try:
                    if is_resource_exhausted_error(exc) or is_model_unavailable_error(exc):
                        continue
                except Exception:
                    pass
                return None, errors[-1]

    return None, (errors[0] if errors else "llm_failed")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_reflection(
    final_report: Dict[str, Any],
    screening_result: Optional[Dict[str, Any]] = None,
    research_result: Optional[Dict[str, Any]] = None,
    evidence_qa_result: Optional[Dict[str, Any]] = None,
    plan_complexity: str = "MEDIUM",
) -> ReflectionResult:
    """Produce a ReflectionResult for the given completed report.

    Strategy
    --------
    1. Always run the deterministic fast-path scorer.
    2. If REFLECTION_ENABLED and complexity >= MEDIUM, call the LLM reviewer.
    3. On LLM failure, fall back to deterministic result (marked 'llm_fallback').
    4. Never raise — always return a valid ReflectionResult.

    Parameters
    ----------
    final_report      : Completed report dict from build_final_report().
    screening_result  : Raw screening payload from ScreeningAgent.
    research_result   : Research payload from ResearchAgent.
    evidence_qa_result: EvidenceQA payload (optional, enriches scoring).
    plan_complexity   : 'LOW' | 'MEDIUM' | 'HIGH' from AgentPlan.
    """
    report = final_report if isinstance(final_report, dict) else {}
    screening = screening_result if isinstance(screening_result, dict) else {}
    research = research_result if isinstance(research_result, dict) else {}
    evidence_qa = (
        evidence_qa_result if isinstance(evidence_qa_result, dict) else None
    )
    complexity = str(plan_complexity or "MEDIUM").upper()

    # Step 1 — Deterministic fast-path (always)
    try:
        deterministic = _run_deterministic_reflection(
            report, screening, research, evidence_qa
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("run_reflection: deterministic scorer failed (%s); returning minimal result", exc)
        return ReflectionResult(
            score=50,
            supported=True,
            missing_evidence=["Reflection scorer encountered an internal error."],
            reflection_source="deterministic",
        )

    logger.info(
        "run_reflection: deterministic score=%d  complexity=%s  enabled=%s",
        deterministic.score,
        complexity,
        _reflection_enabled(),
    )

    # Step 2 — LLM review: only if enabled AND complexity is MEDIUM or HIGH
    use_llm = _reflection_enabled() and complexity in ("MEDIUM", "HIGH")
    if not use_llm:
        return deterministic

    llm_result, status = _call_llm_reflection(
        report, screening, research, evidence_qa, deterministic
    )
    if llm_result is not None:
        logger.info("run_reflection: LLM review succeeded  status=%s", status)
        return llm_result

    logger.warning(
        "run_reflection: LLM review failed (%s); using deterministic fallback", status
    )
    # Mark source as llm_fallback so caller knows LLM was attempted
    object.__setattr__(deterministic, "reflection_source", "llm_fallback")
    return deterministic


# ---------------------------------------------------------------------------
# reflection_agent LlmAgent — ADK wrapper (for future graph integration)
# ---------------------------------------------------------------------------

_REFLECTION_INSTRUCTION = """You are ReflectionAgent.

Your responsibility is to critically evaluate toxicity reports generated by ToxAgent.

You must behave like an independent scientific reviewer.

Review dimensions:

1. Scientific validity
2. Evidence sufficiency
3. Consistency between conclusions and evidence
4. Missing information
5. Reliability concerns
6. OOD warnings
7. Overconfidence detection

For every report:

Determine:

* Are conclusions supported?
* Are there unsupported claims?
* Are important caveats missing?
* Is confidence appropriate?
* Is additional retrieval needed?

Output ONLY JSON.

Schema:

{
"score": 0-100,
"supported": true,
"missing_evidence": [],
"unsupported_claims": [],
"ood_risk": false,
"needs_revision": false,
"recommended_actions": []
}

Scoring:

90-100:
Strong evidence support.

70-89:
Acceptable but minor weaknesses.

50-69:
Insufficient support.

Below 50:
Major scientific concerns.

If unsupported claims exist,
set "needs_revision": true.
"""

reflection_agent = LlmAgent(
    name="ReflectionAgent",
    model=REFLECTION_MODEL,
    description=(
        "Independent scientific reviewer that evaluates a completed toxicity report "
        "and returns a structured quality assessment with score, evidence gaps, "
        "unsupported claims, and recommended corrective actions."
    ),
    instruction=_REFLECTION_INSTRUCTION,
    tools=[],
    output_key="reflection_result",
)
