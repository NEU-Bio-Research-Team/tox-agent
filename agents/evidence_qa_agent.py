from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Tuple

from .adk_compat import LlmAgent

QA_MODEL = os.getenv("AGENT_MODEL_FAST", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

_TOX_TERMS = (
    "toxicity",
    "toxic",
    "hepatotoxicity",
    "cardiotoxicity",
    "nephrotoxicity",
    "genotoxicity",
    "mutagenicity",
    "carcinogenicity",
    "cytotoxicity",
    "mechanism",
    "adverse",
    "safety",
    "risk",
)


def _to_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _to_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    return re.sub(r"\s+", " ", text)


def _normalize_key(value: str) -> str:
    lowered = value.lower().strip()
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


def _extract_compound_terms(research: Dict[str, Any]) -> List[str]:
    compound_info = _to_dict(research.get("compound_info"))
    query_name = _clean_text(research.get("query_name_used"))
    names = [
        query_name,
        _clean_text(compound_info.get("common_name")),
        _clean_text(compound_info.get("iupac_name")),
    ]
    terms: List[str] = []
    for name in names:
        if not name:
            continue
        for token in re.split(r"[\s,;:/()\-]+", name.lower()):
            token = token.strip()
            if len(token) >= 3 and token not in terms:
                terms.append(token)
    return terms


def _score_article(article: Dict[str, Any], compound_terms: List[str]) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    score = 0.0

    title = _clean_text(article.get("title"))
    journal = _clean_text(article.get("journal"))
    pmid = _clean_text(article.get("pmid"))
    year_raw = _clean_text(article.get("year"))
    haystack = f"{title} {journal}".lower()

    if pmid:
        score += 0.15
        reasons.append("has_pmid")
    if title:
        score += 0.1
        reasons.append("has_title")
    if journal:
        score += 0.05
        reasons.append("has_journal")

    tox_hits = sum(1 for term in _TOX_TERMS if term in haystack)
    if tox_hits:
        score += min(0.4, 0.08 * tox_hits)
        reasons.append(f"tox_terms={tox_hits}")

    compound_hits = sum(1 for term in compound_terms if term and term in haystack)
    if compound_hits:
        score += min(0.2, 0.1 * compound_hits)
        reasons.append(f"compound_terms={compound_hits}")

    if year_raw.isdigit():
        year = int(year_raw)
        if year >= 2018:
            score += 0.1
            reasons.append("recent_paper")
        elif year >= 2010:
            score += 0.05
            reasons.append("moderately_recent")

    return min(score, 1.0), reasons


def _dedupe_articles(articles: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    seen: set[str] = set()
    unique: List[Dict[str, Any]] = []
    removed = 0

    for article in articles:
        pmid = _clean_text(article.get("pmid"))
        title = _clean_text(article.get("title"))
        key = pmid if pmid else _normalize_key(title)
        if not key:
            key = f"idx-{len(unique)}"
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        unique.append(article)

    return unique, removed


def _confidence_from_quality(high_rel_count: int, total_curated: int, flags: List[str]) -> str:
    severe_flags = {"research_result_missing", "literature_missing", "literature_error_present"}
    if any(flag in severe_flags for flag in flags):
        return "LOW"
    if total_curated >= 3 and high_rel_count >= 2:
        return "HIGH"
    if total_curated >= 1 and high_rel_count >= 1:
        return "MEDIUM"
    return "LOW"


def run_evidence_qa(
    research_result: Dict[str, Any] | None,
    top_k: int = 5,
    high_relevance_threshold: float = 0.55,
) -> Dict[str, Any]:
    """Quality-gate and normalize literature evidence without LLM calls."""
    research = _to_dict(research_result)
    flags: List[str] = []

    if not research:
        flags.append("research_result_missing")
        return {
            "evidence_qa_result": {
                "research_result_sanitized": {},
                "curated_articles": [],
                "total_articles_in": 0,
                "total_articles_curated": 0,
                "high_relevance_count": 0,
                "evidence_confidence": "LOW",
                "research_quality_flags": flags,
            },
            "evidence_qa_error": "research_result_missing",
        }

    compound_info = _to_dict(research.get("compound_info"))
    literature = _to_dict(research.get("literature"))
    articles_in = _to_list(literature.get("articles"))

    if not compound_info:
        flags.append("compound_info_missing")
    if not literature:
        flags.append("literature_missing")
    if literature.get("error"):
        flags.append("literature_error_present")
    if not articles_in:
        flags.append("no_articles_found")

    normalized_input: List[Dict[str, Any]] = []
    for item in articles_in:
        article = _to_dict(item)
        normalized_input.append(
            {
                "pmid": _clean_text(article.get("pmid")),
                "title": _clean_text(article.get("title")),
                "authors": _to_list(article.get("authors"))[:5],
                "year": _clean_text(article.get("year")),
                "journal": _clean_text(article.get("journal")),
                "pubmed_url": _clean_text(article.get("pubmed_url")),
            }
        )

    deduped, removed_count = _dedupe_articles(normalized_input)
    if removed_count:
        flags.append(f"duplicate_articles_removed:{removed_count}")

    compound_terms = _extract_compound_terms(research)
    scored: List[Dict[str, Any]] = []
    for article in deduped:
        score, reasons = _score_article(article, compound_terms)
        quality = "HIGH" if score >= 0.70 else "MEDIUM" if score >= 0.45 else "LOW"
        enriched = dict(article)
        enriched["relevance_score"] = round(score, 3)
        enriched["relevance_level"] = quality
        enriched["qa_reasons"] = reasons
        scored.append(enriched)

    scored.sort(
        key=lambda x: (
            float(x.get("relevance_score", 0.0)),
            1 if x.get("year", "").isdigit() else 0,
            x.get("year", ""),
        ),
        reverse=True,
    )

    curated = scored[: max(1, int(top_k))]
    high_rel_count = sum(
        1 for article in curated if float(article.get("relevance_score", 0.0)) >= high_relevance_threshold
    )
    if curated and high_rel_count == 0:
        flags.append("low_relevance_evidence")

    confidence = _confidence_from_quality(high_rel_count, len(curated), flags)

    sanitized_research = dict(research)
    sanitized_literature = dict(literature)
    sanitized_literature["articles"] = curated
    sanitized_literature["total_curated"] = len(curated)
    sanitized_literature["total_incoming"] = len(articles_in)
    sanitized_literature["high_relevance_count"] = high_rel_count
    sanitized_research["literature"] = sanitized_literature

    qa_result = {
        "research_result_sanitized": sanitized_research,
        "curated_articles": curated,
        "total_articles_in": len(articles_in),
        "total_articles_curated": len(curated),
        "high_relevance_count": high_rel_count,
        "evidence_confidence": confidence,
        "research_quality_flags": flags,
    }

    return {
        "evidence_qa_result": qa_result,
        "evidence_qa_error": None,
    }


evidence_qa_agent = LlmAgent(
    name="EvidenceQAAgent",
    model=QA_MODEL,
    description="Quality-gate research evidence before report writing.",
    instruction="""
You are an evidence quality controller for toxicity literature context.

Task:
1. Read research_result from session state.
2. Validate and sanitize literature evidence:
   - remove duplicates
   - score relevance
   - assign confidence
3. Return JSON for key evidence_qa_result with:
   - research_result_sanitized
   - curated_articles
   - total_articles_in
   - total_articles_curated
   - high_relevance_count
   - evidence_confidence
   - research_quality_flags

Rules:
- Never invent PMID or citation fields.
- If research_result is missing/invalid, return LOW confidence with clear flags.
""",
    tools=[],
    output_key="evidence_qa_result",
)

