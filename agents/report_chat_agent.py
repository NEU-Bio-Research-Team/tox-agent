from __future__ import annotations

import copy
import inspect
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

from .adk_compat import LlmAgent
from .screening_agent import run_screening
from services.firestore_client import get_firestore_availability
from services.firestore_client import fetch_collection_documents
from services.molecule_retriever import retrieve_similar_molecules

try:
    from rdkit import Chem
except Exception:
    Chem = None

CHAT_MODEL = os.getenv("AGENT_MODEL_FAST", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


@dataclass
class ChatMessage:
    role: str  # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReportChatSession:
    session_id: str
    report_state: Dict[str, Any]  # Frozen after init; tools/chat must treat as read-only.
    system_context: str  # Built once in build_report_context().
    history: List[ChatMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    MAX_HISTORY_TURNS: int = 6  # Sliding window = 6 user + assistant pairs.

    def get_trimmed_messages(self) -> List[Dict[str, str]]:
        """Return last MAX_HISTORY_TURNS*2 messages for context window."""
        tail = self.history[-(self.MAX_HISTORY_TURNS * 2) :]
        return [{"role": m.role, "content": m.content} for m in tail]

    def add_turn(self, user_msg: str, assistant_msg: str) -> None:
        self.history.append(ChatMessage(role="user", content=user_msg))
        self.history.append(ChatMessage(role="assistant", content=assistant_msg))
        self.last_active = time.time()


# In-memory session store (replace with Redis/DB in production)
_SESSION_STORE: Dict[str, ReportChatSession] = {}


def create_chat_session(report_state: Dict[str, Any]) -> str:
    """Create a report-anchored chat session with immutable grounding context."""
    session_id = str(uuid.uuid4())
    frozen_state = copy.deepcopy(report_state or {})
    context = build_report_context(frozen_state)
    _SESSION_STORE[session_id] = ReportChatSession(
        session_id=session_id,
        report_state=frozen_state,
        system_context=context,
    )
    return session_id


def get_session(session_id: str) -> Optional[ReportChatSession]:
    return _SESSION_STORE.get(session_id)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_articles(articles: List[Dict[str, Any]], max_articles: int = 5) -> str:
    if not articles:
        return "  (no curated articles available)"

    lines: List[str] = []
    for i, article in enumerate(articles[:max_articles], 1):
        pmid = article.get("pmid", "N/A")
        title = article.get("title", "Untitled")
        year = article.get("year", "?")
        journal = article.get("journal", "Unknown journal")
        score = _safe_float(article.get("relevance_score"), 0.0)
        level = article.get("relevance_level", "?")
        url = str(article.get("pubmed_url", "") or "").strip()

        line = f"  [{i}] PMID:{pmid} | {title} | {journal} ({year}) | relevance={level}({score:.2f})"
        if url:
            line += f" | {url}"
        lines.append(line)

    return "\n".join(lines)


def _compress_section(data: Any, max_chars: int = 600) -> str:
    """Convert dict/list section to readable flat text and truncate if needed."""
    if isinstance(data, str):
        text = data.strip()
    elif data is None:
        return "(not available)"
    else:
        text = json.dumps(data, ensure_ascii=False, indent=None)

    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "... [truncated — ask for details]"
    return text


def _to_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def build_report_context(report_state: Dict[str, Any]) -> str:
    """
    Build immutable system grounding for one report chat session.

    Consumes:
      - final_report
      - evidence_qa_result
      - smiles_input
    """
    final_report = _to_dict(report_state.get("final_report"))
    qa_result = _to_dict(report_state.get("evidence_qa_result"))
    report_metadata = _to_dict(final_report.get("report_metadata"))
    sections = _to_dict(final_report.get("sections"))
    smiles = report_state.get("smiles_input", "N/A")

    compound_name = (
        final_report.get("compound_name")
        or report_metadata.get("compound_name")
        or report_metadata.get("smiles")
        or "Unknown Compound"
    )
    risk_level = final_report.get("risk_level", "UNKNOWN")
    exec_summary = final_report.get("executive_summary", "(not available)")
    raw_confidence = str(qa_result.get("evidence_confidence") or "LOW").upper()
    confidence = raw_confidence if raw_confidence in {"HIGH", "MEDIUM", "LOW"} else "LOW"
    flags = qa_result.get("research_quality_flags") or []
    curated_articles = qa_result.get("curated_articles") or []
    high_rel = int(qa_result.get("high_relevance_count", 0) or 0)
    total_curated = int(qa_result.get("total_articles_curated", len(curated_articles)) or 0)

    clin_tox = (
        final_report.get("clinical_toxicity")
        or final_report.get("toxicity_predictions")
        or sections.get("clinical_toxicity")
        or {}
    )
    mech_tox = (
        final_report.get("mechanism_toxicity")
        or final_report.get("toxicity_mechanisms")
        or sections.get("mechanism_toxicity")
        or {}
    )
    molrag_evidence = (
        final_report.get("molrag_evidence")
        or sections.get("molrag_evidence")
        or sections.get("molrag")
        or {}
    )
    fusion_result = (
        final_report.get("fusion_result")
        or sections.get("fusion_result")
        or {}
    )
    literature_context = (
        final_report.get("literature_context")
        or sections.get("literature_context")
        or {}
    )
    recommendations = final_report.get("recommendations") or sections.get("recommendations") or ""

    calibration_map = {
        "HIGH": "Evidence is WELL-SUPPORTED. You may express confident conclusions.",
        "MEDIUM": "Evidence is MODERATE. Qualify claims as 'based on limited evidence'.",
        "LOW": "Evidence is WEAK or MISSING. Strongly qualify all claims. Do NOT express certainty.",
    }
    calibration = calibration_map.get(str(confidence), "Evidence quality is UNKNOWN. Be conservative.")
    shown_articles = min(len(curated_articles), 5)

    context = f"""=== TOXICITY REPORT — GROUNDED CONTEXT ===
Compound: {compound_name}
SMILES: {smiles}
Overall Risk Level: {risk_level}
Evidence Confidence: {confidence} ({high_rel}/{total_curated} high-relevance articles)
Calibration: {calibration}

=== EXECUTIVE SUMMARY ===
{_compress_section(exec_summary, max_chars=800)}

=== CLINICAL TOXICITY PREDICTIONS ===
{_compress_section(clin_tox, max_chars=800)}

=== TOXICITY MECHANISMS ===
{_compress_section(mech_tox, max_chars=600)}

=== MOLRAG EVIDENCE ===
{_compress_section(molrag_evidence, max_chars=900)}

=== BASELINE/MOLRAG FUSION ===
{_compress_section(fusion_result, max_chars=500)}

=== LITERATURE CONTEXT ===
{_compress_section(literature_context, max_chars=900)}

=== RECOMMENDATIONS ===
{_compress_section(recommendations, max_chars=400)}

=== CURATED LITERATURE EVIDENCE (top {shown_articles} of {total_curated} articles) ===
{_format_articles(curated_articles)}

=== QUALITY FLAGS ===
{', '.join(flags) if flags else 'none'}

=== END OF REPORT CONTEXT ==="""

    return context


def get_article_detail(pmid: str, session_id: str) -> Dict[str, Any]:
    """
    Look up an article by PMID in this session's curated list.
    Returns full article metadata or an explicit error payload.
    """
    session = get_session(session_id)
    if not session:
        return {"error": "session_not_found"}

    qa_result = _to_dict(session.report_state.get("evidence_qa_result"))
    articles = qa_result.get("curated_articles")
    if not isinstance(articles, list):
        articles = []

    needle = str(pmid).strip()
    for article in articles:
        if not isinstance(article, dict):
            continue
        if str(article.get("pmid", "")).strip() == needle:
            return article

    return {"error": f"PMID {pmid} not found in curated list. Do not invent details."}


def check_claim_support(claim: str, session_id: str) -> Dict[str, Any]:
    """
    Estimate claim support from curated literature via keyword overlap.
    SUPPORT levels:
      - SUPPORTED: >=2 matching articles
      - PARTIAL: 1 matching article
      - UNSUPPORTED_IN_REPORT: no matches found in curated list
    """
    session = get_session(session_id)
    if not session:
        return {"error": "session_not_found"}

    qa_result = _to_dict(session.report_state.get("evidence_qa_result"))
    articles = qa_result.get("curated_articles")
    if not isinstance(articles, list):
        articles = []

    tokens = set(re.findall(r"[a-z]{4,}", str(claim).lower()))
    matches: List[Dict[str, Any]] = []

    for article in articles:
        if not isinstance(article, dict):
            continue
        haystack = f"{article.get('title', '')} {article.get('journal', '')}".lower()
        hit_count = sum(1 for token in tokens if token in haystack)
        if hit_count >= 2:
            matches.append(
                {
                    "pmid": article.get("pmid"),
                    "title": article.get("title"),
                    "relevance_score": article.get("relevance_score"),
                    "token_hits": hit_count,
                }
            )

    matches.sort(
        key=lambda item: (int(item.get("token_hits", 0) or 0), _safe_float(item.get("relevance_score"))),
        reverse=True,
    )

    support_level = (
        "SUPPORTED"
        if len(matches) >= 2
        else "PARTIAL"
        if matches
        else "UNSUPPORTED_IN_REPORT"
    )

    return {
        "claim": claim,
        "support_level": support_level,
        "matching_articles": matches[:3],
        "note": "UNSUPPORTED_IN_REPORT means not found in curated list, not that the claim is false.",
    }


def get_report_section(section_name: str, session_id: str) -> Dict[str, Any]:
    """
    Retrieve raw content of a final report section for this session.

    Valid names include:
      - executive_summary
      - clinical_toxicity
      - mechanism_toxicity
            - molrag_evidence
            - fusion_result
            - literature_context
            - ood_assessment
            - inference_context
      - recommendations
      - compound_info
      - risk_level
    """
    section_map = {
        "executive_summary": ["executive_summary", "summary"],
        "clinical_toxicity": ["clinical_toxicity", "toxicity_predictions"],
        "mechanism_toxicity": ["mechanism_toxicity", "toxicity_mechanisms"],
        "molrag_evidence": ["molrag_evidence", "molrag"],
        "fusion_result": ["fusion_result"],
        "literature_context": ["literature_context"],
        "ood_assessment": ["ood_assessment"],
        "inference_context": ["inference_context"],
        "recommendations": ["recommendations"],
        "compound_info": ["compound_info", "report_metadata", "compound_id"],
        "risk_level": ["risk_level"],
    }

    session = get_session(session_id)
    if not session:
        return {"error": "session_not_found"}

    final_report = _to_dict(session.report_state.get("final_report"))
    report_metadata = _to_dict(final_report.get("report_metadata"))
    sections = _to_dict(final_report.get("sections"))

    normalized = str(section_name).lower().replace(" ", "_")
    candidates = section_map.get(normalized, [section_name])

    for key in candidates:
        if key in final_report:
            return {"section": section_name, "content": final_report[key]}
        if key in sections:
            return {"section": section_name, "content": sections[key]}
        if key in report_metadata:
            return {"section": section_name, "content": report_metadata[key]}

    available = sorted(
        {
            *list(final_report.keys()),
            *list(sections.keys()),
            *list(report_metadata.keys()),
        }
    )
    return {
        "error": f"Section '{section_name}' not found in report.",
        "available_keys": available,
    }


def _session_language(session: ReportChatSession) -> str:
    final_report = _to_dict(session.report_state.get("final_report"))
    report_metadata = _to_dict(final_report.get("report_metadata"))
    language = str(report_metadata.get("language") or "vi").strip().lower()
    return "en" if language.startswith("en") else "vi"


def _session_inference_context(session: ReportChatSession) -> Dict[str, Any]:
    final_report = _to_dict(session.report_state.get("final_report"))
    sections = _to_dict(final_report.get("sections"))
    return _to_dict(sections.get("inference_context"))


def rerun_screening(smiles: str, session_id: str) -> Dict[str, Any]:
    """
    Re-run the deterministic screening pipeline for a new or modified SMILES.
    Returns a compact summary payload suitable for chat context windows.
    """
    session = get_session(session_id)
    if not session:
        return {"error": "session_not_found"}

    normalized_smiles = str(smiles or "").strip()
    if len(normalized_smiles) < 3:
        return {"error": "invalid_smiles"}

    inference_context = _session_inference_context(session)
    result = run_screening(
        smiles_input=normalized_smiles,
        language=_session_language(session),
        clinical_threshold=_safe_float(inference_context.get("clinical_threshold_applied"), 0.35),
        mechanism_threshold=0.5,
        inference_backend=str(inference_context.get("inference_backend") or "xsmiles"),
        binary_tox_model=str(inference_context.get("binary_tox_model") or "pretrained_2head_herg_chemberta_model"),
        tox_type_model=str(inference_context.get("tox_type_model") or "tox21_ensemble_3_best"),
        molrag_enabled=True,
        molrag_top_k=5,
        molrag_min_similarity=0.15,
    )

    screening = _to_dict(result.get("screening_result"))
    if not screening:
        return {"error": result.get("screening_error") or "screening_failed"}

    clinical = _to_dict(screening.get("clinical"))
    molrag = _to_dict(screening.get("molrag"))
    fusion = _to_dict(screening.get("fusion_result"))

    return {
        "input_smiles": normalized_smiles,
        "canonical_smiles": screening.get("canonical_smiles"),
        "risk_label": clinical.get("label"),
        "toxicity_score": clinical.get("p_toxic"),
        "is_toxic": clinical.get("is_toxic"),
        "final_verdict": screening.get("final_verdict"),
        "molrag_label": molrag.get("suggested_label"),
        "molrag_confidence": molrag.get("confidence"),
        "tox_classes": (molrag.get("tox_classes") or [])[:5],
        "fusion": {
            "final_label": fusion.get("final_label"),
            "agreement": fusion.get("agreement"),
            "final_confidence": fusion.get("final_confidence"),
        },
        "ood": _to_dict(screening.get("ood_assessment")),
    }


def query_molrag_live(
    smiles: str,
    top_k: int = 5,
    min_similarity: float = 0.3,
) -> Dict[str, Any]:
    """
    Retrieve similar compounds live from Firestore-backed MolRAG retrieval.
    """
    normalized_smiles = str(smiles or "").strip()
    if len(normalized_smiles) < 3:
        return {"error": "invalid_smiles"}

    safe_top_k = max(1, min(_safe_int(top_k, 5), 20))
    safe_min_similarity = _clamp(_safe_float(min_similarity, 0.3), 0.0, 1.0)
    retrieval = retrieve_similar_molecules(
        normalized_smiles,
        top_k=safe_top_k,
        min_similarity=safe_min_similarity,
    )

    matches = retrieval.get("matches") if isinstance(retrieval.get("matches"), list) else []
    analogs: List[Dict[str, Any]] = []
    for match in matches:
        if not isinstance(match, dict):
            continue
        analogs.append(
            {
                "name": match.get("name"),
                "smiles": match.get("smiles"),
                "label": match.get("label"),
                "tox_class": match.get("tox_class") or [],
                "similarity": match.get("similarity"),
                "source": match.get("source"),
                "is_exact_match": bool(match.get("is_exact_match", False)),
            }
        )

    return {
        "query_smiles": retrieval.get("canonical_smiles") or normalized_smiles,
        "db_source": retrieval.get("db_source"),
        "db_size": retrieval.get("db_size"),
        "firestore": retrieval.get("firestore") or get_firestore_availability(),
        "analogs": analogs,
        "error": retrieval.get("error"),
    }


def compare_with_analogs(smiles: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Summarize nearest analogs and label distribution for quick comparison.
    """
    retrieval = query_molrag_live(smiles=smiles, top_k=top_k, min_similarity=0.25)
    if retrieval.get("error"):
        return retrieval

    analogs = retrieval.get("analogs") if isinstance(retrieval.get("analogs"), list) else []
    toxic_count = 0
    non_toxic_count = 0
    unknown_count = 0

    for analog in analogs:
        label = str((analog or {}).get("label") or "").strip().lower()
        if label == "toxic":
            toxic_count += 1
        elif label in {"non-toxic", "non_toxic"}:
            non_toxic_count += 1
        else:
            unknown_count += 1

    return {
        "query_smiles": retrieval.get("query_smiles"),
        "top_match": analogs[0] if analogs else None,
        "label_distribution": {
            "toxic": toxic_count,
            "non_toxic": non_toxic_count,
            "unknown": unknown_count,
        },
        "analogs": analogs,
        "db_source": retrieval.get("db_source"),
    }


def fetch_pubmed_context(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Fetch real-time PubMed summaries for mechanism or compound follow-up questions.
    """
    normalized_query = str(query or "").strip()
    if len(normalized_query) < 2:
        return {"error": "query_too_short"}

    safe_max = max(1, min(_safe_int(max_results, 5), 10))
    try:
        response = httpx.get(
            f"{PUBMED_BASE}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": f"{normalized_query} toxicity",
                "retmax": safe_max,
                "retmode": "json",
                "sort": "relevance",
            },
            timeout=10.0,
        )
        response.raise_for_status()
        search_payload = response.json().get("esearchresult", {})
        pmids = search_payload.get("idlist", [])
        if not isinstance(pmids, list) or not pmids:
            return {
                "query": normalized_query,
                "articles": [],
                "total_found": _safe_int(search_payload.get("count"), 0),
                "note": "No PubMed hits found for this query.",
            }

        summary_response = httpx.get(
            f"{PUBMED_BASE}/esummary.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "json",
            },
            timeout=10.0,
        )
        summary_response.raise_for_status()
        summary_payload = summary_response.json().get("result", {})

        articles: List[Dict[str, Any]] = []
        for pmid in pmids:
            entry = summary_payload.get(pmid, {}) if isinstance(summary_payload, dict) else {}
            if not isinstance(entry, dict):
                continue
            articles.append(
                {
                    "pmid": pmid,
                    "title": entry.get("title"),
                    "year": str(entry.get("pubdate") or "")[:4],
                    "journal": entry.get("source"),
                    "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                }
            )

        return {
            "query": normalized_query,
            "articles": articles,
            "total_found": _safe_int(search_payload.get("count"), len(articles)),
            "note": "Additional live PubMed results outside the frozen report context.",
        }
    except Exception as exc:
        return {"error": f"pubmed_fetch_failed: {type(exc).__name__}: {str(exc)[:180]}"}


def lookup_structural_alerts(smiles: str, max_results: int = 10) -> Dict[str, Any]:
    """
    Match SMILES against SMARTS structural alerts from molrag_knowledge.
    """
    if Chem is None:
        return {"error": "rdkit_not_available"}

    normalized_smiles = str(smiles or "").strip()
    molecule = Chem.MolFromSmiles(normalized_smiles)
    if molecule is None:
        return {"error": "invalid_smiles"}

    safe_limit = max(1, min(_safe_int(max_results, 10), 50))
    knowledge_docs = fetch_collection_documents("molrag_knowledge")

    severity_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    matches: List[Dict[str, Any]] = []

    for doc in knowledge_docs:
        if not isinstance(doc, dict):
            continue
        doc_type = str(doc.get("type") or "").strip().lower()
        if doc_type != "structural_alert":
            continue

        smarts = str(doc.get("smarts") or "").strip()
        if not smarts:
            continue

        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None or not molecule.HasSubstructMatch(pattern):
            continue

        severity = str(doc.get("severity") or doc.get("risk_level") or "UNKNOWN")
        matches.append(
            {
                "doc_id": doc.get("doc_id"),
                "alert_name": doc.get("name") or doc.get("title"),
                "smarts": smarts,
                "severity": severity,
                "tox_class": doc.get("tox_class") or [],
                "mechanism_ref": doc.get("mechanism_ref"),
                "summary": doc.get("summary"),
                "rank": severity_rank.get(str(severity).lower(), 0),
            }
        )

    matches.sort(key=lambda item: (int(item.get("rank", 0)), str(item.get("alert_name") or "")), reverse=True)

    for item in matches:
        item.pop("rank", None)

    return {
        "smiles": normalized_smiles,
        "alerts_found": len(matches),
        "alerts": matches[:safe_limit],
    }


def explain_mechanism(mechanism_id: str) -> Dict[str, Any]:
    """
    Retrieve mechanism details from molrag_knowledge by id, semantic query, or scaffold keywords.

    Supports natural-language prompts such as:
      - "Why does piperidine ring contribute to toxicity?"
      - "explain mitochondrial toxicity"
    """
    raw_query = str(mechanism_id or "").strip()
    if not raw_query:
        return {"error": "mechanism_id_required"}

    normalized_query = raw_query.lower()

    docs = fetch_collection_documents("molrag_knowledge")
    if not docs:
        return {"error": "knowledge_store_empty"}

    stopwords = {
        "why",
        "does",
        "do",
        "what",
        "is",
        "are",
        "the",
        "a",
        "an",
        "to",
        "of",
        "for",
        "and",
        "in",
        "on",
        "about",
        "contribute",
        "contributes",
        "contributing",
        "toxicity",
        "toxic",
        "ring",
        "please",
        "explain",
    }

    query_terms = [
        token
        for token in re.findall(r"[a-z0-9_+-]{3,}", normalized_query)
        if token not in stopwords
    ]
    query_terms = list(dict.fromkeys(query_terms))
    if normalized_query and normalized_query not in query_terms:
        query_terms.append(normalized_query)

    mechanism_candidates: List[Tuple[float, Dict[str, Any], List[str]]] = []
    structural_candidates: List[Tuple[float, Dict[str, Any], List[str]]] = []

    for doc in docs:
        if not isinstance(doc, dict):
            continue
        doc_type = str(doc.get("type") or "").strip().lower()

        doc_id = str(doc.get("doc_id") or "").strip().lower()
        name = str(doc.get("name") or "").strip().lower()
        title = str(doc.get("title") or "").strip().lower()
        summary = str(doc.get("summary") or "").strip().lower()
        clinical_manifestation = str(doc.get("clinical_manifestation") or "").strip().lower()
        associated_scaffolds = " ".join(str(item).strip().lower() for item in (doc.get("associated_scaffolds") or []))
        structural_alerts = " ".join(str(item).strip().lower() for item in (doc.get("structural_alerts") or []))
        tox_class = " ".join(str(item).strip().lower() for item in (doc.get("tox_class") or []))

        searchable = " ".join(
            [
                doc_id,
                name,
                title,
                summary,
                clinical_manifestation,
                associated_scaffolds,
                structural_alerts,
                tox_class,
            ]
        ).strip()

        if not searchable:
            continue

        score = 0.0
        matched_terms: List[str] = []

        if normalized_query in {doc_id, name, title}:
            score += 100.0
            matched_terms.append(normalized_query)
        elif normalized_query and normalized_query in searchable:
            score += 20.0
            matched_terms.append(normalized_query)

        for term in query_terms:
            if not term:
                continue
            if term in {doc_id, name, title}:
                score += 30.0
                matched_terms.append(term)
            elif term in associated_scaffolds:
                score += 16.0
                matched_terms.append(term)
            elif term in structural_alerts:
                score += 12.0
                matched_terms.append(term)
            elif term in searchable:
                score += 6.0
                matched_terms.append(term)

        if score <= 0:
            continue

        deduped_terms = list(dict.fromkeys(matched_terms))
        if doc_type == "mechanism":
            mechanism_candidates.append((score, doc, deduped_terms))
        elif doc_type == "structural_alert":
            structural_candidates.append((score, doc, deduped_terms))

    mechanism_candidates.sort(key=lambda item: item[0], reverse=True)
    structural_candidates.sort(key=lambda item: item[0], reverse=True)

    if mechanism_candidates:
        best_score, selected, matched_terms = mechanism_candidates[0]
        return {
            "doc_id": selected.get("doc_id"),
            "name": selected.get("name") or selected.get("title"),
            "type": selected.get("type"),
            "tox_class": selected.get("tox_class") or [],
            "summary": selected.get("summary"),
            "clinical_manifestation": selected.get("clinical_manifestation"),
            "risk_level": selected.get("risk_level") or selected.get("severity"),
            "associated_scaffolds": selected.get("associated_scaffolds") or [],
            "structural_alerts": selected.get("structural_alerts") or [],
            "key_refs": selected.get("key_refs") or [],
            "source": selected.get("source"),
            "matched_by": "query_semantic",
            "matched_terms": matched_terms[:8],
            "match_score": round(float(best_score), 3),
            "source_scope": "live_molrag_knowledge",
        }

    if structural_candidates:
        best_score, selected, matched_terms = structural_candidates[0]
        return {
            "doc_id": selected.get("doc_id"),
            "name": selected.get("name") or selected.get("title"),
            "type": selected.get("type"),
            "summary": selected.get("summary") or "Matched structural alert related to query.",
            "risk_level": selected.get("risk_level") or selected.get("severity"),
            "smarts": selected.get("smarts"),
            "tox_class": selected.get("tox_class") or [],
            "mechanism_ref": selected.get("mechanism_ref"),
            "matched_by": "structural_alert_query",
            "matched_terms": matched_terms[:8],
            "match_score": round(float(best_score), 3),
            "source_scope": "live_molrag_knowledge",
            "note": "No direct mechanism doc match; using nearest structural alert evidence.",
        }

    return {
        "error": f"Mechanism '{mechanism_id}' not found.",
        "suggestion": "Try a shorter mechanism keyword (e.g., hERG, DILI, mitochondrial toxicity, reactive metabolite).",
    }


REPORT_CHAT_INSTRUCTION = """
You are a toxicology reasoning assistant with access to both frozen report context and live tools.

## REASONING PROTOCOL

Tier 1 (Report-grounded, highest priority)
1. For report-specific questions, use frozen report context first.
2. If a section is truncated, call get_report_section.
3. If a citation claim needs checking, call check_claim_support.
4. If user asks for a PMID in curated evidence, call get_article_detail.

Tier 2 (Live structural analysis)
1. For questions about structural risk or mechanism, call lookup_structural_alerts.
2. For deeper mechanism detail, call explain_mechanism.
3. For analog comparison, call query_molrag_live or compare_with_analogs.

Tier 3 (Re-analysis)
1. Only when user explicitly provides a new SMILES, call rerun_screening.
2. Clearly label re-analysis output as new run, distinct from frozen report.

Tier 4 (Literature expansion)
1. Only when report evidence is insufficient or user asks for more literature, call fetch_pubmed_context.
2. Label these results as additional literature outside original report curation.

Never fabricate tool outputs.
Prefer fewer tool calls when Tier 1 already provides enough evidence.

## CONFIDENCE CALIBRATION

- The report's `Evidence Confidence` level (HIGH/MEDIUM/LOW) must be reflected in your answer tone.
- LOW confidence → Always prepend "Based on limited evidence, ..." to any mechanistic claim.
- MEDIUM → Use "Evidence suggests..." language.
- HIGH → May state conclusions directly, still with section citation.

## CITATION FORMAT

Every factual answer must end with: `[Source: <Section Name> | Evidence: <Confidence Level>]`
For article citations: use `[PMID:XXXXX]` inline.

## SCOPE ENFORCEMENT

- If asked about something NOT in this report and there is no relevant tool evidence: reply exactly
    "Thông tin này không có trong report hiện tại. Report chỉ bao gồm: [list available sections]."
- If Tier 2/Tier 4 tools return relevant evidence, you may answer using that evidence and clearly label it as
    "Supplemental live evidence outside frozen report context".
- Do NOT synthesize knowledge from general training data for compound-specific claims without report evidence or tool observations.
- Do NOT invent PMIDs, scores, or article titles not in the curated list.

## LANGUAGE

Respond in the same language as the user's question (Vietnamese or English).
Technical terms may remain in English even in Vietnamese responses.

## PROHIBITED ACTIONS

- Fabricating PMIDs or citation metadata
- Claiming HIGH confidence when quality flags include "low_relevance_evidence" or "literature_missing"
- Answering synthesis routes, dosing, or clinical treatment questions
- Making absolute safety claims ("this compound is safe") regardless of report content
- Calling rerun_screening when user did not provide explicit replacement SMILES
"""

report_chat_agent = LlmAgent(
    name="ReportChatAgent",
    model=CHAT_MODEL,
    description="Conversational QA agent for per-report toxicology analysis chat.",
    instruction=REPORT_CHAT_INSTRUCTION,
    tools=[
        get_article_detail,
        check_claim_support,
        get_report_section,
        rerun_screening,
        query_molrag_live,
        fetch_pubmed_context,
        lookup_structural_alerts,
        compare_with_analogs,
        explain_mechanism,
    ],
    output_key="chat_response",
)


def estimate_context_tokens(text: str) -> int:
    """Rough estimate: 1 token ~= 4 chars for English/mixed content."""
    return len(str(text)) // 4


def validate_context_budget(
    system_context: str,
    history_messages: List[Dict[str, str]],
    max_tokens: int = 8000,
) -> bool:
    """
    Returns False if total context would exceed budget.
    Typical target:
      - system_context: ~2000-2800 tokens
      - history (6 turns): ~800-1200 tokens
      - total <= 8000 leaves room for generation
    """
    total = estimate_context_tokens(system_context)
    for msg in history_messages:
        total += estimate_context_tokens(msg.get("content", ""))
    return total <= max_tokens


def _trim_history_to_budget(
    system_context: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 8000,
) -> List[Dict[str, str]]:
    """Drop oldest user+assistant turns until context budget fits."""
    trimmed = list(messages)
    while len(trimmed) > 1 and not validate_context_budget(system_context, trimmed, max_tokens=max_tokens):
        trimmed = trimmed[2:]
    if not validate_context_budget(system_context, trimmed, max_tokens=max_tokens):
        return trimmed[-1:]
    return trimmed


def chat_with_report(
    session_id: str,
    user_message: str,
    llm_caller: Callable[..., str],  # signature: (system_prompt, messages) -> str
    max_tool_calls: int = 3,
) -> Tuple[str, Optional[ReportChatSession]]:
    """
    Execute one chat turn against a frozen report session.

    Design notes:
    - system_context is injected from session (frozen), not rebuilt per-turn.
    - Rolling history applies via get_trimmed_messages().
    - max_tool_calls is forwarded only if llm_caller supports it.
    """
    session = get_session(session_id)
    if session is None:
        return "Session expired or not found. Please restart the analysis.", None

    messages = session.get_trimmed_messages()
    messages.append({"role": "user", "content": user_message})
    messages = _trim_history_to_budget(session.system_context, messages)

    system_prompt = session.system_context + "\n\n" + REPORT_CHAT_INSTRUCTION
    llm_params = inspect.signature(llm_caller).parameters
    if "max_tool_calls" in llm_params:
        response = llm_caller(system_prompt, messages, max_tool_calls=max_tool_calls)
    else:
        response = llm_caller(system_prompt, messages)

    response_text = str(response)
    session.add_turn(user_message, response_text)
    return response_text, session


__all__ = [
    "CHAT_MODEL",
    "ChatMessage",
    "ReportChatSession",
    "build_report_context",
    "chat_with_report",
    "check_claim_support",
    "compare_with_analogs",
    "create_chat_session",
    "estimate_context_tokens",
    "explain_mechanism",
    "fetch_pubmed_context",
    "get_article_detail",
    "get_report_section",
    "get_session",
    "lookup_structural_alerts",
    "query_molrag_live",
    "report_chat_agent",
    "rerun_screening",
    "validate_context_budget",
]
