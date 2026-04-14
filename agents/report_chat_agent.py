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

from .adk_compat import LlmAgent

CHAT_MODEL = os.getenv("AGENT_MODEL_FAST", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))


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


REPORT_CHAT_INSTRUCTION = """
You are a specialized toxicology report assistant. Your ONLY knowledge source is the
specific compound report injected into your system context above.

## REASONING PROTOCOL (follow exactly in order)

1. LOCATE: Identify which section of the grounded report answers the question.
2. VERIFY: If making a specific claim about evidence or citations, call check_claim_support first.
3. RETRIEVE: If user asks about a specific PMID or article, call get_article_detail.
4. EXPAND: If context shows "[truncated]" for a section, call get_report_section to get full content.
5. ANSWER: Synthesize from tool results + report context. Always cite the section name.

## CONFIDENCE CALIBRATION

- The report's `Evidence Confidence` level (HIGH/MEDIUM/LOW) must be reflected in your answer tone.
- LOW confidence → Always prepend "Based on limited evidence, ..." to any mechanistic claim.
- MEDIUM → Use "Evidence suggests..." language.
- HIGH → May state conclusions directly, still with section citation.

## CITATION FORMAT

Every factual answer must end with: `[Source: <Section Name> | Evidence: <Confidence Level>]`
For article citations: use `[PMID:XXXXX]` inline.

## SCOPE ENFORCEMENT

- If asked about something NOT in this report: reply exactly
  "Thông tin này không có trong report hiện tại. Report chỉ bao gồm: [list available sections]."
- Do NOT synthesize knowledge from general training data for compound-specific claims.
- Do NOT invent PMIDs, scores, or article titles not in the curated list.

## LANGUAGE

Respond in the same language as the user's question (Vietnamese or English).
Technical terms may remain in English even in Vietnamese responses.

## PROHIBITED ACTIONS

- Fabricating PMIDs or citation metadata
- Claiming HIGH confidence when quality flags include "low_relevance_evidence" or "literature_missing"
- Answering synthesis routes, dosing, or clinical treatment questions
- Making absolute safety claims ("this compound is safe") regardless of report content
"""

report_chat_agent = LlmAgent(
    name="ReportChatAgent",
    model=CHAT_MODEL,
    description="Conversational QA agent for per-report toxicology analysis chat.",
    instruction=REPORT_CHAT_INSTRUCTION,
    tools=[get_article_detail, check_claim_support, get_report_section],
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
    "create_chat_session",
    "estimate_context_tokens",
    "get_article_detail",
    "get_report_section",
    "get_session",
    "report_chat_agent",
    "validate_context_budget",
]
