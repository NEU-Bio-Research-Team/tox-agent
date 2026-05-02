from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import httpx

from services.genai_runtime import (
    build_genai_client_candidates,
    call_with_retry,
    dedupe_strings,
    is_model_unavailable_error,
    is_resource_exhausted_error,
)

try:
    from google import genai
except Exception:
    genai = None

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY", "").strip()
SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
EUROPE_PMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _truncate_text(value: str, limit: int = 1600) -> str:
    text = _clean_text(value)
    if len(text) <= limit:
        return text
    return f"{text[: max(limit - 3, 0)].rstrip()}..."


def _pubmed_get_with_retry(url: str, timeout: float = 15.0, max_retries: int = 3) -> httpx.Response:
    """GET with light exponential backoff for PubMed rate-limit and overload responses."""
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            response = httpx.get(url, timeout=timeout)
            if response.status_code in {429, 503} and attempt < max_retries - 1:
                time.sleep(2**attempt)
                continue
            return response
        except httpx.RequestError as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
                continue
            raise

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("pubmed_request_failed")


def _pubchem_get_with_retry(url: str, timeout: float = 12.0, max_retries: int = 3) -> httpx.Response:
    """GET with exponential backoff for transient PubChem overload/rate-limit errors."""
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            response = httpx.get(url, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < max_retries - 1:
                time.sleep(2**attempt)
                continue
            return response
        except httpx.RequestError as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
                continue
            raise

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("pubchem_request_failed")


def _looks_like_smiles(value: str) -> bool:
    text = (value or "").strip()
    if len(text) < 8:
        return False
    return re.fullmatch(r"[A-Za-z0-9@+\-\[\]\(\)=#$\\/.%]+", text) is not None


def _get_canonical_smiles(smiles: str) -> Optional[str]:
    """Return RDKit canonical SMILES, or None if RDKit is unavailable or the SMILES is invalid."""
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol)
    except Exception:
        pass
    return None


def _build_rdkit_metadata_fallback(smiles: str) -> Dict[str, Any]:
    """Best-effort local metadata fallback when PubChem is unavailable."""
    output = {
        "cid": None,
        "iupac_name": None,
        "common_name": None,
        "molecular_formula": None,
        "molecular_weight": None,
        "synonyms": [],
        "pubchem_url": None,
        "error": "pubchem_unavailable",
        "fallback_source": "rdkit",
    }
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            output["error"] = "invalid_smiles_for_rdkit_fallback"
            return output

        output["molecular_formula"] = rdMolDescriptors.CalcMolFormula(mol)
        output["molecular_weight"] = round(float(Descriptors.MolWt(mol)), 4)
        output["error"] = "pubchem_unavailable_rdkit_fallback"
        return output
    except Exception as exc:
        output["error"] = f"rdkit_fallback_failed:{type(exc).__name__}:{str(exc)[:120]}"
        return output


def _resolve_literature_query(compound_name: str, compound_smiles: Optional[str]) -> str:
    """Resolve a PubMed-friendly compound query, upgrading SMILES-like input to a known name when possible."""
    preferred = (compound_name or "").strip()
    smiles_hint = (compound_smiles or "").strip()
    candidate_smiles = smiles_hint or preferred

    if _looks_like_smiles(preferred) or _looks_like_smiles(candidate_smiles):
        lookup = get_compound_info_pubchem(candidate_smiles)
        lookup_name = str(lookup.get("common_name") or lookup.get("iupac_name") or "").strip()
        if lookup_name and not _looks_like_smiles(lookup_name):
            preferred = lookup_name
        else:
            # Fallback: use molecular formula as query base — far more meaningful than raw SMILES
            mol_formula = _clean_text(lookup.get("molecular_formula"))
            if mol_formula:
                preferred = mol_formula

    if not preferred:
        preferred = candidate_smiles or "unknown compound"

    return preferred


def _pubmed_search_once(query: str, max_results: int) -> Dict[str, Any]:
    """Run one PubMed search/summarize pass for a single query."""
    encoded_query = urllib.parse.quote(query)
    api_key_param = f"&api_key={PUBMED_API_KEY}" if PUBMED_API_KEY else ""

    search_resp = _pubmed_get_with_retry(
        f"{PUBMED_BASE}/esearch.fcgi?db=pubmed&term={encoded_query}"
        f"&retmax={max_results}&retmode=json&sort=relevance{api_key_param}",
        timeout=15.0,
    )
    search_resp.raise_for_status()
    search_data = search_resp.json().get("esearchresult", {})
    pmids = search_data.get("idlist", [])
    total_found = int(search_data.get("count", 0) or 0)

    if not pmids:
        return {
            "articles": [],
            "total_found": total_found,
            "query_used": query,
            "search_source": "pubmed",
            "fallback_used": False,
            "error": None,
        }

    ids_str = ",".join(pmids)
    summary_resp = _pubmed_get_with_retry(
        f"{PUBMED_BASE}/esummary.fcgi?db=pubmed&id={ids_str}"
        f"&retmode=json{api_key_param}",
        timeout=15.0,
    )
    summary_resp.raise_for_status()
    summary_data = summary_resp.json().get("result", {})
    abstracts = fetch_pubmed_abstracts(pmids)

    articles: List[Dict[str, Any]] = []
    for pmid in pmids:
        article = _article_from_pubmed_summary(str(pmid), summary_data)
        abstract_info = abstracts.get(str(pmid), {})
        abstract_text = _clean_text(abstract_info.get("text")) if isinstance(abstract_info, dict) else ""
        if abstract_text:
            article["abstract"] = abstract_text
            article["abstract_source"] = _clean_text(abstract_info.get("source") or "pubmed")
            article["snippet"] = _truncate_text(abstract_text, 240)
        articles.append(article)

    return {
        "articles": articles,
        "total_found": total_found,
        "query_used": query,
        "search_source": "pubmed",
        "fallback_used": any(
            str(article.get("abstract_source") or "") not in {"", "none", "pubmed"}
            for article in articles
        ),
        "error": None,
    }


def _parse_pubmed_abstracts(xml_text: str) -> Dict[str, Dict[str, str]]:
    result: Dict[str, Dict[str, str]] = {}
    root = ET.fromstring(xml_text)

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        if pmid_el is None or not _clean_text(pmid_el.text):
            continue

        parts: List[str] = []
        for abstract_el in article.findall(".//Abstract/AbstractText"):
            section_label = _clean_text(abstract_el.attrib.get("Label"))
            segment = " ".join(piece.strip() for piece in abstract_el.itertext() if piece and piece.strip())
            if not segment:
                continue
            if section_label:
                parts.append(f"{section_label}: {segment}")
            else:
                parts.append(segment)

        if parts:
            result[_clean_text(pmid_el.text)] = {
                "text": _truncate_text(" ".join(parts), 1800),
                "source": "pubmed",
            }

    return result


def _fetch_abstracts_europe_pmc(pmids: List[str]) -> Dict[str, Dict[str, str]]:
    """Fallback abstract fetch via Europe PMC."""
    result: Dict[str, Dict[str, str]] = {}

    for pmid in pmids[:5]:
        try:
            response = httpx.get(
                f"{EUROPE_PMC_BASE}/search?query=EXT_ID:{pmid}%20AND%20SRC:MED&format=json&resultType=core",
                timeout=10.0,
            )
            if response.status_code != 200:
                continue

            items = response.json().get("resultList", {}).get("result", [])
            if not items:
                continue

            abstract_text = _truncate_text(items[0].get("abstractText") or "", 1800)
            if abstract_text:
                result[str(pmid)] = {
                    "text": abstract_text,
                    "source": "europepmc",
                }
        except Exception:
            continue

    return result


def _fetch_abstracts_semantic_scholar(pmids: List[str]) -> Dict[str, Dict[str, str]]:
    """Last-resort abstract fetch via Semantic Scholar."""
    result: Dict[str, Dict[str, str]] = {}

    for pmid in pmids[:5]:
        try:
            response = httpx.get(
                f"{SEMANTIC_SCHOLAR_BASE}/paper/PMID:{pmid}?fields=title,abstract,year,externalIds",
                timeout=10.0,
                headers={"User-Agent": "tox-agent-research/1.0"},
            )
            if response.status_code != 200:
                continue

            data = response.json()
            abstract_text = _truncate_text(data.get("abstract") or "", 1800)
            if abstract_text:
                result[str(pmid)] = {
                    "text": abstract_text,
                    "source": "semanticscholar",
                }
        except Exception:
            continue

    return result


def fetch_pubmed_abstracts(pmids: List[str]) -> Dict[str, Dict[str, str]]:
    """Fetch abstracts with PubMed primary source and Europe PMC/Semantic Scholar fallback."""
    if not pmids:
        return {}

    primary: Dict[str, Dict[str, str]] = {}
    ids_str = ",".join(str(pmid) for pmid in pmids[:5])
    api_key_param = f"&api_key={PUBMED_API_KEY}" if PUBMED_API_KEY else ""

    try:
        response = _pubmed_get_with_retry(
            f"{PUBMED_BASE}/efetch.fcgi?db=pubmed&id={ids_str}&rettype=abstract&retmode=xml{api_key_param}",
            timeout=20.0,
        )
        if response.status_code == 200:
            primary = _parse_pubmed_abstracts(response.text)
    except Exception:
        primary = {}

    missing_pmids = [str(pmid) for pmid in pmids if str(pmid) not in primary]
    if not missing_pmids:
        return primary

    europe = _fetch_abstracts_europe_pmc(missing_pmids)
    combined = {**primary, **europe}

    missing_after_europe = [pmid for pmid in missing_pmids if pmid not in combined]
    if not missing_after_europe:
        return combined

    semantic = _fetch_abstracts_semantic_scholar(missing_after_europe)
    return {**combined, **semantic}


def _article_from_pubmed_summary(pmid: str, summary_data: Dict[str, Any]) -> Dict[str, Any]:
    article = summary_data.get(pmid, {})
    authors = [author.get("name", "").strip() for author in article.get("authors", [])[:4]]
    authors = [name for name in authors if name]
    title = article.get("title", "N/A")

    return {
        "pmid": str(pmid),
        "title": title,
        "authors": ", ".join(authors) if authors else "N/A",
        "year": str(article.get("pubdate", ""))[:4],
        "journal": article.get("source", "N/A"),
        "snippet": _truncate_text(title, 150),
        "abstract": "",
        "abstract_source": "none",
        "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        "search_source": "pubmed",
    }


def _search_europe_pmc_literature(query: str, max_results: int) -> Dict[str, Any]:
    encoded_query = urllib.parse.quote(query)

    response = httpx.get(
        f"{EUROPE_PMC_BASE}/search?query={encoded_query}&format=json&pageSize={max_results}&resultType=core",
        timeout=15.0,
    )
    response.raise_for_status()
    payload = response.json()
    items = payload.get("resultList", {}).get("result", [])

    articles: List[Dict[str, Any]] = []
    for item in items[:max_results]:
        pmid = _clean_text(item.get("pmid") or item.get("id"))
        abstract_text = _truncate_text(item.get("abstractText") or "", 1800)
        articles.append(
            {
                "pmid": pmid,
                "title": item.get("title") or "N/A",
                "authors": item.get("authorString") or "N/A",
                "year": str(item.get("pubYear") or ""),
                "journal": item.get("journalTitle") or "N/A",
                "snippet": _truncate_text(abstract_text or item.get("title") or "", 240),
                "abstract": abstract_text,
                "abstract_source": "europepmc" if abstract_text else "none",
                "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                "search_source": "europepmc",
            }
        )

    return {
        "articles": articles,
        "total_found": int(payload.get("hitCount", 0) or 0),
        "query_used": query,
        "search_source": "europepmc",
        "fallback_used": bool(articles),
        "error": None,
    }


def _search_semantic_scholar_literature(query: str, max_results: int) -> Dict[str, Any]:
    encoded_query = urllib.parse.quote(query)

    response = httpx.get(
        f"{SEMANTIC_SCHOLAR_BASE}/paper/search?query={encoded_query}&limit={max_results}&fields=title,abstract,year,authors,journal,externalIds",
        timeout=15.0,
        headers={"User-Agent": "tox-agent-research/1.0"},
    )
    response.raise_for_status()
    payload = response.json()
    items = payload.get("data", [])

    articles: List[Dict[str, Any]] = []
    for item in items[:max_results]:
        external_ids = item.get("externalIds") or {}
        pmid = _clean_text(external_ids.get("PubMed") or external_ids.get("PubMedCentral") or "")
        authors = [author.get("name", "").strip() for author in item.get("authors", [])[:4]]
        authors = [name for name in authors if name]
        abstract_text = _truncate_text(item.get("abstract") or "", 1800)
        journal_info = item.get("journal") or {}
        articles.append(
            {
                "pmid": pmid,
                "title": item.get("title") or "N/A",
                "authors": ", ".join(authors) if authors else "N/A",
                "year": str(item.get("year") or ""),
                "journal": journal_info.get("name") or "N/A",
                "snippet": _truncate_text(abstract_text or item.get("title") or "", 240),
                "abstract": abstract_text,
                "abstract_source": "semanticscholar" if abstract_text else "none",
                "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                "search_source": "semanticscholar",
            }
        )

    return {
        "articles": articles,
        "total_found": int(payload.get("total", len(articles)) or 0),
        "query_used": query,
        "search_source": "semanticscholar",
        "fallback_used": bool(articles),
        "error": None,
    }


def _build_synthesis_prompt(
    *,
    articles: List[Dict[str, Any]],
    compound_name: str,
    compound_smiles: str,
    language: str,
    using_title_only: bool,
) -> str:
    paper_blocks: List[str] = []
    for index, article in enumerate(articles[:5], start=1):
        summary_text = _clean_text(article.get("abstract") or article.get("snippet") or article.get("title"))
        evidence_label = "Title/snippet" if using_title_only else "Abstract"
        paper_blocks.append(
            "\n".join(
                [
                    f"[Paper {index}] PMID:{article.get('pmid') or 'N/A'} ({article.get('year') or 'N/A'})",
                    f"Title: {article.get('title') or 'Untitled'}",
                    f"Source: {article.get('abstract_source') or article.get('search_source') or 'unknown'}",
                    f"{evidence_label}: {summary_text}",
                ]
            )
        )

    language_instruction = (
        "Respond entirely in Vietnamese."
        if str(language).strip().lower().startswith("vi")
        else "Respond entirely in English."
    )
    evidence_mode = "title/snippet fallback" if using_title_only else "abstract-backed"
    papers_text = "\n\n".join(paper_blocks)

    return f"""You are a pharmacotoxicology literature synthesis assistant.

Compound: {compound_name}
SMILES: {compound_smiles or 'not provided'}
Evidence mode: {evidence_mode}
{language_instruction}

Papers:
{papers_text}

Return a JSON object with exactly these keys:
{{
  "consensus_mechanisms": ["..."],
  "key_targets": ["..."],
  "dose_response_signals": ["..."],
  "conflicting_findings": ["..."],
  "confidence_level": "high|medium|low",
  "synthesis_text": "long narrative synthesis",
  "pmids_used": ["..."],
  "evidence_basis": "abstract|title_only"
}}

Rules:
- Only use evidence explicitly present in the supplied papers.
- If evidence mode is title/snippet fallback, keep confidence low unless titles/snippets strongly agree.
- Prefer a detailed synthesis_text of 4-6 sentences.
- Mention disagreements when present rather than smoothing them over.
"""


def _derive_confidence_level(
    *,
    papers_with_content: int,
    fallback_count: int,
    using_title_only: bool,
) -> str:
    if using_title_only or papers_with_content <= 1:
        return "low"
    if papers_with_content >= 4 and fallback_count <= 1:
        return "high"
    return "medium"


def _deterministic_literature_synthesis(
    *,
    articles: List[Dict[str, Any]],
    language: str,
    error: str | None = None,
) -> Dict[str, Any]:
    papers_with_abstract = [article for article in articles if _clean_text(article.get("abstract"))]
    using_title_only = not papers_with_abstract and bool(articles)
    source_coverage: Dict[str, int] = {}
    for article in papers_with_abstract or articles:
        source_key = _clean_text(article.get("abstract_source") or article.get("search_source") or "unknown")
        source_coverage[source_key] = source_coverage.get(source_key, 0) + 1

    confidence_level = _derive_confidence_level(
        papers_with_content=len(papers_with_abstract),
        fallback_count=sum(count for source, count in source_coverage.items() if source != "pubmed"),
        using_title_only=using_title_only,
    )
    pmids_used = [str(article.get("pmid") or "").strip() for article in articles[:5] if str(article.get("pmid") or "").strip()]

    if not articles:
        synthesis_text = (
            "Khong tim thay bai bao phu hop de tong hop."
            if str(language).strip().lower().startswith("vi")
            else "No suitable papers were found for synthesis."
        )
    elif using_title_only:
        synthesis_text = (
            "Chi co tieu de hoac snippet nen tong hop nay co do tin cay thap va can doc abstract/day du de xac nhan co che doc tinh."
            if str(language).strip().lower().startswith("vi")
            else "Only titles or snippets were available, so this synthesis has low confidence and should be confirmed with full abstracts or full text."
        )
    else:
        synthesis_text = (
            "Tong hop tam thoi duoc tao tu abstract da lay duoc; nen uu tien review thu cong cac paper dau danh sach de xac nhan co che, target va bat ky mau thuan nao."
            if str(language).strip().lower().startswith("vi")
            else "A provisional synthesis was built from the available abstracts; manual review of the top papers is still recommended to confirm mechanisms, targets, and any contradictions."
        )

    return {
        "consensus_mechanisms": [],
        "key_targets": [],
        "dose_response_signals": [],
        "conflicting_findings": [],
        "confidence_level": confidence_level,
        "synthesis_text": synthesis_text,
        "papers_with_content": len(papers_with_abstract),
        "pmids_used": pmids_used,
        "source_coverage": source_coverage,
        "evidence_basis": "title_only" if using_title_only else "abstract",
        "error": error,
    }


def get_compound_info_pubchem(smiles: str) -> Dict[str, Any]:
    """Resolve compound metadata from PubChem by SMILES.

    Call this first in the research stage to obtain CID and preferred naming
    fields used by downstream literature and bioassay lookups.

    Args:
        smiles: Valid SMILES string.

    Returns:
        Dict including ``cid``, ``iupac_name``, ``common_name``,
        ``molecular_formula``, ``molecular_weight``, ``synonyms``,
        ``pubchem_url``, and ``error``.
    """
    encoded = urllib.parse.quote(smiles, safe="")
    try:
        cid_resp = _pubchem_get_with_retry(
            f"{PUBCHEM_BASE}/compound/smiles/{encoded}/cids/JSON",
            timeout=10.0,
        )
        cid_resp.raise_for_status()
        cid_list = cid_resp.json().get("IdentifierList", {}).get("CID", [])
        if not cid_list:
            # Retry with RDKit canonical SMILES — PubChem may not recognize non-canonical forms
            canonical = _get_canonical_smiles(smiles)
            if canonical and canonical != smiles:
                enc_canonical = urllib.parse.quote(canonical, safe="")
                try:
                    retry_resp = _pubchem_get_with_retry(
                        f"{PUBCHEM_BASE}/compound/smiles/{enc_canonical}/cids/JSON",
                        timeout=10.0,
                    )
                    if retry_resp.status_code == 200:
                        cid_list = retry_resp.json().get("IdentifierList", {}).get("CID", [])
                except Exception:
                    pass
        if not cid_list:
            fallback = _build_rdkit_metadata_fallback(smiles)
            fallback["error"] = "cid_not_found_rdkit_fallback"
            return fallback
        cid = cid_list[0]

        props_resp = _pubchem_get_with_retry(
            f"{PUBCHEM_BASE}/compound/cid/{cid}/property/"
            "IUPACName,MolecularFormula,MolecularWeight/JSON",
            timeout=10.0,
        )
        props_resp.raise_for_status()
        props = props_resp.json().get("PropertyTable", {}).get("Properties", [{}])[0]

        syn_resp = _pubchem_get_with_retry(
            f"{PUBCHEM_BASE}/compound/cid/{cid}/synonyms/JSON",
            timeout=10.0,
        )
        synonyms: List[str] = []
        if syn_resp.status_code == 200:
            synonyms = (
                syn_resp.json()
                .get("InformationList", {})
                .get("Information", [{}])[0]
                .get("Synonym", [])[:5]
            )

        return {
            "cid": cid,
            "iupac_name": props.get("IUPACName"),
            "common_name": synonyms[0] if synonyms else None,
            "molecular_formula": props.get("MolecularFormula"),
            "molecular_weight": props.get("MolecularWeight"),
            "synonyms": synonyms,
            "pubchem_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
            "error": None,
            "fallback_source": "pubchem",
        }
    except Exception as exc:
        fallback = _build_rdkit_metadata_fallback(smiles)
        fallback["error"] = str(exc)
        return fallback


def search_toxicity_literature(
    compound_name: str,
    max_results: int = 5,
    compound_smiles: Optional[str] = None,
) -> Dict[str, Any]:
    """Search toxicity/mechanism literature with PubMed primary source and search fallbacks.

    Args:
        compound_name: Name to query (prefer common name from PubChem).
        max_results: Maximum number of returned article summaries (capped at 10).

    Returns:
        Dict containing ``articles`` (list), ``total_found``, ``query_used``,
        ``search_source``, ``fallback_used`` and ``error``.
    """
    max_results = min(max_results, 10)
    resolved_name = _resolve_literature_query(compound_name, compound_smiles)
    query_candidates: List[str] = []
    primary_query = f"{resolved_name} toxicity mechanism"
    query_candidates.append(primary_query)

    if compound_smiles and _looks_like_smiles(resolved_name):
        retry_info = get_compound_info_pubchem(compound_smiles)
        retry_name = _clean_text(retry_info.get("common_name") or retry_info.get("iupac_name"))
        if retry_name and not _looks_like_smiles(retry_name):
            query_candidates.append(f"{retry_name} toxicity mechanism")
            query_candidates.append(f"{retry_name} adverse effects")

    # Always include a structure-keyword fallback so PubMed still has a chance if naming lookup fails.
    if compound_smiles:
        query_candidates.append(f"{compound_smiles} toxicity mechanism")

    query_candidates = dedupe_strings([q for q in query_candidates if _clean_text(q)])
    pubmed_error: str | None = None
    last_pubmed_result: Dict[str, Any] | None = None

    for query in query_candidates:
        try:
            pubmed_result = _pubmed_search_once(query=query, max_results=max_results)
            last_pubmed_result = pubmed_result
            if pubmed_result.get("articles"):
                return pubmed_result
        except Exception as exc:
            pubmed_error = str(exc)

    for fallback_search in (_search_europe_pmc_literature, _search_semantic_scholar_literature):
        try:
            for query in query_candidates:
                fallback_result = fallback_search(query, max_results)
                if fallback_result.get("articles"):
                    fallback_result["fallback_used"] = True
                    fallback_result["error"] = pubmed_error
                    return fallback_result
        except Exception:
            continue

    return {
        "articles": [],
        "total_found": int((last_pubmed_result or {}).get("total_found") or 0),
        "query_used": (last_pubmed_result or {}).get("query_used") or primary_query,
        "search_source": "pubmed",
        "fallback_used": False,
        "error": pubmed_error or "search_failed",
    }


def synthesize_literature(
    articles: List[Dict[str, Any]],
    compound_name: str,
    compound_smiles: str = "",
    language: str = "en",
) -> Dict[str, Any]:
    """Synthesize toxicity literature findings into a structured summary."""
    sanitized_articles = [article for article in articles if isinstance(article, dict)]
    papers_with_abstract = [article for article in sanitized_articles if _clean_text(article.get("abstract"))]
    prompt_articles = (papers_with_abstract or sanitized_articles)[:5]
    using_title_only = not papers_with_abstract and bool(prompt_articles)

    if not prompt_articles:
        return _deterministic_literature_synthesis(
            articles=[],
            language=language,
            error="no_articles_available",
        )

    source_coverage: Dict[str, int] = {}
    for article in papers_with_abstract or prompt_articles:
        source_key = _clean_text(article.get("abstract_source") or article.get("search_source") or "unknown")
        source_coverage[source_key] = source_coverage.get(source_key, 0) + 1

    default_confidence = _derive_confidence_level(
        papers_with_content=len(papers_with_abstract),
        fallback_count=sum(count for source, count in source_coverage.items() if source != "pubmed"),
        using_title_only=using_title_only,
    )

    if str(os.getenv("RESEARCH_ENABLE_LLM_SYNTHESIS", "1")).strip().lower() in {"0", "false", "no"}:
        return _deterministic_literature_synthesis(
            articles=prompt_articles,
            language=language,
            error="llm_disabled_by_env",
        )

    if genai is None:
        return _deterministic_literature_synthesis(
            articles=prompt_articles,
            language=language,
            error="genai_unavailable",
        )

    model_candidates = dedupe_strings(
        [
            os.getenv("LITERATURE_SYNTHESIS_MODEL"),
            os.getenv("AGENT_MODEL_FAST"),
            os.getenv("GEMINI_MODEL"),
            os.getenv("AGENT_MODEL_PRO"),
            "gemini-2.5-flash",
        ]
    )
    client_candidates = build_genai_client_candidates()
    if not client_candidates:
        return _deterministic_literature_synthesis(
            articles=prompt_articles,
            language=language,
            error="genai_client_unavailable",
        )

    prompt = _build_synthesis_prompt(
        articles=prompt_articles,
        compound_name=compound_name,
        compound_smiles=compound_smiles,
        language=language,
        using_title_only=using_title_only,
    )
    config = genai.types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema={
            "type": "object",
            "properties": {
                "consensus_mechanisms": {"type": "array", "items": {"type": "string"}},
                "key_targets": {"type": "array", "items": {"type": "string"}},
                "dose_response_signals": {"type": "array", "items": {"type": "string"}},
                "conflicting_findings": {"type": "array", "items": {"type": "string"}},
                "confidence_level": {"type": "string"},
                "synthesis_text": {"type": "string"},
                "pmids_used": {"type": "array", "items": {"type": "string"}},
                "evidence_basis": {"type": "string"},
            },
            "required": [
                "consensus_mechanisms",
                "key_targets",
                "dose_response_signals",
                "conflicting_findings",
                "confidence_level",
                "synthesis_text",
                "pmids_used",
                "evidence_basis",
            ],
        },
    )

    errors: List[str] = []
    try:
        for client, auth_mode in client_candidates:
            for model_name in model_candidates:
                try:
                    response = call_with_retry(
                        lambda client=client, model_name=model_name: client.models.generate_content(
                            model=model_name,
                            contents=prompt,
                            config=config,
                        )
                    )
                    result = json.loads(response.text)
                    result["papers_with_content"] = len(papers_with_abstract)
                    result["source_coverage"] = source_coverage
                    result["confidence_level"] = _clean_text(result.get("confidence_level") or default_confidence).lower() or default_confidence
                    result["evidence_basis"] = "title_only" if using_title_only else "abstract"
                    result["error"] = None
                    result["runtime_detail"] = f"{auth_mode}:{model_name}"
                    return result
                except Exception as exc:
                    errors.append(f"{auth_mode}:{model_name}:{type(exc).__name__}:{str(exc)[:180]}")
                    if is_resource_exhausted_error(exc) or is_model_unavailable_error(exc):
                        continue
                    raise
    except Exception:
        return _deterministic_literature_synthesis(
            articles=prompt_articles,
            language=language,
            error=errors[0] if errors else "genai_request_failed",
        )

    return _deterministic_literature_synthesis(
        articles=prompt_articles,
        language=language,
        error=errors[0] if errors else "genai_request_failed",
    )


def get_pubchem_bioassay_data(cid: int) -> Dict[str, Any]:
    """Fetch PubChem bioassay activity summary for a compound CID.

    Args:
        cid: PubChem compound id obtained from ``get_compound_info_pubchem``.

    Returns:
        Dict containing ``active_assays``, ``total_assays_tested``,
        ``tox21_active_count`` and ``error``.
    """
    if not cid:
        return {
            "cid": cid,
            "active_assays": [],
            "total_assays_tested": 0,
            "tox21_active_count": 0,
            "error": "cid_required",
        }

    try:
        resp = httpx.get(
            f"{PUBCHEM_BASE}/compound/cid/{cid}/assaysummary/JSON",
            timeout=15.0,
        )
        resp.raise_for_status()
        data = resp.json()
        summaries = (
            data.get("AssaySummaries", {}).get("AssaySummary", [])
            if isinstance(data, dict)
            else []
        )

        active_assays: List[Dict[str, Any]] = []
        tox21_active_count = 0

        for item in summaries:
            outcome = str(item.get("ActivityOutcome", "")).lower()
            if outcome != "active":
                continue
            assay_name = item.get("AssayName") or item.get("Name") or ""
            active_assays.append(
                {
                    "aid": item.get("AID"),
                    "assay_name": assay_name,
                    "activity_outcome": item.get("ActivityOutcome"),
                }
            )
            if "tox21" in assay_name.lower():
                tox21_active_count += 1

        return {
            "cid": cid,
            "active_assays": active_assays,
            "total_assays_tested": len(summaries),
            "tox21_active_count": tox21_active_count,
            "error": None,
        }
    except Exception as exc:
        return {
            "cid": cid,
            "active_assays": [],
            "total_assays_tested": 0,
            "tox21_active_count": 0,
            "error": str(exc),
        }
