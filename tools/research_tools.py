from __future__ import annotations

import os
import urllib.parse
from typing import Any, Dict, List

import httpx

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY", "").strip()


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
        cid_resp = httpx.get(
            f"{PUBCHEM_BASE}/compound/smiles/{encoded}/cids/JSON",
            timeout=10.0,
        )
        cid_resp.raise_for_status()
        cid_list = cid_resp.json().get("IdentifierList", {}).get("CID", [])
        if not cid_list:
            return {
                "cid": None,
                "iupac_name": None,
                "common_name": None,
                "molecular_formula": None,
                "molecular_weight": None,
                "synonyms": [],
                "pubchem_url": None,
                "error": "cid_not_found",
            }
        cid = cid_list[0]

        props_resp = httpx.get(
            f"{PUBCHEM_BASE}/compound/cid/{cid}/property/"
            "IUPACName,MolecularFormula,MolecularWeight/JSON",
            timeout=10.0,
        )
        props_resp.raise_for_status()
        props = props_resp.json().get("PropertyTable", {}).get("Properties", [{}])[0]

        syn_resp = httpx.get(
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
        }
    except Exception as exc:
        return {
            "cid": None,
            "iupac_name": None,
            "common_name": None,
            "molecular_formula": None,
            "molecular_weight": None,
            "synonyms": [],
            "pubchem_url": None,
            "error": str(exc),
        }


def search_toxicity_literature(compound_name: str, max_results: int = 5) -> Dict[str, Any]:
    """Search PubMed for toxicity/mechanism literature by compound name.

    Args:
        compound_name: Name to query (prefer common name from PubChem).
        max_results: Maximum number of returned article summaries (capped at 10).

    Returns:
        Dict containing ``articles`` (list), ``total_found``, ``query_used`` and
        ``error``. Each article has PMID, title, authors, year, journal and URL.
    """
    max_results = min(max_results, 10)
    query = f"{compound_name} toxicity mechanism"
    encoded_query = urllib.parse.quote(query)
    api_key_param = f"&api_key={PUBMED_API_KEY}" if PUBMED_API_KEY else ""

    try:
        search_resp = httpx.get(
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
                "error": None,
            }

        ids_str = ",".join(pmids)
        summary_resp = httpx.get(
            f"{PUBMED_BASE}/esummary.fcgi?db=pubmed&id={ids_str}"
            f"&retmode=json{api_key_param}",
            timeout=15.0,
        )
        summary_resp.raise_for_status()
        summary_data = summary_resp.json().get("result", {})

        articles = []
        for pmid in pmids:
            art = summary_data.get(pmid, {})
            authors = [a.get("name", "") for a in art.get("authors", [])[:3]]
            title = art.get("title", "N/A")
            articles.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "authors": authors,
                    "year": str(art.get("pubdate", ""))[:4],
                    "journal": art.get("source", "N/A"),
                    "abstract_snippet": title[:150],
                    "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                }
            )

        return {
            "articles": articles,
            "total_found": total_found,
            "query_used": query,
            "error": None,
        }
    except Exception as exc:
        return {
            "articles": [],
            "total_found": 0,
            "query_used": query,
            "error": str(exc),
        }


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
