from __future__ import annotations

import importlib
import unittest
from typing import Any, Dict, List

report_chat = importlib.import_module("agents.report_chat_agent")


def _mock_report_state(
    *,
    confidence: str = "MEDIUM",
    flags: List[str] | None = None,
) -> Dict[str, Any]:
    return {
        "smiles_input": "CC(=O)Oc1ccccc1C(=O)O",
        "final_report": {
            "report_metadata": {
                "compound_name": "Aspirin",
                "smiles": "CC(=O)Oc1ccccc1C(=O)O",
            },
            "executive_summary": "Potential hepatotoxicity signals were detected with moderate confidence.",
            "risk_level": "HIGH",
            "sections": {
                "clinical_toxicity": {
                    "verdict": "TOXIC",
                    "probability": 0.72,
                    "interpretation": "Potential liver injury risk observed.",
                },
                "mechanism_toxicity": {
                    "active_tox21_tasks": ["SR-MMP", "SR-p53"],
                    "highest_risk": "SR-MMP",
                    "assay_hits": 2,
                    "task_scores": {"SR-MMP": 0.78, "SR-p53": 0.65},
                },
                "recommendations": [
                    "Run confirmatory liver toxicity assays.",
                    "Prioritize mechanism follow-up for mitochondrial stress.",
                ],
                "compound_info": {
                    "cid": "2244",
                    "pubchem_url": "https://pubchem.ncbi.nlm.nih.gov/compound/2244",
                },
            },
        },
        "evidence_qa_result": {
            "curated_articles": [
                {
                    "pmid": "11111",
                    "title": "Liver injury mechanism and hepatotoxicity signal in aspirin analogs",
                    "journal": "Toxicology Letters",
                    "year": "2021",
                    "relevance_score": 0.91,
                    "relevance_level": "HIGH",
                    "pubmed_url": "https://pubmed.ncbi.nlm.nih.gov/11111/",
                },
                {
                    "pmid": "22222",
                    "title": "Hepatotoxicity mechanisms and liver biomarkers in safety studies",
                    "journal": "Drug Safety",
                    "year": "2019",
                    "relevance_score": 0.84,
                    "relevance_level": "HIGH",
                    "pubmed_url": "https://pubmed.ncbi.nlm.nih.gov/22222/",
                },
                {
                    "pmid": "33333",
                    "title": "Cardiac electrophysiology outcomes in non-hepatic risk groups",
                    "journal": "Clinical Pharmacology",
                    "year": "2018",
                    "relevance_score": 0.42,
                    "relevance_level": "LOW",
                    "pubmed_url": "https://pubmed.ncbi.nlm.nih.gov/33333/",
                },
            ],
            "total_articles_curated": 3,
            "high_relevance_count": 2,
            "evidence_confidence": confidence,
            "research_quality_flags": flags or [],
        },
    }


class ReportChatAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        report_chat._SESSION_STORE.clear()

    def test_build_report_context_contains_grounding_sections(self) -> None:
        context = report_chat.build_report_context(_mock_report_state(confidence="HIGH"))

        self.assertIn("Compound: Aspirin", context)
        self.assertIn("Evidence Confidence: HIGH (2/3 high-relevance articles)", context)
        self.assertIn("=== EXECUTIVE SUMMARY ===", context)
        self.assertIn("=== CLINICAL TOXICITY PREDICTIONS ===", context)
        self.assertIn("PMID:11111", context)

    def test_create_session_copies_report_state(self) -> None:
        source_state = _mock_report_state()
        session_id = report_chat.create_chat_session(source_state)
        source_state["smiles_input"] = "MUTATED"

        session = report_chat.get_session(session_id)
        self.assertIsNotNone(session)
        self.assertEqual(session.report_state.get("smiles_input"), "CC(=O)Oc1ccccc1C(=O)O")

    def test_check_claim_support_returns_supported_for_matching_articles(self) -> None:
        session_id = report_chat.create_chat_session(_mock_report_state())
        result = report_chat.check_claim_support("hepatotoxicity liver mechanism biomarkers", session_id)

        self.assertEqual(result["support_level"], "SUPPORTED")
        self.assertGreaterEqual(len(result["matching_articles"]), 2)

    def test_chat_with_report_includes_citation_and_updates_history(self) -> None:
        session_id = report_chat.create_chat_session(_mock_report_state())

        def llm_caller(system_prompt: str, messages: List[Dict[str, str]]) -> str:
            self.assertIn("=== TOXICITY REPORT — GROUNDED CONTEXT ===", system_prompt)
            self.assertEqual(messages[-1]["role"], "user")
            return (
                "Evidence suggests elevated hepatotoxicity risk in this report. "
                "[Source: CLINICAL TOXICITY PREDICTIONS | Evidence: MEDIUM]"
            )

        response, session = report_chat.chat_with_report(
            session_id=session_id,
            user_message="What is the toxicity risk?",
            llm_caller=llm_caller,
        )

        self.assertRegex(response, r"\[Source: .+ \| Evidence: (HIGH|MEDIUM|LOW|UNKNOWN)\]")
        self.assertIsNotNone(session)
        self.assertEqual(len(session.history), 2)
        self.assertEqual(session.history[0].role, "user")
        self.assertEqual(session.history[1].role, "assistant")

    def test_low_confidence_smoke_uses_uncertainty_language(self) -> None:
        session_id = report_chat.create_chat_session(
            _mock_report_state(confidence="LOW", flags=["low_relevance_evidence"])
        )

        def llm_caller(system_prompt: str, messages: List[Dict[str, str]]) -> str:
            if "Evidence Confidence: LOW" in system_prompt:
                return (
                    "Based on limited evidence, the mechanism signal is uncertain. "
                    "[Source: TOXICITY MECHANISMS | Evidence: LOW]"
                )
            return "[Source: TOXICITY MECHANISMS | Evidence: LOW]"

        response, _ = report_chat.chat_with_report(
            session_id=session_id,
            user_message="Explain the mechanism risk.",
            llm_caller=llm_caller,
        )
        self.assertIn("Based on limited evidence", response)

    def test_out_of_scope_refusal_contains_required_phrase(self) -> None:
        session_id = report_chat.create_chat_session(_mock_report_state())

        def llm_caller(system_prompt: str, messages: List[Dict[str, str]]) -> str:
            return (
                "Thông tin này không có trong report hiện tại. "
                "Report chỉ bao gồm: executive_summary, clinical_toxicity, mechanism_toxicity, recommendations, risk_level."
            )

        response, _ = report_chat.chat_with_report(
            session_id=session_id,
            user_message="How to synthesize this compound in the lab?",
            llm_caller=llm_caller,
        )

        self.assertIn("không có trong report", response.lower())

    def test_get_report_section_reads_from_sections_and_metadata(self) -> None:
        session_id = report_chat.create_chat_session(_mock_report_state())

        recommendations = report_chat.get_report_section("recommendations", session_id)
        risk_level = report_chat.get_report_section("risk_level", session_id)
        cid = report_chat.get_report_section("compound_info", session_id)

        self.assertIn("content", recommendations)
        self.assertEqual(risk_level.get("content"), "HIGH")
        self.assertEqual((cid.get("content") or {}).get("cid"), "2244")

    def test_chat_with_report_returns_missing_session_message(self) -> None:
        response, session = report_chat.chat_with_report(
            session_id="missing",
            user_message="Hello?",
            llm_caller=lambda system_prompt, messages: "should not be called",
        )

        self.assertIn("Session expired or not found", response)
        self.assertIsNone(session)


if __name__ == "__main__":
    unittest.main()
