"""Comparative Agent Benchmark: ToxAgent vs GPT vs Gemini.

Evaluates hallucination and predictive accuracy across three systems
using a shared dataset of 100 toxicology compounds.

Literature grounding:
  - MedHallu (Pandit et al., ACL 2025) — hallucination taxonomy
  - RAGTruth (Niu et al., ACL 2024) — span-level hallucination detection
  - AgentHallu (2026) — agentic workflow hallucination
  - FActScore (Min et al., EMNLP 2023) — claim-level factual precision
  - SAFE (Google, ICML 2024) — search-augmented fact verification
"""
