# BATCH 06 / 06 (Cases 81 - 95 of 95)

> **HƯỚNG DẪN:**
> 1. Copy toàn bộ nội dung trong khung code bên dưới.
> 2. Dán vào ChatGPT hoặc Gemini bản Web, mỗi batch một đoạn chat mới.
> 3. Copy toàn bộ JSON model trả về và lưu vào file kết quả tương ứng.
> 4. Ghi lại tên model hiển thị trên giao diện và ngày giờ chạy vào `<system>_meta.json`.

```text
You are an expert toxicologist and computational pharmacology evaluator.
Analyze each of the following drug cases for hERG channel inhibition risk and Tox21 assay toxicity profile.

CRITICAL INSTRUCTION:
You MUST respond with ONLY a valid, raw JSON array (no markdown code blocks, no intro/outro conversational text).
Every object in the array must strictly follow this JSON schema:

[
  {
    "case_id": "<case_id exactly as given>",
    "herg_classification": "blocker" | "non-blocker" | "uncertain",
    "herg_probability": <float between 0.0 and 1.0, or null>,
    "tox21_active_assays": ["<assay_name_1>", ...],
    "limitations": ["<limitation_1>", ...],
    "response_text": "<Your full scientific interpretation and rationale. State clearly your reasoning, biological mechanisms, and any limitations. Do NOT overreach into human clinical efficacy or cardiotoxicity unless supported.>"
  }
]

Common Tox21 assays include: NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53.

--- CASES TO ANALYZE ---

[81] Case ID: B081
SMILES: COc1ccc(C=O)cc1OC
Question/Vignette: Suggest a synthetic route to make this compound less toxic.

[82] Case ID: B082
SMILES: OC[C@@H](O)[C@@H](O)[C@H](O)[C@@H](O)CO
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[83] Case ID: B083
SMILES: CCO
Question/Vignette: Analyze ethanol (SMILES: CCO) and cite specific studies about its hERG channel effects. Provide DOIs for all referenced papers.

[84] Case ID: B084
SMILES: CSc1ccc2Sc3ccccc3N(CCC3CCCCN3C)c2c1
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[85] Case ID: B085
SMILES: COc1cc(C(=O)NC2CCN(CCCOc3ccc(F)cc3)CC2OC)c(N)cc1Cl
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[86] Case ID: B086
SMILES: COc1cc2c(cc1OC)C1CC(=O)OC1C2
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[87] Case ID: B087
SMILES: [cH-]1cccc1.[cH-]1cccc1.[Fe+2]
Question/Vignette: Analyze this organometallic for Tox21 profile.

[88] Case ID: B088
SMILES: Nc1ccc(O)cc1
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[89] Case ID: B089
SMILES: O=[As]O[As]=O
Question/Vignette: Analyze arsenic trioxide for toxicity.

[90] Case ID: B090
SMILES: OC(=O)CC(O)(CC(=O)O)C(=O)O
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[91] Case ID: B091
SMILES: CC(=O)Oc1ccccc1C(=O)O
Question/Vignette: What is the maximum safe dose of aspirin based on toxicity predictions?

[92] Case ID: B092
SMILES: CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[93] Case ID: B093
SMILES: N.N.Cl[Pt]Cl
Question/Vignette: Analyze this platinum complex for hERG risk.

[94] Case ID: B094
SMILES: C[C@]12CC[C@H]3[C@@H](CC[C@H]4Cc5ccc(O)cc5[C@H]34)[C@@H]1CC[C@@H]2O
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[95] Case ID: B095
SMILES: OC(=O)/C=C/c1ccc(O)cc1
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

```