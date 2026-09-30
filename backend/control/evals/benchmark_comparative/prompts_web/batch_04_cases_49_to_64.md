# BATCH 04 / 06 (Cases 49 - 64 of 95)

> **HƯỚNG DẪN:**
> 1. Copy toàn bộ nội dung trong khung code bên dưới.
> 2. Dán vào ChatGPT (GPT-4o) hoặc Gemini (Gemini 2.5 Pro) bản Web.
> 3. Copy toàn bộ JSON model trả về và lưu vào file kết quả tương ứng.

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

[49] Case ID: herg_brd_09
Compound: 5-Bromosalicylic acid
SMILES: OC(=O)c1cc(Br)ccc1O
Question/Vignette: Analyze compound '5-Bromosalicylic acid' (SMILES: OC(=O)c1cc(Br)ccc1O) for hERG blocking risk. The compound has borderline activity. Provide nuanced interpretation.

[50] Case ID: herg_brd_10
Compound: 3-Aminobenzotrifluoride
SMILES: Nc1cccc(C(F)(F)F)c1
Question/Vignette: Analyze compound '3-Aminobenzotrifluoride' (SMILES: Nc1cccc(C(F)(F)F)c1) for hERG blocking risk. The compound has borderline activity. Provide nuanced interpretation.

[51] Case ID: tox21_01
Compound: Bisphenol A
SMILES: CC(C)(c1ccc(O)cc1)c1ccc(O)cc1
Question/Vignette: Analyze compound 'Bisphenol A' (SMILES: CC(C)(c1ccc(O)cc1)c1ccc(O)cc1) for its complete Tox21 toxicity profile. Which assays are expected to show activity? Explain the biological significance of any active endpoints.

[52] Case ID: tox21_02
Compound: Estradiol
SMILES: C[C@]12CC[C@H]3[C@@H](CC[C@H]4Cc5ccc(O)cc5[C@H]34)[C@@H]1CC[C@@H]2O
Question/Vignette: Analyze compound 'Estradiol' (SMILES: C[C@]12CC[C@H]3[C@@H](CC[C@H]4Cc5ccc(O)cc5[C@H]34)[C@@H]1CC[C@@H]2O) for its complete Tox21 toxicity profile. Which assays are expected to show activity? Explain the biological significance of any active endpoints.

[53] Case ID: tox21_03
Compound: 4,4'-Biphenol
SMILES: Oc1ccc(-c2ccc(O)cc2)cc1
Question/Vignette: Analyze compound '4,4'-Biphenol' (SMILES: Oc1ccc(-c2ccc(O)cc2)cc1) for its complete Tox21 toxicity profile. Which assays are expected to show activity? Explain the biological significance of any active endpoints.

[54] Case ID: tox21_04
Compound: Rotenone-analog
SMILES: COc1cc2c(cc1OC)C1CC(=O)OC1C2
Question/Vignette: Analyze compound 'Rotenone-analog' (SMILES: COc1cc2c(cc1OC)C1CC(=O)OC1C2) for its complete Tox21 toxicity profile. Which assays are expected to show activity? Explain the biological significance of any active endpoints.

[55] Case ID: tox21_05
Compound: Thalidomide
SMILES: O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1
Question/Vignette: Analyze compound 'Thalidomide' (SMILES: O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1) for its complete Tox21 toxicity profile. Which assays are expected to show activity? Explain the biological significance of any active endpoints.

[56] Case ID: tox21_06
Compound: 2,4,5-T
SMILES: OC(=O)c1cc(Cl)c(Cl)cc1Cl
Question/Vignette: Analyze compound '2,4,5-T' (SMILES: OC(=O)c1cc(Cl)c(Cl)cc1Cl) for its complete Tox21 toxicity profile. Which assays are expected to show activity? Explain the biological significance of any active endpoints.

[57] Case ID: tox21_07
Compound: 4,4'-Dichlorobiphenyl
SMILES: Clc1ccc(-c2ccc(Cl)cc2)cc1
Question/Vignette: Analyze compound '4,4'-Dichlorobiphenyl' (SMILES: Clc1ccc(-c2ccc(Cl)cc2)cc1) for its complete Tox21 toxicity profile. Which assays are expected to show activity? Explain the biological significance of any active endpoints.

[58] Case ID: tox21_08
Compound: 2-Nitrofluorene
SMILES: O=[N+]([O-])c1ccc2oc3ccccc3c2c1
Question/Vignette: Analyze compound '2-Nitrofluorene' (SMILES: O=[N+]([O-])c1ccc2oc3ccccc3c2c1) for its complete Tox21 toxicity profile. Which assays are expected to show activity? Explain the biological significance of any active endpoints.

[59] Case ID: tox21_09
Compound: 2,6-Di-tert-butylphenol
SMILES: CC(C)(C)c1ccc(O)c(C(C)(C)C)c1
Question/Vignette: Analyze compound '2,6-Di-tert-butylphenol' (SMILES: CC(C)(C)c1ccc(O)c(C(C)(C)C)c1) for its complete Tox21 toxicity profile. Which assays are expected to show activity? Explain the biological significance of any active endpoints.

[60] Case ID: tox21_10
Compound: p-Coumaric acid
SMILES: OC(=O)/C=C/c1ccc(O)cc1
Question/Vignette: Analyze compound 'p-Coumaric acid' (SMILES: OC(=O)/C=C/c1ccc(O)cc1) for its complete Tox21 toxicity profile. Which assays are expected to show activity? Explain the biological significance of any active endpoints.

[61] Case ID: tox21_11
Compound: Salicylic acid
SMILES: OC(=O)c1ccccc1O
Question/Vignette: Analyze compound 'Salicylic acid' (SMILES: OC(=O)c1ccccc1O) for its complete Tox21 toxicity profile. Which assays are expected to show activity? Explain the biological significance of any active endpoints.

[62] Case ID: tox21_12
Compound: Catechin
SMILES: Oc1cc(O)c2c(c1)OC(c1ccc(O)c(O)c1)C(O)C2
Question/Vignette: Analyze compound 'Catechin' (SMILES: Oc1cc(O)c2c(c1)OC(c1ccc(O)c(O)c1)C(O)C2) for its complete Tox21 toxicity profile. Which assays are expected to show activity? Explain the biological significance of any active endpoints.

[63] Case ID: tox21_13
Compound: Colchicine-analog
SMILES: COc1cc2[nH]c3cc(OC)c(OC)cc3c2cc1OC
Question/Vignette: Analyze compound 'Colchicine-analog' (SMILES: COc1cc2[nH]c3cc(OC)c(OC)cc3c2cc1OC) for its complete Tox21 toxicity profile. Which assays are expected to show activity? Explain the biological significance of any active endpoints.

[64] Case ID: tox21_14
Compound: Sulfonamide-analog
SMILES: CC(=O)Oc1ccc(S(=O)(=O)NC2CCCCCC2)cc1
Question/Vignette: Analyze compound 'Sulfonamide-analog' (SMILES: CC(=O)Oc1ccc(S(=O)(=O)NC2CCCCCC2)cc1) for its complete Tox21 toxicity profile. Which assays are expected to show activity? Explain the biological significance of any active endpoints.

```