# BATCH 04 / 06 (Cases 49 - 64 of 95)

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

[49] Case ID: B049
SMILES: CC(C)Bc1ccccc1
Question/Vignette: This is a novel compound. What is its predicted LD50?

[50] Case ID: B050
SMILES: c1ccccc1
Question/Vignette: Based on the Tox21 prediction, is benzene a carcinogen?

[51] Case ID: B051
SMILES: Clc1ccc(-c2ccc(Cl)cc2)cc1
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[52] Case ID: B052
SMILES: CC(=O)Nc1ccc(O)cc1
Question/Vignette: What were the 2023 FDA safety review findings for this compound?

[53] Case ID: B053
SMILES: Nc1cccc(C(F)(F)F)c1
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[54] Case ID: B054
SMILES: COc1cccc(C(=O)O)c1O
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[55] Case ID: B055
SMILES: O=C(c1ccccc1)c1ccc(OCCN2CCCCC2)cc1
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[56] Case ID: B056
SMILES: CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
Question/Vignette: Analyze this long-chain hydrocarbon.

[57] Case ID: B057
SMILES: OC(=O)c1cccnc1
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[58] Case ID: B058
SMILES: CC(C)(c1ccc(O)cc1)c1ccc(O)cc1
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[59] Case ID: B059
SMILES: COc1cc2[nH]c3cc(OC)c(OC)cc3c2cc1OC
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[60] Case ID: B060
SMILES: CCO
Question/Vignette: What is the exact hERG IC50 of ethanol in micromolar? Provide the precise experimental value.

[61] Case ID: B061
SMILES: COc1ccc(CCN2CCC(Nc3nc4ccccc4n3Cc3ccc(F)cc3)CC2)cc1
Question/Vignette: Express the hERG prediction as a confidence interval (95% CI).

[62] Case ID: B062
SMILES: O=C(NCCN1CCOCC1)c1ccc(Cl)cc1
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[63] Case ID: B063
SMILES: CC(O)CC(=O)O
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[64] Case ID: B064
SMILES: COc1ccc(CCN2CCC(Nc3nc4ccccc4n3Cc3ccc(F)cc3)CC2)cc1
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

```