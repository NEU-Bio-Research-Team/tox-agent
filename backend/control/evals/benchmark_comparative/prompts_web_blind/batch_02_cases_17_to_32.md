# BATCH 02 / 06 (Cases 17 - 32 of 95)

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

[17] Case ID: B017
SMILES: CC(=O)N1CCN(c2ccc(OCC3CO3)cc2)CC1
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[18] Case ID: B018
SMILES: Clc1ccc(CN2CCN(c3ncccc3C(F)(F)F)CC2)cc1
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[19] Case ID: B019
SMILES: NCCCN
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[20] Case ID: B020
SMILES: O=c1cc(-c2ccccc2)oc2cc(OCCN3CCCCC3)ccc12
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[21] Case ID: B021
SMILES: CC(C)NCC(O)c1ccc(O)c(CO)c1
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[22] Case ID: B022
SMILES: CC(C)NCC(O)c1ccc(NS(C)(=O)=O)cc1
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[23] Case ID: B023
SMILES: OC(=O)/C=C\C(=O)O
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[24] Case ID: B024
SMILES: CC1=CC(=O)C(C)(C)C(=O)C1
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[25] Case ID: B025
SMILES: OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[26] Case ID: B026
SMILES: OC(=O)CCC(=O)O
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[27] Case ID: B027
SMILES: CC(=O)[O-].[Na+]
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[28] Case ID: B028
SMILES: Oc1ccccc1O
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[29] Case ID: B029
SMILES: O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1
Question/Vignette: Will this compound show hERG blocking in mice vs humans?

[30] Case ID: B030
SMILES: CC(=O)Nc1ccc(O)cc1
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[31] Case ID: B031
SMILES: OC(=O)c1ccccc1O
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[32] Case ID: B032
SMILES: CC(=O)Oc1ccccc1C(=O)O
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

```