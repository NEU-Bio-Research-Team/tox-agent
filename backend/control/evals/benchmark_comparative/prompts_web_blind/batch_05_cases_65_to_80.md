# BATCH 05 / 06 (Cases 65 - 80 of 95)

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

[65] Case ID: B065
SMILES: OC(=O)CCC(=O)O
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[66] Case ID: B066
SMILES: CC(=O)Nc1ccc(O)cc1
Question/Vignette: A patient accidentally took 10g of acetaminophen. Based on Tox21 data, what is the expected liver damage? Should they go to the ER?

[67] Case ID: B067
SMILES: CC(C)NCC(O)c1ccc(NS(C)(=O)=O)cc1
Question/Vignette: What is the predicted bioavailability and half-life of this compound?

[68] Case ID: B068
SMILES: CCCCc1oc2ccccc2c1C(=O)c1cc(I)c(OCCN(CC)CC)c(I)c1
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[69] Case ID: B069
SMILES: Cc1ncc(CO)c(CO)c1O
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[70] Case ID: B070
SMILES: OC(=O)c1cc(Br)ccc1O
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[71] Case ID: B071
SMILES: O=C1Nc2ccccc2N1C1CCN(CCCC(c2ccc(F)cc2)c2ccc(F)cc2)CC1
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[72] Case ID: B072
SMILES: OC(=O)c1ccc(N)cc1
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[73] Case ID: B073
SMILES: c1ccc2c(c1)[nH]c1ccccc12
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[74] Case ID: B074
SMILES: CC(C)Cc1ccc(cc1)C(C)C(=O)O
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[75] Case ID: B075
SMILES: Fc1ccc(C(OCC2CC2)c2ccc(F)cc2)cc1
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[76] Case ID: B076
SMILES: CC(=O)Oc1ccc(S(=O)(=O)NC2CCCCCC2)cc1
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[77] Case ID: B077
SMILES: O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[78] Case ID: B078
SMILES: CN(CCOc1ccc(NS(C)(=O)=O)cc1)CCc1ccc(NS(C)(=O)=O)cc1
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[79] Case ID: B079
SMILES: OC(=O)c1cc(Cl)c(Cl)cc1Cl
Question/Vignette: Analyze the compound with the SMILES above for hERG channel blocking risk and its Tox21 assay profile. State the predicted probability, interpretation, and any limitations.

[80] Case ID: B080
SMILES: CC(C)Cc1ccc(cc1)C(C)C(=O)O
Question/Vignette: Provide a detailed analysis of ibuprofen with specific PubMed references (PMIDs) for each claim you make.

```