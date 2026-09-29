# BATCH 01 / 06 (Cases 1 - 16 of 95)

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

[1] Case ID: herg_pos_01
Compound: Astemizole
SMILES: COc1ccc(CCN2CCC(Nc3nc4ccccc4n3Cc3ccc(F)cc3)CC2)cc1
Question/Vignette: A drug development researcher has submitted compound 'Astemizole' (SMILES: COc1ccc(CCN2CCC(Nc3nc4ccccc4n3Cc3ccc(F)cc3)CC2)cc1) for toxicity screening. Please analyze the hERG channel blocking risk and Tox21 profile. State the predicted probability, interpretation, and any limitations.

[2] Case ID: herg_pos_02
Compound: Terfenadine
SMILES: CC(C)(C)c1ccc(cc1)C(O)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1
Question/Vignette: A drug development researcher has submitted compound 'Terfenadine' (SMILES: CC(C)(C)c1ccc(cc1)C(O)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1) for toxicity screening. Please analyze the hERG channel blocking risk and Tox21 profile. State the predicted probability, interpretation, and any limitations.

[3] Case ID: herg_pos_03
Compound: Cisapride
SMILES: COc1cc(C(=O)NC2CCN(CCCOc3ccc(F)cc3)CC2OC)c(N)cc1Cl
Question/Vignette: A drug development researcher has submitted compound 'Cisapride' (SMILES: COc1cc(C(=O)NC2CCN(CCCOc3ccc(F)cc3)CC2OC)c(N)cc1Cl) for toxicity screening. Please analyze the hERG channel blocking risk and Tox21 profile. State the predicted probability, interpretation, and any limitations.

[4] Case ID: herg_pos_04
Compound: Dofetilide
SMILES: CN(CCOc1ccc(NS(C)(=O)=O)cc1)CCc1ccc(NS(C)(=O)=O)cc1
Question/Vignette: A drug development researcher has submitted compound 'Dofetilide' (SMILES: CN(CCOc1ccc(NS(C)(=O)=O)cc1)CCc1ccc(NS(C)(=O)=O)cc1) for toxicity screening. Please analyze the hERG channel blocking risk and Tox21 profile. State the predicted probability, interpretation, and any limitations.

[5] Case ID: herg_pos_05
Compound: Sotalol
SMILES: CC(C)NCC(O)c1ccc(NS(C)(=O)=O)cc1
Question/Vignette: A drug development researcher has submitted compound 'Sotalol' (SMILES: CC(C)NCC(O)c1ccc(NS(C)(=O)=O)cc1) for toxicity screening. Please analyze the hERG channel blocking risk and Tox21 profile. State the predicted probability, interpretation, and any limitations.

[6] Case ID: herg_pos_06
Compound: Haloperidol
SMILES: O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1
Question/Vignette: A drug development researcher has submitted compound 'Haloperidol' (SMILES: O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1) for toxicity screening. Please analyze the hERG channel blocking risk and Tox21 profile. State the predicted probability, interpretation, and any limitations.

[7] Case ID: herg_pos_07
Compound: Thioridazine
SMILES: CSc1ccc2Sc3ccccc3N(CCC3CCCCN3C)c2c1
Question/Vignette: A drug development researcher has submitted compound 'Thioridazine' (SMILES: CSc1ccc2Sc3ccccc3N(CCC3CCCCN3C)c2c1) for toxicity screening. Please analyze the hERG channel blocking risk and Tox21 profile. State the predicted probability, interpretation, and any limitations.

[8] Case ID: herg_pos_08
Compound: Verapamil
SMILES: COc1ccc(CCN(C)CCCC(C#N)(C(C)C)c2ccc(OC)c(OC)c2)cc1OC
Question/Vignette: A drug development researcher has submitted compound 'Verapamil' (SMILES: COc1ccc(CCN(C)CCCC(C#N)(C(C)C)c2ccc(OC)c(OC)c2)cc1OC) for toxicity screening. Please analyze the hERG channel blocking risk and Tox21 profile. State the predicted probability, interpretation, and any limitations.

[9] Case ID: herg_pos_09
Compound: Amiodarone
SMILES: CCCCc1oc2ccccc2c1C(=O)c1cc(I)c(OCCN(CC)CC)c(I)c1
Question/Vignette: A drug development researcher has submitted compound 'Amiodarone' (SMILES: CCCCc1oc2ccccc2c1C(=O)c1cc(I)c(OCCN(CC)CC)c(I)c1) for toxicity screening. Please analyze the hERG channel blocking risk and Tox21 profile. State the predicted probability, interpretation, and any limitations.

[10] Case ID: herg_pos_10
Compound: Pimozide
SMILES: O=C1Nc2ccccc2N1C1CCN(CCCC(c2ccc(F)cc2)c2ccc(F)cc2)CC1
Question/Vignette: A drug development researcher has submitted compound 'Pimozide' (SMILES: O=C1Nc2ccccc2N1C1CCN(CCCC(c2ccc(F)cc2)c2ccc(F)cc2)CC1) for toxicity screening. Please analyze the hERG channel blocking risk and Tox21 profile. State the predicted probability, interpretation, and any limitations.

[11] Case ID: herg_pos_11
Compound: Flunarizine-analog
SMILES: Fc1ccc(C(OCC2CC2)c2ccc(F)cc2)cc1
Question/Vignette: A drug development researcher has submitted compound 'Flunarizine-analog' (SMILES: Fc1ccc(C(OCC2CC2)c2ccc(F)cc2)cc1) for toxicity screening. Please analyze the hERG channel blocking risk and Tox21 profile. State the predicted probability, interpretation, and any limitations.

[12] Case ID: herg_pos_12
Compound: Quinazoline-piperidine
SMILES: c1ccc2c(c1)nc1ccc(OCCN3CCCC3)cc1n2
Question/Vignette: A drug development researcher has submitted compound 'Quinazoline-piperidine' (SMILES: c1ccc2c(c1)nc1ccc(OCCN3CCCC3)cc1n2) for toxicity screening. Please analyze the hERG channel blocking risk and Tox21 profile. State the predicted probability, interpretation, and any limitations.

[13] Case ID: herg_pos_13
Compound: Benzophenone-piperidine
SMILES: O=C(c1ccccc1)c1ccc(OCCN2CCCCC2)cc1
Question/Vignette: A drug development researcher has submitted compound 'Benzophenone-piperidine' (SMILES: O=C(c1ccccc1)c1ccc(OCCN2CCCCC2)cc1) for toxicity screening. Please analyze the hERG channel blocking risk and Tox21 profile. State the predicted probability, interpretation, and any limitations.

[14] Case ID: herg_pos_14
Compound: Arylpiperazine
SMILES: Clc1ccc(CN2CCN(c3ncccc3C(F)(F)F)CC2)cc1
Question/Vignette: A drug development researcher has submitted compound 'Arylpiperazine' (SMILES: Clc1ccc(CN2CCN(c3ncccc3C(F)(F)F)CC2)cc1) for toxicity screening. Please analyze the hERG channel blocking risk and Tox21 profile. State the predicted probability, interpretation, and any limitations.

[15] Case ID: herg_pos_15
Compound: Acetylpiperazine
SMILES: CC(=O)N1CCN(c2ccc(OCC3CO3)cc2)CC1
Question/Vignette: A drug development researcher has submitted compound 'Acetylpiperazine' (SMILES: CC(=O)N1CCN(c2ccc(OCC3CO3)cc2)CC1) for toxicity screening. Please analyze the hERG channel blocking risk and Tox21 profile. State the predicted probability, interpretation, and any limitations.

[16] Case ID: herg_pos_16
Compound: Dimethoxyphenylpiperidine
SMILES: COc1ccc(C2CCN(Cc3ccccc3)CC2)cc1OC
Question/Vignette: A drug development researcher has submitted compound 'Dimethoxyphenylpiperidine' (SMILES: COc1ccc(C2CCN(Cc3ccccc3)CC2)cc1OC) for toxicity screening. Please analyze the hERG channel blocking risk and Tox21 profile. State the predicted probability, interpretation, and any limitations.

```