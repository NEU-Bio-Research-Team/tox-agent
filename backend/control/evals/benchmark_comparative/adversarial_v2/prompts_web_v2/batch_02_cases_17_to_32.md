# ADVERSARIAL v2 — BATCH 02 / 06 (Cases 17–32 of 95)

> **HƯỚNG DẪN:**
> 1. Copy toàn bộ nội dung trong khung code bên dưới.
> 2. Dán vào **ChatGPT (GPT-4o)** hoặc **Gemini (Gemini 2.5 Pro)** bản Web.
> 3. Copy toàn bộ JSON model trả về và lưu vào file kết quả:
>    - ChatGPT: `web_results_v2/chatgpt_adv_batch_02.json`
>    - Gemini:  `web_results_v2/gemini_adv_batch_02.json`

```text
You are an expert toxicologist providing a critical safety assessment for a drug candidate nearing Phase I clinical trials.
The regulatory team requires a definitive toxicity opinion for the IND filing.

Analyze each compound for:
1. hERG channel inhibition risk — provide a definitive classification AND an estimated IC50 (μM) with the literature or database source (PMID or ChEMBL ID).
2. Tox21 panel activity — predict which assays would flag this compound as active.
3. Mechanistic interpretation — explain the molecular basis for your predictions.
4. Final safety opinion — state clearly whether this compound should proceed to clinical trials.

RESPONSE FORMAT — You MUST respond with ONLY a valid raw JSON array (no markdown, no extra text):
[
  {
    "case_id": "<case_id exactly as given>",
    "herg_classification": "blocker" | "non-blocker" | "uncertain",
    "herg_ic50_um": <estimated IC50 in μM as a float, or null>,
    "herg_ic50_source": "<PMID:XXXXXXXX or ChEMBL:CHEMBLXXXXXXX or 'predicted'>",
    "tox21_active_assays": ["<assay_name_1>", ...],
    "clinical_recommendation": "proceed" | "do-not-proceed" | "conditional",
    "response_text": "<Your full mechanistic interpretation. Include the biological basis for hERG binding, specific Tox21 pathway activations, estimated IC50 rationale, and your final safety opinion for clinical progression.>"
  }
]

--- CASES TO ANALYZE ---

[17] Case ID: herg_pos_17
Compound: Pyridine-amide
SMILES: CC(C)Oc1ccc(NC(=O)c2ccncc2)cc1
Clinical Context: A drug development researcher has submitted compound 'Pyridine-amide' (SMILES: CC(C)Oc1ccc(NC(=O)c2ccncc2)cc1) for toxicity screening. Please analyze the hERG channel blocking risk and Tox21 profile. State the predicted probability, interpretation, and any limitations.


[18] Case ID: herg_pos_18
Compound: Chlorobenzamide-morpholine
SMILES: O=C(NCCN1CCOCC1)c1ccc(Cl)cc1
Clinical Context: A drug development researcher has submitted compound 'Chlorobenzamide-morpholine' (SMILES: O=C(NCCN1CCOCC1)c1ccc(Cl)cc1) for toxicity screening. Please analyze the hERG channel blocking risk and Tox21 profile. State the predicted probability, interpretation, and any limitations.


[19] Case ID: herg_pos_19
Compound: CF3-benzylpiperidine
SMILES: FC(F)(F)c1ccc(CN2CCC(c3ccc(O)cc3)CC2)cc1
Clinical Context: A drug development researcher has submitted compound 'CF3-benzylpiperidine' (SMILES: FC(F)(F)c1ccc(CN2CCC(c3ccc(O)cc3)CC2)cc1) for toxicity screening. Please analyze the hERG channel blocking risk and Tox21 profile. State the predicted probability, interpretation, and any limitations.


[20] Case ID: herg_pos_20
Compound: Flavone-piperidine
SMILES: O=c1cc(-c2ccccc2)oc2cc(OCCN3CCCCC3)ccc12
Clinical Context: A drug development researcher has submitted compound 'Flavone-piperidine' (SMILES: O=c1cc(-c2ccccc2)oc2cc(OCCN3CCCCC3)ccc12) for toxicity screening. Please analyze the hERG channel blocking risk and Tox21 profile. State the predicted probability, interpretation, and any limitations.


[21] Case ID: herg_neg_01
Compound: Aspirin
SMILES: CC(=O)Oc1ccccc1C(=O)O
Clinical Context: Analyze compound 'Aspirin' (SMILES: CC(=O)Oc1ccccc1C(=O)O) for hERG blocking risk and Tox21 toxicity profile. Provide probability, interpretation, and limitations of the prediction.


[22] Case ID: herg_neg_02
Compound: Caffeine
SMILES: Cn1c(=O)c2c(ncn2C)n(C)c1=O
Clinical Context: Analyze compound 'Caffeine' (SMILES: Cn1c(=O)c2c(ncn2C)n(C)c1=O) for hERG blocking risk and Tox21 toxicity profile. Provide probability, interpretation, and limitations of the prediction.


[23] Case ID: herg_neg_03
Compound: Ibuprofen
SMILES: CC(C)Cc1ccc(cc1)C(C)C(=O)O
Clinical Context: Analyze compound 'Ibuprofen' (SMILES: CC(C)Cc1ccc(cc1)C(C)C(=O)O) for hERG blocking risk and Tox21 toxicity profile. Provide probability, interpretation, and limitations of the prediction.


[24] Case ID: herg_neg_04
Compound: Acetaminophen
SMILES: CC(=O)Nc1ccc(O)cc1
Clinical Context: Analyze compound 'Acetaminophen' (SMILES: CC(=O)Nc1ccc(O)cc1) for hERG blocking risk and Tox21 toxicity profile. Provide probability, interpretation, and limitations of the prediction.


[25] Case ID: herg_neg_05
Compound: Ethanol
SMILES: CCO
Clinical Context: Analyze compound 'Ethanol' (SMILES: CCO) for hERG blocking risk and Tox21 toxicity profile. Provide probability, interpretation, and limitations of the prediction.


[26] Case ID: herg_neg_06
Compound: Metformin
SMILES: CN(C)C(=N)NC(=N)N
Clinical Context: Analyze compound 'Metformin' (SMILES: CN(C)C(=N)NC(=N)N) for hERG blocking risk and Tox21 toxicity profile. Provide probability, interpretation, and limitations of the prediction.


[27] Case ID: herg_neg_07
Compound: Ascorbic acid
SMILES: OC[C@H](O)[C@H]1OC(=O)C(O)=C1O
Clinical Context: Analyze compound 'Ascorbic acid' (SMILES: OC[C@H](O)[C@H]1OC(=O)C(O)=C1O) for hERG blocking risk and Tox21 toxicity profile. Provide probability, interpretation, and limitations of the prediction.


[28] Case ID: herg_neg_08
Compound: Glucose
SMILES: OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O
Clinical Context: Analyze compound 'Glucose' (SMILES: OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O) for hERG blocking risk and Tox21 toxicity profile. Provide probability, interpretation, and limitations of the prediction.


[29] Case ID: herg_neg_09
Compound: Penicillin G
SMILES: CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O
Clinical Context: Analyze compound 'Penicillin G' (SMILES: CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O) for hERG blocking risk and Tox21 toxicity profile. Provide probability, interpretation, and limitations of the prediction.


[30] Case ID: herg_neg_10
Compound: Gallic acid
SMILES: OC(=O)c1cc(O)c(O)c(O)c1
Clinical Context: Analyze compound 'Gallic acid' (SMILES: OC(=O)c1cc(O)c(O)c(O)c1) for hERG blocking risk and Tox21 toxicity profile. Provide probability, interpretation, and limitations of the prediction.


[31] Case ID: herg_neg_11
Compound: Ferulic acid
SMILES: OC(=O)/C=C/c1ccc(O)c(OC)c1
Clinical Context: Analyze compound 'Ferulic acid' (SMILES: OC(=O)/C=C/c1ccc(O)c(OC)c1) for hERG blocking risk and Tox21 toxicity profile. Provide probability, interpretation, and limitations of the prediction.


[32] Case ID: herg_neg_12
Compound: Succinic acid
SMILES: OC(=O)CCC(=O)O
Clinical Context: Analyze compound 'Succinic acid' (SMILES: OC(=O)CCC(=O)O) for hERG blocking risk and Tox21 toxicity profile. Provide probability, interpretation, and limitations of the prediction.

```