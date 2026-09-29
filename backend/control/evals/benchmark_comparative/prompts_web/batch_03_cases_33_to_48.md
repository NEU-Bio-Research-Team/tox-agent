# BATCH 03 / 06 (Cases 33 - 48 of 95)

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

[33] Case ID: herg_neg_13
Compound: 3-Hydroxybutyric acid
SMILES: CC(O)CC(=O)O
Question/Vignette: Analyze compound '3-Hydroxybutyric acid' (SMILES: CC(O)CC(=O)O) for hERG blocking risk and Tox21 toxicity profile. Provide probability, interpretation, and limitations of the prediction.

[34] Case ID: herg_neg_14
Compound: Nicotinamide
SMILES: NC(=O)c1ccc[nH]1
Question/Vignette: Analyze compound 'Nicotinamide' (SMILES: NC(=O)c1ccc[nH]1) for hERG blocking risk and Tox21 toxicity profile. Provide probability, interpretation, and limitations of the prediction.

[35] Case ID: herg_neg_15
Compound: Nicotinic acid
SMILES: OC(=O)c1cccnc1
Question/Vignette: Analyze compound 'Nicotinic acid' (SMILES: OC(=O)c1cccnc1) for hERG blocking risk and Tox21 toxicity profile. Provide probability, interpretation, and limitations of the prediction.

[36] Case ID: herg_neg_16
Compound: Mannitol
SMILES: OC[C@@H](O)[C@@H](O)[C@H](O)[C@@H](O)CO
Question/Vignette: Analyze compound 'Mannitol' (SMILES: OC[C@@H](O)[C@@H](O)[C@H](O)[C@@H](O)CO) for hERG blocking risk and Tox21 toxicity profile. Provide probability, interpretation, and limitations of the prediction.

[37] Case ID: herg_neg_17
Compound: Sodium acetate
SMILES: CC(=O)[O-].[Na+]
Question/Vignette: Analyze compound 'Sodium acetate' (SMILES: CC(=O)[O-].[Na+]) for hERG blocking risk and Tox21 toxicity profile. Provide probability, interpretation, and limitations of the prediction.

[38] Case ID: herg_neg_18
Compound: Maleic acid
SMILES: OC(=O)/C=C\C(=O)O
Question/Vignette: Analyze compound 'Maleic acid' (SMILES: OC(=O)/C=C\C(=O)O) for hERG blocking risk and Tox21 toxicity profile. Provide probability, interpretation, and limitations of the prediction.

[39] Case ID: herg_neg_19
Compound: 1,3-Diaminopropane
SMILES: NCCCN
Question/Vignette: Analyze compound '1,3-Diaminopropane' (SMILES: NCCCN) for hERG blocking risk and Tox21 toxicity profile. Provide probability, interpretation, and limitations of the prediction.

[40] Case ID: herg_neg_20
Compound: Citric acid
SMILES: OC(=O)CC(O)(CC(=O)O)C(=O)O
Question/Vignette: Analyze compound 'Citric acid' (SMILES: OC(=O)CC(O)(CC(=O)O)C(=O)O) for hERG blocking risk and Tox21 toxicity profile. Provide probability, interpretation, and limitations of the prediction.

[41] Case ID: herg_brd_01
Compound: Albuterol
SMILES: CC(C)NCC(O)c1ccc(O)c(CO)c1
Question/Vignette: Analyze compound 'Albuterol' (SMILES: CC(C)NCC(O)c1ccc(O)c(CO)c1) for hERG blocking risk. The compound has borderline activity. Provide nuanced interpretation.

[42] Case ID: herg_brd_02
Compound: Pyridoxine
SMILES: Cc1ncc(CO)c(CO)c1O
Question/Vignette: Analyze compound 'Pyridoxine' (SMILES: Cc1ncc(CO)c(CO)c1O) for hERG blocking risk. The compound has borderline activity. Provide nuanced interpretation.

[43] Case ID: herg_brd_03
Compound: Phenylbutazone-analog
SMILES: CC1(C)C(=O)N(c2ccccc2)N(c2ccccc2)C1=O
Question/Vignette: Analyze compound 'Phenylbutazone-analog' (SMILES: CC1(C)C(=O)N(c2ccccc2)N(c2ccccc2)C1=O) for hERG blocking risk. The compound has borderline activity. Provide nuanced interpretation.

[44] Case ID: herg_brd_04
Compound: Vanillin
SMILES: COc1cc(C=O)ccc1O
Question/Vignette: Analyze compound 'Vanillin' (SMILES: COc1cc(C=O)ccc1O) for hERG blocking risk. The compound has borderline activity. Provide nuanced interpretation.

[45] Case ID: herg_brd_05
Compound: 4-Aminobenzoic acid
SMILES: OC(=O)c1ccc(N)cc1
Question/Vignette: Analyze compound '4-Aminobenzoic acid' (SMILES: OC(=O)c1ccc(N)cc1) for hERG blocking risk. The compound has borderline activity. Provide nuanced interpretation.

[46] Case ID: herg_brd_06
Compound: 4-Hydroxybenzamide
SMILES: NC(=O)c1ccc(O)cc1
Question/Vignette: Analyze compound '4-Hydroxybenzamide' (SMILES: NC(=O)c1ccc(O)cc1) for hERG blocking risk. The compound has borderline activity. Provide nuanced interpretation.

[47] Case ID: herg_brd_07
Compound: 3,4-Dichlorobenzoic acid
SMILES: OC(=O)c1ccc(Cl)c(Cl)c1
Question/Vignette: Analyze compound '3,4-Dichlorobenzoic acid' (SMILES: OC(=O)c1ccc(Cl)c(Cl)c1) for hERG blocking risk. The compound has borderline activity. Provide nuanced interpretation.

[48] Case ID: herg_brd_08
Compound: 3-Methoxysalicylic acid
SMILES: COc1cccc(C(=O)O)c1O
Question/Vignette: Analyze compound '3-Methoxysalicylic acid' (SMILES: COc1cccc(C(=O)O)c1O) for hERG blocking risk. The compound has borderline activity. Provide nuanced interpretation.

```