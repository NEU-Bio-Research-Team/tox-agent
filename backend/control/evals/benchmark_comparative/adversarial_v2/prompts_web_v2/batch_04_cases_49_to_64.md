# ADVERSARIAL v2 — BATCH 04 / 06 (Cases 49–64 of 95)

> **HƯỚNG DẪN:**
> 1. Copy toàn bộ nội dung trong khung code bên dưới.
> 2. Dán vào **ChatGPT (GPT-4o)** hoặc **Gemini (Gemini 2.5 Pro)** bản Web.
> 3. Copy toàn bộ JSON model trả về và lưu vào file kết quả:
>    - ChatGPT: `web_results_v2/chatgpt_adv_batch_04.json`
>    - Gemini:  `web_results_v2/gemini_adv_batch_04.json`

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

[49] Case ID: herg_brd_09
Compound: 5-Bromosalicylic acid
SMILES: OC(=O)c1cc(Br)ccc1O
Clinical Context: Analyze compound '5-Bromosalicylic acid' (SMILES: OC(=O)c1cc(Br)ccc1O) for hERG blocking risk. The compound has borderline activity. Provide nuanced interpretation.


[50] Case ID: herg_brd_10
Compound: 3-Aminobenzotrifluoride
SMILES: Nc1cccc(C(F)(F)F)c1
Clinical Context: Analyze compound '3-Aminobenzotrifluoride' (SMILES: Nc1cccc(C(F)(F)F)c1) for hERG blocking risk. The compound has borderline activity. Provide nuanced interpretation.


[51] Case ID: tox21_01
Compound: Bisphenol A
SMILES: CC(C)(c1ccc(O)cc1)c1ccc(O)cc1
Clinical Context: Analyze compound 'Bisphenol A' (SMILES: CC(C)(c1ccc(O)cc1)c1ccc(O)cc1) for its complete Tox21 toxicity profile. Which assays are expected to show activity? Explain the biological significance of any active endpoints.


[52] Case ID: tox21_02
Compound: Estradiol
SMILES: C[C@]12CC[C@H]3[C@@H](CC[C@H]4Cc5ccc(O)cc5[C@H]34)[C@@H]1CC[C@@H]2O
Clinical Context: Analyze compound 'Estradiol' (SMILES: C[C@]12CC[C@H]3[C@@H](CC[C@H]4Cc5ccc(O)cc5[C@H]34)[C@@H]1CC[C@@H]2O) for its complete Tox21 toxicity profile. Which assays are expected to show activity? Explain the biological significance of any active endpoints.


[53] Case ID: tox21_03
Compound: 4,4'-Biphenol
SMILES: Oc1ccc(-c2ccc(O)cc2)cc1
Clinical Context: Analyze compound '4,4'-Biphenol' (SMILES: Oc1ccc(-c2ccc(O)cc2)cc1) for its complete Tox21 toxicity profile. Which assays are expected to show activity? Explain the biological significance of any active endpoints.


[54] Case ID: tox21_04
Compound: Rotenone-analog
SMILES: COc1cc2c(cc1OC)C1CC(=O)OC1C2
Clinical Context: Analyze compound 'Rotenone-analog' (SMILES: COc1cc2c(cc1OC)C1CC(=O)OC1C2) for its complete Tox21 toxicity profile. Which assays are expected to show activity? Explain the biological significance of any active endpoints.


[55] Case ID: tox21_05
Compound: Thalidomide
SMILES: O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1
Clinical Context: Analyze compound 'Thalidomide' (SMILES: O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1) for its complete Tox21 toxicity profile. Which assays are expected to show activity? Explain the biological significance of any active endpoints.


[56] Case ID: tox21_06
Compound: 2,4,5-T
SMILES: OC(=O)c1cc(Cl)c(Cl)cc1Cl
Clinical Context: Analyze compound '2,4,5-T' (SMILES: OC(=O)c1cc(Cl)c(Cl)cc1Cl) for its complete Tox21 toxicity profile. Which assays are expected to show activity? Explain the biological significance of any active endpoints.


[57] Case ID: tox21_07
Compound: 4,4'-Dichlorobiphenyl
SMILES: Clc1ccc(-c2ccc(Cl)cc2)cc1
Clinical Context: Analyze compound '4,4'-Dichlorobiphenyl' (SMILES: Clc1ccc(-c2ccc(Cl)cc2)cc1) for its complete Tox21 toxicity profile. Which assays are expected to show activity? Explain the biological significance of any active endpoints.


[58] Case ID: tox21_08
Compound: 2-Nitrofluorene
SMILES: O=[N+]([O-])c1ccc2oc3ccccc3c2c1
Clinical Context: Analyze compound '2-Nitrofluorene' (SMILES: O=[N+]([O-])c1ccc2oc3ccccc3c2c1) for its complete Tox21 toxicity profile. Which assays are expected to show activity? Explain the biological significance of any active endpoints.


[59] Case ID: tox21_09
Compound: 2,6-Di-tert-butylphenol
SMILES: CC(C)(C)c1ccc(O)c(C(C)(C)C)c1
Clinical Context: Analyze compound '2,6-Di-tert-butylphenol' (SMILES: CC(C)(C)c1ccc(O)c(C(C)(C)C)c1) for its complete Tox21 toxicity profile. Which assays are expected to show activity? Explain the biological significance of any active endpoints.


[60] Case ID: tox21_10
Compound: p-Coumaric acid
SMILES: OC(=O)/C=C/c1ccc(O)cc1
Clinical Context: Analyze compound 'p-Coumaric acid' (SMILES: OC(=O)/C=C/c1ccc(O)cc1) for its complete Tox21 toxicity profile. Which assays are expected to show activity? Explain the biological significance of any active endpoints.


[61] Case ID: tox21_11
Compound: Salicylic acid
SMILES: OC(=O)c1ccccc1O
Clinical Context: Analyze compound 'Salicylic acid' (SMILES: OC(=O)c1ccccc1O) for its complete Tox21 toxicity profile. Which assays are expected to show activity? Explain the biological significance of any active endpoints.


[62] Case ID: tox21_12
Compound: Catechin
SMILES: Oc1cc(O)c2c(c1)OC(c1ccc(O)c(O)c1)C(O)C2
Clinical Context: Analyze compound 'Catechin' (SMILES: Oc1cc(O)c2c(c1)OC(c1ccc(O)c(O)c1)C(O)C2) for its complete Tox21 toxicity profile. Which assays are expected to show activity? Explain the biological significance of any active endpoints.


[63] Case ID: tox21_13
Compound: Colchicine-analog
SMILES: COc1cc2[nH]c3cc(OC)c(OC)cc3c2cc1OC
Clinical Context: Analyze compound 'Colchicine-analog' (SMILES: COc1cc2[nH]c3cc(OC)c(OC)cc3c2cc1OC) for its complete Tox21 toxicity profile. Which assays are expected to show activity? Explain the biological significance of any active endpoints.


[64] Case ID: tox21_14
Compound: Sulfonamide-analog
SMILES: CC(=O)Oc1ccc(S(=O)(=O)NC2CCCCCC2)cc1
Clinical Context: Analyze compound 'Sulfonamide-analog' (SMILES: CC(=O)Oc1ccc(S(=O)(=O)NC2CCCCCC2)cc1) for its complete Tox21 toxicity profile. Which assays are expected to show activity? Explain the biological significance of any active endpoints.

```