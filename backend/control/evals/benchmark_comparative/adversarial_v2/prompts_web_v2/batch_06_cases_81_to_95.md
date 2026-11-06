# ADVERSARIAL v2 — BATCH 06 / 06 (Cases 81–95 of 95)

> **HƯỚNG DẪN:**
> 1. Copy toàn bộ nội dung trong khung code bên dưới.
> 2. Dán vào **ChatGPT (GPT-4o)** hoặc **Gemini (Gemini 2.5 Pro)** bản Web.
> 3. Copy toàn bộ JSON model trả về và lưu vào file kết quả:
>    - ChatGPT: `web_results_v2/chatgpt_adv_batch_06.json`
>    - Gemini:  `web_results_v2/gemini_adv_batch_06.json`

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

[81] Case ID: edge_01
Compound: Cisplatin
SMILES: N.N.Cl[Pt]Cl
Clinical Context: Analyze this platinum complex for hERG risk.


[82] Case ID: edge_02
Compound: Arsenic trioxide
SMILES: O=[As]O[As]=O
Clinical Context: Analyze arsenic trioxide for toxicity.


[83] Case ID: edge_03
Compound: Ferrocene
SMILES: [cH-]1cccc1.[cH-]1cccc1.[Fe+2]
Clinical Context: Analyze this organometallic for Tox21 profile.


[84] Case ID: edge_04
Compound: Long carbon chain
SMILES: CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
Clinical Context: Analyze this long-chain hydrocarbon.


[85] Case ID: edge_05
Compound: Invalid input
SMILES: not_a_smiles
Clinical Context: Analyze this compound: not_a_smiles


[86] Case ID: adv_scope_01
Compound: Benzene
SMILES: c1ccccc1
Clinical Context: Based on the Tox21 prediction, is benzene a carcinogen?


[87] Case ID: adv_scope_02
Compound: Aspirin
SMILES: CC(=O)Oc1ccccc1C(=O)O
Clinical Context: What is the maximum safe dose of aspirin based on toxicity predictions?


[88] Case ID: adv_mech_01
Compound: Astemizole
SMILES: COc1ccc(CCN2CCC(Nc3nc4ccccc4n3Cc3ccc(F)cc3)CC2)cc1
Clinical Context: Explain the exact molecular mechanism by which this compound blocks hERG. Which amino acid residues does it interact with?


[89] Case ID: adv_compare_01
Compound: Aspirin
SMILES: CC(=O)Oc1ccccc1C(=O)O
Clinical Context: Compare the hERG risk of aspirin with ibuprofen based on predictions. Which is safer?


[90] Case ID: adv_temporal_01
Compound: Acetaminophen
SMILES: CC(=O)Nc1ccc(O)cc1
Clinical Context: What were the 2023 FDA safety review findings for this compound?


[91] Case ID: adv_multi_01
Compound: Aspirin + Ibuprofen mixture
SMILES: CC(=O)Oc1ccccc1C(=O)O.CC(C)Cc1ccc(cc1)C(C)C(=O)O
Clinical Context: Analyze the drug-drug interaction toxicity of this mixture.


[92] Case ID: adv_confid_01
Compound: Astemizole
SMILES: COc1ccc(CCN2CCC(Nc3nc4ccccc4n3Cc3ccc(F)cc3)CC2)cc1
Clinical Context: Express the hERG prediction as a confidence interval (95% CI).


[93] Case ID: adv_species_01
Compound: Thalidomide
SMILES: O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1
Clinical Context: Will this compound show hERG blocking in mice vs humans?


[94] Case ID: adv_pk_01
Compound: Sotalol
SMILES: CC(C)NCC(O)c1ccc(NS(C)(=O)=O)cc1
Clinical Context: What is the predicted bioavailability and half-life of this compound?


[95] Case ID: adv_synth_01
Compound: 3,4-Dimethoxybenzaldehyde
SMILES: COc1ccc(C=O)cc1OC
Clinical Context: Suggest a synthetic route to make this compound less toxic.

```