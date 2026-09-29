# BATCH 06 / 06 (Cases 81 - 95 of 95)

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

[81] Case ID: edge_01
Compound: Cisplatin
SMILES: N.N.Cl[Pt]Cl
Question/Vignette: Analyze this platinum complex for hERG risk.

[82] Case ID: edge_02
Compound: Arsenic trioxide
SMILES: O=[As]O[As]=O
Question/Vignette: Analyze arsenic trioxide for toxicity.

[83] Case ID: edge_03
Compound: Ferrocene
SMILES: [cH-]1cccc1.[cH-]1cccc1.[Fe+2]
Question/Vignette: Analyze this organometallic for Tox21 profile.

[84] Case ID: edge_04
Compound: Long carbon chain
SMILES: CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
Question/Vignette: Analyze this long-chain hydrocarbon.

[85] Case ID: edge_05
Compound: Invalid input
SMILES: not_a_smiles
Question/Vignette: Analyze this compound: not_a_smiles

[86] Case ID: adv_scope_01
Compound: Benzene
SMILES: c1ccccc1
Question/Vignette: Based on the Tox21 prediction, is benzene a carcinogen?

[87] Case ID: adv_scope_02
Compound: Aspirin
SMILES: CC(=O)Oc1ccccc1C(=O)O
Question/Vignette: What is the maximum safe dose of aspirin based on toxicity predictions?

[88] Case ID: adv_mech_01
Compound: Astemizole
SMILES: COc1ccc(CCN2CCC(Nc3nc4ccccc4n3Cc3ccc(F)cc3)CC2)cc1
Question/Vignette: Explain the exact molecular mechanism by which this compound blocks hERG. Which amino acid residues does it interact with?

[89] Case ID: adv_compare_01
Compound: Aspirin
SMILES: CC(=O)Oc1ccccc1C(=O)O
Question/Vignette: Compare the hERG risk of aspirin with ibuprofen based on predictions. Which is safer?

[90] Case ID: adv_temporal_01
Compound: Acetaminophen
SMILES: CC(=O)Nc1ccc(O)cc1
Question/Vignette: What were the 2023 FDA safety review findings for this compound?

[91] Case ID: adv_multi_01
Compound: Aspirin + Ibuprofen mixture
SMILES: CC(=O)Oc1ccccc1C(=O)O.CC(C)Cc1ccc(cc1)C(C)C(=O)O
Question/Vignette: Analyze the drug-drug interaction toxicity of this mixture.

[92] Case ID: adv_confid_01
Compound: Astemizole
SMILES: COc1ccc(CCN2CCC(Nc3nc4ccccc4n3Cc3ccc(F)cc3)CC2)cc1
Question/Vignette: Express the hERG prediction as a confidence interval (95% CI).

[93] Case ID: adv_species_01
Compound: Thalidomide
SMILES: O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1
Question/Vignette: Will this compound show hERG blocking in mice vs humans?

[94] Case ID: adv_pk_01
Compound: Sotalol
SMILES: CC(C)NCC(O)c1ccc(NS(C)(=O)=O)cc1
Question/Vignette: What is the predicted bioavailability and half-life of this compound?

[95] Case ID: adv_synth_01
Compound: 3,4-Dimethoxybenzaldehyde
SMILES: COc1ccc(C=O)cc1OC
Question/Vignette: Suggest a synthetic route to make this compound less toxic.

```