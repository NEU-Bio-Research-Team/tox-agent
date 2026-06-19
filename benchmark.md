# Benchmark for ToxAgent (draft)

Date: 2026-05-25

## Scope
- Datasets: Tox21, ClinTox, hERG
- Metrics: ROC-AUC, PR-AUC, Accuracy, F1, joint_auc_beta3
- Purpose: identify direct competitors and summarize numeric advantages

## Direct competitors (platform)
### ProTox 3.0 (webserver)
- 61 models for toxicity endpoints (acute toxicity, organ toxicity, Tox21 pathways, toxicity targets, etc).
- Acute toxicity + binding to 16 toxicity targets are always computed; extra models can add ~10s each.
- Input options: PubChem search, SMILES string, drawing.

## Direct competitors (model backbones used in our domain)
- ChemBERTa, Molformer, PubChemBERT dual-head checkpoints (single-model baselines).

## Chosen competitors (final 5)
1) ProTox 3.0
2) ProTox-II
3) ADMETlab 2.0
4) pkCSM
5) SwissADME

## Competitor comparison tables (platform level)

### Table A. Platform scope and endpoint breadth
| Platform | Primary scope | Endpoint breadth (reported) | Output depth | Notes |
| --- | --- | --- | --- | --- |
| ToxAgent | Decision-centric toxicity screening | 14 outputs (ClinTox 1 + Tox21 12 + hERG 1) | dual-head scores + explainability + evidence + grounded chat | Internal config; not a full ADMET suite |
| ProTox 3.0 | Broad toxicity endpoint coverage | 61 models; 16 toxicity targets always computed | endpoint predictions + toxicity class | Webserver, breadth-first |
| ProTox-II | Broad toxicity endpoint coverage | Not retrieved | Not retrieved | Legacy ProTox line |
| ADMETlab 2.0 | ADMET / toxicity prediction | Not retrieved | Not retrieved | Needs source detail |
| pkCSM | ADMET / pharmacokinetics prediction | Not stated on landing page | property predictions | Graph-based signatures |
| SwissADME | Molecular parameters for PK/PD | Not stated on landing page | descriptors and PK-related metrics | Tool focuses on ADME, not tox-specific |

### Table B. Input modalities
| Platform | Text SMILES | Drawing | PubChem search | Image OCR | Free-text extraction | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| ToxAgent | Yes | Yes | Not described | Yes | Yes | OCR via image upload + SMILES extraction |
| ProTox 3.0 | Yes | Yes | Yes | No | No | PubChem search + SMILES + drawing |
| ProTox-II | Not retrieved | Not retrieved | Not retrieved | Not retrieved | Not retrieved | Needs source detail |
| ADMETlab 2.0 | Not retrieved | Not retrieved | Not retrieved | Not retrieved | Not retrieved | Needs source detail |
| pkCSM | Not described | Not described | Not described | Not described | Not described | Input UI not documented in retrieved page |
| SwissADME | Yes | Unavailable (site notice) | No | No | No | Site requests SMILES list input |

### Table C. Benchmark metrics availability (public)
| Platform | Tox21 ROC-AUC/PR-AUC | ClinTox ROC-AUC/PR-AUC | hERG ROC-AUC/PR-AUC | Notes |
| --- | --- | --- | --- | --- |
| ToxAgent | Yes (internal) | Yes (internal) | Yes (internal) | Reported in this repo |
| ProTox 3.0 | Not in retrieved pages | Not in retrieved pages | Not in retrieved pages | Model statistics page exists but not parsed |
| ProTox-II | Not retrieved | Not retrieved | Not retrieved | Needs source detail |
| ADMETlab 2.0 | Not retrieved | Not retrieved | Not retrieved | Needs source detail |
| pkCSM | Not in retrieved pages | Not in retrieved pages | Not in retrieved pages | Focus is PK/ADMET, not benchmarked on Tox21 |
| SwissADME | Not in retrieved pages | Not in retrieved pages | Not in retrieved pages | Focus is ADME descriptors |

## Internal benchmark snapshot (dual-head)
| Model | tox21_macro_auc_roc | tox21_macro_pr_auc | herg_auc_roc | herg_pr_auc | joint_auc_beta3 |
| --- | ---:| ---:| ---:| ---:| ---:|
| dualhead_ensemble6_simple | 0.7795 | 0.3916 | 0.8691 | 0.8702 | 0.8467 |
| pretrained_2head_herg_molformer_model | 0.7387 | 0.3501 | 0.8565 | 0.8603 | 0.8270 |

## Tox21 task-level highlights (tox21_gatv2)
- NR-AhR AUC-ROC 0.797, PR-AUC 0.375
- NR-AR-LBD AUC-ROC 0.782, PR-AUC 0.320
- SR-p53 AUC-ROC 0.779, PR-AUC 0.254
- SR-MMP AUC-ROC 0.766, PR-AUC 0.308
- Mean AUC-ROC (12 tasks): 0.725
- Mean PR-AUC (12 tasks): 0.247

## 5-6 numeric advantage points to highlight
1) Ensemble gain vs best single model (Molformer)
   - joint_auc_beta3: 0.8467 vs 0.8270, delta +0.0197.
2) Tox21 macro gains vs best single model (Molformer)
   - macro AUC-ROC: 0.7795 vs 0.7387, delta +0.0408.
   - macro PR-AUC: 0.3916 vs 0.3501, delta +0.0415.
3) hERG branch gains vs best single model (Molformer)
   - AUC-ROC: 0.8691 vs 0.8565, delta +0.0126.
   - PR-AUC: 0.8702 vs 0.8603, delta +0.0099.
4) ClinTox internal run (XSMILES checkpoint)
   - AUC 0.997826, Accuracy 0.972973, F1 0.818182 on test set 148 (10 positive).
5) Input modality coverage (depth of ingestion)
   - ToxAgent: 4 modes (text SMILES, drawing, image OCR, free-text extraction).
   - ProTox 3.0: 3 modes (PubChem search, SMILES string, drawing).
6) Depth-first decision workflow vs breadth-first endpoint grid
   - ToxAgent: 6-agent pipeline (InputValidator, Screening, Researcher, EvidenceQA, Writer, ReportChat) with 4 core capabilities (dual-head, explainability, evidence retrieval, grounded chat).
   - ProTox 3.0: 61 endpoint models (breadth). This contrast supports the depth-first positioning.

## Notes / limitations
- ClinTox result above is on a small positive set (10 positives); report as internal benchmark and include a plan for multi-seed validation if used in a paper.
- ProTox 3.0 numbers are platform-level (endpoint count, target count, input options). They are not direct ROC-AUC comparisons.

## Sources
- Internal metrics:
  - models/dualhead_model_ranking.csv
  - models/tox21_gatv2_model/tox21_task_metrics.csv
  - docs/archive/XSMILES_PERFORMANCE_BRAINSTORM.md
  - docs/papers/ToxAgent_IEEE_ACM_Full_Paper_vi.md
- Competitor sources:
  - http://tox.charite.de/protox_II/
  - https://tox.charite.de/protox3/index.php?site=compound_input
   - https://admetmesh.scbdd.com/
   - https://biosig.lab.uq.edu.au/pkcsm/
   - https://www.swissadme.ch/
