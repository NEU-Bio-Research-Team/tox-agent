You are assisting a scientist in early-stage drug discovery. Answer their message as you would for a colleague.

For reference, this is the output of our in-house toxicity predictor (ToxPred) for the compound in the question, as JSON:
```json
{
  "applicability": {
    "chemical_guard": {
      "method": "element_rules_v1",
      "reasons": [
        "all elements are common in the training sets; this rule cannot confirm distributional similarity beyond element composition"
      ],
      "status": "ok"
    },
    "method": "element_rules_v1",
    "reasons": [
      "all elements are common in the training sets; this rule cannot confirm distributional similarity beyond element composition"
    ],
    "similarity_domain": null,
    "status": "ok"
  },
  "canonical_smiles": "CC(C)(C)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1",
  "predictions": {
    "herg": {
      "label": "blocker",
      "measurement": "hERG channel blockade liability",
      "model_id": "herg-tox21-chemberta-v1",
      "probability_blocker": 0.6803638935089111,
      "threshold": 0.4133453071117401,
      "threshold_source": "artifact"
    },
    "tox21": {
      "assays": {
        "NR-AR": {
          "active": false,
          "probability_activity": 0.2041947990655899,
          "threshold": 0.9399998188018799,
          "threshold_source": "artifact"
        },
        "NR-AR-LBD": {
          "active": false,
          "probability_activity": 0.44677114486694336,
          "threshold": 0.489999920129776,
          "threshold_source": "artifact"
        },
        "NR-AhR": {
          "active": false,
          "probability_activity": 0.021260729059576988,
          "threshold": 0.6799998879432678,
          "threshold_source": "artifact"
        },
        "NR-Aromatase": {
          "active": false,
          "probability_activity": 0.49875369668006897,
          "threshold": 0.7099998593330383,
          "threshold_source": "artifact"
        },
        "NR-ER": {
          "active": false,
          "probability_activity": 0.45679643750190735,
          "threshold": 0.5099999308586121,
          "threshold_source": "artifact"
        },
        "NR-ER-LBD": {
          "active": false,
          "probability_activity": 0.6124134659767151,
          "threshold": 0.6799998879432678,
          "threshold_source": "artifact"
        },
        "NR-PPAR-gamma": {
          "active": true,
          "probability_activity": 0.4762975573539734,
          "threshold": 0.4199999272823334,
          "threshold_source": "artifact"
        },
        "SR-ARE": {
          "active": true,
          "probability_activity": 0.5408167839050293,
          "threshold": 0.49999991059303284,
          "threshold_source": "artifact"
        },
        "SR-ATAD5": {
          "active": false,
          "probability_activity": 0.12436281144618988,
          "threshold": 0.47999992966651917,
          "threshold_source": "artifact"
        },
        "SR-HSE": {
          "active": false,
          "probability_activity": 0.35787075757980347,
          "threshold": 0.5099999308586121,
          "threshold_source": "artifact"
        },
        "SR-MMP": {
          "active": true,
          "probability_activity": 0.6065913438796997,
          "threshold": 0.5399999022483826,
          "threshold_source": "artifact"
        },
        "SR-p53": {
          "active": false,
          "probability_activity": 0.3152877986431122,
          "threshold": 0.34999996423721313,
          "threshold_source": "artifact"
        }
      },
      "measurement": "Twelve independent Tox21 assay activities",
      "model_id": "herg-tox21-chemberta-v1",
      "task_order_version": "tox21-12task-v1"
    }
  },
  "requested_endpoints": [
    "herg",
    "tox21"
  ],
  "unavailable_endpoints": []
}
```

Compound X (CC(C)(C)C1=CC=C(C=C1)C(CCCN2CCC(CC2)C(C3=CC=CC=C3)(C4=CC=CC=C4)O)O) is an early lead in our allergic-rhinitis programme. Is hERG a concern, and what should we test next?