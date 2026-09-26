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
  "canonical_smiles": "COc1c(N2C[C@@H]3CCCN[C@@H]3C2)c(F)cc2c(=O)c(C(=O)O)cn(C3CC3)c12",
  "predictions": {
    "herg": {
      "label": "non_blocker",
      "measurement": "hERG channel blockade liability",
      "model_id": "herg-tox21-chemberta-v1",
      "probability_blocker": 0.14254340529441833,
      "threshold": 0.4133453071117401,
      "threshold_source": "artifact"
    },
    "tox21": {
      "assays": {
        "NR-AR": {
          "active": false,
          "probability_activity": 0.7929943203926086,
          "threshold": 0.9399998188018799,
          "threshold_source": "artifact"
        },
        "NR-AR-LBD": {
          "active": true,
          "probability_activity": 0.6900485754013062,
          "threshold": 0.489999920129776,
          "threshold_source": "artifact"
        },
        "NR-AhR": {
          "active": false,
          "probability_activity": 0.3516481816768646,
          "threshold": 0.6799998879432678,
          "threshold_source": "artifact"
        },
        "NR-Aromatase": {
          "active": false,
          "probability_activity": 0.3984413146972656,
          "threshold": 0.7099998593330383,
          "threshold_source": "artifact"
        },
        "NR-ER": {
          "active": false,
          "probability_activity": 0.2799506187438965,
          "threshold": 0.5099999308586121,
          "threshold_source": "artifact"
        },
        "NR-ER-LBD": {
          "active": false,
          "probability_activity": 0.3181741237640381,
          "threshold": 0.6799998879432678,
          "threshold_source": "artifact"
        },
        "NR-PPAR-gamma": {
          "active": true,
          "probability_activity": 0.5965786576271057,
          "threshold": 0.4199999272823334,
          "threshold_source": "artifact"
        },
        "SR-ARE": {
          "active": true,
          "probability_activity": 0.5250524282455444,
          "threshold": 0.49999991059303284,
          "threshold_source": "artifact"
        },
        "SR-ATAD5": {
          "active": false,
          "probability_activity": 0.4063534140586853,
          "threshold": 0.47999992966651917,
          "threshold_source": "artifact"
        },
        "SR-HSE": {
          "active": false,
          "probability_activity": 0.29065704345703125,
          "threshold": 0.5099999308586121,
          "threshold_source": "artifact"
        },
        "SR-MMP": {
          "active": false,
          "probability_activity": 0.25407832860946655,
          "threshold": 0.5399999022483826,
          "threshold_source": "artifact"
        },
        "SR-p53": {
          "active": true,
          "probability_activity": 0.489885538816452,
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

Moxifloxacin (COC1=C2C(=CC(=C1N3C[C@@H]4CCCN[C@@H]4C3)F)C(=O)C(=CN2C5CC5)C(=O)O) blocks hERG only at tens of micromolar in patch-clamp studies. Is that a concern at therapeutic exposure, and what information would settle it?