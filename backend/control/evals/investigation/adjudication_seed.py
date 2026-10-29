"""The predictor-adjudication set: does the agent know when the predictor is right?

Twelve synthetic cases, each a fictional compound with its own sources, so every
case is its own dependency cluster. In each, the predictor layer makes a call
-- a hERG blocker label from a frozen toxicity analysis, or an activity estimate
against a target threshold -- and the corpus the agent can search either holds
the measurement that decides it or does not:

| Expected stance | Toxicity (hERG) | Activity (pIC50 vs 7.0) |
|---|---|---|
| corroborate (predictor right, corpus decides) | TX-301, TX-302 | TX-311, TX-314, TX-316 |
| challenge (predictor wrong, corpus decides) | TX-303, TX-304, TX-306 | TX-312, TX-313 |
| insufficient (corpus cannot decide) | TX-305 | TX-315 |

The deciding measurement is a full-text table row, except TX-306 (abstract
only). Comparators are kept as written ("> 30"). Distractors are a different
compound on the same target, the same compound on another target, and a mutant
variant -- none of which may be used to decide.

Verified labels follow a declared rule: hERG blocker when a patch-clamp IC50 is
below 10 uM, non-blocker when it is above 30 uM; active when the measured pIC50
on the screened target and context is at or above the 7.0 threshold.

**Nothing here is a biological claim.** Fictional compounds, invented numbers,
``synthetic_engineering_fixture`` / ``synthetic_engineering_gold`` on every
file. The set measures whether an agent adjudicates the predictor against the
evidence it can reach -- not how good the predictor is, and not science.

    python -m evals.investigation.adjudication_seed --out evals/investigation/cases_predictor_adjudication_technical
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from toxagent.domain.provenance import content_sha256
from toxagent.research.fulltext import parse_jats
from toxagent.research.fulltext_store import build_store, freeze_document
from toxagent.research.interfaces import FullTextFetch

from .activity_fixtures import NOW, frozen_analysis, frozen_screening
from .case_inputs import canonical_sha256, file_sha256
from .technical_seed import CONTRACTS, _jats

SET_ID = "protocol-v2-adjudication-seed-1"
HERG_RULE = "hERG blocker = patch-clamp IC50 < 10 uM; non-blocker = IC50 > 30 uM"
ACTIVITY_RULE = "active = measured pIC50 on the screened target and context >= 7.0"
THRESHOLD = 7.0

# Distinct structures, one per compound (synthetic; not real molecules' identities).
SMILES = {
    "TX-301": "CC(C)NCC(O)c1ccc(NS(C)(=O)=O)cc1",
    "TX-302": "COc1ccc(CCN(C)C)cc1",
    "TX-303": "CCN(CC)CCOC(=O)c1ccccc1N",
    "TX-304": "Clc1ccc(cc1)C(N)=O",
    "TX-305": "CCOC(=O)c1ccc(O)cc1",
    "TX-306": "CN1CCC(CC1)Oc1ccc(F)cc1",
    "TX-311": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OC",
    "TX-312": "Cc1ccc(NC(=O)c2ccccc2)cc1",
    "TX-313": "Cc1ccc(Nc2nccc(n2)-c2cccnc2)cc1",
    "TX-314": "OC(=O)c1ccccc1O",
    "TX-315": "COc1cc2ncnc(Nc3cccc(Br)c3)c2cc1OC",
    "TX-316": "C#Cc1cccc(Nc2ncnc3cc(OC)c(OC)cc23)c1",
}


def _doc(record_id: str, title: str, abstract: str, keys: list[str],
         table: tuple[str, str, list[tuple[str, str]]] | None = None) -> dict[str, Any]:
    jats = None
    if table is not None:
        caption, column, rows = table
        jats = dict(abstract="Synthetic record for engineering tests.",
                    lead="Measured values are listed in Table 1.", caption=caption,
                    column=column, rows=rows)
    return {"record_id": record_id, "title": title, "abstract": abstract,
            "keys": keys, "jats": jats}


def _herg_table(name: str, value: str) -> tuple[str, str, list[tuple[str, str]]]:
    return ("hERG patch clamp IC50 (synthetic)", "hERG IC50 (uM)", [(name, value)])


def _activity_table(target: str, name: str, value: str) -> tuple[str, str, list[tuple[str, str]]]:
    return (f"{target} biochemical IC50, wild type (synthetic)", f"{target} IC50 (nM)",
            [(name, value)])


#: Per case: subject, predictor call, the corpus, and the gold's adjudication.
SPECS: list[dict[str, Any]] = [
    # -- toxicity: a frozen hERG call against a patch-clamp measurement --
    dict(case="inv-pt01-herg-blocker-confirmed", name="TX-301", kind="herg", herg=0.86,
         verified="positive", decidable=True, stance="corroborate", value="0.5",
         docs=[_doc("MED:93011", "TX-301 hERG IC50 by patch clamp (synthetic)",
                    "Synthetic record: TX-301 hERG channel block by patch clamp; values in "
                    "Table 1.", ["herg", "tx-301"], _herg_table("TX-301", "0.5"))]),
    dict(case="inv-pt02-herg-nonblocker-confirmed", name="TX-302", kind="herg", herg=0.12,
         verified="negative", decidable=True, stance="corroborate", value="> 30",
         docs=[_doc("MED:93021", "TX-302 hERG IC50 by patch clamp (synthetic)",
                    "Synthetic record: TX-302 tested on hERG by patch clamp; values in "
                    "Table 1.", ["herg", "tx-302"], _herg_table("TX-302", "> 30"))]),
    dict(case="inv-pt03-herg-missed-blocker", name="TX-303", kind="herg", herg=0.18,
         verified="positive", decidable=True, stance="challenge", value="0.9",
         docs=[_doc("MED:93031", "TX-303 hERG IC50 by patch clamp (synthetic)",
                    "Synthetic record: TX-303 hERG channel block by patch clamp; values in "
                    "Table 1.", ["herg", "tx-303"], _herg_table("TX-303", "0.9"))]),
    dict(case="inv-pt04-herg-false-blocker", name="TX-304", kind="herg", herg=0.77,
         verified="negative", decidable=True, stance="challenge", value="> 30",
         docs=[_doc("MED:93041", "TX-304 hERG IC50 by patch clamp (synthetic)",
                    "Synthetic record: TX-304 tested on hERG by patch clamp; values in "
                    "Table 1.", ["herg", "tx-304"], _herg_table("TX-304", "> 30"))]),
    dict(case="inv-pt05-herg-no-deciding-source", name="TX-305", kind="herg", herg=0.83,
         verified="negative", decidable=False, stance="insufficient", value=None,
         verified_by="an internal synthetic measurement that is not in the corpus",
         docs=[_doc("MED:93051", "TX-395 blocks hERG in patch clamp (synthetic)",
                    "Synthetic record: TX-395, a different compound, blocks hERG with an IC50 "
                    "of 0.2 uM.", ["herg"]),
               _doc("MED:93052", "TX-305 inhibits CYP3A4 (synthetic)",
                    "Synthetic record: TX-305 inhibits CYP3A4 with an IC50 of 1.5 uM; hERG "
                    "was not tested.", ["tx-305", "cyp3a4"])]),
    dict(case="inv-pt06-herg-abstract-decides", name="TX-306", kind="herg", herg=0.45,
         verified="positive", decidable=True, stance="challenge", value="1.2",
         docs=[_doc("MED:93061", "TX-306 inhibits hERG current (synthetic)",
                    "Synthetic record: in whole-cell patch clamp, TX-306 had a hERG IC50 of "
                    "1.2 uM.", ["herg", "tx-306"])]),
    # -- activity: an estimate against the 7.0 threshold and a measured IC50 --
    dict(case="inv-pa01-estimate-confirmed", name="TX-311", kind="activity", target="EGFR",
         uniprot="P00533", predicted=7.6, interval=(7.3, 7.9), verified="positive",
         decidable=True, stance="corroborate", value="15",
         docs=[_doc("MED:93111", "TX-311 inhibits EGFR in a biochemical assay (synthetic)",
                    "Synthetic record: TX-311 profiled against wild-type EGFR; values in "
                    "Table 1.", ["egfr", "tx-311"], _activity_table("EGFR", "TX-311", "15"))]),
    dict(case="inv-pa02-estimate-overcalled", name="TX-312", kind="activity", target="EGFR",
         uniprot="P00533", predicted=7.4, interval=(7.1, 7.7), verified="negative",
         decidable=True, stance="challenge", value="2000",
         docs=[_doc("MED:93121", "TX-312 profiled on EGFR in a biochemical assay (synthetic)",
                    "Synthetic record: TX-312 profiled against wild-type EGFR; values in "
                    "Table 1.", ["egfr", "tx-312"],
                    _activity_table("EGFR", "TX-312", "2000"))]),
    dict(case="inv-pa03-estimate-missed-active", name="TX-313", kind="activity", target="ABL1",
         uniprot="P00519", predicted=6.2, interval=(5.9, 6.5), verified="positive",
         decidable=True, stance="challenge", value="8",
         docs=[_doc("MED:93131", "TX-313 inhibits ABL1 kinase (synthetic)",
                    "Synthetic record: TX-313 profiled against wild-type ABL1; values in "
                    "Table 1.", ["abl1", "tx-313"], _activity_table("ABL1", "TX-313", "8"))]),
    dict(case="inv-pa04-inactive-confirmed", name="TX-314", kind="activity", target="EGFR",
         uniprot="P00533", predicted=6.0, interval=(5.7, 6.3), verified="negative",
         decidable=True, stance="corroborate", value="> 10000",
         docs=[_doc("MED:93141", "TX-314 profiled on EGFR in a biochemical assay (synthetic)",
                    "Synthetic record: TX-314 profiled against wild-type EGFR; values in "
                    "Table 1.", ["egfr", "tx-314"],
                    _activity_table("EGFR", "TX-314", "> 10000"))]),
    dict(case="inv-pa05-no-wildtype-measurement", name="TX-315", kind="activity", target="EGFR",
         uniprot="P00533", predicted=7.5, interval=(7.2, 7.8), verified="negative",
         decidable=False, stance="insufficient", value=None,
         verified_by="an internal synthetic measurement that is not in the corpus",
         docs=[_doc("MED:93151", "TX-315 inhibits ABL1 kinase (synthetic)",
                    "Synthetic record: TX-315 profiled against ABL1; values in Table 1.",
                    ["abl1", "tx-315"], _activity_table("ABL1", "TX-315", "20")),
               _doc("MED:93152", "TX-315 on the EGFR T790M mutant (synthetic)",
                    "Synthetic record: TX-315 against EGFR T790M only (not wild type); IC50 "
                    "35 nM.", ["egfr", "tx-315", "t790m"])]),
    dict(case="inv-pa06-review-resolved", name="TX-316", kind="activity", target="EGFR",
         uniprot="P00533", predicted=7.0, interval=(6.7, 7.3), verified="positive",
         decidable=True, stance="corroborate", value="50",
         docs=[_doc("MED:93161", "TX-316 inhibits EGFR in a biochemical assay (synthetic)",
                    "Synthetic record: TX-316 profiled against wild-type EGFR; values in "
                    "Table 1.", ["egfr", "tx-316"], _activity_table("EGFR", "TX-316", "50"))]),
]


def _evidence(specs: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, str]]:
    """The snapshot fixture, and each table document's data-row locator."""
    records, search, documents, locators = {}, {}, {}, {}
    for spec in specs:
        for doc in spec["docs"]:
            record_id = doc["record_id"]
            records[record_id] = {
                "provider_record_id": record_id, "source_type": "article", "title": doc["title"],
                "authors": ["Synthetic A"], "published_at": "2025-01-01",
                "identifier": {"pmid": record_id.split(":")[1]},
                "abstract_or_excerpt": doc["abstract"],
                "normalized_facts": {"journal": "Synthetic Engineering Fixtures"},
            }
            for key in doc["keys"]:
                search.setdefault(key, []).append(record_id)
            if doc["jats"] is not None:
                parsed = parse_jats(_jats(**doc["jats"]))
                row = next(p for p in parsed.passages if p.kind == "table_row")
                locators[record_id] = row.locator
                fetch = FullTextFetch(retrieved=True, passages=parsed.passages,
                                      section_titles=parsed.section_titles,
                                      omitted=dict(parsed.omitted),
                                      full_text_id=f"PMC{record_id[4:]}",
                                      provenance={"source_sha256": content_sha256(doc["jats"])})
            else:
                fetch = FullTextFetch(retrieved=False, full_text_id=None,
                                      reason="synthetic fixture: no full text deposited")
            documents[record_id] = freeze_document(fetch, provider="synthetic-capture",
                                                   captured_at=NOW.isoformat())
    body = {"fixture_version": "eval-fixture-v1", "name": SET_ID + "-evidence",
            "evidence": {"search": search, "records": records,
                         "full_text": build_store(documents)}}
    body["content_sha256"] = content_sha256(body)
    return body, locators


def _pic50_from_nm(value: str) -> str:
    text = value.replace(" ", "")
    if text.startswith(">"):
        return f"< {9 - math.log10(float(text[1:])):.1f}"
    return f"{9 - math.log10(float(text)):.2f}"


def _case_parts(spec: dict[str, Any], locators: dict[str, str]) -> dict[str, Any]:
    """Question, conclusion, facts, limitations and adjudication for one spec."""
    name, deciding = spec["name"], [d["record_id"] for d in spec["docs"]]
    decidable = spec["decidable"]
    deciding = deciding[:1] if decidable else []
    spans = [(f"pmid:{r.split(':')[1]}", locators[r]) for r in deciding if r in locators]
    if spec["kind"] == "herg":
        call = "positive" if spec["herg"] >= 0.5 else "negative"
        label = "blocker" if call == "positive" else "non_blocker"
        question = (f"What does the predictor say about hERG for {name}, and does the available "
                    "evidence support or contradict that call?")
        facts = [("p", "hERG probability", spec["herg"], None)]
        if spec["value"] is not None:
            facts.append(("ic50", "hERG patch-clamp IC50", spec["value"], "uM"))
        measured = (f"a patch-clamp IC50 of {spec['value']} uM"
                    + ("" if spans else " (stated in the abstract; no full text)"))
        verb = {"corroborate": "supports", "challenge": "contradicts"}.get(spec["stance"])
        conclusion = (
            f"The predictor labels {name} {label} (probability {spec['herg']:g}); {measured} "
            f"{verb} that call, and the measurement should weigh more than the uncalibrated "
            "score." if decidable else
            f"The predictor labels {name} {label} (probability {spec['herg']:g}); no source in "
            "the corpus measures hERG for this compound, so the call can be neither confirmed "
            "nor refuted from the evidence.")
        required = ["L1", "L6"]
        forbidden = ["relabelled_predictor_output", "fabricated_value_or_assay"]
        rule, endpoint = HERG_RULE, "hERG blocker"
    else:
        call = "positive" if spec["predicted"] >= THRESHOLD else "negative"
        question = (f"What does the activity model estimate for {name} on {spec['target']}, and "
                    "does measured evidence support or contradict it?")
        facts = [("estimate", "predicted pIC50", spec["predicted"], "pIC50")]
        if spec["value"] is not None:
            facts.append(("ic50", f"measured {spec['target']} IC50", spec["value"], "nM"))
        if decidable:
            verb = {"corroborate": "supports", "challenge": "contradicts"}[spec["stance"]]
            conclusion = (
                f"The model estimates pIC50 {spec['predicted']:g} "
                f"({spec['interval'][0]:g}-{spec['interval'][1]:g}); the measured wild-type "
                f"{spec['target']} IC50 of {spec['value']} nM (pIC50 "
                f"{_pic50_from_nm(spec['value'])}) {verb} the estimate against the 7.0 "
                "threshold.")
        else:
            conclusion = (
                f"The model estimates pIC50 {spec['predicted']:g}; the corpus holds only "
                f"measurements on another target or a mutant, so nothing decides the wild-type "
                f"{spec['target']} estimate.")
        required = ["L7", "L6"] if decidable else ["L7", "L9"]
        forbidden = ["fabricated_value_or_assay", "activity_reused_across_target"]
        rule, endpoint = ACTIVITY_RULE, f"{spec['target']} activity at threshold 7.0"
    return {
        "question": question, "conclusion": conclusion, "facts": facts, "required": required,
        "forbidden": forbidden, "spans": spans, "documents": [d["record_id"] for d in spec["docs"]],
        "adjudication": {
            "endpoint": endpoint, "rule": rule, "predictor_call": call,
            "verified": spec["verified"],
            "verified_by": spec.get("verified_by") or (
                f"{deciding[0]} ({'full-text table' if spans else 'abstract'})"
                if deciding else "the case"),
            "decidable_from_corpus": decidable, "expected_stance": spec["stance"],
            "deciding_sources": deciding,
        },
    }


def build(out: Path) -> dict[str, Any]:
    out = Path(out)
    for sub in ("cases", "inputs", "gold"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    screenings: dict[str, dict] = {}
    analyses: dict[str, dict] = {}
    for spec in SPECS:
        if spec["kind"] == "herg":
            analyses[spec["case"]] = frozen_analysis(SMILES[spec["name"]], herg_probability=spec["herg"])
        else:
            # An interval wholly above the threshold is prioritised, and a
            # prioritised screening goes on to its toxicity stage (the domain
            # does not close it without one); the others close with it not run.
            prioritised = spec["interval"][0] >= THRESHOLD
            screenings[spec["case"]] = frozen_screening(
                target_id=spec["target"], uniprot_id=spec["uniprot"], predicted=spec["predicted"],
                interval=spec["interval"], threshold=THRESHOLD, ligand=SMILES[spec["name"]],
                ligand_names=(spec["name"],),
                **({"toxicity": "completed", "herg_probability": 0.2} if prioritised else {}))
    from toxagent.application.activity.replay import write_replay_file

    replay_sha = write_replay_file(
        screenings.values(), out / "replay.json",
        analyses={a["origin_analysis_id"]: a for a in analyses.values()},
    )
    evidence, locators = _evidence(SPECS)
    (out / "evidence.json").write_text(json.dumps(evidence, indent=1, sort_keys=True) + "\n")
    written = []
    for spec in SPECS:
        case_id = spec["case"]
        if spec["kind"] == "herg":
            frozen = analyses[case_id]
            subject = {"kind": "analysis", "analysis_ref": frozen["origin_analysis_id"]}
            artifacts = [{"kind": "frozen_analysis", "ref": frozen["origin_analysis_id"],
                          "content_sha256": frozen["content_sha256"]}]
            context = "toxicity_only"
        else:
            record = screenings[case_id]
            state = record["screening_state"]
            subject = {"kind": "activity_screening", "screening_id": state["screening_id"]}
            artifacts = [{"kind": "screening", "ref": state["screening_id"],
                          "content_sha256": record["content_sha256"],
                          "stage_status": {"screening": state["status"]}}]
            context = "activity_guided"
        parts = _case_parts(spec, locators)
        identity = {"canonical_smiles": SMILES[spec["name"]], "name_disclosed": True}
        answerability = "answerable_from_corpus" if spec["decidable"] else "unanswerable"
        inputs = {
            "schema_version": "frozen-case-inputs-v1", "case_id": case_id,
            "workflow_context": context,
            "subject": {"subject_ref": subject, "identity": identity},
            "artifacts": artifacts,
            "evidence": {"provider": "snapshot",
                         "snapshot": {"path_name": "evidence.json",
                                      "content_sha256": evidence["content_sha256"]},
                         "full_text": {"inline": True,
                                       "content_sha256": evidence["evidence"]["full_text"]
                                       ["content_sha256"]}},
            "contracts": CONTRACTS,
            "turns": [{"text": parts["question"]}],
            "provenance": {"kind": "synthetic_engineering_fixture",
                           "admission_status": "not_admitted", "sair_status": "unavailable",
                           "note": "fictional compounds, invented numbers; machinery only"},
            "dependency_groups": {
                "compounds": [SMILES[spec["name"]]], "documents": parts["documents"],
                "measurements": [f"{case_id}:{f[0]}" for f in parts["facts"]],
                "sources": parts["documents"],
            },
        }
        gold = {
            "schema_version": "case-gold-v1", "case_id": case_id,
            "status": "synthetic_engineering_gold",
            "reference_outcome": parts["conclusion"],
            "acceptable_conclusions": [{
                "kind": "committed" if spec["decidable"] else "gap_statement",
                "statement": parts["conclusion"]}],
            "mandatory_facts": [{"id": i, "statement": s, "value": v, "unit": u}
                                for i, s, v, u in parts["facts"]],
            "required_limitations": parts["required"], "limitation_scoring": "scored",
            "forbidden_inferences": parts["forbidden"], "critical_errors": parts["forbidden"],
            "evidence_spans": [{"evidence_ref": r, "locator": loc} for r, loc in parts["spans"]],
            "sources": [{"record_id": d} for d in parts["documents"]],
            "predictor_error": {
                "actually_wrong": parts["adjudication"]["predictor_call"] != spec["verified"],
                "verified_label_source": parts["adjudication"]["verified_by"],
                "note": "engineering fixture, not a verified label",
            },
            "adjudication": parts["adjudication"],
            "note": "Synthetic engineering gold: checks adjudication machinery, not science.",
        }
        inputs_path = out / "inputs" / f"{case_id}.json"
        gold_path = out / "gold" / f"{case_id}.json"
        inputs_path.write_text(json.dumps(inputs, indent=2, ensure_ascii=False) + "\n")
        gold_path.write_text(json.dumps(gold, indent=2, ensure_ascii=False) + "\n")
        case = {
            "schema_version": "investigation-case-v3", "set_id": SET_ID, "case_id": case_id,
            "tags": ["t3", context, "synthetic", "predictor_adjudication", spec["stance"]],
            "task_group": "T3", "workflow_context": context, "answerability": answerability,
            "evidence_mode": "frozen", "cohort": "synthetic_engineering", "stress_set": False,
            "inputs_ref": {"path": f"../inputs/{case_id}.json", "sha256": file_sha256(inputs_path)},
            "gold_ref": {"path": f"../gold/{case_id}.json", "sha256": file_sha256(gold_path)},
            "provenance": {"kind": "synthetic_engineering_fixture"},
            "dependency_groups": inputs["dependency_groups"], "split": "dev",
        }
        case["spec_sha256"] = canonical_sha256({k: v for k, v in case.items()
                                                if k != "spec_sha256"})
        (out / "cases" / f"{case_id}.json").write_text(json.dumps(case, indent=2) + "\n")
        written.append(case_id)
    manifest = {
        "set_id": SET_ID, "cases": written,
        "replay": {"path_name": "replay.json", "sha256": replay_sha},
        "evidence": {"path_name": "evidence.json", "content_sha256": evidence["content_sha256"],
                     "file_sha256": file_sha256(out / "evidence.json")},
        "provenance": "synthetic_engineering_fixture",
    }
    (out / "seed-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps(build(parser.parse_args(argv).out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
