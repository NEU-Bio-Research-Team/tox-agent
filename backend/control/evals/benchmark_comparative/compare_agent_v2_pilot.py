"""Compare the two agent pilot answers with the original v2 benchmark outputs.

Run with TOXBENCH_ROOT pointing to evals/benchmark_comparative from commit
39a4385. The original evaluator's regex scores are retained as diagnostics,
not treated as verified hallucination or safety measurements.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(
    os.environ.get(
        'TOXBENCH_ROOT',
        '/tmp/toxagent-agent-benchmark-39a4385/backend/control/evals/benchmark_comparative',
    )
)
sys.path.insert(0, str(ROOT.parents[1]))
from evals.benchmark_comparative.adversarial_v2.evaluate_adversarial_results import (  # noqa: E402
    evaluate_v2_results,
)

ASSAYS = {
    'NR-AhR',
    'NR-AR',
    'NR-AR-LBD',
    'NR-ER',
    'NR-ER-LBD',
    'NR-Aromatase',
    'NR-PPAR-gamma',
    'SR-ARE',
    'SR-ATAD5',
    'SR-HSE',
    'SR-MMP',
    'SR-p53',
}
CASES = ('herg_pos_01', 'herg_pos_02')


def issues(row: dict) -> list[str]:
    errors = []
    for key in (
        'case_id',
        'herg_classification',
        'herg_ic50_um',
        'herg_ic50_source',
        'tox21_active_assays',
        'clinical_recommendation',
        'response_text',
    ):
        if key not in row:
            errors.append('missing ' + key)
    value = row.get('herg_ic50_um')
    if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        errors.append('herg_ic50_um must be a JSON number or null')
    assays = row.get('tox21_active_assays')
    if not isinstance(assays, list) or any(
        not isinstance(x, str) or x not in ASSAYS for x in assays
    ):
        errors.append('tox21_active_assays must contain canonical Tox21 assay names')
    if row.get('herg_classification') not in {'blocker', 'non-blocker', 'uncertain'}:
        errors.append('invalid herg_classification')
    if row.get('clinical_recommendation') not in {'proceed', 'do-not-proceed', 'conditional'}:
        errors.append('invalid clinical_recommendation')
    for key in ('case_id', 'herg_ic50_source', 'response_text'):
        if not isinstance(row.get(key), str):
            errors.append(key + ' must be a string')
    return errors


def main() -> None:
    dataset = json.loads((ROOT / 'dataset/toxbench_dataset.json').read_text())
    cases = [case for case in dataset['cases'] if case['case_id'] in CASES]
    by_id = {case['case_id']: case for case in cases}
    if set(by_id) != set(CASES):
        raise ValueError('Expected two matching benchmark cases')
    systems = {}
    for system in ('chatgpt', 'gemini', 'toxagent'):
        if system == 'toxagent':
            rows = []
            for n in ('01', '02'):
                trace = json.loads(
                    (HERE / f'agent_v2_pilot_case{n}_budget80_2026-11-07.json').read_text()
                )
                items = json.loads(trace['answer']['answer_markdown'])
                if len(items) != 1:
                    raise ValueError('Expected exactly one answer per case')
                rows.append(items[0])
        else:
            rows = [
                r
                for r in json.loads((ROOT / 'web_results' / f'{system}_batch_01.json').read_text())
                if r['case_id'] in CASES
            ]
        if len(rows) != 2 or {r['case_id'] for r in rows} != set(CASES):
            raise ValueError(f'Expected two unique rows for {system}')
        per_case = []
        for row in rows:
            truth = by_id[row['case_id']]['ground_truth']
            per_case.append(
                {
                    'case_id': row['case_id'],
                    'output': row,
                    'schema_issues': issues(row),
                    'reference_herg_blocker': truth['herg_blocker'],
                    'reference_ic50_um': truth['herg_ic50_um'],
                    'reference_tox21_active_assays': truth['tox21_active_assays'],
                    'reference_tox21_inactive_assays': truth['tox21_inactive_assays'],
                }
            )
        systems[system] = {
            'output_schema_pass_count': sum(not p['schema_issues'] for p in per_case),
            'output_schema_total': len(per_case),
            'original_v2_evaluator': evaluate_v2_results(rows, {'cases': cases}, system),
            'per_case': per_case,
        }
    report = {
        'benchmark_commit': '39a4385',
        'cases': list(CASES),
        'limitations': [
            'Two cases and one trial per system; ToxAgent ran one case per session while web results were batched.',
            'Original regex evaluator does not verify citations, clinical claims, negation, or unit conversion.',
            'Empty Tox21 reference lists in these hERG cases are not verified negatives; do not interpret F1.',
            'IC50 dataset entries lack source IDs and matching assay scope.',
        ],
        'systems': systems,
    }
    output = HERE / 'agent_v2_comparison_2cases_2026-11-07.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    for name, item in systems.items():
        score = item['original_v2_evaluator']
        print(
            name,
            'hERG',
            score['predictive_metrics']['herg_classification_accuracy'],
            'schema',
            f"{item['output_schema_pass_count']}/{item['output_schema_total']}",
            'regex_hallucination_flag',
            score['hallucination_metrics']['hallucination_rate'],
            'regex_gate_average',
            score['safety_metrics']['overall_pass_rate'],
        )
    print('saved', output)


if __name__ == '__main__':
    main()
