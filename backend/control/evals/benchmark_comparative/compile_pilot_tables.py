"""Compile transparent 10-case and 95-case comparative pilot metrics."""

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
from compare_agent_v2_pilot import ASSAYS  # noqa: E402


def main():
    dataset = json.loads((ROOT / 'dataset/toxbench_dataset.json').read_text())
    cases = {case['case_id']: case for case in dataset['cases']}
    systems = {}
    for name in ('chatgpt', 'gemini'):
        rows = []
        for file in sorted((ROOT / 'web_results').glob(f'{name}_batch_*.json')):
            rows.extend(json.loads(file.read_text()))
        if (
            len(rows) != 95
            or len({r['case_id'] for r in rows}) != 95
            or set(r['case_id'] for r in rows) != set(cases)
        ):
            raise ValueError(f'{name} does not have 95 unique, matching case IDs')
        v2 = evaluate_v2_results(rows, dataset, name)
        h_total = h_classified = h_correct = h_uncertain = 0
        tp = fp = fn = complete = invalid_names = 0
        for row in rows:
            gold = cases[row['case_id']]['ground_truth']
            if isinstance(gold.get('herg_blocker'), bool):
                h_total += 1
                label = str(row.get('herg_classification', '')).lower().replace('_', '-')
                if label in ('blocker', 'non-blocker'):
                    h_classified += 1
                    h_correct += (label == 'blocker') == gold['herg_blocker']
                elif label == 'uncertain':
                    h_uncertain += 1
            active = set(gold.get('tox21_active_assays', []))
            inactive = set(gold.get('tox21_inactive_assays', []))
            if active | inactive != ASSAYS or active & inactive:
                continue
            pred = row.get('tox21_active_assays')
            if not isinstance(pred, list) or not all(isinstance(x, str) for x in pred):
                continue
            complete += 1
            pred = set(pred)
            invalid_names += bool(pred - ASSAYS)
            tp += len(pred & active)
            fp += len(pred - active)
            fn += len(active - pred)
        systems[name] = {
            'answer_count': len(rows),
            'regex_flags': v2['hallucination_metrics'],
            'herg_v2_raw': v2['predictive_metrics']['herg_classification_accuracy'],
            'herg_correct': h_correct,
            'herg_classified': h_classified,
            'herg_labeled': h_total,
            'herg_uncertain': h_uncertain,
            'tox21_cases': complete,
            'tox21_bad_name_cases': invalid_names,
            'tox21_tp': tp,
            'tox21_fp': fp,
            'tox21_fn': fn,
            'tox21_exact_micro_f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        }
    out = HERE / 'runs/agent_v2_terra_2026-11-07/comparison_95_web.json'
    out.write_text(
        json.dumps(
            {
                'benchmark_commit': '39a4385',
                'systems': systems,
                'warnings': [
                    'Regex matches are not verified hallucinations.',
                    'V2 treats uncertain as non-blocker.',
                    'Exact matching penalizes noncanonical assay names; references not independently verified.',
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + '\n'
    )
    print(json.dumps(systems, ensure_ascii=False, indent=2))
    print('saved', out)


if __name__ == '__main__':
    main()
