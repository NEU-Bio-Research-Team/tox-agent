"""Compare equal case IDs across ToxAgent and existing v2 web outputs."""

from __future__ import annotations
import json
import os
from pathlib import Path
from compare_agent_v2_pilot import ASSAYS, issues

HERE = Path(__file__).resolve().parent
ROOT = Path(
    os.environ.get(
        'TOXBENCH_ROOT',
        '/tmp/toxagent-agent-benchmark-39a4385/backend/control/evals/benchmark_comparative',
    )
)
RUNS = Path(os.environ.get('TOXBENCH_RUNS_DIR', str(HERE / 'runs/agent_v2_parallel_2026-11-07')))
INDICES = [3, 4, 21, 22, 41, 51, 52, 60, 71, 81]


def main():
    dataset = json.loads((ROOT / 'dataset/toxbench_dataset.json').read_text())['cases']
    cases = [dataset[i - 1] for i in INDICES]
    ids = [x['case_id'] for x in cases]
    systems = {}
    for name in ['toxagent', 'chatgpt', 'gemini']:
        rows, states = {}, {}
        if name == 'toxagent':
            for i in INDICES:
                c = dataset[i - 1]
                path = RUNS / f'case_{i:02}.json'
                retry = RUNS / f'case_{i:02}_retry1.json'
                if retry.exists():
                    path = retry
                if not path.exists():
                    states[c['case_id']] = {'status': 'missing_trace'}
                    continue
                trace = json.loads(path.read_text())
                run = trace.get('agent_run') or {}
                answer = trace.get('answer') or {}
                rel = answer.get('release') or {}
                states[c['case_id']] = {
                    'status': run.get('status'),
                    'failure': run.get('failure_code'),
                    'error': trace.get('error'),
                    'answer': bool(answer),
                    'fallback': answer.get('is_fallback'),
                    'release': rel.get('outcome'),
                    'reviewed': rel.get('semantically_passed'),
                    'started_at': run.get('started_at'),
                    'ended_at': run.get('ended_at'),
                    'usage': (trace.get('decision_state') or {}).get('usage'),
                    'runtime_model': (run.get('runtime') or {}).get('model_id'),
                    'intent': run.get('intent'),
                    'analysis_status': (trace.get('analysis_run') or {}).get('status'),
                    'selected_trace': str(path),
                    'retry_used': path == retry,
                }
                try:
                    data = json.loads(answer.get('answer_markdown') or '')
                    if (
                        isinstance(data, list)
                        and len(data) == 1
                        and data[0].get('case_id') == c['case_id']
                    ):
                        rows[c['case_id']] = data[0]
                except (ValueError, AttributeError, TypeError):
                    pass
        else:
            for p in sorted((ROOT / 'web_results').glob(f'{name}_batch_*.json')):
                for row in json.loads(p.read_text()):
                    if row.get('case_id') in ids:
                        if row['case_id'] in rows:
                            raise ValueError('duplicate case ID')
                        rows[row['case_id']] = row
        if name != 'toxagent' and set(rows) != set(ids):
            raise ValueError(f'{name}: incomplete cases')
        herg_cases = [c for c in cases if isinstance(c['ground_truth']['herg_blocker'], bool)]
        correct = 0
        classified = 0
        for c in herg_cases:
            pred = rows.get(c['case_id'], {}).get('herg_classification')
            if isinstance(pred, str):
                pred = pred.lower().replace('_', '-')
            if pred in ('blocker', 'non-blocker'):
                classified += 1
                correct += (pred == 'blocker') == c['ground_truth']['herg_blocker']
        systems[name] = {
            'rows': rows,
            'states': states,
            'json_objects': len(rows),
            'format_valid': sum(not issues(row) for row in rows.values()),
            'herg_correct': correct,
            'herg_classified': classified,
            'herg_labeled_total': len(herg_cases),
        }
    # A common-case evaluation prevents systems missing responses from being scored on easier subsets.
    gold_cases = [
        c
        for c in cases
        if set(c['ground_truth']['tox21_active_assays'])
        | set(c['ground_truth']['tox21_inactive_assays'])
        == ASSAYS
    ]
    common = []
    for c in gold_cases:
        if all(
            c['case_id'] in v['rows']
            and isinstance(v['rows'][c['case_id']].get('tox21_active_assays'), list)
            and all(
                isinstance(a, str) and a in ASSAYS
                for a in v['rows'][c['case_id']]['tox21_active_assays']
            )
            for v in systems.values()
        ):
            common.append(c)
    for _name, system in systems.items():
        tp = fp = fn = 0
        for c in common:
            pred = set(system['rows'][c['case_id']]['tox21_active_assays'])
            gold = set(c['ground_truth']['tox21_active_assays'])
            tp += len(pred & gold)
            fp += len(pred - gold)
            fn += len(gold - pred)
        system['tox21_common'] = {
            'cases': [c['case_id'] for c in common],
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        }
    report = {
        'benchmark_commit': '39a4385',
        'indices': INDICES,
        'cases': [
            {k: c[k] for k in ['case_id', 'group', 'compound_name', 'ground_truth']} for c in cases
        ],
        'tox21_labeled_case_ids': [c['case_id'] for c in gold_cases],
        'systems': systems,
        'limitations': [
            'One trial; non-random sample; sessions vs batches 16; real tools and model differ.',
            'Reference labels and measurement scope have not been independently verified.',
            'Non-JSON fallback omitted from parsed classifications but counted in total.',
            'Conditional metrics on parsed answers are shown with coverage.',
        ],
    }
    out = RUNS / 'comparison_10_cases.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    for n, s in systems.items():
        print(
            n,
            'json',
            s['json_objects'],
            'format',
            s['format_valid'],
            'herg',
            s['herg_correct'],
            s['herg_classified'],
            s['herg_labeled_total'],
            'tox21',
            s['tox21_common'],
        )
    print('saved', out)


if __name__ == '__main__':
    main()
