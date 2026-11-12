"""Compute original v2 regex flags for the same 10-case experiment."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('TOXBENCH_ROOT', '/tmp/toxagent-agent-benchmark-39a4385/backend/control/evals/benchmark_comparative'))
sys.path.insert(0, str(ROOT.parents[1]))
from evals.benchmark_comparative.adversarial_v2.evaluate_adversarial_results import evaluate_v2_results

RUNS = HERE / 'runs'
luna = json.loads((RUNS / 'agent_v2_parallel_2026-11-07/comparison_10_cases.json').read_text())
terra = json.loads((RUNS / 'agent_v2_terra_2026-11-07/comparison_10_cases.json').read_text())
dataset = json.loads((ROOT / 'dataset/toxbench_dataset.json').read_text())
systems = {'luna': luna['systems']['toxagent']['rows'], 'terra': terra['systems']['toxagent']['rows'],
           'chatgpt': terra['systems']['chatgpt']['rows'], 'gemini': terra['systems']['gemini']['rows']}
common = set.intersection(*(set(rows) for rows in systems.values()))
result = {'common_case_ids': sorted(common), 'systems': {},
          'warning': 'Unmodified v2 regex matches are not verified hallucinations; negation and source validity are not assessed.'}
for name, rows in systems.items():
    result['systems'][name] = {}
    for cohort, ids in [('common', common), ('all_parsed', set(rows))]:
        score = evaluate_v2_results([rows[i] for i in sorted(ids)], dataset, name)
        result['systems'][name][cohort] = {'case_ids': sorted(ids), 'n': len(ids), **score['hallucination_metrics']}
out = RUNS / 'agent_v2_terra_2026-11-07/hallucination_regex_comparison.json'
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
for name, scores in result['systems'].items():
    print(name, scores['common']['n'], scores['common']['hallucination_rate'], scores['all_parsed']['n'], scores['all_parsed']['hallucination_rate'])
print('saved', out)
