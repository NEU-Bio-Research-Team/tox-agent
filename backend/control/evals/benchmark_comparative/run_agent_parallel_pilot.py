"""Run selected ToxBench agent cases concurrently; preserve every run and failure."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE / 'agent_layer_v2_pilot.py'


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_one(index: int, out: Path) -> dict:
    path = out / f'case_{index:02d}.json'
    log = out / f'case_{index:02d}.log'
    env = os.environ.copy()
    env.update(TOXBENCH_CASE_INDEX=str(index), TOXBENCH_PILOT_OUTPUT=str(path))
    start = now()
    with log.open('w') as stream:
        try:
            p = subprocess.run(
                [sys.executable, str(RUNNER)],
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=1500,
                check=False,
            )
            exit_code = p.returncode
        except subprocess.TimeoutExpired:
            exit_code = None
    entry = {
        'case_index': index,
        'started_at': start,
        'ended_at': now(),
        'runner_exit_code': exit_code,
        'trace': str(path),
        'log': str(log),
    }
    if path.exists():
        d = json.loads(path.read_text())
        entry.update(
            case_id=d.get('case_id'),
            error=d.get('error'),
            analysis_status=(d.get('analysis_run') or {}).get('status'),
            run_status=(d.get('agent_run') or {}).get('status'),
            failure_code=(d.get('agent_run') or {}).get('failure_code'),
            answer_recorded=bool(d.get('answer')),
        )
    else:
        entry['error'] = 'runner did not create trace'
    return entry


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--indices', nargs='+', type=int, required=True)
    p.add_argument('--parallel', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.parallel < 1 or len(set(a.indices)) != len(a.indices):
        p.error('parallel must be positive and indices must be unique')
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = {'started_at': now(), 'parallel': a.parallel, 'indices': a.indices, 'runs': []}
    output = a.out / ('wave_' + '_'.join(map(str, a.indices)) + '.json')

    def save() -> None:
        output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')

    save()
    with ThreadPoolExecutor(max_workers=a.parallel) as pool:
        future_map = {pool.submit(run_one, index, a.out): index for index in a.indices}
        while future_map:
            time.sleep(15)
            for f, index in list(future_map.items()):
                if f.done():
                    try:
                        entry = f.result()
                    except Exception as e:
                        entry = {'case_index': index, 'error': str(e)}
                    manifest['runs'].append(entry)
                    del future_map[f]
                    print(
                        f"Case {index}: {entry.get('run_status')} {entry.get('failure_code') or entry.get('error') or ''}",
                        flush=True,
                    )
                    save()
            if future_map:
                print('Pending cases:', ', '.join(map(str, future_map.values())), flush=True)
    manifest['ended_at'] = now()
    save()
    print('Saved', output, flush=True)


if __name__ == '__main__':
    main()
