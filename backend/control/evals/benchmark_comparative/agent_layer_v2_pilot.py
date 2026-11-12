import fcntl
import json
import os
import re
import time
import urllib.request
import urllib.error
from pathlib import Path

root = Path(
    os.environ.get(
        'TOXBENCH_ROOT',
        '/tmp/toxagent-agent-benchmark-39a4385/backend/control/evals/benchmark_comparative',
    )
)
output = Path(os.environ.get('TOXBENCH_PILOT_OUTPUT', 'agent_v2_pilot.json'))
index = int(os.environ.get('TOXBENCH_CASE_INDEX', '1'))
cases = json.loads((root / 'dataset/toxbench_dataset.json').read_text())['cases']
if not 1 <= index <= len(cases):
    raise ValueError('TOXBENCH_CASE_INDEX out of range')
case = cases[index - 1]
batch = sorted((root / 'adversarial_v2/prompts_web_v2').glob('batch_*.md'))[(index - 1) // 16]
text = batch.read_text().split('```text\n', 1)[1].split('```', 1)[0]
header = text.split('\n--- CASES TO ANALYZE ---', 1)[0] + '\n--- CASES TO ANALYZE ---\n'
blocks = re.split(r'(?=\n\[\d+\] Case ID:)', text.split('\n--- CASES TO ANALYZE ---', 1)[1])
block = next(b for b in blocks if re.match(rf'\n\[{index}\] Case ID:', b))
prompt = (header + block).strip()
token = os.environ['TOXAGENT_STATIC_TOKENS'].split(',')[0].split(':')[0]
base = os.environ.get('TOXAGENT_BENCHMARK_URL', 'http://127.0.0.1:8088')


def req(path, data=None, method=None):
    h = {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}
    request = urllib.request.Request(
        base + path,
        data=json.dumps(data).encode() if data is not None else None,
        headers=h,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'{path}: HTTP {e.code}: {e.read().decode()[:500]}') from e


def wait(sid, rid):
    for _ in range(600):
        r = req(f'/v1/sessions/{sid}/runs/{rid}')
        if r['status'] in ('completed', 'failed', 'cancelled'):
            return r
        time.sleep(2)
    raise TimeoutError(rid)


record = {
    'branch': 'origin/Agent_Benchmark',
    'commit': '39a4385',
    'case_id': case['case_id'],
    'prompt': prompt,
    'note': 'One original case; prompt template preserved, other cases omitted to bind one subject.',
}
try:
    s = req('/v1/sessions', {'preferred_language': 'en', 'title': 'ToxBench v2 agent pilot'})[
        'session_id'
    ]
    record['session_id'] = s
    settings = {'ai_profile_id': os.environ['TOXAGENT_CONNECTION_ID'], 'predictor_bindings': {}}
    if os.environ.get('TOXAGENT_MODEL_ID_OVERRIDE'):
        settings['ai_model_id'] = os.environ['TOXAGENT_MODEL_ID_OVERRIDE']
    req(f'/v1/sessions/{s}/settings', settings, 'PATCH')
    with open('/tmp/toxagent-benchmark-analysis.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        a = req(f'/v1/sessions/{s}/messages', {'molecule': {'smiles': case['smiles']}})
        record['analysis_run'] = wait(s, a['run_id'])
        fcntl.flock(lock, fcntl.LOCK_UN)
    print('Analysis:', record['analysis_run']['status'], flush=True)
    if record['analysis_run']['status'] != 'completed':
        raise RuntimeError('analysis failed: ' + str(record['analysis_run'].get('failure_code')))
    a = req(
        f'/v1/sessions/{s}/messages',
        {'intent_hint': 'auto', 'content': [{'type': 'text', 'text': prompt}]},
    )
    rid = a['run_id']
    record['agent_run'] = wait(s, rid)
    print('Agent:', record['agent_run']['status'], flush=True)
    for name, path in [
        ('result', f'/v1/sessions/{s}/runs/{rid}/result'),
        ('messages', f'/v1/sessions/{s}/messages'),
        ('decision_state', f'/v1/sessions/{s}/runs/{rid}/decision-state'),
        ('drafts', f'/v1/sessions/{s}/runs/{rid}/answer-drafts'),
    ]:
        try:
            record[name] = req(path)
        except Exception as exc:
            record[name] = {'error': str(exc)}
    for m in record.get('messages', {}).get('messages', []):
        for part in m.get('parts', []):
            if part.get('type') == 'answer_ref':
                try:
                    answer = req(f"/v1/sessions/{s}/answers/{part['content']['answer_id']}")
                    if answer.get('run_id') == rid:
                        record['answer'] = answer
                except Exception as exc:
                    record['answer_error'] = str(exc)
except Exception as e:
    record['error'] = f'{type(e).__name__}: {e}'
    print(record['error'], flush=True)
finally:
    output.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str))
    print('Saved:', output, flush=True)
