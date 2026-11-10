import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
import yaml
from .checkpointing import Store, atomic, locked
from .dataset import load, benchmark_path
from .discovery import inspect


def resolved_settings(model, settings):
    settings = dict(settings)
    if settings.get('max_model_len') == 'checkpoint':
        settings['max_model_len'] = model['context_window']
    return settings


def experiment(model, settings, sha):
    settings = resolved_settings(model, settings)
    spec = {'model': model, 'generation': settings, 'benchmark_sha256': sha}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(), spec


def run(root, model_id, smoke=False, resume=False, engine_factory=None):
    execution_path = root / 'configs/execution.yaml'
    execution = yaml.safe_load(execution_path.read_text()) if execution_path.exists() else {}
    if not execution.get('run_smoke_tests', True):
        smoke = False
    models = yaml.safe_load((root / 'configs/models.yaml').read_text())
    model = models[model_id]
    if not model['ready']:
        raise ValueError(f'{model_id} blocked: {model["problems"]}')
    local = inspect(model['path'])
    if not local['ready']:
        raise ValueError(f'Checkpoint failed preflight: {local["problems"]}')
    if local['metadata_sha256'] != model['metadata_sha256'] or local['shards'] != model['shard_manifest']:
        raise ValueError('Model metadata or shard manifest changed since discovery')
    settings = resolved_settings(model, yaml.safe_load((root / 'configs/generation.yaml').read_text()))
    rows, _, sha = load(benchmark_path(root))
    fingerprint, spec = experiment(model, settings, sha)
    # Explicit user-authorized continuation policy; retain the original lineage
    # for existing answers, with effective per-answer settings and manifest audit.
    override = execution.get('continuation_overrides', {}).get(model_id)
    if override:
        settings = dict(settings, max_new_tokens=override['max_new_tokens'],
                        retry_max_new_tokens=override['max_new_tokens'],
                        max_model_len=override['max_model_len'])
        model = dict(model, cpu_offload_gb=override['cpu_offload_gb'])
    raw = root / 'outputs/raw' / fingerprint
    store = Store(raw, model_id)
    store.retain_truncated = execution.get('retain_generated_outputs', False)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    manifest_path = root / 'outputs/runs' / (run_id + '.json')
    if smoke:
        selected = [next(r for r in rows if r['language'] == lang) for lang in ('en', 'fr')]
    else:
        selected = rows
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with locked(raw / (model_id + '.lock')):
        old = {r['question_id']: store.get(r['question_id'], fingerprint) for r in selected}
        if not resume and any(old.values()):
            raise ValueError('Results exist: use --resume')
        pending = [r for r in selected if not old[r['question_id']] or old[r['question_id']]['status'] != 'success']
        manifest = dict(spec, run_id=run_id, fingerprint=fingerprint, model_id=model_id,
                        started=datetime.now(timezone.utc).isoformat(), status='running', smoke=smoke,
                        execution_policy=execution,
                        checkpoint_manifest=local, hostname=platform.node(), job_id=os.getenv('SLURM_JOB_ID'),
                        packages={m: importlib.metadata.version(m) for m in ('vllm', 'torch', 'transformers')}
                        if engine_factory is None else {},
                        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in (root / 'src/llm_evals').glob('*.py')})
        atomic(manifest_path, manifest)
        try:
            if engine_factory is None:
                from .inference import VLLM
                engine_factory = VLLM
            if pending:
                engine = (engine_factory(model, settings, batching=execution.get('batching', {}))
                          if execution.get('batching', {}).get('enabled') else engine_factory(model, settings))
            else:
                engine = None
            manifest['effective_chat_template'] = engine.template if engine else 'See earlier run manifest'
            manifest['native_chat_template'] = getattr(engine, 'native_template', None) if engine else None
            atomic(manifest_path, manifest)
            if engine and execution.get('batching', {}).get('enabled'):
                consume_batched(engine, pending, store, fingerprint, model_id, model, settings,
                                run_id, sha, lambda: stopping)
                manifest['batch_runtime'] = getattr(engine, 'batch_runtime', {})
            else:
                for row in pending:
                    if stopping:
                        break
                    qid = row['question_id']
                    previous = store.get(qid, fingerprint)
                    attempts = previous.get('attempts', []) if previous else []
                    # Three retries per invocation; historical attempts are retained on manual resume.
                    for attempt in range(settings['max_attempts']):
                        if stopping:
                            break
                        budget = settings['max_new_tokens'] if not any(a.get('truncated') for a in attempts) else settings['retry_max_new_tokens']
                        record = {'question_id': qid, 'question': row['question'], 'model_id': model_id,
                                  'model_path': model['path'], 'model_revision': model['revision'],
                                  'fingerprint': fingerprint, 'run_id': run_id, 'benchmark_sha256': sha,
                                  'timestamp': datetime.now(timezone.utc).isoformat(),
                                  'generated_answer': '', 'truncated': False, 'status': 'error',
                                  'decoding': dict(settings, actual_max_new_tokens=budget),
                                  'prompt_template': 'effective_chat_template(single user question; add_generation_prompt=True)',
                                  'retry_count': len(attempts), 'error': None}
                        try:
                            answer = engine.answer(row['question'], budget)
                            record.update(answer)
                            if not answer['generated_answer'].strip():
                                raise ValueError('Empty generated answer')
                            if answer['generated_answer'].strip() == row['question'].strip():
                                raise ValueError('Generated answer repeats question verbatim')
                            record['status'] = 'truncated' if answer['truncated'] else 'success'
                        except Exception as exc:
                            record['error'] = f'{type(exc).__name__}: {exc}'
                        attempts.append({k: v for k, v in record.items() if k != 'attempts'})
                        record['attempts'] = attempts.copy()
                        store.put(record)
                        print(json.dumps({'model': model_id, 'question_id': qid, 'status': record['status'],
                                          'attempt': len(attempts)}, ensure_ascii=False), flush=True)
                        if record['status'] == 'success' or (settings['max_new_tokens'] is None and record['status'] == 'truncated'):
                            break
                        if attempt + 1 < settings['max_attempts']:
                            time.sleep(2 ** attempt)
            records = [store.get(r['question_id'], fingerprint) for r in selected]
            successful = sum(bool(r and r['status'] == 'success') for r in records)
            context_limited = sum(bool(r and r['status'] == 'truncated') for r in records)
            captured = successful + context_limited
            manifest.update(successful=successful, context_limited=context_limited, captured=captured,
                            full_answers_complete=successful == len(selected), expected=len(selected),
                            status=('complete' if successful == len(selected) else
                                    'completed_with_context_limits' if captured == len(selected) else 'incomplete'))
            if smoke:
                passed = successful == len(selected)
                atomic(root / 'outputs/runs' / f'smoke-{model_id}-{fingerprint}.json',
                       {'passed': passed, 'run_id': run_id, 'question_ids': [r['question_id'] for r in selected],
                        'review': 'Automated structural checks only; bilingual text is retained for inspection'})
        except BaseException as exc:
            manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            manifest['ended'] = datetime.now(timezone.utc).isoformat()
            atomic(manifest_path, manifest)
        if manifest['status'] not in ('complete', 'completed_with_context_limits'):
            raise RuntimeError(f'Incomplete run: {manifest.get("successful", 0)}/{len(selected)}')
        return manifest



def consume_batched(engine, rows, store, fingerprint, model_id, model, settings, run_id, sha, stopped):
    """Persist every completion before asking the engine for more work."""
    row_map = {r['question_id']: r for r in rows}
    remaining = list(rows)
    for attempt in range(settings['max_attempts']):
        if not remaining or stopped():
            return
        jobs = [{'question_id': r['question_id'], 'question': r['question'],
                 'max_tokens': settings['max_new_tokens']} for r in remaining]
        failed = []
        seen = set()
        for qid, answer in engine.answers(jobs, stopped):
            if qid not in row_map or qid in seen:
                raise ValueError('Unexpected or duplicate batched completion')
            seen.add(qid)
            previous = store.get(qid, fingerprint)
            attempts = previous.get('attempts', []) if previous else []
            record = {'question_id': qid, 'question': row_map[qid]['question'], 'model_id': model_id,
                      'model_path': model['path'], 'model_revision': model['revision'],
                      'fingerprint': fingerprint, 'run_id': run_id, 'benchmark_sha256': sha,
                      'timestamp': datetime.now(timezone.utc).isoformat(), 'status': 'success',
                      'error': None, 'retry_count': len(attempts),
                      'decoding': dict(settings, actual_max_new_tokens=settings['max_new_tokens']),
                      'prompt_template': 'effective_chat_template(single user question; add_generation_prompt=True)',
                      **answer}
            if not answer['generated_answer'].strip():
                record.update(status='error', error='Empty generated answer')
            elif answer['truncated']:
                record['status'] = 'truncated'
            attempts.append({k: v for k, v in record.items() if k != 'attempts'})
            record['attempts'] = attempts
            store.put(record)
            print(json.dumps({'model': model_id, 'question_id': qid, 'status': record['status'],
                              'output_tokens': record.get('output_tokens'),
                              'concurrency': record.get('concurrency_at_submit')}, ensure_ascii=False), flush=True)
            if record['status'] == 'error':
                failed.append(row_map[qid])
        if not stopped() and len(seen) != len(remaining):
            raise RuntimeError('Batch engine omitted unanswered requests')
        remaining = failed
