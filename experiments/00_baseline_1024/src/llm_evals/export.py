"""Strict result association, complete-only final export, explicit partial exports."""
import json
import os
import tempfile
import yaml
from .checkpointing import Store, atomic
from .dataset import load, benchmark_path
from .generation import experiment


def export(root, partial=False):
    import openpyxl
    rows, headers, sha = load(benchmark_path(root))
    models = yaml.safe_load((root / 'configs/models.yaml').read_text())
    settings = yaml.safe_load((root / 'configs/generation.yaml').read_text())
    records, failures, wide = [], [], []
    counts = dict.fromkeys(models, 0)
    for row in rows:
        out = dict(row)
        for key, model in models.items():
            fp, _ = experiment(model, settings, sha)
            record = Store(root / 'outputs/raw' / fp, key).get(row['question_id'], fp)
            answer = ''
            if record:
                if record['question'] != row['question'] or record['benchmark_sha256'] != sha:
                    raise ValueError('Result question/checksum mismatch')
                records.append(record)
                if record['status'] == 'success':
                    if not record['generated_answer'].strip() or (record['truncated'] and not record.get('accepted_truncation')):
                        raise ValueError('Invalid successful result')
                    answer = record['generated_answer']
                    counts[key] += 1
                else:
                    failures.append(record)
            else:
                failures.append({'question_id': row['question_id'], 'model_id': key, 'status': 'missing'})
            out[key + '_answer'] = answer
        wide.append(out)
    folder = root / 'outputs/exports'
    folder.mkdir(parents=True, exist_ok=True)
    atomic(folder / 'integrity.json', {'successful': counts, 'expected_per_model': len(rows),
                                      'benchmark_sha256': sha, 'complete': all(n == len(rows) for n in counts.values())})
    if not partial and not all(n == len(rows) for n in counts.values()):
        raise ValueError(f'Final workbook requires 756 successes/model: {counts}. Use --partial for an explicitly incomplete workbook')
    stem = 'canadianness_756_three_model_answers' + ('_PARTIAL' if partial else '')
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = 'CORE Questions'
    columns = headers + [key + '_answer' for key in models]
    sheet.append(columns)
    for row in wide:
        for col, name in enumerate(columns, 1):
            text = row.get(name)
            if isinstance(text, str) and len(text) > 32767:
                raise ValueError('Excel cell limit exceeded; full answer remains in raw records')
            cell = sheet.cell(sheet.max_row if col > 1 else sheet.max_row + 1, col, text)
            if isinstance(text, str):
                cell.data_type = 's'  # preserve literal '=' answers; no formula injection
    sheet.freeze_panes = 'H2'
    sheet.auto_filter.ref = sheet.dimensions
    destination = folder / (stem + '.xlsx')
    fd, tmp = tempfile.mkstemp(dir=folder, suffix='.xlsx')
    os.close(fd)
    try:
        book.save(tmp)
        os.replace(tmp, destination)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    for suffix, values in [('.jsonl', wide), ('_raw.jsonl', records), ('_failures.jsonl', failures)]:
        fd, tmp = tempfile.mkstemp(dir=folder)
        with os.fdopen(fd, 'w') as f:
            for value in values:
                f.write(json.dumps(value, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, folder / (stem + suffix))
    return {'workbook': str(destination), 'successful': counts, 'rows': len(wide)}


def status(root):
    rows, _, sha = load(benchmark_path(root))
    models = yaml.safe_load((root / 'configs/models.yaml').read_text())
    settings = yaml.safe_load((root / 'configs/generation.yaml').read_text())
    result = {}
    for key, model in models.items():
        fp, _ = experiment(model, settings, sha)
        store = Store(root / 'outputs/raw' / fp, key)
        counts = {'success': 0, 'missing': 0, 'error': 0, 'truncated': 0}
        for row in rows:
            record = store.get(row['question_id'], fp)
            if record and (record['question'] != row['question'] or record['benchmark_sha256'] != sha):
                raise ValueError('Result integrity mismatch')
            if record and record['status'] == 'success' and (not record['generated_answer'].strip() or (record['truncated'] and not record.get('accepted_truncation'))):
                raise ValueError('Invalid success record')
            counts[record['status'] if record else 'missing'] += 1
            if record and record['status'] == 'success' and record.get('truncated'):
                counts['truncated'] += 1
        result[key] = dict(counts, ready=model['ready'], problems=model['problems'])
    execution_path = root / 'configs/execution.yaml'
    execution = yaml.safe_load(execution_path.read_text()) if execution_path.exists() else {}
    submissions_path = root / 'outputs/runs/submissions.json'
    submissions = json.loads(submissions_path.read_text()) if submissions_path.exists() else {}
    for key, details in result.items():
        details['batching'] = dict(execution.get('batching', {}))
        ack = submissions.get(key, {}).get('scheduler_acknowledgement', '')
        job = ack.split(';')[0]
        log = root / 'outputs/runs' / f'canadianness-{job}.out'
        if log.exists():
            for line in reversed(log.read_text(errors='replace').splitlines()):
                if line.startswith('BATCH_RUNTIME '):
                    details['batching']['measured'] = json.loads(line[len('BATCH_RUNTIME '):])
                    break
        details['phase'] = ('complete' if details['success'] == len(rows) else
                            'batched_generation' if details['batching'].get('measured') else
                            'loading_or_initializing' if details['batching'].get('enabled') else 'generation')
    return {'benchmark_sha256': sha, 'expected_per_model': len(rows), 'models': result}
